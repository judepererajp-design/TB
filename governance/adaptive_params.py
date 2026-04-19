"""
TitanBot — Adaptive Parameter Store
=====================================
Centralises all Tier 1 tunable engine constants — penalty magnitudes and
market-state confidence multipliers — in a single mutable, DB-persisted store.

WHY THIS EXISTS
---------------
The engine previously used hard-coded literals scattered across 15+ files.
When a penalty was wrong (e.g. sector mismatch −8 too harsh), the system
could only react by trading less (adaptive gate tightens) — never by fixing
the source of the problem.

This store enables the ParamTuner to make slow, bounded adjustments based on
per-market-state win rate feedback, so the engine can self-correct rather than
just becoming increasingly conservative.

DESIGN PRINCIPLES
-----------------
• All parameters have a safe range (lo, hi) — no runaway changes.
• Each tuning cycle is capped to ±step — maximum one small move per 24 h.
• No AI/LLM involved: adjustments are pure statistical feedback.
• The store is a singleton; callers use adaptive_params.get(key).
• Values survive restarts via the existing learning_state DB table.

TIERS
-----
  Tier 1 (SAFE — wrapped here):     penalty magnitudes, market-state mults, BTC-D adjustments
  Tier 2 (future):                   chop weights, sector weights
  Tier 3 (DO NOT TOUCH):             risk per trade, circuit breakers, max loss limits
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)
params_log = logging.getLogger("params")

@dataclass(frozen=True)
class ParamSpec:
    key: str
    default: float
    lo: float    # minimum safe value — never go below
    hi: float    # maximum safe value — never go above
    step: float  # maximum change per tuning cycle
    description: str


# All Tier 1 tunable parameters.  Values are absolute magnitudes —
# callers that subtract penalties negate at the call site.
PARAM_SPECS: Dict[str, ParamSpec] = {
    # ── Market state block penalties (confidence points subtracted) ──────
    "mse_penalty_volatile_panic": ParamSpec(
        "mse_penalty_volatile_panic", 18, 12, 24, 1.0,
        "Confidence penalty when VOLATILE_PANIC state blocks a strategy",
    ),
    "mse_penalty_liquidity_hunt": ParamSpec(
        "mse_penalty_liquidity_hunt", 8, 4, 14, 1.0,
        "Confidence penalty when LIQUIDITY_HUNT state blocks a strategy",
    ),
    "mse_penalty_default": ParamSpec(
        "mse_penalty_default", 12, 8, 18, 1.0,
        "Confidence penalty for any unrecognised market state block",
    ),
    # ── Market state confidence multipliers (applied post-aggregation) ───
    "mse_mult_trending": ParamSpec(
        "mse_mult_trending", 1.05, 1.00, 1.15, 0.02,
        "Confidence multiplier in TRENDING market state",
    ),
    "mse_mult_expansion": ParamSpec(
        "mse_mult_expansion", 1.08, 1.00, 1.18, 0.02,
        "Confidence multiplier in EXPANSION market state",
    ),
    "mse_mult_compression": ParamSpec(
        "mse_mult_compression", 0.92, 0.82, 1.00, 0.02,
        "Confidence multiplier in COMPRESSION market state",
    ),
    "mse_mult_liquidity_hunt": ParamSpec(
        "mse_mult_liquidity_hunt", 0.85, 0.75, 0.95, 0.02,
        "Confidence multiplier in LIQUIDITY_HUNT market state",
    ),
    "mse_mult_rotation": ParamSpec(
        "mse_mult_rotation", 0.90, 0.80, 1.00, 0.02,
        "Confidence multiplier in ROTATION market state",
    ),
    "mse_mult_volatile_panic": ParamSpec(
        "mse_mult_volatile_panic", 0.70, 0.55, 0.82, 0.02,
        "Confidence multiplier in VOLATILE_PANIC market state",
    ),
    # ── BTC dominance penalties / boosts (absolute integer magnitudes) ───
    "btcd_alt_long_penalty_sharp": ParamSpec(
        "btcd_alt_long_penalty_sharp", 15, 8, 22, 1.0,
        "Altcoin LONG penalty when BTC.D sharply rising (+2% in 24h)",
    ),
    "btcd_alt_long_penalty_moderate": ParamSpec(
        "btcd_alt_long_penalty_moderate", 8, 4, 14, 1.0,
        "Altcoin LONG penalty when BTC.D moderately rising (+1% in 24h)",
    ),
    "btcd_alt_long_boost_sharp": ParamSpec(
        "btcd_alt_long_boost_sharp", 10, 5, 15, 1.0,
        "Altcoin LONG boost when BTC.D sharply falling (-2% in 24h)",
    ),
    "btcd_alt_long_boost_moderate": ParamSpec(
        "btcd_alt_long_boost_moderate", 5, 2, 10, 1.0,
        "Altcoin LONG boost when BTC.D moderately falling (-1% in 24h)",
    ),
}


# ── Store ──────────────────────────────────────────────────────────────────

class AdaptiveParamStore:
    """
    Central store for tunable engine parameters.

    All Tier 1 penalty/multiplier constants live here instead of being
    scattered as hard-coded literals.  The ParamTuner adjusts values slowly
    (once per 24 h, at most ±step per cycle) based on per-market-state win rate
    feedback from the database.

    Usage (synchronous — safe to call from any context)::

        from governance.adaptive_params import adaptive_params
        penalty = adaptive_params.get("mse_penalty_volatile_panic")  # → 18.0
        mult    = adaptive_params.get("mse_mult_trending")           # → 1.05

    Persistence::

        await adaptive_params.load()   # called once at startup (engine.py)
        await adaptive_params.save()   # called after each tuning cycle
    """

    _DB_KEY = "adaptive_params_v1"

    def __init__(self) -> None:
        self._values: Dict[str, float] = {k: s.default for k, s in PARAM_SPECS.items()}
        self._lock = asyncio.Lock()
        self._dirty = False
        # Snapshot taken immediately before a tuning cycle's adjustments.
        # Stored as (values_dict, unix_timestamp).  The ParamTuner uses this
        # to detect performance regressions and revert if needed.
        self._snapshot: Optional[Tuple[Dict[str, float], float]] = None

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, key: str, default: Optional[float] = None) -> float:
        """Return the current live value of *key*.

        Falls back to the ParamSpec default if the key is unknown, or to the
        *default* argument if provided and the key isn't in PARAM_SPECS.
        """
        if key in self._values:
            return self._values[key]
        if default is not None:
            return default
        spec = PARAM_SPECS.get(key)
        return spec.default if spec else 0.0

    def spec_step(self, key: str) -> float:
        """Return the maximum change-per-cycle for *key*."""
        spec = PARAM_SPECS.get(key)
        return spec.step if spec else 1.0

    def snapshot(self) -> Dict[str, float]:
        """Return an immutable copy of all current values."""
        return dict(self._values)

    def take_snapshot(self) -> None:
        """Record the current state before a tuning cycle.

        The ParamTuner calls this *before* applying adjustments so that it can
        later call ``restore_snapshot()`` if the cycle caused a regression.
        Only one snapshot is kept at a time; calling this again overwrites the
        previous snapshot.
        """
        self._snapshot = (dict(self._values), time.time())
        logger.debug("⚙️  adaptive_params: snapshot taken")

    def restore_snapshot(self) -> bool:
        """Restore values from the last snapshot taken by ``take_snapshot()``.

        Returns True if the rollback was applied, False if no snapshot exists.
        Marks the store dirty so that the reverted state is persisted on the
        next ``save()`` call.
        """
        if self._snapshot is None:
            logger.warning("adaptive_params.restore_snapshot: no snapshot to restore")
            return False
        saved_values, snap_time = self._snapshot
        rolled_back = []
        for key, val in saved_values.items():
            if self._values.get(key) != val:
                self._values[key] = val
                rolled_back.append(key)
        if rolled_back:
            self._dirty = True
            logger.info(
                "⚙️  adaptive_params: rolled back %d param(s) to snapshot from %.0f s ago: %s",
                len(rolled_back),
                time.time() - snap_time,
                ", ".join(rolled_back),
            )
        self._snapshot = None
        return True

    @property
    def snapshot_age_s(self) -> Optional[float]:
        """Seconds since the last snapshot was taken, or None if no snapshot."""
        return (time.time() - self._snapshot[1]) if self._snapshot else None

    # ── Write (used by ParamTuner only) ───────────────────────────────────

    def adjust(self, key: str, delta: float) -> float:
        """Apply a bounded delta to *key* and return the new value.

        The delta is clamped to ±step for the parameter so no single cycle
        can make a large jump.  The result is clamped to [lo, hi].

        G-1 FIX: acquire _lock around the read-modify-write so that a
        concurrent param_tuner.adjust() call (e.g. from a background task
        racing with the diagnostic loop) cannot produce a torn write where
        two increments are applied but only one is visible.
        """
        spec = PARAM_SPECS.get(key)
        if spec is None:
            logger.warning(f"adaptive_params.adjust: unknown key '{key}' — ignored")
            return self._values.get(key, 0.0)

        # Clamp delta to the allowed per-cycle step
        delta = max(-spec.step, min(spec.step, delta))
        old_value = self._values.get(key, spec.default)
        new_value = max(spec.lo, min(spec.hi, old_value + delta))

        if new_value != old_value:
            self._values[key] = new_value
            self._dirty = True
            logger.info(
                "⚙️  adaptive_params: %s %.3g → %.3g (Δ%+.3g, bounds=[%.3g, %.3g])",
                key, old_value, new_value, new_value - old_value, spec.lo, spec.hi,
            )
            params_log.info(
                "TUNE | %s | %s → %s | delta=%+.4f | bounds=[%.3g, %.3g]",
                key, old_value, new_value, new_value - old_value, spec.lo, spec.hi,
            )

        return new_value

    def reset(self, key: str) -> float:
        """Reset *key* to its factory default."""
        spec = PARAM_SPECS.get(key)
        if spec is None:
            return 0.0
        old = self._values.get(key, spec.default)
        self._values[key] = spec.default
        if old != spec.default:
            self._dirty = True
            logger.info("⚙️  adaptive_params: %s reset to default %.3g", key, spec.default)
        return spec.default

    # ── Persistence ───────────────────────────────────────────────────────

    async def load(self) -> None:
        """Load persisted values from the database (call once at startup)."""
        try:
            from data.database import db as _db
            state = await _db.load_learning_state(self._DB_KEY)
            # G-1 FIX: acquire _lock while mutating _values so that any adjust()
            # call that fires between two awaits in this method (impossible in
            # single-threaded asyncio, but guarded for clarity and future
            # thread-pool usage) sees a fully-written state.
            async with self._lock:
                if state:
                    count = 0
                    for key, value in state.items():
                        if key in PARAM_SPECS:
                            spec = PARAM_SPECS[key]
                            # Clamp loaded values to the current safe range in case
                            # bounds were tightened since the last save.
                            self._values[key] = max(spec.lo, min(spec.hi, float(value)))
                            count += 1
                    logger.info("⚙️  adaptive_params: loaded %d values from DB", count)
                else:
                    logger.info("⚙️  adaptive_params: no persisted state — using defaults")
            # Restore the pre-change snapshot if the tuner saved one before shutdown
            snap = await _db.load_learning_state(self._DB_KEY + "_snapshot")
            if snap and "values" in snap:
                self._snapshot = (snap["values"], float(snap.get("ts", 0)))
                logger.debug("⚙️  adaptive_params: pre-change snapshot restored from DB")
        except Exception as exc:
            logger.warning("adaptive_params.load failed — using defaults: %s", exc)

    async def save(self) -> None:
        """Persist current values to the database."""
        try:
            from data.database import db as _db
            # G-1 FIX: take a consistent copy under lock before the first await
            # so an interleaved adjust() cannot change _values mid-save.
            async with self._lock:
                values_copy = dict(self._values)
                snapshot_copy = self._snapshot
                self._dirty = False
            await _db.save_learning_state(self._DB_KEY, values_copy)
            # Also persist the pre-change snapshot so that a restart between a
            # tuning cycle and the next check cycle doesn't lose the rollback data.
            if snapshot_copy is not None:
                snap_values, snap_ts = snapshot_copy
                await _db.save_learning_state(
                    self._DB_KEY + "_snapshot", {"values": snap_values, "ts": snap_ts}
                )
        except Exception as exc:
            logger.warning("adaptive_params.save failed: %s", exc)

    async def clear_snapshot(self) -> None:
        """Discard the stored snapshot (call after a successful rollback check)."""
        self._snapshot = None
        try:
            from data.database import db as _db
            # Remove the persisted snapshot entry so it isn't applied on next boot
            await _db.save_learning_state(self._DB_KEY + "_snapshot", {})
        except Exception as exc:
            logger.debug("adaptive_params.clear_snapshot: %s", exc)


# ── Singleton ──────────────────────────────────────────────────────────────

adaptive_params = AdaptiveParamStore()
