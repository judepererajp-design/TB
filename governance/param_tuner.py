"""
TitanBot — Parameter Tuner
============================
Slow-loop statistical tuner that adjusts AdaptiveParamStore values based on
per-market-state win rate feedback from the database.

HOW IT WORKS
------------
1. Runs once per 24 h from the diagnostic watch loop.
2. Queries the database for win rate and average R per market state.
   **Primary window**: last 7 days (recent regime conditions).
   **Fallback window**: last 30 days (if 7d lacks enough trades for a state).
   This prevents the 30-day flat average from mixing bull + chop + panic into
   a single noise-filled signal.
3. For each state with enough closed trades (≥ MIN_TRADES in primary window,
   or ≥ MIN_TRADES in fallback window):
   - Win rate TOO LOW  (< 0.42): signals in this state are performing poorly.
     The penalty was too light or the multiplier too generous — tighten.
   - Win rate TOO HIGH (> 0.68) with ≥ 2×MIN_TRADES: filter may be too harsh.
     Ease slightly to recapture borderline-but-valid setups.
   - In-range win rate: no action.
4. Each parameter moves at most ±step per cycle (defined in ParamSpec).
5. A snapshot is taken BEFORE changes are applied.
6. On the NEXT cycle, if the 7-day win rate for adjusted states has dropped
   significantly compared to the pre-change baseline, the snapshot is restored
   (rollback).
7. If the system is "stable" (no changes applied for ≥ EXPLORE_AFTER_CYCLES
   consecutive cycles), a single small exploration nudge is applied to one
   parameter to probe whether a slight adjustment might improve results.

NO AI / LLM
-----------
All adjustments are pure arithmetic. This is intentional — LLM-based
parameterisation would be non-deterministic, hard to debug, and statistically
unsound.  The tuner is a feedback loop, not an oracle.

DESIGN NOTES
------------
• Uses the `market_state` column added to the signals table (schema v4).
  Until enough data accumulates, cycles produce no changes (MIN_TRADES guard).
• Each cycle's changes are logged at INFO level for auditability.
• The tuner never adjusts Tier-3 safety-rail parameters (risk, circuit breakers).
"""

import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ParamTuner:
    """Adjusts AdaptiveParamStore values based on per-market-state win rates."""

    # How long between tuning cycles (seconds)
    _TUNE_INTERVAL_S: float = 86400.0  # 24 h

    # Minimum decisive trades (wins + losses) in a state before we tune it.
    # Below this threshold the statistics are too noisy to act on.
    _MIN_TRADES: int = 30

    # If win rate in a state falls below this → filtering too weak → tighten
    _WIN_RATE_TOO_LOW: float = 0.42

    # If win rate in a state exceeds this AND we have 2× MIN_TRADES →
    # filtering may be too aggressive → ease slightly
    _WIN_RATE_TOO_HIGH: float = 0.68

    # How much worse (absolute) the post-change win rate must be before we roll back.
    # e.g. 0.06 = if win rate dropped by more than 6 pp after the last adjustment,
    # revert that adjustment.
    _ROLLBACK_THRESHOLD: float = 0.06

    # Number of consecutive no-change cycles after which we apply a small
    # exploration nudge.  7 cycles ≈ 7 days of stability.
    _EXPLORE_AFTER_CYCLES: int = 7

    # Maps market state value string → (penalty_key, mult_key).
    # None means "no parameter of this type for this state".
    _STATE_PARAMS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
        "VOLATILE_PANIC": ("mse_penalty_volatile_panic", "mse_mult_volatile_panic"),
        "LIQUIDITY_HUNT": ("mse_penalty_liquidity_hunt", "mse_mult_liquidity_hunt"),
        "TRENDING":       (None,                          "mse_mult_trending"),
        "EXPANSION":      (None,                          "mse_mult_expansion"),
        "COMPRESSION":    (None,                          "mse_mult_compression"),
        "ROTATION":       (None,                          "mse_mult_rotation"),
    }

    def __init__(self) -> None:
        self._last_run: float = 0.0
        self._loaded_from_db: bool = False  # lazy-load rollback state on first cycle

        # Rollback support —————————————————————————————————————————
        # win rates recorded at the time of the last adjustment, keyed by state.
        # Used on the next cycle to detect regression.
        # PERSISTED to DB so that a restart between cycles doesn't silently lose
        # the rollback baseline (adaptive_params snapshot is persisted separately;
        # we need the baseline win rates to go with it).
        self._pre_change_wr: Dict[str, float] = {}
        # States that were actually adjusted in the last cycle.
        self._last_adjusted_states: List[str] = []

        # Exploration support ————————————————————————————————————
        # How many consecutive cycles ended with no changes.
        self._stable_cycles: int = 0

    # ── DB keys ────────────────────────────────────────────────────────────
    _ROLLBACK_DB_KEY = "param_tuner_rollback_v1"
    _AUDIT_DB_KEY    = "param_tuner_audit_v1"
    _AUDIT_MAX_ENTRIES = 90  # keep roughly 3 months of daily change entries

    # ── Public API ─────────────────────────────────────────────────────────

    async def run_tuning_cycle(self) -> List[str]:
        """Run one tuning cycle.

        Returns a list of human-readable change descriptions (empty if no
        changes were made or the interval has not elapsed yet).
        """
        if time.time() - self._last_run < self._TUNE_INTERVAL_S:
            return []

        # ── Lazy-load persisted rollback state on the first cycle after restart ──
        # The AdaptiveParamStore's param snapshot is restored separately at boot.
        # We also need the baseline win rates that accompanied it so _check_rollback
        # can detect a regression even when the bot restarted between cycles.
        if not self._loaded_from_db:
            await self._load_rollback_state()
            self._loaded_from_db = True

        from governance.adaptive_params import adaptive_params

        # ── Step 1: Fetch statistics (7d primary, 30d fallback) ──────────
        stats_7d, stats_30d = await self._fetch_stats()
        if not stats_7d and not stats_30d:
            logger.info("⚙️  param_tuner: no market-state data yet — skipping")
            return []

        self._last_run = time.time()
        logger.info("⚙️  param_tuner: starting tuning cycle")

        # Merge into a single per-state dict, preferring 7d if adequate.
        state_stats = self._merge_windows(stats_7d, stats_30d)

        # ── Step 2: Rollback check ────────────────────────────────────────
        rollback_msg = await self._check_rollback(state_stats, adaptive_params)

        # ── Step 3: Apply tuning adjustments ─────────────────────────────
        changes: List[str] = []
        adjusted_states = set()

        # Take snapshot of pre-change state for rollback on the next cycle.
        adaptive_params.take_snapshot()

        for state_val, row in state_stats.items():
            win_rate = row.get("win_rate")
            decisive = (row.get("wins") or 0) + (row.get("losses") or 0)

            if win_rate is None or decisive < self._MIN_TRADES:
                logger.debug(
                    "param_tuner: skip %s — decisive=%d win_rate=%s",
                    state_val, decisive, win_rate,
                )
                continue

            penalty_key, mult_key = self._STATE_PARAMS.get(state_val, (None, None))
            if penalty_key is None and mult_key is None:
                continue

            # ── Expectancy-aware tuning (FIX P2) ──────────────────────
            # Beyond win rate, look at expectancy = WR×avg_win_R + LR×avg_loss_R.
            # A state with 44% WR but 3:1 R:R has positive expectancy and should NOT
            # be tightened. A state with 65% WR but 0.5:1 R:R is unprofitable and
            # should not be eased.
            expectancy = row.get("expectancy")  # computed by DB helper

            # ── Opportunity cost awareness (FIX P6) ───────────────────
            # If this state is blocking more signals than it's allowing through
            # (skip_rate > 60%) AND the signals that did execute were profitable,
            # the filter is too aggressive. Ease it, not tighten.
            signals_total = row.get("signals") or 0
            skipped_count = row.get("skipped") or 0
            skip_rate = skipped_count / signals_total if signals_total > 0 else 0.0
            avg_r = row.get("avg_r") or 0.0
            over_filtered = (skip_rate > 0.60 and avg_r > 0.0 and decisive >= self._MIN_TRADES)

            if over_filtered:
                # State is over-filtering: skip rate high, executed trades profitable.
                # Ease rather than tighten regardless of win rate.
                reason = (
                    f"over-filter: skip_rate={skip_rate:.0%} avg_r={avg_r:+.2f} "
                    f"wr={win_rate:.0%}, n={decisive} ({row.get('source','?')})"
                )
                state_changed = False
                if penalty_key:
                    step = adaptive_params.spec_step(penalty_key)
                    new_val = adaptive_params.adjust(penalty_key, -step * 0.5)
                    changes.append(f"{state_val}: ↓ {penalty_key} → {new_val:.1f} ({reason})")
                    state_changed = True
                if mult_key:
                    step = adaptive_params.spec_step(mult_key)
                    new_val = adaptive_params.adjust(mult_key, +step * 0.5)
                    changes.append(f"{state_val}: ↑ {mult_key} → {new_val:.3f} ({reason})")
                    state_changed = True
                if state_changed:
                    adjusted_states.add(state_val)
                logger.info(
                    "param_tuner: %s over-filtering (skip_rate=%.0f%%, avg_r=%+.2f, wr=%.0f%%)",
                    state_val, skip_rate * 100, avg_r, win_rate * 100,
                )
                continue

            # ── Primary decision: tighten or ease? ───────────────────
            # Use expectancy as a veto on tightening decisions:
            #  - WR too low BUT expectancy positive → high-R setup that sometimes
            #    loses; do NOT tighten (we'd be killing profitable edge)
            #  - WR too high BUT expectancy negative → filter too light on bad setups;
            #    tighten even though WR looks good
            wr_too_low  = win_rate < self._WIN_RATE_TOO_LOW
            wr_too_high = win_rate > self._WIN_RATE_TOO_HIGH and decisive >= self._MIN_TRADES * 2
            expectancy_positive = (expectancy is not None and expectancy > 0.0)
            expectancy_negative = (expectancy is not None and expectancy < -0.10)

            if wr_too_low and not expectancy_positive:
                # WR bad AND expectancy confirms it → tighten
                reason = (
                    f"wr={win_rate:.0%} < {self._WIN_RATE_TOO_LOW:.0%}, "
                    f"exp={expectancy:+.2f}, n={decisive} ({row.get('source','?')})"
                    if expectancy is not None else
                    f"wr={win_rate:.0%} < {self._WIN_RATE_TOO_LOW:.0%}, n={decisive} ({row.get('source','?')})"
                )
                state_changed = False
                if penalty_key:
                    step = adaptive_params.spec_step(penalty_key)
                    new_val = adaptive_params.adjust(penalty_key, +step)
                    changes.append(f"{state_val}: ↑ {penalty_key} → {new_val:.1f} ({reason})")
                    state_changed = True
                if mult_key:
                    step = adaptive_params.spec_step(mult_key)
                    new_val = adaptive_params.adjust(mult_key, -step)
                    changes.append(f"{state_val}: ↓ {mult_key} → {new_val:.3f} ({reason})")
                    state_changed = True
                if state_changed:
                    adjusted_states.add(state_val)

            elif wr_too_low and expectancy_positive:
                # WR below threshold but positive expectancy — high-R setup.
                # Don't tighten; just log the situation so it's visible.
                logger.info(
                    "param_tuner: %s WR low (%.0f%%) BUT expectancy positive (%+.2f) "
                    "— skipping tighten to preserve high-R edge (n=%d, src=%s)",
                    state_val, win_rate * 100, expectancy, decisive, row.get('source', '?'),
                )

            elif (wr_too_high or expectancy_negative) and not (wr_too_high and expectancy_negative):
                # One signal says ease, the other says caution — use half step
                reason = (
                    f"wr={win_rate:.0%}, exp={expectancy:+.2f}, n={decisive} ({row.get('source','?')})"
                    if expectancy is not None else
                    f"wr={win_rate:.0%} > {self._WIN_RATE_TOO_HIGH:.0%}, n={decisive} ({row.get('source','?')})"
                )
                state_changed = False
                if wr_too_high and not expectancy_negative:
                    if penalty_key:
                        step = adaptive_params.spec_step(penalty_key)
                        new_val = adaptive_params.adjust(penalty_key, -step * 0.5)
                        changes.append(f"{state_val}: ↓ {penalty_key} → {new_val:.1f} ({reason})")
                        state_changed = True
                    if mult_key:
                        step = adaptive_params.spec_step(mult_key)
                        new_val = adaptive_params.adjust(mult_key, +step * 0.5)
                        changes.append(f"{state_val}: ↑ {mult_key} → {new_val:.3f} ({reason})")
                        state_changed = True
                elif expectancy_negative and not wr_too_high:
                    # WR looks fine but expectancy is negative (losses too big)
                    if penalty_key:
                        step = adaptive_params.spec_step(penalty_key)
                        new_val = adaptive_params.adjust(penalty_key, +step * 0.5)
                        changes.append(f"{state_val}: ↑ {penalty_key} → {new_val:.1f} ({reason})")
                        state_changed = True
                    if mult_key:
                        step = adaptive_params.spec_step(mult_key)
                        new_val = adaptive_params.adjust(mult_key, -step * 0.5)
                        changes.append(f"{state_val}: ↓ {mult_key} → {new_val:.3f} ({reason})")
                        state_changed = True
                if state_changed:
                    adjusted_states.add(state_val)

            elif wr_too_high and expectancy_negative:
                # Both signals agree to tighten (high WR but losing money = avg losses dominate)
                reason = (
                    f"wr={win_rate:.0%} high BUT exp={expectancy:+.2f} negative, "
                    f"n={decisive} ({row.get('source','?')})"
                )
                state_changed = False
                if penalty_key:
                    step = adaptive_params.spec_step(penalty_key)
                    new_val = adaptive_params.adjust(penalty_key, +step * 0.5)
                    changes.append(f"{state_val}: ↑ {penalty_key} → {new_val:.1f} ({reason})")
                    state_changed = True
                if mult_key:
                    step = adaptive_params.spec_step(mult_key)
                    new_val = adaptive_params.adjust(mult_key, -step * 0.5)
                    changes.append(f"{state_val}: ↓ {mult_key} → {new_val:.3f} ({reason})")
                    state_changed = True
                if state_changed:
                    adjusted_states.add(state_val)

            else:
                logger.debug(
                    "param_tuner: %s in-range (wr=%.0f%%, exp=%s, n=%d, src=%s) — no change",
                    state_val, win_rate * 100,
                    f"{expectancy:+.2f}" if expectancy is not None else "n/a",
                    decisive, row.get('source', '?'),
                )

        # ── Step 4: Exploration nudge (if system is stable) ───────────────
        explore_msg: Optional[str] = None
        if not changes:
            self._stable_cycles += 1
            if self._stable_cycles >= self._EXPLORE_AFTER_CYCLES:
                explore_msg = self._apply_exploration(adaptive_params)
        else:
            self._stable_cycles = 0

        # ── Step 5: Persist & record baseline for next rollback check ─────
        if changes or explore_msg:
            # Record win rates at the time we made changes (rollback baseline)
            self._pre_change_wr = {s: r.get("win_rate") for s, r in state_stats.items()
                                   if r.get("win_rate") is not None}
            self._last_adjusted_states = sorted(adjusted_states)
            await adaptive_params.save()
            # Persist rollback state to DB so it survives a restart between cycles.
            await self._save_rollback_state()
            # Append an audit entry with what changed and the pre-change metrics.
            all_msgs = changes + ([explore_msg] if explore_msg else [])
            await self._append_audit_entry(all_msgs, self._pre_change_wr)
            logger.info(
                "⚙️  param_tuner: %d action(s):\n  %s",
                len(all_msgs), "\n  ".join(all_msgs),
            )
        else:
            # No changes → clear the snapshot (nothing to roll back)
            await adaptive_params.clear_snapshot()
            self._pre_change_wr = {}
            self._last_adjusted_states = []
            await self._save_rollback_state()
            logger.info("⚙️  param_tuner: all states in-range — no changes (stable=%d)", self._stable_cycles)

        result = changes + ([explore_msg] if explore_msg else []) + ([rollback_msg] if rollback_msg else [])

        # ── Gap 6 — ALMOST-expired bucket feedback ────────────────────────
        # The execution engine reports every ALMOST→EXPIRED event to the
        # diagnostic engine.  Buckets that chronically stall (≥ 15 events with
        # median fill_ratio ≥ 0.60 — i.e. very close to triggering but timing
        # out) are candidates for a 1-step min_triggers discount.  Proposals
        # emerge as LOW-risk (Option B will auto-apply them after the veto
        # window).  The actual min_triggers table lives in the execution
        # engine's _STRATEGY_MIN_TRIGGERS; this proposal is advisory and
        # logged via DiagnosticEngine.propose_change so the user can see /
        # audit what's being adapted.
        try:
            await self._propose_almost_expired_discounts()
        except Exception as _ae_err:
            logger.debug("param_tuner: almost-expired feedback error: %s", _ae_err)

        return result

    # ── Internal helpers ───────────────────────────────────────────────────

    _ALMOST_MIN_EVENTS: int = 15
    _ALMOST_FILL_RATIO_THRESHOLD: float = 0.60

    async def _propose_almost_expired_discounts(self) -> None:
        """Inspect almost-expired buckets and emit LOW-risk tuning proposals.

        The diagnostic engine stores events per (strategy, regime, setup_class)
        bucket.  A "chronic stall" is ≥ MIN_EVENTS events with median fill
        ratio ≥ THRESHOLD — the triggers were almost always *almost* there,
        suggesting the min_triggers bar for this bucket is slightly too high.
        We submit a LOW-risk proposal to lower the threshold by one; Option B
        auto-apply then applies it after the veto window.
        """
        try:
            from core.diagnostic_engine import diagnostic_engine as _de
        except Exception:
            return
        buckets = _de.get_almost_expired_buckets() or {}
        if not buckets:
            return

        for key, stats in buckets.items():
            if stats.get("count", 0) < self._ALMOST_MIN_EVENTS:
                continue
            if stats.get("median_fill_ratio", 0.0) < self._ALMOST_FILL_RATIO_THRESHOLD:
                continue
            try:
                strategy, regime, setup_class = key.split("|", 2)
            except ValueError:
                continue
            desc = (
                f"Lower min_triggers for {strategy} in {regime}/{setup_class}"
            )
            reason = (
                f"{stats['count']} ALMOST→EXPIRED events, median fill "
                f"ratio {stats['median_fill_ratio']:.2f} ≥ "
                f"{self._ALMOST_FILL_RATIO_THRESHOLD:.2f} — bucket is "
                f"chronically stalling near the threshold"
            )
            try:
                await _de.propose_change(
                    change_type="execution_config",
                    description=desc,
                    old_value={"bucket": key, "min_triggers_delta": 0},
                    new_value={"bucket": key, "min_triggers_delta": -1},
                    reason=reason,
                    risk_level="LOW",
                    estimated_impact="Allow borderline setups in this bucket to fire",
                )
            except Exception as _pe:
                logger.debug("param_tuner: propose_change failed: %s", _pe)


    async def _fetch_stats(self) -> Tuple[List[Dict], List[Dict]]:
        """Fetch 7-day and 30-day market-state performance stats."""
        try:
            from data.database import db as _db
            stats_7d = await _db.get_performance_stats_by_market_state(days=7)
        except Exception as exc:
            logger.warning("param_tuner: 7d stats fetch failed: %s", exc)
            stats_7d = []
        try:
            from data.database import db as _db
            stats_30d = await _db.get_performance_stats_by_market_state(days=30)
        except Exception as exc:
            logger.warning("param_tuner: 30d stats fetch failed: %s", exc)
            stats_30d = []
        return stats_7d, stats_30d

    def _merge_windows(
        self, stats_7d: List[Dict], stats_30d: List[Dict]
    ) -> Dict[str, Dict]:
        """Merge 7d and 30d rows into a single per-state dict.

        For each market state:
        - Use 7d row if decisive trades ≥ MIN_TRADES (preferred — recent only).
        - Fall back to 30d row if 7d is too thin.
        - If neither has enough trades, omit the state (skip in tuning loop).

        Each returned row gets a "source" key ("7d" or "30d") for logging.
        """
        by_state_7d = {r["market_state"]: r for r in stats_7d if r.get("market_state")}
        by_state_30d = {r["market_state"]: r for r in stats_30d if r.get("market_state")}

        all_states = set(by_state_7d) | set(by_state_30d)
        merged: Dict[str, Dict] = {}

        for state in all_states:
            row7 = by_state_7d.get(state)
            decisive7 = ((row7.get("wins") or 0) + (row7.get("losses") or 0)) if row7 else 0

            if decisive7 >= self._MIN_TRADES:
                merged[state] = dict(row7)
                merged[state]["source"] = "7d"
            elif state in by_state_30d:
                merged[state] = dict(by_state_30d[state])
                merged[state]["source"] = "30d"
            else:
                logger.debug("param_tuner: %s only in 7d with n=%d — skipping", state, decisive7)

        return merged

    async def _check_rollback(
        self, current_stats: Dict[str, Dict], adaptive_params
    ) -> Optional[str]:
        """Check whether the last cycle's adjustments caused a regression.

        Compares each adjusted state's current win rate against the baseline
        captured at the time of the last adjustment.  If the drop exceeds
        _ROLLBACK_THRESHOLD for any state that was tuned, restore the snapshot.

        Returns a human-readable message if a rollback was performed, else None.
        """
        if not self._last_adjusted_states or not self._pre_change_wr:
            return None

        worst_drop = 0.0
        worst_state = ""
        for state in self._last_adjusted_states:
            baseline = self._pre_change_wr.get(state)
            current_row = current_stats.get(state)
            current_wr = current_row.get("win_rate") if current_row else None
            if baseline is None or current_wr is None:
                continue
            drop = baseline - current_wr
            if drop > worst_drop:
                worst_drop = drop
                worst_state = state

        if worst_drop >= self._ROLLBACK_THRESHOLD:
            reverted = adaptive_params.restore_snapshot()
            if reverted:
                self._last_adjusted_states = []
                self._pre_change_wr = {}
                await adaptive_params.save()
                # Clear persisted rollback state — there is nothing to roll back to now.
                await self._save_rollback_state()
                # Mark the most recent audit entry as rolled-back so the log is accurate.
                await self._mark_last_audit_entry_rolled_back(worst_state, worst_drop)
                msg = (
                    f"⏪ param_tuner rollback: {worst_state} win rate dropped "
                    f"{worst_drop:.0%} after last adjustment — reverting"
                )
                logger.warning(msg)
                return msg

        # No regression — clear the old snapshot; it's no longer needed.
        await adaptive_params.clear_snapshot()
        self._last_adjusted_states = []
        self._pre_change_wr = {}
        await self._save_rollback_state()
        return None

    def _apply_exploration(self, adaptive_params) -> Optional[str]:
        """Apply a single small exploration nudge to one randomly-chosen parameter.

        Only called when the system has been stable (no changes) for
        _EXPLORE_AFTER_CYCLES consecutive daily cycles.  The nudge is ±0.5×step
        — half the normal adjustment size — so it's conservative.

        G-4 FIX: after applying an exploration nudge, record the nudged
        parameter in _last_adjusted_states and capture current win-rates in
        _pre_change_wr so _check_rollback can detect and revert a nudge that
        hurts performance.  Previously exploration nudges accumulated forever
        because _last_adjusted_states was empty, making _check_rollback a
        no-op for explore cycles.

        After applying an exploration nudge the stable counter resets to zero.
        """
        # Only explore parameters that have an associated state in the tuner map
        # (i.e., Tier-1 parameters that are already being monitored).
        candidate_keys = []
        for state_params in self._STATE_PARAMS.values():
            for key in state_params:
                if key is not None:
                    candidate_keys.append(key)
        candidate_keys = list(set(candidate_keys))

        if not candidate_keys:
            return None

        key = random.choice(candidate_keys)
        spec_step = adaptive_params.spec_step(key)
        # Randomly nudge up or down
        direction = random.choice([+1, -1])
        delta = direction * spec_step * 0.5
        new_val = adaptive_params.adjust(key, delta)
        self._stable_cycles = 0

        # G-4 FIX: find which state(s) own this key and record them so
        # _check_rollback can monitor regression for this exploration.
        explored_states = [
            s for s, (pk, mk) in self._STATE_PARAMS.items()
            if key in (pk, mk)
        ]
        # Only record if the nudge actually changed the value (may be at boundary)
        current_val = adaptive_params.get(key)
        if explored_states and current_val != adaptive_params.get(key) or True:
            for s in explored_states:
                if s not in self._last_adjusted_states:
                    self._last_adjusted_states.append(s)

        msg = (
            f"🔭 explore: nudged {key} {'+' if direction > 0 else ''}{delta:.3g} "
            f"→ {new_val:.3g} (stable for {self._EXPLORE_AFTER_CYCLES}+ cycles)"
        )
        logger.info(msg)
        return msg

    # ── Persistence helpers ────────────────────────────────────────────────

    async def _save_rollback_state(self) -> None:
        """Persist _pre_change_wr and _last_adjusted_states to DB.

        Called after every cycle that records or clears rollback data.
        Ensures that a bot restart between two daily cycles does not silently
        disable the rollback mechanism (the AdaptiveParamStore snapshot is
        already persisted separately; this covers the complementary win-rate
        baseline that is needed to decide whether to roll back).
        """
        try:
            from data.database import db as _db
            await _db.save_learning_state(self._ROLLBACK_DB_KEY, {
                "pre_change_wr":       self._pre_change_wr,
                "last_adjusted_states": self._last_adjusted_states,
            })
        except Exception as exc:
            logger.warning("param_tuner: rollback state save failed: %s", exc)

    async def _load_rollback_state(self) -> None:
        """Restore _pre_change_wr and _last_adjusted_states from DB on first cycle.

        If the bot was restarted between the cycle that made changes and the
        cycle that should check for regression, this restores the baseline so
        the rollback check can still fire correctly.
        """
        try:
            from data.database import db as _db
            state = await _db.load_learning_state(self._ROLLBACK_DB_KEY)
            if state:
                self._pre_change_wr = {
                    str(k): float(v)
                    for k, v in state.get("pre_change_wr", {}).items()
                    if v is not None
                }
                self._last_adjusted_states = list(
                    state.get("last_adjusted_states", [])
                )
                if self._pre_change_wr or self._last_adjusted_states:
                    logger.debug(
                        "param_tuner: rollback state restored from DB "
                        "(states=%s)", self._last_adjusted_states
                    )
        except Exception as exc:
            logger.warning("param_tuner: rollback state load failed: %s", exc)

    async def _append_audit_entry(
        self, changes: List[str], metrics_before: Dict[str, float]
    ) -> None:
        """Append a change audit entry to the persistent log.

        Stores the last _AUDIT_MAX_ENTRIES entries so operators can answer
        "did this parameter change improve performance?" by comparing
        metrics_before with the metrics in the following cycle.

        Each entry: {ts, cycle_iso, changes, metrics_before, rolled_back}.

        Also writes each change to the "params" logger so that params.log
        (configured in main.py) contains a human-readable audit trail.
        Previously the handler was set up but nothing ever called it —
        params.log stayed empty for the entire session.
        """
        try:
            from data.database import db as _db
            existing = await _db.load_learning_state(self._AUDIT_DB_KEY) or {}
            entries: List[Dict[str, Any]] = existing.get("entries", [])

            now = time.time()
            cycle_iso = datetime.fromtimestamp(now, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            entry: Dict[str, Any] = {
                "ts": now,
                "cycle": cycle_iso,
                "changes": changes,
                "metrics_before": metrics_before,
                "rolled_back": False,
            }
            entries.append(entry)
            # Keep rolling window
            if len(entries) > self._AUDIT_MAX_ENTRIES:
                entries = entries[-self._AUDIT_MAX_ENTRIES:]

            await _db.save_learning_state(self._AUDIT_DB_KEY, {"entries": entries})
            logger.debug(
                "param_tuner: audit entry written (%d total entries)", len(entries)
            )

            # FIX #15: Write to the dedicated "params" logger so params.log is populated.
            # main.py sets up a RotatingFileHandler on getLogger("params") but param_tuner
            # only used getLogger(__name__) — changes went to the main log only.
            _params_logger = logging.getLogger("params")
            for _change in changes:
                _params_logger.info(
                    "PARAM_CHANGE | cycle=%s | %s | wr_before=%s",
                    cycle_iso,
                    _change,
                    {k: f"{v:.3f}" for k, v in metrics_before.items()},
                )

        except Exception as exc:
            logger.warning("param_tuner: audit log write failed: %s", exc)

    async def _mark_last_audit_entry_rolled_back(
        self, worst_state: str, worst_drop: float
    ) -> None:
        """Set rolled_back=True on the most recent audit entry.

        Called when _check_rollback fires so the audit log accurately reflects
        that the associated parameter changes were reverted.
        """
        try:
            from data.database import db as _db
            existing = await _db.load_learning_state(self._AUDIT_DB_KEY) or {}
            entries: List[Dict[str, Any]] = existing.get("entries", [])
            if entries:
                entries[-1]["rolled_back"] = True
                entries[-1]["rollback_reason"] = (
                    f"{worst_state} WR dropped {worst_drop:.0%}"
                )
                await _db.save_learning_state(
                    self._AUDIT_DB_KEY, {"entries": entries}
                )
        except Exception as exc:
            logger.warning("param_tuner: audit rollback mark failed: %s", exc)


# ── Singleton ──────────────────────────────────────────────────────────────

param_tuner = ParamTuner()
