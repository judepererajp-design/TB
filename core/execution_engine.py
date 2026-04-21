
"""
TitanBot Pro — Execution Engine (Upgraded)
==========================================
Human‑anticipation execution lifecycle.

States:
  APPROVED   → Signal created
  PREPARING  → Price moving toward entry (early awareness)
  WATCHING   → Price approaching entry
  ENTRY_ZONE → Price inside entry zone
  ALMOST     → 1 trigger detected
  EXECUTE    → 2+ triggers confirmed
  FILLED / EXPIRED / INVALIDATED

Triggers (need 2 of 4):
  1. Rejection candle
  2. Structure shift
  3. Momentum expansion
  4. Liquidity reaction ⭐
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Callable

from config.loader import cfg
from data.api_client import api
from signals.liquidity_reaction import detect_liquidity_reaction
from analyzers.market_state_builder import build_market_state
from analyzers.market_brain import build_market_profile
from governance.execution_journal import log_execution
from governance.regime_tracker import compute_regime_stats
from governance.risk_governor import get_risk_adjustment
from governance.execution_confidence import compute_confidence
from core.price_cache import price_cache
from config.constants import Timing

logger = logging.getLogger(__name__)
try:
    from utils.trade_logger import trade_logger as _tl
except Exception:
    _tl = None


def _safe_ensure_future(coro, *, context: str = "DB write"):
    """Schedule a coroutine as a fire-and-forget task with error logging.

    FIX #2 (AUDIT): All previous ``asyncio.ensure_future()`` calls swallowed
    exceptions silently.  This wrapper attaches a done-callback that logs
    failures so operators can detect persistence problems from the log stream.
    """
    task = asyncio.ensure_future(coro)

    def _on_done(t: asyncio.Task):
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error(f"[{context}] background task failed: {exc}")

    task.add_done_callback(_on_done)
    return task


class SignalState(Enum):
    APPROVED = "APPROVED"
    PREPARING = "PREPARING"
    WATCHING = "WATCHING"
    ENTRY_ZONE = "ENTRY_ZONE"
    ALMOST = "ALMOST"
    EXECUTE = "EXECUTE"
    FILLED = "FILLED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


# FIX #31: Setup-class-aware timeouts.
# Previously a flat 4h timeout applied to ALL signals equally.
# A 15m scalp signal should expire in ~1.5h; a 4h swing signal in 8h.
# Scalp/intraday entries have tight zones that go stale quickly.
# Swing/positional setups legitimately take 6-12h to develop.
EXECUTION_WATCHING_TIMEOUT_BY_SETUP = Timing.EXECUTION_WATCHING_BY_SETUP
EXECUTION_WATCHING_TIMEOUT = Timing.EXECUTION_WATCHING_TIMEOUT_SECS
EXECUTION_ALMOST_TIMEOUT    = Timing.EXECUTION_ALMOST_TIMEOUT_SECS
EXECUTION_CHECK_INTERVAL    = Timing.EXECUTION_CHECK_INTERVAL_SECS
EXECUTION_FAST_CHECK_INTERVAL = Timing.EXECUTION_FAST_CHECK_INTERVAL_SECS
TRIGGER_WINDOW_CONFIRMATION_BARS = 4
TRIGGER_WINDOW_MIN_SECS = 20 * 60
TRIGGER_WINDOW_MAX_SECS = 48 * 3600
MIN_PRICE_DENOMINATOR = 1e-9
NEAR_ENTRY_FAST_POLL_DISTANCE = 0.02
# Minimum decay penalty in confidence points before we require one extra
# confirmation trigger for a still-pending entry.
DECAY_TRIGGER_INCREMENT_THRESHOLD = 5.0
LOCAL_RANGE_EXECUTE_BYPASS_STRATEGIES = frozenset({
    "InstitutionalBreakout",
    "Momentum",
    "MomentumContinuation",
    "RangeScalper",
})
LOCAL_RANGE_WRONG_SIDE_THRESHOLD = 0.20

def _get_watching_timeout(sig) -> int:
    """Return setup-class-aware watching timeout in seconds."""
    if sig is None:
        return EXECUTION_WATCHING_TIMEOUT
    sc = getattr(sig, 'setup_class', None) or 'intraday'
    return EXECUTION_WATCHING_TIMEOUT_BY_SETUP.get(sc, EXECUTION_WATCHING_TIMEOUT)


def _timeframe_to_seconds(timeframe: str) -> int:
    mapping = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    return mapping.get(str(timeframe or "").lower(), 3600)


def _late_local_range_reason(sig, price: float, ohlcv: list) -> Optional[str]:
    """Block live execute calls that have drifted into the wrong side of a tight local range."""
    if getattr(sig, "strategy", "") in LOCAL_RANGE_EXECUTE_BYPASS_STRATEGIES:
        return None
    if price <= 0 or not ohlcv or len(ohlcv) < 10:
        return None

    try:
        highs = [float(bar[2]) for bar in ohlcv]
        lows = [float(bar[3]) for bar in ohlcv]
    except Exception:
        return None

    range_high = max(highs, default=0.0)
    range_low = min(lows, default=0.0)
    if range_high <= range_low:
        return None

    from config.constants import LocalRange as _LR

    range_pct = (range_high - range_low) / max(price, MIN_PRICE_DENOMINATOR)
    if range_pct > _LR.RANGE_PCT_THRESHOLD:
        return None

    eq = (range_high + range_low) / 2
    half_range = (range_high - range_low) / 2
    if half_range <= MIN_PRICE_DENOMINATOR:
        return None

    position = max(-1.0, min(1.0, (price - eq) / half_range))
    if getattr(sig, "direction", "") == "LONG" and position > LOCAL_RANGE_WRONG_SIDE_THRESHOLD:
        return (
            f"Late LONG execute blocked: price drifted into premium of tight local range "
            f"({position:+.2f}, range {range_pct*100:.1f}%)"
        )
    if getattr(sig, "direction", "") == "SHORT" and position < -LOCAL_RANGE_WRONG_SIDE_THRESHOLD:
        return (
            f"Late SHORT execute blocked: price drifted into discount of tight local range "
            f"({position:+.2f}, range {range_pct*100:.1f}%)"
        )
    return None


def _get_trigger_window_secs(setup_class: str) -> int:
    from strategies.base import SETUP_CLASS_CONFIRM_TF

    confirm_tf = SETUP_CLASS_CONFIRM_TF.get(setup_class or "intraday", "15m")
    # Keep triggers coherent within roughly four confirmation bars while retaining
    # a hard 20m floor for fast setups and a 48h cap for very slow ones.
    return max(
        TRIGGER_WINDOW_MIN_SECS,
        min(
            TRIGGER_WINDOW_MAX_SECS,
            _timeframe_to_seconds(confirm_tf) * TRIGGER_WINDOW_CONFIRMATION_BARS,
        ),
    )


# Gap 6 feedback loop — per-(strategy|regime|setup_class) min_triggers deltas
# applied by track() on top of the static _STRATEGY_MIN_TRIGGERS table.
# Populated by DiagnosticEngine._apply_known_change when an "execution_config"
# LOW-risk proposal is approved (manually or via auto-apply).  Deltas are
# clamped to [-2, +2] so a runaway tuner can't drive min_triggers negative
# or absurdly high.  The floor at 1 trigger is re-applied at read time.
_MIN_TRIG_BUCKET_DELTAS: Dict[str, int] = {}
_MIN_TRIG_BUCKET_DELTA_CLAMP: int = 2


def set_min_triggers_bucket_delta(bucket: str, delta: int) -> int:
    """Record a clamped delta for (strategy|regime|setup_class) bucket.

    Returns the clamped value actually stored.  A delta of 0 removes the
    entry so lookups fall back to the static table cleanly.
    """
    try:
        d = int(delta)
    except (TypeError, ValueError):
        return 0
    d = max(-_MIN_TRIG_BUCKET_DELTA_CLAMP,
            min(_MIN_TRIG_BUCKET_DELTA_CLAMP, d))
    if not bucket:
        return 0
    if d == 0:
        _MIN_TRIG_BUCKET_DELTAS.pop(bucket, None)
    else:
        _MIN_TRIG_BUCKET_DELTAS[bucket] = d
    return d


def get_min_triggers_bucket_delta(bucket: str) -> int:
    return int(_MIN_TRIG_BUCKET_DELTAS.get(bucket or "", 0))


def _distance_to_entry_zone(sig, price: float) -> float:
    if price <= 0:
        return 1.0
    if sig.entry_low <= price <= sig.entry_high:
        return 0.0
    nearest_boundary = min(abs(price - sig.entry_low), abs(price - sig.entry_high))
    return nearest_boundary / max(price, MIN_PRICE_DENOMINATOR)


def _resolve_check_interval(tracked: Dict[int, "TrackedExecution"]) -> int:
    for sig in tracked.values():
        if sig.state in (
            SignalState.ENTRY_ZONE,
            SignalState.ALMOST,
            SignalState.PREPARING,
            SignalState.WATCHING,
        ):
            return EXECUTION_FAST_CHECK_INTERVAL
        price = price_cache.get(sig.symbol)
        if price is not None and _distance_to_entry_zone(sig, price) <= NEAR_ENTRY_FAST_POLL_DISTANCE:
            return EXECUTION_FAST_CHECK_INTERVAL

    return EXECUTION_CHECK_INTERVAL


@dataclass
class TrackedExecution:
    signal_id: int
    symbol: str
    direction: str
    strategy: str
    entry_low: float
    entry_high: float
    stop_loss: float
    confidence: float
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: Optional[float] = None
    rr_ratio: float = 0.0
    state: SignalState = SignalState.APPROVED
    created_at: float = field(default_factory=time.time)
    watching_since: Optional[float] = None   # B1: when signal entered WATCHING
    almost_since: Optional[float] = None      # B1: when signal entered ALMOST

    has_rejection_candle: bool = False
    has_structure_shift: bool = False
    has_momentum_expansion: bool = False
    has_liquidity_reaction: bool = False

    # Gap 5 — per-trigger magnitude multipliers (populated by _check_triggers).
    # Each defaults to 1.0 (no scaling) and is clamped to [0.5, 2.0] by
    # core._exec_helpers.magnitude_multiplier.  A small wick on a thin alt
    # gets 0.5; a $5M whale sweep vs a $500K median gets 2.0.  The multiplier
    # is applied to the fixed nominal weight in ``triggers_met``.
    rejection_magnitude: float = 1.0
    structure_magnitude: float = 1.0
    momentum_magnitude:  float = 1.0
    liquidity_magnitude: float = 1.0

    # FIX STICKY-TRIGGERS: record when trigger accumulation window started.
    # Triggers older than TRIGGER_WINDOW_SECS are stale and must be reset —
    # a rejection candle from 2 hours ago + structure shift from now is NOT
    # confluence.  When the window expires without reaching min_triggers, all
    # flags reset and a fresh window begins. track() overwrites TRIGGER_WINDOW_SECS
    # per signal via _get_trigger_window_secs(setup_class).
    trigger_window_since: Optional[float] = None
    TRIGGER_WINDOW_SECS: int = 1200  # 20 minutes

    last_price: float = 0
    message_id: Optional[int] = None  # V10: Telegram message_id for threading
    grade: str = "B"                   # V11: alpha grade (A+/A/B+/B)
    min_triggers: int = 2              # V11: triggers needed (grade-dependent)
    setup_class: str = "intraday"      # FIX #31: scalp/intraday/swing/positional
    staleness_data: Optional[dict] = None  # BUG-13: conditions at signal creation time

    # Gap 7 — cached raw_data snapshot from the signal so the pre-fill
    # semantic-kill pass can evaluate CHoCH/BOS/Wyckoff/sweep fields that live
    # only in raw_data.  Optional; falls back to {} when absent.
    raw_data: Optional[dict] = None

    # Gap 5 — effective required triggers at the most recent _check_triggers
    # evaluation.  Starts at min_triggers and is raised by HTF-bear, low
    # confidence, and decay adjustments in the watch loop.  Used when
    # recording ALMOST→EXPIRED so tuner stats reflect the actual bar, not
    # the nominal floor.  None until first evaluation.
    last_required_triggers: Optional[int] = None

    # FIX #32: Trigger quality weights — structure_shift outweighs rejection_candle
    TRIGGER_WEIGHTS: dict = None  # set as class var below (dataclass limitation)

    @property
    def entry_mid(self):
        return (self.entry_low + self.entry_high) / 2

    @property
    def triggers_met(self) -> float:
        """
        STRATEGIST SCORE SYSTEM (from trade review session):
        Scores each confirmation type by quality:
          Structure shift     = +2  (break of structure — real demand/supply)
          Momentum candle     = +2  (volume + body expansion — real buyers/sellers)
          Liquidity reaction  = +1  (price swept liq — valid but weak alone)
          Rejection candle    = +1  (wick rejection — valid but weak alone)

        LONG entry requires ≥ 3 points  (min: structure+liquidity, or momentum alone)
        SHORT entry requires ≥ 3 points (same rule)
        LONG in HTF BEARISH week requires ≥ 4 points (set externally via min_triggers)

        Old weights: structure=1.5, liquidity=1.3, momentum=1.0, rejection=0.7
        → AAVE entered on liquidity_reaction ONLY (score=1.3) — too permissive
        New weights make liquidity alone (score=1.0) insufficient to trigger entry.
        """
        score = 0.0
        # Gap 5 — magnitude multiplier scales each trigger's nominal weight
        # by how decisively it fired relative to its own baseline.  A full
        # structural break on real volume counts 2.0; a marginal break on
        # thin volume counts 1.0 (multiplier clamped to floor 0.5 at the
        # magnitude source).  This prevents a threshold-grazing shift from
        # carrying the same weight as a decisive one.
        if self.has_structure_shift:
            score += 2.0 * float(self.structure_magnitude or 1.0)
        if self.has_momentum_expansion:
            score += 2.0 * float(self.momentum_magnitude or 1.0)
        if self.has_liquidity_reaction:
            score += 1.0 * float(self.liquidity_magnitude or 1.0)
        if self.has_rejection_candle:
            score += 1.0 * float(self.rejection_magnitude or 1.0)
        return score


class ExecutionEngine:

    def __init__(self):
        self._tracked: Dict[int, TrackedExecution] = {}
        self._running = False
        self._check_interval = Timing.EXECUTION_CHECK_INTERVAL_SECS
        self._task = None
        self.on_stage_change: Optional[Callable] = None
        self._stage_msg_sent: Dict[int, set] = {}  # V10: signal_id -> set of states already messaged

    @staticmethod
    def _validate_levels(direction, entry_low, entry_high, stop_loss,
                         tp1, tp2, tp3) -> Optional[str]:
        """Validate that stop-loss and TPs are on the correct side of the zone.

        Returns an error string if invalid, ``None`` otherwise. Catching this
        before track() stores the signal prevents a corrupted record from
        silently turning into a guaranteed loss once price moves.
        """
        try:
            entry_low = float(entry_low)
            entry_high = float(entry_high)
            stop_loss = float(stop_loss)
        except (TypeError, ValueError):
            return f"non-numeric levels (entry_low={entry_low}, entry_high={entry_high}, stop_loss={stop_loss})"
        if not (entry_low > 0 and entry_high > 0 and stop_loss > 0):
            return f"non-positive levels (entry_low={entry_low}, entry_high={entry_high}, stop_loss={stop_loss})"
        if entry_low > entry_high:
            return f"entry_low {entry_low} > entry_high {entry_high}"

        tps = []
        for label, value in (("tp1", tp1), ("tp2", tp2), ("tp3", tp3)):
            if value in (None, 0, 0.0):
                continue
            try:
                tps.append((label, float(value)))
            except (TypeError, ValueError):
                return f"non-numeric {label}={value}"

        if direction == "LONG":
            if stop_loss >= entry_low:
                return f"LONG stop {stop_loss} not below entry_low {entry_low}"
            for label, v in tps:
                if v <= entry_high:
                    return f"LONG {label} {v} not above entry_high {entry_high}"
            for (la, va), (lb, vb) in zip(tps, tps[1:]):
                if vb <= va:
                    return f"LONG {lb} {vb} not above {la} {va}"
        elif direction == "SHORT":
            if stop_loss <= entry_high:
                return f"SHORT stop {stop_loss} not above entry_high {entry_high}"
            for label, v in tps:
                if v >= entry_low:
                    return f"SHORT {label} {v} not below entry_low {entry_low}"
            for (la, va), (lb, vb) in zip(tps, tps[1:]):
                if vb >= va:
                    return f"SHORT {lb} {vb} not below {la} {va}"
        else:
            return f"unknown direction {direction!r}"
        return None

    def track(self, signal_id, symbol, direction, strategy,
              entry_low, entry_high, stop_loss, confidence,
              tp1=0.0, tp2=0.0, tp3=None, rr_ratio=0.0, message_id=None,
              grade: str = "B", staleness_data: Optional[dict] = None,
              setup_class: str = "intraday",
              raw_data: Optional[dict] = None):
        # Stop-loss / take-profit side sanity check: catches corrupted records
        # before they enter the tracker (e.g. SHORT with stop below entry would
        # silently lose the entire move once price moves the "wrong" way).
        _sides_err = self._validate_levels(
            direction, entry_low, entry_high, stop_loss, tp1, tp2, tp3
        )
        if _sides_err:
            logger.error(
                f"ExecutionEngine.track: rejecting signal #{signal_id} {symbol} "
                f"{direction}: {_sides_err}"
            )
            return
        # FIX #3 (AUDIT): setup_class parameter added so callers can pass the
        # signal's actual setup class (scalp/intraday/swing/positional) through
        # to TrackedExecution. Previously this was always the dataclass default
        # "intraday", causing wrong WATCHING timeouts and max-hold limits for
        # scalp and swing signals routed through execution_engine.
        # ── Strategy-specific trigger requirements ─────────────────────────────
        # Different strategies have fundamentally different entry mechanics.
        # A universal "2 triggers" rule is wrong — it causes slippage on
        # momentum/breakout strategies and redundant waiting on pattern strategies.
        #
        # 0 triggers: enter as soon as price is in zone (momentum/breakout/limit-based)
        # 1 trigger:  one confirmation (most setups — a single 5m confirmation is enough)
        # 2 triggers: two confirmations (B/B+ grade signals need extra validation)
        # STRATEGIST ENTRY RULES:
        # "LONG (Mean Reversion): must have Structure shift + Momentum candle (not just reaction)"
        # "SHORT (Trend/Wyckoff): current logic is GOOD — keep unchanged"
        # With new weights (structure=2, momentum=2, liquidity=1, rejection=1):
        #   Score ≥ 3 = minimum entry (structure+liquidity OR momentum+liquidity OR momentum+rejection)
        #   Score ≥ 4 = quality entry (structure+momentum = both confirm)
        #   Score = 1 = liquidity reaction alone → BLOCKED (was the AAVE problem)
        _STRATEGY_MIN_TRIGGERS = {
            # Enter immediately (score=0 means enter on zone touch)
            "InstitutionalBreakout": 0,    # Breakout IS the signal — no wait
            "Momentum":              0,    # EMA alignment bars = entry
            "MomentumContinuation":  0,    # Legacy alias
            "RangeScalper":          0,    # Boundary touch is the entry
            # Mean reversion / funding squeeze: need ≥ 3 score
            # Strategist: "Must have structure shift + momentum candle, NOT just reaction"
            # Score 3 requires either: structure(2)+liquidity(1) OR momentum(2)+liquidity(1)
            # Score 1 (liquidity alone) is explicitly blocked
            "FundingRateArb":        3,    # STRATEGIST: structure+liq OR momentum+liq minimum
            "MeanReversion":         3,    # Same — Z-score extreme alone not enough
            "ExtremeReversal":       3,    # Same — RSI extreme alone not enough
            # Trend continuation / structure: need ≥ 3 score (already working well)
            "PriceAction":           3,    # Strategist confirmed this needs structure
            "Ichimoku":              3,    # Cloud cross + structure confirmation
            "IchimokuCloud":         3,    # Legacy alias
            "SmartMoneyConcepts":    3,    # OB tap + structure + volume ideally
            # Wave / event strategies: need ≥ 3 score
            "ElliottWave":           3,    # Momentum into Fib zone
            "WyckoffAccDist":        3,    # Volume + structure shift confirms Spring/UTAD
            "Wyckoff":               3,    # Legacy alias
            "HarmonicPattern":       3,    # Pattern completion confirmation
            "GeometricPattern":      3,    # Same
        }
        _strategy_trig = _STRATEGY_MIN_TRIGGERS.get(strategy, None)

        # Grade still acts as a floor — a B+ signal always needs at least what the
        # grade requires, even if strategy says 0.
        _grade_trig = {"A+": 0, "A": 1, "B+": 1, "B": 2, "C": 3}.get(grade, 2)

        if _strategy_trig is not None:
            # Use whichever is higher: strategy requirement or grade floor
            # Exception: if grade is A+ (confident setup), trust the strategy minimum
            if grade == "A+":
                _min_trig = _strategy_trig
            else:
                _min_trig = max(_strategy_trig, _grade_trig)
        else:
            _min_trig = _grade_trig

        # FIX: Symmetric regime-aware trigger adjustment.
        # Previously we only *ratcheted up* for counter-trend (AAVE / KITE losses
        # at 1.3–2.2/1 score in BEAR_TREND).  The inverse is equally valid: a
        # LONG in a confirmed BULL_TREND with weekly ADX≥25 is buying dips in a
        # trending market, and should be allowed to fire on one fewer trigger
        # (floored at 1).  core/_exec_helpers.trigger_adjustment_for_regime
        # encapsulates both sides so the rule is testable and symmetric.
        try:
            from analyzers.regime import regime_analyzer as _ee_ra
            from analyzers.htf_guardrail import htf_guardrail as _ee_htf
            from core._exec_helpers import trigger_adjustment_for_regime as _ee_adj
            _ee_regime = _ee_ra.regime.value
            _adj_min, _adj_note = _ee_adj(
                direction=direction,
                regime=_ee_regime,
                weekly_bias=_ee_htf._weekly_bias,
                weekly_adx=_ee_htf._weekly_adx,
                base_min_triggers=_min_trig,
            )
            if _adj_note:
                _min_trig = _adj_min
                logger.info(
                    f"ExecutionEngine: #{signal_id} {symbol} {direction} "
                    f"min_triggers adjusted — {_adj_note}"
                )
        except Exception:
            pass

        # Gap 6 feedback application — if the param-tuner has accepted a
        # LOW-risk proposal for this (strategy|regime|setup_class) bucket,
        # apply the clamped delta on top.  Floored at 1 so a bucket can never
        # fire on 0 triggers (that's still reserved for immediate-entry
        # strategies, which already have _min_trig=0 from the static table —
        # and we intentionally skip the delta when _min_trig == 0 to preserve
        # that intent).
        try:
            from analyzers.regime import regime_analyzer as _ee_ra2
            _bkt_regime = getattr(_ee_ra2.regime, 'value', 'UNKNOWN')
            _bkt_key = f"{strategy}|{_bkt_regime}|{setup_class or 'intraday'}"
            _bkt_delta = get_min_triggers_bucket_delta(_bkt_key)
            if _bkt_delta != 0 and _min_trig > 0:
                _pre = _min_trig
                _min_trig = max(1, int(_min_trig) + int(_bkt_delta))
                if _min_trig != _pre:
                    logger.info(
                        f"ExecutionEngine: #{signal_id} {symbol} min_triggers "
                        f"tuned {_pre}→{_min_trig} for bucket {_bkt_key} "
                        f"(delta={_bkt_delta:+d})"
                    )
        except Exception:
            pass

        # BUG-13: Snapshot baseline conditions at creation time for pre-fill staleness check.
        # When price finally reaches the zone hours later, we compare current conditions
        # against these baselines to detect if the market has genuinely reversed.
        if staleness_data is None:
            try:
                from analyzers.regime import regime_analyzer
                from analyzers.htf_guardrail import htf_guardrail as _htf
                staleness_data = {
                    'regime': getattr(regime_analyzer.regime, 'value', 'UNKNOWN'),
                    'chop': regime_analyzer.chop_strength,
                    'weekly_bias': getattr(_htf, '_weekly_bias', 'NEUTRAL'),
                    'fear_greed': regime_analyzer.fear_greed,
                    'created_price': price_cache.get(symbol) or ((entry_low + entry_high) / 2),
                }
            except Exception:
                staleness_data = {}

        self._tracked[signal_id] = TrackedExecution(
            signal_id=signal_id, symbol=symbol, direction=direction,
            strategy=strategy, entry_low=entry_low, entry_high=entry_high,
            stop_loss=stop_loss, confidence=confidence,
            tp1=tp1, tp2=tp2, tp3=tp3, rr_ratio=rr_ratio,
            message_id=message_id,
            grade=grade, min_triggers=_min_trig,
            staleness_data=staleness_data,
            setup_class=setup_class,
            raw_data=(dict(raw_data) if isinstance(raw_data, dict) else None),
        )
        self._tracked[signal_id].TRIGGER_WINDOW_SECS = _get_trigger_window_secs(setup_class)
        price_cache.subscribe(symbol)
        # Trade-tape marker (Apr 2026): distinguish TRACKED from PUBLISHED
        # so the tape shows the handoff from publisher to execution engine.
        try:
            if _tl:
                _tl.tracked(
                    signal_id=signal_id, symbol=symbol, direction=direction,
                    grade=grade, strategy=strategy,
                    min_triggers=float(_min_trig),
                    setup_class=setup_class,
                )
        except Exception:
            pass
        # Persist (best-effort, fire-and-forget) so a crash/restart shortly after
        # track() doesn't lose this signal. Errors are surfaced via the
        # _safe_ensure_future done-callback rather than swallowed.
        try:
            from data.database import db as _ee_db
            _safe_ensure_future(
                _ee_db.save_tracked_signal(
                    signal_id=signal_id, symbol=symbol, direction=direction,
                    strategy=strategy, state="APPROVED",
                    entry_low=entry_low, entry_high=entry_high,
                    stop_loss=stop_loss, confidence=confidence,
                    grade=grade, message_id=message_id,
                    tp1=tp1, tp2=tp2, tp3=tp3, rr_ratio=rr_ratio,
                    min_triggers=_min_trig,
                    setup_class=setup_class,
                ),
                context=f"save_tracked APPROVED #{signal_id}",
            )
        except Exception as _persist_err:
            logger.warning(
                f"ExecutionEngine.track: scheduling persistence failed for "
                f"#{signal_id} {symbol}: {_persist_err}"
            )

    def invalidate_regime_cache(self):
        """FIX L2: Called by engine when regime changes to clear stale regime stats."""
        self._regime_cache_invalidated = True

    def untrack(self, signal_id: int):
        """Remove a signal from execution tracking and release price cache subscription."""
        sig = self._tracked.pop(signal_id, None)
        if sig:
            # Only unsubscribe if no other tracked signal needs same symbol
            still_needed = any(s.symbol == sig.symbol for s in self._tracked.values())
            if not still_needed:
                price_cache.unsubscribe(sig.symbol)
            logger.debug(f"ExecutionEngine: untracked #{signal_id} {sig.symbol}")
            try:
                from data.database import db as _ee_db
                _safe_ensure_future(
                    _ee_db.delete_tracked_signal(signal_id),
                    context=f"delete_tracked #{signal_id}",
                )
            except Exception as _persist_err:
                logger.warning(
                    f"ExecutionEngine.untrack: scheduling DB delete failed for "
                    f"#{signal_id}: {_persist_err}"
                )

    async def restore(self) -> None:
        """Reload pre-fill signals from tracked_signals_v1 on startup.

        Recovers signals that were in APPROVED/PREPARING/WATCHING/ALMOST/ENTRY_ZONE
        state when the bot last stopped, including any trigger flags that had
        already been set.  Called before start() so the loop resumes immediately.
        """
        _pre_fill_states = [
            "APPROVED", "PREPARING", "WATCHING", "ALMOST", "ENTRY_ZONE",
            "ARMED",  # legacy: DB may still contain old state name
        ]
        try:
            from data.database import db as _ee_db
            rows = await _ee_db.load_tracked_signals(states=_pre_fill_states)
        except Exception as _re:
            logger.warning(f"ExecutionEngine.restore: DB load failed (non-fatal): {_re}")
            return

        restored = 0
        for row in rows:
            sid = row['signal_id']
            if sid in self._tracked:
                continue  # already present (e.g. added by engine recovery logic)
            try:
                te = TrackedExecution(
                    signal_id=sid,
                    symbol=row['symbol'],
                    direction=row['direction'],
                    strategy=row['strategy'],
                    entry_low=row['entry_low'],
                    entry_high=row['entry_high'],
                    stop_loss=row['stop_loss'],
                    confidence=row['confidence'],
                    tp1=row.get('tp1', 0.0),
                    tp2=row.get('tp2', 0.0),
                    tp3=row.get('tp3'),
                    rr_ratio=row.get('rr_ratio', 0.0),
                    message_id=row.get('message_id'),
                    grade=row.get('grade', 'B'),
                    min_triggers=row.get('min_triggers', 2),
                    setup_class=row.get('setup_class', 'intraday'),
                    has_rejection_candle=bool(row.get('has_rejection_candle', 0)),
                    has_structure_shift=bool(row.get('has_structure_shift', 0)),
                    has_momentum_expansion=bool(row.get('has_momentum_expansion', 0)),
                    has_liquidity_reaction=bool(row.get('has_liquidity_reaction', 0)),
                    created_at=row.get('created_at', time.time()),
                )
                # Restore the state enum
                try:
                    # Migration: map legacy "ARMED" DB values to "ALMOST"
                    _raw_state = row['state']
                    if _raw_state == "ARMED":
                        _raw_state = "ALMOST"
                    te.state = SignalState(_raw_state)
                except ValueError:
                    te.state = SignalState.WATCHING  # safe fallback
                # Restore timing fields so timeout logic is correct.
                # Use the timestamp of the most recent persisted state transition
                # when available. save_tracked_signal() updates updated_at on every
                # state change, so this is the best proxy for "when did we enter
                # WATCHING/ALMOST/ENTRY_ZONE" after a restart.
                _state_changed_at = row.get('updated_at') or te.created_at
                if te.state in (SignalState.WATCHING, SignalState.ENTRY_ZONE):
                    te.watching_since = _state_changed_at
                elif te.state == SignalState.ALMOST:
                    te.almost_since = _state_changed_at

                self._tracked[sid] = te
                price_cache.subscribe(te.symbol)
                restored += 1
            except Exception as _row_err:
                logger.debug(f"ExecutionEngine.restore: skipped row #{sid}: {_row_err}")

        if restored:
            logger.info(f"🔄 ExecutionEngine restored {restored} pre-fill signal(s) from DB")

    def start(self):
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        """Clean shutdown"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_status_summary(self) -> str:
        """Return a brief status string for heartbeat logging"""
        total = len(self._tracked)
        executing = sum(1 for s in self._tracked.values() if s.state == SignalState.EXECUTE)
        watching = sum(1 for s in self._tracked.values() if s.state in (SignalState.WATCHING, SignalState.ENTRY_ZONE, SignalState.ALMOST))
        return f"{total} tracked | {executing} execute | {watching} watching"

    async def _loop(self):
        _regime_stats_cache = {}
        _regime_stats_last = 0.0
        _REGIME_CACHE_TTL = 300  # Fix E5: only recompute every 5min, not every 15s
        self._regime_cache_invalidated = False  # FIX L2

        while self._running:
            now = time.time()

            # Fix E5: cache regime stats, don't read the whole file every cycle
            # FIX L2: also refresh if regime change was detected externally
            if now - _regime_stats_last > _REGIME_CACHE_TTL or self._regime_cache_invalidated:
                self._regime_cache_invalidated = False
                try:
                    _regime_stats_cache = compute_regime_stats()
                    _regime_stats_last = now
                    if _regime_stats_cache:
                        logger.debug(f"[REGIME_STATS] {_regime_stats_cache}")
                except Exception:
                    pass

            await asyncio.sleep(self._check_interval)
            self._check_interval = _resolve_check_interval(self._tracked)

            for sig in list(self._tracked.values()):
                await self._check(sig, _regime_stats_cache)

    async def _check(self, sig: TrackedExecution, regime_stats: dict = None):
        """Fix E3: market_state computed once here and passed down to _check_triggers"""
        # BUG-B FIX: `now` must be defined at the top of _check.
        # The B1 timeout logic (WATCHING/ALMOST expiry) uses `now` but the original
        # code only defined `now` in `_loop`, not here. This caused NameError on
        # every signal that entered WATCHING or ALMOST state.
        now = time.time()
        try:
            from signals.signal_decay import apply_decay
            decayed_confidence = apply_decay(sig.confidence, sig.created_at, now)
        except Exception:
            decayed_confidence = sig.confidence

        price = price_cache.get(sig.symbol)
        if price is None:
            return  # Cache miss — skip this cycle, try next

        sig.last_price = price

        old_state = sig.state

        try:
            from analyzers.btc_news_intelligence import btc_news_intelligence
            _btc_block_reason = btc_news_intelligence.get_signal_block_reason(sig.direction)
            if _btc_block_reason:
                sig.state = SignalState.INVALIDATED
                logger.info(
                    f"ExecutionEngine: #{sig.signal_id} {sig.symbol} invalidated by BTC news "
                    f"block — {_btc_block_reason}"
                )
                # Fall through (no early return) so the terminal-state
                # pathway at the end of _check fires on_stage_change and
                # pops the signal out of self._tracked.  An early return
                # here would leave the INVALIDATED signal sitting in the
                # tracked map and re-checked forever.
        except Exception:
            pass

        # Guard against a corrupted signal record where entry_mid is zero, negative,
        # NaN, or Inf — which would cause ZeroDivisionError or NaN propagation.
        if not (sig.entry_mid > 0):  # catches NaN, Inf, 0, and negatives
            logger.error(
                f"ExecutionEngine: signal #{sig.signal_id} {sig.symbol} has "
                f"entry_mid={sig.entry_mid} — corrupted record, invalidating"
            )
            sig.state = SignalState.INVALIDATED
            return

        distance = abs(price - sig.entry_mid) / sig.entry_mid

        in_zone = sig.entry_low <= price <= sig.entry_high

        # Fix E3: build market_state once for this check cycle (not twice)
        profile = {}
        market_state = {}
        try:
            market_state = await build_market_state(sig.symbol, sig.direction)
            profile = build_market_profile(market_state) if market_state else {}
        except Exception:
            pass

        # Fix E4: compute confidence_score and risk_adj BEFORE the state machine uses them
        try:
            confidence_score = compute_confidence(market_state, profile)
            profile["confidence_score"] = confidence_score
            sig._confidence_score = confidence_score  # expose for logging
        except Exception:
            confidence_score = 50

        try:
            risk_adj = get_risk_adjustment(profile.get("regime"))
            profile["risk_multiplier"] = profile.get("risk_multiplier", 1.0) * risk_adj
            sig._risk_adj = risk_adj  # expose for logging
        except Exception:
            risk_adj = 1.0

        # --- STATE MACHINE ---
        # B1: Expire signals that have been WATCHING or ALMOST too long without firing
        if sig.state == SignalState.WATCHING:
            if sig.watching_since is None:
                sig.watching_since = now
            elif now - sig.watching_since > _get_watching_timeout(sig):
                sig.state = SignalState.EXPIRED
                _timeout_h = _get_watching_timeout(sig) / 3600
                logger.info(
                    f"ExecutionEngine: #{sig.signal_id} {sig.symbol} EXPIRED "
                    f"(in WATCHING for {(now - sig.watching_since)/3600:.1f}h, "
                    f"timeout={_timeout_h:.1f}h setup_class={getattr(sig,'setup_class','?')})"
                )
        if sig.state == SignalState.ALMOST:
            if sig.almost_since is None:
                sig.almost_since = now
            elif now - sig.almost_since > EXECUTION_ALMOST_TIMEOUT:
                sig.state = SignalState.EXPIRED
                logger.info(
                    f"ExecutionEngine: #{sig.signal_id} {sig.symbol} EXPIRED "
                    f"(in ALMOST for {(now - sig.almost_since)/3600:.1f}h)"
                )
                # Gap 6: feed ALMOST→EXPIRED into the diagnostic / param tuner.
                # Chronic "zone reached, triggers never confluenced" means the
                # trigger threshold for this (strategy, regime, setup_class)
                # bucket may be too strict — the tuner accumulates these events
                # and can propose lowering require_triggers for that bucket.
                try:
                    from core.diagnostic_engine import diagnostic_engine as _de
                    from analyzers.regime import regime_analyzer as _ra
                    _regime = getattr(_ra.regime, 'value', 'UNKNOWN')
                    _de.record_almost_expired(
                        strategy=sig.strategy,
                        regime=_regime,
                        setup_class=getattr(sig, 'setup_class', 'intraday'),
                        score=sig.triggers_met,
                        min_triggers=(
                            sig.last_required_triggers
                            if sig.last_required_triggers is not None
                            else sig.min_triggers
                        ),
                    )
                except Exception:
                    pass

        # Gap 3: expire signals whose live RR has decayed below 1.0 after
        # SL tightening.  We only fire this in pre-fill states because once the
        # signal has fired we're in the outcome monitor's domain.
        if (
            sig.state in (SignalState.APPROVED, SignalState.PREPARING,
                          SignalState.WATCHING, SignalState.ENTRY_ZONE,
                          SignalState.ALMOST)
            and getattr(sig, '_rr_decay_flag', False)
        ):
            _live_rr = getattr(sig, '_live_rr', 0.0)
            sig.state = SignalState.EXPIRED
            logger.info(
                f"ExecutionEngine: #{sig.signal_id} {sig.symbol} EXPIRED "
                f"(RR_DECAY live_rr={_live_rr:.2f} < 1.0 after SL tightening)"
            )

        if (
            decayed_confidence <= 50.0
            and sig.state in (SignalState.APPROVED, SignalState.PREPARING, SignalState.WATCHING)
            and now - sig.created_at > 900
        ):
            sig.state = SignalState.EXPIRED
            logger.info(
                f"ExecutionEngine: #{sig.signal_id} {sig.symbol} EXPIRED "
                f"(signal decay {sig.confidence:.1f}→{decayed_confidence:.1f})"
            )

        # Extended-zone kill: price blew past the entry zone boundary without ever
        # entering it (e.g. LONG waiting for pullback but price ran +2% above entry_high).
        # The optimal entry window is gone — further holding risks a bad fill.
        if sig.state in (SignalState.APPROVED, SignalState.PREPARING, SignalState.WATCHING):
            _ext_pct = 0.0
            if sig.direction == "LONG" and price > sig.entry_high:
                _ext_pct = (price - sig.entry_high) / sig.entry_high * 100
            elif sig.direction == "SHORT" and price < sig.entry_low:
                _ext_pct = (sig.entry_low - price) / sig.entry_low * 100
            if _ext_pct > 2.0:
                sig.state = SignalState.INVALIDATED
                logger.info(
                    f"Extended kill: #{sig.signal_id} {sig.symbol} {sig.direction} "
                    f"price {price:.6g} is {_ext_pct:.1f}% beyond entry zone — invalidated"
                )

        if sig.state == SignalState.APPROVED and distance < 0.02:
            sig.state = SignalState.PREPARING

        elif sig.state == SignalState.PREPARING and distance < 0.01:
            sig.state = SignalState.WATCHING
            sig.watching_since = now  # B1: record when we entered WATCHING

        elif sig.state == SignalState.WATCHING and in_zone:
            sig.state = SignalState.ENTRY_ZONE
            # FIX 3B: Record that price reached the entry zone.
            # Enables AI to distinguish "zone never reached" (execution miss)
            # from "zone reached but trigger failed" (trigger quality issue).
            try:
                from data.database import db
                _safe_ensure_future(
                    db.write_zone_reached(sig.signal_id),
                    context=f"write_zone_reached #{sig.signal_id}",
                )
            except Exception as _zr_err:
                logger.warning(
                    f"ExecutionEngine: scheduling write_zone_reached failed for "
                    f"#{sig.signal_id}: {_zr_err}"
                )

        elif sig.state in (SignalState.ENTRY_ZONE, SignalState.ALMOST):
            await self._check_triggers(sig, price, market_state)  # Fix E3: pass pre-computed state

            # V11: Grade-aware trigger requirements (replaces hardcoded 2)
            _required_triggers = sig.min_triggers

            # STRATEGIST FIX B: Dynamic timing filter
            # "If HTF = bearish: LONG requires ≥ 4 score"
            # Prevents AAVE-style entries on single liquidity reaction (score=1.3 old / 1.0 new)
            try:
                from analyzers.htf_guardrail import htf_guardrail as _ee_htf2
                _htf_bear_long = (
                    sig.direction == "LONG" and
                    _ee_htf2._weekly_bias == "BEARISH" and
                    _ee_htf2._weekly_adx >= 25
                )
                if _htf_bear_long:
                    _required_triggers = max(_required_triggers, 4)  # Need structure+momentum BOTH
            except Exception:
                pass

            # Low-confidence upgrade: need 1 extra trigger if confidence is poor.
            # compute_confidence() returns 0-100, not 0-1.
            if confidence_score < 40 and _required_triggers < 3:
                _required_triggers += 1
            if decayed_confidence <= sig.confidence - DECAY_TRIGGER_INCREMENT_THRESHOLD and _required_triggers < 4:
                _required_triggers += 1

            # Gap 5 — remember the effective (post-HTF / post-decay) trigger
            # bar so ALMOST→EXPIRED feedback doesn't understate what was
            # actually required at the moment of evaluation.
            sig.last_required_triggers = _required_triggers

            if sig.triggers_met >= _required_triggers:
                # BUG-13: Pre-fill staleness validation.
                # The signal was analysed hours ago. Before executing, verify
                # the market hasn't genuinely reversed while waiting for the pullback.
                # Three checks in order of severity:
                _stale_reason = await self._check_staleness(sig, price)
                if _stale_reason:
                    sig.state = SignalState.INVALIDATED
                    logger.info(
                        f"🚫 Staleness block | #{sig.signal_id} {sig.symbol} {sig.direction} "
                        f"| Triggers met but setup invalidated: {_stale_reason}"
                    )
                    if self.on_stage_change:
                        _sent_set = self._stage_msg_sent.setdefault(sig.signal_id, set())
                        if "STALENESS_INVALIDATED" not in _sent_set:
                            _sent_set.add("STALENESS_INVALIDATED")
                            # Notify via existing stage_change callback so Telegram gets updated
                            await self.on_stage_change(sig, SignalState.ALMOST, SignalState.INVALIDATED)
                else:
                    sig.state = SignalState.EXECUTE

                # --- Execution Journal Logging ---
                # FIX: only log if we actually reached EXECUTE state (not INVALIDATED)
                if sig.state == SignalState.EXECUTE:
                    try:
                        journal_entry = {
                            "symbol": sig.symbol,
                            "direction": sig.direction,
                            "state": "EXECUTE",
                            "triggers_met": sig.triggers_met,
                            "regime": profile.get("regime"),
                            "confirmation_required": profile.get("confirmation_required"),
                            "risk_multiplier": profile.get("risk_multiplier"),
                            "last_price": price,
                        }
                        log_execution(journal_entry)
                    except Exception:
                        pass

            elif sig.triggers_met >= 1:
                _was_already_almost = sig.state == SignalState.ALMOST
                sig.state = SignalState.ALMOST
                if sig.almost_since is None:
                    sig.almost_since = now  # B1: record when almost ready
                if _tl and not _was_already_almost:
                    try:
                        _tl.trigger(signal_id=sig.signal_id, symbol=sig.symbol, direction=sig.direction, structure_shift=sig.has_structure_shift, momentum=sig.has_momentum_expansion, liquidity=sig.has_liquidity_reaction, rejection=sig.has_rejection_candle, triggers_score=sig.triggers_met, min_triggers=sig.min_triggers, new_state="ALMOST")
                    except Exception:
                        pass

        try:
            regime = profile.get("regime", "UNKNOWN")
            logger.debug(
                f"[EXEC_ENGINE] {sig.symbol} | State={sig.state.value} "
                f"| Triggers={sig.triggers_met} | Conf={confidence_score:.2f} "
                f"| RiskAdj={risk_adj:.2f} | Regime={regime}"
            )
        except Exception:
            pass
    
        if sig.state != old_state:
            # Persist exec_state to DB so dashboard reflects correct state after restart
            try:
                from data.database import db as _exec_db
                _safe_ensure_future(
                    _exec_db.update_signal_exec_state(sig.signal_id, sig.state.value),
                    context=f"exec_state {sig.state.value} #{sig.signal_id}",
                )
                # Also persist full signal state to tracked_signals_v1 so trigger
                # flags and state survive restarts (pre-fill states only — post-fill
                # states are managed by OutcomeMonitor).
                _non_terminal = sig.state not in (
                    SignalState.EXPIRED, SignalState.INVALIDATED,
                    SignalState.FILLED, SignalState.EXECUTE,
                )
                if _non_terminal:
                    _safe_ensure_future(
                        _exec_db.save_tracked_signal(
                            signal_id=sig.signal_id,
                            symbol=sig.symbol,
                            direction=sig.direction,
                            strategy=sig.strategy,
                            state=sig.state.value,
                            entry_low=sig.entry_low,
                            entry_high=sig.entry_high,
                            stop_loss=sig.stop_loss,
                            confidence=sig.confidence,
                            grade=sig.grade,
                            message_id=sig.message_id,
                            created_at=sig.created_at,
                            tp1=sig.tp1,
                            tp2=sig.tp2,
                            tp3=sig.tp3,
                            rr_ratio=sig.rr_ratio,
                            min_triggers=sig.min_triggers,
                            setup_class=sig.setup_class,
                            has_rejection_candle=sig.has_rejection_candle,
                            has_structure_shift=sig.has_structure_shift,
                            has_momentum_expansion=sig.has_momentum_expansion,
                            has_liquidity_reaction=sig.has_liquidity_reaction,
                        ),
                        context=f"save_tracked {sig.state.value} #{sig.signal_id}",
                    )
            except Exception:
                pass
            if self.on_stage_change:
                # V10: Dedup — only send each state transition message once
                _sent_set = self._stage_msg_sent.setdefault(sig.signal_id, set())
                if sig.state.value not in _sent_set:
                    _sent_set.add(sig.state.value)
                    await self.on_stage_change(sig, old_state, sig.state)

        # FIX L1 / BUG-D FIX: Auto-remove from _tracked when signal reaches terminal state.
        # BUG-D: EXECUTE and FILLED were excluded from _tracked.pop(), causing them to
        # loop forever in the check cycle, wasting API calls indefinitely after every
        # confirmed execution. All four terminal states must be fully cleaned up.
        # B10: Also clean up _stage_msg_sent to prevent unbounded memory growth.
        if sig.state in (SignalState.EXPIRED, SignalState.INVALIDATED, SignalState.FILLED, SignalState.EXECUTE):
            self._tracked.pop(sig.signal_id, None)      # BUG-D: was only removing EXPIRED/INVALIDATED
            self._stage_msg_sent.pop(sig.signal_id, None)  # B10: cleanup dedup dict
            logger.debug(f"ExecutionEngine: auto-removed #{sig.signal_id} ({sig.state.value})")
            # Remove from tracked_signals_v1 — OutcomeMonitor will re-insert if it
            # transitions to ACTIVE, so no gap in coverage.
            try:
                from data.database import db as _exec_db_term
                _safe_ensure_future(
                    _exec_db_term.delete_tracked_signal(sig.signal_id),
                    context=f"delete_tracked terminal #{sig.signal_id}",
                )
            except Exception:
                pass

    async def _check_staleness(self, sig: TrackedExecution, price: float) -> Optional[str]:
        """
        BUG-13: Pre-fill staleness validation.

        Called when all entry triggers are met but the signal was created hours ago.
        Returns a reason string if the setup has been invalidated by market changes,
        or None if the setup is still valid and execution should proceed.

        Three checks in order of impact:

        1. REGIME FLIP: If the regime at fill time has flipped against the direction
           (e.g. was BULL_TREND at signal creation, now BEAR_TREND at fill time),
           the structural basis for the signal has changed fundamentally.

        2. VOLUME CHARACTER: A healthy retracement to the entry zone has DECLINING
           volume as price approaches (sellers exhausting). A reversal has EXPANDING
           volume on the approach (new sellers entering). Check the last 3 bars on 5m.

        3. CHOCH SINCE SIGNAL: If a Change of Character has formed on the 1h chart
           since the signal was created, the market structure has flipped. A CHoCH
           on the timeframe the signal was analysed on invalidates the entire setup.
        """
        try:
            baseline = sig.staleness_data or {}
            signal_age_hours = (time.time() - sig.created_at) / 3600
            ohlcv_1h = None

            # ── Gap 4: re-run semantic kills against live context ──────
            # setup_context on the signal was built at creation.  By fill time,
            # a CHoCH may have formed, the weekly trend may have flipped, or
            # Wyckoff phase may have rolled — all things _check_semantic_kills
            # already knows how to detect, and all things that should block an
            # execution regardless of microstructure triggers lining up.
            # This is the second call of the same helper (aggregator already
            # runs it at publish time), this time with a freshly rebuilt
            # setup_context so the semantic read is current.
            try:
                from analyzers.execution_gate import ExecutionQualityGate
                from signals.context_contracts import build_setup_context as _bsc

                class _SignalContextShim:
                    """Minimal SignalResult-shaped adapter for build_setup_context().

                    build_setup_context reads a handful of attributes off the
                    signal object to construct the structure/pattern/location
                    dicts.  We rebuild it here from the live TrackedExecution
                    so Wyckoff-phase flips, CHoCH formations, and trend
                    reversals since publication are visible to the semantic
                    kill pass.
                    """
                    symbol = sig.symbol
                    direction = sig.direction
                    strategy = sig.strategy
                    entry_low = sig.entry_low
                    entry_high = sig.entry_high
                    stop_loss = sig.stop_loss
                    tp1 = sig.tp1
                    tp2 = sig.tp2
                    tp3 = sig.tp3
                    confidence = sig.confidence
                    setup_class = getattr(sig, 'setup_class', 'intraday')
                    raw_data = sig.raw_data if isinstance(sig.raw_data, dict) else {}
                _live_ctx = _bsc(_SignalContextShim())
                _sem_reason = ExecutionQualityGate._check_semantic_kills(
                    direction=sig.direction,
                    setup_context=_live_ctx,
                    execution_context=None,
                )
                if _sem_reason:
                    return f"PRE_FILL_SEMANTIC_KILL: {_sem_reason[:100]}"
            except Exception as _sk_err:
                logger.debug(
                    f"Pre-fill semantic kill check skipped (non-fatal): {_sk_err}"
                )

            # ── Check 0: local-range drift before execute ──────────────
            # A setup can be valid at publish time but become a bad live trade if
            # price runs into the wrong side of a tight local range before triggers complete.
            # This is the "don't buy premium / sell discount after the move already happened"
            # guard, independent of the optional LLM layer.
            try:
                ohlcv_1h = await api.fetch_ohlcv(sig.symbol, "1h", limit=50)
                _late_reason = _late_local_range_reason(sig, price, ohlcv_1h)
                if _late_reason:
                    return _late_reason
            except Exception:
                pass

            # Only run staleness checks if signal is old enough to have meaningful drift
            # (less than 1 hour old = regime/structure haven't had time to change)
            if signal_age_hours < 1.0:
                return None

            # ── Check 1: Regime flip ──────────────────────────────────
            try:
                from analyzers.regime import regime_analyzer
                current_regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                original_regime = baseline.get('regime', current_regime)
                current_chop = regime_analyzer.chop_strength

                # A flip from trending to CHOPPY or opposite trend is a structural change
                _was_trending = original_regime in ("BULL_TREND", "BEAR_TREND")
                _now_trending = current_regime in ("BULL_TREND", "BEAR_TREND")
                _trend_flipped = (
                    _was_trending and _now_trending and original_regime != current_regime
                )
                _went_risk_off = current_regime == "VOLATILE" and original_regime != "VOLATILE"

                if _trend_flipped:
                    return (
                        f"Regime flipped {original_regime}→{current_regime} since signal "
                        f"({signal_age_hours:.1f}h ago) — structural basis changed"
                    )
                if _went_risk_off:
                    return (
                        f"Market entered VOLATILE since signal ({signal_age_hours:.1f}h ago) "
                        f"— panic conditions, not a healthy retracement"
                    )

                # Chop surge: went from low chop (trending) to high chop (range)
                original_chop = baseline.get('chop', 0.5)
                if original_chop < 0.25 and current_chop > 0.55:
                    return (
                        f"Chop surge {original_chop:.2f}→{current_chop:.2f} since signal "
                        f"— trending setup, now ranging market"
                    )
            except Exception:
                pass

            # ── Check 2: Volume character on approach ────────────────
            # Contracting volume on pullback = healthy retracement (sellers exhausting)
            # Expanding volume on pullback = reversal (new sellers entering aggressively)
            try:
                import numpy as np
                ohlcv_5m = await api.fetch_ohlcv(sig.symbol, "5m", limit=12)
                if ohlcv_5m and len(ohlcv_5m) >= 6:
                    closes_5m = np.array([float(b[4]) for b in ohlcv_5m])
                    vols_5m   = np.array([float(b[5]) for b in ohlcv_5m])

                    # Identify last 4 bars where price was moving toward the zone
                    # (falling for LONG entry, rising for SHORT entry)
                    _approach_bars = []
                    for i in range(-4, 0):
                        _moving_toward_zone = (
                            (sig.direction == "LONG" and closes_5m[i] < closes_5m[i-1]) or
                            (sig.direction == "SHORT" and closes_5m[i] > closes_5m[i-1])
                        )
                        if _moving_toward_zone:
                            _approach_bars.append(vols_5m[i])

                    if len(_approach_bars) >= 3:
                        avg_vol_recent = np.mean(vols_5m[-12:])
                        avg_approach_vol = np.mean(_approach_bars)
                        # Expanding volume: approach volume > 1.5× recent average
                        # and each approach bar is larger than the last
                        vol_expanding = avg_approach_vol > avg_vol_recent * 1.5
                        vol_accelerating = (
                            len(_approach_bars) >= 3 and
                            _approach_bars[-1] > _approach_bars[-2] > _approach_bars[-3]
                        )

                        if vol_expanding and vol_accelerating:
                            return (
                                f"Reversal volume pattern on approach: "
                                f"avg approach vol {avg_approach_vol:.0f} > 1.5× recent avg {avg_vol_recent:.0f}, "
                                f"accelerating — new sellers entering, not exhaustion"
                            )
            except Exception:
                pass

            # ── Check 3: CHoCH on 1h since signal was created ────────
            # A CHoCH (Change of Character) means the trend direction has flipped
            # on the analysis timeframe. If a LONG setup had bullish structure,
            # and a bearish CHoCH has formed since then, the setup is void.
            try:
                if ohlcv_1h is None:
                    ohlcv_1h = await api.fetch_ohlcv(sig.symbol, "1h", limit=20)
                if ohlcv_1h and len(ohlcv_1h) >= 6:
                    import numpy as np
                    closes_1h = np.array([float(b[4]) for b in ohlcv_1h])
                    highs_1h  = np.array([float(b[2]) for b in ohlcv_1h])
                    lows_1h   = np.array([float(b[3]) for b in ohlcv_1h])

                    # Simple CHoCH: look at the last N bars since signal creation
                    # N bars = signal_age_hours (roughly)
                    n_bars = max(3, min(int(signal_age_hours) + 1, len(closes_1h) - 2))
                    recent_closes = closes_1h[-n_bars:]
                    recent_highs  = highs_1h[-n_bars:]
                    recent_lows   = lows_1h[-n_bars:]

                    if sig.direction == "LONG":
                        # Bearish CHoCH: a prior swing high was broken lower
                        # (lower high formed after the signal was bullish)
                        prior_high = float(np.max(highs_1h[-(n_bars+3):-n_bars]))
                        current_high = float(np.max(recent_highs))
                        if current_high < prior_high * 0.97:  # Lower high by >3%
                            # Also confirm lower low (not just a pullback)
                            prior_low = float(np.min(lows_1h[-(n_bars+3):-n_bars]))
                            current_low = float(np.min(recent_lows))
                            if current_low < prior_low * 0.98:
                                return (
                                    f"Bearish CHoCH since signal: "
                                    f"lower high ({current_high:.5f} < {prior_high:.5f}) "
                                    f"+ lower low ({current_low:.5f} < {prior_low:.5f}) "
                                    f"on 1h in last {n_bars}h"
                                )
                    else:  # SHORT
                        # Bullish CHoCH: higher low + higher high formed since signal
                        prior_low = float(np.min(lows_1h[-(n_bars+3):-n_bars]))
                        current_low = float(np.min(recent_lows))
                        if current_low > prior_low * 1.03:
                            prior_high = float(np.max(highs_1h[-(n_bars+3):-n_bars]))
                            current_high = float(np.max(recent_highs))
                            if current_high > prior_high * 1.02:
                                return (
                                    f"Bullish CHoCH since signal: "
                                    f"higher low ({current_low:.5f} > {prior_low:.5f}) "
                                    f"+ higher high ({current_high:.5f} > {prior_high:.5f}) "
                                    f"on 1h in last {n_bars}h"
                                )
            except Exception:
                pass

        except Exception as _stale_outer:
            logger.debug(f"Staleness check error (non-fatal): {_stale_outer}")

        return None  # All checks passed — setup still valid

    async def _check_triggers(self, sig: TrackedExecution, price: float, market_state: dict = None):
        """
        Fix E1: structure_shift now actually detects BOS (break of structure).
        Fix E2: momentum_expansion requires real volume + body expansion, not any green candle.
        Fix E3: accepts pre-computed market_state instead of fetching again.
        FIX STICKY-TRIGGERS: trigger flags now expire after TRIGGER_WINDOW_SECS.
          Triggers accumulate within a rolling 20-minute window.  If the window
          expires without reaching min_triggers, all flags reset and a new window
          starts.  This prevents a rejection candle from hour 1 combining with a
          structure shift from hour 3 to produce a false EXECUTE.
        """
        import numpy as np
        now = time.time()

        # ── Stale-trigger reset ────────────────────────────────────────────────
        if sig.trigger_window_since is None:
            sig.trigger_window_since = now
        elif now - sig.trigger_window_since > sig.TRIGGER_WINDOW_SECS:
            # Window expired — were we close enough to fire?  If so, extend rather
            # than wipe (avoids punishing slow momentum setups on illiquid alts).
            # Only reset if triggers_met is still well below the threshold.
            _gap = sig.min_triggers - sig.triggers_met
            if _gap > 1.0:  # More than 1 point away — definitely stale, reset
                _was = (sig.has_rejection_candle, sig.has_structure_shift,
                        sig.has_momentum_expansion, sig.has_liquidity_reaction)
                sig.has_rejection_candle   = False
                sig.has_structure_shift    = False
                sig.has_momentum_expansion = False
                sig.has_liquidity_reaction = False
                # Gap 5: clear magnitudes too so stale multipliers don't leak
                # into the next window's fresh trigger detections.
                sig.rejection_magnitude = 1.0
                sig.structure_magnitude = 1.0
                sig.momentum_magnitude  = 1.0
                sig.liquidity_magnitude = 1.0
                sig.trigger_window_since   = now
                if any(_was):
                    logger.info(
                        f"🔄 Trigger reset | #{sig.signal_id} {sig.symbol} "
                        f"| window={sig.TRIGGER_WINDOW_SECS//60}min expired "
                        f"| gap={_gap:.1f} pts | rej={_was[0]} struct={_was[1]} "
                        f"mom={_was[2]} liq={_was[3]}"
                    )
            else:
                # Close to threshold — extend the window rather than resetting
                sig.trigger_window_since = now

        from strategies.base import SETUP_CLASS_CONFIRM_TF as _SC_TF
        _confirm_tf = _SC_TF.get(getattr(sig, 'setup_class', 'intraday'), '15m')
        ohlcv = await api.fetch_ohlcv(sig.symbol, _confirm_tf, limit=20)
        if not ohlcv or len(ohlcv) < 5:
            return

        last = ohlcv[-1]
        o, h, l, c = float(last[1]), float(last[2]), float(last[3]), float(last[4])

        highs  = np.array([float(x[2]) for x in ohlcv])
        lows   = np.array([float(x[3]) for x in ohlcv])
        closes = np.array([float(x[4]) for x in ohlcv])
        vols   = np.array([float(x[5]) for x in ohlcv])

        body = abs(c - o)
        rng  = h - l if (h - l) > 0 else 1e-9

        # Compute ATR-14 early — used by both rejection candle and momentum checks
        # BUG-1 FIX: use True Range (max of H-L, |H-prev_close|, |L-prev_close|) instead
        # of close-to-close differences, which understate ATR on wick-heavy candles.
        if len(closes) >= 2:
            _prev_c = np.concatenate([[closes[0]], closes[:-1]])
            _tr = np.maximum(highs - lows, np.maximum(np.abs(highs - _prev_c), np.abs(lows - _prev_c)))
            atr_14 = float(np.mean(_tr[-14:])) if len(_tr) >= 14 else float(np.mean(_tr))
        else:
            atr_14 = rng * 0.5
        avg_vol = float(np.mean(vols[-10:])) if len(vols) >= 10 else float(vols[-1])

        # ── Gap 1: Vol-aware trigger window refresh ─────────────────────────
        # Recompute TRIGGER_WINDOW_SECS from live ATR %.  Compare recent ATR %
        # (last 5 bars) to baseline ATR % (last 20 bars).  In fast vol the
        # confluence window shortens; in quiet markets it stretches.  We scale
        # off the setup-class-derived base so the original 4-bar heuristic is
        # preserved as the neutral point.
        try:
            if len(closes) >= 5 and len(_tr) >= 20 and c > 0:
                atr_pct_recent = float(np.mean(_tr[-5:])) / c
                atr_pct_baseline = float(np.mean(_tr[-20:])) / c
                if atr_pct_recent > 0 and atr_pct_baseline > 0:
                    from core._exec_helpers import scale_trigger_window as _vscale
                    _base_w = _get_trigger_window_secs(
                        getattr(sig, 'setup_class', 'intraday')
                    )
                    sig.TRIGGER_WINDOW_SECS = _vscale(
                        _base_w, atr_pct_recent, atr_pct_baseline,
                        min_secs=TRIGGER_WINDOW_MIN_SECS,
                        max_secs=TRIGGER_WINDOW_MAX_SECS,
                    )
        except Exception:
            pass

        # ── Gap 3: SL refresh against newest swing pivot + RR decay check ───
        # Between publication and fill, a new swing pivot may have formed closer
        # to entry.  Tightening SL to that pivot preserves RR integrity and
        # avoids giving away an entire leg of stop distance that the original
        # computation left behind.  Never loosens (max/min with original).
        #
        # After tightening, if the resulting RR to TP1 has decayed below 1.0,
        # we mark the signal RR_DECAY and let the top-level _check expire it.
        try:
            from core._exec_helpers import (
                tightened_sl as _tsl, compute_rr as _crr,
            )
            # Nearest opposite-side pivot in the last ~12 bars of the confirm TF.
            if len(closes) >= 12:
                window = closes[-12:]
                if sig.direction == "LONG":
                    pivot = float(np.min(lows[-12:]))
                    new_sl = _tsl(sig.direction, sig.stop_loss, pivot)
                elif sig.direction == "SHORT":
                    pivot = float(np.max(highs[-12:]))
                    new_sl = _tsl(sig.direction, sig.stop_loss, pivot)
                else:
                    new_sl = sig.stop_loss
                if new_sl != sig.stop_loss:
                    _old = sig.stop_loss
                    sig.stop_loss = float(new_sl)
                    logger.info(
                        f"🎯 SL tightened | #{sig.signal_id} {sig.symbol} "
                        f"{sig.direction} | {_old:.6g} → {sig.stop_loss:.6g} "
                        f"(pivot={pivot:.6g})"
                    )
                # Recompute RR to TP1 from current mid-entry.  If RR floor
                # collapses below 1.0 after repricing, stamp rr_decay so the
                # top-level _check path can expire the signal cleanly.
                if sig.tp1 and sig.entry_mid > 0:
                    _live_rr = _crr(
                        sig.direction, sig.entry_mid, sig.stop_loss, sig.tp1,
                    )
                    sig._live_rr = _live_rr  # exposed for logging / expire
                    if _live_rr > 0 and _live_rr < 1.0:
                        sig._rr_decay_flag = True
        except Exception:
            pass


        # Trigger 1: rejection candle — wick-heavy candle with meaningful size
        #
        # OLD: body/rng < 0.4 AND body > 0
        #   Problem: a 1-tick body on a 3-tick doji qualifies. On thin alts with
        #   a 0.0001 spread this fires on almost every candle in the entry zone.
        #
        # NEW: body/rng < 0.4 AND total range >= 0.5 × ATR14
        #   The ATR floor ensures the candle is large enough to represent real
        #   rejection.  A micro-doji with rng=0.00002 on a coin trading at $0.03
        #   is noise — it should not count as rejection evidence worth +1pt.
        _wick_meaningful = rng >= atr_14 * 0.5
        if body / rng < 0.4 and body > 0 and _wick_meaningful:
            if sig.direction == "LONG" and c > o:
                sig.has_rejection_candle = True
            elif sig.direction == "SHORT" and c < o:
                sig.has_rejection_candle = True
        # Gap 5: magnitude — wick-size relative to ATR.  A rejection on a
        # 2×ATR bar carries more information than one on a 0.5×ATR bar.
        if sig.has_rejection_candle and atr_14 > 0:
            try:
                from core._exec_helpers import magnitude_multiplier as _mm
                sig.rejection_magnitude = _mm(rng, atr_14)
            except Exception:
                pass

        # Fix E2: momentum expansion — requires volume spike AND large body
        vol_spike = float(vols[-1]) > avg_vol * 1.3
        body_expanded = body > atr_14 * 0.6

        if vol_spike and body_expanded:
            if sig.direction == "LONG" and c > o:
                sig.has_momentum_expansion = True
            elif sig.direction == "SHORT" and c < o:
                sig.has_momentum_expansion = True
        # Gap 5: magnitude — body size * vol ratio, normalized by ATR & avg vol.
        # A 2×ATR candle on 3× volume is a much stronger signal than a
        # 0.6×ATR candle on 1.3× volume (the threshold floor).
        if sig.has_momentum_expansion and atr_14 > 0 and avg_vol > 0:
            try:
                from core._exec_helpers import magnitude_multiplier as _mm
                body_mag = _mm(body, atr_14 * 0.6)
                vol_mag = _mm(float(vols[-1]), avg_vol * 1.3)
                # Combined magnitude is the geometric-ish mean, clamped.
                sig.momentum_magnitude = max(0.5, min(2.0, (body_mag * vol_mag) ** 0.5))
            except Exception:
                pass

        # STRATEGIST FIX E: structure shift with volume validation
        # "Structure shift = break of structure (real demand/supply)" → score=2
        #
        # QUALITY UPGRADE (3 changes):
        #   1. Lookback: 6 bars (30min) → 12 bars (60min).
        #      A 30min high on a volatile 5m chart is just noise — price breaks it
        #      constantly.  60min gives a meaningful swing high/low to break.
        #   2. Volume threshold: 1.05× avg → 1.15× avg.
        #      5% above average is trivially easy on most candles.  15% requires
        #      genuine participation above the baseline.
        #   3. Clearance: 0.1% → 0.2%.
        #      Sub-tick clearances cause false breaks on spreads and micro-moves.
        if len(closes) >= 13:
            swing_lookback = closes[-13:-1]  # 12 bars = 60min of 5m structure
            _vol_confirms_structure = (
                float(vols[-1]) >= float(np.mean(vols[-15:-1])) * 1.15
                if len(vols) >= 15 else True
            )
            if sig.direction == "LONG":
                recent_swing_high = float(np.max(swing_lookback))
                if c > recent_swing_high * 1.002 and _vol_confirms_structure:
                    sig.has_structure_shift = True
                    # Gap 5: magnitude — how far past the swing high we broke.
                    # 0.2% clearance = baseline (floor 0.5); 1%+ clearance = 2.0.
                    try:
                        from core._exec_helpers import magnitude_multiplier as _mm
                        clearance_pct = (c - recent_swing_high) / recent_swing_high
                        sig.structure_magnitude = _mm(clearance_pct, 0.002)
                    except Exception:
                        pass
            elif sig.direction == "SHORT":
                recent_swing_low = float(np.min(swing_lookback))
                if c < recent_swing_low * 0.998 and _vol_confirms_structure:
                    sig.has_structure_shift = True
                    try:
                        from core._exec_helpers import magnitude_multiplier as _mm
                        clearance_pct = (recent_swing_low - c) / recent_swing_low
                        sig.structure_magnitude = _mm(clearance_pct, 0.002)
                    except Exception:
                        pass

        # Trigger 4: liquidity reaction — use pre-computed market_state (Fix E3: no double fetch)
        if market_state is None:
            market_state = await build_market_state(sig.symbol, sig.direction)

        if detect_liquidity_reaction({"direction": sig.direction}, market_state):
            sig.has_liquidity_reaction = True
            # Gap 5: magnitude — sweep USD relative to per-symbol baseline.
            # build_market_state typically exposes `liquidity_sweep_usd` in the
            # liquidity block; we fall back gracefully if absent.
            try:
                from core._exec_helpers import magnitude_multiplier as _mm
                liq_block = dict((market_state or {}).get('liquidity') or {})
                sweep_usd = float(liq_block.get('sweep_usd') or 0.0)
                median_usd = float(liq_block.get('median_sweep_usd') or 0.0)
                if sweep_usd > 0 and median_usd > 0:
                    sig.liquidity_magnitude = _mm(sweep_usd, median_usd)
            except Exception:
                pass

        # Persist trigger flags whenever any of them is now set.
        # This is a cheap UPDATE (only touched if at least one flag is True) so
        # we don't spam the DB — no flags means nothing to save.
        if (sig.has_rejection_candle or sig.has_structure_shift
                or sig.has_momentum_expansion or sig.has_liquidity_reaction):
            try:
                from data.database import db as _trig_db
                _safe_ensure_future(
                    _trig_db.update_tracked_signal(
                        sig.signal_id,
                        has_rejection_candle=1 if sig.has_rejection_candle else 0,
                        has_structure_shift=1 if sig.has_structure_shift else 0,
                        has_momentum_expansion=1 if sig.has_momentum_expansion else 0,
                        has_liquidity_reaction=1 if sig.has_liquidity_reaction else 0,
                    ),
                    context=f"trigger_flags #{sig.signal_id}",
                )
            except Exception:
                pass


# ── Singleton ──────────────────────────────────────────────
execution_engine = ExecutionEngine()
