"""
TitanBot Pro — Regime-Aware TP/SL Adjuster (v12)
====================================================
Central module that adjusts strategy-computed TP/SL based on:
  1. Current regime (TRENDING/CHOPPY/VOLATILE/VOLATILE)
  2. Range structure (range_high, range_low, equilibrium)
  3. Setup class (swing/intraday/scalp)

Architecture:
  Strategy computes raw TP1/TP2/TP3/SL → this module adjusts them.
  Called from engine._scan_symbol() AFTER entry refiner, BEFORE aggregator.

Principles:
  - TRENDING: wider TPs (trends deliver), wider SL (avoid whipsaw)
  - CHOPPY: TPs capped by range structure, tighter SL, no TP3
  - VOLATILE: wider everything (big candles need room)
  - VOLATILE: tight everything, TP1 only, survival mode
  - TP1 always >= minimum floor (1.0R swing, 0.8R intraday, 0.6R scalp)
  - In chop, TPs never exceed range boundaries
"""

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Regime multiplier profiles ────────────────────────────────
# These scale the DISTANCE from entry to SL/TP (not the price itself)

REGIME_PROFILES = {
    "BULL_TREND": {
        "sl_mult": 1.20,     # Wider — trends whipsaw before continuing
        "tp1_mult": 1.00,    # TP1 normal
        "tp2_mult": 1.40,    # TP2 extended — trends run
        "tp3_mult": 1.30,    # V14: was 2.0x — too aggressive for structural targets
        "tp3_enabled": True,
        "trail_pct": 0.30,   # Tighter trail — lock in trend profits
        # FIX (P2-C): Direction-aware multipliers in BULL_TREND.
        # Mirror of BEAR_TREND's direction-aware TP/SL.
        # LONG in BULL_TREND = trend-aligned → wider targets (ride the wave)
        # SHORT in BULL_TREND = counter-trend → tighter targets, tighter SL
        "long_tp2_mult": 1.60,    # LONGs: wider TP2 (trend continuation expected)
        "long_tp3_mult": 1.80,    # LONGs: wider TP3 for full trend extension
        "short_tp1_mult": 0.80,   # SHORTs: tighter TP1 (take profit faster in bull)
        "short_tp2_mult": 0.85,   # SHORTs: tighter TP2 (pullback, not reversal)
        "short_tp3_mult": 0.95,   # SHORTs: runner heavily constrained
        "short_sl_mult": 0.95,    # SHORTs: tighter SL (less room to be wrong)
        "short_local_tp1_mult": 0.90,  # Local bearish structure: not a scalp, still cautious
        "short_local_tp2_mult": 1.05,  # Allow modest extension below support
        "short_local_tp3_mult": 1.10,  # Controlled runner only
        "short_local_sl_mult": 0.97,   # Keep risk tighter than full trend
    },
    "BEAR_TREND": {
        "sl_mult": 1.20,
        "tp1_mult": 1.00,
        "tp2_mult": 1.40,
        "tp3_mult": 1.30,
        "tp3_enabled": True,
        "trail_pct": 0.30,
        # STRATEGIST FIX F: Direction-aware multipliers in BEAR_TREND
        # "Trend trades (SHORTs) underutilize R:R — should aim 2.0-4.0R like BANANAS31"
        # SHORT in BEAR_TREND = trend-aligned → wider targets
        # LONG in BEAR_TREND = counter-trend → tighter targets (protect capital)
        "short_tp2_mult": 1.70,   # SHORTs: wider TP2 (trend continuation expected)
        "short_tp3_mult": 2.00,   # SHORTs: wide TP3 for full wave completion
        "long_tp1_mult": 0.75,    # LONGs: tighter TP1 (take profit faster in bear)
        "long_tp2_mult": 0.85,    # LONGs: tighter TP2
        "long_sl_mult": 1.00,     # LONGs: tighter SL (less room to be wrong)
        "long_local_tp1_mult": 0.90,   # Local bullish structure: faster than trend, not scalp
        "long_local_tp2_mult": 1.05,   # Moderate extension only
        "long_local_tp3_mult": 1.10,   # Controlled runner
        "long_local_sl_mult": 0.97,    # Slightly tighter risk than trend-aligned shorts
    },
    "CHOPPY": {
        "sl_mult": 0.85,     # Tighter — ranges have clear boundaries
        "tp1_mult": 0.80,    # TP1 closer — take profit quick
        "tp2_mult": 0.70,    # TP2 compressed — EQ is realistic max
        "tp3_mult": 0.60,    # TP3 barely reachable in a range
        "tp3_enabled": True,  # Still shown but compressed
        "trail_pct": 0.0,    # NO trailing — exit at TP
    },
    "VOLATILE": {
        "sl_mult": 1.50,     # Much wider — big candles need room
        "tp1_mult": 1.30,    # TP1 wider — moves are bigger
        "tp2_mult": 1.30,    # TP2 wider
        "tp3_mult": 1.50,    # TP3 aggressive
        "tp3_enabled": True,
        "trail_pct": 0.50,   # Wide trail — ride explosive moves
    },
    "VOLATILE_PANIC": {
        "sl_mult": 0.70,     # Tight — protect capital
        "tp1_mult": 0.60,    # TP1 very close — take what you can
        "tp2_mult": 0.50,    # TP2 conservative
        "tp3_mult": 0.0,
        "tp3_enabled": False, # No TP3 — don't hold in panic
        "trail_pct": 0.0,    # No trail — exit at TP1
    },
}

DEFAULT_PROFILE = {
    "sl_mult": 1.0, "tp1_mult": 1.0, "tp2_mult": 1.0, "tp3_mult": 1.0,
    "tp3_enabled": True, "trail_pct": 0.40,
}

# ── TP1 minimum floor per setup class (in R-multiples) ───────
TP1_MIN_R = {
    "swing": 1.0,
    "intraday": 0.8,
    "scalp": 0.6,
}

# ── Minimum gap between TPs (in R-multiples) ─────────────────
MIN_TP_GAP_R = 0.3
MAX_TARGET_RR = 8.0
MIN_POSITIVE_PRICE_FRAC = 0.01
STRUCTURE_LOOKBACK_BARS = 30  # Bars inspected for local structure and VWAP context.
STRUCTURE_MIN_BARS = 14  # Minimum bars needed to evaluate 13-bar BOS plus the trigger bar.
STRUCTURE_FALLBACK_BARS = 5  # Consecutive bars used when pivots are too sparse to form swings.
STRUCTURE_MAX_SCORE = 4  # Structure score = structure + BOS + EQ/VWAP + rejection context.
STRUCTURE_SWING_TOLERANCE = 0.002
STRUCTURE_BOS_LOOKBACK = 13
STRUCTURE_EQ_VWAP_TOLERANCE = 0.005
STRUCTURE_REJECTION_LOOKBACK = 5  # Recent candles checked for wick-based rejection evidence.
STRUCTURE_REJECTION_WICK_RATIO = 1.2
STRUCTURE_MIN_REJECTIONS = 2
STRUCTURE_MIN_STRONG_SCORE = 3
STRUCTURE_MIN_BODY_SIZE = 1e-9


@dataclass
class AdjustedLevels:
    """Result of regime-aware TP/SL adjustment"""
    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    rr_ratio: float
    regime: str
    adjustments: list  # Human-readable list of what changed
    trail_pct: float   # Trailing stop percentage for this regime
    trade_type: str = "TREND"
    local_structure_bias: str = "neutral"
    local_structure_score: int = 0
    local_structure_reason: str = ""
    local_structure_bars: int = 0
    local_structure_used_vwap: bool = False
    local_structure_used_rejections: bool = False


@dataclass
class LocalStructureAssessment:
    """Guarded local structure read for trend-conflict trade shaping."""
    bias: str
    score: int
    reason: str = ""
    bars: int = 0
    used_vwap: bool = False
    used_rejections: bool = False


def _recent_swings(values: Sequence[float], kind: str, span: int = 2) -> list:
    """Return recent local pivots for highs/lows."""
    pivots = []
    if len(values) < span * 2 + 1:
        return pivots

    for i in range(span, len(values) - span):
        window = values[i - span:i + span + 1]
        current = values[i]
        if kind == "high" and current >= max(window):
            pivots.append(float(current))
        elif kind == "low" and current <= min(window):
            pivots.append(float(current))
    return pivots


def _check_monotonic_trend(values: Sequence[float], direction: str, bars: int = STRUCTURE_FALLBACK_BARS) -> bool:
    """Fallback for sparse pivots: require consecutive lower highs or consecutive higher lows."""
    if len(values) < bars:
        return False
    if direction == "down":
        return all(values[-i] < values[-i - 1] for i in range(1, bars))
    return all(values[-i] > values[-i - 1] for i in range(1, bars))


def _assess_local_structure(
    direction: str,
    entry_mid: float,
    range_high: float,
    range_low: float,
    range_eq: float,
    recent_opens: Optional[Sequence[float]] = None,
    recent_highs: Optional[Sequence[float]] = None,
    recent_lows: Optional[Sequence[float]] = None,
    recent_closes: Optional[Sequence[float]] = None,
    recent_volumes: Optional[Sequence[float]] = None,
) -> LocalStructureAssessment:
    """Use structure as the hard gate, context as confirmation."""
    if not recent_highs or not recent_lows or not recent_closes:
        return LocalStructureAssessment(bias="neutral", score=0, reason="missing_ohlc")

    highs = [float(v) for v in recent_highs[-STRUCTURE_LOOKBACK_BARS:]]
    lows = [float(v) for v in recent_lows[-STRUCTURE_LOOKBACK_BARS:]]
    closes = [float(v) for v in recent_closes[-STRUCTURE_LOOKBACK_BARS:]]
    opens = [float(v) for v in (recent_opens or [])[-STRUCTURE_LOOKBACK_BARS:]]
    volumes = [float(v) for v in (recent_volumes or [])[-STRUCTURE_LOOKBACK_BARS:]]
    bars = min(len(highs), len(lows), len(closes))

    _min_structure_bars = max(STRUCTURE_BOS_LOOKBACK + 1, STRUCTURE_MIN_BARS)
    if len(highs) < _min_structure_bars or len(lows) < _min_structure_bars or len(closes) < _min_structure_bars:
        return LocalStructureAssessment(
            bias="neutral",
            score=0,
            reason="insufficient_bars",
            bars=bars,
        )

    swing_highs = _recent_swings(highs, "high")
    swing_lows = _recent_swings(lows, "low")
    lower_highs = len(swing_highs) >= 2 and swing_highs[-1] < swing_highs[-2] * (1 - STRUCTURE_SWING_TOLERANCE)
    higher_lows = len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2] * (1 + STRUCTURE_SWING_TOLERANCE)
    if not lower_highs:
        lower_highs = _check_monotonic_trend(highs, "down")
    if not higher_lows:
        higher_lows = _check_monotonic_trend(lows, "up")
    bos_down = closes[-1] < min(lows[-STRUCTURE_BOS_LOOKBACK:-1]) * (1 - STRUCTURE_SWING_TOLERANCE)
    bos_up = closes[-1] > max(highs[-STRUCTURE_BOS_LOOKBACK:-1]) * (1 + STRUCTURE_SWING_TOLERANCE)

    eq = range_eq if range_high > range_low > 0 and range_low <= range_eq <= range_high else (
        (range_high + range_low) / 2 if range_high > range_low > 0 else 0.0
    )
    below_eq = eq > 0 and closes[-1] < eq * (1 - STRUCTURE_EQ_VWAP_TOLERANCE) and entry_mid < eq
    above_eq = eq > 0 and closes[-1] > eq * (1 + STRUCTURE_EQ_VWAP_TOLERANCE) and entry_mid > eq

    below_vwap = above_vwap = False
    used_vwap = False
    if volumes and len(volumes) == len(closes):
        total_vol = sum(max(v, 0.0) for v in volumes)
        if total_vol > 0:
            used_vwap = True
            typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
            vwap = sum(tp * max(v, 0.0) for tp, v in zip(typical, volumes)) / total_vol
            below_vwap = closes[-1] < vwap * (1 - STRUCTURE_EQ_VWAP_TOLERANCE)
            above_vwap = closes[-1] > vwap * (1 + STRUCTURE_EQ_VWAP_TOLERANCE)

    bearish_rejections = bullish_rejections = False
    used_rejections = False
    if opens and len(opens) == len(closes):
        used_rejections = True
        bearish_hits = 0
        bullish_hits = 0
        for o, h, l, c in zip(
            opens[-STRUCTURE_REJECTION_LOOKBACK:],
            highs[-STRUCTURE_REJECTION_LOOKBACK:],
            lows[-STRUCTURE_REJECTION_LOOKBACK:],
            closes[-STRUCTURE_REJECTION_LOOKBACK:],
        ):
            body = max(abs(c - o), STRUCTURE_MIN_BODY_SIZE)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            if upper_wick > body * STRUCTURE_REJECTION_WICK_RATIO and c <= o:
                bearish_hits += 1
            if lower_wick > body * STRUCTURE_REJECTION_WICK_RATIO and c >= o:
                bullish_hits += 1
        bearish_rejections = bearish_hits >= STRUCTURE_MIN_REJECTIONS
        bullish_rejections = bullish_hits >= STRUCTURE_MIN_REJECTIONS

    if direction == "SHORT":
        score = sum([lower_highs, bos_down, (below_eq or below_vwap), bearish_rejections])
        bias = "bearish" if lower_highs and bos_down and score >= STRUCTURE_MIN_STRONG_SCORE else "neutral"
        reason = "confirmed_bearish" if bias == "bearish" else "structure_unconfirmed"
        return LocalStructureAssessment(
            bias=bias,
            score=score,
            reason=reason,
            bars=bars,
            used_vwap=used_vwap,
            used_rejections=used_rejections,
        )

    score = sum([higher_lows, bos_up, (above_eq or above_vwap), bullish_rejections])
    bias = "bullish" if higher_lows and bos_up and score >= STRUCTURE_MIN_STRONG_SCORE else "neutral"
    reason = "confirmed_bullish" if bias == "bullish" else "structure_unconfirmed"
    return LocalStructureAssessment(
        bias=bias,
        score=score,
        reason=reason,
        bars=bars,
        used_vwap=used_vwap,
        used_rejections=used_rejections,
    )


def adjust_levels(
    entry_low: float,
    entry_high: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    tp3: Optional[float],
    direction: str,
    setup_class: str = "intraday",
    regime: str = "CHOPPY",
    chop_strength: float = 0.0,
    range_high: float = 0.0,
    range_low: float = 0.0,
    range_eq: float = 0.0,
    recent_opens: Optional[Sequence[float]] = None,
    recent_highs: Optional[Sequence[float]] = None,
    recent_lows: Optional[Sequence[float]] = None,
    recent_closes: Optional[Sequence[float]] = None,
    recent_volumes: Optional[Sequence[float]] = None,
) -> AdjustedLevels:
    """
    Adjust strategy-computed TP/SL based on regime and range structure.

    Called from engine AFTER strategy + entry refiner, BEFORE aggregator.
    """
    adjustments = []
    entry_mid = (entry_low + entry_high) / 2
    is_long = direction == "LONG"

    # Get regime profile
    profile = REGIME_PROFILES.get(regime, DEFAULT_PROFILE)
    trail_pct = profile["trail_pct"]
    local_structure = _assess_local_structure(
        direction=direction,
        entry_mid=entry_mid,
        range_high=range_high,
        range_low=range_low,
        range_eq=range_eq,
        recent_opens=recent_opens,
        recent_highs=recent_highs,
        recent_lows=recent_lows,
        recent_closes=recent_closes,
        recent_volumes=recent_volumes,
    )
    trade_type = f"TREND_{direction}"
    if regime == "BULL_TREND" and not is_long:
        trade_type = "LOCAL_CONTINUATION_SHORT" if local_structure.bias == "bearish" else "PULLBACK_SHORT"
    elif regime == "BEAR_TREND" and is_long:
        trade_type = "LOCAL_CONTINUATION_LONG" if local_structure.bias == "bullish" else "PULLBACK_LONG"

    # ── 1. Compute raw distances from entry ────────────────────
    sl_dist = abs(entry_mid - stop_loss)
    tp1_dist = abs(tp1 - entry_mid)
    tp2_dist = abs(tp2 - entry_mid)
    tp3_dist = abs(tp3 - entry_mid) if tp3 else 0

    # V14: Minimum SL distance floor — prevents stops too tight for the timeframe
    # These are absolute minimums regardless of ATR calculation
    _MIN_SL_PCT = {"swing": 0.020, "intraday": 0.012, "scalp": 0.008}
    _min_sl_dist = entry_mid * _MIN_SL_PCT.get(setup_class, 0.012)
    if sl_dist < _min_sl_dist and sl_dist > 0:
        adjustments.append(f"SL floor: {sl_dist/entry_mid*100:.1f}% → {_min_sl_dist/entry_mid*100:.1f}%")
        sl_dist = _min_sl_dist

    # FIX-13: Micro-price token guard — when SL is essentially zero (< 0.001% of entry),
    # the strategy computed a degenerate level. Fall back to the minimum.
    # This prevents TP3 values that are 10B% above entry on $0.00001 tokens.
    if sl_dist < entry_mid * 0.00001:
        sl_dist = _min_sl_dist
        adjustments.append("SL near-zero — micro-price fallback applied")

    if sl_dist <= 0:
        # Can't adjust without valid SL
        return AdjustedLevels(
            entry_low=entry_low, entry_high=entry_high,
            stop_loss=stop_loss, tp1=tp1, tp2=tp2, tp3=tp3,
            rr_ratio=0, regime=regime, adjustments=["SL distance zero"],
            trail_pct=trail_pct,
            trade_type=trade_type,
            local_structure_bias=local_structure.bias,
            local_structure_score=local_structure.score,
            local_structure_reason=local_structure.reason,
            local_structure_bars=local_structure.bars,
            local_structure_used_vwap=local_structure.used_vwap,
            local_structure_used_rejections=local_structure.used_rejections,
        )

    # ── 2. Apply regime multipliers ────────────────────────────
    # V13 BUG 2: In CHOPPY, don't blindly tighten SL — anchor to range boundary
    # to avoid stop hunts. Market makers sweep above/below range edges.
    # BUG-5 FIX: has_range MUST be defined before the first use below.
    # Previously defined at step 5 (line ~208) but first referenced here (step 2),
    # causing UnboundLocalError on every CHOPPY signal. Engine silently swallowed
    # this via `except Exception`, so all CHOPPY TP/SL adjustment was dead code.
    #
    # CRITICAL FIX (TP-CORRUPTION): range_high/range_low are BTC's weekly range
    # (e.g. $60k–$70k). For altcoins trading at $0.71, this makes sl_dist = 59,199
    # which then inflates TP1 floor to $47,360. Guard: range is only relevant when
    # entry_mid is actually within 3x of the range boundaries.
    _range_relevant = (
        range_high > range_low > 0
        and entry_mid >= range_low * 0.1   # entry not more than 10x below range_low
        and entry_mid <= range_high * 3.0  # entry not more than 3x above range_high
    )
    has_range = _range_relevant
    if regime == "CHOPPY" and has_range:
        range_size = range_high - range_low
        _range_buffer = range_size * 0.08  # 8% of range beyond the boundary
        if is_long:
            # LONG SL should be below range_low + buffer (beyond the demand zone)
            _range_sl_dist = abs(entry_mid - (range_low - _range_buffer))
            if _range_sl_dist > sl_dist:
                adjustments.append(f"SL widened to range_low-8% (anti-hunt)")
                sl_dist = _range_sl_dist
            # Don't tighten below the range-anchored level
        else:
            # SHORT SL should be above range_high + buffer (beyond supply zone)
            _range_sl_dist = abs((range_high + _range_buffer) - entry_mid)
            if _range_sl_dist > sl_dist:
                adjustments.append(f"SL widened to range_high+8% (anti-hunt)")
                sl_dist = _range_sl_dist
    else:
        sl_dist *= profile["sl_mult"]
        if profile["sl_mult"] != 1.0:
            adjustments.append(f"SL ×{profile['sl_mult']:.2f}")

    tp1_dist *= profile["tp1_mult"]
    tp2_dist *= profile["tp2_mult"]
    if tp3_dist > 0 and profile["tp3_enabled"]:
        tp3_dist *= profile["tp3_mult"]
    if profile["tp1_mult"] != 1.0:
        adjustments.append(f"TP1 ×{profile['tp1_mult']:.2f}")
    if profile["tp2_mult"] != 1.0:
        adjustments.append(f"TP2 ×{profile['tp2_mult']:.2f}")

    # STRATEGIST FIX F: Apply direction-aware TP mults in BEAR_TREND
    # "Trend trades (SHORTs) should aim for 2.0-4.0R like BANANAS31 (3.4R)"
    # SHORT in BEAR → wider targets (trend continuation); LONG in BEAR → tighter (quick exit)
    if regime == "BEAR_TREND" and any(k in profile for k in ("short_tp2_mult", "long_tp1_mult")):
        if not is_long and "short_tp2_mult" in profile:
            _s_tp2_m = profile["short_tp2_mult"]
            _s_tp3_m = profile.get("short_tp3_mult", _s_tp2_m)
            tp2_dist = (tp2_dist / profile["tp2_mult"]) * _s_tp2_m
            if tp3_dist > 0:
                tp3_dist = (tp3_dist / profile.get("tp3_mult", 1.3)) * _s_tp3_m
            adjustments.append(f"SHORT BEAR_TREND: TP2 ×{_s_tp2_m} TP3 ×{_s_tp3_m} (trend run)")
        elif is_long and "long_tp1_mult" in profile:
            _l_tp1_m = profile["long_tp1_mult"]
            _l_tp2_m = profile.get("long_tp2_mult", _l_tp1_m)
            _l_tp3_m = profile.get("long_tp3_mult", 1.0)
            _l_sl_m  = profile.get("long_sl_mult", 1.0)
            if trade_type == "LOCAL_CONTINUATION_LONG":
                _l_tp1_m = profile.get("long_local_tp1_mult", 0.90)
                _l_tp2_m = profile.get("long_local_tp2_mult", 1.05)
                _l_tp3_m = profile.get("long_local_tp3_mult", 1.10)
                _l_sl_m = profile.get("long_local_sl_mult", _l_sl_m)
            tp1_dist = (tp1_dist / profile["tp1_mult"]) * _l_tp1_m
            tp2_dist = (tp2_dist / profile["tp2_mult"]) * _l_tp2_m
            if tp3_dist > 0:
                tp3_dist = (tp3_dist / profile.get("tp3_mult", 1.3)) * _l_tp3_m
            if _l_sl_m != 1.0 and regime != "CHOPPY":
                sl_dist *= _l_sl_m
            _label = "local continuation" if trade_type == "LOCAL_CONTINUATION_LONG" else "quick exit"
            adjustments.append(
                f"LONG BEAR_TREND: TP adjusted TP1×{_l_tp1_m} TP2×{_l_tp2_m} TP3×{_l_tp3_m} ({_label})"
            )

    # FIX (P2-C): Apply direction-aware TP mults in BULL_TREND
    # LONG in BULL → wider targets (trend continuation expected)
    # SHORT in BULL → tighter targets, tighter SL (counter-trend pullback scalp)
    if regime == "BULL_TREND" and any(k in profile for k in ("long_tp2_mult", "short_tp1_mult")):
        if is_long and "long_tp2_mult" in profile:
            _l_tp2_m = profile["long_tp2_mult"]
            _l_tp3_m = profile.get("long_tp3_mult", _l_tp2_m)
            tp2_dist = (tp2_dist / profile["tp2_mult"]) * _l_tp2_m
            if tp3_dist > 0:
                tp3_dist = (tp3_dist / profile.get("tp3_mult", 1.3)) * _l_tp3_m
            adjustments.append(f"LONG BULL_TREND: TP2 ×{_l_tp2_m} TP3 ×{_l_tp3_m} (trend run)")
        elif not is_long and "short_tp1_mult" in profile:
            _s_tp1_m = profile["short_tp1_mult"]
            _s_tp2_m = profile.get("short_tp2_mult", _s_tp1_m)
            _s_tp3_m = profile.get("short_tp3_mult", 1.0)
            _s_sl_m  = profile.get("short_sl_mult", 1.0)
            if trade_type == "LOCAL_CONTINUATION_SHORT":
                _s_tp1_m = profile.get("short_local_tp1_mult", 0.90)
                _s_tp2_m = profile.get("short_local_tp2_mult", 1.05)
                _s_tp3_m = profile.get("short_local_tp3_mult", 1.10)
                _s_sl_m = profile.get("short_local_sl_mult", _s_sl_m)
            tp1_dist = (tp1_dist / profile["tp1_mult"]) * _s_tp1_m
            tp2_dist = (tp2_dist / profile["tp2_mult"]) * _s_tp2_m
            if tp3_dist > 0:
                tp3_dist = (tp3_dist / profile.get("tp3_mult", 1.3)) * _s_tp3_m
            if _s_sl_m != 1.0:
                sl_dist *= _s_sl_m
            _label = "local continuation" if trade_type == "LOCAL_CONTINUATION_SHORT" else "pullback scalp"
            adjustments.append(
                f"SHORT BULL_TREND: TP adjusted TP1×{_s_tp1_m} TP2×{_s_tp2_m} TP3×{_s_tp3_m} SL×{_s_sl_m} ({_label})"
            )

    # ── FIX-5: Cap TP extension for counter-trend LONGs in weekly BEARISH ──
    # When regime=BULL_TREND but weekly bias=BEARISH, LONG signals are counter-trend
    # bounces (the 4h is pushing up against the weekly distribution).
    # Applying ×1.4 TP2 sends the target above the weekly resistance zone that is
    # actively preventing the move from extending. These trades typically stall at
    # the prior swing high, not at TP2×1.4.
    #
    # Fix: cap TP2 extension at ×1.0 (no extension) for counter-trend LONG setups.
    # TP3 is already disallowed for VOLATILE; here we cap to ×1.1 (minimal extension).
    # This doesn't apply to SHORT in BULL_TREND (trend-aligned) or any BEAR_TREND short.
    try:
        from analyzers.htf_guardrail import htf_guardrail as _htf_ct
        _is_counter_trend_long = (
            is_long
            and regime in ("BULL_TREND", "VOLATILE")
            and getattr(_htf_ct, '_weekly_bias', 'NEUTRAL') == "BEARISH"
        )
        if _is_counter_trend_long and profile["tp2_mult"] > 1.0:
            # Undo the extension and apply a tighter cap
            _original_tp2_mult = profile["tp2_mult"]
            _ct_tp2_mult = 1.05  # Tiny extension only — bounce, not trend continuation
            tp2_dist = (tp2_dist / _original_tp2_mult) * _ct_tp2_mult
            tp3_dist = (tp3_dist / profile.get("tp3_mult", 1.3)) * 1.0 if tp3_dist > 0 else 0
            adjustments.append(
                f"TP2 capped ×1.05 (counter-trend LONG in weekly BEARISH — was ×{_original_tp2_mult})"
            )
    except Exception:
        pass  # htf_guardrail unavailable — skip this adjustment

    # ── 2c. Counter-trend pullback targets in trend regimes ─────────────
    # SHORTs inside BULL_TREND and LONGs inside BEAR_TREND are pullback
    # trades, not full reversal bets. Cap TP2 at the symbol's equilibrium
    # and suppress TP3 so the card does not advertise deep trend-reversal
    # targets against the broader market structure.
    if has_range and regime in ("BULL_TREND", "BEAR_TREND"):
        # Fallback to midpoint when caller does not provide a usable EQ.
        _has_valid_eq = isinstance(range_eq, (int, float)) and range_low <= range_eq <= range_high
        _eq = range_eq if _has_valid_eq else (range_high + range_low) / 2
        _is_counter_trend_pullback = (
            (regime == "BULL_TREND" and not is_long) or
            (regime == "BEAR_TREND" and is_long)
        )
        if _is_counter_trend_pullback and trade_type.startswith("PULLBACK_") and _eq > 0:
            _tp2_cap = (_eq - entry_mid) if is_long else (entry_mid - _eq)
            if _tp2_cap > 0 and tp2_dist > _tp2_cap:
                tp2_dist = _tp2_cap
                adjustments.append(f"Counter-trend pullback: TP2 capped at EQ ({_eq:.6f})")
            if tp3_dist > 0:
                tp3_dist = 0
                adjustments.append("Counter-trend pullback: TP3 disabled")

    # ── 3. Enforce TP1 minimum floor ───────────────────────────
    min_tp1_dist = sl_dist * TP1_MIN_R.get(setup_class, 0.8)
    if tp1_dist < min_tp1_dist:
        adjustments.append(
            f"TP1 floor: {tp1_dist / sl_dist:.2f}R → {min_tp1_dist / sl_dist:.2f}R"
        )
        tp1_dist = min_tp1_dist

    # ── 4. Enforce minimum gaps between TPs ────────────────────
    min_gap = sl_dist * MIN_TP_GAP_R
    if tp2_dist - tp1_dist < min_gap:
        tp2_dist = tp1_dist + min_gap
        adjustments.append(f"TP2 spaced +{MIN_TP_GAP_R:.1f}R from TP1")

    if tp3_dist > 0 and tp3_dist - tp2_dist < min_gap:
        tp3_dist = tp2_dist + min_gap

    # ── 5. Range-cap TPs in CHOPPY regime ──────────────────────
    # has_range already defined above at step 2
    if has_range and regime == "CHOPPY" and chop_strength >= 0.30:
        range_size = range_high - range_low
        eq = range_eq if range_eq > 0 else (range_high + range_low) / 2

        if is_long:
            # LONG targets capped by range structure
            # TP1 ≤ EQ, TP2 ≤ supply zone start, TP3 ≤ range_high
            eq_dist = abs(eq - entry_mid) if eq > entry_mid else tp1_dist
            supply_start = range_high - range_size * 0.25
            supply_dist = abs(supply_start - entry_mid) if supply_start > entry_mid else tp2_dist
            range_high_dist = abs(range_high - entry_mid) if range_high > entry_mid else tp3_dist

            if tp1_dist > eq_dist and eq_dist > min_tp1_dist:
                adjustments.append(f"TP1 capped at EQ ({eq:.6f})")
                tp1_dist = eq_dist
            if tp2_dist > supply_dist:
                adjustments.append(f"TP2 capped at supply zone")
                tp2_dist = supply_dist
            if tp3_dist > range_high_dist:
                tp3_dist = range_high_dist
        else:
            # SHORT targets capped by range structure
            eq_dist = abs(entry_mid - eq) if eq < entry_mid else tp1_dist
            demand_start = range_low + range_size * 0.25
            demand_dist = abs(entry_mid - demand_start) if demand_start < entry_mid else tp2_dist
            range_low_dist = abs(entry_mid - range_low) if range_low < entry_mid else tp3_dist

            if tp1_dist > eq_dist and eq_dist > min_tp1_dist:
                adjustments.append(f"TP1 capped at EQ ({eq:.6f})")
                tp1_dist = eq_dist
            if tp2_dist > demand_dist:
                adjustments.append(f"TP2 capped at demand zone")
                tp2_dist = demand_dist
            if tp3_dist > range_low_dist:
                tp3_dist = range_low_dist

    # ── 5b. Local range awareness for non-CHOPPY trending regimes ──
    # When BTC macro says BULL_TREND / BEAR_TREND but the symbol's own
    # 50-bar range is tight (accumulation / distribution box), the trend
    # extension multipliers produce overextended TP2/TP3 targets.
    # Apply a SOFT cap: TP2/TP3 can exceed the range boundary by a small
    # fraction, but not by the full trend-extension distance.
    # TP1 is left untouched — immediate targets are the strategy's job.
    if (
        has_range
        and regime in ("BULL_TREND", "BEAR_TREND")
    ):
        from config.constants import LocalRange as _LR
        range_size = range_high - range_low
        _range_pct = range_size / entry_mid if entry_mid > 0 else 1.0
        if _range_pct < _LR.RANGE_PCT_THRESHOLD and chop_strength >= _LR.MIN_CHOP_FOR_LOCAL:
            # Symbol is locally range-bound despite macro trend
            if is_long:
                # TP2 soft-capped at range_high + overshoot
                _tp2_cap = abs(range_high - entry_mid) + range_size * _LR.TP2_RANGE_OVERSHOOT
                _tp3_cap = abs(range_high - entry_mid) + range_size * _LR.TP3_RANGE_OVERSHOOT
                # Only cap if range_high is above entry — otherwise the cap
                # distance would be negative (entry already above range).
                if range_high > entry_mid:
                    if tp2_dist > _tp2_cap:
                        adjustments.append(
                            f"TP2 local-range cap ({_range_pct*100:.1f}% range)"
                        )
                        tp2_dist = _tp2_cap
                    if tp3_dist > _tp3_cap:
                        adjustments.append(
                            f"TP3 local-range cap ({_range_pct*100:.1f}% range)"
                        )
                        tp3_dist = _tp3_cap
            else:
                # SHORT: TP2/TP3 soft-capped at range_low - overshoot
                _tp2_cap = abs(entry_mid - range_low) + range_size * _LR.TP2_RANGE_OVERSHOOT
                _tp3_cap = abs(entry_mid - range_low) + range_size * _LR.TP3_RANGE_OVERSHOOT
                # Only cap if range_low is below entry — otherwise the cap
                # distance would be negative (entry already below range).
                if range_low < entry_mid:
                    if tp2_dist > _tp2_cap:
                        adjustments.append(
                            f"TP2 local-range cap ({_range_pct*100:.1f}% range)"
                        )
                        tp2_dist = _tp2_cap
                    if tp3_dist > _tp3_cap:
                        adjustments.append(
                            f"TP3 local-range cap ({_range_pct*100:.1f}% range)"
                        )
                        tp3_dist = _tp3_cap

    # ── V14: Cap TP3 at max 50% from entry price ──────────────
    _max_tp3_dist = entry_mid * 0.50
    if tp3_dist > _max_tp3_dist:
        # FIX-13: Cap the percentage display to avoid 10445193735% log noise for micro-price tokens
        _tp3_pct_raw = tp3_dist / entry_mid * 100 if entry_mid > 0 else 0
        _tp3_pct_display = f"{min(_tp3_pct_raw, 9999):.0f}" + ("%" if _tp3_pct_raw < 9999 else "%+ (overflow)")
        adjustments.append(f"TP3 capped: {_tp3_pct_display} → 50%")
        tp3_dist = _max_tp3_dist

    # ── 5c. Hard-cap target ladder to sane R multiples / positive prices ──
    _max_target_dist = sl_dist * MAX_TARGET_RR
    if not is_long:
        _min_positive_price = max(entry_mid * MIN_POSITIVE_PRICE_FRAC, 1e-8)
        _max_target_dist = min(_max_target_dist, max(0.0, entry_mid - _min_positive_price))
    if _max_target_dist > 0:
        _has_tp3 = tp3_dist > 0
        _tp1_cap = _max_target_dist - (2 * min_gap if _has_tp3 else min_gap)
        _tp2_cap = _max_target_dist - (min_gap if _has_tp3 else 0.0)
        if _tp1_cap > 0 and tp1_dist > _tp1_cap:
            adjustments.append(f"TP1 capped at {MAX_TARGET_RR:.1f}R ladder max")
            tp1_dist = _tp1_cap
        if _tp2_cap > 0 and tp2_dist > _tp2_cap:
            adjustments.append(f"TP2 capped at {MAX_TARGET_RR:.1f}R")
            tp2_dist = _tp2_cap
        if tp3_dist > 0 and tp3_dist > _max_target_dist:
            adjustments.append(f"TP3 capped at {MAX_TARGET_RR:.1f}R")
            tp3_dist = _max_target_dist

    # ── 6. Disable TP3 in VOLATILE ─────────────────────────────
    if not profile["tp3_enabled"]:
        tp3_dist = 0
        adjustments.append("TP3 disabled (regime)")

    # ── 7. Re-enforce minimum gaps after capping ───────────────
    if tp2_dist <= tp1_dist:
        tp2_dist = tp1_dist + min_gap
    if tp3_dist > 0 and tp3_dist <= tp2_dist:
        tp3_dist = tp2_dist + min_gap

    # ── 8. Convert distances back to prices ────────────────────
    if is_long:
        new_sl = entry_mid - sl_dist
        new_tp1 = entry_mid + tp1_dist
        new_tp2 = entry_mid + tp2_dist
        new_tp3 = entry_mid + tp3_dist if tp3_dist > 0 else None
    else:
        new_sl = entry_mid + sl_dist
        new_tp1 = entry_mid - tp1_dist
        new_tp2 = entry_mid - tp2_dist
        new_tp3 = entry_mid - tp3_dist if tp3_dist > 0 else None

    # ── 8a. Safety clamp — SL must never cross entry (compounding multipliers
    #        on large sl_dist can produce new_sl >= entry_low for LONGs or
    #        new_sl <= entry_high for SHORTs when adjustments compound).
    _epsilon = entry_mid * 0.0001  # 0.01% buffer
    if is_long:
        # LONG SL must be below the entry zone low
        sl_ceiling = entry_low - _epsilon
        if new_sl >= sl_ceiling:
            new_sl = sl_ceiling
            adjustments.append("SL clamped below entry_low (safety)")
    else:
        # SHORT SL must be above the entry zone high
        sl_floor = entry_high + _epsilon
        if new_sl <= sl_floor:
            new_sl = sl_floor
            adjustments.append("SL clamped above entry_high (safety)")

    # ── 9. Compute final R:R ───────────────────────────────────
    # FIX: use the POST-CLAMP distances so the reported rr_ratio matches the
    # actual SL/TP that will be emitted. Previously this used tp2_dist/sl_dist
    # which are pre-clamp values — after a safety clamp the real SL is tighter
    # and the real RR differs from the reported one, causing downstream gates
    # (RR floor, sizing) to act on stale numbers.
    effective_sl_dist = abs(entry_mid - new_sl)
    effective_tp2_dist = abs(new_tp2 - entry_mid) if new_tp2 is not None else 0.0
    rr = effective_tp2_dist / effective_sl_dist if effective_sl_dist > 0 else 0

    return AdjustedLevels(
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=round(new_sl, 10),
        tp1=round(new_tp1, 10),
        tp2=round(new_tp2, 10),
        tp3=round(new_tp3, 10) if new_tp3 else None,
        rr_ratio=round(rr, 2),
        regime=regime,
        adjustments=adjustments,
        trail_pct=trail_pct,
        trade_type=trade_type,
        local_structure_bias=local_structure.bias,
        local_structure_score=local_structure.score,
        local_structure_reason=local_structure.reason,
        local_structure_bars=local_structure.bars,
        local_structure_used_vwap=local_structure.used_vwap,
        local_structure_used_rejections=local_structure.used_rejections,
    )


def get_trail_pct(regime: str) -> float:
    """Get trailing stop percentage for current regime."""
    return REGIME_PROFILES.get(regime, DEFAULT_PROFILE).get("trail_pct", 0.40)
