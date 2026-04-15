"""
TitanBot Pro — Dynamic Regime Thresholds (v2: Continuous Weight Interpolation)
================================================================================
Instead of hard regime switches, this uses continuous weight interpolation
based on chop_strength (0.0 = trending, 1.0 = pure chop).

Markets transition gradually — hard switches cause whipsaw.
Institutional systems always scale weights continuously.

Key changes from v1:
  - Strategy weights interpolate based on chop_strength
  - Range position detection: only trade outer 25% during high chop
  - Position size scales down proportionally to chop_strength
  - Learning loop decay slows in chop (prevents noise contamination)
"""

import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ── Base regime thresholds (discrete regimes still used as foundation) ──

REGIME_THRESHOLDS: Dict[str, Dict] = {
    "BULL_TREND": {
        "min_confidence": 68,
        "max_signals_per_hour": 6,
        "strategies": None,  # All allowed in trends
        "long_bias": 1.1,
        "short_bias": 0.82,             # FIX (P1-B): was 0.9 — too lenient, allowed 65% SHORTs in bull.
                                         # Now 0.82 (symmetric-ish with BEAR_TREND long_bias=0.75).
                                         # A 78-conf SHORT × 0.82 = 63.9, below 68 floor. Requires ~83+ to pass.
                                         # Shorts are still allowed (GPT: fine IF controlled), just need higher conviction.
        "counter_short_max": 3,          # FIX (P1-A): Portfolio cap — max 3 SHORT positions in bull trend.
                                         # Was 4, reduced per review: correlated alts move together,
                                         # 4 concurrent SHORTs = portfolio-wide stopout risk.
                                         # 3 allows legitimate pullback scalps, limits correlated exposure.
                                         # BEAR_TREND has counter_long_max=2 (asymmetric: longs in bear are riskier).
    },
    "BEAR_TREND": {
        "min_confidence": 75,       # Raised from 72 — require higher bar in bear
        "max_signals_per_hour": 5,
        "strategies": None,
        "long_bias": 0.75,          # Stronger LONG penalty (was 0.88) — counter-trend longs take bigger confidence hit
        "short_bias": 1.15,         # Small boost for shorts (was 1.12) — reward trend-aligned direction
        "counter_long_max": 2,      # Portfolio cap: max 2 LONG positions in bear trend
    },
    "CHOPPY": {
        "min_confidence": 76,       # FIX CHOP-FLOOR: was 82 → 80 → 76 (let HTF-aligned shorts through)
        # Adaptive floor interpolation via get_adaptive_confidence_floor():
        #   floor = trend_floor + (min_confidence - trend_floor) × chop_strength
        # At chop=0.33:  60 + (76-60)*0.33 = 65.3  → allows 65+ signals in mild chop
        # At chop=0.50:  60 + (76-60)*0.50 = 68.0
        # At chop=1.00:  60 + (76-60)*1.00 = 76.0  → full floor in pure chop
        # The weighted scoring pipeline (volume/derivatives/orderflow) still has to
        # agree, so a 65 raw-strategy signal won't pass unless the market structure backs it.
        "trend_floor": 60,          # Floor at no chop — lowered to let more signals reach EV gate
        "max_signals_per_hour": 3,
        "strategies": None,  # Don't hard-block — use weight interpolation instead
        "long_bias": 1.0,
        "short_bias": 1.0,
    },
    "VOLATILE": {
        "min_confidence": 78,
        "max_signals_per_hour": 4,
        "strategies": None,
        "long_bias": 1.0,
        "short_bias": 1.0,
    },
    "VOLATILE_PANIC": {
        # Extreme panic — only highest-quality setups
        "min_confidence": 90,
        "max_signals_per_hour": 2,
        "strategies": ["SmartMoneyConcepts", "WyckoffAccDist", "Wyckoff", "FundingRateArb"],
        "long_bias": 0.85,
        "short_bias": 1.0,
    },
    "LOW_VOL": {
        "min_confidence": 75,
        "max_signals_per_hour": 4,
        "strategies": None,
        "long_bias": 1.0,
        "short_bias": 1.0,
    },
}

DEFAULT_THRESHOLD = {
    "min_confidence": 75,
    "max_signals_per_hour": 5,
    "strategies": None,
    "long_bias": 1.0,
    "short_bias": 1.0,
}


# ── Strategy weight multipliers by chop_strength ──────────────
# These interpolate: weight = base + (chop_mult - base) * chop_strength

STRATEGY_CHOP_WEIGHTS: Dict[str, Dict[str, float]] = {
    # strategy_name: {trend_weight, chop_weight}
    # Trend-following strategies get penalized in chop
    "SmartMoneyConcepts":     {"trend": 1.0, "chop": 0.85},  # SMC works in both, slight penalty
    "InstitutionalBreakout":  {"trend": 1.0, "chop": 0.45},  # Breakouts fail in chop
    "Momentum":               {"trend": 1.0, "chop": 0.40},  # Momentum fails in chop
    "MomentumContinuation":   {"trend": 1.0, "chop": 0.40},  # Legacy alias
    "ElliottWave":            {"trend": 1.0, "chop": 0.35},  # Wave counts unreliable in chop
    "Ichimoku":               {"trend": 1.0, "chop": 0.50},  # Cloud signals whipsaw in chop
    "IchimokuCloud":          {"trend": 1.0, "chop": 0.50},  # Legacy alias

    # Range-friendly strategies get boosted in chop
    "MeanReversion":          {"trend": 0.75, "chop": 1.30},  # Mean reversion thrives in chop
    "ExtremeReversal":        {"trend": 0.80, "chop": 1.20},  # Reversal at range edges
    "PriceAction":            {"trend": 1.0, "chop": 0.90},  # Neutral — works in both
    "FundingRateArb":         {"trend": 1.0, "chop": 1.10},  # Funding arb unaffected by chop
    "RangeScalper":           {"trend": 0.30, "chop": 1.35}, # ONLY trades chop — max boost
    "WyckoffAccDist":         {"trend": 1.0, "chop": 0.80},  # Wyckoff structural, slight chop penalty
}


def get_regime_threshold(regime: str) -> Dict:
    """Get thresholds for a given regime"""
    return REGIME_THRESHOLDS.get(regime, DEFAULT_THRESHOLD)


def get_strategy_chop_weight(strategy: str, chop_strength: float) -> float:
    """
    Get interpolated strategy weight based on chop_strength.

    chop_strength=0.0 → full trend weight
    chop_strength=1.0 → full chop weight
    Anything between → smooth interpolation

    Returns: multiplier (0.25 to 1.35)

    Weights can be overridden in settings.yaml under:
      regime_thresholds.chop_weights.<StrategyName>.trend / .chop
    """
    # Allow settings.yaml overrides (e.g. after paper-trading reveals actual chop performance)
    try:
        from config.loader import cfg
        cfg_weights = cfg.get('regime_thresholds', {}).get('chop_weights', {}).get(strategy)
        if cfg_weights:
            trend_w = cfg_weights.get('trend', STRATEGY_CHOP_WEIGHTS.get(strategy, {}).get('trend', 1.0))
            chop_w  = cfg_weights.get('chop',  STRATEGY_CHOP_WEIGHTS.get(strategy, {}).get('chop',  1.0))
        else:
            weights = STRATEGY_CHOP_WEIGHTS.get(strategy, {"trend": 1.0, "chop": 1.0})
            trend_w = weights["trend"]
            chop_w  = weights["chop"]
    except Exception:
        weights = STRATEGY_CHOP_WEIGHTS.get(strategy, {"trend": 1.0, "chop": 1.0})
        trend_w = weights["trend"]
        chop_w  = weights["chop"]

    # Linear interpolation
    interpolated = trend_w + (chop_w - trend_w) * chop_strength
    return round(max(0.25, min(1.35, interpolated)), 3)


def get_adaptive_confidence_floor(regime: str, chop_strength: float) -> float:
    """
    Interpolate the minimum confidence floor based on chop_strength.

    For CHOPPY regime: blends between trend_floor (mild chop) and
    min_confidence (pure chop) so the floor scales continuously
    — the same way strategy weights already do.

    Example (with current values trend_floor=60, min_confidence=76):
             chop=0.25 → floor = 60 + 16*0.25 = 64.0
             chop=0.50 → floor = 60 + 16*0.50 = 68.0
             chop=1.00 → floor = 60 + 16*1.00 = 76.0

    All other regimes return their flat min_confidence unchanged.
    """
    thresholds = get_regime_threshold(regime)
    min_conf = thresholds["min_confidence"]

    trend_floor = thresholds.get("trend_floor")
    if trend_floor is None:
        # Non-CHOPPY regimes: flat floor, no interpolation needed
        return float(min_conf)

    interpolated = trend_floor + (min_conf - trend_floor) * chop_strength
    return round(interpolated, 1)


def get_chop_size_multiplier(chop_strength: float) -> float:
    """
    Position size reduction in chop.
    chop_strength=0 → full size (1.0)
    chop_strength=1 → half size (0.5)

    Smooth interpolation, not a hard switch.
    """
    return round(1.0 - 0.5 * chop_strength, 3)


def get_chop_learning_decay(chop_strength: float, base_decay: float = 0.995) -> float:
    """
    Slow learning loop decay during chop to prevent noise contamination.
    High chop → slower decay → system doesn't learn from noise.

    chop_strength=0 → normal decay (0.995)
    chop_strength=1 → slow decay (0.999)
    """
    return round(base_decay + 0.004 * chop_strength, 4)


def is_in_range_outer_zone(
    price: float,
    range_high: float,
    range_low: float,
    outer_pct: float = 0.25,
) -> Tuple[bool, str]:
    """
    Check if price is in the outer zone of the trading range.
    Only trade outer 25% of range during chop.

    Returns: (is_in_outer, zone_description)
    """
    if range_high <= range_low or range_high == 0:
        return True, "no_range"  # Can't determine range — allow trade

    range_size = range_high - range_low
    outer_band = range_size * outer_pct

    # Upper zone: near supply
    if price >= range_high - outer_band:
        return True, "supply_zone"

    # Lower zone: near demand
    if price <= range_low + outer_band:
        return True, "demand_zone"

    # Mid-range: avoid in chop
    return False, "mid_range"


def apply_regime_filter(
    confidence: float,
    direction: str,
    strategy: str,
    regime: str,
    chop_strength: float = 0.0,
    htf_aligned: bool = False,
) -> Tuple[float, bool, str]:
    """
    Apply regime-based adjustments to a signal's confidence.
    Now uses continuous weight interpolation for chop handling,
    including an interpolated confidence floor (not a hard discrete value).

    htf_aligned: True when the signal direction matches the weekly HTF bias
    (e.g. SHORT during weekly BEARISH, LONG during weekly BULLISH).
    When True, the directional bias penalty is skipped — it makes no sense
    to penalise a SHORT in BULL_TREND when the weekly trend is BEARISH.
    This prevents the HTF/regime deadlock where LONGs are hard-blocked AND
    SHORTs are penalised below the confidence floor simultaneously.

    Returns:
        (adjusted_confidence, should_block, reason)
    """
    thresholds = get_regime_threshold(regime)
    min_conf = get_adaptive_confidence_floor(regime, chop_strength)  # ← interpolated floor

    # Check if strategy is hard-blocked in this regime (only VOLATILE does this)
    allowed = thresholds.get("strategies")
    if allowed is not None and strategy not in allowed:
        return confidence, True, (
            f"🚫 {strategy} not allowed in {regime} regime "
            f"(allowed: {', '.join(allowed)})"
        )

    # Apply directional bias — but skip the penalty when the signal is aligned
    # with the weekly HTF trend. A SHORT during weekly BEARISH is the *right*
    # trade; applying BULL_TREND's short_bias=0.9 would fight the HTF signal.
    bias_key = "long_bias" if direction == "LONG" else "short_bias"
    raw_bias = thresholds.get(bias_key, 1.0)
    if htf_aligned and raw_bias < 1.0:
        # Use neutral bias — don't penalise HTF-aligned direction
        bias = 1.0
        logger.debug(
            f"HTF-aligned {direction} in {regime}: bias override "
            f"{raw_bias:.2f} → 1.0 (deadlock prevention)"
        )
    else:
        bias = raw_bias
    adjusted = confidence * bias

    # ⭐ Apply chop weight interpolation (continuous, not hard switch)
    chop_weight = get_strategy_chop_weight(strategy, chop_strength)
    if htf_aligned and chop_weight < 1.0:
        # HTF-aligned signals have weekly trend as macro confirmation.
        # Lower-TF chop is noise relative to a weekly directional bias — don't penalise.
        # The weekly guardrail already handles risk management for these signals.
        chop_weight = 1.0
        logger.debug(
            f"HTF-aligned: chop_weight bypassed (deadlock prevention)"
        )
    adjusted *= chop_weight

    # Check minimum confidence.
    # HTF-aligned signals get a reduced floor: their weekly trend confirmation
    # provides macro edge that the lower-TF regime label doesn't capture.
    # Without this, a SHORT that is perfectly aligned with a weekly BEARISH trend
    # can be blocked purely because the 1h/4h regime is BULL_TREND — a deadlock.
    # Floor reduction: 10 pts (e.g. 68 → 58). Still requires meaningful confidence;
    # just doesn't punish signals for the lower-TF/HTF conflict.
    effective_floor = max(52.0, min_conf - 14.0) if htf_aligned else min_conf
    if adjusted < effective_floor:
        floor_note = f" (HTF-aligned floor: {effective_floor:.0f})" if htf_aligned else ""
        return adjusted, True, (
            f"🚫 Confidence {confidence:.0f}→{adjusted:.0f} below {regime} floor ({effective_floor:.0f}){floor_note} "
            f"[chop={chop_strength:.2f}, weight={chop_weight:.2f}]"
        )

    # ── Risk-on modifier: applies when macro confirms strong risk appetite ──
    # F&G>70 + alt_season>65 + macro_risk_on (DXY falling + SPX strong + VIX low)
    # These three together = genuine cross-asset risk appetite, not just BTC noise
    # Effect: raise long_bias 1.1→1.2× and reduce confidence floor 3pt
    try:
        from analyzers.regime import regime_analyzer as _ro_ra
        _ro_fg = _ro_ra.fear_greed
        _ro_alt = _ro_ra.alt_season_score
        _ro_macro = getattr(_ro_ra, '_macro_risk_on', False)
        _ro_active = (
            _ro_fg > 70
            and _ro_alt > 65
            and direction == "LONG"
            and regime in ("BULL_TREND", "CHOPPY")  # Only in non-bear regimes
            and (_ro_macro or (_ro_fg > 80 and _ro_alt > 70))  # Macro OR strong internal signals
        )
        if _ro_active and bias >= 1.0:
            # Risk-on: amplify the long_bias by an additional 0.10× (1.1 → 1.2 in BULL)
            _ro_boost = 0.10
            adjusted = adjusted * (1.0 + _ro_boost / bias)  # Proportional boost
            effective_floor = max(52.0, effective_floor - 3)  # 3pt lower floor
            logger.debug(
                f"Risk-on modifier active: F&G={_ro_fg} alt={_ro_alt} "
                f"macro={_ro_macro} → +{_ro_boost:.2f}× bias, floor-3"
            )
    except Exception:
        pass

    # Build reason string
    reason = (
        f"✅ {regime} regime: conf {confidence:.0f}→{adjusted:.0f} "
        f"(bias={bias:.2f}, chop_w={chop_weight:.2f}"
        + (f", htf_aligned_floor={effective_floor:.0f}" if htf_aligned else "")
        + ")"
    )
    logger.debug(reason)

    return adjusted, False, reason
