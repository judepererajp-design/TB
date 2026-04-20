"""Lightweight helpers for early counter-trend hard rejection."""

COUNTER_TREND_HARD_REJECT_STRATEGIES = {
    "SmartMoneyConcepts",
    "InstitutionalBreakout",
    "Ichimoku",
    "ElliottWave",
    "WyckoffAccDist",
    "GeometricPattern",
}


def should_hard_reject_counter_trend_signal(strategy_name: str, direction: str, regime_name: str) -> bool:
    """Only hard-reject when weekly HTF bias confirms the intraday trend regime."""
    if strategy_name not in COUNTER_TREND_HARD_REJECT_STRATEGIES:
        return False
    try:
        from analyzers.htf_guardrail import htf_guardrail as _ctr_htf
        weekly_bias = getattr(_ctr_htf, "_weekly_bias", "NEUTRAL")
    except Exception:
        weekly_bias = "NEUTRAL"
    if regime_name == "BULL_TREND" and weekly_bias == "BULLISH":
        return direction == "SHORT"
    if regime_name == "BEAR_TREND" and weekly_bias == "BEARISH":
        return direction == "LONG"
    return False
