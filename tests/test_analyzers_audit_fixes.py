"""
Tests covering the analyzers-folder audit fixes:

1. derivatives.funding_delta_trend is wired into bias & squeeze risk
2. execution_gate no longer uses locals().get() — positioning defaults apply
   when called without a context dict.
3. htf_guardrail._btc_4h_resuming_down is consumed by _should_hard_block to
   lower the counter-trend SHORT threshold (symmetric to _btc_4h_bouncing).
4. strategies.base.detect_parabolic optionally delegates to
   analyzers.parabolic_detector when ohlcv is provided — exhaustion fields
   and confidence_penalty appear in the returned dict.
"""
from __future__ import annotations

import numpy as np

from analyzers.derivatives import DerivativesAnalyzer, DerivativesData


# ──────────────────────────────────────────────────────────────────────────
# derivatives.funding_delta_trend wiring
# ──────────────────────────────────────────────────────────────────────────
def test_funding_delta_rising_flips_bias_to_bearish():
    """RISING funding at HIGH level with long-heavy L/S → BEARISH fade bias."""
    analyzer = DerivativesAnalyzer()
    data = DerivativesData(
        symbol="X/USDT",
        funding_trend="HIGH",
        funding_delta_trend="RISING",
        lsr_trend="LONG_HEAVY",
        oi_trend="NEUTRAL",
    )
    assert analyzer._classify_signal_bias(data) == "BEARISH"
    # A note explaining the fade should be appended so operators see it.
    assert any("Funding rising into HIGH" in n for n in data.notes)


def test_funding_delta_falling_while_negative_flips_bias_to_bullish():
    """FALLING funding while NEGATIVE with short-heavy L/S → BULLISH squeeze fuel."""
    analyzer = DerivativesAnalyzer()
    data = DerivativesData(
        symbol="X/USDT",
        funding_trend="NEGATIVE",
        funding_delta_trend="FALLING",
        lsr_trend="SHORT_HEAVY",
        oi_trend="NEUTRAL",
    )
    assert analyzer._classify_signal_bias(data) == "BULLISH"
    assert any("Funding falling" in n for n in data.notes)


def test_funding_delta_boosts_squeeze_risk_when_aligned_with_positioning():
    """FALLING funding + short-heavy L/S should upgrade squeeze_risk vs. the
    same inputs without the delta alignment."""
    analyzer = DerivativesAnalyzer()
    base = DerivativesData(
        symbol="X/USDT",
        funding_trend="NEUTRAL",          # not NEGATIVE — isolate the delta effect
        funding_delta_trend="FLAT",
        lsr_trend="SHORT_HEAVY",
        oi_trend="NEUTRAL",
    )
    with_delta = DerivativesData(
        symbol="X/USDT",
        funding_trend="NEUTRAL",
        funding_delta_trend="FALLING",    # delta now aligned
        lsr_trend="SHORT_HEAVY",
        oi_trend="NEUTRAL",
    )
    base_risk = analyzer._assess_squeeze_risk(base)
    aligned_risk = analyzer._assess_squeeze_risk(with_delta)
    # Ranking: LOW < MEDIUM < HIGH
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    assert order[aligned_risk] > order[base_risk]


def test_funding_delta_classifier_flat_with_single_sample():
    """Delta classifier must return FLAT until ≥2 samples are collected."""
    analyzer = DerivativesAnalyzer()
    # First poll — only one sample in the deque → FLAT.
    assert analyzer._classify_funding_delta("Y/USDT", 0.01) == "FLAT"
    # After a clearly-rising second sample, we should see RISING.
    assert analyzer._classify_funding_delta("Y/USDT", 0.03) == "RISING"


# ──────────────────────────────────────────────────────────────────────────
# execution_gate — no more locals().get() antipattern
# ──────────────────────────────────────────────────────────────────────────
def test_execution_gate_evaluate_without_context_uses_defaults():
    """Calling evaluate() without any context dict must not raise and must
    compute a positioning factor from the up-front defaults (was previously
    relying on locals().get fallbacks which made it unclear what the caller
    would see)."""
    from analyzers.execution_gate import execution_gate

    assessment = execution_gate.evaluate(
        direction="LONG",
        grade="A",
        confidence=75.0,
        symbol="X/USDT",
        session_name="London",
        trigger_quality_score=0.8,
        spread_bps=5.0,
        volume_score=60.0,
    )
    # Positioning factor must be present and a valid 0-100 score
    assert "positioning" in assessment.factors
    assert 0.0 <= assessment.factors["positioning"] <= 100.0


def test_execution_gate_evaluate_source_has_no_locals_antipattern():
    """Guard against regressions: the `locals().get(...)` antipattern must
    not reappear in the positioning block."""
    import inspect
    import analyzers.execution_gate as eg_mod

    src = inspect.getsource(eg_mod.ExecutionQualityGate.evaluate)
    # Strip out comments so the historical note explaining the fix (which
    # mentions locals().get) doesn't trigger the guard.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "locals().get" not in code_only, (
        "execution_gate.evaluate should not use locals().get() — use the "
        "hoisted defaults at the top of the function instead."
    )


# ──────────────────────────────────────────────────────────────────────────
# htf_guardrail._btc_4h_resuming_down is now consumed
# ──────────────────────────────────────────────────────────────────────────
def test_htf_resuming_down_flag_lowers_counter_trend_short_threshold():
    """When the weekly bias is BULLISH (weekly-bull regime) and BTC 4h is
    resuming down, the counter-trend SHORT threshold should be lowered to 78
    — symmetric to the existing bounce-based long reduction."""
    from analyzers.htf_guardrail import htf_guardrail

    # Set up a strong BULL weekly with 4h resuming-down pattern
    htf_guardrail._warmed = True   # is_hard_blocked short-circuits until warmed
    htf_guardrail._weekly_bias = "BULLISH"
    htf_guardrail._weekly_adx = 40.0
    htf_guardrail._btc_4h_bouncing = False
    htf_guardrail._btc_4h_resuming_down = True
    # Confidence of 80 would be rejected under the plain formula (threshold ~82)
    # but allowed under the reduced 78 threshold.
    blocked, reason = htf_guardrail.is_hard_blocked(
        signal_direction="SHORT",
        raw_confidence=80.0,
        strategy_name="PriceAction",
    )
    assert blocked is False, f"Expected override with resuming-down flag, got: {reason}"

    # Without the flag, the same 80 should be blocked at this ADX
    htf_guardrail._btc_4h_resuming_down = False
    blocked2, reason2 = htf_guardrail.is_hard_blocked(
        signal_direction="SHORT",
        raw_confidence=80.0,
        strategy_name="PriceAction",
    )
    assert blocked2 is True, f"Expected block without flag, got: {reason2}"


# ──────────────────────────────────────────────────────────────────────────
# strategies.base.detect_parabolic — parabolic_detector enrichment path
# ──────────────────────────────────────────────────────────────────────────
def test_detect_parabolic_returns_exhaustion_fields_when_ohlcv_provided():
    """detect_parabolic must return the new exhaustion-related keys when
    ohlcv is supplied, delegating to analyzers.parabolic_detector."""
    from strategies.base import BaseStrategy

    # Construct a parabolic UP sequence with a late climactic volume candle
    closes = np.array([100.0 * (1.0 + 0.012) ** i for i in range(40)])
    ohlcv = []
    for i, c in enumerate(closes):
        vol = 1000.0 if i < 38 else 5000.0  # volume climax on the last candle
        ohlcv.append([i, c - 0.05, c + 0.5, c - 0.5, c, vol])

    result = BaseStrategy.detect_parabolic(closes, ohlcv=ohlcv, direction="LONG")
    # The enrichment path must populate these keys (defaults were set).
    assert "is_exhausted" in result
    assert "exhaustion_signals" in result
    assert "confidence_penalty" in result
    assert isinstance(result["exhaustion_signals"], list)
    assert isinstance(result["confidence_penalty"], int)


def test_detect_parabolic_backward_compatible_without_ohlcv():
    """Callers that don't pass ohlcv must get the same basic keys as before,
    with the new fields defaulted (empty/false/0)."""
    from strategies.base import BaseStrategy

    closes = np.array([100.0 + i * 0.5 for i in range(40)])
    result = BaseStrategy.detect_parabolic(closes)
    # Base keys still present
    for key in ("is_parabolic", "acceleration", "roc", "direction"):
        assert key in result
    # New keys default-populated — no error, no enrichment call
    assert result.get("is_exhausted") is False
    assert result.get("exhaustion_signals") == []
    assert result.get("confidence_penalty") == 0


def test_volume_profile_ignores_all_zero_volume_input():
    """Guard against regressions: zero-total-volume must short-circuit before
    fabricating value-area output from empty profile bins."""
    import inspect
    from analyzers.volume import VolumeAnalyzer

    src = inspect.getsource(VolumeAnalyzer._calculate_volume_profile)
    assert "if total_volume <= 0:" in src
