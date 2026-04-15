from dataclasses import replace
from unittest.mock import MagicMock, patch

from analyzers.market_state_engine import (
    BREAKOUT_EXEMPTION_MAX_AGE_SECS,
    MarketState,
    MarketStateEngine,
)
from analyzers.regime_transition import (
    RegimeTransitionDetector,
    TransitionResult,
    amplify_transition_with_news,
    get_trade_type_transition_penalty,
    has_active_breakout_exemption,
    should_block_countertrend_pullback,
)


class TestRegimeTransitionDetectorHardening:
    def test_unstable_transition_penalty_is_hardened_and_escalates(self):
        detector = RegimeTransitionDetector()

        detector.record("BULL_TREND", 0.20, 24.0)
        detector.record("CHOPPY", 0.55, 23.0)
        detector.record("BEAR_TREND", 0.45, 22.0)
        first = detector.evaluate(
            current_regime="BEAR_TREND",
            chop_strength=0.45,
            vol_ratio=1.0,
            adx=22.0,
        )

        detector.record("CHOPPY", 0.60, 21.0)
        second = detector.evaluate(
            current_regime="CHOPPY",
            chop_strength=0.60,
            vol_ratio=1.0,
            adx=21.0,
        )

        assert first.transition_type == "UNSTABLE"
        assert first.confidence_increase >= 12
        assert first.cycles_in_transition == 1
        assert second.transition_type == "UNSTABLE"
        assert second.confidence_increase > first.confidence_increase
        assert second.cycles_in_transition == 2
        assert "persisting 2 cycles" in second.warning

    def test_pullback_blocking_preserves_local_continuations(self):
        transition = TransitionResult(
            in_transition=True,
            transition_type="TREND→CHOP",
            risk_level=0.75,
            confidence_increase=10,
            volume_reduction=0.6,
            warning="warn",
        )

        assert should_block_countertrend_pullback("PULLBACK_SHORT", transition, "noise") is True
        assert should_block_countertrend_pullback("LOCAL_CONTINUATION_SHORT", transition, "noise") is False
        assert should_block_countertrend_pullback(
            "PULLBACK_SHORT",
            replace(transition, breakout_exemption_active=True),
            "breakout",
        ) is False
        assert should_block_countertrend_pullback(
            "PULLBACK_SHORT",
            replace(transition, transition_type="CHOP→EXPANSION"),
            "noise",
        ) is False

    def test_trade_type_penalty_scaling_reduces_trend_penalty_only(self):
        transition = TransitionResult(
            in_transition=True,
            transition_type="UNSTABLE",
            risk_level=0.70,
            confidence_increase=12,
            volume_reduction=0.6,
            warning="warn",
        )

        assert get_trade_type_transition_penalty("TREND_LONG", transition) == 8
        assert get_trade_type_transition_penalty("LOCAL_CONTINUATION_LONG", transition) == 12
        assert get_trade_type_transition_penalty("PULLBACK_LONG", transition) == 12

    def test_escalation_continues_beyond_four_cycles_with_slower_scaling(self):
        detector = RegimeTransitionDetector()
        penalties = []

        for idx in range(7):
            detector.record(f"REGIME_{idx}", 0.50, 20.0)
            result = detector.evaluate(
                current_regime=f"REGIME_{idx}",
                chop_strength=0.50,
                vol_ratio=1.0,
                adx=20.0,
            )
            penalties.append(result.confidence_increase)

        assert penalties[4] > penalties[3]
        assert penalties[5] >= penalties[4]
        assert penalties[6] >= penalties[5]
        assert penalties[6] - penalties[5] <= 1

    def test_news_amplifies_bearish_transition_penalty_when_regime_is_bullish(self):
        transition = TransitionResult(
            in_transition=True,
            transition_type="TREND→CHOP",
            risk_level=0.75,
            confidence_increase=10,
            volume_reduction=0.6,
            warning="warn",
        )

        amplified = amplify_transition_with_news(
            transition_result=transition,
            regime_name="BULL_TREND",
            news_bias="BEARISH",
            news_score=-0.8,
        )

        assert amplified.confidence_increase > transition.confidence_increase
        assert amplified.risk_level > transition.risk_level
        assert "BEARISH news score" in amplified.warning

    def test_news_does_not_amplify_when_bias_matches_regime(self):
        transition = TransitionResult(
            in_transition=True,
            transition_type="TREND→CHOP",
            risk_level=0.75,
            confidence_increase=10,
            volume_reduction=0.6,
            warning="warn",
        )

        amplified = amplify_transition_with_news(
            transition_result=transition,
            regime_name="BEAR_TREND",
            news_bias="BEARISH",
            news_score=-0.8,
        )

        assert amplified == transition


class TestRegimeWarningBackstop:
    def test_transition_warning_uses_internal_backstop_when_external_feeds_missing(self):
        with patch.dict(
            "sys.modules",
            {
                "data.api_client": MagicMock(api=MagicMock()),
                "ccxt": MagicMock(),
                "ccxt.async_support": MagicMock(),
            },
        ):
            from analyzers.regime import Regime, RegimeAnalyzer

            analyzer = RegimeAnalyzer()
            analyzer._regime = Regime.BULL_TREND
            analyzer._chop_strength = 0.55
            analyzer._btc_atr_ratio = 1.0
            analyzer._btc_adx = 21.0
            analyzer._adx_history = [28.0, 27.0, 26.0, 24.0, 21.0]

            analyzer._refresh_transition_warning()
            tw = analyzer.get_transition_warning()

        assert tw["warning"] is True
        assert any("Backstop:TREND→CHOP" in f for f in tw["factors"])


class TestMarketStateBreakoutStaleness:
    def test_recent_compression_resolution_is_breakout(self):
        engine = MarketStateEngine()
        now = 1_000_000.0
        engine._state_history = [
            (now - 60, MarketState.COMPRESSION),
            (now, MarketState.EXPANSION),
        ]

        assert engine.get_transition_type() == "breakout"

    def test_stale_breakout_resolution_falls_back_to_stable(self):
        engine = MarketStateEngine()
        now = 1_000_000.0
        stale_gap = BREAKOUT_EXEMPTION_MAX_AGE_SECS + 1
        engine._state_history = [
            (now - stale_gap, MarketState.COMPRESSION),
            (now - 60, MarketState.EXPANSION),
            (now, MarketState.EXPANSION),
        ]

        assert engine.get_transition_type() == "stable"

    def test_breakout_exemption_expires_after_two_faded_cycles(self):
        detector = RegimeTransitionDetector()

        for adx in (12.0, 14.0, 16.0, 18.0, 20.0):
            detector.record("CHOPPY", 0.60, adx)
        breakout = detector.evaluate(
            current_regime="CHOPPY",
            chop_strength=0.60,
            vol_ratio=1.6,
            adx=20.0,
        )

        detector.record("BULL_TREND", 0.40, 20.0)
        first_fade = detector.evaluate(
            current_regime="BULL_TREND",
            chop_strength=0.40,
            vol_ratio=1.2,
            adx=20.0,
        )

        detector.record("BULL_TREND", 0.38, 19.0)
        second_fade = detector.evaluate(
            current_regime="BULL_TREND",
            chop_strength=0.38,
            vol_ratio=1.1,
            adx=19.0,
        )

        assert breakout.transition_type == "CHOP→EXPANSION"
        assert has_active_breakout_exemption("breakout", breakout) is True
        assert first_fade.in_transition is False
        assert has_active_breakout_exemption("breakout", first_fade) is True
        assert second_fade.transition_type == "EXPANSION→CHOP"
        assert has_active_breakout_exemption("breakout", second_fade) is False
