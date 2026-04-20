from governance.performance_tracker import PerformanceTracker


def test_suppress_strategy_auto_registers_unknown_strategy():
    tracker = PerformanceTracker()

    tracker.suppress_strategy("ElliottWave", duration_mins=5)

    assert "ElliottWave" in tracker._stats
    assert tracker.is_strategy_disabled("ElliottWave") is True


def test_engine_counter_trend_hard_reject_requires_confirmed_weekly_bias(monkeypatch):
    from core.counter_trend_guard import should_hard_reject_counter_trend_signal

    monkeypatch.setattr("analyzers.htf_guardrail.htf_guardrail._weekly_bias", "NEUTRAL")
    assert should_hard_reject_counter_trend_signal("WyckoffAccDist", "SHORT", "BULL_TREND") is False

    monkeypatch.setattr("analyzers.htf_guardrail.htf_guardrail._weekly_bias", "BULLISH")
    assert should_hard_reject_counter_trend_signal("WyckoffAccDist", "SHORT", "BULL_TREND") is True


def test_engine_counter_trend_hard_reject_requires_bearish_weekly_confirmation(monkeypatch):
    from core.counter_trend_guard import should_hard_reject_counter_trend_signal

    monkeypatch.setattr("analyzers.htf_guardrail.htf_guardrail._weekly_bias", "NEUTRAL")
    assert should_hard_reject_counter_trend_signal("SmartMoneyConcepts", "LONG", "BEAR_TREND") is False

    monkeypatch.setattr("analyzers.htf_guardrail.htf_guardrail._weekly_bias", "BEARISH")
    assert should_hard_reject_counter_trend_signal("SmartMoneyConcepts", "LONG", "BEAR_TREND") is True


def test_engine_counter_trend_hard_reject_skips_strategies_not_in_list(monkeypatch):
    from core.counter_trend_guard import should_hard_reject_counter_trend_signal

    monkeypatch.setattr("analyzers.htf_guardrail.htf_guardrail._weekly_bias", "BULLISH")
    assert should_hard_reject_counter_trend_signal("PriceAction", "SHORT", "BULL_TREND") is False
