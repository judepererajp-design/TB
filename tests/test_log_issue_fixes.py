from governance.performance_tracker import PerformanceTracker


def test_suppress_strategy_auto_registers_unknown_strategy():
    tracker = PerformanceTracker()

    tracker.suppress_strategy("ElliottWave", duration_mins=5)

    assert "ElliottWave" in tracker._stats
    assert tracker.is_strategy_disabled("ElliottWave") is True
