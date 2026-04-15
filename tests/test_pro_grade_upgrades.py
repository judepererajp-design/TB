import asyncio
from unittest.mock import AsyncMock


def test_effective_exit_pnl_r_realizes_tp1_partial():
    from signals.outcome_monitor import OutcomeMonitor, TrackedSignal, SignalState

    monitor = OutcomeMonitor()
    tracked = TrackedSignal(
        signal_id=1,
        symbol="TESTUSDT",
        direction="LONG",
        strategy="PriceAction",
        entry_low=100.0,
        entry_high=100.0,
        stop_loss=98.0,
        tp1=102.0,
        tp2=104.0,
        tp3=106.0,
        confidence=80.0,
        state=SignalState.BE_ACTIVE,
        entry_price=100.0,
    )

    pnl_r = monitor._calc_effective_exit_pnl_r(tracked, 101.0)

    assert round(pnl_r, 2) == 0.75


def test_candle_entry_fill_marks_late_entry_when_zone_was_touched():
    from signals.outcome_monitor import OutcomeMonitor, TrackedSignal

    monitor = OutcomeMonitor()
    monitor._fetch_recent_candle = AsyncMock(
        return_value=[0, 100.0, 102.5, 99.5, 102.0, 1_000.0]
    )
    tracked = TrackedSignal(
        signal_id=2,
        symbol="TESTUSDT",
        direction="LONG",
        strategy="PriceAction",
        entry_low=100.0,
        entry_high=101.0,
        stop_loss=98.0,
        tp1=103.0,
        tp2=105.0,
        tp3=None,
        confidence=80.0,
    )

    fill_price, status = asyncio.run(monitor._get_candle_entry_fill(tracked, 102.0))

    assert fill_price == 101.0
    assert status == "LATE"


def test_stop_hunt_wick_is_ignored_when_candle_recovers():
    from signals.outcome_monitor import OutcomeMonitor, TrackedSignal

    monitor = OutcomeMonitor()
    monitor._fetch_recent_candle = AsyncMock(
        return_value=[0, 100.0, 101.0, 97.8, 99.4, 1_000.0]
    )
    tracked = TrackedSignal(
        signal_id=3,
        symbol="TESTUSDT",
        direction="LONG",
        strategy="PriceAction",
        entry_low=100.0,
        entry_high=100.0,
        stop_loss=98.0,
        tp1=102.0,
        tp2=104.0,
        tp3=None,
        confidence=80.0,
    )

    assert asyncio.run(monitor._is_stop_hunt_wick(tracked, 98.0)) is True


def test_learning_loop_stats_report_honest_fill_and_win_rates():
    from core.learning_loop import LearningLoop

    loop = LearningLoop()
    loop.record_trade(1, "A", "S1", "LONG", "BULL_TREND", 80, 0.6, 2.0, True, 2.0, "WIN")
    loop.record_trade(2, "B", "S1", "LONG", "BULL_TREND", 75, 0.55, 2.0, False, -1.0, "LOSS")
    loop.record_trade(3, "C", "S1", "LONG", "BULL_TREND", 70, 0.52, 2.0, False, 0.0, "BREAKEVEN")
    loop.record_trade(4, "D", "S1", "LONG", "BULL_TREND", 70, 0.52, 2.0, False, 0.0, "EXPIRED")
    loop.record_trade(5, "E", "S1", "LONG", "BULL_TREND", 70, 0.52, 2.0, False, 0.0, "INVALIDATED")

    stats = loop.get_stats()

    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["breakevens"] == 1
    assert stats["expired_count"] == 1
    assert stats["invalidated_count"] == 1
    assert stats["decisive_win_rate"] == 0.5
    assert round(stats["execution_win_rate"], 4) == round(1 / 3, 4)
    assert round(stats["fill_rate"], 4) == round(3 / 5, 4)


def test_portfolio_engine_rejects_dead_positions_below_viable_size():
    from core.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine()
    engine.get_dynamic_kelly = lambda *args, **kwargs: 0.002

    decision = engine._size_position_locked(
        symbol="TESTUSDT",
        direction="LONG",
        strategy="PriceAction",
        entry_price=100.0,
        stop_loss=0.0,
        kelly_fraction=0.002,
        p_win=0.55,
        rr_ratio=2.0,
        sector="L1",
        correlation_to_btc=0.7,
        symbol_volatility=0.03,
    )

    assert decision.approved is False
    assert "too small" in (decision.reject_reason or "").lower()
