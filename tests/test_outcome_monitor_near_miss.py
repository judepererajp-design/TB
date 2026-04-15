import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

if 'config.loader' not in sys.modules:
    _loader = types.ModuleType('config.loader')
    _cfg = MagicMock()
    _cfg.aggregator = {'min_confidence': 72}
    _loader.cfg = _cfg
    sys.modules['config.loader'] = _loader

if 'data.api_client' not in sys.modules:
    _api_mod = types.ModuleType('data.api_client')
    _api_mod.api = types.SimpleNamespace(
        fetch_ohlcv=AsyncMock(return_value=[]),
        fetch_ticker=AsyncMock(return_value={}),
    )
    sys.modules['data.api_client'] = _api_mod

if 'utils.formatting' not in sys.modules:
    _fmt_mod = types.ModuleType('utils.formatting')
    _fmt_mod.fmt_price = lambda value: str(value)
    sys.modules['utils.formatting'] = _fmt_mod

if 'core.price_cache' not in sys.modules:
    _pc_mod = types.ModuleType('core.price_cache')
    _pc_mod.price_cache = types.SimpleNamespace(
        get_prices=lambda: {},
        subscribe=lambda symbol: None,
        unsubscribe=lambda symbol: None,
    )
    sys.modules['core.price_cache'] = _pc_mod

from analyzers import near_miss_tracker as near_miss_module
from signals import outcome_monitor as outcome_monitor_module
from signals.outcome_monitor import OutcomeMonitor


def test_update_near_miss_feedback_uses_live_price_and_recent_candle(monkeypatch):
    monitor = OutcomeMonitor()
    fake_tracker = MagicMock()
    fake_tracker.get_pending_keys.return_value = [
        ("BTCUSDT", "LONG"),
        ("BTCUSDT", "SHORT"),
    ]
    monkeypatch.setattr(near_miss_module, "near_miss_tracker", fake_tracker)
    monkeypatch.setattr(
        outcome_monitor_module.api,
        "fetch_ohlcv",
        AsyncMock(return_value=[
            [1, 0, 110.0, 90.0, 100.0],
            [2, 0, 112.0, 88.0, 101.0],
        ]),
    )
    monkeypatch.setattr(
        outcome_monitor_module.api,
        "fetch_ticker",
        AsyncMock(return_value={"last": 101.5}),
    )

    asyncio.run(monitor._update_near_miss_feedback({"BTCUSDT": 101.5}))

    fake_tracker.check_outcome.assert_any_call("BTCUSDT", "LONG", 110.0, 90.0, 101.5)
    fake_tracker.check_outcome.assert_any_call("BTCUSDT", "SHORT", 110.0, 90.0, 101.5)
    outcome_monitor_module.api.fetch_ticker.assert_not_awaited()


def test_update_near_miss_feedback_falls_back_to_ticker(monkeypatch):
    monitor = OutcomeMonitor()
    fake_tracker = MagicMock()
    fake_tracker.get_pending_keys.return_value = [("ETHUSDT", "LONG")]
    monkeypatch.setattr(near_miss_module, "near_miss_tracker", fake_tracker)
    monkeypatch.setattr(
        outcome_monitor_module.api,
        "fetch_ohlcv",
        AsyncMock(return_value=[[1, 0, 3300.0, 3100.0, 3200.0]]),
    )
    monkeypatch.setattr(
        outcome_monitor_module.api,
        "fetch_ticker",
        AsyncMock(return_value={"last": 3205.0}),
    )

    asyncio.run(monitor._update_near_miss_feedback({}))

    fake_tracker.check_outcome.assert_called_once_with(
        "ETHUSDT", "LONG", 3300.0, 3100.0, 3205.0
    )
    outcome_monitor_module.api.fetch_ticker.assert_awaited_once_with("ETHUSDT")
