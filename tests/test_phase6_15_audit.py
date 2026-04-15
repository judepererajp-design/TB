import time
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch


class TestExecutionEngineRestoreTimers:
    async def test_restore_uses_updated_at_for_watching_timeout(self):
        from core.execution_engine import ExecutionEngine

        engine = ExecutionEngine()
        created_at = time.time() - 10_000
        updated_at = time.time() - 120
        rows = [{
            "signal_id": 7,
            "symbol": "BTC/USDT",
            "direction": "LONG",
            "strategy": "PriceAction",
            "entry_low": 100.0,
            "entry_high": 101.0,
            "stop_loss": 95.0,
            "confidence": 80.0,
            "tp1": 103.0,
            "tp2": 105.0,
            "tp3": None,
            "rr_ratio": 2.0,
            "message_id": None,
            "grade": "A",
            "min_triggers": 2,
            "setup_class": "intraday",
            "has_rejection_candle": 0,
            "has_structure_shift": 0,
            "has_momentum_expansion": 0,
            "has_liquidity_reaction": 0,
            "created_at": created_at,
            "updated_at": updated_at,
            "state": "WATCHING",
        }]

        mock_db = AsyncMock()
        mock_db.load_tracked_signals = AsyncMock(return_value=rows)

        with patch.dict("sys.modules", {"data.database": type("M", (), {"db": mock_db})}):
            await engine.restore()

        restored = engine._tracked[7]
        assert restored.watching_since == updated_at
        assert restored.watching_since != created_at

    def test_setup_class_trigger_window_is_timeframe_aware(self):
        from core.execution_engine import _get_trigger_window_secs

        assert _get_trigger_window_secs("scalp") == 1_200
        assert _get_trigger_window_secs("swing") == 14_400

    async def test_check_triggers_uses_setup_class_confirmation_timeframe(self):
        from core.execution_engine import ExecutionEngine, TrackedExecution

        engine = ExecutionEngine()
        sig = TrackedExecution(
            signal_id=12,
            symbol="BTC/USDT",
            direction="LONG",
            strategy="PriceAction",
            entry_low=100.0,
            entry_high=101.0,
            stop_loss=95.0,
            confidence=80.0,
            setup_class="swing",
        )

        fake_ohlcv = [[idx, 100.0, 101.0, 99.5, 100.5 + idx, 1000.0] for idx in range(20)]
        with patch("core.execution_engine.api.fetch_ohlcv", AsyncMock(return_value=fake_ohlcv)) as fetch_mock:
            await engine._check_triggers(sig, price=100.5, market_state={})

        assert fetch_mock.await_args.args[1] == "1h"

    def test_fast_poll_interval_when_signal_is_near_entry(self):
        from core.execution_engine import (
            EXECUTION_FAST_CHECK_INTERVAL,
            SignalState,
            TrackedExecution,
            _resolve_check_interval,
        )

        sig = TrackedExecution(
            signal_id=13,
            symbol="BTC/USDT",
            direction="LONG",
            strategy="PriceAction",
            entry_low=100.0,
            entry_high=101.0,
            stop_loss=95.0,
            confidence=80.0,
        )
        sig.state = SignalState.APPROVED

        with patch("core.execution_engine.price_cache", type("PC", (), {"get": staticmethod(lambda _sym: 100.4)})()):
            interval = _resolve_check_interval({13: sig})

        assert interval == EXECUTION_FAST_CHECK_INTERVAL


class TestCorrelationCacheRespectsDynamicWindow:
    async def test_cache_is_ignored_when_regime_window_changes(self):
        from analyzers.correlation import CorrelationAnalyzer
        from analyzers.regime import regime_analyzer

        analyzer = CorrelationAnalyzer()
        analyzer._cache_ttl = 3600
        analyzer._cache["ETH/USDT"] = (1.1, 0.8, time.time(), 72)

        ohlcv_symbol = [[i, 0, 0, 0, 100 + (i * i), 0] for i in range(40)]
        btc_ohlcv = [[i, 0, 0, 0, 200 + (i * i * 0.5), 0] for i in range(40)]

        with patch("analyzers.correlation.api.fetch_ohlcv", AsyncMock(return_value=btc_ohlcv)) as fetch_mock:
            with patch.object(regime_analyzer, "_regime", type("Regime", (), {"value": "CHOPPY"})()):
                beta, corr = await analyzer.get_btc_beta("ETH/USDT", ohlcv_symbol)

        assert fetch_mock.await_count == 1
        assert isinstance(beta, float)
        assert isinstance(corr, float)


class TestDatabaseSchemaVersion:
    def test_schema_version_matches_latest_migration(self):
        source = Path(__file__).resolve().parents[1].joinpath("data", "database.py").read_text()
        assert re.search(r"_SCHEMA_VERSION\s*=\s*9\b", source)
