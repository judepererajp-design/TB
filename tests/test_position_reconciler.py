"""Tests for core.position_reconciler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import time

from core.position_reconciler import PositionReconciler, RecoveredSignal


class TestPositionReconciler:
    """Unit tests for PositionReconciler."""

    def _make_reconciler(self):
        return PositionReconciler()

    def _make_mock_db(self, rows=None):
        """Create a mock db with a connection pool that returns rows."""
        mock_db = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.description = [
            ("signal_id",), ("symbol",), ("direction",), ("strategy",),
            ("entry_low",), ("entry_high",), ("stop_loss",),
            ("tp1",), ("tp2",), ("tp3",), ("confidence",), ("state",),
            ("entry_price",), ("be_stop",), ("trail_stop",), ("trail_pct",),
            ("regime",), ("created_at",), ("activated_at",), ("message_id",),
        ]
        mock_cursor.fetchall = AsyncMock(return_value=rows or [])

        mock_conn = AsyncMock()
        mock_conn.execute = MagicMock(return_value=mock_cursor)

        # Make mock_conn work as async context manager
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_reader = MagicMock()
        mock_reader.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_reader.__aexit__ = AsyncMock(return_value=False)
        mock_pool.reader = MagicMock(return_value=mock_reader)

        mock_db._pool = mock_pool
        return mock_db

    def _make_row(self, signal_id=1, age_hours=1.0, state="ACTIVE"):
        """Create a fake DB row tuple."""
        now = time.time()
        return (
            signal_id, "BTC/USDT", "LONG", "SMC",
            100.0, 102.0, 98.0,    # entry_low, entry_high, stop_loss
            104.0, 108.0, 112.0,   # tp1, tp2, tp3
            85.0, state,           # confidence, state
            None, None, None, 0.4, # entry_price, be_stop, trail_stop, trail_pct
            "BULL_TREND",          # regime
            now - age_hours * 3600,  # created_at
            None,                  # activated_at
            None,                  # message_id
        )

    # ── row_to_signal ───────────────────────────────────────

    def test_row_to_signal(self):
        r = self._make_reconciler()
        row_dict = {
            "signal_id": 42, "symbol": "ETH/USDT", "direction": "SHORT",
            "strategy": "Breakout", "entry_low": 2000.0, "entry_high": 2050.0,
            "stop_loss": 2100.0, "tp1": 1950.0, "tp2": 1900.0, "tp3": None,
            "confidence": 78.0, "state": "PENDING", "entry_price": None,
            "be_stop": None, "trail_stop": None, "trail_pct": 0.4,
            "regime": "BEAR_TREND", "created_at": time.time(),
            "activated_at": None, "message_id": None,
        }
        sig = r._row_to_signal(row_dict)
        assert isinstance(sig, RecoveredSignal)
        assert sig.signal_id == 42
        assert sig.direction == "SHORT"
        assert sig.tp3 is None

    # ── reconcile_on_startup — no DB ────────────────────────

    async def test_reconcile_no_db_pool(self):
        """When DB pool is not initialized, return zeros."""
        r = self._make_reconciler()
        mock_db = MagicMock()
        mock_db._pool = None
        mock_om = MagicMock()

        # The reconciler does `from data.database import db` inside the method,
        # so we mock the module in sys.modules before the call.
        mock_module = MagicMock()
        mock_module.db = mock_db
        import sys
        old = sys.modules.get("data.database")
        sys.modules["data.database"] = mock_module
        try:
            result = await r.reconcile_on_startup(mock_om)
        finally:
            if old is not None:
                sys.modules["data.database"] = old
            else:
                sys.modules.pop("data.database", None)

        assert result == {"recovered": 0, "expired": 0, "errors": 0}

    # ── properties ──────────────────────────────────────────

    def test_initial_counts_zero(self):
        r = self._make_reconciler()
        assert r.recovered_count == 0
        assert r.expired_count == 0
