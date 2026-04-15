import time
from unittest.mock import AsyncMock

import pytest


def _ohlcv_from_closes(closes):
    rows = []
    for idx, close in enumerate(closes):
        rows.append([idx, close - 0.3, close + 0.5, close - 0.5, close, 1000 + idx])
    return rows


def test_late_local_range_reason_blocks_long_in_premium():
    from core.execution_engine import TrackedExecution, _late_local_range_reason

    sig = TrackedExecution(
        signal_id=1,
        symbol="EDGE/USDT",
        direction="LONG",
        strategy="PriceAction",
        entry_low=1.0184,
        entry_high=1.0361,
        stop_loss=0.9598,
        confidence=82.0,
        created_at=time.time(),
    )

    reason = _late_local_range_reason(sig, 104.0, _ohlcv_from_closes([95, 96, 97, 98, 99, 100, 101, 102, 103, 105]))

    assert reason is not None
    assert "premium" in reason.lower()


def test_late_local_range_reason_skips_breakout_style_entries():
    from core.execution_engine import TrackedExecution, _late_local_range_reason

    sig = TrackedExecution(
        signal_id=2,
        symbol="EDGE/USDT",
        direction="LONG",
        strategy="Momentum",
        entry_low=1.0184,
        entry_high=1.0361,
        stop_loss=0.9598,
        confidence=82.0,
        created_at=time.time(),
    )

    reason = _late_local_range_reason(sig, 104.0, _ohlcv_from_closes([95, 96, 97, 98, 99, 100, 101, 102, 103, 105]))

    assert reason is None


@pytest.mark.asyncio
async def test_check_staleness_blocks_fresh_late_long_in_local_premium(monkeypatch):
    from core.execution_engine import ExecutionEngine, TrackedExecution

    engine = ExecutionEngine()
    sig = TrackedExecution(
        signal_id=3,
        symbol="EDGE/USDT",
        direction="LONG",
        strategy="PriceAction",
        entry_low=1.0184,
        entry_high=1.0361,
        stop_loss=0.9598,
        confidence=82.0,
        created_at=time.time(),
    )

    monkeypatch.setattr(
        "core.execution_engine.api.fetch_ohlcv",
        AsyncMock(return_value=_ohlcv_from_closes([95, 96, 97, 98, 99, 100, 101, 102, 103, 105])),
    )

    reason = await engine._check_staleness(sig, 104.0)

    assert reason is not None
    assert "premium" in reason.lower()
