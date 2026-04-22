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


def test_late_local_range_reason_skips_sub_atr_pullback_drift():
    """Log-audit 2026-04-22: a pullback that drifts <0.5 ATR past EQ inside a
    tight local range should NOT be called late.  Previously this fired on
    range-position alone and blocked valid wedge entries on drifts as small
    as +0.22 with range=7.8%."""
    from core.execution_engine import TrackedExecution, _late_local_range_reason

    sig = TrackedExecution(
        signal_id=10,
        symbol="TIGHT/USDT",
        direction="LONG",
        strategy="PriceAction",
        entry_low=99.5,
        entry_high=100.5,
        stop_loss=97.0,
        confidence=72.0,
        created_at=time.time(),
    )

    # Build a tight, high-ATR range: bars swing between 95 and 105 in a
    # choppy zig-zag so ATR is large relative to the half-range (≈10× bigger).
    # Range ≈ 10.5 / 100 = ~10% (tight), half-range ≈ 5, ATR ≈ 10.
    # Price at 101: position = 1/5 = 0.20 (borderline premium),
    # drift = 1, atr_drift_ratio = 1/10 = 0.10 → well below 0.5 → pass.
    rows = []
    closes = [95, 105, 96, 104, 97, 103, 98, 102, 99, 101]
    for idx, close in enumerate(closes):
        rows.append([idx, close, close + 5.0, close - 5.0, close, 1000 + idx])

    reason = _late_local_range_reason(sig, 101.0, rows)

    assert reason is None, f"sub-ATR pullback should not be blocked, got: {reason}"


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
