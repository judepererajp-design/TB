"""
Tests for Gate 5: post-startup extension filter in engine._scan_symbol.

Gate 5 blocks signals where the recent 4-bar 1h price move is already
≥ POST_STARTUP_EXT_ATR_MULT × ATR (floor: POST_STARTUP_EXT_FLOOR_PCT) in the
signal direction, during the POST_STARTUP_FILTER_SECS window after warmup ends.
"""
import asyncio
import time
import types
import sys
from unittest.mock import AsyncMock, MagicMock


def _make_ohlcv_bars(n=20, price_start=100.0, price_end=100.0):
    """Return `n` synthetic 1h OHLCV bars.

    First n-5 bars are flat at price_start; last 5 bars ramp linearly to price_end.
    Format: [ts, open, high, low, close, volume].
    ATR per bar = 1% of the mid-point close.
    """
    bars = []
    ts = 0
    for i in range(n):
        if i < n - 5:
            c = price_start
        else:
            frac = (i - (n - 5)) / 4  # 0.0 → 1.0 over last 5 bars
            c = price_start + frac * (price_end - price_start)
        atr = c * 0.01  # 1% ATR per bar
        h = c + atr / 2
        lo = c - atr / 2
        bars.append([ts, c, h, lo, c, 1000.0])
        ts += 3600
    return bars


def _build_engine_minimal():
    """
    Return a minimal Engine-like object with just the attributes Gate 5 reads:
      _warmup_end_time, _warmup_active, and the Timing constants.
    """
    from config.constants import Timing

    eng = types.SimpleNamespace(
        _warmup_end_time=time.time() - 60,   # warmup ended 60s ago (within 900s window)
        _warmup_active=False,
        _Timing=Timing,
    )
    return eng


# ── helpers to replicate Gate 5 logic in isolation ────────────────────────────

def _run_gate5(dir_str, bars, warmup_end_time, current_p=None):
    """
    Replicate Gate 5 logic exactly as written in engine._scan_symbol.
    Returns True if the signal is BLOCKED, False if it PASSES.
    """
    from config.constants import Timing

    if current_p is None:
        current_p = float(bars[-1][4]) if bars else 0.0

    _psfilt_active = (
        warmup_end_time > 0
        and time.time() - warmup_end_time < Timing.POST_STARTUP_FILTER_SECS
        and current_p > 0
    )
    if not _psfilt_active:
        return False  # gate inactive

    try:
        _chk_bars = bars
        if len(_chk_bars) >= 8:
            _c_now  = float(_chk_bars[-1][4])
            _c_4ago = float(_chk_bars[-5][4])
            _atr = (
                sum(float(_chk_bars[i][2]) - float(_chk_bars[i][3])
                    for i in range(-14, 0)) / 14
            ) if len(_chk_bars) >= 14 else 0.0
            if _c_4ago > 0 and _atr > 0 and _c_now > 0:
                _4h_chg = (_c_now - _c_4ago) / _c_4ago * 100
                _atr_pct = _atr / _c_now * 100
                _ext_thresh = max(
                    _atr_pct * Timing.POST_STARTUP_EXT_ATR_MULT,
                    Timing.POST_STARTUP_EXT_FLOOR_PCT,
                )
                _is_extended = (
                    (dir_str == "LONG"  and _4h_chg >  _ext_thresh) or
                    (dir_str == "SHORT" and _4h_chg < -_ext_thresh)
                )
                return _is_extended
    except Exception:
        pass
    return False


# ── tests ──────────────────────────────────────────────────────────────────────

def test_gate5_blocks_long_when_price_already_surged():
    """LONG signal blocked when 4h price moved well above ATR-based threshold."""
    # price started at 100 and is now at 115 (+15%) — well above any 6% floor
    bars = _make_ohlcv_bars(20, price_start=100.0, price_end=115.0)
    warmup_end = time.time() - 60  # within the 900s window
    assert _run_gate5("LONG", bars, warmup_end) is True


def test_gate5_blocks_short_when_price_already_crashed():
    """SHORT signal blocked when 4h price dropped far below ATR-based threshold."""
    bars = _make_ohlcv_bars(20, price_start=100.0, price_end=83.0)  # –17%
    warmup_end = time.time() - 60
    assert _run_gate5("SHORT", bars, warmup_end) is True


def test_gate5_passes_long_on_normal_move():
    """LONG signal passes when the 4h move is small (within 1 ATR)."""
    # ATR ≈ 1% of price; 4h move of +1% is well below 6% floor
    bars = _make_ohlcv_bars(20, price_start=100.0, price_end=101.0)
    warmup_end = time.time() - 60
    assert _run_gate5("LONG", bars, warmup_end) is False


def test_gate5_passes_short_on_small_dip():
    """SHORT signal passes when the 4h dip is small."""
    bars = _make_ohlcv_bars(20, price_start=100.0, price_end=99.5)  # –0.5%
    warmup_end = time.time() - 60
    assert _run_gate5("SHORT", bars, warmup_end) is False


def test_gate5_inactive_outside_startup_window():
    """Gate 5 is a no-op after POST_STARTUP_FILTER_SECS has elapsed."""
    from config.constants import Timing
    bars = _make_ohlcv_bars(20, price_start=100.0, price_end=120.0)  # +20%
    # warmup ended well outside the filter window
    warmup_end = time.time() - (Timing.POST_STARTUP_FILTER_SECS + 60)
    assert _run_gate5("LONG", bars, warmup_end) is False


def test_gate5_inactive_when_warmup_end_not_set():
    """Gate 5 is inactive when warmup never ended (warmup_end_time == 0)."""
    bars = _make_ohlcv_bars(20, price_start=100.0, price_end=130.0)  # +30%
    assert _run_gate5("LONG", bars, warmup_end_time=0) is False


def test_gate5_does_not_block_opposite_direction():
    """A large surge should block LONG but NOT block SHORT (opposite direction)."""
    bars = _make_ohlcv_bars(20, price_start=100.0, price_end=115.0)  # +15%
    warmup_end = time.time() - 60
    # LONG blocked
    assert _run_gate5("LONG", bars, warmup_end) is True
    # SHORT passes (price went up, not down)
    assert _run_gate5("SHORT", bars, warmup_end) is False


def test_gate5_survives_insufficient_ohlcv_data():
    """Gate 5 never blocks a signal when OHLCV data is too short."""
    short_bars = _make_ohlcv_bars(5, price_start=100.0, price_end=120.0)
    warmup_end = time.time() - 60
    assert _run_gate5("LONG", short_bars, warmup_end) is False


def test_gate5_survives_empty_ohlcv():
    """Gate 5 never blocks when ohlcv_dict is empty."""
    warmup_end = time.time() - 60
    assert _run_gate5("LONG", [], warmup_end, current_p=105.0) is False
