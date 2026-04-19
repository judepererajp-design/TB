import numpy as np

from strategies.base import BaseStrategy


def _row(ts: int, o: float, h: float, l: float, c: float, v: float = 1000.0):
    return [ts, o, h, l, c, v]


def test_mtf_find_zone_bearish_uses_zone_high_anchor():
    # Build 40 bars; the penultimate candle is the bullish OB and the last is bearish impulse.
    ohlcv = [_row(i, 100.0, 101.0, 99.0, 100.0) for i in range(38)]
    ohlcv.append(_row(38, 105.0, 111.0, 104.0, 110.0))  # bullish OB (open 105, close 110)
    ohlcv.append(_row(39, 110.0, 111.0, 99.0, 100.0))   # bearish impulse, current=100

    zone = BaseStrategy.mtf_find_zone({"1h": ohlcv}, tf="1h", bias="BEARISH")
    # With ATR ~2-3, ob_low(105) sits within +3*ATR but ob_high(110) does not.
    # Correct behavior is to reject this as a near-price supply zone.
    assert zone is None


def test_calculate_bollinger_short_data_returns_nan_triplet():
    mid, up, lo = BaseStrategy.calculate_bollinger(np.array([100.0, 101.0]), period=20)
    assert np.isnan(mid) and np.isnan(up) and np.isnan(lo)


def test_calculate_rsi_keeps_full_precision():
    closes = np.array([100.0, 101.3, 100.4, 102.7, 102.2, 103.9, 102.8, 104.1, 103.7, 105.2,
                       104.6, 106.0, 105.4, 107.3, 106.5, 108.2, 107.9, 109.1, 108.3, 110.0])
    rsi = BaseStrategy.calculate_rsi(closes, period=14)
    assert rsi != round(rsi, 2)


def test_indicator_cache_is_populated_for_repeated_calls():
    BaseStrategy.clear_indicator_cache()
    closes = np.array([100.0 + i * 0.2 for i in range(60)])
    highs = closes + 0.5
    lows = closes - 0.5

    BaseStrategy.calculate_rsi(closes, period=14)
    BaseStrategy.calculate_atr(highs, lows, closes, period=14)
    BaseStrategy.calculate_adx(highs, lows, closes, period=14)
    BaseStrategy.calculate_bollinger(closes, period=20, std_mult=2.0)

    assert len(BaseStrategy._class_indicator_cache) >= 4
