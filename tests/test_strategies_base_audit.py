import inspect

from strategies.base import BaseStrategy


def test_mtf_find_zone_bearish_uses_zone_high_anchor():
    src = inspect.getsource(BaseStrategy.mtf_find_zone)
    assert "if ob_high > current and ob_high < current + atr * 3:" in src


def test_calculate_bollinger_short_data_returns_nan_triplet():
    src = inspect.getsource(BaseStrategy.calculate_bollinger)
    assert 'return float("nan"), float("nan"), float("nan")' in src


def test_calculate_rsi_keeps_full_precision():
    src = inspect.getsource(BaseStrategy.calculate_rsi)
    assert "return round(" not in src
    assert "result = float(100.0 - (100.0 / (1.0 + rs)))" in src


def test_indicator_cache_is_populated_for_repeated_calls():
    src = inspect.getsource(BaseStrategy)
    assert 'cache_key = ("rsi", period, cls._array_fingerprint(closes))' in src
    assert '"atr",' in src and '"adx",' in src and '"bb", period, std_mult' in src
    assert "cls._class_indicator_cache[cache_key] = result" in src
