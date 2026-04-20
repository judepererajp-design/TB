"""
Tests for strategies.base direction helpers and SignalResult.get_raw.

These replace the ~80-site ``getattr(signal.direction, 'value', str(signal.direction))``
idiom and the 9-line ``hasattr(signal, 'raw_data') and signal.raw_data``
defensive-access pattern in ``core/engine.py``.

The helpers tolerate the two concrete representations that flow through the
pipeline — ``SignalDirection`` enum members (produced by strategies) and raw
``"LONG"``/``"SHORT"`` strings (set on mutated/rehydrated signals).
"""
from types import SimpleNamespace

from strategies.base import (
    SignalDirection,
    SignalResult,
    direction_str,
    is_long,
    is_short,
)


# ── direction_str / is_long / is_short ────────────────────────────────────

def test_direction_str_accepts_enum():
    sig = SimpleNamespace(direction=SignalDirection.LONG)
    assert direction_str(sig) == "LONG"


def test_direction_str_accepts_raw_string():
    sig = SimpleNamespace(direction="SHORT")
    assert direction_str(sig) == "SHORT"


def test_direction_str_accepts_bare_enum_value():
    assert direction_str(SignalDirection.LONG) == "LONG"
    assert direction_str("SHORT") == "SHORT"


def test_is_long_and_is_short_on_enum():
    sig = SimpleNamespace(direction=SignalDirection.LONG)
    assert is_long(sig) is True
    assert is_short(sig) is False


def test_is_long_and_is_short_on_string():
    sig = SimpleNamespace(direction="SHORT")
    assert is_long(sig) is False
    assert is_short(sig) is True


# ── SignalResult.get_raw ──────────────────────────────────────────────────

def _make_signal(raw_data=None):
    return SignalResult(
        symbol="BTC/USDT",
        direction=SignalDirection.LONG,
        strategy="Test",
        confidence=50.0,
        entry_low=1.0,
        entry_high=1.0,
        stop_loss=0.9,
        tp1=1.1,
        tp2=1.2,
        raw_data=raw_data,
    )


def test_get_raw_returns_default_when_raw_data_is_none():
    sig = _make_signal(raw_data=None)
    assert sig.get_raw("adx", 0) == 0
    assert sig.get_raw("has_ob", False) is False


def test_get_raw_returns_value_when_present():
    sig = _make_signal(raw_data={"adx": 25, "has_ob": True})
    assert sig.get_raw("adx", 0) == 25
    assert sig.get_raw("has_ob", False) is True


def test_get_raw_returns_default_for_missing_key():
    sig = _make_signal(raw_data={"adx": 25})
    assert sig.get_raw("has_ob", False) is False


def test_get_raw_tolerates_non_dict_raw_data():
    # Some code paths accidentally assign a non-dict into raw_data; the helper
    # must not raise.
    sig = _make_signal(raw_data="corrupt")  # type: ignore[arg-type]
    assert sig.get_raw("adx", 0) == 0
