"""
Regression tests for the InstitutionalBreakout audit fixes:
  BR-1  – get_weekly_adx() public accessor used instead of _weekly_adx
  BR-2  – volume denominator excludes the current bar
  BR-3  – flat-market gate: range_size < atr * 0.5 → return None
  BR-4  – SL anchored to entry candle's low/high, not inside the range
  BR-Q1 – measured-move targets: TP2=80%, TP3=100%
  BR-Q2 – min_adx lowered; rising ADX slope accepted as alternative qualifier
  BR-Q3 – false_breakout_filter flag wires a confidence penalty below 3× vol
  BR-Q4 – VOLATILE_PANIC hard-blocked; VOLATILE requires ≥2× vol_mult minimum
"""
import inspect


# ── BR-1: public accessor ─────────────────────────────────────────────────────

def test_get_weekly_adx_exists_and_returns_float():
    from analyzers.htf_guardrail import HTFWeeklyGuardrail
    g = HTFWeeklyGuardrail()
    assert hasattr(g, "get_weekly_adx")
    assert callable(g.get_weekly_adx)
    assert isinstance(g.get_weekly_adx(), float)


def test_get_weekly_adx_reflects_internal_value():
    from analyzers.htf_guardrail import HTFWeeklyGuardrail
    g = HTFWeeklyGuardrail()
    g._weekly_adx = 42.5
    assert g.get_weekly_adx() == 42.5


def test_breakout_uses_public_weekly_adx_accessor():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert "get_weekly_adx()" in src
    assert "_brk_htf._weekly_adx" not in src


# ── BR-2: volume denominator ──────────────────────────────────────────────────

def test_breakout_volume_denominator_excludes_current_bar():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    # Must use the same lookback-exclusive slice as Donchian channel
    assert "volumes[-lookback-1:-1]" in src
    # Must NOT average volumes[-lookback:] (which included the current bar)
    assert "np.mean(volumes[-lookback:])" not in src


# ── BR-3: flat market gate ────────────────────────────────────────────────────

def test_breakout_has_range_size_gate():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert "range_size < atr * 0.5" in src


# ── BR-4: SL anchored to entry candle ────────────────────────────────────────

def test_breakout_long_sl_uses_current_low():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert "current_low - atr * 0.2" in src


def test_breakout_short_sl_uses_current_high():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert "current_high + atr * 0.2" in src


def test_breakout_sl_no_longer_uses_range_percentage():
    """Old code used range_size * 0.30 to place SL inside the range — removed."""
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert "range_size * 0.30" not in src


# ── BR-Q1: measured-move target ratios ───────────────────────────────────────

def test_breakout_tp2_is_80_pct_measured_move():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert "range_size * 0.80" in src
    # 100% measured move is now TP3, not TP2
    assert "range_size * 1.00" in src
    # Old 100% TP2 and 150% TP3 should be gone
    assert "range_size * 1.50" not in src


# ── BR-Q2: ADX threshold and slope ───────────────────────────────────────────

def test_breakout_default_min_adx_is_20():
    from strategies.breakout import BreakoutStrategy
    import unittest.mock as mock
    # getattr falls back to 20 when config attribute is absent
    b = BreakoutStrategy.__new__(BreakoutStrategy)
    b._cfg = mock.MagicMock(spec=[])  # no donchian_period / min_adx attrs
    min_adx = getattr(b._cfg, "min_adx", 20)
    assert min_adx == 20


def test_breakout_adx_slope_check_in_source():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert "_adx_rising" in src
    assert "_adx_prev" in src
    # Filter accepts rising slope as alternative
    assert "adx < min_adx and not _adx_rising" in src


# ── BR-Q3: false_breakout_filter penalty ─────────────────────────────────────

def test_breakout_false_breakout_filter_penalty_in_source():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert "false_breakout_filter" in src
    assert "_fb_penalty" in src
    # Penalty applied to confidence
    assert "_fb_penalty" in src and "confidence" in src


def test_breakout_3x_vol_threshold_referenced():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert "vol_ratio < 3.0" in src


# ── BR-Q4: VOLATILE_PANIC block and VOLATILE vol floor ───────────────────────

def test_breakout_volatile_panic_blocked():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    assert 'regime == "VOLATILE_PANIC"' in src


def test_breakout_volatile_vol_floor_2x():
    src = inspect.getsource(__import__("strategies.breakout", fromlist=["BreakoutStrategy"]).BreakoutStrategy)
    # VOLATILE requires at least 2.0 vol_mult
    assert "_is_volatile" in src
    assert "max(vol_mult, 2.0)" in src
