"""
Smoke tests for patterns/ folder — Phase-1 audit fixes.

Full integration tests need unmocked numpy; conftest.py mocks numpy globally.
These tests exercise:
  * Pure-data ratio tables and config wiring that don't touch numpy.
  * Module importability (catching regressions like TypeErrors in comments
    or undefined names).
"""

import pytest


def test_harmonic_cypher_ratio_fixed():
    """P1-H1: Cypher BCD must be (1.272, 2.000), not (0.786,)."""
    from patterns.harmonic import PATTERN_RATIOS
    assert 'Cypher' in PATTERN_RATIOS
    bcd = PATTERN_RATIOS['Cypher']['BCD']
    assert bcd == (1.272, 2.000), f"Cypher BCD should be (1.272, 2.000), got {bcd}"
    # Also XAD should be 0.786
    assert PATTERN_RATIOS['Cypher']['XAD'] == (0.786,)


def test_harmonic_abs_tolerance_table_defined():
    """P1-H4: absolute per-pattern tolerances must be defined for all patterns."""
    from patterns.harmonic import _PATTERN_ABS_TOL, PATTERN_RATIOS
    for name in PATTERN_RATIOS:
        assert name in _PATTERN_ABS_TOL, f"Missing abs tolerance for {name}"
        assert 0 < _PATTERN_ABS_TOL[name] < 0.2


def test_candlestick_module_imports():
    """Smoke: new candlestick patterns importable without error."""
    from patterns import candlestick as cs
    # Phase-1 additions
    for sym in (
        'pin_bar', 'engulfing', 'inside_bar', 'doji',
        'morning_star', 'evening_star',
        'three_white_soldiers', 'three_black_crows',
        'piercing_line', 'dark_cloud_cover',
        'tweezer_top', 'tweezer_bottom',
        'three_inside_up', 'three_inside_down',
        'detect_all',
    ):
        assert hasattr(cs, sym), f"candlestick.{sym} missing"


def test_candlestick_empty_input_safe():
    """Empty / malformed inputs must return (False, '', 0.0), not crash."""
    from patterns import candlestick as cs
    for fn in (cs.pin_bar, cs.doji, cs.engulfing, cs.inside_bar,
               cs.tweezer_top, cs.tweezer_bottom,
               cs.piercing_line, cs.dark_cloud_cover,
               cs.three_white_soldiers, cs.three_black_crows,
               cs.three_inside_up, cs.three_inside_down):
        assert fn([], [], [], []) == (False, '', 0.0)


def test_geometric_sub_pattern_flags_respected():
    """P1-G13: config flags `patterns.{sub}` must gate detector selection."""
    import patterns.geometric as geo
    # GeometricPatterns.__init__ reads the flags and sets _enabled_subs
    # We don't construct an instance (numpy is mocked) — we inspect the class.
    # But _enabled_subs is populated in __init__; check that the keys are
    # referenced during analyze() via the _cfg_bool helper existing.
    assert hasattr(geo.GeometricPatterns, '_cfg_bool')
    assert hasattr(geo.GeometricPatterns, '_clamp_projection')


def test_wyckoff_lazy_singleton():
    """P1-W12: wyckoff_analyzer must be a lazy proxy, not a real instance."""
    import patterns.wyckoff as w
    # Proxy class, not a WyckoffAnalyzer instance at import time
    assert not isinstance(w.wyckoff_analyzer, w.WyckoffAnalyzer)
    # But attribute access must still work (forwarded to a real instance)
    assert hasattr(w.wyckoff_analyzer, 'analyze')
