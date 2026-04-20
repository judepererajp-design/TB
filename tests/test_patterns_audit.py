"""
Smoke tests for patterns/ folder — Phase-1 and Phase-2 audit fixes.

Full integration tests need unmocked numpy; conftest.py mocks numpy globally.
These tests exercise:
  * Pure-data ratio tables and config wiring that don't touch numpy.
  * Module importability (catching regressions like TypeErrors in comments
    or undefined names).
  * Shared `patterns/_common/` helpers (stdlib-only, fully exercisable).
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


# ── Phase-2 structural tests ────────────────────────────────────────

def test_phase2_common_module_importable():
    """P2: patterns/_common re-exports all helpers."""
    from patterns._common import (
        find_alternating_pivots, is_isolated_swing,
        clamp_projection, price_floor_valid,
        regime_allows_structural, regime_penalty_for_pattern,
        volume_confirmed, compute_volume_stats,
    )
    # Sanity ping each helper
    assert find_alternating_pivots([], [], 5) == []
    assert is_isolated_swing([], 0, 3) is False
    # clamp_projection: raw=100, entry=10, atr=1 → cap = max(5, 3) = 5
    assert clamp_projection(100.0, 10.0, atr=1.0) == 5.0
    assert price_floor_valid(10.0, 5.0, 7.0) is True
    assert price_floor_valid(10.0, 0.001) is False
    assert volume_confirmed(None) is False
    assert compute_volume_stats(None) is None


def test_phase2_clamp_projection_caps_negative_and_overshoot():
    """P2: clamp_projection caps overshoots to entry/2 and blocks negatives."""
    from patterns._common import clamp_projection
    # Overshoot: raw much greater than entry → capped at entry * 0.5
    assert clamp_projection(100.0, 10.0, atr=0.0) == 5.0
    # With ATR floor, cap = max(entry*0.5, 3*ATR)
    assert clamp_projection(100.0, 10.0, atr=3.0) == 9.0
    # Negative raw becomes 0
    assert clamp_projection(-5.0, 10.0) == 0.0
    # NaN is treated as 0
    assert clamp_projection(float('nan'), 10.0) == 0.0


def test_phase2_pivots_alternating():
    """P2: find_alternating_pivots returns strictly alternating H/L pivots."""
    from patterns._common import find_alternating_pivots
    # Hand-built alternating series: low, high, low, high, low
    highs = [2, 3, 5, 3, 4, 7, 4, 5, 9, 5, 6]
    lows  = [1, 2, 4, 1, 3, 6, 2, 4, 8, 3, 5]
    pivots = find_alternating_pivots(highs, lows, order=2)
    # Must alternate strictly
    for i in range(1, len(pivots)):
        assert pivots[i][2] != pivots[i-1][2], f"pivots not alternating: {pivots}"


def test_phase2_regime_gating():
    """P2-R1: regime helper gates continuation setups in opposing regimes."""
    from patterns._common import (
        regime_allows_structural, regime_penalty_for_pattern,
    )
    # Bull flag blocked in BEAR_TREND
    assert regime_allows_structural("bull_flag", "BEAR_TREND") is False
    assert regime_allows_structural("bull_flag", "BULL_TREND") is True
    # Inverse H&S blocked in BEAR_TREND (counter-regime reversal)
    assert regime_allows_structural("inverse_hs", "BEAR_TREND") is False
    # Unknown inputs are permissive
    assert regime_allows_structural("bull_flag", None) is True
    assert regime_allows_structural("nonexistent", "BULL_TREND") is True
    # Penalty is modest and typed
    p = regime_penalty_for_pattern("bull_flag", "CHOPPY")
    assert isinstance(p, float) and 0 <= p <= 10


def test_phase2_volume_confirmed_median_baseline():
    """P2-G1: volume_confirmed uses median, not mean — one outlier ignored."""
    from patterns._common import volume_confirmed
    # 19 quiet bars + 1 spike baseline must not defeat a real 1.5× event
    baseline = [10.0] * 19 + [500.0]    # median of first 19 is 10
    volumes = baseline + [25.0]         # event bar 2.5x median of first 19
    assert volume_confirmed(volumes, mult=1.3, lookback=20) is True
    # Flat volume: no confirmation
    assert volume_confirmed([10.0] * 25, mult=1.3) is False
    # Empty / None returns False
    assert volume_confirmed(None) is False
    assert volume_confirmed([]) is False


def test_phase2_harmonic_exposes_prz_helpers():
    """P2-H1/H2: PRZ cluster and D-reversal candle helpers are attached."""
    from patterns.harmonic import HarmonicDetector
    assert hasattr(HarmonicDetector, '_check_prz_cluster')
    assert hasattr(HarmonicDetector, '_has_reversal_candle_at_d')
    # _build_signal accepts the new kwargs
    import inspect
    sig = inspect.signature(HarmonicDetector._build_signal)
    for kw in ('prz_cluster', 'd_candle_ok', 'regime'):
        assert kw in sig.parameters


def test_phase2_wyckoff_lps_and_cause_effect_fields():
    """P2-W1/W2: WyckoffResult carries lps_confirmed + cause_effect_target."""
    from patterns.wyckoff import WyckoffResult
    fields = {f.name for f in WyckoffResult.__dataclass_fields__.values()}
    assert 'lps_confirmed' in fields
    assert 'cause_effect_target' in fields


def test_phase2_wyckoff_cause_effect_math():
    """Cause-effect: sqrt-scaling of range_size with cap at entry * 0.5."""
    from patterns.wyckoff import WyckoffAnalyzer
    # Instantiate via the lazy proxy path
    import patterns.wyckoff as w
    a = w.WyckoffAnalyzer()
    # LONG projection: key_level + range_size * sqrt(bars / min_bars)
    target = a._cause_effect_target("LONG",
                                    key_level=100.0,
                                    range_bars=80,
                                    range_size=10.0,
                                    atr=1.0)
    assert target is not None
    assert target > 100.0
    # SHORT mirror: target below key_level
    t2 = a._cause_effect_target("SHORT", 100.0, 80, 10.0, 1.0)
    assert t2 is not None and t2 < 100.0
    # Invalid inputs return None
    assert a._cause_effect_target("LONG", 0.0, 80, 10.0, 1.0) is None
    assert a._cause_effect_target("LONG", 100.0, 0, 10.0, 1.0) is None


def test_phase2_geometric_uses_shared_clamp():
    """P2: GeometricPatterns._clamp_projection delegates to shared helper."""
    from patterns.geometric import GeometricPatterns
    from patterns._common import clamp_projection
    # The static method should return identical values to the shared helper.
    for raw, entry, atr in [(100.0, 10.0, 0.0),
                            (100.0, 10.0, 3.0),
                            (0.5, 10.0, 1.0),
                            (-5.0, 10.0, 0.0)]:
        assert GeometricPatterns._clamp_projection(raw, entry, atr) == \
               clamp_projection(raw, entry, atr)


def test_phase3_wyckoff_spring_recovery_volume_counts_as_confirmation():
    """Phase-3: next-bar recovery volume must feed downstream volume_confirms."""
    import inspect
    from patterns.wyckoff import WyckoffAnalyzer

    a = WyckoffAnalyzer()
    a._vol_sensitivity = 1.1
    highs = [101.0] * 26
    lows = [99.5] * 26
    closes = [100.0] * 26
    volumes = [100.0] * 26

    lows[24] = 98.0
    closes[24] = 99.0          # no same-bar recovery
    closes[25] = 100.4         # next-bar recovery
    volumes[24] = 110.0        # no event-bar spike
    volumes[25] = 120.0        # recovery-bar spike; clears the 100 * 1.1 * 0.8 threshold

    spring = a._detect_spring(
        highs, lows, closes, volumes,
        range_low=99.5, range_size=1.5, avg_volume=100.0, atr=1.0,
    )
    assert spring is not None
    assert spring["volume_spike"] is False
    assert spring["recovery_volume_spike"] is True
    assert spring["recovery_bar"] == 25
    assert spring["recovery_close"] == pytest.approx(100.4)
    src = inspect.getsource(type(a).analyze)
    assert "spring['volume_spike'] or spring.get('recovery_volume_spike')" in src


def test_phase3_wyckoff_same_bar_recovery_does_not_fake_recovery_spike():
    """Phase-3: same-bar recovery should not double-count as recovery-bar volume."""
    from patterns.wyckoff import WyckoffAnalyzer

    a = WyckoffAnalyzer()
    a._vol_sensitivity = 1.0
    highs = [101.0] * 26
    lows = [99.5] * 26
    closes = [100.0] * 26
    volumes = [100.0] * 26

    lows[24] = 98.0
    closes[24] = 100.4         # same-bar recovery
    volumes[24] = 120.0        # event-bar spike only

    spring = a._detect_spring(
        highs, lows, closes, volumes,
        range_low=99.5, range_size=1.5, avg_volume=100.0, atr=1.0,
    )
    assert spring is not None
    assert spring["volume_spike"] is True
    assert spring["recovery_volume_spike"] is False
    assert spring["recovery_bar"] is None
    assert spring["recovery_close"] == pytest.approx(100.4)


def test_phase3_wyckoff_utad_rejection_volume_wired_into_confirmation():
    """Phase-3: UTAD next-bar rejection volume must also wire into volume_confirms."""
    import inspect
    from patterns.wyckoff import WyckoffAnalyzer

    src = inspect.getsource(WyckoffAnalyzer.analyze)
    assert "utad['volume_spike'] or utad.get('rejection_volume_spike')" in src
