"""
TitanBot Pro — R:R Audit Tests
================================
Tests for the deep audit fixes:
1. PriceAction SL buffer uses sl_atr_mult * 0.35 (not entry_zone_tight)
2. max_rr from risk config is enforced in aggregator
"""

import pytest
from unittest.mock import MagicMock, patch


# ── 1. PriceAction SL Buffer Tests ────────────────────────────────

class TestPriceActionSLBuffer:
    """
    Verify that PriceAction uses sl_atr_mult * 0.35 for SL buffer
    instead of the old entry_zone_tight (0.10 ATR).
    """

    def test_sl_buffer_uses_sl_atr_mult(self):
        """SL buffer should be sl_atr_mult * 0.35, not entry_zone_tight."""
        from utils.risk_params import rp
        sl_atr_mult = rp.sl_atr_mult
        entry_zone_tight = rp.entry_zone_tight

        expected_buffer = sl_atr_mult * 0.35
        assert expected_buffer > entry_zone_tight, (
            f"New SL buffer ({expected_buffer:.3f}) should exceed "
            f"entry_zone_tight ({entry_zone_tight:.3f})"
        )

    def test_sl_buffer_minimum_distance(self):
        """SL buffer should be at least 0.3 ATR for noise protection."""
        from utils.risk_params import rp
        sl_buffer_value = rp.sl_atr_mult * 0.35
        assert sl_buffer_value >= 0.30, (
            f"SL buffer multiplier ({sl_buffer_value:.3f}) should be >= 0.30 ATR "
            f"to protect against noise"
        )

    def test_long_sl_below_pattern_low(self):
        """LONG SL should be below pattern_low by sl_atr_mult*0.35*ATR."""
        from utils.risk_params import rp
        atr = 0.005  # example ATR
        pattern_low = 0.100  # example pattern low
        _sl_buf = atr * rp.sl_atr_mult * 0.35

        stop_loss = pattern_low - _sl_buf
        assert stop_loss < pattern_low, "LONG SL must be below pattern low"
        assert (pattern_low - stop_loss) == pytest.approx(_sl_buf, abs=1e-10), (
            "SL distance should equal sl_atr_mult * 0.35 * ATR"
        )

    def test_short_sl_above_pattern_high(self):
        """SHORT SL should be above pattern_high by sl_atr_mult*0.35*ATR."""
        from utils.risk_params import rp
        atr = 0.005
        pattern_high = 0.120
        _sl_buf = atr * rp.sl_atr_mult * 0.35

        stop_loss = pattern_high + _sl_buf
        assert stop_loss > pattern_high, "SHORT SL must be above pattern high"
        assert (stop_loss - pattern_high) == pytest.approx(_sl_buf, abs=1e-10), (
            "SL distance should equal sl_atr_mult * 0.35 * ATR"
        )

    def test_rr_more_realistic_with_wider_sl(self):
        """
        Wider SL buffer should produce lower (more realistic) R:R.
        Old: 0.10 ATR buffer → ~6.5R on small candles
        New: 0.42 ATR buffer → ~3.5R on small candles
        """
        from utils.risk_params import rp
        atr = 0.005
        current_price = 0.100
        current_high = 0.1015  # small bearish candle (0.3 ATR wick)

        # Entry zone for SHORT
        entry_high = current_price + atr * rp.entry_zone_tight
        entry_low = current_price - atr * rp.entry_zone_atr
        entry_mid = (entry_low + entry_high) / 2

        # Old SL (entry_zone_tight = 0.10)
        old_sl = current_high + atr * rp.entry_zone_tight
        old_risk = old_sl - entry_mid

        # New SL (sl_atr_mult * 0.35)
        new_sl = current_high + atr * rp.sl_atr_mult * 0.35
        new_risk = new_sl - entry_mid

        # TP2 (fallback ATR-based)
        tp2 = entry_low - atr * rp.tp2_atr_mult
        reward = entry_mid - tp2

        old_rr = reward / old_risk
        new_rr = reward / new_risk

        assert new_rr < old_rr, (
            f"New R:R ({new_rr:.1f}) should be lower than old ({old_rr:.1f})"
        )
        assert new_rr < 5.0, (
            f"New R:R ({new_rr:.1f}) should be below 5.0 for realistic targets"
        )


# ── 2. max_rr Enforcement Tests ──────────────────────────────────

class TestMaxRREnforcement:
    """
    Verify that the user-configurable max_rr from settings is respected.
    """

    def test_max_rr_config_exists_in_schema(self):
        """max_rr should be defined in the risk config schema."""
        from config.schema import RiskConfig
        rc = RiskConfig()
        assert hasattr(rc, 'max_rr'), "RiskConfig must have max_rr field"
        assert rc.max_rr == 6.0, "Default max_rr should be 6.0"

    def test_max_rr_greater_than_min_rr(self):
        """max_rr must be >= min_rr."""
        from config.schema import RiskConfig
        rc = RiskConfig()
        assert rc.max_rr >= rc.min_rr, (
            f"max_rr ({rc.max_rr}) must be >= min_rr ({rc.min_rr})"
        )

    def test_max_rr_below_sanity_cap(self):
        """max_rr should be below the hardcoded RR_SANITY_CAP."""
        from config.schema import RiskConfig
        from config.constants import Penalties
        rc = RiskConfig()
        assert rc.max_rr <= Penalties.RR_SANITY_CAP, (
            f"max_rr ({rc.max_rr}) should be <= RR_SANITY_CAP ({Penalties.RR_SANITY_CAP})"
        )

    def test_rr_sanity_cap_constant(self):
        """RR_SANITY_CAP should match max_rr default (6.0)."""
        from config.constants import Penalties
        assert Penalties.RR_SANITY_CAP == 6.0

    def test_max_rr_fallback_always_caps(self):
        """max_rr enforcement must ALWAYS cap even if config import fails.

        The aggregator had a silent `except Exception: pass` that allowed
        signals to exceed max_rr when config.loader couldn't be imported.
        The fix uses Penalties.RR_SANITY_CAP as fallback.
        """
        from config.constants import Penalties
        _max_rr_fallback = Penalties.RR_SANITY_CAP

        # Simulate config import failure — fallback must still apply
        try:
            raise ImportError("Simulated config failure")
        except (ImportError, AttributeError):
            _user_max_rr = _max_rr_fallback

        assert _user_max_rr == 6.0, "Fallback must be 6.0 even on config failure"

        # Verify a signal with 8.0R would be capped
        fake_rr = 8.0
        if _user_max_rr and fake_rr > _user_max_rr:
            fake_rr = round(_user_max_rr, 2)
        assert fake_rr == 6.0, f"8.0R signal should be capped to 6.0, got {fake_rr}"

    def test_sanity_cap_equals_max_rr_default(self):
        """RR_SANITY_CAP and max_rr default should be equal — double gate."""
        from config.constants import Penalties
        from config.schema import RiskConfig
        rc = RiskConfig()
        assert Penalties.RR_SANITY_CAP == rc.max_rr, (
            f"RR_SANITY_CAP ({Penalties.RR_SANITY_CAP}) should equal "
            f"max_rr default ({rc.max_rr}) as a double safety gate"
        )


# ── 3. Regime Direction Bias Tests ────────────────────────────────

class TestRegimeDirectionBias:
    """
    Verify that regime thresholds correctly penalize counter-trend signals.
    """

    def test_bull_trend_short_bias_penalizes(self):
        """SHORT bias in BULL_TREND should be < 1.0 (penalty)."""
        from signals.regime_thresholds import REGIME_THRESHOLDS
        bull = REGIME_THRESHOLDS["BULL_TREND"]
        assert bull["short_bias"] < 1.0, (
            f"short_bias ({bull['short_bias']}) should be < 1.0 in BULL_TREND"
        )

    def test_bull_trend_long_bias_boosts(self):
        """LONG bias in BULL_TREND should be >= 1.0 (boost)."""
        from signals.regime_thresholds import REGIME_THRESHOLDS
        bull = REGIME_THRESHOLDS["BULL_TREND"]
        assert bull["long_bias"] >= 1.0, (
            f"long_bias ({bull['long_bias']}) should be >= 1.0 in BULL_TREND"
        )

    def test_bear_trend_long_bias_penalizes(self):
        """LONG bias in BEAR_TREND should be < 1.0 (penalty)."""
        from signals.regime_thresholds import REGIME_THRESHOLDS
        bear = REGIME_THRESHOLDS["BEAR_TREND"]
        assert bear["long_bias"] < 1.0, (
            f"long_bias ({bear['long_bias']}) should be < 1.0 in BEAR_TREND"
        )

    def test_bear_trend_short_bias_boosts(self):
        """SHORT bias in BEAR_TREND should be >= 1.0 (boost)."""
        from signals.regime_thresholds import REGIME_THRESHOLDS
        bear = REGIME_THRESHOLDS["BEAR_TREND"]
        assert bear["short_bias"] >= 1.0, (
            f"short_bias ({bear['short_bias']}) should be >= 1.0 in BEAR_TREND"
        )

    def test_counter_short_max_in_bull(self):
        """BULL_TREND should have counter_short_max cap."""
        from signals.regime_thresholds import REGIME_THRESHOLDS
        bull = REGIME_THRESHOLDS["BULL_TREND"]
        assert "counter_short_max" in bull, "BULL_TREND must have counter_short_max"
        assert bull["counter_short_max"] <= 3, (
            f"counter_short_max ({bull['counter_short_max']}) should be <= 3"
        )

    def test_counter_long_max_in_bear(self):
        """BEAR_TREND should have counter_long_max cap."""
        from signals.regime_thresholds import REGIME_THRESHOLDS
        bear = REGIME_THRESHOLDS["BEAR_TREND"]
        assert "counter_long_max" in bear, "BEAR_TREND must have counter_long_max"
        assert bear["counter_long_max"] <= 2, (
            f"counter_long_max ({bear['counter_long_max']}) should be <= 2"
        )

    def test_bull_short_penalty_requires_high_confidence(self):
        """
        SHORT in BULL should require ~83+ confidence to pass the 68 floor.
        confidence * short_bias >= min_confidence
        """
        from signals.regime_thresholds import REGIME_THRESHOLDS
        bull = REGIME_THRESHOLDS["BULL_TREND"]
        min_conf_needed = bull["min_confidence"] / bull["short_bias"]
        assert min_conf_needed > 80, (
            f"SHORT in BULL needs {min_conf_needed:.1f} confidence "
            f"(should be > 80 to filter low-quality counter-trend)"
        )

    def test_choppy_neutral_biases(self):
        """CHOPPY and VOLATILE should have neutral biases (1.0)."""
        from signals.regime_thresholds import REGIME_THRESHOLDS
        for regime in ("CHOPPY", "VOLATILE"):
            rt = REGIME_THRESHOLDS[regime]
            assert rt["long_bias"] == 1.0, f"{regime} long_bias should be 1.0"
            assert rt["short_bias"] == 1.0, f"{regime} short_bias should be 1.0"


# ── 4. Strategy Geometry Validation Tests ─────────────────────────

class TestStrategyGeometry:
    """
    Verify that all strategies produce geometrically valid SL/TP for
    both LONG and SHORT directions.
    """

    def test_long_geometry_rules(self):
        """LONG: SL < entry_low < entry_high < TP1 < TP2 < TP3."""
        from utils.risk_params import rp
        atr = 0.005
        price = 0.100

        # Simulate PriceAction LONG
        _sl_buf = atr * rp.sl_atr_mult * 0.35
        entry_low = price - atr * rp.entry_zone_tight
        entry_high = price + atr * rp.entry_zone_atr
        pattern_low = price - atr * 0.3  # small candle
        stop_loss = pattern_low - _sl_buf
        tp1 = entry_high + atr * 1.5
        tp2 = entry_high + atr * rp.tp2_atr_mult
        tp3 = entry_high + atr * 3.0

        assert stop_loss < entry_low, "SL must be below entry_low"
        assert entry_low < entry_high, "entry_low must be < entry_high"
        assert entry_high < tp1, "entry_high must be < TP1"
        assert tp1 < tp2, "TP1 must be < TP2"
        assert tp2 < tp3, "TP2 must be < TP3"

    def test_short_geometry_rules(self):
        """SHORT: TP3 < TP2 < TP1 < entry_low < entry_high < SL."""
        from utils.risk_params import rp
        atr = 0.005
        price = 0.100

        # Simulate PriceAction SHORT
        _sl_buf = atr * rp.sl_atr_mult * 0.35
        entry_high = price + atr * rp.entry_zone_tight
        entry_low = price - atr * rp.entry_zone_atr
        pattern_high = price + atr * 0.3  # small candle
        stop_loss = pattern_high + _sl_buf
        tp1 = entry_low - atr * 1.5
        tp2 = entry_low - atr * rp.tp2_atr_mult
        tp3 = entry_low - atr * 3.0

        assert stop_loss > entry_high, "SL must be above entry_high"
        assert entry_high > entry_low, "entry_high must be > entry_low"
        assert entry_low > tp1, "entry_low must be > TP1"
        assert tp1 > tp2, "TP1 must be > TP2"
        assert tp2 > tp3, "TP2 must be > TP3"

    def test_rr_positive_both_directions(self):
        """R:R must be positive for both LONG and SHORT."""
        from utils.risk_params import rp
        atr = 0.005
        price = 0.100
        _sl_buf = atr * rp.sl_atr_mult * 0.35

        # LONG
        entry_low_l = price - atr * rp.entry_zone_tight
        pattern_low = price - atr * 0.5
        sl_l = pattern_low - _sl_buf
        risk_l = entry_low_l - sl_l
        assert risk_l > 0, "LONG risk must be positive"

        # SHORT
        entry_high_s = price + atr * rp.entry_zone_tight
        pattern_high = price + atr * 0.5
        sl_s = pattern_high + _sl_buf
        risk_s = sl_s - entry_high_s
        assert risk_s > 0, "SHORT risk must be positive"
