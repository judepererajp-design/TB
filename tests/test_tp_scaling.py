"""
Tests for TP scaling fixes — verifying that:
1. Base TP multipliers have been lowered (TP1: 1.5, TP2: 2.2)
2. Timeframe scaling produces tighter TPs for scalp timeframes
3. Volatility-scaled TPs exist and work correctly
4. Trailing stop parameters are tightened
5. Max-hold constants are centralized and swing is 72h
"""
import pytest


class TestBaseTPMultipliers:
    """Verify the lowered default TP multipliers."""

    def test_tp1_default_is_1_5(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert rp.tp1_atr_mult == 1.5

    def test_tp2_default_is_2_2(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert rp.tp2_atr_mult == 2.2

    def test_tp3_default_is_4_0(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert rp.tp3_atr_mult == 4.0

    def test_sl_default_unchanged(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert rp.sl_atr_mult == 1.2


class TestTimeframeScaling:
    """Verify that scaled_tp*() applies timeframe factors."""

    def test_5m_scalp_tp2_is_halved(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        # 5m scale = 0.50, so TP2 = 2.2 × 0.50 = 1.1
        assert abs(rp.scaled_tp2('5m') - 2.2 * 0.50) < 0.001

    def test_15m_scalp_tp2_is_reduced(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        # 15m scale = 0.65, so TP2 = 2.2 × 0.65 = 1.43
        assert abs(rp.scaled_tp2('15m') - 2.2 * 0.65) < 0.001

    def test_1h_tp2_is_base(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        # 1h scale = 1.0, so TP2 = 2.2
        assert abs(rp.scaled_tp2('1h') - 2.2) < 0.001

    def test_4h_swing_tp2_is_wider(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        # 4h scale = 1.40, so TP2 = 2.2 × 1.40 = 3.08
        assert abs(rp.scaled_tp2('4h') - 2.2 * 1.40) < 0.001

    def test_1d_position_tp2_is_widest(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        # 1d scale = 2.00, so TP2 = 2.2 × 2.00 = 4.40
        assert abs(rp.scaled_tp2('1d') - 2.2 * 2.00) < 0.001

    def test_tp1_scaled_5m(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert abs(rp.scaled_tp1('5m') - 1.5 * 0.50) < 0.001

    def test_tp3_scaled_4h(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert abs(rp.scaled_tp3('4h') - 4.0 * 1.40) < 0.001

    def test_sl_scaled_5m(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert abs(rp.scaled_sl('5m') - 1.2 * 0.50) < 0.001

    def test_unknown_tf_defaults_to_1x(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert abs(rp.scaled_tp2('3m') - 2.2) < 0.001


class TestVolatilityScaling:
    """Verify volatility-scaled methods produce correct ranges."""

    def test_vol_scaled_tp2_low_vol(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        # vp=0.0: vol_scale = 0.85, tp2 = 2.2 × 1.0 × 0.85 = 1.87
        result = rp.volatility_scaled_tp2('1h', vol_percentile=0.0)
        assert abs(result - 2.2 * 0.85) < 0.01

    def test_vol_scaled_tp2_high_vol(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        # vp=1.0: vol_scale = 0.85 + 0.45 = 1.30, tp2 = 2.2 × 1.0 × 1.30 = 2.86
        result = rp.volatility_scaled_tp2('1h', vol_percentile=1.0)
        assert abs(result - 2.2 * 1.30) < 0.01

    def test_vol_scaled_tp2_mid_vol(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        # vp=0.5: vol_scale = 0.85 + 0.225 = 1.075
        result = rp.volatility_scaled_tp2('1h', vol_percentile=0.5)
        assert abs(result - 2.2 * 1.075) < 0.01

    def test_vol_scaled_combines_with_tf(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        # 5m + low vol: 2.2 × 0.50 × 0.85 = 0.935
        result = rp.volatility_scaled_tp2('5m', vol_percentile=0.0)
        assert abs(result - 2.2 * 0.50 * 0.85) < 0.01

    def test_vol_clamps_above_1(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        result = rp.volatility_scaled_tp2('1h', vol_percentile=5.0)  # Over 1.0
        expected = rp.volatility_scaled_tp2('1h', vol_percentile=1.0)
        assert abs(result - expected) < 0.001

    def test_vol_clamps_below_0(self):
        from utils.risk_params import RiskParams
        rp = RiskParams()
        result = rp.volatility_scaled_tp2('1h', vol_percentile=-1.0)
        expected = rp.volatility_scaled_tp2('1h', vol_percentile=0.0)
        assert abs(result - expected) < 0.001


class TestTrailingStopParams:
    """Verify tightened trailing stop configuration."""

    def test_trail_start_frac(self):
        from config.constants import Timing
        assert Timing.TRAIL_START_FRAC == 0.55

    def test_trail_tp2_frac(self):
        from config.constants import Timing
        assert Timing.TRAIL_TP2_FRAC == 0.80

    def test_trail_cap_frac(self):
        from config.constants import Timing
        assert Timing.TRAIL_CAP_FRAC == 0.90


class TestMaxHoldConstants:
    """Verify centralized max-hold configuration."""

    def test_max_hold_scalp_4h(self):
        from config.constants import Timing
        assert Timing.MAX_HOLD_BY_SETUP['scalp'] == 4 * 3600

    def test_max_hold_intraday_12h(self):
        from config.constants import Timing
        assert Timing.MAX_HOLD_BY_SETUP['intraday'] == 12 * 3600

    def test_max_hold_swing_72h(self):
        from config.constants import Timing
        assert Timing.MAX_HOLD_BY_SETUP['swing'] == 72 * 3600

    def test_max_hold_positional_unlimited(self):
        from config.constants import Timing
        assert Timing.MAX_HOLD_BY_SETUP['positional'] == 0


class TestScalingSanity:
    """Cross-check that scalp TPs are realistic for TP hit rate."""

    def test_5m_scalp_tp2_under_1_5_atr(self):
        """5m scalps should have TP2 well under 1.5× ATR for realistic hit rate."""
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert rp.scaled_tp2('5m') < 1.5

    def test_15m_scalp_tp2_under_2_atr(self):
        """15m scalps should have TP2 under 2× ATR."""
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert rp.scaled_tp2('15m') < 2.0

    def test_4h_swing_tp2_above_2_5_atr(self):
        """4h swings should have TP2 above 2.5× ATR."""
        from utils.risk_params import RiskParams
        rp = RiskParams()
        assert rp.scaled_tp2('4h') > 2.5

    def test_all_tps_ordered_correctly(self):
        """TP1 < TP2 < TP3 for any timeframe."""
        from utils.risk_params import RiskParams
        rp = RiskParams()
        for tf in ('5m', '15m', '1h', '4h', '1d'):
            assert rp.scaled_tp1(tf) < rp.scaled_tp2(tf) < rp.scaled_tp3(tf), \
                f"TP ordering violated for {tf}"
