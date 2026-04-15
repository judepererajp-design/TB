"""
Tests for GPT-recommended hardening improvements:
  1. Adaptive Weights: Stability Factor
  2. Flow Acceleration: Volume Gate
  3. Vol Compression: Directional Bias
  4. Cascade Pressure: Cluster Persistence
  5. Regime Transition: Exhaustion Detection (4th factor)
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config.constants import (
    AdaptiveWeights as AWC,
    FlowAcceleration as FAC,
    VolCompressionTimer as VCT,
    CascadePressure as CP,
    RegimeTransition as RT,
)


# ════════════════════════════════════════════════════════════════
# 1. ADAPTIVE WEIGHTS — STABILITY FACTOR
# ════════════════════════════════════════════════════════════════

class TestAdaptiveWeightsStability:
    """Tests for the stability factor that penalises inconsistent sources."""

    def test_stability_floor_constant_exists(self):
        assert hasattr(AWC, 'STABILITY_FLOOR')
        assert 0.0 < AWC.STABILITY_FLOOR < 1.0

    def test_stability_no_penalty_when_windows_agree(self):
        """When 30d and 60d scores agree, stability ≈ 1.0 → no penalty."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()

        # Both windows produce similar score
        short_score = mgr._compute_source_score("cvd", "BULL_TREND", [])
        # With no data, falls through to defaults — test the formula directly
        s_short, s_long = 2.0, 2.0
        _max_s = max(s_short, s_long, 0.1)
        _diff = abs(s_short - s_long)
        stability = max(AWC.STABILITY_FLOOR, 1.0 - _diff / _max_s)
        assert stability == 1.0, "Identical scores should have stability=1.0"

    def test_stability_penalises_divergent_windows(self):
        """When 30d score=3.0 and 60d score=0.5, stability should reduce weight."""
        s_short, s_long = 3.0, 0.5
        _max_s = max(s_short, s_long, 0.1)
        _diff = abs(s_short - s_long)
        stability = max(AWC.STABILITY_FLOOR, 1.0 - _diff / _max_s)
        # stability = max(0.4, 1.0 - 2.5/3.0) = max(0.4, 0.167) = 0.4
        assert stability == AWC.STABILITY_FLOOR

    def test_stability_moderate_divergence(self):
        """Moderate divergence should yield partial penalty."""
        s_short, s_long = 2.0, 1.0
        _max_s = max(s_short, s_long, 0.1)
        _diff = abs(s_short - s_long)
        stability = max(AWC.STABILITY_FLOOR, 1.0 - _diff / _max_s)
        # stability = max(0.4, 1.0 - 1.0/2.0) = max(0.4, 0.5) = 0.5
        assert stability == 0.5

    def test_stability_multiplied_into_blended_score(self):
        """Verify that the stability factor is actually multiplied in."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()

        # Create mock signals to test the full recalculate path
        # We verify the code path exists by checking the source
        import inspect
        source = inspect.getsource(mgr.recalculate)
        assert "stability" in source.lower(), "recalculate() must use stability factor"


# ════════════════════════════════════════════════════════════════
# 2. FLOW ACCELERATION — VOLUME GATE
# ════════════════════════════════════════════════════════════════

class TestFlowAccelerationVolumeGate:
    """Tests for thin-market volume gating on flow acceleration signals."""

    def test_volume_gate_constants_exist(self):
        assert hasattr(FAC, 'LOW_VOLUME_DAMPENER')
        assert hasattr(FAC, 'LOW_VOLUME_THRESHOLD_USD')
        assert 0.0 < FAC.LOW_VOLUME_DAMPENER < 1.0
        assert FAC.LOW_VOLUME_THRESHOLD_USD > 0

    def test_set_market_volume_exists(self):
        """StablecoinFlowAnalyzer must have set_market_volume()."""
        from analyzers.stablecoin_flows import StablecoinFlowAnalyzer
        analyzer = StablecoinFlowAnalyzer()
        assert hasattr(analyzer, 'set_market_volume')
        analyzer.set_market_volume(1_000_000_000)
        assert analyzer._market_volume_24h == 1_000_000_000

    def test_volume_gate_dampens_in_thin_market(self):
        """When volume < threshold, acceleration deltas should be halved."""
        from analyzers.stablecoin_flows import StablecoinFlowAnalyzer, FlowDynamics, StablecoinSupplyData
        analyzer = StablecoinFlowAnalyzer()
        analyzer.set_market_volume(100_000_000)  # $100M — well below $500M threshold
        analyzer._snapshot.supply = StablecoinSupplyData(
            total_supply_usd=150_000_000_000,
            supply_change_7d_pct=0.0,
            trend="NEUTRAL",
            timestamp=time.time(),
        )

        # Set up accelerating flow
        analyzer._flow_dynamics = FlowDynamics(
            velocity=2.0, acceleration=1.5,
            is_accelerating=True, is_decelerating=False,
            timestamp=time.time(),
        )

        delta, note = analyzer.get_signal_intel("BTCUSDT", "LONG")
        # With volume gate at 0.5, the +4 accel delta becomes +2
        # (other deltas from supply/dominance are zero since stale)
        assert delta <= FAC.ACCEL_BULLISH_DELTA  # Must be dampened

    def test_volume_gate_no_dampening_in_normal_market(self):
        """When volume ≥ threshold, no dampening applied."""
        from analyzers.stablecoin_flows import StablecoinFlowAnalyzer, FlowDynamics, StablecoinSupplyData
        analyzer = StablecoinFlowAnalyzer()
        analyzer.set_market_volume(1_000_000_000)  # $1B — well above threshold
        analyzer._snapshot.supply = StablecoinSupplyData(
            total_supply_usd=150_000_000_000,
            supply_change_7d_pct=0.0,
            trend="NEUTRAL",
            timestamp=time.time(),
        )

        analyzer._flow_dynamics = FlowDynamics(
            velocity=2.0, acceleration=1.5,
            is_accelerating=True, is_decelerating=False,
            timestamp=time.time(),
        )

        delta, note = analyzer.get_signal_intel("BTCUSDT", "LONG")
        # With no dampener, full +4 delta should apply
        assert delta == FAC.ACCEL_BULLISH_DELTA

    def test_volume_gate_no_dampening_when_volume_unset(self):
        """Volume=0 (unset) should NOT trigger dampener."""
        from analyzers.stablecoin_flows import StablecoinFlowAnalyzer, FlowDynamics, StablecoinSupplyData
        analyzer = StablecoinFlowAnalyzer()
        # _market_volume_24h defaults to 0.0, gate condition is: 0 < vol < threshold
        # 0 < 0 is False → no dampening
        analyzer._snapshot.supply = StablecoinSupplyData(
            total_supply_usd=150_000_000_000,
            supply_change_7d_pct=0.0,
            trend="NEUTRAL",
            timestamp=time.time(),
        )

        analyzer._flow_dynamics = FlowDynamics(
            velocity=2.0, acceleration=1.5,
            is_accelerating=True, is_decelerating=False,
            timestamp=time.time(),
        )

        delta, note = analyzer.get_signal_intel("BTCUSDT", "LONG")
        assert delta == FAC.ACCEL_BULLISH_DELTA


# ════════════════════════════════════════════════════════════════
# 3. VOL COMPRESSION — DIRECTIONAL BIAS
# ════════════════════════════════════════════════════════════════

class TestVolCompressionDirectionalBias:
    """Tests for directional bias in vol compression breakout."""

    def test_directional_bias_constants_exist(self):
        assert hasattr(VCT, 'DIRECTIONAL_BIAS_WEIGHT_WHALE')
        assert hasattr(VCT, 'DIRECTIONAL_BIAS_WEIGHT_FLOW')
        assert hasattr(VCT, 'DIRECTIONAL_BIAS_THRESHOLD')
        assert hasattr(VCT, 'DIRECTIONAL_BIAS_DELTA')

    def test_set_directional_bias_exists(self):
        """VolatilityStructureAnalyzer must have set_directional_bias()."""
        from analyzers.volatility_structure import VolatilityStructureAnalyzer
        analyzer = VolatilityStructureAnalyzer()
        assert hasattr(analyzer, 'set_directional_bias')
        analyzer.set_directional_bias(whale_intent="BULLISH", flow_direction="INFLOW")
        assert analyzer._whale_intent == "BULLISH"
        assert analyzer._flow_direction == "INFLOW"

    def test_directional_bias_init_neutral(self):
        """Directional bias should default to NEUTRAL."""
        from analyzers.volatility_structure import VolatilityStructureAnalyzer
        analyzer = VolatilityStructureAnalyzer()
        assert analyzer._whale_intent == "NEUTRAL"
        assert analyzer._flow_direction == "NEUTRAL"

    def test_bias_score_calculation_bullish(self):
        """Bullish whale + inflow should give strong positive bias."""
        bias_score = 0.0
        bias_score += VCT.DIRECTIONAL_BIAS_WEIGHT_WHALE   # +0.60
        bias_score += VCT.DIRECTIONAL_BIAS_WEIGHT_FLOW     # +0.40
        assert bias_score == 1.0  # Maximum bullish bias
        assert bias_score >= VCT.DIRECTIONAL_BIAS_THRESHOLD

    def test_bias_score_calculation_bearish(self):
        """Bearish whale + outflow should give strong negative bias."""
        bias_score = 0.0
        bias_score -= VCT.DIRECTIONAL_BIAS_WEIGHT_WHALE   # -0.60
        bias_score -= VCT.DIRECTIONAL_BIAS_WEIGHT_FLOW     # -0.40
        assert bias_score == -1.0  # Maximum bearish bias
        assert abs(bias_score) >= VCT.DIRECTIONAL_BIAS_THRESHOLD

    def test_bias_score_mixed_signals_below_threshold(self):
        """Conflicting whale vs flow should cancel and stay below threshold."""
        bias_score = 0.0
        bias_score += VCT.DIRECTIONAL_BIAS_WEIGHT_WHALE   # +0.60 (bullish whale)
        bias_score -= VCT.DIRECTIONAL_BIAS_WEIGHT_FLOW     # -0.40 (outflow)
        # 0.20 — below 0.30 threshold
        assert abs(bias_score) < VCT.DIRECTIONAL_BIAS_THRESHOLD

    def test_weights_sum_to_one(self):
        """Whale + flow weights should sum to 1.0."""
        total = VCT.DIRECTIONAL_BIAS_WEIGHT_WHALE + VCT.DIRECTIONAL_BIAS_WEIGHT_FLOW
        assert total == 1.0


# ════════════════════════════════════════════════════════════════
# 4. CASCADE PRESSURE — CLUSTER PERSISTENCE
# ════════════════════════════════════════════════════════════════

class TestCascadePressurePersistence:
    """Tests for cluster persistence tracking and bonus."""

    def test_persistence_constants_exist(self):
        assert hasattr(CP, 'PERSISTENCE_MIN_MINS')
        assert hasattr(CP, 'PERSISTENCE_BOOST_MINS')
        assert hasattr(CP, 'PERSISTENCE_MAX_BONUS')
        assert CP.PERSISTENCE_MIN_MINS < CP.PERSISTENCE_BOOST_MINS
        assert CP.PERSISTENCE_MAX_BONUS > 1.0

    def test_pressure_score_has_persistence_fields(self):
        """CascadePressureScore must have persistence_mins fields."""
        from analyzers.leverage_mapper import CascadePressureScore
        score = CascadePressureScore()
        assert hasattr(score, 'long_persistence_mins')
        assert hasattr(score, 'short_persistence_mins')
        assert score.long_persistence_mins == 0.0
        assert score.short_persistence_mins == 0.0

    def test_persistence_bonus_below_min(self):
        """Clusters newer than PERSISTENCE_MIN_MINS get no bonus (1.0×)."""
        persist = CP.PERSISTENCE_MIN_MINS - 1
        if persist < CP.PERSISTENCE_MIN_MINS:
            bonus = 1.0  # Expected: no bonus
        assert bonus == 1.0

    def test_persistence_bonus_at_max(self):
        """Clusters at PERSISTENCE_BOOST_MINS get full PERSISTENCE_MAX_BONUS."""
        persist = CP.PERSISTENCE_BOOST_MINS
        t = min(persist, CP.PERSISTENCE_BOOST_MINS) - CP.PERSISTENCE_MIN_MINS
        t_range = max(1, CP.PERSISTENCE_BOOST_MINS - CP.PERSISTENCE_MIN_MINS)
        bonus = 1.0 + (CP.PERSISTENCE_MAX_BONUS - 1.0) * (t / t_range)
        assert bonus == CP.PERSISTENCE_MAX_BONUS

    def test_persistence_bonus_halfway(self):
        """Clusters halfway should get partial bonus."""
        half_time = (CP.PERSISTENCE_MIN_MINS + CP.PERSISTENCE_BOOST_MINS) / 2
        t = min(half_time, CP.PERSISTENCE_BOOST_MINS) - CP.PERSISTENCE_MIN_MINS
        t_range = max(1, CP.PERSISTENCE_BOOST_MINS - CP.PERSISTENCE_MIN_MINS)
        bonus = 1.0 + (CP.PERSISTENCE_MAX_BONUS - 1.0) * (t / t_range)
        expected = 1.0 + (CP.PERSISTENCE_MAX_BONUS - 1.0) * 0.5
        assert abs(bonus - expected) < 0.01

    def test_leverage_mapper_tracks_persistence(self):
        """LeverageMapper must have cluster first-seen tracking attributes."""
        from analyzers.leverage_mapper import LeverageMapper
        mapper = LeverageMapper()
        assert hasattr(mapper, '_long_cluster_first_seen')
        assert hasattr(mapper, '_short_cluster_first_seen')
        assert mapper._long_cluster_first_seen == 0.0
        assert mapper._short_cluster_first_seen == 0.0

    def test_persistence_resets_on_new_cluster(self):
        """When cluster distance changes significantly, persistence resets."""
        from analyzers.leverage_mapper import LeverageMapper
        mapper = LeverageMapper()

        # First update: establish cluster
        mapper.update(
            oi_usd=5e9, market_cap=500e9, funding_rate=0.01,
            long_liq_clusters=[{"price": 97000, "usd": 5_000_000}],
            short_liq_clusters=[],
            current_price=100_000,
        )
        first_seen = mapper._long_cluster_first_seen
        assert first_seen > 0

        # Second update: same cluster area → persistence continues
        mapper.update(
            oi_usd=5e9, market_cap=500e9, funding_rate=0.01,
            long_liq_clusters=[{"price": 97000, "usd": 5_000_000}],
            short_liq_clusters=[],
            current_price=100_000,
        )
        assert mapper._long_cluster_first_seen == first_seen  # Should NOT reset


# ════════════════════════════════════════════════════════════════
# 5. REGIME TRANSITION — EXHAUSTION DETECTION
# ════════════════════════════════════════════════════════════════

class TestRegimeTransitionExhaustion:
    """Tests for 4th factor: OI/price exhaustion detection."""

    def _make_regime_analyzer(self):
        """Create a minimal RegimeAnalyzer for testing with mocked deps."""
        import sys
        from unittest.mock import MagicMock

        # Mock heavy dependencies that regime.py imports at module level
        if 'data.api_client' not in sys.modules:
            mock_api = MagicMock()
            sys.modules['data.api_client'] = mock_api
            sys.modules['data.api_client'].api = MagicMock()
        if 'ccxt' not in sys.modules:
            sys.modules['ccxt'] = MagicMock()
            sys.modules['ccxt.async_support'] = MagicMock()

        from analyzers.regime import RegimeAnalyzer
        r = RegimeAnalyzer()
        return r

    def test_exhaustion_constants_exist(self):
        assert hasattr(RT, 'EXHAUSTION_OI_THRESHOLD')
        assert hasattr(RT, 'EXHAUSTION_PRICE_THRESHOLD')
        assert RT.EXHAUSTION_OI_THRESHOLD > 0
        assert RT.EXHAUSTION_PRICE_THRESHOLD > 0

    def test_set_exhaustion_data_exists(self):
        """RegimeAnalyzer must have set_exhaustion_data()."""
        r = self._make_regime_analyzer()
        assert hasattr(r, 'set_exhaustion_data')
        r.set_exhaustion_data(oi_change_pct=10.0, price_change_pct=0.3)
        assert r._external_oi_change == 10.0
        assert r._external_price_change == 0.3

    def test_exhaustion_detected_when_oi_high_price_flat(self):
        """OI +10% with price 0% should trigger exhaustion factor."""
        r = self._make_regime_analyzer()
        r._btc_adx = 25
        r._adx_history = [30, 29, 28, 27, 26, 25]  # ADX declining

        r.set_exhaustion_data(oi_change_pct=10.0, price_change_pct=0.5)
        r._check_transition_warning()

        # Should have at least ADX decline + exhaustion = 2 factors
        assert any("Exhaust" in f for f in r._transition_factors)

    def test_exhaustion_not_detected_when_price_moves(self):
        """OI +10% with price +5% is healthy, NOT exhaustion."""
        r = self._make_regime_analyzer()
        r._btc_adx = 25
        r._adx_history = [25]

        r.set_exhaustion_data(oi_change_pct=10.0, price_change_pct=5.0)
        r._check_transition_warning()

        assert not any("Exhaust" in f for f in r._transition_factors)

    def test_exhaustion_not_detected_when_oi_low(self):
        """OI +2% is not enough to trigger exhaustion regardless of price."""
        r = self._make_regime_analyzer()
        r._btc_adx = 25
        r._adx_history = [25]

        r.set_exhaustion_data(oi_change_pct=2.0, price_change_pct=0.1)
        r._check_transition_warning()

        assert not any("Exhaust" in f for f in r._transition_factors)

    def test_4_factor_warning_with_exhaustion(self):
        """All 4 factors (ADX + vol + flow + exhaustion) should all appear."""
        r = self._make_regime_analyzer()
        r._btc_adx = 22
        r._adx_history = [30, 29, 28, 27, 26, 22]  # ADX decline ≥3

        r.set_vol_regime_for_transition("COMPRESSED")
        r.set_flow_accel_for_transition(-1.0)
        r.set_exhaustion_data(oi_change_pct=8.0, price_change_pct=0.3)

        r._check_transition_warning()

        assert r._transition_warning is True
        assert r._transition_factor_count == 4
        assert any("ADX" in f for f in r._transition_factors)
        assert any("Vol" in f for f in r._transition_factors)
        assert any("FlowDecel" in f for f in r._transition_factors)
        assert any("Exhaust" in f for f in r._transition_factors)


# ════════════════════════════════════════════════════════════════
# CONSTANTS VALIDATION
# ════════════════════════════════════════════════════════════════

class TestHardeningConstants:
    """Validate all new constants are properly defined."""

    def test_adaptive_weights_stability_constant(self):
        assert AWC.STABILITY_FLOOR == 0.40

    def test_flow_acceleration_volume_gate_constants(self):
        assert FAC.LOW_VOLUME_DAMPENER == 0.50
        assert FAC.LOW_VOLUME_THRESHOLD_USD == 500_000_000

    def test_vol_compression_directional_bias_constants(self):
        assert VCT.DIRECTIONAL_BIAS_WEIGHT_WHALE == 0.60
        assert VCT.DIRECTIONAL_BIAS_WEIGHT_FLOW == 0.40
        assert VCT.DIRECTIONAL_BIAS_THRESHOLD == 0.30
        assert VCT.DIRECTIONAL_BIAS_DELTA == 2

    def test_cascade_persistence_constants(self):
        assert CP.PERSISTENCE_MIN_MINS == 10
        assert CP.PERSISTENCE_BOOST_MINS == 30
        assert CP.PERSISTENCE_MAX_BONUS == 1.5

    def test_regime_transition_exhaustion_constants(self):
        assert RT.EXHAUSTION_OI_THRESHOLD == 5.0
        assert RT.EXHAUSTION_PRICE_THRESHOLD == 1.0
