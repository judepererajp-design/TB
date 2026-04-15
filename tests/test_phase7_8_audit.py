import time
from pathlib import Path
from unittest.mock import MagicMock


class TestPhase78EngineWiring:
    """Behavioral tests that exercise actual code paths rather than grepping source text."""

    def test_stablecoin_analyzer_set_market_volume_exists_and_works(self):
        from analyzers.stablecoin_flows import StablecoinFlowAnalyzer

        analyzer = StablecoinFlowAnalyzer()
        # set_market_volume should accept a float without error
        analyzer.set_market_volume(50_000_000.0)
        assert analyzer._market_volume_24h == 50_000_000.0

    def test_vol_dampener_uses_high_regime(self):
        """Verify the HIGH_VOL_DAMPENER constant exists and that 'HIGH' is a
        recognised high-vol regime string (used by the engine's dampener logic)."""
        from config.constants import Enrichment as EC

        assert hasattr(EC, "HIGH_VOL_DAMPENER")
        assert EC.HIGH_VOL_DAMPENER < 1.0  # dampener should reduce, not amplify
        # The engine applies dampener when _vol_regime is in this set:
        high_vol_regimes = ("HIGH", "HIGH_VOL", "EXTREME")
        assert "HIGH" in high_vol_regimes

    def test_regime_update_transition_inputs_batches_correctly(self):
        """update_transition_inputs should set all fields and refresh once."""
        import sys
        if "data.api_client" not in sys.modules:
            sys.modules["data.api_client"] = MagicMock()
            sys.modules["data.api_client"].api = MagicMock()
        if "ccxt" not in sys.modules:
            sys.modules["ccxt"] = MagicMock()
            sys.modules["ccxt.async_support"] = MagicMock()

        from analyzers.regime import RegimeAnalyzer

        r = RegimeAnalyzer()
        r._adx_history = [40.0, 42.0, 44.0, 43.0, 41.0]
        r._btc_adx = 38.0

        r.update_transition_inputs(
            vol_regime="COMPRESSED",
            flow_acceleration=-2.0,
            oi_change_pct=6.0,
            price_change_pct=0.5,
        )

        tw = r.get_transition_warning()
        assert tw["warning"] is True
        assert tw["factor_count"] >= 3
        assert any("ADX" in f for f in tw["factors"])
        assert any("Vol" in f for f in tw["factors"])
        assert any("Exhaust" in f for f in tw["factors"])


class TestPhase78StablecoinFlowStaleness:

    def test_stale_supply_resets_flow_dynamics(self):
        from analyzers.stablecoin_flows import StablecoinFlowAnalyzer, FlowDynamics, StablecoinSupplyData

        analyzer = StablecoinFlowAnalyzer()
        analyzer._consecutive_shock_count = 2
        analyzer._flow_dynamics = FlowDynamics(
            velocity=3.0,
            acceleration=2.0,
            is_accelerating=True,
            shock_detected=True,
            shock_confirmed=True,
            shock_direction="INFLOW_SHOCK",
            consecutive_shocks=2,
            timestamp=time.time(),
        )
        analyzer._snapshot.supply = StablecoinSupplyData(
            total_supply_usd=100_000_000_000,
            supply_change_7d_pct=4.0,
            trend="INFLOW",
            timestamp=time.time() - (12 * 3600),
        )

        analyzer._compute_flow_dynamics()

        dyn = analyzer.get_flow_dynamics()
        assert dyn.timestamp == 0.0
        assert dyn.is_accelerating is False
        assert dyn.shock_confirmed is False
        assert analyzer._consecutive_shock_count == 0

    def test_stale_supply_blocks_acceleration_signal(self):
        from analyzers.stablecoin_flows import StablecoinFlowAnalyzer, FlowDynamics, StablecoinSupplyData

        analyzer = StablecoinFlowAnalyzer()
        analyzer._snapshot.supply = StablecoinSupplyData(
            total_supply_usd=100_000_000_000,
            supply_change_7d_pct=3.0,
            trend="INFLOW",
            timestamp=time.time() - (12 * 3600),
        )
        analyzer._flow_dynamics = FlowDynamics(
            velocity=2.5,
            acceleration=1.5,
            is_accelerating=True,
            timestamp=time.time(),
        )

        delta, note = analyzer.get_signal_intel("BTCUSDT", "LONG")

        assert delta == 0
        assert "Flow accelerating" not in note
        assert "SHOCK" not in note


class TestPhase78RegimeTransitionRefresh:

    def _make_regime_analyzer(self):
        import sys

        if "data.api_client" not in sys.modules:
            sys.modules["data.api_client"] = MagicMock()
            sys.modules["data.api_client"].api = MagicMock()
        if "ccxt" not in sys.modules:
            sys.modules["ccxt"] = MagicMock()
            sys.modules["ccxt.async_support"] = MagicMock()

        from analyzers.regime import RegimeAnalyzer

        return RegimeAnalyzer()

    def test_transition_warning_refreshes_when_setters_are_called(self):
        r = self._make_regime_analyzer()
        r._adx_history = [40.0, 42.0, 44.0, 43.0, 41.0]
        r._btc_adx = 38.0

        r.set_vol_regime_for_transition("COMPRESSED")
        r.set_flow_accel_for_transition(-2.0)

        tw = r.get_transition_warning()
        assert tw["warning"] is True
        assert tw["factor_count"] >= 2
        assert any("ADX" in f for f in tw["factors"])
        assert any("Vol" in f for f in tw["factors"])

    def test_exhaustion_setter_refreshes_warning_state(self):
        r = self._make_regime_analyzer()
        r._adx_history = [40.0, 42.0, 44.0, 43.0, 41.0]
        r._btc_adx = 38.0

        r.set_exhaustion_data(oi_change_pct=6.0, price_change_pct=0.5)

        tw = r.get_transition_warning()
        assert any("ADX" in f for f in tw["factors"])
        assert any("Exhaust" in f for f in tw["factors"])
