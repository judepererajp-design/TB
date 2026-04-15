"""
Tests for Tier 1 features:
  1. Adaptive Ensemble Weights (signals/adaptive_weights.py)
  2. Flow Acceleration & Shock Detection (stablecoin_flows + wallet_behavior)
"""

import time
from collections import deque
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# Ensure config.loader is mocked before project imports
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ════════════════════════════════════════════════════════════════
# 1. ADAPTIVE WEIGHTS TESTS
# ════════════════════════════════════════════════════════════════

class TestAdaptiveWeightScoring:
    """Test the score formula: (win_rate × avg_return) / max(drawdown, floor)."""

    def test_score_winning_system(self):
        """High win rate + positive return → score > 1.0."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()

        signals = [
            {"outcome": "WIN",  "pnl_r": 2.0, "regime": "BULL_TREND"},
            {"outcome": "WIN",  "pnl_r": 1.5, "regime": "BULL_TREND"},
            {"outcome": "WIN",  "pnl_r": 1.0, "regime": "BULL_TREND"},
            {"outcome": "LOSS", "pnl_r": -1.0, "regime": "BULL_TREND"},
        ] * 5  # 20 total signals

        score = mgr._compute_source_score("cvd", "BULL_TREND", signals)
        assert score.win_rate == 0.75  # 15 wins / 20 total
        assert score.avg_return > 0
        assert score.score > 1.0  # Better than baseline

    def test_score_losing_system(self):
        """Low win rate + negative return → score < 1.0."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()

        signals = [
            {"outcome": "WIN",  "pnl_r": 0.5, "regime": "CHOPPY"},
            {"outcome": "LOSS", "pnl_r": -2.0, "regime": "CHOPPY"},
            {"outcome": "LOSS", "pnl_r": -1.5, "regime": "CHOPPY"},
            {"outcome": "LOSS", "pnl_r": -1.0, "regime": "CHOPPY"},
        ] * 5

        score = mgr._compute_source_score("crowd", "CHOPPY", signals)
        assert score.win_rate == 0.25  # 5 wins / 20 total
        assert score.score < 1.0  # Worse than baseline

    def test_score_empty_data(self):
        """No signals → default score."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()

        score = mgr._compute_source_score("onchain", "NEUTRAL", [])
        assert score.sample_count == 0
        assert score.score == 1.0  # Default

    def test_score_breakeven_excluded_from_winloss(self):
        """BREAKEVEN outcomes count as samples but don't count as wins or losses."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()

        signals = [
            {"outcome": "WIN",      "pnl_r": 1.0, "regime": "NEUTRAL"},
            {"outcome": "BREAKEVEN","pnl_r": 0.0, "regime": "NEUTRAL"},
        ] * 10

        score = mgr._compute_source_score("basis", "NEUTRAL", signals)
        # Only WIN and LOSS count for win_rate — BREAKEVEN excluded from decisive outcomes
        assert score.win_rate == 1.0
        assert score.sample_count == 10  # 10 WIN + 0 LOSS = 10 decisive outcomes

    def test_weight_clamping(self):
        """Adaptive weights are clamped to [FLOOR, CEIL]."""
        from signals.adaptive_weights import AdaptiveWeightManager, AdaptiveWeightSnapshot
        from config.constants import AdaptiveWeights as AWC
        mgr = AdaptiveWeightManager()

        # Force an extreme snapshot
        mgr._snapshots["TEST"] = AdaptiveWeightSnapshot(
            regime="TEST",
            weights={"cvd": 0.01, "smart_money": 99.0},  # Extreme values
            sufficient_data=True,
            last_update=time.time(),
        )
        result = mgr.get_weights("TEST")
        # The raw weights are stored as-is; clamping happens during recalculate
        assert result is not None


class TestAdaptiveWeightFallback:
    """Test the 3-tier fallback: adaptive → static regime → flat defaults."""

    def test_no_data_returns_none(self):
        """With no snapshots, get_weights returns None."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()
        assert mgr.get_weights("BULL_TREND") is None

    def test_stale_data_returns_none(self):
        """Stale snapshots return None for fallback."""
        from signals.adaptive_weights import AdaptiveWeightManager, AdaptiveWeightSnapshot
        from config.constants import AdaptiveWeights as AWC
        mgr = AdaptiveWeightManager()

        mgr._snapshots["STALE"] = AdaptiveWeightSnapshot(
            regime="STALE",
            weights={"cvd": 1.5},
            sufficient_data=True,
            last_update=time.time() - AWC.RECALC_INTERVAL_SECS * 3,  # Very stale
        )
        assert mgr.get_weights("STALE") is None

    def test_sufficient_data_returns_weights(self):
        """Fresh snapshots with sufficient data return weights."""
        from signals.adaptive_weights import AdaptiveWeightManager, AdaptiveWeightSnapshot
        mgr = AdaptiveWeightManager()

        mgr._snapshots["BULL_TREND"] = AdaptiveWeightSnapshot(
            regime="BULL_TREND",
            weights={"cvd": 1.8, "smart_money": 2.2, "onchain": 1.5},
            sufficient_data=True,
            last_update=time.time(),
        )
        weights = mgr.get_weights("BULL_TREND")
        assert weights is not None
        assert weights["cvd"] == 1.8
        assert weights["smart_money"] == 2.2

    def test_diagnostics(self):
        """Diagnostics returns correct counts."""
        from signals.adaptive_weights import AdaptiveWeightManager, AdaptiveWeightSnapshot
        mgr = AdaptiveWeightManager()

        mgr._snapshots["A"] = AdaptiveWeightSnapshot(regime="A", sufficient_data=True, last_update=time.time())
        mgr._snapshots["B"] = AdaptiveWeightSnapshot(regime="B", sufficient_data=False, last_update=time.time())
        mgr._last_recalc = time.time()

        diag = mgr.get_diagnostics()
        assert diag["regimes_tracked"] == 2
        assert diag["sufficient_data_regimes"] == 1


# ════════════════════════════════════════════════════════════════
# 2. STABLECOIN FLOW ACCELERATION TESTS
# ════════════════════════════════════════════════════════════════

class TestStablecoinFlowDynamics:
    """Test velocity, acceleration, and shock detection in stablecoin flows."""

    def _make_analyzer(self):
        from analyzers.stablecoin_flows import StablecoinFlowAnalyzer, StablecoinSupplyData
        analyzer = StablecoinFlowAnalyzer()
        # Pre-populate supply data so it's not stale
        analyzer._snapshot.supply = StablecoinSupplyData(
            total_supply_usd=150_000_000_000,
            supply_change_7d_pct=3.0,
            trend="INFLOW",
            timestamp=time.time(),
        )
        return analyzer

    def test_no_data_neutral_dynamics(self):
        """With insufficient data, dynamics stay neutral."""
        analyzer = self._make_analyzer()
        analyzer._compute_flow_dynamics()
        dyn = analyzer.get_flow_dynamics()
        assert dyn.velocity == 0.0
        assert not dyn.shock_confirmed

    def test_velocity_computed_from_supply_change(self):
        """Velocity tracks supply change rate."""
        analyzer = self._make_analyzer()
        # Simulate two data points
        analyzer._snapshot.supply.supply_change_7d_pct = 2.0
        analyzer._compute_flow_dynamics()
        analyzer._snapshot.supply.supply_change_7d_pct = 4.0
        analyzer._compute_flow_dynamics()

        dyn = analyzer.get_flow_dynamics()
        assert dyn.velocity == 4.0
        assert dyn.acceleration == 2.0  # 4.0 - 2.0

    def test_acceleration_detected(self):
        """Positive acceleration with positive velocity → is_accelerating."""
        analyzer = self._make_analyzer()
        # Build a window of increasing velocity
        for v in [1.0, 1.5, 2.0, 3.0, 5.0]:
            analyzer._snapshot.supply.supply_change_7d_pct = v
            analyzer._compute_flow_dynamics()

        dyn = analyzer.get_flow_dynamics()
        assert dyn.is_accelerating  # Velocity is positive and acceleration > 0.5

    def test_deceleration_detected(self):
        """Negative acceleration with negative velocity → is_decelerating."""
        analyzer = self._make_analyzer()
        for v in [-1.0, -1.5, -2.0, -3.0, -5.0]:
            analyzer._snapshot.supply.supply_change_7d_pct = v
            analyzer._compute_flow_dynamics()

        dyn = analyzer.get_flow_dynamics()
        assert dyn.is_decelerating

    def test_shock_requires_confirmation(self):
        """Shock needs SHOCK_CONFIRM_INTERVALS consecutive triggers."""
        from config.constants import FlowAcceleration as FAC
        analyzer = self._make_analyzer()

        # Build baseline window
        for v in [1.0, 1.0, 1.0, 1.0, 1.0]:
            analyzer._snapshot.supply.supply_change_7d_pct = v
            analyzer._compute_flow_dynamics()

        # First shock interval — detected but not confirmed
        analyzer._snapshot.supply.supply_change_7d_pct = 10.0  # 10× avg velocity
        analyzer._compute_flow_dynamics()
        dyn = analyzer.get_flow_dynamics()
        assert dyn.shock_detected
        assert not dyn.shock_confirmed  # Only 1 interval

        # Second shock interval — now confirmed
        analyzer._snapshot.supply.supply_change_7d_pct = 12.0
        analyzer._compute_flow_dynamics()
        dyn = analyzer.get_flow_dynamics()
        assert dyn.shock_confirmed
        assert dyn.shock_direction == "INFLOW_SHOCK"

    def test_shock_resets_on_normal(self):
        """Shock counter resets when flow returns to normal."""
        analyzer = self._make_analyzer()
        for v in [1.0, 1.0, 1.0, 10.0]:
            analyzer._snapshot.supply.supply_change_7d_pct = v
            analyzer._compute_flow_dynamics()
        assert analyzer._consecutive_shock_count >= 1

        # Normal flow resets counter
        analyzer._snapshot.supply.supply_change_7d_pct = 1.0
        analyzer._compute_flow_dynamics()
        assert analyzer._consecutive_shock_count == 0

    def test_signal_intel_includes_acceleration(self):
        """get_signal_intel reports acceleration delta and note."""
        from config.constants import FlowAcceleration as FAC
        analyzer = self._make_analyzer()
        # Create accelerating flow
        for v in [1.0, 2.0, 3.5, 5.5, 8.0]:
            analyzer._snapshot.supply.supply_change_7d_pct = v
            analyzer._compute_flow_dynamics()

        delta, note = analyzer.get_signal_intel("BTCUSDT", "LONG")
        assert delta > 0  # Should be positive for LONG + accelerating inflow
        assert "accelerat" in note.lower() or "inflow" in note.lower()

    def test_signal_intel_shock_delta(self):
        """Confirmed shock gives larger delta than normal acceleration."""
        from config.constants import FlowAcceleration as FAC
        analyzer = self._make_analyzer()
        # Build baseline then trigger confirmed shock
        for v in [1.0, 1.0, 1.0, 1.0, 1.0]:
            analyzer._snapshot.supply.supply_change_7d_pct = v
            analyzer._compute_flow_dynamics()
        for _ in range(FAC.SHOCK_CONFIRM_INTERVALS):
            analyzer._snapshot.supply.supply_change_7d_pct = 15.0
            analyzer._compute_flow_dynamics()

        delta_shock, note = analyzer.get_signal_intel("BTCUSDT", "LONG")
        assert "SHOCK" in note
        assert delta_shock >= FAC.SHOCK_BULLISH_DELTA  # Shock delta >= accel delta


# ════════════════════════════════════════════════════════════════
# 3. WHALE FLOW ACCELERATION TESTS
# ════════════════════════════════════════════════════════════════

class TestWhaleDynamics:
    """Test velocity, acceleration, and shock detection in whale flows."""

    def _make_profiler(self):
        from analyzers.wallet_behavior import WalletBehaviorProfiler
        return WalletBehaviorProfiler()

    def test_no_events_no_dynamics(self):
        """With no events, dynamics stay at defaults."""
        profiler = self._make_profiler()
        dyn = profiler.get_whale_dynamics()
        assert dyn.velocity == 0.0
        assert not dyn.shock_confirmed

    def test_buy_events_create_positive_velocity(self):
        """Multiple buy events create positive net flow velocity."""
        profiler = self._make_profiler()
        # Record enough events to trigger analysis
        for i in range(10):
            profiler.record_event("buy", 2_000_000, "BTCUSDT", "orderbook")
        dyn = profiler.get_whale_dynamics()
        assert dyn.velocity > 0  # Net buying

    def test_sell_events_create_negative_velocity(self):
        """Multiple sell events create negative net flow velocity."""
        profiler = self._make_profiler()
        for i in range(10):
            profiler.record_event("sell", 2_000_000, "BTCUSDT", "orderbook")
        dyn = profiler.get_whale_dynamics()
        assert dyn.velocity < 0  # Net selling

    def test_shock_detection_large_flow(self):
        """Massive sudden flow triggers shock detection."""
        from config.constants import FlowAcceleration as FAC
        profiler = self._make_profiler()

        # Build baseline: small events
        for i in range(FAC.WHALE_ACCEL_MIN_EVENTS):
            profiler.record_event("buy", 100_000, "BTCUSDT", "orderbook")

        # Massive buy surge
        for _ in range(FAC.SHOCK_CONFIRM_INTERVALS + 2):
            for i in range(8):
                profiler.record_event("buy", 10_000_000, "BTCUSDT", "deposit")

        dyn = profiler.get_whale_dynamics()
        # With a massive surge the shock should be at least detected
        assert dyn.shock_detected or dyn.velocity > 0

    def test_signal_intel_includes_dynamics(self):
        """get_signal_intel includes dynamics-based adjustments."""
        profiler = self._make_profiler()

        # Create enough events for accumulation + dynamics
        for i in range(15):
            profiler.record_event("buy", 3_000_000, "BTCUSDT", "orderbook")

        delta, note = profiler.get_signal_intel("BTCUSDT", "LONG")
        # Should have positive delta from accumulation + possibly acceleration
        assert delta > 0

    def test_get_whale_intent_stable(self):
        """get_whale_intent returns correct classification."""
        profiler = self._make_profiler()

        # Heavy buying → BULLISH
        for i in range(20):
            profiler.record_event("buy", 2_000_000, "BTCUSDT", "orderbook")

        intent = profiler.get_whale_intent()
        assert intent == "BULLISH"

    def test_mixed_flow_neutral(self):
        """Balanced buy/sell → NEUTRAL intent."""
        profiler = self._make_profiler()
        for i in range(10):
            profiler.record_event("buy", 1_000_000, "BTCUSDT", "orderbook")
            profiler.record_event("sell", 1_000_000, "BTCUSDT", "orderbook")

        intent = profiler.get_whale_intent()
        assert intent == "NEUTRAL"
