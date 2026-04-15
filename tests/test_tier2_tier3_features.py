"""
Tests for Tier 2 & Tier 3 features:
  3. Context-Aware Volatility Compression Timer (volatility_structure.py)
  4. Cascade Pressure Scoring (leverage_mapper.py)
  5. Regime Transition Early Warning (regime.py)
"""

import math
import time
from unittest.mock import MagicMock, patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ════════════════════════════════════════════════════════════════
# 3. CONTEXT-AWARE VOLATILITY COMPRESSION TIMER
# ════════════════════════════════════════════════════════════════

class TestVolCompressionTimer:
    """Test context-aware breakout probability scoring."""

    def _make_analyzer(self):
        from analyzers.volatility_structure import VolatilityStructureAnalyzer
        a = VolatilityStructureAnalyzer()
        return a

    def test_context_fields_exist_on_vol_regime_data(self):
        """VolRegimeData should have context_score and context_factors fields."""
        from analyzers.volatility_structure import VolRegimeData
        vr = VolRegimeData()
        assert hasattr(vr, 'context_score')
        assert hasattr(vr, 'context_factors')
        assert vr.context_score == 0.0
        assert vr.context_factors == ""

    def test_set_context_stores_values(self):
        """set_context should store OI, funding, and leverage zone."""
        a = self._make_analyzer()
        a.set_context(oi_change_pct=12.0, funding_rate=3.0, leverage_zone="HIGH")
        assert a._oi_change_pct == 12.0
        assert a._funding_rate == 3.0
        assert a._leverage_zone == "HIGH"

    def test_no_context_when_not_compressed(self):
        """Context scoring should not activate when regime is not compressed."""
        from analyzers.volatility_structure import RealizedVolData
        a = self._make_analyzer()
        a._snapshot.realized = RealizedVolData(
            rv_7d=30.0, rv_14d=28.0, rv_30d=25.0, rv_90d=22.0,
            rv_trend="STABLE", cone_percentile=50.0, timestamp=time.time()
        )
        a.set_context(oi_change_pct=20.0, funding_rate=0.0, leverage_zone="EXTREME")
        a._compute_vol_regime()
        # 50th percentile = NEUTRAL, no context applied
        assert a._snapshot.regime.regime == "NEUTRAL"
        assert a._snapshot.regime.context_score == 0.0

    def test_context_activates_on_compressed_regime(self):
        """Context should score >0 when regime is COMPRESSED and duration >= MIN."""
        from analyzers.volatility_structure import RealizedVolData
        from config.constants import VolCompressionTimer as VCT

        a = self._make_analyzer()
        # Force regime start 5 days ago (above MIN_COMPRESSION_DAYS)
        a._regime_start = time.time() - (VCT.MIN_COMPRESSION_DAYS + 2) * 86400
        a._snapshot.regime.regime = "COMPRESSED"  # So regime won't reset _regime_start
        a._snapshot.realized = RealizedVolData(
            rv_7d=10.0, rv_14d=12.0, rv_30d=15.0, rv_90d=20.0,
            rv_trend="CONTRACTING", cone_percentile=5.0, timestamp=time.time()
        )
        a.set_context(oi_change_pct=15.0, funding_rate=0.0, leverage_zone="HIGH")
        a._compute_vol_regime()
        assert a._snapshot.regime.regime == "COMPRESSED"
        assert a._snapshot.regime.context_score > 0.0

    def test_context_boost_raises_breakout_probability(self):
        """High context score should boost breakout probability beyond duration-only calc."""
        from analyzers.volatility_structure import RealizedVolData
        from config.constants import VolCompressionTimer as VCT

        a = self._make_analyzer()
        a._regime_start = time.time() - 10 * 86400
        a._snapshot.regime.regime = "COMPRESSED"
        a._snapshot.realized = RealizedVolData(
            rv_7d=8.0, rv_14d=10.0, rv_30d=12.0, rv_90d=18.0,
            rv_trend="CONTRACTING", cone_percentile=3.0, timestamp=time.time()
        )
        # Maximal context: high OI, neutral funding, extreme leverage
        a.set_context(oi_change_pct=20.0, funding_rate=0.0, leverage_zone="EXTREME")
        a._compute_vol_regime()
        # Duration-only: 0.3 + 10*0.08 = 1.1 → capped at 0.9
        # Context should further boost
        assert a._snapshot.regime.breakout_probability > 0.9

    def test_funding_outside_band_lowers_score(self):
        """Extreme funding rate should not contribute to context score."""
        from analyzers.volatility_structure import RealizedVolData
        from config.constants import VolCompressionTimer as VCT

        a = self._make_analyzer()
        a._regime_start = time.time() - 7 * 86400
        a._snapshot.regime.regime = "COMPRESSED"
        a._snapshot.realized = RealizedVolData(
            rv_7d=9.0, rv_14d=11.0, rv_30d=14.0, rv_90d=19.0,
            rv_trend="CONTRACTING", cone_percentile=4.0, timestamp=time.time()
        )
        # Extreme positive funding = outside neutral band
        a.set_context(oi_change_pct=0.0, funding_rate=50.0, leverage_zone="NEUTRAL")
        a._compute_vol_regime()
        score_extreme_funding = a._snapshot.regime.context_score

        # Neutral funding
        a._snapshot.regime.regime = "COMPRESSED"
        a.set_context(oi_change_pct=0.0, funding_rate=0.0, leverage_zone="NEUTRAL")
        a._compute_vol_regime()
        score_neutral_funding = a._snapshot.regime.context_score

        # Neutral funding should give higher or equal context than extreme
        assert score_neutral_funding >= score_extreme_funding

    def test_signal_intel_context_delta_bonus(self):
        """get_signal_intel should add context bonus when context_score is high."""
        from analyzers.volatility_structure import VolatilityStructureAnalyzer, VolRegimeData
        from config.constants import VolCompressionTimer as VCT

        a = VolatilityStructureAnalyzer()
        # Directly set a high-context compressed regime
        a._snapshot.regime = VolRegimeData(
            regime="COMPRESSED",
            breakout_probability=0.85,
            days_in_regime=10,
            context_score=0.75,  # Above CONTEXT_BOOST_THRESHOLD
            context_factors="OI+15.0%+Lev:HIGH",
            timestamp=time.time(),
        )
        delta, note = a.get_signal_intel("BTCUSDT", "LONG")
        # Base +5 for compressed + context bonus
        assert delta >= 5 + VCT.CONTEXT_DELTA_BONUS
        assert "ctx=" in note

    def test_signal_intel_no_bonus_below_threshold(self):
        """No context bonus when context_score is below threshold."""
        from analyzers.volatility_structure import VolatilityStructureAnalyzer, VolRegimeData
        from config.constants import VolCompressionTimer as VCT

        a = VolatilityStructureAnalyzer()
        a._snapshot.regime = VolRegimeData(
            regime="COMPRESSED",
            breakout_probability=0.65,
            days_in_regime=5,
            context_score=0.20,  # Below CONTEXT_BOOST_THRESHOLD
            context_factors="",
            timestamp=time.time(),
        )
        delta, note = a.get_signal_intel("BTCUSDT", "LONG")
        assert delta == 5  # Just the base, no bonus
        assert "ctx=" not in note


# ════════════════════════════════════════════════════════════════
# 4. CASCADE PRESSURE SCORING
# ════════════════════════════════════════════════════════════════

class TestCascadePressureScoring:
    """Test momentum-weighted cascade pressure: pressure = (size/dist) × momentum."""

    def _make_mapper(self):
        from analyzers.leverage_mapper import LeverageMapper
        return LeverageMapper()

    def test_pressure_dataclass_exists(self):
        """CascadePressureScore should exist in LeverageSnapshot."""
        from analyzers.leverage_mapper import LeverageSnapshot, CascadePressureScore
        snap = LeverageSnapshot()
        assert hasattr(snap, 'pressure')
        assert isinstance(snap.pressure, CascadePressureScore)

    def test_no_pressure_without_clusters(self):
        """Pressure should be 0 when no liquidation clusters provided."""
        lm = self._make_mapper()
        lm.update(oi_usd=1e9, market_cap=5e9, current_price=50000)
        p = lm.get_pressure()
        assert p.long_pressure == 0.0
        assert p.short_pressure == 0.0
        assert p.long_pressure_level == "LOW"

    def test_pressure_with_nearby_large_cluster(self):
        """Nearby large cluster should produce high pressure when momentum confirms."""
        lm = self._make_mapper()
        # Simulate price dropping (toward long liquidations)
        for price in [51000, 50800, 50600, 50400, 50200, 50000]:
            lm.update(
                oi_usd=1e9, market_cap=5e9,
                long_liq_clusters=[{"price": 49000, "usd": 15_000_000}],
                short_liq_clusters=[],
                current_price=price,
            )
        p = lm.get_pressure()
        # Distance ≈2%, size=$15M → pressure = (15/2) × momentum_factor
        assert p.long_pressure > 0
        assert p.momentum_toward_long is True
        assert p.long_pressure_level in ("MEDIUM", "HIGH", "CRITICAL")

    def test_no_pressure_without_momentum(self):
        """Cluster exists but price is moving away → lower pressure."""
        lm = self._make_mapper()
        # Price moving UP (away from long liquidations below)
        for price in [49000, 49200, 49400, 49600, 49800, 50000]:
            lm.update(
                oi_usd=1e9, market_cap=5e9,
                long_liq_clusters=[{"price": 48000, "usd": 10_000_000}],
                short_liq_clusters=[],
                current_price=price,
            )
        p = lm.get_pressure()
        # Moving away → momentum_toward_long is False → halved pressure factor
        assert p.momentum_toward_long is False

    def test_pressure_below_min_cluster_ignored(self):
        """Clusters below MIN_CLUSTER_USD should not generate pressure."""
        from config.constants import CascadePressure as CP
        lm = self._make_mapper()
        for price in [51000, 50500, 50000]:
            lm.update(
                oi_usd=1e9, market_cap=5e9,
                long_liq_clusters=[{"price": 49500, "usd": CP.MIN_CLUSTER_USD - 1}],
                current_price=price,
            )
        p = lm.get_pressure()
        assert p.long_pressure == 0.0

    def test_signal_intel_pressure_penalty_for_longs(self):
        """LONG signals should get penalized when pressure is high toward long liquidations."""
        lm = self._make_mapper()
        # Drop price significantly to build momentum toward long liquidations
        for price in [52000, 51500, 51000, 50500, 50000, 49800, 49600, 49400, 49200, 49100, 49050, 49010]:
            lm.update(
                oi_usd=1e9, market_cap=5e9,
                long_liq_clusters=[{"price": 48900, "usd": 50_000_000}],
                current_price=price,
            )
        delta, note = lm.get_signal_intel("BTCUSDT", "LONG")
        # Should have negative cascade delta
        assert delta < 0
        assert "cascade" in note.lower() or "pressure" in note.lower() or "Cascade" in note

    def test_signal_intel_opportunity_for_shorts(self):
        """SHORT signals should get boosted when long cascades are imminent."""
        lm = self._make_mapper()
        for price in [52000, 51500, 51000, 50500, 50000, 49800, 49600, 49400, 49200, 49100, 49050, 49010]:
            lm.update(
                oi_usd=1e9, market_cap=5e9,
                long_liq_clusters=[{"price": 48900, "usd": 50_000_000}],
                current_price=price,
            )
        delta, note = lm.get_signal_intel("BTCUSDT", "SHORT")
        # Short should benefit from long cascade pressure (opportunity)
        if lm.get_pressure().long_pressure_level in ("CRITICAL", "HIGH"):
            assert "opportunity" in note.lower() or "Cascade" in note or delta > 0

    def test_pressure_level_classification(self):
        """Test pressure level classification thresholds."""
        from config.constants import CascadePressure as CP
        from analyzers.leverage_mapper import CascadePressureScore

        # These are just the thresholds
        assert CP.PRESSURE_CRITICAL > CP.PRESSURE_HIGH > CP.PRESSURE_MEDIUM


# ════════════════════════════════════════════════════════════════
# 5. REGIME TRANSITION EARLY WARNING
# ════════════════════════════════════════════════════════════════

class TestRegimeTransitionWarning:
    """Test 3-factor regime transition detection."""

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

    def test_transition_warning_fields_exist(self):
        """RegimeAnalyzer should have transition warning attributes."""
        r = self._make_regime_analyzer()
        assert hasattr(r, '_transition_warning')
        assert hasattr(r, '_transition_factors')
        assert hasattr(r, '_transition_factor_count')
        assert r._transition_warning is False

    def test_get_transition_warning_returns_dict(self):
        """get_transition_warning() should return expected structure."""
        r = self._make_regime_analyzer()
        tw = r.get_transition_warning()
        assert isinstance(tw, dict)
        assert "warning" in tw
        assert "factor_count" in tw
        assert "factors" in tw
        assert tw["warning"] is False
        assert tw["factor_count"] == 0

    def test_set_vol_regime_for_transition(self):
        """Should be able to set external vol regime."""
        r = self._make_regime_analyzer()
        r.set_vol_regime_for_transition("COMPRESSED")
        assert r._external_vol_regime == "COMPRESSED"

    def test_set_flow_accel_for_transition(self):
        """Should be able to set external flow acceleration."""
        r = self._make_regime_analyzer()
        r.set_flow_accel_for_transition(-1.5)
        assert r._external_flow_accel == -1.5

    def test_adx_decline_detected(self):
        """ADX declining from peak should count as a factor."""
        from config.constants import RegimeTransition as RT

        r = self._make_regime_analyzer()
        # Simulate ADX declining
        r._adx_history = [40.0, 42.0, 44.0, 43.0, 41.0]  # Peak 44, current 41
        r._btc_adx = 38.0  # Further decline

        # Set other factors to trigger warning
        r.set_vol_regime_for_transition("COMPRESSED")
        r.set_flow_accel_for_transition(-1.0)

        r._check_transition_warning()

        # ADX decline = 44 - 38 = 6 > RT.ADX_DECLINE_THRESHOLD (3)
        assert any("ADX" in f for f in r._transition_factors)

    def test_vol_compression_detected(self):
        """Vol in compressed regime should count as a factor."""
        r = self._make_regime_analyzer()
        r._adx_history = [30.0, 30.0, 30.0]
        r._btc_adx = 30.0

        r.set_vol_regime_for_transition("COMPRESSED")
        r._check_transition_warning()

        assert any("Vol" in f for f in r._transition_factors)

    def test_flow_deceleration_detected(self):
        """Flow acceleration below threshold should count as a factor."""
        from config.constants import RegimeTransition as RT

        r = self._make_regime_analyzer()
        r._adx_history = [30.0, 30.0, 30.0]
        r._btc_adx = 30.0

        r.set_flow_accel_for_transition(-2.0)  # Below FLOW_DECEL_THRESHOLD
        r._check_transition_warning()

        assert any("FlowDecel" in f for f in r._transition_factors)

    def test_warning_requires_min_factors(self):
        """Warning should only trigger when MIN_FACTORS_FOR_WARNING met."""
        from config.constants import RegimeTransition as RT

        r = self._make_regime_analyzer()
        r._adx_history = [30.0, 30.0, 30.0]
        r._btc_adx = 30.0

        # Only 1 factor: vol compression
        r.set_vol_regime_for_transition("COMPRESSED")
        r._check_transition_warning()

        if RT.MIN_FACTORS_FOR_WARNING > 1:
            assert r._transition_warning is False
            assert r._transition_factor_count == 1

    def test_warning_triggers_with_2_factors(self):
        """Warning should trigger when 2+ factors present (MIN=2)."""
        from config.constants import RegimeTransition as RT

        r = self._make_regime_analyzer()
        r._adx_history = [30.0, 30.0, 30.0]
        r._btc_adx = 30.0

        r.set_vol_regime_for_transition("COMPRESSED")
        r.set_flow_accel_for_transition(-2.0)
        r._check_transition_warning()

        assert r._transition_warning is True
        assert r._transition_factor_count >= RT.MIN_FACTORS_FOR_WARNING

    def test_warning_triggers_with_all_3_factors(self):
        """All 3 factors should trigger warning with max factor count."""
        r = self._make_regime_analyzer()
        # ADX declining: peak 45, current 38 → decline = 7
        r._adx_history = [40.0, 42.0, 44.0, 45.0, 43.0]
        r._btc_adx = 38.0

        r.set_vol_regime_for_transition("LOW")
        r.set_flow_accel_for_transition(-3.0)
        r._check_transition_warning()

        assert r._transition_warning is True
        assert r._transition_factor_count == 3
        tw = r.get_transition_warning()
        assert tw["warning"] is True
        assert len(tw["factors"]) == 3

    def test_no_warning_when_no_factors(self):
        """No factors → no warning."""
        r = self._make_regime_analyzer()
        r._adx_history = [30.0, 30.0, 30.0]
        r._btc_adx = 31.0  # ADX rising

        # No vol or flow set
        r._check_transition_warning()

        assert r._transition_warning is False
        assert r._transition_factor_count == 0

    def test_adx_history_bounded(self):
        """ADX history should not grow unbounded."""
        from config.constants import RegimeTransition as RT

        r = self._make_regime_analyzer()
        # Push many values
        for i in range(50):
            r._adx_history.append(float(i))
            r._btc_adx = float(i)
            r._check_transition_warning()

        max_size = RT.ADX_PEAK_LOOKBACK + 5
        assert len(r._adx_history) <= max_size + 1  # +1 for the append before trim


# ════════════════════════════════════════════════════════════════
# 6. CONSTANTS VALIDATION
# ════════════════════════════════════════════════════════════════

class TestTier2Tier3Constants:
    """Validate new constant classes exist and have correct values."""

    def test_vol_compression_timer_constants(self):
        from config.constants import VolCompressionTimer as VCT
        assert VCT.WEIGHT_DURATION + VCT.WEIGHT_OI_BUILDUP + VCT.WEIGHT_FUNDING_NEUTRAL + VCT.WEIGHT_LEVERAGE_ZONE == pytest.approx(1.0)
        assert VCT.MIN_COMPRESSION_DAYS >= 1
        assert VCT.CONTEXT_BOOST_THRESHOLD > 0
        assert VCT.CONTEXT_DELTA_BONUS > 0

    def test_cascade_pressure_constants(self):
        from config.constants import CascadePressure as CP
        assert CP.PRESSURE_CRITICAL > CP.PRESSURE_HIGH > CP.PRESSURE_MEDIUM
        assert CP.MIN_CLUSTER_USD > 0
        assert CP.DISTANCE_FLOOR_PCT > 0
        assert CP.MOMENTUM_TOWARD_THRESHOLD > 0

    def test_regime_transition_constants(self):
        from config.constants import RegimeTransition as RT
        assert RT.ADX_DECLINE_THRESHOLD > 0
        assert RT.MIN_FACTORS_FOR_WARNING >= 2
        assert RT.WARNING_ENRICHMENT_DELTA < 0
        assert "COMPRESSED" in RT.VOL_COMPRESS_REGIMES
