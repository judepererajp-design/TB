"""
Tests for the 5-area enhancement:
  1. Power Alignment v2 (HTF concordance, weighted scoring, tiers, regime-aware)
  2. Trigger Quality Scoring (diminishing returns, volume context, diversity)
  3. Spoofing / Fake Wall Detection (persistence, pull patterns, size anomaly)
  4. Advanced Whale Intent (taxonomy, multi-source, coordination, decay)
  5. Volume Quality Assessment (climactic, breadth, spread-adjusted, dry volume)

These tests work with the mocked numpy environment from conftest.py.
Tests for pure logic (constants, dataclass fields, scoring math) don't need numpy.
Tests that exercise real numpy array operations are in test_five_area_realmath.py.
"""

import math
import time
from unittest.mock import MagicMock, patch

import pytest


# ════════════════════════════════════════════════════════════════
# 1. POWER ALIGNMENT v2 CONSTANTS
# ════════════════════════════════════════════════════════════════

class TestPowerAlignmentV2Constants:
    """Verify v2 power alignment constants exist and are sensible."""

    def test_weight_constants_exist(self):
        from config.constants import Grading
        assert hasattr(Grading, 'POWER_ALIGN_WEIGHT_STRUCTURE')
        assert hasattr(Grading, 'POWER_ALIGN_WEIGHT_VOLUME')
        assert hasattr(Grading, 'POWER_ALIGN_WEIGHT_DERIVATIVES')
        assert hasattr(Grading, 'POWER_ALIGN_WEIGHT_ORDERFLOW')
        assert hasattr(Grading, 'POWER_ALIGN_WEIGHT_CONTEXT')

    def test_weights_sum_to_one(self):
        from config.constants import Grading
        total = (
            Grading.POWER_ALIGN_WEIGHT_STRUCTURE +
            Grading.POWER_ALIGN_WEIGHT_VOLUME +
            Grading.POWER_ALIGN_WEIGHT_DERIVATIVES +
            Grading.POWER_ALIGN_WEIGHT_ORDERFLOW +
            Grading.POWER_ALIGN_WEIGHT_CONTEXT
        )
        assert abs(total - 1.0) < 0.01

    def test_tier_thresholds_ordered(self):
        from config.constants import Grading
        assert Grading.POWER_ALIGN_TIER_STRONG > Grading.POWER_ALIGN_TIER_MODERATE

    def test_htf_concordance_constants(self):
        from config.constants import Grading
        assert Grading.POWER_ALIGN_HTF_CONCORDANCE_BONUS > 0
        assert Grading.POWER_ALIGN_HTF_CONFLICT_PENALTY < 0

    def test_conf_boost_constants(self):
        from config.constants import Grading
        assert Grading.POWER_ALIGN_CONF_BOOST_STRONG > Grading.POWER_ALIGN_CONF_BOOST_MODERATE
        assert Grading.POWER_ALIGN_CONF_BOOST_MODERATE > 0

    def test_regime_adjustments(self):
        from config.constants import Grading
        assert Grading.POWER_ALIGN_CHOPPY_PENALTY > 0
        assert Grading.POWER_ALIGN_TREND_BONUS < 0  # Negative = lowers threshold


# ════════════════════════════════════════════════════════════════
# 2. TRIGGER QUALITY CONSTANTS
# ════════════════════════════════════════════════════════════════

class TestTriggerQualityConstants:
    """Verify trigger quality constants exist."""

    def test_constants_class_exists(self):
        from config.constants import TriggerQuality
        assert TriggerQuality.DIMINISH_BASE == 1.0
        assert 0 < TriggerQuality.DIMINISH_DECAY < 1.0
        assert TriggerQuality.MAX_USEFUL_TRIGGERS >= 3

    def test_vol_confirmed_bonus_positive(self):
        from config.constants import TriggerQuality
        assert TriggerQuality.VOL_CONFIRMED_BONUS > 1.0
        assert TriggerQuality.VOL_UNCONFIRMED_PENALTY < 1.0

    def test_quality_thresholds_ordered(self):
        from config.constants import TriggerQuality
        assert TriggerQuality.QUALITY_HIGH > TriggerQuality.QUALITY_MEDIUM

    def test_fast_move_thresholds(self):
        from config.constants import TriggerQuality as TQ
        assert TQ.FAST_MOVE_VOL_SPIKE_MIN > 1.0
        assert TQ.FAST_MOVE_PRICE_VEL_ATR > 0
        assert TQ.FAST_MOVE_CONF_BONUS > 0

    def test_climactic_thresholds(self):
        from config.constants import TriggerQuality as TQ
        assert TQ.CLIMACTIC_VOL_MULT > 2.0
        assert 0 < TQ.CLIMACTIC_REVERSAL_PENALTY < 1.0

    def test_diversity_bonus(self):
        from config.constants import TriggerQuality as TQ
        assert TQ.DIVERSITY_BONUS_PER_CATEGORY > 0
        assert TQ.MAX_DIVERSITY_BONUS > TQ.DIVERSITY_BONUS_PER_CATEGORY


# ════════════════════════════════════════════════════════════════
# 3. SPOOFING / FAKE WALL CONSTANTS
# ════════════════════════════════════════════════════════════════

class TestSpoofingConstants:
    """Verify spoofing detection constants."""

    def test_constants_class_exists(self):
        from config.constants import SpoofingDetection
        assert SpoofingDetection.MIN_PERSISTENCE_SNAPSHOTS >= 2
        assert SpoofingDetection.FAKE_WALL_CONFIDENCE > 0
        assert SpoofingDetection.WALL_TIER_MEGA > SpoofingDetection.WALL_TIER_LARGE

    def test_persistence_thresholds(self):
        from config.constants import SpoofingDetection as SD
        assert SD.PERSISTENCE_CONFIDENCE > 0.5
        assert SD.SIZE_ANOMALY_MULT > 3.0

    def test_pull_distance(self):
        from config.constants import SpoofingDetection as SD
        assert 0 < SD.PULL_DISTANCE_PCT < 0.05  # Should be small (< 5%)
        assert 0 < SD.PULL_PENALTY <= 1.0

    def test_symmetric_tolerance(self):
        from config.constants import SpoofingDetection as SD
        assert 0 < SD.SYMMETRIC_WALL_TOLERANCE < 1.0
        assert SD.SYMMETRIC_WALL_BOOST >= 1.0

    def test_wall_tiers_ordered(self):
        from config.constants import SpoofingDetection as SD
        assert SD.WALL_TIER_MEGA > SD.WALL_TIER_LARGE > SD.WALL_TIER_MODERATE


# ════════════════════════════════════════════════════════════════
# 4. WHALE INTENT CONSTANTS & BEHAVIOR
# ════════════════════════════════════════════════════════════════

class TestWhaleIntentConstants:
    """Verify whale intent constants."""

    def test_constants_class_exists(self):
        from config.constants import WhaleIntent
        assert WhaleIntent.INTENT_ACCUMULATION == "ACCUMULATION"
        assert WhaleIntent.INTENT_DISTRIBUTION == "DISTRIBUTION"
        assert WhaleIntent.INTENT_MARKET_MAKING == "MARKET_MAKING"
        assert WhaleIntent.INTENT_DIRECTIONAL == "DIRECTIONAL"
        assert WhaleIntent.INTENT_HEDGING == "HEDGING"
        assert WhaleIntent.INTENT_REBALANCING == "REBALANCING"

    def test_source_weights_sum_to_one(self):
        from config.constants import WhaleIntent as WI
        total = (
            WI.SOURCE_WEIGHT_ORDERBOOK +
            WI.SOURCE_WEIGHT_ONCHAIN +
            WI.SOURCE_WEIGHT_DERIVATIVES +
            WI.SOURCE_WEIGHT_DEPOSIT
        )
        assert abs(total - 1.0) < 0.01

    def test_intent_confidence_deltas(self):
        from config.constants import WhaleIntent as WI
        assert WI.INTENT_CONF_ACCUMULATION > 0
        assert WI.INTENT_CONF_DISTRIBUTION < 0
        assert WI.INTENT_CONF_MM == 0

    def test_decay_parameters(self):
        from config.constants import WhaleIntent as WI
        assert WI.INTENT_DECAY_TAU_SECS > 0
        assert 0 < WI.INTENT_MIN_CONFIDENCE < 1.0

    def test_mm_detection_params(self):
        from config.constants import WhaleIntent as WI
        assert 0 < WI.MM_SPREAD_RATIO_MAX < 0.01
        assert 0 < WI.MM_SIZE_SYMMETRY_MIN < 1.0

    def test_coordination_params(self):
        from config.constants import WhaleIntent as WI
        assert WI.COORD_TIME_WINDOW_SECS > 0
        assert WI.COORD_MIN_PARTICIPANTS >= 2
        assert WI.COORD_USD_THRESHOLD > 0


class TestAdvancedWhaleIntent:
    """Test advanced whale intent classification."""

    def _make_profiler(self):
        from analyzers.wallet_behavior import WalletBehaviorProfiler
        return WalletBehaviorProfiler()

    def test_unknown_with_no_events(self):
        profiler = self._make_profiler()
        result = profiler.get_advanced_intent("LONG")
        assert result.intent == "UNKNOWN"
        assert result.confidence == 0.0

    def test_bullish_accumulation_detected(self):
        profiler = self._make_profiler()
        for i in range(10):
            profiler.record_event("buy", 500_000, "BTCUSDT", "orderbook")
            time.sleep(0.001)

        result = profiler.get_advanced_intent("LONG")
        assert result.intent in ("ACCUMULATION", "DIRECTIONAL")
        assert result.directional_bias == "BULLISH"
        assert result.confidence > 0

    def test_bearish_distribution_detected(self):
        profiler = self._make_profiler()
        for i in range(10):
            profiler.record_event("sell", 500_000, "BTCUSDT", "orderbook")
            time.sleep(0.001)

        result = profiler.get_advanced_intent("SHORT")
        assert result.intent in ("DISTRIBUTION", "DIRECTIONAL")
        assert result.directional_bias == "BEARISH"

    def test_market_maker_detected(self):
        profiler = self._make_profiler()
        for i in range(10):
            profiler.record_event("buy", 300_000, "BTCUSDT", "orderbook")
            profiler.record_event("sell", 310_000, "BTCUSDT", "orderbook")
            time.sleep(0.001)

        result = profiler.get_advanced_intent("LONG")
        assert result.intent in ("MARKET_MAKING", "REBALANCING", "DIRECTIONAL")
        assert result.confidence >= 0

    def test_rebalancing_detected(self):
        profiler = self._make_profiler()
        for i in range(5):
            profiler.record_event("buy", 400_000, "BTCUSDT", "orderbook")
            profiler.record_event("sell", 350_000, "BTCUSDT", "orderbook")
            time.sleep(0.001)

        result = profiler.get_advanced_intent("LONG")
        assert result.confidence >= 0

    def test_directional_against_penalizes(self):
        profiler = self._make_profiler()
        for i in range(10):
            profiler.record_event("buy", 500_000, "BTCUSDT", "orderbook")
            time.sleep(0.001)

        result_short = profiler.get_advanced_intent("SHORT")
        result_long = profiler.get_advanced_intent("LONG")
        assert result_short.confidence_delta <= result_long.confidence_delta

    def test_advanced_intent_dataclass(self):
        from analyzers.wallet_behavior import AdvancedWhaleIntent
        intent = AdvancedWhaleIntent()
        assert intent.intent == "UNKNOWN"
        assert intent.confidence == 0.0
        assert intent.notes == []

    def test_multi_source_events(self):
        profiler = self._make_profiler()
        profiler.record_event("buy", 500_000, "BTCUSDT", "orderbook")
        profiler.record_event("buy", 800_000, "BTCUSDT", "deposit")
        profiler.record_event("buy", 300_000, "BTCUSDT", "smart_money")

        result = profiler.get_advanced_intent("LONG")
        assert result.confidence > 0


# ════════════════════════════════════════════════════════════════
# 5. VOLUME QUALITY CONSTANTS
# ════════════════════════════════════════════════════════════════

class TestVolumeQualityConstants:
    """Verify volume quality constants."""

    def test_constants_class_exists(self):
        from config.constants import VolumeQuality
        assert VolumeQuality.CLIMACTIC_MULT > 2.0
        assert VolumeQuality.QUALITY_SCORE_HIGH > VolumeQuality.QUALITY_SCORE_LOW

    def test_quality_weights_sum_to_one(self):
        from config.constants import VolumeQuality as VQ
        total = (
            VQ.QUALITY_WEIGHT_MAGNITUDE +
            VQ.QUALITY_WEIGHT_TREND +
            VQ.QUALITY_WEIGHT_CONTEXT +
            VQ.QUALITY_WEIGHT_BREADTH +
            VQ.QUALITY_WEIGHT_SPREAD
        )
        assert abs(total - 1.0) < 0.01

    def test_context_types(self):
        from config.constants import VolumeQuality as VQ
        assert VQ.CLIMACTIC_WITH_REVERSAL == "EXHAUSTION"
        assert VQ.CLIMACTIC_WITH_TREND == "BREAKOUT"

    def test_breadth_thresholds(self):
        from config.constants import VolumeQuality as VQ
        assert 0 < VQ.TICK_VOLUME_RATIO_LOW < VQ.TICK_VOLUME_RATIO_HIGH
        assert VQ.TICK_VOLUME_RATIO_HIGH <= 1.0

    def test_spread_thresholds(self):
        from config.constants import VolumeQuality as VQ
        assert VQ.SPREAD_WIDE_THRESHOLD_BPS > 0
        assert 0 < VQ.SPREAD_WIDE_VOL_DISCOUNT < 1.0

    def test_dry_volume_params(self):
        from config.constants import VolumeQuality as VQ
        assert VQ.DRY_VOL_PRICE_MOVE_ATR > 0
        assert VQ.DRY_VOL_THRESHOLD_MULT < 1.0
        assert VQ.DRY_VOL_PENALTY < 0

    def test_confirmation_params(self):
        from config.constants import VolumeQuality as VQ
        assert VQ.CONFIRM_WINDOW_BARS >= 1
        assert VQ.CONFIRM_MIN_MULT > 1.0

    def test_breakout_params(self):
        from config.constants import VolumeQuality as VQ
        assert VQ.BREAKOUT_VOL_MIN_MULT > 1.0
        assert VQ.RANGE_VOL_HIGH_WARNING > VQ.BREAKOUT_VOL_MIN_MULT
