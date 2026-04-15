"""
Tests for Power Alignment badge — cross-category strength detection.
Covers: constants, feature flag, compute logic, ScoredSignal fields,
and formatter display.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest


# ════════════════════════════════════════════════════════════════
# 1. Constants
# ════════════════════════════════════════════════════════════════

class TestPowerAlignmentConstants:
    """Verify threshold constants are defined and sensible."""

    def test_constants_exist(self):
        from config.constants import Grading
        assert hasattr(Grading, 'POWER_ALIGN_STRUCTURE_MIN')
        assert hasattr(Grading, 'POWER_ALIGN_VOLUME_MIN')
        assert hasattr(Grading, 'POWER_ALIGN_DERIVATIVES_MIN')
        assert hasattr(Grading, 'POWER_ALIGN_ORDERFLOW_MIN')
        assert hasattr(Grading, 'POWER_ALIGN_MIN_CATEGORIES')

    def test_thresholds_are_above_neutral(self):
        from config.constants import Grading
        # All category thresholds should be ≥ neutral (50) to be meaningful
        assert Grading.POWER_ALIGN_STRUCTURE_MIN >= 50
        assert Grading.POWER_ALIGN_VOLUME_MIN >= 50
        assert Grading.POWER_ALIGN_DERIVATIVES_MIN >= 50
        assert Grading.POWER_ALIGN_ORDERFLOW_MIN >= 50

    def test_min_categories_sensible(self):
        from config.constants import Grading
        # Must require at least 3 categories to avoid trivial badge
        assert 3 <= Grading.POWER_ALIGN_MIN_CATEGORIES <= 5


# ════════════════════════════════════════════════════════════════
# 2. Feature Flag
# ════════════════════════════════════════════════════════════════

class TestPowerAlignmentFlag:
    """Verify POWER_ALIGNMENT flag exists in the registry."""

    def test_flag_in_defaults(self):
        from config.feature_flags import FeatureFlags
        assert "POWER_ALIGNMENT" in FeatureFlags._DEFAULTS

    def test_flag_defaults_off(self):
        from config.feature_flags import FeatureFlags
        assert FeatureFlags._DEFAULTS["POWER_ALIGNMENT"] == "off"

    def test_env_override(self):
        os.environ["TITANBOT_FF_POWER_ALIGNMENT"] = "live"
        try:
            from config.feature_flags import FeatureFlags
            ff_test = FeatureFlags()
            assert ff_test.is_enabled("POWER_ALIGNMENT")
        finally:
            del os.environ["TITANBOT_FF_POWER_ALIGNMENT"]

    def test_shadow_mode(self):
        os.environ["TITANBOT_FF_POWER_ALIGNMENT"] = "shadow"
        try:
            from config.feature_flags import FeatureFlags
            ff_test = FeatureFlags()
            assert ff_test.is_shadow("POWER_ALIGNMENT")
            assert ff_test.is_active("POWER_ALIGNMENT")
            assert not ff_test.is_enabled("POWER_ALIGNMENT")
        finally:
            del os.environ["TITANBOT_FF_POWER_ALIGNMENT"]


# ════════════════════════════════════════════════════════════════
# 3. ScoredSignal fields
# ════════════════════════════════════════════════════════════════

class TestScoredSignalFields:
    """Verify the new fields exist with correct defaults."""

    def test_power_aligned_default(self):
        from signals.aggregator import ScoredSignal
        sig = ScoredSignal(base_signal=MagicMock())
        assert sig.power_aligned is False

    def test_power_alignment_reason_default(self):
        from signals.aggregator import ScoredSignal
        sig = ScoredSignal(base_signal=MagicMock())
        assert sig.power_alignment_reason == ""


# ════════════════════════════════════════════════════════════════
# 4. Compute logic
# ════════════════════════════════════════════════════════════════

class TestComputePowerAlignment:
    """Test the _compute_power_alignment static method."""

    def _make_scored(self, tech=50, vol=50, of=50, deriv=50, sent=50, corr=50):
        from signals.aggregator import ScoredSignal
        s = ScoredSignal(base_signal=MagicMock())
        s.technical_score = tech
        s.volume_score = vol
        s.orderflow_score = of
        s.derivatives_score = deriv
        s.sentiment_score = sent
        s.correlation_score = corr
        return s

    def _compute(self, scored):
        from signals.aggregator import SignalAggregator
        return SignalAggregator._compute_power_alignment(scored)

    def test_all_strong_returns_aligned(self):
        """All categories above thresholds → power aligned."""
        scored = self._make_scored(tech=70, vol=65, of=60, deriv=68, sent=60)
        aligned, reason = self._compute(scored)
        assert aligned is True
        assert "5/5" in reason

    def test_all_neutral_returns_not_aligned(self):
        """All categories at neutral 50 → not aligned (most below threshold)."""
        scored = self._make_scored()
        aligned, reason = self._compute(scored)
        assert aligned is False
        assert reason == ""

    def test_four_categories_sufficient(self):
        """4 out of 5 categories above threshold → aligned (default min=4)."""
        # Strong structure, volume, derivatives, orderflow; weak context
        scored = self._make_scored(tech=60, vol=60, of=55, deriv=60, sent=40, corr=40)
        aligned, reason = self._compute(scored)
        assert aligned is True
        assert "4/5" in reason

    def test_three_categories_insufficient(self):
        """Only 3 categories strong → not aligned."""
        scored = self._make_scored(tech=60, vol=60, of=40, deriv=60, sent=40, corr=40)
        aligned, reason = self._compute(scored)
        assert aligned is False

    def test_context_via_sentiment(self):
        """Context counted when sentiment ≥ 55."""
        scored = self._make_scored(tech=60, vol=60, of=55, deriv=30, sent=55, corr=40)
        aligned, reason = self._compute(scored)
        # structure + volume + orderflow + context = 4
        assert aligned is True
        assert "context" in reason

    def test_context_via_correlation(self):
        """Context counted when correlation ≥ 60."""
        scored = self._make_scored(tech=60, vol=60, of=55, deriv=30, sent=40, corr=65)
        aligned, reason = self._compute(scored)
        assert aligned is True
        assert "context" in reason

    def test_high_grade_low_power(self):
        """High technical + single strong category → not power aligned.
        This is the ENJ scenario: high confidence driven by one dimension."""
        scored = self._make_scored(tech=92, vol=45, of=40, deriv=48, sent=40)
        aligned, reason = self._compute(scored)
        assert aligned is False

    def test_low_grade_high_power(self):
        """Moderate confidence but strong across all dimensions → power aligned.
        This is the 'hidden gem' the badge is designed to catch."""
        scored = self._make_scored(tech=58, vol=60, of=55, deriv=60, sent=58)
        aligned, reason = self._compute(scored)
        assert aligned is True

    def test_reason_lists_categories(self):
        """Reason string names the confirmed categories."""
        scored = self._make_scored(tech=70, vol=70, of=60, deriv=70, sent=60)
        aligned, reason = self._compute(scored)
        assert aligned is True
        assert "structure" in reason
        assert "volume" in reason
        assert "orderflow" in reason
        assert "derivatives" in reason
        assert "context" in reason

    def test_exact_threshold_counts(self):
        """Score exactly at threshold should count."""
        from config.constants import Grading
        scored = self._make_scored(
            tech=Grading.POWER_ALIGN_STRUCTURE_MIN,
            vol=Grading.POWER_ALIGN_VOLUME_MIN,
            of=Grading.POWER_ALIGN_ORDERFLOW_MIN,
            deriv=Grading.POWER_ALIGN_DERIVATIVES_MIN,
            sent=55,  # context threshold
        )
        aligned, _ = self._compute(scored)
        assert aligned is True

    def test_just_below_threshold_fails(self):
        """Score 1 below threshold should not count."""
        from config.constants import Grading
        scored = self._make_scored(
            tech=Grading.POWER_ALIGN_STRUCTURE_MIN - 1,
            vol=Grading.POWER_ALIGN_VOLUME_MIN - 1,
            of=Grading.POWER_ALIGN_ORDERFLOW_MIN - 1,
            deriv=Grading.POWER_ALIGN_DERIVATIVES_MIN - 1,
            sent=40, corr=40,
        )
        aligned, _ = self._compute(scored)
        assert aligned is False


# ════════════════════════════════════════════════════════════════
# 5. Formatter display
# ════════════════════════════════════════════════════════════════

class TestFormatterPowerBadge:
    """Verify badge appears in formatted output when power_aligned=True."""

    def _make_scored_for_format(self, power_aligned=False, grade="A"):
        from signals.aggregator import ScoredSignal
        sig_mock = MagicMock()
        sig_mock.direction = MagicMock(value="LONG")
        sig_mock.symbol = "BTCUSDT"
        sig_mock.confidence = 82
        sig_mock.entry_low = 100
        sig_mock.entry_high = 102
        sig_mock.stop_loss = 98
        sig_mock.tp1 = 106
        sig_mock.tp2 = 110
        sig_mock.tp3 = None
        sig_mock.rr_ratio = 3.0
        sig_mock.strategy = "SmartMoneyConcepts"
        sig_mock.raw_data = {}
        sig_mock.timeframe = "1h"
        sig_mock.setup_class = "intraday"
        sig_mock.confluence = []

        scored = ScoredSignal(base_signal=sig_mock)
        scored.technical_score = 70
        scored.volume_score = 65
        scored.orderflow_score = 60
        scored.derivatives_score = 68
        scored.sentiment_score = 55
        scored.final_confidence = 82
        scored.grade = grade
        scored.is_killzone = False
        scored.killzone_bonus = 0
        scored.power_aligned = power_aligned
        scored.power_alignment_reason = "5/5 categories confirmed (structure, volume, derivatives, orderflow, context)" if power_aligned else ""
        scored.derivatives_data = None
        scored.volume_data = None
        scored.all_confluence = []
        return scored

    def test_badge_in_signal_card(self):
        """⚡ badge appears in signal card when power_aligned=True."""
        from tg.formatter import TelegramFormatter
        fmt = TelegramFormatter()
        scored = self._make_scored_for_format(power_aligned=True, grade="A")
        text = fmt.format_signal(scored)
        assert "⚡" in text
        assert "Power Aligned" in text

    def test_no_badge_when_not_aligned(self):
        """No badge when power_aligned=False."""
        from tg.formatter import TelegramFormatter
        fmt = TelegramFormatter()
        scored = self._make_scored_for_format(power_aligned=False, grade="A")
        text = fmt.format_signal(scored)
        assert "Power Aligned" not in text

    def test_badge_in_metrics_panel(self):
        """Metrics panel shows power alignment reason."""
        from tg.formatter import TelegramFormatter
        fmt = TelegramFormatter()
        scored = self._make_scored_for_format(power_aligned=True)
        text = fmt.format_metrics_panel(scored)
        assert "⚡" in text
        assert "Power Aligned" in text
        assert "5/5" in text

    def test_no_badge_in_metrics_when_not_aligned(self):
        """Metrics panel omits badge when not aligned."""
        from tg.formatter import TelegramFormatter
        fmt = TelegramFormatter()
        scored = self._make_scored_for_format(power_aligned=False)
        text = fmt.format_metrics_panel(scored)
        assert "Power Aligned" not in text
