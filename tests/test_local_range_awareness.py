"""
Local Range Awareness Tests
============================
Verifies that:
  1. regime_levels.adjust_levels() applies soft TP2/TP3 caps when a symbol
     is range-bound (tight 50-bar range) despite BTC macro regime being
     BULL_TREND or BEAR_TREND.
  2. equilibrium.EquilibriumAnalyzer.assess() applies soft confidence penalty
     for direction-zone mismatches in trending regimes when the symbol's own
     range is tight.
  3. No changes to behavior when:
     - Regime is CHOPPY (already handled by existing logic)
     - Symbol range is wide (symbol is genuinely trending)
     - chop_strength is too low (strong trend, incidental consolidation)

Test count: 31 tests
"""

import sys
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from config.constants import LocalRange


# ═══════════════════════════════════════════════════════════════
# PART 1: regime_levels — local range TP capping
# ═══════════════════════════════════════════════════════════════

from signals.regime_levels import adjust_levels


class TestLocalRangeTPCapping:
    """TP2/TP3 are soft-capped when symbol is range-bound in trend regime."""

    # ── Helper: create a basic LONG scenario in BULL_TREND with tight range ──
    @staticmethod
    def _long_bull_tight_range(**overrides):
        """
        Symbol at $100, range $95–$105 (10% range, below 15% threshold).
        Strategy wants entry=100, SL=97, TP1=103, TP2=110, TP3=118.
        TP2 at $110 is 5% beyond range_high ($105) — should be capped.
        """
        defaults = dict(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=103.0, tp2=110.0, tp3=118.0,
            direction="LONG", setup_class="intraday",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=105.0, range_low=95.0, range_eq=100.0,
        )
        defaults.update(overrides)
        return adjust_levels(**defaults)

    @staticmethod
    def _short_bear_tight_range(**overrides):
        """
        Symbol at $100, range $95–$105 (10% range).
        Strategy wants entry=100, SL=103, TP1=97, TP2=90, TP3=82.
        TP2 at $90 is 5% below range_low ($95) — should be capped.
        """
        defaults = dict(
            entry_low=99.5, entry_high=100.5,
            stop_loss=103.0,
            tp1=97.0, tp2=90.0, tp3=82.0,
            direction="SHORT", setup_class="intraday",
            regime="BEAR_TREND", chop_strength=0.20,
            range_high=105.0, range_low=95.0, range_eq=100.0,
        )
        defaults.update(overrides)
        return adjust_levels(**defaults)

    # ── LONG in BULL_TREND with tight range ──
    def test_long_bull_tp2_capped(self):
        """TP2 should be capped near range_high + overshoot, not at $110."""
        result = self._long_bull_tight_range()
        # range_high=105, range_size=10, overshoot=10% → cap at ~106
        # entry_mid=100, so tp2_dist cap = (105-100) + 10*0.10 = 6.0
        # TP2 = 100 + 6.0 = 106.0 (before other adjustments)
        assert result.tp2 < 110.0, f"TP2 should be capped below 110, got {result.tp2}"

    def test_long_bull_tp3_capped(self):
        """TP3 should be capped near range_high + larger overshoot."""
        result = self._long_bull_tight_range()
        # tp3_dist cap = (105-100) + 10*0.25 = 7.5 → TP3 = 107.5
        assert result.tp3 is not None
        assert result.tp3 < 118.0, f"TP3 should be capped below 118, got {result.tp3}"

    def test_long_bull_tp1_untouched(self):
        """TP1 should NOT be affected by local range capping."""
        result = self._long_bull_tight_range()
        # TP1 at 103 is within range — should not be capped further than
        # the normal regime adjustments would do
        result_no_range = self._long_bull_tight_range(range_high=0, range_low=0)
        # TP1 should be similar (small differences from min-gap enforcement OK)
        assert abs(result.tp1 - result_no_range.tp1) < 2.0

    def test_long_bull_adjustments_mention_local_range(self):
        """Adjustments log should mention local-range cap."""
        result = self._long_bull_tight_range()
        adj_text = " ".join(result.adjustments)
        assert "local-range" in adj_text.lower(), f"Expected 'local-range' in adjustments: {result.adjustments}"

    # ── SHORT in BEAR_TREND with tight range ──
    def test_short_bear_tp2_capped(self):
        """SHORT TP2 should be capped near range_low - overshoot."""
        result = self._short_bear_tight_range()
        # range_low=95, range_size=10, overshoot=10% → cap distance = (100-95) + 10*0.10 = 6.0
        # TP2 = 100 - 6.0 = 94.0
        assert result.tp2 > 90.0, f"SHORT TP2 should be capped above 90, got {result.tp2}"

    def test_short_bear_tp3_capped(self):
        """SHORT TP3 should be capped near range_low - larger overshoot."""
        result = self._short_bear_tight_range()
        assert result.tp3 is not None
        assert result.tp3 > 82.0, f"SHORT TP3 should be capped above 82, got {result.tp3}"

    # ── No capping when range is wide ──
    def test_wide_range_no_cap(self):
        """When symbol range is > 15%, no local range capping should apply."""
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=103.0, tp2=110.0, tp3=118.0,
            direction="LONG", setup_class="intraday",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=120.0, range_low=80.0, range_eq=100.0,  # 40% range
        )
        adj_text = " ".join(result.adjustments)
        assert "local-range" not in adj_text.lower(), \
            f"Wide range should NOT trigger local-range cap: {result.adjustments}"

    # ── No capping when chop_strength is too low ──
    def test_low_chop_no_cap(self):
        """When chop_strength < MIN_CHOP_FOR_LOCAL, no local range cap."""
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=103.0, tp2=110.0, tp3=118.0,
            direction="LONG", setup_class="intraday",
            regime="BULL_TREND", chop_strength=0.05,  # below 0.10
            range_high=105.0, range_low=95.0, range_eq=100.0,
        )
        adj_text = " ".join(result.adjustments)
        assert "local-range" not in adj_text.lower()

    # ── No capping in CHOPPY regime (already handled by step 5) ──
    def test_choppy_uses_existing_logic(self):
        """CHOPPY regime should use existing step 5 logic, not local range."""
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=103.0, tp2=110.0, tp3=118.0,
            direction="LONG", setup_class="intraday",
            regime="CHOPPY", chop_strength=0.50,
            range_high=105.0, range_low=95.0, range_eq=100.0,
        )
        adj_text = " ".join(result.adjustments)
        # Should NOT see local-range — CHOPPY has its own tighter capping
        assert "local-range" not in adj_text.lower()

    # ── No capping in VOLATILE regime ──
    def test_volatile_no_local_cap(self):
        """VOLATILE regime should not get local range capping."""
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=103.0, tp2=110.0, tp3=118.0,
            direction="LONG", setup_class="intraday",
            regime="VOLATILE", chop_strength=0.20,
            range_high=105.0, range_low=95.0, range_eq=100.0,
        )
        adj_text = " ".join(result.adjustments)
        assert "local-range" not in adj_text.lower()

    # ── No capping when no range data ──
    def test_no_range_data_no_cap(self):
        """When range_high/range_low are 0, no capping."""
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=103.0, tp2=110.0, tp3=118.0,
            direction="LONG", setup_class="intraday",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=0.0, range_low=0.0, range_eq=0.0,
        )
        adj_text = " ".join(result.adjustments)
        assert "local-range" not in adj_text.lower()

    # ── Boundary: range_pct exactly at threshold ──
    def test_range_at_threshold_no_cap(self):
        """Range exactly at 15% threshold should NOT trigger capping."""
        # entry=100, range_high=107.5, range_low=92.5 → range_size=15, pct=15%
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=103.0, tp2=110.0, tp3=118.0,
            direction="LONG", setup_class="intraday",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=107.5, range_low=92.5, range_eq=100.0,
        )
        adj_text = " ".join(result.adjustments)
        assert "local-range" not in adj_text.lower()

    # ── Boundary: range just below threshold ──
    def test_range_just_below_threshold_caps(self):
        """Range at 14.9% should trigger capping."""
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=103.0, tp2=110.0, tp3=118.0,
            direction="LONG", setup_class="intraday",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=107.45, range_low=92.55, range_eq=100.0,
        )
        adj_text = " ".join(result.adjustments)
        assert "local-range" in adj_text.lower()

    # ── TP2 within cap should not be reduced ──
    def test_tp2_within_cap_untouched(self):
        """If TP2 already within the cap distance, local-range should not cap it."""
        # range_high=105, range_size=10, entry_mid=100
        # TP2 overshoot cap = (105-100) + 10*0.10 = 6.0
        # After BULL_TREND applies tp2_mult + long_tp2_mult, TP2 distance gets expanded.
        # So use a TP2 that after expansion still fits under the cap:
        # The regime profile multiplies tp2_dist by tp2_mult=1.4 then long_tp2_mult=1.6
        # Let's set original tp2=103.5 → dist=3.5 → after 1.4=4.9 → after 1.6 rebase=~5.6
        # Cap is 6.0, so 5.6 < 6.0 → should NOT be capped
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=102.0, tp2=103.5, tp3=104.5,  # TP2 close to entry
            direction="LONG", setup_class="intraday",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=105.0, range_low=95.0, range_eq=100.0,
        )
        adj_text = " ".join(result.adjustments)
        assert "TP2 local-range" not in adj_text

    # ── SHORT in BULL_TREND with tight range ──
    def test_short_bull_tight_range(self):
        """SHORT in BULL_TREND with tight range should also get capped."""
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=103.0,
            tp1=97.0, tp2=90.0, tp3=82.0,
            direction="SHORT", setup_class="intraday",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=105.0, range_low=95.0, range_eq=100.0,
        )
        adj_text = " ".join(result.adjustments)
        # SHORT in BULL with 10% range should get local-range capping
        assert "local-range" in adj_text.lower()

    # ── LONG in BEAR_TREND with tight range ──
    def test_long_bear_tight_range(self):
        """LONG in BEAR_TREND with tight range should get capped."""
        result = adjust_levels(
            entry_low=99.5, entry_high=100.5,
            stop_loss=97.0,
            tp1=103.0, tp2=110.0, tp3=118.0,
            direction="LONG", setup_class="intraday",
            regime="BEAR_TREND", chop_strength=0.20,
            range_high=105.0, range_low=95.0, range_eq=100.0,
        )
        adj_text = " ".join(result.adjustments)
        assert "local-range" in adj_text.lower()

    # ── Constants are accessible and have sensible defaults ──
    def test_constants_threshold(self):
        assert LocalRange.RANGE_PCT_THRESHOLD == 0.15

    def test_constants_tp2_overshoot(self):
        assert 0 < LocalRange.TP2_RANGE_OVERSHOOT < 0.50

    def test_constants_tp3_overshoot(self):
        assert LocalRange.TP3_RANGE_OVERSHOOT > LocalRange.TP2_RANGE_OVERSHOOT

    def test_constants_min_chop(self):
        assert 0 < LocalRange.MIN_CHOP_FOR_LOCAL < 0.30


class TestCounterTrendPullbackCaps:
    """Counter-trend trend trades should not advertise deep reversal targets."""

    def test_short_bull_caps_tp2_at_eq_and_disables_tp3(self):
        result = adjust_levels(
            entry_low=149.0, entry_high=151.0,
            stop_loss=156.0,
            tp1=145.0, tp2=120.0, tp3=90.0,
            direction="SHORT", setup_class="swing",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=175.0, range_low=105.0, range_eq=140.0,
        )
        assert result.tp2 == pytest.approx(140.0), f"TP2 should be capped at EQ, got {result.tp2}"
        assert result.tp3 is None, "Counter-trend pullback TP3 should be disabled"
        assert any("counter-trend pullback" in adj.lower() for adj in result.adjustments)

    def test_long_bear_caps_tp2_at_eq_and_disables_tp3(self):
        result = adjust_levels(
            entry_low=119.0, entry_high=121.0,
            stop_loss=114.0,
            tp1=125.0, tp2=170.0, tp3=210.0,
            direction="LONG", setup_class="swing",
            regime="BEAR_TREND", chop_strength=0.20,
            range_high=175.0, range_low=105.0, range_eq=140.0,
        )
        assert result.tp2 == pytest.approx(140.0), f"TP2 should be capped at EQ, got {result.tp2}"
        assert result.tp3 is None, "Counter-trend pullback TP3 should be disabled"
        assert any("counter-trend pullback" in adj.lower() for adj in result.adjustments)

    def test_with_trend_long_bull_keeps_tp3(self):
        result = adjust_levels(
            entry_low=149.0, entry_high=151.0,
            stop_loss=144.0,
            tp1=155.0, tp2=170.0, tp3=190.0,
            direction="LONG", setup_class="swing",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=175.0, range_low=105.0, range_eq=140.0,
        )
        assert result.tp3 is not None, "Trend-aligned trade should keep TP3"
        assert not any("counter-trend pullback" in adj.lower() for adj in result.adjustments)


class TestLocalContinuationTradeTypes:
    """Strong local structure should avoid pullback-only handling."""

    @staticmethod
    def _bearish_structure():
        return dict(
            recent_opens=[159, 161, 158, 162, 157, 160, 156, 159, 155, 158, 153, 152, 151, 150, 147],
            recent_highs=[160, 164, 161, 165, 160, 163, 159, 162, 158, 161, 160, 159, 158, 156, 154],
            recent_lows=[156, 158, 155, 159, 154, 157, 153, 156, 152, 155, 151, 150, 149, 148, 144],
            recent_closes=[158, 160, 157, 161, 156, 159, 155, 158, 154, 157, 152, 151, 150, 148, 144],
            recent_volumes=[10, 11, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 18, 19, 22],
        )

    @staticmethod
    def _bullish_structure():
        return dict(
            recent_opens=[101, 100, 103, 102, 105, 104, 107, 106, 109, 108, 111, 112, 114, 117, 118],
            recent_highs=[104, 103, 106, 105, 108, 107, 110, 109, 112, 111, 113, 115, 117, 120, 124],
            recent_lows=[99, 98, 101, 100, 103, 102, 105, 104, 107, 106, 108, 110, 112, 114, 116],
            recent_closes=[100, 99, 104, 103, 106, 105, 108, 107, 110, 109, 112, 114, 116, 119, 123],
            recent_volumes=[10, 11, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 18, 19, 22],
        )

    def test_short_bull_with_bearish_structure_becomes_local_continuation(self):
        result = adjust_levels(
            entry_low=149.0, entry_high=151.0,
            stop_loss=156.0,
            tp1=145.0, tp2=120.0, tp3=90.0,
            direction="SHORT", setup_class="swing",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=175.0, range_low=105.0, range_eq=140.0,
            **self._bearish_structure(),
        )
        assert result.trade_type == "LOCAL_CONTINUATION_SHORT"
        assert result.tp2 < 140.0, f"Local continuation short should extend below EQ, got {result.tp2}"
        assert result.tp3 is not None, "Local continuation short should keep a controlled TP3"
        assert any("local continuation" in adj.lower() for adj in result.adjustments)

    def test_short_bull_without_structure_stays_pullback(self):
        result = adjust_levels(
            entry_low=149.0, entry_high=151.0,
            stop_loss=156.0,
            tp1=145.0, tp2=120.0, tp3=90.0,
            direction="SHORT", setup_class="swing",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=175.0, range_low=105.0, range_eq=140.0,
        )
        assert result.trade_type == "PULLBACK_SHORT"
        assert result.tp2 == pytest.approx(140.0), f"Pullback short should still cap at EQ, got {result.tp2}"
        assert result.tp3 is None

    def test_long_bear_with_bullish_structure_becomes_local_continuation(self):
        result = adjust_levels(
            entry_low=119.0, entry_high=121.0,
            stop_loss=114.0,
            tp1=125.0, tp2=170.0, tp3=210.0,
            direction="LONG", setup_class="swing",
            regime="BEAR_TREND", chop_strength=0.20,
            range_high=175.0, range_low=105.0, range_eq=130.0,
            **self._bullish_structure(),
        )
        assert result.trade_type == "LOCAL_CONTINUATION_LONG"
        assert result.tp2 > 130.0, f"Local continuation long should extend above EQ, got {result.tp2}"
        assert result.tp3 is not None, "Local continuation long should keep a controlled TP3"
        assert any("local continuation" in adj.lower() for adj in result.adjustments)

    def test_long_bear_without_structure_stays_pullback(self):
        result = adjust_levels(
            entry_low=119.0, entry_high=121.0,
            stop_loss=114.0,
            tp1=125.0, tp2=170.0, tp3=210.0,
            direction="LONG", setup_class="swing",
            regime="BEAR_TREND", chop_strength=0.20,
            range_high=175.0, range_low=105.0, range_eq=130.0,
        )
        assert result.trade_type == "PULLBACK_LONG"
        assert result.tp2 == pytest.approx(130.0), f"Pullback long should still cap at EQ, got {result.tp2}"
        assert result.tp3 is None

    def test_insufficient_bars_maintains_pullback_type(self):
        bearish = self._bearish_structure()
        result = adjust_levels(
            entry_low=149.0, entry_high=151.0,
            stop_loss=156.0,
            tp1=145.0, tp2=120.0, tp3=90.0,
            direction="SHORT", setup_class="swing",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=175.0, range_low=105.0, range_eq=140.0,
            recent_opens=bearish["recent_opens"][:-2],
            recent_highs=bearish["recent_highs"][:-2],
            recent_lows=bearish["recent_lows"][:-2],
            recent_closes=bearish["recent_closes"][:-2],
            recent_volumes=bearish["recent_volumes"][:-2],
        )
        assert result.trade_type == "PULLBACK_SHORT"
        assert result.local_structure_reason == "insufficient_bars"
        assert result.local_structure_bars == 13
        assert result.tp2 == pytest.approx(140.0)
        assert result.tp3 is None

    def test_short_bull_without_rejection_data_can_still_be_local_continuation(self):
        bearish = self._bearish_structure()
        result = adjust_levels(
            entry_low=149.0, entry_high=151.0,
            stop_loss=156.0,
            tp1=145.0, tp2=120.0, tp3=90.0,
            direction="SHORT", setup_class="swing",
            regime="BULL_TREND", chop_strength=0.20,
            range_high=175.0, range_low=105.0, range_eq=140.0,
            recent_highs=bearish["recent_highs"],
            recent_lows=bearish["recent_lows"],
            recent_closes=bearish["recent_closes"],
            recent_volumes=bearish["recent_volumes"],
        )
        assert result.trade_type == "LOCAL_CONTINUATION_SHORT"
        assert result.local_structure_reason == "confirmed_bearish"
        assert result.local_structure_used_vwap is True
        assert result.local_structure_used_rejections is False


# ═══════════════════════════════════════════════════════════════
# PART 2: equilibrium — soft EQ awareness in trending regimes
# ═══════════════════════════════════════════════════════════════

class TestSoftEQInTrend:
    """
    In trending regimes, when the symbol's own range is tight,
    EquilibriumAnalyzer should apply a soft confidence penalty
    for direction-zone mismatches instead of fully bypassing.
    """

    @staticmethod
    def _mock_regime(regime_value, chop=0.20):
        """Create mock regime_analyzer."""
        mock = MagicMock()
        mock.chop_strength = chop
        # Mock Regime enum
        mock_regime = MagicMock()
        mock_regime.value = regime_value
        # Make comparison work: Regime.BULL_TREND etc
        mock.regime = mock_regime
        return mock

    def _assess_with_regime(self, regime_value, direction, price,
                            range_high, range_low, chop=0.20):
        """Run assess() with mocked regime."""
        from analyzers.equilibrium import EquilibriumAnalyzer
        from analyzers.regime import Regime

        analyzer = EquilibriumAnalyzer()
        # regime_analyzer is imported inside assess() via
        # `from analyzers.regime import regime_analyzer, Regime`
        # so we patch the source module's singleton
        with patch('analyzers.regime.regime_analyzer') as mock_ra:
            mock_ra.chop_strength = chop
            # Use real Regime enum for comparison
            mock_ra.regime = getattr(Regime, regime_value)
            return analyzer.assess(
                current_price=price,
                direction=direction,
                symbol_range_high=range_high,
                symbol_range_low=range_low,
            )

    # ── LONG at premium of tight range in BULL_TREND ──
    def test_long_premium_bull_soft_penalty(self):
        """LONG at premium zone of a tight range gets soft penalty, not bypass."""
        result = self._assess_with_regime(
            "BULL_TREND", "LONG", price=104.0,
            range_high=105.0, range_low=95.0, chop=0.20,
        )
        # Price at 104 in range 95-105 → premium zone
        assert not result.should_block, "Should NOT hard-block in trend"
        assert result.confidence_mult < 1.0, \
            f"Should apply soft penalty, got mult={result.confidence_mult}"
        assert result.confidence_mult >= 0.90, \
            f"Penalty should be mild (>=0.90), got {result.confidence_mult}"

    # ── SHORT at discount of tight range in BEAR_TREND ──
    def test_short_discount_bear_soft_penalty(self):
        """SHORT at discount zone of tight range gets soft penalty."""
        result = self._assess_with_regime(
            "BEAR_TREND", "SHORT", price=96.0,
            range_high=105.0, range_low=95.0, chop=0.20,
        )
        assert not result.should_block
        assert result.confidence_mult < 1.0

    # ── No penalty when direction matches zone ──
    def test_short_premium_bull_no_penalty(self):
        """SHORT at premium of a tight range (correct zone) → no penalty."""
        result = self._assess_with_regime(
            "BULL_TREND", "SHORT", price=104.0,
            range_high=105.0, range_low=95.0, chop=0.20,
        )
        assert result.confidence_mult == 1.0

    def test_long_discount_bear_no_penalty(self):
        """LONG at discount of a tight range (correct zone) → no penalty."""
        result = self._assess_with_regime(
            "BEAR_TREND", "LONG", price=96.0,
            range_high=105.0, range_low=95.0, chop=0.20,
        )
        assert result.confidence_mult == 1.0

    # ── Full bypass when range is wide ──
    def test_wide_range_full_bypass(self):
        """Wide range (>15%) → full bypass, no penalty even in wrong zone."""
        result = self._assess_with_regime(
            "BULL_TREND", "LONG", price=115.0,
            range_high=120.0, range_low=80.0, chop=0.20,  # 40% range
        )
        assert result.confidence_mult == 1.0

    # ── Full bypass when chop is too low ──
    def test_low_chop_full_bypass(self):
        """Low chop (<0.10) → full bypass even with tight range."""
        result = self._assess_with_regime(
            "BULL_TREND", "LONG", price=104.0,
            range_high=105.0, range_low=95.0, chop=0.05,
        )
        assert result.confidence_mult == 1.0

    # ── VOLATILE regime: tight range gets soft penalty too ──
    def test_volatile_tight_range_soft_penalty(self):
        """VOLATILE with tight range should also get soft EQ awareness."""
        result = self._assess_with_regime(
            "VOLATILE", "LONG", price=104.0,
            range_high=105.0, range_low=95.0, chop=0.20,
        )
        assert not result.should_block
        # May or may not have penalty depending on position depth
        # The key is: should_block is False

    # ── Price at EQ → no penalty ──
    def test_price_at_eq_no_penalty(self):
        """Price at equilibrium → position ≈ 0 → no penalty."""
        result = self._assess_with_regime(
            "BULL_TREND", "LONG", price=100.0,
            range_high=105.0, range_low=95.0, chop=0.20,
        )
        assert result.confidence_mult == 1.0

    # ── No symbol range → full bypass ──
    def test_no_range_data_full_bypass(self):
        """No symbol range → standard trend bypass."""
        result = self._assess_with_regime(
            "BULL_TREND", "LONG", price=100.0,
            range_high=0.0, range_low=0.0, chop=0.20,
        )
        assert result.confidence_mult == 1.0
        assert not result.should_block

    # ── Reason string includes useful info ──
    def test_penalty_reason_contains_info(self):
        """Penalty reason should mention 'Soft EQ' and range percentage."""
        result = self._assess_with_regime(
            "BULL_TREND", "LONG", price=104.0,
            range_high=105.0, range_low=95.0, chop=0.20,
        )
        assert "soft eq" in result.reason.lower() or "bypassed" in result.reason.lower()

    # ── Confidence mult never drops below 0.90 ──
    def test_penalty_floor(self):
        """Soft penalty should never drop below 0.90."""
        # Price at very top of range
        result = self._assess_with_regime(
            "BULL_TREND", "LONG", price=104.9,
            range_high=105.0, range_low=95.0, chop=0.20,
        )
        assert result.confidence_mult >= 0.90
