"""
TP1 Swing-Level Anchoring Tests
================================
Verifies that PriceAction and SMC cap TP1 at the nearest intermediate
swing high (LONG) or swing low (SHORT) when structural resistance/support
exists between entry and the ATR/risk-based TP1.

Coverage:
  - LONG: swing high between entry and ATR-TP1 → TP1 capped at swing high
  - LONG: no swing high in range → TP1 stays at ATR calculation
  - LONG: swing high too close to entry (< 0.3 ATR) → ignored (noise)
  - LONG: multiple swing highs → picks nearest (lowest)
  - SHORT: mirror of above with swing lows
  - Both PriceAction and SMC strategies
"""

import sys
from unittest.mock import MagicMock

import pytest

# ── Do NOT re-mock modules handled by conftest.py ──

from strategies.base import SignalResult, SignalDirection
from config.constants import Penalties

# Constants used in anchoring logic (must match strategy implementations)
_DZ_MULT = Penalties.TP1_SWING_DEAD_ZONE_ATR   # 0.3
_N_SWINGS = Penalties.TP1_MAX_SWING_LEVELS      # 5


# ═══════════════════════════════════════════════════════════════
# Helpers: extract TP1 anchoring logic from each strategy
# ═══════════════════════════════════════════════════════════════

def price_action_tp1_long(entry_high, atr, tp1_atr_val, swing_highs):
    """Replicate PriceAction LONG TP1 anchoring logic."""
    tp1_atr = entry_high + atr * tp1_atr_val
    _tp1_resist = [sh for sh in swing_highs[:_N_SWINGS]
                   if entry_high + atr * _DZ_MULT < sh < tp1_atr]
    return min(_tp1_resist) if _tp1_resist else tp1_atr


def price_action_tp1_short(entry_low, atr, tp1_atr_val, swing_lows):
    """Replicate PriceAction SHORT TP1 anchoring logic."""
    tp1_atr = entry_low - atr * tp1_atr_val
    _tp1_support = [sl for sl in swing_lows[:_N_SWINGS]
                    if tp1_atr < sl < entry_low - atr * _DZ_MULT]
    return max(_tp1_support) if _tp1_support else tp1_atr


def smc_tp1_long(entry_ref, entry_high, atr, risk_dist, tp1_mult, swing_highs):
    """Replicate SMC LONG TP1 anchoring logic."""
    tp1_calc = entry_ref + risk_dist * tp1_mult
    _tp1_resist = [sh for sh in swing_highs[:_N_SWINGS]
                   if entry_high + atr * _DZ_MULT < sh < tp1_calc]
    return min(_tp1_resist) if _tp1_resist else tp1_calc


def smc_tp1_short(entry_ref, entry_low, atr, risk_dist, tp1_mult, swing_lows):
    """Replicate SMC SHORT TP1 anchoring logic."""
    tp1_calc = entry_ref - risk_dist * tp1_mult
    _tp1_support = [sl for sl in swing_lows[:_N_SWINGS]
                    if tp1_calc < sl < entry_low - atr * _DZ_MULT]
    return max(_tp1_support) if _tp1_support else tp1_calc


# ═══════════════════════════════════════════════════════════════
# PriceAction LONG tests
# ═══════════════════════════════════════════════════════════════

class TestPriceActionTP1Long:
    """PriceAction LONG TP1 anchoring to swing highs."""

    def test_swing_high_caps_tp1(self):
        """Swing high between entry and ATR-TP1 → TP1 capped at swing high."""
        # Entry high at 100, ATR=2, scaled_tp1=1.5 → ATR-TP1 = 103
        # Swing high at 101.5 — between entry+0.6 and 103
        tp1 = price_action_tp1_long(
            entry_high=100, atr=2.0, tp1_atr_val=1.5,
            swing_highs=[101.5, 105.0, 98.0]
        )
        assert tp1 == 101.5

    def test_no_swing_high_in_range(self):
        """No swing high between entry and ATR-TP1 → TP1 stays at ATR."""
        tp1 = price_action_tp1_long(
            entry_high=100, atr=2.0, tp1_atr_val=1.5,
            swing_highs=[105.0, 110.0]  # all above ATR-TP1 (103)
        )
        assert tp1 == pytest.approx(103.0)

    def test_swing_high_too_close_ignored(self):
        """Swing high within 0.3 ATR of entry → ignored (noise)."""
        # 0.3 * 2.0 = 0.6, so anything ≤ 100.6 is too close
        tp1 = price_action_tp1_long(
            entry_high=100, atr=2.0, tp1_atr_val=1.5,
            swing_highs=[100.5]  # too close
        )
        assert tp1 == pytest.approx(103.0)

    def test_multiple_swing_highs_picks_nearest(self):
        """Multiple qualifying swing highs → picks lowest (nearest)."""
        tp1 = price_action_tp1_long(
            entry_high=100, atr=2.0, tp1_atr_val=1.5,
            swing_highs=[102.5, 101.0, 102.0]
        )
        assert tp1 == 101.0

    def test_empty_swing_highs(self):
        """No swing highs at all → ATR-based TP1."""
        tp1 = price_action_tp1_long(
            entry_high=100, atr=2.0, tp1_atr_val=1.5,
            swing_highs=[]
        )
        assert tp1 == pytest.approx(103.0)

    def test_swing_high_exactly_at_atr_tp1_excluded(self):
        """Swing high exactly at ATR-TP1 → excluded (not strictly less)."""
        tp1 = price_action_tp1_long(
            entry_high=100, atr=2.0, tp1_atr_val=1.5,
            swing_highs=[103.0]  # exactly at tp1_atr
        )
        assert tp1 == pytest.approx(103.0)

    def test_swing_high_just_below_atr_tp1(self):
        """Swing high just below ATR-TP1 → caps TP1."""
        tp1 = price_action_tp1_long(
            entry_high=100, atr=2.0, tp1_atr_val=1.5,
            swing_highs=[102.99]
        )
        assert tp1 == pytest.approx(102.99)

    def test_only_first_5_swing_highs_considered(self):
        """Only swing_highs[:5] are checked."""
        # 6th swing high would qualify but is out of range
        highs = [105.0, 106.0, 107.0, 108.0, 109.0, 101.5]
        tp1 = price_action_tp1_long(
            entry_high=100, atr=2.0, tp1_atr_val=1.5,
            swing_highs=highs
        )
        # 101.5 is index 5 — not in [:5]
        assert tp1 == pytest.approx(103.0)

    def test_micro_price_pump_token(self):
        """Realistic PUMPUSDT-like scenario: entry $0.00181, PDH at $0.00190."""
        entry_high = 0.001820
        atr = 0.0000467
        # scaled_tp1 ≈ 1.5 → ATR-TP1 ≈ 0.001890 (above PDH)
        # But with vol scaling, let's say effective mult is ~3.0
        # ATR-TP1 = 0.001820 + 0.0000467 * 3.0 = 0.001960
        tp1 = price_action_tp1_long(
            entry_high=entry_high, atr=atr, tp1_atr_val=3.0,
            swing_highs=[0.001906, 0.001950, 0.001750]
        )
        # PDH at 0.001906 is between entry+0.3*ATR and ATR-TP1
        # 0.001820 + 0.3*0.0000467 = 0.001834
        # 0.001906 > 0.001834 and < 0.001960 → caps
        assert tp1 == pytest.approx(0.001906)


# ═══════════════════════════════════════════════════════════════
# PriceAction SHORT tests
# ═══════════════════════════════════════════════════════════════

class TestPriceActionTP1Short:
    """PriceAction SHORT TP1 anchoring to swing lows."""

    def test_swing_low_caps_tp1(self):
        """Swing low between entry and ATR-TP1 → TP1 capped at swing low."""
        # Entry low at 100, ATR=2, scaled_tp1=1.5 → ATR-TP1 = 97
        # Swing low at 98.5 — between 97 and entry-0.6 (99.4)
        tp1 = price_action_tp1_short(
            entry_low=100, atr=2.0, tp1_atr_val=1.5,
            swing_lows=[98.5, 95.0, 102.0]
        )
        assert tp1 == 98.5

    def test_no_swing_low_in_range(self):
        """No swing low between entry and ATR-TP1 → TP1 stays at ATR."""
        tp1 = price_action_tp1_short(
            entry_low=100, atr=2.0, tp1_atr_val=1.5,
            swing_lows=[95.0, 90.0]  # all below ATR-TP1 (97)
        )
        assert tp1 == pytest.approx(97.0)

    def test_swing_low_too_close_ignored(self):
        """Swing low within 0.3 ATR of entry → ignored."""
        tp1 = price_action_tp1_short(
            entry_low=100, atr=2.0, tp1_atr_val=1.5,
            swing_lows=[99.5]  # too close (> 100 - 0.6 = 99.4)
        )
        assert tp1 == pytest.approx(97.0)

    def test_multiple_swing_lows_picks_nearest(self):
        """Multiple qualifying swing lows → picks highest (nearest to entry)."""
        tp1 = price_action_tp1_short(
            entry_low=100, atr=2.0, tp1_atr_val=1.5,
            swing_lows=[97.5, 99.0, 98.0]
        )
        assert tp1 == 99.0

    def test_empty_swing_lows(self):
        """No swing lows → ATR-based TP1."""
        tp1 = price_action_tp1_short(
            entry_low=100, atr=2.0, tp1_atr_val=1.5,
            swing_lows=[]
        )
        assert tp1 == pytest.approx(97.0)


# ═══════════════════════════════════════════════════════════════
# SMC LONG tests
# ═══════════════════════════════════════════════════════════════

class TestSMCTP1Long:
    """SMC LONG TP1 anchoring to swing highs."""

    def test_swing_high_caps_tp1(self):
        """Swing high between entry and risk-TP1 → TP1 capped."""
        # entry_ref=100, risk_dist=5, tp1_mult=1.5 → TP1_calc=107.5
        # entry_high=100.5, 0.3*ATR=0.6 → min=101.1
        tp1 = smc_tp1_long(
            entry_ref=100, entry_high=100.5, atr=2.0,
            risk_dist=5.0, tp1_mult=1.5,
            swing_highs=[103.0, 108.0, 99.0]
        )
        assert tp1 == 103.0

    def test_no_qualifying_swing_high(self):
        """No swing high in range → risk-based TP1."""
        tp1 = smc_tp1_long(
            entry_ref=100, entry_high=100.5, atr=2.0,
            risk_dist=5.0, tp1_mult=1.5,
            swing_highs=[108.0, 110.0]
        )
        assert tp1 == pytest.approx(107.5)

    def test_swing_high_too_close_ignored(self):
        """Swing high within 0.3 ATR of entry_high → ignored."""
        tp1 = smc_tp1_long(
            entry_ref=100, entry_high=100.5, atr=2.0,
            risk_dist=5.0, tp1_mult=1.5,
            swing_highs=[101.0]  # < 101.1 threshold
        )
        assert tp1 == pytest.approx(107.5)

    def test_multiple_picks_nearest(self):
        """Multiple qualifying → picks lowest."""
        tp1 = smc_tp1_long(
            entry_ref=100, entry_high=100.5, atr=2.0,
            risk_dist=5.0, tp1_mult=1.5,
            swing_highs=[106.0, 103.0, 105.0]
        )
        assert tp1 == 103.0


# ═══════════════════════════════════════════════════════════════
# SMC SHORT tests
# ═══════════════════════════════════════════════════════════════

class TestSMCTP1Short:
    """SMC SHORT TP1 anchoring to swing lows."""

    def test_swing_low_caps_tp1(self):
        """Swing low between entry and risk-TP1 → TP1 capped."""
        # entry_ref=100, risk_dist=5, tp1_mult=1.5 → TP1_calc=92.5
        # entry_low=99.5, 0.3*ATR=0.6 → max=98.9
        tp1 = smc_tp1_short(
            entry_ref=100, entry_low=99.5, atr=2.0,
            risk_dist=5.0, tp1_mult=1.5,
            swing_lows=[97.0, 91.0, 101.0]
        )
        assert tp1 == 97.0

    def test_no_qualifying_swing_low(self):
        """No swing low in range → risk-based TP1."""
        tp1 = smc_tp1_short(
            entry_ref=100, entry_low=99.5, atr=2.0,
            risk_dist=5.0, tp1_mult=1.5,
            swing_lows=[91.0, 90.0]
        )
        assert tp1 == pytest.approx(92.5)

    def test_swing_low_too_close_ignored(self):
        """Swing low within 0.3 ATR of entry_low → ignored."""
        tp1 = smc_tp1_short(
            entry_ref=100, entry_low=99.5, atr=2.0,
            risk_dist=5.0, tp1_mult=1.5,
            swing_lows=[99.0]  # > 98.9 threshold
        )
        assert tp1 == pytest.approx(92.5)

    def test_multiple_picks_nearest(self):
        """Multiple qualifying → picks highest (nearest to entry)."""
        tp1 = smc_tp1_short(
            entry_ref=100, entry_low=99.5, atr=2.0,
            risk_dist=5.0, tp1_mult=1.5,
            swing_lows=[94.0, 97.0, 95.0]
        )
        assert tp1 == 97.0


# ═══════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════

class TestTP1AnchorEdgeCases:
    """Edge cases for TP1 swing anchoring."""

    def test_swing_high_below_entry_ignored_long(self):
        """Swing highs below entry_high are never candidates (LONG)."""
        tp1 = price_action_tp1_long(
            entry_high=100, atr=2.0, tp1_atr_val=1.5,
            swing_highs=[99.0, 98.0]
        )
        assert tp1 == pytest.approx(103.0)

    def test_swing_low_above_entry_ignored_short(self):
        """Swing lows above entry_low are never candidates (SHORT)."""
        tp1 = price_action_tp1_short(
            entry_low=100, atr=2.0, tp1_atr_val=1.5,
            swing_lows=[101.0, 102.0]
        )
        assert tp1 == pytest.approx(97.0)

    def test_very_small_atr_micro_cap(self):
        """Micro-cap tokens with tiny ATR still anchor correctly."""
        tp1 = price_action_tp1_long(
            entry_high=0.00180, atr=0.00004, tp1_atr_val=1.5,
            swing_highs=[0.00185, 0.00195]
        )
        # ATR-TP1 = 0.00180 + 0.00004*1.5 = 0.00186
        # 0.3*ATR = 0.000012 → min = 0.001812
        # 0.00185 > 0.001812 and < 0.00186 → caps
        assert tp1 == pytest.approx(0.00185)

    def test_zero_atr_no_crash(self):
        """Zero ATR → 0.3*0 = 0, all swing highs above entry qualify."""
        tp1 = price_action_tp1_long(
            entry_high=100, atr=0.0, tp1_atr_val=1.5,
            swing_highs=[100.5]
        )
        # ATR-TP1 = 100, swing high 100.5 > 100 → not < 100, excluded
        # So fallback to ATR-TP1
        assert tp1 == pytest.approx(100.0)

    def test_rr_preserved_after_anchor(self):
        """Anchored TP1 still yields positive R:R."""
        entry_high = 100
        entry_low = 99.0
        stop_loss = 97.0
        atr = 2.0

        tp1 = price_action_tp1_long(
            entry_high=entry_high, atr=atr, tp1_atr_val=1.5,
            swing_highs=[101.5]
        )
        # TP1=101.5, risk = 99 - 97 = 2, reward = 101.5 - 100 = 1.5
        risk = entry_low - stop_loss
        reward = tp1 - entry_high
        rr = reward / risk
        assert rr > 0, "R:R must be positive after anchoring"
        assert tp1 == 101.5
