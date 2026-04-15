"""
Direction-Aware Regime Confidence Audit
=======================================
Verifies that strategies with `_REGIME_CONF_WITH_TREND` / `_REGIME_CONF_COUNTER_TREND`
apply the correct confidence delta based on direction × regime.

Fixes audited:
  - SMC: +8 with-trend, -12 counter-trend (was flat +8)
  - PriceAction: +5 with-trend, -8 counter-trend (was flat +5)
  - Ichimoku: +6 with-trend, -10 counter-trend (was 0)
  - ElliottWave: +6 with-trend, -10 counter-trend (was 0)

Production evidence: SMC generated 2 LONG signals in BEAR_TREND (both killed)
because the flat +8 inflated counter-trend confidence instead of penalizing it.
"""

import pytest


# ═══════════════════════════════════════════════════════════════════
# 1. Class-level attribute verification
# ═══════════════════════════════════════════════════════════════════

class TestSMCRegimeConfidence:
    """SmartMoneyConcepts direction-aware regime confidence."""

    def test_has_with_trend_map(self):
        from strategies.smc import SmartMoneyConcepts
        assert hasattr(SmartMoneyConcepts, '_REGIME_CONF_WITH_TREND')

    def test_has_counter_trend_map(self):
        from strategies.smc import SmartMoneyConcepts
        assert hasattr(SmartMoneyConcepts, '_REGIME_CONF_COUNTER_TREND')

    def test_no_flat_delta(self):
        """Old flat _REGIME_CONFIDENCE_DELTA must be removed."""
        from strategies.smc import SmartMoneyConcepts
        assert not hasattr(SmartMoneyConcepts, '_REGIME_CONFIDENCE_DELTA')

    def test_bull_with_trend_positive(self):
        from strategies.smc import SmartMoneyConcepts
        assert SmartMoneyConcepts._REGIME_CONF_WITH_TREND["BULL_TREND"] > 0

    def test_bear_with_trend_positive(self):
        from strategies.smc import SmartMoneyConcepts
        assert SmartMoneyConcepts._REGIME_CONF_WITH_TREND["BEAR_TREND"] > 0

    def test_bull_counter_trend_negative(self):
        from strategies.smc import SmartMoneyConcepts
        assert SmartMoneyConcepts._REGIME_CONF_COUNTER_TREND["BULL_TREND"] < 0

    def test_bear_counter_trend_negative(self):
        from strategies.smc import SmartMoneyConcepts
        assert SmartMoneyConcepts._REGIME_CONF_COUNTER_TREND["BEAR_TREND"] < 0

    def test_with_trend_boost_equals_8(self):
        from strategies.smc import SmartMoneyConcepts
        assert SmartMoneyConcepts._REGIME_CONF_WITH_TREND["BULL_TREND"] == 8

    def test_counter_trend_penalty_equals_neg12(self):
        from strategies.smc import SmartMoneyConcepts
        assert SmartMoneyConcepts._REGIME_CONF_COUNTER_TREND["BEAR_TREND"] == -12

    def test_volatile_same_both_maps(self):
        from strategies.smc import SmartMoneyConcepts
        assert SmartMoneyConcepts._REGIME_CONF_WITH_TREND["VOLATILE"] == \
               SmartMoneyConcepts._REGIME_CONF_COUNTER_TREND["VOLATILE"]

    def test_choppy_same_both_maps(self):
        from strategies.smc import SmartMoneyConcepts
        assert SmartMoneyConcepts._REGIME_CONF_WITH_TREND["CHOPPY"] == \
               SmartMoneyConcepts._REGIME_CONF_COUNTER_TREND["CHOPPY"]


class TestPriceActionRegimeConfidence:
    """PriceAction direction-aware regime confidence."""

    def test_has_with_trend_map(self):
        from strategies.price_action import PriceAction
        assert hasattr(PriceAction, '_REGIME_CONF_WITH_TREND')

    def test_has_counter_trend_map(self):
        from strategies.price_action import PriceAction
        assert hasattr(PriceAction, '_REGIME_CONF_COUNTER_TREND')

    def test_no_flat_delta(self):
        from strategies.price_action import PriceAction
        assert not hasattr(PriceAction, '_REGIME_CONFIDENCE_DELTA')

    def test_bull_with_trend_positive(self):
        from strategies.price_action import PriceAction
        assert PriceAction._REGIME_CONF_WITH_TREND["BULL_TREND"] > 0

    def test_bull_counter_trend_negative(self):
        from strategies.price_action import PriceAction
        assert PriceAction._REGIME_CONF_COUNTER_TREND["BULL_TREND"] < 0

    def test_with_trend_equals_5(self):
        from strategies.price_action import PriceAction
        assert PriceAction._REGIME_CONF_WITH_TREND["BEAR_TREND"] == 5

    def test_counter_trend_equals_neg8(self):
        from strategies.price_action import PriceAction
        assert PriceAction._REGIME_CONF_COUNTER_TREND["BEAR_TREND"] == -8


class TestIchimokuRegimeConfidence:
    """Ichimoku direction-aware regime confidence."""

    def test_has_with_trend_map(self):
        from strategies.ichimoku import Ichimoku
        assert hasattr(Ichimoku, '_REGIME_CONF_WITH_TREND')

    def test_has_counter_trend_map(self):
        from strategies.ichimoku import Ichimoku
        assert hasattr(Ichimoku, '_REGIME_CONF_COUNTER_TREND')

    def test_bull_with_trend_positive(self):
        from strategies.ichimoku import Ichimoku
        assert Ichimoku._REGIME_CONF_WITH_TREND["BULL_TREND"] > 0

    def test_bull_counter_trend_negative(self):
        from strategies.ichimoku import Ichimoku
        assert Ichimoku._REGIME_CONF_COUNTER_TREND["BULL_TREND"] < 0

    def test_with_trend_equals_6(self):
        from strategies.ichimoku import Ichimoku
        assert Ichimoku._REGIME_CONF_WITH_TREND["BEAR_TREND"] == 6

    def test_counter_trend_equals_neg10(self):
        from strategies.ichimoku import Ichimoku
        assert Ichimoku._REGIME_CONF_COUNTER_TREND["BEAR_TREND"] == -10


class TestElliottWaveRegimeConfidence:
    """ElliottWave direction-aware regime confidence."""

    def test_has_with_trend_map(self):
        from strategies.elliott_wave import ElliottWave
        assert hasattr(ElliottWave, '_REGIME_CONF_WITH_TREND')

    def test_has_counter_trend_map(self):
        from strategies.elliott_wave import ElliottWave
        assert hasattr(ElliottWave, '_REGIME_CONF_COUNTER_TREND')

    def test_bull_with_trend_positive(self):
        from strategies.elliott_wave import ElliottWave
        assert ElliottWave._REGIME_CONF_WITH_TREND["BULL_TREND"] > 0

    def test_bull_counter_trend_negative(self):
        from strategies.elliott_wave import ElliottWave
        assert ElliottWave._REGIME_CONF_COUNTER_TREND["BULL_TREND"] < 0

    def test_with_trend_equals_6(self):
        from strategies.elliott_wave import ElliottWave
        assert ElliottWave._REGIME_CONF_WITH_TREND["BEAR_TREND"] == 6

    def test_counter_trend_equals_neg10(self):
        from strategies.elliott_wave import ElliottWave
        assert ElliottWave._REGIME_CONF_COUNTER_TREND["BEAR_TREND"] == -10


# ═══════════════════════════════════════════════════════════════════
# 2. Direction-aware regime bonus computation logic
# ═══════════════════════════════════════════════════════════════════

def _compute_regime_bonus(with_trend_map, counter_trend_map, direction, regime):
    """
    Replicate the direction-aware regime bonus logic used in SMC/PriceAction/Ichimoku/ElliottWave.
    This is the exact pattern from the strategy code.
    """
    _is_with_trend = (
        (direction == "LONG" and regime == "BULL_TREND") or
        (direction == "SHORT" and regime == "BEAR_TREND")
    )
    if _is_with_trend or regime not in ("BULL_TREND", "BEAR_TREND"):
        return with_trend_map.get(regime, 0)
    else:
        return counter_trend_map.get(regime, 0)


class TestRegimeBonusLogic:
    """Verify the direction × regime → bonus mapping is correct."""

    # ── SMC ────────────────────────────────────────────────────────────

    def test_smc_long_in_bull_gets_plus8(self):
        from strategies.smc import SmartMoneyConcepts as S
        bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "LONG", "BULL_TREND")
        assert bonus == 8

    def test_smc_short_in_bear_gets_plus8(self):
        from strategies.smc import SmartMoneyConcepts as S
        bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "SHORT", "BEAR_TREND")
        assert bonus == 8

    def test_smc_long_in_bear_gets_neg12(self):
        """THE FIX: LONG in BEAR_TREND was +8, now -12."""
        from strategies.smc import SmartMoneyConcepts as S
        bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "LONG", "BEAR_TREND")
        assert bonus == -12

    def test_smc_short_in_bull_gets_neg12(self):
        from strategies.smc import SmartMoneyConcepts as S
        bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "SHORT", "BULL_TREND")
        assert bonus == -12

    def test_smc_long_in_volatile_gets_plus3(self):
        from strategies.smc import SmartMoneyConcepts as S
        bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "LONG", "VOLATILE")
        assert bonus == 3

    def test_smc_short_in_volatile_gets_plus3(self):
        from strategies.smc import SmartMoneyConcepts as S
        bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "SHORT", "VOLATILE")
        assert bonus == 3

    def test_smc_long_in_choppy_gets_neg10(self):
        from strategies.smc import SmartMoneyConcepts as S
        bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "LONG", "CHOPPY")
        assert bonus == -10

    def test_smc_unknown_gets_zero(self):
        from strategies.smc import SmartMoneyConcepts as S
        bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "LONG", "UNKNOWN")
        assert bonus == 0

    # ── PriceAction ────────────────────────────────────────────────────

    def test_pa_long_in_bull_gets_plus5(self):
        from strategies.price_action import PriceAction as PA
        bonus = _compute_regime_bonus(PA._REGIME_CONF_WITH_TREND, PA._REGIME_CONF_COUNTER_TREND, "LONG", "BULL_TREND")
        assert bonus == 5

    def test_pa_long_in_bear_gets_neg8(self):
        from strategies.price_action import PriceAction as PA
        bonus = _compute_regime_bonus(PA._REGIME_CONF_WITH_TREND, PA._REGIME_CONF_COUNTER_TREND, "LONG", "BEAR_TREND")
        assert bonus == -8

    def test_pa_short_in_bear_gets_plus5(self):
        from strategies.price_action import PriceAction as PA
        bonus = _compute_regime_bonus(PA._REGIME_CONF_WITH_TREND, PA._REGIME_CONF_COUNTER_TREND, "SHORT", "BEAR_TREND")
        assert bonus == 5

    def test_pa_short_in_bull_gets_neg8(self):
        from strategies.price_action import PriceAction as PA
        bonus = _compute_regime_bonus(PA._REGIME_CONF_WITH_TREND, PA._REGIME_CONF_COUNTER_TREND, "SHORT", "BULL_TREND")
        assert bonus == -8

    # ── Ichimoku ───────────────────────────────────────────────────────

    def test_ichi_long_in_bull_gets_plus6(self):
        from strategies.ichimoku import Ichimoku as IC
        bonus = _compute_regime_bonus(IC._REGIME_CONF_WITH_TREND, IC._REGIME_CONF_COUNTER_TREND, "LONG", "BULL_TREND")
        assert bonus == 6

    def test_ichi_long_in_bear_gets_neg10(self):
        from strategies.ichimoku import Ichimoku as IC
        bonus = _compute_regime_bonus(IC._REGIME_CONF_WITH_TREND, IC._REGIME_CONF_COUNTER_TREND, "LONG", "BEAR_TREND")
        assert bonus == -10

    def test_ichi_short_in_bear_gets_plus6(self):
        from strategies.ichimoku import Ichimoku as IC
        bonus = _compute_regime_bonus(IC._REGIME_CONF_WITH_TREND, IC._REGIME_CONF_COUNTER_TREND, "SHORT", "BEAR_TREND")
        assert bonus == 6

    def test_ichi_short_in_bull_gets_neg10(self):
        from strategies.ichimoku import Ichimoku as IC
        bonus = _compute_regime_bonus(IC._REGIME_CONF_WITH_TREND, IC._REGIME_CONF_COUNTER_TREND, "SHORT", "BULL_TREND")
        assert bonus == -10

    # ── ElliottWave ────────────────────────────────────────────────────

    def test_ew_long_in_bull_gets_plus6(self):
        from strategies.elliott_wave import ElliottWave as EW
        bonus = _compute_regime_bonus(EW._REGIME_CONF_WITH_TREND, EW._REGIME_CONF_COUNTER_TREND, "LONG", "BULL_TREND")
        assert bonus == 6

    def test_ew_long_in_bear_gets_neg10(self):
        from strategies.elliott_wave import ElliottWave as EW
        bonus = _compute_regime_bonus(EW._REGIME_CONF_WITH_TREND, EW._REGIME_CONF_COUNTER_TREND, "LONG", "BEAR_TREND")
        assert bonus == -10

    def test_ew_short_in_bear_gets_plus6(self):
        from strategies.elliott_wave import ElliottWave as EW
        bonus = _compute_regime_bonus(EW._REGIME_CONF_WITH_TREND, EW._REGIME_CONF_COUNTER_TREND, "SHORT", "BEAR_TREND")
        assert bonus == 6

    def test_ew_short_in_bull_gets_neg10(self):
        from strategies.elliott_wave import ElliottWave as EW
        bonus = _compute_regime_bonus(EW._REGIME_CONF_WITH_TREND, EW._REGIME_CONF_COUNTER_TREND, "SHORT", "BULL_TREND")
        assert bonus == -10


# ═══════════════════════════════════════════════════════════════════
# 3. Full direction × regime matrix for all 4 strategies
# ═══════════════════════════════════════════════════════════════════

_DIRECTIONS = ["LONG", "SHORT"]

# Expected sign of the bonus for each (direction, regime) combo
_TRENDING_EXPECTATIONS = {
    ("LONG",  "BULL_TREND"):  "positive",
    ("LONG",  "BEAR_TREND"):  "negative",
    ("SHORT", "BULL_TREND"):  "negative",
    ("SHORT", "BEAR_TREND"):  "positive",
}


class TestDirectionRegimeMatrix:
    """Cross-product: every direction × trending regime must produce correct sign."""

    @pytest.mark.parametrize("direction,regime,expected_sign", [
        ("LONG",  "BULL_TREND", "positive"),
        ("LONG",  "BEAR_TREND", "negative"),
        ("SHORT", "BULL_TREND", "negative"),
        ("SHORT", "BEAR_TREND", "positive"),
    ])
    def test_smc_sign(self, direction, regime, expected_sign):
        from strategies.smc import SmartMoneyConcepts as S
        bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, direction, regime)
        if expected_sign == "positive":
            assert bonus > 0, f"SMC {direction} in {regime} should be positive, got {bonus}"
        else:
            assert bonus < 0, f"SMC {direction} in {regime} should be negative, got {bonus}"

    @pytest.mark.parametrize("direction,regime,expected_sign", [
        ("LONG",  "BULL_TREND", "positive"),
        ("LONG",  "BEAR_TREND", "negative"),
        ("SHORT", "BULL_TREND", "negative"),
        ("SHORT", "BEAR_TREND", "positive"),
    ])
    def test_price_action_sign(self, direction, regime, expected_sign):
        from strategies.price_action import PriceAction as PA
        bonus = _compute_regime_bonus(PA._REGIME_CONF_WITH_TREND, PA._REGIME_CONF_COUNTER_TREND, direction, regime)
        if expected_sign == "positive":
            assert bonus > 0, f"PA {direction} in {regime} should be positive, got {bonus}"
        else:
            assert bonus < 0, f"PA {direction} in {regime} should be negative, got {bonus}"

    @pytest.mark.parametrize("direction,regime,expected_sign", [
        ("LONG",  "BULL_TREND", "positive"),
        ("LONG",  "BEAR_TREND", "negative"),
        ("SHORT", "BULL_TREND", "negative"),
        ("SHORT", "BEAR_TREND", "positive"),
    ])
    def test_ichimoku_sign(self, direction, regime, expected_sign):
        from strategies.ichimoku import Ichimoku as IC
        bonus = _compute_regime_bonus(IC._REGIME_CONF_WITH_TREND, IC._REGIME_CONF_COUNTER_TREND, direction, regime)
        if expected_sign == "positive":
            assert bonus > 0, f"Ichimoku {direction} in {regime} should be positive, got {bonus}"
        else:
            assert bonus < 0, f"Ichimoku {direction} in {regime} should be negative, got {bonus}"

    @pytest.mark.parametrize("direction,regime,expected_sign", [
        ("LONG",  "BULL_TREND", "positive"),
        ("LONG",  "BEAR_TREND", "negative"),
        ("SHORT", "BULL_TREND", "negative"),
        ("SHORT", "BEAR_TREND", "positive"),
    ])
    def test_elliott_wave_sign(self, direction, regime, expected_sign):
        from strategies.elliott_wave import ElliottWave as EW
        bonus = _compute_regime_bonus(EW._REGIME_CONF_WITH_TREND, EW._REGIME_CONF_COUNTER_TREND, direction, regime)
        if expected_sign == "positive":
            assert bonus > 0, f"ElliottWave {direction} in {regime} should be positive, got {bonus}"
        else:
            assert bonus < 0, f"ElliottWave {direction} in {regime} should be negative, got {bonus}"


# ═══════════════════════════════════════════════════════════════════
# 4. Counter-trend penalty magnitude ordering
# ═══════════════════════════════════════════════════════════════════

class TestPenaltyMagnitude:
    """Counter-trend penalty should be larger (more negative) than with-trend boost."""

    def test_smc_penalty_larger_than_boost(self):
        from strategies.smc import SmartMoneyConcepts as S
        boost = S._REGIME_CONF_WITH_TREND["BEAR_TREND"]
        penalty = S._REGIME_CONF_COUNTER_TREND["BEAR_TREND"]
        assert abs(penalty) > abs(boost), "Counter-trend penalty should exceed with-trend boost"

    def test_pa_penalty_larger_than_boost(self):
        from strategies.price_action import PriceAction as PA
        boost = PA._REGIME_CONF_WITH_TREND["BEAR_TREND"]
        penalty = PA._REGIME_CONF_COUNTER_TREND["BEAR_TREND"]
        assert abs(penalty) > abs(boost)

    def test_ichi_penalty_larger_than_boost(self):
        from strategies.ichimoku import Ichimoku as IC
        boost = IC._REGIME_CONF_WITH_TREND["BEAR_TREND"]
        penalty = IC._REGIME_CONF_COUNTER_TREND["BEAR_TREND"]
        assert abs(penalty) > abs(boost)

    def test_ew_penalty_larger_than_boost(self):
        from strategies.elliott_wave import ElliottWave as EW
        boost = EW._REGIME_CONF_WITH_TREND["BEAR_TREND"]
        penalty = EW._REGIME_CONF_COUNTER_TREND["BEAR_TREND"]
        assert abs(penalty) > abs(boost)


# ═══════════════════════════════════════════════════════════════════
# 5. Non-trending regimes are direction-neutral
# ═══════════════════════════════════════════════════════════════════

class TestNonTrendingNeutral:
    """In VOLATILE/CHOPPY/UNKNOWN, bonus is the same for both directions."""

    @pytest.mark.parametrize("regime", ["VOLATILE", "CHOPPY", "UNKNOWN"])
    def test_smc_neutral(self, regime):
        from strategies.smc import SmartMoneyConcepts as S
        long_bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "LONG", regime)
        short_bonus = _compute_regime_bonus(S._REGIME_CONF_WITH_TREND, S._REGIME_CONF_COUNTER_TREND, "SHORT", regime)
        assert long_bonus == short_bonus, f"Non-trending {regime} should be neutral"

    @pytest.mark.parametrize("regime", ["VOLATILE", "CHOPPY", "UNKNOWN"])
    def test_pa_neutral(self, regime):
        from strategies.price_action import PriceAction as PA
        long_bonus = _compute_regime_bonus(PA._REGIME_CONF_WITH_TREND, PA._REGIME_CONF_COUNTER_TREND, "LONG", regime)
        short_bonus = _compute_regime_bonus(PA._REGIME_CONF_WITH_TREND, PA._REGIME_CONF_COUNTER_TREND, "SHORT", regime)
        assert long_bonus == short_bonus

    @pytest.mark.parametrize("regime", ["VOLATILE"])
    def test_ichi_neutral(self, regime):
        from strategies.ichimoku import Ichimoku as IC
        long_bonus = _compute_regime_bonus(IC._REGIME_CONF_WITH_TREND, IC._REGIME_CONF_COUNTER_TREND, "LONG", regime)
        short_bonus = _compute_regime_bonus(IC._REGIME_CONF_WITH_TREND, IC._REGIME_CONF_COUNTER_TREND, "SHORT", regime)
        assert long_bonus == short_bonus


# ═══════════════════════════════════════════════════════════════════
# 6. Strategies that SHOULD NOT have direction-aware regime confidence
# ═══════════════════════════════════════════════════════════════════

class TestUnconstrainedStrategies:
    """
    Some strategies are intentionally direction-neutral (reversal, mean reversion,
    range scalper). They should NOT have direction-aware regime maps.
    """

    def test_reversal_no_regime_dir(self):
        from strategies.reversal import ReversalStrategy
        assert not hasattr(ReversalStrategy, '_REGIME_CONF_WITH_TREND')
        assert not hasattr(ReversalStrategy, '_REGIME_CONF_COUNTER_TREND')

    def test_mean_reversion_no_regime_dir(self):
        from strategies.mean_reversion import MeanReversion
        assert not hasattr(MeanReversion, '_REGIME_CONF_WITH_TREND')
        assert not hasattr(MeanReversion, '_REGIME_CONF_COUNTER_TREND')

    def test_range_scalper_no_regime_dir(self):
        from strategies.range_scalper import RangeScalperStrategy
        assert not hasattr(RangeScalperStrategy, '_REGIME_CONF_WITH_TREND')
        assert not hasattr(RangeScalperStrategy, '_REGIME_CONF_COUNTER_TREND')

    def test_wyckoff_no_regime_dir(self):
        """Wyckoff accumulation/distribution is phase-based, not trend-following."""
        from strategies.wyckoff import WyckoffAccDist
        assert not hasattr(WyckoffAccDist, '_REGIME_CONF_WITH_TREND')
        assert not hasattr(WyckoffAccDist, '_REGIME_CONF_COUNTER_TREND')

    def test_funding_arb_no_regime_dir(self):
        """Funding rate arbitrage is regime-independent."""
        from strategies.funding_arb import FundingRateArb
        assert not hasattr(FundingRateArb, '_REGIME_CONF_WITH_TREND')
        assert not hasattr(FundingRateArb, '_REGIME_CONF_COUNTER_TREND')


# ═══════════════════════════════════════════════════════════════════
# 7. Existing hard-constraint strategies still work
# ═══════════════════════════════════════════════════════════════════

class TestExistingHardConstraints:
    """Momentum and Breakout still have their hard direction constraints."""

    def test_momentum_has_regime_dir_constraint(self):
        from strategies.momentum import Momentum
        assert hasattr(Momentum, '_REGIME_DIR_CONSTRAINT')
        assert Momentum._REGIME_DIR_CONSTRAINT["BULL_TREND"] == "LONG"
        assert Momentum._REGIME_DIR_CONSTRAINT["BEAR_TREND"] == "SHORT"

    def test_breakout_has_regime_preferred_dir(self):
        from strategies.breakout import BreakoutStrategy
        assert hasattr(BreakoutStrategy, '_REGIME_PREFERRED_DIR')
        assert BreakoutStrategy._REGIME_PREFERRED_DIR["BULL_TREND"] == "LONG"
        assert BreakoutStrategy._REGIME_PREFERRED_DIR["BEAR_TREND"] == "SHORT"


# ═══════════════════════════════════════════════════════════════════
# 8. Production scenario: SMC LONG in BEAR_TREND confidence impact
# ═══════════════════════════════════════════════════════════════════

class TestProductionScenario:
    """
    Reproduce the production issue: SMC fires LONG in BEAR_TREND.
    Before fix: confidence_base(75) + regime(+8) + killzone(+10) = 93
    After fix:  confidence_base(75) + regime(-12) + killzone(+10) = 73
    Delta: -20 points — enough to push marginal signals below AGG_THRESHOLD.
    """

    def test_smc_long_in_bear_confidence_drop(self):
        """With-trend gets +8, counter-trend gets -12 → 20-point swing."""
        from strategies.smc import SmartMoneyConcepts as S
        old_bonus = 8   # was flat _REGIME_CONFIDENCE_DELTA[BEAR_TREND]
        new_bonus = S._REGIME_CONF_COUNTER_TREND["BEAR_TREND"]  # -12
        delta = new_bonus - old_bonus
        assert delta == -20, f"Expected 20-point drop, got {delta}"

    def test_smc_short_in_bear_confidence_maintained(self):
        """With-trend SHORT in BEAR_TREND still gets +8 — no regression."""
        from strategies.smc import SmartMoneyConcepts as S
        new_bonus = S._REGIME_CONF_WITH_TREND["BEAR_TREND"]
        assert new_bonus == 8, "With-trend bonus should be unchanged"

    def test_smc_production_counter_trend_long_below_threshold(self):
        """
        Simulate: confidence_base=75, killzone=10, regime=-12
        Result: 73 — below typical AGG_THRESHOLD (~75) → killed correctly.
        """
        confidence_base = 75
        killzone = 10
        from strategies.smc import SmartMoneyConcepts as S
        regime_bonus = S._REGIME_CONF_COUNTER_TREND["BEAR_TREND"]
        total = confidence_base + killzone + regime_bonus
        assert total == 73
        assert total < 75, "Counter-trend LONG should be below AGG_THRESHOLD"

    def test_smc_production_with_trend_short_above_threshold(self):
        """
        Simulate: confidence_base=75, killzone=10, regime=+8
        Result: 93 — well above threshold → passes correctly.
        """
        confidence_base = 75
        killzone = 10
        from strategies.smc import SmartMoneyConcepts as S
        regime_bonus = S._REGIME_CONF_WITH_TREND["BEAR_TREND"]
        total = confidence_base + killzone + regime_bonus
        assert total == 93
        assert total > 75, "With-trend SHORT should pass easily"
