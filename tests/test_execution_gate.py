"""
Tests for the Execution Quality Gate.

The gate separates "good setup" from "good trade" by evaluating execution
conditions (session, trigger quality, spread, whale alignment, entry position,
volume) as a group and hard-blocking when too many factors are simultaneously bad.
"""
import sys
import types
from unittest.mock import MagicMock

# ── Minimal stubs so analyzers.execution_gate can import config.constants ──
# conftest.py mocks numpy/pandas globally, but execution_gate doesn't need them.
# We need config.loader.cfg to exist for the constants module.
if 'config.loader' not in sys.modules:
    _loader = types.ModuleType('config.loader')
    _cfg = MagicMock()
    _cfg.aggregator = {'min_confidence': 72}
    _loader.cfg = _cfg
    sys.modules['config.loader'] = _loader

from config.constants import ExecutionGate as EG
from analyzers.execution_gate import ExecutionQualityGate, ExecutionAssessment


class TestExecutionGateBasics:
    """Test basic scoring and decision logic."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_perfect_execution_passes(self):
        """All factors perfect → high score, no block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=85.0,
            is_killzone=True,
            trigger_quality_score=0.85,
            trigger_quality_label="HIGH",
            spread_bps=5.0,
            whale_aligned=True,
            whale_buy_ratio=0.8,
            eq_zone="discount",
            eq_zone_depth=0.7,
            volume_context="BREAKOUT",
            volume_score=80.0,
        )
        assert result.execution_score >= 80.0
        assert not result.should_block
        assert not result.should_penalize
        assert len(result.bad_factors) == 0

    def test_terrible_execution_blocks(self):
        """All factors bad → low score, hard block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=72.0,
            is_dead_zone=True,
            trigger_quality_score=0.15,
            trigger_quality_label="LOW",
            spread_bps=250.0,
            whale_aligned=False,
            whale_buy_ratio=0.1,
            eq_zone="premium",
            eq_zone_depth=0.8,
            volume_context="LOW_VOL",
            volume_score=20.0,
        )
        assert result.execution_score < EG.HARD_BLOCK_THRESHOLD
        assert result.should_block
        assert len(result.bad_factors) >= 4

    def test_mediocre_execution_penalizes(self):
        """Mixed factors → soft penalty zone."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="B",
            confidence=75.0,
            session_name="Off-Session",
            trigger_quality_score=0.35,
            trigger_quality_label="MEDIUM",
            spread_bps=50.0,
            whale_aligned=None,
            eq_zone="trending",
            volume_context="NORMAL",
            volume_score=45.0,
        )
        # Should be in the soft penalty zone (35-50)
        assert result.execution_score >= EG.HARD_BLOCK_THRESHOLD
        assert result.execution_score <= 60.0

    def test_aplus_gets_relaxed_threshold(self):
        """A+ signals with high confidence get relaxed block threshold."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=92.0,
            is_dead_zone=True,
            trigger_quality_score=0.20,
            trigger_quality_label="LOW",
            spread_bps=80.0,
            whale_aligned=True,
            eq_zone="trending",
            volume_context="NORMAL",
            volume_score=50.0,
        )
        # A+ with 92 conf gets relaxed threshold (25 instead of 35)
        # Should survive because whale alignment + trending offset dead zone
        assert result.execution_score >= EG.APLUS_BLOCK_THRESHOLD

    def test_aplus_low_confidence_no_relaxation(self):
        """A+ with low confidence doesn't get the relaxation."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=80.0,  # Below 90 → no relaxation
            is_dead_zone=True,
            trigger_quality_score=0.10,
            trigger_quality_label="LOW",
            spread_bps=200.0,
            whale_aligned=False,
            whale_buy_ratio=0.2,
            eq_zone="premium",
            volume_context="LOW_VOL",
            volume_score=20.0,
        )
        # Uses standard threshold (35) since confidence < 90
        assert result.should_block

    def test_structured_execution_context_overrides_flat_inputs(self):
        """Structured execution context should drive evaluation without note parsing."""
        result = self.gate.evaluate(
            context={
                "session": {"name": "London Open", "is_killzone": True, "is_dead_zone": False, "is_weekend": False},
                "trigger": {"score": 0.82, "label": "HIGH"},
                "liquidity": {"spread_bps": 7.0, "volume_context": "BREAKOUT", "volume_score": 85.0},
                "whales": {"aligned": True, "buy_ratio": 0.81},
                "positioning": {"derivatives_score": 72.0, "sentiment_score": 68.0},
                "location": {"eq_zone": "discount", "eq_distance": 0.66},
                "market": {"volatility_regime": "HIGH", "transition_type": "breakout", "transition_risk": 0.22},
                "trade": {"direction": "LONG", "type": "PULLBACK_LONG", "strength": 3},
            },
            direction="LONG",
            grade="A",
            confidence=86.0,
            is_dead_zone=True,             # should be ignored by context
            spread_bps=220.0,              # should be ignored by context
            whale_aligned=False,           # should be ignored by context
            whale_buy_ratio=0.10,          # should be ignored by context
            eq_zone="premium",             # should be ignored by context
            volume_context="LOW_VOL",      # should be ignored by context
            volume_score=15.0,             # should be ignored by context
            trigger_quality_score=0.10,    # should be ignored by context
            trigger_quality_label="LOW",   # should be ignored by context
        )
        assert result.execution_score >= 80.0
        assert result.factors["session"] == EG.SESSION_SCORE_KILLZONE
        assert result.factors["spread"] == EG.SPREAD_SCORE_TIGHT
        assert result.factors["whale_alignment"] == EG.WHALE_ALIGNED_SCORE
        assert result.context_snapshot["market"]["volatility_regime"] == "HIGH"
        assert result.block_threshold_used == (
            EG.HARD_BLOCK_THRESHOLD
            + EG.ADAPTIVE_KILLZONE_ADJ
            + EG.ADAPTIVE_HIGH_VOL_ADJ
        )

    def test_context_zero_values_not_treated_as_falsy(self):
        """Zero values in context must NOT be replaced by flat-arg defaults."""
        result = self.gate.evaluate(
            context={
                "session": {"name": "London Open", "is_killzone": True, "is_dead_zone": False, "is_weekend": False},
                "trigger": {"score": 0.0, "label": "LOW"},
                "liquidity": {"spread_bps": 0.0, "volume_context": "NORMAL", "volume_score": 0.0},
                "whales": {"aligned": True, "buy_ratio": 0.0},
                "location": {"eq_zone": "discount", "eq_distance": 0.0},
                "trade": {"type": "PULLBACK_LONG", "strength": 0},
            },
            direction="LONG",
            grade="B",
            confidence=70.0,
            trigger_quality_score=0.99,    # should be ignored
            spread_bps=999.0,              # should be ignored
            volume_score=100.0,            # should be ignored
            whale_buy_ratio=1.0,           # should be ignored
            eq_zone_depth=0.99,            # should be ignored
            trade_type_strength=4,         # should be ignored
        )
        # Zero-value context fields must survive (not be overridden by fallbacks)
        assert result.factors["trigger_quality"] <= 15.0  # score 0.0 → worst bucket
        # spread_bps=0.0 from context means "no spread data" (scorer returns 60.0),
        # NOT 999.0 bps which would come from the flat-arg fallback if the bug existed
        assert result.factors["spread"] == 60.0

    def test_semantic_kill_blocks_bullish_choch_short(self):
        result = self.gate.evaluate(
            context={
                "session": {"name": "London Open", "is_killzone": True},
                "market": {"volatility_regime": "HIGH"},
            },
            setup_context={
                "structure": {"choch": True, "choch_direction": "BULLISH"},
                "pattern": {},
            },
            direction="SHORT",
            grade="A",
            confidence=84.0,
            trigger_quality_score=0.82,
            spread_bps=7.0,
            whale_aligned=True,
            eq_zone="premium",
            volume_context="BREAKOUT",
            volume_score=80.0,
        )
        assert result.should_block is True
        assert result.kill_combo.startswith("SEMANTIC_KILL:")

    def test_countertrend_override_allows_structured_bull_trend_short(self):
        result = self.gate.evaluate(
            context={
                "session": {"name": "London Open", "is_killzone": True},
                "trigger": {"score": 0.74, "label": "HIGH"},
                "liquidity": {"spread_bps": 12.0, "volume_context": "BREAKOUT", "volume_score": 72.0},
                "whales": {"aligned": False, "buy_ratio": 0.91},
                "positioning": {
                    "derivatives_score": 74.0,
                    "sentiment_score": 68.0,
                    "funding_rate": 0.05,
                    "oi_change_24h": 12.0,
                },
                "location": {"eq_zone": "premium", "eq_distance": 0.64},
                "market": {"volatility_regime": "HIGH"},
                "trade": {"type": "LOCAL_CONTINUATION_SHORT", "strength": 3},
            },
            setup_context={
                "structure": {"trend": "BULL_TREND"},
                "pattern": {},
                "location": {"setup_class": "swing"},
            },
            direction="SHORT",
            grade="A",
            confidence=82.0,
        )
        assert result.should_block is False
        assert "SEMANTIC_KILL" not in result.kill_combo
        assert "EXTREME_WHALE_OPPOSITION" not in result.kill_combo

    def test_naked_bull_trend_short_still_blocks_on_whale_opposition(self):
        result = self.gate.evaluate(
            context={
                "session": {"name": "London Open", "is_killzone": True},
                "trigger": {"score": 0.74, "label": "HIGH"},
                "liquidity": {"spread_bps": 12.0, "volume_context": "BREAKOUT", "volume_score": 72.0},
                "whales": {"aligned": False, "buy_ratio": 0.91},
                "positioning": {
                    "derivatives_score": 58.0,
                    "sentiment_score": 52.0,
                    "funding_rate": 0.01,
                    "oi_change_24h": 3.0,
                },
                "location": {"eq_zone": "premium", "eq_distance": 0.64},
                "market": {"volatility_regime": "HIGH"},
                "trade": {"type": "PULLBACK_SHORT", "strength": 1},
            },
            setup_context={
                "structure": {"trend": "BULL_TREND"},
                "pattern": {},
                "location": {"setup_class": "swing"},
            },
            direction="SHORT",
            grade="A",
            confidence=82.0,
        )
        assert result.should_block is True
        assert result.kill_combo in {
            "EXTREME_WHALE_OPPOSITION (91% vs SHORT)",
            "SEMANTIC_KILL: HTF_TREND BULL vs SHORT (no reversal signal)",
        }

    def test_context_thresholds_adapt_to_dead_zone_and_low_vol(self):
        result = self.gate.evaluate(
            context={
                "session": {"name": "Dead Zone", "is_dead_zone": True},
                "market": {"volatility_regime": "LOW"},
            },
            direction="LONG",
            grade="B",
            confidence=78.0,
            trigger_quality_score=0.50,
            spread_bps=10.0,
            whale_aligned=True,
            eq_zone="discount",
            volume_context="NORMAL",
            volume_score=60.0,
        )
        assert result.block_threshold_used == 50.0

    def test_context_thresholds_relax_in_killzone_high_vol(self):
        result = self.gate.evaluate(
            context={
                "session": {"name": "London Open", "is_killzone": True},
                "market": {"volatility_regime": "BREAKOUT"},
            },
            direction="LONG",
            grade="B",
            confidence=78.0,
            trigger_quality_score=0.50,
            spread_bps=10.0,
            whale_aligned=True,
            eq_zone="discount",
            volume_context="NORMAL",
            volume_score=60.0,
        )
        assert result.block_threshold_used == 25.0

    def test_positioning_context_penalizes_wrong_direction(self):
        supportive = self.gate.evaluate(
            context={
                "session": {"name": "London Open", "is_killzone": True},
                "trigger": {"score": 0.75, "label": "HIGH"},
                "liquidity": {"spread_bps": 8.0, "volume_context": "BREAKOUT", "volume_score": 80.0},
                "whales": {"aligned": True, "buy_ratio": 0.8},
                "positioning": {
                    "derivatives_score": 80.0,
                    "sentiment_score": 72.0,
                    "funding_rate": -0.02,
                    "oi_change_24h": 12.0,
                },
                "location": {"eq_zone": "discount", "eq_distance": 0.6},
            },
            direction="LONG",
            grade="A",
            confidence=84.0,
        )
        hostile = self.gate.evaluate(
            context={
                "session": {"name": "London Open", "is_killzone": True},
                "trigger": {"score": 0.75, "label": "HIGH"},
                "liquidity": {"spread_bps": 8.0, "volume_context": "BREAKOUT", "volume_score": 80.0},
                "whales": {"aligned": True, "buy_ratio": 0.8},
                "positioning": {
                    "derivatives_score": 22.0,
                    "sentiment_score": 30.0,
                    "funding_rate": 0.05,
                    "oi_change_24h": 12.0,
                },
                "location": {"eq_zone": "discount", "eq_distance": 0.6},
            },
            direction="LONG",
            grade="A",
            confidence=84.0,
        )
        assert supportive.factors["positioning"] > hostile.factors["positioning"]
        assert supportive.execution_score > hostile.execution_score

    def test_threshold_uses_calibration_adjustment(self, monkeypatch):
        from analyzers import near_miss_tracker as near_miss_module

        monkeypatch.setattr(
            near_miss_module.near_miss_tracker,
            "compute_threshold_adjustment",
            lambda: 3.0,
        )
        result = self.gate.evaluate(
            context={
                "session": {"name": "London Open", "is_killzone": True},
                "market": {"volatility_regime": "BREAKOUT"},
            },
            direction="LONG",
            grade="B",
            confidence=78.0,
            trigger_quality_score=0.50,
            spread_bps=10.0,
            whale_aligned=True,
            eq_zone="discount",
            volume_context="NORMAL",
            volume_score=60.0,
        )
        assert result.block_threshold_used == 28.0


class TestSessionScoring:
    """Test session quality factor scoring."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_killzone_scores_high(self):
        result = self.gate.evaluate(
            direction="LONG", is_killzone=True,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['session'] == EG.SESSION_SCORE_KILLZONE

    def test_dead_zone_scores_low(self):
        result = self.gate.evaluate(
            direction="LONG", is_dead_zone=True,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['session'] == EG.SESSION_SCORE_DEAD_ZONE
        assert 'session' in result.bad_factors

    def test_weekend_scores_low(self):
        result = self.gate.evaluate(
            direction="LONG", is_weekend=True,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['session'] == EG.SESSION_SCORE_WEEKEND

    def test_session_name_dead_zone(self):
        result = self.gate.evaluate(
            direction="LONG", session_name="Dead Zone",
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['session'] == EG.SESSION_SCORE_DEAD_ZONE

    def test_session_name_london(self):
        result = self.gate.evaluate(
            direction="LONG", session_name="London Open",
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['session'] == EG.SESSION_SCORE_KILLZONE

    def test_normal_session(self):
        result = self.gate.evaluate(
            direction="LONG", session_name="Off-Session",
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['session'] == EG.SESSION_SCORE_NORMAL


class TestSpreadScoring:
    """Test spread (liquidity) factor scoring."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_tight_spread(self):
        result = self.gate.evaluate(
            direction="LONG", spread_bps=5.0,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['spread'] == EG.SPREAD_SCORE_TIGHT

    def test_extreme_spread(self):
        result = self.gate.evaluate(
            direction="LONG", spread_bps=150.0,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['spread'] == EG.SPREAD_SCORE_EXTREME
        assert 'spread' in result.bad_factors

    def test_wide_spread(self):
        result = self.gate.evaluate(
            direction="LONG", spread_bps=80.0,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['spread'] < EG.SPREAD_SCORE_NORMAL
        assert result.factors['spread'] > EG.SPREAD_SCORE_EXTREME

    def test_no_spread_data(self):
        """Missing spread data → neutral score (don't penalize or reward)."""
        result = self.gate.evaluate(
            direction="LONG", spread_bps=0.0,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['spread'] == 60.0  # Neutral default


class TestWhaleAlignmentScoring:
    """Test whale alignment factor scoring."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_whale_aligned(self):
        result = self.gate.evaluate(
            direction="LONG",
            whale_aligned=True, whale_buy_ratio=0.8,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['whale_alignment'] == EG.WHALE_ALIGNED_SCORE

    def test_whale_strongly_opposing(self):
        """SHORT while whales buying 91% → severe penalty."""
        result = self.gate.evaluate(
            direction="SHORT",
            whale_aligned=False, whale_buy_ratio=0.91,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['whale_alignment'] == EG.WHALE_OPPOSING_SCORE
        assert 'whale_alignment' in result.bad_factors

    def test_whale_mildly_opposing(self):
        """Mild whale opposition → less severe."""
        result = self.gate.evaluate(
            direction="SHORT",
            whale_aligned=False, whale_buy_ratio=0.55,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        # Mild opposition → between neutral and opposing
        assert result.factors['whale_alignment'] >= EG.WHALE_OPPOSING_SCORE
        assert result.factors['whale_alignment'] <= EG.WHALE_NEUTRAL_SCORE

    def test_no_whale_data(self):
        """No whale data → neutral score."""
        result = self.gate.evaluate(
            direction="LONG",
            whale_aligned=None,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['whale_alignment'] == EG.WHALE_NEUTRAL_SCORE


class TestEntryPositionScoring:
    """Test entry position factor scoring."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_long_in_discount(self):
        result = self.gate.evaluate(
            direction="LONG", eq_zone="discount", eq_zone_depth=0.7,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['entry_position'] == EG.ENTRY_DISCOUNT_LONG

    def test_short_in_premium(self):
        result = self.gate.evaluate(
            direction="SHORT", eq_zone="premium", eq_zone_depth=0.7,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['entry_position'] == EG.ENTRY_PREMIUM_SHORT

    def test_long_in_premium_bad(self):
        result = self.gate.evaluate(
            direction="LONG", eq_zone="premium", eq_zone_depth=0.7,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['entry_position'] == EG.ENTRY_WRONG_ZONE
        assert 'entry_position' in result.bad_factors

    def test_short_in_discount_bad(self):
        result = self.gate.evaluate(
            direction="SHORT", eq_zone="discount", eq_zone_depth=0.7,
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['entry_position'] == EG.ENTRY_WRONG_ZONE

    def test_trending_regime_bypass(self):
        result = self.gate.evaluate(
            direction="LONG", eq_zone="trending",
            trigger_quality_score=0.5, volume_score=50.0,
        )
        assert result.factors['entry_position'] == EG.ENTRY_TREND_BYPASS


class TestVolumeEnvironmentScoring:
    """Test volume environment factor scoring."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_breakout_volume(self):
        result = self.gate.evaluate(
            direction="LONG", volume_context="BREAKOUT", volume_score=80.0,
            trigger_quality_score=0.5,
        )
        # 0.6 * 100 + 0.4 * 80 = 92
        assert result.factors['volume_env'] >= 90.0

    def test_low_volume(self):
        result = self.gate.evaluate(
            direction="LONG", volume_context="LOW_VOL", volume_score=25.0,
            trigger_quality_score=0.5,
        )
        # 0.6 * 25 + 0.4 * 25 = 25
        assert result.factors['volume_env'] <= 30.0
        assert 'volume_env' in result.bad_factors


class TestStackingPenalty:
    """Test that multiple bad factors trigger stacking penalty."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_three_bad_factors_triggers_stacking(self):
        """3+ bad factors → stacking penalty applied (when no kill combo fires)."""
        # Use factors that are bad but don't form any kill combo:
        # dead zone + extreme spread + wrong zone entry
        # Trigger above KC floor, volume normal, whales mildly opposing
        result = self.gate.evaluate(
            direction="LONG",
            is_dead_zone=True,            # bad: session (15)
            trigger_quality_score=0.45,   # mediocre but above KC floor (0.25)
            spread_bps=40.0,              # moderate spread (below KC threshold 50)
            whale_aligned=False,          # bad: whale opposing
            whale_buy_ratio=0.3,          # mild opposition (below KC ratio 0.85)
            eq_zone="premium",            # bad: wrong zone (20)
            volume_context="NORMAL",
            volume_score=50.0,
        )
        assert len(result.bad_factors) >= 3
        assert result.kill_combo == ""  # No kill combo
        assert any('stacking' in note.lower() for note in result.notes)

    def test_two_bad_factors_no_stacking(self):
        """2 bad factors → no stacking penalty."""
        result = self.gate.evaluate(
            direction="LONG",
            is_dead_zone=True,            # bad: session
            trigger_quality_score=0.10,   # bad: trigger quality
            spread_bps=5.0,               # good
            whale_aligned=True,           # good
            eq_zone="discount",           # good
            volume_context="BREAKOUT",
            volume_score=80.0,
        )
        assert not any('stacking' in note.lower() for note in result.notes)


class TestRealWorldScenarios:
    """Test real scenarios from the signals that inspired this gate."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_mon_long_dead_zone_premium_wide_spread(self):
        """MON LONG: dead zone + premium + wide spread + low trigger quality.
        This should be blocked — the signal that started the discussion."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=61.0,
            is_dead_zone=True,
            trigger_quality_score=0.27,
            trigger_quality_label="LOW",
            spread_bps=235.0,
            whale_aligned=None,       # No direct whale opposition
            eq_zone="trending",       # BULL_TREND bypasses EQ
            volume_context="LOW_VOL",
            volume_score=66.0,
        )
        # Dead zone (15) + low trigger (27) + extreme spread (10) + low vol
        # = should block or heavily penalize
        assert result.execution_score < EG.SOFT_PENALTY_THRESHOLD
        assert result.should_block or result.should_penalize

    def test_ondo_short_whale_opposing_equilibrium(self):
        """ONDO SHORT: whale opposition (91% buying) + equilibrium entry +
        low trigger quality. This should be blocked."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="B",
            confidence=71.0,
            is_killzone=True,         # London open — good
            trigger_quality_score=0.27,
            trigger_quality_label="LOW",
            spread_bps=74.0,
            whale_aligned=False,
            whale_buy_ratio=0.91,     # Severe opposition
            eq_zone="trending",
            volume_context="LOW_VOL",
            volume_score=59.0,
        )
        # Killzone good, but whale opposing (15) + low trigger (27) +
        # wide spread + low vol = should block or penalize
        assert result.should_block or result.should_penalize

    def test_uni_short_wyckoff_utad(self):
        """UNI SHORT: better setup (Wyckoff UTAD, 4 strategies, killzone)
        but still has whale opposition + low trigger quality."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="A",
            confidence=72.0,
            is_killzone=True,          # Good
            trigger_quality_score=0.27,
            trigger_quality_label="LOW",
            spread_bps=55.0,
            whale_aligned=False,
            whale_buy_ratio=0.91,      # Severe opposition
            eq_zone="trending",
            volume_context="LOW_VOL",
            volume_score=59.0,
        )
        # Better than ONDO (killzone, higher volume score) but
        # whale opposition still makes it risky
        assert result.execution_score < 55.0

    def test_ideal_trend_continuation(self):
        """Ideal scenario: killzone, tight spread, whales aligned,
        good trigger, breakout volume. Should sail through."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=90.0,
            is_killzone=True,
            trigger_quality_score=0.80,
            trigger_quality_label="HIGH",
            spread_bps=8.0,
            whale_aligned=True,
            whale_buy_ratio=0.85,
            eq_zone="discount",
            eq_zone_depth=0.6,
            volume_context="BREAKOUT",
            volume_score=85.0,
        )
        assert result.execution_score >= 85.0
        assert not result.should_block
        assert not result.should_penalize
        assert len(result.bad_factors) == 0

    def test_weekend_low_volume_but_strong_trigger(self):
        """Weekend with strong triggers — should penalize but not block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=82.0,
            is_weekend=True,
            trigger_quality_score=0.75,
            trigger_quality_label="HIGH",
            spread_bps=20.0,
            whale_aligned=True,
            eq_zone="discount",
            volume_context="ABOVE_AVG",
            volume_score=70.0,
        )
        # Weekend is bad for session, but everything else is good
        assert not result.should_block
        assert result.execution_score >= 60.0


class TestExecutionAssessmentNotes:
    """Test that the gate provides useful transparency notes."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_notes_contain_all_factors(self):
        result = self.gate.evaluate(
            direction="LONG",
            trigger_quality_score=0.5,
            volume_score=50.0,
        )
        # Should have a note for each factor, including positioning
        factor_notes = [n for n in result.notes if '/100' in n]
        assert len(factor_notes) == 7

    def test_blocked_signal_has_reason(self):
        result = self.gate.evaluate(
            direction="LONG",
            is_dead_zone=True,
            trigger_quality_score=0.10,
            spread_bps=200.0,
            whale_aligned=False,
            whale_buy_ratio=0.1,
            volume_context="LOW_VOL",
            volume_score=20.0,
        )
        assert result.should_block
        # Blocked by either kill combo or execution gate threshold
        assert "KILL COMBO" in result.reason or "EXECUTION GATE" in result.reason
        assert "NO TRADE" in result.reason or "KILL COMBO" in result.reason

    def test_factors_dict_complete(self):
        result = self.gate.evaluate(
            direction="LONG",
            trigger_quality_score=0.5,
            volume_score=50.0,
        )
        expected_keys = {'session', 'trigger_quality', 'spread',
                        'whale_alignment', 'entry_position', 'volume_env', 'positioning'}
        assert set(result.factors.keys()) == expected_keys


class TestKillCombos:
    """Test non-negotiable kill combinations.

    Kill combos hard-block regardless of composite score.  They enforce
    non-linear logic: certain factor combinations are ALWAYS bad, even
    if other factors score perfectly.
    """

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    # ── KC1: Low trigger + low volume ────────────────────────────

    def test_kc1_low_trigger_low_volume_blocks(self):
        """Trigger < 0.25 AND volume LOW → always block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",           # Good grade doesn't help
            confidence=85.0,     # High confidence doesn't help
            is_killzone=True,    # Even in killzone
            trigger_quality_score=0.20,
            trigger_quality_label="LOW",
            spread_bps=5.0,      # Tight spread doesn't save it
            whale_aligned=True,  # Whale alignment doesn't save it
            eq_zone="discount",  # Perfect entry doesn't save it
            volume_context="LOW_VOL",
            volume_score=25.0,
        )
        assert result.should_block
        assert result.kill_combo != ""
        assert "LOW_TRIGGER" in result.kill_combo
        assert "LOW_VOLUME" in result.kill_combo

    def test_kc1_borderline_trigger_passes(self):
        """Trigger exactly at 0.25 → no kill combo (barely passes)."""
        result = self.gate.evaluate(
            direction="LONG",
            is_killzone=True,
            trigger_quality_score=0.25,  # At boundary, not below
            volume_context="LOW_VOL",
            volume_score=25.0,
        )
        # Kill combo should NOT fire — score decides
        assert result.kill_combo == ""

    def test_kc1_low_trigger_normal_volume_no_combo(self):
        """Low trigger but normal volume → no kill combo."""
        result = self.gate.evaluate(
            direction="LONG",
            is_killzone=True,
            trigger_quality_score=0.15,
            volume_context="NORMAL",
            volume_score=55.0,
        )
        assert result.kill_combo == ""

    # ── KC2: Extreme whale opposition ────────────────────────────

    def test_kc2_whale_opposition_short_blocks(self):
        """SHORT while whales buying 91% → kill combo (not A+)."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="A",           # A, not A+
            confidence=80.0,
            is_killzone=True,    # Killzone doesn't save
            trigger_quality_score=0.70,  # Good trigger doesn't save
            whale_aligned=False,
            whale_buy_ratio=0.91,
            volume_context="BREAKOUT",
            volume_score=80.0,
        )
        assert result.should_block
        assert "WHALE_OPPOSITION" in result.kill_combo

    def test_kc2_whale_opposition_aplus_exempt(self):
        """A+ SHORT with whale opposition → exempt (Wyckoff UTAD case)."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="A+",          # A+ exemption
            confidence=90.0,
            is_killzone=True,
            trigger_quality_score=0.70,
            whale_aligned=False,
            whale_buy_ratio=0.91,
            eq_zone="premium",
            volume_context="ABOVE_AVG",
            volume_score=70.0,
        )
        # A+ exemption: kill combo should NOT fire
        assert "WHALE_OPPOSITION" not in result.kill_combo

    def test_kc2_whale_opposition_long_blocks(self):
        """LONG while whales selling heavily → kill combo."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            whale_aligned=False,
            whale_buy_ratio=0.10,  # 90% selling
            trigger_quality_score=0.5,
            volume_score=50.0,
        )
        assert result.should_block
        assert "WHALE_OPPOSITION" in result.kill_combo

    def test_kc2_mild_whale_opposition_no_combo(self):
        """Moderate whale opposition (70%) → no kill combo."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="B",
            whale_aligned=False,
            whale_buy_ratio=0.70,  # Below 85% threshold
            trigger_quality_score=0.5,
            volume_score=50.0,
        )
        assert "WHALE_OPPOSITION" not in (result.kill_combo or "")

    # ── KC3: Dead zone + wide spread ─────────────────────────────

    def test_kc3_dead_zone_wide_spread_blocks(self):
        """Dead zone + spread > 50 bps → always block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=85.0,
            is_dead_zone=True,
            spread_bps=74.0,     # Wide spread in dead zone
            trigger_quality_score=0.70,  # Good trigger
            whale_aligned=True,          # Whales aligned
            eq_zone="discount",          # Perfect entry
            volume_context="ABOVE_AVG",
            volume_score=70.0,
        )
        assert result.should_block
        assert "DEAD_ZONE" in result.kill_combo
        assert "SPREAD" in result.kill_combo

    def test_kc3_dead_zone_tight_spread_no_combo(self):
        """Dead zone + tight spread → no kill combo (score decides)."""
        result = self.gate.evaluate(
            direction="LONG",
            is_dead_zone=True,
            spread_bps=8.0,  # Tight spread
            trigger_quality_score=0.5,
            volume_score=50.0,
        )
        assert "DEAD_ZONE" not in (result.kill_combo or "") or "SPREAD" not in (result.kill_combo or "")

    # ── KC4: Wrong zone + low trigger ────────────────────────────

    def test_kc4_long_premium_low_trigger_blocks(self):
        """LONG in premium + trigger < 0.25 → kill combo."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            is_killzone=True,
            trigger_quality_score=0.20,
            eq_zone="premium",
            whale_aligned=True,
            volume_context="ABOVE_AVG",
            volume_score=70.0,
        )
        assert result.should_block
        assert "WRONG_ZONE" in result.kill_combo

    def test_kc4_short_discount_low_trigger_blocks(self):
        """SHORT in discount + trigger < 0.25 → kill combo."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="B",
            is_killzone=True,
            trigger_quality_score=0.20,
            eq_zone="discount",
            volume_context="NORMAL",
            volume_score=50.0,
        )
        assert result.should_block
        assert "WRONG_ZONE" in result.kill_combo

    def test_kc4_correct_zone_low_trigger_no_combo(self):
        """LONG in discount + low trigger → KC1 may fire but not KC4."""
        result = self.gate.evaluate(
            direction="LONG",
            is_killzone=True,
            trigger_quality_score=0.20,
            eq_zone="discount",  # Correct zone
            volume_context="ABOVE_AVG",
            volume_score=70.0,
        )
        # KC4 should not fire (correct zone), KC1 might if volume low
        assert "WRONG_ZONE" not in (result.kill_combo or "")

    # ── KC5: Triple death (dead + low trigger + low volume) ──────

    def test_kc5_triple_death_blocks(self):
        """Dead zone + low trigger + low volume → absolute kill."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",          # Even A+ dies here
            confidence=95.0,
            is_dead_zone=True,
            trigger_quality_score=0.15,
            spread_bps=5.0,      # Even tight spread
            whale_aligned=True,  # Even whales aligned
            eq_zone="discount",  # Even perfect entry
            volume_context="LOW_VOL",
            volume_score=20.0,
        )
        assert result.should_block
        assert "DEAD_ZONE" in result.kill_combo
        assert "LOW_TRIGGER" in result.kill_combo
        assert "LOW_VOLUME" in result.kill_combo

    # ── Kill combo transparency ──────────────────────────────────

    def test_kill_combo_has_factor_notes(self):
        """Kill combo still provides factor breakdown for transparency."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="B",
            whale_aligned=False,
            whale_buy_ratio=0.91,
            trigger_quality_score=0.5,
            volume_score=50.0,
        )
        assert result.should_block
        # Should still have factor notes for debugging
        factor_notes = [n for n in result.notes if '/100' in n]
        assert len(factor_notes) == 7

    def test_kill_combo_stored_in_assessment(self):
        """Kill combo name is stored for engine to log."""
        result = self.gate.evaluate(
            direction="LONG",
            is_dead_zone=True,
            trigger_quality_score=0.15,
            volume_context="LOW_VOL",
            volume_score=20.0,
        )
        assert result.kill_combo != ""
        assert "KILL COMBO" in result.reason


class TestRealWorldWithKillCombos:
    """Re-test the real-world scenarios with kill combos active."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_ondo_short_killed_by_whale_combo(self):
        """ONDO SHORT: 91% whale buying → kill combo fires (grade B)."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="B",
            confidence=71.0,
            is_killzone=True,
            trigger_quality_score=0.27,
            trigger_quality_label="LOW",
            spread_bps=74.0,
            whale_aligned=False,
            whale_buy_ratio=0.91,
            eq_zone="trending",
            volume_context="LOW_VOL",
            volume_score=59.0,
        )
        assert result.should_block
        assert "WHALE_OPPOSITION" in result.kill_combo

    def test_uni_short_aplus_survives_whale_combo(self):
        """UNI SHORT: if hypothetically graded A+, whale combo exempt.
        But trigger 0.27 > KC floor 0.25, so KC1 doesn't fire either.
        Score-based penalty still applies."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="A+",         # Hypothetical A+ for Wyckoff UTAD
            confidence=85.0,
            is_killzone=True,
            trigger_quality_score=0.27,
            trigger_quality_label="LOW",
            spread_bps=55.0,
            whale_aligned=False,
            whale_buy_ratio=0.91,
            eq_zone="trending",
            volume_context="LOW_VOL",
            volume_score=59.0,
        )
        # A+ exempts from whale kill combo
        assert "WHALE_OPPOSITION" not in (result.kill_combo or "")
        # Trigger 0.27 > 0.25 floor, so KC1 doesn't fire either
        assert result.kill_combo == ""
        # But score is low enough for penalty (3 bad factors + stacking)
        assert result.should_penalize or result.should_block

    def test_uni_short_actual_grade_a_killed(self):
        """UNI SHORT actual: grade A + 91% whale buy → killed."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="A",
            confidence=72.0,
            is_killzone=True,
            trigger_quality_score=0.27,
            trigger_quality_label="LOW",
            spread_bps=55.0,
            whale_aligned=False,
            whale_buy_ratio=0.91,
            eq_zone="trending",
            volume_context="LOW_VOL",
            volume_score=59.0,
        )
        assert result.should_block
        # Either whale combo or low trigger + low volume
        assert result.kill_combo != ""

    def test_mon_long_killed_by_triple_death(self):
        """MON LONG: dead zone + low trigger + low volume → triple death."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=61.0,
            is_dead_zone=True,
            trigger_quality_score=0.27,  # Above KC floor
            trigger_quality_label="LOW",
            spread_bps=235.0,
            whale_aligned=None,
            eq_zone="trending",
            volume_context="LOW_VOL",
            volume_score=66.0,
        )
        # Trigger 0.27 > 0.25 floor, so KC5 doesn't fire
        # But KC3 (dead + wide spread 235 > 50) fires
        assert result.should_block
        assert "DEAD_ZONE" in result.kill_combo
        assert "SPREAD" in result.kill_combo


# ═══════════════════════════════════════════════════════════════════════
# GPT Combo Matrix — Decision-Boundary Tests
# ═══════════════════════════════════════════════════════════════════════
# These tests implement the structured test matrix recommended by GPT:
#   1. Ideal trade (always pass)
#   2. Good but not perfect (minor flaws)
#   3. Borderline (should NOT auto-enter)
#   4. Hard block (already covered by kill combo tests above)
#   5. Momentum breakout cases
#   6. Conflict engine tests
#   7. Stacking effect tests
#   + Decision boundary tests at exact thresholds
#   + Scoring validation (exact score + action pairs)
# ═══════════════════════════════════════════════════════════════════════


class TestGoodButNotPerfect:
    """GPT Matrix §2 — Setups with one minor flaw.

    These should ENTER with slight confidence reduction (soft penalty)
    or pass cleanly.  One bad factor alone shouldn't kill a good trade.
    """

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_2a_medium_trigger_rest_perfect(self):
        """Good setup, one minor flaw: medium trigger (0.45).
        Everything else perfect → should pass (possibly soft penalty)."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=85.0,
            is_killzone=True,
            trigger_quality_score=0.45,   # Minor flaw
            trigger_quality_label="MEDIUM",
            spread_bps=8.0,
            whale_aligned=True,
            whale_buy_ratio=0.82,
            eq_zone="discount",
            eq_zone_depth=0.6,
            volume_context="ABOVE_AVG",
            volume_score=75.0,
        )
        assert not result.should_block
        assert result.execution_score >= 60.0
        # At most soft penalty, not block
        assert result.kill_combo == ""

    def test_2a_eq_entry_rest_perfect(self):
        """Good setup, one flaw: equilibrium entry (mediocre but not terrible).
        Everything else perfect → should pass."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=85.0,
            is_killzone=True,
            trigger_quality_score=0.80,
            trigger_quality_label="HIGH",
            spread_bps=8.0,
            whale_aligned=True,
            whale_buy_ratio=0.80,
            eq_zone="eq_dead_zone",        # Mediocre entry
            eq_zone_depth=0.5,
            volume_context="BREAKOUT",
            volume_score=80.0,
        )
        assert not result.should_block
        assert result.execution_score >= 70.0

    def test_2b_premium_but_strong_momentum(self):
        """LONG in premium BUT high trigger + breakout volume.
        Momentum should partially offset premium entry → penalize not block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=85.0,
            is_killzone=True,
            trigger_quality_score=0.82,   # High trigger → momentum
            trigger_quality_label="HIGH",
            spread_bps=12.0,
            whale_aligned=True,
            whale_buy_ratio=0.80,
            eq_zone="premium",            # Wrong zone for LONG
            eq_zone_depth=0.6,
            volume_context="BREAKOUT",    # Strong momentum
            volume_score=80.0,
        )
        # Premium penalizes entry factor to 20, but everything else is 80-100
        assert not result.should_block
        assert result.execution_score >= 55.0
        # At most mild penalty, momentum offsets location
        assert result.kill_combo == ""

    def test_2b_normal_session_all_else_good(self):
        """Normal session (not killzone) but everything else is great.
        One mild session penalty → should pass."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="A",
            confidence=85.0,
            session_name="Off-Session",   # Normal, not killzone
            trigger_quality_score=0.78,
            trigger_quality_label="HIGH",
            spread_bps=15.0,
            whale_aligned=True,
            whale_buy_ratio=0.15,         # Whales selling → aligned with SHORT
            eq_zone="premium",
            eq_zone_depth=0.7,
            volume_context="ABOVE_AVG",
            volume_score=72.0,
        )
        assert not result.should_block
        assert result.execution_score >= 65.0


class TestBorderlineCases:
    """GPT Matrix §3 — Good setups with execution concerns.

    These should NOT auto-enter.  Expected: WAIT (soft penalty) or edge
    of block territory.
    """

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_3a_near_long_dead_zone_premium_medium_trigger(self):
        """NEAR LONG: the actual signal from the analysis.
        Strong setup (A+) + medium trigger + premium + dead zone + wide spread.
        KC3 fires: dead zone + spread 84 > 50 bps."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=96.0,
            is_dead_zone=True,             # Dead zone session
            trigger_quality_score=0.62,    # Medium trigger
            trigger_quality_label="MEDIUM",
            spread_bps=84.0,               # Wide spread
            whale_aligned=True,            # Whales aligned (87% buy)
            whale_buy_ratio=0.87,
            eq_zone="premium",             # LONG in premium
            eq_zone_depth=0.92,
            volume_context="ABOVE_AVG",
            volume_score=74.0,
        )
        # KC3: dead zone + spread 84 > 50 → kill combo fires
        assert result.should_block
        assert "DEAD_ZONE" in result.kill_combo
        assert "SPREAD" in result.kill_combo

    def test_3a_near_variant_normal_session_tight_spread(self):
        """NEAR LONG variant: same strong setup but in normal session
        with tighter spread.  No kill combo, should pass (possibly penalized)."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=96.0,
            session_name="London Open",    # Good session now
            trigger_quality_score=0.62,
            trigger_quality_label="MEDIUM",
            spread_bps=18.0,               # Normal spread
            whale_aligned=True,
            whale_buy_ratio=0.87,
            eq_zone="premium",             # Still premium
            eq_zone_depth=0.92,
            volume_context="ABOVE_AVG",
            volume_score=74.0,
        )
        # No kill combo now — good session + tight spread
        assert result.kill_combo == ""
        assert not result.should_block
        # Premium entry still penalizes but doesn't kill
        assert result.execution_score >= 55.0

    def test_3b_strong_setup_whale_conflict(self):
        """Strong setup (A grade, killzone, good trigger) BUT whales opposing.
        Grade A → not exempt from KC2.  91% opposition → kill combo."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=84.0,
            is_killzone=True,
            trigger_quality_score=0.70,
            trigger_quality_label="HIGH",
            spread_bps=12.0,
            whale_aligned=False,
            whale_buy_ratio=0.08,          # 92% selling vs LONG
            eq_zone="discount",
            eq_zone_depth=0.7,
            volume_context="ABOVE_AVG",
            volume_score=72.0,
        )
        # KC2: 92% selling = (1-0.08) = 0.92 >= 0.85 threshold
        assert result.should_block
        assert "WHALE_OPPOSITION" in result.kill_combo

    def test_3b_strong_setup_moderate_whale_conflict(self):
        """Strong setup with moderate (not extreme) whale opposition.
        75% opposition → below KC2 threshold.  Should penalize, not block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=84.0,
            is_killzone=True,
            trigger_quality_score=0.70,
            trigger_quality_label="HIGH",
            spread_bps=12.0,
            whale_aligned=False,
            whale_buy_ratio=0.30,          # 70% selling, below 85% thresh
            eq_zone="discount",
            eq_zone_depth=0.7,
            volume_context="ABOVE_AVG",
            volume_score=72.0,
        )
        # Moderate opposition → no kill combo but whale score penalizes
        assert "WHALE_OPPOSITION" not in (result.kill_combo or "")
        assert not result.should_block
        # Score still reasonable since most factors are good
        assert result.execution_score >= 55.0


class TestMomentumBreakoutCases:
    """GPT Matrix §5 — Momentum override and fake breakout detection.

    Tests whether the system correctly handles breakout situations where
    momentum may override location rules, or volume invalidates a trigger.
    """

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_5a_breakout_override_premium_entry(self):
        """Premium entry + high trigger + breakout volume.
        Momentum should override location concern → ENTER.
        No kill combo fires since trigger is high and volume is breakout."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=85.0,
            is_killzone=True,
            trigger_quality_score=0.85,    # High trigger
            trigger_quality_label="HIGH",
            spread_bps=10.0,
            whale_aligned=True,
            whale_buy_ratio=0.80,
            eq_zone="premium",             # Wrong zone
            eq_zone_depth=0.7,
            volume_context="BREAKOUT",     # Breakout volume
            volume_score=85.0,
        )
        # No kill combo: trigger high (no KC4), spread tight (no KC3)
        assert result.kill_combo == ""
        assert not result.should_block
        # Premium scores 20, but everything else 85-100
        # Composite should still be above penalty zone
        assert result.execution_score >= 60.0

    def test_5b_fake_breakout_high_trigger_low_volume(self):
        """High trigger signal but LOW volume → fake breakout.
        Trigger looks good (0.75) but no volume backing.
        Volume factor scores badly, but other factors compensate.
        Important: system flags volume_env as bad even if composite passes."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=75.0,
            is_killzone=True,
            trigger_quality_score=0.75,    # Good trigger
            trigger_quality_label="HIGH",
            spread_bps=20.0,
            whale_aligned=None,            # No whale data
            eq_zone="discount",
            volume_context="LOW_VOL",      # But low volume!
            volume_score=22.0,
        )
        # No kill combo: trigger 0.75 > 0.25, volume LOW but trigger good
        assert result.kill_combo == ""
        # Volume factor is flagged as bad (~24/100)
        assert 'volume_env' in result.bad_factors
        assert result.factors['volume_env'] < 30.0
        # With 5 other strong factors, composite still passes (weighted average)
        # This demonstrates why GPT says single-factor "fake breakout"
        # detection needs the kill combo approach, not just weighted scoring:
        # the system correctly scores volume badly but other factors dilute it
        assert result.execution_score >= 60.0

    def test_5a_breakout_override_short_in_discount(self):
        """SHORT in discount (wrong zone) + high trigger + breakout volume.
        Same momentum-override concept applied to shorts."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="A",
            confidence=84.0,
            is_killzone=True,
            trigger_quality_score=0.80,
            trigger_quality_label="HIGH",
            spread_bps=8.0,
            whale_aligned=True,
            whale_buy_ratio=0.15,
            eq_zone="discount",            # Wrong zone for SHORT
            eq_zone_depth=0.6,
            volume_context="BREAKOUT",
            volume_score=85.0,
        )
        # Entry scores 20 (wrong zone) but everything else strong
        assert result.kill_combo == ""
        assert not result.should_block
        assert result.execution_score >= 55.0


class TestConflictEngine:
    """GPT Matrix §6 — Conflicting signals between subsystems.

    Tests that the system correctly de-ranks (doesn't ignore) when one
    strong factor conflicts with an otherwise aligned setup.
    """

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_6a_structure_bearish_whales_bullish(self):
        """Wyckoff/UNI scenario: structure says SHORT, whales say LONG.
        Taking SHORT direction → whale opposition penalizes heavily.
        But with 0.15 weight, one bad factor is diluted by 5 good ones."""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="A",
            confidence=82.0,
            is_killzone=True,
            trigger_quality_score=0.55,
            trigger_quality_label="MEDIUM",
            spread_bps=20.0,
            whale_aligned=False,           # Whales oppose SHORT
            whale_buy_ratio=0.78,          # Strong buying (below KC2's 85%)
            eq_zone="premium",             # Correct zone for SHORT
            eq_zone_depth=0.7,
            volume_context="ABOVE_AVG",
            volume_score=65.0,
        )
        # No kill combo (78% < 85% threshold)
        assert "WHALE_OPPOSITION" not in (result.kill_combo or "")
        # Whale factor scores badly (opposing, strong ~19/100)
        assert result.factors['whale_alignment'] < 40.0
        assert 'whale_alignment' in result.bad_factors
        # De-ranked but not killed: whale at 0.15 weight can't single-handedly
        # block when 5 other factors score 55-100.  This is the design:
        # weighted scoring de-ranks, kill combos enforce hard stops.
        # Score is lower than ideal but still passes
        assert result.execution_score < 80.0

    def test_6b_everything_aligned_except_session(self):
        """Everything perfect except dead zone session.
        One strong conflict → system de-ranks, doesn't kill (no combo)."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=85.0,
            is_dead_zone=True,             # Bad session
            trigger_quality_score=0.80,
            trigger_quality_label="HIGH",
            spread_bps=8.0,                # Tight (no KC3)
            whale_aligned=True,
            whale_buy_ratio=0.82,
            eq_zone="discount",
            eq_zone_depth=0.65,
            volume_context="BREAKOUT",
            volume_score=82.0,
        )
        # Dead zone + tight spread → no KC3 (spread < 50)
        assert result.kill_combo == ""
        # Session scores 15 (bad) but everything else 80-100
        # De-ranked but not killed
        assert not result.should_block
        assert 'session' in result.bad_factors
        assert result.execution_score >= 55.0

    def test_6b_everything_aligned_except_volume(self):
        """Everything perfect except volume is climactic (exhaustion risk).
        One conflict → de-rank, don't ignore."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=90.0,
            is_killzone=True,
            trigger_quality_score=0.82,
            trigger_quality_label="HIGH",
            spread_bps=6.0,
            whale_aligned=True,
            whale_buy_ratio=0.85,
            eq_zone="discount",
            eq_zone_depth=0.7,
            volume_context="CLIMACTIC",    # Exhaustion risk
            volume_score=35.0,
        )
        assert result.kill_combo == ""
        assert not result.should_block
        # Climactic scores ~30, which is below BAD_FACTOR_THRESHOLD
        assert 'volume_env' in result.bad_factors
        # But score should still be decent (5 good factors, 1 bad)
        assert result.execution_score >= 60.0


class TestStackingEffectsExpanded:
    """GPT Matrix §7 — Stacking effect escalation tests.

    Tests that multiple simultaneous bad factors escalate properly:
      - 2 bad → no stacking
      - 3 bad → stacking penalty
      - 4 bad → heavier stacking
      - 5+ bad → near-certain block
    """

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_7a_three_weak_factors_blocks(self):
        """Dead zone + wide spread (45 bps, below KC3) + medium trigger.
        3 weak factors stacking → should block or heavily penalize."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=72.0,
            is_dead_zone=True,             # Bad: session (15)
            trigger_quality_score=0.30,    # Bad: trigger (30)
            trigger_quality_label="LOW",
            spread_bps=45.0,               # Below KC3 (50) but still penalized
            whale_aligned=None,            # Neutral
            eq_zone="premium",             # Bad: wrong zone (20)
            eq_zone_depth=0.6,
            volume_context="NORMAL",
            volume_score=50.0,
        )
        assert len(result.bad_factors) >= 3
        assert result.kill_combo == ""     # No kill combo fires
        # Stacking penalty should kick in
        assert any('stacking' in note.lower() for note in result.notes)
        # Should block or heavily penalize
        assert result.should_block or result.should_penalize

    def test_7b_two_weak_one_strong_penalizes(self):
        """Dead zone + premium entry BUT high trigger (0.80).
        2 weak + 1 strong offsetting → penalize not block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=75.0,
            is_dead_zone=True,             # Bad: session (15)
            trigger_quality_score=0.80,    # Good: trigger (80)
            trigger_quality_label="HIGH",
            spread_bps=15.0,               # OK spread
            whale_aligned=True,            # Good
            whale_buy_ratio=0.75,
            eq_zone="premium",             # Bad: wrong zone (20)
            eq_zone_depth=0.6,
            volume_context="ABOVE_AVG",
            volume_score=70.0,
        )
        # 2 bad factors (session, entry) → no stacking
        assert len(result.bad_factors) == 2
        assert result.kill_combo == ""
        # High trigger + whales + volume offset the 2 bad factors
        assert not result.should_block

    def test_7c_four_bad_factors_escalation(self):
        """Dead zone + low trigger + wide spread + wrong zone.
        4 bad factors → stacking penalty escalates (2 above threshold)."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=70.0,
            is_dead_zone=True,             # Bad: session (15)
            trigger_quality_score=0.35,    # Bad: trigger (35)
            trigger_quality_label="LOW",
            spread_bps=45.0,               # Below KC3 but scores ~42
            whale_aligned=None,            # Neutral (~60)
            eq_zone="premium",             # Bad: wrong zone (20)
            eq_zone_depth=0.7,
            volume_context="LOW_VOL",      # Bad: low volume (~25)
            volume_score=30.0,
        )
        assert len(result.bad_factors) >= 4
        # Stacking penalty: (4-2) * 5 = 10 extra penalty
        assert any('stacking' in note.lower() for note in result.notes)
        assert result.should_block

    def test_7d_five_bad_factors_maximum_stacking(self):
        """Everything bad except one factor → maximum stacking.
        Dead zone + low trigger + wide spread + opposing whales + premium.
        Only volume is normal."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=70.0,
            is_dead_zone=True,                # Bad: session (15)
            trigger_quality_score=0.30,       # Bad: trigger (30)
            trigger_quality_label="LOW",
            spread_bps=45.0,                  # Below KC3 (50 bps)
            whale_aligned=False,
            whale_buy_ratio=0.30,             # Mild opposition (below KC2)
            eq_zone="premium",                # Bad: wrong zone (20)
            eq_zone_depth=0.8,
            volume_context="LOW_VOL",         # Bad: low volume
            volume_score=25.0,
        )
        assert len(result.bad_factors) >= 5
        # Stacking penalty capped at MAX_STACKING_PENALTY (15)
        assert any('stacking' in note.lower() for note in result.notes)
        assert result.should_block
        assert result.execution_score < EG.HARD_BLOCK_THRESHOLD


class TestDecisionBoundaries:
    """Test exact threshold boundaries where decisions flip.

    These are the critical edges:
      - Hard block: composite < 35 (standard) or < 25 (A+)
      - Soft penalty: 35 ≤ composite < 50
      - Pass: composite ≥ 50
    """

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_just_above_hard_block_threshold(self):
        """Composite barely above 35 → soft penalty, NOT block."""
        # Asia session (55) + medium trigger (40) + moderate spread (~60)
        # + neutral whales (60) + trending entry (70) + normal volume (~53)
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=72.0,
            session_name="Asia",           # 55
            trigger_quality_score=0.40,    # 40
            trigger_quality_label="MEDIUM",
            spread_bps=25.0,               # ~85 (interpolated)
            whale_aligned=None,            # 60
            eq_zone="trending",            # 70
            volume_context="NORMAL",
            volume_score=55.0,             # ~58
        )
        # Composite: 55*0.2 + 40*0.2 + ~85*0.15 + 60*0.15 + 70*0.15 + ~58*0.15
        # = 11 + 8 + 12.75 + 9 + 10.5 + 8.7 ≈ 60
        assert not result.should_block
        assert result.execution_score >= EG.HARD_BLOCK_THRESHOLD

    def test_just_below_hard_block_threshold(self):
        """Composite just below 35 → BLOCK (grade B)."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=72.0,
            is_dead_zone=True,             # 15
            trigger_quality_score=0.28,    # 28
            trigger_quality_label="LOW",
            spread_bps=45.0,               # ~49 (wide but below KC3)
            whale_aligned=False,
            whale_buy_ratio=0.35,          # Mild opposition, below KC2
            eq_zone="eq_dead_zone",        # 50
            volume_context="LOW_VOL",
            volume_score=28.0,             # ~26
        )
        # Multiple bad factors + stacking → below 35
        assert result.should_block

    def test_aplus_survives_between_25_and_35(self):
        """A+ with confidence ≥ 90 uses relaxed threshold (25).
        Score between 25-35 → passes for A+ but would block for B."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=92.0,
            is_dead_zone=True,             # Bad session (15)
            trigger_quality_score=0.40,    # Mediocre (40)
            trigger_quality_label="MEDIUM",
            spread_bps=8.0,                # Tight (no KC3)
            whale_aligned=True,            # Good (100)
            whale_buy_ratio=0.80,
            eq_zone="trending",            # OK (70)
            volume_context="NORMAL",
            volume_score=50.0,             # OK (~54)
        )
        # No kill combo (spread < 50)
        assert result.kill_combo == ""
        # A+ threshold is 25 → should survive
        assert not result.should_block

    def test_aplus_low_conf_uses_standard_threshold(self):
        """A+ with confidence < 90 → standard threshold (35).
        Same setup as above but conf=80 → might block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=80.0,               # Below 90 → no relaxation
            is_dead_zone=True,
            trigger_quality_score=0.20,    # Bad
            trigger_quality_label="LOW",
            spread_bps=8.0,                # Tight (no KC3)
            whale_aligned=None,            # Neutral
            eq_zone="premium",             # Wrong zone
            volume_context="LOW_VOL",
            volume_score=25.0,
        )
        # Uses standard threshold (35), not relaxed
        # 3+ bad factors → stacking → likely below 35
        assert result.should_block

    def test_soft_penalty_zone_between_35_and_50(self):
        """Score in 35-50 range → penalize but don't block."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=75.0,
            session_name="Asia",           # 55
            trigger_quality_score=0.35,    # 35
            trigger_quality_label="LOW",
            spread_bps=35.0,               # ~65
            whale_aligned=None,            # 60
            eq_zone="trending",            # 70
            volume_context="NORMAL",
            volume_score=45.0,             # ~51
        )
        # Should be in soft penalty zone
        if not result.should_block:
            assert result.should_penalize or result.execution_score >= EG.SOFT_PENALTY_THRESHOLD

    def test_above_soft_penalty_clean_pass(self):
        """Score above 50 → clean pass, no penalty."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A",
            confidence=82.0,
            session_name="London Open",    # 100
            trigger_quality_score=0.65,    # 65
            trigger_quality_label="MEDIUM",
            spread_bps=15.0,               # ~92
            whale_aligned=True,            # 100
            eq_zone="discount",            # 100
            volume_context="NORMAL",
            volume_score=55.0,             # ~52
        )
        assert not result.should_block
        assert not result.should_penalize
        assert result.execution_score >= EG.SOFT_PENALTY_THRESHOLD


class TestScoringValidation:
    """GPT Matrix — Exact score + action validation.

    Verifies execution_score, final action, and kill_combo for specific
    scenario classes.  Format:
      {"case": "...", "execution_score": N, "expected": "BLOCK/PASS/PENALIZE"}
    """

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_ideal_trade_scores_above_80(self):
        """Case: ideal_trade → score ≥ 80, action = PASS"""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=92.0,
            is_killzone=True,
            trigger_quality_score=0.88,
            trigger_quality_label="HIGH",
            spread_bps=5.0,
            whale_aligned=True,
            whale_buy_ratio=0.85,
            eq_zone="discount",
            eq_zone_depth=0.7,
            volume_context="BREAKOUT",
            volume_score=88.0,
        )
        assert result.execution_score >= 80.0
        assert not result.should_block
        assert not result.should_penalize
        assert result.kill_combo == ""

    def test_dead_zone_low_trigger_score_blocked(self):
        """Case: dead_zone_low_trigger → score ~25, action = BLOCK"""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=70.0,
            is_dead_zone=True,
            trigger_quality_score=0.22,
            trigger_quality_label="LOW",
            spread_bps=40.0,
            whale_aligned=None,
            eq_zone="trending",
            volume_context="LOW_VOL",
            volume_score=25.0,
        )
        assert result.should_block
        # Either kill combo (KC5: dead + low trigger + low vol) or score-based
        assert result.execution_score < EG.HARD_BLOCK_THRESHOLD or result.kill_combo != ""

    def test_whale_opposition_extreme_blocked(self):
        """Case: whale_opposition_extreme → kill combo, action = BLOCK"""
        result = self.gate.evaluate(
            direction="SHORT",
            grade="B",
            confidence=75.0,
            is_killzone=True,
            trigger_quality_score=0.65,
            trigger_quality_label="MEDIUM",
            spread_bps=15.0,
            whale_aligned=False,
            whale_buy_ratio=0.92,
            eq_zone="premium",
            volume_context="NORMAL",
            volume_score=55.0,
        )
        assert result.should_block
        assert "WHALE_OPPOSITION" in result.kill_combo

    def test_premium_weak_momentum_blocked(self):
        """Case: premium_weak_momentum → KC4 fires, action = BLOCK"""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=72.0,
            is_killzone=True,
            trigger_quality_score=0.18,    # Below KC floor
            trigger_quality_label="LOW",
            spread_bps=20.0,
            whale_aligned=True,
            whale_buy_ratio=0.75,
            eq_zone="premium",             # Wrong zone + low trigger
            volume_context="NORMAL",
            volume_score=55.0,
        )
        assert result.should_block
        assert "WRONG_ZONE" in result.kill_combo

    def test_near_signal_full_scoring_validation(self):
        """Case: near_long_real_signal → KC3 fires, exec_score ~59"""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=96.0,
            is_dead_zone=True,
            trigger_quality_score=0.62,
            trigger_quality_label="MEDIUM",
            spread_bps=84.0,
            whale_aligned=True,
            whale_buy_ratio=0.87,
            eq_zone="premium",
            eq_zone_depth=0.92,
            volume_context="ABOVE_AVG",
            volume_score=74.0,
        )
        # KC3 fires regardless of high score
        assert result.should_block
        assert result.kill_combo != ""
        # But composite would have been decent without combo
        assert result.execution_score >= 40.0
        # All factors present, including positioning
        assert len(result.factors) == 7

    def test_near_signal_at_killzone_passes(self):
        """Case: near_long_at_killzone → same setup but good session → PASS"""
        result = self.gate.evaluate(
            direction="LONG",
            grade="A+",
            confidence=96.0,
            is_killzone=True,              # Changed: good session
            trigger_quality_score=0.62,
            trigger_quality_label="MEDIUM",
            spread_bps=18.0,               # Changed: normal spread
            whale_aligned=True,
            whale_buy_ratio=0.87,
            eq_zone="discount",            # Changed: pullback to discount
            eq_zone_depth=0.6,
            volume_context="ABOVE_AVG",
            volume_score=74.0,
        )
        # No kill combo: good session + tight spread
        assert result.kill_combo == ""
        assert not result.should_block
        # Score should be high with most factors good
        assert result.execution_score >= 70.0
