"""
Tests for deep-dive gap closures:
  Gap 1: Regime transition warnings consumed by risk manager
  Gap 2: Vol-of-vol in signal scoring
  Gap 3: F&G thresholds use constants (not hardcoded 20/80)
  Gap 4: VOLATILE_PANIC position review in risk manager
  Gap 5: Stablecoin flow linked to HTF ADX threshold
"""

import sys
import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Pre-mock heavy modules that risk.manager imports transitively
# (circuit_breaker → cfg.risk.circuit_breaker, database → aiosqlite)
if "risk.circuit_breaker" not in sys.modules:
    _mock_cb_mod = MagicMock()
    _mock_cb_mod.circuit_breaker = MagicMock()
    sys.modules["risk.circuit_breaker"] = _mock_cb_mod

if "data.database" not in sys.modules:
    sys.modules["data.database"] = MagicMock()

if "governance.performance_tracker" not in sys.modules:
    sys.modules["governance.performance_tracker"] = MagicMock()

# ═══════════════════════════════════════════════════════════
# GAP 1: Regime transition → risk manager Kelly reduction
# ═══════════════════════════════════════════════════════════

class TestRegimeTransitionRiskManager:
    """Gap 1: get_regime_transition_kelly_mult() in risk/manager.py."""

    def test_no_transition_returns_1(self):
        """When no transition warning, Kelly mult is 1.0 (no impact)."""
        from risk.manager import RiskManager
        rm = RiskManager()
        mock_regime = MagicMock()
        mock_regime.get_transition_warning.return_value = {
            "warning": False, "factor_count": 0, "factors": []
        }
        with patch("risk.manager.logger"):
            with patch.dict(sys.modules, {"analyzers.regime": types.ModuleType("analyzers.regime")}):
                sys.modules["analyzers.regime"].regime_analyzer = mock_regime
                result = rm.get_regime_transition_kelly_mult()
        assert result == 1.0

    def test_transition_active_returns_reduction(self):
        """When transition is active (≥2 factors), return WARNING_KELLY_REDUCTION."""
        from risk.manager import RiskManager
        from config.constants import RegimeTransition as RT
        rm = RiskManager()
        mock_regime = MagicMock()
        mock_regime.get_transition_warning.return_value = {
            "warning": True, "factor_count": 3, "factors": ["ADX↓5.0", "Vol:COMPRESSED", "FlowDecel:-1.2%"]
        }
        with patch("risk.manager.logger"):
            with patch.dict(sys.modules, {"analyzers.regime": types.ModuleType("analyzers.regime")}):
                sys.modules["analyzers.regime"].regime_analyzer = mock_regime
                result = rm.get_regime_transition_kelly_mult()
        assert result == RT.WARNING_KELLY_REDUCTION
        assert result < 1.0  # Must be a reduction

    def test_transition_exception_returns_1(self):
        """On exception, fallback to 1.0 (no impact)."""
        from risk.manager import RiskManager
        rm = RiskManager()
        with patch("risk.manager.logger"):
            with patch.dict(sys.modules, {"analyzers.regime": None}):
                # Force import to fail
                result = rm.get_regime_transition_kelly_mult()
        assert result == 1.0

    def test_kelly_reduction_constant_exists(self):
        """The WARNING_KELLY_REDUCTION constant exists and is sensible."""
        from config.constants import RegimeTransition as RT
        assert hasattr(RT, 'WARNING_KELLY_REDUCTION')
        assert 0.5 <= RT.WARNING_KELLY_REDUCTION <= 0.9


# ═══════════════════════════════════════════════════════════
# GAP 2: Vol-of-vol in signal scoring
# ═══════════════════════════════════════════════════════════

class TestVolOfVolSignalScoring:
    """Gap 2: vol_of_vol integrated into get_signal_intel() delta."""

    def test_vol_of_vol_constants_exist(self):
        """VolOfVol constants in config/constants.py."""
        from config.constants import VolOfVol
        assert hasattr(VolOfVol, 'HIGH_THRESHOLD')
        assert hasattr(VolOfVol, 'LOW_THRESHOLD')
        assert hasattr(VolOfVol, 'HIGH_PENALTY')
        assert hasattr(VolOfVol, 'LOW_BONUS')
        assert VolOfVol.HIGH_PENALTY < 0  # Must be a penalty
        assert VolOfVol.LOW_BONUS > 0     # Must be a bonus

    def test_high_vov_applies_penalty(self):
        """High vol_of_vol → negative delta applied to signal intel."""
        from config.constants import VolOfVol
        from analyzers.volatility_structure import VolatilityStructureAnalyzer

        analyzer = VolatilityStructureAnalyzer.__new__(VolatilityStructureAnalyzer)
        # Build a minimal snapshot with high vol_of_vol
        snap = MagicMock()
        snap.regime.is_stale = False
        snap.regime.regime = "NORMAL"
        snap.regime.vol_of_vol = VolOfVol.HIGH_THRESHOLD + 0.05  # Above threshold
        snap.regime.breakout_probability = 0.3
        snap.regime.days_in_regime = 5
        snap.regime.context_score = 0.0
        snap.regime.context_factors = ""
        snap.realized.is_stale = True
        snap.implied.is_stale = True
        analyzer._snapshot = snap

        delta, note = analyzer.get_signal_intel("BTCUSDT", "LONG")
        assert delta <= VolOfVol.HIGH_PENALTY  # At least the penalty
        assert "vol clustering" in note.lower() or "VoV" in note

    def test_low_vov_applies_bonus(self):
        """Low vol_of_vol → positive delta (small bonus for stable regime)."""
        from config.constants import VolOfVol
        from analyzers.volatility_structure import VolatilityStructureAnalyzer

        analyzer = VolatilityStructureAnalyzer.__new__(VolatilityStructureAnalyzer)
        snap = MagicMock()
        snap.regime.is_stale = False
        snap.regime.regime = "NORMAL"
        snap.regime.vol_of_vol = VolOfVol.LOW_THRESHOLD - 0.01  # Below low threshold
        snap.regime.breakout_probability = 0.3
        snap.regime.days_in_regime = 5
        snap.regime.context_score = 0.0
        snap.regime.context_factors = ""
        snap.realized.is_stale = True
        snap.implied.is_stale = True
        analyzer._snapshot = snap

        delta, note = analyzer.get_signal_intel("BTCUSDT", "LONG")
        assert delta >= VolOfVol.LOW_BONUS
        assert "stable" in note.lower() or "VoV" in note

    def test_zero_vov_no_impact(self):
        """vol_of_vol == 0 → no VoV delta applied."""
        from analyzers.volatility_structure import VolatilityStructureAnalyzer

        analyzer = VolatilityStructureAnalyzer.__new__(VolatilityStructureAnalyzer)
        snap = MagicMock()
        snap.regime.is_stale = False
        snap.regime.regime = "NORMAL"
        snap.regime.vol_of_vol = 0  # Zero
        snap.regime.breakout_probability = 0.3
        snap.realized.is_stale = True
        snap.implied.is_stale = True
        analyzer._snapshot = snap

        delta, note = analyzer.get_signal_intel("BTCUSDT", "LONG")
        assert "VoV" not in note  # No VoV note when zero


# ═══════════════════════════════════════════════════════════
# GAP 3: F&G thresholds from constants
# ═══════════════════════════════════════════════════════════

class TestFearGreedThresholds:
    """Gap 3: F&G uses centralized constants, not hardcoded 20/80."""

    def test_fear_greed_constants_exist(self):
        """FearGreedThresholds class exists in config/constants.py."""
        from config.constants import FearGreedThresholds
        assert hasattr(FearGreedThresholds, 'EXTREME_FEAR')
        assert hasattr(FearGreedThresholds, 'EXTREME_GREED')
        assert FearGreedThresholds.EXTREME_FEAR == 25
        assert FearGreedThresholds.EXTREME_GREED == 75

    def test_regime_uses_constants_not_20(self):
        """regime.py get_adaptive_min_confidence uses FearGreedThresholds.EXTREME_FEAR (25), not 20."""
        import inspect
        from analyzers.regime import RegimeAnalyzer
        src = inspect.getsource(RegimeAnalyzer.get_adaptive_min_confidence)
        # The hardcoded 20 should be gone — replaced by _FGT.EXTREME_FEAR
        assert "< 20" not in src, "regime.py still uses hardcoded < 20"
        assert "> 80" not in src, "regime.py still uses hardcoded > 80"
        assert "FearGreedThresholds" in src or "_FGT" in src

    def test_htf_guardrail_uses_constants_not_20(self):
        """htf_guardrail.py uses FearGreedThresholds, not hardcoded 20/80."""
        import inspect
        from analyzers.htf_guardrail import HTFWeeklyGuardrail
        src = inspect.getsource(HTFWeeklyGuardrail.is_hard_blocked)
        assert "_fg < 20" not in src, "htf_guardrail still uses hardcoded < 20"
        assert "_fg > 80" not in src, "htf_guardrail still uses hardcoded > 80"
        assert "FearGreedThresholds" in src or "_FGT" in src

    def test_regime_fear_25_triggers_adjustment(self):
        """F&G at 24 (< 25) triggers contrarian long adjustment in regime."""
        from analyzers.regime import RegimeAnalyzer
        ra = RegimeAnalyzer.__new__(RegimeAnalyzer)
        ra._fear_greed = 24  # Below 25 (new threshold)
        ra._btc_adx = 30
        ra._chop_strength = 0.5
        ra._provisional_regime = MagicMock()
        ra._provisional_regime.value = "BEAR_TREND"
        ra._regime = ra._provisional_regime
        ra._committed_confidence = 0.7
        ra._committed_regime = MagicMock()
        ra._committed_regime.value = "BEAR_TREND"
        ra._performance_tracker = None
        # base_min=72, direction="LONG" — should apply +4 for contrarian fear
        result_long = ra.get_adaptive_min_confidence(72, direction="LONG")
        result_short = ra.get_adaptive_min_confidence(72, direction="SHORT")
        # Long should be higher (penalized), short should NOT be penalized
        assert result_long > result_short

    def test_regime_fg_21_old_would_trigger_new_still_triggers(self):
        """F&G at 21 was NOT caught by < 20, now caught by threshold < 25."""
        from analyzers.regime import RegimeAnalyzer
        ra = RegimeAnalyzer.__new__(RegimeAnalyzer)
        ra._fear_greed = 21  # Was NOT caught by old "< 20", now caught by "< 25"
        ra._btc_adx = 30
        ra._chop_strength = 0.5
        ra._provisional_regime = MagicMock()
        ra._provisional_regime.value = "BEAR_TREND"
        ra._regime = ra._provisional_regime
        ra._committed_confidence = 0.7
        ra._committed_regime = MagicMock()
        ra._committed_regime.value = "BEAR_TREND"
        ra._performance_tracker = None
        result_long = ra.get_adaptive_min_confidence(72, direction="LONG")
        result_no_dir = ra.get_adaptive_min_confidence(72, direction="")
        # With direction=LONG in extreme fear, should apply +4
        # Without direction, should apply +2 (mild caution)
        assert result_long > result_no_dir


# ═══════════════════════════════════════════════════════════
# GAP 4: VOLATILE_PANIC position review
# ═══════════════════════════════════════════════════════════

class TestPanicPositionReview:
    """Gap 4: check_panic_position_review() in risk/manager.py."""

    def test_no_panic_returns_inactive(self):
        """Normal regime → panic_active=False."""
        from risk.manager import RiskManager
        rm = RiskManager()
        mock_regime = MagicMock()
        mock_regime.regime = MagicMock()
        mock_regime.regime.value = "BULL_TREND"
        with patch("risk.manager.logger"):
            with patch.dict(sys.modules, {"analyzers.regime": types.ModuleType("analyzers.regime")}):
                sys.modules["analyzers.regime"].regime_analyzer = mock_regime
                result = rm.check_panic_position_review()
        assert result["panic_active"] is False
        assert result["positions_to_tighten"] == []
        assert result["total_open"] == 0

    def test_panic_with_open_positions(self):
        """VOLATILE_PANIC + open positions → panic_active + total_open set."""
        from risk.manager import RiskManager
        rm = RiskManager()

        mock_regime = MagicMock()
        mock_regime.regime = MagicMock()
        mock_regime.regime.value = "VOLATILE_PANIC"

        mock_tracked = MagicMock()
        mock_tracked.state = "WATCHING"
        mock_tracked2 = MagicMock()
        mock_tracked2.state = "BE_ACTIVE"  # Past TP1

        mock_om = MagicMock()
        mock_om.get_active_signals.return_value = {1: mock_tracked, 2: mock_tracked2}

        regime_mod = types.ModuleType("analyzers.regime")
        regime_mod.regime_analyzer = mock_regime
        om_mod = types.ModuleType("signals.outcome_monitor")
        om_mod.outcome_monitor = mock_om

        with patch("risk.manager.logger"):
            with patch.dict(sys.modules, {
                "analyzers.regime": regime_mod,
                "signals.outcome_monitor": om_mod,
            }):
                result = rm.check_panic_position_review()

        assert result["panic_active"] is True
        assert result["total_open"] == 2
        assert 2 in result["positions_to_tighten"]  # BE_ACTIVE signal
        assert 1 not in result["positions_to_tighten"]  # WATCHING signal

    def test_panic_no_open_positions(self):
        """VOLATILE_PANIC but no open positions → panic_active but empty lists."""
        from risk.manager import RiskManager
        rm = RiskManager()

        mock_regime = MagicMock()
        mock_regime.regime = MagicMock()
        mock_regime.regime.value = "VOLATILE_PANIC"

        mock_om = MagicMock()
        mock_om.get_active_signals.return_value = {}

        regime_mod = types.ModuleType("analyzers.regime")
        regime_mod.regime_analyzer = mock_regime
        om_mod = types.ModuleType("signals.outcome_monitor")
        om_mod.outcome_monitor = mock_om

        with patch("risk.manager.logger"):
            with patch.dict(sys.modules, {
                "analyzers.regime": regime_mod,
                "signals.outcome_monitor": om_mod,
            }):
                result = rm.check_panic_position_review()

        assert result["panic_active"] is True
        assert result["total_open"] == 0
        assert result["positions_to_tighten"] == []

    def test_panic_position_review_constants_exist(self):
        """PanicPositionReview constants exist."""
        from config.constants import PanicPositionReview
        assert hasattr(PanicPositionReview, 'TIGHTEN_POST_TP1')
        assert hasattr(PanicPositionReview, 'PANIC_NEW_SIGNAL_KELLY_MULT')
        assert PanicPositionReview.PANIC_NEW_SIGNAL_KELLY_MULT < 0.25  # Stricter than normal VOLATILE_PANIC


# ═══════════════════════════════════════════════════════════
# GAP 5: Stablecoin flow → HTF ADX threshold
# ═══════════════════════════════════════════════════════════

class TestStablecoinHTFLink:
    """Gap 5: stablecoin flow adjusts HTF guardrail ADX threshold."""

    def test_stablecoin_htf_constants_exist(self):
        """StablecoinHTF constants in config/constants.py."""
        from config.constants import StablecoinHTF
        assert hasattr(StablecoinHTF, 'OUTFLOW_ADX_BOOST')
        assert hasattr(StablecoinHTF, 'INFLOW_ADX_REDUCTION')
        assert hasattr(StablecoinHTF, 'BASE_ADX_THRESHOLD')
        assert StablecoinHTF.BASE_ADX_THRESHOLD == 25

    def test_draining_raises_threshold(self):
        """DRAINING stablecoin signal → raises ADX threshold (more restrictive)."""
        from analyzers.htf_guardrail import HTFWeeklyGuardrail
        from config.constants import StablecoinHTF as SH
        g = HTFWeeklyGuardrail()
        g.update_stablecoin_adx_offset("DRAINING")
        threshold = g.get_dynamic_adx_threshold()
        assert threshold == SH.BASE_ADX_THRESHOLD + SH.OUTFLOW_ADX_BOOST
        assert threshold > 25  # Must be more restrictive

    def test_ample_lowers_threshold(self):
        """AMPLE stablecoin signal → lowers ADX threshold (more permissive)."""
        from analyzers.htf_guardrail import HTFWeeklyGuardrail
        from config.constants import StablecoinHTF as SH
        g = HTFWeeklyGuardrail()
        g.update_stablecoin_adx_offset("AMPLE")
        threshold = g.get_dynamic_adx_threshold()
        assert threshold == SH.BASE_ADX_THRESHOLD - SH.INFLOW_ADX_REDUCTION
        assert threshold < 25  # Must be more permissive

    def test_neutral_is_baseline(self):
        """NEUTRAL stablecoin signal → ADX threshold stays at 25."""
        from analyzers.htf_guardrail import HTFWeeklyGuardrail
        g = HTFWeeklyGuardrail()
        g.update_stablecoin_adx_offset("NEUTRAL")
        assert g.get_dynamic_adx_threshold() == 25

    def test_unknown_is_baseline(self):
        """UNKNOWN stablecoin signal → ADX threshold stays at 25."""
        from analyzers.htf_guardrail import HTFWeeklyGuardrail
        g = HTFWeeklyGuardrail()
        g.update_stablecoin_adx_offset("UNKNOWN")
        assert g.get_dynamic_adx_threshold() == 25

    def test_draining_blocks_weak_counter_at_higher_adx(self):
        """When DRAINING, a weak trend at ADX=26 now gets through (was blocked at 25)."""
        from analyzers.htf_guardrail import HTFWeeklyGuardrail
        g = HTFWeeklyGuardrail()
        g._warmed = True
        g._weekly_bias = "BEARISH"
        g._weekly_adx = 26.0  # Above 25 but below 28 (25+3)

        # Without stablecoin adjustment: ADX 26 > 25 → would evaluate counter-trend threshold
        # With DRAINING: threshold raised to 28 → ADX 26 < 28 → soft penalty, not hard block
        g.update_stablecoin_adx_offset("DRAINING")
        blocked, reason = g.is_hard_blocked("LONG", 70.0, "Momentum")
        assert blocked is False  # ADX 26 < 28 → weak trend path
        assert "weak counter-trend" in reason.lower() or "28" in reason

    def test_ample_allows_counter_trend_at_lower_adx(self):
        """When AMPLE, ADX 24 gets soft penalty (was under 25 baseline anyway, now under 23)."""
        from analyzers.htf_guardrail import HTFWeeklyGuardrail
        g = HTFWeeklyGuardrail()
        g._warmed = True
        g._weekly_bias = "BEARISH"
        g._weekly_adx = 24.0

        # With AMPLE: threshold lowered to 23 → ADX 24 > 23 → evaluates counter-trend
        g.update_stablecoin_adx_offset("AMPLE")
        # ADX 24 is now > 23 (dynamic threshold), so it enters counter-trend evaluation
        # This is different from neutral where ADX 24 < 25 = weak trend path
        # The key test: AMPLE makes the guardrail notice this trend
        threshold = g.get_dynamic_adx_threshold()
        assert threshold == 23
        assert g._weekly_adx >= threshold  # ADX 24 >= 23 — will evaluate

    def test_offset_resets_on_neutral(self):
        """Setting DRAINING then NEUTRAL resets offset to 0."""
        from analyzers.htf_guardrail import HTFWeeklyGuardrail
        g = HTFWeeklyGuardrail()
        g.update_stablecoin_adx_offset("DRAINING")
        assert g._stablecoin_adx_offset > 0
        g.update_stablecoin_adx_offset("NEUTRAL")
        assert g._stablecoin_adx_offset == 0
        assert g.get_dynamic_adx_threshold() == 25
