"""
Tests for production gap fixes from live log analysis.

Gap 1: Zero-data signal gate
Gap 2: Ensemble SUPPRESS threshold raised from -3.0 to -4.5
Gap 3: Clarity-based hourly limit bypass
Gap 4: Adaptive weights persistence
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ════════════════════════════════════════════════════════════════════
# GAP 1: Zero-data signal gate
# ════════════════════════════════════════════════════════════════════

class TestZeroDataSignalGate:
    """Signals with ALL default scores (tech=50, vol=50, of=50) must be rejected."""

    def test_zero_data_margin_constant_exists(self):
        from config.constants import Grading
        assert hasattr(Grading, 'ZERO_DATA_MARGIN')
        assert Grading.ZERO_DATA_MARGIN == 2.0

    def test_neutral_score_constant(self):
        from config.constants import Grading
        assert Grading.NEUTRAL_SCORE == 50.0

    def test_all_neutral_scores_detected(self):
        """If tech=50, vol=50, of=50 → all within margin → should reject."""
        from config.constants import Grading
        neutral = Grading.NEUTRAL_SCORE
        margin = Grading.ZERO_DATA_MARGIN

        tech, vol, of = 50.0, 50.0, 50.0
        is_zero_data = (
            abs(tech - neutral) <= margin
            and abs(vol - neutral) <= margin
            and abs(of - neutral) <= margin
        )
        assert is_zero_data, "All-neutral scores should be detected as zero-data"

    def test_one_real_score_passes(self):
        """If tech=65 but vol=50, of=50 → tech has real data → should pass."""
        from config.constants import Grading
        neutral = Grading.NEUTRAL_SCORE
        margin = Grading.ZERO_DATA_MARGIN

        tech, vol, of = 65.0, 50.0, 50.0
        is_zero_data = (
            abs(tech - neutral) <= margin
            and abs(vol - neutral) <= margin
            and abs(of - neutral) <= margin
        )
        assert not is_zero_data, "One real score should prevent zero-data rejection"

    def test_slight_above_neutral_passes(self):
        """If tech=53, vol=50, of=50 → tech > neutral + margin → passes."""
        from config.constants import Grading
        neutral = Grading.NEUTRAL_SCORE
        margin = Grading.ZERO_DATA_MARGIN

        tech, vol, of = 53.0, 50.0, 50.0
        is_zero_data = (
            abs(tech - neutral) <= margin
            and abs(vol - neutral) <= margin
            and abs(of - neutral) <= margin
        )
        assert not is_zero_data


# ════════════════════════════════════════════════════════════════════
# GAP 2: Ensemble SUPPRESS threshold
# ════════════════════════════════════════════════════════════════════

class TestEnsembleSuppressThreshold:
    """SUPPRESS threshold raised from -3.0 to -4.5 and high-conf override added."""

    def test_suppress_threshold_constant_exists(self):
        from config.constants import EnsembleVoter
        assert hasattr(EnsembleVoter, 'SUPPRESS_WEIGHTED_THRESHOLD')
        assert EnsembleVoter.SUPPRESS_WEIGHTED_THRESHOLD == -4.5

    def test_suppress_override_constants(self):
        from config.constants import EnsembleVoter
        assert EnsembleVoter.SUPPRESS_OVERRIDE_MIN_CONF == 85
        assert EnsembleVoter.SUPPRESS_OVERRIDE_MIN_RR == 2.5

    def test_reduce_threshold(self):
        from config.constants import EnsembleVoter
        assert EnsembleVoter.REDUCE_WEIGHTED_THRESHOLD == -1.5

    def test_boost_threshold(self):
        from config.constants import EnsembleVoter
        assert EnsembleVoter.BOOST_WEIGHTED_THRESHOLD == 3.0

    def test_score_minus_3_5_is_reduce_not_suppress(self):
        """Score of -3.5 should now REDUCE (not SUPPRESS as before)."""
        from config.constants import EnsembleVoter as EVC
        score = -3.5
        # Old behavior: score <= -3.0 → SUPPRESS
        # New behavior: score > EVC.SUPPRESS_WEIGHTED_THRESHOLD (-4.5) → REDUCE
        assert score > EVC.SUPPRESS_WEIGHTED_THRESHOLD, (
            "Score -3.5 should NOT trigger SUPPRESS (threshold is now -4.5)"
        )

    def test_score_minus_5_0_is_suppress(self):
        """Score of -5.0 should still SUPPRESS."""
        from config.constants import EnsembleVoter as EVC
        score = -5.0
        assert score <= EVC.SUPPRESS_WEIGHTED_THRESHOLD

    def test_high_conf_overrides_suppress(self):
        """Signal with conf=87, rr=3.0 should downgrade SUPPRESS → REDUCE."""
        from config.constants import EnsembleVoter as EVC
        conf = 87
        rr = 3.0
        should_override = (
            conf >= EVC.SUPPRESS_OVERRIDE_MIN_CONF
            and rr >= EVC.SUPPRESS_OVERRIDE_MIN_RR
        )
        assert should_override

    def test_low_conf_does_not_override(self):
        """Signal with conf=70, rr=3.0 should NOT override SUPPRESS."""
        from config.constants import EnsembleVoter as EVC
        conf = 70
        rr = 3.0
        should_override = (
            conf >= EVC.SUPPRESS_OVERRIDE_MIN_CONF
            and rr >= EVC.SUPPRESS_OVERRIDE_MIN_RR
        )
        assert not should_override

    def test_evaluate_accepts_signal_params(self):
        """ensemble_voter.evaluate() should accept signal_confidence and signal_rr."""
        from signals.ensemble_voter import ensemble_voter
        import inspect
        sig = inspect.signature(ensemble_voter.evaluate)
        assert 'signal_confidence' in sig.parameters
        assert 'signal_rr' in sig.parameters


# ════════════════════════════════════════════════════════════════════
# GAP 3: Clarity-based hourly limit bypass
# ════════════════════════════════════════════════════════════════════

class TestClarityHourlyBypass:
    """High-clarity signals should bypass hourly rate limits."""

    def test_clarity_bypass_constants(self):
        from config.constants import RateLimiting
        assert RateLimiting.CLARITY_BYPASS_MIN_SCORE == 95
        assert RateLimiting.CLARITY_BYPASS_MIN_CONF == 80
        assert RateLimiting.CLARITY_BYPASS_MIN_RR == 2.0

    def test_perfect_clarity_bypasses(self):
        """clarity=100, conf=82, rr=2.5 should bypass."""
        from config.constants import RateLimiting as RL
        clarity, conf, rr = 100, 82, 2.5
        exempt = (
            clarity >= RL.CLARITY_BYPASS_MIN_SCORE
            and conf >= RL.CLARITY_BYPASS_MIN_CONF
            and rr >= RL.CLARITY_BYPASS_MIN_RR
        )
        assert exempt

    def test_low_clarity_does_not_bypass(self):
        """clarity=70, conf=90, rr=3.0 should NOT bypass."""
        from config.constants import RateLimiting as RL
        clarity, conf, rr = 70, 90, 3.0
        exempt = (
            clarity >= RL.CLARITY_BYPASS_MIN_SCORE
            and conf >= RL.CLARITY_BYPASS_MIN_CONF
            and rr >= RL.CLARITY_BYPASS_MIN_RR
        )
        assert not exempt

    def test_clarity_95_exact_threshold(self):
        """clarity=95 is the exact boundary — should bypass."""
        from config.constants import RateLimiting as RL
        clarity, conf, rr = 95, 80, 2.0
        exempt = (
            clarity >= RL.CLARITY_BYPASS_MIN_SCORE
            and conf >= RL.CLARITY_BYPASS_MIN_CONF
            and rr >= RL.CLARITY_BYPASS_MIN_RR
        )
        assert exempt

    def test_clarity_94_does_not_bypass(self):
        """clarity=94 is below threshold."""
        from config.constants import RateLimiting as RL
        clarity, conf, rr = 94, 80, 2.0
        exempt = (
            clarity >= RL.CLARITY_BYPASS_MIN_SCORE
            and conf >= RL.CLARITY_BYPASS_MIN_CONF
            and rr >= RL.CLARITY_BYPASS_MIN_RR
        )
        assert not exempt


# ════════════════════════════════════════════════════════════════════
# GAP 4: Adaptive weights persistence
# ════════════════════════════════════════════════════════════════════

class TestAdaptiveWeightsPersistence:
    """Adaptive weights should persist to DB and reload on restart."""

    def test_persist_key_exists(self):
        from signals.adaptive_weights import AdaptiveWeightManager
        assert hasattr(AdaptiveWeightManager, '_PERSIST_KEY')
        assert AdaptiveWeightManager._PERSIST_KEY == "adaptive_weight_snapshots"

    def test_has_persist_method(self):
        from signals.adaptive_weights import AdaptiveWeightManager
        assert hasattr(AdaptiveWeightManager, '_persist_snapshots')
        assert asyncio.iscoroutinefunction(AdaptiveWeightManager._persist_snapshots)

    def test_has_load_method(self):
        from signals.adaptive_weights import AdaptiveWeightManager
        assert hasattr(AdaptiveWeightManager, '_load_persisted_snapshots')
        assert asyncio.iscoroutinefunction(AdaptiveWeightManager._load_persisted_snapshots)

    @pytest.mark.asyncio
    async def test_load_empty_state_is_noop(self):
        """Loading when no state exists should be a no-op (no crash)."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()

        mock_db = MagicMock()
        mock_db.load_learning_state = AsyncMock(return_value=None)

        with patch.dict('sys.modules', {'data.database': MagicMock(db=mock_db)}):
            await mgr._load_persisted_snapshots()

        assert len(mgr._snapshots) == 0

    @pytest.mark.asyncio
    async def test_load_restores_snapshots(self):
        """Loading valid persisted state should restore snapshots."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()

        persisted = {
            "BULL_TREND": {
                "weights": {"cvd": 1.5, "smart_money": 2.0},
                "last_update": time.time() - 3600,
                "sufficient_data": True,
            }
        }
        mock_db = MagicMock()
        mock_db.load_learning_state = AsyncMock(return_value=persisted)

        with patch.dict('sys.modules', {'data.database': MagicMock(db=mock_db)}):
            await mgr._load_persisted_snapshots()

        assert "BULL_TREND" in mgr._snapshots
        snap = mgr._snapshots["BULL_TREND"]
        assert snap.weights == {"cvd": 1.5, "smart_money": 2.0}
        assert snap.sufficient_data is True

    @pytest.mark.asyncio
    async def test_persist_saves_snapshots(self):
        """Persisting should call save_learning_state with correct payload."""
        from signals.adaptive_weights import AdaptiveWeightManager, AdaptiveWeightSnapshot
        mgr = AdaptiveWeightManager()

        mgr._snapshots["BEAR_TREND"] = AdaptiveWeightSnapshot(
            regime="BEAR_TREND",
            weights={"cvd": 1.2},
            last_update=1000.0,
            sufficient_data=True,
        )

        mock_db = MagicMock()
        mock_db.save_learning_state = AsyncMock()

        with patch.dict('sys.modules', {'data.database': MagicMock(db=mock_db)}):
            await mgr._persist_snapshots()

        mock_db.save_learning_state.assert_called_once()
        key, payload = mock_db.save_learning_state.call_args[0]
        assert key == "adaptive_weight_snapshots"
        assert "BEAR_TREND" in payload
        assert payload["BEAR_TREND"]["weights"] == {"cvd": 1.2}

    @pytest.mark.asyncio
    async def test_persist_failure_is_nonfatal(self):
        """Persist failure should log but not crash."""
        from signals.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()
        mgr._snapshots["X"] = MagicMock()
        mgr._snapshots["X"].weights = {}
        mgr._snapshots["X"].last_update = 0
        mgr._snapshots["X"].sufficient_data = False

        mock_db = MagicMock()
        mock_db.save_learning_state = AsyncMock(side_effect=Exception("DB error"))

        with patch.dict('sys.modules', {'data.database': MagicMock(db=mock_db)}):
            # Should not raise
            await mgr._persist_snapshots()


# ════════════════════════════════════════════════════════════════════
# GAP 5: Phase 4 runtime wiring
# ════════════════════════════════════════════════════════════════════

class TestPhase4RuntimeWiring:
    def test_engine_starts_adaptive_weights(self):
        from pathlib import Path

        source = Path(__file__).resolve().parents[1].joinpath("core", "engine.py").read_text()
        assert "adaptive_weight_manager.start" in source

    @pytest.mark.asyncio
    async def test_param_tuner_no_data_does_not_advance_last_run(self):
        from governance.param_tuner import ParamTuner

        tuner = ParamTuner()
        tuner._last_run = 0.0
        tuner._loaded_from_db = True
        tuner._fetch_stats = AsyncMock(return_value=([], []))

        result = await tuner.run_tuning_cycle()

        assert result == []
        assert tuner._last_run == 0.0

    @pytest.mark.asyncio
    async def test_param_tuner_tracks_only_adjusted_states(self):
        from governance.param_tuner import ParamTuner

        tuner = ParamTuner()
        tuner._loaded_from_db = True
        tuner._fetch_stats = AsyncMock(return_value=([
            {
                "market_state": "VOLATILE_PANIC",
                "wins": 10,
                "losses": 30,
                "win_rate": 0.25,
                "expectancy": -0.5,
                "signals": 40,
                "skipped": 0,
                "avg_r": -0.2,
            },
            {
                "market_state": "TRENDING",
                "wins": 20,
                "losses": 20,
                "win_rate": 0.50,
                "expectancy": 0.2,
                "signals": 40,
                "skipped": 0,
                "avg_r": 0.3,
            },
        ], []))
        tuner._check_rollback = AsyncMock(return_value=None)
        tuner._save_rollback_state = AsyncMock()
        tuner._append_audit_entry = AsyncMock()

        fake_params = MagicMock()
        fake_params.take_snapshot = MagicMock()
        fake_params.spec_step.side_effect = lambda _key: 1.0
        fake_params.adjust.side_effect = lambda _key, _delta: 1.0
        fake_params.save = AsyncMock()

        with patch.dict('sys.modules', {'governance.adaptive_params': MagicMock(adaptive_params=fake_params)}):
            await tuner.run_tuning_cycle()

        assert tuner._last_adjusted_states == ["VOLATILE_PANIC"]

    def test_stablecoin_vote_accepts_liquidity_labels(self):
        from signals.ensemble_voter import ensemble_voter

        verdict = ensemble_voter.evaluate(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100_000.0,
            stablecoin_trend="AMPLE",
        )

        stablecoin_vote = next(v for v in verdict.votes if v.source == "stablecoin")
        assert stablecoin_vote.value > 0
        assert "INFLOW" in stablecoin_vote.reason

    def test_phase4_inputs_create_real_votes(self):
        from signals.ensemble_voter import ensemble_voter

        verdict = ensemble_voter.evaluate(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100_000.0,
            mining_health="STRONG",
            network_demand="STRONG",
            vol_regime="HIGH",
        )

        votes = {v.source: v for v in verdict.votes}
        assert votes["mining_health"].value > 0
        assert votes["network_demand"].value > 0
        assert votes["vol_regime"].value < 0

    def test_optional_phase4_sources_use_weight_lookup(self):
        from signals.ensemble_voter import ensemble_voter

        with patch.object(ensemble_voter, "_get_weights", return_value={
            "cvd": 1.0,
            "smart_money": 1.0,
            "oi_trend": 1.0,
            "crowd": 1.0,
            "whale_flow": 1.0,
            "basis": 1.0,
            "onchain": 1.7,
            "stablecoin": 1.6,
            "whale_intent": 1.5,
            "mining_health": 1.4,
            "network_demand": 1.3,
            "vol_regime": 1.2,
        }):
            verdict = ensemble_voter.evaluate(
                symbol="BTCUSDT",
                direction="LONG",
                entry_price=100_000.0,
                onchain_zone="UNDERVALUED",
                stablecoin_trend="INFLOW",
                whale_intent="BULLISH",
                mining_health="STRONG",
                network_demand="STRONG",
                vol_regime="COMPRESSED",
            )

        weights = {v.source: v.weight for v in verdict.votes}
        assert weights["onchain"] == 1.7
        assert weights["stablecoin"] == 1.6
        assert weights["whale_intent"] == 1.5
        assert weights["mining_health"] == 1.4
        assert weights["network_demand"] == 1.3
        assert weights["vol_regime"] == 1.2

    # ── P4-2: LOW vol_regime should be treated as SUPPORT ─────────────
    def test_low_vol_regime_produces_support_vote(self):
        """LOW vol_regime is a compression regime and should SUPPORT, not be NEUTRAL."""
        from signals.ensemble_voter import ensemble_voter

        verdict = ensemble_voter.evaluate(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100_000.0,
            vol_regime="LOW",
        )

        votes = {v.source: v for v in verdict.votes}
        assert "vol_regime" in votes, "LOW vol_regime should create a vote"
        assert votes["vol_regime"].value > 0, "LOW vol_regime should SUPPORT"

    def test_compressed_vol_regime_produces_support_vote(self):
        """COMPRESSED vol_regime should SUPPORT (existing behavior confirmation)."""
        from signals.ensemble_voter import ensemble_voter

        verdict = ensemble_voter.evaluate(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100_000.0,
            vol_regime="COMPRESSED",
        )

        votes = {v.source: v for v in verdict.votes}
        assert "vol_regime" in votes
        assert votes["vol_regime"].value > 0

    def test_neutral_vol_regime_does_not_vote(self):
        """NEUTRAL vol_regime should not produce a vote."""
        from signals.ensemble_voter import ensemble_voter

        verdict = ensemble_voter.evaluate(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100_000.0,
            vol_regime="NEUTRAL",
        )

        sources = {v.source for v in verdict.votes}
        assert "vol_regime" not in sources


# ════════════════════════════════════════════════════════════════════
# GAP 6: Adaptive weight attribution correctness (P4-1)
# ════════════════════════════════════════════════════════════════════

class TestAdaptiveWeightAttributionP4:
    """P4-1: Source absent from ensemble_votes should be skipped, not proxy-credited."""

    def test_absent_source_not_credited(self):
        """Row with ensemble_votes={cvd:...} should NOT credit mining_health."""
        from signals.adaptive_weights import AdaptiveWeightManager

        mgr = AdaptiveWeightManager()
        signals = [{
            "outcome": "WIN",
            "pnl_r": 2.0,
            "ensemble_votes": {
                "cvd": {"value": 1, "weight": 1.8},
            },
        }] * 10

        score = mgr._compute_source_score("mining_health", "BULL_TREND", signals)
        # mining_health was absent from every row — should have zero samples
        assert score.sample_count == 0

    def test_present_source_credited(self):
        """Row where the source IS in ensemble_votes should be credited normally."""
        from signals.adaptive_weights import AdaptiveWeightManager

        mgr = AdaptiveWeightManager()
        signals = [{
            "outcome": "WIN",
            "pnl_r": 1.5,
            "ensemble_votes": {
                "cvd": {"value": 1, "weight": 1.8},
                "mining_health": {"value": 1, "weight": 1.0},
            },
        }] * 10

        score = mgr._compute_source_score("mining_health", "BULL_TREND", signals)
        assert score.sample_count == 10
        assert score.win_rate == 1.0

    def test_legacy_row_still_uses_proxy(self):
        """Rows WITHOUT ensemble_votes should still fall back to proxy attribution."""
        from signals.adaptive_weights import AdaptiveWeightManager

        mgr = AdaptiveWeightManager()
        signals = [
            {"outcome": "WIN", "pnl_r": 1.5},
            {"outcome": "LOSS", "pnl_r": -1.0},
        ] * 5

        score = mgr._compute_source_score("cvd", "NEUTRAL", signals)
        # Legacy path: all 10 rows are proxy-attributed
        assert score.sample_count == 10

    def test_mixed_legacy_and_modern_rows(self):
        """Mix of legacy (no ensemble_votes) and modern rows handles both correctly."""
        from signals.adaptive_weights import AdaptiveWeightManager

        mgr = AdaptiveWeightManager()
        signals = [
            # Legacy row — no ensemble_votes, proxy fallback
            {"outcome": "WIN", "pnl_r": 1.0},
            # Modern row — cvd voted, mining_health absent
            {"outcome": "WIN", "pnl_r": 2.0, "ensemble_votes": {
                "cvd": {"value": 1, "weight": 1.8},
            }},
        ] * 5

        # cvd: legacy rows (proxy) + modern rows (present) → all 10
        cvd_score = mgr._compute_source_score("cvd", "NEUTRAL", signals)
        assert cvd_score.sample_count == 10

        # mining_health: legacy rows (proxy) + modern rows (absent → skip) → 5
        mh_score = mgr._compute_source_score("mining_health", "NEUTRAL", signals)
        assert mh_score.sample_count == 5
