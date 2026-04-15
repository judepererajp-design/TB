"""
Tests for the 5 GPT-identified final gaps:
  1. Signal Promotion Path (Block → Re-activate)
  2. Confidence Calibration Loop (Feedback → Threshold Adjustment)
  3. Signal Priority Ranking Integration
  4. Signal Decay Logic
  5. Higher-Order Semantic Guards (Macro Logic)
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from config.constants import ExecutionGate as EG


# ════════════════════════════════════════════════════════════════
# Gap 1: Signal Promotion Path
# ════════════════════════════════════════════════════════════════

class TestSignalPromotion(unittest.TestCase):
    """Test that blocked near-misses can be promoted when conditions improve."""

    def setUp(self):
        from analyzers.near_miss_tracker import NearMissTracker
        self.tracker = NearMissTracker(max_tracked=50)

    def test_promotable_returns_recent_near_misses(self):
        """Near-misses within age window should be promotable."""
        self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG", strategy="smc",
            execution_score=30.0, block_threshold=35.0,
            entry_price=50000.0, tp_price=51000.0, sl_price=49500.0,
        )
        candidates = self.tracker.get_promotable_near_misses(max_age_secs=3600)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].symbol, "BTCUSDT")

    def test_old_near_misses_not_promotable(self):
        """Near-misses older than max_age should be excluded."""
        record = self.tracker.record_near_miss(
            symbol="ETHUSDT", direction="SHORT", strategy="breakout",
            execution_score=28.0, block_threshold=35.0,
        )
        record.ts = time.time() - 7200  # 2 hours old
        candidates = self.tracker.get_promotable_near_misses(max_age_secs=1800)
        self.assertEqual(len(candidates), 0)

    def test_already_checked_not_promotable(self):
        """Near-misses with outcome_checked=True should be excluded."""
        record = self.tracker.record_near_miss(
            symbol="SOLUSDT", direction="LONG", strategy="momentum",
            execution_score=33.0, block_threshold=35.0,
        )
        record.outcome_checked = True
        candidates = self.tracker.get_promotable_near_misses(max_age_secs=3600)
        self.assertEqual(len(candidates), 0)

    def test_mark_promoted(self):
        """Promoted records should be excluded from future promotion."""
        record = self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG", strategy="smc",
            execution_score=30.0, block_threshold=35.0,
        )
        self.tracker.mark_promoted(record)
        candidates = self.tracker.get_promotable_near_misses(max_age_secs=3600)
        self.assertEqual(len(candidates), 0)

    def test_promotion_constants_exist(self):
        """Verify promotion constants are defined."""
        self.assertTrue(hasattr(EG, 'PROMOTION_ENABLED'))
        self.assertTrue(hasattr(EG, 'PROMOTION_SCORE_THRESHOLD'))
        self.assertTrue(hasattr(EG, 'PROMOTION_MAX_AGE_SECS'))
        self.assertTrue(hasattr(EG, 'PROMOTION_MAX_PER_CYCLE'))
        self.assertEqual(EG.PROMOTION_MAX_AGE_SECS, 1800)
        self.assertEqual(EG.PROMOTION_SCORE_THRESHOLD, 50.0)

    def test_multiple_promotable(self):
        """Multiple near-misses within window should all be returned."""
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            self.tracker.record_near_miss(
                symbol=sym, direction="LONG", strategy="smc",
                execution_score=31.0, block_threshold=35.0,
            )
        candidates = self.tracker.get_promotable_near_misses(max_age_secs=3600)
        self.assertEqual(len(candidates), 3)


# ════════════════════════════════════════════════════════════════
# Gap 2: Confidence Calibration Loop
# ════════════════════════════════════════════════════════════════

class TestConfidenceCalibration(unittest.TestCase):
    """Test that the feedback loop adjusts thresholds based on FNR."""

    def setUp(self):
        from analyzers.near_miss_tracker import NearMissTracker
        self.tracker = NearMissTracker(max_tracked=50)

    def test_insufficient_data_returns_zero(self):
        """With too few samples, no adjustment should be made."""
        adj = self.tracker.compute_threshold_adjustment()
        self.assertEqual(adj, 0.0)

    def test_high_fnr_loosens_threshold(self):
        """If >20% of blocks would have won, gate is too strict → loosen."""
        # Simulate 100 blocks with 25 would-have-won (25% FNR)
        self.tracker._total_blocks = 100
        self.tracker._would_have_won = 25
        self.tracker._would_have_lost = 50
        self.tracker._outcomes_checked = 75
        self.tracker._pass_wins = 10
        self.tracker._pass_losses = 5
        adj = self.tracker.compute_threshold_adjustment()
        self.assertLess(adj, 0, "Should loosen (negative adjustment)")
        self.assertEqual(adj, EG.CALIBRATION_LOOSEN_STEP)

    def test_low_fnr_high_plr_tightens(self):
        """If FNR is low but pass-loss-rate is high, gate is too loose → tighten."""
        self.tracker._total_blocks = 100
        self.tracker._would_have_won = 2   # 2% FNR (low)
        self.tracker._would_have_lost = 80
        self.tracker._outcomes_checked = 82
        self.tracker._pass_wins = 3
        self.tracker._pass_losses = 15     # 83% pass-loss-rate (high)
        adj = self.tracker.compute_threshold_adjustment()
        self.assertGreater(adj, 0, "Should tighten (positive adjustment)")
        self.assertEqual(adj, EG.CALIBRATION_TIGHTEN_STEP)

    def test_moderate_fnr_no_change(self):
        """If FNR is moderate and PLR is acceptable, no adjustment."""
        self.tracker._total_blocks = 100
        self.tracker._would_have_won = 10  # 10% FNR (moderate)
        self.tracker._would_have_lost = 60
        self.tracker._outcomes_checked = 70
        self.tracker._pass_wins = 20
        self.tracker._pass_losses = 8
        adj = self.tracker.compute_threshold_adjustment()
        self.assertEqual(adj, 0.0)

    def test_calibration_constants_exist(self):
        """Verify calibration constants are defined."""
        self.assertTrue(hasattr(EG, 'CALIBRATION_ENABLED'))
        self.assertTrue(hasattr(EG, 'CALIBRATION_FNR_HIGH'))
        self.assertTrue(hasattr(EG, 'CALIBRATION_FNR_LOW'))
        self.assertTrue(hasattr(EG, 'CALIBRATION_LOOSEN_STEP'))
        self.assertTrue(hasattr(EG, 'CALIBRATION_TIGHTEN_STEP'))

    def test_calibration_disabled_returns_zero(self):
        """When calibration is disabled, always return 0."""
        self.tracker._total_blocks = 100
        self.tracker._would_have_won = 30
        self.tracker._pass_wins = 5
        self.tracker._pass_losses = 5
        with patch.object(EG, 'CALIBRATION_ENABLED', False):
            adj = self.tracker.compute_threshold_adjustment()
        self.assertEqual(adj, 0.0)


# ════════════════════════════════════════════════════════════════
# Gap 3: Signal Priority Ranking
# ════════════════════════════════════════════════════════════════

class TestSignalRanking(unittest.TestCase):
    """Test that SignalRanker properly ranks and filters signals."""

    def _make_scored(self, symbol, direction, confidence, rr, volume_score):
        """Create a mock ScoredSignal."""
        from strategies.base import SignalDirection
        base = MagicMock()
        base.symbol = symbol
        base.direction = SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT
        base.rr_ratio = rr
        scored = MagicMock()
        scored.base_signal = base
        scored.final_confidence = confidence
        scored.volume_score = volume_score
        return scored

    def test_rank_returns_sorted_by_score(self):
        """Higher-scored signals should come first."""
        from signals.signal_ranker import SignalRanker
        ranker = SignalRanker()
        # Use uncorrelated symbols to avoid correlation filter
        signals = [
            self._make_scored("AVAXUSDT", "LONG", 60.0, 1.5, 40.0),
            self._make_scored("BTCUSDT", "LONG", 90.0, 3.0, 80.0),
            self._make_scored("DOGEUSDT", "LONG", 75.0, 2.0, 60.0),
        ]
        result = ranker.rank(signals, regime="BULL_TREND", max_signals=3)
        self.assertEqual(len(result), 3)
        # BTC should rank highest (highest confidence + RR + volume)
        self.assertEqual(result[0].base_signal.symbol, "BTCUSDT")

    def test_rank_limits_to_max_signals(self):
        """Only max_signals should be returned."""
        from signals.signal_ranker import SignalRanker
        ranker = SignalRanker()
        signals = [
            self._make_scored(f"SYM{i}USDT", "LONG", 70 + i, 2.0, 60.0)
            for i in range(10)
        ]
        result = ranker.rank(signals, regime="NEUTRAL", max_signals=3)
        self.assertEqual(len(result), 3)

    def test_correlation_filter_removes_duplicates(self):
        """Correlated same-direction signals should be filtered."""
        from signals.signal_ranker import SignalRanker
        ranker = SignalRanker()
        # BTC and ETH have 0.85 correlation — same direction should filter
        signals = [
            self._make_scored("BTCUSDT", "LONG", 90.0, 3.0, 80.0),
            self._make_scored("ETHUSDT", "LONG", 85.0, 2.5, 70.0),
        ]
        result = ranker.rank(signals, regime="BULL_TREND", max_signals=5)
        # ETH should be filtered (correlated with higher-ranked BTC, same dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].base_signal.symbol, "BTCUSDT")

    def test_opposite_direction_not_filtered(self):
        """Correlated symbols in opposite directions should not filter."""
        from signals.signal_ranker import SignalRanker
        ranker = SignalRanker()
        signals = [
            self._make_scored("BTCUSDT", "LONG", 90.0, 3.0, 80.0),
            self._make_scored("ETHUSDT", "SHORT", 85.0, 2.5, 70.0),
        ]
        result = ranker.rank(signals, regime="NEUTRAL", max_signals=5)
        self.assertEqual(len(result), 2)

    def test_empty_signals_returns_empty(self):
        """Empty input should return empty list."""
        from signals.signal_ranker import SignalRanker
        ranker = SignalRanker()
        result = ranker.rank([], regime="NEUTRAL")
        self.assertEqual(result, [])

    def test_regime_alignment_boosts_score(self):
        """Bull-regime should boost LONG signals over SHORTs."""
        from signals.signal_ranker import SignalRanker
        ranker = SignalRanker()
        # Same confidence/RR/volume, different directions
        signals = [
            self._make_scored("BNBUSDT", "SHORT", 80.0, 2.0, 60.0),
            self._make_scored("SOLUSDT", "LONG", 80.0, 2.0, 60.0),
        ]
        result = ranker.rank(signals, regime="BULL_TREND", max_signals=2)
        # LONG should rank higher in BULL regime
        self.assertEqual(result[0].base_signal.symbol, "SOLUSDT")


# ════════════════════════════════════════════════════════════════
# Gap 4: Signal Decay Logic
# ════════════════════════════════════════════════════════════════

class TestSignalDecay(unittest.TestCase):
    """Test that signal confidence degrades over time."""

    def test_no_decay_during_grace(self):
        """No penalty during the grace period."""
        from signals.signal_decay import compute_decay_penalty
        now = time.time()
        created = now - 300  # 5 minutes ago (within 15-min grace)
        penalty = compute_decay_penalty(created, now)
        self.assertEqual(penalty, 0.0)

    def test_decay_after_grace(self):
        """Penalty accumulates after grace period."""
        from signals.signal_decay import compute_decay_penalty
        now = time.time()
        # 1 hour 15 min ago → 1 hour past grace → 2% * 1 = 2.0
        created = now - (EG.DECAY_GRACE_PERIOD_SECS + 3600)
        penalty = compute_decay_penalty(created, now)
        self.assertAlmostEqual(penalty, EG.DECAY_RATE_PER_HOUR, places=1)

    def test_decay_max_cap(self):
        """Penalty should never exceed DECAY_MAX_PENALTY."""
        from signals.signal_decay import compute_decay_penalty
        now = time.time()
        # 24 hours past grace → 2% * 24 = 48%, but capped at 15%
        created = now - (EG.DECAY_GRACE_PERIOD_SECS + 86400)
        penalty = compute_decay_penalty(created, now)
        self.assertEqual(penalty, EG.DECAY_MAX_PENALTY)

    def test_apply_decay_clamps_to_floor(self):
        """Decayed confidence should not go below DECAY_MIN_CONFIDENCE."""
        from signals.signal_decay import apply_decay
        now = time.time()
        created = now - (EG.DECAY_GRACE_PERIOD_SECS + 86400)  # Max decay
        result = apply_decay(55.0, created, now)
        self.assertGreaterEqual(result, EG.DECAY_MIN_CONFIDENCE)

    def test_apply_decay_no_penalty_when_fresh(self):
        """Fresh signals should keep full confidence."""
        from signals.signal_decay import apply_decay
        now = time.time()
        result = apply_decay(85.0, now - 60, now)
        self.assertEqual(result, 85.0)

    def test_decay_disabled_returns_zero(self):
        """When decay is disabled, no penalty applied."""
        from signals.signal_decay import compute_decay_penalty
        now = time.time()
        created = now - 7200
        with patch.object(EG, 'DECAY_ENABLED', False):
            penalty = compute_decay_penalty(created, now)
        self.assertEqual(penalty, 0.0)

    def test_zero_timestamp_returns_zero(self):
        """Invalid timestamp should return 0 penalty."""
        from signals.signal_decay import compute_decay_penalty
        penalty = compute_decay_penalty(0, time.time())
        self.assertEqual(penalty, 0.0)

    def test_decay_constants_exist(self):
        """Verify decay constants are defined."""
        self.assertTrue(hasattr(EG, 'DECAY_ENABLED'))
        self.assertTrue(hasattr(EG, 'DECAY_GRACE_PERIOD_SECS'))
        self.assertTrue(hasattr(EG, 'DECAY_RATE_PER_HOUR'))
        self.assertTrue(hasattr(EG, 'DECAY_MAX_PENALTY'))
        self.assertTrue(hasattr(EG, 'DECAY_MIN_CONFIDENCE'))

    def test_progressive_decay(self):
        """Penalty should increase with time."""
        from signals.signal_decay import compute_decay_penalty
        now = time.time()
        p1 = compute_decay_penalty(now - (EG.DECAY_GRACE_PERIOD_SECS + 1800), now)  # 30 min past grace
        p2 = compute_decay_penalty(now - (EG.DECAY_GRACE_PERIOD_SECS + 3600), now)  # 1h past grace
        p3 = compute_decay_penalty(now - (EG.DECAY_GRACE_PERIOD_SECS + 7200), now)  # 2h past grace
        self.assertLess(p1, p2)
        self.assertLess(p2, p3)


# ════════════════════════════════════════════════════════════════
# Gap 5: Higher-Order Semantic Guards
# ════════════════════════════════════════════════════════════════

class TestMacroSemanticGuards(unittest.TestCase):
    """Test HTF trend-direction and volatility regime semantic kills."""

    def _gate(self):
        from analyzers.execution_gate import ExecutionQualityGate
        return ExecutionQualityGate()

    def test_short_in_bull_trend_no_reversal_blocked(self):
        """SHORT in BULL_TREND without bearish reversal signal → BLOCK."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="SHORT",
            setup_context={
                "structure": {"trend": "BULL_TREND", "choch": False, "choch_direction": ""},
                "pattern": {"wyckoff_phase": None, "utad_detected": False, "spring_detected": False},
                "location": {},
            },
        )
        self.assertIn("HTF_TREND", kill)
        self.assertIn("BULL", kill)

    def test_short_in_bull_trend_with_choch_passes(self):
        """SHORT in BULL_TREND with bearish CHoCH → should pass."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="SHORT",
            setup_context={
                "structure": {"trend": "BULL_TREND", "choch": True, "choch_direction": "BEARISH"},
                "pattern": {"wyckoff_phase": None, "utad_detected": False, "spring_detected": False},
                "location": {},
            },
        )
        # BEARISH CHoCH should actually trigger the existing CHoCH kill for this combo
        # But the HTF trend guard should NOT fire because there IS a reversal signal
        # The CHoCH guard fires first (direction=SHORT + choch_direction=BEARISH → no kill)
        # Wait — SHORT + BEARISH CHoCH is aligned, not contradictory
        self.assertEqual(kill, "", "SHORT with bearish CHoCH should pass trend guard")

    def test_long_in_bear_trend_no_reversal_blocked(self):
        """LONG in BEAR_TREND without bullish reversal → BLOCK."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="LONG",
            setup_context={
                "structure": {"trend": "BEAR_TREND", "choch": False, "choch_direction": ""},
                "pattern": {"wyckoff_phase": None, "utad_detected": False, "spring_detected": False},
                "location": {},
            },
        )
        self.assertIn("HTF_TREND", kill)
        self.assertIn("BEAR", kill)

    def test_long_in_bear_with_spring_passes(self):
        """LONG in BEAR_TREND with Wyckoff spring → should pass."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="LONG",
            setup_context={
                "structure": {"trend": "BEAR_TREND", "choch": False, "choch_direction": ""},
                "pattern": {"wyckoff_phase": "accumulation", "spring_detected": True, "utad_detected": False},
                "location": {},
            },
        )
        # Accumulation is a bullish Wyckoff phase, but the existing Wyckoff guard
        # catches LONG in bearish phases. Here we're in a bullish phase (accumulation)
        # WITH spring_detected → that's a reversal signal → HTF trend guard should not fire
        self.assertEqual(kill, "", "LONG with spring in accumulation should pass trend guard")

    def test_long_in_bull_trend_passes(self):
        """LONG in BULL_TREND → no semantic kill."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="LONG",
            setup_context={
                "structure": {"trend": "BULL_TREND", "choch": False, "choch_direction": ""},
                "pattern": {"wyckoff_phase": None, "utad_detected": False, "spring_detected": False},
                "location": {},
            },
        )
        self.assertEqual(kill, "")

    def test_scalp_in_extreme_volatility_blocked(self):
        """Scalp setup in EXTREME volatility → BLOCK."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="LONG",
            setup_context={
                "structure": {"trend": "NEUTRAL"},
                "pattern": {},
                "location": {"setup_class": "scalp"},
            },
            execution_context={
                "market": {"volatility_regime": "EXTREME"},
            },
        )
        self.assertIn("SCALP", kill)
        self.assertIn("EXTREME", kill)

    def test_intraday_in_extreme_volatility_passes(self):
        """Non-scalp setup in EXTREME volatility → should pass vol guard."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="LONG",
            setup_context={
                "structure": {"trend": "NEUTRAL"},
                "pattern": {},
                "location": {"setup_class": "intraday"},
            },
            execution_context={
                "market": {"volatility_regime": "EXTREME"},
            },
        )
        self.assertEqual(kill, "")

    def test_scalp_in_normal_volatility_passes(self):
        """Scalp in normal volatility → no vol guard kill."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="LONG",
            setup_context={
                "structure": {"trend": "NEUTRAL"},
                "pattern": {},
                "location": {"setup_class": "scalp"},
            },
            execution_context={
                "market": {"volatility_regime": "NORMAL"},
            },
        )
        self.assertEqual(kill, "")

    def test_trend_guard_disabled(self):
        """When SEMANTIC_TREND_GUARD is False, trend kills should not fire."""
        from analyzers.execution_gate import ExecutionQualityGate
        with patch.object(EG, 'SEMANTIC_TREND_GUARD', False):
            kill = ExecutionQualityGate._check_semantic_kills(
                direction="SHORT",
                setup_context={
                    "structure": {"trend": "BULL_TREND", "choch": False, "choch_direction": ""},
                    "pattern": {},
                    "location": {},
                },
            )
        self.assertEqual(kill, "")

    def test_vol_guard_disabled(self):
        """When SEMANTIC_VOLATILITY_GUARD is False, vol kills should not fire."""
        from analyzers.execution_gate import ExecutionQualityGate
        with patch.object(EG, 'SEMANTIC_VOLATILITY_GUARD', False):
            kill = ExecutionQualityGate._check_semantic_kills(
                direction="LONG",
                setup_context={
                    "structure": {"trend": "NEUTRAL"},
                    "pattern": {},
                    "location": {"setup_class": "scalp"},
                },
                execution_context={
                    "market": {"volatility_regime": "EXTREME"},
                },
            )
        self.assertEqual(kill, "")

    def test_semantic_constants_exist(self):
        """Verify macro semantic guard constants are defined."""
        self.assertTrue(hasattr(EG, 'SEMANTIC_TREND_GUARD'))
        self.assertTrue(hasattr(EG, 'SEMANTIC_VOLATILITY_GUARD'))
        self.assertTrue(hasattr(EG, 'SEMANTIC_EXTREME_VOL_LABELS'))

    def test_existing_choch_kills_still_work(self):
        """Existing CHoCH semantic kills should still fire."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="SHORT",
            setup_context={
                "structure": {"choch": True, "choch_direction": "BULLISH", "trend": "NEUTRAL"},
                "pattern": {},
            },
        )
        self.assertIn("BULLISH_CHOCH", kill)

    def test_existing_wyckoff_kills_still_work(self):
        """Existing Wyckoff semantic kills should still fire."""
        from analyzers.execution_gate import ExecutionQualityGate
        kill = ExecutionQualityGate._check_semantic_kills(
            direction="SHORT",
            setup_context={
                "structure": {"choch": False, "choch_direction": "", "trend": "NEUTRAL"},
                "pattern": {"wyckoff_phase": "accumulation", "spring_detected": False, "utad_detected": False},
                "location": {},
            },
        )
        self.assertIn("BULLISH_WYCKOFF", kill)


# ════════════════════════════════════════════════════════════════
# Integration: Execution Gate with New Semantic Guards
# ════════════════════════════════════════════════════════════════

class TestExecutionGateWithMacroGuards(unittest.TestCase):
    """Ensure full evaluate() integration picks up new semantic kills."""

    def test_evaluate_blocks_short_in_bull_trend(self):
        """Full evaluate() should BLOCK SHORT in BULL_TREND without reversal."""
        from analyzers.execution_gate import ExecutionQualityGate
        gate = ExecutionQualityGate()
        result = gate.evaluate(
            direction="SHORT",
            grade="B",
            confidence=80.0,
            symbol="BTCUSDT",
            session_name="London Open",
            is_killzone=True,
            trigger_quality_score=0.8,
            spread_bps=5.0,
            whale_aligned=True,
            whale_buy_ratio=0.5,
            eq_zone="premium",
            volume_context="NORMAL",
            volume_score=70.0,
            setup_context={
                "structure": {"trend": "BULL_TREND", "choch": False, "choch_direction": ""},
                "pattern": {"wyckoff_phase": None, "spring_detected": False, "utad_detected": False},
                "location": {},
            },
        )
        self.assertTrue(result.should_block)
        self.assertIn("HTF_TREND", result.kill_combo)

    def test_evaluate_passes_long_in_bull_trend(self):
        """Full evaluate() should PASS LONG in BULL_TREND with good factors."""
        from analyzers.execution_gate import ExecutionQualityGate
        gate = ExecutionQualityGate()
        result = gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=80.0,
            symbol="BTCUSDT",
            session_name="London Open",
            is_killzone=True,
            trigger_quality_score=0.8,
            spread_bps=5.0,
            whale_aligned=True,
            whale_buy_ratio=0.7,
            eq_zone="discount",
            volume_context="ABOVE_AVG",
            volume_score=70.0,
            setup_context={
                "structure": {"trend": "BULL_TREND", "choch": False, "choch_direction": ""},
                "pattern": {"wyckoff_phase": None, "spring_detected": False, "utad_detected": False},
                "location": {},
            },
        )
        self.assertFalse(result.should_block)


if __name__ == "__main__":
    unittest.main()
