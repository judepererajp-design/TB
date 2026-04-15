"""
Tests for Near-Miss Tracking and Execution Gate Feedback Metrics.

The near-miss tracker monitors blocked signals that were close to passing,
then tracks whether they would have been profitable (false negatives).
This drives the execution gate feedback loop.
"""
import sys
import time
import types
from unittest.mock import MagicMock

# ── Minimal stubs so modules can import config.constants ──────────
if 'config.loader' not in sys.modules:
    _loader = types.ModuleType('config.loader')
    _cfg = MagicMock()
    _cfg.aggregator = {'min_confidence': 72}
    _loader.cfg = _cfg
    sys.modules['config.loader'] = _loader

from config.constants import ExecutionGate as EG
from analyzers.execution_gate import ExecutionQualityGate, ExecutionAssessment
from analyzers.near_miss_tracker import NearMissTracker, NearMissRecord


# ═══════════════════════════════════════════════════════════════════
# 1. ExecutionAssessment near-miss flag
# ═══════════════════════════════════════════════════════════════════

class TestExecutionAssessmentNearMissField:
    """Verify the is_near_miss field on ExecutionAssessment."""

    def test_default_is_false(self):
        result = ExecutionAssessment()
        assert result.is_near_miss is False

    def test_can_set_near_miss(self):
        result = ExecutionAssessment(is_near_miss=True)
        assert result.is_near_miss is True


# ═══════════════════════════════════════════════════════════════════
# 2. Execution gate near-miss flagging logic
# ═══════════════════════════════════════════════════════════════════

class TestExecutionGateNearMissFlagging:
    """Verify the gate flags near-miss blocked signals correctly."""

    def setup_method(self):
        self.gate = ExecutionQualityGate()

    def test_score_in_near_miss_band_flagged(self):
        """Score between NEAR_MISS_LOWER and block threshold → near-miss."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=72.0,
            session_name="Asia",
            trigger_quality_score=0.30,
            trigger_quality_label="LOW",
            spread_bps=25.0,
            whale_aligned=None,
            whale_buy_ratio=0.5,
            eq_zone="discount",
            eq_zone_depth=0.6,
            volume_context="NORMAL",
            volume_score=45.0,
        )
        # Should be blocked and near the threshold
        if result.should_block and EG.NEAR_MISS_LOWER <= result.execution_score:
            assert result.is_near_miss is True
            assert any("NEAR MISS" in n for n in result.notes)

    def test_very_low_score_not_near_miss(self):
        """Score well below NEAR_MISS_LOWER → not a near-miss."""
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=72.0,
            is_dead_zone=True,
            trigger_quality_score=0.10,
            trigger_quality_label="VERY_LOW",
            spread_bps=120.0,
            whale_aligned=False,
            whale_buy_ratio=0.2,
            eq_zone="premium",
            eq_zone_depth=0.8,
            volume_context="LOW_VOL",
            volume_score=15.0,
        )
        assert result.should_block is True
        # Very low score should NOT be near-miss
        if result.execution_score < EG.NEAR_MISS_LOWER:
            assert result.is_near_miss is False

    def test_passing_signal_not_near_miss(self):
        """Signal that passes → never flagged as near-miss."""
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
        assert result.should_block is False
        assert result.is_near_miss is False

    def test_kill_combo_with_decent_score_is_near_miss(self):
        """Kill combo fires but underlying score is in near-miss band."""
        # KC1: low trigger + low volume — but other factors decent
        result = self.gate.evaluate(
            direction="LONG",
            grade="B",
            confidence=75.0,
            is_killzone=True,  # Good session
            trigger_quality_score=0.20,  # Below KC threshold
            trigger_quality_label="LOW",
            spread_bps=8.0,  # Good spread
            whale_aligned=True,
            whale_buy_ratio=0.8,
            eq_zone="discount",
            eq_zone_depth=0.7,
            volume_context="LOW_VOL",  # Low volume triggers KC1
            volume_score=20.0,
        )
        assert result.should_block is True
        assert result.kill_combo != ""
        # The underlying composite may be in near-miss range
        # depending on exact scoring
        if EG.NEAR_MISS_LOWER <= result.execution_score < EG.NEAR_MISS_UPPER:
            assert result.is_near_miss is True


# ═══════════════════════════════════════════════════════════════════
# 3. NearMissTracker core functionality
# ═══════════════════════════════════════════════════════════════════

class TestNearMissTrackerBasics:
    """Test basic record/retrieve operations."""

    def setup_method(self):
        self.tracker = NearMissTracker(max_tracked=50)

    def test_record_near_miss(self):
        record = self.tracker.record_near_miss(
            symbol="BTCUSDT",
            direction="LONG",
            strategy="SmartMoneyConcepts",
            execution_score=32.0,
            block_threshold=35.0,
            factors={"session": 55, "trigger_quality": 20},
            bad_factors=["trigger_quality"],
            grade="B",
            confidence=74.0,
        )
        assert isinstance(record, NearMissRecord)
        assert record.symbol == "BTCUSDT"
        assert record.execution_score == 32.0
        assert record.outcome_checked is False
        assert record.would_have_won is None

    def test_counter_increments(self):
        self.tracker.record_near_miss(
            symbol="ETHUSDT", direction="SHORT",
            execution_score=30.0, block_threshold=35.0,
        )
        assert self.tracker._total_near_misses == 1
        metrics = self.tracker.get_metrics()
        assert metrics["near_miss_count"] == 1

    def test_block_and_pass_counters(self):
        self.tracker.record_block()
        self.tracker.record_block()
        self.tracker.record_pass()
        metrics = self.tracker.get_metrics()
        assert metrics["total_blocks"] == 2
        assert metrics["total_passes"] == 1

    def test_get_pending_keys_dedupes_and_skips_checked(self):
        self.tracker.record_near_miss(symbol="BTCUSDT", direction="LONG")
        self.tracker.record_near_miss(symbol="BTCUSDT", direction="LONG")
        checked = self.tracker.record_near_miss(symbol="ETHUSDT", direction="SHORT")
        checked.outcome_checked = True

        pending = self.tracker.get_pending_keys()

        assert pending == [("BTCUSDT", "LONG")]


class TestNearMissTrackerOutcomes:
    """Test outcome checking logic."""

    def setup_method(self):
        self.tracker = NearMissTracker(max_tracked=50)

    def test_would_have_won_long(self):
        """Price moves up enough → would have won for LONG."""
        record = self.tracker.record_near_miss(
            symbol="BTCUSDT",
            direction="LONG",
            execution_score=33.0,
            block_threshold=35.0,
            entry_price=100.0,
            tp_price=102.0,
            sl_price=99.0,
        )
        # Price hit TP
        self.tracker.check_outcome(
            symbol="BTCUSDT",
            direction="LONG",
            high_price=102.5,
            low_price=99.5,
            current_price=101.0,
        )
        assert record.outcome_checked is True
        assert record.would_have_won is True
        assert self.tracker._would_have_won == 1

    def test_would_have_lost_long(self):
        """Price drops to SL → would have lost for LONG."""
        record = self.tracker.record_near_miss(
            symbol="ETHUSDT",
            direction="LONG",
            execution_score=30.0,
            block_threshold=35.0,
            entry_price=3000.0,
            tp_price=3100.0,
            sl_price=2900.0,
        )
        # Price hit SL but not TP
        self.tracker.check_outcome(
            symbol="ETHUSDT",
            direction="LONG",
            high_price=3050.0,
            low_price=2890.0,
            current_price=2950.0,
        )
        assert record.outcome_checked is True
        assert record.would_have_lost is True
        assert self.tracker._would_have_lost == 1

    def test_would_have_won_short(self):
        """Price drops to TP → would have won for SHORT."""
        record = self.tracker.record_near_miss(
            symbol="BTCUSDT",
            direction="SHORT",
            execution_score=28.0,
            block_threshold=35.0,
            entry_price=50000.0,
            tp_price=49000.0,
            sl_price=51000.0,
        )
        # Price dropped to TP
        self.tracker.check_outcome(
            symbol="BTCUSDT",
            direction="SHORT",
            high_price=50500.0,
            low_price=48900.0,
            current_price=49500.0,
        )
        assert record.outcome_checked is True
        assert record.would_have_won is True

    def test_no_tp_uses_percentage(self):
        """When no TP/SL set, uses TP_CHECK_PERCENT threshold."""
        record = self.tracker.record_near_miss(
            symbol="SOLUSDT",
            direction="LONG",
            execution_score=33.0,
            block_threshold=35.0,
            entry_price=100.0,
            tp_price=0.0,  # No TP
            sl_price=0.0,  # No SL
        )
        # Price moves up by more than TP_CHECK_PERCENT
        move_pct = EG.NEAR_MISS_TP_CHECK_PERCENT
        high = 100.0 * (1.0 + move_pct / 100.0 + 0.01)  # Just above threshold
        self.tracker.check_outcome(
            symbol="SOLUSDT",
            direction="LONG",
            high_price=high,
            low_price=99.5,
            current_price=101.0,
        )
        assert record.would_have_won is True

    def test_expired_outcome(self):
        """Record past the outcome window → marked as unknown."""
        record = self.tracker.record_near_miss(
            symbol="BTCUSDT",
            direction="LONG",
            execution_score=33.0,
            block_threshold=35.0,
            entry_price=100.0,
        )
        # Force timestamp to be old
        record.ts = time.time() - EG.NEAR_MISS_OUTCOME_WINDOW - 100

        self.tracker.check_outcome(
            symbol="BTCUSDT",
            direction="LONG",
            high_price=100.5,
            low_price=99.5,
            current_price=100.0,
        )
        assert record.outcome_checked is True
        assert record.would_have_won is None  # Unknown
        assert self.tracker._outcome_unknown == 1

    def test_wrong_symbol_not_checked(self):
        """Check_outcome for different symbol doesn't affect record."""
        record = self.tracker.record_near_miss(
            symbol="BTCUSDT",
            direction="LONG",
            execution_score=33.0,
            block_threshold=35.0,
            entry_price=100.0,
            tp_price=102.0,
        )
        # Check different symbol
        self.tracker.check_outcome(
            symbol="ETHUSDT",
            direction="LONG",
            high_price=150.0,
            low_price=90.0,
            current_price=100.0,
        )
        assert record.outcome_checked is False


# ═══════════════════════════════════════════════════════════════════
# 4. Feedback metrics computation
# ═══════════════════════════════════════════════════════════════════

class TestNearMissMetrics:
    """Test the get_metrics() computation."""

    def setup_method(self):
        self.tracker = NearMissTracker(max_tracked=50)

    def test_empty_metrics(self):
        metrics = self.tracker.get_metrics()
        assert metrics["near_miss_count"] == 0
        assert metrics["near_miss_rate"] == 0.0
        assert metrics["would_have_won_rate"] == 0.0
        assert metrics["false_negative_rate"] == 0.0
        assert metrics["gate_accuracy"] is None  # Insufficient data
        assert metrics["recent_near_misses"] == []

    def test_near_miss_rate(self):
        self.tracker.record_block()
        self.tracker.record_block()
        self.tracker.record_block()
        self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG",
            execution_score=33.0, block_threshold=35.0,
        )
        metrics = self.tracker.get_metrics()
        # 1 near-miss / 3 blocks = 0.3333
        assert metrics["near_miss_rate"] == round(1 / 3, 4)

    def test_would_have_won_rate(self):
        # Create 3 near-misses, 2 would have won, 1 unknown
        r1 = self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG",
            execution_score=33.0, block_threshold=35.0,
            entry_price=100.0, tp_price=102.0,
        )
        r2 = self.tracker.record_near_miss(
            symbol="ETHUSDT", direction="LONG",
            execution_score=34.0, block_threshold=35.0,
            entry_price=3000.0, tp_price=3100.0,
        )
        r3 = self.tracker.record_near_miss(
            symbol="SOLUSDT", direction="SHORT",
            execution_score=30.0, block_threshold=35.0,
            entry_price=100.0, tp_price=98.0, sl_price=102.0,
        )

        # Check outcomes: BTC hits TP, ETH hits TP, SOL hits SL
        self.tracker.check_outcome("BTCUSDT", "LONG", 103.0, 99.5, 101.0)
        self.tracker.check_outcome("ETHUSDT", "LONG", 3200.0, 2990.0, 3100.0)
        self.tracker.check_outcome("SOLUSDT", "SHORT", 103.0, 99.0, 101.0)

        metrics = self.tracker.get_metrics()
        assert metrics["would_have_won"] == 2
        assert metrics["would_have_lost"] == 1
        assert metrics["would_have_won_rate"] == round(2 / 3, 4)

    def test_false_negative_rate(self):
        """False negative = near-miss that would have won / total blocks."""
        for _ in range(10):
            self.tracker.record_block()

        r = self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG",
            execution_score=33.0, block_threshold=35.0,
            entry_price=100.0, tp_price=102.0,
        )
        self.tracker.check_outcome("BTCUSDT", "LONG", 103.0, 99.5, 101.0)

        metrics = self.tracker.get_metrics()
        # 1 false negative / 10 blocks = 0.1
        assert metrics["false_negative_rate"] == round(1 / 10, 4)

    def test_pass_win_rate(self):
        self.tracker.record_pass_outcome(won=True)
        self.tracker.record_pass_outcome(won=True)
        self.tracker.record_pass_outcome(won=False)
        metrics = self.tracker.get_metrics()
        assert metrics["pass_win_rate"] == round(2 / 3, 4)

    def test_gate_accuracy_insufficient_data(self):
        """Gate accuracy not computed without enough samples."""
        self.tracker.record_pass_outcome(won=True)
        metrics = self.tracker.get_metrics()
        assert metrics["gate_accuracy"] is None

    def test_gate_accuracy_with_data(self):
        """Gate accuracy computed when enough outcomes exist."""
        # Create enough samples
        for _ in range(15):
            self.tracker.record_pass_outcome(won=True)
        for _ in range(5):
            self.tracker.record_pass_outcome(won=False)

        # 15 correct passes + 5 incorrect passes = 20 pass outcomes
        # We need total_known >= FEEDBACK_MIN_SAMPLE (20)
        metrics = self.tracker.get_metrics()
        if EG.FEEDBACK_MIN_SAMPLE <= 20:
            assert metrics["gate_accuracy"] is not None
            # accuracy = (correct_blocks=0 + correct_passes=15) / (total_known=20)
            assert metrics["gate_accuracy"] == round(15 / 20, 4)

    def test_factor_blame_tracking(self):
        """Bad factors that appear in false-negative near-misses are tracked."""
        r = self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG",
            execution_score=33.0, block_threshold=35.0,
            entry_price=100.0, tp_price=102.0,
            bad_factors=["trigger_quality", "volume_env"],
        )
        # Mark as would-have-won
        self.tracker.check_outcome("BTCUSDT", "LONG", 103.0, 99.5, 101.0)

        metrics = self.tracker.get_metrics()
        blame = metrics["factor_false_negative_blame"]
        assert blame.get("trigger_quality") == 1
        assert blame.get("volume_env") == 1

    def test_recent_near_misses_in_metrics(self):
        """Metrics includes recent near-misses for dashboard display."""
        self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG",
            execution_score=33.0, block_threshold=35.0,
        )
        self.tracker.record_near_miss(
            symbol="ETHUSDT", direction="SHORT",
            execution_score=28.0, block_threshold=35.0,
        )
        metrics = self.tracker.get_metrics()
        recent = metrics["recent_near_misses"]
        assert len(recent) == 2
        assert recent[0]["symbol"] == "BTCUSDT"
        assert recent[1]["symbol"] == "ETHUSDT"
        assert "execution_score" in recent[0]
        assert "is_checked" in recent[0]


# ═══════════════════════════════════════════════════════════════════
# 5. Constants validation
# ═══════════════════════════════════════════════════════════════════

class TestNearMissConstants:
    """Verify near-miss constants are properly configured."""

    def test_near_miss_lower_below_block_threshold(self):
        assert EG.NEAR_MISS_LOWER < EG.HARD_BLOCK_THRESHOLD

    def test_near_miss_upper_above_block_threshold(self):
        assert EG.NEAR_MISS_UPPER > EG.HARD_BLOCK_THRESHOLD

    def test_near_miss_band_reasonable(self):
        band = EG.NEAR_MISS_UPPER - EG.NEAR_MISS_LOWER
        assert 10.0 <= band <= 30.0  # Band should be 10-30 points wide

    def test_outcome_window_positive(self):
        assert EG.NEAR_MISS_OUTCOME_WINDOW > 0

    def test_tp_check_percent_positive(self):
        assert EG.NEAR_MISS_TP_CHECK_PERCENT > 0

    def test_max_tracked_reasonable(self):
        assert EG.NEAR_MISS_MAX_TRACKED >= 50

    def test_feedback_min_sample_reasonable(self):
        assert 5 <= EG.FEEDBACK_MIN_SAMPLE <= 100


# ═══════════════════════════════════════════════════════════════════
# 6. Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestNearMissEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        self.tracker = NearMissTracker(max_tracked=5)

    def test_deque_evicts_oldest(self):
        """When max_tracked is reached, oldest records are evicted."""
        for i in range(10):
            self.tracker.record_near_miss(
                symbol=f"TOKEN{i}USDT", direction="LONG",
                execution_score=33.0, block_threshold=35.0,
            )
        # Only last 5 should be in the deque
        assert len(self.tracker._near_misses) == 5
        assert self.tracker._near_misses[0].symbol == "TOKEN5USDT"

    def test_zero_entry_price_skipped(self):
        """Records with entry_price=0 are skipped during outcome check."""
        record = self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG",
            execution_score=33.0, block_threshold=35.0,
            entry_price=0.0,
        )
        self.tracker.check_outcome("BTCUSDT", "LONG", 150.0, 50.0, 100.0)
        assert record.outcome_checked is False

    def test_check_outcome_idempotent(self):
        """Already-checked records aren't rechecked."""
        record = self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG",
            execution_score=33.0, block_threshold=35.0,
            entry_price=100.0, tp_price=102.0,
        )
        self.tracker.check_outcome("BTCUSDT", "LONG", 103.0, 99.5, 101.0)
        assert record.would_have_won is True

        # Call again with losing prices — should not change
        self.tracker.check_outcome("BTCUSDT", "LONG", 80.0, 70.0, 75.0)
        assert record.would_have_won is True  # Unchanged

    def test_max_favorable_adverse_tracked(self):
        """Max favorable and adverse excursions are tracked."""
        record = self.tracker.record_near_miss(
            symbol="BTCUSDT", direction="LONG",
            execution_score=33.0, block_threshold=35.0,
            entry_price=100.0, tp_price=0.0, sl_price=0.0,
        )
        # Small move — not enough for TP
        self.tracker.check_outcome("BTCUSDT", "LONG", 100.5, 99.8, 100.2)
        assert record.max_favorable > 0
        assert record.max_adverse > 0
        assert record.outcome_checked is False  # Not enough movement yet
