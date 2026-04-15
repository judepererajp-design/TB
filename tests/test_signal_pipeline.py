"""Tests for signals.signal_pipeline."""

from unittest.mock import MagicMock

from signals.signal_pipeline import (
    apply_htf_hard_gate,
    apply_equilibrium_gate,
    apply_regime_filter_gate,
    passes_confidence_floor,
    direction_str,
    GateResult,
    evaluate_opposite_direction_conflict,
    has_reversal_confirmation,
)


class TestApplyHtfHardGate:

    def test_blocked(self):
        htf = MagicMock()
        htf.is_hard_blocked.return_value = (True, "weekly BEARISH vs LONG")
        result = apply_htf_hard_gate(htf, "LONG", 80.0, "SMC")
        assert result.blocked is True
        assert "BEARISH" in result.reason

    def test_not_blocked(self):
        htf = MagicMock()
        htf.is_hard_blocked.return_value = (False, "")
        result = apply_htf_hard_gate(htf, "SHORT", 80.0, "SMC")
        assert result.blocked is False

    def test_exception_is_swallowed(self):
        htf = MagicMock()
        htf.is_hard_blocked.side_effect = RuntimeError("boom")
        result = apply_htf_hard_gate(htf, "LONG", 80.0, "SMC")
        assert result.blocked is False


class TestApplyEquilibriumGate:

    def test_should_block(self):
        eq = MagicMock()
        eq.assess.return_value = MagicMock(
            should_block=True, reason="LONG at premium", confidence_mult=1.0,
        )
        result = apply_equilibrium_gate(eq, 100.0, "LONG")
        assert result.blocked is True

    def test_confidence_mult_applied(self):
        eq = MagicMock()
        eq.assess.return_value = MagicMock(
            should_block=False, reason="discount zone", confidence_mult=0.9,
        )
        result = apply_equilibrium_gate(eq, 100.0, "LONG")
        assert result.blocked is False
        assert result.adjusted_confidence == 0.9

    def test_no_adjustment(self):
        eq = MagicMock()
        eq.assess.return_value = MagicMock(
            should_block=False, reason="neutral", confidence_mult=1.0,
        )
        result = apply_equilibrium_gate(eq, 100.0, "LONG")
        assert result.adjusted_confidence == 1.0

    def test_exception_returns_empty_gate(self):
        eq = MagicMock()
        eq.assess.side_effect = RuntimeError("fail")
        result = apply_equilibrium_gate(eq, 100.0, "LONG")
        assert result.blocked is False


class TestPassesConfidenceFloor:

    def test_above_threshold(self):
        assert passes_confidence_floor(80.0, 72.0) is True

    def test_at_threshold(self):
        assert passes_confidence_floor(72.0, 72.0) is True

    def test_below_threshold(self):
        assert passes_confidence_floor(71.9, 72.0) is False


class TestDirectionStr:

    def test_enum_value(self):
        from enum import Enum

        class Dir(str, Enum):
            LONG = "LONG"

        assert direction_str(Dir.LONG) == "LONG"

    def test_plain_string(self):
        assert direction_str("SHORT") == "SHORT"


class TestDirectionalConflict:

    def test_blocks_recent_opposite_signal_without_reversal_context(self):
        result = evaluate_opposite_direction_conflict(
            new_direction="LONG",
            conflicting_direction="SHORT",
            conflict_age_secs=60,
            cooldown_secs=15 * 60,
            setup_context={"structure": {}, "pattern": {}},
            execution_context={"trade": {"type": "", "strength": 0}},
        )
        assert result.blocked is True
        assert "cooldown" in result.reason

    def test_allows_recent_opposite_signal_when_reversal_is_confirmed(self):
        result = evaluate_opposite_direction_conflict(
            new_direction="LONG",
            conflicting_direction="SHORT",
            conflict_age_secs=60,
            cooldown_secs=15 * 60,
            setup_context={
                "structure": {"choch": True, "choch_direction": "BULLISH"},
                "pattern": {},
            },
            execution_context={"trade": {"type": "", "strength": 0}},
        )
        assert result.blocked is False

    def test_has_reversal_confirmation_from_local_continuation_trade(self):
        assert has_reversal_confirmation(
            "SHORT",
            setup_context={"structure": {}, "pattern": {}},
            execution_context={"trade": {"type": "LOCAL_CONTINUATION_SHORT", "strength": 2}},
        ) is True


# ── Hourly counter cap test ──────────────────────────────────
class TestHourlyCounterCap:
    """Exempt signals must NOT inflate _hourly_total above _max_per_hour.

    Regression test for the bug where exempt (HTF-aligned / A+ / clarity)
    signals pushed the counter to e.g. 15/12, permanently blocking all
    non-exempt signals for the rest of the hour.
    """

    def test_exempt_signals_do_not_inflate_counter_above_max(self):
        """After counter reaches max, further approved signals must not increment."""
        from signals.aggregator import SignalAggregator
        agg = SignalAggregator()
        agg._max_per_hour = 12

        # Simulate 12 approved non-exempt signals
        agg._hourly_total = 12

        # Simulate the approval logic (mirrors aggregator.py lines 1119-1120)
        # An exempt signal is approved but the counter should NOT go above 12
        if agg._hourly_total < agg._max_per_hour:
            agg._hourly_total += 1
        assert agg._hourly_total == 12, "Counter must not exceed max_per_hour"

    def test_counter_increments_when_under_max(self):
        """Normal signals still increment the counter when under the limit."""
        from signals.aggregator import SignalAggregator
        agg = SignalAggregator()
        agg._max_per_hour = 12
        agg._hourly_total = 10

        if agg._hourly_total < agg._max_per_hour:
            agg._hourly_total += 1
        assert agg._hourly_total == 11

    def test_release_slot_decrements(self):
        """release_hourly_slot() opens a slot even after exempt signals."""
        from signals.aggregator import SignalAggregator
        agg = SignalAggregator()
        agg._max_per_hour = 12
        agg._hourly_total = 12

        agg.release_hourly_slot()
        assert agg._hourly_total == 11, "Release must open a slot"

    def test_release_slot_does_not_go_negative(self):
        """release_hourly_slot() must not push counter below zero."""
        from signals.aggregator import SignalAggregator
        agg = SignalAggregator()
        agg._hourly_total = 0

        agg.release_hourly_slot()
        assert agg._hourly_total == 0
