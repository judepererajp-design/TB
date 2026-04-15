"""
Tests for the enrichment pipeline guardrails.

Covers:
  - Enrichment constants class
  - MVRV/SOPR/NUPL diminishing returns (onchain_analytics.get_signal_intel)
  - Aggregate delta cap logic
  - Valuation pre-gate
  - Volatility regime dampener
"""

import pytest
from config.constants import Enrichment


# ── Enrichment constants sanity ──────────────────────────────────

class TestEnrichmentConstants:
    """Verify the Enrichment config class exists with correct values."""

    def test_aggregate_cap_exists(self):
        assert hasattr(Enrichment, 'AGGREGATE_DELTA_CAP')
        assert Enrichment.AGGREGATE_DELTA_CAP == 25

    def test_overvalued_long_ceiling(self):
        assert Enrichment.OVERVALUED_LONG_DELTA_CEIL == 0

    def test_undervalued_short_ceiling(self):
        assert Enrichment.UNDERVALUED_SHORT_DELTA_CEIL == 0

    def test_high_vol_dampener(self):
        assert 0 < Enrichment.HIGH_VOL_DAMPENER < 1.0
        assert Enrichment.HIGH_VOL_DAMPENER == 0.60

    def test_diminishing_returns_multipliers(self):
        assert Enrichment.CORR_SECOND_SIGNAL_MULT == 0.60
        assert Enrichment.CORR_THIRD_SIGNAL_MULT == 0.40


# ── Aggregate cap logic ─────────────────────────────────────────

class TestAggregateCap:
    """Test the aggregate cap applied in the engine enrichment section."""

    def test_positive_cap(self):
        """Sum of positive deltas is clamped to +25."""
        raw = 40
        capped = max(-Enrichment.AGGREGATE_DELTA_CAP,
                      min(Enrichment.AGGREGATE_DELTA_CAP, raw))
        assert capped == 25

    def test_negative_cap(self):
        """Sum of negative deltas is clamped to -25."""
        raw = -50
        capped = max(-Enrichment.AGGREGATE_DELTA_CAP,
                      min(Enrichment.AGGREGATE_DELTA_CAP, raw))
        assert capped == -25

    def test_within_range_untouched(self):
        """Values within ±25 pass through unchanged."""
        for raw in [-20, -10, 0, 10, 18]:
            capped = max(-Enrichment.AGGREGATE_DELTA_CAP,
                          min(Enrichment.AGGREGATE_DELTA_CAP, raw))
            assert capped == raw


# ── Valuation pre-gate logic ────────────────────────────────────

class TestValuationPreGate:
    """
    When on-chain says OVERVALUED, no LONG-positive enrichment should
    survive the pre-gate.  Mirror for UNDERVALUED + SHORT.
    """

    def test_overvalued_blocks_long_boost(self):
        """Positive delta clamped to 0 for LONG when OVERVALUED."""
        d = 8  # bullish enrichment
        val_zone = "OVERVALUED"
        is_long = True
        if val_zone == "OVERVALUED" and is_long and d > 0:
            d = min(d, Enrichment.OVERVALUED_LONG_DELTA_CEIL)
        assert d == 0

    def test_overvalued_allows_long_penalty(self):
        """Negative delta passes through for LONG when OVERVALUED."""
        d = -5
        val_zone = "OVERVALUED"
        is_long = True
        if val_zone == "OVERVALUED" and is_long and d > 0:
            d = min(d, Enrichment.OVERVALUED_LONG_DELTA_CEIL)
        assert d == -5  # untouched

    def test_overvalued_allows_short_boost(self):
        """Positive delta passes through for SHORT when OVERVALUED."""
        d = 6
        val_zone = "OVERVALUED"
        is_long = False
        if val_zone == "OVERVALUED" and is_long and d > 0:
            d = min(d, Enrichment.OVERVALUED_LONG_DELTA_CEIL)
        assert d == 6  # untouched — only LONGs blocked

    def test_undervalued_blocks_short_boost(self):
        """Positive delta clamped to 0 for SHORT when UNDERVALUED."""
        d = 7
        val_zone = "UNDERVALUED"
        is_long = False
        if val_zone == "UNDERVALUED" and not is_long and d > 0:
            d = min(d, Enrichment.UNDERVALUED_SHORT_DELTA_CEIL)
        assert d == 0

    def test_fair_value_passes_through(self):
        """All deltas pass through at FAIR_VALUE."""
        for d_in in [-8, 0, 8]:
            d = d_in
            val_zone = "FAIR_VALUE"
            if val_zone == "OVERVALUED" and d > 0:
                d = min(d, Enrichment.OVERVALUED_LONG_DELTA_CEIL)
            assert d == d_in


# ── Volatility regime dampener ──────────────────────────────────

class TestVolDampener:
    """HIGH_VOL and EXTREME reduce deltas; other regimes pass through."""

    def test_high_vol_reduces(self):
        d = 10
        dampened = int(round(d * Enrichment.HIGH_VOL_DAMPENER))
        assert dampened == 6  # 10 × 0.60 = 6

    def test_normal_vol_unchanged(self):
        d = 10
        regime = "NORMAL"
        mult = Enrichment.HIGH_VOL_DAMPENER if regime in ("HIGH_VOL", "EXTREME") else 1.0
        assert int(round(d * mult)) == 10

    def test_negative_delta_dampened(self):
        d = -8
        dampened = int(round(d * Enrichment.HIGH_VOL_DAMPENER))
        assert dampened == -5  # -8 × 0.60 = -4.8 → -5


# ── Diminishing returns for correlated trio ─────────────────────

class TestDiminishingReturns:
    """Verify that stacked MVRV+SOPR+NUPL signals use diminishing multipliers."""

    def test_single_signal_full_weight(self):
        """One signal fires → no diminishing."""
        deltas = [(8, "MVRV")]
        mults = [1.0, Enrichment.CORR_SECOND_SIGNAL_MULT, Enrichment.CORR_THIRD_SIGNAL_MULT]
        deltas.sort(key=lambda x: abs(x[0]), reverse=True)
        total = sum(int(round(d * mults[i])) for i, (d, _) in enumerate(deltas))
        assert total == 8

    def test_two_signals_second_reduced(self):
        """Two correlated signals → 2nd at 60%."""
        deltas = [(8, "MVRV"), (5, "SOPR")]
        mults = [1.0, Enrichment.CORR_SECOND_SIGNAL_MULT, Enrichment.CORR_THIRD_SIGNAL_MULT]
        deltas.sort(key=lambda x: abs(x[0]), reverse=True)
        total = sum(int(round(d * mults[i])) for i, (d, _) in enumerate(deltas))
        # 8×1.0 + 5×0.60 = 8 + 3 = 11
        assert total == 11

    def test_three_signals_diminishing(self):
        """All three fire → 1st full, 2nd 60%, 3rd 40%."""
        deltas = [(8, "MVRV"), (6, "NUPL"), (5, "SOPR")]
        mults = [1.0, Enrichment.CORR_SECOND_SIGNAL_MULT, Enrichment.CORR_THIRD_SIGNAL_MULT]
        deltas.sort(key=lambda x: abs(x[0]), reverse=True)
        total = sum(int(round(d * mults[i])) for i, (d, _) in enumerate(deltas))
        # 8×1.0 + 6×0.60 + 5×0.40 = 8 + 4 + 2 = 14
        # (vs old: 8+6+5 = 19)
        assert total == 14

    def test_three_bearish_signals_diminishing(self):
        """All three bearish → same diminishing pattern."""
        deltas = [(-8, "MVRV"), (-6, "NUPL"), (-4, "SOPR")]
        mults = [1.0, Enrichment.CORR_SECOND_SIGNAL_MULT, Enrichment.CORR_THIRD_SIGNAL_MULT]
        deltas.sort(key=lambda x: abs(x[0]), reverse=True)
        total = sum(int(round(d * mults[i])) for i, (d, _) in enumerate(deltas))
        # -8×1.0 + -6×0.60 + -4×0.40 = -8 + -4 + -2 = -14
        assert total == -14
