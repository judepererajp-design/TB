"""Tests for signals.funding_bridge."""

from signals.funding_bridge import FundingBridge, FundingAdjustment
from config.constants import FundingIntegration as FIC


class TestFundingBridge:
    """Unit tests for FundingBridge."""

    def _make_bridge(self):
        return FundingBridge()

    # ── record_rate / history ───────────────────────────────

    def test_record_rate_creates_history(self):
        b = self._make_bridge()
        b.record_rate("BTC/USDT", 0.0001)
        assert len(b.get_funding_history("BTC/USDT")) == 1

    def test_history_capped_at_max_length(self):
        b = self._make_bridge()
        for i in range(FIC.MAX_HISTORY_LENGTH + 10):
            b.record_rate("BTC/USDT", 0.0001 * i)
        assert len(b.get_funding_history("BTC/USDT")) == FIC.MAX_HISTORY_LENGTH

    def test_empty_history_returns_empty(self):
        b = self._make_bridge()
        assert b.get_funding_history("BTC/USDT") == []

    # ── is_funding_extreme ──────────────────────────────────

    def test_extreme_positive_detected(self):
        b = self._make_bridge()
        b.record_rate("BTC/USDT", FIC.EXTREME_POSITIVE_RATE)
        assert b.is_funding_extreme("BTC/USDT") is True

    def test_extreme_negative_detected(self):
        b = self._make_bridge()
        b.record_rate("BTC/USDT", FIC.EXTREME_NEGATIVE_RATE)
        assert b.is_funding_extreme("BTC/USDT") is True

    def test_normal_rate_not_extreme(self):
        b = self._make_bridge()
        b.record_rate("BTC/USDT", 0.00005)
        assert b.is_funding_extreme("BTC/USDT") is False

    def test_no_data_not_extreme(self):
        b = self._make_bridge()
        assert b.is_funding_extreme("BTC/USDT") is False

    # ── assess_funding_impact — no data ─────────────────────

    def test_no_funding_data_returns_neutral(self):
        b = self._make_bridge()
        adj = b.assess_funding_impact("BTC/USDT", "LONG", None)
        assert adj.confidence_delta == 0.0
        assert adj.severity == "NONE"

    def test_empty_dict_returns_neutral(self):
        b = self._make_bridge()
        adj = b.assess_funding_impact("BTC/USDT", "LONG", {})
        assert adj.confidence_delta == 0.0

    # ── assess_funding_impact — LONG direction ──────────────

    def test_long_extreme_positive_penalized(self):
        """Extreme positive funding opposes LONG → large penalty."""
        b = self._make_bridge()
        adj = b.assess_funding_impact(
            "BTC/USDT", "LONG",
            {"funding_rate": FIC.EXTREME_POSITIVE_RATE, "funding_trend": "RISING"},
        )
        assert adj.confidence_delta < 0
        assert adj.severity == "HIGH"

    def test_long_high_positive_penalized(self):
        """High positive funding opposes LONG → moderate penalty."""
        b = self._make_bridge()
        adj = b.assess_funding_impact(
            "BTC/USDT", "LONG",
            {"funding_rate": FIC.HIGH_POSITIVE_RATE, "funding_trend": "NEUTRAL"},
        )
        assert adj.confidence_delta < 0
        assert adj.severity == "MEDIUM"

    def test_long_extreme_negative_boosted(self):
        """Extreme negative funding supports LONG → boost."""
        b = self._make_bridge()
        adj = b.assess_funding_impact(
            "BTC/USDT", "LONG",
            {"funding_rate": FIC.EXTREME_NEGATIVE_RATE, "funding_trend": "FALLING"},
        )
        assert adj.confidence_delta > 0
        assert adj.severity == "LOW"

    # ── assess_funding_impact — SHORT direction ─────────────

    def test_short_extreme_negative_penalized(self):
        """Extreme negative funding opposes SHORT → large penalty."""
        b = self._make_bridge()
        adj = b.assess_funding_impact(
            "BTC/USDT", "SHORT",
            {"funding_rate": FIC.EXTREME_NEGATIVE_RATE, "funding_trend": "FALLING"},
        )
        assert adj.confidence_delta < 0
        assert adj.severity == "HIGH"

    def test_short_extreme_positive_boosted(self):
        """Extreme positive funding supports SHORT → boost."""
        b = self._make_bridge()
        adj = b.assess_funding_impact(
            "BTC/USDT", "SHORT",
            {"funding_rate": FIC.EXTREME_POSITIVE_RATE, "funding_trend": "RISING"},
        )
        assert adj.confidence_delta > 0
        assert adj.severity == "LOW"

    def test_short_high_positive_modest_boost(self):
        """High positive funding moderately supports SHORT."""
        b = self._make_bridge()
        adj = b.assess_funding_impact(
            "BTC/USDT", "SHORT",
            {"funding_rate": FIC.HIGH_POSITIVE_RATE, "funding_trend": "NEUTRAL"},
        )
        assert adj.confidence_delta > 0
        assert adj.severity == "LOW"

    # ── Trend amplifier ─────────────────────────────────────

    def test_trend_amplifier_worsens_penalty(self):
        """When funding trends against position, penalty is amplified."""
        b = self._make_bridge()
        # Seed history with rising rates to establish positive trend_slope
        for i in range(FIC.TREND_WINDOW + 1):
            b.record_rate("BTC/USDT", 0.0001 + i * 0.0001)

        adj = b.assess_funding_impact(
            "BTC/USDT", "LONG",
            {"funding_rate": FIC.EXTREME_POSITIVE_RATE, "funding_trend": "RISING"},
        )
        # Should be amplified by 1.25x
        assert adj.confidence_delta <= -FIC.EXTREME_PENALTY_PTS

    # ── None funding_rate in dict ───────────────────────────

    def test_none_funding_rate_treated_as_zero(self):
        b = self._make_bridge()
        adj = b.assess_funding_impact(
            "BTC/USDT", "LONG",
            {"funding_rate": None},
        )
        assert adj.confidence_delta == 0.0
