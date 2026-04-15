"""Tests for core.slippage_tracker."""

from core.slippage_tracker import SlippageTracker, SlippageStats


class TestSlippageTracker:
    """Unit tests for SlippageTracker."""

    def _make_tracker(self):
        return SlippageTracker()

    # ── record_fill ─────────────────────────────────────────

    def test_record_long_adverse_slippage(self):
        """LONG that fills above expected → positive (adverse) slippage."""
        t = self._make_tracker()
        t.record_fill(1, "BTC/USDT", "SMC", 100.0, 100.5, "LONG", 1000.0)
        stats = t.get_stats()
        assert stats.total_fills == 1
        assert stats.adverse_fills == 1
        assert stats.avg_slippage_pct > 0  # adverse

    def test_record_long_favorable_slippage(self):
        """LONG that fills below expected → negative (favourable) slippage."""
        t = self._make_tracker()
        t.record_fill(1, "BTC/USDT", "SMC", 100.0, 99.5, "LONG", 1000.0)
        stats = t.get_stats()
        assert stats.total_fills == 1
        assert stats.favorable_fills == 1
        assert stats.avg_slippage_pct < 0

    def test_record_short_adverse_slippage(self):
        """SHORT that fills below expected → positive (adverse) slippage."""
        t = self._make_tracker()
        t.record_fill(1, "ETH/USDT", "Breakout", 2000.0, 1990.0, "SHORT", 500.0)
        stats = t.get_stats()
        assert stats.adverse_fills == 1
        assert stats.avg_slippage_pct > 0

    def test_record_short_favorable_slippage(self):
        """SHORT that fills above expected → negative (favourable) slippage."""
        t = self._make_tracker()
        t.record_fill(1, "ETH/USDT", "Breakout", 2000.0, 2010.0, "SHORT", 500.0)
        stats = t.get_stats()
        assert stats.favorable_fills == 1
        assert stats.avg_slippage_pct < 0

    def test_record_fill_zero_expected_price_ignored(self):
        """Fill with expected_price=0 should be silently dropped."""
        t = self._make_tracker()
        t.record_fill(1, "BTC/USDT", "SMC", 0.0, 100.0, "LONG")
        assert t.get_stats().total_fills == 0

    # ── get_stats filtering ─────────────────────────────────

    def test_filter_by_symbol(self):
        t = self._make_tracker()
        t.record_fill(1, "BTC/USDT", "SMC", 100.0, 100.1, "LONG")
        t.record_fill(2, "ETH/USDT", "SMC", 200.0, 200.2, "LONG")
        btc = t.get_stats(symbol="BTC/USDT")
        assert btc.total_fills == 1

    def test_filter_by_strategy(self):
        t = self._make_tracker()
        t.record_fill(1, "BTC/USDT", "SMC", 100.0, 100.1, "LONG")
        t.record_fill(2, "BTC/USDT", "Breakout", 100.0, 100.2, "LONG")
        smc = t.get_stats(strategy="SMC")
        assert smc.total_fills == 1

    def test_empty_stats_returns_zeroes(self):
        t = self._make_tracker()
        stats = t.get_stats()
        assert stats == SlippageStats()

    # ── get_adjustment_factor ───────────────────────────────

    def test_adjustment_factor_no_data(self):
        t = self._make_tracker()
        assert t.get_adjustment_factor("BTC/USDT", "LONG") == 1.0

    def test_adjustment_factor_favorable(self):
        """Favourable slippage → factor = 1.0 (no penalty)."""
        t = self._make_tracker()
        for i in range(5):
            t.record_fill(i, "BTC/USDT", "SMC", 100.0, 99.9, "LONG")
        assert t.get_adjustment_factor("BTC/USDT", "LONG") == 1.0

    def test_adjustment_factor_adverse(self):
        """Adverse slippage → factor < 1.0."""
        t = self._make_tracker()
        for i in range(10):
            t.record_fill(i, "BTC/USDT", "SMC", 100.0, 101.0, "LONG", 1000.0)
        factor = t.get_adjustment_factor("BTC/USDT", "LONG")
        assert factor < 1.0
        assert factor >= 0.9  # MAX_CONFIDENCE_REDUCTION is 0.10

    # ── to_dict ─────────────────────────────────────────────

    def test_to_dict_keys(self):
        t = self._make_tracker()
        t.record_fill(1, "BTC/USDT", "SMC", 100.0, 100.5, "LONG", 1000.0)
        d = t.to_dict()
        expected_keys = {
            "total_fills", "avg_slippage_pct", "max_slippage_pct",
            "total_slippage_usd", "favorable_fills", "adverse_fills",
            "buffer_size",
        }
        assert set(d.keys()) == expected_keys
