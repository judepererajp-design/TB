"""Tests for PR4 features: funding z-score, liq-sweep, session spread, weekend chop, partial-fill tracker."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from config.constants import (
    FundingZScore as FZ,
    LiqSweep as LSC,
    WeekendChop as WC,
    PartialFill as PFC,
)


# ── (7) Funding z-score ────────────────────────────────────────

class TestFundingZScore:
    def test_insufficient_history_returns_zero(self):
        from analyzers.derivatives import DerivativesAnalyzer
        an = DerivativesAnalyzer()
        # Seed only a few samples (below MIN_SAMPLES_FOR_Z)
        for v in [0.01, 0.012, 0.011]:
            an._classify_funding_delta("BTC/USDT", v)
        z, label = an._compute_funding_z("BTC/USDT")
        assert z == 0.0
        assert label == "NORMAL"

    def test_extreme_hot_z_classified(self):
        from analyzers.derivatives import DerivativesAnalyzer
        an = DerivativesAnalyzer()
        # Steady baseline of 0.01, then a spike to make latest sample ~2.5σ above
        baseline = [0.01] * 10
        for v in baseline:
            an._classify_funding_delta("BTC/USDT", v)
        # Spike: bump above mean by clearly > 2σ. With std≈MIN_STD floor (1e-4),
        # any sizeable jump will produce a huge z, so we use a moderate jump.
        an._classify_funding_delta("BTC/USDT", 0.012)
        z, label = an._compute_funding_z("BTC/USDT")
        assert z > FZ.EXTREME_Z
        assert label in ("EXTREME_HOT", "VERY_HOT")

    def test_extreme_cold_z_classified(self):
        from analyzers.derivatives import DerivativesAnalyzer
        an = DerivativesAnalyzer()
        baseline = [0.01] * 10
        for v in baseline:
            an._classify_funding_delta("BTC/USDT", v)
        an._classify_funding_delta("BTC/USDT", -0.005)
        z, label = an._compute_funding_z("BTC/USDT")
        assert z < -FZ.EXTREME_Z
        assert label in ("EXTREME_COLD", "VERY_COLD")

    def test_normal_z_when_sample_inside_distribution(self):
        from analyzers.derivatives import DerivativesAnalyzer
        an = DerivativesAnalyzer()
        # Wide-but-uniform distribution; a sample at the mean → z≈0
        for v in [0.005, 0.01, 0.015, 0.02, 0.005, 0.012, 0.018, 0.008, 0.015, 0.012]:
            an._classify_funding_delta("BTC/USDT", v)
        z, label = an._compute_funding_z("BTC/USDT")
        assert label == "NORMAL"
        assert abs(z) < FZ.EXTREME_Z


# ── (8) Liq-sweep estimator ────────────────────────────────────

class TestLiqSweepEstimator:
    def test_no_ohlcv_no_round_no_swing_score(self):
        from analyzers.liq_sweep import liq_sweep_estimator
        # Entry well away from any round number; no OHLCV → probability 0
        result = liq_sweep_estimator.estimate(
            direction="LONG", entry=12345.67, stop=12200.00, ohlcv=None,
        )
        assert result.swing_proximity == 0.0

    def test_invalid_direction_returns_zero(self):
        from analyzers.liq_sweep import liq_sweep_estimator
        result = liq_sweep_estimator.estimate(
            direction="WHATEVER", entry=100.0, stop=99.0,
        )
        assert result.probability == 0.0

    def test_long_stop_below_recent_low_high_swing_score(self):
        from analyzers.liq_sweep import liq_sweep_estimator
        # OHLCV with clear local low at 99.5 over 20 bars
        bars = []
        for i in range(20):
            bars.append([i, 100.0, 100.5, 99.6, 100.2, 1000])
        bars[10] = [10, 100.0, 100.5, 99.50, 100.2, 1000]  # the swing low
        # Stop just 0.05 below the swing low (well within PROXIMITY_PCT * entry)
        result = liq_sweep_estimator.estimate(
            direction="LONG", entry=100.5, stop=99.45, ohlcv=bars,
        )
        assert result.swing_proximity > 0.5
        assert result.probability > 0
        assert result.level_kind == "swing_low"

    def test_short_stop_above_recent_high_high_swing_score(self):
        from analyzers.liq_sweep import liq_sweep_estimator
        bars = []
        for i in range(20):
            bars.append([i, 100.0, 100.4, 99.6, 100.2, 1000])
        bars[5] = [5, 100.0, 100.50, 99.6, 100.2, 1000]
        result = liq_sweep_estimator.estimate(
            direction="SHORT", entry=99.5, stop=100.55, ohlcv=bars,
        )
        assert result.swing_proximity > 0.5
        assert result.level_kind == "swing_high"

    def test_round_number_proximity_long(self):
        from analyzers.liq_sweep import liq_sweep_estimator
        # Entry ~$60 100, stop just below round $60 000
        result = liq_sweep_estimator.estimate(
            direction="LONG", entry=60100.0, stop=59950.0, ohlcv=None,
        )
        assert result.round_proximity > 0
        assert result.probability > 0

    def test_confidence_penalty_scales_linearly(self):
        from analyzers.liq_sweep import liq_sweep_estimator
        zero = liq_sweep_estimator.confidence_penalty(0.0)
        half = liq_sweep_estimator.confidence_penalty(0.5)
        full = liq_sweep_estimator.confidence_penalty(1.0)
        assert zero == 0.0
        assert abs(half - LSC.MAX_CONFIDENCE_PENALTY / 2) < 1e-9
        assert abs(full - LSC.MAX_CONFIDENCE_PENALTY) < 1e-9


# ── (9) Session-aware spread multiplier ─────────────────────────
# ── (10) Weekend chop floor bump ────────────────────────────────

class TestSessionSpreadAndWeekendChop:
    def _fake_now(self, weekday: int, hour: int):
        # Build a real datetime that lands on the desired weekday/hour
        # Pick a known Monday: 2024-01-01 was a Monday → weekday 0
        from datetime import timedelta
        base = datetime(2024, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
        return base + timedelta(days=weekday)

    def test_killzone_spread_multiplier_is_one(self):
        from signals import time_filter as tf_mod
        with patch.object(tf_mod, "datetime") as dt_mock:
            dt_mock.now.return_value = self._fake_now(weekday=1, hour=8)  # Tuesday 08 UTC = London open
            assert tf_mod.time_filter.expected_spread_multiplier() == WC.SPREAD_MULT_KILLZONE

    def test_saturday_spread_multiplier_is_weekend(self):
        from signals import time_filter as tf_mod
        with patch.object(tf_mod, "datetime") as dt_mock:
            dt_mock.now.return_value = self._fake_now(weekday=5, hour=10)  # Saturday
            assert tf_mod.time_filter.expected_spread_multiplier() == WC.SPREAD_MULT_SATURDAY

    def test_sunday_early_spread_multiplier_is_worst(self):
        from signals import time_filter as tf_mod
        with patch.object(tf_mod, "datetime") as dt_mock:
            dt_mock.now.return_value = self._fake_now(weekday=6, hour=4)  # Sunday 04 UTC
            mult = tf_mod.time_filter.expected_spread_multiplier()
            assert mult == WC.SPREAD_MULT_SUNDAY_EARLY
            assert mult >= WC.SPREAD_MULT_SATURDAY

    def test_dead_zone_spread_multiplier(self):
        from signals import time_filter as tf_mod
        with patch.object(tf_mod, "datetime") as dt_mock:
            dt_mock.now.return_value = self._fake_now(weekday=2, hour=4)  # Wed 04 UTC = dead zone
            assert tf_mod.time_filter.expected_spread_multiplier() == WC.SPREAD_MULT_DEAD_ZONE

    def test_weekend_floor_bump_on_saturday(self):
        from signals import time_filter as tf_mod
        with patch.object(tf_mod, "datetime") as dt_mock:
            dt_mock.now.return_value = self._fake_now(weekday=5, hour=12)
            assert tf_mod.time_filter.weekend_chop_floor_bump() == WC.WEEKEND_CONF_FLOOR_BUMP

    def test_weekend_floor_bump_sunday_early_is_larger(self):
        from signals import time_filter as tf_mod
        with patch.object(tf_mod, "datetime") as dt_mock:
            dt_mock.now.return_value = self._fake_now(weekday=6, hour=3)
            assert tf_mod.time_filter.weekend_chop_floor_bump() == WC.SUNDAY_EARLY_CONF_FLOOR_BUMP

    def test_weekday_floor_bump_is_zero(self):
        from signals import time_filter as tf_mod
        with patch.object(tf_mod, "datetime") as dt_mock:
            dt_mock.now.return_value = self._fake_now(weekday=2, hour=14)
            assert tf_mod.time_filter.weekend_chop_floor_bump() == 0


# ── (11) Partial-fill tracker ───────────────────────────────────

class TestPartialFillTracker:
    def _fresh(self):
        from core.partial_fill_tracker import partial_fill_tracker
        partial_fill_tracker._records.clear()
        return partial_fill_tracker

    def test_vwap_across_legs(self):
        t = self._fresh()
        for price, qty in [(100.0, 1), (100.5, 2), (101.0, 1)]:
            t.record_leg(signal_id=1, symbol="BTC/USDT", strategy="SMC",
                         direction="LONG", expected_price=100.0,
                         leg_price=price, leg_qty=qty)
        # vwap = (100*1 + 100.5*2 + 101*1) / 4 = 402 / 4 = 100.5
        assert abs(t.weighted_avg_price(1) - 100.5) < 1e-9
        assert t.total_filled_qty(1) == 4

    def test_finalise_pushes_to_slippage_tracker(self):
        t = self._fresh()
        t.record_leg(signal_id=2, symbol="ETH/USDT", strategy="Momentum",
                     direction="LONG", expected_price=3000.0,
                     leg_price=3009.0, leg_qty=1.0)
        t.record_leg(signal_id=2, symbol="ETH/USDT", strategy="Momentum",
                     direction="LONG", expected_price=3000.0,
                     leg_price=3015.0, leg_qty=1.0)
        with patch("core.slippage_tracker.slippage_tracker") as st_mock:
            summary = t.finalise(signal_id=2, size_usd=6000.0)
        assert summary is not None
        assert summary.leg_count == 2
        assert abs(summary.vwap - 3012.0) < 1e-9
        # LONG slippage = (vwap - expected) / expected = 12/3000 = 0.4 %
        assert abs(summary.slippage_pct - 0.004) < 1e-9
        st_mock.record_fill.assert_called_once()

    def test_short_slippage_convention(self):
        t = self._fresh()
        # SHORT received less = adverse → slippage_pct positive
        t.record_leg(signal_id=3, symbol="BTC/USDT", strategy="X",
                     direction="SHORT", expected_price=100.0,
                     leg_price=99.0, leg_qty=1.0)
        with patch("core.slippage_tracker.slippage_tracker"):
            summary = t.finalise(signal_id=3, size_usd=100.0)
        assert summary.slippage_pct > 0  # adverse

    def test_finalised_record_rejects_late_legs(self):
        t = self._fresh()
        t.record_leg(signal_id=4, symbol="X/USDT", strategy="X",
                     direction="LONG", expected_price=10.0,
                     leg_price=10.0, leg_qty=1.0)
        with patch("core.slippage_tracker.slippage_tracker"):
            t.finalise(signal_id=4)
        # Late leg should be ignored
        t.record_leg(signal_id=4, symbol="X/USDT", strategy="X",
                     direction="LONG", expected_price=10.0,
                     leg_price=99.0, leg_qty=99.0)
        rec = t.get_record(4)
        assert len(rec.legs) == 1

    def test_empty_signal_finalise_returns_none(self):
        t = self._fresh()
        assert t.finalise(signal_id=999) is None

    def test_max_legs_enforced(self):
        t = self._fresh()
        for i in range(PFC.MAX_LEGS_PER_SIGNAL + 5):
            t.record_leg(signal_id=5, symbol="X/USDT", strategy="X",
                         direction="LONG", expected_price=10.0,
                         leg_price=10.0, leg_qty=0.1)
        rec = t.get_record(5)
        assert len(rec.legs) == PFC.MAX_LEGS_PER_SIGNAL

    def test_purge_stale_drops_finalised_records(self):
        t = self._fresh()
        t.record_leg(signal_id=6, symbol="X/USDT", strategy="X",
                     direction="LONG", expected_price=10.0,
                     leg_price=10.0, leg_qty=1.0)
        with patch("core.slippage_tracker.slippage_tracker"):
            t.finalise(signal_id=6)
        # Pretend the record is old
        t.get_record(6).started_at = time.time() - 49 * 3600
        purged = t.purge_stale(retention_hours=PFC.RETENTION_HOURS)
        assert purged == 1
        assert t.get_record(6) is None
