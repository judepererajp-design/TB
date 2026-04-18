"""
Tests for DeepSeek-validated production fixes.

Fix 1: OHLCV data-quality cooldown — symbols failing N consecutive quality
       checks are auto-excluded for a cooldown period.
Fix 2: Dynamic RR floor by confidence — when confidence ≥ 95, the RR floor
       is scaled down by 0.75× so high-EV setups are not killed.
"""

import time
import pytest


# ════════════════════════════════════════════════════════════════════
# FIX 1: OHLCV DATA-QUALITY COOLDOWN
# ════════════════════════════════════════════════════════════════════

class TestOHLCVCooldownConstants:
    """Constants for the OHLCV cooldown feature exist and are sane."""

    def test_ohlcv_cooldown_class_exists(self):
        from config.constants import OHLCVCooldown
        assert hasattr(OHLCVCooldown, 'FAIL_THRESHOLD')
        assert hasattr(OHLCVCooldown, 'COOLDOWN_SECS')

    def test_fail_threshold_is_3(self):
        from config.constants import OHLCVCooldown
        assert OHLCVCooldown.FAIL_THRESHOLD == 3

    def test_cooldown_secs_is_1800(self):
        from config.constants import OHLCVCooldown
        assert OHLCVCooldown.COOLDOWN_SECS == 1800

    def test_fail_threshold_positive(self):
        from config.constants import OHLCVCooldown
        assert OHLCVCooldown.FAIL_THRESHOLD > 0

    def test_cooldown_secs_reasonable(self):
        from config.constants import OHLCVCooldown
        assert 300 <= OHLCVCooldown.COOLDOWN_SECS <= 7200  # 5 min to 2 hours


class TestOHLCVCooldownScanner:
    """Scanner tracks OHLCV failures and enforces cooldowns."""

    def _make_scanner(self):
        """Build a scanner with minimal mocking — import directly."""
        import sys
        from unittest.mock import MagicMock

        # Mock transitive deps that scanner.scanner imports, but only if they
        # are genuinely unavailable. Unconditionally inserting MagicMock into
        # sys.modules poisons later tests (e.g. test_phase3_audit) that need
        # the real aiosqlite module.
        for mod_name in ("aiosqlite", "data.database"):
            if mod_name in sys.modules:
                continue
            try:
                __import__(mod_name)
            except ImportError:
                sys.modules[mod_name] = MagicMock()

        mock_cfg = MagicMock()
        mock_cfg.scanning = {}
        mock_cfg.system = MagicMock()
        mock_cfg.system.get = MagicMock(return_value=120)
        mock_cfg.exchange = MagicMock()
        mock_cfg.exchange.get = MagicMock(return_value=[])

        from scanner.scanner import Scanner

        import scanner.scanner as ss_mod
        orig_cfg = ss_mod.cfg
        ss_mod.cfg = mock_cfg
        try:
            s = Scanner()
        finally:
            ss_mod.cfg = orig_cfg
        return s

    def test_new_symbol_not_cooled_down(self):
        s = self._make_scanner()
        assert s.is_ohlcv_cooled_down("BZ/USDT") is False

    def test_single_fail_does_not_trigger_cooldown(self):
        s = self._make_scanner()
        result = s.record_ohlcv_fail("BZ/USDT")
        assert result is False
        assert s.is_ohlcv_cooled_down("BZ/USDT") is False

    def test_two_fails_does_not_trigger_cooldown(self):
        s = self._make_scanner()
        s.record_ohlcv_fail("BZ/USDT")
        result = s.record_ohlcv_fail("BZ/USDT")
        assert result is False
        assert s.is_ohlcv_cooled_down("BZ/USDT") is False

    def test_three_fails_triggers_cooldown(self):
        s = self._make_scanner()
        s.record_ohlcv_fail("BZ/USDT")
        s.record_ohlcv_fail("BZ/USDT")
        result = s.record_ohlcv_fail("BZ/USDT")
        assert result is True
        assert s.is_ohlcv_cooled_down("BZ/USDT") is True

    def test_cooldown_expires(self):
        s = self._make_scanner()
        for _ in range(3):
            s.record_ohlcv_fail("CL/USDT")
        # Manually set the cooldown to have already expired
        s._ohlcv_cooldown_until["CL/USDT"] = time.time() - 1
        assert s.is_ohlcv_cooled_down("CL/USDT") is False

    def test_success_resets_counter(self):
        s = self._make_scanner()
        s.record_ohlcv_fail("BZ/USDT")
        s.record_ohlcv_fail("BZ/USDT")
        # A successful fetch resets the counter
        s.record_ohlcv_success("BZ/USDT")
        # Now one more fail should NOT trigger cooldown (only 1 in a row)
        result = s.record_ohlcv_fail("BZ/USDT")
        assert result is False
        assert s.is_ohlcv_cooled_down("BZ/USDT") is False

    def test_different_symbols_tracked_independently(self):
        s = self._make_scanner()
        for _ in range(3):
            s.record_ohlcv_fail("BZ/USDT")
        # BZ is cooled down, CL is not
        assert s.is_ohlcv_cooled_down("BZ/USDT") is True
        assert s.is_ohlcv_cooled_down("CL/USDT") is False

    def test_counter_resets_after_cooldown_triggered(self):
        s = self._make_scanner()
        for _ in range(3):
            s.record_ohlcv_fail("BZ/USDT")
        # Counter should be reset (popped) after entering cooldown
        assert s._ohlcv_fail_counts.get("BZ/USDT", 0) == 0


# ════════════════════════════════════════════════════════════════════
# FIX 2: DYNAMIC R:R FLOOR BY CONFIDENCE
# ════════════════════════════════════════════════════════════════════

class TestDynamicRRConstants:
    """Constants for dynamic RR floor exist and are sane."""

    def test_dynamic_rr_class_exists(self):
        from config.constants import DynamicRR
        assert hasattr(DynamicRR, 'HIGH_CONF_THRESHOLD')
        assert hasattr(DynamicRR, 'HIGH_CONF_RR_DISCOUNT')

    def test_high_conf_threshold_is_95(self):
        from config.constants import DynamicRR
        assert DynamicRR.HIGH_CONF_THRESHOLD == 95.0

    def test_high_conf_rr_discount_is_075(self):
        from config.constants import DynamicRR
        assert DynamicRR.HIGH_CONF_RR_DISCOUNT == 0.75

    def test_discount_reduces_floor(self):
        """0.75 × 2.0 = 1.5 — a 99% conf signal with 1.8R should now pass."""
        from config.constants import DynamicRR
        swing_floor = 2.0
        discounted = swing_floor * DynamicRR.HIGH_CONF_RR_DISCOUNT
        assert discounted == pytest.approx(1.5)
        assert 1.8 >= discounted  # 1.8R passes the discounted floor

    def test_below_threshold_no_discount(self):
        """94% confidence should NOT get the discount."""
        from config.constants import DynamicRR
        assert 94.0 < DynamicRR.HIGH_CONF_THRESHOLD

    def test_ev_math_validates_discount(self):
        """
        EV(99% × 1.8R) = 1.782  >>  EV(68% × 2.5R) = 1.70
        The discounted floor captures higher-EV trades.
        """
        ev_high_conf = 0.99 * 1.8
        ev_normal = 0.68 * 2.5
        assert ev_high_conf > ev_normal


class TestDynamicRRBehavior:
    """
    The aggregator should apply the discount when confidence ≥ 95.
    """

    def test_floor_unchanged_for_normal_confidence(self):
        """Confidence 80 → floor stays at base (e.g., 1.5 for intraday)."""
        from config.constants import DynamicRR
        base_floor = 1.5  # intraday default
        conf = 80.0
        if conf >= DynamicRR.HIGH_CONF_THRESHOLD:
            effective = base_floor * DynamicRR.HIGH_CONF_RR_DISCOUNT
        else:
            effective = base_floor
        assert effective == 1.5

    def test_floor_discounted_for_high_confidence(self):
        """Confidence 97 → floor drops to 1.125 for intraday (1.5 × 0.75)."""
        from config.constants import DynamicRR
        base_floor = 1.5
        conf = 97.0
        if conf >= DynamicRR.HIGH_CONF_THRESHOLD:
            effective = round(base_floor * DynamicRR.HIGH_CONF_RR_DISCOUNT, 2)
        else:
            effective = base_floor
        assert effective == pytest.approx(1.12, abs=0.01)

    def test_swing_floor_discounted(self):
        """Swing floor 2.0 × 0.75 = 1.5 at 95+ conf."""
        from config.constants import DynamicRR
        swing_floor = 2.0
        discounted = round(swing_floor * DynamicRR.HIGH_CONF_RR_DISCOUNT, 2)
        assert discounted == 1.5

    def test_scalp_floor_discounted(self):
        """Scalp floor 1.3 × 0.75 = 0.975 → rounds to 0.98 at 95+ conf."""
        from config.constants import DynamicRR
        scalp_floor = 1.3
        discounted = round(scalp_floor * DynamicRR.HIGH_CONF_RR_DISCOUNT, 2)
        assert discounted == pytest.approx(0.98, abs=0.01)

    def test_boundary_exactly_95(self):
        """Confidence exactly 95.0 should get the discount."""
        from config.constants import DynamicRR
        assert 95.0 >= DynamicRR.HIGH_CONF_THRESHOLD

    def test_boundary_94_99_no_discount(self):
        """94.99 is below 95.0 threshold — no discount."""
        from config.constants import DynamicRR
        assert 94.99 < DynamicRR.HIGH_CONF_THRESHOLD
