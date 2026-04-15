"""
TP3 ATR Clamp — Comprehensive Audit Tests
==========================================
Tests that the post-strategy TP3 normalization layer correctly clamps
unbounded structural TPs while preserving ATR-based TPs.

Coverage matrix:
  - LONG / SHORT direction
  - All 4 structural strategies (Breakout, Elliott Wave, Mean Reversion, Range Scalper)
  - ATR-based strategies (should never be clamped)
  - Edge cases: TP3=None, atr=None, atr=0, negative TP3 (interaction w/ floor)
  - Boundary: exactly at limit, just over, just under
  - Regime-aware: different vol regimes don't break clamping
"""

import sys
import math
from unittest.mock import MagicMock

import pytest

# ── The conftest.py already handles numpy/pandas mocking and project module
# pre-mocking.  We only need to ensure strategies.base and config.constants
# are importable.  Do NOT re-mock modules that conftest already provides. ──

from strategies.base import SignalResult, SignalDirection
from config.constants import Penalties


# ═══════════════════════════════════════════════════════════════
# Helper: simulate the aggregator's TP3 clamp logic
# ═══════════════════════════════════════════════════════════════
def aggregator_tp3_clamp(signal: SignalResult) -> SignalResult:
    """Replicate the aggregator's TP3 ATR clamp exactly as implemented."""
    entry_mid = (signal.entry_low + signal.entry_high) / 2
    _sig_atr = getattr(signal, 'atr', None)
    if _sig_atr and _sig_atr > 0 and signal.tp3 is not None:
        _max_tp3_dist = _sig_atr * Penalties.TP3_MAX_ATR_MULT
        _tp3_dist = abs(signal.tp3 - entry_mid)
        if _tp3_dist > _max_tp3_dist:
            signal.tp3 = entry_mid + math.copysign(_max_tp3_dist, signal.tp3 - entry_mid)
    return signal


def make_signal(
    direction="LONG", entry=100.0, spread=0.5,
    stop_loss=None, tp1=None, tp2=None, tp3=None,
    atr=2.0, strategy="test",
) -> SignalResult:
    """Build a minimal valid SignalResult for clamping tests."""
    entry_low = entry - spread
    entry_high = entry + spread
    if direction == "LONG":
        stop_loss = stop_loss or (entry - atr * 1.2)
        tp1 = tp1 or (entry + atr * 1.5)
        tp2 = tp2 or (entry + atr * 2.2)
        tp3 = tp3 if tp3 is not None else (entry + atr * 4.0)
    else:
        stop_loss = stop_loss or (entry + atr * 1.2)
        tp1 = tp1 or (entry - atr * 1.5)
        tp2 = tp2 or (entry - atr * 2.2)
        tp3 = tp3 if tp3 is not None else (entry - atr * 4.0)
    return SignalResult(
        symbol="BTCUSDT",
        direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
        strategy=strategy,
        confidence=75.0,
        entry_low=entry_low, entry_high=entry_high,
        stop_loss=stop_loss,
        tp1=tp1, tp2=tp2, tp3=tp3,
        rr_ratio=2.0, atr=atr,
    )


# ═══════════════════════════════════════════════════════════════
# 1. Basic LONG / SHORT clamping
# ═══════════════════════════════════════════════════════════════
class TestBasicClamping:

    def test_long_tp3_clamped_when_exceeds_max(self):
        """LONG: TP3 at 20×ATR gets clamped to 7.5×ATR."""
        sig = make_signal("LONG", entry=100, atr=2.0, tp3=100 + 2.0 * 20)
        entry_mid = 100.0
        assert sig.tp3 == pytest.approx(140.0)  # 20×ATR unclamped

        aggregator_tp3_clamp(sig)
        expected = entry_mid + 2.0 * Penalties.TP3_MAX_ATR_MULT
        assert sig.tp3 == pytest.approx(expected)
        assert sig.tp3 < 140.0

    def test_short_tp3_clamped_when_exceeds_max(self):
        """SHORT: TP3 at 20×ATR below entry gets clamped."""
        sig = make_signal("SHORT", entry=100, atr=2.0, tp3=100 - 2.0 * 20)
        entry_mid = 100.0
        assert sig.tp3 == pytest.approx(60.0)

        aggregator_tp3_clamp(sig)
        expected = entry_mid - 2.0 * Penalties.TP3_MAX_ATR_MULT
        assert sig.tp3 == pytest.approx(expected)
        assert sig.tp3 > 60.0

    def test_long_tp3_not_clamped_within_limit(self):
        """LONG: TP3 at 5×ATR stays untouched (< 7.5 cap)."""
        sig = make_signal("LONG", entry=100, atr=2.0, tp3=110.0)  # 5×ATR
        orig_tp3 = sig.tp3
        aggregator_tp3_clamp(sig)
        assert sig.tp3 == pytest.approx(orig_tp3)

    def test_short_tp3_not_clamped_within_limit(self):
        """SHORT: TP3 at 5×ATR below entry stays untouched."""
        sig = make_signal("SHORT", entry=100, atr=2.0, tp3=90.0)
        orig_tp3 = sig.tp3
        aggregator_tp3_clamp(sig)
        assert sig.tp3 == pytest.approx(orig_tp3)

    def test_long_tp3_at_exact_boundary(self):
        """LONG: TP3 exactly at 7.5×ATR stays untouched."""
        atr = 2.0
        tp3 = 100.0 + atr * Penalties.TP3_MAX_ATR_MULT  # exactly at cap
        sig = make_signal("LONG", entry=100, atr=atr, tp3=tp3)
        aggregator_tp3_clamp(sig)
        assert sig.tp3 == pytest.approx(tp3)

    def test_long_tp3_just_over_boundary(self):
        """LONG: TP3 at 7.51×ATR gets clamped to 7.5."""
        atr = 2.0
        tp3 = 100.0 + atr * (Penalties.TP3_MAX_ATR_MULT + 0.01)
        sig = make_signal("LONG", entry=100, atr=atr, tp3=tp3)
        aggregator_tp3_clamp(sig)
        expected = 100.0 + atr * Penalties.TP3_MAX_ATR_MULT
        assert sig.tp3 == pytest.approx(expected, abs=0.001)


# ═══════════════════════════════════════════════════════════════
# 2. Edge cases: None / missing ATR
# ═══════════════════════════════════════════════════════════════
class TestEdgeCases:

    def test_tp3_none_passthrough(self):
        """TP3=None should not be modified."""
        sig = make_signal("LONG", entry=100, atr=2.0)
        sig.tp3 = None
        aggregator_tp3_clamp(sig)
        assert sig.tp3 is None

    def test_atr_none_no_clamp(self):
        """If signal.atr is None, skip clamping entirely."""
        sig = make_signal("LONG", entry=100, atr=2.0, tp3=200.0)
        sig.atr = None
        aggregator_tp3_clamp(sig)
        assert sig.tp3 == pytest.approx(200.0)  # Unchanged

    def test_atr_zero_no_clamp(self):
        """If signal.atr is 0, skip clamping."""
        sig = make_signal("LONG", entry=100, atr=2.0, tp3=200.0)
        sig.atr = 0.0
        aggregator_tp3_clamp(sig)
        assert sig.tp3 == pytest.approx(200.0)

    def test_atr_missing_no_clamp(self):
        """If signal has no atr attribute, skip clamping (backward compat)."""
        sig = make_signal("LONG", entry=100, atr=2.0, tp3=200.0)
        # Simulate old SignalResult without atr
        object.__delattr__(sig, 'atr')
        aggregator_tp3_clamp(sig)
        assert sig.tp3 == pytest.approx(200.0)


# ═══════════════════════════════════════════════════════════════
# 3. Direction correctness: clamp preserves sign
# ═══════════════════════════════════════════════════════════════
class TestDirectionCorrectness:

    def test_long_clamp_above_entry(self):
        """LONG TP3 must always be ABOVE entry after clamping."""
        sig = make_signal("LONG", entry=100, atr=2.0, tp3=150.0)
        aggregator_tp3_clamp(sig)
        entry_mid = 100.0
        assert sig.tp3 > entry_mid

    def test_short_clamp_below_entry(self):
        """SHORT TP3 must always be BELOW entry after clamping."""
        sig = make_signal("SHORT", entry=100, atr=2.0, tp3=50.0)
        aggregator_tp3_clamp(sig)
        entry_mid = 100.0
        assert sig.tp3 < entry_mid

    def test_long_tp3_above_tp2_after_clamp(self):
        """LONG: after clamping, TP3 should still be ≥ TP2."""
        atr = 2.0
        sig = make_signal("LONG", entry=100, atr=atr, tp2=104.4, tp3=200.0)
        aggregator_tp3_clamp(sig)
        assert sig.tp3 >= sig.tp2

    def test_short_tp3_below_tp2_after_clamp(self):
        """SHORT: after clamping, TP3 should still be ≤ TP2."""
        atr = 2.0
        sig = make_signal("SHORT", entry=100, atr=atr, tp2=95.6, tp3=10.0)
        aggregator_tp3_clamp(sig)
        assert sig.tp3 <= sig.tp2


# ═══════════════════════════════════════════════════════════════
# 4. Structural strategies produce realistic clamps
# ═══════════════════════════════════════════════════════════════
class TestStructuralStrategyClamping:

    def test_breakout_long_measured_move_clamped(self):
        """Breakout LONG: 150% measured-move on wide channel gets clamped."""
        channel_high, channel_low = 110.0, 90.0
        range_size = channel_high - channel_low  # 20
        atr = 2.0
        entry = channel_high  # 110
        tp3 = channel_high + range_size * 1.50  # 110 + 30 = 140 → 15×ATR from entry!

        sig = make_signal("LONG", entry=entry, atr=atr, tp3=tp3, strategy="Breakout")
        aggregator_tp3_clamp(sig)

        max_dist = atr * Penalties.TP3_MAX_ATR_MULT
        assert abs(sig.tp3 - entry) <= max_dist + 0.01
        assert sig.tp3 > entry  # Still LONG direction

    def test_breakout_short_measured_move_clamped(self):
        """Breakout SHORT: 150% measured-move below channel gets clamped."""
        channel_high, channel_low = 110.0, 90.0
        range_size = channel_high - channel_low
        atr = 2.0
        entry = channel_low  # 90
        tp3 = channel_low - range_size * 1.50  # 90 - 30 = 60 → 15×ATR below entry!

        sig = make_signal("SHORT", entry=entry, atr=atr, tp3=tp3, strategy="Breakout")
        aggregator_tp3_clamp(sig)

        max_dist = atr * Penalties.TP3_MAX_ATR_MULT
        assert abs(sig.tp3 - entry) <= max_dist + 0.01
        assert sig.tp3 < entry

    def test_elliott_fibonacci_extension_clamped(self):
        """Elliott Wave: 1.618 Fib extension producing 12×ATR gets clamped."""
        atr = 2.0
        entry = 100.0
        # Simulate: tp_proj - entry = 15, tp3 = entry + 15 * 1.618 = 124.27
        tp3 = entry + 15.0 * 1.618  # = 124.27 → 12.14×ATR
        sig = make_signal("LONG", entry=entry, atr=atr, tp3=tp3, strategy="ElliottWave")
        aggregator_tp3_clamp(sig)

        max_dist = atr * Penalties.TP3_MAX_ATR_MULT
        assert abs(sig.tp3 - entry) <= max_dist + 0.01
        assert sig.tp3 > entry

    def test_elliott_short_fibonacci_clamped(self):
        """Elliott Wave SHORT: 1.618 extension gets clamped."""
        atr = 2.0
        entry = 100.0
        tp3 = entry - 15.0 * 1.618
        sig = make_signal("SHORT", entry=entry, atr=atr, tp3=tp3, strategy="ElliottWave")
        aggregator_tp3_clamp(sig)

        max_dist = atr * Penalties.TP3_MAX_ATR_MULT
        assert abs(sig.tp3 - entry) <= max_dist + 0.01
        assert sig.tp3 < entry

    def test_mean_reversion_zscore_clamped(self):
        """Mean Reversion: z-score TP3 can overshoot when std >> ATR."""
        atr = 1.0
        entry = 100.0
        # Simulate: mean=105, std=8, z=2.0 → tp3 = 105 + 8*2 = 121 → 21×ATR
        tp3 = 121.0
        sig = make_signal("LONG", entry=entry, atr=atr, tp3=tp3, strategy="MeanReversion")
        aggregator_tp3_clamp(sig)

        max_dist = atr * Penalties.TP3_MAX_ATR_MULT
        assert abs(sig.tp3 - entry) <= max_dist + 0.01

    def test_range_scalper_equilibrium_not_clamped(self):
        """Range Scalper: 10% overshoot past equilibrium is typically small → no clamp."""
        atr = 2.0
        entry = 95.0
        equilibrium = 100.0
        range_high = 105.0
        tp3 = equilibrium + (range_high - equilibrium) * 0.10  # 100.5 → 2.75×ATR
        sig = make_signal("LONG", entry=entry, atr=atr, tp3=tp3, strategy="RangeScalper")
        orig_tp3 = sig.tp3
        aggregator_tp3_clamp(sig)
        assert sig.tp3 == pytest.approx(orig_tp3)  # Not clamped


# ═══════════════════════════════════════════════════════════════
# 5. ATR-based strategies are NOT clamped (already bounded)
# ═══════════════════════════════════════════════════════════════
class TestATRBasedNotClamped:

    @pytest.mark.parametrize("strategy,tf_scale,vol_scale", [
        ("Momentum", 1.0, 1.075),     # 1h mid-vol: 4.0 * 1.0 * 1.075 = 4.3
        ("SMC", 1.4, 1.3),            # 4h high-vol: 4.0 * 1.4 * 1.3 = 7.28
        ("Ichimoku", 1.0, 0.85),      # 1h low-vol: 4.0 * 1.0 * 0.85 = 3.4
        ("Reversal", 0.5, 1.3),       # 5m high-vol: 4.0 * 0.5 * 1.3 = 2.6
        ("PriceAction", 1.4, 1.075),  # 4h mid-vol: 4.0 * 1.4 * 1.075 = 6.02
        ("Wyckoff", 1.4, 1.3),        # 4h high-vol: 7.28 — the max legitimate ATR TP3
        ("FundingArb", 1.0, 1.3),     # 1h high-vol: 5.2
    ])
    def test_atr_tp3_within_cap(self, strategy, tf_scale, vol_scale):
        """ATR-based strategies produce TP3 ≤ 7.5×ATR → no clamping occurs."""
        atr = 2.0
        base_mult = 4.0
        effective_mult = base_mult * tf_scale * vol_scale
        entry = 100.0
        tp3 = entry + atr * effective_mult
        sig = make_signal("LONG", entry=entry, atr=atr, tp3=tp3, strategy=strategy)
        orig_tp3 = sig.tp3
        aggregator_tp3_clamp(sig)
        assert sig.tp3 == pytest.approx(orig_tp3), (
            f"{strategy} TP3 at {effective_mult:.2f}×ATR should NOT be clamped"
        )


# ═══════════════════════════════════════════════════════════════
# 6. Regime-aware: different volatility regimes
# ═══════════════════════════════════════════════════════════════
class TestRegimeAwareness:

    def test_trending_high_vol_tp3_not_clamped(self):
        """In TRENDING regime with high vol, 4h swing TP3 ≈ 7.28×ATR → not clamped."""
        atr = 3.5  # Higher ATR in volatile trending market
        entry = 50000.0  # BTC
        tp3 = entry + atr * 7.28  # Maximum legitimate ATR-scaled TP3
        sig = make_signal("LONG", entry=entry, atr=atr, tp3=tp3, strategy="Momentum")
        aggregator_tp3_clamp(sig)
        assert sig.tp3 == pytest.approx(entry + atr * 7.28)

    def test_ranging_structural_tp3_clamped(self):
        """In RANGING regime, structural TP3 can be 15×ATR → clamped to 7.5."""
        atr = 1.5
        entry = 50000.0
        tp3 = entry + atr * 15.0  # Absurd structural target in ranging market
        sig = make_signal("LONG", entry=entry, atr=atr, tp3=tp3, strategy="ElliottWave")
        aggregator_tp3_clamp(sig)
        expected = entry + atr * Penalties.TP3_MAX_ATR_MULT
        assert sig.tp3 == pytest.approx(expected)

    def test_volatile_panic_short_clamped(self):
        """In VOLATILE_PANIC, short-side structural TP3 gets clamped too."""
        atr = 5.0  # Wide ATR in panic
        entry = 45000.0
        tp3 = entry - atr * 20.0  # Absurd crash target
        sig = make_signal("SHORT", entry=entry, atr=atr, tp3=tp3, strategy="Breakout")
        aggregator_tp3_clamp(sig)
        expected = entry - atr * Penalties.TP3_MAX_ATR_MULT
        assert sig.tp3 == pytest.approx(expected)
        assert sig.tp3 > 0  # Must remain positive


# ═══════════════════════════════════════════════════════════════
# 7. Interaction with negative-floor clamp
# ═══════════════════════════════════════════════════════════════
class TestNegativeFloorInteraction:

    def test_negative_floor_then_atr_clamp_order(self):
        """Negative floor runs first, then ATR clamp — both fire on low-price short."""
        atr = 0.005  # Low-price token like $0.03
        entry = 0.03
        # After negative floor, TP3 = 0.003% of entry = 0.00003
        # ATR clamp should not re-expand it
        sig = make_signal("SHORT", entry=entry, atr=atr, tp3=-0.01, strategy="SMC")

        # Simulate negative floor first (as aggregator does)
        _min_tp = entry * Penalties.TP_NEGATIVE_FLOOR_PCT
        if sig.tp3 <= 0:
            sig.tp3 = _min_tp

        # Then ATR clamp
        aggregator_tp3_clamp(sig)
        # TP3 should be the floor value or the ATR-clamped value, whichever is closer to entry
        assert sig.tp3 > 0
        assert sig.tp3 < entry  # SHORT: below entry


# ═══════════════════════════════════════════════════════════════
# 8. SignalResult.atr field exists and works
# ═══════════════════════════════════════════════════════════════
class TestSignalResultATRField:

    def test_atr_field_default_none(self):
        """New SignalResult without atr= kwarg defaults to None."""
        sig = SignalResult(
            symbol="TEST", direction=SignalDirection.LONG, strategy="test",
            confidence=70, entry_low=99, entry_high=101,
            stop_loss=97, tp1=103, tp2=105,
        )
        assert sig.atr is None

    def test_atr_field_set_explicitly(self):
        """atr= kwarg is stored on the SignalResult."""
        sig = SignalResult(
            symbol="TEST", direction=SignalDirection.LONG, strategy="test",
            confidence=70, entry_low=99, entry_high=101,
            stop_loss=97, tp1=103, tp2=105, atr=2.5,
        )
        assert sig.atr == 2.5


# ═══════════════════════════════════════════════════════════════
# 9. Constant value sanity checks
# ═══════════════════════════════════════════════════════════════
class TestConstants:

    def test_tp3_max_atr_mult_is_7_5(self):
        """TP3_MAX_ATR_MULT should be 7.5 (covers 4h high-vol TP3 at 7.28×ATR)."""
        assert Penalties.TP3_MAX_ATR_MULT == 7.5

    def test_tp3_max_exceeds_max_legitimate_atr_tp3(self):
        """Cap (7.5) must exceed the highest legitimate ATR-scaled TP3 (7.28)."""
        max_legitimate = 4.0 * 1.40 * 1.30  # tp3_base × 4h_scale × high_vol_scale
        assert Penalties.TP3_MAX_ATR_MULT > max_legitimate

    def test_rr_sanity_cap_still_exists(self):
        """RR sanity cap is still a safety net even after TP3 clamping."""
        assert Penalties.RR_SANITY_CAP == 6.0
