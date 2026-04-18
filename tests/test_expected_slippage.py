"""Tests for the predictive slippage model (analyzers.expected_slippage)."""
from __future__ import annotations

from analyzers.expected_slippage import (
    estimate_slippage,
    estimate_slippage_pct_from_signal,
)
from config.constants import Backtester, ExpectedSlippage as _ES


# ── Pure model ────────────────────────────────────────────────────────

def test_estimate_with_no_inputs_returns_backtester_default():
    est = estimate_slippage()
    assert est.fallback is True
    assert est.expected_pct == Backtester.DEFAULT_SLIPPAGE_PCT
    assert est.inputs_known == 0


def test_half_spread_dominates_with_only_spread_input():
    est = estimate_slippage(spread_bps=10.0)
    # 10 bps spread → half = 5 bps = 0.0005
    assert abs(est.half_spread_pct - 0.0005) < 1e-9
    assert est.impact_pct == 0.0
    assert est.vol_pct == 0.0
    assert est.fallback is False
    # Final value floored at FLOOR_PCT but never below half_spread+impact+vol
    assert est.expected_pct >= max(_ES.FLOOR_PCT, 0.0005)


def test_thicker_book_lowers_impact():
    thin = estimate_slippage(spread_bps=5.0, size_usd=10_000, top_book_depth_usd=10_000)
    thick = estimate_slippage(spread_bps=5.0, size_usd=10_000, top_book_depth_usd=1_000_000)
    assert thick.impact_pct < thin.impact_pct
    assert thick.expected_pct <= thin.expected_pct


def test_high_volatility_increases_estimate():
    calm = estimate_slippage(spread_bps=5.0, atr_pct=0.005)   # 0.5 % ATR
    fast = estimate_slippage(spread_bps=5.0, atr_pct=0.05)    # 5 % ATR
    assert fast.vol_pct > calm.vol_pct
    assert fast.expected_pct > calm.expected_pct


def test_size_to_depth_ratio_capped():
    """A trade 100x bigger than book should not blow up to 50% slippage —
    the MAX_DEPTH_RATIO cap clamps it."""
    huge = estimate_slippage(size_usd=1_000_000, top_book_depth_usd=1_000)
    assert huge.expected_pct <= _ES.CEILING_PCT


# ── Signal-bridge convenience ─────────────────────────────────────────

class _StubSignal:
    def __init__(self, raw_data, entry_low=100.0, entry_high=101.0):
        self.raw_data = raw_data
        self.entry_low = entry_low
        self.entry_high = entry_high


def test_signal_bridge_consumes_raw_data_fields():
    sig = _StubSignal(raw_data={
        "spread_bps": 8.0,
        "atr_pct": 0.01,
        "top_book_depth_usd": 250_000,
        "intended_size_usd": 5_000,
    })
    pct = estimate_slippage_pct_from_signal(sig)
    # Should be > floor and < ceiling and finite.
    assert _ES.FLOOR_PCT <= pct <= _ES.CEILING_PCT


def test_signal_bridge_derives_atr_pct_from_atr_proxy():
    """If atr_pct is missing but atr_proxy is present, we should derive it
    from atr_proxy / entry_mid so the vol term contributes."""
    sig_with = _StubSignal(raw_data={
        "spread_bps": 5.0,
        "atr_proxy": 2.0,            # entry_mid = 100.5 → atr_pct ≈ 0.0199
        "top_book_depth_usd": 100_000,
    })
    sig_without = _StubSignal(raw_data={
        "spread_bps": 5.0,
        "top_book_depth_usd": 100_000,
    })
    p_with = estimate_slippage_pct_from_signal(sig_with)
    p_without = estimate_slippage_pct_from_signal(sig_without)
    assert p_with >= p_without


def test_signal_bridge_fallback_on_empty_raw_data():
    """No usable inputs → static backtester default."""
    sig = _StubSignal(raw_data={})
    pct = estimate_slippage_pct_from_signal(sig)
    # When no inputs known and we fall back, we use Backtester default.
    assert pct == Backtester.DEFAULT_SLIPPAGE_PCT


# ── fee_adjusted_rr integration ───────────────────────────────────────

def test_fee_adjusted_rr_uses_predicted_friction():
    """A signal with ample microstructure data should produce a realistic
    fee_adjusted_rr that tracks the predictive friction (not just the
    static default)."""
    from utils.signal_guidance import (
        fee_adjusted_rr,
        predicted_round_trip_friction_pct,
        round_trip_friction_pct,
    )

    class _Sig:
        direction = "LONG"
        entry_low = 100.0
        entry_high = 100.0
        stop_loss = 99.0
        tp1 = 102.0
        tp2 = 102.0

    # Wide spread + thin book → predicted friction should EXCEED the static.
    s_wide = _Sig()
    s_wide.raw_data = {
        "spread_bps": 40.0,            # 40 bps spread = very wide
        "top_book_depth_usd": 1_000,   # paper-thin
        "intended_size_usd": 10_000,
        "atr_pct": 0.04,
    }
    pred_wide = predicted_round_trip_friction_pct(s_wide)
    assert pred_wide > round_trip_friction_pct()

    # Tight spread + deep book → predicted should NOT explode (still bounded).
    s_tight = _Sig()
    s_tight.raw_data = {
        "spread_bps": 2.0,
        "top_book_depth_usd": 5_000_000,
        "intended_size_usd": 5_000,
        "atr_pct": 0.005,
    }
    pred_tight = predicted_round_trip_friction_pct(s_tight)
    assert pred_tight < pred_wide
    # And the resulting fee-adjusted RR remains a finite positive number
    rr_tight = fee_adjusted_rr(s_tight)
    assert rr_tight is not None and rr_tight > 0
