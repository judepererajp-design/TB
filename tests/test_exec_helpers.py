"""Tests for core/_exec_helpers — pure execution-lifecycle helpers.

These are stdlib-only pure functions, so they run cleanly under the
global numpy mock in tests/conftest.py without any special fixtures.
"""

import pytest

from core._exec_helpers import (
    AUTO_APPLY_LOW_RISK_VETO_SECS,
    auto_apply_at_for_risk,
    compute_rr,
    magnitude_multiplier,
    scale_trigger_window,
    tightened_sl,
    trigger_adjustment_for_regime,
    vol_scale_factor,
)


# ── vol_scale_factor ───────────────────────────────────────────
def test_vol_scale_factor_neutral_when_equal():
    assert vol_scale_factor(0.02, 0.02) == pytest.approx(1.0)


def test_vol_scale_factor_clamps_high():
    # 5× ratio should clamp to 2.0 (default cap)
    assert vol_scale_factor(0.10, 0.02) == pytest.approx(2.0)


def test_vol_scale_factor_clamps_low():
    # 0.1× ratio should clamp to 0.5 (default floor)
    assert vol_scale_factor(0.002, 0.02) == pytest.approx(0.5)


def test_vol_scale_factor_malformed_returns_one():
    assert vol_scale_factor(0.0, 0.02) == 1.0
    assert vol_scale_factor(-1.0, 0.02) == 1.0
    assert vol_scale_factor("bad", 0.02) == 1.0


# ── scale_trigger_window ───────────────────────────────────────
def test_scale_trigger_window_shortens_on_high_vol():
    # 2× vol → window halves
    w = scale_trigger_window(3600, 0.04, 0.02, min_secs=60, max_secs=36000)
    assert w == 1800


def test_scale_trigger_window_stretches_on_low_vol():
    # 0.5× vol → window doubles
    w = scale_trigger_window(3600, 0.01, 0.02, min_secs=60, max_secs=36000)
    assert w == 7200


def test_scale_trigger_window_respects_min_and_max():
    # vol_scale_factor clamps to [0.5, 2.0]; with s=2.0, window = 3600/2 = 1800.
    # To trigger the min_secs clamp we need min_secs above 1800.
    assert scale_trigger_window(3600, 10.0, 0.02, min_secs=2400) == 2400
    # Extreme calm clamps s to 0.5 so window stretches to 7200; cap at 5000.
    assert scale_trigger_window(3600, 0.001, 0.02, max_secs=5000) == 5000


# ── trigger_adjustment_for_regime ──────────────────────────────
def test_counter_trend_ratchets_up():
    adj, note = trigger_adjustment_for_regime(
        direction="LONG", regime="BEAR_TREND",
        weekly_bias="BEARISH", weekly_adx=30.0, base_min_triggers=1,
    )
    assert adj == 2
    assert "counter-trend" in note


def test_with_trend_discounts_below_base():
    adj, note = trigger_adjustment_for_regime(
        direction="LONG", regime="BULL_TREND",
        weekly_bias="BULLISH", weekly_adx=30.0, base_min_triggers=2,
    )
    assert adj == 1
    assert "with-trend" in note


def test_with_trend_floor_is_one():
    adj, _ = trigger_adjustment_for_regime(
        direction="LONG", regime="BULL_TREND",
        weekly_bias="BULLISH", weekly_adx=30.0, base_min_triggers=1,
    )
    # Can't go below floor of 1 — should stay at 1.
    assert adj == 1


def test_neutral_regime_no_change():
    adj, note = trigger_adjustment_for_regime(
        direction="LONG", regime="CHOPPY",
        weekly_bias="NEUTRAL", weekly_adx=15.0, base_min_triggers=2,
    )
    assert adj == 2
    assert note == ""


def test_low_adx_no_with_trend_discount():
    # ADX < 25 means the weekly trend isn't strong enough to warrant the discount.
    adj, _ = trigger_adjustment_for_regime(
        direction="LONG", regime="BULL_TREND",
        weekly_bias="BULLISH", weekly_adx=18.0, base_min_triggers=2,
    )
    assert adj == 2


# ── magnitude_multiplier ───────────────────────────────────────
def test_magnitude_multiplier_at_baseline_is_one():
    assert magnitude_multiplier(100, 100) == pytest.approx(1.0)


def test_magnitude_multiplier_caps_at_2():
    assert magnitude_multiplier(500, 100) == pytest.approx(2.0)


def test_magnitude_multiplier_floors_at_half():
    assert magnitude_multiplier(10, 100) == pytest.approx(0.5)


def test_magnitude_multiplier_handles_zero_baseline():
    assert magnitude_multiplier(100, 0) == 1.0


# ── compute_rr ─────────────────────────────────────────────────
def test_compute_rr_long_valid():
    # LONG entry 100, SL 98, TP 106 → risk 2, reward 6 → RR=3.0
    assert compute_rr("LONG", 100, 98, 106) == pytest.approx(3.0)


def test_compute_rr_short_valid():
    # SHORT entry 100, SL 102, TP 94 → risk 2, reward 6 → RR=3.0
    assert compute_rr("SHORT", 100, 102, 94) == pytest.approx(3.0)


def test_compute_rr_wrong_side_returns_zero():
    # LONG with SL above entry → no risk, invalid
    assert compute_rr("LONG", 100, 105, 110) == 0.0
    # LONG with TP below entry → no reward
    assert compute_rr("LONG", 100, 95, 90) == 0.0


def test_compute_rr_malformed_returns_zero():
    assert compute_rr("", 100, 95, 105) == 0.0
    assert compute_rr("LONG", "bad", 95, 105) == 0.0


# ── tightened_sl ───────────────────────────────────────────────
def test_tightened_sl_long_moves_closer():
    # Original SL 95, new swing low 97 → tightened to 97 * (1 - 0.0015) ≈ 96.8
    new = tightened_sl("LONG", 95.0, 97.0)
    assert new > 95.0
    assert new < 97.0


def test_tightened_sl_long_never_loosens():
    # Original SL 99, swing low 95 (farther) → must not loosen to 95.
    new = tightened_sl("LONG", 99.0, 95.0)
    assert new == 99.0


def test_tightened_sl_short_moves_closer():
    new = tightened_sl("SHORT", 105.0, 103.0)
    assert new < 105.0
    assert new > 103.0


def test_tightened_sl_short_never_loosens():
    new = tightened_sl("SHORT", 101.0, 105.0)
    assert new == 101.0


# ── auto_apply_at_for_risk ─────────────────────────────────────
def test_auto_apply_low_risk_schedules_future_apply():
    now = 1_000_000.0
    at = auto_apply_at_for_risk("LOW", now)
    assert at == now + AUTO_APPLY_LOW_RISK_VETO_SECS


def test_auto_apply_medium_and_high_never_auto():
    assert auto_apply_at_for_risk("MEDIUM", 1_000_000.0) == 0.0
    assert auto_apply_at_for_risk("HIGH", 1_000_000.0) == 0.0
    assert auto_apply_at_for_risk("", 1_000_000.0) == 0.0
