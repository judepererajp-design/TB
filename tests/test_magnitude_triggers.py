"""Tests for Gap 5 — magnitude-weighted TrackedExecution triggers."""

from core.execution_engine import TrackedExecution, SignalState


def _make(**overrides):
    base = dict(
        signal_id=1, symbol="X/USDT", direction="LONG", strategy="MeanReversion",
        entry_low=100.0, entry_high=101.0, stop_loss=98.0, confidence=70.0,
        tp1=104.0, tp2=106.0, rr_ratio=2.0,
    )
    base.update(overrides)
    return TrackedExecution(**base)


def test_triggers_met_defaults_to_nominal_weights():
    te = _make()
    te.has_structure_shift = True
    te.has_liquidity_reaction = True
    # Default multipliers are 1.0, so score = 2 + 1 = 3.0
    assert te.triggers_met == 3.0


def test_triggers_met_applies_magnitude_multipliers():
    te = _make()
    te.has_structure_shift = True
    te.structure_magnitude = 1.5   # strong break
    te.has_momentum_expansion = True
    te.momentum_magnitude = 0.5    # weak momentum
    # 2.0 * 1.5 + 2.0 * 0.5 = 3.0 + 1.0 = 4.0
    assert te.triggers_met == 4.0


def test_triggers_met_handles_zero_or_none_magnitude():
    te = _make()
    te.has_liquidity_reaction = True
    te.liquidity_magnitude = 0  # invalid → falls back to 1.0
    assert te.triggers_met == 1.0

    te.liquidity_magnitude = None
    assert te.triggers_met == 1.0


def test_magnitude_does_not_fire_without_flag():
    te = _make()
    te.structure_magnitude = 2.0  # set but flag off
    te.has_structure_shift = False
    assert te.triggers_met == 0.0
