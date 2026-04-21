"""Tests for Option B tiered auto-approval + Gap 6 almost-expired feedback."""

import time
from unittest.mock import MagicMock

import pytest

# DiagnosticEngine imports heavy stuff — conftest already mocks it.
from core.diagnostic_engine import DiagnosticEngine, PendingApproval


def _fresh_engine():
    de = DiagnosticEngine()
    # Avoid touching errors.log / AI analyst etc — tests only exercise pure logic.
    de._errors_log_path = "/tmp/_nonexistent_diag_test.log"
    return de


# ── Gap 6 — almost-expired bucket aggregation ──────────────────
def test_record_almost_expired_aggregates_by_bucket():
    de = _fresh_engine()

    for _ in range(3):
        de.record_almost_expired(
            strategy="MeanReversion", regime="CHOPPY",
            setup_class="intraday", score=2, min_triggers=3,
        )
    de.record_almost_expired(
        strategy="Momentum", regime="BULL_TREND",
        setup_class="intraday", score=1, min_triggers=2,
    )

    buckets = de.get_almost_expired_buckets()
    assert "MeanReversion|CHOPPY|intraday" in buckets
    assert buckets["MeanReversion|CHOPPY|intraday"]["count"] == 3
    # score/min = 2/3 ≈ 0.666
    assert buckets["MeanReversion|CHOPPY|intraday"]["median_fill_ratio"] == pytest.approx(2 / 3)
    assert buckets["Momentum|BULL_TREND|intraday"]["count"] == 1


def test_record_almost_expired_handles_zero_min_triggers():
    de = _fresh_engine()
    de.record_almost_expired(
        strategy="X", regime="Y", setup_class="intraday",
        score=2, min_triggers=0,
    )
    buckets = de.get_almost_expired_buckets()
    # Should not divide by zero
    assert buckets["X|Y|intraday"]["median_fill_ratio"] == 0.0


# ── Option B — tiered auto-approval ────────────────────────────
@pytest.mark.asyncio
async def test_low_risk_proposal_sets_auto_apply_at(monkeypatch):
    de = _fresh_engine()
    # Skip AI verdict generation — no LLM available in tests.
    async def _noop(*a, **kw):
        return None
    de.on_send_approval = _noop

    await de.propose_change(
        change_type="rr_floor",
        description="Lower RR floor to 1.35",
        old_value={"setup_class": "swing", "new_floor": 1.5},
        new_value={"setup_class": "swing", "new_floor": 1.35},
        reason="Tuner feedback",
        risk_level="LOW",
    )

    pending = de.get_pending_approvals()
    assert len(pending) == 1
    assert pending[0].auto_apply_at > time.time()


@pytest.mark.asyncio
async def test_high_risk_proposal_never_auto_applies():
    de = _fresh_engine()
    async def _noop(*a, **kw):
        return None
    de.on_send_approval = _noop

    await de.propose_change(
        change_type="suppress_strategy",
        description="Kill switch",
        old_value=None,
        new_value={"strategy": "Momentum", "duration_mins": 60},
        reason="circuit-breaker",
        risk_level="HIGH",
    )

    pending = de.get_pending_approvals()
    assert len(pending) == 1
    assert pending[0].auto_apply_at == 0.0


@pytest.mark.asyncio
async def test_process_auto_apply_fires_only_mature(monkeypatch):
    de = _fresh_engine()

    applied_args = []

    async def _on_apply(change_type, value):
        applied_args.append((change_type, value))

    de.on_apply_override = _on_apply

    # Insert one mature and one not-yet-mature LOW approval directly.
    now = time.time()
    mature = PendingApproval(
        approval_id="m1",
        change_type="rr_floor",
        description="mature",
        old_value=1.5, new_value=1.35, reason="r",
        risk_level="LOW", estimated_impact="",
        expires_at=now + 1000, auto_apply_at=now - 10,
    )
    immature = PendingApproval(
        approval_id="i1",
        change_type="rr_floor",
        description="immature",
        old_value=1.5, new_value=1.35, reason="r",
        risk_level="LOW", estimated_impact="",
        expires_at=now + 1000, auto_apply_at=now + 1000,
    )
    de._pending_approvals["m1"] = mature
    de._pending_approvals["i1"] = immature

    await de._process_auto_apply()

    # Mature one applied; immature still pending.
    assert "m1" not in de._pending_approvals
    assert "i1" in de._pending_approvals
    # Rollback snapshot captured for mature.
    assert "m1" in de._applied_overrides
    assert de._applied_overrides["m1"]["previous"] == 1.5


@pytest.mark.asyncio
async def test_apply_all_low_risk_batches_only_low(monkeypatch):
    de = _fresh_engine()

    async def _on_apply(change_type, value):
        return None

    de.on_apply_override = _on_apply

    now = time.time()
    for i, risk in enumerate(["LOW", "LOW", "MEDIUM", "HIGH"]):
        de._pending_approvals[f"a{i}"] = PendingApproval(
            approval_id=f"a{i}",
            change_type="rr_floor",
            description=f"desc{i}",
            old_value=1.5, new_value=1.35, reason="",
            risk_level=risk, estimated_impact="",
            expires_at=now + 1000, auto_apply_at=0.0,
        )

    applied = await de.apply_all_low_risk()
    assert sorted(applied) == ["a0", "a1"]
    # MEDIUM and HIGH remain pending.
    assert "a2" in de._pending_approvals
    assert "a3" in de._pending_approvals
