import asyncio
import sys
import types
from unittest.mock import AsyncMock


def test_build_inputs_used_captures_signal_market_and_execution_context():
    from signals.signal_decision import _build_inputs_used

    detail = {
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "strategy": "MomentumContinuation",
        "confidence": 78,
        "alpha_grade": "A",
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 98.0,
        "tp1": 104.0,
        "rr_ratio": 2.1,
        "fee_adjusted_rr": 1.9,
        "rsi": 62.4,
        "adx": 27.3,
        "funding_rate": 0.01,
        "execution_score": 58,
        "session_warning": "⚠️ Dead zone — reduce size by 50%",
        "skip_rule": "Skip if price trades above 103.0",
        "confluence": ["✅ EMA stack aligned", "⚡ Volume expansion"],
    }
    ctx = {
        "regime_now": "BULL_TREND",
        "current_price": 100.4,
        "state": "ENTRY_ZONE",
        "in_entry_zone": True,
        "distance_to_zone_pct": 0.0,
        "funding_now": 0.012,
        "reward_to_tp1_pct": 3.6,
        "risk_to_sl_pct": 2.4,
    }

    inputs = _build_inputs_used(detail, ctx)

    assert any("confidence 78/100" in item for item in inputs["signal"])
    assert any("entry 100.0 → 101.0" in item for item in inputs["signal"])
    assert any("regime at decision BULL_TREND" in item for item in inputs["market"])
    assert any("live price 100.4" in item for item in inputs["market"])
    assert any("execution score 58" in item for item in inputs["execution"])
    assert any("confluence:" in item for item in inputs["execution"])


def test_signal_decision_endpoint_returns_review(monkeypatch):
    from web.app import DashboardApp

    sig = {"id": 42, "symbol": "BTC/USDT", "direction": "LONG", "strategy": "MomentumContinuation"}
    detail = {"signal_id": 42, "symbol": "BTC/USDT", "exec_state": "ENTRY_ZONE"}
    review = {
        "verdict": "TAKE",
        "confidence": 83,
        "summary": "Price is in-zone with acceptable execution quality.",
        "strengths": ["In entry zone"],
        "risks": ["Session warning is mild"],
        "entry_guidance": "Enter inside the zone only.",
        "inputs_used": {"signal": ["stub"], "market": [], "execution": []},
    }

    sys.modules["config.loader"] = types.SimpleNamespace(
        cfg=types.SimpleNamespace(ai={}, database=types.SimpleNamespace(path=":memory:"))
    )
    monkeypatch.setattr("web.app._json_response", lambda data, status=200: data)
    monkeypatch.setattr(DashboardApp, "_build_signal_detail_payload", AsyncMock(return_value=detail))
    sys.modules["data.database"] = types.SimpleNamespace(
        db=types.SimpleNamespace(get_signal=AsyncMock(return_value=sig))
    )
    sys.modules["signals.signal_decision"] = types.SimpleNamespace(
        review_signal=AsyncMock(return_value=review)
    )

    app = DashboardApp()
    request = types.SimpleNamespace(match_info={"id": "42"})
    data = asyncio.run(app._handle_signal_decision(request))

    assert data["signal_id"] == 42
    assert data["verdict"] == "TAKE"
    assert data["summary"].startswith("Price is in-zone")


def test_review_signal_timeout_fallback(monkeypatch):
    import signals.signal_decision as signal_decision

    async def slow_call(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return {
            "verdict": "TAKE",
            "confidence": 99,
            "summary": "Should not be returned",
        }

    monkeypatch.setattr(signal_decision, "_AI_DECISION_TIMEOUT_S", 0.01)
    monkeypatch.setattr(signal_decision, "_call_ai", slow_call)
    monkeypatch.setattr(signal_decision, "_gather_context", lambda *_args, **_kwargs: {
        "current_price": 105.0,
        "in_entry_zone": False,
        "distance_to_zone_pct": 4.0,
        "regime_now": "CHOPPY",
        "regime_changed": False,
        "funding_now": None,
        "btc_weekly_bearish": None,
        "btc_4h_bouncing": None,
        "state": "WATCHING",
        "reward_to_tp1_pct": 1.0,
        "risk_to_sl_pct": 4.0,
    })

    detail = {
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "strategy": "MomentumContinuation",
        "confidence": 42,
        "alpha_grade": "C",
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 99.0,
        "tp1": 106.0,
        "rr_ratio": 0.8,
        "fee_adjusted_rr": 0.7,
        "execution_score": 18,
        "session_warning": "Late session",
    }

    review = asyncio.run(signal_decision.review_signal({"symbol": "BTC/USDT"}, detail))

    assert review["verdict"] == "AVOID"
    assert review["summary"] != "Should not be returned"
    assert review["inputs_used"]["signal"]
