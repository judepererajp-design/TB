import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

from utils.signal_guidance import DEAD_ZONE_SIZE_MULTIPLIER, guidance_payload


def _signal_dict(direction="LONG"):
    return {
        "symbol": "BTC/USDT",
        "direction": direction,
        "strategy": "Momentum",
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 98.0 if direction == "LONG" else 103.0,
        "tp1": 103.0 if direction == "LONG" else 98.0,
        "tp2": 106.0 if direction == "LONG" else 95.0,
        "rr_ratio": 2.0,
        "confluence": ["⚠️ Dead zone — B-grade heavy penalty"],
    }


def test_guidance_payload_adds_fee_rr_skip_rule_and_dead_zone_warning():
    payload = guidance_payload(_signal_dict())

    assert payload["fee_adjusted_rr"] is not None
    assert payload["fee_adjusted_rr"] < 2.2
    assert payload["skip_rule"].startswith("Skip if price trades above 103.0")
    assert payload["session_warning"] == "⚠️ Dead zone — reduce size by 50%"
    assert payload["size_modifier"] == DEAD_ZONE_SIZE_MULTIPLIER


def test_guidance_payload_short_skip_rule_uses_lower_bound():
    payload = guidance_payload(_signal_dict(direction="SHORT"))

    assert payload["skip_rule"].startswith("Skip if price trades below 98")


def test_formatter_surfaces_guidance_lines():
    from tg.formatter import TelegramFormatter

    fmt = TelegramFormatter()

    scored = MagicMock()
    scored.base_signal = MagicMock()
    scored.base_signal.symbol = "ETHUSDT"
    scored.base_signal.direction = MagicMock(value="LONG")
    scored.base_signal.strategy = "Momentum"
    scored.base_signal.entry_low = 3000
    scored.base_signal.entry_high = 3050
    scored.base_signal.stop_loss = 2900
    scored.base_signal.tp1 = 3200
    scored.base_signal.tp2 = 3400
    scored.base_signal.tp3 = None
    scored.base_signal.rr_ratio = 2.0
    scored.base_signal.raw_data = {}
    scored.base_signal.confidence = 78
    scored.base_signal.timeframe = "1h"
    scored.base_signal.entry_timeframe = "15m"
    scored.base_signal.setup_class = "intraday"
    scored.base_signal.sector = "L1"
    scored.base_signal.tier = 1
    scored.base_signal.confluence = ["⚠️ Dead zone — B-grade heavy penalty"]
    scored.final_confidence = 79
    scored.grade = "A"
    scored.volume_score = 75
    scored.is_killzone = False
    scored.killzone_bonus = 0
    scored.all_confluence = []

    text = fmt.format_signal(scored, signal_id=7)

    assert "Adj R/R:" in text
    assert "Skip: Skip if price trades above" in text
    assert "Dead zone — reduce size by 50%" in text
    assert "(expected fill)" in text


def test_dashboard_signal_detail_includes_guidance(monkeypatch):
    from web.app import DashboardApp

    sig = _signal_dict()
    sig.update(
        {
            "id": 11,
            "confidence": 78,
            "p_win": 0.61,
            "alpha_grade": "A",
            "regime": "CHOPPY",
            "sector": "L1",
            "setup_class": "intraday",
            "confluence": json.dumps(sig["confluence"]),
            "raw_scores": json.dumps({}),
            "outcome": None,
            "pnl_r": None,
            "entry_price": None,
            "entry_time": None,
            "exit_price": None,
            "exit_reason": None,
            "max_r": 0.0,
            "zone_reached": 0,
            "zone_reached_at": None,
            "publish_price": 100.5,
            "message_id": 123,
            "exec_state": "WATCHING",
            "created_at": "2026-04-14 06:00:00",
        }
    )

    monkeypatch.setattr("data.database.db.get_signal", AsyncMock(return_value=sig))
    monkeypatch.setattr("web.app._json_response", lambda data, status=200: data)
    monkeypatch.setitem(sys.modules, "tg.bot", types.SimpleNamespace(telegram_bot=types.SimpleNamespace(_by_signal_id={})))

    app = DashboardApp()
    request = types.SimpleNamespace(match_info={"id": "11"})
    data = asyncio.run(app._handle_signal_detail(request))

    assert data["fee_adjusted_rr"] is not None
    assert data["skip_rule"].startswith("Skip if price trades above 103.0")
    assert data["session_warning"] == "⚠️ Dead zone — reduce size by 50%"


def test_dashboard_signals_list_includes_guidance(monkeypatch):
    from web.app import DashboardApp

    sig = _signal_dict()
    sig.update(
        {
            "id": 12,
            "alpha_grade": "A",
            "confidence": 74,
            "message_id": 456,
            "exec_state": "WATCHING",
            "created_at": "2026-04-14 06:00:00",
            "confluence": json.dumps(sig["confluence"]),
        }
    )

    monkeypatch.setattr("data.database.db.get_recent_signals", AsyncMock(return_value=[sig]))
    monkeypatch.setattr("web.app._json_response", lambda data, status=200: data)
    monkeypatch.setitem(sys.modules, "core.execution_engine", types.SimpleNamespace(execution_engine=types.SimpleNamespace(_tracked={})))
    monkeypatch.setitem(sys.modules, "tg.bot", types.SimpleNamespace(telegram_bot=types.SimpleNamespace(_by_signal_id={})))

    app = DashboardApp()
    request = types.SimpleNamespace(rel_url=types.SimpleNamespace(query={}))
    data = asyncio.run(app._handle_signals(request))

    assert data["signals"][0]["fee_adjusted_rr"] is not None
    assert data["signals"][0]["session_warning"] == "⚠️ Dead zone — reduce size by 50%"
