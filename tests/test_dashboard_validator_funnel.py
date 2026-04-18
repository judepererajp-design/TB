import asyncio
import sys
import types
from collections import deque
from unittest.mock import AsyncMock


def test_signal_funnel_uses_windowed_validator_counts(monkeypatch):
    from web.app import DashboardApp

    monkeypatch.setitem(sys.modules, "data.database", types.SimpleNamespace(
        db=types.SimpleNamespace(
            get_signal_funnel=AsyncMock(return_value={"published": 4, "taken": 2})
        )
    ))
    monkeypatch.setattr("web.app._json_response", lambda data, status=200: data)
    monkeypatch.setattr("time.time", lambda: 1_000.0)

    diagnostic_engine = types.SimpleNamespace(
        _total_scan_count=12,
        _total_signals_generated=8,
        _death_log=deque([
            {"kill_reason": "VALIDATOR_ERROR", "ts": 950.0},
            {"kill_reason": "VALIDATOR_KILL_SWITCH", "ts": 200.0},
            {"kill_reason": "RR_FLOOR", "ts": 975.0},
        ])
    )
    monkeypatch.setitem(sys.modules, "core.diagnostic_engine", types.SimpleNamespace(
        diagnostic_engine=diagnostic_engine
    ))
    monkeypatch.setitem(sys.modules, "config.feature_flags", types.SimpleNamespace(
        ff=types.SimpleNamespace(get_state=lambda name: "live")
    ))
    monkeypatch.setitem(sys.modules, "signals.signal_validator", types.SimpleNamespace(
        signal_validator=types.SimpleNamespace(get_warning_count=lambda hours=None: 3)
    ))

    app = DashboardApp()
    request = types.SimpleNamespace(rel_url=types.SimpleNamespace(query={"hours": "0"}))
    data = asyncio.run(app._handle_signal_funnel(request))

    assert data["scanned"] == 12
    assert data["raw_generated"] == 8
    assert data["validator_kills"] == 0
    assert data["validator_warnings"] == 3
