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


def test_diagnostics_includes_missed_fill_tracking(monkeypatch):
    from web.app import DashboardApp

    monkeypatch.setattr("web.app._json_response", lambda data, status=200: data)
    monkeypatch.setattr("time.time", lambda: 1_000.0)

    diagnostic_engine = types.SimpleNamespace(
        _total_scan_count=12,
        _scan_count=12,
        _death_log=deque([]),
        get_stats_summary=lambda: {"signals_generated": 2, "signals_published": 1, "win_rate_1h": 0.5, "regime": "CHOPPY"},
        get_death_breakdown=lambda hours=24: {"by_reason": {}, "by_strategy": {}},
    )
    monkeypatch.setitem(sys.modules, "core.diagnostic_engine", types.SimpleNamespace(
        diagnostic_engine=diagnostic_engine
    ))
    ai_analyst = types.SimpleNamespace(
        _session_stats={"regime_at_signal": {}},
        pipeline_diagnostics=AsyncMock(return_value={}),
        get_audit_status=lambda: {"mode": "ok"},
    )
    monkeypatch.setitem(sys.modules, "analyzers.ai_analyst", types.SimpleNamespace(
        ai_analyst=ai_analyst
    ))
    monkeypatch.setitem(sys.modules, "analyzers.veto_system", types.SimpleNamespace(
        veto_system=types.SimpleNamespace(get_proposals=lambda limit=5: [])
    ))
    monkeypatch.setitem(sys.modules, "analyzers.near_miss_tracker", types.SimpleNamespace(
        near_miss_tracker=types.SimpleNamespace(get_metrics=lambda: {"near_miss_count": 0})
    ))
    monkeypatch.setitem(sys.modules, "config.feature_flags", types.SimpleNamespace(
        ff=types.SimpleNamespace(get_state=lambda name: "off")
    ))
    monkeypatch.setitem(sys.modules, "core.missed_fill_tracker", types.SimpleNamespace(
        missed_fill_tracker=types.SimpleNamespace(
            to_dict=lambda: {
                "active_count": 2,
                "finalised_count": 7,
                "stats_24h": {"would_have_won_pct": 50.0, "would_have_lost_pct": 25.0, "never_entered_pct": 25.0},
            }
        )
    ))
    monkeypatch.setitem(sys.modules, "analyzers.regime", types.SimpleNamespace(
        regime_analyzer=types.SimpleNamespace(fear_greed=55)
    ))
    monkeypatch.setitem(sys.modules, "analyzers.htf_guardrail", types.SimpleNamespace(
        htf_guardrail=types.SimpleNamespace(_weekly_bias="BULLISH")
    ))
    monkeypatch.setitem(sys.modules, "core.engine", types.SimpleNamespace(
        engine=types.SimpleNamespace(_start_time=700.0)
    ))

    app = DashboardApp()
    data = asyncio.run(app._handle_diagnostics(types.SimpleNamespace()))

    assert data["missed_fill_tracking"]["active_count"] == 2
    assert data["missed_fill_tracking"]["finalised_count"] == 7
