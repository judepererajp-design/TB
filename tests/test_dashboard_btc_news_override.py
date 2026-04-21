import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock


def test_btc_news_endpoint_includes_effective_override_context(monkeypatch):
    from web.app import DashboardApp

    monkeypatch.setattr("web.app._json_response", lambda data, status=200: data)
    monkeypatch.setattr("time.time", lambda: 1_100.0)

    ctx = types.SimpleNamespace(
        is_active=True,
        event_type=types.SimpleNamespace(value="REGULATORY"),
        direction="BEARISH",
        confidence=0.71,
        severity="HIGH",
        headline="Operator override: SEC headline",
        explanation="Manual override active",
        impact_type="Operator Override",
        impact_detail="Dashboard-set override",
        age_minutes=5.0,
        headline_age_minutes=7.0,
        headline_published_at=1_000.0,
        expires_at=1_400.0,
        detected_at=1_050.0,
        block_longs=True,
        block_shorts=False,
        confidence_mult=0.9,
        reduce_size_mult=0.8,
        require_low_corr=False,
        sources=["operator"],
        is_mixed_signal=False,
        reaction_validated=False,
        reaction_confirmed=False,
        net_news_score=-0.4,
    )
    btc_news_intelligence = types.SimpleNamespace(
        get_event_context=lambda: ctx,
        get_status=lambda: {"active": False, "event_type": "UNKNOWN"},
        _move_analyzer=types.SimpleNamespace(
            get_move=lambda window: (1.25, "UP"),
            get_move_since=lambda ts: (-0.55, "DOWN"),
        ),
        _classifier=types.SimpleNamespace(
            classify=lambda title: (types.SimpleNamespace(value="REGULATORY"), "BEARISH", 0.6, False)
        ),
    )
    override_store = types.SimpleNamespace(
        load=AsyncMock(),
        status=lambda: {
            "active": {
                "event_type": "REGULATORY",
                "direction": "BEARISH",
                "confidence_mult": 0.9,
                "size_mult": 0.8,
                "set_by": "dashboard",
                "reason": "Manual override active",
            },
            "expires_in_minutes": 5.0,
            "history_tail": [],
            "trust": {},
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "analyzers.btc_news_intelligence",
        types.SimpleNamespace(
            btc_news_intelligence=btc_news_intelligence,
            BTCEventType=[
                types.SimpleNamespace(value="MACRO_RISK_OFF"),
                types.SimpleNamespace(value="REGULATORY"),
                types.SimpleNamespace(value="UNKNOWN"),
            ],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "analyzers.news_override",
        types.SimpleNamespace(news_override_store=override_store),
    )
    monkeypatch.setitem(
        sys.modules,
        "analyzers.news_scraper",
        types.SimpleNamespace(
            news_scraper=types.SimpleNamespace(
                get_all_stories=lambda **kwargs: [
                    {"title": "Bitcoin regulation headline", "source": "Reuters", "published_at": 1_000.0}
                ]
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "config.constants",
        types.SimpleNamespace(
            NewsOverrideDefaults=types.SimpleNamespace(
                DEFAULT_TTL_MINUTES=60,
                MIN_TTL_MINUTES=5,
                MAX_TTL_MINUTES=240,
                MIN_CONF_MULT=0.5,
                MAX_CONF_MULT=1.5,
                MIN_SIZE_MULT=0.5,
                MAX_SIZE_MULT=1.5,
            )
        ),
    )

    app = DashboardApp()
    data = asyncio.run(app._handle_btc_news(types.SimpleNamespace()))

    assert data["context"]["active"] is True
    assert data["context"]["event_type"] == "REGULATORY"
    assert data["context"]["block_longs"] is True
    assert data["override"]["active"]["set_by"] == "dashboard"
    assert "REGULATORY" in data["event_types"]
    override_store.load.assert_awaited_once()


def test_btc_news_override_post_sets_override(monkeypatch):
    from web.app import DashboardApp

    monkeypatch.setattr("web.app._json_response", lambda data, status=200: {"_status": status, **data})
    monkeypatch.setattr(DashboardApp, "_auth_required", lambda self, request: None)

    override = types.SimpleNamespace(
        to_dict=lambda: {
            "event_type": "MACRO_RISK_OFF",
            "direction": "BEARISH",
            "confidence_mult": 0.95,
            "size_mult": 0.9,
            "reason": "Fed risk-off",
            "set_by": "dashboard",
        }
    )
    store = types.SimpleNamespace(
        set_override=AsyncMock(return_value=override),
        status=lambda: {"active": override.to_dict(), "expires_in_minutes": 60.0},
    )
    monkeypatch.setitem(
        sys.modules,
        "analyzers.news_override",
        types.SimpleNamespace(news_override_store=store),
    )

    app = DashboardApp()
    request = types.SimpleNamespace(
        json=AsyncMock(
            return_value={
                "event_type": "MACRO_RISK_OFF",
                "direction": "BEARISH",
                "confidence_mult": 0.95,
                "size_mult": 0.9,
                "ttl_minutes": 60,
                "reason": "Fed risk-off",
            }
        )
    )
    data = asyncio.run(app._handle_btc_news_override(request))

    assert data["ok"] is True
    assert data["_status"] == 200
    assert data["override"]["event_type"] == "MACRO_RISK_OFF"
    assert store.set_override.await_args.kwargs["set_by"] == "dashboard"
    assert store.set_override.await_args.kwargs["ttl_minutes"] == 60


def test_btc_news_override_clear_removes_override(monkeypatch):
    from web.app import DashboardApp

    monkeypatch.setattr("web.app._json_response", lambda data, status=200: {"_status": status, **data})
    monkeypatch.setattr(DashboardApp, "_auth_required", lambda self, request: None)

    cleared = types.SimpleNamespace(
        to_dict=lambda: {
            "event_type": "REGULATORY",
            "direction": "BEARISH",
            "reason": "Manual clear",
        }
    )
    store = types.SimpleNamespace(clear=AsyncMock(return_value=cleared))
    monkeypatch.setitem(
        sys.modules,
        "analyzers.news_override",
        types.SimpleNamespace(news_override_store=store),
    )

    app = DashboardApp()
    data = asyncio.run(app._handle_btc_news_override_clear(types.SimpleNamespace()))

    assert data["ok"] is True
    assert data["_status"] == 200
    assert data["cleared"]["event_type"] == "REGULATORY"
    assert store.clear.await_args.kwargs["reason"] == "dashboard clear"


def test_dashboard_html_contains_btc_override_controls():
    html = Path("/home/runner/work/TB/TB/web/static/index.html").read_text(encoding="utf-8")

    assert "openBtcOverride()" in html
    assert "clearBtcOverride()" in html
    assert "btc-override-body" in html
    assert "/api/btc-news/override" in html
