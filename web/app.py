"""
TitanBot Pro - Local Web Dashboard
====================================
Runs on localhost:8080. Read-only view of the SQLite database +
live data from HyperTracker and real-time RSS news.

Start automatically with TitanBot or manually:
    python -m web.app

Requires: pip install aiohttp aiohttp-cors
"""

import asyncio
import hmac
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from aiohttp import web
    import aiohttp_cors
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False
    logger.warning("Web dashboard disabled - run: pip install aiohttp aiohttp-cors")


def _json_response(data: Any, status: int = 200) -> "web.Response":
    return web.Response(
        text=json.dumps(data, default=str),
        content_type="application/json",
        status=status,
    )


# ── Embedded dashboard HTML (v2.10c) ─────────────────────────
_DASHBOARD_HTML = ''
try:
    _DASHBOARD_HTML = (Path(__file__).parent / 'static' / 'index.html').read_text(encoding='utf-8')
except Exception:
    logger.warning('Could not load static/index.html for embedded fallback')


# ── Notification event queue (engine → SSE → browser) ────────────────────
from collections import deque as _deque
_NOTIF_QUEUE: "_deque[dict]" = _deque(maxlen=100)  # rolling 100 events

def push_notification(event: dict) -> None:
    """Push a notification event to all SSE clients. Thread-safe."""
    import time as _t
    event.setdefault("ts", _t.time())
    _NOTIF_QUEUE.append(event)


def _from_cache(app: "DashboardApp", key: str):
    """Return cached response data if still valid, else None."""
    import time as _time
    entry = getattr(app, '_response_cache', {}).get(key)
    if entry:
        data, ts = entry
        ttl = getattr(app, '_CACHE_TTL', {}).get(key, 30)
        if _time.time() - ts < ttl:
            return data
    return None

def _cached(app: "DashboardApp", key: str, data) -> None:
    """Store data in the response cache with current timestamp."""
    import time as _time
    if not hasattr(app, '_response_cache'):
        app._response_cache = {}
    app._response_cache[key] = (data, _time.time())


class DashboardApp:

    # TTL cache config (seconds per endpoint)
    _CACHE_TTL = {
        "/api/stats": 60, "/api/equity-curve": 120, "/api/monthly-pnl": 300,
        "/api/regime-performance": 120, "/api/quant-metrics": 120,
        "/api/signal-funnel": 60, "/api/learning": 180, "/api/cohorts": 120,
        "/api/portfolio-risk": 15, "/api/positions": 10,
        "/api/markets": 30, "/api/institutional": 300,
        "/api/smart-money": 120, "/api/microstructure": 30,
    }

    @staticmethod
    def _sanitize_symbol(raw: str) -> str:
        """
        Validate and return the symbol from a URL path parameter.
        Raises ValueError if the symbol contains characters outside the
        expected set (A-Z, 0-9, '/', ':') or exceeds 30 characters.
        This prevents path traversal, injection, and log-injection payloads
        from reaching the exchange API or database queries.
        """
        symbol = raw.upper().strip()
        if not symbol or not re.match(r'^[A-Z0-9/:]{1,30}$', symbol):
            raise ValueError(f"Invalid symbol: {raw!r}")
        return symbol

    def __init__(self):
        self._app = None
        self._runner = None
        self._site = None
        self._port = 8080
        self._response_cache: dict = {}  # path -> (timestamp, data)

    async def _handle_signals(self, request: "web.Request") -> "web.Response":
        """GET /api/signals - recent signals from DB."""
        try:
            from data.database import db
            from utils.signal_guidance import guidance_payload
            hours = int(request.rel_url.query.get("hours", 168))
            outcome = request.rel_url.query.get("outcome", "")
            grade   = request.rel_url.query.get("grade", "")
            symbol_filter = request.rel_url.query.get("symbol", "").upper().strip()
            signals = await db.get_recent_signals(hours=hours, exclude_c_grade=False)
            if outcome:
                signals = [s for s in signals if s.get("outcome", "") == outcome.upper()]
            if grade:
                signals = [s for s in signals if s.get("alpha_grade", "") == grade.upper()]
            if symbol_filter:
                signals = [s for s in signals if symbol_filter in s.get("symbol", "").upper()]
            # Enrich with live execution state (in-memory > DB fallback)
            try:
                from core.execution_engine import execution_engine as _ee
                for _s in signals:
                    _t = _ee._tracked.get(_s.get("id"))
                    if _t:
                        _s["exec_state"] = _t.state.value
                    elif not _s.get("exec_state"):
                        # Not tracked in memory - use DB value (WATCHING if missing)
                        _s["exec_state"] = _s.get("exec_state") or "WATCHING"
                    # Migration: map legacy "ARMED" DB values to "ALMOST"
                    if _s.get("exec_state") == "ARMED":
                        _s["exec_state"] = "ALMOST"
                    # Mark whether the signal was actually sent to Telegram
                    _s["published"] = _s.get("message_id") is not None
                    try:
                        _confluence = json.loads(_s.get("confluence") or "[]")
                        if not isinstance(_confluence, list):
                            _confluence = []
                    except Exception:
                        _confluence = []
                    _guidance = guidance_payload(_s, confluence=_confluence)
                    _s["fee_adjusted_rr"] = _guidance.get("fee_adjusted_rr")
                    _s["skip_rule"] = _guidance.get("skip_rule")
                    _s["session_warning"] = _guidance.get("session_warning")
                    _s["size_modifier"] = _guidance.get("size_modifier")
            except Exception as _e:
                logger.debug("Failed to enrich execution state: %s", _e)
            # Enrich with power alignment from in-memory scored signals
            try:
                from tg.bot import telegram_bot as _tb
                for _s in signals:
                    _rec = _tb._by_signal_id.get(_s.get("id"))
                    if _rec and _rec.scored:
                        _s["power_aligned"] = getattr(_rec.scored, 'power_aligned', False)
            except Exception as _e:
                logger.debug("Failed to enrich power alignment: %s", _e)
            return _json_response({"signals": signals[:200], "total": len(signals)})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_stats(self, request: "web.Request") -> "web.Response":
        """GET /api/stats - performance statistics with strategy breakdown for charts."""
        # Check cache first
        cached = _from_cache(self, "/api/stats")
        if cached is not None:
            return _json_response(cached)
        try:
            from data.database import db
            from config.constants import OutcomeTracking
            stats = await db.get_performance_stats(days=30)
            if stats:
                wins = stats.get("wins") or 0
                losses = stats.get("losses") or 0
                breakevens = stats.get("breakevens") or 0
                decisive = wins + losses
                resolved = decisive + breakevens
                stats["win_rate"] = round(wins / decisive * 100, 1) if decisive > 0 else None
                stats["decisive_win_rate"] = stats["win_rate"]
                stats["execution_win_rate"] = round(wins / resolved * 100, 1) if resolved > 0 else None
                stats["best_r"]   = stats.get("avg_r")

            # Fill rate + direction accuracy
            try:
                quality_rows = await db.get_signal_quality_rows(days=30)
                fills = expired = invalidated = correct = evaluated = 0
                for row in quality_rows:
                    outcome = row.get("outcome")
                    if outcome in ("WIN", "LOSS", "BREAKEVEN"):
                        fills += 1
                    elif outcome == "EXPIRED":
                        expired += 1
                    elif outcome == "INVALIDATED":
                        invalidated += 1

                    direction = row.get("direction")
                    if direction not in ("LONG", "SHORT"):
                        continue

                    entry_low = row.get("entry_low")
                    entry_high = row.get("entry_high")
                    zone_mid = None
                    if entry_low is not None and entry_high is not None:
                        zone_mid = (entry_low + entry_high) / 2

                    publish_price = row.get("publish_price") or row.get("entry_price") or zone_mid
                    if not publish_price:
                        continue

                    is_correct = None
                    if outcome in ("WIN", "BREAKEVEN"):
                        is_correct = True
                    elif outcome == "LOSS":
                        is_correct = float(row.get("max_r") or 0.0) >= OutcomeTracking.DIRECTION_ACCURACY_MIN_R
                    elif outcome == "EXPIRED" and row.get("post_expiry_price"):
                        post_expiry_price = float(row.get("post_expiry_price") or 0.0)
                        if post_expiry_price > 0:
                            is_correct = (
                                post_expiry_price > publish_price
                                if direction == "LONG"
                                else post_expiry_price < publish_price
                            )
                    elif outcome == "INVALIDATED":
                        is_correct = False

                    if is_correct is not None:
                        evaluated += 1
                        correct += 1 if is_correct else 0

                fill_denominator = fills + expired + invalidated
                stats["fill_rate"] = round(fills / fill_denominator * 100, 1) if fill_denominator > 0 else None
                stats["expired"] = expired
                stats["invalidated"] = invalidated
                stats["direction_accuracy"] = round(correct / evaluated * 100, 1) if evaluated > 0 else None
            except Exception as _e:
                logger.debug("Failed to build fill/accuracy stats: %s", _e)
                stats["fill_rate"] = None
                stats["direction_accuracy"] = None

            try:
                from core.learning_loop import learning_loop
                ll_stats = learning_loop.get_stats()
                stats["learning"] = {
                    "decisive_win_rate": round((ll_stats.get("decisive_win_rate") or 0) * 100, 1),
                    "execution_win_rate": round((ll_stats.get("execution_win_rate") or 0) * 100, 1),
                    "fill_rate": round((ll_stats.get("fill_rate") or 0) * 100, 1),
                    "expired_count": ll_stats.get("expired_count", 0),
                    "invalidated_count": ll_stats.get("invalidated_count", 0),
                    "breakevens": ll_stats.get("breakevens", 0),
                }
            except Exception as _e:
                logger.debug("Failed to attach learning stats: %s", _e)

            # Add per-strategy breakdown for Performance page charts
            try:
                strat_rows = await db.get_strategy_performance(days=30)
                strategy_stats = {}
                for row in strat_rows:
                    name = row.get("strategy", "")
                    if not name:
                        continue
                    w = row.get("wins") or 0
                    l = row.get("losses") or 0
                    dec = w + l
                    # Pull advanced metrics from performance tracker if available
                    _sharpe = _sortino = _calmar = _mdd = 0.0
                    try:
                        from governance.performance_tracker import performance_tracker
                        _st = performance_tracker._stats.get(name)
                        if _st:
                            _sharpe  = _st.sharpe_ratio
                            _sortino = _st.sortino_ratio
                            _calmar  = _st.calmar_ratio
                            _mdd     = _st.max_drawdown_r
                    except Exception as _e:
                        logger.debug("Failed to load performance_tracker stats: %s", _e)
                    strategy_stats[name] = {
                        "signals":  row.get("signals") or 0,
                        "wins":     w,
                        "losses":   l,
                        "win_rate": round(w / dec * 100, 1) if dec > 0 else 0,
                        "avg_r":    row.get("avg_r"),
                        "sharpe":   round(_sharpe, 2),
                        "sortino":  round(_sortino, 2),
                        "calmar":   round(_calmar, 2),
                        "max_drawdown_r": round(_mdd, 2),
                    }
                stats["strategy_stats"] = strategy_stats
            except Exception as _e:
                logger.debug("Failed to build strategy_stats: %s", _e)
                stats["strategy_stats"] = {}

            # Add per-sector breakdown for Performance page sector chart
            try:
                sector_rows = await db.get_sector_performance(days=30)
                sector_stats = {}
                for row in (sector_rows or []):
                    sector = row.get("sector", "Other") or "Other"
                    w = row.get("wins") or 0
                    l = row.get("losses") or 0
                    dec = w + l
                    sector_stats[sector] = {
                        "signals":  row.get("signals") or 0,
                        "wins":     w,
                        "losses":   l,
                        "win_rate": round(w / dec * 100, 1) if dec > 0 else 0,
                        "total_r":  round(float(row.get("total_r") or 0), 2),
                        "avg_r":    round(float(row.get("avg_r") or 0), 2),
                    }
                # Fallback: derive from strategy_stats if sector DB method not available
                if not sector_stats and strategy_stats:
                    from config.loader import cfg
                    _sec_map = getattr(cfg, 'sectors', {})
                    for strat, sv in strategy_stats.items():
                        sec = "Other"
                        for _sec, _strats in (_sec_map.__dict__ if hasattr(_sec_map, "__dict__") else {}).items():
                            if strat in (str(_strats) if not isinstance(_strats, (list, tuple)) else _strats):
                                sec = _sec
                                break
                        if sec not in sector_stats:
                            sector_stats[sec] = {"signals": 0, "wins": 0, "losses": 0,
                                                  "win_rate": 0, "total_r": 0.0, "avg_r": 0.0}
                        sector_stats[sec]["signals"] += sv.get("signals", 0)
                        sector_stats[sec]["wins"]    += sv.get("wins", 0)
                        sector_stats[sec]["total_r"] = round(
                            sector_stats[sec]["total_r"] + sv.get("avg_r", 0) * sv.get("signals", 0), 2
                        )
                stats["sector_stats"] = sector_stats
            except Exception as _e:
                logger.debug("Failed to build sector_stats: %s", _e)
                stats["sector_stats"] = {}

            _cached(self, "/api/stats", stats)
            return _json_response(stats)
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_cohorts(self, request: "web.Request") -> "web.Response":
        """GET /api/cohorts - live HyperTracker cohort data. 20s timeout."""
        try:
            from analyzers.hypertracker_client import hypertracker
            # Timeout after 20s - 6 parallel HyperTracker requests on slow connection
            # needs more than the original 5s
            cohorts = await asyncio.wait_for(hypertracker.get_all_cohorts(), timeout=20.0)
            data = [
                {
                    "id":            c.cohort_id,
                    "name":          c.name,
                    "bias":          c.bias,
                    "bias_score":    c.bias_score,
                    "leverage":      c.leverage,
                    "in_positions":  c.in_positions_pct,
                    "trader_count":  c.trader_count,
                    "top_coins":     c.top_coins,
                    "stale":         c.is_stale,
                }
                for c in cohorts
            ]
            return _json_response({"cohorts": data, "enabled": hypertracker._enabled})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_news(self, request: "web.Request") -> "web.Response":
        """GET /api/news - RSS news feed (real-time, no API key needed)."""
        try:
            from analyzers.news_scraper import news_scraper
            symbol = request.rel_url.query.get("symbol", "")
            if symbol:
                items = news_scraper.get_news_for_symbol(
                    symbol.upper(), max_age_mins=120
                )
                # enrich with sentiment
                for item in items:
                    score = news_scraper._score_headline(item.get("title", ""))
                    item["sentiment_score"] = 50 + score * 35
                    item["urgency"] = "NORMAL"
                    item["currencies"] = []
                    item["votes_bullish"] = 1 if score > 0 else 0
                    item["votes_bearish"] = 1 if score < 0 else 0
            else:
                items = news_scraper.get_all_stories(max_age_mins=120, limit=50)
            summary = news_scraper.get_market_sentiment_summary()
            return _json_response({"items": items, "summary": summary})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_market_intel(self, request: "web.Request") -> "web.Response":
        """GET /api/market-intel - aggregate OI + liquidation clusters from 7 exchanges."""
        try:
            from analyzers.liquidation_analyzer import liquidation_analyzer
            data = liquidation_analyzer.get_summary_for_dashboard()
            ready = liquidation_analyzer.is_ready if hasattr(liquidation_analyzer, 'is_ready') else bool(data)
            return _json_response({
                "coins":   data,
                "enabled": True,
                "ready":   bool(data),
                "count":   len(data),
            })
        except Exception as e:
            return _json_response({"error": str(e), "enabled": False})

    async def _handle_smart_money(self, request: "web.Request") -> "web.Response":
        """GET /api/smart-money - Hyperliquid top trader positions aggregated."""
        try:
            from analyzers.smart_money_client import smart_money
            data   = smart_money.get_summary_for_dashboard()
            # Build traders list shaped to match dashboard JS expectations:
            # frontend looks for sm.traders[].wallet and sm.traders[].net_long_value
            traders = [
                {
                    "wallet":        c.get("coin", "-"),
                    "net_long_value": c.get("long_usd", 0) - c.get("short_usd", 0),
                    "long_count":    c.get("long_count", 0),
                    "short_count":   c.get("short_count", 0),
                    "total_wallets": c.get("total_wallets", 0),
                    "bias_label":    c.get("bias_label", ""),
                }
                for c in data if c.get("total_wallets", 0) > 0
            ]
            return _json_response({
                "coins":        data,
                "traders":      traders,
                "enabled":      True,
                "ready":        smart_money.is_ready,
                "wallet_count": smart_money.wallet_count,
            })
        except Exception as e:
            return _json_response({"error": str(e), "enabled": False})

    async def _handle_tag_signal(self, request: "web.Request") -> "web.Response":
        """POST /api/signals/{id}/tag - toggle user_taken, save notes."""
        try:

            _auth_err = self._auth_required(request)
            if _auth_err: return _auth_err
            sig_id = int(request.match_info["id"])
            body   = await request.json()
            taken  = int(bool(body.get("taken", False)))
            notes  = str(body.get("notes", ""))[:500]
            from data.database import db
            async with db._lock:
                await db._exec(
                    "UPDATE signals SET user_taken=?, user_notes=? WHERE id=?",
                    (taken, notes, sig_id)
                )
                await db._conn.commit()
            return _json_response({"ok": True, "id": sig_id, "taken": taken})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_microstructure(self, request: "web.Request") -> "web.Response":
        """GET /api/microstructure - CVD, basis, options, netflow."""
        try:
            from analyzers.market_microstructure import microstructure
            return _json_response(microstructure.get_dashboard_data())
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_wallets(self, request: "web.Request") -> "web.Response":
        """GET /api/wallets - tracked wallet positions."""
        try:
            from analyzers.hypertracker_client import hypertracker
            cache = hypertracker.get_watched_wallet_positions()
            result = {}
            for addr, positions in cache.items():
                result[addr] = [
                    {
                        "symbol":         p.symbol,
                        "direction":      p.direction,
                        "size_usd":       p.size_usd,
                        "entry_price":    p.entry_price,
                        "pnl_usd":        p.pnl_usd,
                        "leverage":       p.leverage,
                        "dist_to_liq":    p.dist_to_liq_pct,
                    }
                    for p in positions
                ]
            return _json_response({"wallets": result, "last_updated": hypertracker._wallet_last_fetch})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_engine_status(self, request: "web.Request") -> "web.Response":
        """GET /api/status - live engine stats. Never returns 500 - always a valid object."""
        result = {
            "scan_count": 0, "signal_count": 0, "paused": False,
            "regime": "STARTING", "fear_greed": 50, "alt_season": 50,
            "uptime_s": 0, "trending": [], "ready": False,
            "circuit_breaker_active": False,
            "long_signals": 0, "short_signals": 0,
            "current_session": "LOADING",
        }
        try:
            from core.engine import engine
            result["scan_count"]   = getattr(engine, '_scan_count', 0)
            result["signal_count"] = getattr(engine, '_signal_count', 0)
            result["paused"]       = getattr(engine, '_paused', False)
            result["uptime_s"]     = int(time.time() - engine._start_time) if hasattr(engine, "_start_time") else 0
            result["ready"]               = result["scan_count"] > 0
            result["circuit_breaker_active"] = getattr(engine, '_circuit_breaker_active', False)
            result["long_signals"]            = getattr(engine, '_long_signal_count', 0)
            result["short_signals"]           = getattr(engine, '_short_signal_count', 0)
        except Exception as _e:
            logger.debug("Failed to load engine status: %s", _e)
        try:
            from analyzers.regime import regime_analyzer
            result["regime"]     = regime_analyzer.regime.value if regime_analyzer.regime else "STARTING"
            result["fear_greed"] = regime_analyzer.fear_greed
            result["alt_season"] = regime_analyzer.alt_season_score
            result["macro_risk_off"] = getattr(regime_analyzer, '_macro_risk_off', False)
            result["macro_risk_on"]  = getattr(regime_analyzer, '_macro_risk_on', False)
            result["dxy_rising"]     = getattr(regime_analyzer, '_dxy_rising', False)
            result["vix_elevated"]   = getattr(regime_analyzer, '_vix_elevated', False)
            result["spx_weak"]       = getattr(regime_analyzer, '_spx_below_ma20', False)
            # FIX: was reading non-existent "current_session" attr - property is "session"
            try:
                result["current_session"] = getattr(regime_analyzer.session, 'value', 'UNKNOWN')
            except Exception as _e:
                logger.debug("Failed to read regime session: %s", _e)
                result["current_session"] = "UNKNOWN"
        except Exception as _e:
            logger.debug("Failed to load regime_analyzer status: %s", _e)
        try:
            from analyzers.coingecko_client import coingecko
            result["trending"] = coingecko.get_trending_list()
        except Exception as _e:
            logger.debug("Failed to load trending data: %s", _e)
        try:
            from core.portfolio_engine import portfolio_engine
            positions = portfolio_engine.get_all_positions()
            result["open_positions"] = [
                {
                    "symbol":      p.get("symbol", ""),
                    "direction":   p.get("direction", ""),
                    "entry_price": p.get("entry_price", 0),
                    "signal_id":   p.get("signal_id", 0),
                    "size_usdt":   p.get("size_usdt", 0),
                }
                for p in (positions or [])
            ]
        except Exception as _e:
            logger.debug("Failed to load open positions: %s", _e)
            result["open_positions"] = []
        return _json_response(result)

    async def _handle_prices(self, request: "web.Request") -> "web.Response":
        """GET /api/prices - live prices for the ticker bar.
        
        Uses individual fetch_ticker() calls per symbol (5s cache in api_client).
        fetch_tickers() returns BTC/USDT:USDT keys (futures format) which don't
        match the ticker bar element IDs. Individual fetch_ticker("BTC/USDT") works.
        Runs concurrently for all 12 symbols.
        """
        try:
            from data.api_client import api as exchange_api

            # These must match the tp-XXX ids in the dashboard ticker bar
            symbols = [
                "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "ARB/USDT", "OP/USDT",
                "SUI/USDT", "DOT/USDT",
            ]

            # Fetch all concurrently - each has a 5s TTL cache in api_client
            results = await asyncio.gather(
                *[exchange_api.fetch_ticker(sym) for sym in symbols],
                return_exceptions=True
            )

            prices = {}
            for sym, result in zip(symbols, results):
                if isinstance(result, Exception) or not result:
                    continue
                last = float(result.get("last") or result.get("close") or 0)
                pct  = float(result.get("percentage") or 0)
                if last > 0:
                    prices[sym] = {"price": last, "change_24h": round(pct, 2)}

            return _json_response({"prices": prices, "ts": time.time()})
        except Exception as e:
            return _json_response({"prices": {}, "error": str(e)})


    async def _handle_sentinel_status(self, request: "web.Request") -> "web.Response":
        """GET /api/sentinel - sentinel audit status."""
        try:
            from analyzers.ai_analyst import ai_analyst
            from analyzers.sentinel_features import sentinel
            audit_status = ai_analyst.get_audit_status()
            blacklist = [{"symbol": b.symbol, "reason": b.reason, "fpr": b.false_positive_rate,
                          "hours_left": round((b.banned_until - time.time()) / 3600, 1)}
                         for b in sentinel.get_blacklist()]
            # Build session stats: HTF block counts + strategy direction bias
            _session_stats = {"htf_blocked_long": 0, "htf_blocked_short": 0,
                               "strategy_direction": {}}
            try:
                from analyzers.htf_guardrail import htf_guardrail
                _session_stats["htf_blocked_long"]  = getattr(htf_guardrail, '_blocks_long_session', 0)
                _session_stats["htf_blocked_short"] = getattr(htf_guardrail, '_blocks_short_session', 0)
            except Exception as _e:
                logger.debug("Failed to load htf_guardrail session stats: %s", _e)
            try:
                from data.database import db
                dir_rows = await db.get_strategy_direction_breakdown(hours=24)
                for row in (dir_rows or []):
                    strat = row.get("strategy", "")
                    if strat:
                        key = f"{strat}:session"
                        _session_stats["strategy_direction"][key] = {
                            "LONG":  row.get("long_count", 0),
                            "SHORT": row.get("short_count", 0),
                        }
            except Exception as _e:
                logger.debug("Failed to load strategy direction breakdown: %s", _e)

            return _json_response({"audit": audit_status, "blacklist": blacklist,
                                   "ai_status": ai_analyst.get_status(),
                                   "session_stats": _session_stats})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_sse(self, request: "web.Request") -> "web.Response":
        """GET /api/stream - Server-Sent Events for real-time updates.
        ALWAYS writes a valid JSON object with ts field so JS sets 'Live'.
        Runs indefinitely until client disconnects.

        FIX #29: Added connection limit. Each open browser tab holds an SSE
        connection indefinitely. Without a cap, repeated tab open/close cycles
        (or failed client cleanup) accumulate connections in memory.
        """
        # FIX #29: enforce connection limit
        MAX_SSE_CONNECTIONS = 20
        if not hasattr(self, '_sse_connection_count'):
            self._sse_connection_count = 0
        if self._sse_connection_count >= MAX_SSE_CONNECTIONS:
            return _json_response({"error": "Too many SSE connections"}, 429)
        self._sse_connection_count += 1
        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Access-Control-Allow-Origin"] = "*"
        await response.prepare(request)
        try:
            while True:
                # Always build a valid data object - NEVER send bare {}
                data = {"ts": time.time(), "scan_count": 0, "signal_count": 0,
                        "regime": "STARTING", "fear_greed": 50, "ready": False}
                try:
                    from core.engine import engine
                    data["scan_count"]   = getattr(engine, '_scan_count', 0)
                    data["signal_count"] = getattr(engine, '_signal_count', 0)
                    data["paused"]       = getattr(engine, '_paused', False)
                    data["ready"]        = data["scan_count"] > 0
                except Exception as _e:
                    logger.debug("SSE: failed to load engine status: %s", _e)
                try:
                    from analyzers.regime import regime_analyzer
                    data["regime"]     = regime_analyzer.regime.value if regime_analyzer.regime else "STARTING"
                    data["fear_greed"] = regime_analyzer.fear_greed
                    data["alt_season"] = regime_analyzer.alt_season_score
                    data["macro_risk_off"] = getattr(regime_analyzer, '_macro_risk_off', False)
                    data["macro_risk_on"]  = getattr(regime_analyzer, '_macro_risk_on', False)
                    data["dxy_rising"]     = getattr(regime_analyzer, '_dxy_rising', False)
                    data["vix_elevated"]   = getattr(regime_analyzer, '_vix_elevated', False)
                    data["chop"]       = round(getattr(regime_analyzer, '_chop_strength', 0), 2)
                    data["session"]    = getattr(regime_analyzer.session, 'value', str(getattr(regime_analyzer, 'current_session', 'UNKNOWN')))
                except Exception as _e:
                    logger.debug("SSE: failed to load regime data: %s", _e)
                try:
                    from analyzers.htf_guardrail import htf_guardrail as _htf
                    data["weekly_bias"] = getattr(_htf, '_weekly_bias', 'NEUTRAL')
                except Exception as _e:
                    logger.debug("SSE: failed to load htf_guardrail data: %s", _e)
                try:
                    from core.price_cache import price_cache as _pc
                    for _sym in ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT"]:
                        _p = _pc.get(_sym)
                        if _p: data[_sym.replace("/USDT","").lower()+"_price"] = _p
                    from core.portfolio_engine import portfolio_engine
                    from risk.circuit_breaker import circuit_breaker
                    data["open_positions"] = len(portfolio_engine._positions)
                    data["cb_active"] = circuit_breaker._is_active
                    data["total_risk_pct"] = round(
                        sum(p.risk_usdt for p in portfolio_engine._positions.values())
                        / max(1, portfolio_engine._capital) * 100, 1
                    )
                except Exception as _e:
                    logger.debug("SSE: failed to load price/portfolio data: %s", _e)
                try:
                    if not hasattr(self, "_last_sig_id"):
                        self._last_sig_id = 0
                    from data.database import db
                    _recent = await db.get_recent_signals(hours=1)
                    if _recent:
                        _lat = _recent[0]
                        _lid = _lat.get("id", 0)
                        if _lid and _lid != self._last_sig_id:
                            self._last_sig_id = _lid
                            data["new_signal"] = {
                                "id": _lid,
                                "symbol": _lat.get("symbol"),
                                "direction": _lat.get("direction"),
                                "grade": _lat.get("alpha_grade"),
                                "confidence": _lat.get("confidence"),
                                "strategy": _lat.get("strategy","").replace("Strategy",""),
                            }
                except Exception as _e:
                    logger.debug("SSE: failed to load recent signals: %s", _e)
                # Flush pending notification events from engine
                if _NOTIF_QUEUE:
                    data["notifications"] = list(_NOTIF_QUEUE)
                    _NOTIF_QUEUE.clear()
                msg = f"data: {json.dumps(data)}\n\n"
                await response.write(msg.encode())
                await asyncio.sleep(3)
        except (ConnectionResetError, asyncio.CancelledError):
            pass  # Normal client disconnect
        except Exception as e:
            logger.debug("SSE connection error: %s", e)
        finally:
            # FIX #29: decrement on disconnect so slot is freed
            self._sse_connection_count = max(0, getattr(self, '_sse_connection_count', 1) - 1)
        return response

    async def _handle_debug(self, request: "web.Request") -> "web.Response":
        """GET /api/debug - diagnostic endpoint, shows what each module returns."""
        diag = {"ts": time.time(), "modules": {}}
        try:
            from core.engine import engine
            diag["modules"]["engine"] = {
                "ok": True,
                "scan_count": getattr(engine, '_scan_count', 'MISSING'),
                "signal_count": getattr(engine, '_signal_count', 'MISSING'),
                "uptime_s": int(time.time() - engine._start_time) if hasattr(engine, "_start_time") else "MISSING",
            }
        except Exception as e:
            diag["modules"]["engine"] = {"ok": False, "error": str(e)}
        try:
            from analyzers.regime import regime_analyzer
            diag["modules"]["regime"] = {
                "ok": True,
                "regime": regime_analyzer.regime.value,
                "fear_greed": regime_analyzer.fear_greed,
                "alt_season":      regime_analyzer.alt_season_score,
                "macro_risk_off":  getattr(regime_analyzer, '_macro_risk_off', False),
                "macro_risk_on":   getattr(regime_analyzer, '_macro_risk_on', False),
                "dxy_rising":      getattr(regime_analyzer, '_dxy_rising', False),
                "vix_elevated":    getattr(regime_analyzer, '_vix_elevated', False),
            }
        except Exception as e:
            diag["modules"]["regime"] = {"ok": False, "error": str(e)}
        try:
            from data.database import db
            stats = await db.get_performance_stats(days=30)
            diag["modules"]["database"] = {"ok": True, "stats_keys": list(stats.keys())}
        except Exception as e:
            diag["modules"]["database"] = {"ok": False, "error": str(e)}
        try:
            from analyzers.hypertracker_client import hypertracker
            diag["modules"]["hypertracker"] = {
                "ok": True, "enabled": hypertracker._enabled,
                "watched_wallets": len(hypertracker._watched_wallets),
            }
        except Exception as e:
            diag["modules"]["hypertracker"] = {"ok": False, "error": str(e)}
        try:
            from analyzers.news_scraper import news_scraper
            stories = news_scraper.get_all_stories(max_age_mins=60, limit=5)
            diag["modules"]["news_scraper"] = {"ok": True, "recent_stories": len(stories)}
        except Exception as e:
            diag["modules"]["news_scraper"] = {"ok": False, "error": str(e)}
        try:
            from analyzers.coingecko_client import coingecko
            diag["modules"]["coingecko"] = {
                "ok": True, "trending_count": len(coingecko.get_trending_list())
            }
        except Exception as e:
            diag["modules"]["coingecko"] = {"ok": False, "error": str(e)}
        try:
            from analyzers.ai_analyst import ai_analyst
            diag["modules"]["ai_analyst"] = ai_analyst.get_status()
            diag["modules"]["ai_analyst"]["ok"] = True
        except Exception as e:
            diag["modules"]["ai_analyst"] = {"ok": False, "error": str(e)}
        return _json_response(diag)


    # ── APPROVALS ──────────────────────────────────────────────
    async def _handle_approvals_list(self, request: "web.Request") -> "web.Response":
        """GET /api/approvals - pending + recent applied approvals."""
        try:
            from core.diagnostic_engine import diagnostic_engine
            from analyzers.veto_system import veto_system
            from analyzers.veto_system import PLAIN_ENGLISH_GUIDE
            pending = [
                {
                    "id":          a.approval_id,
                    "type":        a.change_type,
                    "description": a.description,
                    "old_value":   str(a.old_value),
                    "new_value":   str(a.new_value),
                    "reason":      a.reason,
                    "risk_level":  a.risk_level,
                    "estimated_impact": a.estimated_impact,
                    "created_at":  a.created_at,
                    "plain_english": PLAIN_ENGLISH_GUIDE.get(a.change_type, {}),
                    "user_verdict":  getattr(a, 'user_verdict', ''),
                    "verdict_approve": getattr(a, 'verdict_approve', True),
                }
                for a in diagnostic_engine.get_pending_approvals()
            ]
            history = diagnostic_engine.get_applied_overrides()
            veto_proposals = veto_system.get_proposals(limit=10)
            veto_status = veto_system.get_status()

            # FIX: proposals_total in veto_status only counts veto system proposals.
            # Manual approvals applied via the web UI are tracked separately in
            # diagnostic_engine._applied_overrides. Add them to the total so the
            # counter reflects ALL changes that have been applied, not just AI ones.
            _manual_count = len(history)
            veto_status["proposals_total"] = veto_status.get("proposals_total", 0) + _manual_count
            veto_status["manual_applied"] = _manual_count

            return _json_response({
                "pending": pending,
                "history": history,
                "veto_proposals": veto_proposals,
                "veto_status": veto_status,
            })
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_approval_apply(self, request: "web.Request") -> "web.Response":
        """POST /api/approvals/{id}/apply - apply a pending approval."""
        try:

            _auth_err = self._auth_required(request)
            if _auth_err: return _auth_err
            approval_id = request.match_info.get("id", "")
            from core.diagnostic_engine import diagnostic_engine
            success = await diagnostic_engine.apply_approval(approval_id)
            return _json_response({"success": success, "id": approval_id})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_approval_reject(self, request: "web.Request") -> "web.Response":
        """POST /api/approvals/{id}/reject - reject a pending approval."""
        try:

            _auth_err = self._auth_required(request)
            if _auth_err: return _auth_err
            approval_id = request.match_info.get("id", "")
            from core.diagnostic_engine import diagnostic_engine
            diagnostic_engine.reject_approval(approval_id)
            return _json_response({"success": True, "id": approval_id})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_veto_revert(self, request: "web.Request") -> "web.Response":
        """POST /api/veto/revert - revert last auto-applied veto change."""
        try:

            _auth_err = self._auth_required(request)
            if _auth_err: return _auth_err
            from analyzers.veto_system import veto_system
            success = await veto_system.manual_revert_last()
            return _json_response({"success": success})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_markets(self, request: "web.Request") -> "web.Response":
        """GET /api/markets - live prices + metadata for all scanned symbols.
        
        Fetches tickers in batches directly from exchange so prices are always fresh.
        """
        try:
            from core.price_cache import price_cache
            from scanner.scanner import scanner
            from data.api_client import api as exchange_api

            symbols = scanner.get_all_symbols()
            result = []
            for sym in symbols:
                price = price_cache.get(sym)
                state = scanner._symbols.get(sym)
                tier = state.tier.value if state else 2
                vol = state.volume_24h if state else 0
                result.append({
                    "symbol": sym,
                    "price": price,
                    "change_24h": None,
                    "tier": tier,
                    "volume_24h": vol,
                })

            # Fetch tickers in batches of 100 to get live prices + 24h change
            batch_size = 100
            all_syms = [r["symbol"] for r in result]
            try:
                for i in range(0, len(all_syms), batch_size):
                    batch = all_syms[i:i+batch_size]
                    try:
                        tickers = await asyncio.wait_for(
                            exchange_api.fetch_tickers(batch), timeout=10.0
                        )
                        if tickers:
                            for item in result[i:i+batch_size]:
                                t = (tickers.get(item["symbol"]) or tickers.get(item["symbol"] + ":USDT") or tickers.get(item["symbol"].split("/")[0] + "/USDT:USDT"))
                                if t:
                                    last = float(t.get("last") or t.get("close") or 0)
                                    if last > 0:
                                        item["price"] = last
                                    item["change_24h"] = round(float(t.get("percentage") or 0), 2)
                                    item["high_24h"] = float(t.get("high") or 0)
                                    item["low_24h"] = float(t.get("low") or 0)
                    except Exception as _e:
                        logger.debug("Failed to fetch ticker batch: %s", _e)
            except Exception as _e:
                logger.debug("Failed to enrich market data with prices: %s", _e)

            result.sort(key=lambda x: x.get("volume_24h", 0), reverse=True)
            return _json_response({"symbols": result, "count": len(result)})
        except Exception as e:
            return _json_response({"error": str(e)})


    async def _handle_settings_get(self, request: "web.Request") -> "web.Response":
        """GET /api/settings - read key bot parameters."""
        try:
            from config.loader import cfg
            from signals.aggregator import signal_aggregator
            from strategies.base import cfg_min_rr, get_rr_floor_overrides
            settings = {
                "min_confidence":    getattr(signal_aggregator, '_min_confidence', 62),
                "max_signals_hour":  cfg.aggregator.get("max_signals_per_hour", 6),
                "rr_floors":         get_rr_floor_overrides(),
                "rr_floor_swing":    cfg_min_rr("swing"),
                "rr_floor_intraday": cfg_min_rr("intraday"),
                "rr_floor_scalp":    cfg_min_rr("scalp"),
                "dead_zone_start":   cfg.time_filters.get("dead_zone_start_utc", 3) if hasattr(cfg, "time_filters") else 3,
                "dead_zone_end":     cfg.time_filters.get("dead_zone_end_utc", 7) if hasattr(cfg, "time_filters") else 7,
                "tier1_interval":    cfg.system.get("tier1_interval", 120),
                "tier2_interval":    cfg.system.get("tier2_interval", 300),
                "tier3_interval":    cfg.system.get("tier3_interval", 900),
                "tier1_max":         cfg.scanning.tier1.get("max_symbols", 120) if hasattr(cfg, "scanning") else 120,
                "tier2_max":         cfg.scanning.tier2.get("max_symbols", 120) if hasattr(cfg, "scanning") else 120,
                "tier3_max":         cfg.scanning.tier3.get("max_symbols", 80) if hasattr(cfg, "scanning") else 80,
            }
            return _json_response(settings)
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_settings_post(self, request: "web.Request") -> "web.Response":
        """POST /api/settings - apply a single parameter change at runtime."""
        try:

            _auth_err = self._auth_required(request)
            if _auth_err: return _auth_err
            body = await request.json()
            key = body.get("key", "")
            value = body.get("value")
            if not key or value is None:
                return _json_response({"error": "key and value required"}, 400)

            applied = False
            if key == "min_confidence":
                from signals.aggregator import signal_aggregator
                old = signal_aggregator._min_confidence
                signal_aggregator._min_confidence = float(value)
                applied = True
                logger.info(f"[SETTINGS] min_confidence: {old} → {value}")

            elif key == "rr_floor_swing":
                from strategies.base import set_rr_floor_override
                set_rr_floor_override("swing", float(value))
                applied = True
                logger.info(f"[SETTINGS] rr_floor_swing → {value}")

            elif key == "rr_floor_intraday":
                from strategies.base import set_rr_floor_override
                set_rr_floor_override("intraday", float(value))
                applied = True

            elif key == "rr_floor_scalp":
                from strategies.base import set_rr_floor_override
                set_rr_floor_override("scalp", float(value))
                applied = True

            elif key == "max_signals_hour":
                from signals.aggregator import signal_aggregator
                signal_aggregator._max_per_hour = int(value)
                applied = True
                logger.info(f"[SETTINGS] max_signals_hour → {value}")

            else:
                return _json_response({"error": f"Unknown setting: {key}"}, 400)

            return _json_response({"success": applied, "key": key, "value": value})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_coin_narrative(self, request: "web.Request") -> "web.Response":
        """GET /api/narrative/{symbol} - AI-generated coin narrative."""
        try:
            try:
                symbol = self._sanitize_symbol(request.match_info.get("symbol", ""))
            except ValueError as e:
                return _json_response({"error": str(e)}, status=400)
            if "/" not in symbol:
                symbol += "/USDT"

            from core.price_cache import price_cache
            from analyzers.regime import regime_analyzer
            from analyzers.news_scraper import news_scraper

            price = price_cache.get(symbol) or 0
            regime = regime_analyzer.regime.value
            fg = regime_analyzer.fear_greed
            news = news_scraper.get_news_for_symbol(symbol.split("/")[0], max_age_mins=360)[:3]
            news_titles = [n.get("title", "") for n in news]

            # Build narrative via AI analyst
            try:
                from analyzers.ai_analyst import ai_analyst
                prompt = (
                    f"You are a crypto market analyst. Write a single paragraph (3-4 sentences) "
                    f"narrative for {symbol.split('/')[0]}. Current price: ${price:,.6f}. "
                    f"Market regime: {regime}. Fear & Greed: {fg}/100. "
                    f"Recent news: {'; '.join(news_titles[:2]) if news_titles else 'none'}. "
                    f"Be specific, insightful and trader-focused. No fluff. "
                    f"End with one key level to watch."
                )
                narrative = await ai_analyst.fast_analyse(prompt, max_tokens=200)
                if not narrative or len(narrative) < 30:
                    raise ValueError("empty narrative")
            except Exception as _e:
                logger.debug("Failed to generate AI narrative: %s", _e)
                # Fallback: generate without AI
                sym = symbol.split("/")[0]
                rg_desc = {"BULL_TREND": "bullish trend", "BEAR_TREND": "bearish trend",
                           "CHOPPY": "ranging/choppy conditions", "VOLATILE": "high volatility",
                           "VOLATILE_PANIC": "risk-off environment"}.get(regime, regime)
                narrative = (
                    f"{sym} is trading at ${price:,.6f} amid {rg_desc}. "
                    f"Market sentiment reads {fg}/100 on Fear & Greed. "
                    f"{'Recent news is negative.' if news_titles and any(w in ' '.join(news_titles).lower() for w in ['drop','fall','crash','bear']) else 'No major negative catalysts in recent news.'} "
                    f"Watch the current price level for directional confirmation."
                )

            return _json_response({"symbol": symbol, "narrative": narrative, "price": price})
        except Exception as e:
            return _json_response({"error": str(e)})

    # ── MANUAL SCAN ────────────────────────────────────────────
    async def _handle_scan_symbol(self, request: "web.Request") -> "web.Response":
        """POST /api/scan/{symbol} - run full analysis on one symbol."""
        try:
            try:
                symbol = self._sanitize_symbol(request.match_info.get("symbol", ""))
            except ValueError as e:
                return _json_response({"error": str(e)}, status=400)
            if "/" not in symbol:
                symbol += "/USDT"

            result = {"symbol": symbol, "signals": [], "ohlcv": [], "news": [], "whale": {}, "regime": "UNKNOWN", "error": None}

            # Fetch OHLCV
            try:
                from data.api_client import api as exchange_api
                ohlcv_1h = await exchange_api.fetch_ohlcv(symbol, "1h", limit=50)
                ohlcv_4h = await exchange_api.fetch_ohlcv(symbol, "4h", limit=30)
                result["ohlcv_1h"] = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in (ohlcv_1h or [])]
                result["ohlcv_4h"] = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in (ohlcv_4h or [])]
                result["ohlcv"] = result["ohlcv_1h"]
            except Exception as e:
                result["error"] = f"OHLCV fetch failed: {e}"

            # Get regime
            try:
                from analyzers.regime import regime_analyzer
                result["regime"] = regime_analyzer.regime.value if regime_analyzer.regime else "UNKNOWN"
            except Exception as _e:
                logger.debug("Failed to load regime for scan: %s", _e)

            # Run strategies
            if result["ohlcv_1h"]:
                try:
                    from strategies.smc import SMCStrategy
                    from strategies.breakout import BreakoutStrategy
                    from strategies.reversal import ReversalStrategy
                    from strategies.mean_reversion import MeanReversionStrategy
                    from strategies.price_action import PriceActionStrategy
                    from strategies.momentum import MomentumStrategy
                    from strategies.ichimoku import IchimokuStrategy
                    from strategies.elliott_wave import ElliottWaveStrategy
                    from strategies.funding_arb import FundingArbStrategy
                    from strategies.range_scalper import RangeScalperStrategy
                    from strategies.wyckoff import WyckoffStrategy
                    from patterns.harmonic import HarmonicDetector
                    from patterns.geometric import GeometricPatterns
                    from strategies.base import BaseStrategy as _BS
                    _BS.clear_indicator_cache()
                    ohlcv_dict = {"1h": ohlcv_1h or [], "4h": ohlcv_4h or [], "15m": []}
                    for strat in [SMCStrategy(), BreakoutStrategy(), ReversalStrategy(),
                                  MeanReversionStrategy(), PriceActionStrategy(), MomentumStrategy(),
                                  IchimokuStrategy(), ElliottWaveStrategy(), FundingArbStrategy(),
                                  RangeScalperStrategy(), WyckoffStrategy(), HarmonicDetector(),
                                  GeometricPatterns()]:
                        try:
                            sig = await strat.analyze(symbol, ohlcv_dict)
                            if sig:
                                result["signals"].append({
                                    "strategy":    sig.strategy,
                                    "direction":   sig.direction.value,
                                    "confidence":  sig.confidence,
                                    "rr_ratio":    sig.rr_ratio,
                                    "alpha_grade": getattr(sig, 'alpha_grade', 'B'),
                                    "entry_low":   sig.entry_low,
                                    "entry_high":  sig.entry_high,
                                    "stop_loss":   sig.stop_loss,
                                    "tp1":         sig.tp1,
                                    "tp2":         sig.tp2,
                                    "confluence":  sig.confluence[:3] if sig.confluence else [],
                                })
                        except Exception as _e:
                            logger.debug("Strategy %s analyze failed: %s", type(strat).__name__, _e)
                except Exception as e:
                    result["strategy_error"] = str(e)

            # Whale data
            try:
                from signals.whale_aggregator import whale_aggregator
                recent_w = whale_aggregator.get_recent_events(symbol=symbol, max_age_secs=600)
                buy_usd  = sum(w.order_usd for w in recent_w if w.side == "buy")
                sell_usd = sum(w.order_usd for w in recent_w if w.side == "sell")
                result["whale"] = {"buy_usd": buy_usd, "sell_usd": sell_usd, "events": len(recent_w)}
            except Exception as _e:
                logger.debug("Failed to load whale data: %s", _e)

            # News
            try:
                result["news"] = news_scraper.get_news_for_symbol(symbol, max_age_mins=120)[:5]
            except Exception as _e:
                logger.debug("Failed to load news for scan: %s", _e)

            # AI analysis - capped at 45s so total scan stays under 90s client timeout
            try:
                analysis = await asyncio.wait_for(
                    ai_analyst.analyse_symbol(
                        symbol=symbol,
                        ohlcv_data=result["ohlcv"],
                        signals=result["signals"],
                        whale_data=result["whale"],
                        news=result["news"],
                        regime=result["regime"],
                    ),
                    timeout=45.0
                )
                result["ai_analysis"] = analysis
            except asyncio.TimeoutError:
                result["ai_analysis"] = "AI analysis timed out - scan data above is still valid."
            except Exception as e:
                result["ai_analysis"] = f"Analysis unavailable: {e}"

            return _json_response(result)
        except Exception as e:
            # Always return 200 so the JS api() helper doesn't discard the response.
            # api() returns null on non-ok status, which shows "Scan failed: unknown".
            return _json_response({"error": str(e) or "Scan error", "signals": [], "ohlcv_1h": []})

    async def _handle_chat(self, request: "web.Request") -> "web.Response":
        """POST /api/chat - AI chat with symbol context."""
        try:

            _auth_err = self._auth_required(request)
            if _auth_err: return _auth_err
            body = await request.json()
            symbol   = body.get("symbol", "BTC/USDT")
            question = body.get("question", "")
            history  = body.get("history", [])
            context  = body.get("context", {})

            if not question:
                return _json_response({"error": "question required"}, 400)

            from analyzers.ai_analyst import ai_analyst
            answer = await ai_analyst.chat_about_symbol(
                symbol=symbol,
                question=question,
                context=context,
                history=history,
            )
            return _json_response({"answer": answer, "symbol": symbol})
        except Exception as e:
            return _json_response({"error": str(e)})

    # ── DIAGNOSTICS ────────────────────────────────────────────
    async def _handle_diagnostics(self, request: "web.Request") -> "web.Response":
        """GET /api/diagnostics - AI pipeline diagnostics and recommendations."""
        try:
            from core.diagnostic_engine import diagnostic_engine
            from analyzers.ai_analyst import ai_analyst
            from analyzers.veto_system import veto_system
            from collections import Counter

            stats = diagnostic_engine.get_stats_summary()
            # FIX #22: Use get_death_breakdown() for aggregated, sorted kill reason data
            death_breakdown = diagnostic_engine.get_death_breakdown(hours=24)
            kill_reasons = death_breakdown["by_reason"]
            strat_breakdown_agg = death_breakdown["by_strategy"]
            death_log = diagnostic_engine._death_log[-200:]
            kill_reasons_raw = dict(Counter(d.get("kill_reason","?") for d in death_log).most_common(8))
            strat_breakdown = {}
            for d in death_log:
                s = d.get("strategy","?")
                if s not in strat_breakdown:
                    strat_breakdown[s] = {"deaths": 0, "directions": {"LONG":0,"SHORT":0}}
                strat_breakdown[s]["deaths"] += 1
                _dir = d.get("direction", "?")
                strat_breakdown[s]["directions"][_dir] = (
                    strat_breakdown[s]["directions"].get(_dir, 0) + 1
                )

            # Build session data for pipeline diagnostics
            from analyzers.ai_analyst import ai_analyst as _ai
            ss = _ai._session_stats
            sd = {
                "symbols_scanned":   getattr(diagnostic_engine, '_total_scan_count', 0) or getattr(diagnostic_engine, '_scan_count', 0),
                "raw_signals":       stats.get("signals_generated", 0),
                "published":         stats.get("signals_published", 0),
                "win_rate":          stats.get("win_rate_1h", 0),
                "kill_reasons":      kill_reasons,
                "killed_rr":         kill_reasons.get("RR_FLOOR", 0),
                "killed_htf":        kill_reasons.get("HTF_GUARDRAIL", 0),
                "killed_conf":       kill_reasons.get("AGG_THRESHOLD", 0),
                "killed_ntze":       kill_reasons.get("NO_TRADE_ZONE", 0),
                "killed_agg":        kill_reasons.get("CONFLUENCE", 0),
                "long_count":        ss.get("regime_at_signal", {}).get("BULL_TREND", {}).get("LONG", 0),
                "short_count":       ss.get("regime_at_signal", {}).get("BULL_TREND", {}).get("SHORT", 0),
                "htf_long_blocks":   ss.get("htf_blocked_long", 0),
                "htf_short_blocks":  ss.get("htf_blocked_short", 0),
                "strategy_breakdown": strat_breakdown,
                "regime":            stats.get("regime", "UNKNOWN"),
            }
            try:
                from analyzers.regime import regime_analyzer as _ra
                sd["fear_greed"] = _ra.fear_greed
            except Exception as _e:
                logger.debug("Failed to load fear_greed for diagnostics: %s", _e)
                sd["fear_greed"] = 50
            try:
                from analyzers.htf_guardrail import htf_guardrail as _htf
                sd["weekly_bias"] = getattr(_htf, '_weekly_bias', 'UNKNOWN')
            except Exception as _e:
                logger.debug("Failed to load weekly_bias for diagnostics: %s", _e)
                sd["weekly_bias"] = 'UNKNOWN'
            try:
                from core.engine import engine as _eng
                sd["uptime_min"] = round((time.time() - getattr(_eng, '_start_time', time.time())) / 60)
            except Exception as _e:
                logger.debug("Failed to load uptime for diagnostics: %s", _e)
                sd["uptime_min"] = 0

            pipeline_diag = await ai_analyst.pipeline_diagnostics(sd)

            # If AI is unavailable (no API key), generate a rule-based diagnosis
            # so the Diagnostics page is never empty
            if not pipeline_diag:
                total_kills = sum(kill_reasons.values()) if kill_reasons else 0
                published = sd.get("published", 0)
                scanned = sd.get("symbols_scanned", 0)
                top_killer = max(kill_reasons, key=kill_reasons.get) if kill_reasons else "NONE"
                top_killer_count = kill_reasons.get(top_killer, 0)

                # Determine health score from funnel efficiency
                funnel_pct = (published / max(total_kills + published, 1)) * 100
                if funnel_pct >= 20:
                    health = "GOOD"; score = 75
                elif funnel_pct >= 10:
                    health = "FAIR"; score = 55
                elif funnel_pct >= 5:
                    health = "POOR"; score = 35
                else:
                    health = "CRITICAL"; score = 15

                # Map kill reason to plain-English bottleneck
                reason_labels = {
                    "RR_FLOOR": "R:R floor - setups not meeting minimum reward-to-risk ratio",
                    "HTF_GUARDRAIL": "HTF guardrail - signals blocked by weekly trend direction",
                    "AGG_THRESHOLD": "Confidence threshold - signals below minimum confidence score",
                    "EQ_ZONE_BLOCK": "Equilibrium zone - price too close to midrange (no edge)",
                    "DAILY_SYMBOL_LIMIT": "Daily symbol limit - same coin already fired max signals today",
                    "DAILY_TRADE_LIMIT": "Daily trade limit - maximum published signals reached",
                    "HOURLY_TOTAL_LIMIT": "Hourly cap - too many signals in the last hour",
                    "DEDUP_WINDOW": "Dedup cooldown - same setup recently published, waiting for cooldown",
                    "RISK_LIMIT": "Risk limit - daily trade counter exhausted",
                    "CONFLUENCE": "Confluence threshold - not enough confirming factors",
                    "VALIDATOR_ERROR": "Data validator - signal data failed programmatic or LLM integrity checks",
                    "VALIDATOR_KILL_SWITCH": "Validator kill-switch - critical data confidence failure or multiple violations",
                }
                bottleneck_desc = reason_labels.get(top_killer, f"{top_killer} - {top_killer_count} signals killed")

                issues = []
                if kill_reasons.get("DAILY_TRADE_LIMIT", 0) > 0 or kill_reasons.get("RISK_LIMIT", 0) > 0:
                    issues.append("Daily trade limit was hit - consider raising max_daily_trades in settings.yaml")
                if kill_reasons.get("EQ_ZONE_BLOCK", 0) > total_kills * 0.4:
                    issues.append("Equilibrium zone blocking >40% of signals - market is ranging, this is expected in CHOPPY regime")
                if kill_reasons.get("DAILY_SYMBOL_LIMIT", 0) > 5:
                    issues.append("Symbol daily limit frequently hit - same setups firing repeatedly, consider raising max_signals_per_symbol")
                if published == 0 and scanned > 100:
                    issues.append("Zero signals published despite high scan count - check logs for gate stacking")
                if not issues:
                    issues.append("No critical structural issues detected")

                improvements = []
                if top_killer == "RR_FLOOR":
                    improvements.append({"change": "Lower rr_floor_intraday from 1.2 to 1.1", "expected": "+20-30% more signals pass the RR gate", "priority": "MEDIUM"})
                if top_killer == "EQ_ZONE_BLOCK":
                    improvements.append({"change": "EQ dead zone already at 7% in CHOPPY - wait for breakout conditions", "expected": "More signals when price moves from mid-range", "priority": "LOW"})
                if top_killer == "AGG_THRESHOLD":
                    improvements.append({"change": "Lower min_confidence by 2-3 points in settings", "expected": "+15% more signals at cost of slightly lower average quality", "priority": "MEDIUM"})
                improvements.append({"change": "Review Kill Reasons chart below to identify the dominant gate", "expected": "Targeted tuning of the biggest bottleneck", "priority": "HIGH"})

                pipeline_diag = {
                    "bottleneck": bottleneck_desc if top_killer != "NONE" else "No signal deaths recorded yet - bot may be warming up",
                    "structural_issues": issues,
                    "filter_verdict": "too aggressive" if funnel_pct < 5 else ("balanced" if funnel_pct < 25 else "permissive"),
                    "improvements": improvements,
                    "overall_health": health,
                    "health_score": score,
                    "_source": "rule_based"  # flag that this is not AI-generated
                }

            veto_proposals = veto_system.get_proposals(limit=5)
            audit_status = ai_analyst.get_audit_status()

            # ── Execution Gate diagnostics section ─────────────────────
            exec_gate_data = {}
            try:
                # Pull execution gate deaths from death_log
                exec_deaths = [
                    d for d in death_log
                    if d.get("kill_reason") == "EXECUTION_GATE"
                ]
                exec_gate_data["blocked_count"] = len(exec_deaths)
                exec_gate_data["recent_blocks"] = exec_deaths[-20:]  # last 20

                # Count by symbol
                _eg_sym_counter = Counter(d.get("symbol", "?") for d in exec_deaths)
                exec_gate_data["top_blocked_symbols"] = dict(_eg_sym_counter.most_common(10))

                # Count by direction
                _eg_dir_counter = Counter(d.get("direction", "?") for d in exec_deaths)
                exec_gate_data["by_direction"] = dict(_eg_dir_counter)

                # Count by strategy
                _eg_strat_counter = Counter(d.get("strategy", "?") for d in exec_deaths)
                exec_gate_data["by_strategy"] = dict(_eg_strat_counter.most_common(10))
            except Exception as _eg_err:
                logger.debug("Failed to build exec gate diagnostics: %s", _eg_err)
                exec_gate_data["error"] = str(_eg_err)

            # ── Near-miss tracking + feedback metrics ──────────────────
            near_miss_data = {}
            try:
                from analyzers.near_miss_tracker import near_miss_tracker
                near_miss_data = near_miss_tracker.get_metrics()
            except Exception as _nm_err:
                logger.debug("Failed to build near-miss metrics: %s", _nm_err)
                near_miss_data["error"] = str(_nm_err)

            # ── Signal Validator stats ────────────────────────────────
            validator_stats = {}
            try:
                from config.feature_flags import ff
                _vff = ff.get_state("SIGNAL_VALIDATOR")
                if _vff in ("live", "shadow"):
                    from signals.signal_validator import signal_validator
                    validator_stats = signal_validator.get_stats()
                    validator_stats["mode"] = _vff
            except Exception as _vs_err:
                logger.debug("Failed to load validator stats: %s", _vs_err)

            return _json_response({
                "pipeline": pipeline_diag,
                "kill_reasons": kill_reasons,           # from get_death_breakdown (sorted)
                "strategy_breakdown": strat_breakdown_agg,  # FIX #22: use aggregated version
                "veto_proposals": veto_proposals,
                "audit_status": audit_status,
                "stats": stats,
                "execution_gate": exec_gate_data,
                "near_miss_tracking": near_miss_data,
                "validator_stats": validator_stats,
            })
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_index(self, request: "web.Request") -> "web.Response":
        """Serve dashboard HTML.
        Always reads and serves as text/html - avoids FileResponse caching issues
        that were causing 239-byte responses in some browser/aiohttp combinations.
        """
        try:
            static_dir = Path(__file__).parent / "static"
            index_path = static_dir / "index.html"
            if index_path.exists():
                file_size = index_path.stat().st_size
                if file_size > 10000:
                    html_content = index_path.read_text(encoding="utf-8")
                    return web.Response(
                        text=html_content,
                        content_type="text/html",
                        charset="utf-8",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                                 "Pragma": "no-cache"}
                    )
        except Exception as e:
            logger.warning(f"Static file read error: {e} - using embedded HTML")
        # Fallback: embedded HTML
        return web.Response(
            text=_DASHBOARD_HTML,
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )


    async def _handle_equity_curve(self, request: "web.Request") -> "web.Response":
        """GET /api/equity-curve - daily cumulative R for equity chart."""
        try:
            from data.database import db
            days = int(request.rel_url.query.get("days", 60))
            data = await db.get_equity_curve(days=days)
            return _json_response({"curve": data, "days": days})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_quant_metrics(self, request: "web.Request") -> "web.Response":
        """GET /api/quant-metrics - Sharpe, Sortino, Calmar, drawdown, profit factor."""
        try:
            from data.database import db
            days = int(request.rel_url.query.get("days", 30))
            metrics = await db.get_quant_metrics(days=days)
            return _json_response(metrics)
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_monthly_pnl(self, request: "web.Request") -> "web.Response":
        """GET /api/monthly-pnl - monthly P&L for heatmap calendar."""
        try:
            from data.database import db
            data = await db.get_monthly_pnl(months=12)
            return _json_response({"months": data})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_regime_perf(self, request: "web.Request") -> "web.Response":
        """GET /api/regime-performance - performance by regime × direction."""
        try:
            from data.database import db
            days = int(request.rel_url.query.get("days", 60))
            data = await db.get_regime_performance(days=days)
            return _json_response({"rows": data})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_signal_funnel(self, request: "web.Request") -> "web.Response":
        """GET /api/signal-funnel - pipeline funnel stats."""
        try:
            from data.database import db
            from core.diagnostic_engine import diagnostic_engine
            hours = int(request.rel_url.query.get("hours", 24))
            db_funnel = await db.get_signal_funnel(hours=hours)
            # Add engine-level pre-filter stats
            db_funnel["scanned"] = getattr(diagnostic_engine, '_total_scan_count', 0)
            db_funnel["raw_generated"] = getattr(diagnostic_engine, '_total_signals_generated', 0)
            # Add validator kill/warning counts from death log
            try:
                death_log = diagnostic_engine._death_log
                import time as _t
                _cutoff = _t.time() - hours * 3600
                _recent = [d for d in death_log if d.get("ts", 0) >= _cutoff]
                db_funnel["validator_kills"] = sum(
                    1 for d in _recent
                    if d.get("kill_reason", "").startswith("VALIDATOR_")
                )
            except Exception:
                db_funnel["validator_kills"] = 0
            try:
                from config.feature_flags import ff
                if ff.get_state("SIGNAL_VALIDATOR") in ("live", "shadow"):
                    from signals.signal_validator import signal_validator
                    db_funnel["validator_warnings"] = signal_validator.get_warning_count(hours=hours)
                else:
                    db_funnel["validator_warnings"] = 0
            except Exception:
                db_funnel["validator_warnings"] = 0
            return _json_response(db_funnel)
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_positions(self, request: "web.Request") -> "web.Response":
        """GET /api/positions - open positions with live P&L from price cache."""
        try:
            from data.database import db
            from core.price_cache import price_cache
            positions = await db.load_open_positions()
            result = []
            for pos in positions:
                sym = pos.get("symbol", "")
                live_price = price_cache.get(sym)
                entry = pos.get("entry_price", 0)
                direction = pos.get("direction", "LONG")
                sl = pos.get("stop_loss", 0)
                tp1 = pos.get("tp1", 0)
                tp2 = pos.get("tp2", 0)
                risk = pos.get("risk_usdt", 0)

                pnl_r = 0.0
                pnl_usdt = 0.0
                dist_tp1_pct = None
                dist_sl_pct = None

                if live_price and entry and risk:
                    sl_dist = abs(entry - sl) if sl else 0
                    if direction == "LONG":
                        # pnl_r: direct from entry/sl geometry - not via pnl_usdt/risk
                        pnl_r = round((live_price - entry) / sl_dist, 3) if sl_dist else 0
                        pnl_usdt = round(pnl_r * risk, 2)
                        if sl:
                            # % distance remaining to SL, relative to entry (not live_price)
                            dist_sl_pct = round((live_price - sl) / entry * 100, 2)
                        if tp1:
                            dist_tp1_pct = round((tp1 - live_price) / entry * 100, 2)
                    else:  # SHORT
                        pnl_r = round((entry - live_price) / sl_dist, 3) if sl_dist else 0
                        pnl_usdt = round(pnl_r * risk, 2)
                        if sl:
                            dist_sl_pct = round((sl - live_price) / entry * 100, 2)
                        if tp1:
                            dist_tp1_pct = round((live_price - tp1) / entry * 100, 2)

                result.append({
                    **pos,
                    "live_price": live_price,
                    "pnl_r": pnl_r,
                    "pnl_usdt": pnl_usdt,
                    "dist_sl_pct": dist_sl_pct,
                    "dist_tp1_pct": dist_tp1_pct,
                    "age_seconds": int(time.time() - pos["created_at"]) if pos.get("created_at") else 0,
                })
            return _json_response({"positions": result, "count": len(result)})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_portfolio_risk(self, request: "web.Request") -> "web.Response":
        """GET /api/portfolio-risk - portfolio concentration, correlation, direction split."""
        try:
            from core.portfolio_engine import portfolio_engine
            from core.price_cache import price_cache
            positions = list(portfolio_engine._positions.values())
            total_risk = sum(p.risk_usdt for p in positions)
            capital = portfolio_engine._capital
            long_risk = sum(p.risk_usdt for p in positions if p.direction == "LONG")
            short_risk = sum(p.risk_usdt for p in positions if p.direction == "SHORT")
            high_corr = sum(1 for p in positions if p.correlation_to_btc >= 0.7)

            sectors = {}
            for p in positions:
                sectors[p.sector or "Other"] = sectors.get(p.sector or "Other", 0) + p.risk_usdt

            return _json_response({
                "position_count": len(positions),
                "total_risk_usdt": round(total_risk, 2),
                "total_risk_pct": round(total_risk / capital * 100, 2) if capital else 0,
                "long_risk_usdt": round(long_risk, 2),
                "short_risk_usdt": round(short_risk, 2),
                "high_btc_corr_count": high_corr,
                "sector_breakdown": sectors,
                "capital": round(capital, 2),
                "max_positions": portfolio_engine._max_positions,
                "max_total_risk_pct": round(portfolio_engine._max_total_risk_pct * 100, 1),
            })
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_learning_state(self, request: "web.Request") -> "web.Response":
        """GET /api/learning - Bayesian posteriors, evidence ratios, calibration."""
        try:
            from core.probability_engine import probability_engine
            from core.alpha_model import alpha_model
            from core.learning_loop import learning_loop

            # Top strategy posteriors
            posteriors = probability_engine.get_all_posteriors()
            # Sort by count (most data first)
            sorted_post = sorted(posteriors.items(), key=lambda x: -x[1].get("count", 0))

            # Evidence likelihoods
            likelihoods = dict(probability_engine._likelihood_ratios)

            # Strategy weights
            weights = alpha_model.get_all_weights()

            # Learning stats
            ll_stats = learning_loop.get_stats()

            # Calibration bins
            cal_bins = probability_engine._calibration_bins

            return _json_response({
                "posteriors": dict(sorted_post[:50]),
                "likelihoods": likelihoods,
                "strategy_weights": weights,
                "stats": ll_stats,
                "calibration_bins": cal_bins,
            })
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_backtest_run(self, request: "web.Request") -> "web.Response":
        """POST /api/backtest - run a backtest with optional IS/OOS split."""
        _auth_err = self._auth_required(request)
        if _auth_err: return _auth_err
        try:
            body = await request.json()
            symbol    = body.get("symbol", "BTC/USDT")
            days      = int(body.get("days", 90))
            strategies = body.get("strategies", None)
            oos_split  = float(body.get("oos_split", 0.0))   # e.g. 0.20 = last 20% held out

            from backtester.engine import BacktestEngine
            from data.api_client import api as exchange_api

            tfs = ["4h", "1h", "15m"]
            ohlcv_data = {}
            # Bars-per-day varies by timeframe; compute per-tf so we always
            # fetch enough history for the full requested window.
            _tf_bars_per_day = {"4h": 6, "1h": 24, "15m": 96}
            for tf in tfs:
                try:
                    _bpd = _tf_bars_per_day.get(tf, 24)
                    _tf_limit = min(days * _bpd + 300, 2000)
                    bars = await exchange_api.fetch_ohlcv(symbol, tf, limit=_tf_limit)
                    if bars:
                        ohlcv_data[tf] = bars
                except Exception as _e:
                    logger.debug("Failed to fetch OHLCV for timeframe %s: %s", tf, _e)

            if not ohlcv_data:
                return _json_response({"error": "No historical data available"})

            engine = BacktestEngine()
            raw = await engine.run(symbol=symbol, ohlcv_data=ohlcv_data, oos_split=oos_split)

            # IS/OOS split returns a dict; plain run returns BacktestResult
            if isinstance(raw, dict) and "in_sample" in raw:
                is_r  = raw["in_sample"]
                oos_r = raw["out_of_sample"]

                def _ser_result(r):
                    return {
                        "total_trades":    r.total_trades,
                        "win_rate":        round(r.win_rate, 1),
                        "total_r":         round(r.total_r, 2),
                        "avg_r":           round(r.avg_r, 3),
                        "sharpe":          round(getattr(r, 'sharpe_ratio', 0) or 0, 2),
                        "sortino":         round(getattr(r, 'sortino_ratio', 0) or 0, 2),
                        "calmar":          round(getattr(r, 'calmar_ratio', 0) or 0, 2),
                        "max_drawdown":    round(getattr(r, 'max_drawdown_r', 0) or 0, 3),
                        "profit_factor":   round(getattr(r, 'profit_factor', 0) or 0, 2),
                        "avg_mae_r":       round(getattr(r, 'avg_mae_r', 0) or 0, 3),
                        "avg_mfe_r":       round(getattr(r, 'avg_mfe_r', 0) or 0, 3),
                        "strategy_breakdown": getattr(r, 'strategy_breakdown', {}),
                        "regime_breakdown":   getattr(r, 'regime_breakdown', {}),
                        "monte_carlo":        getattr(r, 'monte_carlo', {}),
                        "walk_forward":       getattr(r, 'walk_forward', []),
                        "trades": [
                            {"symbol": t.symbol, "strategy": t.strategy, "direction": t.direction,
                             "pnl_r": round(t.pnl_r, 3), "outcome": t.outcome,
                             "confidence": t.confidence}
                            for t in (r.trades or [])[-50:]
                        ],
                    }

                return _json_response({
                    "mode":              "oos_split",
                    "oos_split":         oos_split,
                    "overfitting_decay": raw.get("overfitting_decay"),
                    "overfitting_warn":  raw.get("overfitting_decay", 1.0) < 0.5,
                    "in_sample":         _ser_result(is_r),
                    "out_of_sample":     _ser_result(oos_r),
                    "symbol":            symbol,
                })

            # Plain (non-OOS) result
            result = raw
            if not result:
                return _json_response({"error": "Backtest produced no results"})

            trades = [
                {"symbol": t.symbol, "strategy": t.strategy, "direction": t.direction,
                 "entry_price": round(t.entry_price, 6),
                 "exit_price":  round(t.exit_price, 6) if t.exit_price else None,
                 "pnl_r": round(t.pnl_r, 3), "outcome": t.outcome,
                 "confidence": t.confidence, "bars_to_outcome": t.bars_to_outcome}
                for t in (result.trades or [])
            ]

            walk_windows = engine.run_walk_forward(
                result.trades or [],
                window_trades=max(20, len(result.trades) // 8),
                step_trades=max(5, len(result.trades) // 20),
            )

            return _json_response({
                "mode":             "full",
                "symbol":           symbol,
                "total_trades":     result.total_trades,
                "win_rate":         round(result.win_rate, 1),
                "total_r":          round(result.total_r, 2),
                "avg_r":            round(result.avg_r, 3),
                "sharpe":           round(getattr(result, 'sharpe_ratio', 0) or 0, 2),
                "sortino":          round(getattr(result, 'sortino_ratio', 0) or 0, 2),
                "calmar":           round(getattr(result, 'calmar_ratio', 0) or 0, 2),
                "max_drawdown":     round(getattr(result, 'max_drawdown_r', 0) or 0, 3),
                "profit_factor":    round(getattr(result, 'profit_factor', 0) or 0, 2),
                "avg_mae_r":        round(getattr(result, 'avg_mae_r', 0) or 0, 3),
                "avg_mfe_r":        round(getattr(result, 'avg_mfe_r', 0) or 0, 3),
                "best_trade_r":     round(getattr(result, 'best_trade_r', 0) or 0, 2),
                "worst_trade_r":    round(getattr(result, 'worst_trade_r', 0) or 0, 2),
                "max_consec_wins":  getattr(result, 'max_consecutive_wins', 0),
                "max_consec_losses": getattr(result, 'max_consecutive_losses', 0),
                "strategy_breakdown": getattr(result, 'strategy_breakdown', {}),
                "regime_breakdown":   getattr(result, 'regime_breakdown', {}),
                "monte_carlo":        getattr(result, 'monte_carlo', {}),
                "walk_forward":       walk_windows,
                "trades":             trades[-100:],
            })
        except Exception as e:
            return _json_response({"error": str(e)})



    async def _handle_signal_note(self, request: "web.Request") -> "web.Response":
        """POST /api/signals/{id}/note - save or clear a trade note."""
        _auth_err = self._auth_required(request)
        if _auth_err: return _auth_err
        try:
            sid = int(request.match_info["id"])
            body = await request.json()
            note = str(body.get("note", "")).strip()
            from data.database import db
            ok = await db.set_signal_note(sid, note)
            return _json_response({"success": ok, "signal_id": sid, "note": note})
        except Exception as e:
            return _json_response({"success": False, "error": str(e)})

    async def _handle_journal(self, request: "web.Request") -> "web.Response":
        """GET /api/journal - signals with trader notes."""
        try:
            from data.database import db
            rows = await db.get_signal_notes(limit=200)
            return _json_response({"notes": rows, "entries": rows, "count": len(rows)})
        except Exception as e:
            return _json_response({"error": str(e)})


    async def _handle_institutional(self, request: "web.Request") -> "web.Response":
        """GET /api/institutional - institutional flow engine data for dashboard."""
        try:
            from analyzers.institutional_flow import institutional_flow
            return _json_response(institutional_flow.get_dashboard_data())
        except Exception as e:
            return _json_response({"error": str(e), "signals": [], "macro": {}})

    async def _handle_drawdown(self, request: "web.Request") -> "web.Response":
        """GET /api/drawdown?days=90 - equity drawdown curve for performance analytics."""
        try:
            days = int(request.rel_url.query.get("days", 90))
            from data.database import db
            cutoff = time.time() - days * 86400
            async with db._lock:
                rows = await db._fetchall(
                    "SELECT created_at, final_r FROM signals "
                    "WHERE outcome IN ('WIN','LOSS','PARTIAL') AND created_at > ? "
                    "ORDER BY created_at ASC",
                    (cutoff,)
                )

            cum_r = 0.0
            peak_r = 0.0
            points = []
            for created_at, final_r in rows:
                cum_r += (final_r or 0)
                if cum_r > peak_r:
                    peak_r = cum_r
                drawdown = cum_r - peak_r  # always <= 0
                points.append({
                    "t": created_at,
                    "cum_r": round(cum_r, 3),
                    "dd": round(drawdown, 3),
                })

            max_dd = min((p["dd"] for p in points), default=0.0)
            return _json_response({"points": points, "max_dd": round(max_dd, 3)})
        except Exception as e:
            return _json_response({"error": str(e), "points": []})

    async def _handle_btc_intel_history(self, request: "web.Request") -> "web.Response":
        """GET /api/btc-intel-history - recent BTC event classifications."""
        try:
            from analyzers.btc_news_intelligence import btc_news_intelligence
            ctx = btc_news_intelligence.get_event_context()
            history = getattr(btc_news_intelligence, '_event_history', [])
            return _json_response({
                "current": {
                    "event_type": ctx.event_type.value if ctx else "UNKNOWN",
                    "direction": ctx.direction if ctx else "NEUTRAL",
                    "confidence": round(ctx.confidence, 3) if ctx else 0,
                    "headline": ctx.triggering_headline if ctx else "",
                    "detected_at": ctx.detected_at if ctx else 0,
                    "expires_at": ctx.expires_at if ctx else 0,
                    "confidence_mult": ctx.confidence_mult if ctx else 1.0,
                },
                "history": history[-50:] if history else [],
            })
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_approval_impact_preview(self, request: "web.Request") -> "web.Response":
        """GET /api/approvals/impact-preview/{id} - simulate impact of applying a pending approval."""
        try:
            approval_id = request.match_info.get("id", "")
            from core.diagnostic_engine import diagnostic_engine
            approval = diagnostic_engine._pending_approvals.get(approval_id)
            if not approval:
                return _json_response({"error": "not found"})

            from data.database import db
            cutoff = time.time() - 7 * 86400  # last 7 days
            async with db._lock:
                total = await db._fetchone(
                    "SELECT COUNT(*) FROM signals WHERE created_at > ?", (cutoff,)
                )
                total = (total or [0])[0]

            ct = approval.change_type
            nv = str(approval.new_value)
            ov = str(approval.old_value)

            preview_text = f"Over the last 7 days ({total} signals):\n"
            if ct == "adjust_rr_floor":
                try:
                    new_rr = float(nv)
                    async with db._lock:
                        blocked = await db._fetchone(
                            "SELECT COUNT(*) FROM signals WHERE created_at > ? AND json_extract(raw_data,'$.rr') < ?",
                            (cutoff, new_rr)
                        )
                        blocked = (blocked or [0])[0]
                    preview_text += f"~{blocked} signals would have been blocked by RR floor {ov} → {nv}"
                except Exception:
                    preview_text += f"RR floor change: {ov} → {nv}"
            elif ct == "adjust_confidence_floor":
                try:
                    new_cf = float(nv)
                    async with db._lock:
                        blocked = await db._fetchone(
                            "SELECT COUNT(*) FROM signals WHERE created_at > ? AND confidence < ?",
                            (cutoff, new_cf)
                        )
                        blocked = (blocked or [0])[0]
                    preview_text += f"~{blocked} signals would have been blocked by confidence floor → {nv}"
                except Exception:
                    preview_text += f"Confidence floor change: {ov} → {nv}"
            else:
                preview_text += f"Change type: {ct}, {ov} → {nv}"

            return _json_response({
                "approval_id": approval_id,
                "change_type": ct,
                "old_value": ov,
                "new_value": nv,
                "preview": preview_text,
                "signals_last_7d": total,
            })
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_journal_manual(self, request: "web.Request") -> "web.Response":
        """POST /api/journal/manual - add a manual trade entry to the journal."""
        try:
            _auth_err = self._auth_required(request)
            if _auth_err: return _auth_err
            body = await request.json()
            symbol   = str(body.get("symbol", "")).upper().strip()[:20]
            direction = str(body.get("direction", "LONG"))
            outcome  = str(body.get("outcome", "WIN"))
            pnl_r    = float(body.get("pnl_r", 0))
            strategy = str(body.get("strategy", "Manual"))[:60]
            notes    = str(body.get("user_notes", ""))[:500]
            if not symbol:
                return _json_response({"ok": False, "error": "symbol required"})
            from data.database import db
            now = time.time()
            async with db._lock:
                await db._exec(
                    "INSERT INTO signals (symbol, direction, strategy, outcome, final_r, "
                    "confidence, created_at, user_taken, user_notes) "
                    "VALUES (?,?,?,?,?,0,?,1,?)",
                    (symbol, direction, strategy, outcome, pnl_r, now, notes)
                )
                await db._conn.commit()
            return _json_response({"ok": True})
        except Exception as e:
            logger.warning(f"journal manual entry error: {e}")
            return _json_response({"ok": False, "error": str(e)})

    async def _handle_sentinel_audit_symbol(self, request: "web.Request") -> "web.Response":
        """POST /api/sentinel/audit - queue a fast audit for a specific symbol."""
        try:
            _auth_err = self._auth_required(request)
            if _auth_err: return _auth_err
            body   = await request.json()
            symbol = str(body.get("symbol", "")).upper().strip()
            if not symbol:
                return _json_response({"ok": False, "error": "symbol required"})
            from analyzers.ai_analyst import ai_analyst
            # Schedule a fast audit in the background (non-blocking)
            # Note: structural_audit is session-wide; symbol is logged for context only
            import asyncio
            task = asyncio.create_task(ai_analyst.structural_audit(force=True))
            task.add_done_callback(
                lambda t: logger.warning(f"Sentinel audit task failed: {t.exception()}")
                if not t.cancelled() and t.exception() else None
            )
            return _json_response({
                "ok": True, "symbol": symbol, "queued": True,
                "note": "Runs session-wide structural audit (symbol-specific audit not yet supported)",
            })
        except Exception as e:
            logger.warning(f"sentinel audit symbol error: {e}")
            return _json_response({"ok": False, "error": str(e)})

    async def _handle_ai_decisions(self, request: "web.Request") -> "web.Response":
        """GET /api/ai-decisions - last 50 structured AI decision records."""
        try:
            from analyzers.ai_analyst import ai_analyst
            history = ai_analyst.get_decision_history()
            ok_count   = sum(1 for d in history if d.get("outcome") == "ok")
            fail_count = sum(1 for d in history if d.get("outcome") in ("error", "timeout"))
            avg_latency = (
                sum(d.get("latency_ms", 0) for d in history if d.get("outcome") == "ok")
                / max(ok_count, 1)
            )
            return _json_response({
                "decisions": history,
                "count": len(history),
                "ok": ok_count,
                "failed": fail_count,
                "avg_latency_ms": int(avg_latency),
            })
        except Exception as e:
            return _json_response({"error": str(e), "decisions": []})

    async def _handle_btc_news(self, request: "web.Request") -> "web.Response":
        """GET /api/btc-news - BTC news event classification and altcoin impact context."""
        try:
            from analyzers.btc_news_intelligence import btc_news_intelligence, BTCEventType
            from analyzers.news_scraper import news_scraper

            status = btc_news_intelligence.get_status()

            # Attach recent BTC/macro headlines for the dashboard feed
            recent_btc = [
                n for n in news_scraper.get_all_stories(max_age_mins=120, limit=50)
                if any(kw in n.get("title", "").lower()
                       for kw in ["bitcoin", "btc", "fed", "inflation", "rate",
                                  "crypto", "sec", "etf", "hack", "regulation"])
            ][:20]

            # Build event type history from news cache scores for the chart
            from collections import Counter
            type_counts: Counter = Counter()
            bull_pressure = 0
            bear_pressure = 0
            for h in recent_btc:
                etype, direction, conf, _is_mixed = btc_news_intelligence._classifier.classify(h.get("title", ""))
                if conf > 0.12:
                    type_counts[etype.value] += 1
                    if direction == "BULLISH":
                        bull_pressure += 1
                    elif direction == "BEARISH":
                        bear_pressure += 1

            return _json_response({
                "context":       status,
                "headlines":     recent_btc,
                "type_counts":   dict(type_counts),
                "bull_pressure": bull_pressure,
                "bear_pressure": bear_pressure,
                "event_types":   [t.value for t in BTCEventType],
            })
        except Exception as e:
            return _json_response({"error": str(e), "context": {}, "headlines": []})

    async def _build_signal_detail_payload(self, signal_id: int, sig: Dict[str, Any]) -> Dict[str, Any]:
        # Get live scored signal from bot memory if available
        scored = None
        try:
            from tg.bot import telegram_bot
            rec = telegram_bot._by_signal_id.get(signal_id)
            if rec and rec.scored:
                scored = rec.scored
        except Exception as _e:
            logger.debug("Failed to load scored signal from bot memory: %s", _e)

        from utils.signal_guidance import guidance_payload

        raw = {}
        try:
            raw = json.loads(sig.get("raw_scores") or "{}")
        except Exception as _e:
            logger.debug("Failed to parse raw_scores: %s", _e)

        confluence = []
        try:
            _conf_raw = sig.get("confluence") or "[]"
            confluence = json.loads(_conf_raw)
            if isinstance(confluence, list):
                confluence = [str(c) for c in confluence]
        except Exception as _e:
            logger.debug("Failed to parse confluence: %s", _e)

        def _score(attr, raw_key, default=0):
            if scored and hasattr(scored, attr):
                val = getattr(scored, attr)
                if val is not None:
                    return val
            return raw.get(raw_key, raw.get(attr, default))

        result = {
            "signal_id": signal_id,
            "symbol": sig.get("symbol"),
            "direction": sig.get("direction"),
            "strategy": sig.get("strategy"),
            "confidence": sig.get("confidence"),
            "p_win": sig.get("p_win"),
            "alpha_grade": sig.get("alpha_grade"),
            "regime": sig.get("regime"),
            "sector": sig.get("sector"),
            "setup_class": sig.get("setup_class", "intraday"),
            "confluence": confluence,
            "technical_score": _score("technical_score", "technical", 50),
            "volume_score": _score("volume_score", "volume", 50),
            "orderflow_score": _score("orderflow_score", "orderflow", 50),
            "derivatives_score": _score("derivatives_score", "derivatives", 50),
            "sentiment_score": _score("sentiment_score", "sentiment", 50),
            "has_ob": raw.get("has_ob", False),
            "has_fvg": raw.get("has_fvg", False),
            "has_sweep": raw.get("has_sweep", False),
            "wave_type": raw.get("wave_type", ""),
            "wyckoff_event": raw.get("wyckoff_event", ""),
            "htf_structure": raw.get("htf_structure", ""),
            "adx": float(raw.get("adx", 0) or 0),
            "rsi": float(raw.get("rsi", 0) or 0),
            "funding_rate": float(raw.get("funding_rate", 0) or 0),
            "vol_ratio": float(raw.get("vol_ratio", 0) or 0),
            "ichimoku_above_cloud": any("above cloud" in str(c).lower() for c in confluence),
            "ichimoku_below_cloud": any("below cloud" in str(c).lower() for c in confluence),
            "ichimoku_tk_aligned": any("tk aligned" in str(c).lower() for c in confluence),
            "ichimoku_chikou": any("chikou" in str(c).lower() for c in confluence),
            "entry_low": sig.get("entry_low"),
            "entry_high": sig.get("entry_high"),
            "stop_loss": sig.get("stop_loss"),
            "tp1": sig.get("tp1"),
            "tp2": sig.get("tp2"),
            "tp3": sig.get("tp3"),
            "rr_ratio": sig.get("rr_ratio"),
            "ev_r": sig.get("ev_r", 0),
            "confidence_adjustments": raw.get("confidence_adjustments") or [],
            "outcome": sig.get("outcome"),
            "pnl_r": sig.get("pnl_r"),
            "entry_price": sig.get("entry_price"),
            "entry_time": sig.get("entry_time"),
            "exit_price": sig.get("exit_price"),
            "exit_reason": sig.get("exit_reason"),
            "max_r": sig.get("max_r"),
            "zone_reached": sig.get("zone_reached", 0),
            "zone_reached_at": sig.get("zone_reached_at"),
            "publish_price": sig.get("publish_price"),
            "published": sig.get("message_id") is not None,
            "exec_state": "ALMOST" if sig.get("exec_state", "WATCHING") == "ARMED" else sig.get("exec_state", "WATCHING"),
            "created_at": sig.get("created_at"),
            "power_aligned": getattr(scored, "power_aligned", False) if scored else False,
            "power_alignment_reason": getattr(scored, "power_alignment_reason", "") if scored else "",
            "setup_context": raw.get("setup_context"),
            "execution_context": raw.get("execution_context"),
            "execution_score": raw.get("execution_score"),
            "execution_factors": raw.get("execution_factors"),
            "execution_bad_factors": raw.get("execution_bad_factors"),
            "execution_kill_combo": raw.get("execution_kill_combo"),
        }
        guidance = guidance_payload(sig, confluence=confluence)
        result["fee_adjusted_rr"] = guidance.get("fee_adjusted_rr")
        result["round_trip_friction_pct"] = guidance.get("round_trip_friction_pct")
        result["expected_fill_low"] = guidance.get("expected_fill_low")
        result["expected_fill_high"] = guidance.get("expected_fill_high")
        result["expected_fill_mid"] = guidance.get("expected_fill_mid")
        result["skip_rule"] = guidance.get("skip_rule")
        result["skip_level"] = guidance.get("skip_level")
        result["session_warning"] = guidance.get("session_warning")
        result["size_modifier"] = guidance.get("size_modifier")
        return result

    async def _handle_signal_detail(self, request: "web.Request") -> "web.Response":
        """GET /api/signals/{id}/detail - full scored signal data for dashboard panels."""
        try:
            signal_id = int(request.match_info.get("id", 0))
            from data.database import db
            sig = await db.get_signal(signal_id)
            if not sig:
                return _json_response({"error": "Signal not found"})
            return _json_response(await self._build_signal_detail_payload(signal_id, sig))
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_signal_decision(self, request: "web.Request") -> "web.Response":
        """GET /api/signals/{id}/decision - run an on-demand AI decision review."""
        try:
            signal_id = int(request.match_info.get("id", 0))
            from data.database import db
            sig = await db.get_signal(signal_id)
            if not sig:
                return _json_response({"error": "Signal not found"}, 404)
            from config.loader import cfg
            from signals.signal_decision import review_signal
            detail = await self._build_signal_detail_payload(signal_id, sig)
            review = await review_signal(
                sig,
                detail,
                openrouter_api_key=cfg.ai.get("openrouter_api_key", ""),
            )
            return _json_response({"signal_id": signal_id, **review})
        except Exception as e:
            return _json_response({"error": str(e)})

    async def _handle_signal_enter(self, request: "web.Request") -> "web.Response":
        """POST /api/signals/{id}/enter - user confirms entry from dashboard (Option B flow)."""
        try:
            # R5-S4 FIX: Missing authentication — this endpoint modifies DB and
            # registers positions in portfolio engine. All other POST handlers
            # use _auth_required(). Without this, any network user can execute trades.
            _auth_err = self._auth_required(request)
            if _auth_err:
                return _auth_err

            signal_id = int(request.match_info.get("id", 0))
            from data.database import db
            from core.price_cache import price_cache
            from core.portfolio_engine import portfolio_engine

            sig = await db.get_signal(signal_id)
            if not sig:
                return _json_response({"ok": False, "error": "Signal not found"})

            symbol    = sig.get("symbol", "")
            direction = sig.get("direction", "LONG")
            entry_now = price_cache.get(symbol) or ((sig.get("entry_low",0)+sig.get("entry_high",0))/2)
            stop_loss = sig.get("stop_loss", 0) or entry_now * 0.98
            stop_dist = abs(entry_now - stop_loss) / entry_now if entry_now > 0 else 0.02
            risk_usdt = sig.get("risk_amount") or (portfolio_engine._capital * 0.01)
            size_usdt = sig.get("position_size") or (risk_usdt / max(stop_dist, 0.001))

            # Update DB - mark as user-entered
            async with db._lock:
                await db._conn.execute(
                    "UPDATE signals SET user_taken=1, user_notes=? WHERE id=?",
                    (f"Dashboard entry @ {entry_now:.6g}", signal_id)
                )
                await db._conn.commit()

            # Save open position
            await db.save_open_position(
                signal_id=signal_id, symbol=symbol, direction=direction,
                strategy=sig.get("strategy",""), entry_price=entry_now,
                size_usdt=size_usdt, risk_usdt=risk_usdt, stop_loss=stop_loss,
                tp1=sig.get("tp1",0), tp2=sig.get("tp2",0), tp3=sig.get("tp3"),
                rr_ratio=sig.get("rr_ratio",0), sector=sig.get("sector","") or "",
                message_id=sig.get("message_id"),
            )

            # Register with portfolio engine
            try:
                portfolio_engine.open_position(
                    symbol=symbol, direction=direction,
                    strategy=sig.get("strategy",""), entry_price=entry_now,
                    size_usdt=size_usdt, risk_usdt=risk_usdt, stop_loss=stop_loss,
                    sector=sig.get("sector","") or "", signal_id=signal_id,
                )
            except Exception as _e:
                logger.debug("Failed to register position with portfolio engine: %s", _e)

            return _json_response({"ok": True, "entry_price": entry_now, "signal_id": signal_id})
        except Exception as e:
            return _json_response({"ok": False, "error": str(e)})

    async def _handle_signal_check(self, request: "web.Request") -> "web.Response":
        """GET /api/signals/{id}/check - run mid-trade thesis check via AI."""
        try:
            signal_id = int(request.match_info.get("id", 0))
            from data.database import db
            signal = await db.get_signal(signal_id)
            if not signal:
                return _json_response({"error": "Signal not found"}, 404)

            from signals.outcome_monitor import outcome_monitor
            from core.price_cache import price_cache
            from config.loader import cfg
            from signals.thesis_checker import check_thesis

            tracked = outcome_monitor.get_active_signals().get(signal_id)
            sym = signal.get("symbol", "")
            current_price = price_cache.get(sym) or 0.0
            current_r = 0.0
            max_r = 0.0
            if tracked:
                current_r = outcome_monitor.calc_pnl_r(tracked, current_price)
                max_r     = tracked.max_r

            raw = {}
            try:
                import json as _json
                raw = _json.loads(signal.get("raw_data") or "{}")
            except Exception as _e:
                logger.debug("Failed to parse raw_data for thesis check: %s", _e)

            text = await check_thesis(
                signal_id=signal_id,
                symbol=sym,
                direction=signal.get("direction", "LONG"),
                strategy=signal.get("strategy", ""),
                entry_price=tracked.entry_price if tracked else (
                    (signal.get("entry_low", 0) + signal.get("entry_high", 0)) / 2
                ),
                stop_loss=signal.get("stop_loss", 0),
                tp1=signal.get("tp1", 0),
                tp2=signal.get("tp2", 0),
                created_at=signal.get("created_at", 0),
                activated_at=tracked.activated_at if tracked else None,
                raw_data=raw,
                current_price=current_price,
                current_r=current_r,
                max_r=max_r,
                openrouter_api_key=cfg.ai.get("openrouter_api_key", ""),
            )
            return _json_response({"text": text, "signal_id": signal_id})
        except Exception as e:
            return _json_response({"error": str(e)})

    def _check_auth(self, request) -> bool:
        """
        FIX #28: Validate DASHBOARD_AUTH_TOKEN on all mutating POST endpoints.
        Token must be passed via the X-Auth-Token header only.
        URL query-param tokens (?token=...) are intentionally NOT supported:
        they appear in server access logs and browser history, leaking secrets.
        Returns True if auth passes or no token is configured (open dev mode).

        Uses hmac.compare_digest for constant-time comparison to prevent
        timing-based token oracle attacks.
        """
        from config.loader import cfg
        expected = getattr(cfg, '_dashboard_auth_token', '') or ''
        if not expected:
            return True  # No token configured → open mode (local dev)
        provided = request.headers.get('X-Auth-Token', '')
        # Reject immediately if no token was provided - avoids compare_digest
        # with an empty string (compare_digest(b'', b'secret') returns False
        # correctly, but an explicit guard is clearer and future-safe).
        if not provided:
            return False
        # compare_digest requires both operands to be the same type
        return hmac.compare_digest(
            provided.encode('utf-8'),
            expected.encode('utf-8'),
        )

    def _auth_required(self, request):
        """Return 401 response if auth fails, None if passes."""
        from aiohttp import web as _web
        if not self._check_auth(request):
            return _web.Response(
                status=401,
                content_type='application/json',
                text='{"error":"Unauthorized - set X-Auth-Token header"}',
            )
        return None

    async def _handle_adaptive_state(self, request: "web.Request") -> "web.Response":
        """GET /api/adaptive-state - expose all adaptive system feature states."""
        result = {}
        # Adaptive weights
        try:
            from signals.adaptive_weights import adaptive_weight_manager
            result["adaptive_weights"] = adaptive_weight_manager.get_diagnostics()
        except Exception as e:
            result["adaptive_weights"] = {"error": str(e)}

        # Stablecoin flow dynamics
        try:
            from analyzers.stablecoin_flows import stablecoin_analyzer
            snap = stablecoin_analyzer.get_snapshot()
            dynamics = stablecoin_analyzer.get_flow_dynamics()
            result["stablecoin_flows"] = {
                "supply": snap.supply.__dict__ if hasattr(snap.supply, '__dict__') else str(snap.supply),
                "dominance": snap.dominance.__dict__ if hasattr(snap.dominance, '__dict__') else str(snap.dominance),
                "dynamics": dynamics.__dict__ if hasattr(dynamics, '__dict__') else str(dynamics),
                "last_update": snap.last_update,
            }
        except Exception as e:
            result["stablecoin_flows"] = {"error": str(e)}

        # Whale behavior
        try:
            from analyzers.wallet_behavior import wallet_profiler
            snap = wallet_profiler.get_snapshot()
            result["whale_behavior"] = {
                "phase": snap.phase.__dict__ if hasattr(snap.phase, '__dict__') else str(snap.phase),
                "pattern": snap.pattern.__dict__ if hasattr(snap.pattern, '__dict__') else str(snap.pattern),
                "coordination": snap.coordination.__dict__ if hasattr(snap.coordination, '__dict__') else str(snap.coordination),
                "intent": wallet_profiler.get_whale_intent(),
                "last_update": snap.last_update,
            }
        except Exception as e:
            result["whale_behavior"] = {"error": str(e)}

        # Volatility compression / regime
        try:
            from analyzers.volatility_structure import volatility_analyzer
            snap = volatility_analyzer.get_snapshot()
            result["volatility"] = {
                "realized": snap.realized.__dict__ if hasattr(snap.realized, '__dict__') else str(snap.realized),
                "implied": snap.implied.__dict__ if hasattr(snap.implied, '__dict__') else str(snap.implied),
                "regime": snap.regime.__dict__ if hasattr(snap.regime, '__dict__') else str(snap.regime),
                "last_update": snap.last_update,
            }
        except Exception as e:
            result["volatility"] = {"error": str(e)}

        # Cascade pressure
        try:
            from analyzers.leverage_mapper import leverage_mapper
            snap = leverage_mapper.get_snapshot()
            result["cascade_pressure"] = {
                "leverage": snap.leverage.__dict__ if hasattr(snap.leverage, '__dict__') else str(snap.leverage),
                "cascade": snap.cascade.__dict__ if hasattr(snap.cascade, '__dict__') else str(snap.cascade),
                "pressure": snap.pressure.__dict__ if hasattr(snap.pressure, '__dict__') else str(snap.pressure),
                "last_update": snap.last_update,
            }
        except Exception as e:
            result["cascade_pressure"] = {"error": str(e)}

        # Regime transition
        try:
            from analyzers.regime import regime_analyzer
            result["regime_transition"] = {
                "regime": getattr(regime_analyzer.regime, 'value', 'UNKNOWN'),
                "last_update": getattr(regime_analyzer, '_last_update', 0),
            }
            # Add early warning if available
            if hasattr(regime_analyzer, 'get_transition_warning'):
                result["regime_transition"]["warning"] = regime_analyzer.get_transition_warning()
        except Exception as e:
            result["regime_transition"] = {"error": str(e)}

        return _json_response(result)


    def _build_app(self) -> "web.Application":
        app = web.Application()

        # Routes
        app.router.add_get("/",              self._handle_index)
        app.router.add_get("/api/signals",          self._handle_signals)
        app.router.add_post("/api/signals/{id}/tag", self._handle_tag_signal)
        app.router.add_get("/api/stats",     self._handle_stats)
        app.router.add_get("/api/cohorts",   self._handle_cohorts)
        app.router.add_get("/api/news",      self._handle_news)
        app.router.add_get("/api/wallets",   self._handle_wallets)
        app.router.add_get("/api/status",    self._handle_engine_status)
        app.router.add_get("/api/prices",     self._handle_prices)
        app.router.add_get("/api/sentinel",   self._handle_sentinel_status)
        app.router.add_get("/api/stream",     self._handle_sse)
        app.router.add_get("/api/debug",      self._handle_debug)
        app.router.add_get("/api/approvals",             self._handle_approvals_list)
        app.router.add_post("/api/approvals/{id}/apply", self._handle_approval_apply)
        app.router.add_post("/api/approvals/{id}/reject",self._handle_approval_reject)
        app.router.add_get("/api/market-intel",     self._handle_market_intel)
        app.router.add_get("/api/smart-money",      self._handle_smart_money)
        app.router.add_get("/api/microstructure",   self._handle_microstructure)
        app.router.add_post("/api/veto/revert",          self._handle_veto_revert)
        app.router.add_get("/api/scan/{symbol}",         self._handle_scan_symbol)  # FIX: was POST
        app.router.add_post("/api/chat",                 self._handle_chat)
        app.router.add_get("/api/diagnostics",           self._handle_diagnostics)
        # NEW endpoints
        app.router.add_get("/api/markets",               self._handle_markets)
        app.router.add_get("/api/settings",              self._handle_settings_get)
        app.router.add_post("/api/settings",             self._handle_settings_post)
        app.router.add_get("/api/narrative/{symbol}",    self._handle_coin_narrative)

        # New exchange-quality dashboard endpoints
        app.router.add_get("/api/equity-curve",          self._handle_equity_curve)
        app.router.add_get("/api/quant-metrics",         self._handle_quant_metrics)
        app.router.add_get("/api/monthly-pnl",           self._handle_monthly_pnl)
        app.router.add_get("/api/regime-performance",    self._handle_regime_perf)
        app.router.add_get("/api/signal-funnel",         self._handle_signal_funnel)
        app.router.add_get("/api/positions",             self._handle_positions)
        app.router.add_get("/api/portfolio-risk",        self._handle_portfolio_risk)
        app.router.add_get("/api/learning",              self._handle_learning_state)
        app.router.add_post("/api/backtest",             self._handle_backtest_run)
        app.router.add_post("/api/signals/{id}/note",    self._handle_signal_note)
        app.router.add_get("/api/signals/{id}/check",    self._handle_signal_check)
        app.router.add_get("/api/signals/{id}/detail",   self._handle_signal_detail)
        app.router.add_get("/api/signals/{id}/decision", self._handle_signal_decision)
        app.router.add_post("/api/signals/{id}/enter",   self._handle_signal_enter)
        app.router.add_get("/api/journal",               self._handle_journal)
        app.router.add_get("/api/institutional",          self._handle_institutional)
        app.router.add_get("/api/btc-news",               self._handle_btc_news)   # NEW: BTC event intelligence
        app.router.add_get("/api/drawdown",                       self._handle_drawdown)
        app.router.add_get("/api/btc-intel-history",              self._handle_btc_intel_history)
        app.router.add_get("/api/approvals/impact-preview/{id}",  self._handle_approval_impact_preview)
        app.router.add_post("/api/journal/manual",               self._handle_journal_manual)
        app.router.add_post("/api/sentinel/audit",               self._handle_sentinel_audit_symbol)
        app.router.add_get("/api/ai-decisions",                  self._handle_ai_decisions)
        app.router.add_get("/api/adaptive-state",                 self._handle_adaptive_state)

        # Static files
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            app.router.add_static("/static", static_dir)

        # CORS - allow common localhost ports
        try:
            _opts = aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*")
            cors = aiohttp_cors.setup(app, defaults={
                "http://localhost:3000": _opts,
                "http://localhost:5173": _opts,
                "http://127.0.0.1:8080": _opts,
            })
            for route in list(app.router.routes()):
                try: cors.add(route)
                except Exception as _e: logger.debug("Failed to add CORS route: %s", _e)
        except Exception as _e:
            logger.debug("Failed to setup CORS: %s", _e)

        return app

    async def start(self, port: int = 8080):
        if not _AIOHTTP_AVAILABLE:
            return
        self._port = port
        self._app = self._build_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", port)
        try:
            await self._site.start()
            logger.info(f"🌐 Web dashboard: http://localhost:{port}")
        except OSError as e:
            logger.warning(f"Web dashboard failed to start on port {port}: {e}")

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()


dashboard = DashboardApp()


if __name__ == "__main__":
    # Standalone mode
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    async def main():
        await dashboard.start(8080)
        print("Dashboard running at http://localhost:8080 - Ctrl+C to stop")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await dashboard.stop()

    asyncio.run(main())
