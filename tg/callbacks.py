"""
CallbacksMixin — all callback-query handlers for TelegramBot.

Extracted from tg/bot.py to keep the main bot class lean.
Each method is an instance method that accesses TelegramBot
attributes through ``self`` at runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from data.database import db
from tg import keyboards
from tg.formatter import formatter
from tg.session_state import TradeState

if TYPE_CHECKING:
    pass  # no direct reference needed; mixin merges at runtime

logger = logging.getLogger(__name__)


class CallbacksMixin:
    """Mixin providing ``_handle_callback``, ``_handle_ai_callbacks``, and all ``_cb_*`` helpers."""

    # Callback dispatcher
    # ─────────────────────────────────────────────────────────

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not self._auth_query(query):
            return
        await query.answer()

        data    = query.data or ""
        chat_id = str(query.message.chat_id)

        try:
            # ── AI analyst callbacks ─────────────────────────────────────
            if (data.startswith("ai_") or data.startswith("approve_")):
                await self._handle_ai_callbacks(query, data)
                return

            # ── Inline panel routing (new v6 format: panel:{sid}:{action}) ──
            if data.startswith("panel:") and not data.startswith("panel:close:"):
                await self._cb_panel_inline(query, data)
                return

            # ── Signal analysis buttons (legacy format: action_SYM_TF_ID) ───
            if data.startswith("dismiss_"):
                # V17: Dismiss signal — remove keyboard, mark as ignored
                try:
                    await query.edit_message_reply_markup(reply_markup=None)
                    await query.answer("Signal dismissed")
                except Exception:
                    await query.answer("Already dismissed")
                return

            elif data.startswith("market_"):
                await self._cb_panel_button(query, data, "market")
            elif data.startswith("plan_"):
                await self._cb_panel_button(query, data, "plan")
            elif data.startswith("logic_"):
                await self._cb_panel_button(query, data, "logic")
            elif data.startswith("metrics_"):
                await self._cb_panel_button(query, data, "metrics")
            elif data.startswith("profile_"):
                await self._cb_panel_button(query, data, "profile")

            # ── Active trade buttons ─────────────────────────────────────
            elif data.startswith("live_"):
                await self._cb_live_update(query, data)
            elif data.startswith("targets_") or data.startswith("exit_"):
                action = data.split("_")[0]
                await self._cb_lifecycle_button(query, data, action)

            # ── Menu navigation ──────────────────────────────────────────
            elif data == "menu:main" or data == "nav:main":
                from tg.keyboards import main_menu
                text = self._render_dashboard_text()
                await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=main_menu())

            elif data == "menu:dashboard":
                from tg.keyboards import dashboard_menu
                stats = await self._collect_stats()
                text  = formatter.format_status(stats)
                await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                               reply_markup=dashboard_menu(paused=self.paused))

            elif data == "menu:market":
                from tg.keyboards import market_menu
                text = formatter.format_market_digest()
                await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=market_menu())

            elif data == "menu:signals":
                from tg.keyboards import signals_menu
                await query.edit_message_text(
                    "🎯 <b>Signals</b>\n\nSelect a filter:",
                    parse_mode=ParseMode.HTML, reply_markup=signals_menu())

            elif data == "menu:performance" or data == "nav:performance":
                from tg.keyboards import performance_menu
                await query.edit_message_text(
                    "📈 <b>Performance</b>\n\nSelect period:",
                    parse_mode=ParseMode.HTML, reply_markup=performance_menu())

            elif data == "menu:bot_health":
                from tg.keyboards import bot_health_menu
                chat_id = str(query.message.chat_id)
                self.session_state.remember_view(chat_id, "health", message_id=query.message.message_id)
                await query.edit_message_text(
                    await self.health_service.render_dashboard(),
                    parse_mode=ParseMode.HTML,
                    reply_markup=bot_health_menu(),
                )

            elif data == "menu:search":
                from tg.keyboards import search_menu
                chat_id = str(query.message.chat_id)
                self.session_state.remember_view(chat_id, "search", message_id=query.message.message_id)
                await query.edit_message_text(
                    "🔎 <b>Search</b>\n\nFind any signal, symbol, strategy, note, or health incident.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=search_menu(),
                )

            elif data == "menu:watchlist":
                from tg.keyboards import watchlist_menu
                watchlist = await db.get_watchlist()
                count = len(watchlist)
                text = (f"👁 <b>Watchlist</b>  —  {count} setups forming\n\nTap View Symbols to browse."
                        if count else "👁 <b>Watchlist</b>\n\nNothing coiling yet. Scanning…")
                await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                               reply_markup=watchlist_menu())

            elif data == "menu:settings" or data == "nav:config":
                from tg.keyboards import settings_menu
                await query.edit_message_text(
                    "⚙️ <b>Settings</b>\n\nSelect a section:",
                    parse_mode=ParseMode.HTML, reply_markup=settings_menu())

            elif data == "menu:help":
                from tg.keyboards import help_menu
                text = (
                    "❓ <b>Help</b>\n\n"
                    "All features are available through the menus.\n"
                    "/start always returns to the main menu.\n\n"
                    "Select a topic:"
                )
                await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=help_menu())

            # ── Dashboard sub-actions ────────────────────────────────────
            elif data == "dash:refresh":
                from tg.keyboards import dashboard_menu
                stats = await self._collect_stats()
                text  = formatter.format_status(stats)
                await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                               reply_markup=dashboard_menu(paused=self.paused))

            elif data == "dash:regime":
                from tg.keyboards import market_menu
                r = regime_analyzer
                regime  = r.regime.value if r.regime else "—"
                chop    = getattr(r, 'chop_strength', 0)
                adx     = getattr(r, 'adx', 0)
                session = r.session.value if hasattr(r,"session") and r.session else "—"
                fg      = getattr(r, 'fear_greed', '—')
                alt     = getattr(r, 'alt_season_score', '—')
                text = (
                    f"📈 <b>Current Regime</b>\n\n"
                    f"Regime    <b>{regime}</b>\n"
                    f"Chop      <b>{chop:.2f}</b>\n"
                    f"ADX       <b>{adx:.1f}</b>\n"
                    f"Session   <b>{session}</b>\n"
                    f"F&G       <b>{fg}</b>\n"
                    f"Alt       <b>{alt}/100</b>"
                )
                await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=market_menu())

            elif data == "dash:market_full":
                from tg.keyboards import market_menu
                text = formatter.format_market_digest()
                await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=market_menu())

            elif data in ("dash:altseason", "dash:fg", "dash:whales", "dash:rotation"):
                from tg.keyboards import market_menu
                r = regime_analyzer
                if data == "dash:altseason":
                    score = getattr(r, 'alt_season_score', 50)
                    phase = "🔥 Alt Season" if score > 70 else ("🌊 Transition" if score > 40 else "🥶 BTC Season")
                    text  = f"🌊 <b>Alt Season Index</b>\n\n<b>{score}/100</b>  —  {phase}"
                elif data == "dash:fg":
                    fg = getattr(r, 'fear_greed', 50)
                    mood = "🤑 Greed" if fg > 60 else ("😨 Fear" if fg < 40 else "😐 Neutral")
                    text = f"😨 <b>Fear & Greed</b>\n\n<b>{fg}/100</b>  —  {mood}"
                elif data == "dash:whales":
                    try:
                        from signals.whale_aggregator import whale_aggregator
                        stats = whale_aggregator.get_session_stats()
                        text = (f"🐋 <b>Whale Flow — Session</b>\n\n"
                                f"Buy orders:   <b>${stats.get('total_buy',0)/1e6:.1f}M</b>\n"
                                f"Sell orders:  <b>${stats.get('total_sell',0)/1e6:.1f}M</b>\n"
                                f"Net flow:     <b>${stats.get('net_flow',0)/1e6:.1f}M</b>\n"
                                f"Signals:      <b>{stats.get('count',0)}</b>")
                    except Exception:
                        text = "🐋 <b>Whale Flow</b>\n\nNo data available yet."
                else:  # rotation
                    try:
                        phase = rotation_tracker.current_phase
                        hot   = rotation_tracker.hot_sectors[:3]
                        cold  = rotation_tracker.cold_sectors[:3]
                        text = (f"🔄 <b>Sector Rotation</b>\n\n"
                                f"Phase:  <b>{phase}</b>\n"
                                f"Hot:    {', '.join(hot) or '—'}\n"
                                f"Cold:   {', '.join(cold) or '—'}")
                    except Exception:
                        text = "🔄 <b>Rotation</b>\n\nNo data yet."
                await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=market_menu())

            elif data == "dash:whales_ht":
                # HyperTracker smart-money cohort table
                from analyzers.hypertracker_client import hypertracker
                from tg.keyboards import market_menu
                await query.answer("Fetching smart money data…")
                cohorts = await hypertracker.get_all_cohorts()
                text = hypertracker.format_whales_summary()
                wallet_text = hypertracker.format_wallet_positions()
                if wallet_text and "No tracked wallets" not in wallet_text:
                    text += f"\n\n{wallet_text}"
                await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=market_menu())

            elif data == "dash:news_feed":
                # RSS news feed (real-time, no API key needed)
                from analyzers.news_scraper import news_scraper
                from tg.keyboards import market_menu
                summary = news_scraper.get_market_sentiment_summary()
                total = summary.get("story_count", 0)
                if total == 0:
                    text = "📰 <b>Market News</b>\n\nNo news loaded yet — RSS scraper warming up. Try again in 2 minutes."
                else:
                    score     = summary.get("score", 50)
                    label     = summary.get("label", "Neutral")
                    bull      = summary.get("bull_count", 0)
                    bear      = summary.get("bear_count", 0)
                    s_emoji   = "🟢" if score >= 65 else ("🔴" if score <= 35 else "⚪")
                    text = (
                        f"📰 <b>Market News — RSS</b>\n\n"
                        f"Stories (1h):  {total}\n"
                        f"Sentiment:     {s_emoji} {label} ({score:.0f}/100)\n"
                        f"Bull/Bear:     {bull} / {bear}\n\n"
                        f"<b>Top stories:</b>"
                    )
                    for h in news_scraper.get_all_stories(max_age_mins=60, limit=5):
                        sc = h.get("sentiment_score", 50)
                        em = "🟢" if sc >= 65 else ("🔴" if sc <= 35 else "⚪")
                        text += f"\n{em} <i>{h['title'][:75]}</i>  <code>{h['source']}</code>"
                    text += "\n\n<i>Use /news SYMBOL for coin-specific news</i>"
                await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                               reply_markup=market_menu(),
                                               disable_web_page_preview=True)

            elif data.startswith("health:"):
                from tg.keyboards import bot_health_menu
                section = data.split(":", 1)[1]
                await query.edit_message_text(
                    await self.health_service.render_section(section),
                    parse_mode=ParseMode.HTML,
                    reply_markup=bot_health_menu(),
                )

            elif data.startswith("search:mode:"):
                from tg.keyboards import search_menu
                mode = data.split(":")[-1]
                pending = await self._begin_search_prompt(str(query.message.chat_id), mode)
                await query.edit_message_text(
                    f"🔎 <b>Search</b>\n\n{pending.prompt}\n\nType it now.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=search_menu(),
                )

            elif data.startswith("search:quick:symbol:"):
                from tg.keyboards import search_results_menu
                symbol = data.split(":", 3)[-1]
                results = await self.search_service.search("symbol", symbol)
                await query.edit_message_text(
                    self.search_service.format_results(results),
                    parse_mode=ParseMode.HTML,
                    reply_markup=search_results_menu(results),
                    disable_web_page_preview=True,
                )

            elif data.startswith("search:open_signal:"):
                from tg.keyboards import search_signal_actions
                signal_id = int(data.split(":")[-1])
                signal = await db.get_signal(signal_id)
                if not signal:
                    await query.answer("Signal not found.", show_alert=True)
                    return
                await query.edit_message_text(
                    self._format_signal_lookup_detail(signal),
                    parse_mode=ParseMode.HTML,
                    reply_markup=search_signal_actions(signal_id, signal.get("symbol", "")),
                    disable_web_page_preview=True,
                )

            # ── Signals sub-actions ──────────────────────────────────────
            elif data.startswith("sig:list:"):
                from tg.keyboards import signals_menu
                hours = int(data.split(":")[-1])
                signals = await db.get_recent_signals(hours=hours)
                if not signals:
                    await query.edit_message_text(
                        f"📋 No signals in the last {hours}h.",
                        reply_markup=signals_menu())
                    return
                lines = [f"📋 <b>Signals — last {hours}h</b>\n"]
                for s in signals[:10]:
                    d_e = "🟢" if s.get("direction") == "LONG" else "🔴"
                    lines.append(f"{d_e} <b>{s.get('symbol')}</b> {s.get('direction')} — {s.get('confidence',0):.0f}% — {str(s.get('created_at',''))[:16]}")
                await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.HTML,
                                               reply_markup=signals_menu())

            elif data.startswith("sig:filter:"):
                from tg.keyboards import signals_menu
                filt = data.split(":")[-1]
                label = {"win":"✅ Wins","loss":"🔴 Losses","pending":"⏳ Pending"}.get(filt, filt)
                signals = await db.get_recent_signals(hours=168)
                filtered = []
                for s in signals:
                    out = s.get("outcome","")
                    if filt == "win" and out == "WIN": filtered.append(s)
                    elif filt == "loss" and out == "LOSS": filtered.append(s)
                    elif filt == "pending" and not out: filtered.append(s)
                if not filtered:
                    await query.edit_message_text(f"No {label} in last 7 days.", reply_markup=signals_menu())
                    return
                lines = [f"<b>{label} — 7d</b>\n"]
                for s in filtered[:10]:
                    lines.append(f"<b>{s.get('symbol')}</b> {s.get('direction')} — {str(s.get('created_at',''))[:16]}")
                await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.HTML,
                                               reply_markup=signals_menu())

            elif data.startswith("sig:status:"):
                from tg.keyboards import signals_menu
                status = data.split(":")[-1]
                signals = await db.get_recent_signals_all(hours=168)
                if status == "live":
                    active_ids = {
                        sid for sid, rec in self._by_signal_id.items()
                        if rec.state in (
                            TradeState.SETUP,
                            TradeState.ENTRY_ACTIVE,
                            TradeState.TRADE_ACTIVE,
                            TradeState.TP1_HIT,
                        )
                    }
                    filtered = [s for s in signals if s.get("id") in active_ids]
                    label = "🟢 Live Trades"
                elif status == "closed":
                    filtered = [
                        s for s in signals
                        if (s.get("outcome") or "").upper() in ("WIN", "LOSS", "BREAKEVEN", "EXPIRED", "INVALIDATED")
                    ]
                    label = "✅ Closed Trades"
                elif status == "skipped":
                    filtered = [s for s in signals if (s.get("outcome") or "").upper() == "SKIPPED"]
                    label = "⏭ Skipped Trades"
                else:
                    filtered = []
                    label = status
                if not filtered:
                    await query.edit_message_text(f"No {label.lower()} found in the last 7 days.", reply_markup=signals_menu())
                    return
                lines = [f"<b>{label}</b>\n"]
                for s in filtered[:10]:
                    outcome = s.get("outcome") or "PENDING"
                    lines.append(
                        f"#{s.get('id')} <b>{s.get('symbol')}</b> {s.get('direction')} · {s.get('alpha_grade','?')} · {outcome}"
                    )
                await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.HTML, reply_markup=signals_menu())

            elif data == "sig:upgrades":
                from tg.keyboards import back_to_main
                from signals.upgrade_tracker import upgrade_tracker
                stats = upgrade_tracker.get_stats()
                text = (
                    f"⬆️ <b>Upgrade Tracker</b>\n\n"
                    f"C signals seen   {stats['total_c_signals']}\n"
                    f"Sent to you      {stats['sent_to_user']}\n"
                    f"Upgraded         {stats['upgraded']}  ({stats['upgrade_rate']})\n"
                    f"Avg time         {stats['avg_upgrade_min']}\n"
                )
                await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                               reply_markup=back_to_main())

            elif data.startswith("sig:summary:"):
                signal_id = int(data.split(":")[-1])
                sig = await db.get_signal(signal_id)
                if not sig:
                    await query.answer("Signal not found.", show_alert=True)
                    return
                outcome = sig.get("outcome") or "PENDING"
                pnl_r   = sig.get("pnl_r") or 0.0
                r_str   = f"{pnl_r:+.2f}R" if pnl_r else "—"
                created = str(sig.get("created_at", ""))[:16]
                dur     = sig.get("duration_h")
                dur_str = f"{dur:.1f}h" if dur else "—"
                o_emoji = {"WIN": "✅", "LOSS": "🔴", "BREAKEVEN": "➡️", "EXPIRED": "⏰"}.get(outcome, "📋")
                text = (
                    f"📊 <b>Trade #{signal_id} Summary</b>\n\n"
                    f"<b>{sig.get('symbol')} {sig.get('direction')}</b>\n"
                    f"Strategy: {sig.get('strategy', '?')}\n"
                    f"Grade:    {sig.get('alpha_grade', '?')}\n\n"
                    f"Opened:   {created}\n"
                    f"Duration: {dur_str}\n\n"
                    f"Outcome:  {o_emoji} <b>{outcome}</b>\n"
                    f"P&L:      <b>{r_str}</b>"
                )
                sent = await query.message.reply_text(text, parse_mode=ParseMode.HTML)
                if sent:
                    try:
                        await self.gateway.edit_reply_markup(
                            chat_id=sent.chat_id, message_id=sent.message_id,
                            reply_markup=keyboards.panel_close(sent.message_id),
                        )
                    except TelegramError:
                        pass

            elif data.startswith("sig:explain:"):
                signal_id = int(data.split(":")[-1])
                sig = await db.get_signal(signal_id)
                if not sig:
                    await query.answer(f"Signal #{signal_id} not found.", show_alert=True); return
                import json as _json
                confluence_raw = sig.get('confluence','[]')
                try: clist = _json.loads(confluence_raw) if isinstance(confluence_raw,str) else confluence_raw
                except (ValueError, TypeError): clist = []
                lines = [
                    f"🔍 <b>Signal #{signal_id} — Explain</b>",
                    f"<b>{sig.get('symbol')} {sig.get('direction')} · {sig.get('strategy')}</b>",
                    f"Confidence: {sig.get('confidence',0):.0f}  Grade: {sig.get('alpha_grade','?')}",
                    f"Regime: {sig.get('regime','?')}  Created: {str(sig.get('created_at',''))[:16]}",
                ]
                if clist:
                    lines.append(f"\n<b>Confluence ({len(clist)}):</b>")
                    for c in clist[:8]: lines.append(f"· {str(c)[:80]}")
                await query.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

            # ── Performance ──────────────────────────────────────────────
            elif data.startswith("perf:"):
                await self._cb_performance_period(query, data)

            # ── Watchlist ────────────────────────────────────────────────
            elif data in ("wl:view", "wl:refresh"):
                from tg.keyboards import watchlist_symbols, watchlist_menu
                watchlist = await db.get_watchlist()
                if not watchlist:
                    await query.edit_message_text(
                        "👁 <b>Watchlist</b>\n\nNothing coiling yet.",
                        parse_mode=ParseMode.HTML, reply_markup=watchlist_menu())
                    return
                symbols = [w.get("symbol","?") for w in watchlist[:27]]
                await query.edit_message_text(
                    f"👁 <b>Watchlist</b>  —  {len(symbols)} setups forming\n\nTap a symbol:",
                    parse_mode=ParseMode.HTML, reply_markup=watchlist_symbols(symbols, page=0))

            elif data.startswith("wl:page:"):
                from tg.keyboards import watchlist_symbols
                page     = int(data.split(":")[-1])
                watchlist = await db.get_watchlist()
                symbols  = [w.get("symbol","?") for w in watchlist[:27]]
                await query.edit_message_text(
                    f"👁 <b>Watchlist</b>  —  {len(symbols)} setups forming\n\nTap a symbol:",
                    parse_mode=ParseMode.HTML, reply_markup=watchlist_symbols(symbols, page=page))

            elif data.startswith("wl:snap:"):
                from tg.keyboards import watchlist_snapshot_actions
                symbol   = data[8:]
                watchlist = await db.get_watchlist()
                entry    = next((w for w in watchlist if w.get("symbol") == symbol), None)
                if entry:
                    import json as _j
                    reasons = _j.loads(entry.get("reasons","[]"))
                    text    = formatter.format_watchlist_snapshot(symbol, {"score": entry.get("score",0), "reasons": reasons})
                else:
                    text = f"👁 <b>{symbol}</b>\n\nSnapshot not available."
                await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                               reply_markup=watchlist_snapshot_actions(symbol))

            elif data.startswith("wl:watch:"):
                symbol = data[9:]
                await db.mark_user_watching(symbol, query.message.message_id)
                await query.answer(f"👀 {symbol} prioritised.")

            elif data.startswith("wl:ignore:"):
                symbol = data[10:]
                await db.remove_from_watchlist(symbol)
                await query.answer(f"🚫 {symbol} removed.")

            # Legacy watchlist callbacks — redirect to canonical wl: behaviour
            elif data.startswith("wl_snap:"):
                symbol = data[8:]
                from tg.keyboards import watchlist_snapshot_actions
                watchlist = await db.get_watchlist()
                entry = next((w for w in watchlist if w.get("symbol") == symbol), None)
                if entry:
                    import json as _j
                    reasons = _j.loads(entry.get("reasons","[]"))
                    text = formatter.format_watchlist_snapshot(symbol, {"score": entry.get("score",0), "reasons": reasons})
                    try:
                        await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                                       reply_markup=watchlist_snapshot_actions(symbol))
                    except TelegramError:
                        await query.message.reply_text(text, parse_mode=ParseMode.HTML,
                                                        reply_markup=watchlist_snapshot_actions(symbol))
                else:
                    await query.answer(f"👁 {symbol} — not available.", show_alert=True)
            elif data.startswith("watch:"):
                symbol = data.split(":",1)[1]
                await db.mark_user_watching(symbol, query.message.message_id)
                await query.answer(f"👀 {symbol} prioritised.")
            elif data.startswith("ignore:"):
                symbol = data.split(":",1)[1]
                await db.remove_from_watchlist(symbol)
                await query.answer(f"🚫 {symbol} removed.")

            # ── Settings ─────────────────────────────────────────────────
            elif data == "cfg:risk":
                await self._cb_show_risk_settings(query)
            elif data == "cfg:strategies":
                await self._cb_show_strategy_menu(query)
            elif data == "cfg:scanning":
                await self._cb_show_scanning_settings(query)
            elif data == "cfg:notifications":
                await self._cb_show_notification_settings(query)
            elif data == "cfg:reload":
                # Hot-reload config from menu button
                try:
                    from config.loader import cfg as _cfg
                    _cfg.reload()
                    await query.edit_message_text(
                        "✅ <b>Config reloaded</b>\n\n"
                        "settings.yaml changes are now active.\n"
                        "<i>Telegram token and chat IDs require a full restart.</i>",
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboards.settings_menu(),
                    )
                except Exception as e:
                    await query.edit_message_text(
                        f"❌ Reload failed: {e}",
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboards.settings_menu(),
                    )
            elif data == "replay:prompt":
                await query.edit_message_text(
                    "📼 <b>Replay Mode</b>\n\n"
                    "Type a command to replay scan logic on a past date:\n"
                    "<code>/replay YYYY-MM-DD</code>\n\n"
                    "Example: <code>/replay 2026-03-07</code>\n\n"
                    "<i>Results are paper-only — no real signals published.</i>",
                    parse_mode=ParseMode.HTML,
                    reply_markup=keyboards.performance_menu(),
                )
            elif data.startswith("strat:toggle:") or data.startswith("strat_toggle:"):
                strat = data.split(":", 2)[-1] if "toggle:" in data else data.split(":", 1)[1]
                strat_lower = strat.lower()
                current = self._strategy_overrides.get(strat_lower, cfg.is_strategy_enabled(strat_lower))
                new_state = not current
                self._strategy_overrides[strat_lower] = new_state
                state_word = "enabled ✅" if new_state else "disabled ❌"
                await query.answer(f"{strat}: {state_word} (session only)", show_alert=False)
                strategy_names = [
                    "smc", "breakout", "reversal", "mean_reversion",
                    "price_action", "momentum", "ichimoku", "elliott_wave",
                    "funding_arb", "wyckoff", "harmonic", "geometric",
                    "range_scalper",
                ]
                strategies = {
                    n: self._strategy_overrides.get(n, cfg.is_strategy_enabled(n))
                    for n in strategy_names
                }
                try:
                    await query.edit_message_reply_markup(
                        reply_markup=keyboards.strategy_menu(strategies)
                    )
                except TelegramError:
                    pass
            elif data.startswith("edit:"):
                await self._cb_edit_setting(query, chat_id, data)

            # ── Help sub-pages ────────────────────────────────────────────
            elif data.startswith("help:"):
                from tg.keyboards import help_menu
                topic = data.split(":",1)[1]
                pages = {
                    "grades": (
                        "📊 <b>Signal Grades</b>\n\n"
                        "⚡ <b>A+</b>  Maximum conviction. All filters passed. Rare.\n"
                        "🔥 <b>A</b>   High conviction. Strong multi-factor confluence.\n"
                        "📊 <b>B</b>   Solid setup. Trade with standard sizing.\n"
                        "👀 <b>C</b>   Context only. High upgrade potential (UP score).\n\n"
                        "Grade is based on confidence score, alpha model, and evidence factors."
                    ),
                    "signal_types": (
                        "📋 <b>Signal Types</b>\n\n"
                        "📊 <b>Execution card</b>\n"
                        "Full trade card with entry zone, SL, TP1/2/3.\n"
                        "Action required.\n\n"
                        "👀 <b>Context signal</b>\n"
                        "Awareness only — no trade numbers.\n"
                        "Upgrades to execution if conditions confirm.\n\n"
                        "⬆️ <b>Upgrade</b>\n"
                        "C → B/A/A+ promotion. Full card follows."
                    ),
                    "reading_cards": (
                        "📈 <b>Reading Signal Cards</b>\n\n"
                        "🌍 <b>Market</b>  — Regime context for this trade\n"
                        "💰 <b>Trade Plan</b>  — Entry, SL, TP levels + sizing\n"
                        "🧠 <b>Logic</b>  — Confluence factors that fired\n"
                        "📊 <b>Metrics</b>  — Confidence score breakdown\n"
                        "🧬 <b>Profile</b>  — Strategy personality + coaching\n\n"
                        "During active trades:\n"
                        "📊 <b>Live P&L</b>  — Current R pop-up\n"
                        "🎯 <b>Targets</b>  — TP levels\n"
                        "🚪 <b>Exit Rules</b>  — When and how to exit"
                    ),
                    "settings": (
                        "⚙️ <b>Settings Guide</b>\n\n"
                        "<b>Risk settings</b>\n"
                        "Account size, risk %, min confidence, max daily loss.\n"
                        "Session changes only — edit settings.yaml to persist.\n\n"
                        "<b>Strategies</b>\n"
                        "Toggle each strategy on/off for the session.\n\n"
                        "<b>Scanning</b>\n"
                        "Tier sizes and scan intervals — view only.\n"
                        "Edit settings.yaml to change.\n\n"
                        "<b>Alerts</b>\n"
                        "Notification settings — view only."
                    ),
                    "commands": (
                        "📱 <b>All Commands</b>\n\n"
                        "/start     — Dashboard\n"
                        "/status    — Bot status\n"
                        "/signals   — Recent signals\n"
                        "/watchlist — Setups forming\n"
                        "/market    — Market digest\n"
                        "/performance — Stats\n"
                        "/report    — 30-day report\n"
                        "/ranking   — Strategy ranking\n"
                        "/panel     — Admin panel\n"
                        "/scan      — Force scan\n"
                        "/pause     — Pause / resume\n"
                        "/upgrades  — Upgrade tracker\n"
                        "/explain N — Explain signal #N\n"
                        "/whales    — HyperTracker smart money\n"
                        "/news      — Top RSS crypto headlines\n"
            "/trending  — CoinGecko trending coins\n"
                        "/news BTC  — News for specific symbol\n"
                        "/help      — Help menu"
                    ),
                    "grade_actions": (
                        "🔢 <b>What each grade means for you</b>\n\n"
                        "⚡ <b>A+</b>  Execute now. All levels shown. Rare multi-cluster setup.\n"
                        "    → Buttons: Market · Logic · Metrics · Trade Plan · Profile · Chart\n\n"
                        "🔥 <b>A</b>   High conviction. All levels shown. Prepare your order.\n"
                        "    → Buttons: Market · Logic · Metrics · Trade Plan · Profile · Chart\n\n"
                        "📊 <b>B+</b>  Solid setup. All levels shown. Wait for first trigger.\n"
                        "    → Buttons: Market · Logic · Metrics · Trade Plan · Chart\n\n"
                        "📊 <b>B</b>   Developing. Zone + SL shown. Tap 🎯 Full Levels to reveal TPs.\n"
                        "    → Buttons: Market · Logic · 🎯 Full Levels · Metrics · Chart\n\n"
                        "📋 <b>C</b>   Context only. No levels yet. Upgrades to B/A if confirmed.\n"
                        "    → Buttons: Market Context · Chart"
                    ),
                }
                text = pages.get(topic, "❓ Topic not found.")
                await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=help_menu())

            # ── Bot control ───────────────────────────────────────────────
            elif data == "bot:pause_ask":
                from tg.keyboards import pause_confirm
                await query.edit_message_text(
                    "⏸ <b>Pause signals?</b>\n\nScanning continues — only signal delivery stops.",
                    parse_mode=ParseMode.HTML, reply_markup=pause_confirm())

            elif data in ("bot:pause_confirm", "bot:pause", "panel_pause"):
                from tg.keyboards import main_menu
                self.paused = True
                if self.on_pause:
                    await self.on_pause()
                await query.edit_message_text(
                    "⏸ <b>Signals paused.</b>\nUse Resume to restart.",
                    parse_mode=ParseMode.HTML, reply_markup=main_menu())

            elif data in ("bot:resume", "bot:resume_confirm"):
                from tg.keyboards import main_menu
                self.paused = False
                if self.on_resume:
                    await self.on_resume()
                await query.edit_message_text(
                    "▶️ <b>Signals resumed.</b>",
                    parse_mode=ParseMode.HTML, reply_markup=main_menu())

            elif data in ("bot:scan", "panel_scan", "bot:force_scan"):
                if self.on_force_scan:
                    if not self._force_scan_task or self._force_scan_task.done():
                        self._force_scan_task = asyncio.create_task(self.on_force_scan())
                        self._force_scan_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                        await query.answer("🔄 Scan triggered")
                    else:
                        await query.answer("⏳ Scan already running")

            elif data == "bot:cancel":
                from tg.keyboards import main_menu
                await query.edit_message_text("✅ Cancelled.", reply_markup=main_menu())

            # Legacy panel_status
            elif data == "panel_status":
                from tg.keyboards import dashboard_menu
                stats = await self._collect_stats()
                text  = formatter.format_status(stats)
                await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                               reply_markup=dashboard_menu(paused=self.paused))

            elif data == "panel_resume":
                self.paused = False
                if self.on_resume:
                    await self.on_resume()
                await query.answer("▶ Signals resumed")

            # ── Circuit breaker ───────────────────────────────────────────
            elif data == "cb:resume_confirm":
                state = await db.get_circuit_breaker_state()
                if state.get("is_active"):
                    await db.set_circuit_breaker(False)
                    self.paused = False
                    await query.edit_message_text(
                        "⚠️ <b>Circuit breaker overridden.</b> Trade carefully.",
                        parse_mode=ParseMode.HTML)
                else:
                    await query.edit_message_text("Circuit breaker was already cleared.")
            elif data == "cb:keep_paused":
                await query.edit_message_text("✅ Staying paused. Let the market settle.")

            # ── Chart ─────────────────────────────────────────────────────
            elif data.startswith("chart:") or data.startswith("chart_sym:"):
                await self._cb_chart(query, data)

            # ── Force scan ────────────────────────────────────────────────
            elif data.startswith("force:sym:") or data.startswith("force_scan:"):
                symbol = data.split(":")[-1]
                if self.on_force_scan:
                    if not self._force_scan_task or self._force_scan_task.done():
                        self._force_scan_task = asyncio.create_task(self.on_force_scan(symbol))
                        self._force_scan_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                        await query.answer(f"⚡ Scanning {symbol}…")
                    else:
                        await query.answer("⏳ Scan already running")

            # ── Outcome recording ─────────────────────────────────────────
            elif data.startswith("outcome:"):
                await self._cb_outcome(query, data)
            elif data.startswith("skip_reason:"):
                await self._cb_skip_reason(query, data)

            # ── Panel popup management ────────────────────────────────────
            elif data.startswith("panel:close:"):
                try:
                    msg_id = int(data.split(":")[-1])
                    await self.gateway.delete_message(
                        chat_id=query.message.chat_id,
                        message_id=msg_id,
                    )
                except TelegramError:
                    await query.answer("Already closed.")

            elif data.startswith("panel:in:"):
                # Legacy path — redirect to taken:{sid}:enter
                signal_id = int(data.split(":")[-1])
                await query.answer("✅ Logged as entered!", show_alert=False)
                rec = self._by_signal_id.get(signal_id)
                if rec:
                    try:
                        kb = keyboards.signal_in_trade(signal_id, rec.symbol, confirmed=True)
                        await query.edit_message_reply_markup(reply_markup=kb)
                    except TelegramError:
                        pass

            elif data.startswith("panel:skip:"):
                signal_id = int(data.split(":")[-1])
                signal = await db.get_signal(signal_id)
                if signal:
                    await db.save_outcome({
                        "signal_id":   signal_id,
                        "symbol":      signal.get("symbol"),
                        "direction":   signal.get("direction"),
                        "outcome":     "SKIPPED",
                        "skip_reason": "user_skipped_at_execute",
                    })
                await query.answer("⏭ Skipped — logged for learning.", show_alert=False)
                try:
                    await self.gateway.edit_reply_markup(
                        chat_id=query.message.chat_id,
                        message_id=query.message.message_id,
                        reply_markup=None,
                    )
                except TelegramError:
                    pass

            elif data.startswith("taken:"):
                # taken:{signal_id}          — toggle taken on/off (C/B grade)
                # taken:{signal_id}:enter    — "I'm In" confirm + switch to active keyboard
                parts_taken = data.split(":")
                signal_id   = int(parts_taken[1])
                is_enter    = len(parts_taken) > 2 and parts_taken[2] == "enter"
                signal      = await db.get_signal(signal_id)
                rec         = self._by_signal_id.get(signal_id)

                if signal:
                    sym       = signal.get("symbol", "?")
                    entry_mid = ((signal.get("entry_low", 0) + signal.get("entry_high", 0)) / 2)

                    if is_enter:
                        # ── "I'm In" pressed ─────────────────────────────────
                        # Record entry, switch keyboard to active trade with confirmed=True
                        note = f"User entered @ ~{entry_mid:.6g}"
                        async with db._lock:
                            await db._conn.execute(
                                "UPDATE signals SET user_taken=1, user_notes=? WHERE id=?",
                                (note, signal_id)
                            )
                            await db._conn.commit()
                        await query.answer(
                            f"✅ {sym} — you're in! Entry: ~{entry_mid:.6g}. Card now shows P&L + Targets.",
                            show_alert=True
                        )
                        # Switch card to confirmed active trade keyboard
                        try:
                            kb = keyboards.signal_in_trade(signal_id, sym, confirmed=True)
                            await query.edit_message_reply_markup(reply_markup=kb)
                        except TelegramError:
                            pass
                        # Persist confirmed state
                        try:
                            from data.database import db as _db
                            asyncio.create_task(_db.update_card_state(
                                signal_id, "ENTRY_ACTIVE", confirmed=True))
                        except Exception as _e:
                            logger.debug("Failed to persist confirmed state for signal %s: %s", signal_id, _e)

                    elif signal.get("user_taken"):
                        # ── Toggle off (already taken) ────────────────────────
                        async with db._lock:
                            await db._conn.execute(
                                "UPDATE signals SET user_taken=0, user_notes='' WHERE id=?",
                                (signal_id,)
                            )
                            await db._conn.commit()
                        await query.answer(f"↩️ {sym} un-marked", show_alert=False)

                    else:
                        # ── Toggle on (C/B grade simple mark) ────────────────
                        note = f"Manual entry @ ~{entry_mid:.6f}"
                        async with db._lock:
                            await db._conn.execute(
                                "UPDATE signals SET user_taken=1, user_notes=? WHERE id=?",
                                (note, signal_id)
                            )
                            await db._conn.commit()
                        await query.answer(f"✅ {sym} marked! Entry: ~{entry_mid:.6f}. Tracked in dashboard.", show_alert=True)
                        # Switch keyboard to active trade management
                        try:
                            kb = keyboards.signal_in_trade(signal_id, sym, confirmed=True)
                            await self.gateway.edit_reply_markup(
                                chat_id=query.message.chat_id,
                                message_id=query.message.message_id,
                                reply_markup=kb,
                            )
                        except Exception as _e:
                            logger.debug("Failed to update keyboard for signal %s: %s", signal_id, _e)
                else:
                    await query.answer("Signal not found.", show_alert=False)

            elif data.startswith("sig:cancel:"):
                await query.answer("Signal cancelled.", show_alert=False)
                try:
                    await self.gateway.edit_reply_markup(
                        chat_id=query.message.chat_id,
                        message_id=query.message.message_id,
                        reply_markup=None,
                    )
                except TelegramError:
                    pass

            else:
                logger.debug(f"Unhandled callback: {data}")

        except Exception as e:
            logger.error(f"Callback error [{data}]: {e}", exc_info=True)
            try:
                await query.message.reply_text("⚠️ Something went wrong.")
            except Exception as _e:
                logger.debug("Failed to send error reply: %s", _e)
    # ─────────────────────────────────────────────────────────
    # Callback sub-handlers
    # ─────────────────────────────────────────────────────────

    async def _cb_panel_inline(self, query, data: str):
        """
        v6 panel handler — edits the signal card IN PLACE instead of spawning reply messages.

        Callback format: panel:{signal_id}:{action}
        Actions:
          back     — restore original card text + grade keyboard
          market   — show Market Context panel
          logic    — show Logic panel
          metrics  — show Metrics panel
          profile  — show Profile panel
          plan     — show Trade Plan panel (Full Levels for grade B)
          live     — P&L toast popup (no edit, reuses _cb_live_update logic)
          targets  — show Targets panel (active trade)
          exit     — show Exit Rules panel (active trade)
          in       — user confirmed entry
          skip     — user skipped signal
        """
        parts  = data.split(":", 2)          # ["panel", signal_id, action]
        if len(parts) < 3:
            await query.answer("Unknown action.")
            return
        try:
            signal_id = int(parts[1])
        except ValueError:
            # Legacy: panel:close:MSG_ID format — handled by old path, shouldn't reach here
            await query.answer()
            return
        action = parts[2]

        rec = self._by_signal_id.get(signal_id)

        # ── Back: restore the card ────────────────────────────────────────
        if action == "back":
            if not rec:
                await query.answer("Signal no longer active.")
                return
            # Determine correct keyboard for current state
            if rec.state in (TradeState.ENTRY_ACTIVE, TradeState.TRADE_ACTIVE, TradeState.TP1_HIT):
                kb = keyboards.signal_in_trade(signal_id, rec.symbol)
            elif rec.state == TradeState.CLOSED:
                kb = keyboards.signal_closed(signal_id)
            else:
                kb = keyboards.signal_card(signal_id, rec.symbol, rec.grade)
            try:
                await query.edit_message_text(
                    rec.card_text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=kb,
                    disable_web_page_preview=True,
                )
            except TelegramError:
                await query.answer("Already showing card.")
            return

        # ── Live P&L: toast popup, no edit ────────────────────────────────
        if action == "live":
            # Reuse existing live-update logic but via new callback format
            legacy_data = f"live_{rec.symbol if rec else ''}__{signal_id}" if rec else data
            await self._cb_live_update(query, legacy_data)
            return

        # ── Signal doesn't exist or no scored data ────────────────────────
        if not rec or not rec.scored:
            await query.answer("Signal data not available.", show_alert=True)
            return

        scored = rec.scored
        sig    = scored.base_signal
        symbol = rec.symbol

        # ── in / skip (stage_execute buttons) ────────────────────────────
        if action == "in":
            # Transition card to active trade keyboard
            await query.answer("✅ You're in! Good luck.", show_alert=False)
            try:
                kb = keyboards.signal_in_trade(signal_id, symbol, confirmed=True)
                await query.edit_message_reply_markup(reply_markup=kb)
            except TelegramError:
                pass
            return
        if action == "skip":
            # Show skip reason keyboard — edit card inline
            try:
                await query.edit_message_reply_markup(
                    reply_markup=keyboards.skip_reason(signal_id)
                )
            except TelegramError:
                pass
            return

        # ── Resolve panel text ────────────────────────────────────────────
        panel_text = None

        if action == "market":
            panel_text = formatter.format_market_panel(scored)

        elif action == "logic":
            panel_text = formatter.format_logic_panel(scored)

        elif action == "metrics":
            panel_text = formatter.format_metrics_panel(scored)

        elif action == "profile":
            panel_text = formatter.format_profile_panel(scored)

        elif action == "plan":
            panel_text = formatter.format_trade_plan_panel(
                scored, rec.alpha_score, rec.prob_estimate, rec.sizing
            )

        elif action == "check":
            # ── Thesis check — AI mid-trade signal validation ──────────────
            sid_str = str(signal_id)
            # Avoid duplicate concurrent checks
            if self._check_signal_tasks.get(signal_id) and                not self._check_signal_tasks[signal_id].done():
                await query.answer("⏳ Check already running…", show_alert=False)
                return
            await query.answer("🔍 Checking thesis… (~8s)", show_alert=False)

            async def _run_check():
                try:
                    from signals.thesis_checker import check_thesis
                    from signals.outcome_monitor import outcome_monitor
                    from core.price_cache import price_cache as _pc
                    from config.loader import cfg

                    tracked = outcome_monitor.get_active_signals().get(signal_id)
                    api_key = cfg.ai.get("openrouter_api_key", "")

                    current_price = _pc.get(symbol) or 0.0
                    current_r = 0.0
                    max_r_val = 0.0

                    if tracked:
                        current_r = outcome_monitor.calc_pnl_r(tracked, current_price)
                        max_r_val = tracked.max_r

                    raw = sig.raw_data or {}
                    text = await check_thesis(
                        signal_id=signal_id,
                        symbol=symbol,
                        direction=rec.direction,
                        strategy=sig.strategy or "",
                        entry_price=tracked.entry_price if tracked else (sig.entry_low + sig.entry_high) / 2,
                        stop_loss=sig.stop_loss,
                        tp1=sig.tp1,
                        tp2=sig.tp2,
                        created_at=tracked.created_at if tracked else time.time(),
                        activated_at=tracked.activated_at if tracked else None,
                        raw_data=raw,
                        current_price=current_price,
                        current_r=current_r,
                        max_r=max_r_val,
                        openrouter_api_key=api_key,
                    )
                    # Send as reply to the signal card
                    await self.gateway.send_text(
                        chat_id=query.message.chat_id,
                        text=text,
                        parse_mode=ParseMode.HTML,
                        reply_to_message_id=rec.message_id,
                        disable_web_page_preview=True,
                    )
                except Exception as _e:
                    logger.error(f"thesis check failed: {_e}")
                    try:
                        await self.gateway.send_text(
                            chat_id=query.message.chat_id,
                            text=f"⚠️ Check failed: {str(_e)[:100]}",
                            reply_to_message_id=rec.message_id,
                        )
                    except Exception as _e:
                        logger.debug("Failed to send thesis check error message: %s", _e)

            task = asyncio.create_task(_run_check())
            self._check_signal_tasks[signal_id] = task
            task.add_done_callback(
                lambda t: t.exception() if not t.cancelled() and t.exception() else None
            )
            return

        elif action == "targets":
            # Active trade: show TP levels inline
            from utils.formatting import fmt_price as _fp
            from signals.outcome_monitor import outcome_monitor
            tracked = outcome_monitor.get_active_signals().get(signal_id)
            be_label = "BE Stop" if (tracked and tracked.be_stop) else "SL"
            active_sl_price = (tracked.be_stop if (tracked and tracked.be_stop) else sig.stop_loss)
            sl_line = f"🛡 {be_label}: <code>{_fp(active_sl_price)}</code>  (risk-free)" \
                      if (tracked and tracked.be_stop) else \
                      f"🛑 SL: <code>{_fp(sig.stop_loss)}</code>"
            panel_text = (
                f"🎯 <b>TARGETS — {symbol}</b>\n\n"
                f"TP1:  <code>{_fp(sig.tp1)}</code>"
                + ("  ✓ taken" if rec.state == TradeState.TP1_HIT else "") + "\n"
                f"TP2:  <code>{_fp(sig.tp2)}</code>\n"
                + (f"TP3:  <code>{_fp(sig.tp3)}</code>\n" if sig.tp3 else "")
                + f"\n{sl_line}\n"
                f"R/R:  {sig.rr_ratio:.1f}R"
            )

        elif action == "exit":
            # Active trade: show exit rules inline
            from utils.formatting import fmt_price as _fp
            from tg.formatter import STRATEGY_PERSONALITY
            from signals.outcome_monitor import outcome_monitor
            strategy = sig.strategy or "Unknown"
            p = STRATEGY_PERSONALITY.get(strategy, {
                'exit_note': 'Follow TP levels. Exit if invalidation condition met.',
                'pullbacks': 'Variable',
            })
            tracked = outcome_monitor.get_active_signals().get(signal_id)
            sl_note = (
                f"🛡 SL at BE: <code>{_fp(tracked.be_stop)}</code>  (risk-free)"
                if tracked and tracked.be_stop else
                f"🛑 SL: <code>{_fp(sig.stop_loss)}</code>"
            )
            panel_text = (
                f"🚪 <b>EXIT RULES — {symbol}</b>\n\n"
                f"{sl_note}\n\n"
                f"<b>Strategy:</b> {strategy}\n"
                f"<b>Pullbacks:</b> {p.get('pullbacks', 'Variable')}\n\n"
                f"<i>{p.get('exit_note', 'Follow TP levels.')}</i>\n\n"
                f"<b>Invalidation:</b> Close past <code>{_fp(sig.stop_loss)}</code>"
            )

        else:
            await query.answer("Unknown panel action.")
            return

        if not panel_text:
            await query.answer("Panel data not available.", show_alert=True)
            return

        # ── Edit the card in-place, add ← Back button ─────────────────────
        try:
            await query.edit_message_text(
                panel_text,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboards.signal_panel_back(signal_id, symbol),
                disable_web_page_preview=True,
            )
        except TelegramError as e:
            # Edit failed (e.g. message too old) — fall back to toast
            await query.answer(panel_text[:180], show_alert=True)

    async def _cb_panel_button(self, query, data: str, action: str):
        """Handle the 5 analysis panel buttons.
        Sends a temporary reply with a Close button so it can be dismissed — no permanent clutter.
        """
        # Parse: action_SYMBOL_TF_SIGNALID
        parts = data.split("_", 3)
        signal_id = int(parts[3]) if len(parts) == 4 else 0
        rec = self._by_signal_id.get(signal_id)

        if not rec or not rec.scored:
            await query.answer("Signal data not available.", show_alert=True)
            return

        scored = rec.scored
        if action == "market":
            text = formatter.format_market_panel(scored)
        elif action == "plan":
            # V17: Always show full trade plan — signal card already shows zone+SL
            text = formatter.format_trade_plan_panel(
                scored, rec.alpha_score, rec.prob_estimate, rec.sizing
            )
        elif action == "logic":
            text = formatter.format_logic_panel(scored)
        elif action == "metrics":
            text = formatter.format_metrics_panel(scored)
        elif action == "profile":
            text = formatter.format_profile_panel(scored)
        else:
            await query.answer("Unknown action.")
            return

        sent = await query.message.reply_text(
            text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
        )
        if sent:
            try:
                await self.gateway.edit_reply_markup(
                    chat_id=sent.chat_id,
                    message_id=sent.message_id,
                    reply_markup=keyboards.panel_close(sent.message_id),
                )
            except TelegramError:
                pass

    async def _cb_lifecycle_button(self, query, data: str, action: str):
        """Handle live / targets / exit buttons during active trade.
        All responses edit the message in place — zero new messages, zero spam.
        """
        parts = data.split("_", 3)
        signal_id = int(parts[3]) if len(parts) == 4 else 0
        rec = self._by_signal_id.get(signal_id)

        if action == "live":
            # Pull current P&L from outcome monitor
            try:
                from signals.outcome_monitor import outcome_monitor
                from utils.formatting import fmt_price
                tracked = outcome_monitor.get_active_signals().get(signal_id)
                if tracked and tracked.entry_price:
                    current_r = outcome_monitor._calc_pnl_r_by_id(signal_id)
                    pnl_emoji = "📈" if current_r >= 0 else "📉"
                    be_label = "BE Stop" if tracked.be_stop else "SL"
                    active_sl = fmt_price(tracked.be_stop if tracked.be_stop else tracked.stop_loss)
                    state_label = "🛡 Risk-free" if tracked.be_stop else "⚠️ At risk"
                    next_target = (
                        f"TP2: <code>{fmt_price(tracked.tp2)}</code>"
                        if tracked.be_stop else
                        f"TP1: <code>{fmt_price(tracked.tp1)}</code>"
                    )
                    await query.answer(
                        f"{pnl_emoji} {current_r:+.2f}R  |  {state_label}",
                        show_alert=True
                    )
                else:
                    await query.answer("Trade data not available yet.", show_alert=True)
            except Exception as e:
                await query.answer("Could not fetch live P&L.", show_alert=True)
            return

        if not rec or not rec.scored:
            await query.answer("Signal data not available.", show_alert=True)
            return

        if action == "targets":
            text = formatter.format_trade_plan_panel(rec.scored)
        elif action == "exit":
            # Show actual exit rules for this strategy
            sig = rec.scored.base_signal
            strategy = sig.strategy or "Unknown"
            from tg.formatter import STRATEGY_PERSONALITY, fmt_price as fp
            p = STRATEGY_PERSONALITY.get(strategy, {
                'exit_note': 'Follow TP levels. Exit if invalidation condition met.',
                'pullbacks': 'Variable',
            })
            from signals.outcome_monitor import outcome_monitor
            tracked = outcome_monitor.get_active_signals().get(signal_id)
            sl_note = (
                f"🛡 SL at BE: <code>{fp(tracked.be_stop)}</code> (risk-free)"
                if tracked and tracked.be_stop else
                f"🛑 SL: <code>{fp(sig.stop_loss)}</code>"
            )
            text = (
                f"🚪 <b>EXIT RULES — {sig.symbol}</b>\n\n"
                f"{sl_note}\n\n"
                f"<b>Strategy:</b> {strategy}\n"
                f"<b>Pullbacks:</b> {p.get('pullbacks','Variable')}\n\n"
                f"<i>{p.get('exit_note','Follow TP levels.')}</i>\n\n"
                f"<b>Invalidation:</b> Close past <code>{fp(sig.stop_loss)}</code>"
            )
        else:
            await query.answer("Unknown action.")
            return

        await query.answer()
        sent = await query.message.reply_text(text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        if sent:
            try:
                await self.gateway.edit_reply_markup(
                    chat_id=sent.chat_id,
                    message_id=sent.message_id,
                    reply_markup=keyboards.panel_close(sent.message_id),
                )
            except TelegramError:
                pass

    async def _cb_live_update(self, query, data: str):
        """
        📊 Live Update button — fetches real-time P&L from outcome monitor.
        Responds as a pop-up alert (no new message, zero spam).
        """
        parts = data.split("_", 3)
        signal_id = int(parts[3]) if len(parts) == 4 else 0

        try:
            from signals.outcome_monitor import outcome_monitor
            from data.api_client import api
            from utils.formatting import fmt_price

            active = outcome_monitor.get_active_signals()
            tracked = active.get(signal_id)

            if not tracked:
                await query.answer("Trade already closed or not found.", show_alert=True)
                return

            # Fetch current price
            ticker = await api.fetch_ticker(tracked.symbol)
            current_price = float(ticker["last"]) if ticker and "last" in ticker else None

            if current_price is None:
                await query.answer("Price fetch failed. Try again.", show_alert=True)
                return

            current_r = outcome_monitor._calc_pnl_r_public(tracked, current_price)
            active_sl = tracked.be_stop if tracked.be_stop else tracked.stop_loss
            be_label = "BE" if tracked.be_stop else "SL"
            pnl_emoji = "📈" if current_r >= 0 else "📉"

            # Show as a Telegram pop-up toast — no new message
            msg = (
                f"{pnl_emoji} {tracked.symbol} {tracked.direction}\n"
                f"P&L: {current_r:+.2f}R  (peak: {tracked.max_r:+.2f}R)\n"
                f"Price: {fmt_price(current_price)}\n"
                f"{be_label}: {fmt_price(active_sl)}"
            )
            await query.answer(msg, show_alert=True)

        except Exception as e:
            logger.debug(f"Live update button error: {e}")
            await query.answer("Could not fetch live data.", show_alert=True)

    async def _cb_chart(self, query, data: str):
        """Chart buttons are now direct URL buttons — no callback needed.
        This handler is kept only for legacy signal cards still using chart:id: format."""
        if data.startswith("chart_sym:"):
            symbol = data.split(":", 1)[1]
            from tg.keyboards import _tv_url
            await query.answer(f"📈 {_tv_url(symbol)[:60]}")
        else:
            await query.answer("📈 Use the Chart button on the signal card.")

    async def _cb_outcome(self, query, data: str):
        parts     = data.split(":")
        signal_id = int(parts[1])
        outcome   = parts[2]
        signal    = await db.get_signal(signal_id)
        if signal:
            await db.save_outcome({
                "signal_id": signal_id,
                "symbol":    signal.get("symbol"),
                "direction": signal.get("direction"),
                "outcome":   outcome,
            })
        emojis = {"WIN": "✅", "LOSS": "❌", "BREAKEVEN": "➡️"}
        await query.edit_message_text(
            f"{emojis.get(outcome,'📋')} <b>{outcome}</b> recorded.",
            parse_mode=ParseMode.HTML,
        )

    async def _cb_skip_reason(self, query, data: str):
        parts     = data.split(":")
        signal_id = int(parts[1])
        reason    = parts[2]
        signal    = await db.get_signal(signal_id)
        if signal:
            await db.save_outcome({
                "signal_id":   signal_id,
                "symbol":      signal.get("symbol"),
                "direction":   signal.get("direction"),
                "outcome":     "SKIPPED",
                "skip_reason": reason,
            })
        await query.edit_message_text("✅ Recorded. Thanks — helps improve filtering.")

    async def _cb_show_risk_settings(self, query):
        risk    = cfg.risk
        account = risk.get("account_size", 5000)
        r_pct   = risk.get("risk_per_trade", 0.005)
        min_c   = cfg.aggregator.get("min_confidence", 72)
        max_l   = risk.get("max_daily_loss_pct", 0.03)
        min_rr  = risk.get("min_rr", 1.3)
        text = (
            "⚖️ <b>RISK SETTINGS</b>\n\n"
            f"Account Size:    <code>${account:,.0f}</code>\n"
            f"Risk Per Trade:  <code>{r_pct*100:.2f}%</code>  (${account * r_pct:,.0f})\n"
            f"Min Confidence:  <code>{min_c}/100</code>\n"
            f"Max Daily Loss:  <code>{max_l*100:.1f}%</code>  (${account * max_l:,.0f})\n"
            f"Min R/R:         <code>{min_rr}</code>"
        )
        await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                       reply_markup=keyboards.risk_settings_menu())

    async def _cb_show_strategy_menu(self, query):
        strategy_names = [
            "smc", "breakout", "reversal", "mean_reversion",
            "price_action", "momentum", "ichimoku", "elliott_wave",
            "funding_arb", "wyckoff", "harmonic", "geometric",
            "range_scalper",
        ]
        strategies = {
            n: self._strategy_overrides.get(n, cfg.is_strategy_enabled(n))
            for n in strategy_names
        }
        await query.edit_message_text(
            "🧠 <b>STRATEGIES</b>\n\nTap to toggle (session only):",
            parse_mode=ParseMode.HTML,
            reply_markup=keyboards.strategy_menu(strategies),
        )

    async def _cb_show_scanning_settings(self, query):
        sc = cfg.scanning
        t1 = sc.tier1; t2 = sc.tier2; t3 = sc.tier3
        text = (
            "📊 <b>SCANNING SETTINGS</b>\n\n"
            f"Tier 1: max {getattr(t1,'max_symbols',80)} symbols  "
            f"(vol > ${getattr(t1,'min_volume_24h',5000000)/1e6:.0f}M)\n"
            f"Tier 2: max {getattr(t2,'max_symbols',80)} symbols  "
            f"(vol > ${getattr(t2,'min_volume_24h',1000000)/1e6:.1f}M)\n"
            f"Tier 3: max {getattr(t3,'max_symbols',40)} symbols  "
            f"(vol > ${getattr(t3,'min_volume_24h',200000)/1e3:.0f}K)\n\n"
            f"Tier1 interval: {cfg.system.get('tier1_interval',120)//60} min\n"
            f"Tier2 interval: {cfg.system.get('tier2_interval',300)//60} min\n"
            f"Tier3 interval: {cfg.system.get('tier3_interval',900)//60} min\n\n"
            "To change: edit <code>config/settings.yaml</code>"
        )
        await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                       reply_markup=keyboards.back_to_main())

    async def _cb_show_notification_settings(self, query):
        tg = cfg.telegram
        def yn(key, default=True): return "✅" if tg.get(key, default) else "❌"
        text = (
            "🔔 <b>NOTIFICATION SETTINGS</b>\n\n"
            f"{yn('send_signals')}         Signals (push)\n"
            f"{yn('send_daily_summary')}   Daily summary\n"
            f"{yn('send_error_alerts')}    Error alerts\n\n"
            "Admin channel receives:\n"
            "• Watchlist forming\n"
            "• Whale alerts\n"
            "• Tier promotions\n"
            "• All system logs\n\n"
            "To toggle: edit <code>settings.yaml → telegram</code>"
        )
        await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                       reply_markup=keyboards.back_to_main())

    async def _cb_edit_setting(self, query, chat_id: str, data: str):
        field = data.split(":", 1)[1]
        self._awaiting_edit[chat_id] = field
        descriptions = {
            "account_size":       ("Account size in USDT", "5000"),
            "risk_per_trade":     ("Risk per trade (0.002–0.02)", "0.005"),
            "min_confidence":     ("Min confidence (50–95)", "72"),
            "max_daily_loss_pct": ("Max daily loss % (0.01–0.10)", "0.03"),
            "min_rr":             ("Minimum R/R (1.0–3.0)", "1.3"),
        }
        desc, example = descriptions.get(field, (field, "?"))
        current = self._get_config_value(field)
        await query.message.reply_text(
            f"✏️ <b>Edit: {field}</b>\n\n{desc}\n"
            f"Current: <code>{current}</code>\n"
            f"Example: <code>{example}</code>\n\nReply with the new value:",
            parse_mode=ParseMode.HTML,
        )

    async def _cb_show_performance(self, query, days: int = 30):
        stats       = await db.get_performance_stats(days=days)
        by_strategy = await db.get_strategy_performance(days=days)
        by_sector   = await db.get_sector_performance(days=days)
        stats["by_strategy"] = by_strategy
        stats["by_sector"]   = by_sector
        text = formatter.format_performance(stats, f"{days} days")
        await query.message.reply_text(text, parse_mode=ParseMode.HTML,
                                        reply_markup=keyboards.performance_period())

    async def _cb_performance_period(self, query, data: str):
        period = data.split(":", 1)[1]
        if period == "export":
            await query.message.reply_text(
                "📤 Data lives in <code>data/titanbot.db</code>. "
                "Open with any SQLite browser to export CSV.",
                parse_mode=ParseMode.HTML,
            )
            return
        days_map = {"7": 7, "30": 30, "all": 365}
        await self._cb_show_performance(query, days=days_map.get(period, 30))


    async def _handle_ai_callbacks(self, query, data: str):
        """Handle AI-related inline button callbacks."""
        from analyzers.ai_analyst import ai_analyst
        from core.diagnostic_engine import diagnostic_engine

        if data == "ai_mode_full":
            ai_analyst.set_mode("full")
            await query.answer("🟢 AI set to Full mode")
            await query.message.reply_text("🤖 AI mode: FULL", parse_mode=ParseMode.HTML)

        elif data == "ai_mode_diagnosis":
            ai_analyst.set_mode("diagnosis_only")
            await query.answer("🟡 AI set to Diagnosis Only")
            await query.message.reply_text("🔬 AI mode: DIAGNOSIS ONLY", parse_mode=ParseMode.HTML)

        elif data == "ai_mode_off":
            ai_analyst.set_mode("off")
            await query.answer("🔴 AI turned OFF")
            await query.message.reply_text("🤖 AI mode: OFF — pure rule-based", parse_mode=ParseMode.HTML)

        elif data == "ai_report":
            await query.answer("Generating report…")
            await diagnostic_engine.get_report_on_demand()

        elif data == "ai_approvals":
            await query.answer()
            pending = diagnostic_engine.get_pending_approvals()
            if not pending:
                await query.message.reply_text("✅ No pending approvals.")
            else:
                for a in pending[:3]:
                    await self.send_approval_request(a)

        elif data == "ai_history":
            await query.answer()
            overrides = diagnostic_engine.get_applied_overrides()
            if not overrides:
                await query.message.reply_text("No active overrides.")
            else:
                lines = ["⚙️ <b>Active Overrides</b>"]
                for ov in overrides[-5:]:
                    import time
                    age_h = (time.time() - ov.get("applied_at", 0)) / 3600
                    lines.append(f"• {ov['description']} ({age_h:.1f}h ago)")
                await query.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

        elif data.startswith("sen_"):
            # Sentinel quick-action callbacks from feature menu
            from analyzers.sentinel_features import sentinel
            feature = data[4:]  # strip "sen_"
            if feature == "postmortem":
                await query.answer("Running postmortem…")
                result = await sentinel.postmortem()
                if not result:
                    await query.message.reply_text("⚠️ Not enough closed trade data yet.")
                else:
                    wr = result.win_rate * 100
                    await query.message.reply_text(
                        f"📊 <b>Postmortem</b> ({result.total_trades} trades, {wr:.1f}% WR)\n\n"
                        f"{result.insight}\n\n"
                        f"Top action: {result.actions[0] if result.actions else 'None'}\n\n"
                        f"Use /sentinel postmortem for full report.",
                        parse_mode=ParseMode.HTML,
                    )
            elif feature == "oracle":
                await query.answer("Querying regime oracle…")
                try:
                    from analyzers.regime import regime_analyzer
                    ctx = {"regime": regime_analyzer.regime.value if hasattr(regime_analyzer, "regime") else "UNKNOWN"}
                    result = await sentinel.regime_oracle(ctx)
                    if result:
                        await query.message.reply_text(
                            f"🔮 <b>Oracle:</b> {result.current_regime} → <b>{result.predicted_next}</b> "
                            f"({result.confidence:.0f}% conf, {result.estimated_time_to_shift})\n"
                            f"{result.trade_implication}",
                            parse_mode=ParseMode.HTML,
                        )
                except Exception as _e:
                    await query.message.reply_text(f"Oracle error: {_e}")
            elif feature.startswith("whale_"):
                sym = feature[6:] + "/USDT"
                await query.answer(f"Classifying whale intent for {sym}…")
                result = await sentinel.whale_intent(sym)
                if result:
                    await query.message.reply_text(
                        f"🐋 <b>{sym}:</b> {result.classification} ({result.confidence:.0f}%) → {result.signal_implication}",
                        parse_mode=ParseMode.HTML,
                    )
                else:
                    await query.message.reply_text(f"No whale data for {sym} right now.")
            elif feature == "blacklist":
                await query.answer("Scanning blacklist…")
                active = sentinel.get_blacklist()
                if not active:
                    await query.message.reply_text("✅ No symbols currently blacklisted.")
                else:
                    lines = [f"🚫 <b>Blacklisted ({len(active)}):</b>"]
                    import time as _t
                    for b in active[:5]:
                        hrs = (b.banned_until - _t.time()) / 3600
                        lines.append(f"  • {b.symbol} ({hrs:.1f}h) — {b.reason}")
                    await query.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            elif feature == "calibrate":
                await query.answer("Running calibration…")
                suggestions = await sentinel.threshold_calibration()
                if not suggestions:
                    await query.message.reply_text("⚠️ Need 20+ closed trades for calibration.")
                else:
                    top = suggestions[0]
                    await query.message.reply_text(
                        f"🔧 Top suggestion: <b>{top.parameter}</b> {top.current_value} → {top.suggested_value}\n"
                        f"{top.evidence}\n\nUse /sentinel calibrate for all suggestions.",
                        parse_mode=ParseMode.HTML,
                    )
            elif feature == "news":
                await query.answer("Scanning news…")
                await query.message.reply_text("Use /sentinel news for full batch scan.")

        elif data in ("audit_fast", "audit_deep", "audit_status"):
            await query.answer("Running Sentinel audit…" if data != "audit_status" else "Loading audit status…")
            if data == "audit_status":
                astat = ai_analyst.get_audit_status()
                ss = astat["session_stats"]
                fa = astat["fast_last_ran"]
                da = astat["deep_last_ran"]
                def _a(s): return "never" if s>86000 else (f"{s//60}m ago" if s<3600 else f"{s//3600}h ago")
                strat_lines = []
                for strat, counts in list(ss.get("strategy_direction", {}).items())[:6]:
                    tot = counts.get("LONG", 0) + counts.get("SHORT", 0)
                    if tot >= 3:
                        sp = counts.get("SHORT", 0) / tot * 100
                        bias = " ⚠️" if sp > 85 or sp < 15 else ""
                        strat_lines.append(f"  {strat.split(':')[-1]}: S:{counts.get('SHORT',0)} L:{counts.get('LONG',0)}{bias}")
                await query.message.reply_text(
                    f"🔬 <b>Sentinel Audit Status</b>\n"
                    f"Fast last ran: {_a(fa)} | Deep last ran: {_a(da)}\n"
                    f"HTF: LONG blocked={ss.get('htf_blocked_long',0)} SHORT blocked={ss.get('htf_blocked_short',0)}\n"
                    f"Whale conflicts: {ss.get('signals_vs_whale_conflict',0)}\n\n"
                    f"Strategy direction:\n" + ("\n".join(strat_lines) or "  Accumulating…"),
                    parse_mode=ParseMode.HTML
                )
            else:
                deep_mode = data == "audit_deep"
                try:
                    audit = await ai_analyst.structural_audit(force=True, deep=deep_mode)
                    if not audit:
                        await query.message.reply_text("⚠️ Not enough session data yet. Let the bot scan a while first.")
                        return
                    emoji_map = {"LOW": "ℹ️", "MEDIUM": "⚠️", "HIGH": "🚨", "CRITICAL": "🔥"}
                    sev_emoji = emoji_map.get(audit.severity, "ℹ️")
                    lines = [f"{sev_emoji} <b>Sentinel [{audit.severity}]</b>"]
                    if audit.biased_strategies:
                        lines.append(f"⚠️ Biased: {', '.join(audit.biased_strategies)}")
                    if audit.htf_asymmetry:
                        lines.append("⚠️ HTF guardrail asymmetric")
                    if audit.whale_signal_conflict:
                        lines.append("⚠️ Signals fighting whale flow")
                    for rec in audit.recommendations[:3]:
                        lines.append(f"→ {rec}")
                    if audit.severity == "LOW" and not audit.biased_strategies:
                        lines.append("✅ No structural issues found.")
                    await query.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
                except Exception as _e:
                    await query.message.reply_text(f"❌ Audit error: {_e}")

        elif data.startswith("approve_yes_"):
            approval_id = data[len("approve_yes_"):]
            success = await diagnostic_engine.apply_approval(approval_id)
            if success:
                await query.answer("✅ Applied!")
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                undo_kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("↩️ UNDO", callback_data=f"approve_undo_{approval_id}")
                ]])
                await query.message.reply_text(
                    f"✅ Change applied. Tap UNDO to revert.",
                    reply_markup=undo_kb,
                )
            else:
                await query.answer("❌ Failed to apply")

        elif data.startswith("approve_no_"):
            approval_id = data[len("approve_no_"):]
            diagnostic_engine.reject_approval(approval_id)
            await query.answer("❌ Rejected")
            await query.message.reply_text("Change ignored.")

        elif data.startswith("approve_snooze_"):
            approval_id = data[len("approve_snooze_"):]
            diagnostic_engine.snooze_approval(approval_id, minutes=60)
            await query.answer("⏰ Will ask again in 1 hour")

        elif data.startswith("approve_undo_"):
            approval_id = data[len("approve_undo_"):]
            success = await diagnostic_engine.undo_change(approval_id)
            if success:
                await query.answer("↩️ Reverted!")
                await query.message.reply_text("↩️ Change reverted.")
            else:
                await query.answer("Nothing to revert")

