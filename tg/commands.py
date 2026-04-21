"""
CommandsMixin — all /command handlers for TelegramBot.

Extracted from tg/bot.py to keep the main bot class lean.
Each method is an instance method that accesses TelegramBot
attributes through ``self`` at runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional, TYPE_CHECKING

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

if TYPE_CHECKING:
    pass  # no direct reference needed; mixin merges at runtime

logger = logging.getLogger(__name__)


class CommandsMixin:
    """Mixin providing all ``_cmd_*`` handlers and ``send_approval_request``."""

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import main_menu
        text = self._render_dashboard_text()
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=main_menu())

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import dashboard_menu
        stats = await self._collect_stats()
        text  = formatter.format_status(stats)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML,
                                         reply_markup=dashboard_menu(paused=self.paused))

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import signals_menu
        await update.message.reply_text(
            "🎯 <b>Signals</b>\n\nSelect a filter:",
            parse_mode=ParseMode.HTML, reply_markup=signals_menu())

    async def _cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import watchlist_symbols, watchlist_menu
        watchlist = await db.get_watchlist()
        if not watchlist:
            await update.message.reply_text(
                "👁 <b>Watchlist</b>\n\nNothing coiling yet. Scanning…",
                parse_mode=ParseMode.HTML, reply_markup=watchlist_menu())
            return
        symbols = [w.get("symbol", "?") for w in watchlist[:27]]
        await update.message.reply_text(
            f"👁 <b>Watchlist</b>  —  {len(symbols)} setups forming\n\nTap a symbol for a snapshot:",
            parse_mode=ParseMode.HTML, reply_markup=watchlist_symbols(symbols, page=0))

    async def _cmd_market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import market_menu
        text = formatter.format_market_digest()
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=market_menu())

    async def _cmd_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import search_menu
        chat_id = str(update.message.chat_id)
        self.session_state.remember_view(chat_id, "search", message_id=getattr(update.message, "message_id", None))
        await update.message.reply_text(
            "🔎 <b>Search</b>\n\nFind any signal, symbol, strategy, note, or health incident.",
            parse_mode=ParseMode.HTML,
            reply_markup=search_menu(),
        )

    async def _cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import bot_health_menu
        chat_id = str(update.message.chat_id)
        self.session_state.remember_view(chat_id, "health", message_id=getattr(update.message, "message_id", None))
        await update.message.reply_text(
            await self.health_service.render_dashboard(),
            parse_mode=ParseMode.HTML,
            reply_markup=bot_health_menu(),
        )


    async def _cmd_sentinel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Sentinel intelligence commands.

        /sentinel                   — show feature menu
        /sentinel postmortem        — analyse last 30 closed trades
        /sentinel oracle            — predict next regime shift
        /sentinel doctor STRATEGY   — diagnose a specific strategy
        /sentinel whale SYMBOL      — classify whale intent on a symbol
        /sentinel blacklist         — scan + show current blacklist
        /sentinel unban SYMBOL      — remove symbol from blacklist
        /sentinel calibrate         — threshold calibration suggestions
        /sentinel corr              — correlation radar (top breaks)
        /sentinel news              — batch news spike scan
        """
        if not self._auth(update):
            return

        from analyzers.sentinel_features import sentinel
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        args = context.args or []
        sub = args[0].lower() if args else ""

        # ── postmortem ────────────────────────────────────────
        if sub == "postmortem":
            await update.message.reply_text(
                "🔍 <b>Running postmortem on last 30 trades…</b>\n"
                "<i>Reading outcome log, classifying loss patterns…</i>",
                parse_mode=ParseMode.HTML,
            )
            result = await sentinel.postmortem()
            if not result:
                await update.message.reply_text("⚠️ Not enough trade data yet (need 5+ closed trades).")
                return
            wr_emoji = "✅" if result.win_rate >= 0.55 else ("⚠️" if result.win_rate >= 0.40 else "🔴")
            lines = [
                f"📊 <b>Trade Postmortem</b> ({result.total_trades} trades)",
                f"━━━━━━━━━━━━━━━━━━━━━━━━",
                f"{wr_emoji} Win rate: <b>{result.win_rate*100:.1f}%</b>",
                f"",
                f"<b>Dominant loss pattern:</b>",
                f"  {result.dominant_loss_pattern}",
                f"",
                f"<b>Worst strategy:</b> {result.worst_strategy} ({result.worst_strategy_loss_rate*100:.0f}% loss rate)",
                f"",
                f"<b>Analysis:</b>",
                f"{result.insight}",
                f"",
                f"<b>Actions:</b>",
            ]
            for i, action in enumerate(result.actions, 1):
                lines.append(f"  {i}. {action}")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            return

        # ── oracle ────────────────────────────────────────────
        if sub == "oracle":
            await update.message.reply_text(
                "🔮 <b>Querying Regime Oracle…</b>\n"
                "<i>Analysing funding momentum, OI velocity, whale accumulation…</i>",
                parse_mode=ParseMode.HTML,
            )
            try:
                from analyzers.regime import regime_analyzer
                from analyzers.derivatives import derivatives_analyzer
                from signals.whale_aggregator import whale_aggregator

                ctx = {
                    "regime": regime_analyzer.regime.value if hasattr(regime_analyzer, "regime") else "UNKNOWN",
                    "adx": getattr(regime_analyzer, '_adx', 0),
                    "chop": getattr(regime_analyzer, '_chop', 0),
                }
                result = await sentinel.regime_oracle(ctx)
                if not result:
                    await update.message.reply_text("⚠️ Oracle needs more market data. Try again in a scan cycle.")
                    return

                conf_bar = "█" * int(result.confidence / 10) + "░" * (10 - int(result.confidence / 10))
                lines = [
                    f"🔮 <b>Regime Oracle</b>",
                    f"━━━━━━━━━━━━━━━━━━━━━━━━",
                    f"Current:  <b>{result.current_regime}</b>",
                    f"Predicted: <b>{result.predicted_next}</b>",
                    f"Confidence: {conf_bar} {result.confidence:.0f}%",
                    f"Time to shift: <b>{result.estimated_time_to_shift}</b>",
                    f"",
                    f"<b>Leading signals:</b>",
                ]
                for sig in result.leading_signals:
                    lines.append(f"  • {sig}")
                if result.trade_implication:
                    lines.append(f"\n<b>Action now:</b> {result.trade_implication}")
                await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            except Exception as _e:
                await update.message.reply_text(f"❌ Oracle error: {_e}")
            return

        # ── doctor ────────────────────────────────────────────
        if sub == "doctor":
            if len(args) < 2:
                await update.message.reply_text(
                    "Usage: /sentinel doctor <strategy>\n"
                    "Examples:\n"
                    "  /sentinel doctor RisingWedge\n"
                    "  /sentinel doctor FundingRateArb\n"
                    "  /sentinel doctor SMC",
                    parse_mode=ParseMode.HTML,
                )
                return
            strat = args[1]
            await update.message.reply_text(
                f"🩺 <b>Diagnosing {strat}…</b>\n"
                "<i>Reading kill reasons, outcome log, regime breakdown…</i>",
                parse_mode=ParseMode.HTML,
            )
            result = await sentinel.strategy_doctor(strat)
            if not result:
                await update.message.reply_text(f"⚠️ Not enough data for {strat} yet (need 5+ signals).")
                return
            wr_emoji = "✅" if result.win_rate >= 0.55 else ("⚠️" if result.win_rate >= 0.40 else "🔴")
            lines = [
                f"🩺 <b>Strategy Doctor: {result.strategy}</b>",
                f"━━━━━━━━━━━━━━━━━━━━━━━━",
                f"{wr_emoji} Win rate: <b>{result.win_rate*100:.1f}%</b> ({result.total_signals} outcomes)",
                f"Best regime: <b>{result.best_regime}</b>  Worst: <b>{result.worst_regime}</b>",
                f"",
                f"<b>Top kill reasons:</b>",
            ]
            for reason, count in result.top_kill_reasons[:4]:
                lines.append(f"  • {reason}: {count}×")
            lines += [
                f"",
                f"<b>Diagnosis:</b>",
                f"{result.diagnosis}",
                f"",
                f"<b>Prescription:</b>",
            ]
            for i, rx in enumerate(result.prescription, 1):
                lines.append(f"  {i}. {rx}")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            return

        # ── whale ─────────────────────────────────────────────
        if sub == "whale":
            if len(args) < 2:
                await update.message.reply_text(
                    "Usage: /sentinel whale <SYMBOL>\nExample: /sentinel whale DOGE",
                    parse_mode=ParseMode.HTML,
                )
                return
            sym = args[1].upper()
            if "/" not in sym:
                sym += "/USDT"
            await update.message.reply_text(
                f"🐋 <b>Classifying whale intent for {sym}…</b>",
                parse_mode=ParseMode.HTML,
            )
            result = await sentinel.whale_intent(sym)
            if not result:
                await update.message.reply_text(f"⚠️ No recent whale activity for {sym} (last 10 min).")
                return
            class_emoji = {
                "ACCUMULATION": "📈", "DISTRIBUTION": "📉",
                "SHORT_SQUEEZE": "💥", "PANIC_SELL": "🔴", "NOISE": "〰️"
            }.get(result.classification, "❓")
            impl_emoji = {"LONG_BIAS": "📈", "SHORT_BIAS": "📉", "AVOID": "⚠️", "NEUTRAL": "➡️"}.get(result.signal_implication, "➡️")
            lines = [
                f"🐋 <b>Whale Intent: {sym}</b>",
                f"━━━━━━━━━━━━━━━━━━━━━━━━",
                f"{class_emoji} Classification: <b>{result.classification}</b>",
                f"Confidence: {result.confidence:.0f}% | Urgency: {result.urgency}",
                f"Signal bias: {impl_emoji} <b>{result.signal_implication}</b>",
                f"",
                f"<b>Evidence:</b>",
            ]
            for ev in result.evidence:
                lines.append(f"  • {ev}")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            return

        # ── blacklist ─────────────────────────────────────────
        if sub == "blacklist":
            await update.message.reply_text(
                "🚫 <b>Running blacklist scan…</b>\n"
                "<i>Finding symbols with chronic false positive rates…</i>",
                parse_mode=ParseMode.HTML,
            )
            banned = await sentinel.symbol_blacklist_scan()
            active = sentinel.get_blacklist()
            lines = [f"🚫 <b>Symbol Blacklist</b>", "━━━━━━━━━━━━━━━━━━━━━━━━"]
            if banned:
                lines.append(f"<b>Newly banned ({len(banned)}):</b>")
                for b in banned[:5]:
                    lines.append(f"  🔴 {b.symbol} — {b.reason} (FPR={b.false_positive_rate:.0%}, {b.sample_size} signals)")
            if active:
                lines.append(f"\n<b>Currently blacklisted ({len(active)}):</b>")
                import time as _t
                for b in active[:8]:
                    hrs = (b.banned_until - _t.time()) / 3600
                    lines.append(f"  • {b.symbol} ({hrs:.1f}h left) — {b.reason}")
            else:
                lines.append("✅ No symbols currently blacklisted.")
            lines.append("\nUse /sentinel unban SYMBOL to remove manually.")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            return

        # ── unban ─────────────────────────────────────────────
        if sub == "unban":
            if len(args) < 2:
                await update.message.reply_text("Usage: /sentinel unban <SYMBOL>")
                return
            sym = args[1].upper()
            if "/" not in sym:
                sym += "/USDT"
            removed = sentinel.remove_from_blacklist(sym)
            await update.message.reply_text(
                f"{'✅ ' + sym + ' removed from blacklist.' if removed else '⚠️ ' + sym + ' was not blacklisted.'}",
                parse_mode=ParseMode.HTML,
            )
            return

        # ── calibrate ─────────────────────────────────────────
        if sub == "calibrate":
            await update.message.reply_text(
                "🔧 <b>Running threshold calibration…</b>\n"
                "<i>Analysing outcome data for parameter improvement opportunities…</i>",
                parse_mode=ParseMode.HTML,
            )
            suggestions = await sentinel.threshold_calibration()
            if not suggestions:
                await update.message.reply_text("⚠️ Not enough outcome data yet (need 20+ closed trades).")
                return
            lines = [f"🔧 <b>Threshold Calibration ({len(suggestions)} suggestions)</b>", "━━━━━━━━━━━━━━━━━━━━━━━━"]
            for s in suggestions:
                risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(s.risk, "⚪")
                lines += [
                    f"",
                    f"{risk_emoji} <b>{s.parameter}</b> ({s.strategy})",
                    f"  {s.current_value} → <b>{s.suggested_value}</b>",
                    f"  {s.evidence}",
                    f"  Expected: <i>{s.expected_improvement}</i>",
                ]
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            return

        # ── corr ──────────────────────────────────────────────
        if sub == "corr":
            await update.message.reply_text(
                "📡 <b>Correlation Radar scanning…</b>\n"
                "<i>Checking BTC correlation breakdowns across active symbols…</i>",
                parse_mode=ParseMode.HTML,
            )
            try:
                # Use whatever symbols the engine has loaded
                from core.engine import TitanBotEngine
                # Fallback: just show message if engine not accessible
                await update.message.reply_text(
                    "📡 Correlation radar runs automatically during scans.\n"
                    "Correlation breaks are flagged in the signal confluence notes\n"
                    "(look for 📡 Correlation Break in signal cards).\n\n"
                    "Use /sentinel corr live for real-time scan (starts next cycle).",
                    parse_mode=ParseMode.HTML,
                )
            except Exception as _e:
                await update.message.reply_text(f"❌ Corr error: {_e}")
            return

        # ── news ──────────────────────────────────────────────
        if sub == "news":
            await update.message.reply_text(
                "📰 <b>News Spike Scanner running…</b>\n"
                "<i>Batch-scanning headlines across all symbols in one AI call…</i>",
                parse_mode=ParseMode.HTML,
            )
            try:
                from analyzers.news_scraper import news_scraper
                from core.engine import _active_symbols  # noqa — may not exist, handled
                symbols = list(getattr(news_scraper, '_symbol_list', []))[:120]
                if not symbols:
                    await update.message.reply_text("⚠️ No symbol list loaded yet — wait for first scan cycle.")
                    return
                spikes = await sentinel.news_spike_scan(symbols)
                if not spikes:
                    await update.message.reply_text("✅ No high-impact news catalysts detected right now.")
                    return
                lines = [f"📰 <b>News Spikes ({len(spikes)} detected)</b>", "━━━━━━━━━━━━━━━━━━━━━━━━"]
                for s in spikes[:8]:
                    sent_emoji = "🟢" if s.sentiment == "BULLISH" else "🔴"
                    urg_emoji = "🔥" if s.urgency == "HIGH" else "⚡"
                    lines.append(f"{urg_emoji}{sent_emoji} <b>{s.symbol}</b> — {s.key_headline[:70]}")
                    lines.append(f"  Impact window: {s.estimated_move_window}")
                await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            except Exception as _e:
                await update.message.reply_text(f"❌ News scan error: {_e}")
            return

        # ── Default: feature menu ─────────────────────────────
        menu_text = (
            "🛡️ <b>Sentinel Intelligence</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Eight AI-powered analysis tools:\n\n"
            "📊 <b>/sentinel postmortem</b>\n"
            "   Analyses last 30 closed trades. Finds losing patterns.\n\n"
            "🔮 <b>/sentinel oracle</b>\n"
            "   Predicts next regime shift 15-30min ahead.\n\n"
            "🩺 <b>/sentinel doctor</b> <code>STRATEGY</code>\n"
            "   Deep-dives one strategy. Diagnosis + prescription.\n\n"
            "🐋 <b>/sentinel whale</b> <code>SYMBOL</code>\n"
            "   Classifies whale flow: accumulation / squeeze / panic.\n\n"
            "🚫 <b>/sentinel blacklist</b>\n"
            "   Scans for chronically bad symbols, temp-bans them.\n\n"
            "🔧 <b>/sentinel calibrate</b>\n"
            "   Suggests specific parameter numbers from your data.\n\n"
            "📡 <b>/sentinel corr</b>\n"
            "   BTC correlation breakdowns = independent move alerts.\n\n"
            "📰 <b>/sentinel news</b>\n"
            "   Batch scans all symbols for news catalysts in 1 call."
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Postmortem",  callback_data="sen_postmortem"),
                InlineKeyboardButton("🔮 Oracle",      callback_data="sen_oracle"),
            ],
            [
                InlineKeyboardButton("🐋 Whale DOGE",  callback_data="sen_whale_DOGE"),
                InlineKeyboardButton("🚫 Blacklist",   callback_data="sen_blacklist"),
            ],
            [
                InlineKeyboardButton("🔧 Calibrate",  callback_data="sen_calibrate"),
                InlineKeyboardButton("📰 News Spikes", callback_data="sen_news"),
            ],
        ])
        await update.message.reply_text(menu_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

    async def _cmd_whales(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show HyperTracker smart-money cohort table + tracked wallet positions."""
        if not self._auth(update):
            return
        from analyzers.hypertracker_client import hypertracker
        from tg.keyboards import market_menu
        await update.message.reply_text("🐋 Fetching smart money data…")
        # Refresh all cohorts on demand
        cohorts = await hypertracker.get_all_cohorts()
        if cohorts:
            # Update internal cache with fresh data before formatting
            pass
        text = hypertracker.format_whales_summary()
        wallet_text = hypertracker.format_wallet_positions()
        if wallet_text and "No tracked wallets" not in wallet_text:
            text += f"\n\n{wallet_text}"
        await update.message.reply_text(
            text, parse_mode=ParseMode.HTML, reply_markup=market_menu()
        )

    async def _cmd_trending(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show currently trending coins on CoinGecko."""
        if not self._auth(update):
            return
        from analyzers.coingecko_client import coingecko
        from tg.keyboards import market_menu
        tickers = coingecko.get_trending_list()
        last_upd = coingecko.last_updated
        import time
        age_min = int((time.time() - last_upd) / 60) if last_upd > 0 else None

        if not tickers:
            await update.message.reply_text(
                "📈 <b>CoinGecko Trending</b>\n\n"
                "Warming up — first fetch takes ~15s after start. Try again in a moment.",
                parse_mode=ParseMode.HTML, reply_markup=market_menu()
            )
            return

        age_str = f"{age_min}m ago" if age_min is not None else "just now"
        text = f"🔥 <b>CoinGecko Trending</b>  <i>(updated {age_str})</i>\n\n"
        text += "Top 7 coins by search volume in the last 24h:\n\n"
        for i, ticker in enumerate(tickers, 1):
            text += f"{i}. <b>{ticker}</b>\n"
        text += "\n<i>Signals on these coins get a +6 confidence boost.</i>"
        await update.message.reply_text(
            text, parse_mode=ParseMode.HTML, reply_markup=market_menu()
        )

    async def _cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show recent news for a symbol. Usage: /news BTC"""
        if not self._auth(update):
            return
        from analyzers.news_scraper import news_scraper
        from tg.keyboards import market_menu
        args = context.args or []
        if not args:
            # Show market-wide top headlines
            summary = news_scraper.get_market_sentiment_summary()
            total = summary.get("story_count", 0)
            if total == 0:
                await update.message.reply_text(
                    "📰 No news loaded yet — RSS scraper is still warming up. Try again in 2 minutes.",
                    reply_markup=market_menu()
                )
                return
            score    = summary.get("score", 50)
            label    = summary.get("label", "Neutral")
            bull     = summary.get("bull_count", 0)
            bear     = summary.get("bear_count", 0)
            s_emoji  = "🟢" if score >= 65 else ("🔴" if score <= 35 else "⚪")
            text = (
                f"📰 <b>Market News Summary</b>\n\n"
                f"Stories (1h):   {total}\n"
                f"Sentiment:      {s_emoji} {label} ({score:.0f}/100)\n"
                f"Bullish:        {bull}  |  Bearish: {bear}\n\n"
                f"<b>Top headlines:</b>"
            )
            for h in news_scraper.get_all_stories(max_age_mins=60, limit=5):
                sc = h.get("sentiment_score", 50)
                em = "🟢" if sc >= 65 else ("🔴" if sc <= 35 else "⚪")
                text += f"\n{em} <i>{h['title'][:80]}</i>  <code>{h['source']}</code>"
            text += "\n\n<i>Use /news BTC for symbol-specific news</i>"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=market_menu())
            return

        symbol = args[0].upper().replace("USDT", "").replace("/", "")
        news = news_scraper.get_news_for_symbol(symbol, max_age_mins=120)[:8]
        if not news:
            await update.message.reply_text(
                f"📰 No recent news for <b>{symbol}</b> in the last 2 hours.\n"
                f"Try a more common symbol or check again later.",
                parse_mode=ParseMode.HTML, reply_markup=market_menu()
            )
            return
        text = f"📰 <b>News: {symbol}</b>  ({len(news)} stories)\n"
        for item in news:
            score = news_scraper._score_headline(item.get("title", ""))
            s_emoji = "🟢" if score > 0 else ("🔴" if score < 0 else "⚪")
            text += f"\n{s_emoji} <i>{item['title'][:75]}</i>\n    <code>{item['source']}</code>\n"
        await update.message.reply_text(
            text, parse_mode=ParseMode.HTML,
            reply_markup=market_menu(),
            disable_web_page_preview=True
        )

    async def _cmd_news_override(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Manually override the automated BTC news classification.

        Usage:
          /news_override status
          /news_override clear
          /news_override set <event_type> <direction> <conf_mult> <size_mult>
                            [ttl_minutes] [reason...]

        event_type: MACRO_RISK_OFF | MACRO_RISK_ON | BTC_FUNDAMENTAL |
                    BTC_TECHNICAL | EXCHANGE_EVENT | REGULATORY | UNKNOWN
        direction:  BULLISH | BEARISH | NEUTRAL
        conf_mult:  e.g. 1.0 = unchanged, 0.7 = soften, 1.1 = boost
        size_mult:  e.g. 1.0 = unchanged, 0.5 = half size

        Example:
          /news_override set MACRO_RISK_OFF BEARISH 0.95 0.95 60 Iran \
              delegation is de-escalation, not escalation
        """
        if not self._auth(update):
            return
        from analyzers.news_override import news_override_store
        args = context.args or []
        sub = (args[0].lower() if args else "status")

        if sub in ("status", "show"):
            st = news_override_store.status()
            active = st.get("active")
            if not active:
                await update.message.reply_text(
                    "📝 <b>News Override</b>\n\nNo active override.",
                    parse_mode=ParseMode.HTML,
                )
                return
            remaining = st.get("expires_in_minutes", 0)
            text = (
                f"📝 <b>News Override — Active</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Event:   <b>{active.get('event_type')}</b>\n"
                f"Dir:     <b>{active.get('direction')}</b>\n"
                f"Conf×:   <b>{active.get('confidence_mult')}</b>\n"
                f"Size×:   <b>{active.get('size_mult')}</b>\n"
                f"Blocks:  LONG={active.get('block_longs')} SHORT={active.get('block_shorts')}\n"
                f"Set by:  <b>{active.get('set_by')}</b>\n"
                f"Expires: <b>{remaining:.0f}m</b>\n"
                f"Next-news consume: <b>{active.get('consume_on_next_event')}</b>\n"
                f"Reason:  <i>{active.get('reason') or '—'}</i>"
            )
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return

        if sub in ("clear", "remove", "cancel"):
            prev = await news_override_store.clear(reason="/news_override clear")
            if prev is None:
                await update.message.reply_text("📝 No override to clear.")
            else:
                await update.message.reply_text(
                    f"📝 Override cleared: <b>{prev.event_type}/{prev.direction}</b>",
                    parse_mode=ParseMode.HTML,
                )
            return

        if sub != "set" or len(args) < 5:
            await update.message.reply_text(
                "Usage:\n"
                "<code>/news_override status</code>\n"
                "<code>/news_override clear</code>\n"
                "<code>/news_override set &lt;event_type&gt; &lt;direction&gt; "
                "&lt;conf_mult&gt; &lt;size_mult&gt; [ttl_minutes] [reason...]</code>",
                parse_mode=ParseMode.HTML,
            )
            return

        event_type = args[1].upper()
        direction  = args[2].upper()
        try:
            conf_mult = float(args[3])
            size_mult = float(args[4])
        except ValueError:
            await update.message.reply_text("❌ conf_mult and size_mult must be numbers.")
            return
        ttl_minutes: Optional[int] = None  # default — store fills in
        reason_start = 5
        if len(args) >= 6:
            try:
                ttl_minutes = int(args[5])
                reason_start = 6
            except ValueError:
                ttl_minutes = None
                reason_start = 5
        reason = " ".join(args[reason_start:]).strip()

        # Derive a user-facing identity.
        set_by = "operator"
        try:
            set_by = update.effective_user.username or str(update.effective_user.id)
        except Exception:
            pass

        ov = await news_override_store.set_override(
            event_type=event_type,
            direction=direction,
            confidence_mult=conf_mult,
            size_mult=size_mult,
            reason=reason,
            set_by=set_by,
            ttl_minutes=ttl_minutes,
        )
        await update.message.reply_text(
            f"📝 Override installed\n"
            f"Event: <b>{ov.event_type}/{ov.direction}</b>\n"
            f"Conf×{ov.confidence_mult:.2f} Size×{ov.size_mult:.2f}\n"
            f"TTL: <b>{(ov.expires_at_utc - ov.set_at_utc)/60:.0f}m</b>",
            parse_mode=ParseMode.HTML,
        )

    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import performance_menu
        await update.message.reply_text(
            "📈 <b>Performance</b>\n\nSelect period:",
            parse_mode=ParseMode.HTML, reply_markup=performance_menu())

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import pause_confirm, resume_confirm
        if self.paused:
            await update.message.reply_text(
                "⏸ <b>Bot is paused.</b>\n\nResume signal sending?",
                parse_mode=ParseMode.HTML, reply_markup=resume_confirm())
        else:
            await update.message.reply_text(
                "⏸ <b>Pause signals?</b>\n\nScanning continues — only signal delivery stops.",
                parse_mode=ParseMode.HTML, reply_markup=pause_confirm())

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        if not self.paused:
            await update.message.reply_text("▶️ Bot is already running.")
            return
        self.paused = False
        if self.on_resume:
            await self.on_resume()
        from tg.keyboards import main_menu
        await update.message.reply_text(
            "▶️ <b>Bot resumed.</b> Signals active.", parse_mode=ParseMode.HTML,
            reply_markup=main_menu())

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import settings_menu
        await update.message.reply_text(
            "⚙️ <b>Settings</b>\n\nSelect a section:",
            parse_mode=ParseMode.HTML, reply_markup=settings_menu())

    async def _cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        # FIX #19: Per-user 30s cooldown prevents scan spam creating 100 concurrent tasks.
        if not hasattr(self, '_scan_cooldowns'):
            self._scan_cooldowns: dict = {}
        import time as _time
        user_id = update.effective_user.id if update.effective_user else 0
        last_scan = self._scan_cooldowns.get(user_id, 0)
        elapsed = _time.time() - last_scan
        if elapsed < 30:
            remaining = int(30 - elapsed)
            await update.message.reply_text(f"⏳ Scan cooldown: {remaining}s remaining.")
            return
        self._scan_cooldowns[user_id] = _time.time()
        await update.message.reply_text("🔄 Scan triggered…")
        if self.on_force_scan:
            if not self._force_scan_task or self._force_scan_task.done():
                self._force_scan_task = asyncio.create_task(self.on_force_scan())
                self._force_scan_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            else:
                await update.message.reply_text("⏳ Scan already running…")

    async def _cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import performance_menu
        stats       = await db.get_performance_stats(days=30)
        by_strategy = await db.get_strategy_performance(days=30)
        by_sector   = await db.get_sector_performance(days=30)
        stats["by_strategy"] = by_strategy
        stats["by_sector"]   = by_sector
        text = formatter.format_performance(stats, "30 days")
        await update.message.reply_text(text, parse_mode=ParseMode.HTML,
                                         reply_markup=performance_menu())

    async def _cmd_ranking(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from governance.performance_tracker import performance_tracker
        from tg.keyboards import back_to_main
        text = performance_tracker.get_ranking()
        await update.message.reply_text(text, parse_mode=ParseMode.HTML,
                                         reply_markup=back_to_main())

    async def _cmd_panel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Redirects to the unified main menu — /start is the single home."""
        if not self._auth(update):
            return
        from tg.keyboards import main_menu
        text = self._render_dashboard_text()
        msg = await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=main_menu())
        self._panel_message_id = msg.message_id

    async def _cmd_upgrades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show Phase 2 upgrade learning status — posteriors + velocity."""
        if not self._auth(update):
            return
        from signals.upgrade_tracker import upgrade_tracker
        stats = upgrade_tracker.get_stats()
        posterior_report = upgrade_tracker.get_posterior_report()
        velocity_report  = upgrade_tracker.get_velocity_report()

        try:
            db_stats = await db.get_upgrade_stats()
        except Exception:
            db_stats = {}

        text = (
            f"⬆️ <b>UPGRADE TRACKER</b>\n\n"
            f"<b>Session Stats</b>\n"
            f"C signals seen:   {stats['total_c_signals']}\n"
            f"Sent to you:      {stats['sent_to_user']}\n"
            f"Suppressed:       {stats['suppressed']}  ({stats['suppress_rate']})\n"
            f"Upgraded:         {stats['upgraded']}  ({stats['upgrade_rate']})\n"
            f"Avg upgrade time: {stats['avg_upgrade_min']}\n\n"
            f"Phase 2 contexts: {stats['phase2_active_keys']} active "
            f"/ {stats['total_context_keys']} total\n\n"
        )

        if db_stats:
            text += (
                f"<b>30-Day DB Stats</b>\n"
                f"Total C signals:  {db_stats.get('total_30d', '—')}\n"
                f"Sent to user:     {db_stats.get('sent_30d', '—')}\n"
                f"Upgraded:         {db_stats.get('upgraded_30d', '—')}  "
                f"({db_stats.get('upgrade_rate', '—')})\n"
                f"Avg latency:      {db_stats.get('avg_latency_min', '—')}min\n\n"
            )

        text += posterior_report + "\n\n" + velocity_report
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def _cmd_explain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show full life/death reason stack for any signal ID."""
        if not self._auth(update):
            return

        args = context.args if context.args else []
        if not args:
            await update.message.reply_text(
                "Usage: /explain <signal_id>\n"
                "Example: /explain 142\n\n"
                "Shows why a signal was published or rejected, with full scoring stack.",
                parse_mode=ParseMode.HTML,
            )
            return

        try:
            signal_id = int(args[0])
        except ValueError:
            await update.message.reply_text("❌ Invalid signal ID. Use a number, e.g. /explain 142")
            return

        sig = await db.get_signal(signal_id)
        if not sig:
            await update.message.reply_text(f"❌ Signal #{signal_id} not found in database.")
            return

        import json as _json

        symbol     = sig.get('symbol', '?')
        direction  = sig.get('direction', '?')
        strategy   = sig.get('strategy', '?')
        confidence = sig.get('confidence', 0)
        grade      = sig.get('alpha_grade', '?')
        p_win      = sig.get('p_win', 0)
        ev_r       = sig.get('ev_r', 0)
        outcome    = sig.get('outcome', 'PENDING')
        pnl_r      = sig.get('pnl_r', None)
        reject     = sig.get('reject_reason', None)
        regime     = sig.get('regime', '?')
        created    = sig.get('created_at', '?')

        confluence_raw = sig.get('confluence', '[]')
        try:
            confluence_list = _json.loads(confluence_raw) if isinstance(confluence_raw, str) else confluence_raw
        except Exception:
            confluence_list = []

        evidence_raw = sig.get('evidence', '{}')
        try:
            evidence_dict = _json.loads(evidence_raw) if isinstance(evidence_raw, str) else evidence_raw
            active_evidence = [k for k, v in evidence_dict.items() if v]
        except Exception:
            active_evidence = []

        outcome_str = f"{outcome}"
        if pnl_r is not None and pnl_r != 0:
            outcome_str += f" ({pnl_r:+.2f}R)"

        if reject:
            header = f"🚫 <b>REJECTED</b>"
        elif outcome == 'PENDING' or not outcome:
            header = f"⏳ <b>PUBLISHED — awaiting outcome</b>"
        else:
            o_emoji = {"WIN": "✅", "LOSS": "🔴", "BREAKEVEN": "➡️", "EXPIRED": "⏰"}.get(outcome, "📊")
            header = f"{o_emoji} <b>{outcome_str}</b>"

        lines = [
            f"🔍 <b>Signal #{signal_id} — Explain</b>",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━",
            header,
            f"",
            f"<b>Signal:</b> {symbol} {direction} · {strategy}",
            f"<b>Created:</b> {created}",
            f"<b>Regime:</b> {regime}",
            f"",
            f"<b>Scoring:</b>",
            f"  Confidence: {confidence:.0f}",
            f"  Grade: {grade}",
            f"  P(win): {p_win*100:.0f}%",
            f"  EV: {ev_r:+.2f}R",
        ]

        if reject:
            lines += [f"", f"<b>Rejection reason:</b>", f"  {reject}"]

        if active_evidence:
            lines += [f"", f"<b>Evidence present ({len(active_evidence)}):</b>"]
            for ev in active_evidence[:8]:
                lines.append(f"  ✅ {ev}")

        if confluence_list:
            lines += [f"", f"<b>Confluence factors ({len(confluence_list)}):</b>"]
            for c in confluence_list[:10]:
                lines.append(f"  · {c[:80]}")

        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def _cmd_quick(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """V17: Quick one-line status."""
        try:
            from core.engine import engine as _eng
            uptime_m = int((time.time() - self._start_time) / 60)
            regime = regime_analyzer.regime.value if regime_analyzer.regime else "?"
            status = "⏸" if self.paused else "▶️"
            fg = regime_analyzer.fear_greed
            text = (
                f"{status} {regime} | F&G:{fg} | "
                f"Scans:{_eng._scan_count} Sigs:{_eng._signal_count} | "
                f"Up:{uptime_m}m"
            )
            await update.message.reply_text(text)
        except Exception as e:
            await update.message.reply_text(f"Status: {e}")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        from tg.keyboards import help_menu
        text = (
            "❓ <b>TitanBot Pro — Help</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "All features are available through the inline menus.\n"
            "Type /start for the main menu at any time.\n\n"
            "<b>Quick commands</b>\n"
            "/start   — Dashboard + main menu\n"
            "/status  — Live bot status\n"
            "/signals — Recent signals\n"
            "/search  — Manual lookup\n"
            "/health  — Bot health console\n"
            "/market  — Market digest\n"
            "/watchlist — Setups forming\n"
            "/scan    — Force scan\n"
            "/panel   — Admin control panel\n"
            "/pause   — Pause / resume\n"
            "/reload  — Hot-reload config (no restart)\n"
            "/replay  — Replay signals on a past date\n\n"
            "Select a help topic below:"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=help_menu())

    async def _cmd_reload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """EC8: Hot-reload settings.yaml without restarting the bot."""
        if not self._auth(update):
            return
        try:
            from config.loader import cfg
            cfg.reload()
            await update.message.reply_text(
                "✅ <b>Config reloaded</b>\n"
                "settings.yaml changes are now active.\n"
                "<i>Note: Telegram token and chat IDs require a restart to take effect.</i>",
                parse_mode=ParseMode.HTML,
            )
            logger.info(f"Config hot-reloaded by user {update.effective_user.id}")
        except Exception as e:
            await update.message.reply_text(
                f"❌ Config reload failed: {e}", parse_mode=ParseMode.HTML
            )

    async def _cmd_replay(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """I7: Replay scan logic against a historical date (paper mode — no real signals sent)."""
        if not self._auth(update):
            return
        usage = (
            "📼 <b>Replay Mode</b>\n\n"
            "Replays the scan logic against historical OHLCV data.\n"
            "Usage: <code>/replay YYYY-MM-DD</code>\n"
            "Example: <code>/replay 2026-03-07</code>\n\n"
            "<i>Results are paper-only — no real signals published.</i>"
        )
        args = context.args or []
        if not args:
            await update.message.reply_text(usage, parse_mode=ParseMode.HTML)
            return
        date_str = args[0].strip()
        await update.message.reply_text(
            f"⏳ Replaying signals for <b>{date_str}</b>…\n"
            f"This may take 1-2 minutes.",
            parse_mode=ParseMode.HTML,
        )
        try:
            from backtester.engine import BacktestEngine
            engine = BacktestEngine()
            results = await engine.run_replay(date_str)
            if not results:
                await update.message.reply_text(
                    f"No historical data available for {date_str}.", parse_mode=ParseMode.HTML
                )
                return
            lines = [f"📼 <b>Replay: {date_str}</b>\n"]
            for r in results[:15]:
                _emoji = "✅" if r.get('would_win') else "🔴"
                lines.append(
                    f"{_emoji} {r.get('symbol')} {r.get('direction')} "
                    f"({r.get('strategy')}) — {r.get('grade', '?')}"
                )
            lines.append(f"\n<i>{len(results)} signal(s) found in replay window.</i>")
            await update.message.reply_text(
                "\n".join(lines), parse_mode=ParseMode.HTML
            )
        except NotImplementedError:
            await update.message.reply_text(
                "📼 Replay via <code>/replay</code> is available.\n"
                "For full historical backtests, run: <code>python run_backtest.py</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            await update.message.reply_text(
                f"❌ Replay failed: {e}", parse_mode=ParseMode.HTML
            )
            logger.error(f"Replay error: {e}", exc_info=True)
    # ─────────────────────────────────────────────────────────

    async def _cmd_ai(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        /ai                      — show AI status panel
        /ai on                   — set mode to full
        /ai off                  — disable AI completely
        /ai diagnosis            — diagnosis only (no signal AI)
        /ai report               — trigger immediate diagnostic report
        /ai approve              — show pending approvals
        /ai approve_all_low      — batch-apply every LOW-risk pending approval
        /ai why SYMBOL           — explain why SYMBOL signals keep dying
        /ai history              — last applied overrides
        /ai audit                — run Sentinel audit now (fast tier, Llama)
        /ai audit deep           — run deep audit (Nemotron, max 1x/24h)
        /ai audit status         — show last audit time + session stats
        """
        if not self._auth(update):
            return

        from analyzers.ai_analyst import ai_analyst
        from core.diagnostic_engine import diagnostic_engine
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        args = context.args or []
        sub = args[0].lower() if args else ""

        # ── Subcommands ──────────────────────────────────────
        if sub in ("on", "full"):
            ai_analyst.set_mode("full")
            await update.message.reply_text(
                "🤖 <b>AI Analyst: FULL MODE</b>\n"
                "Signal AI + diagnostic engine active.",
                parse_mode=ParseMode.HTML,
            )
            return

        if sub == "off":
            ai_analyst.set_mode("off")
            await update.message.reply_text(
                "🤖 <b>AI Analyst: OFF</b>\n"
                "Bot running in pure rule-based mode (v19 behaviour).\n"
                "Diagnostic engine still active.\n"
                "Type /ai on to re-enable.",
                parse_mode=ParseMode.HTML,
            )
            return

        if sub == "diagnosis":
            ai_analyst.set_mode("diagnosis_only")
            await update.message.reply_text(
                "🔬 <b>AI: Diagnosis Only</b>\n"
                "Self-healing diagnostic active.\n"
                "Signal AI (sentiment/narrative) disabled.",
                parse_mode=ParseMode.HTML,
            )
            return

        if sub == "report":
            await update.message.reply_text(
                "⏳ Generating diagnostic report…", parse_mode=ParseMode.HTML
            )
            await diagnostic_engine.get_report_on_demand()
            return

        if sub == "approve":
            pending = diagnostic_engine.get_pending_approvals()
            if not pending:
                await update.message.reply_text(
                    "✅ No pending approvals.", parse_mode=ParseMode.HTML
                )
                return
            for a in pending[:5]:
                await self.send_approval_request(a)
            return

        # Option B — batch-apply every pending LOW-risk proposal at once.
        # MEDIUM and HIGH are never touched by this command; they still need
        # individual taps.  Each applied change gets its own rollback snapshot
        # so /undo <id> remains available per item.
        if sub in ("approve_all_low", "approve-all-low", "approveall"):
            pending = diagnostic_engine.get_pending_approvals()
            low = [a for a in pending if str(a.risk_level or "").upper() == "LOW"]
            if not low:
                await update.message.reply_text(
                    "✅ No LOW-risk approvals pending.", parse_mode=ParseMode.HTML
                )
                return
            applied = await diagnostic_engine.apply_all_low_risk()
            await update.message.reply_text(
                f"✅ <b>Applied {len(applied)} LOW-risk change(s)</b>\n"
                f"IDs: <code>{', '.join(applied) if applied else '—'}</code>\n"
                f"Use /ai history to review or tap UNDO on the per-item cards.",
                parse_mode=ParseMode.HTML,
            )
            return

        if sub == "history":
            overrides = diagnostic_engine.get_applied_overrides()
            if not overrides:
                await update.message.reply_text(
                    "No active overrides.", parse_mode=ParseMode.HTML
                )
                return
            lines = ["⚙️ <b>Active Overrides</b>\n"]
            for ov in overrides[-10:]:
                age_h = (time.time() - ov.get("applied_at", 0)) / 3600
                lines.append(
                    f"• {ov['description']} "
                    f"(<i>{age_h:.1f}h ago</i>)"
                )
            lines.append("\nType /ai approve to see pending proposals.")
            await update.message.reply_text(
                "\n".join(lines), parse_mode=ParseMode.HTML
            )
            return

        if sub == "why" and len(args) >= 2:
            symbol_query = args[1].upper()
            if "/" not in symbol_query:
                symbol_query += "/USDT"
            await update.message.reply_text(
                f"🤖 Analysing why {symbol_query} signals keep dying…",
                parse_mode=ParseMode.HTML,
            )
            # Pull recent deaths for this symbol
            import time as _t
            now = _t.time()
            recent = [
                d for d in diagnostic_engine._death_log
                if d.get("symbol") == symbol_query and now - d["ts"] < 21600
            ]
            # Build log context
            log_lines = [
                f"{d['kill_reason']} | {d['strategy']} | conf={d['confidence']:.0f} rr={d['rr']:.2f} | {d['regime']}"
                for d in recent[:15]
            ]
            # FIX #33: Run AI call in background task, don't block Telegram event loop.
            # explain_signal_death calls OpenRouter and can take 3-8 seconds.
            # Blocking here backs up the update queue for all other users.
            msg = update.message
            _sym = symbol_query
            _recent = recent[:10]
            _log = log_lines

            async def _explain_async():
                try:
                    explanation = await ai_analyst.explain_signal_death(
                        symbol=_sym,
                        recent_deaths=_recent,
                        recent_log_lines=_log,
                    )
                    await msg.reply_text(
                        f"🤖 <b>Why is {_sym} dying?</b>\n\n{explanation}",
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as _e:
                    await msg.reply_text(f"❌ AI analysis failed: {_e}")

            asyncio.create_task(_explain_async())
            return

        if sub == "audit":
            subsub = args[1].lower() if len(args) >= 2 else ""

            if subsub == "status":
                astat = ai_analyst.get_audit_status()
                ss = astat["session_stats"]
                fast_ago = astat["fast_last_ran"]
                deep_ago = astat["deep_last_ran"]
                fast_next = astat["fast_next_in"]
                deep_next = astat["deep_next_in"]

                def _fmt_age(secs):
                    if secs > 86000: return "never"
                    if secs < 120:   return f"{secs}s ago"
                    if secs < 3600:  return f"{secs//60}m ago"
                    return f"{secs//3600}h {(secs%3600)//60}m ago"

                def _fmt_next(secs):
                    if secs <= 0:   return "ready now"
                    if secs < 120:  return f"in {secs}s"
                    if secs < 3600: return f"in {secs//60}m"
                    return f"in {secs//3600}h {(secs%3600)//60}m"

                strat_lines = []
                for strat, counts in list(ss.get("strategy_direction", {}).items())[:8]:
                    total = counts.get("LONG", 0) + counts.get("SHORT", 0)
                    if total >= 3:
                        sp = counts.get("SHORT", 0) / total * 100
                        bias = " ⚠️" if sp > 85 or sp < 15 else ""
                        strat_lines.append(
                            f"  <code>{strat.split(':')[-1]:<20}</code> "
                            f"S:{counts.get('SHORT',0)} L:{counts.get('LONG',0)}{bias}"
                        )

                text = (
                    f"🔬 <b>Sentinel Audit Status</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"<b>Fast audit</b> (Llama, 1h auto)\n"
                    f"  Last ran: {_fmt_age(fast_ago)}\n"
                    f"  Next auto: {_fmt_next(fast_next)}\n\n"
                    f"<b>Deep audit</b> (Nemotron, on-demand)\n"
                    f"  Last ran: {_fmt_age(deep_ago)}\n"
                    f"  Next allowed: {_fmt_next(deep_next)}\n\n"
                    f"<b>Session accumulation</b>\n"
                    f"  HTF blocked: LONG={ss.get('htf_blocked_long',0)} SHORT={ss.get('htf_blocked_short',0)}\n"
                    f"  Whale flow: BUY={ss.get('whale_buy_events',0)}(${ss.get('whale_buy_usd',0)/1000:.0f}k) "
                    f"SELL={ss.get('whale_sell_events',0)}(${ss.get('whale_sell_usd',0)/1000:.0f}k)\n"
                    f"  vs-whale conflicts: {ss.get('signals_vs_whale_conflict',0)}\n\n"
                    f"<b>Strategy direction (this session)</b>\n"
                    + ("\n".join(strat_lines) if strat_lines else "  No data yet — session accumulating")
                )
                await update.message.reply_text(text, parse_mode=ParseMode.HTML)
                return

            deep_mode = subsub == "deep"
            tier_name = "Nemotron (deep)" if deep_mode else "Llama (fast)"
            await update.message.reply_text(
                f"🔬 <b>Running Sentinel audit…</b>\n"
                f"Model: <code>{tier_name}</code>\n"
                f"<i>Analysing strategy direction bias, HTF asymmetry, whale conflicts…</i>",
                parse_mode=ParseMode.HTML,
            )
            try:
                audit = await ai_analyst.structural_audit(force=True, deep=deep_mode)
                if not audit:
                    await update.message.reply_text(
                        "⚠️ Audit returned no result — not enough session data yet.\n"
                        "Let the bot scan a few hundred symbols first.",
                        parse_mode=ParseMode.HTML,
                    )
                    return
                emoji_map = {"LOW": "ℹ️", "MEDIUM": "⚠️", "HIGH": "🚨", "CRITICAL": "🔥"}
                sev_emoji = emoji_map.get(audit.severity, "ℹ️")
                lines = [
                    f"{sev_emoji} <b>Sentinel Audit — {audit.severity}</b>",
                    "━━━━━━━━━━━━━━━━━━━━━━━━",
                ]
                if audit.biased_strategies:
                    lines.append(f"⚠️ <b>Biased strategies:</b> {', '.join(audit.biased_strategies)}")
                if audit.htf_asymmetry:
                    lines.append("⚠️ <b>HTF guardrail</b> blocking one direction 3× more")
                if audit.whale_signal_conflict:
                    lines.append("⚠️ <b>Signals frequently trading against whale flow</b>")
                if audit.regime_signal_conflict:
                    lines.append("⚠️ <b>Signals firing opposite to regime direction</b>")
                if audit.root_causes:
                    lines.append("\n<b>Root causes:</b>")
                    for rc in audit.root_causes[:4]:
                        lines.append(f"  • {rc}")
                if audit.recommendations:
                    lines.append("\n<b>Recommendations:</b>")
                    for rec in audit.recommendations[:4]:
                        lines.append(f"  → {rec}")
                if audit.severity == "LOW" and not audit.biased_strategies:
                    lines.append("\n✅ No structural issues found this session.")
                await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            except Exception as _ae:
                await update.message.reply_text(f"❌ Audit failed: {_ae}", parse_mode=ParseMode.HTML)
            return

        # ── Default: status panel ─────────────────────────────
        import time
        status = ai_analyst.get_status()
        diag_stats = diagnostic_engine.get_stats_summary()

        mode_emoji = {"full": "🟢", "diagnosis_only": "🟡", "off": "🔴"}.get(status["mode"], "⚪")
        mode_label = {"full": "FULL (signal AI + diagnosis)", "diagnosis_only": "DIAGNOSIS ONLY", "off": "OFF"}.get(status["mode"], status["mode"])

        text = (
            f"🤖 <b>AI Analyst — Status</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Mode: {mode_emoji} <b>{mode_label}</b>\n"
            f"Models: fast=<code>{status.get('model_fast','?').split('/')[-1]}</code> deep=<code>{status.get('model_deep','?').split('/')[-1]}</code>\n"
            f"Initialized: {'✅' if status['initialized'] else '❌ Add nvidia_api_key to settings.yaml'}\n\n"
            f"<b>Today</b>\n"
            f"Calls today: {status.get('calls_fast_today',0)} fast + {status.get('calls_deep_today',0)} deep = {status.get('calls_total_today',0)}/50 free limit\n"
            f"Avg latency: {status['avg_latency_ms']}ms\n\n"
            f"<b>Diagnostics (last hour)</b>\n"
            f"Scans: {diag_stats['scan_count']}\n"
            f"Signals generated: {diag_stats['signals_generated']}\n"
            f"Signals published: {diag_stats['signals_published']}\n"
            f"Signal deaths: {diag_stats['death_count_1h']}\n"
            f"Top kill: {diag_stats['top_kill_reason']}\n"
            f"Error rate: {diag_stats['error_rate_per_min']}/min\n\n"
            f"Pending approvals: {diag_stats['pending_approvals']}\n"
            f"Active overrides: {diag_stats['active_overrides']}\n"
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🟢 Full ON",  callback_data="ai_mode_full"),
                InlineKeyboardButton("🟡 Diagnosis", callback_data="ai_mode_diagnosis"),
                InlineKeyboardButton("🔴 OFF",       callback_data="ai_mode_off"),
            ],
            [
                InlineKeyboardButton("📋 Report",    callback_data="ai_report"),
                InlineKeyboardButton("⏳ Approvals", callback_data="ai_approvals"),
                InlineKeyboardButton("📜 History",   callback_data="ai_history"),
            ],
            [
                InlineKeyboardButton("🔬 Audit (Fast)", callback_data="audit_fast"),
                InlineKeyboardButton("🧠 Audit Deep",   callback_data="audit_deep"),
                InlineKeyboardButton("📊 Audit Status", callback_data="audit_status"),
            ],
        ])
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

    async def send_approval_request(self, approval):
        """Send a self-healing approval request to Telegram with YES/NO buttons."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(
            approval.risk_level, "⚪"
        )
        # HTML-escape all dynamic fields to prevent parse errors
        # (values may contain <, >, & characters e.g. "1.0 < x < 2.0")
        from html import escape as _he
        # Option B — hint at auto-apply countdown for LOW risk.
        _auto_line = ""
        try:
            _auto_at = float(getattr(approval, 'auto_apply_at', 0) or 0)
            if _auto_at > 0:
                import time as _t
                _mins = max(0, int((_auto_at - _t.time()) / 60))
                _auto_line = (
                    f"\n<i>🤖 Auto-applies in ~{_mins} min unless you tap NO "
                    f"(LOW risk + reversible)</i>"
                )
        except Exception:
            pass
        text = (
            f"⚠️ <b>Self-Healing Approval Required</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>{_he(str(approval.description))}</b>\n\n"
            f"From: <code>{_he(str(approval.old_value))}</code>\n"
            f"To:   <code>{_he(str(approval.new_value))}</code>\n\n"
            f"Reason: {_he(str(approval.reason))}\n\n"
            f"Risk: {risk_emoji} {_he(str(approval.risk_level))}\n"
            f"Impact: {_he(str(approval.estimated_impact))}\n"
            f"Reversible: ✅ UNDO button always available"
            f"{_auto_line}\n"
            f"\n<i>ID: {_he(str(approval.approval_id))}</i>"
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "✅ YES — Apply Now",
                    callback_data=f"approve_yes_{approval.approval_id}"
                ),
                InlineKeyboardButton(
                    "❌ NO — Ignore",
                    callback_data=f"approve_no_{approval.approval_id}"
                ),
            ],
            [
                InlineKeyboardButton(
                    "⏰ Ask in 1 hour",
                    callback_data=f"approve_snooze_{approval.approval_id}"
                ),
            ],
        ])
        try:
            chat_id = self._admin_ids[0] if self._admin_ids else self._chat_id
            await self.gateway.send_text(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.warning(f"Failed to send approval request: {e}")

        # Push approval to dashboard notification queue
        try:
            from web.app import push_notification
            from html import escape as _he
            push_notification({
                "type":        "approval_required",
                "approval_id": str(approval.approval_id),
                "description": str(approval.description),
                "risk_level":  str(approval.risk_level),
                "reason":      str(approval.reason)[:100],
            })
        except Exception as _e:
            logger.debug("Failed to push approval notification: %s", _e)

