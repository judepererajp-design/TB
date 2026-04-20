"""
TitanBot Pro — Telegram Bot (v4 — Interactive Trading Terminal)
===============================================================
Channel architecture:
  SIGNALS channel  — executable setups + trade lifecycle only
  MARKET  channel  — scheduled market digests (every 4-6 h)
  ADMIN   channel  — all system chatter, debug, watchlist forming

Push notifications ONLY for:
  ✅ New executable setup
  ✅ Entry triggered
  ✅ TP1 hit
  ✅ SL hit
  ✅ Trade closed

Everything else: silent edits.

Message lifecycle per trade (ONE message, always edited):
  SETUP → ENTRY_ACTIVE → TRADE_ACTIVE → TP1_HIT → CLOSED

Commands:
  /start       — Welcome + status
  /status      — Live dashboard
  /signals     — Last 10 signals (24h)
  /watchlist   — Symbol buttons → tap for snapshot
  /market      — Market digest
  /performance — Stats
  /pause       — Pause signals
  /resume      — Resume
  /config      — Settings
  /scan        — Force scan
  /report      — 30-day report
  /ranking     — Strategy ranking
  /panel       — Persistent control panel
  /help        — Command list
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Coroutine, Dict, Optional

import psutil
from telegram import Bot, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter, Conflict, NetworkError, TimedOut
# PTB v20+: FloodControl was merged into RetryAfter — alias for compatibility
try:
    from telegram.error import FloodControl
except ImportError:
    FloodControl = RetryAfter  # Same class in PTB v20+
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from analyzers.altcoin_rotation import rotation_tracker
from analyzers.regime import regime_analyzer
from config.loader import cfg
from data.database import db
from tg.formatter import formatter
from tg.gateway import TelegramGateway
from tg.health import BotHealthService
from tg.keyboards import keyboards, control_panel
from tg.search import TelegramSearchService
from tg.session_state import TelegramSessionState
from utils.formatting import fmt_price

logger = logging.getLogger(__name__)
from tg.commands import CommandsMixin
from tg.callbacks import CallbacksMixin
from tg.session_state import TradeState
from strategies.base import direction_str


class SignalRecord:
    """Tracks one signal's Telegram message and lifecycle state."""
    __slots__ = ('message_id', 'state', 'symbol', 'direction',
                 'signal_id', 'scored', 'alpha_score', 'prob_estimate', 'sizing',
                 'created_at', 'card_text', 'grade')

    def __init__(self, message_id: int, signal_id: int, symbol: str, direction: str,
                 scored=None, alpha_score=None, prob_estimate=None, sizing=None,
                 card_text: str = "", grade: str = "A", state: TradeState = None):
        self.message_id    = message_id
        self.signal_id     = signal_id
        self.symbol        = symbol
        self.direction     = direction
        self.state         = state if state is not None else TradeState.SETUP
        self.scored        = scored
        self.alpha_score   = alpha_score
        self.prob_estimate = prob_estimate
        self.sizing        = sizing
        self.created_at    = time.time()
        self.card_text     = card_text   # current card body (for Back button restore)
        self.grade         = grade       # A+/A/B+/B/C (for keyboard selection)


class TelegramBot(CommandsMixin, CallbacksMixin):
    """
    Main Telegram bot — Interactive Trading Terminal.
    V10: Includes message throttling to prevent Telegram rate limit issues.
    One message per trade, always edited in place.
    System chatter routed to admin channel only.
    """

    def __init__(self):
        tg = cfg.telegram
        self._token      = tg.get("bot_token", "")
        self._chat_id    = str(tg.get("chat_id", ""))          # SIGNALS channel
        self._market_id  = str(tg.get("market_chat_id",        # MARKET channel
                                       tg.get("chat_id", "")))
        self._admin_id   = str(tg.get("admin_chat_id",          # ADMIN channel
                                       tg.get("chat_id", "")))
        self._admin_ids  = tg.get("admin_ids", [])

        self.paused       = False
        self._start_time  = time.time()
        self._app: Optional[Application] = None
        self._bot: Optional[Bot]         = None
        self.gateway = TelegramGateway(self._safe_send)
        self.session_state = TelegramSessionState()
        self.search_service = TelegramSearchService()
        self.health_service = BotHealthService(self._runtime_snapshot)

        # message_id → SignalRecord
        self._signals: Dict[int, SignalRecord] = {}
        # signal_id  → SignalRecord (lookup by signal_id)
        self._by_signal_id: Dict[int, SignalRecord] = {}

        # Session strategy overrides: strategy_name → bool (None = use config)
        self._strategy_overrides: Dict[str, bool] = {}

        # Real universe size — set by engine after scanner.build_universe()
        self._universe_size: int = 0

        # Control panel
        self._panel_message_id: Optional[int] = None

        # Multi-step state for text input
        self._awaiting_entry:   dict = {}
        self._awaiting_outcome: dict = {}
        self._awaiting_edit:    dict = {}

        # Counters (for admin panel)
        self._scan_count   = 0
        self._signal_count = 0
        # V10: Message throttle — prevent Telegram rate limit (30 msg/sec)
        self._last_msg_time: float = 0
        self._msg_throttle_secs: float = 1.5  # Min gap between signal messages
        # FIX: guard against concurrent force-scan tasks from rapid /scan button presses
        self._force_scan_task: Optional[asyncio.Task] = None
        # Thesis-check tasks keyed by signal_id — prevents duplicate concurrent checks
        self._check_signal_tasks: Dict[int, asyncio.Task] = {}

        # External callbacks
        self.on_force_scan: Optional[Callable[..., Coroutine]] = None
        self.on_pause:      Optional[Callable[..., Coroutine]] = None
        self.on_resume:     Optional[Callable[..., Coroutine]] = None

    async def _safe_send(self, coro, max_retries: int = 3):
        """
        E1: FloodControl / RetryAfter-aware wrapper for all Telegram API calls.
        Catches Telegram's rate-limit exception and waits the specified retry_after
        period before retrying. Falls back to TelegramError on other failures.
        """
        import asyncio as _aio
        for attempt in range(max_retries):
            try:
                return await coro
            except (RetryAfter, FloodControl) as e:
                wait_secs = getattr(e, 'retry_after', 30) + 1
                logger.warning(
                    f"Telegram FloodControl — waiting {wait_secs}s "
                    f"(attempt {attempt+1}/{max_retries})"
                )
                await _aio.sleep(wait_secs)
            except TelegramError as e:
                logger.error(f"Telegram API error: {e}")
                return None
        return None

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    async def initialize(self):
        if not self._token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")

        self._app = Application.builder().token(self._token).build()
        self._bot = self._app.bot
        self.gateway.bind(self._bot)

        command_map = {
            "start":       self._cmd_start,
            "status":      self._cmd_status,
            "signals":     self._cmd_signals,
            "search":      self._cmd_search,
            "health":      self._cmd_health,
            "watchlist":   self._cmd_watchlist,
            "market":      self._cmd_market,
            "performance": self._cmd_performance,
            "pause":       self._cmd_pause,
            "resume":      self._cmd_resume,
            "config":      self._cmd_config,
            "scan":        self._cmd_scan,
            "report":      self._cmd_report,
            "ranking":     self._cmd_ranking,
            "panel":       self._cmd_panel,
            "upgrades":    self._cmd_upgrades,
            "explain":     self._cmd_explain,    # Fix U3
            "help":        self._cmd_help,
            "quick":       self._cmd_quick,      # V17: one-liner status
            "reload":      self._cmd_reload,     # EC8: hot-reload config
            "replay":      self._cmd_replay,     # I7: replay historical date
            "ai":          self._cmd_ai,         # V20: AI analyst control
            "whales":      self._cmd_whales,     # v21: HyperTracker smart money
            "sentinel":    self._cmd_sentinel,   # v209: Sentinel intelligence features
            "news":        self._cmd_news,       # v21: RSS news feed
            "trending":    self._cmd_trending,   # v21: CoinGecko trending coins
        }

        for cmd, handler in command_map.items():
            self._app.add_handler(CommandHandler(cmd, handler))

        self._app.add_handler(CallbackQueryHandler(self._handle_callback))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )

        # Error handler: catches all unhandled PTB errors including polling Conflicts
        # Conflict = another bot instance is running (previous session still active on Telegram)
        # This recovers automatically once the old session expires (~30s)
        self._app.add_error_handler(self._handle_ptb_error)

        logger.info("Telegram handlers registered")

        # Restore card states from previous session
        try:
            from data.database import db as _db
            saved_states = await _db.load_active_card_states()
            restored = 0
            for cs in saved_states:
                sid = cs['signal_id']
                if sid not in self._by_signal_id:
                    # Reconstruct minimal SignalRecord for button routing
                    rec = SignalRecord(
                        signal_id=sid,
                        symbol=cs.get('symbol', ''),
                        direction=cs.get('direction', 'LONG'),
                        grade=cs.get('grade', 'B'),
                        message_id=cs.get('message_id'),
                        card_text='',
                        state=TradeState.PENDING,
                        scored=None,
                        alpha_score=None,
                        prob_estimate=None,
                        sizing=None,
                    )
                    self._by_signal_id[sid] = rec
                    restored += 1
            if restored:
                logger.info(f"✅ Restored {restored} card states from previous session")
        except Exception as _e:
            logger.debug(f"Card state restore failed (non-fatal): {_e}")

        # Wire proactive alert callback
        try:
            from signals.proactive_alerts import proactive_alerts as _pa
            async def _on_proactive_alert(signal_id: int, text: str):
                rec = self._by_signal_id.get(signal_id)
                if not rec:
                    return
                try:
                    await self.gateway.send_text(
                        chat_id=self._chat_id,
                        text=text,
                        parse_mode=ParseMode.HTML,
                        reply_to_message_id=rec.message_id,
                        disable_web_page_preview=True,
                    )
                except Exception as _e:
                    logger.debug(f"proactive alert send failed: {_e}")
            _pa.on_alert = _on_proactive_alert
        except Exception as _e:
            logger.debug(f"ProactiveAlert callback wire failed: {_e}")

    async def start(self):
        if not self._app:
            await self.initialize()

        await self._app.initialize()
        await self._app.start()

        # FIX: Call deleteWebhook before polling to clear any stale session from a
        # previous run. Without this, restarting the bot within 30s of the last
        # stop causes: "Conflict: terminated by other getUpdates request".
        # drop_pending_updates=True also clears queued messages from the dead session.
        try:
            await self._bot.delete_webhook(drop_pending_updates=True)
            logger.debug("Telegram: cleared stale webhook/session")
        except Exception as _wh_err:
            logger.debug(f"Telegram: deleteWebhook skipped ({_wh_err})")

        await self._app.updater.start_polling(drop_pending_updates=True)

        # Startup message is sent by the engine after scanner.build_universe()
        # so we have the real symbol count. Do NOT call _send_startup_message() here.

        # FIX: store task references so stop() can cleanly cancel them and exceptions
        # are visible. Unrooted create_task on infinite loops means if _panel_live_updater
        # raises, the exception is swallowed and the panel silently goes dark forever.
        self._panel_task   = asyncio.create_task(self._panel_live_updater())
        self._digest_task  = asyncio.create_task(self._market_digest_scheduler())

        logger.info("✅ Telegram bot polling started")


    async def _handle_ptb_error(self, update, context) -> None:
        """
        PTB application-level error handler.
        Catches Conflict errors (another bot instance) and logs them cleanly
        instead of filling the console with stack traces.
        """
        err = context.error

        if isinstance(err, Conflict):
            # Another instance is polling — this self-resolves in ~30s
            # Log once at WARNING level, not ERROR with full traceback
            logger.warning(
                "⚠️  Telegram Conflict: another bot instance was recently running. "
                "Auto-recovering — this clears in ~30 seconds. "
                "If it persists: check for other TitanBot processes (kill them) "
                "or delete titanbot.lock and restart."
            )
        elif isinstance(err, TimedOut):
            logger.debug(f"Telegram timeout (non-fatal, will retry): {err}")
        elif isinstance(err, NetworkError):
            logger.warning(f"Telegram network error (will retry): {err}")
        else:
            logger.warning(f"Telegram error in update handler: {type(err).__name__}: {err}")

    async def stop(self):
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        logger.info("Telegram bot stopped")

    @property
    def signals_chat_id(self) -> str:
        return self._chat_id

    @property
    def market_chat_id(self) -> str:
        return self._market_id

    @property
    def admin_chat_id(self) -> str:
        return self._admin_id or self._chat_id

    def has_signal_channel(self) -> bool:
        return bool(self._chat_id)

    async def send_text(self, chat_id: str, text: str, **kwargs):
        kwargs.setdefault("parse_mode", ParseMode.HTML)
        return await self.gateway.send_text(chat_id=chat_id, text=text, **kwargs)

    async def send_signals_text(self, text: str, **kwargs):
        return await self.send_text(self._chat_id, text, **kwargs)

    async def send_market_text(self, text: str, **kwargs):
        return await self.send_text(self._market_id, text, **kwargs)

    async def send_admin_text(self, text: str, **kwargs):
        return await self.send_text(self.admin_chat_id, text, **kwargs)

    async def edit_text(self, chat_id: str, message_id: int, text: str, **kwargs):
        kwargs.setdefault("parse_mode", ParseMode.HTML)
        return await self.gateway.edit_text(chat_id=chat_id, message_id=message_id, text=text, **kwargs)

    async def edit_signals_text(self, message_id: int, text: str, **kwargs):
        return await self.edit_text(self._chat_id, message_id, text, **kwargs)

    async def edit_reply_markup(self, chat_id: str, message_id: int, reply_markup=None):
        return await self.gateway.edit_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)

    async def delete_message(self, chat_id: str, message_id: int):
        return await self.gateway.delete_message(chat_id=chat_id, message_id=message_id)

    # ─────────────────────────────────────────────────────────
    # Background tasks
    # ─────────────────────────────────────────────────────────

    async def _panel_live_updater(self):
        """Live admin panel — edits same message every 30s."""
        while True:
            try:
                await asyncio.sleep(30)
                if not self._panel_message_id:
                    continue

                uptime  = int(time.time() - self._start_time)
                h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60
                uptime_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
                regime  = regime_analyzer.regime.value if regime_analyzer.regime else "UNKNOWN"
                session = getattr(regime_analyzer.session, 'value', 'UNKNOWN') if hasattr(regime_analyzer, "session") and regime_analyzer.session else "—"
                cpu     = psutil.cpu_percent()
                mem     = psutil.virtual_memory().percent

                try:
                    from core.engine import engine as _eng
                    scans  = _eng._scan_count
                    sigs   = _eng._signal_count
                except Exception:
                    scans = sigs = 0

                sig_rate = round(sigs / max(uptime / 60, 1), 3)
                regime_opp = {"BULL_TREND":2,"BEAR_TREND":2,"VOLATILE":1,"CHOPPY":0,"LOW_VOL":1}.get(regime, 1)
                opp = ("🔥 HIGH"   if sig_rate > 0.1 or regime_opp == 2 else
                       "⚡ MEDIUM" if sig_rate > 0.02 or regime_opp == 1 else
                       "🌙 LOW")
                regime_emoji = {"BULL_TREND":"📈","BEAR_TREND":"📉","CHOPPY":"〰️","VOLATILE":"⚡","LOW_VOL":"😴"}.get(regime,"🔮")
                status_icon = "⏸" if self.paused else "▶️"

                text = (
                    f"⚡ <b>TitanBot Pro — Admin Panel</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"🕐 Uptime    <b>{uptime_str}</b>\n"
                    f"{regime_emoji} Regime    <b>{regime}</b>\n"
                    f"📅 Session   <b>{session}</b>\n"
                    f"{status_icon} Signals   <b>{'Paused' if self.paused else 'Active'}</b>\n\n"
                    f"🔍 Scans     <b>{scans:,}</b>\n"
                    f"📊 Signals   <b>{sigs}</b>\n"
                    f"🎯 Opp       <b>{opp}</b>\n\n"
                    f"💻 CPU       <b>{cpu:.0f}%</b>\n"
                    f"🧩 Memory    <b>{mem:.0f}%</b>\n\n"
                    f"<i>Updates every 30s</i>"
                )

                from tg.keyboards import control_panel
                await self.gateway.edit_text_quiet(
                    chat_id=self._admin_id,
                    message_id=self._panel_message_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=control_panel(paused=self.paused),
                )
            except Exception as e:
                logger.debug(f"Panel updater skipped: {e}")

    async def _market_digest_scheduler(self):
        """Send market digest to MARKET channel every 4-6 hours."""
        DIGEST_INTERVAL = 4 * 3600  # 4 hours
        await asyncio.sleep(300)  # Wait 5 min after startup before first digest
        while True:
            try:
                # Force regime refresh so Fear & Greed is current when digest fires
                try:
                    from analyzers.regime import regime_analyzer as _ra
                    await _ra.force_update()
                except Exception as _e:
                    logger.debug("Failed to force-update regime analyzer: %s", _e)
                text = formatter.format_market_digest()
                await self.gateway.send_text_quiet(
                    chat_id=self._market_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    disable_notification=True,
                )
                logger.info("Market digest sent")
            except Exception as e:
                logger.error(f"Market digest failed: {e}")
            await asyncio.sleep(DIGEST_INTERVAL)

    # ─────────────────────────────────────────────────────────
    # Outbound: Publishing signals and lifecycle updates
    # ─────────────────────────────────────────────────────────

    async def publish_signal(self, scored, signal_id: int,
                              alpha_score=None, prob_estimate=None,
                              sizing=None, confluence=None,
                              current_price: float = 0.0) -> Optional[int]:
        """
        Send new executable setup to SIGNALS channel.
        Stores SignalRecord for lifecycle editing.
        Returns message_id or None.
        """
        if self.paused:
            logger.debug("Bot paused — signal not sent")
            return None
        if not cfg.telegram.get("send_signals", True):
            return None

        try:
            sig = scored.base_signal
            symbol = sig.symbol
            tf = sig.timeframe
            direction = direction_str(sig)
            # PHASE 3 AUDIT FIX (P3-4): use alpha grade (execution grade) for
            # keyboard buttons and internal records, matching the formatter card.
            grade = (alpha_score.grade if alpha_score and hasattr(alpha_score, 'grade')
                     else scored.grade) or "A"

            text = formatter.format_signal(
                scored, signal_id,
                alpha_score=alpha_score,
                prob_estimate=prob_estimate,
                sizing=sizing,
                confluence=confluence,
                current_price=current_price,
            )
            kb = keyboards.signal_card(signal_id, symbol, grade)

            # V10: Throttle signal messages to avoid Telegram rate limits
            _elapsed = time.time() - self._last_msg_time
            if _elapsed < self._msg_throttle_secs:
                await asyncio.sleep(self._msg_throttle_secs - _elapsed)

            sent = await self.gateway.send_text(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
                disable_web_page_preview=True,
            )
            if not sent:
                logger.error("publish_signal: _safe_send returned None (FloodControl or error)")
                return None
            self._last_msg_time = time.time()

            # Store record
            rec = SignalRecord(
                message_id=sent.message_id,
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
                scored=scored,
                alpha_score=alpha_score,
                prob_estimate=prob_estimate,
                sizing=sizing,
                card_text=text,
                grade=grade,
            )
            self._signals[sent.message_id]   = rec
            self._by_signal_id[signal_id]    = rec
            # Persist card state for restart recovery
            try:
                from data.database import db as _db
                asyncio.create_task(_db.save_card_state(
                    signal_id=signal_id,
                    message_id=sent.message_id,
                    chat_id=self._chat_id,
                    symbol=rec.symbol,
                    direction=rec.direction,
                    grade=rec.grade,
                    state="PENDING", confirmed=False,
                ))
            except Exception as e:
                logger.warning("Card state save failed for signal %s: %s", signal_id, e)
            self._signal_count += 1

            logger.info(
                f"Signal sent: {symbol} {direction} "
                f"conf={scored.final_confidence:.0f} grade={grade} msg_id={sent.message_id}"
            )
            return sent.message_id

        except TelegramError as e:
            logger.error(f"publish_signal failed: {e}")
            return None

    async def update_signal_entry(self, signal_id: int, fill_price: float):
        """
        Stage 2: Entry triggered. Edit message in place.
        Keeps all levels visible — scored passed to formatter.
        """
        rec = self._by_signal_id.get(signal_id)
        if not rec:
            return
        try:
            rec.state = TradeState.ENTRY_ACTIVE
            text = formatter.format_entry_active(
                rec.symbol, rec.direction, fill_price, signal_id,
                scored=rec.scored
            )
            rec.card_text = text
            kb = keyboards.signal_in_trade(signal_id, rec.symbol)
            await self.gateway.edit_text(
                chat_id=self._chat_id,
                message_id=rec.message_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
        except TelegramError as e:
            logger.warning(f"update_signal_entry failed: {e}")

    async def update_signal_pnl(self, signal_id: int, pnl_pct: float):
        """
        Stage 3: Trade active. Silent edit (no push notification).
        Keeps all levels visible — scored passed to formatter.
        """
        rec = self._by_signal_id.get(signal_id)
        if not rec or rec.state not in (TradeState.ENTRY_ACTIVE, TradeState.TRADE_ACTIVE):
            return
        try:
            rec.state = TradeState.TRADE_ACTIVE
            text = formatter.format_trade_active(
                rec.symbol, rec.direction, pnl_pct, signal_id,
                scored=rec.scored
            )
            rec.card_text = text
            kb = keyboards.signal_in_trade(signal_id, rec.symbol)
            await self.gateway.edit_text(
                chat_id=self._chat_id,
                message_id=rec.message_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
        except TelegramError as e:
            logger.debug(f"update_signal_pnl skipped: {e}")

    async def update_signal_tp1(self, signal_id: int, price: float, be_stop: float):
        """
        Stage 4: TP1 hit. Edit + push notification.
        Keeps remaining levels visible in card body.
        """
        rec = self._by_signal_id.get(signal_id)
        if not rec:
            return
        try:
            rec.state = TradeState.TP1_HIT
            text = formatter.format_tp1_hit(
                rec.symbol, rec.direction, price, be_stop, signal_id,
                scored=rec.scored
            )
            rec.card_text = text
            kb = keyboards.signal_in_trade(signal_id, rec.symbol)
            await self.gateway.edit_text(
                chat_id=self._chat_id,
                message_id=rec.message_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
            await self.gateway.send_text(
                chat_id=self._chat_id,
                text=f"🎯 TP1 hit — <b>{rec.symbol}</b>  SL → BE",
                parse_mode=ParseMode.HTML,
                reply_to_message_id=rec.message_id,
                disable_web_page_preview=True,
            )
        except TelegramError as e:
            logger.warning(f"update_signal_tp1 failed: {e}")

    async def update_signal_closed(self, signal_id: int, result_pct: float,
                                    duration_h: float, grade: str = "A"):
        """
        Stage 5: Trade closed. Edit + push notification.
        """
        rec = self._by_signal_id.get(signal_id)
        if not rec:
            return
        try:
            rec.state = TradeState.CLOSED
            text = formatter.format_trade_closed(
                rec.symbol, rec.direction, result_pct, duration_h, grade, signal_id
            )
            rec.card_text = text
            kb = keyboards.signal_closed(signal_id)
            await self.gateway.edit_text(
                chat_id=self._chat_id,
                message_id=rec.message_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
            result_emoji = "✅" if result_pct > 0 else ("🔴" if result_pct < 0 else "➡️")
            await self.gateway.send_text(
                chat_id=self._chat_id,
                text=f"{result_emoji} Closed — <b>{rec.symbol}</b>  {result_pct:+.2f}%",
                parse_mode=ParseMode.HTML,
                reply_to_message_id=rec.message_id,
                disable_web_page_preview=True,
            )
        except TelegramError as e:
            logger.warning(f"update_signal_closed failed: {e}")

    async def update_signal_sl_hit(self, signal_id: int, sl_price: float):
        """SL hit — edit + push."""
        rec = self._by_signal_id.get(signal_id)
        if not rec:
            return
        try:
            rec.state = TradeState.CLOSED
            text = formatter.format_sl_hit(rec.symbol, rec.direction, sl_price, signal_id)
            kb = keyboards.signal_closed(signal_id)
            await self.gateway.edit_text(
                chat_id=self._chat_id,
                message_id=rec.message_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
            await self.gateway.send_text(
                chat_id=self._chat_id,
                text=f"🔴 SL hit — <b>{rec.symbol}</b>  <code>{fmt_price(sl_price)}</code>",
                parse_mode=ParseMode.HTML,
                reply_to_message_id=rec.message_id,
                disable_web_page_preview=True,
            )
        except TelegramError as e:
            logger.warning(f"update_signal_sl_hit failed: {e}")

    async def update_signal_invalidated(self, signal_id: int, reason: str):
        """Signal invalidated — silent edit (no push for pre-entry invalidation)."""
        rec = self._by_signal_id.get(signal_id)
        if not rec or rec.state != TradeState.SETUP:
            return
        try:
            rec.state = TradeState.CLOSED
            text = formatter.format_signal_invalidated(
                rec.symbol, rec.direction, reason, signal_id
            )
            await self.gateway.edit_text(
                chat_id=self._chat_id,
                message_id=rec.message_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
        except TelegramError as e:
            logger.debug(f"update_signal_invalidated skipped: {e}")

    async def update_signal_expired(self, signal_id: int, candles: int):
        """Signal expired — silent edit."""
        rec = self._by_signal_id.get(signal_id)
        if not rec or rec.state != TradeState.SETUP:
            return
        try:
            rec.state = TradeState.CLOSED
            text = formatter.format_signal_expired(
                rec.symbol, rec.direction, candles, signal_id
            )
            await self.gateway.edit_text(
                chat_id=self._chat_id,
                message_id=rec.message_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
        except TelegramError as e:
            logger.debug(f"update_signal_expired skipped: {e}")

    # ── Admin channel helpers ──────────────────────────────────────────

    async def _admin(self, text: str, notify: bool = False):
        """Send message to admin channel. No push by default."""
        await self.gateway.send_text_quiet(
            chat_id=self._admin_id,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_notification=not notify,
            disable_web_page_preview=True,
        )

    async def send_watchlist_alert(self, symbol: str, score: float, reasons: list) -> Optional[int]:
        """Watchlist forming → admin channel only."""
        if not cfg.telegram.get("send_watchlist_alerts", True):
            return None
        try:
            text = formatter.format_watchlist_alert(symbol, score, reasons)
            sent = await self.gateway.send_text(
                chat_id=self._admin_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_notification=True,
            )
            return sent.message_id
        except TelegramError as e:
            logger.error(f"send_watchlist_alert failed: {e}")
            return None

    async def send_whale_alert(self, symbol: str, order_usd: float, vol_mult: float):
        """Whale alert → admin channel."""
        if not cfg.telegram.get("send_whale_alerts", True):
            return
        try:
            text = formatter.format_whale_alert(symbol, order_usd, vol_mult)
            await self.gateway.send_text(
                chat_id=self._admin_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_notification=True,
            )
        except TelegramError as e:
            logger.error(f"send_whale_alert failed: {e}")

    async def send_tier_promotion(self, symbol: str, from_tier: int, to_tier: int,
                                   vol_mult: float, vol_24h: float):
        """Tier promotion → admin channel."""
        if not cfg.telegram.get("send_tier_promotions", True):
            return
        try:
            text = formatter.format_tier_promotion(symbol, from_tier, to_tier, vol_mult, vol_24h)
            await self.gateway.send_text(
                chat_id=self._admin_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_notification=True,
            )
        except TelegramError as e:
            logger.error(f"send_tier_promotion failed: {e}")

    async def send_circuit_breaker_alert(self, active: bool, reason: str = "", resume_time: str = ""):
        """Circuit breaker → signals channel (important)."""
        if not cfg.telegram.get("send_circuit_breaker_alerts", True):
            return
        try:
            if active:
                text = formatter.format_circuit_breaker_active(reason, resume_time)
                kb = keyboards.confirm_resume()
            else:
                text = formatter.format_circuit_breaker_cleared()
                kb = None
            await self.gateway.send_text(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
        except TelegramError as e:
            logger.error(f"send_circuit_breaker_alert failed: {e}")

    async def send_error_alert(self, error: str, recovering: bool = True):
        """Error → admin channel only."""
        if not cfg.telegram.get("send_error_alerts", True):
            return
        try:
            text = formatter.format_error(error, recovering)
            await self.gateway.send_text(
                chat_id=self._admin_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_notification=True,
            )
        except TelegramError:
            pass

    async def send_news_risk_alert(self, ctx):
        """
        🚨 Proactive alert when bearish macro/geopolitical news is detected.
        Warns the user about the event and lists any active LONG trades at risk.

        Called via btc_news_intelligence.on_risk_event callback when a
        MACRO_RISK_OFF or other bearish event context is created.
        """
        try:
            # Build event header
            _severity_emoji = {"EXTREME": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪"}.get(
                ctx.severity, "⚪"
            )
            _is_mixed = getattr(ctx, 'is_mixed_signal', False)
            _alert_title = "⚖️ MIXED MACRO SIGNAL" if _is_mixed else "⚠️ BEARISH NEWS ALERT"

            # Show headline age so users know how fresh the news is
            _age_str = ctx.format_headline_age() if hasattr(ctx, 'format_headline_age') else "🕐 Just now"

            lines = [
                f"{_severity_emoji} <b>{_alert_title}</b>",
                f"━━━━━━━━━━━━━━━━━━━━━━━━",
                f"📰 <b>{ctx.headline[:120]}</b>" if ctx.headline else "",
                f"<i>{_age_str}</i>",
                f"",
                f"🏷️ Type: <b>{ctx.event_type.value}</b>",
                f"📊 Severity: <b>{ctx.severity}</b> (confidence: {ctx.confidence:.0%})",
                f"💡 {ctx.explanation}",
            ]
            if ctx.impact_detail:
                lines.append(f"📉 {ctx.impact_detail}")
            lines.append("")

            # Show what the bot is doing
            actions = []
            if ctx.block_longs:
                actions.append("🚫 All new LONG signals <b>BLOCKED</b>")
            if ctx.confidence_mult < 1.0:
                actions.append(f"📉 Signal confidence reduced to <b>{ctx.confidence_mult:.0%}</b>")
            if ctx.reduce_size_mult < 1.0:
                actions.append(f"📐 Position sizes reduced to <b>{ctx.reduce_size_mult:.0%}</b>")
            if actions:
                lines.append("🤖 <b>Bot actions:</b>")
                lines.extend(f"  {a}" for a in actions)
                lines.append("")

            # List active LONG trades at risk
            try:
                from core.execution_engine import execution_engine
                _tracked = execution_engine._tracked
                _at_risk = [
                    s for s in _tracked.values()
                    if getattr(s, 'direction', '') == 'LONG'
                ]
                if _at_risk:
                    lines.append(f"⚠️ <b>Active LONG trades at risk ({len(_at_risk)}):</b>")
                    for sig in _at_risk[:5]:  # Max 5 to avoid message overflow
                        _state = getattr(sig, 'state', '?')
                        _state_str = _state.value if hasattr(_state, 'value') else str(_state)
                        lines.append(
                            f"  • <b>{sig.symbol}</b> — {_state_str} "
                            f"(entry: {sig.entry_low:.4g}–{sig.entry_high:.4g}, "
                            f"SL: {sig.stop_loss:.4g})"
                        )
                    if len(_at_risk) > 5:
                        lines.append(f"  ... and {len(_at_risk) - 5} more")
                    lines.append("")
                    lines.append("💡 <b>Consider:</b> tightening stop-losses, taking partial profit, or closing LONGs manually.")
                else:
                    lines.append("✅ No active LONG trades currently at risk.")
            except Exception:
                pass  # execution_engine not available yet — skip trade listing

            # Expiry info
            _remaining_mins = max(0, (ctx.expires_at - time.time()) / 60)
            lines.append(f"\n⏱️ Protection active for ~{_remaining_mins:.0f} more minutes")

            text = "\n".join(l for l in lines if l is not None)
            await self.gateway.send_text(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_notification=False,  # Sound ON for risk alerts
            )
        except Exception as e:
            logger.error(f"send_news_risk_alert failed: {e}")

    async def send_daily_summary(self):
        """V15: Enhanced daily summary with per-strategy breakdown and paper P&L."""
        if not cfg.telegram.get("send_daily_summary", True):
            return
        try:
            from core.learning_loop import learning_loop
            import time as _ds_time

            ll_stats = learning_loop.get_stats()
            _real = ll_stats.get('real_trades', 0)
            _wr = ll_stats.get('win_rate', 0)
            _total_r = ll_stats.get('total_r', 0)
            _exp = ll_stats.get('expired_count', 0)

            # I10: Paper P&L in bot's own units ($230 position size default)
            _position_size = cfg.risk.get('position_size_usdt', 230)
            _paper_pnl_usd = _total_r * _position_size * cfg.risk.get('risk_pct', 0.01)

            # V10: Trade recap from today's history
            _today_trades = [
                t for t in learning_loop._trade_history
                if t.timestamp > _ds_time.time() - 86400
                and t.outcome not in ("EXPIRED", "INVALIDATED")
            ]
            _recap_lines = []
            for t in _today_trades[-10:]:
                _emoji = "✅" if t.won else "🔴"
                _recap_lines.append(
                    f"  {_emoji} {t.symbol} {t.direction} ({t.strategy}) {t.pnl_r:+.1f}R"
                )
            _recap = "\n".join(_recap_lines) if _recap_lines else "  No completed trades today."

            # I3: Per-strategy breakdown
            from collections import defaultdict as _dd
            _strat_results = _dd(lambda: {'wins': 0, 'losses': 0, 'r': 0.0})
            for t in _today_trades:
                key = t.strategy or 'Unknown'
                _strat_results[key]['wins' if t.won else 'losses'] += 1
                _strat_results[key]['r'] += t.pnl_r
            _strat_lines = []
            for strat, res in sorted(_strat_results.items(), key=lambda x: -x[1]['r']):
                _total_s = res['wins'] + res['losses']
                _wr_s = res['wins'] / _total_s if _total_s > 0 else 0
                _strat_lines.append(
                    f"  {strat}: {res['wins']}W/{res['losses']}L  "
                    f"{_wr_s:.0%} WR  {res['r']:+.1f}R"
                )
            _strat_section = "\n".join(_strat_lines) if _strat_lines else "  No data yet."

            _summary_text = (
                f"📊 <b>Daily Summary</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Trades: <b>{_real}</b> | WR: <b>{_wr:.0%}</b> | Net: <b>{_total_r:+.1f}R</b>\n"
                f"Paper P&L: <b>${_paper_pnl_usd:+.0f}</b> (at ${_position_size} size)\n"
            )
            if _exp > 0:
                _summary_text += f"Expired: {_exp} signals (no entry)\n"
            _summary_text += f"\n<b>By Strategy:</b>\n{_strat_section}\n"
            _summary_text += f"\n<b>Trade Log:</b>\n{_recap}\n"

            # Also include original digest
            watchlist = await db.get_watchlist()
            if watchlist:
                _wl_names = ", ".join(w.get('symbol', '?') for w in watchlist[:5])
                _summary_text += f"\n👁 <b>Watchlist:</b> {_wl_names}"

            await self.gateway.send_text(
                chat_id=self._market_id or self._chat_id,
                text=_summary_text,
                parse_mode=ParseMode.HTML,
                disable_notification=True,
            )
        except Exception as e:
            logger.error(f"send_daily_summary failed: {e}")

    # Backward compat — edit_signal_outcome (called by OutcomeMonitor)
    async def edit_signal_outcome(self, message_id: int, outcome: str,
                                   pnl_r: float, note: str):
        """Map outcome types to lifecycle updates. Looks up by message_id first,
        falls back to scanning _by_signal_id to find the right record."""
        # Primary lookup by message_id
        rec = self._signals.get(message_id)
        # Fallback: scan _by_signal_id if primary misses (e.g. after restart/edit)
        if not rec:
            rec = next(
                (r for r in self._by_signal_id.values() if r.message_id == message_id),
                None,
            )
        if not rec:
            logger.warning(f"edit_signal_outcome: no record for message_id={message_id}")
            return

        # FIX #6: Calculate real duration from record creation time
        import time as _time_outcome
        _created = getattr(rec, 'created_at', 0)
        _duration_h = (_time_outcome.time() - _created) / 3600.0 if _created > 0 else 0.0

        # FIX #10/#12: Display pnl_r directly instead of hardcoded 5x leverage.
        # Show R-multiple as percentage proxy: pnl_r * risk_pct_per_trade.
        # Use a config-aware calculation if available, else show R directly.
        _result_display = pnl_r  # R-multiple (e.g., +1.5R)

        if outcome == "WIN":
            await self.update_signal_closed(rec.signal_id, _result_display, _duration_h, "A")
        elif outcome == "LOSS":
            await self.update_signal_sl_hit(rec.signal_id, 0)
        elif outcome == "BREAKEVEN":
            await self.update_signal_closed(rec.signal_id, 0, _duration_h, "B")
        elif outcome == "EXPIRED":
            await self.update_signal_expired(rec.signal_id, 6)

    # Legacy send_reply (used by some engine callbacks)
    async def send_reply(self, reply_to_message_id: int, text: str) -> Optional[int]:
        try:
            sent = await self.gateway.send_text(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=True,
            )
            return sent.message_id
        except TelegramError as e:
            logger.warning(f"send_reply failed: {e}")
            return None

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled, respecting session overrides from Telegram toggles."""
        key = strategy_name.lower()
        if key in self._strategy_overrides:
            return self._strategy_overrides[key]
        return cfg.is_strategy_enabled(key)

    async def _send_startup_message(self, num_symbols: int = 0):
        """Send startup card. Called by engine after scanner.build_universe() so
        num_symbols reflects the real scanned universe, not the settings.yaml cap."""
        try:
            from analyzers.regime import regime_analyzer as _ra
            from tg.keyboards import main_menu
            version  = cfg.global_cfg.get('version', '1.0.0')
            mode     = "🧪 Paper" if cfg.global_cfg.get("paper_trading") else ("🧪 Test" if cfg.global_cfg.get("test_mode") else "🔴 Live")
            regime   = _ra.regime.value if _ra.regime else "Analyzing…"
            r_emoji  = {"BULL_TREND":"📈","BEAR_TREND":"📉","CHOPPY":"〰️","VOLATILE":"⚡","LOW_VOL":"😴"}.get(regime,"🔮")
            # Use real scanned count if provided; fall back to stored value or config
            universe = num_symbols or self._universe_size or cfg.system.get('max_symbols', 200)
            if num_symbols:
                self._universe_size = num_symbols  # cache for future reference
            # Count enabled strategies dynamically
            _strat_names = ["smc","breakout","reversal","mean_reversion","price_action",
                            "momentum","ichimoku","elliott_wave","funding_arb","wyckoff",
                            "harmonic","geometric","range_scalper"]
            _n_strats = sum(1 for s in _strat_names if cfg.is_strategy_enabled(s))
            text = (
                f"⚡ <b>TitanBot Pro v{version} — Online</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Mode       {mode}\n"
                f"Regime     {r_emoji} <b>{regime}</b>\n"
                f"Universe   {universe} symbols · {_n_strats} strategies\n"
                f"Min conf   {cfg.aggregator.get('min_confidence', 70)}/100\n\n"
                f"<b>Channels</b>\n"
                f"📊 Signals channel  — Executable setups\n"
                f"📡 Market channel   — Digests every 4h\n"
                f"🔧 Admin channel    — System & logs\n\n"
                f"Use the menu or /help for commands."
            )
            await self.gateway.send_text(
                chat_id=self._chat_id, text=text,
                parse_mode=ParseMode.HTML, reply_markup=main_menu(),
            )
        except Exception as e:
            logger.error(f"Startup message failed: {e}")
    # ─────────────────────────────────────────────────────────
    # Command handlers
    # ─────────────────────────────────────────────────────────


    async def publish_context_signal(self, scored, signal_id: int,
                                      up_score: float) -> Optional[int]:
        """
        Send C signal in CONTEXT format (no trade numbers).
        Push notification ON — user needs to be aware, not to act.
        """
        if self.paused:
            return None
        try:
            sig = scored.base_signal
            symbol = sig.symbol
            tf = sig.timeframe
            direction = direction_str(sig)

            text = formatter.format_context_signal(scored, signal_id, up_score)

            # Minimal keyboard — just watchlist and chart, no analysis buttons
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("👀 Watch", callback_data=f"watch:{symbol}"),
                    InlineKeyboardButton("📊 Chart", callback_data=f"chart_sym:{symbol}"),
                ]
            ])

            sent = await self.gateway.send_text(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
                disable_web_page_preview=True,
            )

            # Store minimal record so upgrade can edit this message
            rec = SignalRecord(
                message_id=sent.message_id,
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
                scored=scored,
            )
            self._signals[sent.message_id] = rec
            self._by_signal_id[signal_id]  = rec
            # Persist card state for restart recovery
            try:
                from data.database import db as _db
                asyncio.create_task(_db.save_card_state(
                    signal_id=signal_id,
                    message_id=sent.message_id,
                    chat_id=self._chat_id,
                    symbol=rec.symbol,
                    direction=rec.direction,
                    grade=rec.grade,
                    state="PENDING", confirmed=False,
                ))
            except Exception as e:
                logger.warning("Card state save failed for signal %s: %s", signal_id, e)

            logger.info(f"Context signal sent: {symbol} {direction} UP={up_score:.2f} msg_id={sent.message_id}")
            return sent.message_id

        except TelegramError as e:
            logger.error(f"publish_context_signal failed: {e}")
            return None

    async def send_admin_c_signal(self, scored, signal_id: int, up_score: float):
        """
        Send suppressed C signal to admin channel only. No push notification.
        """
        try:
            text = formatter.format_admin_c_signal(scored, signal_id, up_score)
            await self.gateway.send_text(
                chat_id=self._admin_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_notification=True,
                disable_web_page_preview=True,
            )
        except TelegramError as e:
            logger.debug(f"send_admin_c_signal failed: {e}")

    async def publish_upgrade(self, signal_id: int, new_grade: str, new_scored):
        """
        Signal upgraded from C to B/A/A+.
        1. Send upgrade notification (push)
        2. Follow immediately with full execution card
        """
        rec = self._by_signal_id.get(signal_id)
        old_grade = "C"

        try:
            sig = new_scored.base_signal
            symbol = sig.symbol
            direction = direction_str(sig)

            # Upgrade notification message
            upgrade_text = formatter.format_upgrade_message(
                symbol, direction, old_grade, new_grade, signal_id
            )
            await self.gateway.send_text(
                chat_id=self._chat_id,
                text=upgrade_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )

            # Follow immediately with full execution card
            execution_text = formatter.format_signal(new_scored, signal_id)
            new_grade = new_scored.grade or "A"
            kb = keyboards.signal_card(signal_id, symbol, new_grade)
            sent = await self.gateway.send_text(
                chat_id=self._chat_id,
                text=execution_text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
                disable_web_page_preview=True,
                disable_notification=True,
            )

            # Update tracking record to the new execution message
            new_rec = SignalRecord(
                message_id=sent.message_id,
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
                scored=new_scored,
                card_text=execution_text,
                grade=new_grade,
            )
            self._signals[sent.message_id] = new_rec
            self._by_signal_id[signal_id]  = new_rec

            logger.info(f"Upgrade published: {symbol} C→{new_grade} msg_id={sent.message_id}")

        except TelegramError as e:
            logger.error(f"publish_upgrade failed: {e}")

    # ─────────────────────────────────────────────────────────
    # Text handler (entry prices / setting edits)
    # ─────────────────────────────────────────────────────────

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._auth(update):
            return
        chat_id = str(update.message.chat_id)
        text    = update.message.text.strip()

        pending = self.session_state.peek_input(chat_id)
        if pending and pending.kind == "search":
            from tg.keyboards import search_results_menu
            self.session_state.pop_input(chat_id)
            results = await self.search_service.search(pending.context.get("mode", "any"), text)
            await update.message.reply_text(
                self.search_service.format_results(results),
                parse_mode=ParseMode.HTML,
                reply_markup=search_results_menu(results),
                disable_web_page_preview=True,
            )
            return

        if chat_id in self._awaiting_entry:
            signal_id = self._awaiting_entry.pop(chat_id)
            try:
                price  = float(text.replace("$", "").replace(",", ""))
                signal = await db.get_signal(signal_id)
                if signal:
                    await db.save_outcome({
                        "signal_id":   signal_id,
                        "symbol":      signal.get("symbol"),
                        "direction":   signal.get("direction"),
                        "outcome":     "OPEN",
                        "entry_price": price,
                    })
                await update.message.reply_text(
                    f"✅ Entry at <code>{fmt_price(price)}</code> recorded.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=keyboards.outcome_result(signal_id),
                )
            except ValueError:
                await update.message.reply_text(
                    "⚠️ Invalid price. Send a number, e.g. <code>67420.50</code>",
                    parse_mode=ParseMode.HTML,
                )
            return

        if chat_id in self._awaiting_edit:
            field = self._awaiting_edit.pop(chat_id)
            try:
                value = float(text.replace(",", ""))
                await update.message.reply_text(
                    f"✅ <b>{field}</b> set to <code>{value}</code>\n\n"
                    "⚠️ Session only. Edit <code>settings.yaml</code> to persist.",
                    parse_mode=ParseMode.HTML,
                )
            except ValueError:
                await update.message.reply_text("⚠️ Invalid value. Send a number.")
            return

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _runtime_snapshot(self) -> dict:
        try:
            from core.engine import engine as _eng
            scan_count = _eng._scan_count
            signal_count = _eng._signal_count
        except Exception:
            scan_count = self._scan_count
            signal_count = self._signal_count
        return {
            "paused": self.paused,
            "scan_count": scan_count,
            "signal_count": signal_count,
            "uptime_seconds": max(0, int(time.time() - self._start_time)),
        }

    _REGIME_EMOJI = {
        "BULL_TREND": "📈", "BEAR_TREND": "📉",
        "CHOPPY": "〰️", "VOLATILE": "⚡", "LOW_VOL": "😴",
    }

    def _render_dashboard_text(self) -> str:
        """Single source of truth for the main dashboard card text."""
        try:
            from core.engine import engine as _eng
            scans = _eng._scan_count
            sigs = _eng._signal_count
        except Exception:
            scans = sigs = 0
        uptime_s = int(time.time() - self._start_time)
        h, m = uptime_s // 3600, (uptime_s % 3600) // 60
        uptime_str = f"{h}h {m}m" if h else f"{m}m"
        regime = regime_analyzer.regime.value if regime_analyzer.regime else "Analyzing…"
        r_emoji = self._REGIME_EMOJI.get(regime, "🔮")
        status_icon = "⏸ Paused" if self.paused else "▶️ Active"
        return (
            f"⚡ <b>TitanBot Pro — Dashboard</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🕐 Uptime     <b>{uptime_str}</b>\n"
            f"{r_emoji} Regime     <b>{regime}</b>\n"
            f"🔍 Scans      <b>{scans:,}</b>\n"
            f"📊 Signals    <b>{sigs}</b>\n"
            f"🤖 Status     <b>{status_icon}</b>\n\n"
            f"Select an option:"
        )

    async def _begin_search_prompt(self, chat_id: str, mode: str):
        return self.session_state.begin_input(
            chat_id,
            "search",
            self.search_service.prompt_for(mode),
            mode=mode,
        )

    def _format_signal_lookup_detail(self, signal: dict) -> str:
        outcome = signal.get("outcome") or "PENDING"
        pnl_r = signal.get("pnl_r")
        pnl_str = f"{pnl_r:+.2f}R" if pnl_r not in (None, "") else "—"
        lines = [
            f"🧾 <b>Signal #{signal.get('id')}</b>",
            f"<b>{signal.get('symbol', '?')} {signal.get('direction', '?')}</b> · {signal.get('alpha_grade', '?')}",
            f"Strategy: {signal.get('strategy', '?')}",
            f"Regime: {signal.get('regime', '?')}",
            f"Outcome: <b>{outcome}</b> · P&L: <b>{pnl_str}</b>",
            "",
            f"Entry: <code>{fmt_price(signal.get('entry_low', 0))}</code> → <code>{fmt_price(signal.get('entry_high', 0))}</code>",
            f"SL: <code>{fmt_price(signal.get('stop_loss', 0))}</code>",
            f"TP1: <code>{fmt_price(signal.get('tp1', 0))}</code>",
        ]
        if signal.get("tp2"):
            lines.append(f"TP2: <code>{fmt_price(signal.get('tp2'))}</code>")
        if signal.get("tp3"):
            lines.append(f"TP3: <code>{fmt_price(signal.get('tp3'))}</code>")
        return "\n".join(lines)

    async def _collect_stats(self) -> dict:
        uptime_s   = int(time.time() - self._start_time)
        hours      = uptime_s // 3600
        minutes    = (uptime_s % 3600) // 60
        uptime_str = f"{hours}h {minutes}m"

        try:
            cpu    = psutil.cpu_percent(interval=0.1)
            ram    = psutil.virtual_memory()
            ram_gb = ram.used / 1e9
        except Exception:
            cpu, ram_gb = 0.0, 0.0

        signals_today = await db.get_signals_today()
        watchlist     = await db.get_watchlist()

        return {
            "uptime_str":      uptime_str,
            "cpu_pct":         round(cpu, 1),
            "ram_gb":          round(ram_gb, 2),
            "tier1_count":     cfg.scanning.tier1.get("max_symbols", 80),
            "tier2_count":     cfg.scanning.tier2.get("max_symbols", 80),
            "tier3_count":     cfg.scanning.tier3.get("max_symbols", 40),
            "watchlist_count": len(watchlist),
            "signals_today":   signals_today,
        }

    def _get_config_value(self, field: str) -> str:
        mapping = {
            "account_size":       cfg.risk.get("account_size", 5000),
            "risk_per_trade":     cfg.risk.get("risk_per_trade", 0.005),
            "min_confidence":     cfg.aggregator.get("min_confidence", 72),
            "max_daily_loss_pct": cfg.risk.get("max_daily_loss_pct", 0.03),
            "min_rr":             cfg.risk.get("min_rr", 1.3),
        }
        return str(mapping.get(field, "?"))

    def _auth(self, update: Update) -> bool:
        if not update.message:
            return False
        chat_id = str(update.message.chat_id)
        user_id = update.message.from_user.id
        if chat_id == self._chat_id or chat_id == self._admin_id:
            return True
        if self._admin_ids and user_id in self._admin_ids:
            return True
        logger.warning(f"Unauthorised: user_id={user_id} chat_id={chat_id}")
        return False

    def _auth_query(self, query) -> bool:
        if not self._admin_ids:
            return True
        return query.from_user.id in self._admin_ids

    # ── /ai command ───────────────────────────────────────────────────────


# ── Singleton ──────────────────────────────────────────────
telegram_bot = TelegramBot()
