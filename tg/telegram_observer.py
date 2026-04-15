"""
TitanBot Pro — Telegram Observer
==================================
A thin wrapper around every bot.send_message() call.

Captures:
  - Message length (flags near-limit messages)
  - HTML parse errors (broken tags, special chars)
  - Send failures (network, rate limits, chat errors)
  - Message type classification (signal, alert, digest, etc.)
  - Delivery latency

All observations feed directly into diagnostic_engine for the
6-hour Telegram Health Report section.

Usage:
  from tg.telegram_observer import TelegramObserver
  observed = TelegramObserver(bot_instance)
  await observed.send(chat_id, text, **kwargs)

  # Or wrap the bot._bot directly at startup:
  telegram_observer.attach(telegram_bot._bot)
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Message type classifier ───────────────────────────────────

_TYPE_KEYWORDS = {
    "signal":      ["🟢", "🔴", "Grade", "Entry:", "Zone:", "TP1:", "TP2:", "R/R:"],
    "alert":       ["🚨", "Circuit breaker", "⚠️ Strategy Health", "Network"],
    "digest":      ["Market Digest", "Alt Season", "Fear", "Regime"],
    "watchlist":   ["👀 Stalker", "Watchlist", "breakdown alert", "breakout alert"],
    "whale":       ["🐋", "Whale"],
    "heartbeat":   ["HEARTBEAT", "Uptime", "Scans"],
    "approval":    ["Self-Healing Approval", "approve_yes", "approve_no"],
    "diagnostic":  ["Diagnostic Report", "Performance Report", "Narrative Quality"],
    "system":      ["TitanBot Pro Starting", "ALL SYSTEMS GO", "Restarted"],
    "ai":          ["AI Analyst", "🤖"],
}


def _classify_message(text: str) -> str:
    """Classify a Telegram message by its content."""
    for msg_type, keywords in _TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return msg_type
    return "other"


# ── HTML validation ───────────────────────────────────────────

def _check_html_balance(text: str) -> Optional[str]:
    """
    Quick check for unbalanced HTML tags.
    Returns error description or None if OK.
    Common issues: unclosed <code>, <b>, <i> tags.
    """
    tags = ["b", "i", "code", "pre", "a", "s", "u"]
    for tag in tags:
        opens = text.count(f"<{tag}>") + text.count(f"<{tag} ")
        closes = text.count(f"</{tag}>")
        if opens != closes:
            return f"Unbalanced <{tag}> tag: {opens} open, {closes} close"
    return None


# ── Observer ─────────────────────────────────────────────────

class TelegramObserver:
    """
    Wraps Telegram bot sends to observe formatting and delivery health.
    Feeds all observations to diagnostic_engine.
    """

    def __init__(self):
        self._attached = False
        self._total_sent = 0
        self._total_failed = 0

    def attach(self, bot_instance):
        """
        Monkey-patch the bot's send_message to route through observer.
        Called once at startup after bot is initialized.
        """
        if self._attached:
            return
        _original_send = bot_instance.send_message

        async def _observed_send(chat_id, text, **kwargs):
            return await self._intercept_send(
                _original_send, chat_id, text, **kwargs
            )

        bot_instance.send_message = _observed_send
        self._attached = True
        logger.info("📊 Telegram observer attached")

    async def _intercept_send(self, original_fn, chat_id, text, **kwargs):
        """Intercept, observe, then forward every send_message call."""
        t0 = time.time()
        msg_type = _classify_message(text or "")
        send_ok = True
        error_str = ""
        result = None
        try:
            result = await original_fn(chat_id, text, **kwargs)
            self._total_sent += 1
        except Exception as e:
            send_ok = False
            error_str = str(e)[:200]
            self._total_failed += 1
            logger.error(f"📊 Telegram send failed ({msg_type}): {error_str}")
            raise  # Re-raise so caller handles it

        finally:
            self.record(
                action="send",
                text=text,
                parse_mode=kwargs.get("parse_mode", ""),
                has_keyboard=kwargs.get("reply_markup") is not None,
                send_ok=send_ok,
                error=error_str,
                latency_ms=(time.time() - t0) * 1000,
            )

        return result

    def record(
        self,
        *,
        action: str,
        text: str,
        parse_mode: str = "",
        has_keyboard: bool = False,
        send_ok: bool = True,
        error: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        char_count = len(text) if text else 0
        msg_type = _classify_message(text or "")

        html_error = None
        if parse_mode == "HTML" and text:
            html_error = _check_html_balance(text)
            if html_error:
                logger.warning(f"📊 HTML issue in {msg_type} {action}: {html_error}")

        if char_count > 4000:
            logger.warning(
                f"📊 Large Telegram {action}: {msg_type} = {char_count} chars "
                f"(Telegram limit: 4096)"
            )

        try:
            from core.diagnostic_engine import diagnostic_engine
            diagnostic_engine.record_telegram_message(
                message_type=msg_type,
                char_count=char_count,
                parse_mode=parse_mode,
                has_keyboard=has_keyboard,
                send_ok=send_ok,
                error=error or (html_error or ""),
            )
        except Exception:
            pass

        try:
            from core.extended_health import ext_health
            ext_health.record_telegram_latency(latency_ms, success=send_ok)
        except Exception:
            pass

    def get_stats(self) -> dict:
        return {
            "total_sent": self._total_sent,
            "total_failed": self._total_failed,
            "attached": self._attached,
        }


# ── Singleton ─────────────────────────────────────────────────
telegram_observer = TelegramObserver()
