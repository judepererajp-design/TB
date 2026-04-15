from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable, Optional

from tg.telegram_observer import telegram_observer

logger = logging.getLogger(__name__)


class TelegramGateway:
    """Single Telegram transport surface for sends, edits, deletes, and markup changes.

    All outbound Telegram I/O MUST flow through this gateway so that
    telemetry (observer), rate-limit handling (_safe_send), and error
    logging are applied uniformly.
    """

    def __init__(self, safe_send: Callable[[Awaitable], Awaitable]) -> None:
        self._safe_send = safe_send
        self._bot = None

    def bind(self, bot) -> None:
        self._bot = bot

    def is_ready(self) -> bool:
        return self._bot is not None

    # ── Send ──────────────────────────────────────────────────

    async def send_text(self, chat_id, text: str, **kwargs):
        """Send a new message.  Returns the Message object or None."""
        self._require_bot()
        return await self._run(
            "send",
            self._bot.send_message(chat_id=chat_id, text=text, **kwargs),
            text=text,
            parse_mode=kwargs.get("parse_mode", ""),
            has_keyboard=kwargs.get("reply_markup") is not None,
        )

    # ── Edit ──────────────────────────────────────────────────

    async def edit_text(self, chat_id, message_id: int, text: str, **kwargs):
        """Edit an existing message's text (and optionally its markup)."""
        self._require_bot()
        return await self._run(
            "edit",
            self._bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, **kwargs),
            text=text,
            parse_mode=kwargs.get("parse_mode", ""),
            has_keyboard=kwargs.get("reply_markup") is not None,
        )

    async def edit_reply_markup(self, chat_id, message_id: int, reply_markup=None):
        """Edit only the inline keyboard of an existing message."""
        self._require_bot()
        return await self._run(
            "reply_markup",
            self._bot.edit_message_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=reply_markup),
            text="",
            parse_mode="",
            has_keyboard=reply_markup is not None,
        )

    # ── Delete ────────────────────────────────────────────────

    async def delete_message(self, chat_id, message_id: int):
        self._require_bot()
        return await self._run(
            "delete",
            self._bot.delete_message(chat_id=chat_id, message_id=message_id),
            text="",
            parse_mode="",
            has_keyboard=False,
        )

    # ── Convenience: fire-and-forget send (logs errors instead of raising) ──

    async def send_text_quiet(self, chat_id, text: str, **kwargs):
        """Like send_text but swallows exceptions (for admin/notification messages)."""
        try:
            return await self.send_text(chat_id, text, **kwargs)
        except Exception as exc:
            logger.debug("send_text_quiet failed: %s", exc)
            return None

    async def edit_text_quiet(self, chat_id, message_id: int, text: str, **kwargs):
        """Like edit_text but swallows exceptions."""
        try:
            return await self.edit_text(chat_id, message_id, text, **kwargs)
        except Exception as exc:
            logger.debug("edit_text_quiet failed: %s", exc)
            return None

    async def edit_reply_markup_quiet(self, chat_id, message_id: int, reply_markup=None):
        """Like edit_reply_markup but swallows exceptions."""
        try:
            return await self.edit_reply_markup(chat_id, message_id, reply_markup)
        except Exception as exc:
            logger.debug("edit_reply_markup_quiet failed: %s", exc)
            return None

    # ── Internal ──────────────────────────────────────────────

    async def _run(self, action: str, coro, *, text: str, parse_mode: str, has_keyboard: bool):
        started = time.time()
        error = ""
        ok = True
        result = None
        try:
            result = await self._safe_send(coro)
            ok = result is not None
            if not ok:
                error = "telegram_safe_send_returned_none"
            return result
        except Exception as exc:
            ok = False
            error = str(exc)[:200]
            raise
        finally:
            telegram_observer.record(
                action=action,
                text=text,
                parse_mode=parse_mode,
                has_keyboard=has_keyboard,
                send_ok=ok,
                error=error,
                latency_ms=(time.time() - started) * 1000,
            )

    def _require_bot(self) -> None:
        if not self._bot:
            raise RuntimeError("Telegram gateway is not bound to a bot instance")
