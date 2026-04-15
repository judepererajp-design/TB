from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


# ── Trade lifecycle states ─────────────────────────────────────────────

class TradeState(Enum):
    PENDING      = "PENDING"       # restored from DB, not yet active in this session
    SETUP        = "SETUP"
    ENTRY_ACTIVE = "ENTRY_ACTIVE"
    TRADE_ACTIVE = "TRADE_ACTIVE"
    TP1_HIT      = "TP1_HIT"
    CLOSED       = "CLOSED"

@dataclass
class PendingInput:
    kind: str
    prompt: str
    context: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ViewState:
    screen: str
    parent: Optional[str] = None
    context: dict = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


class TelegramSessionState:
    """Small in-memory state store for pending input and current views."""

    def __init__(self) -> None:
        self._pending_input: Dict[str, PendingInput] = {}
        self._views: Dict[str, ViewState] = {}

    def begin_input(self, chat_id: str, kind: str, prompt: str, **context) -> PendingInput:
        pending = PendingInput(kind=kind, prompt=prompt, context=context)
        self._pending_input[chat_id] = pending
        return pending

    def peek_input(self, chat_id: str) -> Optional[PendingInput]:
        return self._pending_input.get(chat_id)

    def pop_input(self, chat_id: str) -> Optional[PendingInput]:
        return self._pending_input.pop(chat_id, None)

    def clear_input(self, chat_id: str) -> None:
        self._pending_input.pop(chat_id, None)

    def remember_view(
        self,
        chat_id: str,
        screen: str,
        *,
        message_id: Optional[int] = None,
        parent: Optional[str] = None,
        **context,
    ) -> ViewState:
        key = self._view_key(chat_id, message_id)
        view = ViewState(screen=screen, parent=parent, context=context)
        self._views[key] = view
        return view

    def get_view(self, chat_id: str, message_id: Optional[int] = None) -> Optional[ViewState]:
        return self._views.get(self._view_key(chat_id, message_id))

    def _view_key(self, chat_id: str, message_id: Optional[int]) -> str:
        return f"{chat_id}:{message_id or 0}"
