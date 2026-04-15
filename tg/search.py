from __future__ import annotations

from typing import Any, Dict, List, Optional


class TelegramSearchService:
    """Guided manual lookup for signals, strategies, notes, and health events."""

    def __init__(self, database=None) -> None:
        if database is None:
            from data.database import db as database
        self._db = database

    def prompt_for(self, mode: str) -> str:
        prompts = {
            "signal": "Send a signal ID, e.g. <code>142</code>",
            "symbol": "Send a symbol, e.g. <code>BTC</code> or <code>BTC/USDT</code>",
            "strategy": "Send a strategy name, e.g. <code>Wyckoff</code>",
            "notes": "Send any note text to search signal notes",
            "event": "Send an error/event keyword, e.g. <code>telegram</code> or <code>circuit</code>",
            "any": "Send anything: signal ID, symbol, strategy, or health keyword",
        }
        return prompts.get(mode, prompts["any"])

    async def search(self, mode: str, query: str) -> Dict[str, Any]:
        mode = (mode or "any").lower()
        query = (query or "").strip()
        if not query:
            return {"mode": mode, "query": query, "signals": [], "notes": [], "events": []}

        if mode == "signal":
            signal = await self._search_signal_id(query)
            return {"mode": mode, "query": query, "signals": [signal] if signal else [], "notes": [], "events": []}

        if mode == "symbol":
            signals = await self._search_symbol(query)
            return {"mode": mode, "query": query, "signals": signals, "notes": [], "events": []}

        if mode == "strategy":
            signals = await self._search_strategy(query)
            return {"mode": mode, "query": query, "signals": signals, "notes": [], "events": []}

        if mode == "notes":
            notes = await self._search_notes(query)
            return {"mode": mode, "query": query, "signals": [], "notes": notes, "events": []}

        if mode == "event":
            events = await self._search_events(query)
            return {"mode": mode, "query": query, "signals": [], "notes": [], "events": events}

        signal = await self._search_signal_id(query)
        symbol_hits = await self._search_symbol(query)
        strategy_hits = await self._search_strategy(query)
        note_hits = await self._search_notes(query)
        event_hits = await self._search_events(query)
        if signal and not note_hits:
            note_hits = [
                note for note in await self._db.get_signal_notes(limit=100)
                if note.get("signal_id") == signal.get("id")
            ]

        merged_signals: List[Dict[str, Any]] = []
        seen = set()
        for item in ([signal] if signal else []) + symbol_hits + strategy_hits:
            sid = item.get("id")
            if sid in seen:
                continue
            seen.add(sid)
            merged_signals.append(item)

        return {
            "mode": mode,
            "query": query,
            "signals": merged_signals[:10],
            "notes": note_hits[:5],
            "events": event_hits[:5],
        }

    def format_results(self, results: Dict[str, Any]) -> str:
        mode = results.get("mode", "any")
        query = results.get("query", "")
        signals = results.get("signals", [])
        notes = results.get("notes", [])
        events = results.get("events", [])

        title = {
            "signal": "Signal Lookup",
            "symbol": "Symbol Lookup",
            "strategy": "Strategy Lookup",
            "notes": "Signal Notes Search",
            "event": "Bot Health Search",
            "any": "Universal Search",
        }.get(mode, "Search")

        lines = [f"🔎 <b>{title}</b>", f"<i>Query:</i> <code>{query}</code>", ""]
        if not signals and not notes and not events:
            lines.append("No matches found.")
            lines.append("Try signal ID, symbol, strategy, or a health keyword.")
            return "\n".join(lines)

        if signals:
            lines.append(f"<b>Signals ({len(signals)})</b>")
            for sig in signals[:6]:
                outcome = sig.get("outcome") or "PENDING"
                grade = sig.get("alpha_grade") or sig.get("grade") or "?"
                direction = sig.get("direction") or "?"
                lines.append(
                    f"• #{sig.get('id')} <b>{sig.get('symbol')}</b> {direction} · {grade} · {outcome}"
                )
            lines.append("")

        if notes:
            lines.append(f"<b>Notes ({len(notes)})</b>")
            for note in notes[:3]:
                lines.append(
                    f"• #{note.get('signal_id')} <b>{note.get('symbol') or '?'}</b> — {str(note.get('notes') or '')[:90]}"
                )
            lines.append("")

        if events:
            lines.append(f"<b>Events ({len(events)})</b>")
            for event in events[:3]:
                lines.append(
                    f"• <b>{event.get('event_type', '?')}</b> — {str(event.get('message') or '')[:90]}"
                )

        return "\n".join(line for line in lines if line is not None).strip()

    async def _search_signal_id(self, query: str) -> Optional[Dict[str, Any]]:
        if not query.isdigit():
            return None
        return await self._db.get_signal(int(query))

    async def _search_symbol(self, query: str) -> List[Dict[str, Any]]:
        symbol = self._normalize_symbol(query)
        hits = await self._db.get_recent_signals(hours=24 * 30, symbol=symbol, exclude_c_grade=False)
        if hits or "/" in query or "USDT" in query.upper():
            return hits[:10]
        all_hits = await self._db.get_recent_signals_all(hours=24 * 30)
        needle = query.upper()
        return [s for s in all_hits if needle in str(s.get("symbol", "")).upper()][:10]

    async def _search_strategy(self, query: str) -> List[Dict[str, Any]]:
        all_hits = await self._db.get_recent_signals_all(hours=24 * 30)
        needle = query.lower()
        return [s for s in all_hits if needle in str(s.get("strategy", "")).lower()][:10]

    async def _search_notes(self, query: str) -> List[Dict[str, Any]]:
        notes = await self._db.get_signal_notes(limit=100)
        needle = query.lower()
        return [n for n in notes if needle in str(n.get("notes", "")).lower()][:10]

    async def _search_events(self, query: str) -> List[Dict[str, Any]]:
        events = await self._db.get_recent_events(limit=100)
        needle = query.lower()
        return [
            e for e in events
            if needle in str(e.get("event_type", "")).lower()
            or needle in str(e.get("message", "")).lower()
        ][:10]

    def _normalize_symbol(self, query: str) -> str:
        symbol = query.upper().strip()
        if not symbol:
            return symbol
        if "/" not in symbol and not symbol.endswith("USDT"):
            symbol = f"{symbol}/USDT"
        elif "/" not in symbol:
            symbol = f"{symbol[:-4]}/USDT"
        return symbol
