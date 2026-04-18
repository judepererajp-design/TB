"""
TitanBot Pro — Missed-Fill Tracker
====================================
Records signals that expire `PENDING` (price never reached the entry zone)
and *replays the post-expiry tape* to determine whether the setup would
have hit TP1 or SL had it filled. Feeds the learning loop with a class
of trades that are otherwise invisible: "good ideas the market never
gave us."

Lifecycle:
    record_missed(...)             ← called from engine._handle_signal_expired
        │
        ▼  (within EVAL_WINDOW_SECS, evaluated each tick or via tick(price))
    update_with_price(...)
        │
        ▼  (when entry_reached or window_expired)
    finalise() → outcome ∈ {HIT_TP1, HIT_SL_FIRST, NEVER_ENTERED, NO_DATA}

Outputs are exposed via:
    - to_dict() snapshot for the dashboard
    - get_stats(window_hours) summary for the learning loop / health metrics

Design notes:
    • Stateless apart from the in-memory tracking dict — no DB persistence
      yet so it survives only as long as the process; this is intentional
      for v1 because the data is short-lived (we replay over hours, not
      days) and the learning-loop sink already persists the outcome.
    • Decoupled from price_cache: the tracker accepts ticks via
      ``update_with_price(symbol, price)`` so it can be driven from the
      existing polling loop without adding another fetch.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────
# Replay each missed signal for this many seconds after expiry. Chosen
# to be ~half a typical signal lifetime so we capture late fills + the
# first leg toward TP1, without holding state forever.
EVAL_WINDOW_SECS: int = 4 * 3600      # 4 hours

# Cap the rolling history of finalised records so the dashboard query
# stays cheap.
MAX_HISTORY: int = 500


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class MissedSignal:
    signal_id: int
    symbol: str
    direction: str          # LONG | SHORT
    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float
    tp2: float
    confidence: float
    strategy: str
    expired_at: float = field(default_factory=time.time)

    # Mutable state populated as ticks come in
    entered: bool = False
    entered_price: Optional[float] = None
    extreme_after: Optional[float] = None      # best favourable price post-expiry
    worst_after: Optional[float] = None         # worst adverse price post-expiry
    outcome: str = "PENDING"   # PENDING|HIT_TP1|HIT_SL_FIRST|NEVER_ENTERED|NO_DATA
    finalised_at: Optional[float] = None


# ── Tracker ───────────────────────────────────────────────────────────

class MissedFillTracker:
    """In-memory replay of expired/unfilled signals."""

    def __init__(self, eval_window_secs: int = EVAL_WINDOW_SECS,
                 max_history: int = MAX_HISTORY) -> None:
        self._eval_window = eval_window_secs
        self._active: Dict[int, MissedSignal] = {}
        self._history: Deque[MissedSignal] = deque(maxlen=max_history)

    # ── Recording ─────────────────────────────────────────────

    def record_missed(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        tp1: float,
        tp2: float = 0.0,
        confidence: float = 0.0,
        strategy: str = "",
    ) -> Optional[MissedSignal]:
        """Register an expired-unfilled signal for post-expiry replay.

        Returns the new ``MissedSignal`` record (or ``None`` if inputs are
        invalid — e.g. zero entry/SL).
        """
        if direction not in ("LONG", "SHORT"):
            return None
        if entry_low <= 0 or entry_high <= 0 or stop_loss <= 0 or tp1 <= 0:
            return None
        if signal_id in self._active:
            return self._active[signal_id]
        rec = MissedSignal(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_low=float(entry_low),
            entry_high=float(entry_high),
            stop_loss=float(stop_loss),
            tp1=float(tp1),
            tp2=float(tp2),
            confidence=float(confidence),
            strategy=strategy,
        )
        self._active[signal_id] = rec
        # Keep the symbol polled by price_cache for the eval window.
        # InvalidationMonitor will have unsubscribed it on expiry; without
        # this re-subscribe we'd never see a tick.
        try:
            from core.price_cache import price_cache as _pc
            _pc.subscribe(symbol)
        except Exception:
            pass
        logger.info(
            f"📋 MissedFillTracker: tracking expired signal #{signal_id} "
            f"{symbol} {direction} | entry={entry_low}-{entry_high} "
            f"sl={stop_loss} tp1={tp1}"
        )
        return rec

    # ── Tick ingestion ────────────────────────────────────────

    def update_with_price(self, symbol: str, price: float) -> List[MissedSignal]:
        """Feed a price tick. Returns the list of records that were
        finalised on this tick (so callers can forward them to a learning
        loop without re-iterating the whole table).

        Idempotent and safe to call from any monitor's polling loop.
        """
        if not symbol or price is None or price <= 0:
            return []
        finalised: List[MissedSignal] = []
        now = time.time()
        # Iterate over a snapshot since we mutate during the loop.
        for sid, rec in list(self._active.items()):
            if rec.symbol != symbol:
                # Still check the time window for stale records that no
                # longer receive their own price ticks.
                if now - rec.expired_at >= self._eval_window:
                    self._finalise(rec, now, reason="window_expired")
                    finalised.append(rec)
                continue

            self._update_one(rec, price, now)

            if rec.outcome != "PENDING":
                finalised.append(rec)

        return finalised

    def _update_one(self, rec: MissedSignal, price: float, now: float) -> None:
        # Track the favourable / adverse extremes regardless of entry.
        if rec.direction == "LONG":
            if rec.extreme_after is None or price > rec.extreme_after:
                rec.extreme_after = price
            if rec.worst_after is None or price < rec.worst_after:
                rec.worst_after = price
        else:
            if rec.extreme_after is None or price < rec.extreme_after:
                rec.extreme_after = price
            if rec.worst_after is None or price > rec.worst_after:
                rec.worst_after = price

        # Step 1: did price reach the entry zone?
        if not rec.entered:
            if rec.direction == "LONG" and rec.entry_low <= price <= rec.entry_high * 1.001:
                rec.entered = True
                rec.entered_price = price
            elif rec.direction == "SHORT" and rec.entry_high >= price >= rec.entry_low * 0.999:
                rec.entered = True
                rec.entered_price = price

        # Step 2: post-entry, did TP1 or SL fire first?
        if rec.entered:
            if rec.direction == "LONG":
                if price >= rec.tp1:
                    self._finalise(rec, now, "HIT_TP1")
                    return
                if price <= rec.stop_loss:
                    self._finalise(rec, now, "HIT_SL_FIRST")
                    return
            else:
                if price <= rec.tp1:
                    self._finalise(rec, now, "HIT_TP1")
                    return
                if price >= rec.stop_loss:
                    self._finalise(rec, now, "HIT_SL_FIRST")
                    return

        # Step 3: window expired without resolution.
        if now - rec.expired_at >= self._eval_window:
            outcome = "NEVER_ENTERED" if not rec.entered else "NO_DATA"
            self._finalise(rec, now, outcome)

    def _finalise(self, rec: MissedSignal, now: float, reason: str) -> None:
        """Move record from active → history with the given outcome."""
        if rec.outcome != "PENDING":
            return  # already finalised
        # Map "window_expired" → NEVER_ENTERED if no entry, else NO_DATA.
        if reason == "window_expired":
            reason = "NEVER_ENTERED" if not rec.entered else "NO_DATA"
        rec.outcome = reason
        rec.finalised_at = now
        self._history.append(rec)
        self._active.pop(rec.signal_id, None)
        # Release the price_cache reference we took in record_missed so the
        # symbol stops being polled if no other consumer needs it.
        try:
            from core.price_cache import price_cache as _pc
            _pc.unsubscribe(rec.symbol)
        except Exception:
            pass
        logger.info(
            f"📋 MissedFillTracker: finalised #{rec.signal_id} {rec.symbol} "
            f"{rec.direction} → {reason} (entered={rec.entered})"
        )

    # ── Querying ──────────────────────────────────────────────

    def get_record(self, signal_id: int) -> Optional[MissedSignal]:
        """Return the active or finalised record for a signal_id, or None."""
        if signal_id in self._active:
            return self._active[signal_id]
        for rec in reversed(self._history):
            if rec.signal_id == signal_id:
                return rec
        return None

    def get_stats(self, window_hours: int = 24) -> Dict[str, float]:
        """Aggregate stats over finalised records in the lookback window."""
        cutoff = time.time() - window_hours * 3600
        records = [
            r for r in self._history
            if (r.finalised_at or 0) >= cutoff
        ]
        if not records:
            return {
                "total": 0,
                "would_have_won_pct": 0.0,
                "would_have_lost_pct": 0.0,
                "never_entered_pct": 0.0,
            }
        n = len(records)
        won = sum(1 for r in records if r.outcome == "HIT_TP1")
        lost = sum(1 for r in records if r.outcome == "HIT_SL_FIRST")
        never = sum(1 for r in records if r.outcome == "NEVER_ENTERED")
        return {
            "total": n,
            "would_have_won_pct": 100.0 * won / n,
            "would_have_lost_pct": 100.0 * lost / n,
            "never_entered_pct": 100.0 * never / n,
        }

    def to_dict(self) -> Dict:
        """Snapshot for dashboard / persistence."""
        stats_24h = self.get_stats(window_hours=24)
        return {
            "active_count": len(self._active),
            "finalised_count": len(self._history),
            "stats_24h": stats_24h,
        }

    # ── Maintenance ───────────────────────────────────────────

    def purge_stale(self) -> int:
        """Force-finalise active records whose window has expired.

        Returns the number of records purged. Useful when no price tick
        has arrived for an inactive symbol (e.g. delisted) within the
        evaluation window.
        """
        now = time.time()
        purged = 0
        for sid, rec in list(self._active.items()):
            if now - rec.expired_at >= self._eval_window:
                self._finalise(rec, now, reason="window_expired")
                purged += 1
        return purged

    def reset(self) -> None:
        """Clear all state. Test helper."""
        self._active.clear()
        self._history.clear()


# ── Singleton ─────────────────────────────────────────────────────────
missed_fill_tracker = MissedFillTracker()
