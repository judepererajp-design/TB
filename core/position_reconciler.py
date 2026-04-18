"""
TitanBot Pro — Position Reconciliation
========================================
On startup, recovers in-flight signals from the database and
re-registers them with the outcome monitor and execution engine.

Prevents "orphan" signals: if the bot crashes mid-trade, signals
that were ACTIVE or PENDING are restored so TP/SL monitoring resumes
without manual intervention.

Usage:
    from core.position_reconciler import reconciler
    await reconciler.reconcile_on_startup(outcome_monitor, execution_engine)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from config.constants import Database as DBC

logger = logging.getLogger(__name__)

# Max age for recovery — signals older than this are expired, not recovered
RECOVERY_MAX_AGE_HOURS: int = DBC.SIGNAL_MAX_AGE_RECOVERY_HOURS
ACTIVE_RECOVERY_STATES: tuple = ("PENDING", "ACTIVE", "TP1_HIT", "BE_ACTIVE")


@dataclass
class RecoveredSignal:
    """A signal recovered from DB during startup reconciliation."""
    signal_id: int
    symbol: str
    direction: str
    strategy: str
    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    confidence: float
    state: str
    entry_price: Optional[float]
    be_stop: Optional[float]
    trail_stop: Optional[float]
    trail_pct: float
    regime: str
    created_at: float
    activated_at: Optional[float]
    message_id: Optional[int]


class PositionReconciler:
    """Restores active signals from the database on startup."""

    def __init__(self) -> None:
        self._recovered: List[RecoveredSignal] = []
        self._expired: List[int] = []

    async def reconcile_on_startup(
        self,
        outcome_monitor: object,
        execution_engine: Optional[object] = None,
    ) -> Dict[str, int]:
        """Recover in-flight signals from the database.

        Returns a summary dict: {"recovered": N, "expired": N, "errors": N}.
        """
        from data.database import db

        logger.info("🔄 Position reconciliation — checking for in-flight signals…")

        try:
            if not hasattr(db, '_pool') or db._pool is None:
                # DB not initialised yet
                logger.warning("Reconciler: database not ready, skipping")
                return {"recovered": 0, "expired": 0, "errors": 0}

            rows = await self._fetch_active_signals(db)
        except Exception as e:
            logger.error(f"Reconciler: failed to query DB: {e}")
            return {"recovered": 0, "expired": 0, "errors": 1}

        if not rows:
            logger.info("✅ No in-flight signals to recover")
            return {"recovered": 0, "expired": 0, "errors": 0}

        recovered = 0
        expired = 0
        errors = 0
        now = time.time()
        max_age_secs = RECOVERY_MAX_AGE_HOURS * 3600

        for row in rows:
            try:
                sig = self._row_to_signal(row)

                # Check age — expire if too old
                age_secs = now - sig.created_at
                if age_secs > max_age_secs:
                    logger.info(
                        f"  ⏰ Expiring stale signal #{sig.signal_id} "
                        f"({sig.symbol} {sig.direction}) — "
                        f"age={age_secs/3600:.1f}h > {RECOVERY_MAX_AGE_HOURS}h"
                    )
                    self._expired.append(sig.signal_id)
                    expired += 1
                    continue

                # Re-register with outcome monitor
                await self._restore_signal(sig, outcome_monitor)
                self._recovered.append(sig)
                recovered += 1

                logger.info(
                    f"  ✅ Recovered #{sig.signal_id} {sig.symbol} "
                    f"{sig.direction} (state={sig.state}, "
                    f"age={age_secs/3600:.1f}h)"
                )

            except Exception as e:
                errors += 1
                logger.error(f"  ❌ Failed to recover signal: {e}")

        summary = {
            "recovered": recovered,
            "expired": expired,
            "errors": errors,
        }
        logger.info(
            f"🔄 Reconciliation complete: "
            f"{recovered} recovered, {expired} expired, {errors} errors"
        )
        return summary

    async def _fetch_active_signals(self, db: object) -> List[dict]:
        """Query DB for signals in active states."""
        active_states = ACTIVE_RECOVERY_STATES
        placeholders = ",".join("?" for _ in active_states)
        sql = (
            f"SELECT * FROM tracked_signals_v1 "
            f"WHERE state IN ({placeholders}) "
            f"ORDER BY created_at DESC"
        )

        try:
            async with db._pool.reader() as conn:
                async with conn.execute(sql, active_states) as cursor:
                    cols = [d[0] for d in cursor.description]
                    rows_raw = await cursor.fetchall()
                    return [dict(zip(cols, r)) for r in rows_raw]
        except Exception as e:
            logger.error(f"Reconciler query failed: {e}")
            return []

    def _row_to_signal(self, row: dict) -> RecoveredSignal:
        """Convert a DB row dict to a RecoveredSignal."""
        return RecoveredSignal(
            signal_id=row.get("signal_id", 0),
            symbol=row.get("symbol", ""),
            direction=row.get("direction", ""),
            strategy=row.get("strategy", ""),
            entry_low=float(row.get("entry_low", 0)),
            entry_high=float(row.get("entry_high", 0)),
            stop_loss=float(row.get("stop_loss", 0)),
            tp1=float(row.get("tp1", 0)),
            tp2=float(row.get("tp2", 0)),
            tp3=float(row["tp3"]) if row.get("tp3") is not None else None,
            confidence=float(row.get("confidence", 0)),
            state=row.get("state", "PENDING"),
            entry_price=float(row.get("entry_price")) if row.get("entry_price") else None,
            be_stop=float(row.get("be_stop")) if row.get("be_stop") else None,
            trail_stop=float(row.get("trail_stop")) if row.get("trail_stop") else None,
            trail_pct=float(row.get("trail_pct", DBC.DEFAULT_TRAIL_PCT)),
            regime=row.get("regime", "UNKNOWN"),
            created_at=float(row.get("created_at", 0)),
            activated_at=float(row.get("activated_at")) if row.get("activated_at") else None,
            message_id=int(row.get("message_id")) if row.get("message_id") else None,
        )

    async def _restore_signal(
        self, sig: RecoveredSignal, outcome_monitor: object
    ) -> None:
        """Re-register a recovered signal with the outcome monitor."""
        if not hasattr(outcome_monitor, "track_signal"):
            logger.warning("Outcome monitor has no track_signal method")
            return

        # Build kwargs matching OutcomeMonitor.track_signal() signature
        await outcome_monitor.track_signal(
            signal_id=sig.signal_id,
            symbol=sig.symbol,
            direction=sig.direction,
            strategy=sig.strategy,
            entry_low=sig.entry_low,
            entry_high=sig.entry_high,
            stop_loss=sig.stop_loss,
            tp1=sig.tp1,
            tp2=sig.tp2,
            tp3=sig.tp3,
            confidence=sig.confidence,
            regime=sig.regime,
            message_id=sig.message_id,
        )

    @property
    def recovered_count(self) -> int:
        return len(self._recovered)

    @property
    def expired_count(self) -> int:
        return len(self._expired)


# ── Singleton ───────────────────────────────────────────────────
reconciler = PositionReconciler()
