"""
TitanBot Pro — Partial Fill Tracker
====================================
Records each leg of a partially-filled order and computes a true
volume-weighted average price (VWAP) across all legs of the same
``signal_id``. Once the order is finalised, ``finalise()`` reports
the VWAP to the slippage tracker so per-symbol slippage statistics
reflect the *actual* aggregate fill rather than just the last leg.

Background
----------
``core.slippage_tracker`` records a single ``(expected, actual)``
pair per fill event, treating each leg as if it were the entire
entry. When a 10 000-USD order fills in three legs at 100.00 /
100.05 / 100.10, the previously-recorded "actual price" was just
100.10 (the final leg) — slippage was reported as 10 bps when the
true VWAP-weighted slippage was ~5 bps. For a venue with deep books
this difference is small; for thin alts in adversarial flow it can
be 50 % off.

Usage
-----
    from core.partial_fill_tracker import partial_fill_tracker

    # Record each leg as it lands
    partial_fill_tracker.record_leg(
        signal_id=123, symbol="BTC/USDT", strategy="SMC",
        direction="LONG", expected_price=100.0,
        leg_price=100.05, leg_qty=2.5,
    )

    # When the order is fully filled (or cancelled with partial fills),
    # finalise to push the VWAP into the slippage tracker.
    summary = partial_fill_tracker.finalise(signal_id=123, size_usd=10_000)
    if summary:
        print(summary.vwap, summary.total_qty)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.constants import PartialFill as PFC

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────

@dataclass
class FillLeg:
    """A single executed leg of a (potentially) larger order."""
    leg_price: float
    leg_qty: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PartialFillRecord:
    """All recorded legs for a single ``signal_id``."""
    signal_id: int
    symbol: str
    strategy: str
    direction: str          # LONG | SHORT
    expected_price: float   # Zone midpoint (constant across legs)
    legs: List[FillLeg] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finalised: bool = False


@dataclass
class FillSummary:
    """Result of finalising a partially-filled order."""
    signal_id: int
    symbol: str
    direction: str
    vwap: float
    total_qty: float
    leg_count: int
    expected_price: float
    slippage_pct: float     # Positive = adverse, negative = favourable


# ── Tracker ─────────────────────────────────────────────────────

class PartialFillTracker:
    """Per-signal aggregator of partial fill legs."""

    def __init__(self) -> None:
        self._records: Dict[int, PartialFillRecord] = {}

    # ── Recording ───────────────────────────────────────────

    def record_leg(
        self,
        signal_id: int,
        symbol: str,
        strategy: str,
        direction: str,
        expected_price: float,
        leg_price: float,
        leg_qty: float,
    ) -> None:
        """Append a single fill leg to ``signal_id``'s record.

        Subsequent calls with the same ``signal_id`` re-use the existing
        record. A finalised record cannot accept new legs (call
        ``reset(signal_id)`` first if you really need to overwrite).
        """
        if leg_price <= 0 or leg_qty <= 0:
            return
        if direction not in ("LONG", "SHORT"):
            logger.warning(
                f"partial_fill_tracker.record_leg: invalid direction "
                f"{direction!r} for {symbol} (signal {signal_id}); skipping"
            )
            return

        rec = self._records.get(signal_id)
        if rec is None:
            rec = PartialFillRecord(
                signal_id=signal_id,
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                expected_price=expected_price,
            )
            self._records[signal_id] = rec
        elif rec.finalised:
            logger.warning(
                f"partial_fill_tracker: signal {signal_id} already finalised; "
                f"ignoring late leg @ {leg_price}"
            )
            return

        if len(rec.legs) >= PFC.MAX_LEGS_PER_SIGNAL:
            logger.warning(
                f"partial_fill_tracker: signal {signal_id} reached "
                f"MAX_LEGS_PER_SIGNAL ({PFC.MAX_LEGS_PER_SIGNAL}); dropping leg"
            )
            return

        rec.legs.append(FillLeg(leg_price=leg_price, leg_qty=leg_qty))

    # ── Querying ────────────────────────────────────────────

    def total_filled_qty(self, signal_id: int) -> float:
        """Sum of all leg quantities for ``signal_id``. 0.0 if unknown."""
        rec = self._records.get(signal_id)
        if rec is None or not rec.legs:
            return 0.0
        return sum(leg.leg_qty for leg in rec.legs)

    def weighted_avg_price(self, signal_id: int) -> Optional[float]:
        """Volume-weighted average fill price, or ``None`` if no legs."""
        rec = self._records.get(signal_id)
        if rec is None or not rec.legs:
            return None
        total_qty = sum(leg.leg_qty for leg in rec.legs)
        if total_qty <= 0:
            return None
        return sum(leg.leg_price * leg.leg_qty for leg in rec.legs) / total_qty

    def get_record(self, signal_id: int) -> Optional[PartialFillRecord]:
        return self._records.get(signal_id)

    # ── Finalisation ────────────────────────────────────────

    def finalise(
        self,
        signal_id: int,
        size_usd: float = 0.0,
        push_to_slippage_tracker: bool = True,
    ) -> Optional[FillSummary]:
        """
        Mark ``signal_id`` complete and (by default) push the VWAP into
        the slippage tracker. Returns the ``FillSummary`` or ``None``
        if the signal has no recorded legs.
        """
        rec = self._records.get(signal_id)
        if rec is None or not rec.legs:
            return None
        if rec.finalised:
            logger.debug(
                f"partial_fill_tracker.finalise: signal {signal_id} already finalised"
            )
            return None

        vwap = self.weighted_avg_price(signal_id) or 0.0
        total_qty = self.total_filled_qty(signal_id)

        # Adverse slippage convention matches core.slippage_tracker:
        # LONG paid more, SHORT received less.
        if rec.expected_price > 0:
            if rec.direction == "LONG":
                slip_pct = (vwap - rec.expected_price) / rec.expected_price
            else:
                slip_pct = (rec.expected_price - vwap) / rec.expected_price
        else:
            slip_pct = 0.0

        rec.finalised = True
        summary = FillSummary(
            signal_id=signal_id,
            symbol=rec.symbol,
            direction=rec.direction,
            vwap=vwap,
            total_qty=total_qty,
            leg_count=len(rec.legs),
            expected_price=rec.expected_price,
            slippage_pct=slip_pct,
        )

        logger.info(
            f"📦 Partial fill finalised | {rec.symbol} {rec.direction} "
            f"signal={signal_id} legs={len(rec.legs)} vwap={vwap:.6g} "
            f"qty={total_qty:.6g} slip={slip_pct:+.4%}"
        )

        if push_to_slippage_tracker:
            try:
                from core.slippage_tracker import slippage_tracker as _st
                _st.record_fill(
                    signal_id=signal_id,
                    symbol=rec.symbol,
                    strategy=rec.strategy,
                    expected_price=rec.expected_price,
                    actual_price=vwap,
                    direction=rec.direction,
                    size_usd=size_usd,
                )
            except Exception as exc:
                logger.debug(
                    f"partial_fill_tracker: slippage_tracker.record_fill skipped ({exc})"
                )

        return summary

    # ── Maintenance ─────────────────────────────────────────

    def reset(self, signal_id: int) -> None:
        """Drop all state for ``signal_id``."""
        self._records.pop(signal_id, None)

    def purge_stale(self, retention_hours: int = PFC.RETENTION_HOURS) -> int:
        """Drop finalised records older than ``retention_hours``.

        Returns the number of records purged. Active (non-finalised)
        records are *never* purged — a slow venue may still be filling
        a leg several hours after the order was placed.
        """
        cutoff = time.time() - retention_hours * 3600
        stale = [
            sid for sid, rec in self._records.items()
            if rec.finalised and rec.started_at < cutoff
        ]
        for sid in stale:
            del self._records[sid]
        if stale:
            logger.debug(
                f"partial_fill_tracker.purge_stale: purged {len(stale)} records"
            )
        return len(stale)


# ── Singleton ─────────────────────────────────────────────────
partial_fill_tracker = PartialFillTracker()
