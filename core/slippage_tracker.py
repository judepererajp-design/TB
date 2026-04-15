"""
TitanBot Pro — Live Slippage Tracker
======================================
Records the difference between expected entry price (zone midpoint)
and actual fill price.  Maintains per-symbol and per-strategy history
so the engine can penalise symbols with consistently adverse slippage.

Usage:
    from core.slippage_tracker import slippage_tracker
    slippage_tracker.record_fill(signal_id, symbol, strategy,
                                  expected_price, actual_price, direction, size_usd)
    stats = slippage_tracker.get_stats(symbol="BTC/USDT")
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.constants import SlippageTracking as STC

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────

@dataclass
class FillRecord:
    """Single recorded fill event."""
    signal_id: int
    symbol: str
    strategy: str
    direction: str          # LONG | SHORT
    expected_price: float   # Zone midpoint
    actual_price: float     # Observed fill
    size_usd: float
    slippage_pct: float     # Positive = adverse, negative = favourable
    timestamp: float = field(default_factory=time.time)


@dataclass
class SlippageStats:
    """Aggregated slippage statistics."""
    avg_slippage_pct: float = 0.0
    max_slippage_pct: float = 0.0
    total_fills: int = 0
    total_slippage_usd: float = 0.0
    favorable_fills: int = 0
    adverse_fills: int = 0


# ── Tracker ─────────────────────────────────────────────────────

class SlippageTracker:
    """Tracks fill quality across all signals."""

    def __init__(self) -> None:
        self._history: deque[FillRecord] = deque(maxlen=STC.MAX_HISTORY_SIZE)

    # ── Recording ───────────────────────────────────────────

    def record_fill(
        self,
        signal_id: int,
        symbol: str,
        strategy: str,
        expected_price: float,
        actual_price: float,
        direction: str,
        size_usd: float = 0.0,
    ) -> None:
        """Record a fill event and compute slippage."""
        if expected_price <= 0:
            return

        # Adverse slippage: paid more for LONG, received less for SHORT
        if direction == "LONG":
            slip_pct = (actual_price - expected_price) / expected_price
        else:
            slip_pct = (expected_price - actual_price) / expected_price

        rec = FillRecord(
            signal_id=signal_id,
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            expected_price=expected_price,
            actual_price=actual_price,
            size_usd=size_usd,
            slippage_pct=slip_pct,
        )
        self._history.append(rec)

        label = "adverse" if slip_pct > STC.NEUTRAL_THRESHOLD else ("favourable" if slip_pct < -STC.NEUTRAL_THRESHOLD else "neutral")
        logger.info(
            f"📊 Slippage | {symbol} {direction} | "
            f"{slip_pct:+.4%} ({label}) | "
            f"expected={expected_price:.6g} actual={actual_price:.6g}"
        )

    # ── Querying ────────────────────────────────────────────

    def get_stats(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        window_hours: int = STC.DEFAULT_WINDOW_HOURS,
    ) -> SlippageStats:
        """Get aggregated slippage stats, optionally filtered."""
        cutoff = time.time() - window_hours * 3600
        records = [
            r for r in self._history
            if r.timestamp >= cutoff
            and (symbol is None or r.symbol == symbol)
            and (strategy is None or r.strategy == strategy)
        ]

        if not records:
            return SlippageStats()

        slips = [r.slippage_pct for r in records]
        return SlippageStats(
            avg_slippage_pct=sum(slips) / len(slips),
            max_slippage_pct=max(slips),
            total_fills=len(records),
            total_slippage_usd=sum(
                abs(r.slippage_pct) * r.size_usd for r in records
            ),
            favorable_fills=sum(1 for s in slips if s < 0),
            adverse_fills=sum(1 for s in slips if s > 0),
        )

    def get_adjustment_factor(
        self, symbol: str, direction: str
    ) -> float:
        """Return a confidence multiplier based on recent slippage.

        If recent fills on *symbol* in *direction* show consistent
        adverse slippage, return a value < 1.0 to reduce confidence.
        E.g. 0.97 means "reduce confidence by 3%".
        """
        stats = self.get_stats(symbol=symbol, window_hours=STC.DEFAULT_WINDOW_HOURS)
        if stats.total_fills == 0:
            return 1.0

        avg = stats.avg_slippage_pct
        if avg <= 0:
            # Favourable or zero slippage — no penalty
            return 1.0

        reduction = min(
            avg * STC.ADVERSE_SLIPPAGE_PENALTY_SCALE,
            STC.MAX_CONFIDENCE_REDUCTION,
        )
        return 1.0 - reduction

    # ── Serialisation ───────────────────────────────────────

    def to_dict(self) -> Dict:
        """Snapshot for persistence / dashboard display."""
        overall = self.get_stats()
        return {
            "total_fills": overall.total_fills,
            "avg_slippage_pct": round(overall.avg_slippage_pct, 6),
            "max_slippage_pct": round(overall.max_slippage_pct, 6),
            "total_slippage_usd": round(overall.total_slippage_usd, 2),
            "favorable_fills": overall.favorable_fills,
            "adverse_fills": overall.adverse_fills,
            "buffer_size": len(self._history),
        }


# ── Singleton ───────────────────────────────────────────────────
slippage_tracker = SlippageTracker()
