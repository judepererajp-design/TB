"""
TitanBot Pro — Funding Rate Bridge
====================================
Bridges the derivatives analyzer's funding-rate data into the
aggregator scoring pipeline so that extreme funding conditions
directly affect signal confidence.

Usage:
    from signals.funding_bridge import funding_bridge
    adj = funding_bridge.assess_funding_impact(symbol, "LONG", funding_data)
    signal.confidence += adj.confidence_delta
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.constants import FundingIntegration as FIC

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────

@dataclass
class FundingAdjustment:
    """Result of a funding-rate impact assessment."""
    confidence_delta: float = 0.0   # Points to add/subtract
    rr_adjustment: float = 0.0      # Multiplier for R:R (1.0 = no change)
    reason: str = ""
    severity: str = "NONE"          # NONE | LOW | MEDIUM | HIGH


# ── Bridge ──────────────────────────────────────────────────────

class FundingBridge:
    """Maintains per-symbol funding history and assesses impact on signals."""

    def __init__(self) -> None:
        # symbol -> deque of (timestamp, rate) tuples
        self._history: Dict[str, deque] = {}

    # ── History management ──────────────────────────────────

    def record_rate(self, symbol: str, rate: float) -> None:
        """Append the latest funding rate for *symbol*."""
        self._history.setdefault(
            symbol, deque(maxlen=FIC.MAX_HISTORY_LENGTH)
        ).append((time.time(), rate))

    def get_funding_history(self, symbol: str) -> List[float]:
        """Return the raw rate values for *symbol* (oldest first)."""
        buf = self._history.get(symbol)
        if not buf:
            return []
        return [r for _, r in buf]

    def is_funding_extreme(self, symbol: str) -> bool:
        """Quick check: is the most recent rate in the extreme zone?"""
        buf = self._history.get(symbol)
        if not buf:
            return False
        _, latest = buf[-1]
        return latest >= FIC.EXTREME_POSITIVE_RATE or latest <= FIC.EXTREME_NEGATIVE_RATE

    # ── Impact assessment ───────────────────────────────────

    def assess_funding_impact(
        self,
        symbol: str,
        direction: str,
        funding_data: Optional[dict] = None,
    ) -> FundingAdjustment:
        """Compute a confidence adjustment based on funding conditions.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. "BTC/USDT".
        direction : str
            "LONG" or "SHORT".
        funding_data : dict, optional
            Dict with at least ``funding_rate`` (float) and optionally
            ``funding_trend`` (str: RISING / FALLING / NEUTRAL).
        """
        if funding_data is None:
            return FundingAdjustment()

        rate = funding_data.get("funding_rate", 0.0)
        if rate is None:
            rate = 0.0
        trend = funding_data.get("funding_trend", "NEUTRAL")

        # Record for history tracking
        self.record_rate(symbol, rate)

        # ── Compute trend slope from history ────────────────
        history = self.get_funding_history(symbol)
        trend_slope = 0.0
        if len(history) >= FIC.TREND_WINDOW:
            recent = history[-FIC.TREND_WINDOW:]
            trend_slope = recent[-1] - recent[0]

        # ── Direction-aware assessment ──────────────────────
        delta = 0.0
        rr_adj = 1.0
        reason_parts: List[str] = []
        severity = "NONE"

        if direction == "LONG":
            if rate >= FIC.EXTREME_POSITIVE_RATE:
                delta = -FIC.EXTREME_PENALTY_PTS
                severity = "HIGH"
                reason_parts.append(
                    f"extreme positive funding {rate:.4%} opposes LONG"
                )
            elif rate >= FIC.HIGH_POSITIVE_RATE:
                delta = -FIC.OPPOSED_PENALTY_PTS
                severity = "MEDIUM"
                reason_parts.append(
                    f"high positive funding {rate:.4%} opposes LONG"
                )
            elif rate <= FIC.EXTREME_NEGATIVE_RATE:
                delta = FIC.ALIGNED_BOOST_PTS
                severity = "LOW"
                reason_parts.append(
                    f"negative funding {rate:.4%} supports LONG (shorts paying)"
                )
        else:  # SHORT
            if rate <= FIC.EXTREME_NEGATIVE_RATE:
                delta = -FIC.EXTREME_PENALTY_PTS
                severity = "HIGH"
                reason_parts.append(
                    f"extreme negative funding {rate:.4%} opposes SHORT"
                )
            elif rate >= FIC.EXTREME_POSITIVE_RATE:
                delta = FIC.ALIGNED_BOOST_PTS
                severity = "LOW"
                reason_parts.append(
                    f"extreme positive funding {rate:.4%} supports SHORT (longs paying)"
                )
            elif rate >= FIC.HIGH_POSITIVE_RATE:
                delta = FIC.ALIGNED_BOOST_PTS * 0.5
                severity = "LOW"
                reason_parts.append(
                    f"high positive funding {rate:.4%} moderately supports SHORT"
                )

        # ── Trend amplifier ─────────────────────────────────
        # If funding is trending AGAINST position, amplify the penalty
        if delta < 0 and trend_slope != 0:
            if (direction == "LONG" and trend_slope > 0) or \
               (direction == "SHORT" and trend_slope < 0):
                delta *= 1.25  # 25% worse when trending against
                reason_parts.append("funding trend worsening")

        reason = "; ".join(reason_parts) if reason_parts else "funding neutral"

        if delta != 0:
            logger.info(
                f"💰 Funding | {symbol} {direction} | "
                f"rate={rate:.4%} trend={trend} | "
                f"Δconf={delta:+.1f} | {reason}"
            )

        return FundingAdjustment(
            confidence_delta=delta,
            rr_adjustment=rr_adj,
            reason=reason,
            severity=severity,
        )


# ── Singleton ───────────────────────────────────────────────────
funding_bridge = FundingBridge()
