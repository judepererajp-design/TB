"""
TitanBot Pro — Funding Rate Bridge
====================================
Bridges the derivatives analyzer's funding-rate data into the
aggregator scoring pipeline so that extreme funding conditions
directly affect signal confidence.

Also tracks *accrued* funding (cumulative funding paid/received) per
active position so callers can compute a funding-inclusive net PnL.

Usage:
    from signals.funding_bridge import funding_bridge
    adj = funding_bridge.assess_funding_impact(symbol, "LONG", funding_data)
    signal.confidence += adj.confidence_delta

    # Accrual tracking (one call on activation, then each funding cycle):
    funding_bridge.start_accrual(signal_id, symbol, "LONG", entry_ts)
    funding_bridge.record_cycle(signal_id, rate=0.0001)
    pct = funding_bridge.get_accrued_pct(signal_id)   # signed, in fraction (0.0001 = +1 bp gain)
    r   = funding_bridge.get_accrued_r(signal_id, entry_price, stop_loss)
    funding_bridge.stop_accrual(signal_id)
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


@dataclass
class FundingAccrual:
    """Cumulative funding paid/received for a single active position.

    ``cumulative_rate`` is the *signed* sum of per-cycle funding rates
    that have been charged against this position. Sign convention
    follows the Binance perps side convention:

      * LONG pays funding when ``rate > 0`` (positive rate reduces PnL).
      * SHORT pays funding when ``rate < 0`` (negative rate reduces PnL).

    ``accrued_pct`` (property) converts ``cumulative_rate`` into a
    *signed P&L fraction* by direction:

      * LONG:  accrued_pct = -cumulative_rate   (negative = paid)
      * SHORT: accrued_pct = +cumulative_rate

    So a LONG charged a 0.03% positive rate once yields accrued_pct
    = -0.0003 (i.e. a 3bp drag on PnL).
    """
    signal_id: int
    symbol: str
    direction: str                     # "LONG" | "SHORT"
    entry_ts: float
    cumulative_rate: float = 0.0       # signed sum of rates per cycle
    cycles_recorded: int = 0
    last_cycle_ts: float = 0.0

    @property
    def accrued_pct(self) -> float:
        """Signed PnL impact as a fraction (e.g. -0.0003 = -3bps)."""
        sign = 1.0 if self.direction == "SHORT" else -1.0
        return sign * self.cumulative_rate


# ── Bridge ──────────────────────────────────────────────────────

class FundingBridge:
    """Maintains per-symbol funding history and assesses impact on signals.

    Also owns the per-signal accrual tracker (see ``start_accrual`` /
    ``record_cycle`` / ``get_accrued_pct`` / ``stop_accrual``).
    """

    def __init__(self) -> None:
        # symbol -> deque of (timestamp, rate) tuples
        self._history: Dict[str, deque] = {}

        # Per-signal cumulative funding tracker.
        # signal_id -> FundingAccrual
        self._accrual: Dict[int, FundingAccrual] = {}

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

    # ── Accrual tracking ────────────────────────────────────

    def start_accrual(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        entry_ts: Optional[float] = None,
    ) -> FundingAccrual:
        """Begin tracking cumulative funding for an activated position.

        Idempotent: if an accrual already exists for ``signal_id`` it is
        returned unchanged (so re-calling after an engine restore does
        not reset the counter).
        """
        existing = self._accrual.get(signal_id)
        if existing is not None:
            return existing
        direction_u = (direction or "").upper()
        if direction_u not in ("LONG", "SHORT"):
            # Log and default to LONG so the tracker never silently
            # reverses sign on bad input.
            logger.warning(
                f"funding_bridge.start_accrual: invalid direction "
                f"{direction!r} for {symbol} sid={signal_id} — defaulting to LONG"
            )
            direction_u = "LONG"
        accrual = FundingAccrual(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction_u,
            entry_ts=float(entry_ts) if entry_ts else time.time(),
        )
        self._accrual[signal_id] = accrual
        return accrual

    def record_cycle(self, signal_id: int, rate: float) -> Optional[FundingAccrual]:
        """Record one funding-cycle charge against ``signal_id``.

        ``rate`` is the signed per-cycle rate as reported by the exchange
        (e.g. 0.0001 for +0.01% positive). Returns the updated
        ``FundingAccrual`` or ``None`` if the signal isn't tracked.
        """
        acc = self._accrual.get(signal_id)
        if acc is None:
            return None
        try:
            r = float(rate)
        except (TypeError, ValueError):
            return acc
        acc.cumulative_rate += r
        acc.cycles_recorded += 1
        acc.last_cycle_ts = time.time()
        return acc

    def get_accrual(self, signal_id: int) -> Optional[FundingAccrual]:
        """Return the raw accrual record or None."""
        return self._accrual.get(signal_id)

    def get_accrued_pct(self, signal_id: int) -> float:
        """Return signed P&L fraction from cumulative funding, or 0.0."""
        acc = self._accrual.get(signal_id)
        return acc.accrued_pct if acc is not None else 0.0

    def get_accrued_r(
        self,
        signal_id: int,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """Return the P&L impact of accrued funding in R-multiples.

        R is the entry→SL distance. A LONG with entry=100, SL=98 has 1R
        = 2.00. If this position has accrued 0.06% funding drag
        (LONG paying 3×0.02%), the R impact is -0.0006*100 / 2 = -0.03R.
        """
        acc = self._accrual.get(signal_id)
        if acc is None or not entry_price or not stop_loss:
            return 0.0
        risk = abs(float(entry_price) - float(stop_loss))
        if risk <= 0:
            return 0.0
        pnl_price_delta = acc.accrued_pct * float(entry_price)
        return pnl_price_delta / risk

    def stop_accrual(self, signal_id: int) -> Optional[FundingAccrual]:
        """Remove and return the accrual record for ``signal_id``."""
        return self._accrual.pop(signal_id, None)


# ── Singleton ───────────────────────────────────────────────────
funding_bridge = FundingBridge()
