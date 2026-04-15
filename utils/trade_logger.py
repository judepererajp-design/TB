"""
utils/trade_logger.py — Structured trade audit logger.

Writes one pipe-delimited line per trading event to logs/trades.log.
Four event types cover the full signal lifecycle:

  SIGNAL  — every signal the aggregator considers, approved OR rejected
  TRIGGER — every execution engine state change that leads to ALMOST
  TRAIL   — every trailing stop ratchet after TP1
  OUTCOME — every final trade result with full entry/exit context

Usage (from any module):
    from utils.trade_logger import trade_logger
    trade_logger.signal(...)
    trade_logger.trigger(...)
    trade_logger.trail(...)
    trade_logger.outcome(...)

The dedicated "trades" logger is wired up in setup_logging() (main.py) so
that all writes go to logs/trades.log independently of the main log level.
"""

import logging
from typing import Optional

_tlog = logging.getLogger("trades")


class _TradeLogger:
    """Thin wrapper that formats and emits structured trade audit lines."""

    # ──────────────────────────────────────────────────────────────
    # 1. Signal considered (approved OR rejected)
    # ──────────────────────────────────────────────────────────────
    def signal(
        self,
        *,
        symbol: str,
        direction: str,
        grade: str,
        confidence: float,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        rr: float,
        strategy: str,
        regime: str,
        tech: float = 0.0,
        vol: float = 0.0,
        flow: float = 0.0,
        deriv: float = 0.0,
        sent: float = 0.0,
        corr: float = 0.0,
        result: str = "APPROVED",   # "APPROVED" or "REJECTED(REASON ...)"
    ) -> None:
        """Log a signal that was either approved or rejected by the aggregator."""
        _tlog.info(
            "SIGNAL | %s %s | grade=%s conf=%.1f"
            " | entry=%.4f-%.4f sl=%.4f tp1=%.4f tp2=%.4f"
            " | rr=%.2f | strategy=%s | regime=%s"
            " | tech=%.0f vol=%.0f flow=%.0f deriv=%.0f sent=%.0f corr=%.0f"
            " → %s",
            symbol, direction, grade, confidence,
            entry_low, entry_high, stop_loss, tp1, tp2,
            rr, strategy, regime,
            tech, vol, flow, deriv, sent, corr,
            result,
        )

    # ──────────────────────────────────────────────────────────────
    # 2. Execution trigger update (→ ALMOST)
    # ──────────────────────────────────────────────────────────────
    def trigger(
        self,
        *,
        signal_id: int,
        symbol: str,
        direction: str,
        structure_shift: bool,
        momentum: bool,
        liquidity: bool,
        rejection: bool,
        triggers_score: float,
        min_triggers: float,
        new_state: str,
    ) -> None:
        """Log an execution engine state transition (typically → ALMOST)."""
        _tlog.info(
            "TRIGGER | %s %s #%d"
            " | structure=%s momentum=%s liq=%s rejection=%s"
            " → triggers=%.1f/%.1f → %s",
            symbol, direction, signal_id,
            "✅" if structure_shift else "❌",
            "✅" if momentum else "❌",
            "✅" if liquidity else "❌",
            "✅" if rejection else "❌",
            triggers_score, min_triggers,
            new_state,
        )

    # ──────────────────────────────────────────────────────────────
    # 3. Trailing stop ratchet
    # ──────────────────────────────────────────────────────────────
    def trail(
        self,
        *,
        signal_id: int,
        symbol: str,
        direction: str,
        old_stop: float,
        new_stop: float,
        max_r: float,
        progress_pct: float,
        reason: str = "trail",   # "TP1_hit" or "trail"
    ) -> None:
        """Log a trailing stop ratchet (be_stop moved upward/downward)."""
        _tlog.info(
            "TRAIL | %s %s #%d | %s %.4f → %.4f (max_r=+%.1fR progress=%d%%)",
            symbol, direction, signal_id,
            reason, old_stop, new_stop,
            max_r, int(progress_pct * 100),
        )

    # ──────────────────────────────────────────────────────────────
    # 4. Final trade outcome
    # ──────────────────────────────────────────────────────────────
    def outcome(
        self,
        *,
        signal_id: int,
        symbol: str,
        direction: str,
        result: str,            # WIN | LOSS | BREAKEVEN | EXPIRED
        entry_price: float,
        exit_price: float,
        pnl_r: float,
        max_r: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        duration_min: float,
        strategy: str,
        regime: Optional[str] = None,
        entry_status: Optional[str] = None,   # IN_ZONE | LATE | EXTENDED
        exit_reason: Optional[str] = None,    # SL | TP1 | TP2 | TP3 | BE | TRAIL | EXPIRED
    ) -> None:
        """Log a terminal trade outcome with full context."""
        _tlog.info(
            "OUTCOME | %s %s #%d %s"
            " | entry=%.4f exit=%.4f"
            " | pnl=%+.2fR max_r=+%.2fR"
            " | sl=%.4f tp1=%.4f tp2=%.4f"
            " | duration=%.0fmin"
            " | strategy=%s regime=%s"
            " | entry_status=%s exit_reason=%s",
            symbol, direction, signal_id, result,
            entry_price, exit_price,
            pnl_r, max_r,
            stop_loss, tp1, tp2,
            duration_min,
            strategy, regime or "UNKNOWN",
            entry_status or "UNKNOWN", exit_reason or "UNKNOWN",
        )


trade_logger = _TradeLogger()
