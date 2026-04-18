"""
TitanBot Pro — Risk Manager
==============================
Handles all risk calculation and circuit breaker logic.

Responsibilities:
  - Position size calculation (fixed % + Kelly adjustment)
  - Per-signal risk validation (is R/R acceptable?)
  - Daily loss limit tracking
  - Circuit breaker (pause trading after drawdown)
  - Tier-based risk reduction for less liquid coins
  - Correlation adjustment (reduce size in same sector)
"""

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

from config.loader import cfg
from data.database import db
from signals.aggregator import ScoredSignal
from risk.circuit_breaker import circuit_breaker as CircuitBreaker
from governance.performance_tracker import performance_tracker

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Calculates position sizes and validates risk for each signal.
    All position sizing logic lives here.
    """

    def __init__(self):
        self._risk_cfg = cfg.risk
        self.circuit_breaker = CircuitBreaker
        self._daily_trades = 0          # Loaded from DB on startup via load_daily_count()
        self._day_start = self._get_day_start()
        # OrderedDict with a max-size cap: O(1) insertion and eviction of the oldest
        # entry vs the previous plain-dict + O(n) slice-and-delete approach.
        self._position_history: "OrderedDict[str, float]" = OrderedDict()
        self._position_history_max = 500

    async def load_daily_count(self):
        """FIX-5: Load today's published signal count from DB on restart.
        Prevents bot from exceeding daily limit after restart by forgetting
        signals sent earlier in the same UTC day."""
        try:
            from data.database import db
            count = await db.get_signals_today(exclude_c_grade=True)
            self._daily_trades = max(0, count)
            if count > 0:
                logger.info(f"Risk manager: loaded {count} signals from today → daily_trades={self._daily_trades}")
        except Exception as e:
            logger.debug(f"load_daily_count failed (non-fatal): {e}")

    def _get_day_start(self) -> float:
        """Start of current UTC day"""
        now = datetime.now(timezone.utc)
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()

    async def _reset_daily_if_needed(self):
        """Reset daily counters at midnight UTC"""
        if time.time() >= self._day_start + 86400:
            self._daily_trades = 0
            self._day_start = self._get_day_start()
            # FIX P1-C: also reset circuit breaker daily loss on midnight reset
            await self.circuit_breaker.reset_daily()
            logger.info("Daily risk counters reset (risk_manager + circuit_breaker)")

    # Fix R1: calculate_position_size removed — dead code. Use portfolio_engine.size_position() instead.

    async def validate_signal(self, scored: ScoredSignal) -> Tuple[bool, str]:
        """
        Final gate before a signal is published.
        Returns (approved, reason_if_rejected)
        """
        await self._reset_daily_if_needed()

        # Circuit breaker check
        if self.circuit_breaker.is_active:
            return False, "Circuit breaker active — trading paused"

        # Daily loss limit — use the public property to avoid coupling to internals
        max_daily_loss = self._risk_cfg.get('max_daily_loss_pct', 0.03)
        cb_daily_loss = self.circuit_breaker.daily_loss_pct
        if cb_daily_loss >= max_daily_loss:
            return False, f"Daily loss limit reached ({cb_daily_loss*100:.1f}%)"

        # Daily trade count — default matches settings.yaml (150), not 10
        max_daily = self._risk_cfg.get('max_daily_trades', 150)
        if self._daily_trades >= max_daily:
            return False, f"Daily trade limit reached ({self._daily_trades})"

        # LIQ-GUARD: Reject signals where the stop loss is too close to or below
        # the estimated liquidation price at configured leverage.  This prevents
        # the exchange liquidating the position before the stop loss fires —
        # which would mean the bot's % loss logic never executes.
        try:
            _base = scored.base_signal
            _leverage = float(self._risk_cfg.get('default_leverage', 10))
            _liq_buf  = float(self._risk_cfg.get('liq_min_buffer_pct', 0.02))
            if _leverage > 1 and _base.entry_low > 0 and _base.entry_high > 0:
                _entry_mid = (_base.entry_low + _base.entry_high) / 2.0
                _dir = getattr(_base.direction, 'value', str(_base.direction))
                if _dir == "LONG":
                    _liq_price = _entry_mid * (1.0 - 1.0 / _leverage)
                    _required_sl = _liq_price + _liq_buf * _entry_mid
                    if _base.stop_loss < _required_sl:
                        return False, (
                            f"Stop loss ${_base.stop_loss:.4f} is at or below estimated "
                            f"liquidation price ${_liq_price:.4f} (+ {_liq_buf*100:.0f}% buffer) "
                            f"at {_leverage:.0f}x leverage — position would be liquidated before SL fires"
                        )
                else:  # SHORT
                    _liq_price = _entry_mid * (1.0 + 1.0 / _leverage)
                    _required_sl = _liq_price - _liq_buf * _entry_mid
                    if _base.stop_loss > _required_sl:
                        return False, (
                            f"Stop loss ${_base.stop_loss:.4f} is at or above estimated "
                            f"liquidation price ${_liq_price:.4f} (- {_liq_buf*100:.0f}% buffer) "
                            f"at {_leverage:.0f}x leverage — position would be liquidated before SL fires"
                        )
        except Exception as _liq_err:
            logger.debug(f"Liquidation distance check skipped: {_liq_err}")

        # R/R minimum is checked in aggregator (setup_class-aware) — not duplicated here
        # NOTE: counter is NOT incremented here. Call commit_signal() after successful publish.
        return True, ""

    def commit_signal(self):
        """
        Increment the daily trade counter after a signal is successfully published.
        Called by engine AFTER publisher.publish() succeeds — not on attempt.
        Separating check from commit prevents self-DDoS where 320 symbols × 13 strategies
        exhaust the daily limit with attempts, blocking real executions.
        """
        self._daily_trades += 1

    async def record_outcome(self, pnl_r: float, symbol: str, signal_id: str = "",
                            current_capital: float = -1.0):
        """
        Record a trade outcome — updates daily tracking and circuit breaker.
        PHASE 1 FIX (P1-C): pass signal_id for dedup in circuit_breaker.
        FIX AUDIT-10: Accept and forward current_capital so the peak-equity
        drawdown check in circuit_breaker uses a fresh value. Previously
        this path always passed the default (-1.0) which meant the CB's
        peak-equity check used a stale _current_capital.
        """
        if pnl_r < 0:
            triggered = await self.circuit_breaker.record_loss(
                loss_r=abs(pnl_r), signal_id=signal_id,
                current_capital=current_capital, symbol=symbol,
            )
            if triggered:
                logger.warning(f"Circuit breaker triggered by losses on {symbol}")
        elif abs(pnl_r) < 1e-6:  # EC2/B7: float equality fix (was == 0.0)
            self.circuit_breaker.record_breakeven()
        else:
            self.circuit_breaker.record_win()

        self._position_history[symbol] = pnl_r
        # Move to end (most-recently-used) and evict oldest entry when over cap.
        # OrderedDict.move_to_end + popitem(last=False) is O(1) — no slice copy.
        self._position_history.move_to_end(symbol)
        while len(self._position_history) > self._position_history_max:
            self._position_history.popitem(last=False)

    # ── Regime transition risk adjustment ────────────────────────
    def get_regime_transition_kelly_mult(self) -> float:
        """
        Query regime_analyzer for transition warnings and return Kelly multiplier.

        When a regime transition is detected (≥2 of 4 factors: ADX decline,
        vol compressing, flow deceleration, OI/price exhaustion), reduce
        Kelly sizing to protect against inflection-point whipsaws.

        Returns 1.0 when no transition is active (no impact).
        """
        try:
            from analyzers.regime import regime_analyzer
            from config.constants import RegimeTransition as RT
            tw = regime_analyzer.get_transition_warning()
            if tw["warning"]:
                logger.info(
                    f"⚠️ Risk manager: regime transition active "
                    f"({tw['factor_count']} factors) — reducing Kelly to "
                    f"{RT.WARNING_KELLY_REDUCTION:.0%}"
                )
                return RT.WARNING_KELLY_REDUCTION
        except Exception as e:
            logger.debug(f"get_regime_transition_kelly_mult: {e}")
        return 1.0

    # ── VOLATILE_PANIC position review ──────────────────────────
    def check_panic_position_review(self) -> dict:
        """
        Check if VOLATILE_PANIC is active and return position review actions.

        When VOLATILE_PANIC is detected with existing open positions:
        - Positions past TP1 → recommend tighten SL to breakeven
        - All positions → flag for manual review

        Returns dict with:
            panic_active: bool
            positions_to_tighten: list of signal IDs past TP1
            total_open: int
        """
        result = {
            "panic_active": False,
            "positions_to_tighten": [],
            "total_open": 0,
        }
        try:
            from analyzers.regime import regime_analyzer
            regime_val = getattr(regime_analyzer.regime, 'value', '') if regime_analyzer.regime else ''
            if "VOLATILE_PANIC" not in regime_val:
                return result

            result["panic_active"] = True

            from signals.outcome_monitor import outcome_monitor
            active = outcome_monitor.get_active_signals()
            result["total_open"] = len(active)

            for sig_id, tracked in active.items():
                # Check if TP1 has been hit (BE_ACTIVE state means TP1 was reached)
                if getattr(tracked, 'state', '') == 'BE_ACTIVE':
                    result["positions_to_tighten"].append(sig_id)

            if result["total_open"] > 0:
                logger.warning(
                    f"🚨 VOLATILE_PANIC position review: {result['total_open']} open, "
                    f"{len(result['positions_to_tighten'])} past TP1 to tighten"
                )
        except Exception as e:
            logger.debug(f"check_panic_position_review: {e}")
        return result


# ── Singleton ──────────────────────────────────────────────
risk_manager = RiskManager()
