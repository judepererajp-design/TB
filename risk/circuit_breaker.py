"""
TitanBot Pro — Circuit Breaker
================================
Protects capital by halting signals when:
  - N consecutive losses in an hour
  - Daily drawdown exceeds threshold
  - Intraday peak-equity drawdown exceeds threshold
  - Absolute daily loss cap breached (hard kill — no auto-resume)
  - Manual pause via Telegram

PHASE 1 FIX (P1-C):
  record_loss() now deduplicates by signal_id within a 10-second window.
  Previously a single trade outcome triggered two consecutive record_loss()
  calls (once from outcome_monitor, once from engine.py outcome handler),
  doubling all risk counters — consecutive losses, hourly losses, and daily
  drawdown were all 2× the real values.

BUG-FIX batch (risk audit):
  BUG-1: Peak-equity drawdown check was dead — actual_loss_pct was never
    passed by callers so the condition (actual_loss_pct >= 0) was permanently
    False. Fixed by accepting current_capital instead; CB computes the
    drawdown internally from _peak_capital so callers cannot bypass the check.
  BUG-3: _daily_loss_pct was reset on every cooldown expiry — effectively
    resetting the daily limit after each 45-min pause, allowing unlimited
    intraday loss. Fixed: removed the reset from the is_active auto-clear path.
    reset_daily() (midnight) is the only correct reset point.
  BUG-4: _auto_clear() was dead code (logic duplicated into is_active
    property). Removed to eliminate confusion.
  ISSUE-8: No hard kill switch — on a very bad day consecutive cooldowns
    could add up to far more than max_drawdown_pct. Added
    max_absolute_day_loss_pct (default 20%) which fires CB with no cooldown
    (requires midnight reset_daily() to clear automatically).

C3 FIX (persistence):
  _daily_loss_pct, _peak_capital, _hard_kill_active, and _resume_at are now
  persisted to risk_state_v1 after every state change and restored via
  restore() at startup. Previously a restart after a hard-kill would silently
  clear all these values and resume trading with a blank slate.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from config.loader import cfg

logger = logging.getLogger(__name__)
risk_log = logging.getLogger("risk")


class CircuitBreaker:
    """
    Monitors trade outcomes and halts signal publishing when
    risk thresholds are breached. Safe-fail design.
    """

    def __init__(self):
        self._cb_cfg = cfg.risk.circuit_breaker

        self._is_active        = False
        self._reason           = ""
        self._activated_at: Optional[float] = None
        self._resume_at: Optional[float]    = None
        self._consecutive_losses = 0
        self._losses_this_hour   = 0
        self._hour_window_start  = time.time()
        self._daily_loss_pct     = 0.0

        # T3-FIX: Peak-equity intraday drawdown kill-switch.
        self._peak_capital: float    = 0.0
        self._current_capital: float = 0.0
        self._max_peak_drawdown: float = self._cb_cfg.get('max_peak_drawdown_pct', 0.12)

        # ISSUE-8: Hard kill switch — absolute daily loss cap with no cooldown.
        self._max_absolute_day_loss: float = self._cb_cfg.get('max_absolute_day_loss_pct', 0.20)
        self._hard_kill_active: bool = False

        self._auto_clear_scheduled = False

        # P1-C: dedup guard — signal_id -> timestamp of last record_loss call
        self._recent_loss_ids: Dict[str, float] = {}
        self._dedup_window: float = 10.0  # seconds
        # R-1 FIX: protect _recent_loss_ids with a lock so that simultaneous
        # calls from outcome_monitor and the engine outcome handler cannot both
        # pass the dedup check before either writes back the updated timestamp.
        # Without the lock, two concurrent record_loss() calls for the same
        # signal_id can read the same (missing) entry, both decide "not seen",
        # and both write — doubling every counter.
        self._dedup_lock: asyncio.Lock = asyncio.Lock()

        self._last_loss_time: float = 0.0

        self.on_triggered: Optional[Callable] = None
        self.on_cleared:   Optional[Callable] = None

        self._max_consecutive = self._cb_cfg.get('max_consecutive_losses', 3)
        self._max_per_hour    = self._cb_cfg.get('max_losses_per_hour', 4)
        self._cooldown_mins   = self._cb_cfg.get('cooldown_minutes', 45)
        self._max_drawdown    = self._cb_cfg.get('max_drawdown_pct', 0.08)

    # ── C3 FIX: Persistent state restore / save ───────────────

    async def restore(self) -> None:
        """Restore circuit breaker risk state from DB after a restart.

        Called once at engine startup (after db.initialize()).  Reads
        risk_state_v1 and reinstates daily_loss_pct, peak_capital,
        hard_kill_active, resume_at, and consecutive_losses so a restart
        mid-day cannot silently clear a hard-kill or ongoing drawdown.

        Edge cases handled:
          - resume_at in the past  → state expired during downtime, clear safely.
          - hard_kill + past midnight → still kept (reset_daily clears it).
          - No row in DB yet       → silent no-op (first ever startup).
        """
        try:
            from data.database import db as _db
            row = await _db.load_risk_state()
        except Exception as e:
            logger.warning(f"CircuitBreaker.restore: DB load failed ({e}) — starting fresh")
            return

        if row is None:
            return  # First startup, no state to restore

        self._daily_loss_pct     = float(row.get('daily_loss_pct', 0.0))
        self._peak_capital       = float(row.get('peak_capital',   0.0))
        self._hard_kill_active   = bool(row.get('hard_kill',       0))
        self._consecutive_losses = int(row.get('consecutive',      0))

        resume_at = row.get('resume_at')
        now = time.time()

        if resume_at is not None:
            resume_at = float(resume_at)
            if resume_at > now:
                # Active cooldown that hasn't expired yet — reinstate it.
                self._is_active  = True
                self._resume_at  = resume_at
                self._reason     = "Restored from pre-restart circuit breaker state"
                logger.warning(
                    f"🚨 Circuit breaker restored: active cooldown, "
                    f"resumes at {self.resume_time_str}"
                )
            else:
                # Cooldown window passed while bot was down — clear gracefully.
                logger.info(
                    "Circuit breaker: cooldown window expired during downtime — "
                    "state cleared (restart-safe)"
                )
                # Keep daily_loss_pct and hard_kill — those are day-scoped, not
                # cooldown-scoped. Only clear the active flag and resume_at.
                self._is_active = False
                self._resume_at = None
        elif self._hard_kill_active:
            # Hard kill with no resume_at (indefinite) — reinstate.
            self._is_active = True
            self._reason    = "Restored: hard kill active (daily loss cap breached)"
            logger.warning("🚨 Circuit breaker restored: hard kill still active")

        logger.info(
            f"CircuitBreaker restored — daily_loss: {self._daily_loss_pct*100:.2f}%, "
            f"hard_kill: {self._hard_kill_active}, consecutive: {self._consecutive_losses}, "
            f"active: {self._is_active}"
        )

    async def _save_state(self) -> None:
        """Persist current risk state to risk_state_v1.

        Called after every state-changing operation so restarts always
        have a fresh snapshot.  Failures are logged but never raised —
        the circuit breaker must not be blocked by a DB write error.
        """
        try:
            from data.database import db as _db
            await _db.save_risk_state(
                daily_loss_pct=self._daily_loss_pct,
                peak_capital=self._peak_capital,
                hard_kill=self._hard_kill_active,
                resume_at=self._resume_at,
                consecutive=self._consecutive_losses,
            )
        except Exception as e:
            logger.warning(f"CircuitBreaker._save_state failed (non-fatal): {e}")

    @property
    def daily_loss_pct(self) -> float:
        """Public read-only view of today's accumulated loss percentage."""
        return self._daily_loss_pct

    # ── T7: Drawdown-based size reduction ─────────────────────
    @property
    def position_size_multiplier(self) -> float:
        """
        T7: Returns a position size multiplier based on consecutive losses.
        - 0 losses     → 1.0x (full size)
        - 2 cons. losses → 0.5x (half size — reduce aggression after 2 losses)
        - 3+ losses    → circuit breaker fires (no new signals)
        """
        if self._consecutive_losses >= 2:
            return 0.5
        return 1.0

    @property
    def is_reduced_size(self) -> bool:
        """T7: True if position size is currently reduced due to drawdown."""
        return self._consecutive_losses >= 2

    # ── State checks ─────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        if self._is_active and self._resume_at:
            if time.time() >= self._resume_at:
                self._is_active = False
                self._reason = ""
                self._resume_at = None
                self._consecutive_losses = 0
                self._losses_this_hour = 0
                # BUG-3 FIX: do NOT reset _daily_loss_pct here.
                self._auto_clear_scheduled = True
                logger.info("Circuit breaker auto-cleared (cooldown elapsed)")
        return self._is_active

    async def tick(self):
        """Call periodically to fire on_cleared callback after auto-reset."""
        if not self._is_active and self._auto_clear_scheduled:
            self._auto_clear_scheduled = False
            if self.on_cleared:
                try:
                    await self.on_cleared()
                except Exception as e:
                    logger.error(f"Circuit breaker clear callback failed: {e}")

    @property
    def resume_time_str(self) -> str:
        if self._resume_at:
            dt = datetime.fromtimestamp(self._resume_at, tz=timezone.utc)
            return dt.strftime("%H:%M UTC")
        return "—"

    @property
    def reason(self) -> str:
        return self._reason

    # ── Loss recording ────────────────────────────────────────

    async def record_loss(
        self,
        loss_r: float = 1.0,
        signal_id: str = "",
        current_capital: float = -1.0,
    ) -> bool:
        """
        Record a trade loss. Returns True if circuit breaker triggered.
        """
        now = time.time()

        # ── 1. SIGNAL DEDUP ──────────────────────────────────────────────
        if signal_id:
            # R-1 FIX: hold the dedup lock for the entire read-check-write
            # sequence so two concurrent callers (outcome_monitor + engine
            # outcome handler) cannot both pass the "not seen" check before
            # either writes back the timestamp, which would double-count the loss.
            async with self._dedup_lock:
                last_seen = self._recent_loss_ids.get(signal_id, 0)
                if now - last_seen < self._dedup_window:
                    logger.warning(
                        f"Circuit breaker: duplicate loss record ignored "
                        f"(signal_id={signal_id}, {now - last_seen:.1f}s ago)"
                    )
                    return False
                self._recent_loss_ids[signal_id] = now

        # Always prune stale dedup entries (not just when signal_id is provided)
        if self._recent_loss_ids:
            cutoff = now - 3600
            stale = [k for k, v in self._recent_loss_ids.items() if v <= cutoff]
            for k in stale:
                del self._recent_loss_ids[k]

        # ── 2. LOSS COUNTING ──────────────────────────────────────────────
        if now - self._hour_window_start > 3600:
            self._losses_this_hour  = 0
            self._hour_window_start = now

        self._consecutive_losses += 1
        self._last_loss_time      = now
        self._losses_this_hour   += 1

        if current_capital >= 0:
            self._current_capital = current_capital

        self._daily_loss_pct += cfg.risk.get('risk_per_trade', 0.005) * loss_r

        logger.info(
            f"Loss recorded — consecutive: {self._consecutive_losses}, "
            f"this hour: {self._losses_this_hour}, "
            f"daily loss: {self._daily_loss_pct*100:.2f}%"
        )

        # C3 FIX: persist updated counters after every unique loss so restarts
        # pick up the correct daily_loss_pct, consecutive count, etc.
        await self._save_state()
        risk_log.info(
            "LOSS | consecutive=%d | hourly=%d | daily_loss=%.2f%% | hard_kill=%s",
            self._consecutive_losses, self._losses_this_hour,
            self._daily_loss_pct * 100, self._hard_kill_active,
        )

        if self._consecutive_losses >= self._max_consecutive:
            await self._trigger(
                f"{self._consecutive_losses} consecutive losses — "
                f"taking a break to let market settle"
            )
            return True

        if self._losses_this_hour >= self._max_per_hour:
            await self._trigger(
                f"{self._losses_this_hour} losses in one hour — "
                f"market conditions unfavorable"
            )
            return True

        # ── ISSUE-8: Hard absolute daily loss kill switch ────────────────────
        # FIX AUDIT-1: Check absolute daily loss cap FIRST — it must fire before
        # the softer max_drawdown check (which only applies a cooldown).  Previously
        # max_drawdown (e.g. 8%) always returned True before max_absolute_day_loss
        # (e.g. 20%) could ever execute, so _hard_kill_active was never set and
        # trading resumed after cooldown even on catastrophic loss days.
        if self._daily_loss_pct >= self._max_absolute_day_loss:
            self._hard_kill_active = True
            await self._trigger(
                f"Absolute daily loss cap breached ({self._daily_loss_pct*100:.1f}%) — "
                f"trading suspended until midnight",
                cooldown_minutes=0,
            )
            return True

        if self._daily_loss_pct >= self._max_drawdown:
            await self._trigger(
                f"Daily drawdown reached {self._daily_loss_pct*100:.1f}% — "
                f"daily limit hit, resuming tomorrow"
            )
            return True

        # ── T3-FIX (BUG-1 FIX): Peak-equity drawdown check ─────────────────
        if self._peak_capital > 0 and self._current_capital > 0:
            _peak_dd = (self._peak_capital - self._current_capital) / self._peak_capital
            if _peak_dd >= self._max_peak_drawdown:
                await self._trigger(
                    f"Peak-equity drawdown reached {_peak_dd*100:.1f}% "
                    f"(from intraday high ${self._peak_capital:,.0f}) — "
                    f"protecting gains"
                )
                return True

        return False

    def update_peak_capital(self, current_capital: float):
        """Call whenever capital changes to track intraday peak.

        R-2 FIX: when _peak_capital is 0 (first call after startup or midnight
        reset, before the scan loop refreshes the value), seed it from
        current_capital so the peak-equity drawdown gate is live immediately.
        Without this, the guard ``if self._peak_capital > 0`` in record_loss()
        is permanently False until the first update_peak_capital() call — leaving
        the kill-switch dead for an unknown window that can span the bot's entire
        startup phase.
        """
        self._current_capital = current_capital
        if current_capital > self._peak_capital:
            self._peak_capital = current_capital

    def record_win(self):
        """Record a winning trade — resets consecutive loss counter."""
        self._consecutive_losses = 0
        self._last_loss_time  = 0.0
        logger.debug("Win recorded — consecutive loss counter reset")

    def record_breakeven(self):
        """Record a breakeven trade — does NOT reset consecutive loss counter."""
        logger.debug("Breakeven recorded — consecutive loss counter NOT reset")

    async def reset_daily(self):
        """Call at midnight to reset daily counters.
        C3 FIX: persists cleared state so the next startup sees a clean slate.
        """
        self._daily_loss_pct     = 0.0
        self._losses_this_hour   = 0
        self._consecutive_losses = 0
        self._hour_window_start  = time.time()
        self._last_loss_time     = 0.0
        self._peak_capital       = 0.0
        self._current_capital    = 0.0
        if self._hard_kill_active:
            self._hard_kill_active = False
            self._is_active        = False
            self._reason           = ""
            self._resume_at        = None
            logger.info("Hard kill cleared — new trading day started")
            risk_log.info("DAILY_RESET | hard_kill_cleared=True")
        else:
            risk_log.info("DAILY_RESET | hard_kill_cleared=False")
        logger.info("Circuit breaker daily counters reset")

        # C3 FIX: persist the reset so a restart after midnight never
        # restores yesterday's hard-kill or daily_loss_pct.
        try:
            await self._save_state()
        except Exception as _save_err:
            logger.warning(f"Failed to persist daily reset state: {_save_err}")

    # ── Manual control ────────────────────────────────────────

    async def manually_pause(self, reason: str = "Manual pause via Telegram"):
        await self._trigger(reason, cooldown_minutes=0)

    async def manually_resume(self):
        if self._is_active:
            self._is_active  = False
            self._reason     = ""
            self._resume_at  = None
            # Clear dedup dict so fresh loss records aren't blocked by stale entries
            self._recent_loss_ids.clear()
            logger.warning("Circuit breaker manually overridden by user")
            risk_log.info("MANUAL_RESUME | circuit breaker cleared by user")
            # C3 FIX: persist cleared state immediately.
            await self._save_state()
            if self.on_cleared:
                await self.on_cleared()

    # ── Internal ──────────────────────────────────────────────

    async def _trigger(self, reason: str, cooldown_minutes: Optional[int] = None):
        if self._is_active:
            return
        cooldown = cooldown_minutes if cooldown_minutes is not None else self._cooldown_mins
        self._is_active    = True
        self._reason       = reason
        self._activated_at = time.time()
        self._resume_at    = (time.time() + cooldown * 60) if cooldown else None

        resume_str = self.resume_time_str if self._resume_at else "Manual resume required"
        logger.warning(f"🚨 Circuit breaker activated: {reason} | Resume: {resume_str}")
        risk_log.info(
            "CB_TRIP | %s | resume=%s | hard_kill=%s | daily_loss=%.2f%%",
            reason, resume_str, self._hard_kill_active, self._daily_loss_pct * 100,
        )

        # C3 FIX: persist the triggered state (including resume_at) so restarts
        # within the cooldown window keep the breaker active.
        await self._save_state()

        if self.on_triggered:
            try:
                await self.on_triggered(reason, resume_str)
            except Exception as e:
                logger.error(f"Circuit breaker callback failed: {e}")

    def get_status(self) -> dict:
        return {
            'is_active':           self._is_active,
            'reason':              self._reason,
            'resume_time':         self.resume_time_str,
            'consecutive_losses':  self._consecutive_losses,
            'losses_this_hour':    self._losses_this_hour,
            'daily_loss_pct':      round(self._daily_loss_pct * 100, 2),
            'peak_capital':        round(self._peak_capital, 2),
            'current_capital':     round(self._current_capital, 2),
            'hard_kill_active':    self._hard_kill_active,
        }


# ── Singleton ──────────────────────────────────────────────
circuit_breaker = CircuitBreaker()
