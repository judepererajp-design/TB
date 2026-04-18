"""
TitanBot Pro — Signal Publisher
=================================
Decouples signal publishing from the engine.

Architecture:
  Engine → builds signal data object
  Engine → calls publisher.publish(signal_data)
  Publisher → handles ALL output: Telegram, DB update, outcome tracking, logging

This ensures:
  1. Formatting bugs never crash the engine
  2. Engine only deals with pure data objects
  3. Publisher can be swapped (Telegram → Discord, webhook, etc.)
  4. All publish logic is in one place (single responsibility)
"""

import logging
import time
from typing import Dict, Optional

from config.constants import RateLimiting
from utils.formatting import fmt_price

logger = logging.getLogger(__name__)


class SignalPublisher:
    # V14→V17: Signal burst throttle — was 2/15min, far too aggressive.
    # 2 per 15min means a max of 8 signals/hour even if 200+ symbols are scanning.
    # Raised to 4/10min = max 24/hour which is realistic for a large universe.
    MAX_SIGNALS_PER_WINDOW = RateLimiting.PUBLISHER_MAX_PER_WINDOW
    WINDOW_SECONDS = RateLimiting.PUBLISHER_WINDOW_SECS
    """
    Publishes scored signals to Telegram and registers them
    with outcome/invalidation monitors.

    The engine calls publish() with a pure data dict —
    no Telegram objects or formatter dependencies leak into engine.
    """

    def __init__(self):
        self._published_count = 0
        self._recent_publish_times: list = []  # V14: timestamps of recent publishes
        self._bot = None  # optional bot reference (lazy import used in publish)
        # Per-symbol publish timestamps for the telegram.min_signal_interval gate.
        # Maps symbol -> last successful publish time (unix seconds).
        self._last_symbol_publish: Dict[str, float] = {}

    @staticmethod
    def _per_symbol_cooldown_secs() -> int:
        """Return telegram.min_signal_interval in seconds (0 disables the gate)."""
        try:
            from config.loader import cfg
            raw = cfg.telegram.get("min_signal_interval", 0)
            val = int(raw)
            return val if val > 0 else 0
        except Exception:
            return 0

    def _is_symbol_cooldown(self, symbol: str, grade: str) -> tuple:
        """Check the per-symbol cooldown gate.

        Returns (blocked: bool, remaining_secs: int). A+ signals bypass
        the cooldown (same policy as the burst throttle) so high-conviction
        setups are never dropped by a lower-tier alert that fired seconds earlier.
        """
        cooldown = self._per_symbol_cooldown_secs()
        if cooldown <= 0 or grade == "A+":
            return False, 0
        import time as _t
        last = self._last_symbol_publish.get(symbol, 0.0)
        if last <= 0:
            return False, 0
        elapsed = _t.time() - last
        if elapsed >= cooldown:
            return False, 0
        return True, int(cooldown - elapsed)

    def set_bot(self, bot) -> None:
        """Accept bot reference for API compatibility with publisher.py shim.
        signal_publisher.py imports the bot lazily inside publish() so this
        is stored but not strictly required."""
        self._bot = bot

    def _is_throttled(self, grade: str) -> bool:
        """Return True if a signal with the given grade would be throttled.

        Shared logic used by both would_throttle() (pre-save check) and
        publish() (actual publish gate) to avoid duplication.
        """
        import time as _t
        _now = _t.time()
        self._recent_publish_times = [
            t for t in self._recent_publish_times
            if _now - t < self.WINDOW_SECONDS
        ]
        _is_priority = grade == "A+"
        return not _is_priority and len(self._recent_publish_times) >= self.MAX_SIGNALS_PER_WINDOW

    def would_throttle(self, grade: str) -> bool:
        """Check if a signal with the given grade would be throttled.

        Called by the engine BEFORE db.save_signal() so that throttled
        signals are never written to the database (prevents "NOT SENT"
        phantom entries in Signal History).
        """
        return self._is_throttled(grade)

    async def publish(
        self,
        scored,              # ScoredSignal object
        signal,              # Original SignalResult
        signal_id: int,      # DB signal ID
        sig_data: Dict,      # Full signal data dict
        alpha_score,         # AlphaScore from alpha model
        prob_estimate,       # ProbabilityEstimate
        sizing,              # PositionSizing result
        confluence,          # ConfluenceResult
    ) -> Optional[int]:
        """
        Publish a signal: Telegram → outcome tracking → invalidation tracking.

        Returns message_id if published successfully, None otherwise.
        Signal is ALWAYS in the database regardless of publish success.
        """
        # V14: Burst throttle — max signals per window. Uses shared _is_throttled()
        # to keep logic in sync with would_throttle() pre-check.
        import time as _pub_time
        _now = _pub_time.time()
        _grade = getattr(alpha_score, 'grade', None) or "?"
        _dir_str = getattr(signal.direction, 'value', str(signal.direction))

        # Per-symbol cooldown (telegram.min_signal_interval). Prevents two back-to-back
        # alerts on the same symbol (e.g. from different strategies scoring the same
        # setup within seconds). A+ signals bypass (see _is_symbol_cooldown).
        _sym_blocked, _sym_remain = self._is_symbol_cooldown(signal.symbol, _grade)
        if _sym_blocked:
            logger.info(
                f"⏸️ Signal throttled (per-symbol): {signal.symbol} {_dir_str} "
                f"grade={_grade} — last publish {_sym_remain}s ago "
                f"(min_signal_interval={self._per_symbol_cooldown_secs()}s)"
            )
            return None

        if self._is_throttled(_grade):
            logger.info(
                f"⏸️ Signal throttled: {signal.symbol} {_dir_str} "
                f"grade={_grade} — {len(self._recent_publish_times)} signals in "
                f"last {self.WINDOW_SECONDS//60}min (max {self.MAX_SIGNALS_PER_WINDOW})"
            )
            return None
        # NOTE: _recent_publish_times.append() moved to AFTER confirmed publish below

        from tg.bot import telegram_bot
        from data.database import db

        # ── 1. Publish to Telegram (sandboxed) ────────────────
        _publish_price = float(sig_data.get('publish_price') or 0.0)
        msg_id = None
        try:
            msg_id = await telegram_bot.publish_signal(
                scored, signal_id,
                alpha_score=alpha_score,
                prob_estimate=prob_estimate,
                sizing=sizing,
                confluence=confluence,
                current_price=_publish_price,
            )
        except Exception as pub_err:
            logger.error(
                f"⚠️ Telegram publish failed for {signal.symbol} "
                f"(signal #{signal_id} saved to DB): {pub_err}"
            )
            # Send raw fallback so signal is not lost
            try:
                import time as _fb_time
                _grade    = alpha_score.grade if alpha_score else scored.grade
                _pwin     = f"{prob_estimate.p_win*100:.0f}%" if prob_estimate else "?"
                _ev       = f"{alpha_score.expected_value_r:+.2f}R" if alpha_score else "?"
                _regime   = getattr(scored, 'regime', None) or getattr(signal, 'regime', 'UNKNOWN')
                _dir      = getattr(signal.direction, 'value', str(signal.direction))
                _grade_hd = {"A+": "🟢 A+", "A": "🟡 A", "B": "🔵 B"}.get(_grade, "⚪")
                _dir_arr  = "LONG ↑" if _dir == "LONG" else "SHORT ↓"
                _entry_mid = (signal.entry_low + signal.entry_high) / 2
                # Compute expiry
                try:
                    from tg.formatter import STRATEGY_EXPIRY_CANDLES, TF_MINUTES, _expiry_str
                    _entry_tf = getattr(signal, 'entry_timeframe', '') or getattr(signal, 'timeframe', '1h')
                    _expiry   = _expiry_str(signal.strategy, _entry_tf, _fb_time.time())
                except Exception:
                    _expiry = "N/A"
                # TP lines
                _tp_lines = [f"🎯 TP1: {fmt_price(signal.tp1)}"]
                if signal.tp2:
                    _tp_lines.append(f"🎯 TP2: {fmt_price(signal.tp2)}  ← main target")
                if signal.tp3:
                    _tp_lines.append(f"🎯 TP3: {fmt_price(signal.tp3)}")
                fallback = (
                    f"{_grade_hd} <b>SIGNAL</b> | {signal.symbol} {_dir_arr}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Strategy: {signal.strategy}\n"
                    f"Regime: {_regime}\n"
                    f"P(Win): {_pwin} │ EV: {_ev} │ Grade: <b>{_grade}</b>\n"
                    f"\n"
                    f"📍 Entry: {fmt_price(signal.entry_low)} – {fmt_price(signal.entry_high)}\n"
                    f"🛑 Stop:  {fmt_price(signal.stop_loss)}\n"
                    + "\n".join(_tp_lines) + "\n"
                    f"R:R: {signal.rr_ratio:.1f}:1\n"
                    f"\n"
                    f"⏳ Valid until: {_expiry}\n"
                    f"Signal #{signal_id}\n"
                    f"<i>(Fallback format — check logs)</i>"
                )
                msg = await telegram_bot.send_signals_text(text=fallback)
                msg_id = msg.message_id
            except Exception:
                pass  # Even fallback failed — signal still in DB

        if not msg_id:
            return None

        # FIX: record throttle slot only after confirmed successful publish
        self._recent_publish_times.append(_now)
        # Track per-symbol publish time for the min_signal_interval gate.
        self._last_symbol_publish[signal.symbol] = _now

        # ── 2. Update DB with message ID ──────────────────────
        try:
            await db.update_signal_message_id(signal_id, msg_id)
        except Exception as e:
            logger.debug(f"DB message_id update failed: {e}")

        # ── 3. Grade-based tracking registration ──────────────
        # PHASE 3 AUDIT FIX: outcome_monitor and execution_engine registration
        # is handled by engine.py AFTER publish succeeds. Previously, the
        # publisher also registered A/B signals with outcome_monitor, creating
        # a duplicate PENDING row in tracked_signals_v1 that raced with the
        # execution_engine row written by engine.py. Removed to eliminate the
        # dual-tracking bug (Finding P3-1).
        #
        # The engine routes:
        #   C  → no tracking (informational)
        #   A+ → outcome_monitor directly (immediate entry) or execution_engine in VOLATILE
        #   A/B+/B → execution_engine → outcome_monitor on EXECUTE state
        _grade = alpha_score.grade if alpha_score else "C"
        if _grade == "C":
            logger.debug(f"Signal #{signal_id} is grade C — skipping tracking")
            return msg_id

        # ── 4. Register with invalidation monitor ─────────────
        try:
            from signals.invalidation_monitor import invalidation_monitor
            from tg.formatter import STRATEGY_EXPIRY_CANDLES, TF_MINUTES
            # Sync timeout with the expiry timer shown in the signal card
            expiry_candles = STRATEGY_EXPIRY_CANDLES.get(signal.strategy, 6)
            tf_mins = TF_MINUTES.get(getattr(signal, 'entry_timeframe', signal.timeframe), 60)
            timeout_secs = expiry_candles * tf_mins * 60
            invalidation_monitor.track_signal(
                signal_id=signal_id,
                symbol=signal.symbol,
                direction=getattr(signal.direction, 'value', str(signal.direction)),
                strategy=signal.strategy,
                entry_low=signal.entry_low,
                entry_high=signal.entry_high,
                stop_loss=signal.stop_loss,
                confidence=scored.final_confidence,
                message_id=msg_id,
                grade=alpha_score.grade if alpha_score else "B",
                setup_class=getattr(signal, 'setup_class', 'intraday'),
                timeout_seconds=timeout_secs,
                raw_data=getattr(signal, 'raw_data', None),    # BUG-14: for strategy re-validation
                regime_at_publish=getattr(signal, 'regime', None),
                tp1=signal.tp1, tp2=signal.tp2, tp3=signal.tp3,
                rr_ratio=signal.rr_ratio,
            )
            # Notify of potential counter-signals
            invalidation_monitor.notify_counter_signal(
                signal.symbol, getattr(signal.direction, 'value', str(signal.direction))
            )
        except Exception as e:
            logger.error(f"Invalidation monitor registration failed: {e}")

        # ── 5. Log published signal ───────────────────────────
        self._published_count += 1
        # Notify proactive alert engine — checks if any open trade is opposite direction
        try:
            import asyncio as _asyncio
            from signals.proactive_alerts import proactive_alerts as _pa
            _asyncio.create_task(
                _pa.notify_opposing_signal(
                    signal.symbol,
                    getattr(signal.direction, 'value', str(signal.direction)),
                    signal.strategy or "",
                    scored.final_confidence,
                )
            )
        except Exception:
            pass
        _grade_log = alpha_score.grade if alpha_score else "?"
        _ev_log = alpha_score.expected_value_r if alpha_score else 0.0
        _pwin_log = prob_estimate.p_win if prob_estimate else 0.0
        _size_log = sizing.position_size_usdt if sizing else 0.0
        _conf_log = len(confluence.agreeing_strategies) if confluence else 0
        logger.info(
            f"📊 Signal published: {signal.symbol} "
            f"{getattr(signal.direction, 'value', str(signal.direction))} "
            f"grade={_grade_log} "
            f"P(win)={_pwin_log:.2f} "
            f"EV={_ev_log:+.3f}R "
            f"size=${_size_log:,.0f} "
            f"confluence={_conf_log} strategies"
        )

        return msg_id

    @property
    def published_count(self) -> int:
        return self._published_count


# ── Singleton ──────────────────────────────────────────────
signal_publisher = SignalPublisher()
