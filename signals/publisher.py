"""
TitanBot Pro — Signal Publisher (v2 — Interrupt Worthiness Filtering)
======================================================================
DEPRECATED (B11 fix): This file is a legacy routing shim.
The active publisher is signals/signal_publisher.py.
New code should import from signals.signal_publisher instead.
======================================================================
Routing logic:

  A+ / A / B  →  Signals channel immediately (EXECUTION format)
  C (high UP) →  Signals channel (CONTEXT format, no trade numbers)
  C (low UP)  →  Admin channel only (silent, no user interruption)

The Upgrade Tracker handles:
  - Computing Upgrade Probability for C signals
  - Logging all decisions for Phase 2 learning
  - Firing upgrade messages when C → B/A/A+

All other signals (whale, watchlist, tier) → admin channel unchanged.
"""

import logging
from typing import Optional

from config.loader import cfg
from data.database import db
from signals.aggregator import ScoredSignal
from signals.upgrade_tracker import upgrade_tracker

logger = logging.getLogger(__name__)


class SignalPublisher:
    """
    Coordinates between the signal pipeline and Telegram.
    Saves every signal to DB regardless of routing decision.
    """

    def __init__(self):
        self._bot = None

    def set_bot(self, bot):
        self._bot = bot

    async def publish(self, scored: ScoredSignal) -> bool:
        """
        Save signal to DB and route to appropriate channel.
        Returns True if sent to signals channel (user-facing).
        """
        if self._bot is None:
            logger.error("Publisher has no bot — call set_bot() first")
            return False

        sig = scored.base_signal
        direction = getattr(sig.direction, 'value', str(sig.direction)) if hasattr(sig.direction, 'value') else str(sig.direction)

        # ── 1. Save to database first (always) ───────────────────────
        signal_record = {
            'symbol':     sig.symbol,
            'direction':  direction,
            'strategy':   sig.strategy,
            'confidence': scored.final_confidence or sig.confidence,
            'entry_low':  sig.entry_low,
            'entry_high': sig.entry_high,
            'stop_loss':  sig.stop_loss,
            'tp1':        sig.tp1,
            'tp2':        sig.tp2,
            'tp3':        sig.tp3,
            'rr_ratio':   sig.rr_ratio,
            'setup_class': getattr(sig, 'setup_class', 'intraday'),
            'entry_timeframe': getattr(sig, 'entry_timeframe', '15m'),
            'timeframe': getattr(sig, 'entry_timeframe', sig.timeframe),  # back-compat
            'regime':     sig.regime or scored.regime,
            'sector':     sig.sector,
            'tier':       sig.tier,
            'confluence': scored.all_confluence,
            'grade':      scored.grade,
            'raw_scores': {
                'technical':   scored.technical_score,
                'volume':      scored.volume_score,
                'orderflow':   scored.orderflow_score,
                'derivatives': scored.derivatives_score,
                'sentiment':   scored.sentiment_score,
                'grade':       scored.grade,
            },
        }
        signal_id = await db.save_signal(signal_record)
        logger.info(f"Signal saved: id={signal_id} {sig.symbol} {direction} grade={scored.grade}")

        # ── 2. Route based on grade ───────────────────────────────────

        if scored.grade in ("A+", "A", "B"):
            return await self._publish_execution(scored, signal_id)

        elif scored.grade == "C":
            return await self._publish_c_signal(scored, signal_id)

        return False

    # ── Execution signal (A+/A/B) ─────────────────────────────────────

    async def _publish_execution(self, scored: ScoredSignal, signal_id: int) -> bool:
        """Send full execution card to signals channel with push notification."""
        try:
            message_id = await self._bot.publish_signal(scored, signal_id)
            if not message_id:
                logger.warning(f"Signal saved but Telegram delivery failed: id={signal_id}")
                return False
            # Update DB after a confirmed Telegram send; handle DB failure separately
            # so a transient write error does not mask the fact the message was sent.
            try:
                await db.update_signal_message_id(signal_id, message_id)
            except Exception as db_err:
                logger.error(
                    f"Execution signal sent (id={signal_id} msg_id={message_id}) "
                    f"but DB message_id update failed — manual recovery may be needed: {db_err}"
                )
                return False
            logger.info(f"Execution signal sent: id={signal_id} msg_id={message_id}")
            return True
        except Exception as e:
            logger.error(f"Execution publish error: {e}", exc_info=True)
            return False

    # ── C signal routing ──────────────────────────────────────────────

    async def _publish_c_signal(self, scored: ScoredSignal, signal_id: int) -> bool:
        """
        Evaluate C signal with upgrade tracker.
        High UP → CONTEXT message to signals channel.
        Low UP  → admin channel only, silent.
        """
        send_to_user, up_score = upgrade_tracker.evaluate(scored, signal_id)

        try:
            if send_to_user:
                # Send CONTEXT format (no trade numbers) to signals channel
                message_id = await self._bot.publish_context_signal(
                    scored, signal_id, up_score
                )
                if message_id:
                    await db.update_signal_message_id(signal_id, message_id)
                    return True
                return False
            else:
                # Admin channel only — no push, compact format
                await self._bot.send_admin_c_signal(scored, signal_id, up_score)
                return False

        except Exception as e:
            logger.error(f"C signal publish error: {e}", exc_info=True)
            return False

    async def notify_upgrade(self, signal_id: int, new_grade: str, new_scored: ScoredSignal):
        """
        Called when a previously-sent C signal is upgraded.
        Edits the context message and sends upgrade notification.
        """
        if self._bot is None:
            return

        # Log to upgrade tracker for Phase 2 learning
        upgrade_tracker.on_upgrade(signal_id, new_grade)

        try:
            await self._bot.publish_upgrade(signal_id, new_grade, new_scored)
        except Exception as e:
            logger.error(f"Upgrade notification error: {e}", exc_info=True)

    async def notify_c_expired(self, signal_id: int):
        """Called when a tracked C signal expires without upgrading."""
        upgrade_tracker.on_expired(signal_id)
        # Silent — no message to user

    # ── Unchanged helpers ─────────────────────────────────────────────

    async def publish_watchlist(self, symbol: str, score: float, reasons: list):
        if self._bot is None:
            return
        try:
            await db.upsert_watchlist(symbol, score, reasons)
            await self._bot.send_watchlist_alert(symbol, score, reasons)
        except Exception as e:
            logger.error(f"Watchlist publish error: {e}")

    async def publish_whale_alert(self, symbol: str, order_usd: float, vol_mult: float):
        if self._bot is None:
            return
        try:
            await self._bot.send_whale_alert(symbol, order_usd, vol_mult)
        except Exception as e:
            logger.error(f"Whale alert publish error: {e}")

    async def publish_tier_promotion(self, symbol: str, from_tier: int, to_tier: int,
                                      vol_mult: float, vol_24h: float):
        if self._bot is None:
            return
        try:
            await db.record_promotion(symbol, from_tier, to_tier)
            await self._bot.send_tier_promotion(symbol, from_tier, to_tier, vol_mult, vol_24h)
        except Exception as e:
            logger.error(f"Tier promotion publish error: {e}")


# ── Singleton ──────────────────────────────────────────────
publisher = SignalPublisher()
