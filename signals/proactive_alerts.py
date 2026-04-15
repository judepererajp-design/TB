"""
TitanBot Pro — Proactive Mid-Trade Alerts
==========================================
Monitors active trades and fires Telegram alerts when definitive
thesis-changing events occur. These are all RULE-BASED — no AI,
no ambiguity.

Alert types (in priority order):
  1. VOLATILE_PANIC    — Regime escalated to VOLATILE_PANIC while holding positions
  2. OPPOSING_SIGNAL   — Bot generated a signal in opposite direction on same coin
  3. FUNDING_FLIPPED   — Funding rate reversed for FundingRateArb trades
  4. REGIME_CHANGED    — Regime flipped to/from VOLATILE since entry
  5. BTC_MACRO_DROP    — BTC dropped >2% while holding an alt long (actionable TP1 hint)
  6. NEAR_SL           — Price within 0.3R of SL (proactive warning)

Non-alerts (by design):
  - Normal drawdown/adverse movement — that's what the SL is for
  - Minor regime changes (CHOPPY↔BULL_TREND) — not definitive enough
  - AI "things look different" — too noisy, erodes trust

Macro pressure flag:
  When a BTC_MACRO_DROP fires, is_macro_pressure_active() returns True for
  _MACRO_PRESSURE_TTL seconds. The engine queries this flag to reduce confidence
  on new alt-long signals during the window, biasing toward smaller size and TP1.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

# How often to run the alert check loop
_CHECK_INTERVAL = 60  # seconds

# BTC drop threshold to alert on alt longs
_BTC_DROP_ALERT_PCT = -2.0  # -2% on 15m candle

# How recently we need to have seen a BTC price to compare (60 min)
_BTC_PRICE_WINDOW = 3600

# Funding flip: from negative to positive (or vice versa) — only for FundingRateArb
_FUNDING_FLIP_THRESHOLD = 0.0001  # 0.01%/8h — clearly positive or negative

# Minimum time in trade before near-SL alert fires (avoid alerting on initial entry volatility)
_MIN_TRADE_AGE_FOR_SL_ALERT = 1800  # 30 minutes

# How long the macro pressure flag stays active after a BTC drop alert fires
_MACRO_PRESSURE_TTL = 3600  # 60 minutes


@dataclass
class AlertRecord:
    """Tracks which alerts have been sent per signal to avoid repeats."""
    signal_id: int
    sent: Set[str] = field(default_factory=set)  # alert type names sent
    last_btc_price: Optional[float] = None
    last_btc_time:  Optional[float] = None
    entry_regime:   Optional[str]   = None
    entry_funding:  Optional[float] = None


class ProactiveAlertEngine:
    """
    Runs every 60 seconds, checks all active tracked signals,
    and fires alerts for definitive thesis-changing events.
    """

    def __init__(self):
        self._running   = False
        self._task: Optional[asyncio.Task] = None
        self._records:  Dict[int, AlertRecord] = {}

        # Macro pressure tracking (set by BTC drop alerts)
        self._macro_pressure_until: float = 0.0

        # VOLATILE_PANIC guard — track which signal IDs already received the panic alert
        self._panic_alerted: Set[int] = set()

        # Callbacks — set by bot.py at startup
        self.on_alert:           Optional[Callable] = None  # async fn(signal_id, text)
        self.on_opposing_signal: Optional[Callable] = None  # set from engine

    def start(self):
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("⚡ ProactiveAlertEngine started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def is_macro_pressure_active(self) -> bool:
        """
        Returns True if a BTC macro drop has fired within the last hour.
        Queried by the engine to reduce confidence on new alt-long signals
        and bias toward TP1 rather than TP2 during macro stress windows.
        """
        return time.time() < self._macro_pressure_until

    def register_signal(self, signal_id: int, regime: str, funding: float = 0.0):
        """Called when a signal enters active tracking."""
        rec = AlertRecord(signal_id=signal_id, entry_regime=regime, entry_funding=funding)
        # Snapshot current BTC price
        try:
            from core.price_cache import price_cache
            btc = price_cache.get("BTC/USDT")
            if btc:
                rec.last_btc_price = btc
                rec.last_btc_time  = time.time()
        except Exception:
            pass
        self._records[signal_id] = rec
        logger.debug(f"ProactiveAlert: registered signal #{signal_id}")

    def deregister_signal(self, signal_id: int):
        """Called when a signal closes."""
        self._records.pop(signal_id, None)
        self._panic_alerted.discard(signal_id)

    async def notify_opposing_signal(self, symbol: str, opposing_direction: str,
                                      strategy: str, confidence: float):
        """
        Called by engine when a new signal is published.
        Checks if any active trade on same coin is in the opposite direction.
        """
        if not self.on_alert:
            return
        try:
            from signals.outcome_monitor import outcome_monitor
            active = outcome_monitor.get_active_signals()
            for sig_id, tracked in active.items():
                if tracked.symbol == symbol and tracked.direction != opposing_direction:
                    rec = self._records.get(sig_id)
                    if rec and "opposing_signal" not in rec.sent:
                        rec.sent.add("opposing_signal")
                        text = (
                            f"⚡ <b>OPPOSING SIGNAL — {symbol}</b>\n\n"
                            f"A new <b>{opposing_direction}</b> signal just fired on {symbol} "
                            f"({strategy}, conf={confidence:.0f}).\n\n"
                            f"You hold a <b>{tracked.direction}</b> position.\n\n"
                            f"This doesn't mean exit — it means the bot sees {opposing_direction} "
                            f"structure forming. If price is near your TP, consider taking profit.\n\n"
                            f"<i>Your SL at <code>{tracked.stop_loss:.6g}</code> is still your exit rule.</i>"
                        )
                        await self.on_alert(sig_id, text)
        except Exception as e:
            logger.debug(f"notify_opposing_signal error: {e}")

    async def _loop(self):
        """Main alert monitoring loop."""
        while self._running:
            try:
                await self._check_all()
            except Exception as e:
                logger.debug(f"ProactiveAlert loop error: {e}")
            await asyncio.sleep(_CHECK_INTERVAL)

    async def _check_volatile_panic_positions(self):
        """
        FIX P1: Check if regime has escalated to VOLATILE_PANIC while
        positions are open. This is the highest-priority alert — tail events
        where the biggest losses happen. Fires once per signal per panic episode.
        """
        if not self.on_alert:
            return
        try:
            from analyzers.regime import regime_analyzer
            current_regime = getattr(regime_analyzer.regime, 'value', '') if regime_analyzer.regime else ''
            if "VOLATILE_PANIC" not in current_regime:
                return

            from signals.outcome_monitor import outcome_monitor
            active = outcome_monitor.get_active_signals()
            if not active:
                return

            for sig_id, tracked in active.items():
                if sig_id in self._panic_alerted:
                    continue  # already warned this signal in this panic episode
                self._panic_alerted.add(sig_id)
                is_long = tracked.direction == "LONG"
                action_line = (
                    f"<b>⚡ Action: Consider taking TP1 (<code>{tracked.tp1:.6g}</code>) NOW.</b>\n"
                    f"This locks in gains before selling pressure accelerates."
                ) if is_long else (
                    f"<b>⚡ Action: Consider securing profits on your SHORT — "
                    f"panic selloffs can reverse violently.</b>"
                )
                text = (
                    f"🚨 <b>VOLATILE PANIC — OPEN POSITION AT RISK</b>\n\n"
                    f"Market just escalated to <b>VOLATILE_PANIC</b>. "
                    f"This is a tail-risk event: rapid selling, cascading liquidations, "
                    f"correlation spikes across all altcoins.\n\n"
                    f"Symbol: <b>{tracked.symbol}</b> {tracked.direction}\n"
                    f"Entry: <code>{tracked.entry_price:.6g}</code>  "
                    f"SL: <code>{tracked.stop_loss:.6g}</code>\n\n"
                    f"{action_line}\n\n"
                    f"<i>If TP1 is already hit, tighten mentally to TP1 value. "
                    f"SL at <code>{tracked.stop_loss:.6g}</code> remains your hard exit.</i>"
                )
                await self.on_alert(sig_id, text)
                logger.info(f"🚨 VOLATILE_PANIC alert fired for signal #{sig_id} ({tracked.symbol})")

        except Exception as e:
            logger.debug(f"_check_volatile_panic_positions error: {e}")

    async def _check_all(self):
        """Check all active signals for alert conditions."""
        if not self.on_alert:
            return

        # Always check for VOLATILE_PANIC first — it affects ALL open positions
        await self._check_volatile_panic_positions()

        try:
            from signals.outcome_monitor import outcome_monitor
            active = outcome_monitor.get_active_signals()
        except Exception:
            return

        for sig_id, tracked in list(active.items()):
            try:
                rec = self._records.get(sig_id)
                if not rec:
                    # New signal we haven't registered yet
                    from analyzers.regime import regime_analyzer
                    regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN') if regime_analyzer.regime else "UNKNOWN"
                    self.register_signal(sig_id, regime)
                    rec = self._records[sig_id]

                await self._check_signal(tracked, rec)

            except Exception as e:
                logger.debug(f"ProactiveAlert check #{sig_id}: {e}")

    async def _check_signal(self, tracked, rec: AlertRecord):
        """Run all alert checks for one signal."""
        now = time.time()
        is_long = tracked.direction == "LONG"

        # ── 1. Regime change ─────────────────────────────────────────────
        if "regime_changed" not in rec.sent:
            try:
                from analyzers.regime import regime_analyzer
                current_regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                entry_regime   = rec.entry_regime or current_regime

                # Alert when regime flips to/from risk-off (VOLATILE or VOLATILE_PANIC)
                was_risk_off = "VOLATILE" in entry_regime
                is_risk_off  = "VOLATILE" in current_regime

                if is_long and is_risk_off and not was_risk_off:
                    rec.sent.add("regime_changed")
                    text = (
                        f"🔄 <b>REGIME CHANGE — {tracked.symbol} LONG</b>\n\n"
                        f"Market just entered <b>{current_regime}</b> (was {entry_regime}).\n\n"
                        f"Your LONG signal was generated in a different environment. "
                        f"Risk-Off means broad selling pressure across crypto.\n\n"
                        f"<b>Suggested response:</b> Lower your ambition. If TP1 "
                        f"(<code>{tracked.tp1:.6g}</code>) is close, take it rather than "
                        f"waiting for TP2.\n\n"
                        f"<i>SL at <code>{tracked.stop_loss:.6g}</code> unchanged.</i>"
                    )
                    await self.on_alert(tracked.signal_id, text)

            except Exception:
                pass

        # ── 2. Funding rate flip (FundingRateArb signals only) ───────────
        if "funding_flipped" not in rec.sent and tracked.strategy == "FundingRateArb":
            try:
                from analyzers.derivatives import derivatives_analyzer
                coin   = tracked.symbol.replace("/USDT", "")
                deriv  = derivatives_analyzer.get_coin_intel(coin)
                if deriv:
                    funding_now    = deriv.funding_rate
                    funding_entry  = rec.entry_funding or 0
                    flipped = (
                        (funding_entry < -_FUNDING_FLIP_THRESHOLD and funding_now > _FUNDING_FLIP_THRESHOLD)
                        or (funding_entry > _FUNDING_FLIP_THRESHOLD and funding_now < -_FUNDING_FLIP_THRESHOLD)
                    )
                    if flipped:
                        rec.sent.add("funding_flipped")
                        fn_pct = funding_now * 100
                        fe_pct = funding_entry * 100
                        text = (
                            f"⚡ <b>FUNDING REVERSED — {tracked.symbol}</b>\n\n"
                            f"The FundingRateArb thesis <b>no longer exists</b>.\n\n"
                            f"Funding at entry: <code>{fe_pct:+.4f}%/8h</code>\n"
                            f"Funding now:      <code>{fn_pct:+.4f}%/8h</code>\n\n"
                            f"The squeeze that drove this signal has resolved. "
                            f"The tailwind is gone — you're now holding a plain directional trade.\n\n"
                            f"<b>If price is near TP1 (<code>{tracked.tp1:.6g}</code>), "
                            f"take it. If not, your SL is still your risk limit.</b>\n\n"
                            f"<i>SL at <code>{tracked.stop_loss:.6g}</code> unchanged.</i>"
                        )
                        await self.on_alert(tracked.signal_id, text)
            except Exception:
                pass

        # ── 3. BTC macro drop (alt longs only) ──────────────────────────
        if "btc_macro_drop" not in rec.sent and is_long:
            try:
                from core.price_cache import price_cache
                btc_now = price_cache.get("BTC/USDT")
                if btc_now and rec.last_btc_price and rec.last_btc_time:
                    age = now - rec.last_btc_time
                    if age <= _BTC_PRICE_WINDOW:
                        btc_chg_pct = (btc_now - rec.last_btc_price) / rec.last_btc_price * 100
                        if btc_chg_pct <= _BTC_DROP_ALERT_PCT:
                            rec.sent.add("btc_macro_drop")
                            # Activate macro pressure flag — engine will reduce confidence
                            # on new alt-long signals for the next hour
                            self._macro_pressure_until = now + _MACRO_PRESSURE_TTL
                            logger.info(
                                f"⚠️  Macro pressure flag activated for {_MACRO_PRESSURE_TTL//60}min "
                                f"(BTC dropped {btc_chg_pct:.1f}%)"
                            )
                            text = (
                                f"₿ <b>BTC MACRO PRESSURE — {tracked.symbol}</b>\n\n"
                                f"BTC dropped <b>{btc_chg_pct:.1f}%</b> while you hold an alt long.\n\n"
                                f"Alt-BTC correlation during sharp drops: ~0.80\n"
                                f"Your position peak: <b>{(tracked.max_r):+.2f}R</b>\n\n"
                                f"<b>⚡ Suggested action: Target TP1 (<code>{tracked.tp1:.6g}</code>) "
                                f"rather than waiting for TP2.</b>\n"
                                f"BTC drops drag alts hard in the first 30–60 min. "
                                f"Securing TP1 now locks in real P&L.\n\n"
                                f"<i>SL at <code>{tracked.stop_loss:.6g}</code> is your hard floor.</i>"
                            )
                            await self.on_alert(tracked.signal_id, text)

                # Update BTC snapshot every 15m
                if btc_now and (not rec.last_btc_time or now - rec.last_btc_time > 900):
                    rec.last_btc_price = btc_now
                    rec.last_btc_time  = now

            except Exception:
                pass

        # ── 4. Near SL warning ───────────────────────────────────────────
        trade_age = now - (tracked.activated_at or tracked.created_at or now)
        if ("near_sl" not in rec.sent and trade_age > _MIN_TRADE_AGE_FOR_SL_ALERT):
            try:
                from core.price_cache import price_cache
                price = price_cache.get(tracked.symbol)
                if price and tracked.stop_loss and tracked.entry_price:
                    risk = abs(tracked.entry_price - tracked.stop_loss)
                    if risk > 0:
                        dist_to_sl_r = (
                            (price - tracked.stop_loss) / risk if is_long
                            else (tracked.stop_loss - price) / risk
                        )
                        # Within 0.3R of SL and price is moving against us
                        if 0 < dist_to_sl_r < 0.3:
                            rec.sent.add("near_sl")
                            text = (
                                f"⚠️ <b>APPROACHING SL — {tracked.symbol}</b>\n\n"
                                f"Price <code>{price:.6g}</code> is within "
                                f"<b>{dist_to_sl_r:.2f}R</b> of your stop.\n\n"
                                f"SL: <code>{tracked.stop_loss:.6g}</code>\n\n"
                                f"This is a heads-up, not an instruction. "
                                f"If you're watching, check the chart for structure.\n\n"
                                f"<i>The SL exists for this exact scenario.</i>"
                            )
                            await self.on_alert(tracked.signal_id, text)
            except Exception:
                pass


# ── Singleton ────────────────────────────────────────────────────────────────
proactive_alerts = ProactiveAlertEngine()
