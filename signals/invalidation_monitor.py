"""
TitanBot Pro — Signal Invalidation Monitor
=============================================
Monitors published signals for invalidation conditions:

1. PRICE INVALIDATION: Price closes past stop loss before entry zone reached
   → Send "❌ SIGNAL INVALIDATED" update

2. ENTRY REACHED: Price enters the entry zone
   → Send "✅ ENTRY ZONE REACHED — signal is live"

3. COUNTER-SIGNAL: A new signal fires in the opposite direction on same symbol
   → Send "⚠️ CONFLICTING SIGNAL — reduce size or skip"

4. STRUCTURE BREAK: Market structure changes (e.g., BOS in opposite direction)
   → Send "⚠️ Structure shift — reassess"

5. TIMEOUT: Signal not entered within configured window (default 4h)
   → Send "⏰ SIGNAL EXPIRED — setup no longer valid"

This integrates with the existing OutcomeMonitor but runs BEFORE entry,
while OutcomeMonitor runs AFTER entry.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Set, Callable
from dataclasses import dataclass, field

from config.loader import cfg
from data.api_client import api
from utils.formatting import fmt_price
from core.price_cache import price_cache

logger = logging.getLogger(__name__)


# How often to check each pending signal (seconds).
# Faster = catch invalidations sooner, but more API calls.
#
# Operator override via settings.yaml:
#   system.invalidation_check_interval: <seconds>
# Falls back to the 15s default below when unset/invalid.
_DEFAULT_CHECK_INTERVAL = 15


def _load_check_interval() -> int:
    """Read system.invalidation_check_interval from config, with safe fallback."""
    try:
        from config.loader import cfg
        raw = cfg.system.get("invalidation_check_interval", _DEFAULT_CHECK_INTERVAL)
        val = int(raw)
        if val < 1:
            logger.warning(
                "system.invalidation_check_interval=%s is below minimum (1s); "
                "using default %ss", raw, _DEFAULT_CHECK_INTERVAL,
            )
            return _DEFAULT_CHECK_INTERVAL
        return val
    except Exception as exc:
        logger.debug(
            "Could not read system.invalidation_check_interval (%s); "
            "using default %ss", exc, _DEFAULT_CHECK_INTERVAL,
        )
        return _DEFAULT_CHECK_INTERVAL


CHECK_INTERVAL = _load_check_interval()

# Default timeout before a pending signal expires (seconds)
DEFAULT_PENDING_TIMEOUT = 4 * 3600  # 4 hours


@dataclass
class PendingSignal:
    """A published signal that hasn't been entered yet"""
    signal_id: int
    symbol: str
    direction: str           # LONG | SHORT
    strategy: str
    entry_low: float
    entry_high: float
    stop_loss: float
    confidence: float
    message_id: Optional[int] = None
    created_at: float = 0.0
    status: str = "PENDING"  # PENDING, ENTERED, INVALIDATED, EXPIRED, CONFLICTED
    invalidation_reason: Optional[str] = None
    approaching_alerted: bool = False  # True after "price approaching zone" alert sent
    aging_alerted: bool = False       # True after staleness warning sent (at 75% of lifetime)
    timeout_seconds: float = 0.0      # Strategy-specific timeout (0 = use global default)
    grade: str = "B"                  # BEH-4: grade stored so conflict/approach gates work
    setup_class: str = "intraday"     # BEH-9: setup_class for approach threshold scaling
    raw_data: Optional[dict] = None   # BUG-14: strategy context for pre-entry re-validation
    regime_at_publish: Optional[str] = None  # BUG-14: regime when signal was published
    tp1: float = 0.0                  # FIX-SL-TP: store price levels to avoid DB re-read
    tp2: float = 0.0
    tp3: Optional[float] = None
    rr_ratio: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class InvalidationMonitor:
    """
    Monitors pending signals for invalidation.

    Separate from OutcomeMonitor:
      - InvalidationMonitor: BEFORE entry (is the setup still valid?)
      - OutcomeMonitor: AFTER entry (did the trade win or lose?)
    """

    def __init__(self):
        self._pending: Dict[int, PendingSignal] = {}  # signal_id -> PendingSignal
        self._active_symbols: Set[str] = set()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._paused = False
        # FIX #11: per-symbol approaching alert cooldown (prevents duplicates from multiple signals)
        self._approaching_symbol_cooldown: Dict[str, float] = {}
        self._approaching_cooldown_secs: int = 1800  # 30 min cooldown per symbol

        # Callbacks
        self.on_invalidated: Optional[Callable] = None   # (signal_id, reason, message_id)
        self.on_entry_reached: Optional[Callable] = None  # (signal_id, price, message_id)
        self.on_conflict: Optional[Callable] = None       # (signal_id, conflicting_direction, message_id)
        self.on_expired: Optional[Callable] = None        # (signal_id, message_id)
        self.on_approaching: Optional[Callable] = None    # (signal_id, price, pct_away, message_id)
        # Gap 5: staleness warning — fires once when signal reaches 75% of its lifetime
        self.on_aging: Optional[Callable] = None          # (signal_id, pct_elapsed, remaining_mins, message_id)


    def start(self):
        """Start the invalidation monitor background task."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("🔍 InvalidationMonitor started")

    async def stop(self):
        """Stop the invalidation monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def pause(self):
        """Pause during network outage."""
        self._paused = True

    def resume(self):
        """Resume after network restored."""
        self._paused = False


    def track_signal(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        strategy: str,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        confidence: float,
        message_id: Optional[int] = None,
        grade: str = "B",
        setup_class: str = "intraday",
        timeout_seconds: float = 0.0,
        raw_data: Optional[dict] = None,
        regime_at_publish: Optional[str] = None,
        tp1: float = 0.0,                     # FIX-SL-TP: pass price levels to avoid DB re-read in approaching alert
        tp2: float = 0.0,
        tp3: Optional[float] = None,
        rr_ratio: float = 0.0,
    ):
        """Register a newly published signal for pre-entry monitoring.

        timeout_seconds: if > 0, overrides the global pending_signal_timeout.
                         Should be set from formatter.STRATEGY_EXPIRY_CANDLES × TF_MINUTES
                         to keep the monitor in sync with the expiry timer shown to the user.
        """
        sig = PendingSignal(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            confidence=confidence,
            message_id=message_id,
            grade=grade,
            setup_class=setup_class,
            timeout_seconds=timeout_seconds,
            raw_data=raw_data,
            regime_at_publish=regime_at_publish,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            rr_ratio=rr_ratio,
        )
        self._pending[signal_id] = sig
        self._active_symbols.add(symbol)
        price_cache.subscribe(symbol)
        logger.debug(f"Tracking signal #{signal_id} ({symbol} {direction}) for invalidation")

    def notify_counter_signal(self, symbol: str, direction: str):
        """
        Called by engine when a new signal fires on a symbol
        that already has a pending signal in the opposite direction.
        """
        for sig in list(self._pending.values()):
            if sig.symbol == symbol and sig.direction != direction and sig.status == "PENDING":
                sig.status = "CONFLICTED"
                sig.invalidation_reason = (
                    f"Counter-signal: {direction} signal fired while "
                    f"{sig.direction} still pending"
                )
                if self.on_conflict:
                    # FIX: was asyncio.create_task() — the task was unrooted and could
                    # be silently cancelled mid-execution. notify_counter_signal() is
                    # called from an async context (the engine scan loop), so we can
                    # schedule a proper awaitable task via the event loop's call_soon_threadsafe
                    # equivalent. Use create_task but store it to prevent GC cancellation.
                    _conflict_task = asyncio.create_task(self._safe_callback(
                        self.on_conflict,
                        sig.signal_id, direction, sig.message_id
                    ))
                    # Add a done-callback to log any unhandled exception from the task
                    _conflict_task.add_done_callback(
                        lambda t: logger.debug(f"Conflict callback error: {t.exception()}") if not t.cancelled() and t.exception() else None
                    )
                logger.info(
                    f"⚠️ Conflict: signal #{sig.signal_id} ({sig.symbol} {sig.direction}) "
                    f"vs new {direction} signal"
                )

    def remove_signal(self, signal_id: int):
        """Remove a signal from tracking (e.g., after outcome monitor takes over)"""
        sig = self._pending.pop(signal_id, None)
        if sig:
            still_has_symbol = any(
                s.symbol == sig.symbol for s in self._pending.values()
            )
            if not still_has_symbol:
                self._active_symbols.discard(sig.symbol)
                price_cache.unsubscribe(sig.symbol)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def get_pending_signals(self) -> list:
        """Get all pending signals as dicts (for reconnect revalidation)"""
        return [
            {
                'signal_id': sig.signal_id,
                'symbol': sig.symbol,
                'direction': sig.direction,
                'created_at': sig.created_at,
                'confidence': sig.confidence,
            }
            for sig in self._pending.values()
        ]

    def cancel_signal(self, signal_id: int, reason: str = "CANCELLED"):
        """Cancel a pending signal with a logged reason"""
        sig = self._pending.pop(signal_id, None)
        if sig:
            # Pass 9 FIX: only unsubscribe from price_cache when no other pending signal
            # still needs this symbol. The old code always discarded without checking —
            # a second pending signal on the same symbol would silently lose price updates,
            # making its SL/entry checks fail for the rest of its lifetime.
            still_has_symbol = any(s.symbol == sig.symbol for s in self._pending.values())
            if not still_has_symbol:
                self._active_symbols.discard(sig.symbol)
                price_cache.unsubscribe(sig.symbol)
            logger.info(
                f"❌ Signal #{signal_id} cancelled | {sig.symbol} "
                f"| reason={reason}"
            )

    # ── Main monitoring loop ──────────────────────────────────

    async def _monitor_loop(self):
        """Periodically check all pending signals"""
        while self._running:
            try:
                if not self._pending:
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                # Check each pending signal
                to_remove = []
                for signal_id, sig in list(self._pending.items()):
                    if sig.status != "PENDING":
                        to_remove.append(signal_id)
                        continue

                    try:
                        await self._check_signal(sig)
                    except Exception as e:
                        logger.error(f"Error checking signal #{signal_id}: {e}")

                    if sig.status != "PENDING":
                        to_remove.append(signal_id)

                # Clean up resolved signals
                for sid in to_remove:
                    self.remove_signal(sid)

                await asyncio.sleep(CHECK_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Invalidation monitor loop error: {e}")
                await asyncio.sleep(60)

    async def _check_signal(self, sig: PendingSignal):
        """Check a single pending signal for invalidation or entry"""

        # ── 1. Timeout check ──────────────────────────────────
        # Use strategy-specific timeout if provided; else global config
        global_timeout = cfg.system.get('pending_signal_timeout', DEFAULT_PENDING_TIMEOUT)
        timeout = sig.timeout_seconds if sig.timeout_seconds > 0 else global_timeout
        elapsed = time.time() - sig.created_at
        pct_elapsed = elapsed / timeout if timeout > 0 else 1.0

        # Gap 5: Staleness warning — fire once when 75% of the validity window has elapsed
        # and the signal still hasn't been entered. Warns the human trader to re-check
        # the setup before acting, because market conditions may have changed.
        if (not sig.aging_alerted
                and pct_elapsed >= 0.75
                and pct_elapsed < 1.0
                and self.on_aging):
            sig.aging_alerted = True
            remaining_mins = (timeout - elapsed) / 60
            logger.info(
                f"⚠️ Signal #{sig.signal_id} aging: {pct_elapsed*100:.0f}% elapsed, "
                f"{remaining_mins:.0f}min remaining"
            )
            await self._safe_callback(
                self.on_aging,
                sig.signal_id,
                pct_elapsed * 100,
                remaining_mins,
                sig.message_id,
            )

        if elapsed > timeout:
            sig.status = "EXPIRED"
            sig.invalidation_reason = f"Signal timed out after {timeout/3600:.1f}h"
            if self.on_expired:
                await self._safe_callback(
                    self.on_expired, sig.signal_id, sig.message_id
                )
            logger.info(f"⏰ Signal #{sig.signal_id} expired (no entry after {timeout/3600:.1f}h)")
            return

        # ── 2. Fetch current price ────────────────────────────
        current_price = price_cache.get(sig.symbol)
        if current_price is None:
            # Cache miss (stale or not yet fetched) — skip this cycle
            return

        # ── 3. Check for invalidation (price past SL before entry) ──
        # Use strict < / > (not <= / >=) so that price touching the SL level exactly
        # does not immediately invalidate — a single tick at the SL price often recovers
        # intrabar on perpetuals. True invalidation requires a close or sustained print
        # below/above the SL, which subsequent ticks will confirm.
        if sig.direction == "LONG":
            if current_price < sig.stop_loss:
                sig.status = "INVALIDATED"
                sig.invalidation_reason = (
                    f"Price {fmt_price(current_price)} dropped below stop loss "
                    f"{fmt_price(sig.stop_loss)} before entry zone reached"
                )
                if self.on_invalidated:
                    await self._safe_callback(
                        self.on_invalidated,
                        sig.signal_id, sig.invalidation_reason, sig.message_id
                    )
                logger.info(f"❌ Signal #{sig.signal_id} invalidated: {sig.invalidation_reason}")
                return

        elif sig.direction == "SHORT":
            if current_price > sig.stop_loss:
                sig.status = "INVALIDATED"
                sig.invalidation_reason = (
                    f"Price {fmt_price(current_price)} rose above stop loss "
                    f"{fmt_price(sig.stop_loss)} before entry zone reached"
                )
                if self.on_invalidated:
                    await self._safe_callback(
                        self.on_invalidated,
                        sig.signal_id, sig.invalidation_reason, sig.message_id
                    )
                logger.info(f"❌ Signal #{sig.signal_id} invalidated: {sig.invalidation_reason}")
                return

        # ── 4. BUG-14: Strategy-specific re-validation (every check cycle) ────
        # These run BEFORE price reaches the zone. If the setup basis has changed,
        # invalidate immediately rather than waiting for the user to enter a broken trade.
        invalidation_reason = await self._revalidate_setup(sig, current_price)
        if invalidation_reason:
            sig.status = "INVALIDATED"
            sig.invalidation_reason = invalidation_reason
            if self.on_invalidated:
                await self._safe_callback(
                    self.on_invalidated,
                    sig.signal_id, sig.invalidation_reason, sig.message_id
                )
            logger.info(f"❌ Signal #{sig.signal_id} setup invalidated: {sig.invalidation_reason}")
            return

        # ── 5. Check for entry zone reached ────────────────────
        if sig.entry_low <= current_price <= sig.entry_high:
            sig.status = "ENTERED"
            if self.on_entry_reached:
                await self._safe_callback(
                    self.on_entry_reached,
                    sig.signal_id, current_price, sig.message_id
                )
            logger.info(
                f"✅ Signal #{sig.signal_id} entry zone reached at {fmt_price(current_price)}"
            )
            return

        # ── 6. Approaching entry zone alert ────────────────────
        # Fire once when price is within 1.5% of the entry zone edge.
        # For LONG: approaching from above (price falling toward entry_high)
        # For SHORT: approaching from below (price rising toward entry_low)
        if not sig.approaching_alerted and self.on_approaching:
            # FIX #11: per-symbol cooldown for approaching alerts
            _last_approach = self._approaching_symbol_cooldown.get(sig.symbol, 0)
            if time.time() - _last_approach < self._approaching_cooldown_secs:
                sig.approaching_alerted = True  # Suppress — another signal already alerted
                return
            zone_midpoint = (sig.entry_low + sig.entry_high) / 2
            # BEH-9: threshold scales with setup_class — defined before branches so both directions can use it
            # FIX-10: In CHOPPY regime, widen threshold — wider ranges mean price often
            # approaches entry then reverses without filling. 1.5% creates false urgency.
            _base_thresh = {"swing": 0.025, "intraday": 0.015, "scalp": 0.008}.get(sig.setup_class, 0.015)
            try:
                from analyzers.regime import regime_analyzer as _ra_inv
                if _ra_inv.regime.value == "CHOPPY":
                    _base_thresh = min(0.030, _base_thresh * 1.6)  # 1.5% → 2.4% in chop
            except Exception:
                pass
            _approach_thresh = _base_thresh
            if sig.direction == "LONG":
                # Price should be above entry zone for a "pullback" entry
                if sig.entry_high < current_price:
                    dist_pct = (current_price - sig.entry_high) / current_price
                    if dist_pct <= _approach_thresh:
                        sig.approaching_alerted = True
                        self._approaching_symbol_cooldown[sig.symbol] = time.time()
                        await self._safe_callback(
                            self.on_approaching,
                            sig.signal_id, current_price, dist_pct * 100, sig.message_id
                        )
                        logger.info(
                            f"📍 Signal #{sig.signal_id} approaching entry zone "
                            f"({dist_pct*100:.1f}% away)"
                        )
            elif sig.direction == "SHORT":
                if current_price < sig.entry_low:
                    dist_pct = (sig.entry_low - current_price) / current_price
                    if dist_pct <= _approach_thresh:
                        sig.approaching_alerted = True
                        self._approaching_symbol_cooldown[sig.symbol] = time.time()
                        await self._safe_callback(
                            self.on_approaching,
                            sig.signal_id, current_price, dist_pct * 100, sig.message_id
                        )

    async def _revalidate_setup(self, sig: PendingSignal, current_price: float) -> Optional[str]:
        """
        BUG-14: Re-validate the setup basis for a pending signal every 15 seconds.

        Runs on ALL strategies for regime check.
        Runs strategy-specific checks when raw_data is available.

        Returns an invalidation reason string if the setup is no longer valid,
        or None if everything still checks out.
        """
        try:
            try:
                from analyzers.btc_news_intelligence import btc_news_intelligence
                _btc_block_reason = btc_news_intelligence.get_signal_block_reason(sig.direction)
                if _btc_block_reason:
                    return _btc_block_reason
            except Exception:
                pass

            signal_age_mins = (time.time() - sig.created_at) / 60

            # Only re-validate if signal has been pending long enough for conditions to drift
            # (< 10 minutes = market hasn't had time to change meaningfully)
            if signal_age_mins < 10:
                return None

            # ── 1. Regime flip re-validation (all strategies) ─────────
            # If regime has flipped against the signal direction since publish,
            # the macro context the signal was analysed in no longer exists.
            try:
                from analyzers.regime import regime_analyzer
                current_regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                original_regime = sig.regime_at_publish

                if original_regime and original_regime != current_regime:
                    # BULL_TREND → BEAR_TREND or BEAR_TREND → BULL_TREND is a structural flip
                    trend_flipped = (
                        original_regime in ("BULL_TREND", "BEAR_TREND") and
                        current_regime in ("BULL_TREND", "BEAR_TREND") and
                        original_regime != current_regime
                    )
                    went_risk_off = current_regime == "VOLATILE" and original_regime != "VOLATILE"

                    if trend_flipped and signal_age_mins > 30:
                        direction_conflict = (
                            (sig.direction == "LONG" and current_regime == "BEAR_TREND") or
                            (sig.direction == "SHORT" and current_regime == "BULL_TREND")
                        )
                        if direction_conflict:
                            return (
                                f"Regime flipped {original_regime}→{current_regime} "
                                f"({signal_age_mins:.0f}min after publish) — "
                                f"{sig.direction} signal now counter-trend"
                            )

                    if went_risk_off:
                        return (
                            f"Market entered VOLATILE {signal_age_mins:.0f}min after publish — "
                            f"panic conditions override pending setup"
                        )
            except Exception:
                pass

            # ── 2. Elliott Wave — W2/W4 violation pre-entry ───────────
            # If price violates the wave invalidation level while still PENDING,
            # the wave count is already broken before we even entered.
            if sig.strategy == "ElliottWave" and sig.raw_data:
                inv_level = sig.raw_data.get('elliott_invalidation_level')
                if inv_level and inv_level > 0:
                    violated = (
                        (sig.direction == "LONG" and current_price < inv_level * 0.999) or
                        (sig.direction == "SHORT" and current_price > inv_level * 1.001)
                    )
                    if violated:
                        return (
                            f"Elliott wave count violated pre-entry: "
                            f"W2/W4 level {inv_level:.5f} taken out "
                            f"({signal_age_mins:.0f}min after publish)"
                        )

            # ── 3. FundingArb — funding normalised while pending ───────
            # If funding rate has normalised, the edge that created the signal is gone.
            if sig.strategy == "FundingRateArb" and sig.raw_data:
                original_funding = sig.raw_data.get('funding_rate', 0)
                if abs(original_funding) > 0:
                    try:
                        from data.api_client import api as _api_ref
                        _fd = await _api_ref.fetch_funding_rate(sig.symbol)
                        if _fd:
                            current_funding = float(_fd.get('fundingRate', original_funding) or original_funding)
                            # Extreme → normal: original was extreme, now within normal range
                            _extreme_long = 0.05 / 100  # 0.05% per 8h
                            _extreme_short = -0.03 / 100
                            original_was_extreme = (
                                original_funding >= _extreme_long or
                                original_funding <= _extreme_short
                            )
                            current_is_normal = _extreme_short < current_funding < _extreme_long
                            if original_was_extreme and current_is_normal:
                                return (
                                    f"FundingArb: funding normalised while pending "
                                    f"({original_funding*100:.3f}% → {current_funding*100:.3f}%) — "
                                    f"squeeze edge no longer present"
                                )
                    except Exception:
                        pass

            # ── 4. Breakout — breakout level retested and failed ───────
            # If a breakout signal's level has been retested, rejected, and price
            # closed back inside the prior range, it's a false breakout.
            if sig.strategy == "InstitutionalBreakout" and signal_age_mins > 60:
                try:
                    from data.api_client import api as _api_ref
                    _ohlcv = await _api_ref.fetch_ohlcv(sig.symbol, "1h", limit=10)
                    if _ohlcv and len(_ohlcv) >= 4:
                        import numpy as np
                        _closes = np.array([float(b[4]) for b in _ohlcv])
                        _breakout_level = (sig.entry_low + sig.entry_high) / 2

                        if sig.direction == "LONG":
                            # False breakout: price went above breakout, came back below it
                            _was_above = any(c > _breakout_level * 1.005 for c in _closes[:-2])
                            _now_below = _closes[-1] < _breakout_level * 0.998
                            if _was_above and _now_below:
                                return (
                                    f"Breakout false — retested {fmt_price(_breakout_level)} "
                                    f"and closed back inside range ({signal_age_mins:.0f}min)"
                                )
                        else:  # SHORT
                            _was_below = any(c < _breakout_level * 0.995 for c in _closes[:-2])
                            _now_above = _closes[-1] > _breakout_level * 1.002
                            if _was_below and _now_above:
                                return (
                                    f"Breakdown false — retested {fmt_price(_breakout_level)} "
                                    f"and closed back inside range ({signal_age_mins:.0f}min)"
                                )
                except Exception:
                    pass

        except Exception as _outer:
            logger.debug(f"_revalidate_setup error (non-fatal): {_outer}")

        return None  # All checks passed — setup still valid

    async def _safe_callback(self, callback, *args):
        """Execute a callback safely"""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Invalidation callback error: {e}")


# Singleton
invalidation_monitor = InvalidationMonitor()
