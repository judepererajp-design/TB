"""
TitanBot Pro — Signal Outcome Monitor
========================================
Automatically tracks published signals and records outcomes.
This is the KEY upgrade that enables the entire adaptive system.

After a signal is published:
  1. Spawns a background monitoring task
  2. Checks price every 60 seconds against TP/SL levels
  3. Auto-records WIN, LOSS, BREAKEVEN, or EXPIRED
  4. Feeds results into PerformanceTracker for EWMA adaptation
  5. Updates the original Telegram message with outcome

Lifecycle:
  PENDING → ENTRY_REACHED → (TP1_HIT → BE_ACTIVE) → TP2_HIT (WIN) | SL_HIT (LOSS) | EXPIRED
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from config.loader import cfg
from config.constants import Timing, OutcomeTracking
from data.api_client import api
from utils.formatting import fmt_price
from core.price_cache import price_cache

logger = logging.getLogger(__name__)
try:
    from utils.trade_logger import trade_logger as _tl
except Exception:
    _tl = None


def _safe_ensure_future(coro, *, context: str = "DB write"):
    """Schedule a coroutine as a fire-and-forget task with error logging.

    FIX #2 (AUDIT): All previous ``asyncio.ensure_future()`` calls swallowed
    exceptions silently.  If the DB write failed, no code path was notified and
    on restart ``restore()`` would find nothing — causing orphaned positions and
    broken learning stats.  This wrapper attaches a done-callback that logs
    failures so operators can detect persistence problems from the log stream.
    """
    task = asyncio.ensure_future(coro)

    def _on_done(t: asyncio.Task):
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error(f"[{context}] background task failed: {exc}")

    task.add_done_callback(_on_done)
    return task


class SignalState(str, Enum):
    PENDING       = "PENDING"         # Waiting for price to enter entry zone
    ACTIVE        = "ACTIVE"          # Price entered entry zone — trade is live
    TP1_HIT       = "TP1_HIT"         # TP1 reached, SL moved to breakeven
    BE_ACTIVE     = "BE_ACTIVE"       # Breakeven stop active
    COMPLETED_WIN = "COMPLETED_WIN"   # TP2 or TP3 hit
    COMPLETED_LOSS= "COMPLETED_LOSS"  # SL hit
    COMPLETED_BE  = "COMPLETED_BE"    # SL at breakeven hit
    EXPIRED       = "EXPIRED"         # Signal timed out


@dataclass
class TrackedSignal:
    """A signal being monitored for outcome"""
    signal_id: int
    symbol: str
    direction: str              # LONG | SHORT
    strategy: str
    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    confidence: float
    state: SignalState = SignalState.PENDING
    entry_price: Optional[float] = None    # Actual entry price when zone reached
    be_stop: Optional[float] = None        # Breakeven stop (set after TP1)
    created_at: float = 0.0
    activated_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_checkin_at: Optional[float] = None   # Last hourly progress update
    last_price: Optional[float] = None        # Most recent price seen (for button queries)
    pnl_r: float = 0.0                    # R-multiple achieved
    max_r: float = 0.0                    # Maximum R reached during trade
    trail_stop: Optional[float] = None      # V10: trailing stop price (after TP1)
    trail_pct: float = OutcomeTracking.DEFAULT_TRAIL_PCT  # V12: trailing fraction (regime-dependent, 0=no trail)
    regime: str = "UNKNOWN"                   # FIX (P2-A): regime at signal creation time
    message_id: Optional[int] = None       # Telegram message ID for editing
    raw_data: Optional[dict] = None        # FIX-1/FIX-2: strategy raw data for in-trade validation
    entry_status: Optional[str] = None     # IN_ZONE | LATE | EXTENDED
    # MAX-HOLD FIX: setup class determines how long a trade can stay open before
    # being force-closed at current price.  A scalp that runs 8h is no longer a scalp —
    # it has become an accidental position trade without sizing for it.
    # scalp=4h, intraday=12h, swing=48h, positional=no limit (0=disabled)
    setup_class: str = "intraday"


class OutcomeMonitor:
    """
    Background monitor that tracks all active signals and auto-records outcomes.
    Feeds into PerformanceTracker to enable adaptive strategy weights.
    """

    def __init__(self):
        self._active: Dict[int, TrackedSignal] = {}   # signal_id -> TrackedSignal
        self._check_interval = Timing.OUTCOME_CHECK_INTERVAL_SECS  # PHASE 3 FIX (OUTCOME-60): reduced from 60s to 15s.
        # Crypto wicks can trigger SL within seconds. 60s polling missed stop-hunt
        # wicks that recovered within the polling window, leaving losers open.
        # 15s is fast enough to catch most intra-candle SL touches.
        self._pending_timeout = Timing.PENDING_ENTRY_TIMEOUT_SECS   # 2 hours to reach entry zone
        self._signal_timeout = Timing.SIGNAL_MAX_LIFETIME_SECS  # 48 hours max signal lifetime
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._paused: bool = False   # B6: pause during network outage
        # FIX RATE: TTL gate for API price fallback — stores last fetch timestamp per symbol
        self._api_fallback_times: Dict[str, float] = {}

        # Callbacks — set by engine
        self.on_outcome: Optional[Callable] = None
        self.on_state_change: Optional[Callable] = None
        self.on_entry_reached: Optional[Callable] = None
        self.on_checkin: Optional[Callable] = None

    async def restore(self) -> None:
        """Reload active trades from tracked_signals_v1 on startup.

        Recovers PENDING/ACTIVE/TP1_HIT/BE_ACTIVE signals with their full
        protection state (be_stop, trail_stop, max_r, entry_price) so that
        SL monitoring continues correctly after a restart.  Existing entries
        in self._active (from open_positions fallback) are not overwritten so
        the two recovery paths can coexist safely.
        """
        _trade_states = ["PENDING", "ACTIVE", "TP1_HIT", "BE_ACTIVE"]
        try:
            from data.database import db as _om_db
            rows = await _om_db.load_tracked_signals(states=_trade_states)
        except Exception as _re:
            logger.warning(f"OutcomeMonitor.restore: DB load failed (non-fatal): {_re}")
            return

        restored = 0
        for row in rows:
            sid = row['signal_id']
            if sid in self._active:
                # Already registered (e.g. via open_positions recovery) — update
                # the protection fields that the legacy path doesn't supply.
                existing = self._active[sid]
                if row.get('entry_price') and not existing.entry_price:
                    existing.entry_price = row['entry_price']
                if row.get('be_stop') and existing.be_stop is None:
                    existing.be_stop = row['be_stop']
                if row.get('trail_stop') and existing.trail_stop is None:
                    existing.trail_stop = row['trail_stop']
                if row.get('max_r', 0) > existing.max_r:
                    existing.max_r = row['max_r']
                if row.get('entry_status') and not existing.entry_status:
                    existing.entry_status = row['entry_status']
                # Restore state (TP1_HIT → BE_ACTIVE state enum)
                try:
                    existing.state = SignalState(row['state'])
                except ValueError:
                    pass
                continue

            try:
                tracked = TrackedSignal(
                    signal_id=sid,
                    symbol=row['symbol'],
                    direction=row['direction'],
                    strategy=row['strategy'],
                    entry_low=row['entry_low'],
                    entry_high=row['entry_high'],
                    stop_loss=row['stop_loss'],
                    tp1=row.get('tp1', 0.0),
                    tp2=row.get('tp2', 0.0),
                    tp3=row.get('tp3'),
                    confidence=row.get('confidence', 0.0),
                    message_id=row.get('message_id'),
                    created_at=row.get('created_at', time.time()),
                    trail_pct=row.get('trail_pct', 0.40),
                    entry_price=row.get('entry_price'),
                    be_stop=row.get('be_stop'),
                    trail_stop=row.get('trail_stop'),
                    max_r=row.get('max_r', 0.0),
                    entry_status=row.get('entry_status'),
                    # FIX #4 (AUDIT): restore setup_class from DB so max-hold
                    # timeouts use the correct limit after restart.  The DB schema
                    # and save_tracked_signal() already persisted this column, but
                    # restore() never read it back — defaulting to "intraday".
                    setup_class=row.get('setup_class', 'intraday'),
                )
                # Restore activation timestamp
                if row.get('entry_price'):
                    tracked.activated_at = row.get('activated_at') or row.get('updated_at', time.time())
                # Restore state enum
                try:
                    tracked.state = SignalState(row['state'])
                except ValueError:
                    tracked.state = SignalState.PENDING

                self._active[sid] = tracked
                price_cache.subscribe(tracked.symbol)
                restored += 1
            except Exception as _row_err:
                logger.debug(f"OutcomeMonitor.restore: skipped row #{sid}: {_row_err}")

        if restored:
            logger.info(
                f"🔄 OutcomeMonitor restored {restored} active trade(s) from DB "
                f"(be_stop/trail_stop/max_r preserved)"
            )

    def start(self):
        """Start the outcome monitor background task."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("📡 OutcomeMonitor started")

    async def stop(self):
        """Stop the outcome monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def pause(self):
        """B6: Pause outcome monitoring during network outage (prevents false SL hits on stale prices)."""
        self._paused = True
        logger.warning("⏸ OutcomeMonitor paused (network offline — SL checks suspended)")

    def resume(self):
        """B6: Resume outcome monitoring after network restored."""
        self._paused = False
        logger.info("▶ OutcomeMonitor resumed (network online)")


    def track_signal(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        strategy: str,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: Optional[float],
        confidence: float,
        message_id: Optional[int] = None,
        trail_pct: float = OutcomeTracking.DEFAULT_TRAIL_PCT,
        raw_data: Optional[dict] = None,
        created_at: Optional[float] = None,  # FIX: accept original timestamp for position recovery
        regime: str = "UNKNOWN",             # FIX (P2-A): regime at signal creation time
        setup_class: str = "intraday",       # MAX-HOLD FIX: scalp/intraday/swing/positional
    ):
        """Register a new signal for tracking"""
        tracked = TrackedSignal(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            confidence=confidence,
            # FIX: use original created_at when recovering positions after restart.
            # Without this, a 20-hour-old signal that timed out before restart gets
            # a fresh 48-hour window instead of expiring as expected.
            created_at=created_at if created_at is not None else time.time(),
            trail_pct=trail_pct,
            message_id=message_id,
            raw_data=raw_data,
            regime=regime,
            setup_class=setup_class,
        )
        self._active[signal_id] = tracked
        price_cache.subscribe(symbol)
        logger.info(f"📡 Tracking signal #{signal_id}: {symbol} {direction}")
        # Persist immediately so a restart before the first price check doesn't
        # lose this active trade.
        try:
            from data.database import db as _om_db
            _safe_ensure_future(
                _om_db.save_tracked_signal(
                    signal_id=signal_id, symbol=symbol, direction=direction,
                    strategy=strategy, state="PENDING",
                    entry_low=entry_low, entry_high=entry_high,
                    stop_loss=stop_loss, confidence=confidence,
                    message_id=message_id,
                    tp1=tp1, tp2=tp2, tp3=tp3,
                    created_at=tracked.created_at,
                    setup_class=setup_class,
                    trail_pct=trail_pct,
                ),
                context=f"save_tracked #{signal_id}",
            )
        except Exception:
            pass

    async def _fetch_recent_candle(self, symbol: str):
        """Return the latest closed candle for wick/zone validation."""
        try:
            candles = await api.fetch_ohlcv(
                symbol,
                OutcomeTracking.WICK_CONFIRM_TIMEFRAME,
                limit=2,
                use_cache=False,
            )
            if not candles:
                return None
            if len(candles) >= 2:
                return candles[-2]
            return candles[-1]
        except Exception:
            return None

    async def _update_near_miss_feedback(self, prices: Dict[str, float]):
        """Feed live price data back into the near-miss tracker."""
        try:
            from analyzers.near_miss_tracker import near_miss_tracker
            pending_keys = near_miss_tracker.get_pending_keys()
        except Exception:
            return

        if not pending_keys:
            return

        snapshots: Dict[str, tuple] = {}
        now = time.time()

        for symbol, direction in pending_keys:
            snapshot = snapshots.get(symbol)
            if snapshot is None:
                current_price = prices.get(symbol)
                if current_price is None:
                    cache_key = f"near_miss:{symbol}"
                    last_fetch = self._api_fallback_times.get(cache_key, 0)
                    if now - last_fetch < Timing.API_FALLBACK_THROTTLE_SECS:
                        continue
                    try:
                        ticker = await api.fetch_ticker(symbol)
                        current_price = ticker.get('last') if ticker else None
                        if current_price is not None:
                            self._api_fallback_times[cache_key] = now
                    except Exception:
                        current_price = None

                if current_price is None:
                    continue

                try:
                    current_price = float(current_price)
                except (TypeError, ValueError):
                    continue

                candle = await self._fetch_recent_candle(symbol)
                if candle and len(candle) >= 4:
                    try:
                        high_price = float(candle[2])
                        low_price = float(candle[3])
                    except (TypeError, ValueError):
                        high_price = current_price
                        low_price = current_price
                else:
                    high_price = current_price
                    low_price = current_price

                snapshot = (high_price, low_price, current_price)
                snapshots[symbol] = snapshot

            near_miss_tracker.check_outcome(
                symbol, direction, snapshot[0], snapshot[1], snapshot[2]
            )

    @staticmethod
    def _derive_entry_status(tracked: TrackedSignal, fill_price: float) -> str:
        """Classify entry quality relative to the original zone."""
        if tracked.entry_low <= fill_price <= tracked.entry_high:
            return "IN_ZONE"

        zone_width = max(
            abs(tracked.entry_high - tracked.entry_low),
            ((tracked.entry_high + tracked.entry_low) / 2) * 0.001,
        )
        if tracked.direction == "LONG":
            overshoot = max(0.0, fill_price - tracked.entry_high) / zone_width
        else:
            overshoot = max(0.0, tracked.entry_low - fill_price) / zone_width
        return "LATE" if overshoot <= 1.0 else "EXTENDED"

    async def _get_candle_entry_fill(self, tracked: TrackedSignal, current_price: float):
        """Detect missed zone touches from candle ranges and infer a conservative fill."""
        candle = await self._fetch_recent_candle(tracked.symbol)
        if not candle or len(candle) < 5:
            return None, None

        _, _open, high, low, close, *_rest = candle
        touched_zone = not (high < tracked.entry_low or low > tracked.entry_high)
        if not touched_zone:
            return None, None

        if tracked.direction == "LONG":
            fill_price = min(max(current_price, tracked.entry_low), tracked.entry_high)
            if current_price > tracked.entry_high:
                fill_price = tracked.entry_high
        else:
            fill_price = min(max(current_price, tracked.entry_low), tracked.entry_high)
            if current_price < tracked.entry_low:
                fill_price = tracked.entry_low

        # If price is already back in the zone, treat as in-zone.
        if tracked.entry_low <= current_price <= tracked.entry_high:
            fill_price = current_price

        status_price = current_price if current_price is not None else close
        return fill_price, self._derive_entry_status(tracked, status_price)

    async def _is_stop_hunt_wick(self, tracked: TrackedSignal, stop_price: float) -> bool:
        """Ignore wick-only stop breaches when the last closed candle recovered."""
        candle = await self._fetch_recent_candle(tracked.symbol)
        if not candle or len(candle) < 5:
            return False

        _, _open, high, low, close, *_rest = candle
        if tracked.direction == "LONG":
            return low <= stop_price < close
        return high >= stop_price > close

    @staticmethod
    def _tp1_has_been_realized(tracked: TrackedSignal) -> bool:
        return tracked.state in (SignalState.TP1_HIT, SignalState.BE_ACTIVE)

    def _calc_effective_exit_pnl_r(self, tracked: TrackedSignal, exit_price: float) -> float:
        """Blend realised TP1 profit with the remainder left open after TP1."""
        exit_r = self._calc_pnl_r(tracked, exit_price)
        if not self._tp1_has_been_realized(tracked):
            return exit_r

        entry = tracked.entry_price or (tracked.entry_low + tracked.entry_high) / 2
        risk = abs(entry - tracked.stop_loss)
        if risk <= 0 or not tracked.tp1:
            return exit_r

        partial = max(0.0, min(1.0, OutcomeTracking.PARTIAL_CLOSE_AT_TP1_PCT))
        if partial <= 0:
            return exit_r

        tp1_r = abs(tracked.tp1 - entry) / risk
        return partial * tp1_r + (1.0 - partial) * exit_r

    def get_active_count(self) -> int:
        return len(self._active)

    def get_active_signals(self) -> Dict[int, TrackedSignal]:
        return dict(self._active)

    def remove_signal(self, signal_id: int):
        """Remove a signal from tracking (e.g., when invalidated before entry)"""
        removed = self._active.pop(signal_id, None)
        if removed:
            # FIX: only unsubscribe from price_cache when no other active signal still
            # needs this symbol. The old code always unsubscribed, causing concurrent
            # signals on the same symbol (e.g., two strategies both fire on ETHUSDT)
            # to lose price updates when the first one completes or is removed.
            still_needed = any(s.symbol == removed.symbol for s in self._active.values())
            if not still_needed:
                price_cache.unsubscribe(removed.symbol)
            logger.debug(f"Removed signal #{signal_id} from outcome monitoring")

    # ── Main monitoring loop ─────────────────────────────────

    async def _monitor_loop(self):
        """Check all active signals on a loop"""
        while self._running:
            try:
                if not self._active or self._paused:  # B6: skip if network offline
                    await asyncio.sleep(self._check_interval)
                    continue

                # Read prices from shared cache; fall back to API for active signals
                # FIX M6: stale/missing cache must not silently skip SL checks
                # FIX RATE: API fallback is now TTL-gated (60s minimum between calls
                # per symbol) to prevent hammering Binance when price_cache is cold.
                prices = {}
                for symbol in set(s.symbol for s in self._active.values()):
                    p = price_cache.get(symbol)
                    if p is not None:
                        prices[symbol] = p

                # Check each signal
                # EC5: Iterate over a snapshot to avoid dict-changed-size errors
                # when track_signal/remove_signal are called during await points.
                completed = []
                _now_check = time.time()
                for sig_id, tracked in list(self._active.items()):
                    current_price = prices.get(tracked.symbol)
                    if current_price is None:
                        # Fallback: only fetch from API if we haven't recently done so
                        _last_fetch = self._api_fallback_times.get(tracked.symbol, 0)
                        if _now_check - _last_fetch >= Timing.API_FALLBACK_THROTTLE_SECS:  # at most once per 60s
                            try:
                                from data.api_client import api as _api_fallback
                                ticker = await _api_fallback.fetch_ticker(tracked.symbol)
                                current_price = ticker.get('last') if ticker else None
                                if current_price is not None:
                                    self._api_fallback_times[tracked.symbol] = _now_check
                            except Exception:
                                current_price = None
                    if current_price is None:
                        continue

                    result = await self._check_signal(tracked, current_price)
                    if result:
                        completed.append(sig_id)
                    else:
                        tracked.last_price = current_price  # keep fresh for button queries

                        # R5-S2 FIX: Update unrealized P&L so get_effective_capital()
                        # reflects underwater positions. Without this, effective capital
                        # always equals raw capital and sizing ignores open losses.
                        try:
                            from core.portfolio_engine import portfolio_engine as _pe
                            if tracked.state in (SignalState.ACTIVE, SignalState.TP1_HIT,
                                                 SignalState.BE_ACTIVE):
                                _entry = tracked.entry_price or (
                                    (tracked.entry_low + tracked.entry_high) / 2
                                )
                                if _entry > 0:
                                    _pos = _pe._positions.get(sig_id)
                                    if _pos and _pos.size_usdt > 0:
                                        if tracked.direction == "LONG":
                                            _unr = (current_price - _entry) / _entry * _pos.size_usdt
                                        else:
                                            _unr = (_entry - current_price) / _entry * _pos.size_usdt
                                        await _pe.update_unrealized(sig_id, _unr)
                        except Exception:
                            pass

                # Remove completed signals
                for sig_id in completed:
                    del self._active[sig_id]

                await self._update_near_miss_feedback(prices)

                # Prune _api_fallback_times for symbols no longer tracked.
                # Without this, the dict grows forever since symbols are added
                # when the price-cache misses but never removed.
                _active_symbols = {t.symbol for t in self._active.values()}
                for _sym in list(self._api_fallback_times):
                    if _sym.startswith("near_miss:"):
                        # Near-miss snapshots are polled independently of active trades.
                        # Keep their throttle entries so blocked-signal feedback can
                        # continue even when no live trade is tracking the symbol.
                        continue
                    if _sym not in _active_symbols:
                        del self._api_fallback_times[_sym]

                await asyncio.sleep(self._check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Outcome monitor error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _check_signal(self, tracked: TrackedSignal, current_price: float) -> bool:
        """
        Check a single signal against current price.
        Returns True if signal is completed (should be removed from tracking).
        """
        now = time.time()
        is_long = tracked.direction == "LONG"

        # ── Timeout checks ────────────────────────────────────
        # Pending timeout: entry zone never reached
        if tracked.state == SignalState.PENDING:
            if now - tracked.created_at > self._pending_timeout:
                await self._complete(tracked, "EXPIRED", 0.0,
                                     "⏰ Signal expired — entry zone never reached")
                return True

            # Check if price entered entry zone
            _entry_fill = None
            _entry_status_val = None
            if tracked.entry_low <= current_price <= tracked.entry_high:
                _entry_fill = current_price
                _entry_status_val = "IN_ZONE"
            else:
                _entry_fill, _entry_status_val = await self._get_candle_entry_fill(tracked, current_price)

            if _entry_fill is not None:
                tracked.state = SignalState.ACTIVE
                tracked.entry_price = _entry_fill
                tracked.entry_status = _entry_status_val
                tracked.activated_at = now
                logger.info(
                    f"✅ Signal #{tracked.signal_id} ACTIVE: {tracked.symbol} "
                    f"entered at {_entry_fill} ({_entry_status_val})"
                )
                # Start funding-accrual tracker for this live position so
                # cumulative funding can later be folded into net PnL.
                try:
                    from signals.funding_bridge import funding_bridge as _fb
                    _fb.start_accrual(
                        tracked.signal_id,
                        tracked.symbol,
                        tracked.direction,
                        entry_ts=now,
                    )
                except Exception:
                    pass
                # Persist entry — entry_price is critical for all future P&L and
                # trailing stop calculations after a restart.
                try:
                    from data.database import db as _om_db_act
                    _safe_ensure_future(
                        _om_db_act.update_tracked_signal(
                            tracked.signal_id,
                            state="ACTIVE",
                            entry_price=_entry_fill,
                            entry_status=_entry_status_val,
                            activated_at=now,
                        ),
                        context=f"update_tracked ACTIVE #{tracked.signal_id}",
                    )
                except Exception:
                    pass
                # Write execution analytics — entry_price, entry_time, and entry_status
                # to signals table so the dashboard can show actual fill vs signal zone,
                # time-to-entry, and whether the entry was in-zone or late/extended.
                try:
                    _safe_ensure_future(
                        _om_db_act.update_signal_entry(
                            tracked.signal_id, _entry_fill, now, _entry_status_val
                        ),
                        context=f"update_signal_entry #{tracked.signal_id}",
                    )
                except Exception:
                    pass
                if self.on_entry_reached:
                    try:
                        await self.on_entry_reached(tracked.signal_id, current_price)
                    except Exception:
                        pass
                # Register with proactive alert engine
                try:
                    from signals.proactive_alerts import proactive_alerts as _pa
                    from analyzers.regime import regime_analyzer
                    _regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN') if regime_analyzer.regime else "UNKNOWN"
                    _pa.register_signal(tracked.signal_id, _regime)
                except Exception:
                    pass
                # Price is IN the entry zone — cannot also be below SL. Skip invalidation check.
                return False

            # Check if price blew through entry zone past SL (invalidated before entry).
            # FIX: was calling _complete with "EXPIRED" — wrong outcome. EXPIRED means
            # the signal timed out without entry; this is a structural invalidation.
            # The learning loop and engine callback both distinguish these outcomes.
            if is_long and current_price < tracked.stop_loss:
                await self._complete(tracked, "INVALIDATED", 0.0,
                                     "❌ Signal invalidated — SL hit before entry zone reached")
                return True
            elif not is_long and current_price > tracked.stop_loss:
                await self._complete(tracked, "INVALIDATED", 0.0,
                                     "❌ Signal invalidated — SL hit before entry zone reached")
                return True

            return False

        # ── Max-hold time exit (active trades only) ───────────
        # A scalp that runs 8+ hours is no longer a scalp — it has become an
        # accidental position trade without appropriate sizing.  Force-close at
        # current price so capital is freed and the loss/gain is realised cleanly.
        # Limits: scalp=4h, intraday=12h, swing=72h, positional=disabled (0)
        if tracked.state in (SignalState.ACTIVE, SignalState.BE_ACTIVE) and tracked.activated_at:
            from config.constants import Timing
            _MAX_HOLD_SECS = Timing.MAX_HOLD_BY_SETUP
            _hold_limit = _MAX_HOLD_SECS.get(getattr(tracked, 'setup_class', 'intraday'), 12 * 3600)
            if _hold_limit > 0 and (now - tracked.activated_at) > _hold_limit:
                _hold_hours = (now - tracked.activated_at) / 3600
                _pnl_r = self._calc_pnl_r(tracked, current_price)
                _outcome = "WIN" if _pnl_r > 0 else "LOSS"
                await self._complete(
                    tracked, _outcome, _pnl_r,
                    f"⏰ Max hold time reached ({_hold_hours:.1f}h > "
                    f"{_hold_limit//3600}h limit for {tracked.setup_class}) "
                    f"— closed at market"
                )
                return True

        # Overall signal timeout
        if now - tracked.created_at > self._signal_timeout:
            # Calculate current P&L at timeout
            pnl_r = self._calc_effective_exit_pnl_r(tracked, current_price)
            if pnl_r > 0.5:
                await self._complete(tracked, "WIN", pnl_r,
                                     f"⏰ Signal expired in profit (+{pnl_r:.1f}R)")
            elif tracked.state in (SignalState.TP1_HIT, SignalState.BE_ACTIVE):
                # FIX: use actual pnl_r — trade may still be in profit after TP1
                if pnl_r > 0.05:
                    await self._complete(tracked, "WIN", pnl_r,
                                         f"⏰ Signal expired in profit after TP1 (+{pnl_r:.1f}R)")
                else:
                    await self._complete(tracked, "BREAKEVEN", max(0.0, pnl_r),
                                         "⏰ Signal expired at breakeven (TP1 was hit)")
            else:
                await self._complete(tracked, "EXPIRED", 0.0,
                                     "⏰ Signal expired without resolution")
            return True

        # ── Price level checks ────────────────────────────────
        entry = tracked.entry_price or (tracked.entry_low + tracked.entry_high) / 2
        risk = abs(entry - tracked.stop_loss) if entry != tracked.stop_loss else 1.0

        # Track max R reached
        current_r = self._calc_pnl_r(tracked, current_price)
        tracked.max_r = max(tracked.max_r, current_r)

        # ── Early profit protection before TP1 ──────────────────
        if (
            tracked.state == SignalState.ACTIVE
            and tracked.be_stop is None
            and risk > 0
            and tracked.max_r >= OutcomeTracking.EARLY_TRAIL_START_R
        ):
            locked_r = tracked.max_r * OutcomeTracking.EARLY_TRAIL_FRACTION
            if is_long:
                _early_stop = entry + locked_r * risk
                _should_move = _early_stop > tracked.stop_loss
            else:
                _early_stop = entry - locked_r * risk
                _should_move = _early_stop < tracked.stop_loss

            if _should_move:
                tracked.be_stop = _early_stop
                tracked.trail_stop = _early_stop
                try:
                    from data.database import db as _om_db_early
                    _safe_ensure_future(
                        _om_db_early.update_tracked_signal(
                            tracked.signal_id,
                            be_stop=_early_stop,
                            trail_stop=_early_stop,
                            max_r=tracked.max_r,
                        ),
                        context=f"early_trail #{tracked.signal_id}",
                    )
                except Exception:
                    pass

        # ── Hourly check-in (only for active trades) ──────────────────────
        checkin_interval = 3600  # 1 hour
        last_checkin = tracked.last_checkin_at or tracked.activated_at or now
        # PHASE 3 FIX (CHECKIN-DARK): fire checkin regardless of message_id.
        # API-executed trades (no Telegram message) were previously silenced because
        # the engine's _handle_trade_checkin only edited existing messages.
        # The engine handler now sends a fresh message when message_id is None.
        if (tracked.state in (SignalState.ACTIVE, SignalState.BE_ACTIVE)
                and now - last_checkin >= checkin_interval
                and self.on_checkin):
            tracked.last_checkin_at = now
            try:
                await self.on_checkin(
                    tracked.signal_id,
                    tracked.symbol,
                    tracked.direction,
                    current_r,
                    tracked.max_r,
                    tracked.state.value,
                    tracked.tp1,
                    tracked.tp2,
                    tracked.tp3,
                    tracked.be_stop,
                    tracked.stop_loss,
                    tracked.message_id,
                )
            except Exception as e:
                logger.debug(f"Check-in callback error: {e}")

        # ── R8-F3: Enhanced Trailing TP System ────────────────────────
        # After TP1 is hit, instead of only static TP2/TP3 targets,
        # a trailing stop ratchets upward with price to capture extended moves.
        # Near TP2 zone, the trail tightens to lock in more profit.
        if tracked.state == SignalState.BE_ACTIVE and tracked.be_stop is not None:
            _trail_frac = getattr(tracked, 'trail_pct', 0.55)
            if _trail_frac > 0 and tracked.max_r > 0.5 and risk > 0:
                # Calculate distance to TP2 as a fraction (0 = at TP1, 1 = at TP2)
                tp1_r = abs(tracked.tp1 - entry) / risk if risk > 0 else 1.0
                tp2_r = abs(tracked.tp2 - entry) / risk if risk > 0 else 2.0
                progress_to_tp2 = min(1.0, (tracked.max_r - tp1_r) / max(0.1, tp2_r - tp1_r))

                # Tighten trail as price approaches TP2:
                #   At TP1: trail_frac stays at configured value (e.g. 0.55)
                #   At TP2: trail_frac tightens to 0.80 (lock in more profit)
                effective_trail = _trail_frac + (0.80 - _trail_frac) * progress_to_tp2
                effective_trail = min(0.90, effective_trail)  # Cap at 90%

                if is_long:
                    _new_trail = entry + (tracked.max_r * risk * effective_trail)
                    if _new_trail > tracked.be_stop:
                        _old_trail_long = tracked.be_stop
                        tracked.be_stop = _new_trail
                        tracked.trail_stop = _new_trail
                        if _tl:
                            try:
                                _tl.trail(signal_id=tracked.signal_id, symbol=tracked.symbol, direction=tracked.direction, old_stop=_old_trail_long, new_stop=_new_trail, max_r=tracked.max_r, progress_pct=progress_to_tp2)
                            except Exception:
                                pass
                        # Persist ratcheted trail so a restart preserves profit protection.
                        try:
                            from data.database import db as _om_db_trail
                            _safe_ensure_future(
                                _om_db_trail.update_tracked_signal(
                                    tracked.signal_id,
                                    be_stop=_new_trail,
                                    trail_stop=_new_trail,
                                    max_r=tracked.max_r,
                                ),
                                context=f"trail_long #{tracked.signal_id}",
                            )
                        except Exception:
                            pass
                else:
                    _new_trail = entry - (tracked.max_r * risk * effective_trail)
                    if _new_trail < tracked.be_stop:
                        _old_trail_short = tracked.be_stop
                        tracked.be_stop = _new_trail
                        tracked.trail_stop = _new_trail
                        if _tl:
                            try:
                                _tl.trail(signal_id=tracked.signal_id, symbol=tracked.symbol, direction=tracked.direction, old_stop=_old_trail_short, new_stop=_new_trail, max_r=tracked.max_r, progress_pct=progress_to_tp2)
                            except Exception:
                                pass
                        try:
                            from data.database import db as _om_db_trail
                            _safe_ensure_future(
                                _om_db_trail.update_tracked_signal(
                                    tracked.signal_id,
                                    be_stop=_new_trail,
                                    trail_stop=_new_trail,
                                    max_r=tracked.max_r,
                                ),
                                context=f"trail_short #{tracked.signal_id}",
                            )
                        except Exception:
                            pass

        # ── FIX-1: Elliott Wave count invalidation ────────────────────
        # If the wave count is violated (price takes out W2 for a W3 entry,
        # or W4 for a W5 entry), the Elliott basis for the trade is gone.
        # Exit immediately at current price rather than holding to the SL.
        # This avoids sitting in a structurally-invalid trade for hours.
        if (tracked.strategy == "ElliottWave"
                and tracked.state == SignalState.ACTIVE
                and tracked.raw_data):
            inv_level = tracked.raw_data.get('elliott_invalidation_level')
            if inv_level and inv_level > 0:
                violated = (
                    (is_long and current_price < inv_level * 0.999) or
                    (not is_long and current_price > inv_level * 1.001)
                )
                if violated:
                    pnl_r = self._calc_pnl_r(tracked, current_price)
                    await self._complete(
                        tracked, "LOSS", pnl_r,
                        f"📉 Elliott wave count invalidated — W2/W4 level {inv_level:.4f} taken out "
                        f"(exiting early at {pnl_r:.2f}R instead of holding to SL)"
                    )
                    return True

        # ── FIX-2: Wyckoff double-spring tolerance ────────────────────
        # A second, deeper spring below the first is a normal Wyckoff event.
        # Without this check the bot exits at SL; with it we widen the SL
        # dynamically to accommodate one additional spring test.
        # Only applies before TP1 is hit (after TP1, normal BE management takes over).
        if (tracked.strategy == "Wyckoff"
                and is_long
                and tracked.state == SignalState.ACTIVE
                and tracked.raw_data):
            spring_low = tracked.raw_data.get('spring_low')
            range_low = tracked.raw_data.get('range_low')
            if spring_low and range_low and current_price < tracked.stop_loss:
                # Check if this looks like a second spring (not just an SL hit):
                # price went below spring_low but is still above range_low × 0.97
                second_spring_floor = range_low * 0.97
                if current_price > second_spring_floor:
                    # Extend SL to accommodate the second spring
                    new_sl = second_spring_floor * 0.998
                    if new_sl < tracked.stop_loss:
                        tracked.stop_loss = new_sl
                        logger.info(
                            f"🔄 Wyckoff #{tracked.signal_id} {tracked.symbol}: "
                            f"second spring detected — SL extended to {new_sl:.4f} "
                            f"(second_spring_floor={second_spring_floor:.4f})"
                        )
                        # Don't exit — let the trade continue with the widened SL

        # ── R8-F2: Whale deposit protection — tighten stops ───────────
        # When whale deposits spike to critical levels, tighten all active
        # stops to breakeven to protect against coordinated sell-offs.
        try:
            from analyzers.whale_deposit_monitor import whale_deposit_monitor
            if whale_deposit_monitor.should_tighten_stops():
                if tracked.state == SignalState.ACTIVE and tracked.be_stop is None:
                    # Move stop to just below entry for protection
                    if is_long:
                        tracked.be_stop = entry - risk * 0.05
                    else:
                        tracked.be_stop = entry + risk * 0.05
                    logger.warning(
                        f"🐋 Whale deposit protection: #{tracked.signal_id} "
                        f"SL tightened to near-BE ({tracked.be_stop:.4f})"
                    )
        except ImportError:
            pass

        # ── Stop Loss hit ─────────────────────────────────────
        active_sl = tracked.be_stop if tracked.be_stop is not None else tracked.stop_loss

        if is_long and current_price <= active_sl:
            if await self._is_stop_hunt_wick(tracked, active_sl):
                return False
            effective_pnl_r = self._calc_effective_exit_pnl_r(tracked, current_price)
            if tracked.state in (SignalState.TP1_HIT, SignalState.BE_ACTIVE) or effective_pnl_r >= 0:
                # FIX: Use actual realized pnl_r — a trailing stop that fires
                # above entry is a PARTIAL_WIN, not a flat breakeven.  This
                # prevents distorting stats when trails lock in +0.5R–+1.5R profit.
                if effective_pnl_r > 0.05:
                    await self._complete(tracked, "WIN", effective_pnl_r,
                                         f"📈 Protective stop hit in profit (+{effective_pnl_r:.2f}R)")
                else:
                    await self._complete(tracked, "BREAKEVEN", max(0.0, effective_pnl_r),
                                         "⚡ Protective stop hit near breakeven")
            else:
                # Use actual realized pnl_r instead of hardcoded -1.0.
                # Entry at top of zone vs SL at entry_mid means real loss
                # can be -1.2R or more. Bayesian system learns from actual R.
                pnl_r = min(-0.1, effective_pnl_r)
                await self._complete(tracked, "LOSS", pnl_r,
                                     f"🛑 Stop loss hit ({pnl_r:.2f}R)")
            return True

        elif not is_long and current_price >= active_sl:
            if await self._is_stop_hunt_wick(tracked, active_sl):
                return False
            effective_pnl_r = self._calc_effective_exit_pnl_r(tracked, current_price)
            if tracked.state in (SignalState.TP1_HIT, SignalState.BE_ACTIVE) or effective_pnl_r >= 0:
                if effective_pnl_r > 0.05:
                    await self._complete(tracked, "WIN", effective_pnl_r,
                                         f"📈 Protective stop hit in profit (+{effective_pnl_r:.2f}R)")
                else:
                    await self._complete(tracked, "BREAKEVEN", max(0.0, effective_pnl_r),
                                         "⚡ Protective stop hit near breakeven")
            else:
                pnl_r = min(-0.1, effective_pnl_r)
                await self._complete(tracked, "LOSS", pnl_r,
                                     f"🛑 Stop loss hit ({pnl_r:.2f}R)")
            return True

        # ── TP3 hit (best outcome) ────────────────────────────
        if tracked.tp3:
            if (is_long and current_price >= tracked.tp3) or \
               (not is_long and current_price <= tracked.tp3):
                pnl_r = self._calc_effective_exit_pnl_r(tracked, current_price)
                await self._complete(tracked, "WIN", pnl_r,
                                     f"🏆 TP3 hit! (+{pnl_r:.1f}R)")
                return True

        # ── TP2 hit (primary win) ─────────────────────────────
        if (is_long and current_price >= tracked.tp2) or \
           (not is_long and current_price <= tracked.tp2):
            pnl_r = self._calc_effective_exit_pnl_r(tracked, current_price)
            await self._complete(tracked, "WIN", pnl_r,
                                 f"🎯 TP2 hit! (+{pnl_r:.1f}R)")
            return True

        # ── TP1 hit → move SL to breakeven ────────────────────
        if tracked.state == SignalState.ACTIVE:
            if (is_long and current_price >= tracked.tp1) or \
               (not is_long and current_price <= tracked.tp1):
                # Move stop to breakeven (entry price + small buffer)
                _existing_stop = tracked.be_stop
                if is_long:
                    _tp1_stop = entry + risk * 0.05  # Tiny buffer above entry
                    tracked.be_stop = max(_existing_stop or tracked.stop_loss, _tp1_stop)
                else:
                    _tp1_stop = entry - risk * 0.05
                    tracked.be_stop = min(_existing_stop or tracked.stop_loss, _tp1_stop)
                tracked.trail_stop = tracked.be_stop
                tracked.state = SignalState.BE_ACTIVE
                logger.info(f"✅ Signal #{tracked.signal_id} TP1 hit — SL moved to BE")
                if _tl:
                    try:
                        _tl.trail(signal_id=tracked.signal_id, symbol=tracked.symbol, direction=tracked.direction, old_stop=tracked.stop_loss, new_stop=tracked.be_stop, max_r=tracked.max_r, progress_pct=0.0, reason="TP1_hit")
                    except Exception:
                        pass
                # Persist be_stop immediately — this is the critical protection that
                # must survive a restart.  Without this, a recovered trade after TP1
                # uses the original hard SL instead of the breakeven stop.
                try:
                    from data.database import db as _om_db_tp1
                    _safe_ensure_future(
                        _om_db_tp1.update_tracked_signal(
                            tracked.signal_id,
                            state="BE_ACTIVE",
                            be_stop=tracked.be_stop,
                            max_r=tracked.max_r,
                        ),
                        context=f"TP1 BE_ACTIVE #{tracked.signal_id}",
                    )
                except Exception:
                    pass
                if self.on_state_change:
                    try:
                        await self.on_state_change(
                            tracked.signal_id,
                            "TP1_HIT",
                            tracked.symbol,
                            tracked.direction,
                            tracked.entry_price or ((tracked.entry_low + tracked.entry_high) / 2),
                            tracked.be_stop,
                            tracked.strategy,
                            tracked.message_id,
                        )
                    except Exception:
                        pass

        return False

    def _calc_pnl_r(self, tracked: TrackedSignal, current_price: float) -> float:
        """Calculate P&L in R-multiples"""
        entry = tracked.entry_price or (tracked.entry_low + tracked.entry_high) / 2
        risk = abs(entry - tracked.stop_loss)
        if risk == 0:
            return 0.0
        if tracked.direction == "LONG":
            return (current_price - entry) / risk
        else:
            return (entry - current_price) / risk

    def _calc_pnl_r_by_id(self, signal_id: int) -> float:
        """Public helper: calculate current R for a tracked signal using last known price."""
        tracked = self._active.get(signal_id)
        if not tracked or not tracked.entry_price:
            return 0.0
        # Use last_price if available, else entry (will show 0R)
        last_price = getattr(tracked, 'last_price', tracked.entry_price)
        return self._calc_pnl_r(tracked, last_price)

    def _calc_pnl_r_public(self, tracked: TrackedSignal, current_price: float) -> float:
        """Public alias for _calc_pnl_r — used by bot.py live update button."""
        return self._calc_pnl_r(tracked, current_price)

    def get_net_pnl_r(self, tracked: TrackedSignal, current_price: float) -> float:
        """Return ``_calc_pnl_r`` minus accrued-funding drag (also in R).

        Introduced for the funding-accrual tracker: lets callers display a
        funding-inclusive PnL without changing the semantics of the
        existing ``_calc_pnl_r`` (which remains the "price-only" figure
        used throughout the outcome pipeline and stats).

        Safe to call for signals with no accrual record — returns the raw
        R value unchanged.
        """
        raw_r = self._calc_pnl_r(tracked, current_price)
        try:
            from signals.funding_bridge import funding_bridge as _fb
            entry = tracked.entry_price or (tracked.entry_low + tracked.entry_high) / 2
            accrued_r = _fb.get_accrued_r(tracked.signal_id, entry, tracked.stop_loss)
        except Exception:
            accrued_r = 0.0
        return raw_r + accrued_r

    async def _complete(self, tracked: TrackedSignal, outcome: str, pnl_r: float, message: str):
        """Complete a signal with an outcome.

        FIX: EXPIRED signals now fire on_outcome so the engine always calls
        portfolio_engine.close_position() and risk_manager.record_outcome().
        Previously EXPIRED skipped the callback — portfolio slots stayed consumed
        indefinitely, eventually blocking all new A/B signals.
        """
        tracked.completed_at = time.time()
        tracked.pnl_r = pnl_r
        # Release the funding-accrual record — the position is closed and
        # no further cycles should be charged against this signal_id.
        try:
            from signals.funding_bridge import funding_bridge as _fb
            _fb.stop_accrual(tracked.signal_id)
        except Exception:
            pass
        # FIX: unsubscribe happens here for ALL outcomes including EXPIRED.
        # Previously remove_signal() was the only unsub path, but EXPIRED
        # signals bypassed remove_signal() — growing the price cache subscriber
        # set unboundedly over time.
        # FIX #1 (AUDIT): Only unsubscribe when no OTHER active signal still
        # needs this symbol's price feed. Without this check, completing one
        # signal on ETHUSDT kills the price feed for concurrent signals on
        # the same symbol (e.g. two strategies both tracking ETHUSDT).
        try:
            from core.price_cache import price_cache as _pc
            still_needed = any(
                s.symbol == tracked.symbol and s.signal_id != tracked.signal_id
                for s in self._active.values()
            )
            if not still_needed:
                _pc.unsubscribe(tracked.symbol)
        except Exception:
            pass  # Non-fatal — unsubscribe is a cleanup hint only

        # Remove from tracked_signals_v1 — the signal has reached a terminal state
        # and no longer needs to be recovered after a restart.
        try:
            from data.database import db as _om_db_cmp
            _safe_ensure_future(
                _om_db_cmp.delete_tracked_signal(tracked.signal_id),
                context=f"delete_tracked #{tracked.signal_id}",
            )
        except Exception:
            pass

        if outcome == "WIN":
            tracked.state = SignalState.COMPLETED_WIN
        elif outcome == "LOSS":
            tracked.state = SignalState.COMPLETED_LOSS
        elif outcome == "BREAKEVEN":
            tracked.state = SignalState.COMPLETED_BE
        else:
            tracked.state = SignalState.EXPIRED

        duration_min = (tracked.completed_at - tracked.created_at) / 60

        # Deregister from proactive alert engine
        try:
            from signals.proactive_alerts import proactive_alerts as _pa
            _pa.deregister_signal(tracked.signal_id)
        except Exception:
            pass

        logger.info(
            f"📊 Signal #{tracked.signal_id} {outcome}: {tracked.symbol} "
            f"{tracked.direction} | {pnl_r:+.1f}R | {duration_min:.0f}min | "
            f"Strategy: {tracked.strategy}"
        )
        # Derive exit_reason from the close message — used by both the trade audit
        # log and the DB update below so both always agree on the value.
        _ea_exit_price = tracked.last_price or tracked.entry_price or None
        _ea_msg_lower = message.lower()
        if "tp3" in _ea_msg_lower:
            _ea_exit_reason = "TP3"
        elif "tp2" in _ea_msg_lower:
            _ea_exit_reason = "TP2"
        elif "tp1" in _ea_msg_lower or "breakeven stop" in _ea_msg_lower:
            _ea_exit_reason = "BE"
        elif "trail" in _ea_msg_lower:
            _ea_exit_reason = "TRAIL"
        elif "stop loss" in _ea_msg_lower or "stop-loss" in _ea_msg_lower:
            _ea_exit_reason = "SL"
        elif outcome == "EXPIRED":
            _ea_exit_reason = "EXPIRED"
        elif "invalid" in _ea_msg_lower:
            _ea_exit_reason = "INVALIDATED"
        else:
            _ea_exit_reason = outcome

        if _tl:
            try:
                _exit_price = _ea_exit_price or 0.0
                _tl.outcome(
                    signal_id=tracked.signal_id, symbol=tracked.symbol,
                    direction=tracked.direction, result=outcome,
                    entry_price=tracked.entry_price or 0.0, exit_price=_exit_price,
                    pnl_r=pnl_r, max_r=tracked.max_r,
                    stop_loss=tracked.stop_loss, tp1=tracked.tp1, tp2=tracked.tp2,
                    duration_min=duration_min, strategy=tracked.strategy,
                    regime=tracked.regime,  # FIX (P2-A): pass regime stored at creation
                    entry_status=getattr(tracked, 'entry_status', None),
                    exit_reason=_ea_exit_reason,
                )
            except Exception:
                pass

        # Write execution analytics to signals table — exit_price, exit_reason, max_r.
        # These answer "how did the trade close and what was the best R reached?".
        # The COALESCE-safe update_signal_outcome() ensures a subsequent call from
        # engine._handle_signal_outcome() (which lacks these values) won't overwrite them.
        try:
            from data.database import db as _om_db_exit
            _safe_ensure_future(
                _om_db_exit.update_signal_outcome(
                    tracked.signal_id, outcome, pnl_r,
                    exit_price=_ea_exit_price,
                    exit_reason=_ea_exit_reason,
                    max_r=tracked.max_r,
                ),
                context=f"update_signal_outcome #{tracked.signal_id}",
            )
        except Exception:
            pass

        # FIX: fire on_outcome for ALL terminal outcomes including EXPIRED.
        # This ensures engine._handle_signal_outcome() always runs, which:
        #   - calls portfolio_engine.close_position() (frees the position slot)
        #   - calls risk_manager.record_outcome() (keeps daily loss counter accurate)
        #   - calls learning_loop.record_trade() (feeds the feedback loop)
        # EXPIRED trades have pnl_r=0.0 and outcome="EXPIRED" so the engine
        # handles them correctly (no win/loss, no circuit breaker increment).
        if self.on_outcome:
            try:
                await self.on_outcome(
                    tracked.signal_id,
                    outcome,
                    pnl_r,
                    tracked.strategy,
                    tracked.message_id,
                    message,
                    tracked.symbol,
                    tracked.direction,
                    tracked.confidence,
                    tracked.stop_loss,
                )
            except Exception as e:
                logger.error(f"Outcome callback error: {e}")

        # FIX 3C + GROUP 4B: On EXPIRED, fetch current price and store as post_expiry_price.
        # This gives the AI everything it needs to audit direction accuracy on missed entries.
        if outcome == "EXPIRED":
            # Post-expiry cooldown: extend dedup window to 4h so near-identical signals
            # don't immediately re-appear (prevents WLFI SHORT regenerating every hour)
            try:
                from signals.aggregator import aggregator as _agg
                _key = f"{tracked.symbol}:{tracked.direction}"
                # Mark as "just expired" — dedup will use extended window
                import time as _t
                from signals.aggregator import _DedupEntry
                _agg._deduplicator._recent[_key] = _DedupEntry(
                    ts=_t.time(),
                    conf=getattr(tracked, 'confidence', 0),
                    grade=getattr(tracked, 'grade', 'B'),
                )
                # Override conf_breakthrough requirement to +20 pts (vs normal +15)
                # so near-identical levels can't slip through for 4 hours
                _agg._deduplicator._post_expiry[_key] = _t.time()
            except Exception:
                pass
            try:
                from data.database import db
                from core.price_cache import price_cache

                _current_price = price_cache.get(tracked.symbol)
                if _current_price and _current_price > 0:
                    await db.write_post_expiry_price(tracked.signal_id, _current_price)
                    logger.debug(
                        f"📊 Post-expiry price stored: {tracked.symbol} "
                        f"@ {_current_price:.6f} (signal #{tracked.signal_id})"
                    )

                # Trigger GROUP 4B direction accuracy audit when we accumulate 3+ expiries
                _expired_today = [
                    s for s in await db.get_recent_signals(hours=24)
                    if s.get("outcome") == "EXPIRED" and s.get("post_expiry_price")
                ]
                if len(_expired_today) >= 3:
                    try:
                        from analyzers.ai_analyst import ai_analyst
                        asyncio.create_task(
                            ai_analyst.post_expiry_direction_audit(_expired_today)
                        )
                    except Exception:
                        pass
            except Exception as _pep_err:
                logger.debug(f"Post-expiry price fetch error: {_pep_err}")

    def get_stats(self) -> Dict:
        """Stats for /status command"""
        states = {}
        for tracked in self._active.values():
            state = tracked.state.value
            states[state] = states.get(state, 0) + 1
        return {
            'active_signals': len(self._active),
            'states': states,
        }


# ── Singleton ──────────────────────────────────────────────
outcome_monitor = OutcomeMonitor()
