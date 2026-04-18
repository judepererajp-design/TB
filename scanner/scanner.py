"""
TitanBot Pro — Scanner
========================
Manages the full scanning pipeline:
  - Universe building (which symbols to scan)
  - Tier classification (Tier 1/2/3 based on volume)
  - Auto-promotion when volume spikes
  - Stalker engine (pre-breakout watchlist)
  - Whale detection
  - Priority queue for scan scheduling
"""

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from config.loader import cfg
from data.api_client import api
from data.database import db
from utils.formatting import fmt_price

# Lazy import to avoid circular dependency at module load time.
# Accessed via _get_regime() below.
def _get_regime() -> str:
    """Return the current regime string, or 'UNKNOWN' if not yet initialised."""
    try:
        from analyzers.regime import regime_analyzer
        return str(regime_analyzer.regime.value)
    except Exception:
        return "UNKNOWN"


def _regime_recently_changed(within_secs: int = 600) -> bool:
    """Return True iff the regime changed within the last ``within_secs`` seconds."""
    try:
        from analyzers.regime import regime_analyzer
        return regime_analyzer.is_recently_transitioned(within_secs)
    except Exception:
        return False

logger = logging.getLogger(__name__)


class Tier(int, Enum):
    TIER1 = 1   # $5M+ volume — scanned every 2 min
    TIER2 = 2   # $1M-$5M — scanned every 5 min
    TIER3 = 3   # $200K-$1M — scanned every 15 min


@dataclass
class SymbolState:
    symbol: str
    tier: Tier = Tier.TIER2
    volume_24h: float = 0.0
    volume_ma: float = 0.0          # 7-day MA for spike detection
    last_scan: float = 0.0
    last_signal: float = 0.0
    activity_score: float = 0.0
    promoted_at: float = 0.0        # Timestamp of last promotion
    scan_count: int = 0
    category: str = "UNKNOWN"


class Scanner:
    """
    Manages the symbol universe and scan scheduling.
    The engine calls scan_due_symbols() in a loop.
    """

    def __init__(self):
        self._scan_cfg = cfg.scanning
        self._sys_cfg  = cfg.system

        # Symbol registry
        self._symbols: Dict[str, SymbolState] = {}

        # Scan intervals (seconds)
        self._tier_intervals = {
            Tier.TIER1: self._sys_cfg.get('tier1_interval', 120),
            Tier.TIER2: self._sys_cfg.get('tier2_interval', 300),
            Tier.TIER3: self._sys_cfg.get('tier3_interval', 900),
        }

        # Whale cooldown tracker
        self._whale_cooldown: Dict[str, float] = {}
        # Whale persistence cache: symbol → {side: last_snapshot_usd}.
        # We require a whale to appear in 2 consecutive order-book snapshots
        # before firing so that spoofed / fleeting walls are filtered out.
        self._whale_snapshot: Dict[str, Dict[str, float]] = {}

        # Watchlist alert cooldown — prevents same symbol spamming every scan cycle
        self._watchlist_cooldown: Dict[str, float] = {}
        self._watchlist_cooldown_secs: int = 3600  # 1 hour between alerts per symbol

        # Cross-path signal dedup: (symbol, setup_type) → last_alert_ts.
        # Prevents scanner.stalker_scan and stalker.StalkerEngine from both
        # firing an alert for the same symbol / setup within the same window.
        self._signal_dedup: Dict[Tuple[str, str], float] = {}
        self._signal_dedup_ttl: int = 1800  # 30-minute dedup window

        # OHLCV data-quality cooldown — symbols that fail N times are auto-excluded
        self._ohlcv_fail_counts: Dict[str, int] = {}      # symbol → consecutive failures
        self._ohlcv_cooldown_until: Dict[str, float] = {} # symbol → cooldown expiry ts

        # Stalker watchlist
        self._stalker_scores: Dict[str, float] = {}

        # Universe refresh
        self._last_universe_refresh: float = 0
        self._universe_ttl = self._sys_cfg.get('universe_refresh', 3600)

        self._lock = asyncio.Lock()

    # ── Universe Management ───────────────────────────────────

    async def build_universe(self):
        """
        Build/refresh the symbol universe from Binance.
        Called at startup and every hour.
        """
        # Fast path: skip lock acquisition entirely when no refresh is due.
        # Callers that arrive while a refresh is already running will get an
        # early return here or on the second check inside the lock below.
        if time.time() - self._last_universe_refresh < self._universe_ttl:
            return

        async with self._lock:
            # Re-check after acquiring the lock so a concurrent caller that was
            # waiting does not trigger a redundant back-to-back refresh.
            now = time.time()
            if now - self._last_universe_refresh < self._universe_ttl:
                return

            logger.info("Building symbol universe...")

            # Fetch all tickers to get volumes
            all_tickers = await api.fetch_tickers()
            if not all_tickers:
                logger.error("Failed to fetch tickers for universe build")
                return

            tier1_cfg = self._scan_cfg.tier1
            tier2_cfg = self._scan_cfg.tier2
            tier3_cfg = self._scan_cfg.tier3

            t1_max = getattr(tier1_cfg, 'max_symbols', 80)
            t2_max = getattr(tier2_cfg, 'max_symbols', 80)
            t3_max = getattr(tier3_cfg, 'max_symbols', 40)

            t1_min_vol = getattr(tier1_cfg, 'min_volume_24h', 5_000_000)
            t2_min_vol = getattr(tier2_cfg, 'min_volume_24h', 1_000_000)
            t3_min_vol = getattr(tier3_cfg, 'min_volume_24h', 200_000)

            excluded = set(cfg.exchange.get('excluded_symbols', [])
                           if isinstance(cfg.exchange.get('excluded_symbols', []), list)
                           else [])

            # Sort by volume descending
            qualified = []
            for symbol, ticker in all_tickers.items():
                base_symbol = symbol.split(':')[0]
                if not base_symbol.endswith('/USDT'):
                    continue
                if base_symbol in excluded:
                    continue
                vol = float(ticker.get('quoteVolume', 0) or 0)
                if vol >= t3_min_vol:
                    qualified.append((base_symbol, vol))

            seen = {}
            for sym, vol in qualified:
                if sym not in seen or vol > seen[sym]:
                    seen[sym] = vol
            qualified = sorted(seen.items(), key=lambda x: x[1], reverse=True)

            # Assign tiers
            new_symbols: Dict[str, SymbolState] = {}
            t1_count = t2_count = t3_count = 0

            for symbol, vol in qualified:
                if vol >= t1_min_vol and t1_count < t1_max:
                    tier = Tier.TIER1
                    t1_count += 1
                elif vol >= t2_min_vol and t2_count < t2_max:
                    tier = Tier.TIER2
                    t2_count += 1
                elif vol >= t3_min_vol and t3_count < t3_max:
                    tier = Tier.TIER3
                    t3_count += 1
                else:
                    continue

                # Preserve existing state if symbol was already tracked
                if symbol in self._symbols:
                    existing = self._symbols[symbol]
                    existing.volume_24h = vol
                    existing.tier = tier
                    new_symbols[symbol] = existing
                else:
                    new_symbols[symbol] = SymbolState(
                        symbol=symbol,
                        tier=tier,
                        volume_24h=vol,
                        volume_ma=vol
                    )

                # Persist to DB
                await db.upsert_symbol_tier(symbol, tier.value, vol)

            self._symbols = new_symbols
            self._last_universe_refresh = now

            # Prune cooldown entries that belong to symbols no longer in the
            # universe so the dicts don't grow unboundedly on long-running instances.
            active_symbols = set(new_symbols)
            for _cd in (
                self._whale_cooldown,
                self._watchlist_cooldown,
                self._ohlcv_fail_counts,
                self._ohlcv_cooldown_until,
                self._whale_snapshot,
                self._signal_dedup,
            ):
                for _k in [k for k in _cd if k not in active_symbols]:
                    _cd.pop(_k, None)

            logger.info(
                f"Universe built: {t1_count} T1 | {t2_count} T2 | {t3_count} T3 "
                f"| Total: {len(new_symbols)}"
            )

    def get_symbols_by_tier(self, tier: Tier) -> List[str]:
        """Get all symbols in a specific tier"""
        return [s for s, state in self._symbols.items() if state.tier == tier]

    def get_all_symbols(self) -> List[str]:
        return list(self._symbols.keys())

    # ── Scan Scheduling ────────────────────────────────────────

    def get_due_symbols(self) -> List[str]:
        """
        Return symbols that are due for scanning right now.
        Priority: Tier 1 first, then 2, then 3.
        """
        now = time.time()
        due = []

        # FIX: iterate over a snapshot of items(). build_universe() runs concurrently
        # in an asyncio task and can mutate _symbols while we iterate, raising
        # RuntimeError: dictionary changed size during iteration.
        for symbol, state in list(self._symbols.items()):
            interval = self._tier_intervals.get(state.tier, 300)
            if now - state.last_scan >= interval:
                due.append(symbol)

        # Sort by tier priority (T1 first) then by how overdue they are
        def priority(sym):
            state = self._symbols.get(sym)
            if not state:
                return (99, 0)
            overdue = (time.time() - state.last_scan) / self._tier_intervals.get(state.tier, 300)
            return (state.tier.value, -overdue)  # Lower tier = higher priority

        due.sort(key=priority)
        return due

    def mark_scanned(self, symbol: str):
        """Mark a symbol as just scanned"""
        # FIX: _symbols is written from multiple concurrent scan tasks. Python's GIL
        # protects individual dict operations, but the read-then-write sequence
        # (check `in` then assign) is not atomic across the GIL. Use direct dict
        # access with a default to make each write a single atomic operation.
        state = self._symbols.get(symbol)
        if state:
            state.last_scan = time.time()
            state.scan_count += 1

    def exclude_symbol(self, symbol: str):
        """E3: Permanently remove a delisted/bad symbol from the scan universe."""
        # FIX: pop() is atomic at the GIL level — safe without a lock
        self._symbols.pop(symbol, None)
        logger.info(f"Scanner: excluded {symbol} from universe (delisted or bad symbol)")

    # ── OHLCV data-quality cooldown ──────────────────────────────

    def record_ohlcv_fail(self, symbol: str) -> bool:
        """
        Record a data-quality failure for *symbol*.
        Returns True if the symbol just entered cooldown (caller should skip it).
        """
        from config.constants import OHLCVCooldown

        count = self._ohlcv_fail_counts.get(symbol, 0) + 1
        self._ohlcv_fail_counts[symbol] = count

        if count >= OHLCVCooldown.FAIL_THRESHOLD:
            self._ohlcv_cooldown_until[symbol] = time.time() + OHLCVCooldown.COOLDOWN_SECS
            self._ohlcv_fail_counts.pop(symbol, None)  # reset counter
            logger.warning(
                f"📊 OHLCV cooldown: {symbol} failed {count}× — "
                f"excluding for {OHLCVCooldown.COOLDOWN_SECS}s"
            )
            return True
        return False

    def record_ohlcv_success(self, symbol: str):
        """Clear the failure counter on a successful OHLCV fetch."""
        self._ohlcv_fail_counts.pop(symbol, None)

    def is_ohlcv_cooled_down(self, symbol: str) -> bool:
        """Return True if *symbol* is still in OHLCV cooldown."""
        until = self._ohlcv_cooldown_until.get(symbol)
        if until is None:
            return False
        if time.time() >= until:
            self._ohlcv_cooldown_until.pop(symbol, None)
            logger.info(f"📊 OHLCV cooldown expired: {symbol} — re-enabling")
            return False
        return True

    def mark_signal(self, symbol: str):
        """Mark that a signal was generated for this symbol"""
        # FIX: same as mark_scanned — use direct state mutation via .get()
        state = self._symbols.get(symbol)
        if state:
            state.last_signal = time.time()
            state.activity_score += 10

    # ── Auto-Promotion ─────────────────────────────────────────

    async def check_promotions(self, symbol: str, current_volume: float) -> Optional[Tuple[int, int]]:
        """
        Check if a symbol should be promoted based on volume spike.
        Returns (from_tier, to_tier) if promoted, else None.
        """
        # Use .get() to avoid a TOCTOU race between the membership test and
        # the dict lookup — build_universe() can replace _symbols concurrently.
        state = self._symbols.get(symbol)
        if state is None:
            return None

        prom_cfg = self._scan_cfg.auto_promotion

        if not getattr(prom_cfg, 'enabled', True):
            return None

        # Always update the volume MA *before* computing the spike ratio so it
        # stays current whether or not a promotion fires this cycle.
        # Use a time-weighted EMA so the effective half-life is ~7 days for all
        # tiers (Tier-1 scans every 2 min, Tier-3 every 15 min).
        elapsed = time.time() - state.last_scan if state.last_scan > 0 else 300.0
        # Use a regime-dependent time constant:
        # CHOPPY/VOLATILE → shorter τ (1 day) to react faster to volume spikes.
        # Trending regimes → longer τ (3 days) to avoid false promotions on spikes.
        _regime = _get_regime()
        if _regime in ("CHOPPY", "VOLATILE", "VOLATILE_PANIC"):
            tau = 1 * 24 * 3600.0   # 1-day time constant
        else:
            tau = 3 * 24 * 3600.0   # 3-day time constant
        alpha = 1.0 - math.exp(-elapsed / tau)
        state.volume_ma = state.volume_ma * (1.0 - alpha) + current_volume * alpha

        vol_ma = state.volume_ma if state.volume_ma > 0 else current_volume
        vol_spike_mult = current_volume / vol_ma if vol_ma > 0 else 1.0
        promotion_threshold = getattr(prom_cfg, 'volume_spike_multiplier', 3.0)
        min_vol = getattr(prom_cfg, 'min_volume_for_promotion', 500_000)

        # Cooldown check
        cooldown_hours = getattr(prom_cfg, 'cooldown_hours', 24)
        if time.time() - state.promoted_at < cooldown_hours * 3600:
            return None

        if (vol_spike_mult >= promotion_threshold and
                current_volume >= min_vol and
                state.tier != Tier.TIER1):

            from_tier = state.tier.value
            new_tier = Tier(max(1, state.tier.value - 1))  # Promote one tier
            to_tier = new_tier.value

            state.tier = new_tier
            state.promoted_at = time.time()
            state.volume_24h = current_volume
            # Increase scan priority immediately
            state.last_scan = 0  # Force immediate rescan

            await db.record_promotion(symbol, from_tier, to_tier)
            logger.info(f"🚀 {symbol} promoted: Tier {from_tier} → Tier {to_tier} ({vol_spike_mult:.1f}x volume)")

            return (from_tier, to_tier)

        return None

    # ── Whale Detection ────────────────────────────────────────

    async def check_whale_activity(
        self, symbol: str, order_book: Dict
    ) -> Optional[Dict]:
        """
        Detect whale orders in the order book.
        Returns whale info dict if detected.
        """
        whale_cfg = self._scan_cfg.whale_detection
        if not getattr(whale_cfg, 'enabled', True):
            return None

        min_order_usd = getattr(whale_cfg, 'min_order_usd', 75_000)
        cooldown_min  = getattr(whale_cfg, 'cooldown_minutes', 45)

        # Cooldown check
        last_whale = self._whale_cooldown.get(symbol, 0)
        if time.time() - last_whale < cooldown_min * 60:
            return None

        if not order_book:
            return None

        # Fix S6: get current price from symbol state (set during universe build from ticker data)
        state = self._symbols.get(symbol)
        if not state:
            return None

        # Use last known price from order book or symbol state
        # order_book bids/asks have prices — use mid of best bid/ask
        best_bid = float(order_book.get('bids', [[]])[0][0]) if order_book.get('bids') else 0.0
        best_ask = float(order_book.get('asks', [[]])[0][0]) if order_book.get('asks') else 0.0
        if best_bid > 0 and best_ask > 0:
            current_price = (best_bid + best_ask) / 2
        elif best_bid > 0:
            current_price = best_bid
        elif best_ask > 0:
            current_price = best_ask
        else:
            # Cannot derive price from an empty order book — skip rather than
            # using a bogus fallback that would corrupt USD threshold comparisons.
            return None

        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        # ── compute spread for fill-probability scoring ─────────
        spread_pct = abs(best_ask - best_bid) / current_price if best_ask > 0 and best_bid > 0 else 0.0

        def _fill_prob(order_price: float, side: str, order_usd: float, threshold: float) -> float:
            """
            Quick fill-probability heuristic (0.0–1.0).
            Factors: distance from mid, relative spread, size relative to threshold.
            """
            dist_pct  = abs(order_price - current_price) / current_price
            size_ratio = min(order_usd / (threshold * 3), 1.0)   # saturates at 3× threshold
            # Large spread or far-from-mid → lower fill probability
            fill_score = max(0.0, 1.0 - dist_pct * 10 - spread_pct * 5) * (0.5 + size_ratio * 0.5)
            return round(min(1.0, max(0.0, fill_score)), 3)

        # Check up to 50 levels.  Detect both concentrated single-level whale
        # walls AND distributed / iceberg orders accumulated within 1% of mid.
        price_band = current_price * 0.01
        for side, orders in [('buy', bids), ('sell', asks)]:
            cumulative_usd = 0.0
            for price, qty in orders[:50]:
                price     = float(price)
                qty       = float(qty)
                order_usd = price * qty

                # Single-level whale wall
                if order_usd >= min_order_usd:
                    # ── Persistence check (relative ratio) ─────────
                    # Require prev_snapshot >= 50% of current snapshot so that the
                    # persistence bar scales with order size rather than being
                    # anchored to the fixed min_order_usd threshold.
                    # This filters spoofed walls that vanish between polls while
                    # still allowing genuine whales that grew since last snapshot.
                    prev = self._whale_snapshot.setdefault(symbol, {}).get(side, 0.0)
                    self._whale_snapshot[symbol][side] = order_usd
                    if prev == 0.0 or (prev / order_usd) < 0.5:
                        # First sighting or too small relative to current — record but don't fire yet
                        logger.debug(
                            f"🐋 Whale candidate (first sighting): "
                            f"{symbol} {side} ${order_usd:,.0f} @ {price}"
                        )
                        break  # still check iceberg on same side below

                    self._whale_cooldown[symbol] = time.time()
                    fp = _fill_prob(price, side, order_usd, min_order_usd)
                    logger.info(
                        f"🐋 Whale order: {symbol} {side} ${order_usd:,.0f} @ {price}"
                        f"  fill_prob={fp:.2f}"
                    )
                    return {
                        'symbol':    symbol,
                        'side':      side,
                        'order_usd': order_usd,
                        'price':     price,
                        'qty':       qty,
                        'fill_prob': fp,
                        'spread_pct': spread_pct,
                    }

                # Accumulate within the 1% mid-price band to catch iceberg orders
                if abs(price - current_price) <= price_band:
                    cumulative_usd += order_usd

            # Distributed / iceberg whale — same relative persistence check
            if cumulative_usd >= min_order_usd:
                prev = self._whale_snapshot.setdefault(symbol, {}).get(f"iceberg_{side}", 0.0)
                self._whale_snapshot[symbol][f"iceberg_{side}"] = cumulative_usd
                if prev == 0.0 or (prev / cumulative_usd) < 0.5:
                    logger.debug(
                        f"🐋 Iceberg candidate (first sighting): "
                        f"{symbol} {side} ${cumulative_usd:,.0f} cumulative"
                    )
                    continue

                self._whale_cooldown[symbol] = time.time()
                fp = _fill_prob(current_price, side, cumulative_usd, min_order_usd)
                logger.info(
                    f"🐋 Iceberg whale: {symbol} {side} "
                    f"${cumulative_usd:,.0f} cumulative within 1% band"
                    f"  fill_prob={fp:.2f}"
                )
                return {
                    'symbol':    symbol,
                    'side':      side,
                    'order_usd': cumulative_usd,
                    'price':     current_price,
                    'qty':       cumulative_usd / current_price,
                    'fill_prob': fp,
                    'spread_pct': spread_pct,
                }

        # Order book checked; update snapshot for sides that saw no whale this cycle
        # (so previous sightings don't carry indefinitely).
        self._whale_snapshot.setdefault(symbol, {})
        for side in ('buy', 'sell', 'iceberg_buy', 'iceberg_sell'):
            if side not in self._whale_snapshot[symbol]:
                self._whale_snapshot[symbol][side] = 0.0

        return None

    # ── Stalker Engine ────────────────────────────────────────

    async def stalker_scan(self, symbol: str, ohlcv: List) -> Optional[Dict]:
        """
        Pre-breakout detection for watchlist.
        Looks for coiling, compression, and pre-breakout conditions.
        Returns dict with score and reasons if interesting.
        """
        if not ohlcv or len(ohlcv) < 30:
            return None

        # Cooldown gate — check before any expensive computation
        _last = self._watchlist_cooldown.get(symbol, 0)
        if time.time() - _last < self._watchlist_cooldown_secs:
            return None

        # Dynamic threshold: raise the bar in CHOPPY (signal flood risk),
        # lower it in trending regimes (signals are scarcer but higher quality).
        # Hysteresis: damp adjustments 25% within 10 min of a regime flip to
        # prevent violent over-correction right after a transition.
        _regime = _get_regime()
        _recently_flipped = _regime_recently_changed(600)
        if _regime == "CHOPPY":
            _raw_adj = 8
        elif _regime in ("VOLATILE", "VOLATILE_PANIC"):
            _raw_adj = 5
        elif _regime in ("BULL_TREND", "BEAR_TREND"):
            _raw_adj = -5
        else:
            _raw_adj = 0
        if _recently_flipped:
            _raw_adj = round(_raw_adj * 0.75)   # damp by 25% during transition
        _min_score = 50 + _raw_adj

        # Regime-dependent dedup TTL:
        # TREND → signals evolve slowly, 30 min window avoids duplicate noise.
        # CHOPPY → signals flip quickly, 15 min window allows valid re-entries.
        if _regime in ("BULL_TREND", "BEAR_TREND"):
            _dedup_ttl = self._signal_dedup_ttl        # 30 min
        else:
            _dedup_ttl = self._signal_dedup_ttl // 2   # 15 min in chop/volatile

        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df = df.astype({'open':float,'high':float,'low':float,'close':float,'volume':float})

        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values

        score = 0
        reasons = []

        # ── 1. Bollinger Band squeeze (volatility compression) ─
        bb_period = 20
        if len(closes) >= bb_period:
            window = closes[-bb_period:]
            bb_mean = np.mean(window)
            bb_std  = np.std(window)
            bb_width = (bb_std * 4) / bb_mean if bb_mean > 0 else 0.0  # Standard Bollinger Bandwidth

            # Lowest bandwidth in 48 bars = squeeze
            if len(closes) >= 48:
                widths_hist = []
                for i in range(len(closes) - 48, len(closes)):
                    w = closes[max(0,i-20):i]
                    if len(w) >= 10:
                        m = np.mean(w)
                        widths_hist.append((np.std(w) * 4) / m if m > 0 else 0.0)
                if widths_hist and bb_width < np.percentile(widths_hist, 20):
                    score += 30
                    reasons.append("🌀 Bollinger squeeze — lowest volatility in 48 bars")

        # ── 2. Key level proximity ─────────────────────────────
        period_high = np.max(highs[-20:])
        period_low  = np.min(lows[-20:])
        current     = closes[-1]

        high_dist = (period_high - current) / current
        low_dist  = (current - period_low) / current

        if high_dist < 0.015:   # Within 1.5% of 20-bar high
            score += 20
            reasons.append(f"📈 Testing 20-bar high ({fmt_price(period_high)}) — breakout alert")
        elif low_dist < 0.015:
            score += 20
            reasons.append(f"📉 Testing 20-bar low ({fmt_price(period_low)}) — breakdown alert")

        # ── 3. Volume declining (coiling) ─────────────────────
        avg_vol_20 = np.mean(volumes[-20:])
        avg_vol_5  = np.mean(volumes[-5:])
        if avg_vol_5 < avg_vol_20 * 0.6:
            score += 15
            reasons.append("📊 Volume declining — coiling for breakout")

        # ── 4. RSI divergence ─────────────────────────────────
        if len(closes) >= 28:
            rsi_now  = self._quick_rsi(closes, 14)
            rsi_prev = self._quick_rsi(closes[:-5], 14)
            price_up = closes[-1] > closes[-6]

            if price_up and rsi_now < rsi_prev - 5:
                score += 15
                reasons.append("⚡ RSI bearish divergence — momentum weakening")
            elif not price_up and rsi_now > rsi_prev + 5:
                score += 15
                reasons.append("⚡ RSI bullish divergence — accumulation signal")

        # ── 5. ATR compression ────────────────────────────────
        if len(closes) >= 30:
            atr_now = self._quick_atr(highs[-14:], lows[-14:], closes[-14:])
            atr_old = self._quick_atr(highs[-28:-14], lows[-28:-14], closes[-28:-14])
            if atr_old > 0 and atr_now < atr_old * 0.5:
                score += 10
                reasons.append(f"🔇 ATR compressed {atr_now/atr_old:.0%} of normal")

        if score >= _min_score:
            # Cross-path dedup: if stalker.StalkerEngine already raised this
            # symbol / setup within the regime-dependent TTL window, don't double-fire.
            setup_type = "pre_breakout"
            dedup_key  = (symbol, setup_type)
            now        = time.time()
            if now - self._signal_dedup.get(dedup_key, 0) < _dedup_ttl:
                return None

            await db.upsert_watchlist(symbol, float(score), reasons)
            self._watchlist_cooldown[symbol] = now
            self._signal_dedup[dedup_key]    = now
            return {'symbol': symbol, 'score': score, 'reasons': reasons, 'regime': _regime}

        return None

    @staticmethod
    def _quick_rsi(closes, period=14) -> float:
        """RSI using Wilder's smoothed moving average (matches standard chart RSI)."""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        # Seed with SMA of the first `period` bars, then apply Wilder's EMA
        avg_g = float(np.mean(gains[:period]))
        avg_l = float(np.mean(losses[:period]))
        for g, l in zip(gains[period:], losses[period:]):
            avg_g = (avg_g * (period - 1) + float(g)) / period
            avg_l = (avg_l * (period - 1) + float(l)) / period
        if avg_l == 0:
            return 100.0
        return float(100 - 100 / (1 + avg_g / avg_l))

    @staticmethod
    def _quick_atr(highs, lows, closes, period=14) -> float:
        if len(closes) < 2:
            return 0.0
        trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
               for i in range(1, len(closes))]
        return float(np.mean(trs[-period:])) if trs else 0.0

    def get_stats(self) -> Dict:
        """Stats for status display"""
        t1 = sum(1 for s in self._symbols.values() if s.tier == Tier.TIER1)
        t2 = sum(1 for s in self._symbols.values() if s.tier == Tier.TIER2)
        t3 = sum(1 for s in self._symbols.values() if s.tier == Tier.TIER3)
        return {'tier1': t1, 'tier2': t2, 'tier3': t3, 'total': len(self._symbols)}


# ── Singleton ──────────────────────────────────────────────
scanner = Scanner()
