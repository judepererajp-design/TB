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
PERMA_EXCLUDE_OHLCV_CYCLES = 2

# ── Shared numeric helpers ────────────────────────────────
# Incremental Wilder RSI that produces the full series in a single pass —
# replaces the O(N²) recomputation pattern `[rsi(closes[:i]) for i in ...]`.

def _wilder_rsi_series(closes, period: int = 14):
    """
    Return a list of RSI values, one per close, using Wilder's smoothing.
    Leading `period` values are NaN because RSI is undefined until the first
    averaging window is filled. Complexity: O(N).
    """
    n = len(closes)
    out = [float('nan')] * n
    if n < period + 1:
        return out
    # Seed: first `period` gains/losses averaged simply (standard Wilder seed).
    gains_sum = 0.0
    losses_sum = 0.0
    for i in range(1, period + 1):
        d = float(closes[i]) - float(closes[i - 1])
        if d > 0:
            gains_sum += d
        else:
            losses_sum += -d
    avg_g = gains_sum / period
    avg_l = losses_sum / period
    if avg_l == 0:
        out[period] = 100.0 if avg_g > 0 else 50.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    # Wilder smoothing for the remainder.
    for i in range(period + 1, n):
        d = float(closes[i]) - float(closes[i - 1])
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
        if avg_l == 0:
            out[i] = 100.0 if avg_g > 0 else 50.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


def _wilder_rsi_last(closes, period: int = 14) -> float:
    """O(N) last-value Wilder RSI. Returns 50.0 when insufficient data."""
    series = _wilder_rsi_series(closes, period)
    if not series:
        return 50.0
    last = series[-1]
    return 50.0 if last != last else float(last)  # NaN → 50


def _true_range(highs, lows, closes, start: int, end: int):
    """Iterator over true ranges in [start, end)."""
    for i in range(start, end):
        yield max(
            float(highs[i]) - float(lows[i]),
            abs(float(highs[i]) - float(closes[i - 1])),
            abs(float(lows[i]) - float(closes[i - 1])),
        )


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
    activity_score_ts: float = 0.0  # Last decay timestamp for activity_score
    promoted_at: float = 0.0        # Timestamp of last promotion
    low_volume_streak: int = 0
    low_volume_since: float = 0.0   # Timestamp when current low-vol streak began
    tier3_underfloor_since: float = 0.0  # Timestamp when T3 went below floor
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
        self._ohlcv_fail_cycles: Dict[str, int] = {}      # symbol → cooldown cycles without recovery
        self._ohlcv_cooldown_until: Dict[str, float] = {} # symbol → cooldown expiry ts
        self._perma_excluded_symbols: Set[str] = set()    # symbol → permanently excluded from rebuilds
        # Temporary exclusion cooldown (edge-case spam control)
        self._temp_excluded_until: Dict[str, float] = {}  # symbol → temporary exclusion expiry ts

        # Stalker watchlist
        self._stalker_scores: Dict[str, float] = {}

        # Universe refresh
        self._last_universe_refresh: float = 0
        self._universe_ttl = self._sys_cfg.get('universe_refresh', 3600)

        self._lock = asyncio.Lock()

        # Per-path dedup TTL age guard — any key whose ts is older than this
        # is evicted on read so the dicts can't grow unboundedly.
        self._dedup_max_age = 4 * 3600   # 4 h covers any regime-dependent TTL

        # Global alert-rate circuit breaker (token-bucket).
        # Protects Telegram from flooding in violent markets.
        self._alert_tokens: Dict[str, list] = {}   # path → list of emit ts

        # Observability counters (emitted via get_stats()).
        self._counters: Dict[str, int] = {
            'alerts_watchlist': 0,
            'alerts_watchlist_suppressed_rate': 0,
            'alerts_watchlist_suppressed_dedup': 0,
            'whales_fired': 0,
            'icebergs_fired': 0,
            'promotions': 0,
            'demotions': 0,
            'stalker_panic_skip': 0,
        }

        # Persistence restore flag — populated on first await of
        # _restore_persisted_state(). Tolerant of DB unavailability.
        self._persistence_loaded: bool = False
        self._persistence_dirty: bool = False
        self._last_persist_at: float = 0.0
        self._persist_debounce_secs: float = 60.0

    # ── Persistence (uses db.learning_state key/value store) ─────
    #
    # `volume_ma`, `promoted_at`, the low-volume streak timer and the
    # whale / stalker cooldown dicts all used to live only in memory.
    # Every deploy or crash cold-started the scanner, so:
    #   • `volume_ma` = current 24 h vol → auto-promotion effectively off
    #     for 1–3 days (the EMA τ) right when operators need it.
    #   • `_whale_snapshot` = empty → 2-snapshot persistence gate can't fire
    #     for ~1 cycle, re-firing already-alerted walls.
    #   • `_watched` / `_watchlist_cooldown` / `_signal_dedup` = empty →
    #     identical watchlist alerts re-emit after every restart.
    # We persist these as a single JSON blob under the learning_state kv
    # store to avoid schema migrations.

    _PERSIST_KEY = "scanner_runtime_state_v1"

    async def _restore_persisted_state(self):
        """Best-effort load of runtime state from DB. Silent on failure."""
        if self._persistence_loaded:
            return
        self._persistence_loaded = True
        try:
            state = await db.load_learning_state(self._PERSIST_KEY)
        except Exception:
            return
        if not isinstance(state, dict):
            return
        try:
            now = time.time()
            # Only restore entries < 24 h old — older ones are stale.
            max_age = 24 * 3600
            sym_state = state.get('symbols', {})
            if isinstance(sym_state, dict):
                for sym, blob in sym_state.items():
                    if sym in self._symbols and isinstance(blob, dict):
                        st = self._symbols[sym]
                        st.volume_ma = float(blob.get('volume_ma', st.volume_ma) or 0.0)
                        st.promoted_at = float(blob.get('promoted_at', st.promoted_at) or 0.0)
                        st.low_volume_since = float(blob.get('low_volume_since', 0.0) or 0.0)
                        st.activity_score = float(blob.get('activity_score', 0.0) or 0.0)
                        st.activity_score_ts = float(blob.get('activity_score_ts', 0.0) or 0.0)
            for src_key, dst in (
                ('whale_cooldown', self._whale_cooldown),
                ('watchlist_cooldown', self._watchlist_cooldown),
                ('temp_excluded_until', self._temp_excluded_until),
            ):
                blob = state.get(src_key)
                if isinstance(blob, dict):
                    for k, v in blob.items():
                        try:
                            ts = float(v)
                        except (TypeError, ValueError):
                            continue
                        if now - ts < max_age:
                            dst[k] = ts
            # signal_dedup: tuple keys encoded as "symbol|setup"
            sd = state.get('signal_dedup')
            if isinstance(sd, dict):
                for encoded, v in sd.items():
                    if '|' not in str(encoded):
                        continue
                    sym, setup = str(encoded).split('|', 1)
                    try:
                        ts = float(v)
                    except (TypeError, ValueError):
                        continue
                    if now - ts < max_age:
                        self._signal_dedup[(sym, setup)] = ts
            wsnap = state.get('whale_snapshot')
            if isinstance(wsnap, dict):
                for sym, sides in wsnap.items():
                    if isinstance(sides, dict):
                        self._whale_snapshot[sym] = {k: float(v or 0.0) for k, v in sides.items()}
            logger.info("Scanner: restored persisted runtime state")
        except Exception as e:
            logger.debug(f"Scanner: persisted state partially restored: {e}")

    async def _persist_state(self):
        """Best-effort snapshot of runtime state to DB."""
        try:
            symbols_blob = {
                sym: {
                    'volume_ma': st.volume_ma,
                    'promoted_at': st.promoted_at,
                    'low_volume_since': st.low_volume_since,
                    'activity_score': st.activity_score,
                    'activity_score_ts': st.activity_score_ts,
                }
                for sym, st in self._symbols.items()
            }
            payload = {
                'symbols': symbols_blob,
                'whale_cooldown': dict(self._whale_cooldown),
                'watchlist_cooldown': dict(self._watchlist_cooldown),
                'temp_excluded_until': dict(self._temp_excluded_until),
                'signal_dedup': {
                    f"{k[0]}|{k[1]}": v for k, v in self._signal_dedup.items()
                    if isinstance(k, tuple) and len(k) == 2
                },
                'whale_snapshot': self._whale_snapshot,
            }
            await db.save_learning_state(self._PERSIST_KEY, payload)
            self._persistence_dirty = False
            self._last_persist_at = time.time()
        except Exception as e:
            logger.debug(f"Scanner: persist failed (non-fatal): {e}")

    def _mark_persistence_dirty(self):
        """Mark runtime state as needing a persistence flush."""
        self._persistence_dirty = True

    async def _persist_state_if_due(self, force: bool = False):
        """Flush dirty runtime state, optionally bypassing the debounce window."""
        if not self._persistence_dirty:
            return
        if not force and self._last_persist_at > 0:
            if time.time() - self._last_persist_at < self._persist_debounce_secs:
                return
        await self._persist_state()

    # ── Shared cross-path dedup ──────────────────────────────
    def try_mark_signal_dedup(self, key: Tuple[str, str], ttl_secs: float) -> bool:
        """
        Atomically check-and-mark the cross-path signal dedup.
        Returns True if the caller should proceed (no recent duplicate);
        False if a duplicate alert was emitted within ``ttl_secs``.
        """
        now = time.time()
        last = self._signal_dedup.get(key, 0.0)
        if now - last < ttl_secs:
            return False
        self._signal_dedup[key] = now
        self._mark_persistence_dirty()
        # Opportunistic eviction of stale keys on the write path.
        if len(self._signal_dedup) > 256:
            cutoff = now - self._dedup_max_age
            stale = [k for k, ts in self._signal_dedup.items() if ts < cutoff]
            for k in stale:
                self._signal_dedup.pop(k, None)
        return True

    # ── Alert-rate circuit breaker ───────────────────────────
    def _rate_allow(self, path: str, max_alerts: int, window_secs: int) -> bool:
        """Token-bucket: ≤ ``max_alerts`` per ``window_secs`` per path."""
        now = time.time()
        bucket = self._alert_tokens.setdefault(path, [])
        cutoff = now - window_secs
        # Drop stale timestamps in-place
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) >= max_alerts:
            return False
        bucket.append(now)
        return True

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

            t1_enabled = getattr(tier1_cfg, 'enabled', True)
            t2_enabled = getattr(tier2_cfg, 'enabled', True)
            t3_enabled = getattr(tier3_cfg, 'enabled', True)

            t1_max = getattr(tier1_cfg, 'max_symbols', 80) if t1_enabled else 0
            t2_max = getattr(tier2_cfg, 'max_symbols', 80) if t2_enabled else 0
            t3_max = getattr(tier3_cfg, 'max_symbols', 40) if t3_enabled else 0

            t1_min_vol = getattr(tier1_cfg, 'min_volume_24h', 5_000_000)
            t2_min_vol = getattr(tier2_cfg, 'min_volume_24h', 1_000_000)
            t3_min_vol = getattr(tier3_cfg, 'min_volume_24h', 200_000)
            active_min_vols = []
            if t1_enabled:
                active_min_vols.append(t1_min_vol)
            if t2_enabled:
                active_min_vols.append(t2_min_vol)
            if t3_enabled:
                active_min_vols.append(t3_min_vol)
            if not active_min_vols:
                self._symbols = {}
                self._last_universe_refresh = now
                logger.info("Universe build skipped: all scanning tiers are disabled")
                return
            min_qual_vol = min(active_min_vols)

            excluded = set(cfg.exchange.get('excluded_symbols', [])
                           if isinstance(cfg.exchange.get('excluded_symbols', []), list)
                           else [])
            excluded.update(self._perma_excluded_symbols)
            # Also exclude symbols flagged as delisted/settling by the exchange
            # via the APIClient's delisted-set (binance error -4108 etc).
            try:
                from data.api_client import api as _api
                _delisted = getattr(_api, '_delisted_symbols', None)
                if _delisted:
                    excluded.update(_delisted)
            except Exception:
                pass

            # Sort by volume descending
            qualified = []
            for symbol, ticker in all_tickers.items():
                base_symbol = symbol.split(':')[0]
                if not base_symbol.endswith('/USDT'):
                    continue
                if base_symbol in excluded:
                    continue
                vol = float(ticker.get('quoteVolume', 0) or 0)
                if vol >= min_qual_vol:
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

            # Restore persisted runtime state (volume_ma, cooldowns, dedup)
            # after the first universe build. Best-effort; no-op on failure.
            if not self._persistence_loaded:
                try:
                    await self._restore_persisted_state()
                except Exception:
                    self._persistence_loaded = True  # don't retry storm

            # Prune cooldown entries that belong to symbols no longer in the
            # universe so the dicts don't grow unboundedly on long-running instances.
            active_symbols = set(new_symbols)
            for _cd in (
                self._whale_cooldown,
                self._watchlist_cooldown,
                self._ohlcv_fail_counts,
                self._ohlcv_fail_cycles,
                self._ohlcv_cooldown_until,
                self._temp_excluded_until,
                self._whale_snapshot,
                self._signal_dedup,
            ):
                if _cd is self._signal_dedup:
                    _stale_keys = [
                        k for k in _cd
                        if not isinstance(k, tuple) or len(k) < 1 or k[0] not in active_symbols
                    ]
                else:
                    _stale_keys = [k for k in _cd if k not in active_symbols]
                for _k in _stale_keys:
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
        Priority: Tier 1 first, then 2, then 3. Ties are broken by
        how-overdue and recent signal activity.
        """
        now = time.time()
        due = []

        # FIX: iterate over a snapshot of items(). build_universe() runs concurrently
        # in an asyncio task and can mutate _symbols while we iterate, raising
        # RuntimeError: dictionary changed size during iteration.
        for symbol, state in list(self._symbols.items()):
            if self.is_temporarily_excluded(symbol):
                continue
            interval = self._tier_intervals.get(state.tier, 300)
            if now - state.last_scan >= interval:
                due.append(symbol)

        # Sort by tier priority (T1 first), then by how overdue, then by
        # activity_score (decayed). Snapshot `now` once instead of per-compare.
        snapshot = {
            sym: self._symbols.get(sym) for sym in due
        }

        def priority(sym):
            state = snapshot.get(sym)
            if not state:
                return (99, 0.0, 0.0)
            interval = self._tier_intervals.get(state.tier, 300)
            overdue = (now - state.last_scan) / interval if interval > 0 else 0.0
            # activity_score is decayed so a dead coin from last week doesn't
            # keep stealing slots. Half-life ~6 h.
            age = max(0.0, now - (state.activity_score_ts or 0.0))
            decayed = state.activity_score * math.exp(-age / (6 * 3600.0))
            # Lower tier → higher priority. Negative overdue/score so Python's
            # stable ascending sort gives the desired order.
            return (state.tier.value, -overdue, -decayed)

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
        self._perma_excluded_symbols.add(symbol)
        # FIX: pop() is atomic at the GIL level — safe without a lock
        self._symbols.pop(symbol, None)
        self._ohlcv_fail_counts.pop(symbol, None)
        self._ohlcv_fail_cycles.pop(symbol, None)
        self._ohlcv_cooldown_until.pop(symbol, None)
        self._temp_excluded_until.pop(symbol, None)
        logger.info(f"Scanner: excluded {symbol} from universe (delisted or bad symbol)")

    def temporarily_exclude_symbol(self, symbol: str, duration_secs: float = 3600, reason: str = "edge case"):
        """Temporarily pause scans for a symbol without removing it from the universe.

        If called again before expiry, the exclusion window is extended to the
        later timestamp (it never shortens an active cooldown).
        """
        if symbol not in self._symbols:
            return
        now = time.time()
        until = now + max(1.0, float(duration_secs))
        prev_until = self._temp_excluded_until.get(symbol, 0.0)
        self._temp_excluded_until[symbol] = max(prev_until, until)
        self._mark_persistence_dirty()
        remaining = int(max(0.0, self._temp_excluded_until[symbol] - now))
        logger.info(
            f"Scanner: temporarily excluded {symbol} for {remaining}s ({reason})"
        )

    def is_temporarily_excluded(self, symbol: str) -> bool:
        """Return True if symbol is under temporary exclusion cooldown."""
        until = self._temp_excluded_until.get(symbol)
        if until is None:
            return False
        if time.time() >= until:
            self._temp_excluded_until.pop(symbol, None)
            self._mark_persistence_dirty()
            logger.info(f"Scanner: temporary exclusion expired for {symbol}")
            return False
        return True

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
            self._ohlcv_fail_counts.pop(symbol, None)  # reset the per-cycle counter before cooldown/exclusion
            cycles = self._ohlcv_fail_cycles.get(symbol, 0) + 1
            self._ohlcv_fail_cycles[symbol] = cycles
            if cycles >= PERMA_EXCLUDE_OHLCV_CYCLES:
                logger.warning(
                    f"📊 OHLCV perma-exclusion: {symbol} hit {cycles} cooldown cycles "
                    f"without recovery — removing from universe"
                )
                self.exclude_symbol(symbol)
                return True
            self._ohlcv_cooldown_until[symbol] = time.time() + OHLCVCooldown.COOLDOWN_SECS
            logger.warning(
                f"📊 OHLCV cooldown: {symbol} failed {count}× — "
                f"excluding for {OHLCVCooldown.COOLDOWN_SECS}s (cycle {cycles})"
            )
            return True
        return False

    def record_ohlcv_success(self, symbol: str):
        """Clear the failure counter on a successful OHLCV fetch."""
        self._ohlcv_fail_counts.pop(symbol, None)
        self._ohlcv_fail_cycles.pop(symbol, None)

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
        """Mark that a signal was generated for this symbol.

        Adds to a time-decayed activity score used by get_due_symbols() to
        prioritise historically productive symbols in dense scan cycles.
        """
        state = self._symbols.get(symbol)
        if state:
            now = time.time()
            # Decay the existing score first so `+= 10` is not unbounded.
            age = max(0.0, now - (state.activity_score_ts or now))
            decayed = state.activity_score * math.exp(-age / (6 * 3600.0))
            # Cap activity_score to keep priority stable; 30 = 3 signals worth.
            state.activity_score = min(30.0, decayed + 10.0)
            state.activity_score_ts = now
            state.last_signal = now

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

        tier_cfgs = {
            Tier.TIER1: self._scan_cfg.tier1,
            Tier.TIER2: self._scan_cfg.tier2,
            Tier.TIER3: self._scan_cfg.tier3,
        }
        tier_enabled = {tier: getattr(cfg_t, 'enabled', True) for tier, cfg_t in tier_cfgs.items()}
        tier_min_vol = {
            Tier.TIER1: float(getattr(self._scan_cfg.tier1, 'min_volume_24h', 5_000_000)),
            Tier.TIER2: float(getattr(self._scan_cfg.tier2, 'min_volume_24h', 1_000_000)),
            Tier.TIER3: float(getattr(self._scan_cfg.tier3, 'min_volume_24h', 200_000)),
        }

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

        # Symmetric low-volume demotion: if a symbol remains below its current
        # tier's minimum volume for at least N minutes *continuously*, demote
        # one enabled tier. Time-based rather than call-based prevents Tier-1
        # from demoting in a few minutes of quiet (it scans every 2 min).
        demotion_minutes = float(getattr(prom_cfg, 'demotion_minutes', 60) or 60)
        current_tier_min = tier_min_vol.get(state.tier, 0.0)
        under_floor_now = current_volume < current_tier_min
        # Tier-3 floor: below T3 min for a sustained period → temporary exile.
        # Without this a dead symbol stays in T3 until the next hourly universe
        # rebuild (up to 4 wasted 15-min scans).
        if state.tier == Tier.TIER3:
            t3_floor_minutes = float(getattr(prom_cfg, 'tier3_exile_minutes', 180) or 180)
            if under_floor_now:
                if state.tier3_underfloor_since == 0.0:
                    state.tier3_underfloor_since = time.time()
                elif time.time() - state.tier3_underfloor_since >= t3_floor_minutes * 60:
                    # Exile for 6 h; next universe rebuild reassesses eligibility.
                    self.temporarily_exclude_symbol(
                        symbol, duration_secs=6 * 3600,
                        reason=f"T3 below ${current_tier_min:,.0f} for ≥{int(t3_floor_minutes)} min",
                    )
                    state.tier3_underfloor_since = 0.0
                    state.low_volume_since = 0.0
                    await self._persist_state_if_due(force=True)
                    return None
            else:
                state.tier3_underfloor_since = 0.0
            # Reset legacy streak counter — T3 has no lower tier.
            state.low_volume_streak = 0
            state.low_volume_since = 0.0
        elif under_floor_now:
            # Start or continue the streak timer.
            if state.low_volume_since == 0.0:
                state.low_volume_since = time.time()
            state.low_volume_streak += 1  # kept for stats / back-compat
            elapsed_min = (time.time() - state.low_volume_since) / 60.0
            if elapsed_min >= demotion_minutes:
                from_tier = state.tier.value
                next_tier_val = state.tier.value + 1
                while next_tier_val <= Tier.TIER3.value and not tier_enabled.get(Tier(next_tier_val), True):
                    next_tier_val += 1
                if next_tier_val <= Tier.TIER3.value:
                    new_tier = Tier(next_tier_val)
                    state.tier = new_tier
                    state.volume_24h = current_volume
                    state.low_volume_streak = 0
                    state.low_volume_since = 0.0
                    self._counters['demotions'] = self._counters.get('demotions', 0) + 1
                    await db.upsert_symbol_tier(symbol, new_tier.value, current_volume)
                    self._mark_persistence_dirty()
                    logger.info(
                        f"📉 {symbol} demoted: Tier {from_tier} → Tier {new_tier.value} "
                        f"(volume ${current_volume:,.0f} < ${current_tier_min:,.0f} "
                        f"for {elapsed_min:.0f} min)"
                    )
                    await self._persist_state_if_due(force=True)
                    return (from_tier, new_tier.value)
        else:
            state.low_volume_streak = 0
            state.low_volume_since = 0.0

        # Cooldown check
        cooldown_hours = getattr(prom_cfg, 'cooldown_hours', 24)
        if time.time() - state.promoted_at < cooldown_hours * 3600:
            return None

        if (vol_spike_mult >= promotion_threshold and
                current_volume >= min_vol and
                state.tier != Tier.TIER1):

            from_tier = state.tier.value
            next_tier_val = state.tier.value - 1
            while next_tier_val >= Tier.TIER1.value and not tier_enabled.get(Tier(next_tier_val), True):
                next_tier_val -= 1
            if next_tier_val < Tier.TIER1.value:
                return None
            new_tier = Tier(next_tier_val)  # Promote one enabled tier
            to_tier = new_tier.value

            state.tier = new_tier
            state.promoted_at = time.time()
            state.volume_24h = current_volume
            state.low_volume_streak = 0
            state.low_volume_since = 0.0
            # Increase scan priority immediately
            state.last_scan = 0  # Force immediate rescan
            self._counters['promotions'] = self._counters.get('promotions', 0) + 1

            await db.record_promotion(symbol, from_tier, to_tier)
            self._mark_persistence_dirty()
            logger.info(f"🚀 {symbol} promoted: Tier {from_tier} → Tier {to_tier} ({vol_spike_mult:.1f}x volume)")

            # Persist so a crash right after promotion doesn't forget the cooldown.
            try:
                await self._persist_state_if_due(force=True)
            except Exception:
                pass

            return (from_tier, to_tier)

        return None

    # ── Whale Detection ────────────────────────────────────────

    async def check_whale_activity(
        self, symbol: str, order_book: Dict
    ) -> Optional[Dict]:
        """
        Detect whale orders in the order book.
        Evaluates *both* single-level walls and cumulative (iceberg)
        accumulations per side, fires the stronger.
        Returns whale info dict if detected.
        """
        whale_cfg = self._scan_cfg.whale_detection
        if not getattr(whale_cfg, 'enabled', True):
            return None

        # Absolute USD floor; actual threshold scales with 24 h volume so the
        # bar is equally meaningful on BTC (huge book) and mid-caps.
        min_order_usd_floor = getattr(whale_cfg, 'min_order_usd', 75_000)
        vol_scale_frac      = float(getattr(whale_cfg, 'vol_threshold_frac', 0.005) or 0.005)
        cooldown_min        = getattr(whale_cfg, 'cooldown_minutes', 45)

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

        # Dynamic threshold: max(floor, vol_scale_frac × 24h quote volume).
        # 0.5 % of daily volume = a real "needle-mover" across cap tiers.
        min_order_usd = max(
            float(min_order_usd_floor),
            float(state.volume_24h or 0.0) * vol_scale_frac,
        )

        # Use last known price from order book or symbol state
        # order_book bids/asks have prices — use mid of best bid/ask
        _bids_raw = order_book.get('bids', [])
        _asks_raw = order_book.get('asks', [])
        best_bid = float(_bids_raw[0][0]) if _bids_raw and _bids_raw[0] else 0.0
        best_ask = float(_asks_raw[0][0]) if _asks_raw and _asks_raw[0] else 0.0
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
        # Order book invariant: asks ascending, bids descending, so a negative
        # (ask-bid) indicates a crossed/stale book — skip.
        if best_ask > 0 and best_bid > 0:
            raw_spread = best_ask - best_bid
            if raw_spread < 0:
                logger.debug(
                    f"🐋 Skipping {symbol}: crossed book "
                    f"(ask={best_ask} < bid={best_bid}) — stale snapshot"
                )
                return None
            spread_pct = raw_spread / current_price
        else:
            spread_pct = 0.0

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

        # Adaptive price band for iceberg accumulation — scale with spread so
        # tight-book BTC uses a narrow band (avoiding noise) and wide-spread
        # mid-caps use a proportionate window.
        price_band = current_price * max(0.005, min(0.02, spread_pct * 20.0))
        if price_band <= 0:
            price_band = current_price * 0.01

        def _accept_persistence(storage_key: str, side: str, current_usd: float) -> bool:
            """
            Two-snapshot persistence gate.
            A wall must appear in >= 2 snapshots *AND* the previous snapshot
            must itself have been at or above a meaningful fraction of the
            threshold (prevents spoof walls that vanish before the next poll).
            Growing walls (prev < cur) are explicitly ACCEPTED — the bug was
            anchoring the ratio to the larger snapshot, which rejected healthy
            accumulation patterns.

            The 0.75 ratio on each branch means:
              • Growing:   prev must already be ≥ 75 % of the current
                           threshold (so we don't count a one-off $10k print
                           as a "previous sighting" of a $100k wall).
              • Shrinking: the wall must retain ≥ 75 % of its prior size
                           (a classic spoof pulls > 50 % between polls; this
                           catches that while tolerating partial fills).
            """
            snap = self._whale_snapshot.setdefault(symbol, {})
            prev = float(snap.get(storage_key, 0.0) or 0.0)
            snap[storage_key] = current_usd
            self._mark_persistence_dirty()
            # Growing wall: accept if prev also meaningfully crossed threshold.
            if prev >= min_order_usd * 0.75:
                return True
            # Shrinking wall: must retain ≥ 75 % of its previous magnitude.
            if prev > 0 and current_usd >= prev * 0.75:
                return True
            return False

        # Scan both sides; track strongest candidate for this book snapshot.
        best_candidate: Optional[Dict] = None
        best_score = 0.0
        best_kind = ""   # "wall" | "iceberg"
        seen_snapshot_keys = set()

        for side, orders in [('buy', bids), ('sell', asks)]:
            cumulative_usd = 0.0
            strongest_wall_usd = 0.0
            strongest_wall_px  = current_price
            strongest_wall_qty = 0.0

            for price, qty in orders[:50]:
                price     = float(price)
                qty       = float(qty)
                order_usd = price * qty

                # Track strongest single-level wall
                if order_usd >= min_order_usd and order_usd > strongest_wall_usd:
                    strongest_wall_usd = order_usd
                    strongest_wall_px  = price
                    strongest_wall_qty = qty

                # Accumulate within the adaptive band for iceberg detection
                if abs(price - current_price) <= price_band:
                    cumulative_usd += order_usd

            # Evaluate single-level wall for this side
            if strongest_wall_usd >= min_order_usd:
                seen_snapshot_keys.add(side)
                if _accept_persistence(side, side, strongest_wall_usd):
                    fp = _fill_prob(strongest_wall_px, side, strongest_wall_usd, min_order_usd)
                    score = strongest_wall_usd * (0.5 + 0.5 * fp)
                    if score > best_score:
                        best_score = score
                        best_kind  = 'wall'
                        best_candidate = {
                            'symbol':    symbol,
                            'side':      side,
                            'order_usd': strongest_wall_usd,
                            'price':     strongest_wall_px,
                            'qty':       strongest_wall_qty,
                            'fill_prob': fp,
                            'spread_pct': spread_pct,
                            'kind':       'wall',
                            'threshold':  min_order_usd,
                        }

            # Evaluate iceberg (cumulative within band) for this side
            if cumulative_usd >= min_order_usd:
                iceberg_key = f"iceberg_{side}"
                seen_snapshot_keys.add(iceberg_key)
                if _accept_persistence(iceberg_key, side, cumulative_usd):
                    fp = _fill_prob(current_price, side, cumulative_usd, min_order_usd)
                    score = cumulative_usd * (0.5 + 0.5 * fp)
                    if score > best_score:
                        best_score = score
                        best_kind  = 'iceberg'
                        best_candidate = {
                            'symbol':    symbol,
                            'side':      side,
                            'order_usd': cumulative_usd,
                            'price':     current_price,
                            'qty':       cumulative_usd / current_price if current_price > 0 else 0.0,
                            'fill_prob': fp,
                            'spread_pct': spread_pct,
                            'kind':       'iceberg',
                            'threshold':  min_order_usd,
                        }

        # Reset opposite-side snapshot entries that saw no wall this cycle,
        # so stale prior sightings from the other side can't persist indefinitely.
        snap = self._whale_snapshot.setdefault(symbol, {})
        for key in ('buy', 'sell', 'iceberg_buy', 'iceberg_sell'):
            if key not in seen_snapshot_keys and float(snap.get(key, 0.0) or 0.0) != 0.0:
                snap[key] = 0.0
                self._mark_persistence_dirty()
            else:
                snap.setdefault(key, 0.0)

        if best_candidate is None:
            await self._persist_state_if_due()
            return None

        # Apply cooldown on firing side only (other side can still fire later).
        self._whale_cooldown[symbol] = time.time()
        self._mark_persistence_dirty()
        await self._persist_state_if_due(force=True)
        if best_kind == 'iceberg':
            self._counters['icebergs_fired'] = self._counters.get('icebergs_fired', 0) + 1
            logger.info(
                f"🐋 Iceberg whale: {symbol} {best_candidate['side']} "
                f"${best_candidate['order_usd']:,.0f} cumulative within adaptive band"
                f"  fill_prob={best_candidate['fill_prob']:.2f}"
                f"  threshold=${min_order_usd:,.0f}"
            )
        else:
            self._counters['whales_fired'] = self._counters.get('whales_fired', 0) + 1
            logger.info(
                f"🐋 Whale order: {symbol} {best_candidate['side']} "
                f"${best_candidate['order_usd']:,.0f} @ {best_candidate['price']}"
                f"  fill_prob={best_candidate['fill_prob']:.2f}"
                f"  threshold=${min_order_usd:,.0f}"
            )
        return best_candidate

    # ── Stalker Engine ────────────────────────────────────────

    async def stalker_scan(self, symbol: str, ohlcv: List) -> Optional[Dict]:
        """
        Pre-breakout detection for watchlist.
        Looks for coiling, compression, and pre-breakout conditions.
        Returns dict with score and reasons if interesting.

        Detectors operate on *closed* bars only (the last OHLCV element is
        treated as the currently-forming bar and excluded from every
        comparison). Earlier versions contaminated squeeze / key-level /
        divergence signals with live tick data and produced false alerts.
        """
        # Need 30 closed bars + 1 still-forming = 31 rows
        if not ohlcv or len(ohlcv) < 31:
            return None

        # Cooldown gate — check before any expensive computation
        _last = self._watchlist_cooldown.get(symbol, 0)
        if time.time() - _last < self._watchlist_cooldown_secs:
            return None

        # VOLATILE_PANIC: pre-breakout compression alerts in this regime are
        # overwhelmingly bull traps (squeeze → resolution down). Skip entirely;
        # reversal strategies handle the direction-aware side of the book.
        _regime = _get_regime()
        if _regime == "VOLATILE_PANIC":
            self._counters['stalker_panic_skip'] = self._counters.get('stalker_panic_skip', 0) + 1
            return None

        # Dynamic threshold: raise the bar in CHOPPY (signal flood risk),
        # lower it in trending regimes (signals are scarcer but higher quality).
        # Hysteresis: damp adjustments 25% within 10 min of a regime flip to
        # prevent violent over-correction right after a transition.
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
        # Hysteresis: within 10 min of a regime flip, damp the TTL reduction by
        # 25% so we don't instantly halve the dedup window on a noisy transition.
        if _regime in ("BULL_TREND", "BEAR_TREND"):
            _dedup_ttl = self._signal_dedup_ttl        # 30 min
        else:
            _raw_ttl_reduction = self._signal_dedup_ttl // 2
            if _recently_flipped:
                # Damp: interpolate 75% of the way from full TTL to half TTL
                _damped_ttl = self._signal_dedup_ttl - round(_raw_ttl_reduction * 0.75)
                _dedup_ttl = _damped_ttl
            else:
                _dedup_ttl = self._signal_dedup_ttl - _raw_ttl_reduction  # 15 min in chop/volatile

        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df = df.astype({'open':float,'high':float,'low':float,'close':float,'volume':float})

        # IMPORTANT: drop the still-forming bar from every detector.
        # Earlier iterations used arr[-1] = the live candle, whose close
        # moves tick-by-tick and contaminates squeeze / key-level / divergence
        # signals (stalker.py._check_ltf_alignment already does this correctly).
        highs_all   = df['high'].values
        lows_all    = df['low'].values
        closes_all  = df['close'].values
        volumes_all = df['volume'].values
        highs   = highs_all[:-1]
        lows    = lows_all[:-1]
        closes  = closes_all[:-1]
        volumes = volumes_all[:-1]

        score = 0
        reasons = []

        # ── 1. Bollinger Band squeeze (volatility compression) ─
        bb_period = 20
        if len(closes) >= bb_period:
            window = closes[-bb_period:]
            bb_mean = np.mean(window)
            bb_std  = np.std(window)
            bb_width = (bb_std * 4) / bb_mean if bb_mean > 0 else 0.0  # Full band width (upper - lower) / mean

            # Extended history: up to 200 closed bars, stride 2, for a stable
            # 20th-percentile gate (~90 samples instead of ~6). Prior 48-bar
            # window produced a noisy percentile with only a handful of points.
            if len(closes) >= bb_period + 20:
                widths_hist = []
                start = max(bb_period, len(closes) - 200)
                for end in range(start, len(closes), 2):
                    w = closes[end - bb_period:end]
                    if len(w) == bb_period:
                        m = np.mean(w)
                        widths_hist.append((np.std(w) * 4) / m if m > 0 else 0.0)
                if widths_hist and bb_width < np.percentile(widths_hist, 20):
                    score += 30
                    reasons.append("🌀 Bollinger squeeze — lowest volatility in recent history")

        # ── 2. Key level proximity ─────────────────────────────
        # Require that the 20-bar extreme has been *touched* within the last 5
        # closed bars. A close 1.5 % below a distant old-high-that-was-never-
        # retested was previously mis-flagged as "testing high".
        period_high = np.max(highs[-20:])
        period_low  = np.min(lows[-20:])
        current     = closes[-1]

        high_dist = (period_high - current) / current if current > 0 else 1.0
        low_dist  = (current - period_low) / current if current > 0 else 1.0

        recent_touch_lookback = 5
        recent_high_touched = bool(np.any(np.isclose(
            highs[-recent_touch_lookback:], period_high, rtol=0, atol=period_high * 0.001
        ))) or bool(np.max(highs[-recent_touch_lookback:]) >= period_high * 0.999)
        recent_low_touched = bool(np.any(np.isclose(
            lows[-recent_touch_lookback:], period_low, rtol=0, atol=period_low * 0.001
        ))) or bool(np.min(lows[-recent_touch_lookback:]) <= period_low * 1.001)

        near_key_level = False
        if high_dist < 0.015 and recent_high_touched:
            score += 20
            near_key_level = True
            reasons.append(f"📈 Testing 20-bar high ({fmt_price(period_high)}) — breakout alert")
        elif low_dist < 0.015 and recent_low_touched:
            score += 20
            near_key_level = True
            reasons.append(f"📉 Testing 20-bar low ({fmt_price(period_low)}) — breakdown alert")

        # ── 3. Volume declining (coiling) ─────────────────────
        avg_vol_20 = np.mean(volumes[-20:])
        avg_vol_5  = np.mean(volumes[-5:])
        range_mid = (np.max(highs[-10:]) + np.min(lows[-10:])) / 2.0
        range_pct = (np.max(highs[-10:]) - np.min(lows[-10:])) / range_mid if range_mid > 0 else 1.0
        is_tight_range = range_pct < 0.03
        # Use a softer volume-drop threshold (0.7) only because we now require
        # simultaneous tight-range + key-level proximity confirmation.
        if avg_vol_5 < avg_vol_20 * 0.7 and is_tight_range and near_key_level:
            score += 15
            reasons.append("📊 Volume cooling inside tight key-level range — coiling setup")

        # ── 4. RSI divergence ─────────────────────────────────
        # Incremental Wilder RSI (O(N)); prior impl was O(N²) and dominated
        # stalker CPU on hot paths.
        # Stricter gates to cut noise divergence:
        #   • pivots must be ≥ 5 bars apart (prevents micro-wiggle matches),
        #   • pivot RSI must sit in the extreme band (>=60 for bearish pivot
        #     highs, <=40 for bullish pivot lows) so we ignore mid-range noise,
        #   • divergence magnitude raised to 12 RSI-points (was 10).
        if len(closes) >= 40:
            def _swing_idxs(arr, is_high: bool, look: int = 2) -> List[int]:
                idxs: List[int] = []
                for i in range(look, len(arr) - look):
                    left = arr[i - look:i]
                    right = arr[i + 1:i + look + 1]
                    if is_high and arr[i] >= np.max(left) and arr[i] >= np.max(right):
                        idxs.append(i)
                    if not is_high and arr[i] <= np.min(left) and arr[i] <= np.min(right):
                        idxs.append(i)
                return idxs

            rsi_series = _wilder_rsi_series(list(closes), 14)

            window = closes[-30:]
            hi_swings = _swing_idxs(window, is_high=True)
            lo_swings = _swing_idxs(window, is_high=False)
            base = len(closes) - 30

            min_pivot_sep = 5
            if len(hi_swings) >= 2:
                a, b = hi_swings[-2], hi_swings[-1]
                if (b - a) >= min_pivot_sep:
                    pa, pb = closes[base + a], closes[base + b]
                    ra, rb = rsi_series[base + a], rsi_series[base + b]
                    if (
                        pb > pa
                        and ra == ra and rb == rb   # NaN check
                        and ra >= 60 and rb <= ra - 12
                    ):
                        score += 15
                        reasons.append("⚡ RSI bearish divergence on swing highs")
            if len(lo_swings) >= 2:
                a, b = lo_swings[-2], lo_swings[-1]
                if (b - a) >= min_pivot_sep:
                    pa, pb = closes[base + a], closes[base + b]
                    ra, rb = rsi_series[base + a], rsi_series[base + b]
                    if (
                        pb < pa
                        and ra == ra and rb == rb
                        and ra <= 40 and rb >= ra + 12
                    ):
                        score += 15
                        reasons.append("⚡ RSI bullish divergence on swing lows")

        # ── 5. ATR compression ────────────────────────────────
        if len(closes) >= 60:
            atr_now = self._quick_atr(highs[-15:], lows[-15:], closes[-15:])
            atr_base = self._quick_atr(highs[-51:], lows[-51:], closes[-51:], period=50)
            if atr_base > 0 and atr_now < atr_base * 0.6:
                score += 10
                reasons.append(f"🔇 ATR compressed {atr_now/atr_base:.0%} of 50-bar baseline")

        if score >= _min_score:
            # Cross-path dedup via shared helper — both scanner.stalker_scan
            # and stalker.StalkerEngine consult the same dict so the same
            # symbol/setup can't double-fire within the regime-aware TTL.
            setup_type = "pre_breakout"
            dedup_key  = (symbol, setup_type)
            if not self.try_mark_signal_dedup(dedup_key, _dedup_ttl):
                self._counters['alerts_watchlist_suppressed_dedup'] += 1
                return None

            # Global alert-rate circuit breaker: ≤ 20 watchlist alerts / 10 min.
            if not self._rate_allow('watchlist', max_alerts=20, window_secs=600):
                self._counters['alerts_watchlist_suppressed_rate'] += 1
                logger.info(
                    f"Watchlist rate-limit hit — suppressing {symbol} "
                    f"(score={score})"
                )
                return None

            await db.upsert_watchlist(symbol, float(score), reasons)
            self._watchlist_cooldown[symbol] = time.time()
            self._counters['alerts_watchlist'] += 1
            self._mark_persistence_dirty()
            await self._persist_state_if_due(force=True)
            return {'symbol': symbol, 'score': score, 'reasons': reasons, 'regime': _regime}

        return None

    @staticmethod
    def _quick_rsi(closes, period=14) -> float:
        """RSI using Wilder's smoothed moving average (matches standard chart RSI).

        Single-value convenience wrapper around ``_wilder_rsi_last``. Callers
        that need the full series should prefer ``_wilder_rsi_series`` to
        avoid O(N²) recomputation.
        """
        return _wilder_rsi_last(closes, period)

    @staticmethod
    def _quick_atr(highs, lows, closes, period=14) -> float:
        """Wilder-style ATR over the last `period` true ranges.

        Convention: `period` is the number of TRs to average, **not** the
        number of bars. N bars produce N-1 TRs, so callers that want an
        "N-bar ATR" should pass N+1 bars. Prior impl silently capped at
        `len(closes)-1` TRs regardless of the requested period, making a
        14-bar tail produce a 13-TR ATR. The fix lives at the call sites
        (pass period+1 bars); this function itself simply averages the last
        `period` TRs of whatever window it receives, or fewer if not enough
        bars are supplied.
        """
        if len(closes) < 2:
            return 0.0
        trs = list(_true_range(highs, lows, closes, 1, len(closes)))
        return float(np.mean(trs[-period:])) if trs else 0.0

    def get_stats(self) -> Dict:
        """Stats for status display (includes operational counters)."""
        t1 = sum(1 for s in self._symbols.values() if s.tier == Tier.TIER1)
        t2 = sum(1 for s in self._symbols.values() if s.tier == Tier.TIER2)
        t3 = sum(1 for s in self._symbols.values() if s.tier == Tier.TIER3)
        stats: Dict = {'tier1': t1, 'tier2': t2, 'tier3': t3, 'total': len(self._symbols)}
        # Merge counters so observability is available in one call.
        for k, v in self._counters.items():
            stats[f'counter_{k}'] = v
        return stats


# ── Singleton ──────────────────────────────────────────────
scanner = Scanner()
