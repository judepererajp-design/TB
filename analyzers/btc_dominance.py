"""
TitanBot Pro — BTC Dominance Regime Layer
==========================================
R8 Feature #9: Tracks BTC.D (dominance) to modify altcoin signal behavior.

When BTC.D is rising sharply (+2% in 24h), altcoins weaken:
  - Reduce altcoin LONG confidence by -15
  - Boost BTC-correlated SHORT confidence by +10

When BTC.D is falling (-2% in 24h), altcoin rotation:
  - Boost altcoin LONG confidence by +10
  - Reduce BTC SHORT confidence by -5

Data source: CoinGecko free API (BTC dominance chart).
Fallback: CoinPaprika, or manual BTC.D from exchange data.

This captures the altcoin rotation dynamics that pure
price/volume analysis misses.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import aiohttp

from config.constants import BTCDominance as BTCDConst

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────
# Kept as module-level aliases for backward compatibility with any
# external callers that import the bare names.  Source of truth now
# lives in config.constants.BTCDominance.

BTCD_CHECK_INTERVAL     = BTCDConst.CHECK_INTERVAL_SECS
BTCD_SHARP_RISE_PCT     = BTCDConst.SHARP_RISE_PCT
BTCD_SHARP_FALL_PCT     = BTCDConst.SHARP_FALL_PCT
BTCD_MODERATE_RISE_PCT  = BTCDConst.MODERATE_RISE_PCT
BTCD_MODERATE_FALL_PCT  = BTCDConst.MODERATE_FALL_PCT

# Confidence adjustments
ALT_LONG_PENALTY_SHARP    = BTCDConst.ALT_LONG_PENALTY_SHARP
ALT_LONG_PENALTY_MODERATE = BTCDConst.ALT_LONG_PENALTY_MODERATE
ALT_LONG_BOOST_SHARP      = BTCDConst.ALT_LONG_BOOST_SHARP
ALT_LONG_BOOST_MODERATE   = BTCDConst.ALT_LONG_BOOST_MODERATE
BTC_SHORT_BOOST           = BTCDConst.BTC_SHORT_BOOST
BTC_SHORT_PENALTY         = BTCDConst.BTC_SHORT_PENALTY


@dataclass
class BTCDominanceState:
    """Current BTC dominance tracking state"""
    current_btcd: float = 0.0          # Current BTC.D percentage
    btcd_24h_ago: float = 0.0          # BTC.D 24 hours ago
    btcd_change_24h: float = 0.0       # Change in BTC.D over 24h
    trend: str = "NEUTRAL"             # RISING, FALLING, NEUTRAL
    last_update: float = 0.0
    data_available: bool = False


class BTCDominanceTracker:
    """
    Tracks BTC dominance and provides confidence adjustments for signals.

    Integration points:
      - Engine calls get_confidence_adj(symbol, direction) before publishing
      - Adds to signal metadata for transparency
    """

    def __init__(self):
        self._state = BTCDominanceState()
        # AUDIT FIX (PR-2 / PR-6): bounded history — the plain list would
        # grow without bound on a long-running process.  Cap at the
        # HISTORY_MAX_POINTS constant; at a 10-minute poll that is >14
        # days of history which is far more than the 24h window we need.
        self._history: "deque[Tuple[float, float]]" = deque(
            maxlen=BTCDConst.HISTORY_MAX_POINTS
        )
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        # AUDIT FIX: guard session creation — two concurrent poll loops
        # (primary + fallback) could each open a ClientSession, leaking
        # one.  The lock serialises the check-and-create.
        self._session_lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Return the shared aiohttp session, creating it if needed.

        Lock-guarded so concurrent callers do not race on session
        creation and leak a duplicate.
        """
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                )
            return self._session

    async def start(self):
        """Start the background tracking loop"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._track_loop())
        logger.info("📊 BTC dominance tracker started")

    async def stop(self):
        """Stop tracking"""
        self._running = False
        if self._task:
            self._task.cancel()
            # AUDIT FIX: await the cancelled task so the session close below
            # doesn't race with an in-flight request.
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._session and not self._session.closed:
            await self._session.close()

    def get_state(self) -> BTCDominanceState:
        """Get current BTC.D state"""
        return self._state

    def get_confidence_adj(self, symbol: str, direction: str) -> int:
        """
        Get confidence adjustment based on BTC dominance trend.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT", "ETH/USDT")
            direction: "LONG" or "SHORT"

        Returns:
            Confidence adjustment (can be negative)
        """
        if not self._state.data_available:
            return 0  # No data — no adjustment

        from governance.adaptive_params import adaptive_params as _ap
        is_btc = symbol.upper().startswith("BTC")
        change = self._state.btcd_change_24h

        if is_btc:
            # BTC pairs: BTC.D rising means BTC is strengthening.
            # These adjustments are BTC-specific and not in the adaptive param store —
            # keep the original constants to avoid using mismatched altcoin param keys.
            if direction == "SHORT" and change >= BTCD_SHARP_RISE_PCT:
                return BTC_SHORT_BOOST
            elif direction == "SHORT" and change <= BTCD_SHARP_FALL_PCT:
                return BTC_SHORT_PENALTY
            return 0
        else:
            # Altcoin pairs: BTC.D rising means altcoins weakening
            if direction == "LONG":
                if change >= BTCD_SHARP_RISE_PCT:
                    return -int(_ap.get("btcd_alt_long_penalty_sharp"))
                elif change >= BTCD_MODERATE_RISE_PCT:
                    return -int(_ap.get("btcd_alt_long_penalty_moderate"))
                elif change <= BTCD_SHARP_FALL_PCT:
                    return int(_ap.get("btcd_alt_long_boost_sharp"))
                elif change <= BTCD_MODERATE_FALL_PCT:
                    return int(_ap.get("btcd_alt_long_boost_moderate"))
            elif direction == "SHORT":
                if change >= BTCD_SHARP_RISE_PCT:
                    return +5  # BTC.D rising = altcoins weak = shorts stronger
                elif change <= BTCD_SHARP_FALL_PCT:
                    return -5  # BTC.D falling = altcoins strong = shorts weaker
            return 0

    def get_metadata(self) -> dict:
        """Get BTC.D metadata for signal cards"""
        return {
            'btc_dominance': round(self._state.current_btcd, 2),
            'btcd_change_24h': round(self._state.btcd_change_24h, 2),
            'btcd_trend': self._state.trend,
        }

    # ── Background Loop ──────────────────────────────────────

    async def _track_loop(self):
        """Main tracking loop"""
        while self._running:
            try:
                await self._update_btcd()
            except Exception as e:
                logger.error(f"BTC dominance update error: {e}")
            await asyncio.sleep(BTCD_CHECK_INTERVAL)

    def _compute_24h_change(
        self, now: float, current_btcd: float
    ) -> Tuple[float, float, bool]:
        """Return (btcd_24h_ago, change, have_24h) from ``self._history``.

        Shared by the CoinGecko and CoinPaprika paths so the two update
        routines can never disagree on the semantics of "24h ago".

        The 24h-ago window is tightened to ``[now - (24h + tol),
        now - (24h - tol)]`` (default ±5 min) instead of "closest sample
        that predates now-24h", because under the previous rule a
        single-sample history after a long outage would return a 30h-old
        value and silently label it "24h ago".
        """
        cutoff = now - 2 * BTCDConst.WINDOW_TARGET_SECS
        # Drop anything older than 48h — the rolling cap.  deque has a
        # maxlen but we prune by age here to keep the history tight for
        # the 24h walk below.
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        target_earliest = now - (BTCDConst.WINDOW_TARGET_SECS
                                  + BTCDConst.WINDOW_TOLERANCE_SECS)
        target_latest = now - (BTCDConst.WINDOW_TARGET_SECS
                                - BTCDConst.WINDOW_TOLERANCE_SECS)

        match_val: Optional[float] = None
        for t, v in self._history:
            if target_earliest <= t <= target_latest:
                match_val = v  # prefer the latest in-window sample
        if match_val is None:
            return current_btcd, 0.0, False
        return match_val, current_btcd - match_val, True

    def _apply_btcd_update(self, now: float, btcd: float, source_label: str) -> None:
        """Persist a new BTC.D sample and derive state/trend fields."""
        self._history.append((now, btcd))
        btcd_24h_ago, change, have_24h = self._compute_24h_change(now, btcd)

        self._state.current_btcd    = btcd
        self._state.btcd_24h_ago    = btcd_24h_ago
        self._state.btcd_change_24h = change
        self._state.last_update     = now
        self._state.data_available  = have_24h

        if not have_24h:
            self._state.trend = "NEUTRAL"
        elif change >= BTCD_SHARP_RISE_PCT:
            self._state.trend = "RISING_SHARP"
        elif change >= BTCD_MODERATE_RISE_PCT:
            self._state.trend = "RISING"
        elif change <= BTCD_SHARP_FALL_PCT:
            self._state.trend = "FALLING_SHARP"
        elif change <= BTCD_MODERATE_FALL_PCT:
            self._state.trend = "FALLING"
        else:
            self._state.trend = "NEUTRAL"

        logger.debug(
            f"BTC.D{source_label}: {btcd:.2f}% "
            f"(24h: {change:+.2f}%, trend: {self._state.trend}, "
            f"have_24h={have_24h})"
        )

    async def _update_btcd(self):
        """Fetch BTC dominance from CoinGecko free API"""
        try:
            session = await self._ensure_session()

            # CoinGecko free API — no key required
            url = "https://api.coingecko.com/api/v3/global"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    market_data = data.get("data", {})
                    btcd = market_data.get("market_cap_percentage", {}).get("btc", 0)

                    if btcd > 0:
                        self._apply_btcd_update(time.time(), float(btcd), "")
                else:
                    logger.debug(f"CoinGecko BTC.D request failed: {resp.status}")

        except Exception as e:
            logger.debug(f"BTC dominance fetch error: {e}")
            # Try fallback
            await self._update_btcd_fallback()

    async def _update_btcd_fallback(self):
        """Fallback: CoinPaprika API (free, no key)"""
        try:
            session = await self._ensure_session()

            url = "https://api.coinpaprika.com/v1/global"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    btcd = data.get("bitcoin_dominance_percentage", 0)
                    if btcd > 0:
                        self._apply_btcd_update(
                            time.time(), float(btcd), " (fallback)"
                        )
        except Exception as e:
            logger.debug(f"CoinPaprika BTC.D fallback error: {e}")


# ── Singleton ─────────────────────────────────────────────────
btc_dominance_tracker = BTCDominanceTracker()
