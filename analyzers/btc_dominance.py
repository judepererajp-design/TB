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
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────

BTCD_CHECK_INTERVAL = 600          # Check every 10 minutes
BTCD_SHARP_RISE_PCT = 2.0          # +2% in 24h = BTC dominance rising
BTCD_SHARP_FALL_PCT = -2.0         # -2% in 24h = BTC dominance falling
BTCD_MODERATE_RISE_PCT = 1.0       # +1% = moderate rise
BTCD_MODERATE_FALL_PCT = -1.0      # -1% = moderate fall

# Confidence adjustments
ALT_LONG_PENALTY_SHARP = -15       # Sharp BTC.D rise → altcoin longs penalized
ALT_LONG_PENALTY_MODERATE = -8     # Moderate BTC.D rise → smaller penalty
ALT_LONG_BOOST_SHARP = +10         # Sharp BTC.D fall → altcoin longs boosted
ALT_LONG_BOOST_MODERATE = +5       # Moderate fall → smaller boost
BTC_SHORT_BOOST = +10              # BTC.D rising → BTC shorts stronger
BTC_SHORT_PENALTY = -5             # BTC.D falling → BTC shorts weaker


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
        self._history: list = []    # (timestamp, btcd_value) pairs
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

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

    async def _update_btcd(self):
        """Fetch BTC dominance from CoinGecko free API"""
        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                )

            # CoinGecko free API — no key required
            url = "https://api.coingecko.com/api/v3/global"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    market_data = data.get("data", {})
                    btcd = market_data.get("market_cap_percentage", {}).get("btc", 0)

                    if btcd > 0:
                        now = time.time()
                        self._history.append((now, btcd))

                        # Keep 48 hours of history
                        cutoff = now - 172800
                        self._history = [
                            (t, v) for t, v in self._history if t > cutoff
                        ]

                        # Calculate change vs the closest pre-target sample.
                        # AUDIT FIX: until we actually have a sample ≥24h old
                        # we don't have a real 24h change — surface that by
                        # keeping data_available=False until the window fills.
                        target_time = now - 86400
                        oldest_ts = self._history[0][0] if self._history else now
                        have_24h = oldest_ts <= target_time
                        btcd_24h_ago = btcd  # Default
                        for t, v in self._history:
                            if t <= target_time:
                                btcd_24h_ago = v

                        change = btcd - btcd_24h_ago if have_24h else 0.0

                        # Update state
                        self._state.current_btcd = btcd
                        self._state.btcd_24h_ago = btcd_24h_ago
                        self._state.btcd_change_24h = change
                        self._state.last_update = now
                        # Only advertise data as available once we have a true
                        # 24h window; otherwise get_confidence_adj would apply
                        # zero adjustment while claiming we have real data.
                        self._state.data_available = have_24h

                        # Determine trend
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
                            f"BTC.D: {btcd:.2f}% (24h: {change:+.2f}%, "
                            f"trend: {self._state.trend}, have_24h={have_24h})"
                        )
                else:
                    logger.debug(f"CoinGecko BTC.D request failed: {resp.status}")

        except Exception as e:
            logger.debug(f"BTC dominance fetch error: {e}")
            # Try fallback
            await self._update_btcd_fallback()

    async def _update_btcd_fallback(self):
        """Fallback: CoinPaprika API (free, no key)"""
        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                )

            url = "https://api.coinpaprika.com/v1/global"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    btcd = data.get("bitcoin_dominance_percentage", 0)
                    if btcd > 0:
                        now = time.time()
                        self._history.append((now, btcd))
                        # AUDIT FIX: also recompute change / trend / have_24h
                        # so a prolonged primary outage doesn't leave stale
                        # fields from yesterday's fetch driving live trading.
                        cutoff = now - 172800
                        self._history = [
                            (t, v) for t, v in self._history if t > cutoff
                        ]
                        target_time = now - 86400
                        oldest_ts = self._history[0][0] if self._history else now
                        have_24h = oldest_ts <= target_time
                        btcd_24h_ago = btcd
                        for t, v in self._history:
                            if t <= target_time:
                                btcd_24h_ago = v
                        change = btcd - btcd_24h_ago if have_24h else 0.0

                        self._state.current_btcd = btcd
                        self._state.btcd_24h_ago = btcd_24h_ago
                        self._state.btcd_change_24h = change
                        self._state.last_update = now
                        self._state.data_available = have_24h
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
                            f"BTC.D (fallback): {btcd:.2f}% "
                            f"(24h: {change:+.2f}%, have_24h={have_24h})"
                        )
        except Exception as e:
            logger.debug(f"CoinPaprika BTC.D fallback error: {e}")


# ── Singleton ─────────────────────────────────────────────────
btc_dominance_tracker = BTCDominanceTracker()
