"""
TitanBot Pro — Shared Price Cache
==================================
Single polling loop that feeds all three monitors (InvalidationMonitor,
OutcomeMonitor, ExecutionEngine) so we don't make 3× the API calls.

Usage:
    from core.price_cache import price_cache

    # Register a symbol for polling
    price_cache.subscribe("BTC/USDT")

    # Read latest price (non-blocking, returns last known or None)
    price = price_cache.get("BTC/USDT")

    # Unsubscribe when no longer needed
    price_cache.unsubscribe("BTC/USDT")

Architecture:
    - Single asyncio task polls all subscribed symbols every POLL_INTERVAL seconds
    - Deduplicates: 10 signals on 8 symbols = 8 API calls/cycle, not 30
    - All monitors call price_cache.get() instead of fetch_ticker() directly
    - Zero contention — cache is read-only from consumer side
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

POLL_INTERVAL = 10  # seconds — faster than any individual monitor was polling


class PriceCache:
    """
    Shared ticker cache. One polling loop, many readers.

    Thread-safety: asyncio single-threaded — no locks needed.
    """

    def __init__(self):
        self._prices: Dict[str, float] = {}          # symbol → last price
        self._timestamps: Dict[str, float] = {}      # symbol → last update time
        self._subscribers: Dict[str, int] = {}       # symbol → subscriber count
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stale_threshold = 120                  # seconds before price considered stale
        # EC4: During exchange maintenance windows (can last 10-30 min),
        # callers should check get_age() before trusting SL/TP decisions.
        # Use get_with_age() to get price + staleness together.

    def subscribe(self, symbol: str):
        """Register interest in a symbol. Reference-counted."""
        self._subscribers[symbol] = self._subscribers.get(symbol, 0) + 1
        logger.debug(f"PriceCache: subscribed {symbol} (refs={self._subscribers[symbol]})")

    def unsubscribe(self, symbol: str):
        """Release interest in a symbol. Stops polling when refcount hits 0."""
        count = self._subscribers.get(symbol, 0)
        if count <= 1:
            self._subscribers.pop(symbol, None)
            self._prices.pop(symbol, None)
            self._timestamps.pop(symbol, None)
            logger.debug(f"PriceCache: unsubscribed {symbol} (no more refs)")
        else:
            self._subscribers[symbol] = count - 1

    def get(self, symbol: str) -> Optional[float]:
        """
        Return the latest cached price for a symbol, or None if:
          - Symbol not subscribed
          - No price fetched yet
          - Price is stale (> stale_threshold seconds old)
        """
        ts = self._timestamps.get(symbol)
        if ts is None:
            return None
        if time.time() - ts > self._stale_threshold:
            return None
        return self._prices.get(symbol)

    def get_age(self, symbol: str) -> Optional[float]:
        """Return seconds since last update, or None if never fetched."""
        ts = self._timestamps.get(symbol)
        return (time.time() - ts) if ts else None

    def get_with_age(self, symbol: str) -> tuple:
        """
        EC4: Return (price, age_seconds) or (None, None).
        Outcome monitor uses this to skip SL checks when price is
        maintenance-stale (>120s old). A maintenance window can last 10-30 min.
        """
        ts = self._timestamps.get(symbol)
        if ts is None:
            return None, None
        age = time.time() - ts
        if age > self._stale_threshold:
            return None, age  # Return age so caller can log why it's being skipped
        price = self._prices.get(symbol)
        return price, age

    @property
    def subscribed_symbols(self) -> Set[str]:
        return set(self._subscribers.keys())

    def start(self):
        """Start the polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("📡 PriceCache started")

    async def warm_up(self, symbols: list):
        """V10: Pre-fetch prices for all symbols before monitors start."""
        from data.api_client import api
        if not symbols:
            return
        logger.info(f"📡 PriceCache warming up {len(symbols)} symbols...")
        batch_size = 20
        now = __import__('time').time()
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = {sym: api.fetch_ticker(sym) for sym in batch}
            results = await __import__('asyncio').gather(*tasks.values(), return_exceptions=True)
            for sym, result in zip(tasks.keys(), results):
                if not isinstance(result, Exception) and result and "last" in result:
                    self._prices[sym] = float(result["last"])
                    self._timestamps[sym] = now
        logger.info(f"📡 PriceCache warmed: {len(self._prices)} prices loaded")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("PriceCache stopped")

    async def _poll_loop(self):
        from data.api_client import api

        while self._running:
            symbols = list(self._subscribers.keys())

            if symbols:
                # Fetch all subscribed symbols concurrently
                tasks = {sym: api.fetch_ticker(sym) for sym in symbols}
                results = await asyncio.gather(
                    *tasks.values(), return_exceptions=True
                )
                now = time.time()
                for sym, result in zip(tasks.keys(), results):
                    if isinstance(result, Exception):
                        logger.debug(f"PriceCache fetch failed for {sym}: {result}")
                        continue
                    if result and "last" in result:
                        self._prices[sym] = float(result["last"])
                        self._timestamps[sym] = now

                logger.debug(
                    f"PriceCache: updated {len(symbols)} symbols "
                    f"({len([r for r in results if not isinstance(r, Exception)])} ok)"
                )

            await asyncio.sleep(POLL_INTERVAL)


# ── Singleton ──────────────────────────────────────────────
price_cache = PriceCache()
