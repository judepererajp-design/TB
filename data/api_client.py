"""
TitanBot Pro — API Client
=========================
Exchange-agnostic API client for perpetual futures PUBLIC endpoints.
Delegates exchange-specific logic to an ExchangeInterface implementation
(see data/exchange_interface.py).

NO API KEY REQUIRED — TitanBot is signal-only. All data is read
from public exchange APIs:
  - OHLCV candles (market data)
  - Order book depth
  - Funding rates
  - Open interest
  - Long/short ratios
  - Tickers and market info

Features:
  - Pluggable exchange back-end via ExchangeInterface / ExchangeFactory
  - Automatic rate limiting (respects exchange limits)
  - Retry logic with exponential backoff
  - In-memory cache with TTL (prevents duplicate fetches)
  - Clean error handling and logging

Usage:
    from data.api_client import api
    ohlcv = await api.fetch_ohlcv("BTC/USDT", "1h", limit=200)
    ticker = await api.fetch_ticker("BTC/USDT")
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import ccxt.async_support as ccxt

from config.loader import cfg
from data.exchange_interface import ExchangeInterface, ExchangeFactory
from data.websocket_feed import ws_feed

logger = logging.getLogger(__name__)


class Cache:
    """Simple TTL cache to avoid hammering the API with duplicate requests"""

    def __init__(self):
        self._store: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            value, expires_at = self._store[key]
            if time.time() < expires_at:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any, ttl: int):
        self._store[key] = (value, time.time() + ttl)

    def delete(self, key: str):
        self._store.pop(key, None)

    def clear_expired(self):
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now >= exp]
        for k in expired:
            del self._store[k]

    def size(self) -> int:
        return len(self._store)


class RateLimiter:
    """Simple rate limiter — ensures we don't exceed Binance's API limits"""

    def __init__(self, calls_per_second: float = 5.0):
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_call = time.time()


class APIClient:
    """
    Async exchange client for perpetual futures.
    Single instance shared across all modules.
    Delegates exchange-specific calls to an ExchangeInterface implementation.
    """

    _instance: Optional['APIClient'] = None

    def __new__(cls) -> 'APIClient':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._exchange_impl: Optional[ExchangeInterface] = None
            self._cache = Cache()
            self._rate_limiter = RateLimiter(calls_per_second=4.0)
            self._initialized = True
            self._healthy = False

            # Request telemetry
            self._total_requests: int = 0
            self._total_retries: int = 0
            self._total_errors: int = 0
            self._latency_sum: float = 0.0
            self._latency_max: float = 0.0
            self._cache_hits: int = 0

    def get_request_stats(self) -> Dict:
        """Get API request telemetry for heartbeat display"""
        avg_lat = (self._latency_sum / self._total_requests * 1000
                   if self._total_requests > 0 else 0)
        return {
            'total_requests': self._total_requests,
            'total_retries': self._total_retries,
            'total_errors': self._total_errors,
            'avg_latency_ms': round(avg_lat, 1),
            'max_latency_ms': round(self._latency_max * 1000, 1),
            'cache_hits': self._cache_hits,
            'cache_size': self._cache.size(),
        }

    async def initialize(self):
        """
        Create and initialize the exchange implementation.
        The exchange back-end is chosen from ``cfg.exchange.name``
        (defaults to ``'binance'``).
        Optionally starts a WebSocket feed for real-time prices.
        """
        exchange_name = cfg.exchange.get('name', 'binance')
        self._exchange_impl = ExchangeFactory.create(exchange_name)
        await self._exchange_impl.initialize()
        self._healthy = self._exchange_impl.is_healthy

        # Start WebSocket feed if enabled in config
        if cfg.exchange.get('use_websocket', False):
            try:
                symbols = self.get_symbol_list()
                if symbols:
                    await ws_feed.start(symbols)
                    logger.info("WebSocket feed started alongside REST client")
            except Exception as exc:
                logger.warning(f"WebSocket feed failed to start, REST-only mode: {exc}")

    async def close(self):
        """Clean up connections"""
        if ws_feed.is_connected:
            await ws_feed.stop()
        if self._exchange_impl:
            await self._exchange_impl.close()
            logger.info("API client closed")

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    # ── Core fetch methods ──────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
        use_cache: bool = True
    ) -> Optional[List]:
        """
        Fetch OHLCV candles for a symbol.
        Returns list of [timestamp, open, high, low, close, volume]
        """
        cache_key = f"ohlcv:{symbol}:{timeframe}:{limit}"
        ttl = cfg.cache.get('ohlcv_ttl', 60)

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached

        result = await self._fetch_with_retry(
            self._exchange_impl.fetch_ohlcv,
            symbol, timeframe, limit=limit
        )

        if result and use_cache:
            self._cache.set(cache_key, result, ttl)

        return result

    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data. Checks WebSocket feed first, REST fallback."""
        # Fast path: WebSocket provides sub-100ms price if connected
        if ws_feed.is_connected:
            ws_price = ws_feed.get_price(symbol)
            if ws_price is not None:
                return {"symbol": symbol, "last": ws_price, "source": "websocket"}

        cache_key = f"ticker:{symbol}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = await self._fetch_with_retry(
            self._exchange_impl.fetch_ticker, symbol
        )

        if result:
            self._cache.set(cache_key, result, 5)  # 5s TTL for tickers

        return result

    async def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict:
        """Fetch all tickers (used for universe building)"""
        cache_key = "all_tickers"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = await self._fetch_with_retry(
            self._exchange_impl.fetch_tickers, symbols
        )

        if result:
            self._cache.set(cache_key, result, 30)

        return result or {}

    async def fetch_order_book(self, symbol: str, limit: int = 50) -> Optional[Dict]:
        """Fetch order book depth"""
        cache_key = f"orderbook:{symbol}"
        ttl = cfg.cache.get('orderflow_ttl', 3)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = await self._fetch_with_retry(
            self._exchange_impl.fetch_order_book, symbol, limit=limit
        )

        if result:
            self._cache.set(cache_key, result, ttl)

        return result

    async def fetch_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Fetch current funding rate for perpetual futures"""
        cache_key = f"funding:{symbol}"
        ttl = cfg.cache.get('derivatives_ttl', 30)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = await self._fetch_with_retry(
                self._exchange_impl.fetch_funding_rate, symbol
            )
            if result:
                self._cache.set(cache_key, result, ttl)
            return result
        except Exception:
            return None  # Not all symbols have funding rates

    async def fetch_open_interest(self, symbol: str) -> Optional[Dict]:
        """Fetch open interest"""
        cache_key = f"oi:{symbol}"
        ttl = cfg.cache.get('derivatives_ttl', 30)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = await self._fetch_with_retry(
                self._exchange_impl.fetch_open_interest, symbol
            )
            if result:
                self._cache.set(cache_key, result, ttl)
            return result
        except Exception:
            return None

    async def fetch_long_short_ratio(self, symbol: str, period: str = "1h") -> Optional[Dict]:
        """Fetch long/short ratio from exchange"""
        cache_key = f"lsr:{symbol}:{period}"
        ttl = cfg.cache.get('derivatives_ttl', 30)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = await self._fetch_with_retry(
                self._exchange_impl.fetch_long_short_ratio, symbol, period
            )
            if result:
                self._cache.set(cache_key, result, ttl)
            return result
        except Exception:
            return None

    async def fetch_markets(self) -> Dict:
        """Get all available markets"""
        if self._exchange_impl:
            return await self._exchange_impl.fetch_markets()
        return {}

    # ── Retry logic ─────────────────────────────────────────

    async def _fetch_with_retry(self, func, *args, **kwargs) -> Optional[Any]:
        """
        Execute an API call with retry logic and rate limiting.
        Uses exponential backoff on failure.
        """
        max_retries = cfg.exchange.get('retry_attempts', 3)
        base_delay = cfg.exchange.get('retry_delay', 2.0)

        for attempt in range(max_retries):
            try:
                await self._rate_limiter.wait()
                t0 = time.time()
                # Timeout protection: prevent hung coroutines if Binance stalls
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=12.0
                )
                latency = time.time() - t0

                # Track telemetry
                self._total_requests += 1
                self._latency_sum += latency
                if latency > self._latency_max:
                    self._latency_max = latency

                # Feed diagnostic engine (non-fatal)
                try:
                    from core.diagnostic_engine import diagnostic_engine
                    _ep = getattr(func, '__name__', 'unknown')
                    diagnostic_engine.record_api_call(
                        endpoint=_ep,
                        latency_ms=latency * 1000,
                        success=True,
                    )
                except Exception:
                    pass

                self._healthy = True
                return result

            except asyncio.TimeoutError:
                self._total_retries += 1
                self._total_requests += 1
                # FIX: record the full 12s timeout as the latency for this call so
                # avg_latency_ms reflects reality. Previously timed-out calls didn't
                # update _latency_sum, making the average appear healthy when the
                # exchange was consistently hanging workers for 12 seconds.
                self._latency_sum += 12.0
                if 12.0 > self._latency_max:
                    self._latency_max = 12.0
                logger.warning(
                    f"API timeout (attempt {attempt+1}/{max_retries}) — "
                    f"call hung >12s, releasing worker"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay)
                continue

            except ccxt.RateLimitExceeded:
                self._total_retries += 1
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit — waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

            except ccxt.NetworkError as e:
                self._total_retries += 1
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"Network error (attempt {attempt+1}/{max_retries}): {e} — retrying in {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

            except ccxt.AuthenticationError as e:
                # E2: API key expired/revoked — stop retrying, alert operator
                self._total_errors += 1
                self._healthy = False
                logger.critical(
                    f"🔑 AuthenticationError — API key may be expired or revoked: {e}. "
                    f"Check your API credentials in .env"
                )
                return None  # Don't retry — auth failures won't self-heal

            except ccxt.ExchangeError as e:
                self._total_errors += 1
                logger.error(f"Exchange error: {e}")
                self._healthy = False
                return None

            except Exception as e:
                if attempt == max_retries - 1:
                    self._total_errors += 1
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    self._healthy = False
                    return None
                self._total_retries += 1
                await asyncio.sleep(base_delay)

        return None

    # ── Utility ─────────────────────────────────────────────

    def get_symbol_list(self, min_volume: float = 0) -> List[str]:
        """Get list of USDT perpetual symbols from loaded markets"""
        if self._exchange_impl:
            return self._exchange_impl.get_symbol_list(min_volume)
        return []

    def cache_stats(self) -> Dict[str, int]:
        """Debug: show cache size"""
        self._cache.clear_expired()
        return {'size': self._cache.size()}


# ── Singleton instance ─────────────────────────────────────
api = APIClient()
