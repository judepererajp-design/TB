"""
TitanBot Pro — Exchange Abstraction Layer
==========================================
Abstract interface for exchange implementations, allowing TitanBot
to support multiple exchanges through a common API.

Currently supported:
  - Binance Perpetual Futures (via CCXT)

Usage:
    from data.exchange_interface import ExchangeFactory, ExchangeInterface

    exchange = ExchangeFactory.create("binance")
    await exchange.initialize()
    ohlcv = await exchange.fetch_ohlcv("BTC/USDT", "1h", limit=200)
"""

from __future__ import annotations

import abc
import asyncio
import logging
from typing import Dict, List, Optional

import ccxt.async_support as ccxt

from config.loader import cfg

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Abstract Interface
# ════════════════════════════════════════════════════════════════

class ExchangeInterface(abc.ABC):
    """Abstract base class for exchange implementations.

    Defines the contract that all exchange adapters must implement.
    Each method corresponds to a public market-data endpoint.
    """

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Connect to the exchange and load market metadata."""
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the exchange connection and release resources."""
        ...

    @abc.abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> Optional[List]:
        """Fetch OHLCV candles for a symbol.

        Returns list of [timestamp, open, high, low, close, volume].
        """
        ...

    @abc.abstractmethod
    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data for a symbol."""
        ...

    @abc.abstractmethod
    async def fetch_tickers(
        self, symbols: Optional[List[str]] = None
    ) -> Dict:
        """Fetch tickers for multiple symbols."""
        ...

    @abc.abstractmethod
    async def fetch_order_book(
        self, symbol: str, limit: int = 50
    ) -> Optional[Dict]:
        """Fetch order book depth for a symbol."""
        ...

    @abc.abstractmethod
    async def fetch_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Fetch current funding rate for a perpetual futures symbol."""
        ...

    @abc.abstractmethod
    async def fetch_open_interest(self, symbol: str) -> Optional[Dict]:
        """Fetch open interest for a symbol."""
        ...

    @abc.abstractmethod
    async def fetch_long_short_ratio(
        self, symbol: str, period: str = "1h"
    ) -> Optional[Dict]:
        """Fetch long/short ratio for a symbol."""
        ...

    @abc.abstractmethod
    async def fetch_markets(self) -> Dict:
        """Get all available markets."""
        ...

    @abc.abstractmethod
    def get_symbol_list(self, min_volume: float = 0) -> List[str]:
        """Get list of tradable symbols, optionally filtered by volume."""
        ...

    @property
    @abc.abstractmethod
    def is_healthy(self) -> bool:
        """Whether the exchange connection is healthy."""
        ...

    @property
    @abc.abstractmethod
    def exchange_name(self) -> str:
        """Name identifier for this exchange."""
        ...


# ════════════════════════════════════════════════════════════════
# Binance Implementation
# ════════════════════════════════════════════════════════════════

class BinanceExchange(ExchangeInterface):
    """
    Binance Perpetual Futures implementation of ExchangeInterface.

    Wraps CCXT's async Binance client for USDT-margined perpetual
    futures public endpoints.  No API key required.
    """

    def __init__(self) -> None:
        self._exchange: Optional[ccxt.binance] = None
        self._healthy: bool = False

    # ── Lifecycle ───────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Set up CCXT Binance Futures connection — public endpoints only,
        no API key needed.  Retries up to configured attempts with
        linear backoff.  ``load_markets()`` downloads all 300+ contract
        specs so the timeout must be generous.
        """
        ex_cfg = cfg.exchange
        timeout_ms = ex_cfg.get('timeout_ms', 30000)
        retry_attempts = ex_cfg.get('retry_attempts', 5)
        retry_delay = ex_cfg.get('retry_delay', 3.0)

        last_error: Exception = RuntimeError('No retry attempts configured')
        for attempt in range(1, retry_attempts + 1):
            try:
                # Fresh exchange instance each attempt
                self._exchange = ccxt.binance({
                    'timeout': timeout_ms,
                    'enableRateLimit': True,
                    'adjustForTimeDifference': True,  # handles clock skew
                    'options': {
                        'defaultType': 'future',      # USDT-margined perpetuals
                        'fetchMarkets': {'type': 'future'},
                        'warnOnFetchOpenOrdersWithoutSymbol': False,
                    },
                })

                logger.info(
                    f"Connecting to Binance Futures "
                    f"(attempt {attempt}/{retry_attempts}, "
                    f"timeout {timeout_ms / 1000:.0f}s)…"
                )
                await self._exchange.load_markets()
                self._healthy = True

                # Count only perpetual futures (exclude quarterly, options)
                perps = [
                    m for m in self._exchange.markets.values()
                    if m.get('type') == 'swap' and m.get('active')
                ]
                logger.info(
                    f"✅ Connected to Binance Futures — "
                    f"{len(self._exchange.markets)} markets loaded "
                    f"({len(perps)} active perpetuals)"
                )
                return  # success

            except Exception as e:
                last_error = e
                err_type = type(e).__name__
                err_msg = str(e)

                # Log the FULL error so we can see exactly what's wrong
                logger.error(
                    f"❌ Binance connection attempt "
                    f"{attempt}/{retry_attempts} failed\n"
                    f"   Type:    {err_type}\n"
                    f"   Detail:  {err_msg[:400]}"
                )

                # Diagnosis hints
                if 'Timeout' in err_type or 'timeout' in err_msg.lower():
                    logger.warning(
                        f"   → Timeout. Increase exchange.timeout_ms in "
                        f"settings.yaml (currently {timeout_ms}ms). "
                        f"Or check internet speed."
                    )
                elif ('NetworkError' in err_type
                      or 'ConnectionRefused' in err_msg):
                    logger.warning(
                        "   → Network unreachable. Check internet "
                        "connection. Run: curl -I "
                        "https://api.binance.com/api/v3/ping"
                    )
                elif '451' in err_msg:
                    logger.warning(
                        "   → HTTP 451: Binance geo-restricted in your "
                        "region. Use a VPN."
                    )
                elif 'ExchangeNotAvailable' in err_type:
                    logger.warning(
                        "   → Binance returned a server error (503/502). "
                        "Will retry — this is usually temporary."
                    )

                if attempt < retry_attempts:
                    wait = retry_delay * attempt  # 3s, 6s, 9s, 12s …
                    logger.info(f"   Retrying in {wait:.0f}s…")
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "   All retry attempts exhausted. "
                        "Check logs above for the error type and fix it."
                    )

                # Close the failed exchange object before next attempt
                try:
                    if self._exchange:
                        await self._exchange.close()
                except Exception:
                    pass
                self._exchange = None

        raise last_error

    async def close(self) -> None:
        """Close the Binance exchange connection."""
        if self._exchange:
            await self._exchange.close()
            logger.info("Binance exchange connection closed")

    # ── Properties ──────────────────────────────────────────

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @is_healthy.setter
    def is_healthy(self, value: bool) -> None:
        self._healthy = value

    @property
    def exchange_name(self) -> str:
        return "binance"

    # ── Market data fetchers ────────────────────────────────

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> Optional[List]:
        """Fetch OHLCV candles from Binance Futures."""
        return await self._exchange.fetch_ohlcv(
            symbol, timeframe, limit=limit
        )

    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker from Binance Futures."""
        return await self._exchange.fetch_ticker(symbol)

    async def fetch_tickers(
        self, symbols: Optional[List[str]] = None
    ) -> Dict:
        """Fetch tickers from Binance Futures."""
        return await self._exchange.fetch_tickers(symbols)

    async def fetch_order_book(
        self, symbol: str, limit: int = 50
    ) -> Optional[Dict]:
        """Fetch order book depth from Binance Futures."""
        return await self._exchange.fetch_order_book(symbol, limit=limit)

    async def fetch_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Fetch funding rate from Binance Futures."""
        return await self._exchange.fetch_funding_rate(symbol)

    async def fetch_open_interest(self, symbol: str) -> Optional[Dict]:
        """Fetch open interest from Binance Futures."""
        return await self._exchange.fetch_open_interest(symbol)

    async def fetch_long_short_ratio(
        self, symbol: str, period: str = "1h"
    ) -> Optional[Dict]:
        """Fetch long/short ratio via Binance-specific API endpoint."""
        clean_symbol = symbol.replace('/USDT', 'USDT').replace('/', '')
        return await self._exchange.fapiDataGetGlobalLongShortAccountRatio(
            {'symbol': clean_symbol, 'period': period, 'limit': 1}
        )

    async def fetch_markets(self) -> Dict:
        """Get all loaded Binance Futures markets."""
        return self._exchange.markets if self._exchange else {}

    # ── Utility ─────────────────────────────────────────────

    def get_symbol_list(self, min_volume: float = 0) -> List[str]:
        """Get list of USDT perpetual symbols from loaded markets."""
        if not self._exchange or not self._exchange.markets:
            return []

        excluded_raw = cfg.exchange.get('excluded_symbols', [])
        # Normalise to a plain set of strings regardless of whether the
        # config value is a list, a dict (old YAML format), or a
        # ConfigNode wrapper.
        if hasattr(excluded_raw, 'to_dict'):
            excluded_raw = list(excluded_raw.to_dict().values())
        if isinstance(excluded_raw, dict):
            excluded_raw = list(excluded_raw.values())
        excluded = (
            set(excluded_raw)
            if isinstance(excluded_raw, (list, tuple))
            else set()
        )

        symbols: List[str] = []
        for symbol, market in self._exchange.markets.items():
            if (market.get('quote') == 'USDT'
                    and market.get('type') == 'swap'
                    and market.get('active', True)
                    and symbol not in excluded):
                symbols.append(symbol)

        return symbols


# ════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════

class ExchangeFactory:
    """Factory for creating exchange instances by name."""

    _EXCHANGES: Dict[str, type] = {
        'binance': BinanceExchange,
    }

    @staticmethod
    def create(exchange_name: str) -> ExchangeInterface:
        """
        Create an exchange instance by name.

        Args:
            exchange_name: Exchange identifier (e.g. ``'binance'``).

        Returns:
            An uninitialised :class:`ExchangeInterface` implementation.
            Call ``await instance.initialize()`` before use.

        Raises:
            ValueError: If the exchange name is not supported.
        """
        key = exchange_name.lower().strip()
        exchange_cls = ExchangeFactory._EXCHANGES.get(key)
        if exchange_cls is None:
            supported = sorted(ExchangeFactory._EXCHANGES.keys())
            raise ValueError(
                f"Unsupported exchange: '{exchange_name}'. "
                f"Supported: {supported}"
            )
        return exchange_cls()
