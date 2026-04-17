"""
TitanBot Pro — CoinGecko Trending Client
=========================================
Polls CoinGecko's free trending endpoint every 10 minutes.
No API key required. Rate limit: ~50 calls/minute (free tier).

Provides:
  - Trending coins list (updates ~every 10 min)
  - +6 confidence boost for signals on trending coins
  - Listed as a confluence factor on signal cards

Endpoint: https://api.coingecko.com/api/v3/search/trending
Returns top 7 trending coins by search volume on CoinGecko in the last 24h.
"""

import asyncio
import logging
import time
from typing import Set, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.coingecko.com/api/v3"
_POLL_INTERVAL = 600  # 10 minutes — trending list doesn't change rapidly


# Map CoinGecko coin IDs / symbols to our trading pair format
# CoinGecko returns: {"item": {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", ...}}
_CG_SYMBOL_MAP: Dict[str, str] = {
    # CoinGecko symbol → our uppercase ticker
    "btc": "BTC", "eth": "ETH", "sol": "SOL", "bnb": "BNB",
    "xrp": "XRP", "ada": "ADA", "doge": "DOGE", "avax": "AVAX",
    "dot": "DOT", "link": "LINK", "uni": "UNI", "aave": "AAVE",
    "matic": "MATIC", "op": "OP", "arb": "ARB", "sui": "SUI",
    "apt": "APT", "hbar": "HBAR", "near": "NEAR", "fil": "FIL",
    "ltc": "LTC", "etc": "ETC", "trx": "TRX", "xlm": "XLM",
    "fet": "FET", "wld": "WLD", "rndr": "RNDR", "ocean": "OCEAN",
    "axs": "AXS", "gala": "GALA", "imx": "IMX",
    "pepe": "PEPE", "shib": "SHIB", "bonk": "BONK", "wif": "WIF",
    "floki": "FLOKI", "moodeng": "MOODENG", "pengu": "PENGU",
    "virtual": "VIRTUAL", "bera": "BERA", "hype": "HYPE",
    "tao": "TAO", "ena": "ENA", "mkr": "MKR", "crv": "CRV",
    "icp": "ICP", "xmr": "XMR", "bch": "BCH", "flow": "FLOW",
    "trump": "TRUMP", "fartcoin": "FARTCOIN",
}


class CoinGeckoClient:
    """
    Polls CoinGecko trending endpoint.
    Stores a set of currently-trending tickers for fast lookup.
    """

    def __init__(self):
        self._trending: Set[str] = set()       # e.g. {"BTC", "SOL", "HYPE"}
        self._trending_names: Dict[str, str] = {}  # ticker → display name
        self._last_fetch: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._poll_loop())
            logger.info("📈 CoinGecko trending client started (poll every 10min)")

    async def stop(self):
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

    async def _get_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "TitanBot/2.0 (trading-signals-bot)"},
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def _poll_loop(self):
        await asyncio.sleep(15)  # stagger after startup
        while self._running:
            try:
                await self._fetch_trending()
                await asyncio.sleep(_POLL_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"CoinGecko poll error: {e}")
                await asyncio.sleep(120)

    async def _fetch_trending(self):
        session = await self._get_session()
        try:
            url = f"{_BASE_URL}/search/trending"
            async with session.get(url) as resp:
                if resp.status == 429:
                    logger.warning("CoinGecko rate limited — will retry in 10 min")
                    return
                if resp.status != 200:
                    logger.warning(f"CoinGecko trending returned HTTP {resp.status}")
                    return
                data = await resp.json()

            coins = data.get("coins", [])
            new_trending: Set[str] = set()
            new_names: Dict[str, str] = {}

            for entry in coins:
                item = entry.get("item", {})
                cg_symbol = item.get("symbol", "").lower()
                cg_name   = item.get("name", "")
                ticker = _CG_SYMBOL_MAP.get(cg_symbol, cg_symbol.upper())
                new_trending.add(ticker)
                new_names[ticker] = cg_name

            self._trending = new_trending
            self._trending_names = new_names
            self._last_fetch = time.time()

            if new_trending:
                logger.info(
                    f"📈 CoinGecko trending: {', '.join(sorted(new_trending))}"
                )

        except Exception as e:
            logger.warning(f"CoinGecko fetch error: {e}")

    def is_trending(self, symbol: str) -> bool:
        """Returns True if symbol is currently trending on CoinGecko."""
        # AUDIT FIX: use split('/') so perpetual-future suffixes like
        # "SOL/USDT:USDT" don't leave trailing ":USDT" on the ticker and
        # silently miss every lookup.
        ticker = symbol.split("/")[0].upper()
        return ticker in self._trending

    def get_trending_boost(self, symbol: str) -> tuple[int, str]:
        """
        Returns (confidence_boost, note) for a symbol.
        Boost = +6 if trending, 0 otherwise.
        """
        if self.is_trending(symbol):
            ticker = symbol.split("/")[0].upper()
            name = self._trending_names.get(ticker, ticker)
            return 6, f"📈 Trending on CoinGecko: {name}"
        return 0, ""

    def get_trending_list(self) -> list:
        """Returns list of currently trending tickers."""
        return sorted(self._trending)

    @property
    def last_updated(self) -> float:
        return self._last_fetch


# ── Singleton ─────────────────────────────────────────────────
coingecko = CoinGeckoClient()
