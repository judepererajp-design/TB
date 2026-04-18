"""
TitanBot Pro — News & Sentiment Scraper
=========================================
Pulls from free real-time data sources — no paid APIs required.

Sources (15 RSS feeds, all real-time, no API key):
  Tier 1: CoinTelegraph, CoinDesk, Decrypt, TheBlock, Blockworks
  Tier 2: CryptoSlate, BeInCrypto, NewsBTC, AMBCrypto, CoinGape
  Tier 3: BinanceBlog, BitcoinMagazine
  Tier 4: U.Today, CryptoBriefing, InsideBitcoins

Plus free on-chain/market data:
  - CoinPaprika events API (free, no key)
  - DeFiLlama TVL changes (free, no key)
  - Alternative.me Fear & Greed (already in bot, extended here)

All sources are polled on a schedule and cached.
The cache is read by ai_analyst.py to build signal context.
"""

import asyncio
import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import aiohttp

from config.constants import NewsIntelligence
from config.feature_flags import ff
from config.shadow_mode import shadow_log

logger = logging.getLogger(__name__)

_DEDUP_ENTITY_TOKENS = {
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "ripple",
    "doge", "dogecoin", "pepe", "shib", "shiba", "sec", "fed", "fomc",
    "cpi", "etf", "blackrock", "coinbase", "binance", "microstrategy",
}


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published_at: float     # Unix timestamp
    coins_mentioned: List[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class TVLChange:
    protocol: str
    chain: str
    tvl_usd: float
    change_1h_pct: float
    change_24h_pct: float
    related_tokens: List[str] = field(default_factory=list)


# ── RSS feed sources ──────────────────────────────────────────
# All free, real-time (no API key, no delay).
# Polled every ~15 minutes. 15 sources gives broad coverage
# across news, analysis, and exchange announcements.
RSS_FEEDS = [
    # ── Tier 1: Major crypto media (highest signal quality) ──
    ("CoinTelegraph",   "https://cointelegraph.com/rss"),
    ("CoinDesk",        "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Decrypt",         "https://decrypt.co/feed"),
    ("TheBlock",        "https://www.theblock.co/rss.xml"),
    ("Blockworks",      "https://blockworks.co/feed"),

    # ── Tier 2: Quality secondary sources ────────────────────
    ("CryptoSlate",     "https://cryptoslate.com/feed/"),
    ("BeInCrypto",      "https://beincrypto.com/feed/"),
    ("NewsBTC",         "https://www.newsbtc.com/feed/"),
    ("AMBCrypto",       "https://ambcrypto.com/feed/"),
    ("CoinGape",        "https://coingape.com/feed/"),

    # ── Tier 3: Exchange / protocol official feeds ────────────
    ("BinanceBlog",     "https://www.binance.com/en/blog/rss"),
    ("BitcoinMagazine", "https://bitcoinmagazine.com/feed"),

    # ── Tier 4: Broader financial / macro (crypto-relevant) ──
    ("U.Today",         "https://u.today/rss"),
    ("CryptoBriefing",  "https://cryptobriefing.com/feed/"),
    ("Utoday",          "https://insidebitcoins.com/feed"),
]

# Common coin name/ticker variations for mention detection
# Covers all major symbols in the 320-symbol scanning universe
_COIN_ALIASES: Dict[str, List[str]] = {
    # ── Layer 1s ─────────────────────────────────────────────
    "BTC":    ["bitcoin", "btc", "satoshi"],
    "ETH":    ["ethereum", "eth", "ether"],
    "SOL":    ["solana", "sol"],
    "BNB":    ["binance coin", "bnb", "bsc", "binance smart chain"],
    "XRP":    ["ripple", "xrp"],
    "ADA":    ["cardano", "ada"],
    "AVAX":   ["avalanche", "avax"],
    "DOT":    ["polkadot", "dot"],
    "NEAR":   ["near protocol", "near"],
    "ICP":    ["internet computer", "icp", "dfinity"],
    "FIL":    ["filecoin", "fil"],
    "LTC":    ["litecoin", "ltc"],
    "ETC":    ["ethereum classic", "etc"],
    "XLM":    ["stellar", "xlm"],
    "HBAR":   ["hedera", "hbar", "hedera hashgraph"],
    "XMR":    ["monero", "xmr"],
    "TAO":    ["bittensor", "tao"],
    "SUI":    ["sui"],
    "APT":    ["aptos", "apt"],
    "TRX":    ["tron", "trx"],
    "BCH":    ["bitcoin cash", "bch"],

    # ── DeFi / L2s ───────────────────────────────────────────
    "LINK":   ["chainlink", "link"],
    "UNI":    ["uniswap", "uni"],
    "AAVE":   ["aave"],
    "CRV":    ["curve", "crv", "curve finance"],
    "MKR":    ["maker", "mkr", "makerdao", "dai"],
    "MATIC":  ["polygon", "matic"],
    "OP":     ["optimism", "op"],
    "ARB":    ["arbitrum", "arb"],
    "FTM":    ["fantom", "ftm"],
    "LYN":    ["lyn", "lineage"],
    "ENA":    ["ethena", "ena"],
    "PENDLE": ["pendle"],
    "COS":    ["contentos", "cos"],

    # ── Memes / Social ────────────────────────────────────────
    "DOGE":   ["dogecoin", "doge"],
    "SHIB":   ["shiba inu", "shib", "shiba"],
    "PEPE":   ["pepe", "pepecoin"],
    "WIF":    ["dogwifhat", "wif"],
    "BONK":   ["bonk"],
    "FLOKI":  ["floki"],
    "TRUMP":  ["trump coin", "official trump", "trump"],
    "MOODENG":["moodeng"],
    "FARTCOIN":["fartcoin"],
    "PENGU":  ["pudgy penguins", "pengu", "pudgy"],

    # ── AI / Tech ─────────────────────────────────────────────
    "FET":    ["fetch.ai", "fet", "fetch ai", "artificial superintelligence"],
    "RNDR":   ["render", "rndr", "render network"],
    "GRT":    ["the graph", "grt"],
    "OCEAN":  ["ocean protocol", "ocean"],
    "AGIX":   ["singularitynet", "agix"],
    "TAO":    ["bittensor", "tao"],
    "WLD":    ["worldcoin", "wld", "world coin"],
    "VIRTUAL":["virtual protocol", "virtual"],

    # ── Gaming / NFT ─────────────────────────────────────────
    "AXS":    ["axie infinity", "axs", "axie"],
    "GALA":   ["gala games", "gala"],
    "SAND":   ["the sandbox", "sand", "sandbox"],
    "IMX":    ["immutable", "imx", "immutable x"],
    "APE":    ["apecoin", "ape"],

    # ── Stablecoins / Yield ───────────────────────────────────
    "USDT":   ["tether", "usdt"],
    "USDC":   ["usd coin", "usdc", "circle"],
    "DAI":    ["dai"],
    "FRAX":   ["frax"],

    # ── Exchange tokens ───────────────────────────────────────
    "OKB":    ["okx", "okb", "okex"],
    "CRO":    ["crypto.com", "cro", "cronos"],
    "HT":     ["huobi", "ht"],

    # ── Other top-100 ─────────────────────────────────────────
    "HYPE":   ["hyperliquid", "hype"],
    "XAU":    ["gold", "xau", "xauusd"],
    "XAG":    ["silver", "xag"],
    "ICP":    ["internet computer", "icp"],
    "FLOW":   ["flow blockchain", "flow"],
    "ENA":    ["ethena", "ena"],
    "RESOLV": ["resolv"],
    "BERA":   ["berachain", "bera"],
}


class NewsScraper:
    """
    Async news and market context scraper.
    All data is cached — polling respects source rate limits.
    """

    def __init__(self):
        # News cache: source -> list of NewsItems
        self._news_cache: List[NewsItem] = []
        self._news_last_fetch: float = 0.0
        self._news_ttl: float = 120.0      # refresh every 2 minutes

        # TVL cache
        self._tvl_cache: List[TVLChange] = []
        self._tvl_last_fetch: float = 0.0
        self._tvl_ttl: float = 300.0        # refresh every 5 minutes

        # CoinPaprika events cache: symbol -> list of events
        self._events_cache: Dict[str, List[dict]] = {}
        self._events_last_fetch: float = 0.0
        self._events_ttl: float = 3600.0    # refresh every hour

        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._task = None
        # AUDIT FIX (news_scraper fire-and-forget): keep strong references to
        # background dispatch tasks so they aren't garbage collected mid-run
        # and any failure is visible via the done-callback installed at spawn.
        self._bni_tasks: set = set()

        # Feature 3: Title dedup — tracks clusters of similar headlines
        self._dedup_clusters: List[Dict] = []  # [{titles: set, source_count: int, first_seen: float}]

        # Feature 4: Headline evolution — URL -> (title_hash, original_title, timestamp)
        self._headline_tracker: Dict[str, Tuple[str, str, float]] = {}
        # Titles whose confidence should be downgraded due to evolution
        self._evolved_titles: Dict[str, float] = {}  # original_title -> penalty multiplier

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": "TitanBot/1.0"},
            )
        return self._session

    def start(self):
        """Start the background polling loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._poll_loop())
            logger.info("📰 News scraper started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    async def _poll_loop(self):
        """Background loop — stagger fetches to avoid burst requests."""
        await asyncio.sleep(5)  # Let bot fully start first
        while self._running:
            try:
                await self._fetch_news()
                await asyncio.sleep(30)
                await self._fetch_tvl()
                await asyncio.sleep(30)
                await self._fetch_events()
                await asyncio.sleep(60)  # Total ~2min cycle
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"News scraper poll error: {e}")
                await asyncio.sleep(60)

    # ── RSS News ──────────────────────────────────────────────

    async def _fetch_news(self):
        """Fetch and parse all RSS feeds."""
        if time.time() - self._news_last_fetch < self._news_ttl:
            return
        session = await self._get_session()
        new_items = []

        _rss_successes = 0
        _rss_total = len(RSS_FEEDS)
        for source_name, url in RSS_FEEDS:
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
                    items = self._parse_rss(text, source_name)
                    new_items.extend(items)
                    _rss_successes += 1
            except Exception as e:
                logger.debug(f"RSS fetch error ({source_name}): {e}")

        # Wire degradation engine: report RSS success rate every cycle
        try:
            from utils.degradation import degradation_engine
            degradation_engine.update_rss_success_rate(_rss_successes, _rss_total)
        except Exception:
            pass

        if new_items:
            # Merge with existing cache, keep last 6 hours
            cutoff = time.time() - 21600
            self._news_cache = [n for n in self._news_cache if n.published_at > cutoff]
            # Add new items, deduplicate by title
            existing_titles = {n.title for n in self._news_cache}
            fresh = []
            for item in new_items:
                if item.title not in existing_titles:
                    self._news_cache.append(item)
                    fresh.append(item)
                    existing_titles.add(item.title)
                    # Feature 4: Track headline evolution by URL
                    if ff.is_active("HEADLINE_EVOLUTION") and item.url:
                        self._track_headline_evolution(item.url, item.title)

            # ── Supplemental source: free-crypto-news API ──────────
            # https://github.com/nirholas/free-crypto-news
            # No API key required. 200+ sources, AI-enriched.
            # Catches headlines that RSS feeds miss (e.g. geopolitical).
            try:
                _api_items = await self._fetch_free_crypto_news(session)
                for item in _api_items:
                    if item.title not in existing_titles:
                        self._news_cache.append(item)
                        fresh.append(item)
                        existing_titles.add(item.title)
            except Exception as _fcn_err:
                logger.debug(f"free-crypto-news API error (non-fatal): {_fcn_err}")

            self._news_cache.sort(key=lambda x: x.published_at, reverse=True)
            logger.debug(f"📰 News cache: {len(self._news_cache)} items from {len(RSS_FEEDS)} feeds + API")

            # Feed fresh headlines to BTC News Intelligence for event classification
            if fresh:
                try:
                    from analyzers.btc_news_intelligence import btc_news_intelligence
                    from analyzers.narrative_tracker import narrative_tracker
                    fresh_dicts = [
                        {"title": n.title, "published_at": n.published_at, "source": n.source}
                        for n in fresh
                    ]
                    for n in fresh:
                        narrative_tracker.process_headline(n.title, n.published_at)
                    # AUDIT FIX (news_scraper fire-and-forget): track the
                    # spawned task and log any failure from the background
                    # coroutine.  Previously `create_task(...)` with bare
                    # `except Exception: pass` swallowed both spawn errors
                    # AND downstream coroutine errors silently.
                    _bni_task = asyncio.create_task(
                        btc_news_intelligence.process_headlines(fresh_dicts),
                        name="btc_news_intelligence.process_headlines",
                    )
                    self._bni_tasks.add(_bni_task)

                    def _bni_done(t: asyncio.Task) -> None:
                        self._bni_tasks.discard(t)
                        if t.cancelled():
                            return
                        exc = t.exception()
                        if exc is not None:
                            # Include the exception traceback via exc_info so
                            # log consumers get a full stack, not just str(exc).
                            logger.warning(
                                "btc_news_intelligence.process_headlines failed",
                                exc_info=exc,
                            )

                    _bni_task.add_done_callback(_bni_done)
                except Exception as _bni_err:
                    logger.warning("btc_news_intelligence dispatch failed: %s", _bni_err)

        self._news_last_fetch = time.time()

    def _parse_rss(self, xml_text: str, source: str) -> List[NewsItem]:
        """Parse RSS XML into NewsItem list. No external XML libraries needed."""
        items = []
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            # Handle both RSS 2.0 and Atom formats
            entries = root.findall(".//item") or root.findall(".//atom:entry", ns)

            for entry in entries[:20]:  # Max 20 per source
                # Title
                title_el = entry.find("title")
                title = ""
                if title_el is not None:
                    title = (title_el.text or "").strip()
                    # Strip CDATA if present
                    if title.startswith("<![CDATA["):
                        title = title[9:-3].strip()
                if not title:
                    continue

                # URL
                link_el = entry.find("link")
                url = ""
                if link_el is not None:
                    url = (link_el.text or link_el.get("href", "")).strip()

                # Published time
                pub_time = time.time()
                for tag in ("pubDate", "published", "updated", "dc:date"):
                    el = entry.find(tag)
                    if el is not None and el.text:
                        try:
                            from email.utils import parsedate_to_datetime
                            pub_time = parsedate_to_datetime(el.text).timestamp()
                            break
                        except Exception:
                            try:
                                from datetime import datetime
                                pub_time = datetime.fromisoformat(
                                    el.text.replace("Z", "+00:00")
                                ).timestamp()
                                break
                            except Exception:
                                pass

                # Only include items from last 6 hours
                if time.time() - pub_time > 21600:
                    continue

                # Detect coin mentions
                text_lower = title.lower()
                mentioned = []
                for ticker, aliases in _COIN_ALIASES.items():
                    if any(alias in text_lower for alias in aliases):
                        mentioned.append(ticker)

                items.append(NewsItem(
                    title=title,
                    source=source,
                    url=url,
                    published_at=pub_time,
                    coins_mentioned=mentioned,
                ))
        except Exception as e:
            logger.debug(f"RSS parse error ({source}): {e}")
        return items

    # ── Supplemental: free-crypto-news API ────────────────────
    # https://github.com/nirholas/free-crypto-news
    # No API key, 200+ sources, AI-enriched, 1-minute freshness.
    # Falls back gracefully if the API is unavailable.

    _FREE_CRYPTO_NEWS_URL = "https://cryptocurrency.cv/api/news"

    async def _fetch_free_crypto_news(self, session: aiohttp.ClientSession) -> List[NewsItem]:
        """Fetch recent headlines from the free-crypto-news aggregator."""
        items = []
        try:
            async with session.get(
                self._FREE_CRYPTO_NEWS_URL,
                params={"limit": "30", "lang": "en"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return items
                data = await resp.json()

            articles = data.get("articles") or data.get("items") or []
            if isinstance(data, list):
                articles = data

            for article in articles[:30]:
                title = (article.get("title") or "").strip()
                if not title:
                    continue

                source = article.get("source") or "FreeCryptoNews"
                url = article.get("link") or article.get("url") or ""

                # Parse publication time
                pub_time = time.time()
                raw_date = article.get("pubDate") or article.get("published") or ""
                if raw_date:
                    try:
                        from email.utils import parsedate_to_datetime
                        pub_time = parsedate_to_datetime(raw_date).timestamp()
                    except Exception:
                        try:
                            from datetime import datetime
                            pub_time = datetime.fromisoformat(
                                str(raw_date).replace("Z", "+00:00")
                            ).timestamp()
                        except Exception:
                            pass

                # Only include items from last 6 hours
                if time.time() - pub_time > 21600:
                    continue

                # Detect coin mentions
                text_lower = title.lower()
                mentioned = []
                for ticker, aliases in _COIN_ALIASES.items():
                    if any(alias in text_lower for alias in aliases):
                        mentioned.append(ticker)

                items.append(NewsItem(
                    title=title,
                    source=f"FCN:{source}" if source != "FreeCryptoNews" else source,
                    url=url,
                    published_at=pub_time,
                    coins_mentioned=mentioned,
                ))
        except asyncio.TimeoutError:
            logger.debug("free-crypto-news API timeout (8s)")
        except Exception as e:
            logger.debug(f"free-crypto-news parse error: {e}")
        return items

    # ── DeFiLlama TVL ─────────────────────────────────────────

    async def _fetch_tvl(self):
        """Fetch protocol TVL changes from DeFiLlama (free, no key)."""
        if time.time() - self._tvl_last_fetch < self._tvl_ttl:
            return
        session = await self._get_session()
        try:
            async with session.get("https://api.llama.fi/protocols") as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            changes = []
            for p in data[:100]:  # Top 100 protocols by TVL
                tvl = float(p.get("tvl") or 0)
                change_1h = float(p.get("change_1h") or 0)
                change_24h = float(p.get("change_1d") or 0)

                # Only track meaningful TVL changes
                if tvl < 1_000_000 or (abs(change_1h) < 2 and abs(change_24h) < 5):
                    continue

                # Map protocol to token symbols
                symbol = (p.get("symbol") or "").upper()
                related = [symbol] if symbol else []

                changes.append(TVLChange(
                    protocol=p.get("name", ""),
                    chain=p.get("chain", ""),
                    tvl_usd=tvl,
                    change_1h_pct=change_1h,
                    change_24h_pct=change_24h,
                    related_tokens=related,
                ))

            self._tvl_cache = sorted(changes, key=lambda x: abs(x.change_24h_pct), reverse=True)
            self._tvl_last_fetch = time.time()
            logger.debug(f"📊 TVL cache: {len(self._tvl_cache)} protocols with notable changes")

        except Exception as e:
            logger.debug(f"DeFiLlama TVL fetch error: {e}")

    # ── CoinPaprika Events ────────────────────────────────────

    async def _fetch_events(self):
        """
        Fetch upcoming events from CoinPaprika (free, no key).
        Used to populate no-trade zones for token unlock events.
        """
        if time.time() - self._events_last_fetch < self._events_ttl:
            return
        session = await self._get_session()
        try:
            # Get top coins list first
            async with session.get("https://api.coinpaprika.com/v1/coins?limit=200") as resp:
                if resp.status != 200:
                    return
                coins = await resp.json()

            events_by_symbol = defaultdict(list)
            # Fetch events for top 20 coins to stay within free tier
            for coin in coins[:20]:
                coin_id = coin.get("id")
                symbol = (coin.get("symbol") or "").upper()
                if not coin_id:
                    continue
                try:
                    async with session.get(
                        f"https://api.coinpaprika.com/v1/coins/{coin_id}/events"
                    ) as resp:
                        if resp.status == 200:
                            coin_events = await resp.json()
                            # Only upcoming events in next 7 days
                            now = time.time()
                            for ev in (coin_events or [])[:5]:
                                ev_date = ev.get("date") or ""
                                try:
                                    from datetime import datetime
                                    ev_ts = datetime.fromisoformat(ev_date.replace("Z", "+00:00")).timestamp()
                                    if now < ev_ts < now + 604800:  # Next 7 days
                                        events_by_symbol[symbol].append({
                                            "name": ev.get("name", ""),
                                            "description": ev.get("description", "")[:100],
                                            "date": ev_date,
                                            "ts": ev_ts,
                                        })
                                except Exception:
                                    pass
                except Exception:
                    pass
                await asyncio.sleep(0.2)  # Gentle rate limiting

            self._events_cache = dict(events_by_symbol)
            self._events_last_fetch = time.time()
            total = sum(len(v) for v in self._events_cache.values())
            logger.debug(f"📅 Events cache: {total} upcoming events for {len(self._events_cache)} coins")

        except Exception as e:
            logger.debug(f"CoinPaprika events fetch error: {e}")

    # ── Public Query API ──────────────────────────────────────

    def get_news_for_symbol(self, symbol: str, max_age_mins: int = 60) -> List[dict]:
        """
        Get recent news headlines relevant to a symbol.
        Returns list of {title, source, published_at} dicts.
        """
        cutoff = time.time() - (max_age_mins * 60)
        result = []
        symbol_upper = symbol.replace("/USDT", "").replace("/BUSD", "").upper()

        for item in self._news_cache:
            if item.published_at < cutoff:
                continue
            # Include if: coin explicitly mentioned OR headline has the token name
            if (symbol_upper in item.coins_mentioned or
                    symbol_upper.lower() in item.title.lower()):
                result.append({
                    "title": item.title,
                    "source": item.source,
                    "published_at": item.published_at,
                    "url": item.url,
                })

        return sorted(result, key=lambda x: x["published_at"], reverse=True)[:10]

    def get_all_recent_news(self, max_age_mins: int = 30) -> List[dict]:
        """Get all recent headlines for market state context."""
        cutoff = time.time() - (max_age_mins * 60)
        return [
            {"title": n.title, "source": n.source}
            for n in self._news_cache
            if n.published_at > cutoff
        ][:15]

    def get_tvl_change_for_symbol(self, symbol: str) -> Optional[float]:
        """
        Get 24h TVL change % for a symbol.
        Returns None if no TVL data found.
        """
        symbol_upper = symbol.replace("/USDT", "").upper()
        for change in self._tvl_cache:
            if symbol_upper in change.related_tokens:
                return change.change_24h_pct
        return None

    def get_upcoming_events(self, symbol: str) -> List[dict]:
        """Get upcoming events for a symbol (from CoinPaprika)."""
        symbol_upper = symbol.replace("/USDT", "").upper()
        return self._events_cache.get(symbol_upper, [])

    def has_dangerous_event_soon(self, symbol: str, within_hours: int = 48) -> bool:
        """
        Returns True if there's a major upcoming event (e.g. token unlock)
        within the next N hours that should suppress trading.
        """
        symbol_upper = symbol.replace("/USDT", "").upper()
        events = self._events_cache.get(symbol_upper, [])
        now = time.time()
        horizon = now + (within_hours * 3600)
        dangerous_keywords = ["unlock", "vesting", "release", "listing", "delisting", "halving"]
        for ev in events:
            ev_ts = ev.get("ts", 0)
            if now < ev_ts < horizon:
                name_lower = ev.get("name", "").lower()
                if any(kw in name_lower for kw in dangerous_keywords):
                    return True
        return False

    # ── Sentiment helpers (replaces CryptoPanic) ──────────────

    # Simple keyword-based sentiment applied to RSS headlines.
    # Not as good as vote-based, but real-time and free.
    _BULLISH_WORDS = {
        "surge", "surges", "surging", "rally", "rallies", "rallying",
        "breakout", "breaks out", "broke out", "all-time high", "ath",
        "bullish", "adoption", "partnership", "launch", "mainnet",
        "upgrade", "institutional", "etf approved", "etf approval",
        "buy", "accumulate", "outperform", "record high", "milestone",
        "bull run", "bullrun", "recovery", "recovering", "rebound",
        "rebounds", "bounces", "bounce", "inflow", "inflows",
        "accumulation", "staking", "listing", "positive",
        "growth", "growing", "expands", "expansion", "wins",
        "approved", "approval", "regulation clarity",
        "bitcoin reserve", "strategic reserve", "hodl",
    }
    _BEARISH_WORDS = {
        "crash", "crashes", "crashing", "collapse", "collapses",
        "dump", "dumps", "dumping", "bearish", "hack", "hacked",
        "exploit", "exploited", "rug pull", "rug", "sec", "lawsuit",
        "ban", "banned", "banning", "regulatory crackdown", "warning",
        "sell", "selloff", "sell-off", "liquidat", "bankruptcy",
        "insolvency", "insolvent", "fraud", "ponzi", "scam",
        "delisting", "suspended", "suspension", "fud", "fear",
        "outflow", "outflows", "exit", "warning", "concern",
        "investigation", "probe", "charges", "indicted",
        "security breach", "vulnerability", "exploit", "stolen",
        "falls", "fell", "plunges", "plunge", "drops", "slump",
        "slumps", "loses", "loss", "losing", "bearish",
        "shutdown", "shut down", "halted", "halt",
        # ── Geopolitical / macro crisis keywords ──────────────────
        # A headline like "Iran-US deal fails" or "JD Vance: no deal"
        # must register as bearish — geopolitical escalation hits BTC.
        "war", "conflict", "escalat", "military", "strike",
        "missile", "nuclear", "invasion", "invade", "sanction",
        "sanctions", "tariff", "tariffs", "trade war",
        "geopolit", "tension", "diplomatic", "retaliat",
        "deal fail", "deal collapse", "no deal", "not make a deal",
        "could not reach", "could not agree", "failed to reach",
        "talks fail", "talks collapse", "talks break down",
        "negotiation fail", "breakdown",
        "recession", "rate hike", "hawkish", "inflation surge",
        "default", "debt crisis", "contagion", "panic",
    }
    _BREAKING_WORDS = {
        "breaking", "urgent", "emergency", "crash", "crashed", "hack",
        "hacked", "exploit", "exploited", "rug pull", "bankrupt", "insolvent",
        "sec charges", "sec sues", "arrested", "seized", "shutdown", "shut down",
        "halted", "suspended", "delisted", "delisting", "all-time high", "ath",
        "etf approved", "etf approval", "strategic reserve",
    }

    def _score_headline(self, title: str) -> int:
        """Returns +1 (bullish), -1 (bearish), or 0 (neutral) for a headline."""
        low = title.lower()
        bull = sum(1 for w in self._BULLISH_WORDS if w in low)
        bear = sum(1 for w in self._BEARISH_WORDS if w in low)
        if bull > bear:
            return 1
        if bear > bull:
            return -1
        return 0

    def _urgency_level(self, title: str) -> str:
        """Return 'HIGH' for breaking/urgent news, else 'NORMAL'."""
        low = title.lower()
        bull_hits = sum(1 for w in self._BULLISH_WORDS if w in low)
        bear_hits = sum(1 for w in self._BEARISH_WORDS if w in low)
        if bull_hits >= 3 or bear_hits >= 3:
            return "HIGH"
        if any(w in low for w in self._BREAKING_WORDS):
            return "HIGH"
        return "NORMAL"

    # ── Feature 2: Clickbait Detection ──────────────────────────
    # 15 regex patterns that detect clickbait signals.
    _CLICKBAIT_PATTERNS = [
        re.compile(r"you won'?t believe", re.IGNORECASE),
        re.compile(r"shocking", re.IGNORECASE),
        re.compile(r"this is (huge|massive|insane|crazy)", re.IGNORECASE),
        re.compile(r"!!!+"),
        re.compile(r"\?\?\?+"),
        re.compile(r"(BREAKING|URGENT|ALERT).*!!!", re.IGNORECASE),
        re.compile(r"secret.*(revealed|exposed|trick)", re.IGNORECASE),
        re.compile(r"(moon|lambo|1000x|100x|10000%)", re.IGNORECASE),
        re.compile(r"experts? (hate|don'?t want)", re.IGNORECASE),
        re.compile(r"(last chance|act now|don'?t miss)", re.IGNORECASE),
        re.compile(r"guaranteed|free money|risk.?free", re.IGNORECASE),
        re.compile(r"number \d+ will (shock|surprise|blow)", re.IGNORECASE),
        re.compile(r"what happens next", re.IGNORECASE),
        re.compile(r"(crypto|bitcoin) (dead|dying|finished|over)", re.IGNORECASE),
        re.compile(r"one simple trick", re.IGNORECASE),
        # Additional patterns for real-world clickbait headlines
        re.compile(r"slides? \d{2,3}%", re.IGNORECASE),         # "slides 90%"
        re.compile(r"(explodes?|surges?|soars?|skyrockets?) \d{2,3}%", re.IGNORECASE),  # "volume explodes 148%"
        re.compile(r"(crash|plunge|tank|dump|spike)s?!+", re.IGNORECASE),  # "crashes!" with exclamation
        re.compile(r"\b(huge|massive|insane|crazy|unbelievable) (pump|dump|crash|rally|surge)", re.IGNORECASE),
        re.compile(r"\bbut why\b", re.IGNORECASE),              # Rhetorical "But why are..."
        re.compile(r"here'?s (what|why|how)", re.IGNORECASE),   # "Here's what you need to know"
        re.compile(r"(can|will|could) .{0,20}\d{3,}[%x]", re.IGNORECASE),  # "can it 500x?"
        re.compile(r"(millionaire|rich|wealthy) (overnight|fast|quick)", re.IGNORECASE),
        re.compile(r"\b(rekt|ngmi|wagmi|wen)\b", re.IGNORECASE),  # Degen slang in "news"
    ]

    def _score_clickbait(self, title: str) -> Tuple[float, List[str]]:
        """
        Returns (clickbait_score, matched_patterns) for a headline.
        Score: 0.0 (clean) to 1.0+ (definite clickbait).

        Each pattern match adds 0.3 to the score.
        ALL-CAPS ratio > 30% adds 0.4.
        Excessive punctuation (!!! or ???) adds 0.2.
        """
        score = 0.0
        matched = []

        for pattern in self._CLICKBAIT_PATTERNS:
            if pattern.search(title):
                score += 0.3
                matched.append(pattern.pattern)

        # ALL-CAPS ratio check
        alpha_chars = [c for c in title if c.isalpha()]
        if alpha_chars:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio > 0.30:
                score += 0.4
                matched.append(f"ALL_CAPS_RATIO={caps_ratio:.0%}")

        return score, matched

    # ── Feature 3: Title Deduplication ──────────────────────────

    @staticmethod
    def _jaccard_ngram_similarity(a: str, b: str, n: int = 3) -> float:
        """
        Compute Jaccard similarity on character n-grams of two strings.
        Returns 0.0–1.0.
        """
        a_low = a.lower().strip()
        b_low = b.lower().strip()
        if not a_low or not b_low:
            return 0.0
        a_ngrams = {a_low[i:i + n] for i in range(len(a_low) - n + 1)}
        b_ngrams = {b_low[i:i + n] for i in range(len(b_low) - n + 1)}
        if not a_ngrams or not b_ngrams:
            return 0.0
        intersection = a_ngrams & b_ngrams
        union = a_ngrams | b_ngrams
        return len(intersection) / len(union)

    def _deduplicate_title(self, title: str, source: str) -> Optional[Dict]:
        """
        Check if a title is a duplicate of an existing cluster.
        Returns the cluster dict if it's a duplicate, None if it's new.

        When duplicate found, cluster source_count is incremented.
        """
        threshold = NewsIntelligence.TITLE_DEDUP_SIMILARITY_THRESHOLD
        for cluster in self._dedup_clusters:
            for existing_title in cluster["titles"]:
                sim = self._jaccard_ngram_similarity(title, existing_title)
                if sim >= threshold:
                    title_entities = self._extract_dedup_entities(title)
                    existing_entities = self._extract_dedup_entities(existing_title)
                    if (
                        title_entities
                        and existing_entities
                        and title_entities.isdisjoint(existing_entities)
                    ):
                        logger.debug(
                            "📰 Dedup skip: '%s…' kept separate from '%s…' "
                            "(sim=%.2f, entity_mismatch=%s vs %s)",
                            title[:60],
                            existing_title[:60],
                            sim,
                            sorted(title_entities),
                            sorted(existing_entities),
                        )
                        continue
                    cluster["titles"].add(title)
                    cluster["source_count"] += 1
                    cluster["sources"].add(source)
                    logger.debug(
                        f"📰 Dedup: '{title[:60]}…' clustered with "
                        f"'{existing_title[:60]}…' (sim={sim:.2f}, "
                        f"sources={cluster['source_count']})"
                    )
                    return cluster
        # New unique title — create new cluster
        cluster = {
            "titles": {title},
            "sources": {source},
            "source_count": 1,
            "first_seen": time.time(),
            "representative": title,
        }
        self._dedup_clusters.append(cluster)
        # AUDIT FIX: dedup clusters previously pruned after 2 h while the
        # news cache itself keeps items for 6 h (see _merge cutoff=21600).
        # A headline dropped from the cluster store would reappear from
        # the still-live cache and be treated as a unique new article,
        # double-counting it in sentiment weighting.  Align the prune
        # window with the news-cache TTL so the two stores agree.
        cutoff = time.time() - 21600
        self._dedup_clusters = [
            c for c in self._dedup_clusters if c["first_seen"] > cutoff
        ]
        return None

    @staticmethod
    def _extract_dedup_entities(title: str) -> set:
        """Extract salient entity tokens to avoid over-merging similar templates."""
        tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\.-]+", title)
        }
        return tokens & _DEDUP_ENTITY_TOKENS

    # ── Feature 4: Headline Evolution Tracking ──────────────────

    def _track_headline_evolution(self, url: str, title: str):
        """
        Track headline changes by URL. If title changed significantly,
        downgrade confidence of signals from the original headline.
        """
        title_hash = hashlib.md5(title.encode()).hexdigest()

        if url in self._headline_tracker:
            old_hash, old_title, old_ts = self._headline_tracker[url]
            if old_hash != title_hash:
                sim = self._jaccard_ngram_similarity(old_title, title)
                if sim < NewsIntelligence.HEADLINE_EVOLUTION_SIMILARITY_THRESHOLD:
                    penalty = NewsIntelligence.HEADLINE_EVOLUTION_CONFIDENCE_PENALTY
                    self._evolved_titles[old_title] = penalty
                    logger.warning(
                        f"📰 Headline evolved: '{old_title[:60]}…' → "
                        f"'{title[:60]}…' (sim={sim:.2f}, penalty={penalty})"
                    )
                    if ff.is_shadow("HEADLINE_EVOLUTION"):
                        shadow_log("HEADLINE_EVOLUTION", {
                            "url": url,
                            "old_title": old_title,
                            "new_title": title,
                            "similarity": sim,
                            "penalty": penalty,
                        })
                    from config.fcn_logger import fcn_log
                    _mode = "shadow" if ff.is_shadow("HEADLINE_EVOLUTION") else "live"
                    fcn_log("HEADLINE_EVOLUTION", f"{_mode} | evolved: '{old_title[:40]}…' → '{title[:40]}…' sim={sim:.2f} penalty={penalty}")
                # Update tracker with new title
                self._headline_tracker[url] = (title_hash, title, time.time())
        else:
            self._headline_tracker[url] = (title_hash, title, time.time())

        # Prune old tracker entries (>3 hours)
        cutoff = time.time() - 10800
        self._headline_tracker = {
            u: v for u, v in self._headline_tracker.items() if v[2] > cutoff
        }

    # ── PRE-WORK 2: Time Decay ──────────────────────────────────

    @staticmethod
    def _is_slow_decay_event(title: str) -> bool:
        """
        Returns True if the headline describes a slow-burn event type
        (regulatory, ETF, macro, scheduled) that should decay more slowly.
        """
        title_lower = title.lower()
        return any(kw in title_lower for kw in NewsIntelligence.SLOW_DECAY_EVENT_KEYWORDS)

    @staticmethod
    def _time_decay_weight(age_minutes: float, title: str = "") -> float:
        """
        Exponential time decay for headline weight.
        Formula: exp(-lambda * age_minutes)

        Event-type-aware: slow-burn events (regulatory, ETF, macro)
        use a gentler lambda (0.03) and longer max age (180 min).
        Standard events use lambda=0.08, max age=90 min.

        Returns 0.0 if headline exceeds the applicable max age.
        """
        if title and NewsScraper._is_slow_decay_event(title):
            max_age = NewsIntelligence.HEADLINE_MAX_AGE_MINUTES_SLOW
            lam = NewsIntelligence.HEADLINE_DECAY_LAMBDA_SLOW
        else:
            max_age = NewsIntelligence.HEADLINE_MAX_AGE_MINUTES
            lam = NewsIntelligence.HEADLINE_DECAY_LAMBDA

        if age_minutes > max_age:
            return 0.0
        return math.exp(-lam * age_minutes)

    def get_symbol_sentiment_score(self, symbol: str, max_age_mins: int = 60) -> Optional[float]:
        """
        Returns a 0-100 sentiment score for a symbol based on recent headlines.
        50 = neutral, >50 = bullish, <50 = bearish.
        Returns None if no relevant news found.

        Applies (when feature flags are live):
        - Time decay: recent headlines weighted more than old ones
        - Source credibility: per-source reliability multiplier
        - Clickbait filter: downweights clickbait headlines by 50%
        - Title dedup: similar titles counted as one signal
        - Headline evolution: downgrade confidence on changed titles
        """
        from analyzers.source_credibility import apply_source_credibility

        news = self.get_news_for_symbol(symbol, max_age_mins=max_age_mins)
        if not news:
            return None

        now = time.time()
        weighted_scores: List[float] = []
        total_weight = 0.0
        seen_clusters: set = set()  # Track dedup cluster representatives

        for n in news:
            title = n["title"]
            source = n.get("source", "unknown")
            published_at = n.get("published_at", now)

            raw_score = self._score_headline(title)  # -1, 0, +1
            weight = 1.0

            # PRE-WORK 2: Time decay (event-type-aware)
            if ff.is_active("TIME_DECAY"):
                age_minutes = (now - published_at) / 60.0
                decay = self._time_decay_weight(age_minutes, title=title)
                if ff.is_enabled("TIME_DECAY"):
                    weight *= decay
                    from config.fcn_logger import fcn_log
                    fcn_log("TIME_DECAY", f"live | '{title[:50]}…' age={age_minutes:.0f}m decay={decay:.3f} slow={self._is_slow_decay_event(title)}")
                else:
                    shadow_log("TIME_DECAY", {
                        "title": title[:80],
                        "age_minutes": round(age_minutes, 1),
                        "decay_factor": round(decay, 3),
                        "is_slow_decay": self._is_slow_decay_event(title),
                        "live_weight": 1.0,
                        "shadow_weight": decay,
                    })
                    from config.fcn_logger import fcn_log
                    fcn_log("TIME_DECAY", f"shadow | '{title[:50]}…' age={age_minutes:.0f}m decay={decay:.3f} slow={self._is_slow_decay_event(title)}")

            # Feature 1: Source credibility
            weight = apply_source_credibility(source, weight)

            # Feature 2: Clickbait filter
            if ff.is_active("CLICKBAIT_FILTER"):
                cb_score, cb_matched = self._score_clickbait(title)
                _cb_mode = "shadow" if ff.is_shadow("CLICKBAIT_FILTER") else "live"
                if cb_score >= NewsIntelligence.CLICKBAIT_SCORE_THRESHOLD:
                    if ff.is_enabled("CLICKBAIT_FILTER"):
                        weight *= NewsIntelligence.CLICKBAIT_WEIGHT_PENALTY
                    logger.debug(
                        f"📰 Clickbait: '{title[:60]}…' score={cb_score:.2f} "
                        f"patterns={cb_matched}"
                    )
                    from config.fcn_logger import fcn_log
                    fcn_log("CLICKBAIT_FILTER", f"{_cb_mode} | '{title[:50]}…' score={cb_score:.2f} patterns={cb_matched}")
                    if ff.is_shadow("CLICKBAIT_FILTER"):
                        shadow_log("CLICKBAIT_FILTER", {
                            "title": title[:80],
                            "clickbait_score": cb_score,
                            "patterns": cb_matched,
                            "live_weight": weight / NewsIntelligence.CLICKBAIT_WEIGHT_PENALTY
                                if ff.is_enabled("CLICKBAIT_FILTER") else weight,
                            "shadow_weight": weight * NewsIntelligence.CLICKBAIT_WEIGHT_PENALTY,
                        })
                else:
                    from config.fcn_logger import fcn_log
                    fcn_log("CLICKBAIT_FILTER", f"{_cb_mode} | '{title[:50]}…' score={cb_score:.2f} — below threshold")

            # Feature 3: Title dedup
            if ff.is_active("TITLE_DEDUP"):
                _td_mode = "shadow" if ff.is_shadow("TITLE_DEDUP") else "live"
                cluster = self._deduplicate_title(title, source)
                if cluster is not None:
                    rep = cluster["representative"]
                    if rep in seen_clusters:
                        # Already counted this cluster — skip
                        if ff.is_enabled("TITLE_DEDUP"):
                            from config.fcn_logger import fcn_log
                            fcn_log("TITLE_DEDUP", f"live | skipped duplicate '{title[:50]}…' cluster='{rep[:50]}…' sources={cluster['source_count']}")
                            continue
                        else:
                            shadow_log("TITLE_DEDUP", {
                                "title": title[:80],
                                "cluster_representative": rep[:80],
                                "source_count": cluster["source_count"],
                                "action": "would_skip",
                            })
                            from config.fcn_logger import fcn_log
                            fcn_log("TITLE_DEDUP", f"shadow | would skip '{title[:50]}…' cluster='{rep[:50]}…' sources={cluster['source_count']}")
                    else:
                        seen_clusters.add(rep)
                        from config.fcn_logger import fcn_log
                        fcn_log("TITLE_DEDUP", f"{_td_mode} | '{title[:50]}…' — unique (new cluster)")
                else:
                    from config.fcn_logger import fcn_log
                    fcn_log("TITLE_DEDUP", f"{_td_mode} | '{title[:50]}…' — unique (no cluster)")

            # Feature 4: Headline evolution penalty
            if ff.is_active("HEADLINE_EVOLUTION"):
                _he_mode = "shadow" if ff.is_shadow("HEADLINE_EVOLUTION") else "live"
                if title in self._evolved_titles:
                    penalty = self._evolved_titles[title]
                    if ff.is_enabled("HEADLINE_EVOLUTION"):
                        weight *= penalty
                        from config.fcn_logger import fcn_log
                        fcn_log("HEADLINE_EVOLUTION", f"live | '{title[:50]}…' penalty={penalty:.3f} weight→{weight:.3f}")
                    else:
                        shadow_log("HEADLINE_EVOLUTION", {
                            "title": title[:80],
                            "live_weight": weight,
                            "shadow_weight": weight * penalty,
                        })
                        from config.fcn_logger import fcn_log
                        fcn_log("HEADLINE_EVOLUTION", f"shadow | '{title[:50]}…' penalty={penalty:.3f} live_w={weight:.3f} shadow_w={weight*penalty:.3f}")
                else:
                    from config.fcn_logger import fcn_log
                    fcn_log("HEADLINE_EVOLUTION", f"{_he_mode} | '{title[:50]}…' — no evolution detected")

            weighted_scores.append(raw_score * weight)
            total_weight += weight

        if not weighted_scores or total_weight == 0:
            return None
        avg = sum(weighted_scores) / total_weight  # -1.0 to +1.0
        return round(50 + avg * 35, 1)  # map to 0-100

    def get_market_sentiment_summary(self) -> dict:
        """
        Returns an overall market sentiment summary from recent headlines.
        Used by the dashboard and Telegram /news command.
        """
        cutoff = time.time() - 3600  # last hour
        recent = [n for n in self._news_cache if n.published_at > cutoff]
        if not recent:
            return {"score": 50, "label": "Neutral", "story_count": 0, "sources": []}

        bull = sum(1 for n in recent if self._score_headline(n.title) > 0)
        bear = sum(1 for n in recent if self._score_headline(n.title) < 0)
        total = len(recent)
        score = round(50 + ((bull - bear) / max(total, 1)) * 35, 1)

        if score >= 65:
            label = "Bullish"
        elif score <= 35:
            label = "Bearish"
        else:
            label = "Neutral"

        sources = list({n.source for n in recent})[:5]
        return {
            "score": score,
            "label": label,
            "story_count": total,
            "bull_count": bull,
            "bear_count": bear,
            "sources": sources,
        }

    def get_all_stories(self, max_age_mins: int = 120, limit: int = 50) -> list:
        """
        Return all recent stories for the dashboard news feed.
        Each item is a dict compatible with the former CryptoPanic format.
        """
        cutoff = time.time() - (max_age_mins * 60)
        results = []
        for item in self._news_cache:
            if item.published_at < cutoff:
                continue
            sentiment = self._score_headline(item.title)
            results.append({
                "title":          item.title,
                "source":         item.source,
                "url":            item.url,
                "published_at":   item.published_at,
                "sentiment_score": 50 + sentiment * 35,   # 15, 50, or 85
                "urgency":        self._urgency_level(item.title),
                "currencies":     item.coins_mentioned,
                "votes_bullish":  1 if sentiment > 0 else 0,
                "votes_bearish":  1 if sentiment < 0 else 0,
            })
        results.sort(key=lambda x: x["published_at"], reverse=True)
        return results[:limit]

    # ── R8-F6: News Impact Velocity Detector ──────────────────────────

    def detect_news_storm(self, window_minutes: int = 15,
                           min_negative_count: int = 3,
                           current_regime: str = "") -> dict:
        """
        Detect coordinated negative news bursts (news storm).

        When 3+ negative news items appear within 15 minutes, it signals
        a coordinated dump narrative. Returns storm state for engine integration.

        FIX (P1-C): News storms now apply regime-aware confidence penalties
        instead of hard-blocking LONGs. GPT feedback: "Sentiment should amplify,
        not hard-block." In BULL_TREND, negative news means LONGs need more
        conviction (bigger penalty) but aren't killed — the trend may absorb
        the news. SHORTs also get a penalty (smaller) since the trend opposes them.
        In BEAR_TREND, negative news AMPLIFIES shorts (small boost) and penalizes
        longs harder. This makes sentiment a direction-aware amplifier.

        Returns:
            {
                'is_storm': bool,        # True if storm detected
                'severity': str,         # 'NONE', 'MODERATE', 'SEVERE'
                'negative_count': int,   # Number of negative items in window
                'total_count': int,      # Total items in window
                'long_penalty': int,     # Confidence penalty for LONGs (always negative)
                'short_penalty': int,    # Confidence penalty for SHORTs (negative or zero)
                'tighten_profitable': bool,  # Whether to tighten profitable stops
            }
        """
        now = time.time()
        window_cutoff = now - (window_minutes * 60)

        # Get recent headlines in the window
        recent = [
            item for item in self._news_cache
            if item.published_at >= window_cutoff
        ]

        # Score each headline
        negative_items = []
        for item in recent:
            score = self._score_headline(item.title)
            if score < 0:
                negative_items.append(item)

        negative_count = len(negative_items)
        total_count = len(recent)

        # Determine storm severity
        if negative_count >= min_negative_count + 3:  # 6+ = severe
            severity = "SEVERE"
            base_long_penalty = -25
            base_short_penalty = -10
        elif negative_count >= min_negative_count:  # 3+ = moderate
            severity = "MODERATE"
            base_long_penalty = -15
            base_short_penalty = -5
        else:
            return {
                'is_storm': False,
                'severity': 'NONE',
                'negative_count': negative_count,
                'total_count': total_count,
                'long_penalty': 0,
                'short_penalty': 0,
                'tighten_profitable': False,
            }

        # FIX (P1-C): Regime-aware penalty scaling.
        # Negative news in BULL_TREND: LONGs get base penalty, SHORTs get reduced penalty
        #   (the bull trend may absorb the news — shorts still fight the trend).
        # Negative news in BEAR_TREND: LONGs get amplified penalty, SHORTs get zero/boost
        #   (news confirms the trend — shorts are amplified, longs are dangerous).
        # CHOPPY/VOLATILE/other: both get base penalty (no direction bias).
        if current_regime == "BEAR_TREND":
            long_penalty = int(base_long_penalty * 1.3)   # Amplified: -19 or -32
            short_penalty = 0                               # News confirms trend — no SHORT penalty
        elif current_regime == "BULL_TREND":
            long_penalty = base_long_penalty                # Standard penalty
            short_penalty = base_short_penalty              # SHORTs also penalized (fighting trend)
        else:
            long_penalty = base_long_penalty
            short_penalty = base_short_penalty

        result = {
            'is_storm': True,
            'severity': severity,
            'negative_count': negative_count,
            'total_count': total_count,
            'long_penalty': long_penalty,
            'short_penalty': short_penalty,
            'tighten_profitable': severity == "SEVERE",
        }

        logger.warning(
            f"📰🔴 NEWS STORM {severity}: {negative_count}/{total_count} negative "
            f"items in {window_minutes}min — LONG penalty {long_penalty}, "
            f"SHORT penalty {short_penalty} (regime={current_regime or 'UNKNOWN'})"
        )

        return result

    def get_news_velocity(self, window_minutes: int = 30) -> dict:
        """
        Measure news publication velocity and sentiment direction.
        Useful for detecting acceleration in narrative shifts.

        Returns:
            {
                'items_per_hour': float,
                'sentiment_direction': float,  # -1.0 to +1.0
                'is_accelerating': bool,       # More news in last 15m vs previous 15m
            }
        """
        now = time.time()
        window = now - (window_minutes * 60)
        half = now - (window_minutes * 30)

        recent_all = [i for i in self._news_cache if i.published_at >= window]
        recent_half = [i for i in self._news_cache if i.published_at >= half]
        older_half = [
            i for i in self._news_cache
            if window <= i.published_at < half
        ]

        items_per_hour = len(recent_all) / max(0.1, window_minutes / 60)

        # Average sentiment
        if recent_all:
            scores = [self._score_headline(i.title) for i in recent_all]
            sentiment_direction = sum(scores) / len(scores)
        else:
            sentiment_direction = 0.0

        # Acceleration: more items in recent half than older half
        is_accelerating = len(recent_half) > len(older_half) * 1.5

        return {
            'items_per_hour': round(items_per_hour, 1),
            'sentiment_direction': round(sentiment_direction, 3),
            'is_accelerating': is_accelerating,
        }


# ── Singleton ─────────────────────────────────────────────────
news_scraper = NewsScraper()
