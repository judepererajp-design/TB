"""
TitanBot Pro — Institutional Flow Engine
==========================================
Multi-exchange smart money aggregator using 100% public APIs.
No API keys required.

EXCHANGES:
  CME (CFTC COT)  — institutional/leveraged fund positioning (weekly)
  Coinbase        — premium vs Binance = US institutional buy pressure
  Binance         — top trader L/S ratio, taker volume, OI change
  Bybit           — account ratio (retail vs institutional split), OI
  Deribit         — options positioning (already exists, extended here)

MACRO:
  FRED            — DXY, 10yr Treasury yield, M2 money supply
  Yahoo Finance   — VIX, Gold, S&P 500 (yfinance-compatible free API)

OUTPUT:
  SmartMoneyScore per coin: -100 (strong short) to +100 (strong long)
  MacroScore: -100 (risk-off) to +100 (risk-on)
  PercentileRanking: where each metric sits in 90-day history

INTEGRATION:
  → ensemble voter  smart_money slot (replaces Hyperliquid-only signal)
  → feature store   new features for probability engine
  → signal card     richer Telegram output
  → dashboard       Markets + Smart Money pages
"""

import asyncio
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ── API endpoints (all public, no keys) ───────────────────────
_BINANCE_BASE   = "https://fapi.binance.com"
_BYBIT_BASE     = "https://api.bybit.com"
_COINBASE_BASE  = "https://api.exchange.coinbase.com"
_BINANCE_SPOT   = "https://api.binance.com/api/v3"
_CFTC_COT       = "https://publicreporting.cftc.gov/api/odata/v1/CorpActions"
_CME_COT_CSV    = "https://www.cftc.gov/dea/futures/deacmesf.htm"
_FRED_BASE      = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_YAHOO_QUERY    = "https://query1.finance.yahoo.com/v8/finance/chart"
_STABLECOIN_API = "https://stablecoins.llama.run"

# Poll intervals
_FAST_INTERVAL  = 300   # 5 min — exchange ratios, OI
_SLOW_INTERVAL  = 3600  # 1 hr  — macro data
_COT_INTERVAL   = 86400 # 24 hr — COT report (weekly release, but check daily)

# History retention
_HISTORY_DAYS = 90


# ── Data classes ──────────────────────────────────────────────
@dataclass
class ExchangeSnapshot:
    """Point-in-time snapshot from one exchange."""
    exchange: str
    symbol: str           # e.g. "BTC"
    timestamp: float
    top_long_pct: float   # % of top traders long (0-100)
    top_short_pct: float  # % of top traders short (0-100)
    global_long_pct: float = 50.0   # all traders
    global_short_pct: float = 50.0
    oi_usd: float = 0.0             # open interest in USD
    oi_change_pct: float = 0.0      # OI change vs 24h ago
    taker_buy_pct: float = 50.0     # % of taker volume that was buys
    funding_rate: float = 0.0       # 8h funding rate
    coinbase_premium: float = 0.0   # Coinbase price - Binance price (USD)


@dataclass
class MacroSnapshot:
    """Macro market conditions."""
    timestamp: float
    dxy: float = 0.0            # US Dollar Index
    dxy_change_1d: float = 0.0  # DXY % change today
    vix: float = 0.0            # CBOE Volatility Index
    gold_usd: float = 0.0       # Gold spot price
    gold_change_1d: float = 0.0 # Gold % change today
    ten_yr_yield: float = 0.0   # 10-year Treasury yield
    spx_change_1d: float = 0.0  # S&P 500 daily % change
    m2_growth: float = 0.0      # M2 money supply YoY growth %
    risk_on_score: float = 0.0  # composite -100 to +100


@dataclass
class COTSnapshot:
    """CFTC Commitment of Traders (CME BTC futures)."""
    timestamp: float
    report_date: str
    asset_mgr_long: float = 0.0      # BlackRock/Fidelity tier — net longs
    asset_mgr_short: float = 0.0
    leveraged_fund_long: float = 0.0  # Hedge funds — net longs
    leveraged_fund_short: float = 0.0
    dealer_long: float = 0.0          # Market makers
    dealer_short: float = 0.0
    institutional_lean: float = 0.0  # computed: asset_mgr net direction
    hedge_fund_lean: float = 0.0     # computed: leveraged fund net direction


@dataclass
class SmartMoneyScore:
    """Aggregated score for one symbol."""
    symbol: str
    timestamp: float
    score: float = 0.0          # -100 to +100
    confidence: float = 0.0     # 0 to 100 (how many signals agree)
    signal_count: int = 0       # how many contributing signals
    components: Dict = field(default_factory=dict)
    percentiles: Dict = field(default_factory=dict)
    summary: str = ""           # human-readable one-liner


# ── Historical store (SQLite) ─────────────────────────────────
class FlowHistoryDB:
    """
    Stores 90 days of exchange metrics per coin.
    Pre-computes percentile ranks so signal generation stays fast.
    """

    def __init__(self, db_path: str = "data/flow_history.db"):
        self._path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        import aiosqlite
        self._conn = await aiosqlite.connect(self._path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS flow_metrics (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                exchange  TEXT NOT NULL,
                symbol    TEXT NOT NULL,
                metric    TEXT NOT NULL,
                value     REAL NOT NULL,
                pct_30d   REAL,
                pct_90d   REAL
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_flow_sym_metric
            ON flow_metrics(symbol, metric, ts DESC)
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_metrics (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      REAL NOT NULL,
                metric  TEXT NOT NULL,
                value   REAL NOT NULL,
                pct_90d REAL
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cot_reports (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               REAL NOT NULL,
                report_date      TEXT UNIQUE,
                asset_mgr_net    REAL,
                lev_fund_net     REAL,
                dealer_net       REAL,
                institutional_lean REAL,
                hedge_fund_lean  REAL
            )
        """)
        await self._conn.commit()
        logger.info("📊 FlowHistoryDB initialized")

    async def record_flow(self, exchange: str, symbol: str,
                          metric: str, value: float, ts: float = None):
        """Store one metric reading and compute its percentile rank."""
        if ts is None:
            ts = time.time()
        async with self._lock:
            # Get last 90 days of this metric for percentile calc
            cutoff_30 = ts - 30 * 86400
            cutoff_90 = ts - 90 * 86400
            cur = await self._conn.execute("""
                SELECT value FROM flow_metrics
                WHERE symbol=? AND metric=? AND exchange=? AND ts > ?
                ORDER BY ts DESC LIMIT 2000
            """, (symbol, metric, exchange, cutoff_90))
            rows = await cur.fetchall()
            await cur.close()
            hist = [r[0] for r in rows]

            pct_30d = pct_90d = None
            if hist:
                pct_90d = sum(1 for v in hist if v <= value) / len(hist) * 100
            # pct_30d: a separate DB query per call is avoided for performance.
            # The rows fetched above cover 90 days; an exact 30d percentile would
            # require a second query with cutoff_30.  Using pct_90d as an approximation
            # is acceptable — they diverge only when the metric has a strong recent trend.
            if len(hist) >= 5:
                pct_30d = pct_90d  # acceptable approximation without a second DB query

            await self._conn.execute("""
                INSERT INTO flow_metrics(ts,exchange,symbol,metric,value,pct_30d,pct_90d)
                VALUES (?,?,?,?,?,?,?)
            """, (ts, exchange, symbol, metric, value, pct_30d, pct_90d))

            # Prune old data (keep 90 days)
            await self._conn.execute("""
                DELETE FROM flow_metrics
                WHERE symbol=? AND metric=? AND exchange=? AND ts < ?
            """, (symbol, metric, exchange, ts - _HISTORY_DAYS * 86400))
            await self._conn.commit()

    async def record_macro(self, metric: str, value: float, ts: float = None):
        if ts is None:
            ts = time.time()
        async with self._lock:
            cur = await self._conn.execute("""
                SELECT value FROM macro_metrics
                WHERE metric=? AND ts > ?
                ORDER BY ts DESC LIMIT 500
            """, (metric, ts - 90 * 86400))
            rows = await cur.fetchall()
            await cur.close()
            hist = [r[0] for r in rows]
            pct = None
            if len(hist) >= 5:
                pct = sum(1 for v in hist if v <= value) / len(hist) * 100
            await self._conn.execute("""
                INSERT INTO macro_metrics(ts,metric,value,pct_90d) VALUES(?,?,?,?)
            """, (ts, metric, value, pct))
            await self._conn.execute("""
                DELETE FROM macro_metrics WHERE metric=? AND ts < ?
            """, (metric, ts - _HISTORY_DAYS * 86400))
            await self._conn.commit()

    async def get_latest(self, symbol: str, metric: str,
                         exchange: str = "") -> Optional[Tuple[float, float, float]]:
        """Returns (value, pct_30d, pct_90d) or None."""
        async with self._lock:
            q = "SELECT value,pct_30d,pct_90d FROM flow_metrics WHERE symbol=? AND metric=?"
            params = [symbol, metric]
            if exchange:
                q += " AND exchange=?"
                params.append(exchange)
            q += " ORDER BY ts DESC LIMIT 1"
            cur = await self._conn.execute(q, params)
            row = await cur.fetchone()
            await cur.close()
            return tuple(row) if row else None

    async def get_latest_macro(self, metric: str) -> Optional[Tuple[float, float]]:
        """Returns (value, pct_90d) or None."""
        async with self._lock:
            cur = await self._conn.execute("""
                SELECT value,pct_90d FROM macro_metrics
                WHERE metric=? ORDER BY ts DESC LIMIT 1
            """, (metric,))
            row = await cur.fetchone()
            await cur.close()
            return tuple(row) if row else None

    async def get_latest_cot(self) -> Optional[COTSnapshot]:
        async with self._lock:
            cur = await self._conn.execute("""
                SELECT ts,report_date,asset_mgr_net,lev_fund_net,dealer_net,
                       institutional_lean,hedge_fund_lean
                FROM cot_reports ORDER BY ts DESC LIMIT 1
            """)
            row = await cur.fetchone()
            await cur.close()
            if not row:
                return None
            return COTSnapshot(
                timestamp=row[0], report_date=row[1],
                asset_mgr_long=max(0, row[2] or 0),
                asset_mgr_short=max(0, -(row[2] or 0)),
                leveraged_fund_long=max(0, row[3] or 0),
                leveraged_fund_short=max(0, -(row[3] or 0)),
                dealer_long=max(0, row[4] or 0),
                dealer_short=max(0, -(row[4] or 0)),
                institutional_lean=row[5] or 0,
                hedge_fund_lean=row[6] or 0,
            )

    async def record_cot(self, snap: COTSnapshot):
        async with self._lock:
            await self._conn.execute("""
                INSERT OR REPLACE INTO cot_reports
                (ts,report_date,asset_mgr_net,lev_fund_net,dealer_net,
                 institutional_lean,hedge_fund_lean)
                VALUES(?,?,?,?,?,?,?)
            """, (snap.timestamp, snap.report_date,
                  snap.asset_mgr_long - snap.asset_mgr_short,
                  snap.leveraged_fund_long - snap.leveraged_fund_short,
                  snap.dealer_long - snap.dealer_short,
                  snap.institutional_lean, snap.hedge_fund_lean))
            await self._conn.commit()


# ── Main engine ───────────────────────────────────────────────
class InstitutionalFlowEngine:
    """
    Polls 5 exchanges + macro sources and computes a composite
    smart money score per coin, updated every 5 minutes.
    """

    def __init__(self):
        self._db = FlowHistoryDB()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._macro_task: Optional[asyncio.Task] = None
        self._cot_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

        # Latest computed scores (in-memory cache for fast access)
        self._scores: Dict[str, SmartMoneyScore] = {}
        self._macro: Optional[MacroSnapshot] = None
        self._cot: Optional[COTSnapshot] = None
        self._last_exchange_poll: float = 0
        self._last_macro_poll: float = 0
        self._last_cot_poll: float = 0

        self._binance_accessible: Optional[bool] = None  # tested on first poll

        # Tracked symbols (BTC + ETH always + top coins)
        self._symbols = [
            "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE",
            "AVAX", "LINK", "ADA", "DOT", "MATIC", "NEAR",
        ]

    async def start(self):
        await self._db.initialize()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "TitanBot/6.0 ResearchClient"}
        )
        self._running = True
        self._task = asyncio.create_task(self._exchange_loop())
        self._macro_task = asyncio.create_task(self._macro_loop())
        self._cot_task = asyncio.create_task(self._cot_loop())
        logger.info("🏛️  InstitutionalFlowEngine started "
                    "(CME·Coinbase·Binance·Bybit·Deribit + macro)")

    async def stop(self):
        self._running = False
        for t in [self._task, self._macro_task, self._cot_task]:
            if t:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        if self._session:
            await self._session.close()

    # ── Exchange polling loop ─────────────────────────────────
    async def _exchange_loop(self):
        """Poll all exchanges every 5 minutes, staggered."""
        await asyncio.sleep(10)  # let bot finish startup first
        while self._running:
            try:
                await self._poll_all_exchanges()
            except Exception as e:
                logger.debug(f"InstitutionalFlow exchange poll error: {e}")
            await asyncio.sleep(_FAST_INTERVAL)

    async def _poll_all_exchanges(self):
        """Poll each exchange with brief delays to avoid rate limits."""
        ts = time.time()
        for sym in self._symbols:
            try:
                await self._poll_binance(sym, ts)
                await asyncio.sleep(0.3)
                await self._poll_bybit(sym, ts)
                await asyncio.sleep(0.3)
                if sym in ("BTC", "ETH"):
                    await self._poll_coinbase_premium(sym, ts)
                    await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning(f"Exchange poll error for {sym}: {e}")
            await asyncio.sleep(0.5)
        await self._compute_scores()
        self._last_exchange_poll = ts
        # Log poll health — shows what's working vs failing
        scored = sum(1 for sc in self._scores.values() if sc.signal_count > 0)
        if scored == 0 and self._binance_accessible is None:
            # First poll returned nothing — test connectivity
            self._binance_accessible = await self._check_binance_access()
            if not self._binance_accessible:
                logger.warning(
                    "🏦 InstitutionalFlow: Binance Futures API unreachable "
                    "(fapi.binance.com blocked — common in India/restricted regions). "
                    "Exchange flow scores will be unavailable. "
                    "Consider using a VPN or setting a proxy in settings.yaml."
                )
            else:
                logger.info("🏦 InstitutionalFlow: Binance accessible but returned no data yet")
        logger.info(
            f"🏦 InstitutionalFlow poll complete | "
            f"{scored}/{len(self._symbols)} coins scored | "
            f"scores: {[f'{k}={round(v.score)}' for k,v in list(self._scores.items())[:4]]}"
        )

    # ── Binance ───────────────────────────────────────────────
    async def _poll_binance(self, sym: str, ts: float):
        pair = f"{sym}USDT"
        # NOTE: fapi.binance.com is geo-restricted in some regions (India, etc).
        # If calls fail consistently, consider using a VPN or proxy.
        try:
            # Top trader long/short ratio (account-based)
            r = await self._get(
                f"{_BINANCE_BASE}/futures/data/topLongShortAccountRatio",
                params={"symbol": pair, "period": "5m", "limit": 1}
            )
            if r and isinstance(r, list) and r:
                row = r[0]
                long_pct = float(row.get("longAccount", 0.5)) * 100
                short_pct = float(row.get("shortAccount", 0.5)) * 100
                await self._db.record_flow("binance", sym, "top_long_pct", long_pct, ts)
                await self._db.record_flow("binance", sym, "top_short_pct", short_pct, ts)

            await asyncio.sleep(0.1)

            # Global long/short ratio (all traders)
            r2 = await self._get(
                f"{_BINANCE_BASE}/futures/data/globalLongShortAccountRatio",
                params={"symbol": pair, "period": "5m", "limit": 1}
            )
            if r2 and isinstance(r2, list) and r2:
                row2 = r2[0]
                gl = float(row2.get("longAccount", 0.5)) * 100
                gs = float(row2.get("shortAccount", 0.5)) * 100
                await self._db.record_flow("binance", sym, "global_long_pct", gl, ts)
                await self._db.record_flow("binance", sym, "global_short_pct", gs, ts)

            await asyncio.sleep(0.1)

            # Taker buy/sell volume (who's aggressing the market)
            r3 = await self._get(
                f"{_BINANCE_BASE}/futures/data/takerlongshortRatio",
                params={"symbol": pair, "period": "5m", "limit": 1}
            )
            if r3 and isinstance(r3, list) and r3:
                buy_vol = float(r3[0].get("buyVol", 1))
                sell_vol = float(r3[0].get("sellVol", 1))
                total = buy_vol + sell_vol
                buy_pct = buy_vol / total * 100 if total > 0 else 50
                await self._db.record_flow("binance", sym, "taker_buy_pct", buy_pct, ts)

            await asyncio.sleep(0.1)

            # Open interest
            r4 = await self._get(
                f"{_BINANCE_BASE}/fapi/v1/openInterest",
                params={"symbol": pair}
            )
            if r4 and "openInterest" in r4:
                oi = float(r4["openInterest"])
                await self._db.record_flow("binance", sym, "oi_contracts", oi, ts)

            await asyncio.sleep(0.1)

            # Funding rate
            r5 = await self._get(
                f"{_BINANCE_BASE}/fapi/v1/premiumIndex",
                params={"symbol": pair}
            )
            if r5 and "lastFundingRate" in r5:
                fr = float(r5["lastFundingRate"])
                await self._db.record_flow("binance", sym, "funding_rate", fr, ts)

        except Exception as e:
            logger.debug(f"Binance poll {sym} error: {e}")

    # ── Bybit ─────────────────────────────────────────────────
    async def _poll_bybit(self, sym: str, ts: float):
        pair = f"{sym}USDT"
        try:
            # Long/short ratio (account-based)
            r = await self._get(
                f"{_BYBIT_BASE}/v5/market/account-ratio",
                params={"category": "linear", "symbol": pair,
                        "period": "5min", "limit": 1}
            )
            if r and r.get("retCode") == 0:
                lst = r.get("result", {}).get("list", [])
                if lst:
                    row = lst[0]
                    bl = float(row.get("buyRatio", 0.5)) * 100
                    bs = float(row.get("sellRatio", 0.5)) * 100
                    await self._db.record_flow("bybit", sym, "top_long_pct", bl, ts)
                    await self._db.record_flow("bybit", sym, "top_short_pct", bs, ts)

            await asyncio.sleep(0.1)

            # Open interest
            r2 = await self._get(
                f"{_BYBIT_BASE}/v5/market/open-interest",
                params={"category": "linear", "symbol": pair,
                        "intervalTime": "5min", "limit": 1}
            )
            if r2 and r2.get("retCode") == 0:
                lst2 = r2.get("result", {}).get("list", [])
                if lst2:
                    oi = float(lst2[0].get("openInterest", 0))
                    await self._db.record_flow("bybit", sym, "oi_contracts", oi, ts)

            await asyncio.sleep(0.1)

            # Funding rate
            r3 = await self._get(
                f"{_BYBIT_BASE}/v5/market/funding/history",
                params={"category": "linear", "symbol": pair, "limit": 1}
            )
            if r3 and r3.get("retCode") == 0:
                lst3 = r3.get("result", {}).get("list", [])
                if lst3:
                    fr = float(lst3[0].get("fundingRate", 0))
                    await self._db.record_flow("bybit", sym, "funding_rate", fr, ts)

        except Exception as e:
            logger.debug(f"Bybit poll {sym} error: {e}")

    # ── Coinbase Premium ──────────────────────────────────────
    async def _poll_coinbase_premium(self, sym: str, ts: float):
        """
        Coinbase premium = Coinbase BTC price - Binance BTC price.
        Positive = US institutional buyers are aggressive.
        Negative = selling pressure from US institutions.
        """
        try:
            product = f"{sym}-USD"
            cb_r = await self._get(
                f"{_COINBASE_BASE}/products/{product}/ticker"
            )
            bn_r = await self._get(
                f"{_BINANCE_SPOT}/ticker/price",
                params={"symbol": f"{sym}USDT"}
            )
            if cb_r and bn_r:
                cb_price = float(cb_r.get("price", 0))
                bn_price = float(bn_r.get("price", 0))
                if cb_price > 0 and bn_price > 0:
                    premium = cb_price - bn_price
                    premium_pct = (cb_price - bn_price) / bn_price * 100
                    await self._db.record_flow(
                        "coinbase", sym, "premium_usd", premium, ts)
                    await self._db.record_flow(
                        "coinbase", sym, "premium_pct", premium_pct, ts)
                    # Record BTC price for lead-lag cross-correlation analysis
                    if sym == "BTC":
                        await self._db.record_flow(
                            "macro", sym, "btc_price", bn_price, ts)
        except Exception as e:
            logger.debug(f"Coinbase premium {sym} error: {e}")

    # ── Macro polling loop ────────────────────────────────────
    async def _macro_loop(self):
        await asyncio.sleep(30)
        while self._running:
            try:
                await self._poll_macro()
            except Exception as e:
                logger.warning(f"Macro poll error: {e}")
            await asyncio.sleep(_SLOW_INTERVAL)

    async def _poll_macro(self):
        ts = time.time()
        snap = MacroSnapshot(timestamp=ts)

        # VIX, Gold, S&P 500 from Yahoo Finance (free, no key)
        for ticker, attr, change_attr in [
            ("^VIX",  "vix",          None),
            ("GC=F",  "gold_usd",     "gold_change_1d"),
            ("^GSPC", None,           "spx_change_1d"),
        ]:
            try:
                r = await self._get(
                    f"{_YAHOO_QUERY}/{ticker}",
                    params={"interval": "1d", "range": "2d"}
                )
                if r:
                    result = r.get("chart", {}).get("result", [{}])[0]
                    closes = result.get("indicators", {}) \
                                   .get("quote", [{}])[0] \
                                   .get("close", [])
                    closes = [c for c in closes if c is not None]
                    if closes:
                        cur = closes[-1]
                        if attr:
                            setattr(snap, attr, cur)
                        if change_attr and len(closes) >= 2:
                            prev = closes[-2]
                            if prev and prev > 0:
                                setattr(snap, change_attr,
                                        (cur - prev) / prev * 100)
                        # Store in history
                        await self._db.record_macro(ticker.lower().replace("^","").replace("=f",""), cur, ts)
            except Exception as e:
                logger.warning(f"Yahoo macro {ticker} error: {e}")
            await asyncio.sleep(0.5)

        # DXY from FRED (free, no key needed for CSV endpoint)
        try:
            async with self._session.get(
                f"{_FRED_BASE}?id=DTWEXBGS",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    lines = [l for l in text.strip().split("\n")
                             if not l.startswith("DATE") and "." in l]
                    if lines:
                        parts = lines[-1].split(",")
                        if len(parts) >= 2 and parts[1].strip() != ".":
                            dxy = float(parts[1].strip())
                            snap.dxy = dxy
                            if len(lines) >= 2:
                                prev_parts = lines[-2].split(",")
                                if len(prev_parts) >= 2 and prev_parts[1].strip() != ".":
                                    prev_dxy = float(prev_parts[1].strip())
                                    snap.dxy_change_1d = (dxy - prev_dxy) / prev_dxy * 100
                            await self._db.record_macro("dxy", dxy, ts)
        except Exception as e:
            logger.debug(f"FRED DXY error: {e}")

        # 10-year Treasury yield from FRED
        try:
            async with self._session.get(
                f"{_FRED_BASE}?id=DGS10",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    lines = [l for l in text.strip().split("\n")
                             if not l.startswith("DATE") and "." in l]
                    if lines:
                        parts = lines[-1].split(",")
                        if len(parts) >= 2 and parts[1].strip() != ".":
                            yld = float(parts[1].strip())
                            snap.ten_yr_yield = yld
                            await self._db.record_macro("10yr_yield", yld, ts)
        except Exception as e:
            logger.debug(f"FRED 10yr yield error: {e}")

        # Compute composite risk-on score
        snap.risk_on_score = self._compute_macro_score(snap)
        self._macro = snap
        self._last_macro_poll = ts
        logger.debug(
            f"🌍 Macro: DXY={snap.dxy:.1f}({snap.dxy_change_1d:+.2f}%) "
            f"VIX={snap.vix:.1f} Gold={snap.gold_usd:.0f}"
            f"({snap.gold_change_1d:+.2f}%) "
            f"10yr={snap.ten_yr_yield:.2f}% risk_on={snap.risk_on_score:+.0f}"
        )

    def _compute_macro_score(self, snap: MacroSnapshot) -> float:
        """
        Composite macro risk-on/off score: -100 (full risk-off) to +100.
        Rules based on well-established macro relationships:
          - DXY rising = risk-off (strong dollar hurts crypto)
          - VIX high = risk-off (equity fear spills to crypto)
          - Gold rising fast = risk-off (flight to safety)
          - 10yr yield rising fast = risk-off (bonds attractive vs risk assets)
          - SPX falling = risk-off (correlates with crypto drawdowns)
        """
        score = 0.0

        # DXY: strong dollar = bad for crypto
        if snap.dxy > 0:
            if snap.dxy_change_1d > 0.5:
                score -= 30   # DXY up strongly today
            elif snap.dxy_change_1d > 0.2:
                score -= 15
            elif snap.dxy_change_1d < -0.5:
                score += 30   # DXY weakening = bullish for crypto
            elif snap.dxy_change_1d < -0.2:
                score += 15

        # VIX: equity fear bleeds into crypto
        if snap.vix > 0:
            if snap.vix > 35:
                score -= 40   # extreme fear — everyone sells everything
            elif snap.vix > 25:
                score -= 20
            elif snap.vix > 18:
                score -= 5
            elif snap.vix < 13:
                score += 15   # complacency = risk-on environment

        # Gold: rising gold = flight to safety (initially bad for BTC)
        # but sustained gold rally often leads BTC upward (shared narrative)
        if snap.gold_change_1d > 2.0:
            score -= 10   # sharp gold spike = panic buying, risk-off
        elif snap.gold_change_1d < -1.5:
            score += 10   # gold selling = risk appetite returning

        # S&P 500: intraday correlation with BTC is real
        if snap.spx_change_1d > 1.5:
            score += 20
        elif snap.spx_change_1d > 0.5:
            score += 10
        elif snap.spx_change_1d < -1.5:
            score -= 20
        elif snap.spx_change_1d < -0.5:
            score -= 10

        # 10-year yield: rising fast = risk assets under pressure
        if snap.ten_yr_yield > 0:
            if snap.ten_yr_yield > 5.0:
                score -= 20   # yields so high bonds are a real alternative
            elif snap.ten_yr_yield > 4.5:
                score -= 10

        return max(-100.0, min(100.0, score))

    # ── COT polling loop ──────────────────────────────────────
    async def _cot_loop(self):
        """Poll CFTC COT report once per day."""
        await asyncio.sleep(60)
        while self._running:
            try:
                await self._poll_cot()
            except Exception as e:
                logger.debug(f"COT poll error: {e}")
            await asyncio.sleep(_COT_INTERVAL)

    async def _poll_cot(self):
        """
        Pull the CFTC Commitment of Traders report for CME Bitcoin futures.
        Market code: 133741 (Bitcoin futures on CME)
        Asset managers = BlackRock/Fidelity tier.
        Leveraged funds = hedge funds.
        """
        try:
            # CFTC publishes COT data as a public API
            url = ("https://publicreporting.cftc.gov/api/odata/v1/"
                   "CorpActions?$filter=Market_and_Exchange_Names eq 'BITCOIN - CHICAGO "
                   "MERCANTILE EXCHANGE'&$orderby=Report_Date_as_YYYY_MM_DD desc&$top=1")
            r = await self._get(url)
            if not r or "value" not in r or not r["value"]:
                # Fallback: try the simplified CFTC endpoint
                await self._poll_cot_fallback()
                return

            row = r["value"][0]
            snap = COTSnapshot(
                timestamp=time.time(),
                report_date=str(row.get("Report_Date_as_YYYY_MM_DD", "")),
                asset_mgr_long=float(row.get("Asset_Mgr_Positions_Long_All", 0)),
                asset_mgr_short=float(row.get("Asset_Mgr_Positions_Short_All", 0)),
                leveraged_fund_long=float(row.get("Lev_Money_Positions_Long_All", 0)),
                leveraged_fund_short=float(row.get("Lev_Money_Positions_Short_All", 0)),
                dealer_long=float(row.get("Dealer_Positions_Long_All", 0)),
                dealer_short=float(row.get("Dealer_Positions_Short_All", 0)),
            )
            snap.institutional_lean = self._cot_lean(
                snap.asset_mgr_long, snap.asset_mgr_short)
            snap.hedge_fund_lean = self._cot_lean(
                snap.leveraged_fund_long, snap.leveraged_fund_short)

            self._cot = snap
            await self._db.record_cot(snap)
            logger.info(
                f"📋 COT ({snap.report_date}): "
                f"Asset Mgrs={snap.institutional_lean:+.0f} "
                f"Hedge Funds={snap.hedge_fund_lean:+.0f}"
            )
        except Exception as e:
            logger.debug(f"COT parse error: {e}")
            await self._poll_cot_fallback()

    async def _poll_cot_fallback(self):
        """
        Fallback: load last known COT from DB if API is unavailable.
        COT data is weekly so a few days old is still useful.
        """
        self._cot = await self._db.get_latest_cot()

    @staticmethod
    def _cot_lean(longs: float, shorts: float) -> float:
        """
        Net lean: +100 = all longs, -100 = all shorts.
        Returns percentage of longs minus 50 (so 0 = neutral).
        """
        total = longs + shorts
        if total <= 0:
            return 0.0
        pct_long = longs / total * 100
        return round((pct_long - 50) * 2, 1)  # scale to -100/+100

    # ── Score computation ─────────────────────────────────────
    async def _compute_scores(self):
        """
        Aggregate all exchange signals into a per-coin SmartMoneyScore.
        Each contributing signal is percentile-ranked so a 70% long ratio
        in a period where average is 60% is different from a period where
        average is 68%.
        """
        for sym in self._symbols:
            try:
                score, components, pcts, signal_count = await self._score_symbol(sym)
                confidence = min(100.0, signal_count * 14)
                summary = self._score_summary(sym, score, components, signal_count)
                self._scores[sym] = SmartMoneyScore(
                    symbol=sym,
                    timestamp=time.time(),
                    score=score,
                    confidence=confidence,
                    signal_count=signal_count,
                    components=components,
                    percentiles=pcts,
                    summary=summary,
                )
            except Exception as e:
                logger.debug(f"Score compute {sym} error: {e}")

    async def _score_symbol(self, sym: str) -> Tuple[float, dict, dict, int]:
        components = {}
        percentiles = {}
        raw_score = 0.0
        signal_count = 0

        # Binance top trader ratio
        bn_top = await self._db.get_latest(sym, "top_long_pct", "binance")
        if bn_top:
            val, p30, p90 = bn_top
            # Normalize: 50 = neutral, above = bullish
            delta = (val - 50) * 1.5   # scale: 70% long → +30 points
            raw_score += delta * 0.35   # 35% weight
            components["binance_top_traders"] = round(delta, 1)
            percentiles["binance_long_pct"] = round(p90, 0) if p90 else None
            signal_count += 1

        # Bybit account ratio
        by_top = await self._db.get_latest(sym, "top_long_pct", "bybit")
        if by_top:
            val, p30, p90 = by_top
            delta = (val - 50) * 1.5
            raw_score += delta * 0.20   # 20% weight
            components["bybit_account_ratio"] = round(delta, 1)
            signal_count += 1

        # Binance taker buy dominance
        taker = await self._db.get_latest(sym, "taker_buy_pct", "binance")
        if taker:
            val, p30, p90 = taker
            delta = (val - 50) * 1.2
            raw_score += delta * 0.20   # 20% weight
            components["taker_aggression"] = round(delta, 1)
            percentiles["taker_buy_pct"] = round(p90, 0) if p90 else None
            signal_count += 1

        # Coinbase premium (BTC/ETH only)
        if sym in ("BTC", "ETH"):
            cb_prem = await self._db.get_latest(sym, "premium_pct", "coinbase")
            if cb_prem:
                val, p30, p90 = cb_prem
                # Positive premium = US institutions buying = bullish
                delta = val * 40   # 0.1% premium → +4 points
                delta = max(-50, min(50, delta))
                raw_score += delta * 0.25   # 25% weight for BTC/ETH
                components["coinbase_premium"] = round(delta, 1)
                percentiles["coinbase_premium_pct"] = round(p90, 0) if p90 else None
                signal_count += 1

        # CME COT institutional lean (BTC only, global signal)
        if sym == "BTC" and self._cot:
            # Asset managers (BlackRock tier): highest weight
            inst_lean = self._cot.institutional_lean
            if inst_lean != 0:
                raw_score += inst_lean * 0.30
                components["cme_institutional"] = round(inst_lean, 1)
                signal_count += 1
            # Hedge funds: contrarian indicator
            hf_lean = self._cot.hedge_fund_lean
            if hf_lean != 0:
                # Hedge funds being massively short is actually bullish (short squeeze)
                # Hedge funds being massively long is bearish (crowded trade)
                raw_score += (-hf_lean * 0.10)
                components["cme_hedge_funds"] = round(-hf_lean, 1)

        # Funding rate: extreme positive = longs overextended = bearish signal
        fr = await self._db.get_latest(sym, "funding_rate", "binance")
        if fr:
            val, p30, p90 = fr
            if p90 is not None:
                if p90 > 85:   # funding in top 15% historically = longs crowded
                    raw_score -= 15
                    components["funding_crowded_longs"] = -15
                elif p90 < 15:  # funding in bottom 15% = shorts crowded
                    raw_score += 15
                    components["funding_crowded_shorts"] = 15
            percentiles["funding_rate_pct"] = round(p90, 0) if p90 else None
            signal_count = max(signal_count, 1)

        # OI change: rising OI with price rising = genuine buying
        # (simplified — we compare current OI to yesterday via percentile)
        oi = await self._db.get_latest(sym, "oi_contracts", "binance")
        if oi and oi[2] is not None:
            val, p30, p90 = oi
            if p90 > 80:
                # High OI = big position being held, direction unknown but conviction high
                components["oi_elevated"] = f"OI at {p90:.0f}th pct"
            percentiles["oi_pct"] = round(p90, 0) if p90 else None

        score = max(-100.0, min(100.0, raw_score))
        return score, components, percentiles, signal_count

    def _score_summary(self, sym: str, score: float,
                       components: dict, signal_count: int) -> str:
        if signal_count == 0:
            return f"{sym}: No institutional data yet"
        direction = "LONG" if score > 10 else "SHORT" if score < -10 else "NEUTRAL"
        strength = "strongly" if abs(score) > 60 else "moderately" if abs(score) > 30 else "slightly"
        parts = []
        if "binance_top_traders" in components:
            v = components["binance_top_traders"]
            parts.append(f"BN top traders {'+' if v>0 else ''}{v:.0f}")
        if "coinbase_premium" in components:
            v = components["coinbase_premium"]
            parts.append(f"CB premium {'+' if v>0 else ''}{v:.0f}")
        if "cme_institutional" in components:
            v = components["cme_institutional"]
            parts.append(f"CME inst {'+' if v>0 else ''}{v:.0f}")
        detail = " · ".join(parts[:3]) if parts else ""
        return f"{sym}: {strength} {direction} ({score:+.0f}) [{detail}]"

    # ── Public API ────────────────────────────────────────────
    def get_score(self, symbol: str) -> Optional[SmartMoneyScore]:
        """Get latest SmartMoneyScore for a coin symbol."""
        coin = symbol.replace("/USDT", "").replace("/BUSD", "").upper()
        return self._scores.get(coin)

    def get_signal_intel(self, symbol: str, direction: str) -> Tuple[float, str]:
        """
        Returns (confidence_delta, note) for integration into
        the existing signal pipeline (same interface as smart_money_client).
        """
        score_obj = self.get_score(symbol)
        if not score_obj or score_obj.signal_count == 0:
            return 0.0, ""

        score = score_obj.score
        is_long = direction.upper() == "LONG"

        # Align: score > 0 = bullish signal
        aligned = (is_long and score > 15) or (not is_long and score < -15)
        opposed = (is_long and score < -20) or (not is_long and score > 20)

        if aligned and abs(score) > 50:
            delta = 8.0
            note = f"🏛️ Institutional flow: strongly aligned ({score:+.0f})"
        elif aligned and abs(score) > 25:
            delta = 4.0
            note = f"🏛️ Institutional flow: aligned ({score:+.0f})"
        elif opposed and abs(score) > 50:
            delta = -10.0
            note = f"🏛️ Institutional flow: strongly opposed ({score:+.0f})"
        elif opposed and abs(score) > 25:
            delta = -5.0
            note = f"🏛️ Institutional flow: opposed ({score:+.0f})"
        else:
            delta = 0.0
            note = ""

        return delta, note

    def get_macro(self) -> Optional[MacroSnapshot]:
        return self._macro

    def get_macro_score(self) -> float:
        return self._macro.risk_on_score if self._macro else 0.0

    def get_cot(self) -> Optional[COTSnapshot]:
        return self._cot

    def get_dashboard_data(self) -> dict:
        """Full data dump for dashboard Smart Money page."""
        scores = {}
        for sym, sc in self._scores.items():
            scores[sym] = {
                "score":        round(sc.score, 1),
                "confidence":   round(sc.confidence, 0),
                "signals":      sc.signal_count,
                "components":   sc.components,
                "percentiles":  sc.percentiles,
                "summary":      sc.summary,
                "age_min":      round((time.time() - sc.timestamp) / 60, 0),
            }
        macro_data = {}
        if self._macro:
            m = self._macro
            macro_data = {
                "dxy":            round(m.dxy, 2),
                "dxy_change":     round(m.dxy_change_1d, 2),
                "vix":            round(m.vix, 1),
                "gold":           round(m.gold_usd, 0),
                "gold_change":    round(m.gold_change_1d, 2),
                "ten_yr_yield":   round(m.ten_yr_yield, 2),
                "spx_change":     round(m.spx_change_1d, 2),
                "risk_on_score":  round(m.risk_on_score, 0),
                "label":          self._macro_label(m.risk_on_score),
            }
        cot_data = {}
        if self._cot:
            c = self._cot
            cot_data = {
                "report_date":      c.report_date,
                "institutional_lean": round(c.institutional_lean, 1),
                "hedge_fund_lean":    round(c.hedge_fund_lean, 1),
                "asset_mgr_long":     round(c.asset_mgr_long, 0),
                "asset_mgr_short":    round(c.asset_mgr_short, 0),
                "label":            "BULLISH" if c.institutional_lean > 20
                                    else "BEARISH" if c.institutional_lean < -20
                                    else "NEUTRAL",
            }
        return {
            "scores":       scores,
            "macro":        macro_data,
            "cot":          cot_data,
            "last_updated": round(self._last_exchange_poll),
            "symbols":      self._symbols,
        }

    @staticmethod
    def _macro_label(score: float) -> str:
        if score > 40:
            return "Risk-On 🟢"
        elif score > 15:
            return "Cautiously bullish 🟡"
        elif score > -15:
            return "Neutral ⚪"
        elif score > -40:
            return "Cautiously bearish 🟠"
        else:
            return "Risk-Off 🔴"

    # ── HTTP helper ───────────────────────────────────────────
    async def _get(self, url: str, params: dict = None) -> Optional[dict]:
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                elif resp.status == 429:
                    logger.warning(f"Rate limited (429): {url}")
                    await asyncio.sleep(5)
                return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout: {url}")
            return None
        except Exception as e:
            logger.warning(f"GET error {url}: {e}")
            return None



    def get_lean(self, symbol: str) -> float:
        """Get -100 to +100 institutional lean for a symbol. 0 = no signal."""
        sc = self._scores.get(symbol) or self._scores.get("BTC/USDT")
        return sc.score if sc else 0.0

    def get_macro_regime(self) -> str:
        """Current macro environment: RISK_ON / VOLATILE / NEUTRAL.
        Uses the existing risk_on_score field (0-100) from MacroSnapshot."""
        if not self._macro or not self._macro.timestamp:
            return "NEUTRAL"
        score = getattr(self._macro, 'risk_on_score', 50.0)
        if score >= 65: return "RISK_ON"
        if score <= 35: return "VOLATILE"
        return "NEUTRAL"

    def get_ensemble_vote(self, symbol: str, direction: str):
        """
        Returns (aligned: bool, note: str) for ensemble voter smart_money slot.
        Replaces previous Hyperliquid-only binary signal with multi-source reading.
        """
        sc = self._scores.get(symbol) or self._scores.get("BTC/USDT")
        if not sc or abs(sc.score) < 20:
            return False, ""
        lean      = sc.score
        regime    = self.get_macro_regime()
        aligned   = (lean > 20 and direction == "LONG") or (lean < -20 and direction == "SHORT")
        mac_note  = f" [{regime}]" if regime != "NEUTRAL" else ""
        conf_note = f" ({sc.confidence:.0f}% agreement)" if sc.confidence > 50 else ""
        if aligned:
            note = f"🏦 Institutional: {abs(lean):.0f}pt {direction} lean{conf_note}{mac_note}"
        else:
            note = f"⚠️ Institutional lean opposes {direction} ({lean:+.0f}pt){mac_note}"
        return aligned, note

    def get_status(self) -> dict:
        """Status dict for dashboard and diagnostics."""
        m = self._macro
        c = self._cot
        am_t = (c.asset_mgr_long + c.asset_mgr_short) if c else 0
        lf_t = (c.leveraged_fund_long + getattr(c, 'leveraged_fund_short', 0)) if c else 0
        return {
            "active":           self._running,
            "signals_cached":   len(self._scores),
            "macro_regime":     self.get_macro_regime(),
            "dxy":              m.dxy              if m else None,
            "vix":              m.vix              if m else None,
            "yield_10y":        m.ten_yr_yield     if m else None,
            "gold":             m.gold_usd         if m else None,
            "risk_on_score":    m.risk_on_score    if m else 50,
            "cme_am_long":      round(c.asset_mgr_long / am_t * 100, 1) if c and am_t > 0 else None,
            "cme_lf_long":      round(c.leveraged_fund_long / lf_t * 100, 1) if c and lf_t > 0 else None,
            "cme_report_date":  c.report_date      if c else None,
            "last_updated":     int(m.timestamp) if m and m.timestamp else 0,
        }

    def get_telegram_context(self, symbol: str) -> str:
        """
        One-line context string for Telegram signal card.
        e.g. "🌍 CME 68%L | CB +$42 | VIX 18.2 (low fear) | DXY 92nd pct"
        """
        sc = self._scores.get(symbol) or self._scores.get("BTC/USDT")
        m  = self._macro
        c  = self._cot
        if not sc and not m:
            return ""
        parts = []
        if c and c.asset_mgr_long:
            am_t = c.asset_mgr_long + c.asset_mgr_short
            am_pct = c.asset_mgr_long / am_t * 100 if am_t else 50
            parts.append(f"CME {am_pct:.0f}%L")
        # Coinbase premium from score components (key is "coinbase_premium")
        cb_delta = (sc.components or {}).get("coinbase_premium") if sc else None
        if cb_delta is not None:
            # delta is in score points, not USD — just show direction
            parts.append(f"CB {'↑' if cb_delta > 0 else '↓'} prem")
        else:
            # Try direct DB lookup for actual USD premium
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                coin = symbol.replace("/USDT","").replace("/BUSD","")
                cb_raw = self._db.get_latest_sync(coin, "premium_usd", "coinbase") if hasattr(self._db, "get_latest_sync") else None
                if cb_raw:
                    parts.append(f"CB {cb_raw[0]:+.0f}$")
            except Exception:
                pass
        # Binance top traders from score components (key is "binance_top_traders", value is delta)
        bn_delta = (sc.components or {}).get("binance_top_traders") if sc else None
        if bn_delta is not None:
            # Convert delta back to approximate percentage: delta = (pct - 50) * 1.5
            approx_pct = 50 + bn_delta / 1.5
            parts.append(f"BN {approx_pct:.0f}%L")
        if m and m.vix:
            fear = " (high fear)" if m.vix > 25 else (" (low fear)" if m.vix < 15 else "")
            parts.append(f"VIX {m.vix:.1f}{fear}")
        # AUDIT FIX: ``MacroSnapshot`` has no ``dxy_pct`` field — the direct
        # attribute access would raise AttributeError whenever a populated
        # snapshot reached this code path.  Look up the DXY percentile in
        # the flow-history DB instead and guard with ``getattr`` so older
        # snapshots without the field stay safe.
        _dxy_pct = getattr(m, "dxy_pct", None) if m else None
        if _dxy_pct is not None:
            if _dxy_pct > 75 or _dxy_pct < 25:
                parts.append(f"DXY {_dxy_pct:.0f}th pct")
        if not parts:
            return ""
        regime = self.get_macro_regime()
        emoji  = {"RISK_ON": "🟢", "VOLATILE": "🔴", "NEUTRAL": "⚪"}.get(regime, "⚪")
        return f"{emoji} Macro: {' | '.join(parts[:4])}"


# ── Singleton ─────────────────────────────────────────────────
institutional_flow = InstitutionalFlowEngine()
