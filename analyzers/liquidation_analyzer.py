"""
TitanBot Pro — Multi-Exchange OI & Liquidation Intelligence
============================================================
Aggregates open interest, funding rates, long/short ratios and
liquidation cluster estimates from 7 exchanges:

  Binance · Bybit · OKX · Bitget · Gate.io · BitMEX · Hyperliquid

Zero API keys required. All public endpoints.

How it helps signals:
  • OI rising + aligned signal direction → +4 confidence (conviction)
  • Overcrowded longs/shorts (high funding + extreme L/S) → -8 confidence
  • Nearest liquidation cluster = natural TP target (price hunts liquidity)
  • Aggregate OI context shown on dashboard

Liquidation cluster sources:
  • Bybit / OKX order book walls (large walls = stop/liq clusters)
  • Binance force orders feed (real liquidations accumulated by price level)
  • Price levels with $1M+ wall density flagged as significant clusters

Refresh strategy (lightweight for MacBook):
  • OI + funding: every 5 min (staggered bulk calls, 1 per exchange)
  • Liq clusters: every 10 min (order book analysis top coins)
  • All data gracefully degrades — bot works fine if any exchange is down
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ── Exchange base URLs ─────────────────────────────────────────────────────────
_BINANCE = "https://fapi.binance.com"
_BYBIT   = "https://api.bybit.com"
_OKX     = "https://www.okx.com"
_BITGET  = "https://api.bitget.com"
_GATE    = "https://fx.gate.io"
_BITMEX  = "https://www.bitmex.com"
_HL      = "https://api.hyperliquid.xyz"

_OI_REFRESH  = 300   # 5 min — full OI sweep
_LIQ_REFRESH = 600   # 10 min — cluster estimation
_STAGGER     = 1.5   # seconds between exchange calls (polite)
_WALL_MIN    = 1_000_000  # $1M minimum wall to count as cluster


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class CoinIntelligence:
    """Aggregated market structure data for one coin across all exchanges."""
    coin:           str
    total_oi_usd:   float = 0.0   # aggregate OI in USD notional
    by_exchange:    Dict[str, float] = field(default_factory=dict)
    long_ratio:     float = 0.5   # fraction of positions that are long (0–1)
    funding_rate:   float = 0.0   # weighted average funding rate
    oi_change_1h:   float = 0.0   # % change vs 1h ago (positive = rising OI)
    current_price:  float = 0.0
    liq_above:      List[Tuple[float, float]] = field(default_factory=list)  # (price, USD)
    liq_below:      List[Tuple[float, float]] = field(default_factory=list)
    exchange_count: int   = 0
    fetched_at:     float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > 1800  # 30 min

    @property
    def oi_trend(self) -> str:
        c = self.oi_change_1h
        if c > 10:  return "RISING_FAST"
        if c > 3:   return "RISING"
        if c < -10: return "FALLING_FAST"
        if c < -3:  return "FALLING"
        return "FLAT"

    @property
    def crowd_sentiment(self) -> str:
        """Crowding signal from funding rate + L/S ratio combined."""
        fr = self.funding_rate
        lr = self.long_ratio
        if fr > 0.001 and lr > 0.65:  return "OVERCROWDED_LONG"
        if fr < -0.001 and lr < 0.35: return "OVERCROWDED_SHORT"
        if lr > 0.62:                  return "LONG_HEAVY"
        if lr < 0.38:                  return "SHORT_HEAVY"
        return "BALANCED"

    def nearest_liq_tp(self, direction: str, entry: float,
                       min_usd: float = 500_000) -> Optional[float]:
        """
        Nearest significant liquidation cluster in trade direction.
        LONG: cluster ABOVE entry (shorts get squeezed → price spikes through)
        SHORT: cluster BELOW entry (longs get liquidated → price crashes through)
        """
        if direction == "LONG":
            candidates = [(p, u) for p, u in self.liq_above
                          if p > entry * 1.003 and u >= min_usd]
            return min(candidates, key=lambda x: x[0])[0] if candidates else None
        else:
            candidates = [(p, u) for p, u in self.liq_below
                          if p < entry * 0.997 and u >= min_usd]
            return max(candidates, key=lambda x: x[0])[0] if candidates else None

    def confidence_delta(self, direction: str) -> Tuple[int, str]:
        """
        Returns (confidence_delta, note) for engine to apply before publishing.
        Positive = boost, negative = penalty.
        """
        delta = 0
        notes: List[str] = []

        if direction == "LONG":
            if self.oi_trend in ("RISING", "RISING_FAST"):
                delta += 4
                notes.append(f"📈 OI↑{self.oi_change_1h:+.1f}% conviction")
            elif self.oi_trend in ("FALLING", "FALLING_FAST"):
                delta -= 3
                notes.append(f"📉 OI↓{self.oi_change_1h:+.1f}% unwinding")
            if self.crowd_sentiment == "OVERCROWDED_LONG":
                delta -= 8
                notes.append(
                    f"⚠️ Overcrowded long "
                    f"(fr={self.funding_rate:.4f}, {self.long_ratio:.0%} long)"
                )
            elif self.crowd_sentiment == "LONG_HEAVY":
                delta += 2
                notes.append(f"🐂 Market leans long ({self.long_ratio:.0%})")
            elif self.crowd_sentiment == "SHORT_HEAVY":
                delta -= 2
                notes.append(f"🔻 Market leans short ({self.long_ratio:.0%})")
        else:  # SHORT
            if self.oi_trend in ("FALLING", "FALLING_FAST"):
                delta += 4
                notes.append(f"📉 OI↓{self.oi_change_1h:+.1f}% unwinding")
            elif self.oi_trend in ("RISING", "RISING_FAST"):
                delta -= 3
                notes.append(f"📈 OI↑{self.oi_change_1h:+.1f}% building")
            if self.crowd_sentiment == "OVERCROWDED_SHORT":
                delta -= 8
                notes.append(
                    f"⚠️ Overcrowded short "
                    f"(fr={self.funding_rate:.4f}, {self.long_ratio:.0%} long)"
                )
            elif self.crowd_sentiment == "SHORT_HEAVY":
                delta += 2
                notes.append(f"🐻 Market leans short ({self.long_ratio:.0%})")
            elif self.crowd_sentiment == "LONG_HEAVY":
                delta -= 2
                notes.append(f"🔺 Market leans long ({self.long_ratio:.0%})")

        return delta, " | ".join(notes)


# ── Main Analyzer ─────────────────────────────────────────────────────────────

class LiquidationAnalyzer:
    """
    Multi-exchange OI aggregator and liquidation cluster estimator.
    No API keys required. Runs as a background async task.
    """

    def __init__(self):
        self._cache:    Dict[str, CoinIntelligence] = {}
        self._session:  Optional[aiohttp.ClientSession] = None
        self._running   = False
        self._task:     Optional[asyncio.Task] = None
        # OI history for trend calculation: coin → list of (timestamp, oi_usd)
        self._oi_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        # Accumulated Binance force orders: coin → list of (price, usd)
        self._force_orders: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    @property
    def is_ready(self) -> bool:
        return bool(self._cache)

    @property
    def history_depth_days(self) -> float:
        """How many days of OI history we have (for logging)."""
        if not self._oi_history:
            return 0.0
        oldest = min(
            (h[0][0] for h in self._oi_history.values() if h),
            default=time.time()
        )
        return (time.time() - oldest) / 86400

    def load_history(self, history: Dict[str, List[Tuple[float, float]]],
                     clusters: Dict[str, List[Tuple[float, float, str]]]):
        """
        Load pre-bootstrapped OI history and liquidation clusters.
        Called from bootstrap script before start().

        history:  coin → [(timestamp, oi_usd), ...]
        clusters: coin → [(price, usd_size, 'above'|'below'), ...]
        """
        now = time.time()
        loaded_coins = 0
        for coin, points in history.items():
            if points:
                # Normalize: bootstrap writes [ts, oi, price] 3-tuples,
                # internal format is (ts, oi) 2-tuples
                normalized = [(p[0], p[1]) for p in points if len(p) >= 2]
                self._oi_history[coin.upper()] = normalized
                loaded_coins += 1

        # Build CoinIntelligence objects from cluster data
        for coin, cluster_points in clusters.items():
            coin = coin.upper()
            ci = self._cache.get(coin, CoinIntelligence(coin=coin))
            above = [(p, u) for p, u, side in cluster_points if side == 'above']
            below = [(p, u) for p, u, side in cluster_points if side == 'below']
            ci.liq_above = sorted(above, key=lambda x: x[0])[:15]
            ci.liq_below = sorted(below, key=lambda x: x[0], reverse=True)[:15]
            ci.fetched_at = now
            self._cache[coin] = ci

        logger.info(
            f"📊 LiquidationAnalyzer: loaded history for {loaded_coins} coins, "
            f"{len(clusters)} with cluster data "
            f"(depth ~{self.history_depth_days:.1f} days)"
        )

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._main_loop())
            logger.info(
                "📊 LiquidationAnalyzer started "
                "(Binance·Bybit·OKX·Bitget·Gate·BitMEX·Hyperliquid — no API keys)"
            )

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    def get(self, coin: str) -> Optional[CoinIntelligence]:
        """Get cached intelligence for a coin. Returns None if not yet fetched."""
        data = self._cache.get(coin.upper())
        return data if data and not data.is_stale else None

    def get_all(self) -> Dict[str, CoinIntelligence]:
        """Get all cached coin intelligence (for dashboard)."""
        return {k: v for k, v in self._cache.items() if not v.is_stale}

    def get_summary_for_dashboard(self) -> list:
        """
        Return a serializable list of coin intelligence summaries for the dashboard.
        Each entry is a plain dict with the key fields the JS needs.
        """
        result = []
        for coin, ci in self._cache.items():
            if ci.is_stale:
                continue
            result.append({
                "coin":          coin,
                "oi_usd":        ci.oi_usd,
                "funding_rate":  ci.funding_rate,
                "long_ratio":    ci.long_ratio,
                "short_ratio":   1.0 - ci.long_ratio,
                "oi_trend":      ci.oi_trend,
                "crowd_bias":    ci.crowd_sentiment,
                "liq_above":     [[p, u] for p, u in (ci.liq_above or [])[:5]],
                "liq_below":     [[p, u] for p, u in (ci.liq_below or [])[:5]],
                "current_price": ci.current_price,
                "exchanges":     ci.exchanges,
                "fetched_at":    ci.fetched_at,
            })
        return sorted(result, key=lambda x: x["oi_usd"], reverse=True)

    # ── HTTP session ──────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "TitanBot/3.0"},
            )
        return self._session

    async def _get(self, url: str, params: dict = None) -> Optional[dict]:
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            pass
        return None

    async def _post(self, url: str, payload: dict) -> Optional[dict]:
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            pass
        return None

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _main_loop(self):
        await asyncio.sleep(25)  # let bot warm up first
        oi_due  = 0.0
        liq_due = 0.0

        while self._running:
            try:
                now = time.time()

                if now >= oi_due:
                    await self._refresh_all_oi()
                    oi_due = now + _OI_REFRESH

                if now >= liq_due:
                    await self._refresh_liq_clusters()
                    liq_due = now + _LIQ_REFRESH

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"LiquidationAnalyzer loop error: {e}")
                await asyncio.sleep(60)

    # ── OI Collection (one bulk call per exchange) ────────────────────────────

    async def _refresh_all_oi(self):
        """Fetch OI from all 7 exchanges and merge per coin."""

        # Staggered calls to avoid hammering all exchanges at once
        fetchers = [
            ("bybit",       self._fetch_bybit_oi()),
            ("okx",         self._fetch_okx_oi()),
            ("bitget",      self._fetch_bitget_oi()),
            ("gate",        self._fetch_gate_oi()),
            ("hyperliquid", self._fetch_hl_oi()),
            ("binance",     self._fetch_binance_funding()),   # funding + prices
            ("bitmex",      self._fetch_bitmex_oi()),
        ]

        # Merge: coin → {oi_usd, funding_weighted, lr_weighted, price, count, exchanges}
        merged: Dict[str, dict] = {}

        for exch_name, coro in fetchers:
            try:
                result = await coro
                if not result:
                    continue
                for coin, data in result.items():
                    coin = coin.upper()
                    oi = float(data.get("oi_usd", 0) or 0)
                    if coin not in merged:
                        merged[coin] = {
                            "oi_usd": 0.0, "fund_w": 0.0, "lr_w": 0.0,
                            "price": 0.0,  "count": 0, "exchanges": []
                        }
                    m = merged[coin]
                    if oi > 0:
                        m["oi_usd"]  += oi
                        m["fund_w"]  += data.get("funding", 0) * oi
                        m["lr_w"]    += data.get("long_ratio", 0.5) * oi
                        m["count"]   += 1
                        m["exchanges"].append(exch_name)
                    if data.get("price", 0) > 0:
                        m["price"] = data["price"]
            except Exception as e:
                logger.debug(f"LiqAnalyzer {exch_name} error: {e}")
            await asyncio.sleep(_STAGGER)

        # Enrich with Bybit L/S account ratio for all coins (fixes 50% default)
        try:
            bybit_ls = await self._fetch_bybit_ls_ratio()
            if bybit_ls:
                coins_enriched = 0
                for coin, lr in bybit_ls.items():
                    coin = coin.upper()
                    if coin in merged:
                        # Only override if we had no good L/S data (i.e. lr_w / oi_usd ≈ 0.5)
                        m = merged[coin]
                        if m["oi_usd"] > 0:
                            existing_lr = m["lr_w"] / m["oi_usd"]
                            if abs(existing_lr - 0.5) < 0.01:  # was at default
                                m["lr_w"] = lr * m["oi_usd"]
                                coins_enriched += 1
                        else:
                            # No OI data yet, just store the ratio
                            m["lr_w"] = lr * max(m["oi_usd"], 1)
                logger.debug(f"Bybit L/S ratio enriched {coins_enriched} coins")
        except Exception as e:
            logger.debug(f"Bybit L/S ratio enrich error: {e}")

        # Build / update CoinIntelligence cache
        now = time.time()
        updated = 0
        for coin, m in merged.items():
            oi = m["oi_usd"]
            if oi <= 0:
                continue

            funding = m["fund_w"] / oi
            lr      = max(0.0, min(1.0, m["lr_w"] / oi))

            # OI 1h trend
            hist = self._oi_history[coin]
            hist.append((now, oi))
            hist[:] = [(t, v) for t, v, *_ in hist if now - t < 5400]  # keep 90 min

            oi_change = 0.0
            hour_ago = [(t, v) for t, v, *_ in hist if now - t >= 3300]
            if hour_ago:
                old_oi = hour_ago[0][1]
                # Guard: require minimum OI value to avoid division-by-near-zero
                # and cap percentage to prevent overflow spikes (e.g. 478M%)
                if old_oi >= 1000.0:  # Minimum $1000 OI to compute meaningful %
                    oi_change = (oi - old_oi) / old_oi * 100
                    oi_change = max(-500.0, min(500.0, oi_change))  # Cap at ±500%

            ci = self._cache.get(coin, CoinIntelligence(coin=coin))
            ci.total_oi_usd   = oi
            ci.by_exchange     = {ex: 1 for ex in m["exchanges"]}
            ci.funding_rate    = funding
            ci.long_ratio      = lr
            ci.oi_change_1h    = oi_change
            ci.exchange_count  = m["count"]
            if m["price"] > 0:
                ci.current_price = m["price"]
            ci.fetched_at      = now
            self._cache[coin]  = ci
            updated += 1

        logger.info(
            f"📊 OI refreshed: {updated} coins from "
            f"{sum(1 for r in [True] if True)} exchanges"  # simplified
        )

    # ── Per-exchange OI fetchers ───────────────────────────────────────────────

    async def _fetch_bybit_oi(self) -> Dict[str, dict]:
        """Bybit: one call returns all linear symbols with OI."""
        out = {}
        try:
            data = await self._get(f"{_BYBIT}/v5/market/tickers",
                                   params={"category": "linear"})
            if not data:
                return out
            for item in data.get("result", {}).get("list", []):
                sym = item.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                coin = sym[:-4]
                oi_usd = float(item.get("openInterestValue", 0) or 0)
                price  = float(item.get("markPrice", 0) or 0)
                # Bybit funding rate in item
                funding = float(item.get("fundingRate", 0) or 0)
                if oi_usd > 0:
                    out[coin] = {
                        "oi_usd": oi_usd, "funding": funding,
                        "long_ratio": 0.5, "price": price
                    }
        except Exception as e:
            logger.debug(f"Bybit OI error: {e}")
        return out

    async def _fetch_okx_oi(self) -> Dict[str, dict]:
        """OKX: one call returns all USDT perpetual swap OI."""
        out = {}
        try:
            data = await self._get(f"{_OKX}/api/v5/public/open-interest",
                                   params={"instType": "SWAP"})
            if not data:
                return out

            # Also get tickers for prices + funding
            tickers = await self._get(f"{_OKX}/api/v5/market/tickers",
                                      params={"instType": "SWAP"}) or {}
            price_map = {}
            funding_map = {}
            for t in tickers.get("data", []):
                inst = t.get("instId", "")
                if inst.endswith("-USDT-SWAP"):
                    coin = inst.replace("-USDT-SWAP", "")
                    price_map[coin]   = float(t.get("last", 0) or 0)
                    funding_map[coin] = float(t.get("fundingRate", 0) or 0)

            for item in data.get("data", []):
                inst = item.get("instId", "")
                if not inst.endswith("-USDT-SWAP"):
                    continue
                coin = inst.replace("-USDT-SWAP", "")
                oi_contracts = float(item.get("oi", 0) or 0)
                price = price_map.get(coin, 0)
                oi_usd = oi_contracts * price if price > 0 else 0
                if oi_usd > 0:
                    out[coin] = {
                        "oi_usd": oi_usd,
                        "funding": funding_map.get(coin, 0),
                        "long_ratio": 0.5,
                        "price": price
                    }
        except Exception as e:
            logger.debug(f"OKX OI error: {e}")
        return out

    async def _fetch_bitget_oi(self) -> Dict[str, dict]:
        """Bitget: one call returns all USDT-FUTURES tickers with OI."""
        out = {}
        try:
            data = await self._get(f"{_BITGET}/api/v2/mix/market/tickers",
                                   params={"productType": "USDT-FUTURES"})
            if not data:
                return out
            for item in data.get("data", []):
                sym = item.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                coin = sym[:-4]
                # holdingAmount = OI in contracts, last = price
                oi_ct  = float(item.get("holdingAmount", 0) or 0)
                price  = float(item.get("last", 0) or 0)
                oi_usd = oi_ct * price if price > 0 else 0
                funding = float(item.get("fundingRate", 0) or 0)
                if oi_usd > 0:
                    out[coin] = {
                        "oi_usd": oi_usd, "funding": funding,
                        "long_ratio": 0.5, "price": price
                    }
        except Exception as e:
            logger.debug(f"Bitget OI error: {e}")
        return out

    async def _fetch_gate_oi(self) -> Dict[str, dict]:
        """Gate.io: one call returns all USDT futures contracts with OI."""
        out = {}
        try:
            # Use tickers endpoint
            data = await self._get(f"{_GATE}/api/v4/futures/usdt/tickers")
            if not data:
                return out
            for item in (data if isinstance(data, list) else []):
                contract = item.get("contract", "")
                if not contract.endswith("_USDT"):
                    continue
                coin = contract[:-5]
                # last = last price, open_interest = OI in contracts
                price     = float(item.get("last", 0) or 0)
                oi_ct     = float(item.get("open_interest", 0) or 0)
                oi_usd    = oi_ct * price if price > 0 else 0
                funding   = float(item.get("funding_rate", 0) or 0)
                long_size = float(item.get("long_liq_size", 0) or 0)
                shrt_size = float(item.get("short_liq_size", 0) or 0)
                lr = 0.5
                total_liq = long_size + shrt_size
                if total_liq > 0:
                    lr = shrt_size / total_liq  # short liq = long positions
                if oi_usd > 0:
                    out[coin] = {
                        "oi_usd": oi_usd, "funding": funding,
                        "long_ratio": lr, "price": price
                    }
        except Exception as e:
            logger.debug(f"Gate.io OI error: {e}")
        return out

    async def _fetch_hl_oi(self) -> Dict[str, dict]:
        """Hyperliquid: one call returns all perps with OI and funding."""
        out = {}
        try:
            data = await self._post(f"{_HL}/info", {"type": "metaAndAssetCtxs"})
            if not data or not isinstance(data, list) or len(data) < 2:
                return out
            meta_list  = data[0].get("universe", [])  # list of {name, szDecimals, ...}
            ctx_list   = data[1]                       # list of asset contexts

            for meta, ctx in zip(meta_list, ctx_list):
                coin = meta.get("name", "")
                if not coin:
                    continue
                price   = float(ctx.get("markPx", 0) or 0)
                oi_ct   = float(ctx.get("openInterest", 0) or 0)
                oi_usd  = oi_ct * price if price > 0 else 0
                funding = float(ctx.get("funding", 0) or 0)
                if oi_usd > 0:
                    out[coin] = {
                        "oi_usd": oi_usd, "funding": funding,
                        "long_ratio": 0.5, "price": price
                    }
        except Exception as e:
            logger.debug(f"Hyperliquid OI error: {e}")
        return out

    async def _fetch_binance_funding(self) -> Dict[str, dict]:
        """
        Binance: bulk premium index gives funding rates + prices for all perps.
        Also fetches L/S account ratio for major coins.
        """
        out = {}
        try:
            data = await self._get(f"{_BINANCE}/fapi/v1/premiumIndex")
            if not data:
                return out
            for item in (data if isinstance(data, list) else []):
                sym = item.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                coin    = sym[:-4]
                price   = float(item.get("markPrice", 0) or 0)
                funding = float(item.get("lastFundingRate", 0) or 0)
                out[coin] = {
                    "oi_usd": 0,
                    "funding": funding,
                    "long_ratio": 0.5,
                    "price": price
                }

            # Fetch Binance L/S ratio for major coins
            major_coins = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
                           "MATIC", "LINK", "DOT", "UNI", "ARB", "OP", "SUI", "APT"]
            ls_tasks = [
                self._get(
                    f"{_BINANCE}/futures/data/globalLongShortAccountRatio",
                    params={"symbol": f"{c}USDT", "period": "5m", "limit": 1}
                )
                for c in major_coins
            ]
            ls_results = await asyncio.gather(*ls_tasks, return_exceptions=True)
            for coin, result in zip(major_coins, ls_results):
                if isinstance(result, list) and result:
                    lr = float(result[0].get("longAccount", 0.5) or 0.5)
                    if coin in out:
                        out[coin]["long_ratio"] = lr
                    else:
                        out[coin] = {"oi_usd": 0, "funding": 0,
                                     "long_ratio": lr, "price": 0}

        except Exception as e:
            logger.debug(f"Binance funding error: {e}")
        return out

    async def _fetch_bybit_ls_ratio(self) -> Dict[str, float]:
        """
        Bybit global account L/S ratio — covers ALL listed perp coins.
        Returns {coin: long_ratio} mapping.
        Free, no API key needed.
        """
        out: Dict[str, float] = {}
        try:
            # First get all active symbols
            symbols_data = await self._get(
                f"{_BYBIT}/v5/market/instruments-info",
                params={"category": "linear", "status": "Trading", "limit": 1000}
            )
            if not symbols_data:
                return out
            symbols = [
                i["symbol"] for i in
                symbols_data.get("result", {}).get("list", [])
                if i.get("symbol", "").endswith("USDT")
            ]

            # Fetch L/S ratio in batches of 10 concurrent requests
            batch_size = 10
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                tasks = [
                    self._get(
                        f"{_BYBIT}/v5/market/account-ratio",
                        params={"category": "linear", "symbol": sym,
                                "period": "5min", "limit": 1}
                    )
                    for sym in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for sym, result in zip(batch, results):
                    if isinstance(result, dict) and result.get("retCode") == 0:
                        rows = result.get("result", {}).get("list", [])
                        if rows:
                            try:
                                buy_ratio = float(rows[0].get("buyRatio", 0.5) or 0.5)
                                coin = sym[:-4]  # strip USDT
                                out[coin] = buy_ratio
                            except (ValueError, TypeError):
                                pass
                await asyncio.sleep(0.05)  # polite

        except Exception as e:
            logger.debug(f"Bybit L/S ratio error: {e}")
        return out

    async def _fetch_bitmex_oi(self) -> Dict[str, dict]:
        """BitMEX: active instruments endpoint. Good for BTC/ETH primarily."""
        out = {}
        try:
            data = await self._get(
                f"{_BITMEX}/api/v1/instrument/active",
                params={"columns": "symbol,openInterest,lastPrice,fundingRate"}
            )
            if not data:
                return out
            for item in (data if isinstance(data, list) else []):
                sym = item.get("symbol", "")
                # XBTUSD = BTC, ETHUSD = ETH, altcoins have different format
                coin_map = {"XBTUSD": "BTC", "ETHUSD": "ETH",
                            "SOLUSD": "SOL", "XRPUSD": "XRP"}
                # Also handle USDTM contracts
                if sym.endswith("USDTM"):
                    coin = sym[:-5]
                elif sym in coin_map:
                    coin = coin_map[sym]
                elif sym.endswith("USD") and not sym.startswith("XBT"):
                    coin = sym[:-3]
                else:
                    coin = coin_map.get(sym, "")
                if not coin:
                    continue
                oi_ct   = float(item.get("openInterest", 0) or 0)
                price   = float(item.get("lastPrice", 0) or 0)
                oi_usd  = oi_ct / price if price > 0 and sym.endswith("USD") else oi_ct * price
                funding = float(item.get("fundingRate", 0) or 0)
                if oi_usd > 0:
                    out[coin] = {
                        "oi_usd": oi_usd, "funding": funding,
                        "long_ratio": 0.5, "price": price
                    }
        except Exception as e:
            logger.debug(f"BitMEX OI error: {e}")
        return out

    # ── Liquidation Cluster Estimation ────────────────────────────────────────

    async def _refresh_liq_clusters(self):
        """
        Build liquidation cluster map from two sources:
        1. Bybit + OKX order book walls (large walls = stop/liq clusters)
        2. Binance force orders feed (real liquidation events by price)
        """
        # Only analyze top coins by OI (too expensive for all 320)
        top_coins = sorted(
            self._cache.items(),
            key=lambda x: x[1].total_oi_usd,
            reverse=True
        )[:40]  # Top 40 by aggregate OI

        tasks = []
        coins_list = []
        for coin, ci in top_coins:
            if ci.current_price > 0:
                coins_list.append(coin)
                tasks.append(self._fetch_liq_clusters_for_coin(coin, ci.current_price))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for coin, result in zip(coins_list, results):
            if isinstance(result, Exception) or not result:
                continue
            # FIX 1A: _fetch_liq_clusters_for_coin always returns (above, below)
            # but if the API response format changes or partial execution occurs,
            # result may be a flat iterable with 3+ items causing ValueError.
            try:
                above, below = result
            except (TypeError, ValueError):
                logger.debug(f"LiqAnalyzer unpack skip {coin}: result shape unexpected ({type(result).__name__})")
                continue
            ci = self._cache.get(coin)
            if ci:
                ci.liq_above = above
                ci.liq_below = below

        # Also accumulate Binance force orders (all coins)
        await self._fetch_binance_force_orders()

        logger.debug(f"📊 Liq clusters updated for {len(coins_list)} coins")

    async def _fetch_liq_clusters_for_coin(
        self, coin: str, price: float
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Get order book for this coin from Bybit (large walls = liq clusters).
        Returns (liq_above, liq_below) each as list of (price, usd_size).
        """
        above: List[Tuple[float, float]] = []
        below: List[Tuple[float, float]] = []

        try:
            # Bybit order book — 200 levels, good depth
            data = await self._get(
                f"{_BYBIT}/v5/market/orderbook",
                params={"category": "linear", "symbol": f"{coin}USDT", "limit": 200}
            )
            if data and data.get("retCode") == 0:
                result = data.get("result", {})
                asks = result.get("a", [])  # [[price, size], ...]
                bids = result.get("b", [])

                # Aggregate into price buckets (0.5% wide)
                def bucket_walls(orders, side):
                    walls = []
                    for entry in orders:
                        try:
                            px  = float(entry[0])
                            qty = float(entry[1])
                            usd = px * qty
                            if usd >= _WALL_MIN:
                                walls.append((px, usd))
                        except (IndexError, ValueError):
                            pass
                    return walls

                ask_walls = bucket_walls(asks, "ask")  # above price
                bid_walls = bucket_walls(bids, "bid")  # below price

                above = sorted(ask_walls, key=lambda x: x[0])
                below = sorted(bid_walls, key=lambda x: x[0], reverse=True)

        except Exception as e:
            logger.debug(f"Liq cluster fetch error {coin}: {e}")

        # Merge with accumulated force orders
        fo = self._force_orders.get(coin, [])
        if fo:
            # Bin force orders into price buckets
            price_bins: Dict[float, float] = defaultdict(float)
            for liq_price, liq_usd in fo:
                # Round to nearest 0.5% bucket
                bucket = round(liq_price / (price * 0.005)) * (price * 0.005)
                price_bins[bucket] += liq_usd

            for bucket_price, total_usd in price_bins.items():
                if total_usd >= _WALL_MIN:
                    if bucket_price > price:
                        above.append((bucket_price, total_usd))
                    else:
                        below.append((bucket_price, total_usd))

        # Sort and deduplicate
        above = sorted(set(above), key=lambda x: x[0])[:10]
        below = sorted(set(below), key=lambda x: x[0], reverse=True)[:10]

        return above, below

    async def _fetch_binance_force_orders(self):
        """
        Fetch recent Binance forced liquidation orders.
        Accumulate per coin for cluster building.
        Keep last 2 hours.
        """
        try:
            data = await self._get(
                f"{_BINANCE}/fapi/v1/forceOrders",
                params={"limit": 100}
            )
            if not data:
                return
            now = time.time()
            for order in (data if isinstance(data, list) else []):
                sym = order.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                coin  = sym[:-4]
                price = float(order.get("averagePrice", 0) or 0)
                qty   = float(order.get("origQty", 0) or 0)
                usd   = price * qty
                if price > 0 and usd > 0:
                    self._force_orders[coin].append((price, usd))

            # Prune old entries (keep 2 hours)
            for coin in list(self._force_orders.keys()):
                # We don't have timestamps, keep last 500 entries per coin
                if len(self._force_orders[coin]) > 500:
                    self._force_orders[coin] = self._force_orders[coin][-500:]

        except Exception as e:
            logger.debug(f"Binance force orders error: {e}")

    # ── Public query interface ─────────────────────────────────────────────────

    def get_signal_intel(
        self, symbol: str, direction: str, entry_price: float
    ) -> Tuple[int, str, Optional[float]]:
        """
        Main interface for engine.py:
        Returns (confidence_delta, note, liq_tp_price_or_None)

        confidence_delta: points to add/subtract from signal confidence
        note: human-readable reason
        liq_tp_price: price of nearest liquidation cluster (suggested TP)
        """
        coin = symbol.replace("/USDT", "").replace("/BUSD", "").upper()
        ci = self.get(coin)
        if not ci:
            return 0, "", None

        delta, note = ci.confidence_delta(direction)
        liq_tp = ci.nearest_liq_tp(direction, entry_price, min_usd=500_000)

        return delta, note, liq_tp

    

    def build_deep_clusters(self, coin: str, current_price: float) -> None:
        """
        Build deep liquidation clusters from OI history.
        This is what lets us see levels like $69k even when we're at $83k.

        Logic:
          When OI increased significantly as price was at level X,
          it means positions were opened there. Those positions will
          liquidate at:
            Long entries at X → liq at X * (1 - 1/leverage)  ≈ X * 0.90  (10x)
            Short entries at X → liq at X * (1 + 1/leverage) ≈ X * 1.10  (10x)

          We can't know exact leverage, so we use a probabilistic model:
            - Use 5x, 10x, 20x leverage buckets (most common retail leverage)
            - Weight 10x most heavily
          
          When a large OI increase happened at a price level:
            → Long liquidation cluster below that level
            → Short squeeze cluster above that level
        """
        hist = self._oi_history.get(coin.upper(), [])
        if len(hist) < 10:
            return

        # Sort by timestamp
        hist = sorted(hist, key=lambda x: x[0])

        # Build price→OI accumulation map
        # We need price at each OI snapshot — use the stored price or estimate
        ci = self._cache.get(coin.upper())
        if not ci or ci.current_price <= 0:
            return

        # Bucket OI changes by price level
        # Since we store (timestamp, oi_usd), we pair consecutive points
        # to get OI deltas. Positive delta = positions opened at that price.
        price_buckets: Dict[float, float] = defaultdict(float)
        bucket_pct = 0.015  # 1.5% price buckets

        for i in range(1, len(hist)):
            # History entries can be [ts, oi_usd] or [ts, oi_usd, price]
            _h_prev = hist[i-1]
            _h_curr = hist[i]
            t_prev, oi_prev = _h_prev[0], _h_prev[1]
            t_curr, oi_curr = _h_curr[0], _h_curr[1]
            oi_delta = oi_curr - oi_prev

            if abs(oi_delta) < 1_000_000:  # ignore < $1M changes
                continue

            # We need price at t_curr to place the cluster
            # Look up in the stored price snapshots if available
            # For now use linear interpolation from current price + time
            # A rough but functional approximation
            age_hours = (time.time() - t_curr) / 3600
            # Use the OI-weighted price estimate stored in history if we
            # enriched it, otherwise skip price estimation for old data
            if hasattr(hist[i], '__len__') and len(hist[i]) > 2:
                est_price = hist[i][2]  # (timestamp, oi, price) format
            else:
                continue  # need price data — skip if not enriched

            if est_price <= 0:
                continue

            # Round to nearest bucket
            bucket = round(est_price / (est_price * bucket_pct)) * (est_price * bucket_pct)
            bucket = round(bucket / (current_price * bucket_pct)) * (current_price * bucket_pct)

            price_buckets[bucket] += abs(oi_delta)

        if not price_buckets:
            return

        # Convert to liq clusters
        above_clusters: List[Tuple[float, float]] = []
        below_clusters: List[Tuple[float, float]] = []

        for price_level, accumulated_oi in price_buckets.items():
            if accumulated_oi < 5_000_000:  # minimum $5M to matter
                continue

            # Long liquidation zones: price where longs opened → they liq below
            # At 10x leverage: long opened at P liquidates at P * 0.90
            # At 5x leverage:  long opened at P liquidates at P * 0.80
            # These are BELOW current price — short targets / bear magnets
            long_liq_10x = price_level * 0.90
            long_liq_5x  = price_level * 0.80
            long_weight   = accumulated_oi * 0.7  # 70% assumed long (typical bull market)

            # Short liquidation zones: shorts opened at P → they liq above
            # At 10x: short opened at P liquidates at P * 1.10
            short_liq_10x = price_level * 1.10
            short_weight  = accumulated_oi * 0.3

            if long_liq_10x < current_price:
                below_clusters.append((long_liq_10x, long_weight))
            elif long_liq_10x > current_price:
                above_clusters.append((long_liq_10x, long_weight))

            if short_liq_10x > current_price:
                above_clusters.append((short_liq_10x, short_weight))
            elif short_liq_10x < current_price:
                below_clusters.append((short_liq_10x, short_weight))

        # Merge with existing order-book clusters
        if ci:
            all_above = ci.liq_above + above_clusters
            all_below = ci.liq_below + below_clusters

            # Deduplicate: merge clusters within 0.5% of each other
            ci.liq_above = _merge_clusters(all_above, merge_pct=0.005)[:15]
            ci.liq_below = _merge_clusters(all_below, merge_pct=0.005)[:15]


def _fmt_usd(v: float) -> str:
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    if v >= 1e3:  return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def _merge_clusters(
    clusters: List[Tuple[float, float]],
    merge_pct: float = 0.005
) -> List[Tuple[float, float]]:
    """
    Merge nearby price clusters. If two clusters are within merge_pct of
    each other, combine them into one (weighted by USD size).
    Returns sorted by USD size descending.
    """
    if not clusters:
        return []

    # Sort by price
    sorted_c = sorted(clusters, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []

    i = 0
    while i < len(sorted_c):
        price, usd = sorted_c[i]
        j = i + 1
        while j < len(sorted_c):
            next_price, next_usd = sorted_c[j]
            if abs(next_price - price) / max(price, 0.001) <= merge_pct:
                # Merge: weighted average price, sum USD
                total_usd = usd + next_usd
                price = (price * usd + next_price * next_usd) / total_usd
                usd = total_usd
                j += 1
            else:
                break
        merged.append((price, usd))
        i = j

    return sorted(merged, key=lambda x: x[1], reverse=True)


# ── Singleton ──────────────────────────────────────────────────────────────────
liquidation_analyzer = LiquidationAnalyzer()
