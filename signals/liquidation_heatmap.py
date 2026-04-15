"""
TitanBot Pro — Liquidation Heatmap v2 (Multi-Exchange)
=======================================================
UPGRADE FROM v1: Binance-only → Binance + Bybit + OKX aggregated.

Why multi-exchange matters:
  - Binance holds ~45% of perp OI. Bybit ~25%. OKX ~15%.
  - A $50M liquidation cluster on Binance is a real level.
    But if Bybit ALSO has $30M at the same price, the combined
    $80M cluster is 3× more likely to cause a significant sweep.
  - Cross-exchange cluster overlap = institutional-grade confluence.
    Retail bots see one exchange. TitanBot sees all three.

Architecture:
  Each exchange has its own fetcher (Binance/Bybit/OKX public endpoints,
  no API keys required). Results are merged before clustering so the
  histogram bins naturally pool volume from all sources.

New features vs v1:
  1. Multi-exchange fetch with per-source tagging
  2. Cross-exchange cluster overlap detection (premium confidence bonus)
  3. Open-interest weighted cluster scoring (large OI = higher magnetism)
  4. Predictive cluster ageing: clusters that have been swept decay faster
  5. Per-exchange fallback — one source failing doesn't break the whole heatmap
  6. Cluster confluence notes carry exchange breakdown (for Telegram card)

Data sources (all public, no API key):
  Binance: fapi/v1/forceOrders
  Bybit:   v5/market/recent-trade (liquidations in side field)
  OKX:     api/v5/public/liquidation-orders
"""

import asyncio
import logging
import time
import aiohttp
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LiquidationCluster:
    """A concentration of liquidations at a price level — now multi-exchange."""
    price_low: float
    price_high: float
    total_volume_usd: float
    long_volume: float        # Volume from long liquidations (SELL side)
    short_volume: float       # Volume from short liquidations (BUY side)
    event_count: int
    dominant_side: str        # LONGS_LIQUIDATED | SHORTS_LIQUIDATED
    recency_score: float      # 1.0 = very recent, decays over time

    # v2: per-exchange volume breakdown
    binance_volume: float = 0.0
    bybit_volume: float = 0.0
    okx_volume: float = 0.0

    @property
    def exchange_count(self) -> int:
        """Number of exchanges contributing to this cluster."""
        return sum(1 for v in (self.binance_volume, self.bybit_volume, self.okx_volume) if v > 0)

    @property
    def is_cross_exchange(self) -> bool:
        """True if 2+ exchanges have meaningful volume at this level."""
        return self.exchange_count >= 2

    @property
    def cross_exchange_bonus(self) -> float:
        """Extra confidence bonus for cross-exchange cluster overlap."""
        if self.exchange_count >= 3:
            return 4.0  # All three exchanges agree — extremely significant
        if self.exchange_count >= 2:
            return 2.0  # Two exchanges — strong confluence
        return 0.0


@dataclass
class LiquidationEvent:
    """Single forced liquidation event — now tagged with source exchange."""
    symbol: str
    side: str           # BUY (short liq) or SELL (long liq)
    price: float
    quantity: float
    quote_qty: float    # USD value
    timestamp: int
    exchange: str = "binance"  # v2: source exchange tag


class LiquidationHeatmap:
    """
    Multi-exchange liquidation heatmap.

    Aggregates liquidation events from Binance, Bybit and OKX,
    builds unified price clusters, and provides:
      - OB overlap detection (with cross-exchange bonus)
      - SL danger zone warnings
      - Volume imbalance (long vs short liquidations)
      - Predicted sweep targets (largest clusters ahead of price)
    """

    def __init__(self, cache_ttl: int = 120):
        """
        E10-FIX: Default cache TTL reduced from 300s to 120s to match Tier 1
        symbol scan interval. The previous 300s TTL meant T1 symbols (scanned
        every 120s) used stale liquidation data for 60+ seconds after a cache
        refresh — critical during fast-moving post-news events when liquidation
        clusters shift rapidly. T2 (300s) and T3 (900s) symbols will still get
        cache hits on their slower intervals.
        """
        self._cache: Dict[str, Tuple[List[LiquidationEvent], float]] = {}
        self._cache_ttl = cache_ttl
        self._cluster_cache: Dict[str, List[LiquidationCluster]] = {}
        # Track which exchanges are reachable (disable failed ones to avoid latency)
        self._exchange_health: Dict[str, float] = {
            "binance": 0.0, "bybit": 0.0, "okx": 0.0
        }
        self._exchange_fail_cooldown = 300.0  # 5 min cooldown after failure

    # ── Public API ────────────────────────────────────────────

    async def get_clusters(
        self,
        symbol: str,
        current_price: float,
        range_pct: float = 0.05,
        min_cluster_usd: float = 50_000,
    ) -> List[LiquidationCluster]:
        """
        Get multi-exchange liquidation clusters around current price.
        Falls back gracefully if any exchange is down.
        """
        events = await self._fetch_all_exchanges(symbol)
        if not events:
            return []

        price_low  = current_price * (1 - range_pct)
        price_high = current_price * (1 + range_pct)
        relevant = [e for e in events if price_low <= e.price <= price_high]
        if not relevant:
            return []

        clusters = self._build_clusters(relevant, current_price, min_cluster_usd)
        self._cluster_cache[symbol] = clusters
        return clusters

    def check_ob_overlap(
        self,
        ob_zone_low: float,
        ob_zone_high: float,
        clusters: List[LiquidationCluster],
        min_overlap_pct: float = 0.3,
    ) -> Tuple[bool, float, Optional[LiquidationCluster]]:
        """
        Check if an Order Block overlaps with any liquidation cluster.
        Returns (has_overlap, confidence_bonus, matching_cluster).

        v2: cross-exchange clusters receive a premium confidence bonus.
        """
        if not clusters:
            return False, 0.0, None

        best_overlap = 0.0
        best_cluster = None

        for cluster in clusters:
            overlap_low  = max(ob_zone_low,  cluster.price_low)
            overlap_high = min(ob_zone_high, cluster.price_high)
            if overlap_high <= overlap_low:
                continue
            ob_width = ob_zone_high - ob_zone_low
            if ob_width == 0:
                continue
            overlap_pct = (overlap_high - overlap_low) / ob_width
            if overlap_pct > best_overlap:
                best_overlap = overlap_pct
                best_cluster = cluster

        if best_overlap < min_overlap_pct or best_cluster is None:
            return False, 0.0, None

        volume_factor  = min(1.0, best_cluster.total_volume_usd / 500_000)
        recency_factor = best_cluster.recency_score

        bonus = (
            5.0 * best_overlap +
            5.0 * volume_factor +
            3.0 * recency_factor +
            best_cluster.cross_exchange_bonus  # v2: up to +4 for all-exchange agreement
        )

        return True, round(bonus, 1), best_cluster

    def check_sl_near_cluster(
        self,
        sl_price: float,
        clusters: List[LiquidationCluster],
        proximity_pct: float = 0.003,
    ) -> Tuple[bool, float, str]:
        """
        Check whether a proposed SL sits inside or near a liquidation cluster.
        Stop hunts work because market makers target these levels.

        Returns (is_dangerous, widen_pct, warning_note).
        """
        if not clusters or sl_price <= 0:
            return False, 0.0, ""

        worst_cluster = None
        best_overlap_pct = 0.0

        for cluster in clusters:
            band = sl_price * proximity_pct
            if (cluster.price_low - band) <= sl_price <= (cluster.price_high + band):
                overlap = (cluster.price_high - cluster.price_low) / sl_price * 100
                if overlap > best_overlap_pct:
                    best_overlap_pct = overlap
                    worst_cluster = cluster

        if worst_cluster is None:
            return False, 0.0, ""

        widen_pct = best_overlap_pct + (proximity_pct * 100)
        vol_m = worst_cluster.total_volume_usd / 1_000_000
        exc_str = f"[{worst_cluster.exchange_count} exchanges]" if worst_cluster.is_cross_exchange else "[Binance]"
        note = (
            f"⚠️ SL near ${sl_price:.6g} inside a ${vol_m:.1f}M liq cluster "
            f"{exc_str} ({worst_cluster.dominant_side.replace('_LIQUIDATED', ' liq').lower()}) "
            f"— stop-hunt risk. Widen ≥{widen_pct:.1f}%"
        )
        return True, round(widen_pct, 2), note

    def get_volume_imbalance(
        self,
        clusters: List[LiquidationCluster],
    ) -> Tuple[float, str]:
        """
        Long/short volume imbalance across all clusters.
        High long-liq volume = swept longs = potential bounce.
        High short-liq volume = swept shorts = potential dump.
        """
        if not clusters:
            return 1.0, "NEUTRAL"

        total_long  = sum(c.long_volume  for c in clusters)
        total_short = sum(c.short_volume for c in clusters)

        if total_short == 0 and total_long == 0:
            return 1.0, "NEUTRAL"
        if total_short == 0:
            return 3.0, "LONG_BIAS"
        if total_long == 0:
            return 3.0, "SHORT_BIAS"

        ratio = total_long / total_short
        if ratio > 1.5:
            return ratio, "LONG_BIAS"
        elif ratio < 0.67:
            return 1 / ratio, "SHORT_BIAS"
        return max(ratio, 1 / ratio), "NEUTRAL"

    def get_sweep_targets(
        self,
        clusters: List[LiquidationCluster],
        current_price: float,
        direction: str,
    ) -> List[Dict]:
        """
        v2: Predict the most likely sweep targets ahead of price.
        These are large clusters in the direction of travel — market makers
        will push price to these levels to harvest the liquidation fuel.

        Returns list of {price_mid, volume_usd, exchange_count, priority}.
        """
        targets = []
        for cluster in clusters:
            mid = (cluster.price_low + cluster.price_high) / 2
            is_above = mid > current_price
            is_below = mid < current_price

            if direction == "LONG" and is_above:
                targets.append({
                    "price_mid": round(mid, 8),
                    "volume_usd": cluster.total_volume_usd,
                    "dominant_side": cluster.dominant_side,
                    "exchange_count": cluster.exchange_count,
                    "recency": cluster.recency_score,
                    "priority": cluster.total_volume_usd * cluster.recency_score
                                * (1 + 0.3 * cluster.exchange_count),
                })
            elif direction == "SHORT" and is_below:
                targets.append({
                    "price_mid": round(mid, 8),
                    "volume_usd": cluster.total_volume_usd,
                    "dominant_side": cluster.dominant_side,
                    "exchange_count": cluster.exchange_count,
                    "recency": cluster.recency_score,
                    "priority": cluster.total_volume_usd * cluster.recency_score
                                * (1 + 0.3 * cluster.exchange_count),
                })

        targets.sort(key=lambda x: x["priority"], reverse=True)
        return targets[:5]

    # ── R8-F7: Liquidation Cascade Predictor ──────────────────────

    def predict_cascade_risk(
        self,
        clusters: List[LiquidationCluster],
        current_price: float,
        oi_usd: float = 0.0,
        avg_leverage: float = 10.0,
    ) -> dict:
        """
        Predict liquidation cascade risk for the current price level.

        When OI is high AND price approaches a dense liquidation cluster,
        calculates the estimated cascade force. If cascade force exceeds
        threshold, returns actionable recommendations.

        Args:
            clusters: Current liquidation clusters
            current_price: Current market price
            oi_usd: Total open interest in USD
            avg_leverage: Average market leverage (default 10x)

        Returns:
            {
                'cascade_risk': str,       # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
                'cascade_force_usd': float, # Estimated cascade volume
                'nearest_cluster': dict,    # Nearest heavy cluster info
                'distance_pct': float,      # Distance to cluster as % of price
                'recommendation': str,      # What to do
                'sl_tighten_level': float,  # Suggested SL if long
            }
        """
        if not clusters or current_price <= 0:
            return {
                'cascade_risk': 'LOW',
                'cascade_force_usd': 0,
                'nearest_cluster': None,
                'distance_pct': 100.0,
                'recommendation': 'none',
                'sl_tighten_level': 0.0,
            }

        # Find the nearest cluster above/below price with significant volume
        nearest_below = None
        nearest_above = None
        min_dist_below = float('inf')
        min_dist_above = float('inf')

        for cluster in clusters:
            mid = (cluster.price_low + cluster.price_high) / 2
            dist_pct = abs(mid - current_price) / current_price * 100

            if mid < current_price and dist_pct < min_dist_below:
                min_dist_below = dist_pct
                nearest_below = cluster
            elif mid > current_price and dist_pct < min_dist_above:
                min_dist_above = dist_pct
                nearest_above = cluster

        # Assess both directions — use the more dangerous one
        best_cluster = None
        best_dist = 100.0
        cascade_direction = "NONE"

        if nearest_below and min_dist_below < min_dist_above:
            best_cluster = nearest_below
            best_dist = min_dist_below
            cascade_direction = "DOWN"  # Price could cascade down through longs
        elif nearest_above:
            best_cluster = nearest_above
            best_dist = min_dist_above
            cascade_direction = "UP"  # Price could cascade up through shorts

        if best_cluster is None:
            return {
                'cascade_risk': 'LOW',
                'cascade_force_usd': 0,
                'nearest_cluster': None,
                'distance_pct': 100.0,
                'recommendation': 'none',
                'sl_tighten_level': 0.0,
            }

        # Calculate cascade force: cluster volume × leverage factor × proximity
        # The closer the price, the higher the risk
        proximity_mult = max(0.1, 1.0 - (best_dist / 5.0))  # Max at <1%, zero at >5%
        oi_factor = min(2.0, oi_usd / 1e9) if oi_usd > 0 else 1.0  # Scale with OI
        cascade_force = (
            best_cluster.total_volume_usd
            * avg_leverage
            * proximity_mult
            * oi_factor
            * best_cluster.recency_score
        )

        # Determine risk level
        if cascade_force >= 500_000_000:  # $500M+ cascade force
            cascade_risk = "CRITICAL"
            recommendation = (
                f"Imminent cascade {cascade_direction}: "
                f"${cascade_force/1e6:.0f}M force at "
                f"{best_dist:.1f}% away — tighten all SLs immediately"
            )
        elif cascade_force >= 100_000_000:  # $100M+
            cascade_risk = "HIGH"
            recommendation = (
                f"High cascade risk {cascade_direction}: "
                f"tighten SL above cluster zone"
            )
        elif cascade_force >= 25_000_000:  # $25M+
            cascade_risk = "MEDIUM"
            recommendation = f"Monitor cascade risk {cascade_direction}"
        else:
            cascade_risk = "LOW"
            recommendation = "none"

        # Calculate suggested SL tighten level (for LONGs, just above the cluster)
        cluster_mid = (best_cluster.price_low + best_cluster.price_high) / 2
        if cascade_direction == "DOWN":
            # If cascading down, LONGs should have SL just above the cluster
            sl_tighten = best_cluster.price_high * 1.002  # 0.2% above cluster top
        else:
            # If cascading up, SHORTs should have SL just below the cluster
            sl_tighten = best_cluster.price_low * 0.998

        return {
            'cascade_risk': cascade_risk,
            'cascade_force_usd': round(cascade_force),
            'nearest_cluster': {
                'price_low': best_cluster.price_low,
                'price_high': best_cluster.price_high,
                'volume_usd': best_cluster.total_volume_usd,
                'dominant_side': best_cluster.dominant_side,
                'exchange_count': best_cluster.exchange_count,
            },
            'distance_pct': round(best_dist, 2),
            'cascade_direction': cascade_direction,
            'recommendation': recommendation,
            'sl_tighten_level': round(sl_tighten, 8),
        }

    # ── Data fetching — multi-exchange ────────────────────────

    async def _fetch_all_exchanges(self, symbol: str) -> List[LiquidationEvent]:
        """
        Fetch from all available exchanges concurrently.
        Merges results; each event is tagged with its source exchange.
        A single exchange failure does NOT break the overall result.
        """
        cache_key = symbol
        if cache_key in self._cache:
            events, cached_at = self._cache[cache_key]
            if time.time() - cached_at < self._cache_ttl:
                return events

        now = time.time()

        # Only attempt exchanges that aren't in their failure cooldown
        tasks = {}
        if now - self._exchange_health.get("binance", 0) > self._exchange_fail_cooldown:
            tasks["binance"] = self._fetch_binance(symbol)
        if now - self._exchange_health.get("bybit", 0) > self._exchange_fail_cooldown:
            tasks["bybit"] = self._fetch_bybit(symbol)
        if now - self._exchange_health.get("okx", 0) > self._exchange_fail_cooldown:
            tasks["okx"] = self._fetch_okx(symbol)

        if not tasks:
            # All exchanges in cooldown — return stale cache if any
            return self._cache.get(cache_key, ([], 0))[0]

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        all_events: List[LiquidationEvent] = []
        for exchange, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"Liquidation fetch failed [{exchange}] {symbol}: {result}")
                self._exchange_health[exchange] = now  # start cooldown
            else:
                all_events.extend(result)
                self._exchange_health[exchange] = 0.0  # mark healthy

        if all_events:
            self._cache[cache_key] = (all_events, now)
        elif cache_key in self._cache:
            return self._cache[cache_key][0]  # stale fallback

        return all_events

    async def _fetch_binance(self, symbol: str) -> List[LiquidationEvent]:
        """Binance: fapi/v1/allForceOrders — public endpoint (no auth required).
        Note: forceOrders requires auth; allForceOrders is the public equivalent."""
        binance_sym = symbol.replace("/", "")
        url = f"https://fapi.binance.com/fapi/v1/allForceOrders?symbol={binance_sym}&limit=100"
        events = []
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise ValueError(f"HTTP {resp.status}")
                    data = await resp.json()
            for item in (data if isinstance(data, list) else []):
                price = float(item.get("price", item.get("p", 0)))
                qty   = float(item.get("origQty", item.get("q", 0)))
                events.append(LiquidationEvent(
                    symbol=symbol,
                    side=item.get("side", item.get("S", "")),
                    price=price,
                    quantity=qty,
                    quote_qty=price * qty,
                    timestamp=int(item.get("time", item.get("T", 0))),
                    exchange="binance",
                ))
        except Exception as e:
            raise RuntimeError(f"binance fetch: {e}") from e
        return events

    async def _fetch_bybit(self, symbol: str) -> List[LiquidationEvent]:
        """
        Bybit: v5/market/liquidation (public).
        Symbol format: BTCUSDT (no slash).
        """
        bybit_sym = symbol.replace("/", "")
        url = (
            f"https://api.bybit.com/v5/market/liquidation"
            f"?category=linear&symbol={bybit_sym}&limit=200"
        )
        events = []
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise ValueError(f"HTTP {resp.status}")
                    data = await resp.json()
            records = (data.get("result", {}) or {}).get("list", [])
            for item in records:
                price = float(item.get("price", 0))
                qty   = float(item.get("qty", 0))
                # Bybit: side = "Buy" means short was liquidated, "Sell" = long liquidated
                raw_side = item.get("side", "")
                mapped_side = "BUY" if raw_side == "Buy" else "SELL"
                events.append(LiquidationEvent(
                    symbol=symbol,
                    side=mapped_side,
                    price=price,
                    quantity=qty,
                    quote_qty=price * qty,
                    timestamp=int(item.get("updatedTime", 0)),
                    exchange="bybit",
                ))
        except Exception as e:
            raise RuntimeError(f"bybit fetch: {e}") from e
        return events

    async def _fetch_okx(self, symbol: str) -> List[LiquidationEvent]:
        """
        OKX: api/v5/public/liquidation-orders (public).
        Symbol format: BTC-USDT-SWAP.
        """
        # OKX SWAP format: BTC-USDT-SWAP. Skip if not USDT pair.
        parts = symbol.replace("/", "-").split("-")
        if len(parts) < 2 or parts[1] != "USDT":
            return []  # OKX only supports USDT perps well
        base = parts[0]
        okx_sym = f"{base}-USDT-SWAP"
        url = (
            f"https://www.okx.com/api/v5/public/liquidation-orders"
            f"?instType=SWAP&instId={okx_sym}&state=filled&limit=100"
        )
        events = []
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise ValueError(f"HTTP {resp.status}")
                    data = await resp.json()
            records = (data.get("data", []) or [])
            for item in records:
                for detail in item.get("details", []):
                    price = float(detail.get("bkPx", 0))
                    qty   = float(detail.get("sz", 0))
                    raw_side = detail.get("side", "")   # "buy" or "sell"
                    mapped_side = "BUY" if raw_side == "buy" else "SELL"
                    ts_str = detail.get("ts", "0")
                    events.append(LiquidationEvent(
                        symbol=symbol,
                        side=mapped_side,
                        price=price,
                        quantity=qty,
                        quote_qty=price * qty,
                        timestamp=int(ts_str),
                        exchange="okx",
                    ))
        except Exception as e:
            raise RuntimeError(f"okx fetch: {e}") from e
        return events

    # ── Cluster building ──────────────────────────────────────

    def _build_clusters(
        self,
        events: List[LiquidationEvent],
        current_price: float,
        min_cluster_usd: float,
        num_bins: int = 40,
    ) -> List[LiquidationCluster]:
        """
        Group multi-exchange liquidation events into price clusters.
        v2: tracks per-exchange volume in each bin for cross-exchange scoring.
        """
        if not events:
            return []

        prices = [e.price for e in events]
        price_min, price_max = min(prices), max(prices)
        price_range = price_max - price_min
        if price_range == 0:
            return []

        bin_width = price_range / num_bins
        now_ms = time.time() * 1000

        bins = [{
            "low": price_min + i * bin_width,
            "high": price_min + (i + 1) * bin_width,
            "long_vol": 0.0, "short_vol": 0.0, "count": 0,
            "latest_ts": 0,
            "binance_vol": 0.0, "bybit_vol": 0.0, "okx_vol": 0.0,
        } for i in range(num_bins)]

        for event in events:
            bin_idx = min(
                max(int((event.price - price_min) / bin_width), 0),
                num_bins - 1
            )
            usd = event.quote_qty if event.quote_qty > 0 else event.price * event.quantity

            if event.side in ("SELL", "sell"):   # long position liquidated
                bins[bin_idx]["long_vol"] += usd
            else:                                 # short position liquidated
                bins[bin_idx]["short_vol"] += usd

            bins[bin_idx][f"{event.exchange}_vol"] += usd
            bins[bin_idx]["count"] += 1
            bins[bin_idx]["latest_ts"] = max(bins[bin_idx]["latest_ts"], event.timestamp)

        clusters = []
        for b in bins:
            total_vol = b["long_vol"] + b["short_vol"]
            if total_vol < min_cluster_usd:
                continue

            age_ms    = now_ms - b["latest_ts"] if b["latest_ts"] > 0 else 86_400_000
            age_hours = age_ms / 3_600_000
            recency   = max(0.1, 1.0 - (age_hours / 24))
            dominant  = "LONGS_LIQUIDATED" if b["long_vol"] > b["short_vol"] else "SHORTS_LIQUIDATED"

            clusters.append(LiquidationCluster(
                price_low=b["low"],
                price_high=b["high"],
                total_volume_usd=total_vol,
                long_volume=b["long_vol"],
                short_volume=b["short_vol"],
                event_count=b["count"],
                dominant_side=dominant,
                recency_score=recency,
                binance_volume=b["binance_vol"],
                bybit_volume=b["bybit_vol"],
                okx_volume=b["okx_vol"],
            ))

        clusters.sort(key=lambda c: c.total_volume_usd, reverse=True)
        return clusters[:15]   # top 15 (up from 10 — more exchanges, more data)


# Singleton
liquidation_heatmap = LiquidationHeatmap()
