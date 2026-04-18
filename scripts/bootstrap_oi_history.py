"""
TitanBot Pro — Historical OI Bootstrap
=======================================
Fetches 90 days of historical Open Interest + price data from:
  • Bybit    — up to 200 data points per request, months of history
  • Binance  — /futures/data/openInterestHist, 30 days at 1h intervals
  • OKX      — contracts/open-interest-volume, 3 months history

Builds liquidation cluster map immediately on first run so the bot
can see deep levels like BTC $69k from day 1.

Run automatically at engine startup (async), or manually:
  python scripts/bootstrap_oi_history.py

Saves processed history to data/oi_history.json so subsequent
restarts are instant (no re-fetching needed unless file > 24 hours old).
"""

import asyncio
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import aiohttp

# Allow running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

_BINANCE = "https://fapi.binance.com"
_BYBIT   = "https://api.bybit.com"
_OKX     = "https://www.okx.com"

_CACHE_FILE  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "oi_history.json")
_CACHE_TTL   = 24 * 3600  # re-bootstrap if file > 24 hours old
_REQUEST_GAP = 0.3        # seconds between API calls

# Top coins to bootstrap (covers 95%+ of trading volume)
# We also do ALL coins the bot scans, but these get priority
_PRIORITY_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
    "MATIC", "LINK", "DOT", "UNI", "ARB", "OP", "SUI", "APT",
    "NEAR", "FIL", "ATOM", "INJ", "TIA", "SEI", "BLUR", "PEPE",
    "WIF", "BONK", "FLOKI", "JUP", "ONDO", "ENA", "HYPE", "PENGU",
    "FARTCOIN", "VIRTUAL", "BERA", "WLD", "RENDER", "FET", "TAO",
    "LTC", "BCH", "XMR", "ETC", "DASH", "ZEC", "MKR", "AAVE",
    "CRV", "GMX", "DYDX", "ICP", "HBAR", "XLM", "TRX", "VET",
]


async def _get(session: aiohttp.ClientSession, url: str,
               params: dict = None, retries: int = 2) -> Optional[any]:
    for attempt in range(retries):
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status == 200:
                    return await r.json()
                elif r.status == 429:
                    await asyncio.sleep(5)
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2)
    return None


# ── Bybit historical OI ────────────────────────────────────────────────────────

async def fetch_bybit_oi_history(
    session: aiohttp.ClientSession, coin: str
) -> List[Tuple[float, float, float]]:
    """
    Returns list of (timestamp, oi_usd, price) tuples — hourly, up to 90 days.
    Bybit GET /v5/market/open-interest supports startTime/endTime.
    """
    results = []
    symbol  = f"{coin}USDT"
    now_ms  = int(time.time() * 1000)
    start_ms = now_ms - 90 * 24 * 3600 * 1000  # 90 days ago

    # Bybit returns max 200 per request at 1h interval
    # 90 days = 2160 hours → need ~11 requests
    cursor_start = start_ms
    batch_size   = 200 * 3600 * 1000  # 200 hours in ms

    while cursor_start < now_ms:
        cursor_end = min(cursor_start + batch_size, now_ms)
        data = await _get(session, f"{_BYBIT}/v5/market/open-interest", {
            "category":    "linear",
            "symbol":      symbol,
            "intervalTime": "1h",
            "startTime":   str(cursor_start),
            "endTime":     str(cursor_end),
            "limit":       200,
        })
        if data and data.get("retCode") == 0:
            items = data.get("result", {}).get("list", [])
            for item in items:
                try:
                    ts  = int(item.get("timestamp", 0)) / 1000
                    oi  = float(item.get("openInterest", 0) or 0)
                    if ts > 0 and oi > 0:
                        results.append((ts, oi, 0.0))  # price filled separately
                except (ValueError, TypeError):
                    pass
        cursor_start = cursor_end + 1
        await asyncio.sleep(_REQUEST_GAP)

    return results


# ── Binance historical OI ──────────────────────────────────────────────────────

async def fetch_binance_oi_history(
    session: aiohttp.ClientSession, coin: str
) -> List[Tuple[float, float, float]]:
    """
    Binance /futures/data/openInterestHist — 30 days at 1h intervals.
    Returns (timestamp, oi_usd, 0.0) — price enriched separately.
    """
    results = []
    symbol = f"{coin}USDT"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 30 * 24 * 3600 * 1000  # 30 days

    # Binance: max 500 per request
    cursor = start_ms
    while cursor < now_ms:
        data = await _get(session, f"{_BINANCE}/futures/data/openInterestHist", {
            "symbol":    symbol,
            "period":    "1h",
            "limit":     500,
            "startTime": cursor,
            "endTime":   min(cursor + 500 * 3600 * 1000, now_ms),
        })
        if data and isinstance(data, list):
            for item in data:
                try:
                    ts = int(item.get("timestamp", 0)) / 1000
                    # sumOpenInterestValue is in USDT
                    oi = float(item.get("sumOpenInterestValue", 0) or 0)
                    if ts > 0 and oi > 0:
                        results.append((ts, oi, 0.0))
                except (ValueError, TypeError):
                    pass
            if len(data) < 500:
                break
            cursor = int(data[-1]["timestamp"]) + 1
        else:
            break
        await asyncio.sleep(_REQUEST_GAP)

    return results


# ── OKX historical OI ──────────────────────────────────────────────────────────

async def fetch_okx_oi_history(
    session: aiohttp.ClientSession, coin: str
) -> List[Tuple[float, float, float]]:
    """
    OKX GET /api/v5/rubik/stat/contracts/open-interest-volume
    Returns list of (timestamp_s, oi_usd, 0.0) tuples — hourly, up to 3 months.
    Response rows: [ts_ms, openInterestCoin, volume24hCoin, openInterestUsd, volume24hUsd]
    """
    results = []
    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - 90 * 24 * 3600 * 1000

    # OKX paginates backwards via the 'after' cursor (exclusive upper bound on ts)
    after: str = ""
    while True:
        params: dict = {
            "ccy":    coin,
            "period": "1H",
            "limit":  "100",
        }
        if after:
            params["after"] = after

        data = await _get(
            session,
            f"{_OKX}/api/v5/rubik/stat/contracts/open-interest-volume",
            params=params,
        )
        if not data or data.get("code") != "0":
            break

        rows = data.get("data", [])
        if not rows:
            break

        for row in rows:
            try:
                ts_ms  = int(row[0])
                oi_usd = float(row[3] or 0)   # index 3 = openInterestUsd
                if ts_ms >= start_ms and oi_usd > 0:
                    results.append((ts_ms / 1000, oi_usd, 0.0))
            except (IndexError, ValueError, TypeError):
                pass

        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break

        after = rows[-1][0]
        await asyncio.sleep(_REQUEST_GAP)

    return results


# ── Binance historical klines for price data ───────────────────────────────────

async def fetch_binance_price_history(
    session: aiohttp.ClientSession, coin: str, days: int = 90
) -> Dict[int, float]:
    """
    Fetches hourly close prices for the past `days` days.
    Returns {timestamp_hour: close_price} mapping.
    """
    prices: Dict[int, float] = {}
    symbol   = f"{coin}USDT"
    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000

    cursor = start_ms
    while cursor < now_ms:
        data = await _get(session, f"{_BINANCE}/fapi/v1/klines", {
            "symbol":    symbol,
            "interval":  "1h",
            "startTime": cursor,
            "endTime":   min(cursor + 1500 * 3600 * 1000, now_ms),
            "limit":     1500,
        })
        if data and isinstance(data, list):
            for kline in data:
                try:
                    ts_ms    = int(kline[0])
                    close_px = float(kline[4])
                    ts_hour  = ts_ms // (3600 * 1000)
                    prices[ts_hour] = close_px
                except (IndexError, ValueError, TypeError):
                    pass
            if len(data) < 1500:
                break
            cursor = int(data[-1][6]) + 1  # closeTime + 1
        else:
            break
        await asyncio.sleep(_REQUEST_GAP)

    return prices


# ── Enrich OI history with prices ─────────────────────────────────────────────

def enrich_with_prices(
    oi_points: List[Tuple[float, float, float]],
    price_map: Dict[int, float]
) -> List[Tuple[float, float, float]]:
    """
    Merge OI snapshots with price data.
    Returns (timestamp, oi_usd, price) tuples, skipping points with no price.
    """
    enriched = []
    for ts, oi, _ in oi_points:
        ts_hour = int(ts) // 3600
        # Search for nearest available hourly price within a ±2h window.
        # Expand outward (0, 1, -1, 2, -2) so the closest match wins.
        price = price_map.get(ts_hour)
        if price is None:
            for offset in (1, -1, 2, -2):
                price = price_map.get(ts_hour + offset)
                if price is not None:
                    break
        if price:
            enriched.append((ts, oi, price))
    return enriched


# ── Build liquidation clusters from enriched history ──────────────────────────

def build_clusters(
    coin: str,
    history: List[Tuple[float, float, float]],
    current_price: float,
) -> List[Tuple[float, float, str]]:
    """
    Compute deep liquidation clusters from OI + price history.

    When OI increased significantly at price level P:
      → Longs were opened at P → will liquidate at P*(1-1/lev)
      → Shorts were opened at P → will liquidate at P*(1+1/lev)

    We use a mix of 5x, 10x, 20x leverage assumptions weighted realistically.
    Returns list of (price, usd_size, 'above'|'below') relative to current price.
    """
    if len(history) < 5 or current_price <= 0:
        return []

    # Sort by timestamp
    history = sorted(history, key=lambda x: x[0])

    # Bucket price levels (1% wide buckets)
    bucket_pct = 0.01
    buckets: Dict[float, float] = defaultdict(float)

    prev_oi = history[0][1]
    for ts, oi, price in history[1:]:
        oi_delta = oi - prev_oi
        prev_oi  = oi

        if oi_delta < 2_000_000:  # ignore small OI changes < $2M
            continue

        # Log-space 1% bucketing: groups prices within ~1% of each other
        # into the same cluster regardless of the absolute price level.
        # This works correctly across all price scales (e.g. BTC at $30k vs $70k).
        log_q = math.log(1.0 + bucket_pct)           # ≈ 0.00995 for 1%
        bucket_idx = round(math.log(price) / log_q)
        bucket = math.exp(bucket_idx * log_q)
        buckets[bucket] += abs(oi_delta)

    if not buckets:
        return []

    clusters: List[Tuple[float, float, str]] = []
    for price_level, oi_accumulated in buckets.items():
        if oi_accumulated < 10_000_000:  # need $10M+ to be significant
            continue

        # Leverage distribution: 20% at 5x, 50% at 10x, 30% at 20x
        # Long liquidation prices (below entry)
        long_liq_5x  = price_level * (1 - 1/5)   # -20%
        long_liq_10x = price_level * (1 - 1/10)  # -10%
        long_liq_20x = price_level * (1 - 1/20)  # -5%

        # Short liquidation prices (above entry)
        short_liq_5x  = price_level * (1 + 1/5)   # +20%
        short_liq_10x = price_level * (1 + 1/10)  # +10%
        short_liq_20x = price_level * (1 + 1/20)  # +5%

        # Assume 65% long / 35% short (typical bull market)
        long_oi  = oi_accumulated * 0.65
        short_oi = oi_accumulated * 0.35

        long_clusters = [
            (long_liq_5x,  long_oi * 0.20),
            (long_liq_10x, long_oi * 0.50),
            (long_liq_20x, long_oi * 0.30),
        ]
        short_clusters = [
            (short_liq_5x,  short_oi * 0.20),
            (short_liq_10x, short_oi * 0.50),
            (short_liq_20x, short_oi * 0.30),
        ]

        for p, usd in long_clusters:
            side = "below" if p < current_price else "above"
            clusters.append((p, usd, side))

        for p, usd in short_clusters:
            side = "above" if p > current_price else "below"
            clusters.append((p, usd, side))

    return clusters


# ── Main bootstrap function ────────────────────────────────────────────────────

async def bootstrap(
    coins: Optional[List[str]] = None,
    force_refresh: bool = False,
    progress_cb=None
) -> Tuple[Dict, Dict]:
    """
    Main bootstrap entry point.
    Returns (history_dict, clusters_dict) for loading into LiquidationAnalyzer.

    Saves/loads from cache file to avoid re-fetching on every restart.
    """
    # Load from cache if fresh enough
    if not force_refresh and os.path.exists(_CACHE_FILE):
        try:
            file_age = time.time() - os.path.getmtime(_CACHE_FILE)
            if file_age < _CACHE_TTL:
                with open(_CACHE_FILE, "r") as f:
                    cached = json.load(f)
                history  = {k: [tuple(p) for p in v]
                            for k, v in cached.get("history", {}).items()}
                clusters = {k: [tuple(p) for p in v]
                            for k, v in cached.get("clusters", {}).items()}
                age_h = file_age / 3600
                logger.info(
                    f"📊 OI history loaded from cache "
                    f"({len(history)} coins, {age_h:.1f}h old)"
                )
                return history, clusters
        except Exception as e:
            logger.warning(f"OI cache load failed, re-fetching: {e}")

    target_coins = coins or _PRIORITY_COINS
    logger.info(f"📊 Bootstrapping OI history for {len(target_coins)} coins...")

    history:  Dict[str, List[Tuple]] = {}
    clusters: Dict[str, List[Tuple]] = {}

    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": "TitanBot/3.0"},
    ) as session:

        for i, coin in enumerate(target_coins):
            try:
                if progress_cb:
                    progress_cb(coin, i, len(target_coins))

                # 1. Fetch price history first (needed to enrich OI)
                price_map = await fetch_binance_price_history(session, coin, days=90)
                if not price_map:
                    logger.debug(f"OI bootstrap: no price data for {coin}, skipping")
                    continue

                current_price = 0
                # Use most recent price
                if price_map:
                    latest_hour = max(price_map.keys())
                    current_price = price_map[latest_hour]

                # 2. Fetch OI history (Bybit primary, Binance + OKX supplement)
                bybit_oi = await fetch_bybit_oi_history(session, coin)
                await asyncio.sleep(_REQUEST_GAP)

                binance_oi = await fetch_binance_oi_history(session, coin)
                await asyncio.sleep(_REQUEST_GAP)

                okx_oi = await fetch_okx_oi_history(session, coin)
                await asyncio.sleep(_REQUEST_GAP)

                # 3. Enrich with prices
                bybit_enriched   = enrich_with_prices(bybit_oi, price_map)
                binance_enriched = enrich_with_prices(binance_oi, price_map)
                okx_enriched     = enrich_with_prices(okx_oi, price_map)

                # Merge: prefer Bybit (longest history), supplement with Binance and OKX.
                # Deduplicate by rounding timestamps to nearest hour.
                all_points: Dict[int, Tuple] = {}
                for ts, oi, px in bybit_enriched:
                    bucket = int(ts) // 3600
                    all_points[bucket] = (ts, oi, px)
                for ts, oi, px in binance_enriched:
                    bucket = int(ts) // 3600
                    if bucket not in all_points:
                        all_points[bucket] = (ts, oi, px)
                for ts, oi, px in okx_enriched:
                    bucket = int(ts) // 3600
                    if bucket not in all_points:
                        all_points[bucket] = (ts, oi, px)

                merged = sorted(all_points.values(), key=lambda x: x[0])

                if len(merged) < 5:
                    logger.debug(f"OI bootstrap: insufficient data for {coin}")
                    continue

                history[coin] = merged

                # 4. Build liquidation clusters
                coin_clusters = build_clusters(coin, merged, current_price)
                if coin_clusters:
                    clusters[coin] = coin_clusters

                logger.debug(
                    f"OI bootstrap {coin}: {len(merged)} points, "
                    f"{len(coin_clusters)} clusters, "
                    f"price=${current_price:.4f}"
                )

            except Exception as e:
                logger.warning(f"OI bootstrap error for {coin}: {e}")
                continue

            # Brief pause between coins to be polite
            await asyncio.sleep(0.5)

    # Save to cache
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump({"history": history, "clusters": clusters,
                       "bootstrapped_at": time.time()}, f)
        logger.info(
            f"📊 OI bootstrap complete: {len(history)} coins, "
            f"{len(clusters)} with cluster data. Saved to cache."
        )
    except Exception as e:
        logger.warning(f"OI cache save failed: {e}")

    return history, clusters


async def bootstrap_and_load(coins: Optional[List[str]] = None,
                              force: bool = False) -> bool:
    """
    Convenience function: bootstrap + load into the singleton analyzer.
    Returns True if successful.
    """
    try:
        from analyzers.liquidation_analyzer import liquidation_analyzer

        def _progress(coin, i, total):
            if i % 10 == 0:
                logger.info(f"📊 OI bootstrap: {i}/{total} ({coin})")

        history, clusters = await bootstrap(
            coins=coins, force_refresh=force, progress_cb=_progress
        )
        if history:
            liquidation_analyzer.load_history(history, clusters)
            return True
        return False
    except Exception as e:
        logger.warning(f"bootstrap_and_load failed: {e}")
        return False


# ── Standalone execution ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    parser = argparse.ArgumentParser(description="Bootstrap OI history for TitanBot")
    parser.add_argument("--force",  action="store_true",
                        help="Force re-fetch even if cache is fresh")
    parser.add_argument("--coins",  nargs="+", default=None,
                        help="Specific coins to bootstrap (default: all priority coins)")
    parser.add_argument("--top",    type=int, default=None,
                        help="Bootstrap only top N priority coins")
    args = parser.parse_args()

    coins = args.coins
    if args.top and not coins:
        coins = _PRIORITY_COINS[:args.top]

    async def _main():
        def progress(coin, i, total):
            pct = (i + 1) / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r[{bar}] {pct:.0f}% — {coin:<12}", end="", flush=True)

        print(f"Bootstrapping OI history for {len(coins or _PRIORITY_COINS)} coins...")
        history, clusters = await bootstrap(
            coins=coins, force_refresh=args.force, progress_cb=progress
        )
        print()
        print(f"\n✅ Done: {len(history)} coins bootstrapped, "
              f"{len(clusters)} with cluster data")

        # Print top clusters for BTC as a sample
        if "BTC" in clusters:
            btc = clusters["BTC"]
            above = sorted([(p, u) for p, u, s in btc if s == "above"], key=lambda x: x[0])[:5]
            below = sorted([(p, u) for p, u, s in btc if s == "below"],
                           key=lambda x: x[0], reverse=True)[:5]
            print(f"\nBTC Liquidation Clusters:")
            print(f"  Above current price:")
            for p, u in above:
                bar = "█" * int(u / 5e8)
                print(f"    ${p:,.0f}  ${u/1e6:.0f}M  {bar}")
            print(f"  Below current price:")
            for p, u in below:
                bar = "█" * int(u / 5e8)
                print(f"    ${p:,.0f}  ${u/1e6:.0f}M  {bar}")

    asyncio.run(_main())
