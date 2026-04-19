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
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
_ONE_HOUR_MS = 3600 * 1000

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
               params: dict = None, retries: int = 2) -> Optional[Any]:
    for attempt in range(retries):
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status == 200:
                    return await r.json()
                elif r.status == 429:
                    await asyncio.sleep(5)
                else:
                    logger.warning(f"HTTP {r.status} from {url} params={params}")
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
        if data and str(data.get("retCode")) == "0":
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
            last_ts = int(data[-1].get("timestamp", 0))
            next_cursor = last_ts + 1
            if next_cursor <= cursor or next_cursor - cursor < _ONE_HOUR_MS:
                logger.warning(
                    f"Binance OI cursor stalled for {symbol}: cursor={cursor}, last_ts={last_ts}"
                )
                break
            cursor = next_cursor
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

        next_after = rows[-1][0]
        if after and next_after == after:
            logger.warning(f"OKX OI cursor stalled for {coin}: after={after}")
            break
        after = next_after
        await asyncio.sleep(_REQUEST_GAP)

    return results


# ── Long/short ratio helpers ────────────────────────────────────────────────────

def _ratio_to_long_share(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    if v <= 1:
        return v
    return v / (1.0 + v)


async def fetch_binance_long_ratio(
    session: aiohttp.ClientSession, coin: str
) -> Optional[float]:
    symbol = f"{coin}USDT"
    data = await _get(session, f"{_BINANCE}/futures/data/globalLongShortAccountRatio", {
        "symbol": symbol,
        "period": "1h",
        "limit": 30,
    })
    if not data or not isinstance(data, list):
        return None

    samples: List[float] = []
    for item in data[-12:]:
        long_share = _ratio_to_long_share(item.get("longAccount"))
        if long_share is None:
            long_share = _ratio_to_long_share(item.get("longShortRatio"))
        if long_share is not None and 0.0 < long_share < 1.0:
            samples.append(long_share)
    return float(statistics.median(samples)) if samples else None


async def fetch_bybit_long_ratio(
    session: aiohttp.ClientSession, coin: str
) -> Optional[float]:
    symbol = f"{coin}USDT"
    data = await _get(session, f"{_BYBIT}/v5/market/account-ratio", {
        "category": "linear",
        "symbol": symbol,
        "period": "1h",
        "limit": 30,
    })
    if not data or str(data.get("retCode")) != "0":
        return None

    rows = data.get("result", {}).get("list", []) or []
    samples: List[float] = []
    for row in rows[-12:]:
        buy_ratio = row.get("buyRatio") or row.get("longAccountRatio") or row.get("longAccount")
        sell_ratio = row.get("sellRatio") or row.get("shortAccountRatio") or row.get("shortAccount")
        long_share = _ratio_to_long_share(buy_ratio)
        short_share = _ratio_to_long_share(sell_ratio)
        if long_share is not None and short_share is not None and (long_share + short_share) > 0:
            samples.append(long_share / (long_share + short_share))
            continue
        if long_share is not None and 0.0 < long_share < 1.0:
            samples.append(long_share)
            continue
        ratio_share = _ratio_to_long_share(row.get("buySellRatio"))
        if ratio_share is not None and 0.0 < ratio_share < 1.0:
            samples.append(ratio_share)

    return float(statistics.median(samples)) if samples else None


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
            next_cursor = int(data[-1][6]) + 1  # closeTime + 1
            if next_cursor <= cursor:
                logger.warning(
                    f"Binance klines cursor stalled for {symbol}: cursor={cursor}"
                )
                break
            cursor = next_cursor
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


# ── Merge near-identical liquidation clusters ─────────────────────────────────

def _merge_clusters(
    clusters: List[Tuple[float, float, str]],
    band_pct: float = 0.01,
) -> List[Tuple[float, float, str]]:
    """
    Merge clusters that fall within `band_pct` of each other (default ±1%).
    Price is the weighted average of merged members; USD size is summed;
    side is taken from the member with the largest USD contribution.
    """
    if not clusters:
        return []
    sorted_c = sorted(clusters, key=lambda x: x[0])
    merged: List[Tuple[float, float, str]] = []
    band_price, band_usd, band_side = sorted_c[0]
    for price, usd, side in sorted_c[1:]:
        if abs(price - band_price) / band_price <= band_pct:
            total_usd  = band_usd + usd
            band_price = (band_price * band_usd + price * usd) / total_usd
            if usd > band_usd:
                band_side = side
            band_usd = total_usd
        else:
            merged.append((band_price, band_usd, band_side))
            band_price, band_usd, band_side = price, usd, side
    merged.append((band_price, band_usd, band_side))
    return merged


# ── Build liquidation clusters from enriched history ──────────────────────────

def build_clusters(
    coin: str,
    history: List[Tuple[float, float, float]],
    current_price: float,
    long_ratio: float = 0.65,
) -> List[Tuple[float, float, str]]:
    """
    Compute deep liquidation clusters from OI + price history.

    When OI increased significantly at price level P:
      → Longs were opened at P → will liquidate at P*(1-1/lev)
      → Shorts were opened at P → will liquidate at P*(1+1/lev)

    We use asset-aware leverage mixes (BTC/ETH lower leverage, smaller alts higher leverage).
    OI deltas are age-weighted (exponential decay) so stale data contributes
    less than recent accumulation.  The decay constant τ is asset-aware:
      • BTC/ETH: τ=30 days (structurally stable; older OI stays relevant)
      • Alts:    τ=21 days (faster cycling; stale accumulation decays sooner)

    Returns list of (price, usd_size, 'above'|'below') relative to current price.
    Clusters with effective OI < $10M are filtered out.
    """
    if len(history) < 5 or current_price <= 0:
        return []

    # Sort by timestamp
    history = sorted(history, key=lambda x: x[0])
    oi_values = [float(p[1]) for p in history if len(p) >= 2 and float(p[1]) > 0]
    if not oi_values:
        return []
    median_oi = float(statistics.median(oi_values))
    # Scale sensitivity by each coin's baseline OI:
    # 0.5% of median OI, capped at $2M and floored at $100k.
    min_oi_delta = max(100_000.0, min(2_000_000.0, median_oi * 0.005))

    # Age-decay parameters — asset-aware:
    # BTC/ETH are more structurally stable; OI from 30 days ago is still relevant.
    # Alts cycle faster; use a 21-day τ so stale accumulation decays sooner.
    now_ts      = time.time()
    _coin_upper = coin.upper() if coin else ""
    if any(_coin_upper.startswith(m) for m in ("BTC", "ETH")):
        tau_days = 30.0
    else:
        tau_days = 21.0
    tau_secs = tau_days * 24 * 3600.0

    # Bucket price levels (1% wide log-space buckets)
    bucket_pct = 0.01
    log_q      = math.log(1.0 + bucket_pct)   # ≈ 0.00995
    buckets: Dict[float, float] = defaultdict(float)

    prev_oi = history[0][1]
    for ts, oi, price in history[1:]:
        oi_delta = oi - prev_oi
        prev_oi  = oi

        if oi_delta < min_oi_delta:
            continue

        # Exponential age weight: recent data → weight ≈ 1.0,
        # data τ days old → weight ≈ 0.37, data 2τ days old → weight ≈ 0.14.
        # τ=30d for BTC/ETH, τ=21d for alts (see asset-aware block above).
        age_secs = max(0.0, now_ts - ts)
        weight   = math.exp(-age_secs / tau_secs)

        # Log-space bucketing (price-scale invariant)
        bucket_idx = round(math.log(price) / log_q)
        bucket     = math.exp(bucket_idx * log_q)
        buckets[bucket] += abs(oi_delta) * weight

    if not buckets:
        return []

    clusters: List[Tuple[float, float, str]] = []
    for price_level, oi_accumulated in buckets.items():
        if oi_accumulated < 10_000_000:  # need $10M+ (age-weighted) to be significant
            continue

        if any(_coin_upper.startswith(m) for m in ("BTC", "ETH")):
            # BTC/ETH tend to carry lower average leverage than smaller-cap alts.
            leverage_mix = [(5, 0.35), (10, 0.45), (20, 0.20)]
        elif any(_coin_upper.startswith(m) for m in ("SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK")):
            # Large liquid alts: balanced mix with a small high-leverage tail.
            leverage_mix = [(5, 0.20), (10, 0.45), (20, 0.30), (50, 0.05)]
        else:
            # Smaller caps: leverage skews higher, including 50x/100x activity.
            leverage_mix = [(10, 0.30), (20, 0.45), (50, 0.20), (100, 0.05)]
        if abs(sum(w for _, w in leverage_mix) - 1.0) > 1e-9:
            logger.warning(f"Invalid leverage mix weights for {coin}: {leverage_mix}")
            continue

        safe_long_ratio = min(0.8, max(0.2, float(long_ratio)))
        long_oi  = oi_accumulated * safe_long_ratio
        short_oi = oi_accumulated * (1.0 - safe_long_ratio)

        long_clusters = [
            (price_level * (1 - 1 / lev), long_oi * w)
            for lev, w in leverage_mix
        ]
        short_clusters = [
            (price_level * (1 + 1 / lev), short_oi * w)
            for lev, w in leverage_mix
        ]

        for p, usd in long_clusters:
            side = "below" if p < current_price else "above"
            clusters.append((p, usd, side))

        for p, usd in short_clusters:
            side = "above" if p > current_price else "below"
            clusters.append((p, usd, side))

    return _merge_clusters(clusters)


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

                # 2b. Pull live long/short account ratios to avoid hardcoded bias.
                binance_long_ratio = await fetch_binance_long_ratio(session, coin)
                await asyncio.sleep(_REQUEST_GAP)
                bybit_long_ratio = await fetch_bybit_long_ratio(session, coin)
                await asyncio.sleep(_REQUEST_GAP)
                ratio_samples = [r for r in (binance_long_ratio, bybit_long_ratio) if r is not None]
                long_ratio = float(statistics.median(ratio_samples)) if ratio_samples else 0.65

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
                coin_clusters = build_clusters(coin, merged, current_price, long_ratio=long_ratio)
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
