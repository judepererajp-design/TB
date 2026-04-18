"""
TitanBot Pro — Backtest Data Loader
=====================================
Loads historical OHLCV data from:
  1. Local CSV files (fastest — use for repeat runs)
  2. Binance Futures API (downloads and caches to CSV)
  3. SQLite database (existing TitanBot data)

Output format matches what strategies expect:
  Dict[timeframe, List[candle]]
  where candle = [timestamp, open, high, low, close, volume]
"""

import os
import logging
import asyncio
import csv
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import numpy as np

logger = logging.getLogger(__name__)

# Default data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Filename-safe symbol sanitiser (handles BTC/USDT, BTC/USDT:USDT, paths on Windows)
_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_symbol(symbol: str) -> str:
    """Return a filename-safe rendering of a trading pair."""
    return _FILENAME_UNSAFE.sub("_", symbol).strip("_")


class BacktestDataLoader:
    """
    Load OHLCV data for backtesting.
    Handles multiple symbols and timeframes.
    """

    # Retry knobs for transient Binance / network errors. We back off on every
    # failure rather than silently truncating the result set, which is what the
    # previous implementation did (a single 429 / TLS reset would short-circuit
    # the entire history fetch and produce a partial CSV that then poisoned
    # every subsequent cached run).
    MAX_FETCH_RETRIES = 5
    BACKOFF_BASE_SECS = 1.0
    BACKOFF_MAX_SECS = 30.0

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    async def load(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: str,      # "2024-01-01"
        end_date: str = None,  # "2024-12-31" or None for today
        source: str = "binance",
    ) -> Dict[str, List]:
        """
        Load OHLCV data for a symbol across multiple timeframes.

        Returns:
            Dict mapping timeframe -> list of candles
        """
        result = {}
        for tf in timeframes:
            data = await self._load_timeframe(
                symbol, tf, start_date, end_date, source
            )
            if data:
                result[tf] = data

        return result

    async def _load_timeframe(
        self, symbol, tf, start_date, end_date, source
    ) -> List:
        """Load a single timeframe"""

        # Check for cached CSV first.
        # FIX: when end_date is None we resolve to *today's* UTC date so that
        # "latest" doesn't permanently freeze a stale cache. Older "latest"
        # files are still picked up as a fallback, but only when no cache for
        # today exists — that lets reruns the next day hit Binance for the
        # missing tail rather than silently returning yesterday's history.
        csv_path = self._csv_path(symbol, tf, start_date, end_date)
        if os.path.exists(csv_path):
            logger.info(f"Loading cached data: {csv_path}")
            return self._read_csv(csv_path)

        # Backwards-compat: pick up legacy "..._latest.csv" if present and the
        # caller did not pin an explicit end_date.
        if end_date is None:
            legacy = os.path.join(
                self.data_dir,
                f"{_safe_symbol(symbol)}_{tf}_{start_date}_latest.csv",
            )
            if os.path.exists(legacy) and legacy != csv_path:
                logger.info(f"Loading legacy cached data: {legacy}")
                return self._read_csv(legacy)

        if source == "binance":
            data = await self._fetch_binance(symbol, tf, start_date, end_date)
            if data:
                self._write_csv(csv_path, data)
            return data
        elif source == "csv":
            # User must provide CSV in data_dir
            alt_path = os.path.join(
                self.data_dir,
                f"{_safe_symbol(symbol)}_{tf}.csv"
            )
            if os.path.exists(alt_path):
                return self._read_csv(alt_path)
            logger.warning(f"CSV not found: {alt_path}")
            return []
        else:
            logger.warning(f"Unknown source: {source}")
            return []

    async def _fetch_binance(
        self, symbol, tf, start_date, end_date
    ) -> List:
        """
        Fetch historical data from Binance Futures.
        Uses CCXT's fetch_ohlcv with pagination and retries with exponential
        back-off on transient errors so we never silently return a truncated
        history (which would otherwise be cached and poison future runs).
        """
        exchange = None
        try:
            import ccxt.async_support as ccxt

            exchange = ccxt.binanceusdm({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
            })

            # Parse dates
            start_ts = int(
                datetime.strptime(start_date, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc)
                .timestamp() * 1000
            )
            if end_date:
                end_ts = int(
                    datetime.strptime(end_date, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp() * 1000
                )
            else:
                end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

            # Validate date range
            if start_ts >= end_ts:
                logger.error(
                    f"Invalid date range: start {start_date} >= end {end_date or 'now'}"
                )
                return []

            all_candles: List = []
            since = start_ts
            limit = 1000
            consecutive_empty = 0

            logger.info(
                f"Fetching {symbol} {tf} from {start_date} to "
                f"{end_date or 'now'}..."
            )

            while since < end_ts:
                candles = await self._fetch_with_retry(
                    exchange, symbol, tf, since, limit,
                )

                if candles is None:
                    # Hard failure after retries. Stop and surface partial
                    # data with a loud warning rather than caching it.
                    logger.error(
                        f"Aborting {symbol} {tf} fetch at {since}: "
                        f"all {self.MAX_FETCH_RETRIES} retries exhausted"
                    )
                    return []

                if not candles:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        # Three empty pages in a row => no more data on the
                        # exchange (e.g. symbol listed mid-range). Done.
                        break
                    # Skip ahead one bar-width to avoid an infinite loop on
                    # gaps in Binance's history.
                    since += self._bar_width_ms(tf)
                    await asyncio.sleep(0.2)
                    continue
                consecutive_empty = 0

                all_candles.extend(candles)

                # Move to next batch
                last_ts = candles[-1][0]
                if last_ts <= since:
                    # No forward progress (would loop forever). Stop here
                    # rather than silently re-requesting the same window.
                    break
                since = last_ts + 1

                # Rate limit
                await asyncio.sleep(0.2)

            # Filter to date range, deduplicate, and sort
            seen_ts = set()
            filtered = []
            for c in all_candles:
                ts = c[0]
                if ts in seen_ts:
                    continue
                if start_ts <= ts <= end_ts:
                    seen_ts.add(ts)
                    filtered.append(c)

            filtered.sort(key=lambda x: x[0])

            # Sanity-check monotonicity (post-sort, post-dedupe).
            if filtered:
                bar_ms = self._bar_width_ms(tf)
                gaps = sum(
                    1 for i in range(1, len(filtered))
                    if filtered[i][0] - filtered[i - 1][0] > bar_ms * 2
                )
                if gaps:
                    logger.warning(
                        f"{symbol} {tf}: {gaps} multi-bar gaps detected in "
                        f"{len(filtered)} candles (possible exchange downtime)"
                    )

            logger.info(f"Loaded {len(filtered)} candles for {symbol} {tf}")
            return filtered

        except ImportError:
            logger.error("CCXT not installed. pip install ccxt")
            return []
        except Exception as e:
            logger.error(f"Binance fetch failed: {e}")
            return []
        finally:
            # Always close the exchange connection to prevent resource leaks
            if exchange is not None:
                try:
                    await exchange.close()
                except Exception:
                    pass

    async def _fetch_with_retry(
        self, exchange, symbol: str, tf: str, since: int, limit: int,
    ) -> Optional[List]:
        """
        Fetch one page of OHLCV with exponential back-off on transient errors.

        Returns the candle list on success (possibly empty), or ``None`` after
        all retries are exhausted so the caller can fail-loud.
        """
        delay = self.BACKOFF_BASE_SECS
        for attempt in range(1, self.MAX_FETCH_RETRIES + 1):
            try:
                return await exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
            except Exception as e:  # ccxt.NetworkError, RateLimitExceeded, etc.
                msg = str(e)
                # Treat 4xx parameter errors as terminal — retrying won't help.
                if any(tok in msg for tok in ("BadSymbol", "InvalidSymbol", "BadRequest")):
                    logger.error(f"Terminal fetch error for {symbol} {tf}: {e}")
                    return None
                if attempt >= self.MAX_FETCH_RETRIES:
                    logger.error(
                        f"Fetch failed at since={since} after {attempt} attempts: {e}"
                    )
                    return None
                logger.warning(
                    f"Fetch error at since={since} (attempt {attempt}/"
                    f"{self.MAX_FETCH_RETRIES}): {e}; sleeping {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.BACKOFF_MAX_SECS)
        return None

    @staticmethod
    def _bar_width_ms(tf: str) -> int:
        """Map a CCXT-style timeframe like '1h' / '15m' / '1d' to milliseconds."""
        unit = tf[-1].lower()
        try:
            n = int(tf[:-1])
        except ValueError:
            return 60_000  # default to 1m to be safe
        mult = {
            's': 1_000,
            'm': 60_000,
            'h': 3_600_000,
            'd': 86_400_000,
            'w': 604_800_000,
        }.get(unit, 60_000)
        return max(1, n) * mult

    def _csv_path(self, symbol, tf, start_date, end_date) -> str:
        """Generate CSV filename for caching.

        ``end_date=None`` resolves to today's UTC date so that the cache is
        bucketed by day; rerunning tomorrow misses today's file and goes back
        to the exchange to fetch the new tail.
        """
        sym = _safe_symbol(symbol)
        if end_date:
            end = end_date
        else:
            end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.data_dir, f"{sym}_{tf}_{start_date}_{end}.csv")

    def _write_csv(self, path: str, data: List):
        """Write OHLCV data to CSV"""
        parent = os.path.dirname(path)
        if parent:
            # Only call makedirs when there is a directory component.
            # If path is a bare filename (no directory), os.path.dirname returns ''
            # and os.makedirs('') raises FileNotFoundError on all platforms.
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for candle in data:
                writer.writerow(candle[:6])
        logger.info(f"Saved {len(data)} candles to {path}")

    def _read_csv(self, path: str) -> List:
        """Read OHLCV data from CSV.

        Enforces strict timestamp monotonicity and de-duplicates on the way in
        — a misordered or duplicate row in the cache would otherwise cause the
        backtest engine to compute negative bar deltas and wildly wrong
        equity curves.
        """
        data = []
        last_ts: Optional[int] = None
        seen_ts: set = set()
        out_of_order = 0
        duplicates = 0
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for lineno, row in enumerate(reader, start=2):
                    if len(row) < 6:
                        continue
                    try:
                        ts = int(float(row[0]))
                        candle = [
                            ts,
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                        ]
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping malformed row {lineno} in {path}: {e}")
                        continue

                    if ts in seen_ts:
                        duplicates += 1
                        continue
                    if last_ts is not None and ts < last_ts:
                        out_of_order += 1
                        # Keep going but track — we'll re-sort at the end.
                    seen_ts.add(ts)
                    last_ts = ts if last_ts is None else max(last_ts, ts)
                    data.append(candle)
        except OSError as e:
            logger.error(f"Cannot read CSV {path}: {e}")
            return []

        if out_of_order:
            logger.warning(
                f"{path}: {out_of_order} out-of-order rows; sorting by timestamp"
            )
            data.sort(key=lambda c: c[0])
        if duplicates:
            logger.warning(f"{path}: dropped {duplicates} duplicate rows")
        return data

    def create_walk_forward_splits(
        self,
        data: List,
        train_pct: float = 0.7,
        n_splits: int = 3,
    ) -> List[Tuple[List, List]]:
        """
        Create walk-forward train/test splits for robust optimization.

        FIX: previously windows overlapped (``start = i * split_size // 2``)
        which caused leakage — a sample appeared in two test sets and three
        train sets, inflating apparent OOS performance. We now build
        non-overlapping anchored walk-forward windows: every test set is
        strictly later than its train set, and no two test sets share any
        bars.
        """
        total = len(data)
        if total < 100 or n_splits < 1:
            return []

        # Reserve room so each split's test window is at least ~20 bars and
        # train >= 50 bars, then carve the dataset into ``n_splits`` slabs.
        slab_size = total // n_splits
        if slab_size < 70:
            return []

        splits: List[Tuple[List, List]] = []
        for i in range(n_splits):
            start = i * slab_size
            end = total if i == n_splits - 1 else (i + 1) * slab_size
            slab = data[start:end]

            train_end = int(len(slab) * train_pct)
            train = slab[:train_end]
            test = slab[train_end:]

            if len(train) > 50 and len(test) > 20:
                splits.append((train, test))

        return splits


# Singleton
data_loader = BacktestDataLoader()
