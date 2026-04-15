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
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import numpy as np

logger = logging.getLogger(__name__)

# Default data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class BacktestDataLoader:
    """
    Load OHLCV data for backtesting.
    Handles multiple symbols and timeframes.
    """

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

        # Check for cached CSV first
        csv_path = self._csv_path(symbol, tf, start_date, end_date)
        if os.path.exists(csv_path):
            logger.info(f"Loading cached data: {csv_path}")
            return self._read_csv(csv_path)

        if source == "binance":
            data = await self._fetch_binance(symbol, tf, start_date, end_date)
            if data:
                self._write_csv(csv_path, data)
            return data
        elif source == "csv":
            # User must provide CSV in data_dir
            alt_path = os.path.join(
                self.data_dir,
                f"{symbol.replace('/', '_')}_{tf}.csv"
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
        Uses CCXT's fetch_ohlcv with pagination.
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

            all_candles = []
            since = start_ts
            limit = 1000

            logger.info(
                f"Fetching {symbol} {tf} from {start_date} to "
                f"{end_date or 'now'}..."
            )

            while since < end_ts:
                try:
                    candles = await exchange.fetch_ohlcv(
                        symbol, tf, since=since, limit=limit
                    )
                except Exception as e:
                    logger.error(f"Fetch error at {since}: {e}")
                    break

                if not candles:
                    break

                all_candles.extend(candles)

                # Move to next batch
                last_ts = candles[-1][0]
                if last_ts <= since:
                    break
                since = last_ts + 1

                # Rate limit
                await asyncio.sleep(0.2)

            # Filter to date range and deduplicate
            seen_ts = set()
            filtered = []
            for c in all_candles:
                if c[0] not in seen_ts and start_ts <= c[0] <= end_ts:
                    seen_ts.add(c[0])
                    filtered.append(c)

            filtered.sort(key=lambda x: x[0])
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

    def _csv_path(self, symbol, tf, start_date, end_date) -> str:
        """Generate CSV filename for caching"""
        sym = symbol.replace("/", "_")
        end = end_date or "latest"
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
        """Read OHLCV data from CSV"""
        data = []
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for lineno, row in enumerate(reader, start=2):
                    if len(row) < 6:
                        continue
                    try:
                        data.append([
                            int(float(row[0])),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                        ])
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping malformed row {lineno} in {path}: {e}")
        except OSError as e:
            logger.error(f"Cannot read CSV {path}: {e}")
        return data

    def create_walk_forward_splits(
        self,
        data: List,
        train_pct: float = 0.7,
        n_splits: int = 3,
    ) -> List[Tuple[List, List]]:
        """
        Create walk-forward train/test splits for robust optimization.
        Each split has a training window followed by an out-of-sample test.
        """
        total = len(data)
        split_size = total // n_splits
        splits = []

        for i in range(n_splits):
            start = i * (split_size // 2)
            end = min(start + split_size, total)
            split_data = data[start:end]

            train_end = int(len(split_data) * train_pct)
            train = split_data[:train_end]
            test = split_data[train_end:]

            if len(train) > 50 and len(test) > 20:
                splits.append((train, test))

        return splits


# Singleton
data_loader = BacktestDataLoader()
