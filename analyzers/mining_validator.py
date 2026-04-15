"""
TitanBot Pro — Mining & Validator Data Analyzer
=================================================
Monitors Bitcoin mining economics and PoS validator metrics.

Miners are "informed natural sellers" — their behavior predicts:
  - Miner capitulation → market bottoms (hash ribbons)
  - Miner accumulation → market strength
  - Hash rate surges → network confidence
  - Difficulty adjustments → mining profitability shifts

Data sources (all free, no API key):
  - Blockchain.com (hash rate, difficulty, miner revenue)
  - Mempool.space (fee market, block data)
  - CoinGecko (ETH staking yield proxy)

Metrics:
  1. Hash Rate Trend     — network security indicator
  2. Hash Ribbons        — 30d MA vs 60d MA (capitulation detector)
  3. Difficulty Trend     — mining economics pressure
  4. Miner Revenue       — revenue vs cost proxy (Puell in onchain_analytics)
  5. Fee Market          — mempool congestion as demand proxy
  6. ETH Staking Yield   — PoS validator economics

Signal impact:
  Hash ribbon buy signal:   +5 confidence on LONGs
  Miner capitulation:       +4 LONGs (contrarian bottom)
  Extreme fees:             -3 LONGs (congestion = near-term sell pressure)
  Hash rate surge:          +3 LONGs (network confidence)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ── API endpoints ─────────────────────────────────────────────
_BLOCKCHAIN_COM  = "https://api.blockchain.info"
_MEMPOOL_SPACE   = "https://mempool.space/api"

# Refresh intervals
_HASH_REFRESH    = 1800   # 30 min (slow-moving)
_FEE_REFRESH     = 300    # 5 min (fast-moving)

# Hash Ribbon thresholds
_HASH_RIBBON_CAPITULATION = True   # When 30d MA < 60d MA
_HASH_RIBBON_RECOVERY     = True   # When 30d MA crosses above 60d MA

# Fee thresholds (sat/vB)
_FEE_EXTREME    = 100     # > 100 sat/vB = extreme congestion
_FEE_HIGH       = 50      # > 50 sat/vB = high
_FEE_MODERATE   = 20      # > 20 sat/vB = moderate


@dataclass
class HashRateData:
    """Bitcoin hash rate and difficulty metrics."""
    hash_rate_eh: float = 0.0         # Current hash rate in EH/s
    hash_rate_30d_avg: float = 0.0    # 30-day moving average
    hash_rate_60d_avg: float = 0.0    # 60-day moving average
    hash_rate_trend: str = "NEUTRAL"  # SURGING | RISING | NEUTRAL | DECLINING | CAPITULATING
    difficulty: float = 0.0           # Current difficulty
    difficulty_change_pct: float = 0.0  # Last difficulty adjustment %
    hash_ribbon_signal: str = "NEUTRAL"  # BUY | NEUTRAL | CAPITULATION
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _HASH_REFRESH * 3


@dataclass
class FeeMarketData:
    """Bitcoin fee market / mempool metrics."""
    fastest_fee: int = 0        # sat/vB for next block
    half_hour_fee: int = 0      # sat/vB for ~3 blocks
    hour_fee: int = 0           # sat/vB for ~6 blocks
    mempool_size_mb: float = 0.0  # Mempool size in MB
    mempool_tx_count: int = 0
    congestion_level: str = "LOW"  # LOW | MODERATE | HIGH | EXTREME
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _FEE_REFRESH * 3


@dataclass
class MinerRevenueData:
    """Miner revenue metrics."""
    daily_revenue_usd: float = 0.0
    revenue_30d_avg: float = 0.0
    miner_selling_pressure: str = "NEUTRAL"  # LOW | NEUTRAL | HIGH | EXTREME
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _HASH_REFRESH * 3


@dataclass
class MiningSnapshot:
    """Complete mining & validator snapshot."""
    hash_rate: HashRateData = field(default_factory=HashRateData)
    fees: FeeMarketData = field(default_factory=FeeMarketData)
    revenue: MinerRevenueData = field(default_factory=MinerRevenueData)
    last_update: float = 0.0


class MiningValidatorAnalyzer:
    """
    Monitors mining economics and network health.

    Integration points:
      - Engine calls get_signal_intel() for per-signal confidence adjustments
      - On-chain analytics uses hash ribbon data
      - Market state builder uses fee market data
    """

    def __init__(self):
        self._snapshot = MiningSnapshot()
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._hash_history: List[Tuple[float, float]] = []  # (ts, hash_rate)

    async def start(self):
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("⛏️ MiningValidatorAnalyzer started (HashRate·Difficulty·Fees·Revenue)")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if self._session:
            await self._session.close()

    async def _poll_loop(self):
        while self._running:
            try:
                await self._update_all()
                self._snapshot.last_update = time.time()
            except Exception as e:
                logger.warning(f"Mining poll error: {e}")
            await asyncio.sleep(_FEE_REFRESH)

    async def _update_all(self):
        now = time.time()
        tasks = [self._fetch_fees()]

        # Hash rate and revenue update less frequently
        if now - self._snapshot.hash_rate.timestamp > _HASH_REFRESH:
            tasks.append(self._fetch_hash_rate())
            tasks.append(self._fetch_miner_revenue())

        await asyncio.gather(*tasks, return_exceptions=True)

    # ── Hash Rate ─────────────────────────────────────────────

    async def _fetch_hash_rate(self):
        """Fetch hash rate and compute hash ribbons."""
        if not self._session:
            return
        try:
            url = f"{_BLOCKCHAIN_COM}/charts/hash-rate?timespan=90days&format=json"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if len(values) >= 60:
                        current_hr = values[-1].get("y", 0)

                        # Compute 30d and 60d MAs
                        hr_30d = [v.get("y", 0) for v in values[-30:]]
                        hr_60d = [v.get("y", 0) for v in values[-60:]]
                        ma_30 = sum(hr_30d) / len(hr_30d) if hr_30d else 0
                        ma_60 = sum(hr_60d) / len(hr_60d) if hr_60d else 0

                        # Hash ribbon signal
                        if ma_30 < ma_60 * 0.95:
                            ribbon_signal = "CAPITULATION"
                        elif ma_30 > ma_60 and ma_30 > ma_60 * 1.02:
                            ribbon_signal = "BUY"
                        else:
                            ribbon_signal = "NEUTRAL"

                        # Trend classification
                        hr_7d = sum(v.get("y", 0) for v in values[-7:]) / 7
                        hr_14d = sum(v.get("y", 0) for v in values[-14:]) / 14

                        if hr_7d > hr_14d * 1.10:
                            trend = "SURGING"
                        elif hr_7d > hr_14d * 1.03:
                            trend = "RISING"
                        elif hr_7d < hr_14d * 0.90:
                            trend = "CAPITULATING"
                        elif hr_7d < hr_14d * 0.97:
                            trend = "DECLINING"
                        else:
                            trend = "NEUTRAL"

                        # Convert to EH/s (blockchain.com returns in GH/s)
                        current_eh = current_hr / 1e9 if current_hr > 1e6 else current_hr

                        self._snapshot.hash_rate = HashRateData(
                            hash_rate_eh=current_eh,
                            hash_rate_30d_avg=ma_30 / 1e9 if ma_30 > 1e6 else ma_30,
                            hash_rate_60d_avg=ma_60 / 1e9 if ma_60 > 1e6 else ma_60,
                            hash_rate_trend=trend,
                            hash_ribbon_signal=ribbon_signal,
                            timestamp=time.time(),
                        )
                        self._hash_history.append((time.time(), current_hr))
                        cutoff = time.time() - 86400 * 90
                        self._hash_history = [(t, v) for t, v in self._hash_history if t > cutoff]

                        logger.debug(
                            f"HashRate={current_eh:.1f}EH/s trend={trend} "
                            f"ribbon={ribbon_signal} 30dMA={ma_30/1e9:.1f}"
                        )
        except Exception as e:
            logger.debug(f"Hash rate fetch error: {e}")

    # ── Fee Market ────────────────────────────────────────────

    async def _fetch_fees(self):
        """Fetch current fee market data from mempool.space."""
        if not self._session:
            return
        try:
            # Fee estimates
            url = f"{_MEMPOOL_SPACE}/v1/fees/recommended"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    fastest = data.get("fastestFee", 0)
                    half_hour = data.get("halfHourFee", 0)
                    hour = data.get("hourFee", 0)

                    if fastest >= _FEE_EXTREME:
                        level = "EXTREME"
                    elif fastest >= _FEE_HIGH:
                        level = "HIGH"
                    elif fastest >= _FEE_MODERATE:
                        level = "MODERATE"
                    else:
                        level = "LOW"

                    self._snapshot.fees = FeeMarketData(
                        fastest_fee=fastest,
                        half_hour_fee=half_hour,
                        hour_fee=hour,
                        congestion_level=level,
                        timestamp=time.time(),
                    )

            # Mempool stats
            url2 = f"{_MEMPOOL_SPACE}/mempool"
            async with self._session.get(url2) as resp2:
                if resp2.status == 200:
                    mp_data = await resp2.json()
                    self._snapshot.fees.mempool_tx_count = mp_data.get("count", 0)
                    vsize = mp_data.get("vsize", 0)
                    self._snapshot.fees.mempool_size_mb = vsize / 1e6 if vsize else 0

                    logger.debug(
                        f"Fees: fastest={fastest}sat/vB level={level} "
                        f"mempool={self._snapshot.fees.mempool_tx_count}tx"
                    )
        except Exception as e:
            logger.debug(f"Fee market fetch error: {e}")

    # ── Miner Revenue ─────────────────────────────────────────

    async def _fetch_miner_revenue(self):
        """Fetch miner revenue data."""
        if not self._session:
            return
        try:
            url = f"{_BLOCKCHAIN_COM}/charts/miners-revenue?timespan=60days&format=json"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if len(values) >= 30:
                        current_rev = values[-1].get("y", 0)
                        avg_30d = sum(v.get("y", 0) for v in values[-30:]) / 30

                        # Miner selling pressure estimation:
                        # If revenue drops below cost (proxy: if rev < 60% of avg),
                        # miners must sell more BTC → selling pressure
                        if avg_30d > 0:
                            ratio = current_rev / avg_30d
                            if ratio < 0.5:
                                pressure = "EXTREME"
                            elif ratio < 0.7:
                                pressure = "HIGH"
                            elif ratio > 1.5:
                                pressure = "LOW"
                            else:
                                pressure = "NEUTRAL"
                        else:
                            pressure = "NEUTRAL"

                        self._snapshot.revenue = MinerRevenueData(
                            daily_revenue_usd=current_rev,
                            revenue_30d_avg=avg_30d,
                            miner_selling_pressure=pressure,
                            timestamp=time.time(),
                        )
                        logger.debug(
                            f"Miner revenue=${current_rev/1e6:.1f}M/day "
                            f"avg=${avg_30d/1e6:.1f}M pressure={pressure}"
                        )
        except Exception as e:
            logger.debug(f"Miner revenue fetch error: {e}")

    # ── Signal Intelligence ───────────────────────────────────

    def get_signal_intel(self, symbol: str, direction: str) -> Tuple[int, str]:
        """Returns (confidence_delta, note) for a given signal."""
        delta = 0
        notes = []
        is_long = direction == "LONG"
        snap = self._snapshot

        # Hash ribbon impact
        if not snap.hash_rate.is_stale:
            if snap.hash_rate.hash_ribbon_signal == "BUY":
                delta += +5 if is_long else -3
                notes.append("⛏️ Hash ribbon BUY signal")
            elif snap.hash_rate.hash_ribbon_signal == "CAPITULATION":
                # Miner capitulation is contrarian bullish (bottoming)
                delta += +4 if is_long else -2
                notes.append("⛏️ Miner capitulation (contrarian bullish)")

            if snap.hash_rate.hash_rate_trend == "SURGING":
                delta += +3 if is_long else -1
                notes.append("⛏️ Hash rate surging (network confidence)")
            elif snap.hash_rate.hash_rate_trend == "CAPITULATING":
                delta += +2 if is_long else -1  # Contrarian — capitulation = near bottom
                notes.append("⛏️ Hash rate declining (potential bottom)")

        # Fee market impact
        if not snap.fees.is_stale:
            if snap.fees.congestion_level == "EXTREME":
                delta += -3 if is_long else +1
                notes.append(f"⛏️ Extreme fees ({snap.fees.fastest_fee}sat/vB)")
            elif snap.fees.congestion_level == "HIGH":
                delta += -2 if is_long else +1
                notes.append(f"⛏️ High fees ({snap.fees.fastest_fee}sat/vB)")

        # Miner revenue impact
        if not snap.revenue.is_stale:
            if snap.revenue.miner_selling_pressure == "EXTREME":
                delta += -3 if is_long else +2
                notes.append("⛏️ Extreme miner selling pressure")
            elif snap.revenue.miner_selling_pressure == "LOW":
                delta += +2 if is_long else -1
                notes.append("⛏️ Low miner selling pressure (hodling)")

        delta = max(-10, min(10, delta))
        note = " | ".join(notes) if notes else ""
        return delta, note

    def get_network_health(self) -> str:
        """Returns network health classification."""
        snap = self._snapshot
        if snap.hash_rate.is_stale:
            return "UNKNOWN"
        if snap.hash_rate.hash_rate_trend in ("SURGING", "RISING"):
            return "STRONG"
        if snap.hash_rate.hash_rate_trend in ("CAPITULATING", "DECLINING"):
            return "WEAK"
        return "NEUTRAL"

    def get_snapshot(self) -> MiningSnapshot:
        return self._snapshot


# Module-level singleton
mining_analyzer = MiningValidatorAnalyzer()
