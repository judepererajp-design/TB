"""
TitanBot Pro — Network Activity Analyzer
==========================================
Monitors blockchain-level network activity as a demand proxy.

Active addresses, transaction counts, and network velocity
provide fundamental confirmation of price trends:
  - Rising activity + rising price = genuine demand
  - Rising price + falling activity = speculative pump (fragile)
  - Falling price + rising activity = accumulation

Data sources (all free, no API key):
  - Blockchain.com (BTC active addresses, transaction count)
  - Mempool.space (BTC transaction data)
  - Etherscan-like public APIs (ETH activity)

Signal impact:
  Activity surge (>20% above avg): +5 confidence on trend-following signals
  Activity collapse (<70% of avg):  -4 confidence on LONGs
  Divergence (price up, activity down): -3 LONGs (fragile rally)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ── API endpoints ─────────────────────────────────────────────
_BLOCKCHAIN_COM = "https://api.blockchain.info"

# Refresh intervals
_ACTIVITY_REFRESH = 600   # 10 min

# Thresholds
_ACTIVITY_SURGE_MULT    = 1.20   # 20% above average
_ACTIVITY_COLLAPSE_MULT = 0.70   # 30% below average
_TX_SURGE_MULT          = 1.25   # 25% above average tx count
_TX_COLLAPSE_MULT       = 0.70   # 30% below average


@dataclass
class ActiveAddressData:
    """Active address metrics."""
    active_addresses: int = 0         # Current unique active addresses
    active_7d_avg: int = 0            # 7-day average
    active_30d_avg: int = 0           # 30-day average
    trend: str = "NEUTRAL"            # SURGING | RISING | NEUTRAL | DECLINING | COLLAPSING
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _ACTIVITY_REFRESH * 3


@dataclass
class TransactionData:
    """Transaction count and volume metrics."""
    tx_count: int = 0                 # Daily transaction count
    tx_count_7d_avg: int = 0
    tx_volume_usd: float = 0.0       # Estimated on-chain volume
    tx_trend: str = "NEUTRAL"         # SURGING | RISING | NEUTRAL | DECLINING | COLLAPSING
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _ACTIVITY_REFRESH * 3


@dataclass
class NetworkVelocityData:
    """Network value transfer velocity."""
    nvt_ratio: float = 0.0            # Network Value to Transactions ratio
    nvt_signal: str = "NEUTRAL"       # OVERVALUED | NEUTRAL | UNDERVALUED
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _ACTIVITY_REFRESH * 3


@dataclass
class NetworkActivitySnapshot:
    """Complete network activity snapshot."""
    addresses: ActiveAddressData = field(default_factory=ActiveAddressData)
    transactions: TransactionData = field(default_factory=TransactionData)
    velocity: NetworkVelocityData = field(default_factory=NetworkVelocityData)
    last_update: float = 0.0


class NetworkActivityAnalyzer:
    """
    Monitors blockchain-level network activity metrics.

    Integration points:
      - Engine calls get_signal_intel() for confirmation
      - Market state builder uses get_demand_signal() for state enrichment
      - On-chain analytics uses get_nvt() for valuation
    """

    def __init__(self):
        self._snapshot = NetworkActivitySnapshot()
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("🌐 NetworkActivityAnalyzer started (Addresses·Transactions·NVT)")

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
                logger.warning(f"Network activity poll error: {e}")
            await asyncio.sleep(_ACTIVITY_REFRESH)

    async def _update_all(self):
        await asyncio.gather(
            self._fetch_active_addresses(),
            self._fetch_transactions(),
            self._compute_nvt(),
            return_exceptions=True,
        )

    # ── Active Addresses ──────────────────────────────────────

    async def _fetch_active_addresses(self):
        """Fetch unique active addresses from blockchain.com."""
        if not self._session:
            return
        try:
            url = f"{_BLOCKCHAIN_COM}/charts/n-unique-addresses?timespan=60days&format=json"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if len(values) >= 30:
                        current = int(values[-1].get("y", 0))
                        avg_7d = sum(v.get("y", 0) for v in values[-7:]) / 7
                        avg_30d = sum(v.get("y", 0) for v in values[-30:]) / 30

                        # Classify trend
                        if avg_30d > 0:
                            ratio = current / avg_30d
                            if ratio >= _ACTIVITY_SURGE_MULT * 1.1:
                                trend = "SURGING"
                            elif ratio >= _ACTIVITY_SURGE_MULT:
                                trend = "RISING"
                            elif ratio <= _ACTIVITY_COLLAPSE_MULT * 0.9:
                                trend = "COLLAPSING"
                            elif ratio <= _ACTIVITY_COLLAPSE_MULT:
                                trend = "DECLINING"
                            else:
                                trend = "NEUTRAL"
                        else:
                            trend = "NEUTRAL"

                        self._snapshot.addresses = ActiveAddressData(
                            active_addresses=current,
                            active_7d_avg=int(avg_7d),
                            active_30d_avg=int(avg_30d),
                            trend=trend,
                            timestamp=time.time(),
                        )
                        logger.debug(
                            f"Active addresses={current:,} trend={trend} "
                            f"7d_avg={int(avg_7d):,}"
                        )
        except Exception as e:
            logger.debug(f"Active addresses fetch error: {e}")

    # ── Transactions ──────────────────────────────────────────

    async def _fetch_transactions(self):
        """Fetch transaction count and estimated volume."""
        if not self._session:
            return
        try:
            # Transaction count
            url = f"{_BLOCKCHAIN_COM}/charts/n-transactions?timespan=60days&format=json"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if len(values) >= 30:
                        current = int(values[-1].get("y", 0))
                        avg_7d = sum(v.get("y", 0) for v in values[-7:]) / 7

                        if avg_7d > 0:
                            ratio = current / avg_7d
                            if ratio >= _TX_SURGE_MULT:
                                trend = "SURGING"
                            elif ratio >= 1.10:
                                trend = "RISING"
                            elif ratio <= _TX_COLLAPSE_MULT:
                                trend = "COLLAPSING"
                            elif ratio <= 0.85:
                                trend = "DECLINING"
                            else:
                                trend = "NEUTRAL"
                        else:
                            trend = "NEUTRAL"

                        self._snapshot.transactions = TransactionData(
                            tx_count=current,
                            tx_count_7d_avg=int(avg_7d),
                            tx_trend=trend,
                            timestamp=time.time(),
                        )

            # Estimated tx volume
            url2 = f"{_BLOCKCHAIN_COM}/charts/estimated-transaction-volume-usd?timespan=7days&format=json"
            async with self._session.get(url2) as resp2:
                if resp2.status == 200:
                    data2 = await resp2.json()
                    values2 = data2.get("values", [])
                    if values2:
                        self._snapshot.transactions.tx_volume_usd = values2[-1].get("y", 0)

                    logger.debug(
                        f"Transactions={self._snapshot.transactions.tx_count:,} "
                        f"trend={self._snapshot.transactions.tx_trend} "
                        f"vol=${self._snapshot.transactions.tx_volume_usd/1e9:.1f}B"
                    )
        except Exception as e:
            logger.debug(f"Transaction fetch error: {e}")

    # ── NVT ───────────────────────────────────────────────────

    async def _compute_nvt(self):
        """
        Network Value to Transactions ratio.
        High NVT = network overvalued relative to usage.
        Low NVT = undervalued or high utility.
        """
        tx = self._snapshot.transactions
        if tx.tx_volume_usd <= 0:
            return

        try:
            # Get BTC market cap
            url = f"{_BLOCKCHAIN_COM}/charts/market-cap?timespan=1days&format=json"
            if not self._session:
                return
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if values:
                        market_cap = values[-1].get("y", 0)
                        if tx.tx_volume_usd > 0 and market_cap > 0:
                            # NVT = market_cap / daily_tx_volume
                            # Annualized for smoother signal
                            nvt = market_cap / (tx.tx_volume_usd * 365)

                            if nvt > 2.0:
                                signal = "OVERVALUED"
                            elif nvt < 0.5:
                                signal = "UNDERVALUED"
                            else:
                                signal = "NEUTRAL"

                            self._snapshot.velocity = NetworkVelocityData(
                                nvt_ratio=nvt,
                                nvt_signal=signal,
                                timestamp=time.time(),
                            )
                            logger.debug(f"NVT={nvt:.2f} signal={signal}")
        except Exception as e:
            logger.debug(f"NVT compute error: {e}")

    # ── Signal Intelligence ───────────────────────────────────

    def get_signal_intel(self, symbol: str, direction: str) -> Tuple[int, str]:
        """Returns (confidence_delta, note) for a given signal."""
        delta = 0
        notes = []
        is_long = direction == "LONG"
        snap = self._snapshot

        # Active address impact
        if not snap.addresses.is_stale:
            if snap.addresses.trend == "SURGING":
                delta += +5 if is_long else -2
                notes.append(f"🌐 Active addresses surging ({snap.addresses.active_addresses:,})")
            elif snap.addresses.trend == "RISING":
                delta += +3 if is_long else -1
                notes.append(f"🌐 Active addresses rising")
            elif snap.addresses.trend == "COLLAPSING":
                delta += -4 if is_long else +2
                notes.append(f"🌐 Active addresses collapsing ({snap.addresses.active_addresses:,})")
            elif snap.addresses.trend == "DECLINING":
                delta += -2 if is_long else +1
                notes.append(f"🌐 Active addresses declining")

        # Transaction trend impact
        if not snap.transactions.is_stale:
            if snap.transactions.tx_trend == "SURGING":
                delta += +3 if is_long else -1
                notes.append(f"🌐 TX count surging ({snap.transactions.tx_count:,})")
            elif snap.transactions.tx_trend == "COLLAPSING":
                delta += -3 if is_long else +2
                notes.append(f"🌐 TX count collapsing")

        # NVT impact
        if not snap.velocity.is_stale:
            if snap.velocity.nvt_signal == "OVERVALUED":
                delta += -3 if is_long else +2
                notes.append(f"🌐 NVT={snap.velocity.nvt_ratio:.2f} overvalued")
            elif snap.velocity.nvt_signal == "UNDERVALUED":
                delta += +3 if is_long else -2
                notes.append(f"🌐 NVT={snap.velocity.nvt_ratio:.2f} undervalued")

        delta = max(-10, min(10, delta))
        note = " | ".join(notes) if notes else ""
        return delta, note

    def get_demand_signal(self) -> str:
        """Returns network demand signal for market state builder."""
        snap = self._snapshot
        if snap.addresses.is_stale:
            return "UNKNOWN"
        if snap.addresses.trend in ("SURGING", "RISING"):
            return "STRONG"
        if snap.addresses.trend in ("COLLAPSING", "DECLINING"):
            return "WEAK"
        return "NEUTRAL"

    def get_snapshot(self) -> NetworkActivitySnapshot:
        return self._snapshot


# Module-level singleton
network_activity = NetworkActivityAnalyzer()
