"""
TitanBot Pro — On-Chain Analytics Engine
==========================================
Brings institutional-grade on-chain metrics into the signal pipeline:

  1. MVRV Ratio       — Market Value vs Realized Value (top/bottom detector)
  2. SOPR             — Spent Output Profit Ratio (holder profit/loss)
  3. Realized Cap     — Aggregate cost basis of all holders
  4. HODL Waves       — Long-term vs short-term holder distribution
  5. NUPL             — Net Unrealized Profit/Loss
  6. Puell Multiple   — Miner revenue vs 365d average

Data sources (all free, no API key):
  - Blockchain.com API (BTC-specific metrics)
  - CryptoQuant free tier (exchange flows, MVRV proxy)
  - Glassnode free endpoints (limited metrics)
  - Mempool.space (BTC mempool/fees)
  - Fallback: derived metrics from price + supply data

Signal impact:
  MVRV > 3.5  → overheated zone, -8 confidence on LONGs
  MVRV < 1.0  → undervalued zone, +6 confidence on LONGs
  SOPR < 1.0  → holders selling at loss (capitulation), +5 LONGs
  SOPR > 1.05 → profit-taking pressure, -4 LONGs
  NUPL > 0.75 → euphoria zone, -6 LONGs
  NUPL < 0    → capitulation zone, +6 LONGs
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ── API endpoints (all free) ─────────────────────────────────
_BLOCKCHAIN_COM    = "https://api.blockchain.info"
_CRYPTOQUANT_FREE  = "https://api.cryptoquant.com/v1"
_MEMPOOL_SPACE     = "https://mempool.space/api"
_COINGECKO_BASE    = "https://api.coingecko.com/api/v3"

# Refresh intervals
_ONCHAIN_REFRESH   = 600    # 10 min for most metrics
_MVRV_REFRESH      = 1800   # 30 min (slow-moving)
_HODL_REFRESH      = 3600   # 1 hour (very slow-moving)

# ── Thresholds ────────────────────────────────────────────────
_MVRV_OVERHEATED       = 3.5
_MVRV_EUPHORIA         = 3.0
_MVRV_UNDERVALUED      = 1.0
_MVRV_DEEP_VALUE       = 0.8

_SOPR_PROFIT_TAKING    = 1.05
_SOPR_CAPITULATION     = 1.0

_NUPL_EUPHORIA         = 0.75
_NUPL_GREED            = 0.50
_NUPL_CAPITULATION     = 0.0
_NUPL_DEEP_FEAR        = -0.25

_PUELL_OVERHEATED      = 4.0
_PUELL_UNDERVALUED     = 0.5


# ── Data models ───────────────────────────────────────────────

@dataclass
class MVRVData:
    """Market Value to Realized Value ratio."""
    mvrv_ratio: float = 0.0           # Current MVRV
    market_cap_usd: float = 0.0       # Market cap
    realized_cap_usd: float = 0.0     # Realized cap (aggregate cost basis)
    zone: str = "NEUTRAL"             # DEEP_VALUE | UNDERVALUED | NEUTRAL | OVERHEATED | EUPHORIA
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _MVRV_REFRESH * 2


@dataclass
class SOPRData:
    """Spent Output Profit Ratio."""
    sopr: float = 1.0                 # Current SOPR (1.0 = breakeven)
    sopr_7d_avg: float = 1.0          # 7-day average
    zone: str = "NEUTRAL"             # CAPITULATION | LOSS_TAKING | NEUTRAL | PROFIT_TAKING | EXTREME_PROFIT
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _ONCHAIN_REFRESH * 2


@dataclass
class NUPLData:
    """Net Unrealized Profit/Loss."""
    nupl: float = 0.0                 # -1 to 1 range
    zone: str = "NEUTRAL"             # CAPITULATION | FEAR | NEUTRAL | GREED | EUPHORIA
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _ONCHAIN_REFRESH * 2


@dataclass
class HODLWaveData:
    """Holder distribution by age."""
    short_term_pct: float = 0.0       # Coins held < 155 days
    long_term_pct: float = 0.0        # Coins held > 155 days
    sth_mvrv: float = 1.0             # Short-term holder MVRV
    lth_mvrv: float = 1.0             # Long-term holder MVRV
    accumulation_phase: bool = False   # LTH increasing, STH decreasing
    distribution_phase: bool = False   # STH increasing, LTH decreasing
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _HODL_REFRESH * 2


@dataclass
class PuellMultipleData:
    """Miner revenue relative to yearly average."""
    puell: float = 1.0
    zone: str = "NEUTRAL"             # UNDERVALUED | NEUTRAL | OVERHEATED
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _ONCHAIN_REFRESH * 2


@dataclass
class OnChainSnapshot:
    """Complete on-chain analytics snapshot."""
    mvrv: MVRVData = field(default_factory=MVRVData)
    sopr: SOPRData = field(default_factory=SOPRData)
    nupl: NUPLData = field(default_factory=NUPLData)
    hodl: HODLWaveData = field(default_factory=HODLWaveData)
    puell: PuellMultipleData = field(default_factory=PuellMultipleData)
    last_update: float = 0.0


@dataclass
class OnChainEventState:
    """Derived on-chain anomaly state for correlation and diagnostics."""
    last_anomaly_time: float = 0.0
    anomaly_keywords: List[str] = field(default_factory=list)
    anomaly_bias: str = "NEUTRAL"
    anomaly_detail: str = ""


class OnChainAnalytics:
    """
    Fetches and interprets on-chain metrics for BTC.

    Integration points:
      - Engine calls get_signal_intel() for per-signal confidence adjustments
      - Ensemble voter uses get_market_phase() for macro context
      - Market state builder uses get_valuation_zone() for regime enrichment
    """

    def __init__(self):
        self._snapshot = OnChainSnapshot()
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

        # Historical data for trend analysis
        self._mvrv_history: List[Tuple[float, float]] = []   # (ts, mvrv)
        self._sopr_history: List[Tuple[float, float]] = []   # (ts, sopr)
        self._nupl_history: List[Tuple[float, float]] = []   # (ts, nupl)

    # ── Lifecycle ─────────────────────────────────────────────

    async def start(self):
        """Begin background polling."""
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("📊 OnChainAnalytics started (MVRV·SOPR·NUPL·HODL·Puell)")

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        if self._task:
            self._task.cancel()
        if self._session:
            await self._session.close()

    # ── Poll loop ─────────────────────────────────────────────

    async def _poll_loop(self):
        """Main polling loop — staggers fetches to avoid rate limits."""
        while self._running:
            try:
                await self._update_all()
                self._snapshot.last_update = time.time()
            except Exception as e:
                logger.warning(f"OnChain poll error: {e}")
            await asyncio.sleep(_ONCHAIN_REFRESH)

    async def _update_all(self):
        """Fetch all on-chain metrics concurrently."""
        tasks = [
            self._fetch_mvrv(),
            self._fetch_sopr(),
            self._fetch_nupl(),
            self._fetch_hodl_waves(),
            self._fetch_puell(),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # ── MVRV ──────────────────────────────────────────────────

    async def _fetch_mvrv(self):
        """
        Fetch MVRV ratio. Uses CoinGecko market cap + estimated realized cap.

        Realized cap is approximated from supply * average acquisition price,
        derived from blockchain.com data. This is a simplified model that
        captures the directional signal without requiring paid API access.
        """
        if not self._session:
            return
        try:
            # Get current BTC market cap from CoinGecko
            url = f"{_COINGECKO_BASE}/coins/bitcoin?localization=false&tickers=false&community_data=false&developer_data=false"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    market_cap = data.get("market_data", {}).get("market_cap", {}).get("usd", 0)
                    current_price = data.get("market_data", {}).get("current_price", {}).get("usd", 0)
                    circulating = data.get("market_data", {}).get("circulating_supply", 0)

                    if market_cap > 0 and circulating > 0 and current_price > 0:
                        # Estimate realized cap from blockchain.com average transaction value
                        # Realized cap ≈ price at which each coin last moved * supply
                        # Approximation: use 200-day SMA as proxy for average cost basis
                        realized_cap = await self._estimate_realized_cap(current_price, circulating)

                        if realized_cap > 0:
                            mvrv = market_cap / realized_cap

                            # Classify zone
                            if mvrv >= _MVRV_OVERHEATED:
                                zone = "EUPHORIA"
                            elif mvrv >= _MVRV_EUPHORIA:
                                zone = "OVERHEATED"
                            elif mvrv <= _MVRV_DEEP_VALUE:
                                zone = "DEEP_VALUE"
                            elif mvrv <= _MVRV_UNDERVALUED:
                                zone = "UNDERVALUED"
                            else:
                                zone = "NEUTRAL"

                            self._snapshot.mvrv = MVRVData(
                                mvrv_ratio=mvrv,
                                market_cap_usd=market_cap,
                                realized_cap_usd=realized_cap,
                                zone=zone,
                                timestamp=time.time(),
                            )
                            self._mvrv_history.append((time.time(), mvrv))
                            # Keep 30 days of history
                            cutoff = time.time() - 86400 * 30
                            self._mvrv_history = [(t, v) for t, v in self._mvrv_history if t > cutoff]

                            logger.debug(f"MVRV={mvrv:.2f} zone={zone} mcap=${market_cap/1e9:.1f}B rcap=${realized_cap/1e9:.1f}B")

        except Exception as e:
            logger.debug(f"MVRV fetch error: {e}")

    async def _estimate_realized_cap(self, current_price: float, circulating: float) -> float:
        """
        Estimate realized cap using a multi-factor approach.
        Uses blockchain.com 200-day transaction price average as a proxy
        for the average cost basis of holders.
        """
        if not self._session:
            return 0.0
        try:
            # Use blockchain.com 200-day average market price
            url = f"{_BLOCKCHAIN_COM}/charts/market-price?timespan=200days&format=json"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if values:
                        # Weight recent prices more heavily (proxy for when coins last moved)
                        total_weight = 0
                        weighted_price = 0.0
                        for i, v in enumerate(values):
                            weight = (i + 1) ** 0.5  # sqrt weighting (recency bias)
                            weighted_price += v.get("y", 0) * weight
                            total_weight += weight
                        if total_weight > 0:
                            avg_cost_basis = weighted_price / total_weight
                            return avg_cost_basis * circulating
        except Exception as e:
            logger.debug(f"Realized cap estimation error: {e}")

        # Fallback: use 0.7 * current price as rough realized cap proxy
        # (empirically BTC realized cap is roughly 60-80% of market cap during neutral periods)
        return current_price * 0.72 * circulating

    # ── SOPR ──────────────────────────────────────────────────

    async def _fetch_sopr(self):
        """
        Fetch Spent Output Profit Ratio.
        SOPR > 1 = holders selling at profit; SOPR < 1 = selling at loss.

        Uses CryptoQuant free tier or derives from price action.
        """
        if not self._session:
            return
        try:
            # Try CryptoQuant free API first
            url = f"{_CRYPTOQUANT_FREE}/btc/market-data/sopr?window=day&limit=7"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {}).get("data", [])
                    if result:
                        current = result[-1].get("sopr", 1.0)
                        avg_7d = sum(r.get("sopr", 1.0) for r in result) / len(result)
                        self._update_sopr(current, avg_7d)
                        return
        except Exception:
            pass

        # Fallback: derive SOPR proxy from price behavior
        # If price > 200-day average → SOPR > 1 (profit zone)
        # If price < 200-day average → SOPR < 1 (loss zone)
        await self._derive_sopr_from_price()

    async def _derive_sopr_from_price(self):
        """Derive SOPR proxy from current price vs cost basis estimate."""
        if not self._session:
            return
        try:
            url = f"{_BLOCKCHAIN_COM}/charts/market-price?timespan=200days&format=json"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if len(values) >= 7:
                        current_price = values[-1].get("y", 0)
                        avg_200d = sum(v.get("y", 0) for v in values) / len(values)
                        avg_7d_prices = sum(v.get("y", 0) for v in values[-7:]) / 7

                        if avg_200d > 0:
                            sopr_proxy = current_price / avg_200d
                            sopr_7d = avg_7d_prices / avg_200d
                            self._update_sopr(sopr_proxy, sopr_7d)
        except Exception as e:
            logger.debug(f"SOPR derivation error: {e}")

    def _update_sopr(self, sopr: float, sopr_7d: float):
        """Update SOPR data and classify zone."""
        if sopr >= 1.10:
            zone = "EXTREME_PROFIT"
        elif sopr >= _SOPR_PROFIT_TAKING:
            zone = "PROFIT_TAKING"
        elif sopr < 0.95:
            zone = "CAPITULATION"
        elif sopr < _SOPR_CAPITULATION:
            zone = "LOSS_TAKING"
        else:
            zone = "NEUTRAL"

        self._snapshot.sopr = SOPRData(
            sopr=sopr, sopr_7d_avg=sopr_7d, zone=zone, timestamp=time.time()
        )
        self._sopr_history.append((time.time(), sopr))
        cutoff = time.time() - 86400 * 30
        self._sopr_history = [(t, v) for t, v in self._sopr_history if t > cutoff]
        logger.debug(f"SOPR={sopr:.4f} zone={zone}")

    # ── NUPL ──────────────────────────────────────────────────

    async def _fetch_nupl(self):
        """
        Net Unrealized Profit/Loss.
        NUPL = (Market Cap - Realized Cap) / Market Cap
        Range: -1 (all at loss) to 1 (all at profit)
        """
        mvrv = self._snapshot.mvrv
        if mvrv.market_cap_usd > 0 and mvrv.realized_cap_usd > 0:
            nupl = (mvrv.market_cap_usd - mvrv.realized_cap_usd) / mvrv.market_cap_usd

            if nupl >= _NUPL_EUPHORIA:
                zone = "EUPHORIA"
            elif nupl >= _NUPL_GREED:
                zone = "GREED"
            elif nupl <= _NUPL_DEEP_FEAR:
                zone = "CAPITULATION"
            elif nupl <= _NUPL_CAPITULATION:
                zone = "FEAR"
            else:
                zone = "NEUTRAL"

            self._snapshot.nupl = NUPLData(
                nupl=nupl, zone=zone, timestamp=time.time()
            )
            self._nupl_history.append((time.time(), nupl))
            cutoff = time.time() - 86400 * 30
            self._nupl_history = [(t, v) for t, v in self._nupl_history if t > cutoff]
            logger.debug(f"NUPL={nupl:.4f} zone={zone}")

    # ── HODL Waves ────────────────────────────────────────────

    async def _fetch_hodl_waves(self):
        """
        Fetch holder distribution by age (HODL waves).
        Uses blockchain.com + CoinGecko supply data.
        """
        if not self._session:
            return
        try:
            # Approximate using blockchain.com days-destroyed metric
            # High coin-days destroyed = old coins moving (distribution)
            # Low coin-days destroyed  = old coins holding (accumulation)
            url = f"{_BLOCKCHAIN_COM}/charts/bitcoin-days-destroyed?timespan=30days&format=json"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if len(values) >= 7:
                        recent_cdd = sum(v.get("y", 0) for v in values[-7:]) / 7
                        older_cdd = sum(v.get("y", 0) for v in values[:7]) / max(7, len(values[:7]))

                        # If recent CDD is high relative to history → distribution
                        # If recent CDD is low → accumulation
                        if older_cdd > 0:
                            cdd_ratio = recent_cdd / older_cdd
                            distribution_phase = cdd_ratio > 1.5
                            accumulation_phase = cdd_ratio < 0.7

                            self._snapshot.hodl = HODLWaveData(
                                short_term_pct=min(100, max(0, cdd_ratio * 30)),
                                long_term_pct=min(100, max(0, 100 - cdd_ratio * 30)),
                                accumulation_phase=accumulation_phase,
                                distribution_phase=distribution_phase,
                                timestamp=time.time(),
                            )
                            logger.debug(
                                f"HODL: cdd_ratio={cdd_ratio:.2f} "
                                f"accum={accumulation_phase} dist={distribution_phase}"
                            )
        except Exception as e:
            logger.debug(f"HODL waves error: {e}")

    # ── Puell Multiple ────────────────────────────────────────

    async def _fetch_puell(self):
        """
        Puell Multiple = daily miner revenue / 365-day MA of miner revenue.
        Uses blockchain.com miner revenue data.
        """
        if not self._session:
            return
        try:
            url = f"{_BLOCKCHAIN_COM}/charts/miners-revenue?timespan=1year&format=json"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if len(values) >= 30:
                        current_rev = values[-1].get("y", 0)
                        avg_365d = sum(v.get("y", 0) for v in values) / len(values)

                        if avg_365d > 0:
                            puell = current_rev / avg_365d

                            if puell >= _PUELL_OVERHEATED:
                                zone = "OVERHEATED"
                            elif puell <= _PUELL_UNDERVALUED:
                                zone = "UNDERVALUED"
                            else:
                                zone = "NEUTRAL"

                            self._snapshot.puell = PuellMultipleData(
                                puell=puell, zone=zone, timestamp=time.time()
                            )
                            logger.debug(f"Puell={puell:.2f} zone={zone}")
        except Exception as e:
            logger.debug(f"Puell fetch error: {e}")

    # ── Signal Intelligence ───────────────────────────────────

    def get_signal_intel(self, symbol: str, direction: str) -> Tuple[int, str]:
        """
        Returns (confidence_delta, note) for a given signal.
        Positive delta = supports direction. Negative = opposes.

        Diminishing returns: MVRV, SOPR, and NUPL all reflect holder
        cost-basis dynamics.  When multiple fire in the same direction
        the 2nd/3rd correlated signals are scaled down to avoid
        double-counting the same underlying insight.
        """
        from config.constants import Enrichment

        delta = 0
        notes = []
        is_long = direction == "LONG"
        snap = self._snapshot

        # ── Correlated valuation trio: MVRV · SOPR · NUPL ──────────
        # Collect raw deltas for the correlated trio, then apply
        # diminishing returns before adding to the running total.
        _trio_deltas: list = []  # list of (raw_delta, note_str)

        # MVRV impact
        if not snap.mvrv.is_stale:
            mvrv = snap.mvrv.mvrv_ratio
            if snap.mvrv.zone == "EUPHORIA":
                _trio_deltas.append((-8 if is_long else +5, f"MVRV={mvrv:.2f} euphoria"))
            elif snap.mvrv.zone == "OVERHEATED":
                _trio_deltas.append((-5 if is_long else +3, f"MVRV={mvrv:.2f} overheated"))
            elif snap.mvrv.zone == "DEEP_VALUE":
                _trio_deltas.append((+6 if is_long else -4, f"MVRV={mvrv:.2f} deep value"))
            elif snap.mvrv.zone == "UNDERVALUED":
                _trio_deltas.append((+4 if is_long else -2, f"MVRV={mvrv:.2f} undervalued"))

        # SOPR impact
        if not snap.sopr.is_stale:
            if snap.sopr.zone == "CAPITULATION":
                _trio_deltas.append((+5 if is_long else -3, f"SOPR={snap.sopr.sopr:.3f} capitulation"))
            elif snap.sopr.zone == "EXTREME_PROFIT":
                _trio_deltas.append((-4 if is_long else +3, f"SOPR={snap.sopr.sopr:.3f} profit-taking"))

        # NUPL impact
        if not snap.nupl.is_stale:
            if snap.nupl.zone == "EUPHORIA":
                _trio_deltas.append((-6 if is_long else +4, f"NUPL={snap.nupl.nupl:.2f} euphoria"))
            elif snap.nupl.zone == "CAPITULATION":
                _trio_deltas.append((+6 if is_long else -4, f"NUPL={snap.nupl.nupl:.2f} capitulation"))

        # Apply diminishing returns: 1st signal at full weight, 2nd at 60%, 3rd at 40%
        # Sort by absolute magnitude descending so the strongest fires at 100%.
        _trio_deltas.sort(key=lambda x: abs(x[0]), reverse=True)
        _dr_mults = [1.0, Enrichment.CORR_SECOND_SIGNAL_MULT, Enrichment.CORR_THIRD_SIGNAL_MULT]
        for idx, (raw_d, note_str) in enumerate(_trio_deltas):
            mult = _dr_mults[idx] if idx < len(_dr_mults) else Enrichment.CORR_THIRD_SIGNAL_MULT
            scaled = int(round(raw_d * mult))
            delta += scaled
            suffix = f" ×{mult:.0%}" if mult < 1.0 else ""
            notes.append(f"{note_str}{suffix}")

        # ── Independent signals (not correlated with the trio) ──────
        # HODL wave impact
        if not snap.hodl.is_stale:
            if snap.hodl.accumulation_phase:
                delta += +3 if is_long else -2
                notes.append("HODL: accumulation phase")
            elif snap.hodl.distribution_phase:
                delta += -3 if is_long else +2
                notes.append("HODL: distribution phase")

        # Puell impact
        if not snap.puell.is_stale:
            if snap.puell.zone == "OVERHEATED":
                delta += -3 if is_long else +2
                notes.append(f"Puell={snap.puell.puell:.2f} overheated")
            elif snap.puell.zone == "UNDERVALUED":
                delta += +3 if is_long else -2
                notes.append(f"Puell={snap.puell.puell:.2f} undervalued")

        # Cap total impact to avoid overwhelming other signals
        delta = max(-15, min(15, delta))
        note = " | ".join(notes) if notes else ""
        return delta, note

    def get_valuation_zone(self) -> str:
        """Returns overall on-chain valuation zone for regime enrichment."""
        snap = self._snapshot
        if snap.mvrv.is_stale:
            return "UNKNOWN"

        zones = [snap.mvrv.zone, snap.sopr.zone, snap.nupl.zone]
        bearish_count = sum(1 for z in zones if z in ("EUPHORIA", "OVERHEATED", "EXTREME_PROFIT"))
        bullish_count = sum(1 for z in zones if z in ("DEEP_VALUE", "UNDERVALUED", "CAPITULATION"))

        if bearish_count >= 2:
            return "OVERVALUED"
        elif bullish_count >= 2:
            return "UNDERVALUED"
        return "FAIR_VALUE"

    def get_market_phase(self) -> str:
        """Returns on-chain market phase for ensemble voter context."""
        snap = self._snapshot
        if snap.hodl.is_stale:
            return "UNKNOWN"
        if snap.hodl.accumulation_phase:
            return "ACCUMULATION"
        if snap.hodl.distribution_phase:
            return "DISTRIBUTION"
        return "NEUTRAL"

    def get_snapshot(self) -> OnChainSnapshot:
        """Return current snapshot for dashboard/logging."""
        return self._snapshot

    def get_state(self) -> OnChainEventState:
        """Return the latest meaningful on-chain event for correlation logic."""
        snap = self._snapshot
        candidates: List[tuple] = []

        if not snap.mvrv.is_stale and snap.mvrv.zone in {"EUPHORIA", "OVERHEATED"}:
            candidates.append((
                snap.mvrv.timestamp,
                ["mvrv", "overheated", "euphoria", "profit-taking"],
                "SHORT",
                f"MVRV {snap.mvrv.zone.lower()} ({snap.mvrv.mvrv_ratio:.2f})",
            ))
        elif not snap.mvrv.is_stale and snap.mvrv.zone in {"UNDERVALUED", "DEEP_VALUE"}:
            candidates.append((
                snap.mvrv.timestamp,
                ["mvrv", "undervalued", "deep value", "accumulation"],
                "LONG",
                f"MVRV {snap.mvrv.zone.lower()} ({snap.mvrv.mvrv_ratio:.2f})",
            ))

        if not snap.sopr.is_stale and snap.sopr.zone == "CAPITULATION":
            candidates.append((
                snap.sopr.timestamp,
                ["sopr", "capitulation", "sell-off", "washout"],
                "LONG",
                f"SOPR capitulation ({snap.sopr.sopr:.3f})",
            ))
        elif not snap.sopr.is_stale and snap.sopr.zone == "EXTREME_PROFIT":
            candidates.append((
                snap.sopr.timestamp,
                ["sopr", "profit-taking", "distribution", "sell pressure"],
                "SHORT",
                f"SOPR profit-taking ({snap.sopr.sopr:.3f})",
            ))

        if not snap.nupl.is_stale and snap.nupl.zone == "CAPITULATION":
            candidates.append((
                snap.nupl.timestamp,
                ["nupl", "capitulation", "fear", "washout"],
                "LONG",
                f"NUPL capitulation ({snap.nupl.nupl:.2f})",
            ))
        elif not snap.nupl.is_stale and snap.nupl.zone == "EUPHORIA":
            candidates.append((
                snap.nupl.timestamp,
                ["nupl", "euphoria", "overheated", "distribution"],
                "SHORT",
                f"NUPL euphoria ({snap.nupl.nupl:.2f})",
            ))

        if not snap.hodl.is_stale and snap.hodl.accumulation_phase:
            candidates.append((
                snap.hodl.timestamp,
                ["hodl", "accumulation", "holder strength"],
                "LONG",
                "HODL accumulation phase",
            ))
        elif not snap.hodl.is_stale and snap.hodl.distribution_phase:
            candidates.append((
                snap.hodl.timestamp,
                ["hodl", "distribution", "holder selling"],
                "SHORT",
                "HODL distribution phase",
            ))

        if not snap.puell.is_stale and snap.puell.zone == "UNDERVALUED":
            candidates.append((
                snap.puell.timestamp,
                ["puell", "undervalued", "miner stress"],
                "LONG",
                f"Puell undervalued ({snap.puell.puell:.2f})",
            ))
        elif not snap.puell.is_stale and snap.puell.zone == "OVERHEATED":
            candidates.append((
                snap.puell.timestamp,
                ["puell", "overheated", "miner distribution"],
                "SHORT",
                f"Puell overheated ({snap.puell.puell:.2f})",
            ))

        if not candidates:
            return OnChainEventState()

        ts, keywords, bias, detail = max(candidates, key=lambda item: item[0])
        return OnChainEventState(
            last_anomaly_time=ts,
            anomaly_keywords=keywords,
            anomaly_bias=bias,
            anomaly_detail=detail,
        )


# ── Module-level singleton ────────────────────────────────────
onchain_analytics = OnChainAnalytics()
