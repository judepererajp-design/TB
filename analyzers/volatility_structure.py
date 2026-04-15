"""
TitanBot Pro — Volatility Structure Analyzer
===============================================
Deep volatility analysis beyond simple ATR — captures the institutional
understanding of options-derived volatility signals.

Metrics:
  1. Realized Volatility (RV)  — actual historical vol (multiple windows)
  2. Volatility Cone           — is current vol high/low vs history?
  3. Volatility Regime          — compression/expansion cycle detection
  4. Vol-of-Vol                — volatility clustering (GARCH-like)
  5. Implied vs Realized       — IV premium/discount (Deribit)
  6. Term Structure            — near-term vs far-term IV

Data sources (all free):
  - OHLCV price data (realized vol calculation)
  - Deribit public API (BTC/ETH implied vol)

Signal impact:
  Vol crush imminent (high IV, low RV):   -4 LONGs (gamma squeeze risk)
  Vol expansion (low RV, breakout setup): +4 trend signals
  IV premium > 30%:                       -3 all directions (expensive)
  Vol compression extreme:                +5 breakout signals
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ── API endpoints ─────────────────────────────────────────────
_DERIBIT = "https://www.deribit.com/api/v2/public"

# Refresh intervals
_RV_REFRESH  = 300    # 5 min (uses cached OHLCV)
_IV_REFRESH  = 1800   # 30 min (Deribit rate limits)

# Volatility cone percentiles
_VOL_EXTREME_HIGH_PCT = 90   # Above 90th percentile = extremely high vol
_VOL_HIGH_PCT         = 75   # Above 75th percentile
_VOL_LOW_PCT          = 25   # Below 25th percentile
_VOL_EXTREME_LOW_PCT  = 10   # Below 10th percentile = extremely compressed


@dataclass
class RealizedVolData:
    """Multi-window realized volatility."""
    rv_7d: float = 0.0           # 7-day realized vol (annualized %)
    rv_14d: float = 0.0          # 14-day
    rv_30d: float = 0.0          # 30-day
    rv_90d: float = 0.0          # 90-day (for cone baseline)
    rv_trend: str = "NEUTRAL"    # EXPANDING | NEUTRAL | COMPRESSING
    cone_percentile: float = 50  # Where current RV sits in historical range
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _RV_REFRESH * 3


@dataclass
class ImpliedVolData:
    """Options-derived implied volatility."""
    iv_current: float = 0.0       # Current ATM IV (annualized %)
    iv_7d: float = 0.0            # 7-day expiry IV
    iv_30d: float = 0.0           # 30-day expiry IV
    iv_rv_premium: float = 0.0    # IV - RV spread (+ = expensive options)
    term_structure: str = "FLAT"  # CONTANGO | FLAT | BACKWARDATION
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _IV_REFRESH * 3


@dataclass
class VolRegimeData:
    """Volatility regime classification."""
    regime: str = "NEUTRAL"          # COMPRESSED | LOW | NEUTRAL | HIGH | EXTREME
    vol_of_vol: float = 0.0          # Volatility of volatility (clustering)
    days_in_regime: int = 0          # How long current regime has lasted
    breakout_probability: float = 0.5  # P(breakout) based on compression duration
    context_score: float = 0.0       # Context-enhanced breakout score (0-1)
    context_factors: str = ""        # Human-readable active context factors
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _RV_REFRESH * 3


@dataclass
class VolatilitySnapshot:
    """Complete volatility structure snapshot."""
    realized: RealizedVolData = field(default_factory=RealizedVolData)
    implied: ImpliedVolData = field(default_factory=ImpliedVolData)
    regime: VolRegimeData = field(default_factory=VolRegimeData)
    last_update: float = 0.0


class VolatilityStructureAnalyzer:
    """
    Deep volatility analysis for institutional-grade signal filtering.

    Integration points:
      - Engine calls get_signal_intel() for vol-aware confidence
      - Market state builder uses get_vol_regime() for state enrichment
      - Entry refiner uses get_breakout_probability() for timing
    """

    def __init__(self):
        self._snapshot = VolatilitySnapshot()
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._rv_history: List[Tuple[float, float]] = []     # (ts, rv_30d)
        self._MIN_RV_SAMPLES = 10  # Minimum RV observations before trusting percentile
        self._regime_start: float = 0.0
        # Context inputs for compression timer (set externally via set_context)
        self._oi_change_pct: float = 0.0      # OI change % (from leverage mapper)
        self._funding_rate: float = 0.0        # Annualized funding rate %
        self._leverage_zone: str = "NEUTRAL"   # From leverage mapper
        # Directional bias (set externally via set_directional_bias)
        self._whale_intent: str = "NEUTRAL"    # "BULLISH" | "BEARISH" | "NEUTRAL"
        self._flow_direction: str = "NEUTRAL"  # "INFLOW" | "OUTFLOW" | "NEUTRAL"

    async def start(self):
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("📈 VolatilityStructure started (RV·IV·VolRegime·TermStructure)")

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
                logger.warning(f"Vol structure poll error: {e}")
            await asyncio.sleep(_RV_REFRESH)

    async def _update_all(self):
        tasks = [self._compute_realized_vol()]
        now = time.time()
        if now - self._snapshot.implied.timestamp > _IV_REFRESH:
            tasks.append(self._fetch_implied_vol())
        await asyncio.gather(*tasks, return_exceptions=True)
        self._compute_vol_regime()

    # ── Realized Volatility ───────────────────────────────────

    async def _compute_realized_vol(self):
        """
        Compute realized volatility from BTC OHLCV data.
        Uses close-to-close log returns, annualized.
        """
        try:
            from data.api_client import api
            candles = await api.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=2160)  # 90 days
            if not candles or len(candles) < 168:  # Need at least 7 days
                return

            closes = [c[4] for c in candles if c[4] and c[4] > 0]
            if len(closes) < 168:
                return

            # Log returns
            returns = []
            for i in range(1, len(closes)):
                if closes[i] > 0 and closes[i - 1] > 0:
                    returns.append(math.log(closes[i] / closes[i - 1]))

            if len(returns) < 168:
                return

            # Compute RV for different windows (annualized)
            annualize = math.sqrt(365 * 24)  # hourly data → annual

            def _rv(rets, window):
                if len(rets) < window:
                    return 0.0
                subset = rets[-window:]
                mean = sum(subset) / len(subset)
                var = sum((r - mean) ** 2 for r in subset) / (len(subset) - 1)
                return math.sqrt(var) * annualize * 100  # as percentage

            rv_7d = _rv(returns, 168)     # 7 * 24
            rv_14d = _rv(returns, 336)    # 14 * 24
            rv_30d = _rv(returns, 720)    # 30 * 24
            rv_90d = _rv(returns, min(len(returns), 2160))

            # Trend: compare short-term to long-term
            if rv_14d > 0 and rv_30d > 0:
                if rv_7d > rv_30d * 1.3:
                    trend = "EXPANDING"
                elif rv_7d < rv_30d * 0.7:
                    trend = "COMPRESSING"
                else:
                    trend = "NEUTRAL"
            else:
                trend = "NEUTRAL"

            # Volatility cone: where does current RV sit in history?
            self._rv_history.append((time.time(), rv_30d))
            cutoff = time.time() - 86400 * 365
            self._rv_history = [(t, v) for t, v in self._rv_history if t > cutoff]

            # Compute percentile (with warmup guard)
            all_rvs = sorted([v for _, v in self._rv_history])
            if len(all_rvs) >= self._MIN_RV_SAMPLES:
                rank = sum(1 for v in all_rvs if v <= rv_30d) / len(all_rvs) * 100
            else:
                # Not enough history — return neutral to avoid false EXTREME
                rank = 50

            self._snapshot.realized = RealizedVolData(
                rv_7d=rv_7d,
                rv_14d=rv_14d,
                rv_30d=rv_30d,
                rv_90d=rv_90d,
                rv_trend=trend,
                cone_percentile=rank,
                timestamp=time.time(),
            )
            logger.debug(
                f"RV: 7d={rv_7d:.1f}% 30d={rv_30d:.1f}% 90d={rv_90d:.1f}% "
                f"trend={trend} pctile={rank:.0f}"
            )
        except Exception as e:
            logger.error("Realized vol computation failed: %s", e, exc_info=True)

    # ── Implied Volatility (Deribit) ──────────────────────────

    async def _fetch_implied_vol(self):
        """Fetch implied volatility from Deribit for BTC."""
        if not self._session:
            return
        try:
            # Get BTC DVOL (Deribit Volatility Index)
            url = f"{_DERIBIT}/get_volatility_index_data?currency=BTC&resolution=3600&start_timestamp={int((time.time()-86400)*1000)}&end_timestamp={int(time.time()*1000)}"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {}).get("data", [])
                    if result:
                        # DVOL returns [timestamp, open, high, low, close]
                        latest = result[-1]
                        iv_current = latest[4] if len(latest) > 4 else latest[-1]

                        # Compute IV-RV premium
                        rv_30d = self._snapshot.realized.rv_30d
                        iv_rv_premium = iv_current - rv_30d if rv_30d > 0 else 0

                        # Term structure approximation
                        # If only DVOL, approximate from volatility trend
                        if len(result) >= 24:
                            iv_recent = sum(r[4] for r in result[-6:]) / 6
                            iv_older = sum(r[4] for r in result[-24:-18]) / 6
                            if iv_recent > iv_older * 1.05:
                                term = "BACKWARDATION"  # Near-term IV rising = fear
                            elif iv_recent < iv_older * 0.95:
                                term = "CONTANGO"  # Normal — far > near
                            else:
                                term = "FLAT"
                        else:
                            term = "FLAT"

                        self._snapshot.implied = ImpliedVolData(
                            iv_current=iv_current,
                            iv_30d=iv_current,  # DVOL is ~30d equivalent
                            iv_rv_premium=iv_rv_premium,
                            term_structure=term,
                            timestamp=time.time(),
                        )
                        logger.debug(
                            f"IV: DVOL={iv_current:.1f}% premium={iv_rv_premium:+.1f}% "
                            f"term={term}"
                        )
        except Exception as e:
            logger.error("IV fetch from Deribit failed: %s", e, exc_info=True)

    # ── Vol Regime ────────────────────────────────────────────

    def set_context(self, oi_change_pct: float = 0.0, funding_rate: float = 0.0,
                    leverage_zone: str = "NEUTRAL"):
        """Set external context inputs for compression timer scoring."""
        self._oi_change_pct = oi_change_pct
        self._funding_rate = funding_rate
        self._leverage_zone = leverage_zone

    def set_directional_bias(self, whale_intent: str = "NEUTRAL",
                             flow_direction: str = "NEUTRAL"):
        """Set external directional hints for breakout direction bias.

        Args:
            whale_intent: "BULLISH" | "BEARISH" | "NEUTRAL"
            flow_direction: "INFLOW" | "OUTFLOW" | "NEUTRAL"
        """
        self._whale_intent = whale_intent
        self._flow_direction = flow_direction

    def _compute_vol_regime(self):
        """Classify current volatility regime with context-aware breakout scoring."""
        from config.constants import VolCompressionTimer as VCT

        rv = self._snapshot.realized
        if rv.is_stale:
            return

        pct = rv.cone_percentile

        if pct >= _VOL_EXTREME_HIGH_PCT:
            regime = "EXTREME"
        elif pct >= _VOL_HIGH_PCT:
            regime = "HIGH"
        elif pct <= _VOL_EXTREME_LOW_PCT:
            regime = "COMPRESSED"
        elif pct <= _VOL_LOW_PCT:
            regime = "LOW"
        else:
            regime = "NEUTRAL"

        # Track regime duration
        old_regime = self._snapshot.regime.regime
        if regime != old_regime:
            self._regime_start = time.time()

        days_in = (time.time() - self._regime_start) / 86400 if self._regime_start else 0

        # Breakout probability increases with compression duration
        if regime == "COMPRESSED":
            breakout_p = min(0.9, 0.3 + days_in * 0.08)
        elif regime == "LOW":
            breakout_p = min(0.7, 0.2 + days_in * 0.05)
        else:
            breakout_p = 0.3

        # ── Context-aware compression scoring ──────────────────
        context_score = 0.0
        context_factors = []

        if regime in ("COMPRESSED", "LOW") and days_in >= VCT.MIN_COMPRESSION_DAYS:
            # Factor 1: Duration score (saturating exponential with half-life)
            _LN2 = 0.693  # ln(2) — ensures score=0.5 at DURATION_HALF_LIFE_DAYS
            duration_score = min(
                VCT.DURATION_MAX_SCORE,
                1.0 - math.exp(-_LN2 * days_in / max(VCT.DURATION_HALF_LIFE_DAYS, 1))
            )

            # Factor 2: OI buildup score
            oi_score = 0.0
            if self._oi_change_pct > VCT.OI_BUILDING_THRESHOLD:
                oi_score = min(1.0, self._oi_change_pct / (VCT.OI_BUILDING_THRESHOLD * 3))
                context_factors.append(f"OI+{self._oi_change_pct:.1f}%")

            # Factor 3: Funding neutrality score (closer to 0 = more neutral = higher)
            funding_score = 0.0
            if VCT.FUNDING_NEUTRAL_LOW <= self._funding_rate <= VCT.FUNDING_NEUTRAL_HIGH:
                # Linearly map: center=1.0, edges=0.0
                half_band = (VCT.FUNDING_NEUTRAL_HIGH - VCT.FUNDING_NEUTRAL_LOW) / 2
                if half_band > 0:
                    center = (VCT.FUNDING_NEUTRAL_HIGH + VCT.FUNDING_NEUTRAL_LOW) / 2
                    funding_score = max(0.0, 1.0 - abs(self._funding_rate - center) / half_band)
                    if funding_score > 0.3:
                        context_factors.append("FundNeutral")

            # Factor 4: Leverage zone score
            leverage_score = 0.0
            if self._leverage_zone in ("HIGH", "EXTREME"):
                leverage_score = 1.0 if self._leverage_zone == "EXTREME" else 0.7
                context_factors.append(f"Lev:{self._leverage_zone}")
            elif self._leverage_zone == "NEUTRAL" and self._oi_change_pct > 0:
                leverage_score = 0.3

            # Weighted blend
            context_score = (
                VCT.WEIGHT_DURATION * duration_score +
                VCT.WEIGHT_OI_BUILDUP * oi_score +
                VCT.WEIGHT_FUNDING_NEUTRAL * funding_score +
                VCT.WEIGHT_LEVERAGE_ZONE * leverage_score
            )

            # Context can boost breakout probability
            if context_score >= VCT.CONTEXT_BOOST_THRESHOLD:
                breakout_p = min(0.95, breakout_p + context_score * 0.15)

        # Vol-of-vol: standard deviation of recent RV changes
        if len(self._rv_history) >= 10:
            recent_rvs = [v for _, v in self._rv_history[-20:]]
            if len(recent_rvs) >= 5:
                mean_rv = sum(recent_rvs) / len(recent_rvs)
                vol_of_vol = math.sqrt(
                    sum((r - mean_rv) ** 2 for r in recent_rvs) / (len(recent_rvs) - 1)
                )
            else:
                vol_of_vol = 0
        else:
            vol_of_vol = 0

        self._snapshot.regime = VolRegimeData(
            regime=regime,
            vol_of_vol=vol_of_vol,
            days_in_regime=int(days_in),
            breakout_probability=breakout_p,
            context_score=round(context_score, 3),
            context_factors="+".join(context_factors) if context_factors else "",
            timestamp=time.time(),
        )

    # ── Signal Intelligence ───────────────────────────────────

    def get_signal_intel(self, symbol: str, direction: str) -> Tuple[int, str]:
        """Returns (confidence_delta, note) for a given signal."""
        from config.constants import VolCompressionTimer as VCT

        delta = 0
        notes = []
        snap = self._snapshot

        # Vol regime impact
        if not snap.regime.is_stale:
            if snap.regime.regime == "COMPRESSED" and snap.regime.breakout_probability > 0.6:
                delta += +5  # Breakout signals benefit from compression
                ctx_label = ""
                if snap.regime.context_score >= VCT.CONTEXT_BOOST_THRESHOLD:
                    delta += VCT.CONTEXT_DELTA_BONUS
                    ctx_label = f" ctx={snap.regime.context_score:.0%}"
                    if snap.regime.context_factors:
                        ctx_label += f"[{snap.regime.context_factors}]"

                # Directional bias: whale intent + flow direction hint breakout direction
                bias_score = 0.0
                is_long = direction == "LONG"
                _w_intent = getattr(self, '_whale_intent', 'NEUTRAL')
                _f_dir = getattr(self, '_flow_direction', 'NEUTRAL')

                if _w_intent == "BULLISH":
                    bias_score += VCT.DIRECTIONAL_BIAS_WEIGHT_WHALE
                elif _w_intent == "BEARISH":
                    bias_score -= VCT.DIRECTIONAL_BIAS_WEIGHT_WHALE

                if _f_dir in ("STRONG_INFLOW", "INFLOW"):
                    bias_score += VCT.DIRECTIONAL_BIAS_WEIGHT_FLOW
                elif _f_dir in ("STRONG_OUTFLOW", "OUTFLOW"):
                    bias_score -= VCT.DIRECTIONAL_BIAS_WEIGHT_FLOW

                # If signal direction aligns with bias → bonus
                if abs(bias_score) >= VCT.DIRECTIONAL_BIAS_THRESHOLD:
                    bias_is_long = bias_score > 0
                    if bias_is_long == is_long:
                        delta += VCT.DIRECTIONAL_BIAS_DELTA
                        ctx_label += f" bias={'LONG' if bias_is_long else 'SHORT'}"
                    else:
                        delta -= VCT.DIRECTIONAL_BIAS_DELTA
                        ctx_label += f" bias={'LONG' if bias_is_long else 'SHORT'}≠dir"

                notes.append(
                    f"📈 Vol compressed {snap.regime.days_in_regime}d "
                    f"(P(breakout)={snap.regime.breakout_probability:.0%}{ctx_label})"
                )
            elif snap.regime.regime == "EXTREME":
                delta += -4  # Extreme vol → harder to profit from direction
                notes.append(f"📈 Extreme vol (RV pctile={snap.realized.cone_percentile:.0f})")

        # IV-RV premium impact (only for BTC-related)
        if not snap.implied.is_stale and "BTC" in symbol.upper():
            if snap.implied.iv_rv_premium > 30:
                delta += -3  # Options very expensive → risky to enter
                notes.append(f"📈 IV premium +{snap.implied.iv_rv_premium:.0f}% (expensive)")
            elif snap.implied.iv_rv_premium < -10:
                delta += +3  # Options cheap → implied vol low, potential for expansion
                notes.append(f"📈 IV discount {snap.implied.iv_rv_premium:.0f}% (cheap)")

        # Backwardation warning
        if not snap.implied.is_stale:
            if snap.implied.term_structure == "BACKWARDATION":
                delta += -2
                notes.append("📈 IV backwardation (near-term fear)")

        # Realized vol trend
        if not snap.realized.is_stale:
            if snap.realized.rv_trend == "EXPANDING":
                delta += -2  # Expanding vol = harder environment
                notes.append(f"📈 Vol expanding (7d={snap.realized.rv_7d:.0f}% vs 30d={snap.realized.rv_30d:.0f}%)")

        # Vol-of-vol (volatility clustering) — erratic vol regimes are harder to trade
        if not snap.regime.is_stale and snap.regime.vol_of_vol > 0:
            from config.constants import VolOfVol as _VoV
            if snap.regime.vol_of_vol >= _VoV.HIGH_THRESHOLD:
                delta += _VoV.HIGH_PENALTY
                notes.append(f"📈 High vol clustering (VoV={snap.regime.vol_of_vol:.3f})")
            elif snap.regime.vol_of_vol <= _VoV.LOW_THRESHOLD:
                delta += _VoV.LOW_BONUS
                notes.append(f"📈 Stable vol regime (VoV={snap.regime.vol_of_vol:.3f})")

        delta = max(-10, min(10, delta))
        note = " | ".join(notes) if notes else ""
        return delta, note

    def get_vol_regime(self) -> str:
        """Returns volatility regime for market state builder."""
        snap = self._snapshot
        if snap.regime.is_stale:
            return "UNKNOWN"
        return snap.regime.regime

    def get_breakout_probability(self) -> float:
        """Returns breakout probability for entry refiner."""
        return self._snapshot.regime.breakout_probability

    def get_snapshot(self) -> VolatilitySnapshot:
        return self._snapshot


# Module-level singleton
volatility_analyzer = VolatilityStructureAnalyzer()
