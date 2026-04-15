"""
TitanBot Pro — Stablecoin Flow Analyzer
=========================================
Monitors stablecoin supply, minting/burning, and exchange flows.
Stablecoins are the "fuel" of crypto markets — rising supply = bullish.

Data sources (all free, no API key):
  - DefiLlama Stablecoins API (aggregate supply data)
  - CoinGecko (individual stablecoin market caps)
  - Blockchain.com / Etherscan (USDT/USDC supply changes)

Metrics:
  1. Total Stablecoin Supply   — aggregate across USDT, USDC, DAI, BUSD, TUSD
  2. Supply Change (7d, 30d)   — minting vs burning trend
  3. Exchange Stablecoin Ratio — stablecoin balance on exchanges vs total
  4. Dominance                 — stablecoin market cap as % of total crypto

Signal impact:
  Supply growing > 2%/week:   +5 confidence on LONGs (fresh capital inflow)
  Supply shrinking > 2%/week: -5 confidence on LONGs (capital flight)
  High exchange ratio:        +4 LONGs (ready to buy)
  Low exchange ratio:         -3 LONGs (no dry powder)
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

from config.constants import FlowAcceleration as FAC

logger = logging.getLogger(__name__)

# ── API endpoints ─────────────────────────────────────────────
_DEFILLAMA_STABLES  = "https://stablecoins.llama.fi"
_COINGECKO_BASE     = "https://api.coingecko.com/api/v3"

# Refresh intervals
_SUPPLY_REFRESH     = 1800   # 30 min for supply data
_EXCHANGE_REFRESH   = 900    # 15 min for exchange ratio

# Thresholds
_SUPPLY_GROWTH_BULLISH     = 0.02    # 2% weekly growth = bullish
_SUPPLY_SHRINK_BEARISH     = -0.02   # 2% weekly shrink = bearish
_SUPPLY_GROWTH_STRONG      = 0.05    # 5% weekly growth = very bullish
_SUPPLY_SHRINK_STRONG      = -0.05   # 5% weekly shrink = very bearish
_EXCHANGE_RATIO_HIGH       = 0.25    # > 25% on exchanges = high buying power
_EXCHANGE_RATIO_LOW        = 0.10    # < 10% on exchanges = low dry powder


@dataclass
class StablecoinSupplyData:
    """Aggregate stablecoin supply metrics."""
    total_supply_usd: float = 0.0
    supply_change_7d_pct: float = 0.0
    supply_change_30d_pct: float = 0.0

    # Individual stablecoin supplies
    usdt_supply: float = 0.0
    usdc_supply: float = 0.0
    dai_supply: float = 0.0

    # Trend classification
    trend: str = "NEUTRAL"   # STRONG_INFLOW | INFLOW | NEUTRAL | OUTFLOW | STRONG_OUTFLOW
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _SUPPLY_REFRESH * 3


@dataclass
class StablecoinExchangeData:
    """Stablecoin presence on exchanges."""
    exchange_ratio: float = 0.0       # % of stables on exchanges
    exchange_supply_usd: float = 0.0
    zone: str = "NEUTRAL"             # HIGH_DRY_POWDER | NEUTRAL | LOW_DRY_POWDER
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _EXCHANGE_REFRESH * 3


@dataclass
class StablecoinDominanceData:
    """Stablecoin as % of total crypto market."""
    dominance_pct: float = 0.0
    dominance_change_7d: float = 0.0
    # Rising dominance = risk-off (people moving to stables)
    # Falling dominance = risk-on (people moving to crypto)
    signal: str = "NEUTRAL"  # RISK_ON | NEUTRAL | RISK_OFF
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _SUPPLY_REFRESH * 3


@dataclass
class StablecoinSnapshot:
    """Complete stablecoin flow snapshot."""
    supply: StablecoinSupplyData = field(default_factory=StablecoinSupplyData)
    exchange: StablecoinExchangeData = field(default_factory=StablecoinExchangeData)
    dominance: StablecoinDominanceData = field(default_factory=StablecoinDominanceData)
    last_update: float = 0.0


@dataclass
class FlowDynamics:
    """Velocity, acceleration, and shock detection for stablecoin flows."""
    velocity: float = 0.0           # Rate of change of supply (1st derivative)
    acceleration: float = 0.0       # Rate of change of velocity (2nd derivative)
    is_accelerating: bool = False   # True when acceleration is positive and significant
    is_decelerating: bool = False   # True when acceleration is negative and significant
    shock_detected: bool = False    # True when flow > SHOCK_MULT × rolling avg
    shock_confirmed: bool = False   # True when shock persists for SHOCK_CONFIRM_INTERVALS
    shock_direction: str = "NEUTRAL"  # "INFLOW_SHOCK" | "OUTFLOW_SHOCK" | "NEUTRAL"
    consecutive_shocks: int = 0     # How many consecutive intervals triggered shock
    timestamp: float = 0.0


class StablecoinFlowAnalyzer:
    """
    Monitors stablecoin supply dynamics and exchange positioning.

    Integration points:
      - Engine calls get_signal_intel() for per-signal confidence adjustments
      - Ensemble voter uses get_liquidity_signal() for capital availability
      - Macro score enrichment via get_macro_context()
    """

    def __init__(self):
        self._snapshot = StablecoinSnapshot()
        self._flow_dynamics = FlowDynamics()
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._supply_history: List[Tuple[float, float]] = []  # (ts, total_supply)
        # Flow acceleration tracking: deque of (timestamp, supply_change_pct)
        self._supply_velocity_window: deque = deque(maxlen=FAC.VELOCITY_WINDOW)
        self._consecutive_shock_count: int = 0
        # Volume gate: external 24h market volume (set by engine)
        self._market_volume_24h: float = 0.0

    async def start(self):
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20)
        )
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("💵 StablecoinFlowAnalyzer started (USDT·USDC·DAI supply tracking)")

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
                logger.warning(f"Stablecoin poll error: {e}")
            await asyncio.sleep(_SUPPLY_REFRESH)

    async def _update_all(self):
        await asyncio.gather(
            self._fetch_supply(),
            self._fetch_dominance(),
            return_exceptions=True,
        )
        # Compute flow dynamics after supply data updates
        self._compute_flow_dynamics()

    # ── Supply Tracking ───────────────────────────────────────

    async def _fetch_supply(self):
        """Fetch aggregate stablecoin supply from DefiLlama."""
        if not self._session:
            return
        try:
            url = f"{_DEFILLAMA_STABLES}/stablecoins?includePrices=false"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    stables = data.get("peggedAssets", [])

                    total = 0.0
                    usdt = usdc = dai = 0.0

                    for s in stables:
                        name = s.get("name", "").upper()
                        symbol = s.get("symbol", "").upper()
                        chains = s.get("chainCirculating", {})

                        # Sum across all chains
                        supply = 0.0
                        for chain_data in chains.values():
                            peg = chain_data.get("current", {}).get("peggedUSD", 0)
                            if isinstance(peg, (int, float)):
                                supply += peg

                        total += supply
                        if symbol == "USDT" or name == "TETHER":
                            usdt = supply
                        elif symbol == "USDC" or name == "USD COIN":
                            usdc = supply
                        elif symbol == "DAI":
                            dai = supply

                    if total > 0:
                        # Calculate changes from history
                        change_7d = 0.0
                        change_30d = 0.0
                        now = time.time()

                        self._supply_history.append((now, total))
                        # Clean old entries (keep 45 days)
                        cutoff = now - 86400 * 45
                        self._supply_history = [(t, v) for t, v in self._supply_history if t > cutoff]

                        # Find 7d and 30d ago values
                        for ts, val in reversed(self._supply_history):
                            age_days = (now - ts) / 86400
                            if 6.5 < age_days < 7.5 and change_7d == 0:
                                change_7d = (total - val) / val if val > 0 else 0
                            elif 29 < age_days < 31 and change_30d == 0:
                                change_30d = (total - val) / val if val > 0 else 0

                        # Classify trend
                        if change_7d >= _SUPPLY_GROWTH_STRONG:
                            trend = "STRONG_INFLOW"
                        elif change_7d >= _SUPPLY_GROWTH_BULLISH:
                            trend = "INFLOW"
                        elif change_7d <= _SUPPLY_SHRINK_STRONG:
                            trend = "STRONG_OUTFLOW"
                        elif change_7d <= _SUPPLY_SHRINK_BEARISH:
                            trend = "OUTFLOW"
                        else:
                            trend = "NEUTRAL"

                        self._snapshot.supply = StablecoinSupplyData(
                            total_supply_usd=total,
                            supply_change_7d_pct=change_7d * 100,
                            supply_change_30d_pct=change_30d * 100,
                            usdt_supply=usdt,
                            usdc_supply=usdc,
                            dai_supply=dai,
                            trend=trend,
                            timestamp=time.time(),
                        )
                        logger.debug(
                            f"Stablecoin supply=${total/1e9:.1f}B "
                            f"7d={change_7d*100:+.2f}% trend={trend}"
                        )
        except Exception as e:
            logger.error("Stablecoin supply fetch failed: %s", e, exc_info=True)

    # ── Dominance ─────────────────────────────────────────────

    async def _fetch_dominance(self):
        """Fetch stablecoin dominance relative to total crypto market cap."""
        if not self._session:
            return
        try:
            url = f"{_COINGECKO_BASE}/global"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    global_data = data.get("data", {})
                    total_mcap = global_data.get("total_market_cap", {}).get("usd", 0)

                    if total_mcap > 0 and not self._snapshot.supply.is_stale:
                        stable_supply = self._snapshot.supply.total_supply_usd
                        dominance = (stable_supply / total_mcap) * 100

                        # Rising dominance = capital moving to safety (bearish for crypto)
                        # Falling dominance = capital moving to risk (bullish)
                        if dominance > 12:  # historically high
                            signal = "RISK_OFF"
                        elif dominance < 6:  # historically low
                            signal = "RISK_ON"
                        else:
                            signal = "NEUTRAL"

                        self._snapshot.dominance = StablecoinDominanceData(
                            dominance_pct=dominance,
                            signal=signal,
                            timestamp=time.time(),
                        )
                        logger.debug(f"Stablecoin dominance={dominance:.1f}% signal={signal}")
        except Exception as e:
            logger.error("Stablecoin dominance computation failed: %s", e, exc_info=True)

    # ── Flow Dynamics (velocity / acceleration / shock) ──────

    def _compute_flow_dynamics(self):
        """
        Compute velocity (1st derivative) and acceleration (2nd derivative)
        of stablecoin supply changes.  Detects shock events when current
        flow exceeds SHOCK_MULT × rolling average, confirmed after
        SHOCK_CONFIRM_INTERVALS consecutive intervals.
        """
        snap = self._snapshot.supply
        if snap.is_stale or snap.total_supply_usd <= 0:
            self._consecutive_shock_count = 0
            self._flow_dynamics = FlowDynamics()
            return

        # Record current supply change as a velocity data point
        now = time.time()
        self._supply_velocity_window.append((now, snap.supply_change_7d_pct))

        window = list(self._supply_velocity_window)
        if len(window) < 2:
            self._flow_dynamics = FlowDynamics(timestamp=now)
            return

        # Velocity: latest supply change rate
        velocity = window[-1][1]

        # Acceleration: change in velocity between last two data points
        prev_velocity = window[-2][1]
        acceleration = velocity - prev_velocity

        # Rolling average of absolute velocity for shock detection
        abs_velocities = [abs(v) for _, v in window]
        rolling_avg = sum(abs_velocities) / len(abs_velocities) if abs_velocities else FAC.MIN_ROLLING_AVG_FLOOR

        # Shock detection: current absolute velocity > SHOCK_MULT × rolling avg
        current_abs = abs(velocity)
        shock_detected = current_abs > FAC.SHOCK_MULT * max(rolling_avg, FAC.MIN_ROLLING_AVG_FLOOR)

        if shock_detected:
            self._consecutive_shock_count += 1
        else:
            self._consecutive_shock_count = 0

        shock_confirmed = self._consecutive_shock_count >= FAC.SHOCK_CONFIRM_INTERVALS

        # Determine shock direction
        shock_direction = "NEUTRAL"
        if shock_confirmed:
            shock_direction = "INFLOW_SHOCK" if velocity > 0 else "OUTFLOW_SHOCK"

        # Determine acceleration state (threshold: > 0.5% change in rate)
        is_accelerating = acceleration > 0.5 and velocity > 0
        is_decelerating = acceleration < -0.5 and velocity < 0

        self._flow_dynamics = FlowDynamics(
            velocity=velocity,
            acceleration=acceleration,
            is_accelerating=is_accelerating,
            is_decelerating=is_decelerating,
            shock_detected=shock_detected,
            shock_confirmed=shock_confirmed,
            shock_direction=shock_direction,
            consecutive_shocks=self._consecutive_shock_count,
            timestamp=now,
        )

        if shock_confirmed:
            logger.info(
                f"⚡ Stablecoin SHOCK {shock_direction}: velocity={velocity:+.2f}%/wk "
                f"accel={acceleration:+.2f} ({self._consecutive_shock_count} consecutive)"
            )
        elif is_accelerating or is_decelerating:
            logger.debug(
                f"📈 Stablecoin flow {'accelerating' if is_accelerating else 'decelerating'}: "
                f"velocity={velocity:+.2f}%/wk accel={acceleration:+.2f}"
            )

    # ── Volume gate ─────────────────────────────────────────────

    def set_market_volume(self, volume_24h_usd: float):
        """Set 24h market volume for thin-market volume gating."""
        self._market_volume_24h = volume_24h_usd

    # ── Signal Intelligence ───────────────────────────────────

    def get_signal_intel(self, symbol: str, direction: str) -> Tuple[int, str]:
        """
        Returns (confidence_delta, note) for a given signal.
        Now includes flow acceleration and shock detection signals.
        """
        delta = 0
        notes = []
        is_long = direction == "LONG"
        snap = self._snapshot
        dyn = self._flow_dynamics

        # Supply trend impact (existing logic)
        if not snap.supply.is_stale:
            if snap.supply.trend == "STRONG_INFLOW":
                delta += +6 if is_long else -3
                notes.append(f"💵 Stablecoin supply +{snap.supply.supply_change_7d_pct:.1f}%/wk (strong inflow)")
            elif snap.supply.trend == "INFLOW":
                delta += +4 if is_long else -2
                notes.append(f"💵 Stablecoin inflow +{snap.supply.supply_change_7d_pct:.1f}%/wk")
            elif snap.supply.trend == "STRONG_OUTFLOW":
                delta += -6 if is_long else +3
                notes.append(f"💵 Stablecoin outflow {snap.supply.supply_change_7d_pct:.1f}%/wk (capital flight)")
            elif snap.supply.trend == "OUTFLOW":
                delta += -4 if is_long else +2
                notes.append(f"💵 Stablecoin outflow {snap.supply.supply_change_7d_pct:.1f}%/wk")

        # Dominance impact (existing logic)
        if not snap.dominance.is_stale:
            if snap.dominance.signal == "RISK_OFF":
                delta += -3 if is_long else +2
                notes.append(f"Stable dominance {snap.dominance.dominance_pct:.1f}% (risk-off)")
            elif snap.dominance.signal == "RISK_ON":
                delta += +3 if is_long else -2
                notes.append(f"Stable dominance {snap.dominance.dominance_pct:.1f}% (risk-on)")

        # Flow acceleration impact (NEW — Tier 1 upgrade)
        # Volume gate: dampen acceleration signals in thin markets.
        # When _market_volume_24h is 0 (unset/unavailable), the gate
        # is bypassed (0 < 0 is False) to avoid false dampening.
        _vol_gate = (
            FAC.LOW_VOLUME_DAMPENER
            if 0 < self._market_volume_24h < FAC.LOW_VOLUME_THRESHOLD_USD
            else 1.0
        )

        if not snap.supply.is_stale and dyn.timestamp > 0:
            if dyn.shock_confirmed:
                # Confirmed shock event — strongest signal
                if dyn.shock_direction == "INFLOW_SHOCK":
                    _raw = +FAC.SHOCK_BULLISH_DELTA if is_long else -3
                    delta += int(round(_raw * _vol_gate))
                    notes.append(
                        f"⚡ INFLOW SHOCK: velocity={dyn.velocity:+.1f}%/wk "
                        f"accel={dyn.acceleration:+.1f} (confirmed {dyn.consecutive_shocks}×)"
                    )
                elif dyn.shock_direction == "OUTFLOW_SHOCK":
                    _raw = -FAC.SHOCK_BEARISH_DELTA if is_long else +3
                    delta += int(round(_raw * _vol_gate))
                    notes.append(
                        f"⚡ OUTFLOW SHOCK: velocity={dyn.velocity:+.1f}%/wk "
                        f"accel={dyn.acceleration:+.1f} (confirmed {dyn.consecutive_shocks}×)"
                    )
            elif dyn.is_accelerating:
                # Inflow acceleration — capital deploying faster
                _raw = +FAC.ACCEL_BULLISH_DELTA if is_long else -2
                delta += int(round(_raw * _vol_gate))
                notes.append(
                    f"📈 Flow accelerating: velocity={dyn.velocity:+.1f}%/wk "
                    f"accel={dyn.acceleration:+.1f}"
                )
            elif dyn.is_decelerating:
                # Outflow acceleration — capital fleeing faster
                _raw = -FAC.ACCEL_BEARISH_DELTA if is_long else +2
                delta += int(round(_raw * _vol_gate))
                notes.append(
                    f"📉 Flow decelerating: velocity={dyn.velocity:+.1f}%/wk "
                    f"accel={dyn.acceleration:+.1f}"
                )

        delta = max(-15, min(15, delta))
        note = " | ".join(notes) if notes else ""
        return delta, note

    def get_liquidity_signal(self) -> str:
        """Returns stablecoin liquidity state for ensemble voter."""
        snap = self._snapshot
        if snap.supply.is_stale:
            return "UNKNOWN"
        if snap.supply.trend in ("STRONG_INFLOW", "INFLOW"):
            return "AMPLE"
        if snap.supply.trend in ("STRONG_OUTFLOW", "OUTFLOW"):
            return "DRAINING"
        return "NEUTRAL"

    def get_macro_context(self) -> dict:
        """Returns stablecoin data for AI analyst context."""
        snap = self._snapshot
        dyn = self._flow_dynamics
        return {
            "stablecoin_supply_usd": snap.supply.total_supply_usd,
            "stablecoin_change_7d_pct": snap.supply.supply_change_7d_pct,
            "stablecoin_trend": snap.supply.trend,
            "stablecoin_dominance_pct": snap.dominance.dominance_pct,
            "stablecoin_dominance_signal": snap.dominance.signal,
            "flow_velocity": dyn.velocity,
            "flow_acceleration": dyn.acceleration,
            "flow_shock_confirmed": dyn.shock_confirmed,
            "flow_shock_direction": dyn.shock_direction,
        }

    def get_snapshot(self) -> StablecoinSnapshot:
        return self._snapshot

    def get_flow_dynamics(self) -> FlowDynamics:
        """Return current flow dynamics for ensemble voter / diagnostics."""
        return self._flow_dynamics


# Module-level singleton
stablecoin_analyzer = StablecoinFlowAnalyzer()
