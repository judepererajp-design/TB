"""
TitanBot Pro — Capital Efficiency & Leverage Mapper
=====================================================
Maps leverage concentration, margin usage, and liquidation cascade
potential across the market. Predicts WHERE price is forced to move.

This enhances the existing liquidation_analyzer with:
  1. Leverage Concentration Mapping — where is leverage piled up?
  2. Cascade Prediction — which liquidation levels trigger chain reactions?
  3. Margin Utilization — how much of the market's margin is deployed?
  4. Effective Leverage Ratio — market-wide leverage estimation

Data sources:
  - Binance futures (OI, funding, liquidations)
  - Existing liquidation_analyzer data
  - Existing derivatives analyzer data

Signal impact:
  High leverage zone near entry:  -4 confidence (wipeout risk)
  Leverage unwinding:             +3 trend signals (clean moves)
  Cascade risk above/below:       adjust TP/SL targets
  Extreme effective leverage:     -5 all signals (fragile market)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Thresholds
_EFFECTIVE_LEVERAGE_EXTREME = 0.40   # > 40% = extreme
_EFFECTIVE_LEVERAGE_HIGH    = 0.30   # > 30% = high
_CASCADE_MIN_USD           = 5_000_000  # $5M+ liquidation cluster = cascade risk


@dataclass
class LeverageMapData:
    """Market-wide leverage mapping."""
    effective_leverage: float = 0.0      # OI / Market Cap ratio (proxy for leverage)
    leverage_trend: str = "NEUTRAL"      # BUILDING | NEUTRAL | UNWINDING
    oi_to_mcap_pct: float = 0.0          # OI as % of market cap
    oi_change_pct: float = 0.0           # Actual OI change % (e.g. oi_change_1h)
    funding_weighted: float = 0.0        # Volume-weighted funding rate
    zone: str = "NEUTRAL"                # LOW | NEUTRAL | HIGH | EXTREME
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > 600


@dataclass
class CascadeRisk:
    """Liquidation cascade risk assessment."""
    cascade_risk_long: str = "LOW"    # LOW | MEDIUM | HIGH | CRITICAL
    cascade_risk_short: str = "LOW"
    nearest_long_cluster_pct: float = 0.0   # Distance to nearest long liq cluster (%)
    nearest_short_cluster_pct: float = 0.0  # Distance to nearest short liq cluster (%)
    estimated_long_cascade_usd: float = 0.0   # USD at risk in nearest LONG cascade
    estimated_short_cascade_usd: float = 0.0  # USD at risk in nearest SHORT cascade
    timestamp: float = 0.0

    @property
    def estimated_cascade_usd(self) -> float:
        """Backward-compatible: returns max of both sides."""
        return max(self.estimated_long_cascade_usd, self.estimated_short_cascade_usd)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > 600


@dataclass
class CascadePressureScore:
    """Momentum-weighted cascade pressure scoring.

    pressure = (cluster_size / distance) × momentum_factor × persistence_bonus
    """
    long_pressure: float = 0.0          # Pressure score for long cascade
    short_pressure: float = 0.0         # Pressure score for short cascade
    long_pressure_level: str = "LOW"    # LOW | MEDIUM | HIGH | CRITICAL
    short_pressure_level: str = "LOW"
    momentum_toward_long: bool = False  # Price moving toward long cluster
    momentum_toward_short: bool = False # Price moving toward short cluster
    long_persistence_mins: float = 0.0  # How long long cluster has persisted
    short_persistence_mins: float = 0.0 # How long short cluster has persisted
    timestamp: float = 0.0


@dataclass
class LeverageSnapshot:
    """Complete leverage mapping snapshot."""
    leverage: LeverageMapData = field(default_factory=LeverageMapData)
    cascade: CascadeRisk = field(default_factory=CascadeRisk)
    pressure: CascadePressureScore = field(default_factory=CascadePressureScore)
    last_update: float = 0.0


class LeverageMapper:
    """
    Maps leverage concentration and cascade risks.

    Integration:
      - Engine calls get_signal_intel() for leverage-aware confidence
      - Entry refiner uses get_cascade_risk() for TP/SL adjustment
      - Works with liquidation_analyzer and derivatives data
    """

    def __init__(self):
        self._snapshot = LeverageSnapshot()
        # Price history for momentum calculation
        from collections import deque
        self._price_history: deque = deque(maxlen=20)
        # Cluster persistence tracking: (first_seen_ts, last_dist_pct)
        self._long_cluster_first_seen: float = 0.0
        self._short_cluster_first_seen: float = 0.0
        self._long_cluster_last_dist: float = float('inf')
        self._short_cluster_last_dist: float = float('inf')

    def update(self, oi_usd: float = 0, market_cap: float = 0,
               funding_rate: float = 0, oi_change_pct: float = 0,
               long_liq_clusters: Optional[List[Dict]] = None,
               short_liq_clusters: Optional[List[Dict]] = None,
               current_price: float = 0):
        """
        Update leverage map with latest data.
        Called by the engine with data from multiple analyzers.
        """
        try:
            # Effective leverage ratio
            if market_cap > 0 and oi_usd > 0:
                eff_lev = oi_usd / market_cap
            else:
                eff_lev = 0.0

            # Classify zone
            if eff_lev >= _EFFECTIVE_LEVERAGE_EXTREME:
                zone = "EXTREME"
            elif eff_lev >= _EFFECTIVE_LEVERAGE_HIGH:
                zone = "HIGH"
            elif eff_lev < 0.15:
                zone = "LOW"
            else:
                zone = "NEUTRAL"

            # Leverage trend from OI change
            if oi_change_pct > 5:
                trend = "BUILDING"
            elif oi_change_pct < -5:
                trend = "UNWINDING"
            else:
                trend = "NEUTRAL"

            self._snapshot.leverage = LeverageMapData(
                effective_leverage=eff_lev,
                leverage_trend=trend,
                oi_to_mcap_pct=eff_lev * 100,
                oi_change_pct=oi_change_pct,
                funding_weighted=funding_rate,
                zone=zone,
                timestamp=time.time(),
            )

            if zone in ("EXTREME", "HIGH"):
                logger.warning(
                    "LeverageMapper: zone=%s eff_lev=%.3f trend=%s OI_chg=%.1f%%",
                    zone, eff_lev, trend, oi_change_pct,
                )
            else:
                logger.debug(
                    "LeverageMapper update: zone=%s eff_lev=%.3f trend=%s",
                    zone, eff_lev, trend,
                )

            # Cascade risk assessment
            if current_price > 0:
                self._price_history.append(current_price)
                self._assess_cascade(current_price, long_liq_clusters or [], short_liq_clusters or [])
                self._compute_pressure(current_price)

            self._snapshot.last_update = time.time()
        except Exception as e:
            logger.error("LeverageMapper.update() failed: %s", e, exc_info=True)

    def _assess_cascade(self, price: float, long_clusters: List[Dict], short_clusters: List[Dict]):
        """Assess liquidation cascade risks."""
        try:
            return self.__assess_cascade_impl(price, long_clusters, short_clusters)
        except Exception as e:
            logger.error("LeverageMapper._assess_cascade() failed: %s", e, exc_info=True)

    def __assess_cascade_impl(self, price: float, long_clusters: List[Dict], short_clusters: List[Dict]):
        # Find nearest long liquidation cluster (below current price)
        nearest_long_dist = float('inf')
        nearest_long_usd = 0.0
        for cluster in long_clusters:
            cluster_price = cluster.get("price", 0)
            cluster_usd = cluster.get("usd", 0)
            if 0 < cluster_price < price:
                dist_pct = (price - cluster_price) / price * 100
                if dist_pct < nearest_long_dist:
                    nearest_long_dist = dist_pct
                    nearest_long_usd = cluster_usd

        # Find nearest short liquidation cluster (above current price)
        nearest_short_dist = float('inf')
        nearest_short_usd = 0.0
        for cluster in short_clusters:
            cluster_price = cluster.get("price", 0)
            cluster_usd = cluster.get("usd", 0)
            if cluster_price > price:
                dist_pct = (cluster_price - price) / price * 100
                if dist_pct < nearest_short_dist:
                    nearest_short_dist = dist_pct
                    nearest_short_usd = cluster_usd

        # Classify cascade risk
        def _risk_level(dist: float, usd: float) -> str:
            if dist == float('inf'):
                return "LOW"
            if dist < 2 and usd > _CASCADE_MIN_USD * 3:
                return "CRITICAL"
            if dist < 3 and usd > _CASCADE_MIN_USD:
                return "HIGH"
            if dist < 5 and usd > _CASCADE_MIN_USD * 0.5:
                return "MEDIUM"
            return "LOW"

        self._snapshot.cascade = CascadeRisk(
            cascade_risk_long=_risk_level(nearest_long_dist, nearest_long_usd),
            cascade_risk_short=_risk_level(nearest_short_dist, nearest_short_usd),
            nearest_long_cluster_pct=nearest_long_dist if nearest_long_dist != float('inf') else 0,
            nearest_short_cluster_pct=nearest_short_dist if nearest_short_dist != float('inf') else 0,
            estimated_long_cascade_usd=nearest_long_usd,
            estimated_short_cascade_usd=nearest_short_usd,
            timestamp=time.time(),
        )
        risk_long = self._snapshot.cascade.cascade_risk_long
        risk_short = self._snapshot.cascade.cascade_risk_short
        if risk_long in ("CRITICAL", "HIGH") or risk_short in ("CRITICAL", "HIGH"):
            logger.info(
                "Cascade risk: long=%s (%.1f%% below, $%.0fM) short=%s (%.1f%% above, $%.0fM)",
                risk_long, nearest_long_dist if nearest_long_dist != float('inf') else 0,
                nearest_long_usd / 1_000_000,
                risk_short, nearest_short_dist if nearest_short_dist != float('inf') else 0,
                nearest_short_usd / 1_000_000,
            )

    def _compute_pressure(self, current_price: float):
        """Compute momentum-weighted cascade pressure scores with persistence."""
        from config.constants import CascadePressure as CP

        cascade = self._snapshot.cascade
        prices = list(self._price_history)
        window = min(CP.MOMENTUM_WINDOW, len(prices))
        now = time.time()

        # Compute recent price momentum (% change over window)
        if window >= 2:
            price_change_pct = (prices[-1] - prices[-window]) / prices[-window] * 100
        else:
            price_change_pct = 0.0

        # Moving DOWN → toward long liquidations (below price)
        momentum_toward_long = price_change_pct < -CP.MOMENTUM_TOWARD_THRESHOLD
        # Moving UP → toward short liquidations (above price)
        momentum_toward_short = price_change_pct > CP.MOMENTUM_TOWARD_THRESHOLD

        # ── Cluster persistence tracking ──────────────────────────
        # A "cluster" persists if distance stays within ~50% of original
        _long_dist = cascade.nearest_long_cluster_pct
        if _long_dist > 0 and _long_dist < 10:
            if (self._long_cluster_first_seen == 0
                    or (self._long_cluster_last_dist > 0
                        and abs(_long_dist - self._long_cluster_last_dist) > self._long_cluster_last_dist * 0.5)):
                # New cluster or significantly shifted → reset
                self._long_cluster_first_seen = now
            self._long_cluster_last_dist = _long_dist
        else:
            self._long_cluster_first_seen = 0.0

        _short_dist = cascade.nearest_short_cluster_pct
        if _short_dist > 0 and _short_dist < 10:
            if (self._short_cluster_first_seen == 0
                    or (self._short_cluster_last_dist > 0
                        and abs(_short_dist - self._short_cluster_last_dist) > self._short_cluster_last_dist * 0.5)):
                self._short_cluster_first_seen = now
            self._short_cluster_last_dist = _short_dist
        else:
            self._short_cluster_first_seen = 0.0

        long_persist_mins = (now - self._long_cluster_first_seen) / 60.0 if self._long_cluster_first_seen else 0.0
        short_persist_mins = (now - self._short_cluster_first_seen) / 60.0 if self._short_cluster_first_seen else 0.0

        def _persistence_bonus(persist_mins: float) -> float:
            """Stable clusters get up to PERSISTENCE_MAX_BONUS."""
            if persist_mins < CP.PERSISTENCE_MIN_MINS:
                return 1.0
            t = min(persist_mins, CP.PERSISTENCE_BOOST_MINS) - CP.PERSISTENCE_MIN_MINS
            t_range = max(1, CP.PERSISTENCE_BOOST_MINS - CP.PERSISTENCE_MIN_MINS)
            return 1.0 + (CP.PERSISTENCE_MAX_BONUS - 1.0) * (t / t_range)

        # Long cascade pressure: cluster_size / distance × momentum × persistence
        long_pressure = 0.0
        if cascade.nearest_long_cluster_pct > 0 and cascade.estimated_long_cascade_usd >= CP.MIN_CLUSTER_USD:
            dist = max(cascade.nearest_long_cluster_pct, CP.DISTANCE_FLOOR_PCT)
            size_factor = cascade.estimated_long_cascade_usd / 1_000_000  # Normalize to millions
            momentum_factor = max(1.0, abs(price_change_pct)) if momentum_toward_long else 0.5
            long_pressure = (size_factor / dist) * momentum_factor * _persistence_bonus(long_persist_mins)

        # Short cascade pressure
        short_pressure = 0.0
        if cascade.nearest_short_cluster_pct > 0 and cascade.estimated_short_cascade_usd >= CP.MIN_CLUSTER_USD:
            dist = max(cascade.nearest_short_cluster_pct, CP.DISTANCE_FLOOR_PCT)
            size_factor = cascade.estimated_short_cascade_usd / 1_000_000
            momentum_factor = max(1.0, abs(price_change_pct)) if momentum_toward_short else 0.5
            short_pressure = (size_factor / dist) * momentum_factor * _persistence_bonus(short_persist_mins)

        def _pressure_level(score: float) -> str:
            if score >= CP.PRESSURE_CRITICAL:
                return "CRITICAL"
            if score >= CP.PRESSURE_HIGH:
                return "HIGH"
            if score >= CP.PRESSURE_MEDIUM:
                return "MEDIUM"
            return "LOW"

        self._snapshot.pressure = CascadePressureScore(
            long_pressure=round(long_pressure, 2),
            short_pressure=round(short_pressure, 2),
            long_pressure_level=_pressure_level(long_pressure),
            short_pressure_level=_pressure_level(short_pressure),
            momentum_toward_long=momentum_toward_long,
            momentum_toward_short=momentum_toward_short,
            long_persistence_mins=round(long_persist_mins, 1),
            short_persistence_mins=round(short_persist_mins, 1),
            timestamp=time.time(),
        )
        lp_level = _pressure_level(long_pressure)
        sp_level = _pressure_level(short_pressure)
        if lp_level in ("CRITICAL", "HIGH") or sp_level in ("CRITICAL", "HIGH"):
            logger.info(
                "Cascade pressure: long=%.2f(%s) short=%.2f(%s) persist_L=%.0fm persist_S=%.0fm",
                long_pressure, lp_level, short_pressure, sp_level,
                long_persist_mins, short_persist_mins,
            )

    def get_signal_intel(self, symbol: str, direction: str) -> Tuple[int, str]:
        """Returns (confidence_delta, note) for a given signal."""
        from config.constants import CascadePressure as CP

        delta = 0
        notes = []
        is_long = direction == "LONG"
        snap = self._snapshot

        # Effective leverage zone
        if not snap.leverage.is_stale:
            if snap.leverage.zone == "EXTREME":
                delta += -5
                notes.append(f"⚡ Extreme leverage ({snap.leverage.oi_to_mcap_pct:.0f}% OI/MCap)")
            elif snap.leverage.zone == "HIGH":
                delta += -3
                notes.append(f"⚡ High leverage ({snap.leverage.oi_to_mcap_pct:.0f}% OI/MCap)")
            elif snap.leverage.zone == "LOW" and snap.leverage.leverage_trend == "UNWINDING":
                delta += +3
                notes.append("⚡ Leverage unwinding (clean moves)")

        # Cascade risk (existing)
        if not snap.cascade.is_stale:
            if is_long and snap.cascade.cascade_risk_long in ("CRITICAL", "HIGH"):
                delta += -4
                notes.append(
                    f"⚡ Long cascade risk {snap.cascade.cascade_risk_long} "
                    f"({snap.cascade.nearest_long_cluster_pct:.1f}% below)"
                )
            elif not is_long and snap.cascade.cascade_risk_short in ("CRITICAL", "HIGH"):
                delta += -4
                notes.append(
                    f"⚡ Short cascade risk {snap.cascade.cascade_risk_short} "
                    f"({snap.cascade.nearest_short_cluster_pct:.1f}% above)"
                )

        # Cascade pressure scoring (momentum-confirmed)
        p = snap.pressure
        if p.timestamp > 0:
            if is_long and p.momentum_toward_long and p.long_pressure_level in ("CRITICAL", "HIGH"):
                # Price falling toward long liquidations with momentum → danger for longs
                pressure_delta = CP.PRESSURE_CRITICAL_DELTA if p.long_pressure_level == "CRITICAL" else CP.PRESSURE_HIGH_DELTA
                delta += -pressure_delta
                notes.append(
                    f"💥 Cascade pressure {p.long_pressure_level} "
                    f"(score={p.long_pressure:.1f}, momentum↓)"
                )
            elif not is_long and p.momentum_toward_short and p.short_pressure_level in ("CRITICAL", "HIGH"):
                # Price rising toward short liquidations → danger for shorts
                pressure_delta = CP.PRESSURE_CRITICAL_DELTA if p.short_pressure_level == "CRITICAL" else CP.PRESSURE_HIGH_DELTA
                delta += -pressure_delta
                notes.append(
                    f"💥 Cascade pressure {p.short_pressure_level} "
                    f"(score={p.short_pressure:.1f}, momentum↑)"
                )
            # Opportunity: if going WITH the cascade direction
            elif not is_long and p.momentum_toward_long and p.long_pressure_level in ("CRITICAL", "HIGH"):
                # SHORT trades benefit when price is cascading down through long liqs
                delta += CP.PRESSURE_HIGH_DELTA
                notes.append(
                    f"💥 Cascade opportunity SHORT "
                    f"(long liq pressure={p.long_pressure:.1f})"
                )
            elif is_long and p.momentum_toward_short and p.short_pressure_level in ("CRITICAL", "HIGH"):
                # LONG trades benefit when price is cascading up through short liqs
                delta += CP.PRESSURE_HIGH_DELTA
                notes.append(
                    f"💥 Cascade opportunity LONG "
                    f"(short liq pressure={p.short_pressure:.1f})"
                )

        delta = max(-10, min(10, delta))
        note = " | ".join(notes) if notes else ""
        if delta != 0:
            logger.debug(
                "LeverageMapper signal_intel: %s %s → delta=%+d %s",
                symbol, direction, delta, note,
            )
        return delta, note

    def get_leverage_zone(self) -> str:
        """Returns current leverage zone."""
        return self._snapshot.leverage.zone if not self._snapshot.leverage.is_stale else "UNKNOWN"

    def get_pressure(self) -> CascadePressureScore:
        """Returns current cascade pressure scoring."""
        return self._snapshot.pressure

    def get_price_history(self) -> list:
        """Returns recent price history (public accessor for cross-module use)."""
        return list(self._price_history)

    def get_snapshot(self) -> LeverageSnapshot:
        return self._snapshot


# Module-level singleton
leverage_mapper = LeverageMapper()
