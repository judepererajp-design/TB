"""
TitanBot Pro — Wallet Behavior Profiler
==========================================
Next-level whale tracking that goes beyond movement detection to
understand whale INTENT through behavioral pattern analysis.

This module enhances the existing whale_deposit_monitor with:
  1. Accumulation vs Distribution Phase Detection
  2. Wallet Holding Time Classification (STH vs LTH behavior)
  3. Behavioral Pattern Recognition (DCA, lump-sum, panic sell)
  4. Whale Cluster Analysis (coordinated movements)

Data sources:
  - Whale deposit events (from whale_deposit_monitor)
  - Smart money positions (from smart_money_client)
  - HyperTracker cohorts (from hypertracker_client)
  - Exchange order flow (from orderflow analyzer)

Signal impact:
  Accumulation phase detected:  +5 confidence LONGs
  Distribution phase detected:  -5 confidence LONGs
  Coordinated whale buy:        +4 LONGs
  Coordinated whale sell:       -4 LONGs
  DCA pattern (steady buying):  +3 LONGs (strong conviction)
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.constants import FlowAcceleration as FAC

logger = logging.getLogger(__name__)

# Phase detection window
_PHASE_WINDOW_HOURS = 24    # 24h window for phase detection
_PATTERN_MIN_EVENTS = 3     # Minimum events to detect a pattern
_COORDINATION_WINDOW = 600  # 10 min — whale events within this window = coordinated

# Confidence thresholds
_ACCUMULATION_THRESHOLD = 0.65  # 65%+ buy events = accumulation
_DISTRIBUTION_THRESHOLD = 0.65  # 65%+ sell events = distribution


@dataclass
class WhaleEvent:
    """Enriched whale event with behavioral context."""
    timestamp: float
    side: str           # "buy" or "sell"
    size_usd: float
    symbol: str
    source: str         # "orderbook" | "deposit" | "smart_money" | "hypertracker"


@dataclass
class WalletPhase:
    """Market-wide accumulation/distribution phase."""
    phase: str = "NEUTRAL"         # ACCUMULATION | DISTRIBUTION | NEUTRAL | TRANSITION
    confidence: float = 0.0        # How confident we are (0-1)
    buy_ratio: float = 0.5         # Ratio of buy events
    total_buy_usd: float = 0.0
    total_sell_usd: float = 0.0
    event_count: int = 0
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _PHASE_WINDOW_HOURS * 3600


@dataclass
class BehaviorPattern:
    """Detected behavioral pattern."""
    pattern: str = "NONE"          # DCA | LUMP_SUM | PANIC_SELL | STEADY_SELL | NONE
    frequency: str = "NONE"        # HIGH | MEDIUM | LOW | NONE
    avg_interval_min: float = 0.0  # Average time between events
    consistency: float = 0.0       # How regular the pattern is (0-1)
    timestamp: float = 0.0


@dataclass
class CoordinationSignal:
    """Whale coordination detection."""
    is_coordinated: bool = False
    direction: str = "NEUTRAL"     # BUY_CLUSTER | SELL_CLUSTER | NEUTRAL
    cluster_size: int = 0          # Number of whales acting together
    cluster_usd: float = 0.0      # Total USD in cluster
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _COORDINATION_WINDOW * 2


@dataclass
class WhaleDynamics:
    """Velocity, acceleration, and shock detection for whale flows."""
    velocity: float = 0.0           # Rate of change of net whale flow (USD/interval)
    acceleration: float = 0.0       # Rate of change of velocity (2nd derivative)
    is_accelerating: bool = False   # Buy flow accelerating
    is_decelerating: bool = False   # Sell flow accelerating
    shock_detected: bool = False    # Current flow > SHOCK_MULT × rolling avg
    shock_confirmed: bool = False   # Shock persists for SHOCK_CONFIRM_INTERVALS
    shock_direction: str = "NEUTRAL"  # "BUY_SHOCK" | "SELL_SHOCK" | "NEUTRAL"
    consecutive_shocks: int = 0
    timestamp: float = 0.0


@dataclass
class WalletBehaviorSnapshot:
    """Complete wallet behavior snapshot."""
    phase: WalletPhase = field(default_factory=WalletPhase)
    pattern: BehaviorPattern = field(default_factory=BehaviorPattern)
    coordination: CoordinationSignal = field(default_factory=CoordinationSignal)
    dynamics: WhaleDynamics = field(default_factory=WhaleDynamics)
    last_update: float = 0.0


class WalletBehaviorProfiler:
    """
    Analyzes whale behavior patterns to determine intent.

    Integration:
      - Engine calls get_signal_intel() for confidence adjustments
      - Ensemble voter uses get_whale_intent() for richer whale vote
      - Works alongside whale_deposit_monitor and smart_money_client
    """

    def __init__(self):
        self._snapshot = WalletBehaviorSnapshot()
        self._events: List[WhaleEvent] = []  # Rolling event buffer
        self._max_events = 500
        self._per_symbol_events: Dict[str, List[WhaleEvent]] = defaultdict(list)
        # Flow acceleration tracking: deque of (timestamp, net_flow_usd)
        self._flow_velocity_window: deque = deque(maxlen=FAC.VELOCITY_WINDOW)
        self._consecutive_shock_count: int = 0
        # Historical intent tracking for behavior profiling
        self._intent_history: List[Tuple[float, str]] = []  # (timestamp, bias)

    def record_event(self, side: str, size_usd: float, symbol: str, source: str = "orderbook"):
        """Record a whale event from any source."""
        try:
            event = WhaleEvent(
                timestamp=time.time(),
                side=side,
                size_usd=size_usd,
                symbol=symbol,
                source=source,
            )
            self._events.append(event)
            self._per_symbol_events[symbol].append(event)

            # Trim to max size
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
            for sym in self._per_symbol_events:
                if len(self._per_symbol_events[sym]) > 100:
                    self._per_symbol_events[sym] = self._per_symbol_events[sym][-100:]

            # Re-analyze after each event
            self._analyze_phase()
            self._analyze_patterns()
            self._detect_coordination()
            self._compute_whale_dynamics()
            self._snapshot.last_update = time.time()
        except Exception as e:
            logger.error("WalletBehavior.record_event() failed: %s", e, exc_info=True)

    def _analyze_phase(self):
        """Detect market-wide accumulation/distribution phase."""
        cutoff = time.time() - _PHASE_WINDOW_HOURS * 3600
        recent = [e for e in self._events if e.timestamp > cutoff]

        if len(recent) < _PATTERN_MIN_EVENTS:
            self._snapshot.phase = WalletPhase(timestamp=time.time())
            return

        buy_count = sum(1 for e in recent if e.side == "buy")
        sell_count = sum(1 for e in recent if e.side == "sell")
        total = buy_count + sell_count
        buy_ratio = buy_count / total if total > 0 else 0.5

        buy_usd = sum(e.size_usd for e in recent if e.side == "buy")
        sell_usd = sum(e.size_usd for e in recent if e.side == "sell")

        # Volume-weighted ratio (more important than count)
        total_usd = buy_usd + sell_usd
        vol_buy_ratio = buy_usd / total_usd if total_usd > 0 else 0.5

        # Combine count and volume ratios
        combined_ratio = (buy_ratio + vol_buy_ratio) / 2

        if combined_ratio >= _ACCUMULATION_THRESHOLD:
            phase = "ACCUMULATION"
            confidence = min(1.0, (combined_ratio - 0.5) * 4)
        elif combined_ratio <= (1 - _DISTRIBUTION_THRESHOLD):
            phase = "DISTRIBUTION"
            confidence = min(1.0, (0.5 - combined_ratio) * 4)
        else:
            phase = "NEUTRAL"
            confidence = 0.3

        self._snapshot.phase = WalletPhase(
            phase=phase,
            confidence=confidence,
            buy_ratio=combined_ratio,
            total_buy_usd=buy_usd,
            total_sell_usd=sell_usd,
            event_count=total,
            timestamp=time.time(),
        )
        if phase != "NEUTRAL":
            logger.debug(
                "WhaleBehavior phase=%s conf=%.2f buy_ratio=%.2f events=%d buy=$%.0fM sell=$%.0fM",
                phase, confidence, combined_ratio, total,
                buy_usd / 1_000_000, sell_usd / 1_000_000,
            )

    def _analyze_patterns(self):
        """Detect behavioral patterns (DCA, lump sum, panic, etc.)."""
        cutoff = time.time() - _PHASE_WINDOW_HOURS * 3600
        recent = [e for e in self._events if e.timestamp > cutoff]

        if len(recent) < _PATTERN_MIN_EVENTS:
            self._snapshot.pattern = BehaviorPattern(timestamp=time.time())
            return

        # Calculate intervals between events
        intervals = []
        for i in range(1, len(recent)):
            intervals.append(recent[i].timestamp - recent[i - 1].timestamp)

        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        avg_interval_min = avg_interval / 60

        # Calculate interval consistency (coefficient of variation)
        if intervals and avg_interval > 0:
            std = (sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)) ** 0.5
            cv = std / avg_interval  # Lower CV = more regular
            consistency = max(0, 1 - cv)
        else:
            consistency = 0

        # Detect pattern
        buy_sizes = [e.size_usd for e in recent if e.side == "buy"]
        sell_sizes = [e.size_usd for e in recent if e.side == "sell"]

        if buy_sizes and consistency > 0.5 and len(buy_sizes) >= 3:
            # Regular buying with consistent sizes → DCA
            size_cv = 0
            if buy_sizes:
                mean_size = sum(buy_sizes) / len(buy_sizes)
                if mean_size > 0:
                    size_std = (sum((s - mean_size) ** 2 for s in buy_sizes) / len(buy_sizes)) ** 0.5
                    size_cv = size_std / mean_size
            if size_cv < 0.5:
                pattern = "DCA"
            else:
                pattern = "LUMP_SUM"
        elif sell_sizes and len(sell_sizes) >= 3:
            if avg_interval_min < 5 and sum(sell_sizes) > 1_000_000:
                pattern = "PANIC_SELL"
            elif consistency > 0.3:
                pattern = "STEADY_SELL"
            else:
                pattern = "LUMP_SUM"
        else:
            pattern = "NONE"

        # Frequency classification
        if avg_interval_min < 10:
            frequency = "HIGH"
        elif avg_interval_min < 60:
            frequency = "MEDIUM"
        elif avg_interval_min < 360:
            frequency = "LOW"
        else:
            frequency = "NONE"

        self._snapshot.pattern = BehaviorPattern(
            pattern=pattern,
            frequency=frequency,
            avg_interval_min=avg_interval_min,
            consistency=consistency,
            timestamp=time.time(),
        )

    def _detect_coordination(self):
        """Detect coordinated whale activity (multiple large events in short window)."""
        cutoff = time.time() - _COORDINATION_WINDOW
        recent = [e for e in self._events if e.timestamp > cutoff]

        if len(recent) < 2:
            self._snapshot.coordination = CoordinationSignal(timestamp=time.time())
            return

        # Check if multiple large events cluster together
        buy_events = [e for e in recent if e.side == "buy"]
        sell_events = [e for e in recent if e.side == "sell"]

        buy_usd = sum(e.size_usd for e in buy_events)
        sell_usd = sum(e.size_usd for e in sell_events)

        is_coordinated = False
        direction = "NEUTRAL"
        cluster_size = 0
        cluster_usd = 0.0

        # Coordinated buy: 3+ buy events from different sources
        sources_buy = set(e.source for e in buy_events)
        sources_sell = set(e.source for e in sell_events)

        if len(buy_events) >= 3 or (len(buy_events) >= 2 and len(sources_buy) >= 2):
            is_coordinated = True
            direction = "BUY_CLUSTER"
            cluster_size = len(buy_events)
            cluster_usd = buy_usd

        if len(sell_events) >= 3 or (len(sell_events) >= 2 and len(sources_sell) >= 2):
            # If both sides coordinated, go with the larger
            if not is_coordinated or sell_usd > buy_usd:
                is_coordinated = True
                direction = "SELL_CLUSTER"
                cluster_size = len(sell_events)
                cluster_usd = sell_usd

        self._snapshot.coordination = CoordinationSignal(
            is_coordinated=is_coordinated,
            direction=direction,
            cluster_size=cluster_size,
            cluster_usd=cluster_usd,
            timestamp=time.time(),
        )

    # ── Whale Dynamics (velocity / acceleration / shock) ─────

    def _compute_whale_dynamics(self):
        """
        Compute velocity and acceleration of whale flows.
        Detects shock events when flow surges beyond normal range.
        """
        # Calculate net flow in the last hour
        cutoff = time.time() - 3600  # 1-hour window for flow measurement
        recent = [e for e in self._events if e.timestamp > cutoff]

        if len(recent) < FAC.WHALE_ACCEL_MIN_EVENTS:
            return

        buy_usd = sum(e.size_usd for e in recent if e.side == "buy")
        sell_usd = sum(e.size_usd for e in recent if e.side == "sell")
        net_flow = buy_usd - sell_usd  # Positive = net buying

        now = time.time()
        self._flow_velocity_window.append((now, net_flow))

        window = list(self._flow_velocity_window)
        if len(window) < 2:
            return

        # Velocity: current net flow
        velocity = window[-1][1]

        # Acceleration: change in net flow
        prev_velocity = window[-2][1]
        acceleration = velocity - prev_velocity

        # Rolling average of absolute flow for shock detection
        abs_flows = [abs(v) for _, v in window]
        rolling_avg = sum(abs_flows) / len(abs_flows) if abs_flows else 1.0

        # Shock detection
        current_abs = abs(velocity)
        shock_detected = (
            current_abs > FAC.SHOCK_MULT * max(rolling_avg, 1.0) and
            current_abs > FAC.WHALE_SHOCK_USD_THRESHOLD
        )

        if shock_detected:
            self._consecutive_shock_count += 1
        else:
            self._consecutive_shock_count = 0

        shock_confirmed = self._consecutive_shock_count >= FAC.SHOCK_CONFIRM_INTERVALS

        # Shock direction
        shock_direction = "NEUTRAL"
        if shock_confirmed:
            shock_direction = "BUY_SHOCK" if velocity > 0 else "SELL_SHOCK"

        # Acceleration state (threshold from constants)
        is_accelerating = acceleration > FAC.WHALE_ACCEL_USD_THRESHOLD and velocity > 0
        is_decelerating = acceleration < -FAC.WHALE_ACCEL_USD_THRESHOLD and velocity < 0

        self._snapshot.dynamics = WhaleDynamics(
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
                f"⚡ Whale SHOCK {shock_direction}: net_flow=${velocity/1e6:+.1f}M "
                f"accel=${acceleration/1e6:+.1f}M ({self._consecutive_shock_count} consecutive)"
            )

    # ── Signal Intelligence ───────────────────────────────────

    def get_signal_intel(self, symbol: str, direction: str) -> Tuple[int, str]:
        """Returns (confidence_delta, note) for a given signal.
        Now includes whale flow acceleration and shock detection."""
        delta = 0
        notes = []
        is_long = direction == "LONG"
        snap = self._snapshot

        # Phase impact (existing)
        if not snap.phase.is_stale and snap.phase.confidence > 0.4:
            if snap.phase.phase == "ACCUMULATION":
                adj = int(5 * snap.phase.confidence)
                delta += +adj if is_long else -int(adj * 0.6)
                notes.append(f"🐋 Whale accumulation phase ({snap.phase.confidence:.0%})")
            elif snap.phase.phase == "DISTRIBUTION":
                adj = int(5 * snap.phase.confidence)
                delta += -adj if is_long else +int(adj * 0.6)
                notes.append(f"🐋 Whale distribution phase ({snap.phase.confidence:.0%})")

        # Pattern impact (existing)
        if snap.pattern.pattern == "DCA":
            delta += +3 if is_long else -1
            notes.append("🐋 DCA pattern (strong conviction buying)")
        elif snap.pattern.pattern == "PANIC_SELL":
            delta += -4 if is_long else +3
            notes.append("🐋 Panic selling detected")

        # Coordination impact (existing)
        if not snap.coordination.is_stale and snap.coordination.is_coordinated:
            if snap.coordination.direction == "BUY_CLUSTER":
                delta += +4 if is_long else -2
                notes.append(
                    f"🐋 Coordinated whale buy "
                    f"({snap.coordination.cluster_size} whales, "
                    f"${snap.coordination.cluster_usd/1e6:.1f}M)"
                )
            elif snap.coordination.direction == "SELL_CLUSTER":
                delta += -4 if is_long else +2
                notes.append(
                    f"🐋 Coordinated whale sell "
                    f"({snap.coordination.cluster_size} whales, "
                    f"${snap.coordination.cluster_usd/1e6:.1f}M)"
                )

        # Whale flow dynamics (NEW — Tier 1 upgrade)
        dyn = snap.dynamics
        if dyn.timestamp > 0:
            if dyn.shock_confirmed:
                if dyn.shock_direction == "BUY_SHOCK":
                    delta += +FAC.SHOCK_BULLISH_DELTA if is_long else -3
                    notes.append(
                        f"⚡ WHALE BUY SHOCK: net=${dyn.velocity/1e6:+.1f}M "
                        f"(confirmed {dyn.consecutive_shocks}×)"
                    )
                elif dyn.shock_direction == "SELL_SHOCK":
                    delta += -FAC.SHOCK_BEARISH_DELTA if is_long else +3
                    notes.append(
                        f"⚡ WHALE SELL SHOCK: net=${dyn.velocity/1e6:+.1f}M "
                        f"(confirmed {dyn.consecutive_shocks}×)"
                    )
            elif dyn.is_accelerating:
                delta += +FAC.ACCEL_BULLISH_DELTA if is_long else -2
                notes.append(
                    f"📈 Whale buying accelerating: "
                    f"net=${dyn.velocity/1e6:+.1f}M"
                )
            elif dyn.is_decelerating:
                delta += -FAC.ACCEL_BEARISH_DELTA if is_long else +2
                notes.append(
                    f"📉 Whale selling accelerating: "
                    f"net=${dyn.velocity/1e6:+.1f}M"
                )

        delta = max(-15, min(15, delta))
        note = " | ".join(notes) if notes else ""
        return delta, note

    def get_whale_intent(self) -> str:
        """Returns whale intent classification for ensemble voter."""
        snap = self._snapshot
        if snap.phase.is_stale:
            return "UNKNOWN"
        if snap.phase.phase == "ACCUMULATION" and snap.phase.confidence > 0.5:
            return "BULLISH"
        if snap.phase.phase == "DISTRIBUTION" and snap.phase.confidence > 0.5:
            return "BEARISH"
        return "NEUTRAL"

    def get_advanced_intent(self, direction: str = "") -> 'AdvancedWhaleIntent':
        """
        Advanced whale intent classification — goes beyond simple buy/sell
        to determine the underlying purpose of whale activity.

        Intent taxonomy:
          ACCUMULATION  — Steady buying, holding for directional move
          DISTRIBUTION  — Steady selling, offloading position
          REBALANCING   — Both sides active, no directional bias
          MARKET_MAKING — Tight symmetric flow, providing liquidity
          DIRECTIONAL   — Strong one-sided flow with conviction
          HEDGING       — Counter-position activity (often paired with derivatives)
        """
        from config.constants import WhaleIntent as WI
        import math

        snap = self._snapshot
        result = AdvancedWhaleIntent()

        if snap.phase.is_stale or snap.phase.event_count < _PATTERN_MIN_EVENTS:
            result.intent = "UNKNOWN"
            result.confidence = 0.0
            return result

        # ── Compute multi-source weighted scores ─────────────
        # Orderbook events
        ob_events = [e for e in self._events if e.source == "orderbook"]
        ob_buy = sum(e.size_usd for e in ob_events if e.side == "buy")
        ob_sell = sum(e.size_usd for e in ob_events if e.side == "sell")

        # On-chain events
        onchain_events = [e for e in self._events if e.source in ("deposit", "smart_money", "hypertracker")]
        onchain_buy = sum(e.size_usd for e in onchain_events if e.side == "buy")
        onchain_sell = sum(e.size_usd for e in onchain_events if e.side == "sell")

        # Total weighted
        total_buy = (
            ob_buy * WI.SOURCE_WEIGHT_ORDERBOOK +
            onchain_buy * WI.SOURCE_WEIGHT_ONCHAIN
        )
        total_sell = (
            ob_sell * WI.SOURCE_WEIGHT_ORDERBOOK +
            onchain_sell * WI.SOURCE_WEIGHT_ONCHAIN
        )
        total = total_buy + total_sell

        if total <= 0:
            result.intent = "NEUTRAL"
            result.confidence = 0.2
            return result

        buy_ratio = total_buy / total

        # ── Market-maker detection ───────────────────────────
        if total_buy > 0 and total_sell > 0:
            size_symmetry = min(total_buy, total_sell) / max(total_buy, total_sell)
            if size_symmetry >= WI.MM_SIZE_SYMMETRY_MIN:
                result.intent = WI.INTENT_MARKET_MAKING
                result.confidence = size_symmetry
                result.is_market_maker = True
                result.notes.append(
                    f"🏦 Market-maker detected: symmetry={size_symmetry:.0%}"
                )
                result.confidence_delta = WI.INTENT_CONF_MM
                return result

        # ── Directional intent ───────────────────────────────
        if buy_ratio >= 0.75:
            result.intent = WI.INTENT_DIRECTIONAL
            result.directional_bias = "BULLISH"
            result.confidence = min(1.0, (buy_ratio - 0.5) * 4)

            # Check if this is accumulation (persistent) or directional (sudden)
            if snap.pattern.pattern == "DCA" and snap.pattern.consistency > 0.4:
                result.intent = WI.INTENT_ACCUMULATION
                result.notes.append(
                    f"🐋 Smart accumulation: DCA pattern, "
                    f"consistency={snap.pattern.consistency:.0%}"
                )
                result.confidence_delta = WI.INTENT_CONF_ACCUMULATION
            else:
                result.notes.append(
                    f"🐋 Directional whale buying: "
                    f"buy ratio={buy_ratio:.0%}"
                )
                if direction == "LONG":
                    result.confidence_delta = WI.INTENT_CONF_DIRECTIONAL_WITH
                elif direction == "SHORT":
                    result.confidence_delta = WI.INTENT_CONF_DIRECTIONAL_AGAINST

        elif buy_ratio <= 0.25:
            result.intent = WI.INTENT_DIRECTIONAL
            result.directional_bias = "BEARISH"
            result.confidence = min(1.0, (0.5 - buy_ratio) * 4)

            if snap.pattern.pattern in ("PANIC_SELL", "STEADY_SELL"):
                result.intent = WI.INTENT_DISTRIBUTION
                result.notes.append(
                    f"🐋 Distribution phase: {snap.pattern.pattern}"
                )
                result.confidence_delta = WI.INTENT_CONF_DISTRIBUTION
            else:
                result.notes.append(
                    f"🐋 Directional whale selling: "
                    f"sell ratio={1 - buy_ratio:.0%}"
                )
                if direction == "SHORT":
                    result.confidence_delta = WI.INTENT_CONF_DIRECTIONAL_WITH
                elif direction == "LONG":
                    result.confidence_delta = WI.INTENT_CONF_DIRECTIONAL_AGAINST

        else:
            # Mixed flow
            result.intent = WI.INTENT_REBALANCING
            result.confidence = 0.4
            result.notes.append(
                f"🐋 Whale rebalancing: buy={buy_ratio:.0%} sell={1 - buy_ratio:.0%}"
            )

        # ── Coordination amplifier ───────────────────────────
        if not snap.coordination.is_stale and snap.coordination.is_coordinated:
            result.is_coordinated = True
            result.confidence = min(1.0, result.confidence + 0.15)
            result.notes.append(
                f"⚡ Coordinated whale activity: {snap.coordination.cluster_size} participants"
            )

        # ── Shock amplifier ──────────────────────────────────
        if snap.dynamics.shock_confirmed:
            result.shock_active = True
            result.confidence = min(1.0, result.confidence + 0.20)
            result.notes.append(
                f"⚡ WHALE SHOCK: {snap.dynamics.shock_direction}"
            )

        # ── Apply confidence decay ───────────────────────────
        age_secs = time.time() - snap.phase.timestamp
        decay = math.exp(-age_secs / WI.INTENT_DECAY_TAU_SECS)
        result.confidence = max(WI.INTENT_MIN_CONFIDENCE, result.confidence * decay)

        # ── Historical behavior profiling ────────────────────
        # Track cumulative directional consistency of this wallet cluster
        history_score = self._get_historical_pattern_score(result.directional_bias)
        if abs(history_score) > 0:
            result.confidence = min(1.0, result.confidence + abs(history_score))
            if history_score > 0.05:
                result.notes.append(
                    f"📈 Historical pattern: consistent {result.directional_bias.lower()} "
                    f"activity (+{history_score:.0%} trust)"
                )

        # Clamp delta
        result.confidence_delta = max(-10, min(10, result.confidence_delta))

        return result

    def _get_historical_pattern_score(self, current_bias: str) -> float:
        """Score historical directional consistency of whale behavior.

        Tracks past intent classifications and rewards consistent patterns.
        Returns 0.0-WI.HISTORICAL_CONSISTENCY_BONUS based on how consistently
        this wallet cluster has shown the same directional bias historically.
        """
        from config.constants import WhaleIntent as WI

        if current_bias not in ("BULLISH", "BEARISH"):
            return 0.0

        now = time.time()
        # Record current bias
        self._intent_history.append((now, current_bias))

        # Prune old entries (beyond decay window)
        cutoff = now - WI.HISTORICAL_DECAY_DAYS * 86400
        self._intent_history = [
            (ts, bias) for ts, bias in self._intent_history if ts > cutoff
        ]

        if len(self._intent_history) < WI.HISTORICAL_MIN_EVENTS:
            return 0.0

        # Count how many past entries match current bias (with time decay)
        matching = 0
        total = 0
        for ts, bias in self._intent_history:
            age_days = (now - ts) / 86400
            weight = math.exp(-age_days / max(WI.HISTORICAL_DECAY_DAYS, 1.0))
            total += weight
            if bias == current_bias:
                matching += weight

        if total <= 0:
            return 0.0

        consistency = matching / total
        # Only reward strong consistency (>60%)
        if consistency < 0.6:
            return 0.0

        return min(WI.HISTORICAL_CONSISTENCY_BONUS, (consistency - 0.6) * 0.5)

    def get_snapshot(self) -> WalletBehaviorSnapshot:
        return self._snapshot

    def get_whale_dynamics(self) -> WhaleDynamics:
        """Return current whale flow dynamics for diagnostics."""
        return self._snapshot.dynamics


@dataclass
class AdvancedWhaleIntent:
    """Result of advanced whale intent classification."""
    intent: str = "UNKNOWN"             # One of WhaleIntent.INTENT_* constants
    directional_bias: str = "NEUTRAL"   # BULLISH | BEARISH | NEUTRAL
    confidence: float = 0.0             # 0-1 confidence in classification
    confidence_delta: int = 0           # Signal confidence adjustment
    is_market_maker: bool = False       # True if MM activity detected
    is_coordinated: bool = False        # True if coordinated whale action
    shock_active: bool = False          # True if flow shock confirmed
    notes: List[str] = field(default_factory=list)


# Module-level singleton
wallet_profiler = WalletBehaviorProfiler()
