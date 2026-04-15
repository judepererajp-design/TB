"""
TitanBot Pro — Order Flow Analyzer
=====================================
Real-time order book analysis for signal confirmation.

Analyzes:
  - Bid/Ask imbalance (who has more firepower)
  - Liquidity walls (large orders as support/resistance)
  - Order book delta (net buying vs selling pressure)
  - Anti-spoofing (detect and filter fake walls)
  - Price impact estimation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.loader import cfg
from data.api_client import api
from utils.formatting import fmt_price

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowData:
    symbol: str
    bid_volume: float = 0.0         # Total bid volume in book
    ask_volume: float = 0.0
    delta: float = 0.0              # bid_volume - ask_volume
    delta_pct: float = 0.0          # delta as % of total
    bid_walls: List[float] = None   # Prices with large bid orders
    ask_walls: List[float] = None   # Prices with large ask orders
    nearest_bid_wall: Optional[float] = None
    nearest_ask_wall: Optional[float] = None
    imbalance: str = "BALANCED"     # BID_HEAVY | ASK_HEAVY | BALANCED
    score: float = 50.0
    notes: List[str] = None

    def __post_init__(self):
        if self.bid_walls is None:
            self.bid_walls = []
        if self.ask_walls is None:
            self.ask_walls = []
        if self.notes is None:
            self.notes = []


class OrderFlowAnalyzer:
    """
    Analyzes order book for signal confirmation.
    Lighter implementation optimized for MacBook (fewer API calls).
    """

    def __init__(self):
        self._cfg = cfg.analyzers.orderflow
        self._wall_mult  = getattr(self._cfg, 'wall_threshold_mult', 4.0)
        self._anti_spoof = getattr(self._cfg, 'anti_spoofing', True)
        self._depth      = getattr(self._cfg, 'depth_limit', 50)
        self._spoof_detector = SpoofDetector()

    async def analyze(self, symbol: str, current_price: float) -> OrderFlowData:
        """Full order flow analysis for a symbol"""
        data = OrderFlowData(symbol=symbol)

        order_book = await api.fetch_order_book(symbol, limit=self._depth)
        if not order_book:
            data.score = 50.0
            return data

        bids = order_book.get('bids', [])  # [[price, size], ...]
        asks = order_book.get('asks', [])

        if not bids or not asks:
            return data

        # ── Volume calculations ────────────────────────────────
        data.bid_volume = sum(float(q) for _, q in bids)
        data.ask_volume = sum(float(q) for _, q in asks)
        total = data.bid_volume + data.ask_volume

        data.delta = data.bid_volume - data.ask_volume
        data.delta_pct = (data.delta / total * 100) if total > 0 else 0

        # ── Imbalance classification ──────────────────────────
        if data.delta_pct > 20:
            data.imbalance = "BID_HEAVY"
            data.notes.append(f"✅ Bid-heavy book ({data.delta_pct:.0f}% imbalance)")
        elif data.delta_pct < -20:
            data.imbalance = "ASK_HEAVY"
            data.notes.append(f"⚠️ Ask-heavy book ({abs(data.delta_pct):.0f}% imbalance)")
        else:
            data.imbalance = "BALANCED"

        # ── Wall detection ────────────────────────────────────
        all_sizes = ([float(q) for _, q in bids[:20]] +
                     [float(q) for _, q in asks[:20]])
        if not all_sizes:
            return data

        avg_size = sum(all_sizes) / len(all_sizes)
        wall_threshold = avg_size * self._wall_mult

        bid_walls = [
            float(p) for p, q in bids[:20]
            if float(q) >= wall_threshold
        ]
        ask_walls = [
            float(p) for p, q in asks[:20]
            if float(q) >= wall_threshold
        ]

        # Anti-spoofing: walls that disappear between snapshots are fake
        # Simple version: walls further from price are less trustworthy
        if self._anti_spoof:
            bid_walls = [p for p in bid_walls
                         if p >= current_price * 0.97]  # Only trust walls within 3%
            ask_walls = [p for p in ask_walls
                         if p <= current_price * 1.03]

        # ── Spoofing detection (v2): wall persistence & fake wall scoring ──
        wall_analysis = self._spoof_detector.analyze_walls(
            bid_walls=[(p, avg_size * self._wall_mult) for p in bid_walls],
            ask_walls=[(p, avg_size * self._wall_mult) for p in ask_walls],
            current_price=current_price,
            avg_size=avg_size,
        )
        # Filter out detected fake walls
        if wall_analysis.fake_bid_walls:
            bid_walls = [p for p in bid_walls if p not in wall_analysis.fake_bid_walls]
            for fp in wall_analysis.fake_bid_walls:
                data.notes.append(
                    f"🎭 Fake bid wall detected at {fmt_price(fp)} — likely spoofing"
                )
        if wall_analysis.fake_ask_walls:
            ask_walls = [p for p in ask_walls if p not in wall_analysis.fake_ask_walls]
            for fp in wall_analysis.fake_ask_walls:
                data.notes.append(
                    f"🎭 Fake ask wall detected at {fmt_price(fp)} — likely spoofing"
                )

        data.bid_walls = bid_walls
        data.ask_walls = ask_walls

        if bid_walls:
            data.nearest_bid_wall = max(bid_walls)  # Closest bid wall (highest price)
            tier = wall_analysis.get_wall_tier(data.nearest_bid_wall, avg_size)
            data.notes.append(f"🛡 Bid wall at {fmt_price(data.nearest_bid_wall)} — {tier} support")

        if ask_walls:
            data.nearest_ask_wall = min(ask_walls)  # Closest ask wall (lowest price)
            tier = wall_analysis.get_wall_tier(data.nearest_ask_wall, avg_size)
            data.notes.append(f"🚧 Ask wall at {fmt_price(data.nearest_ask_wall)} — {tier} resistance")

        # ── Score calculation ─────────────────────────────────
        score = 50.0

        # Imbalance score
        score += data.delta_pct * 0.3  # ±30 max from imbalance

        # Bid wall = support → bullish
        if data.nearest_bid_wall:
            proximity = (current_price - data.nearest_bid_wall) / current_price
            if proximity < 0.02:  # Within 2%
                score += 10

        # Ask wall = resistance → bearish if no bid walls
        if data.nearest_ask_wall and not data.nearest_bid_wall:
            score -= 8

        # Spoofing discount: reduce score if many fake walls detected
        spoof_discount = wall_analysis.overall_spoof_score * 5.0  # Max ~5 pts
        score -= spoof_discount

        data.score = max(0.0, min(100.0, score))
        return data

    def get_direction_adjustment(
        self, data: OrderFlowData, direction: str
    ) -> Tuple[float, str]:
        """
        Get confidence adjustment for signal based on order flow.
        Returns (adjustment, note)
        """
        if direction == "LONG":
            if data.imbalance == "BID_HEAVY":
                return 8.0, "✅ Order flow supports long (bid-heavy)"
            elif data.imbalance == "ASK_HEAVY":
                return -6.0, "⚠️ Order flow opposes long (ask-heavy)"
            if data.nearest_bid_wall:
                return 5.0, f"✅ Bid wall support at {fmt_price(data.nearest_bid_wall)}"
            if data.nearest_ask_wall:
                return -3.0, f"🚧 Ask wall resistance at {fmt_price(data.nearest_ask_wall)}"

        elif direction == "SHORT":
            if data.imbalance == "ASK_HEAVY":
                return 8.0, "✅ Order flow supports short (ask-heavy)"
            elif data.imbalance == "BID_HEAVY":
                return -6.0, "⚠️ Order flow opposes short (bid-heavy)"
            if data.nearest_ask_wall:
                return 5.0, f"✅ Ask wall resistance at {fmt_price(data.nearest_ask_wall)}"
            if data.nearest_bid_wall:
                return -3.0, f"🛡 Bid wall below at {fmt_price(data.nearest_bid_wall)}"

        return 0.0, ""

    def get_spoof_detector(self) -> 'SpoofDetector':
        """Expose spoof detector for external queries."""
        return self._spoof_detector


# ════════════════════════════════════════════════════════════════
# Spoofing / Fake Wall Detector
# ════════════════════════════════════════════════════════════════

@dataclass
class WallSnapshot:
    """A single point-in-time record of a wall."""
    price: float
    size: float
    side: str            # "bid" or "ask"
    timestamp: float


@dataclass
class WallAnalysis:
    """Result of spoofing analysis on order book walls."""
    fake_bid_walls: List[float] = field(default_factory=list)
    fake_ask_walls: List[float] = field(default_factory=list)
    persistent_bid_walls: List[float] = field(default_factory=list)
    persistent_ask_walls: List[float] = field(default_factory=list)
    overall_spoof_score: float = 0.0   # 0-1, higher = more spoofing detected
    absorption_detected: bool = False  # True if price hits wall repeatedly without breaking
    absorption_side: str = ""          # "bid" or "ask" — which wall absorbed
    absorption_touches: int = 0        # Number of price touches at the absorption level
    notes: List[str] = field(default_factory=list)

    def get_wall_tier(self, price: float, avg_size: float) -> str:
        """Classify wall tier based on size relative to average."""
        from config.constants import SpoofingDetection as SD
        # We don't have the actual size here, so return based on persistence
        if price in self.persistent_bid_walls or price in self.persistent_ask_walls:
            return "strong persistent"
        return "moderate"


class SpoofDetector:
    """
    Detects fake/spoofed order book walls by tracking wall persistence
    across multiple snapshots.

    Key detection methods:
      1. Persistence: real walls appear in multiple consecutive snapshots
      2. Pull-before-price: walls removed when price approaches them
      3. Size anomaly: walls absurdly larger than surrounding levels
      4. Symmetric detection: genuine market-maker walls are symmetric
    """

    def __init__(self):
        self._snapshots: List[WallSnapshot] = []
        self._max_snapshots_age: float = 300.0  # 5 min of history
        self._wall_history: Dict[str, List[float]] = {}  # price_key → [timestamps]
        self._price_touch_history: List[Tuple[float, float]] = []  # (timestamp, price)

    def record_snapshot(
        self,
        bid_walls: List[Tuple[float, float]],
        ask_walls: List[Tuple[float, float]],
        current_price: float,
    ):
        """Record current wall positions for persistence tracking."""
        now = time.time()

        # Record price touch for absorption detection
        self._price_touch_history.append((now, current_price))

        for price, size in bid_walls:
            key = f"bid_{price:.6g}"
            if key not in self._wall_history:
                self._wall_history[key] = []
            self._wall_history[key].append(now)

        for price, size in ask_walls:
            key = f"ask_{price:.6g}"
            if key not in self._wall_history:
                self._wall_history[key] = []
            self._wall_history[key].append(now)

        # Prune old history
        cutoff = now - self._max_snapshots_age
        self._wall_history = {
            k: [t for t in ts if t > cutoff]
            for k, ts in self._wall_history.items()
            if any(t > cutoff for t in ts)
        }
        self._price_touch_history = [
            (t, p) for t, p in self._price_touch_history if t > cutoff
        ]

    def analyze_walls(
        self,
        bid_walls: List[Tuple[float, float]],
        ask_walls: List[Tuple[float, float]],
        current_price: float,
        avg_size: float,
    ) -> WallAnalysis:
        """
        Analyze walls for spoofing patterns.

        Parameters
        ----------
        bid_walls : list of (price, size)
        ask_walls : list of (price, size)
        current_price : float
        avg_size : float  — average order size for context
        """
        from config.constants import SpoofingDetection as SD

        result = WallAnalysis()

        # Record this snapshot for persistence tracking
        self.record_snapshot(bid_walls, ask_walls, current_price)

        all_walls = (
            [(p, s, "bid") for p, s in bid_walls] +
            [(p, s, "ask") for p, s in ask_walls]
        )

        fake_count = 0
        total_count = max(len(all_walls), 1)

        for price, size, side in all_walls:
            fake_score = 0.0

            # 1. Persistence check
            key = f"{side}_{price:.6g}"
            appearances = len(self._wall_history.get(key, []))
            if appearances < SD.MIN_PERSISTENCE_SNAPSHOTS:
                fake_score += 0.3  # New wall = somewhat suspicious

            # 2. Size anomaly
            if avg_size > 0 and size / avg_size >= SD.SIZE_ANOMALY_MULT:
                fake_score += 0.3  # Abnormally large = suspicious

            # 3. Proximity to price (spoofed walls tend to be placed near price)
            distance_pct = abs(price - current_price) / current_price
            if distance_pct < SD.PULL_DISTANCE_PCT:
                # Very close to price — could be about to be pulled
                if appearances < SD.MIN_PERSISTENCE_SNAPSHOTS:
                    fake_score += 0.2

            # 4. Classify
            if fake_score >= SD.FAKE_WALL_CONFIDENCE:
                if side == "bid":
                    result.fake_bid_walls.append(price)
                else:
                    result.fake_ask_walls.append(price)
                fake_count += 1
            elif fake_score <= SD.REAL_WALL_CONFIDENCE:
                if side == "bid":
                    result.persistent_bid_walls.append(price)
                else:
                    result.persistent_ask_walls.append(price)

        # 5. Check for symmetric walls (market-maker vs spoof)
        for bp, bs, _ in [(p, s, sd) for p, s, sd in all_walls if sd == "bid"]:
            for ap, as_, _ in [(p, s, sd) for p, s, sd in all_walls if sd == "ask"]:
                bid_dist = abs(current_price - bp) / current_price
                ask_dist = abs(ap - current_price) / current_price
                if bid_dist > 0 and ask_dist > 0:
                    dist_ratio = min(bid_dist, ask_dist) / max(bid_dist, ask_dist)
                    if dist_ratio > (1.0 - SD.SYMMETRIC_WALL_TOLERANCE):
                        # Symmetric — likely market maker, not spoof
                        # Remove from fake walls if present
                        if bp in result.fake_bid_walls:
                            result.fake_bid_walls.remove(bp)
                            fake_count = max(0, fake_count - 1)
                        if ap in result.fake_ask_walls:
                            result.fake_ask_walls.remove(ap)
                            fake_count = max(0, fake_count - 1)
                        result.notes.append(
                            f"🏦 Symmetric walls detected — likely market maker"
                        )

        result.overall_spoof_score = fake_count / total_count

        # ── 6. Absorption detection ─────────────────────────────
        # Price touches a persistent wall repeatedly without breaking through
        # → strong evidence of real liquidity intent (institutions defending a level)
        result = self._detect_absorption(result, all_walls, current_price)

        return result

    def _detect_absorption(
        self,
        result: WallAnalysis,
        all_walls: list,
        current_price: float,
    ) -> WallAnalysis:
        """Detect absorption: price repeatedly touches a wall level without breaking."""
        from config.constants import SpoofingDetection as SD

        if not self._price_touch_history:
            return result

        # Check each persistent wall for repeated price touches
        persistent_walls = (
            [(p, "bid") for p in result.persistent_bid_walls] +
            [(p, "ask") for p in result.persistent_ask_walls]
        )

        best_touches = 0
        best_side = ""

        for wall_price, side in persistent_walls:
            touch_count = sum(
                1 for _, hist_price in self._price_touch_history
                if abs(hist_price - wall_price) / max(wall_price, 1e-10)
                < SD.ABSORPTION_TOUCH_DISTANCE_PCT
            )
            if touch_count >= SD.ABSORPTION_MIN_TOUCHES and touch_count > best_touches:
                best_touches = touch_count
                best_side = side

        if best_touches >= SD.ABSORPTION_MIN_TOUCHES:
            result.absorption_detected = True
            result.absorption_side = best_side
            result.absorption_touches = best_touches
            result.notes.append(
                f"🧲 Absorption detected: price touched {best_side} wall "
                f"{best_touches}× without breaking — real institutional defense"
            )

        return result


# ── Singleton ──────────────────────────────────────────────
orderflow_analyzer = OrderFlowAnalyzer()
