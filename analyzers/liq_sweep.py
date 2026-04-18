"""
Liquidity Sweep Probability Estimator
======================================
Proactive estimate of the probability that price is set up to sweep
nearby resting liquidity *before* the trade thesis plays out.

Existing code only detects sweeps **after** the fact
(`outcome_monitor._is_stop_hunt_wick`), which prevents misclassifying
the loss but doesn't prevent taking the loss in the first place.

We use two cheap, high-signal proxies:

  1. **Swing-extreme proximity**  — the stop sits just beyond a
     recent local high (for SHORT) or low (for LONG). Resting stops
     cluster on the wrong side of swings; aggressive market-makers
     and predatory flow target them.

  2. **Round-number proximity**   — the stop sits just beyond a
     "round" price (e.g. 60 000 BTC, 0.50 ALT). Retail orders cluster
     on these levels; same hunting dynamic.

The output is a probability in ``[0.0, 1.0]`` that a sweep is likely.
The aggregator translates this into a (small) confidence penalty,
linearly scaled. We never hard-block — sweeps and continuations both
happen, and stops survive most of the time. The penalty just makes
sweep-prone setups need a slightly stronger thesis to publish.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from config.constants import LiqSweep as LSC

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    probability: float                    # 0.0–1.0
    swing_proximity: float                # 0.0–1.0 component
    round_proximity: float                # 0.0–1.0 component
    nearest_level: Optional[float] = None
    level_kind: str = ""                  # "swing_low" / "swing_high" / "round" / ""
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


class LiqSweepEstimator:
    """Stateless estimator. Cheap enough to call per-signal."""

    def estimate(
        self,
        direction: str,
        entry: float,
        stop: float,
        ohlcv: Optional[Sequence] = None,
    ) -> SweepResult:
        """
        Estimate sweep probability for a single signal.

        Args:
            direction: ``"LONG"`` or ``"SHORT"``.
            entry:     Entry price (mid of the entry zone is fine).
            stop:      Stop-loss price.
            ohlcv:     Optional OHLCV bars (any timeframe).
                       Each row may be either ``[ts, o, h, l, c, v]``
                       (CCXT format) or an object with attributes
                       ``high``/``low``. When omitted, the swing
                       proximity component is 0.

        Returns:
            SweepResult with ``probability`` ∈ [0, 1].
        """
        if entry <= 0 or stop <= 0:
            return SweepResult(0.0, 0.0, 0.0, notes=["invalid entry/stop"])
        direction = direction.upper()
        if direction not in ("LONG", "SHORT"):
            return SweepResult(0.0, 0.0, 0.0, notes=[f"unknown direction {direction!r}"])

        # ── Component 1: swing-extreme proximity ────────────
        swing_score = 0.0
        nearest_level: Optional[float] = None
        level_kind = ""
        if ohlcv:
            highs, lows = self._extract_highs_lows(ohlcv)
            if highs and lows:
                lookback = max(2, min(LSC.SWING_LOOKBACK_BARS, len(highs)))
                window_highs = highs[-lookback:]
                window_lows = lows[-lookback:]
                # For a LONG, swept stops cluster *below* recent low.
                # For a SHORT, swept stops cluster *above* recent high.
                if direction == "LONG":
                    extreme = min(window_lows)
                    # Only score if the stop is *below* the extreme
                    # (genuine resting-stop zone). A stop above the
                    # local low is wider than the swing and not a
                    # natural hunt target.
                    if stop <= extreme:
                        swing_score = self._proximity_score(stop, extreme, entry)
                        if swing_score > 0:
                            nearest_level = extreme
                            level_kind = "swing_low"
                else:
                    extreme = max(window_highs)
                    if stop >= extreme:
                        swing_score = self._proximity_score(stop, extreme, entry)
                        if swing_score > 0:
                            nearest_level = extreme
                            level_kind = "swing_high"

        # ── Component 2: round-number proximity ─────────────
        round_score = 0.0
        round_level = self._nearest_round(stop)
        if round_level is not None and round_level > 0:
            # Resting stops cluster on the round number that sits between
            # the stop and the entry — i.e. for a LONG with entry=60 100
            # and stop=59 950, the $60 000 round level is the natural
            # hunt target. Sweeping it triggers retail/algo stops parked
            # just below.
            #   LONG  : entry > round_level >= stop  (round above stop, below entry)
            #   SHORT : entry < round_level <= stop  (round below stop, above entry)
            if direction == "LONG" and stop <= round_level <= entry:
                round_score = self._proximity_score(stop, round_level, entry)
            elif direction == "SHORT" and entry <= round_level <= stop:
                round_score = self._proximity_score(stop, round_level, entry)
            if round_score > 0 and not nearest_level:
                nearest_level = round_level
                level_kind = "round"

        # Probability is a weighted combination, clipped to [0, 1].
        probability = min(
            1.0,
            swing_score * LSC.SWING_WEIGHT + round_score * LSC.ROUND_WEIGHT,
        )

        notes: List[str] = []
        if probability >= LSC.WARN_PROBABILITY:
            notes.append(
                f"⚠️ Sweep risk {probability:.0%} "
                f"(swing={swing_score:.2f}, round={round_score:.2f})"
            )

        return SweepResult(
            probability=probability,
            swing_proximity=swing_score,
            round_proximity=round_score,
            nearest_level=nearest_level,
            level_kind=level_kind,
            notes=notes,
        )

    def confidence_penalty(self, probability: float) -> float:
        """Translate a sweep probability into a confidence penalty (points)."""
        p = max(0.0, min(1.0, float(probability or 0.0)))
        return p * LSC.MAX_CONFIDENCE_PENALTY

    # ── Internals ──────────────────────────────────────────

    @staticmethod
    def _proximity_score(stop: float, level: float, entry: float) -> float:
        """
        Score 0–1 for how close ``stop`` sits to ``level``.

        Returns 1.0 when stop is exactly on the level, 0.0 when it's
        ≥ ``PROXIMITY_PCT`` away (relative to entry). We use entry as
        the scale so a 0.5 % proximity means the same thing on a
        $60 000 chart and a $0.05 chart.
        """
        if entry <= 0:
            return 0.0
        max_dist = entry * LSC.PROXIMITY_PCT
        if max_dist <= 0:
            return 0.0
        dist = abs(stop - level)
        if dist >= max_dist:
            return 0.0
        return 1.0 - (dist / max_dist)

    @staticmethod
    def _extract_highs_lows(ohlcv: Sequence) -> tuple:
        """Coerce CCXT-style or object-style OHLCV into (highs, lows)."""
        highs: List[float] = []
        lows: List[float] = []
        for row in ohlcv:
            try:
                if hasattr(row, "high") and hasattr(row, "low"):
                    h = float(row.high)
                    lo = float(row.low)
                else:
                    # CCXT: [ts, open, high, low, close, volume]
                    h = float(row[2])
                    lo = float(row[3])
                if h > 0 and lo > 0:
                    highs.append(h)
                    lows.append(lo)
            except (TypeError, ValueError, IndexError):
                continue
        return highs, lows

    @staticmethod
    def _nearest_round(price: float) -> Optional[float]:
        """
        Return the nearest "round" price for *price*.

        We pick the largest granularity ≤ 1 % of price so the level
        is materially meaningful. E.g. on BTC $60 000 we round to
        $1 000 ticks; on a $0.05 alt we round to $0.001 ticks.
        """
        if price <= 0:
            return None
        # Granularity is the largest power-of-10 slice ≤ 1% of price,
        # with mid-decade fractions for finer resolution.
        magnitude = 10 ** math.floor(math.log10(price))
        granularity = None
        max_size = price * LSC.ROUND_NUMBER_MAX_FRACTION_OF_PRICE
        for frac in LSC.ROUND_NUMBER_FRACTIONS:
            candidate = magnitude * frac
            if candidate > 0 and candidate <= max_size:
                granularity = candidate
                break
        if granularity is None or granularity <= 0:
            return None
        # Round to the nearest multiple
        return round(price / granularity) * granularity


# ── Singleton ─────────────────────────────────────────────────
liq_sweep_estimator = LiqSweepEstimator()
