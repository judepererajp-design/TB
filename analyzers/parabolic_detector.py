"""
TitanBot Pro — Parabolic & Exhaustion Detector
================================================
Standalone analyzer that flags parabolic price acceleration and
exhaustion patterns. Strategies and the aggregator can query this
for early warnings before committing to a trade.

Usage:
    from analyzers.parabolic_detector import parabolic_detector
    result = parabolic_detector.analyze(ohlcv_list)
    if result.is_parabolic:
        confidence -= 10  # entering at the top
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParabolicResult:
    """Result of parabolic/exhaustion analysis."""
    is_parabolic: bool = False
    is_exhausted: bool = False
    acceleration: float = 0.0       # ROC-of-ROC
    roc: float = 0.0                # current ROC
    direction: str = "FLAT"         # UP / DOWN / FLAT
    exhaustion_signals: List[str] = field(default_factory=list)
    exhaustion_score: float = 0.0   # 0-1
    confidence_penalty: int = 0     # suggested penalty for signal confidence
    notes: List[str] = field(default_factory=list)


class ParabolicDetector:
    """
    Analyzes OHLCV data for parabolic moves and exhaustion patterns.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 120  # 2 minutes

    def analyze(
        self,
        ohlcv: list,
        direction: str = "LONG",
        roc_period: int = 10,
        accel_threshold: float = 0.005,
    ) -> ParabolicResult:
        """
        Analyze OHLCV data for parabolic moves and exhaustion.

        Args:
            ohlcv: List of [ts, open, high, low, close, volume] candles
            direction: "LONG" or "SHORT" — direction of the proposed trade
            roc_period: lookback for ROC calculation
            accel_threshold: minimum acceleration to flag as parabolic

        Returns:
            ParabolicResult with analysis
        """
        result = ParabolicResult()

        if not ohlcv or len(ohlcv) < roc_period * 2 + 5:
            return result

        try:
            opens   = np.array([float(c[1]) for c in ohlcv], dtype=float)
            highs   = np.array([float(c[2]) for c in ohlcv], dtype=float)
            lows    = np.array([float(c[3]) for c in ohlcv], dtype=float)
            closes  = np.array([float(c[4]) for c in ohlcv], dtype=float)
            volumes = np.array([float(c[5]) for c in ohlcv], dtype=float)
        except (IndexError, TypeError, ValueError):
            return result

        # ── ROC-of-ROC (acceleration) ───────────────────────────────
        roc_series = []
        for i in range(roc_period, len(closes)):
            prev = closes[i - roc_period]
            if prev == 0:
                roc_series.append(0.0)
            else:
                roc_series.append((closes[i] - prev) / prev)

        if len(roc_series) < roc_period + 1:
            return result

        result.roc = roc_series[-1]

        # Acceleration = change in ROC
        roc_of_roc = [roc_series[i] - roc_series[i - 1] for i in range(1, len(roc_series))]
        if not roc_of_roc:
            return result

        result.acceleration = roc_of_roc[-1]

        # Check for sustained acceleration (3+ bars same direction)
        recent_accel = roc_of_roc[-3:] if len(roc_of_roc) >= 3 else roc_of_roc
        avg_accel = float(np.mean(recent_accel))
        all_same_sign = all(a > 0 for a in recent_accel) or all(a < 0 for a in recent_accel)

        result.is_parabolic = all_same_sign and abs(avg_accel) > accel_threshold

        if result.roc > 0.01:
            result.direction = "UP"
        elif result.roc < -0.01:
            result.direction = "DOWN"
        else:
            result.direction = "FLAT"

        # ── Exhaustion Detection ───────────────────────────────────
        exhaust_signals = []
        exhaust_score = 0.0

        # Histogram deceleration proxy (using closes as momentum proxy)
        if len(closes) >= 5:
            _mom = [closes[-i] - closes[-i - 1] for i in range(1, 4)]
            if all(abs(_mom[i]) < abs(_mom[i + 1]) for i in range(len(_mom) - 1)):
                exhaust_signals.append("momentum_deceleration")
                exhaust_score += 0.3

        # Candle range contraction
        ranges = highs - lows
        if len(ranges) >= 5:
            if ranges[-1] < ranges[-2] < ranges[-3]:
                exhaust_signals.append("range_contraction")
                exhaust_score += 0.25
            avg_range = float(np.mean(ranges[-20:]))
            if avg_range > 0 and ranges[-1] < avg_range * 0.4:
                exhaust_signals.append("doji_exhaustion")
                exhaust_score += 0.15

        # Volume climax
        if len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            if avg_vol > 0:
                vol_ratio = volumes[-1] / avg_vol
                if vol_ratio > 3.0:
                    exhaust_signals.append("volume_climax")
                    exhaust_score += 0.35
                elif vol_ratio > 2.0:
                    exhaust_signals.append("volume_spike")
                    exhaust_score += 0.15

        result.is_exhausted = len(exhaust_signals) >= 2
        result.exhaustion_signals = exhaust_signals
        result.exhaustion_score = min(1.0, round(exhaust_score, 3))

        # ── Confidence penalty calculation ──────────────────────────
        penalty = 0
        notes = []

        if result.is_parabolic:
            # Buying into a parabolic UP move or shorting a parabolic DOWN move
            if (direction == "LONG" and result.direction == "UP"):
                penalty -= 12
                notes.append("⚠️ Parabolic UP — entering LONG at potential blow-off top")
            elif (direction == "SHORT" and result.direction == "DOWN"):
                penalty -= 12
                notes.append("⚠️ Parabolic DOWN — entering SHORT at potential capitulation")
            # Trading against parabolic = good (reversal)
            elif (direction == "SHORT" and result.direction == "UP"):
                penalty += 5
                notes.append("✅ Parabolic UP exhaustion — SHORT entry supported")
            elif (direction == "LONG" and result.direction == "DOWN"):
                penalty += 5
                notes.append("✅ Parabolic DOWN exhaustion — LONG entry supported")

        if result.is_exhausted:
            penalty -= 8
            notes.append(f"⚠️ Exhaustion detected: {', '.join(exhaust_signals)}")

        result.confidence_penalty = penalty
        result.notes = notes
        return result


# Module-level singleton
parabolic_detector = ParabolicDetector()
