"""
TitanBot Pro — Trigger Quality Analyzer
=========================================
More triggers ≠ better/faster moves.  What matters is:

  1. **Quality** of each trigger (volume-confirmed, regime-aligned)
  2. **Diversity** of trigger sources (not all momentum-based)
  3. **Volume context** (climactic volume at end of a move = exhaustion,
     not continuation; low-vol triggers are noise-prone)
  4. **Diminishing returns** — 3 high-quality triggers > 7 mediocre ones

The TriggerQualityAnalyzer produces a composite quality score (0-1) that
the engine and aggregator can use for confidence adjustment.
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.constants import TriggerQuality as TQ

logger = logging.getLogger(__name__)


# ── Trigger Categories ────────────────────────────────────────

CATEGORY_STRUCTURE = "structure"      # Swing highs/lows, S/R, pattern breaks
CATEGORY_MOMENTUM = "momentum"        # RSI, MACD, stochastic signals
CATEGORY_VOLUME = "volume"            # OBV divergence, volume spike, VWAP
CATEGORY_ORDERFLOW = "orderflow"      # OB imbalance, wall breaks, delta
CATEGORY_DERIVATIVES = "derivatives"  # Funding, OI changes, liquidations

ALL_CATEGORIES = {
    CATEGORY_STRUCTURE, CATEGORY_MOMENTUM, CATEGORY_VOLUME,
    CATEGORY_ORDERFLOW, CATEGORY_DERIVATIVES,
}


@dataclass
class Trigger:
    """A single trigger event with quality metadata."""
    name: str                          # e.g. "bullish_engulfing", "RSI_oversold"
    category: str                      # One of the CATEGORY_* constants
    raw_strength: float = 0.5          # 0-1 raw signal strength
    volume_confirmed: bool = False     # Was this trigger accompanied by volume?
    volume_mult: float = 1.0           # Volume relative to 20-bar average


@dataclass
class TriggerQualityResult:
    """Output of trigger quality analysis."""
    quality_score: float = 0.0         # 0-1 composite quality score
    quality_label: str = "LOW"         # HIGH | MEDIUM | LOW
    effective_trigger_count: float = 0.0  # Diminishing-returns-adjusted count
    diversity_bonus: float = 0.0       # Bonus from category diversity
    volume_context: str = "NORMAL"     # NORMAL | LOW_VOL | CLIMACTIC | BREAKOUT
    fast_move: bool = False            # True if fast-move setup detected
    confidence_delta: int = 0          # Confidence adjustment recommendation
    notes: List[str] = field(default_factory=list)


class TriggerQualityAnalyzer:
    """
    Evaluates the quality (not just quantity) of signal triggers.

    Key insight: research and backtesting consistently show that
    *fewer, higher-quality* triggers with volume confirmation produce
    better risk-adjusted returns than many weak/redundant triggers.
    High volume is beneficial for breakouts but can signal exhaustion
    at the end of extended moves.
    """

    def analyze(
        self,
        triggers: List[Trigger],
        volumes: Optional[np.ndarray] = None,
        closes: Optional[np.ndarray] = None,
        atr: float = 0.0,
    ) -> TriggerQualityResult:
        """
        Analyze a list of triggers and produce a quality assessment.

        Parameters
        ----------
        triggers : list of Trigger
            All triggers that fired for this signal.
        volumes : np.ndarray, optional
            Recent volume bars (for context analysis).
        closes : np.ndarray, optional
            Recent close prices (for fast-move detection).
        atr : float
            Current ATR for price velocity calculation.
        """
        result = TriggerQualityResult()

        if not triggers:
            result.notes.append("No triggers provided")
            return result

        # ── 1. Diminishing returns on trigger count ─────────────
        weighted_count = 0.0
        for i, t in enumerate(sorted(triggers, key=lambda x: x.raw_strength, reverse=True)):
            if i >= TQ.MAX_USEFUL_TRIGGERS:
                break
            # Each subsequent trigger contributes less
            decay = TQ.DIMINISH_BASE * (TQ.DIMINISH_DECAY ** i)

            # Volume confirmation multiplier
            if t.volume_confirmed:
                decay *= TQ.VOL_CONFIRMED_BONUS
            else:
                decay *= TQ.VOL_UNCONFIRMED_PENALTY

            weighted_count += decay * t.raw_strength

        result.effective_trigger_count = weighted_count

        # ── 2. Category diversity ───────────────────────────────
        categories_present = set(t.category for t in triggers if t.category in ALL_CATEGORIES)
        diversity_bonus = min(
            TQ.MAX_DIVERSITY_BONUS,
            len(categories_present) * TQ.DIVERSITY_BONUS_PER_CATEGORY,
        )
        result.diversity_bonus = diversity_bonus

        # ── 3. Volume context analysis ──────────────────────────
        vol_context = "NORMAL"
        vol_discount = 1.0

        if volumes is not None and len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            current_vol = float(volumes[-1])

            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol

                # Low-volume environment
                sorted_vols = np.sort(volumes[-20:])
                low_vol_cutoff = sorted_vols[max(0, int(len(sorted_vols) * TQ.LOW_VOL_PERCENTILE))]
                if current_vol <= low_vol_cutoff:
                    vol_context = "LOW_VOL"
                    vol_discount = TQ.LOW_VOL_TRIGGER_DISCOUNT
                    result.notes.append(TQ.LOW_VOL_WARNING_NOTE)

                # Climactic volume detection
                elif vol_ratio >= TQ.CLIMACTIC_VOL_MULT:
                    if closes is not None and len(closes) >= 4:
                        # Check if this is a reversal bar (close in opposite direction of prior trend)
                        # Use bars BEFORE the current to determine the prior trend
                        prior_trend = closes[-2] - closes[-4]
                        bar_direction = closes[-1] - closes[-2]
                        is_reversal = (prior_trend > 0 and bar_direction < 0) or \
                                      (prior_trend < 0 and bar_direction > 0)
                        if is_reversal:
                            vol_context = "CLIMACTIC"
                            vol_discount = TQ.CLIMACTIC_REVERSAL_PENALTY
                            result.notes.append(
                                f"⚠️ Climactic volume ({vol_ratio:.1f}×) with reversal bar — "
                                f"possible exhaustion"
                            )
                        else:
                            vol_context = "BREAKOUT"
                            result.notes.append(
                                f"🚀 Breakout volume ({vol_ratio:.1f}×) with trend bar — "
                                f"genuine momentum"
                            )
                    else:
                        vol_context = "HIGH"
                elif vol_ratio >= 1.5:
                    vol_context = "ABOVE_AVG"

        result.volume_context = vol_context

        # ── 4. Fast-move detection ──────────────────────────────
        if (volumes is not None and closes is not None
                and len(volumes) >= 2 and len(closes) >= 2 and atr > 0):
            avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
            current_vol = float(volumes[-1])
            price_change = abs(closes[-1] - closes[-2])

            if avg_vol > 0:
                vol_spike = current_vol / avg_vol
                price_velocity = price_change / atr

                if (vol_spike >= TQ.FAST_MOVE_VOL_SPIKE_MIN
                        and price_velocity >= TQ.FAST_MOVE_PRICE_VEL_ATR):
                    result.fast_move = True
                    result.confidence_delta += TQ.FAST_MOVE_CONF_BONUS
                    result.notes.append(
                        f"⚡ Fast-move setup: vol {vol_spike:.1f}× + "
                        f"price velocity {price_velocity:.2f} ATR/bar"
                    )

        # ── 5. Trigger sequence detection ───────────────────────
        sequence_bonus = self._detect_sequence_pattern(triggers)
        if sequence_bonus > 0:
            result.notes.append(
                f"🔗 Trigger sequence detected (+{sequence_bonus:.0%} quality)"
            )

        # ── 6. Composite quality score ──────────────────────────
        # Normalize effective trigger count (capped at ~2.5 for perfect triggers)
        max_possible = sum(
            TQ.DIMINISH_BASE * (TQ.DIMINISH_DECAY ** i) * TQ.VOL_CONFIRMED_BONUS
            for i in range(TQ.MAX_USEFUL_TRIGGERS)
        )
        count_score = min(1.0, weighted_count / max(max_possible, 0.01))

        # Diversity score (0-1)
        diversity_score = diversity_bonus / max(TQ.MAX_DIVERSITY_BONUS, 0.01)

        # Volume context score (0-1)
        vol_context_score = {
            "BREAKOUT": 1.0,
            "ABOVE_AVG": 0.7,
            "NORMAL": 0.5,
            "HIGH": 0.65,
            "LOW_VOL": 0.2,
            "CLIMACTIC": 0.3,
        }.get(vol_context, 0.5)

        # Fast-move bonus
        fast_move_bonus = 0.15 if result.fast_move else 0.0

        quality = (
            count_score * 0.35 +
            diversity_score * 0.20 +
            vol_context_score * 0.25 +
            fast_move_bonus * 0.10 +
            sequence_bonus * 0.10
        ) * vol_discount

        quality = max(0.0, min(1.0, quality))
        result.quality_score = quality

        # ── 6. Classify ────────────────────────────────────────
        if quality >= TQ.QUALITY_HIGH:
            result.quality_label = "HIGH"
        elif quality >= TQ.QUALITY_MEDIUM:
            result.quality_label = "MEDIUM"
        else:
            result.quality_label = "LOW"
            result.confidence_delta -= 2  # Penalty for low quality triggers

        result.notes.append(
            f"Trigger quality: {result.quality_label} ({quality:.2f}) — "
            f"{len(triggers)} triggers, {result.effective_trigger_count:.2f} effective, "
            f"{len(categories_present)} categories"
        )

        return result

    @staticmethod
    def _detect_sequence_pattern(triggers: List[Trigger]) -> float:
        """Detect ordered trigger sequences that indicate high-quality setups.

        High-quality entries follow predictable sequences:
          sweep → BOS → retest  (best — full institutional pattern)
          BOS → retest          (good — structure + confirmation)
          random/unordered      (neutral — no bonus)

        Returns a bonus multiplier (0.0 to TQ.SEQUENCE_SWEEP_BOS_RETEST_BONUS).
        """
        if len(triggers) < TQ.SEQUENCE_MIN_TRIGGERS:
            return 0.0

        # Classify triggers into sequence roles by name patterns
        names = [t.name.lower() for t in triggers]
        categories = [t.category for t in triggers]

        has_sweep = any(
            "sweep" in n or "liquidity" in n or "stop_hunt" in n
            for n in names
        )
        has_bos = any(
            "bos" in n or "break_of_structure" in n or "breakout" in n
            or "engulf" in n
            for n in names
        )
        has_retest = any(
            "retest" in n or "pullback" in n or "reentry" in n
            or "fvg" in n or "order_block" in n
            for n in names
        )

        # Check for structural ordering (sweep before BOS before retest)
        if has_sweep and has_bos and has_retest:
            return TQ.SEQUENCE_SWEEP_BOS_RETEST_BONUS
        if has_bos and has_retest:
            return TQ.SEQUENCE_BOS_RETEST_BONUS
        # Partial: at least BOS with structure category trigger = minor bonus
        if has_bos and CATEGORY_STRUCTURE in categories:
            return TQ.SEQUENCE_BOS_RETEST_BONUS * 0.5

        return 0.0


# ── Singleton ──────────────────────────────────────────────
trigger_quality_analyzer = TriggerQualityAnalyzer()
