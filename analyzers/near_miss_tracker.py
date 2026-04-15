"""
TitanBot Pro — Near-Miss Tracker
==================================
Tracks blocked signals that were close to passing the execution gate
("near misses") and monitors whether they would have been profitable.

This drives the execution gate feedback loop:
  - If near-misses frequently "would have won" → gate is too strict
  - If near-misses frequently "would have lost" → gate is correctly calibrated

Tracks:
  - Near-miss count and rate
  - Outcome monitoring (did price hit TP after block?)
  - False positive / false negative rates for the gate
  - Per-factor accuracy (which factors most often caused wrong blocks?)

Integration:
  - engine.py records near-misses when execution gate blocks with is_near_miss=True
  - outcome_monitor.py feeds live price snapshots back into check_outcome()
  - /api/diagnostics exposes near-miss metrics
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional

from config.constants import ExecutionGate as EG

logger = logging.getLogger(__name__)


@dataclass
class NearMissRecord:
    """A single near-miss event."""
    symbol: str
    direction: str                        # LONG / SHORT
    strategy: str
    execution_score: float                # The composite score at block time
    block_threshold: float                # The threshold it failed to meet
    factors: Dict[str, float] = field(default_factory=dict)
    bad_factors: List[str] = field(default_factory=list)
    kill_combo: str = ""
    grade: str = "B"
    confidence: float = 0.0               # Setup confidence at block time
    entry_price: float = 0.0              # Price when blocked
    tp_price: float = 0.0                 # Target price (for outcome check)
    sl_price: float = 0.0                 # Stop loss price (for outcome check)
    ts: float = field(default_factory=time.time)  # Timestamp of block

    # Outcome fields (filled later by check_outcomes)
    outcome_checked: bool = False
    would_have_won: Optional[bool] = None  # True if price hit TP
    would_have_lost: Optional[bool] = None  # True if price hit SL first
    max_favorable: float = 0.0            # Max favorable move after block (%)
    max_adverse: float = 0.0              # Max adverse move after block (%)
    outcome_ts: float = 0.0               # When outcome was determined

    # Promotion tracking
    promoted: bool = False                 # True after re-activation


class NearMissTracker:
    """
    Tracks near-miss blocked signals and their post-block outcomes.

    The tracker maintains a rolling window of near-miss events and computes
    feedback metrics that can be used to tune execution gate thresholds.
    All public methods take an ``RLock`` before mutating or reading shared
    tracker state so callers can safely use the singleton from concurrent
    engine and monitoring paths.
    """

    def __init__(self, max_tracked: int = 0):
        max_sz = max_tracked or EG.NEAR_MISS_MAX_TRACKED
        self._near_misses: deque = deque(maxlen=max_sz)
        self._lock = RLock()

        # Counters for feedback metrics
        self._total_blocks: int = 0       # All blocks (near-miss or not)
        self._total_passes: int = 0       # All passes
        self._total_near_misses: int = 0
        self._outcomes_checked: int = 0
        self._would_have_won: int = 0     # Near-misses that would have hit TP
        self._would_have_lost: int = 0    # Near-misses that would have hit SL
        self._outcome_unknown: int = 0    # Near-misses where price didn't move enough

        # Track pass outcomes (from learning_loop feedback)
        self._pass_wins: int = 0
        self._pass_losses: int = 0

    def record_block(self):
        """Called for every execution gate block (near-miss or not)."""
        with self._lock:
            self._total_blocks += 1

    def record_pass(self):
        """Called for every execution gate pass."""
        with self._lock:
            self._total_passes += 1

    def record_pass_outcome(self, won: bool):
        """Called when a passed signal's trade closes."""
        with self._lock:
            if won:
                self._pass_wins += 1
            else:
                self._pass_losses += 1

    def get_pending_keys(self) -> List[tuple]:
        """Return unique (symbol, direction) pairs with unresolved outcomes."""
        with self._lock:
            pending = []
            seen = set()
            for record in self._near_misses:
                if record.outcome_checked:
                    continue
                key = (record.symbol, record.direction)
                if key in seen:
                    continue
                seen.add(key)
                pending.append(key)
            return pending

    def record_near_miss(
        self,
        *,
        symbol: str,
        direction: str,
        strategy: str = "",
        execution_score: float = 0.0,
        block_threshold: float = 35.0,
        factors: Optional[Dict[str, float]] = None,
        bad_factors: Optional[List[str]] = None,
        kill_combo: str = "",
        grade: str = "B",
        confidence: float = 0.0,
        entry_price: float = 0.0,
        tp_price: float = 0.0,
        sl_price: float = 0.0,
    ) -> NearMissRecord:
        """
        Record a near-miss blocked signal for outcome tracking.

        Called by engine.py when execution gate blocks with is_near_miss=True.
        """
        record = NearMissRecord(
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            execution_score=execution_score,
            block_threshold=block_threshold,
            factors=factors or {},
            bad_factors=bad_factors or [],
            kill_combo=kill_combo,
            grade=grade,
            confidence=confidence,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
        )
        with self._lock:
            self._near_misses.append(record)
            self._total_near_misses += 1

        logger.info(
            f"📍 Near-miss recorded: {symbol} {direction} | "
            f"exec_score={execution_score:.0f} (threshold={block_threshold:.0f}) | "
            f"bad=[{', '.join(bad_factors or [])}]"
        )
        return record

    def check_outcome(
        self,
        symbol: str,
        direction: str,
        high_price: float,
        low_price: float,
        current_price: float,
    ):
        """
        Check pending near-misses for the given symbol against observed prices.

        Called periodically (e.g., each analysis cycle) with recent price data.
        Updates near-miss records with outcome information.
        """
        now = time.time()
        window = EG.NEAR_MISS_OUTCOME_WINDOW

        if not all(math.isfinite(v) for v in (high_price, low_price, current_price)):
            logger.debug(
                "Invalid prices for %s %s: high=%s low=%s current=%s",
                symbol,
                direction,
                high_price,
                low_price,
                current_price,
            )
            return

        with self._lock:
            for record in self._near_misses:
                if record.outcome_checked:
                    continue
                if record.symbol != symbol:
                    continue
                if record.direction != direction:
                    continue
                if now - record.ts > window:
                    # Expired — mark as checked with unknown outcome
                    record.outcome_checked = True
                    record.outcome_ts = now
                    self._outcomes_checked += 1
                    self._outcome_unknown += 1
                    continue

                if record.entry_price <= 0 or not math.isfinite(record.entry_price):
                    continue

                # Compute max favorable/adverse excursion
                if direction == "LONG":
                    favorable = ((high_price - record.entry_price) / record.entry_price) * 100
                    adverse = ((record.entry_price - low_price) / record.entry_price) * 100
                else:  # SHORT
                    favorable = ((record.entry_price - low_price) / record.entry_price) * 100
                    adverse = ((high_price - record.entry_price) / record.entry_price) * 100

                record.max_favorable = max(record.max_favorable, max(0.0, favorable))
                record.max_adverse = max(record.max_adverse, max(0.0, adverse))

                # Check if TP or SL would have been hit
                tp_hit = False
                sl_hit = False

                if record.tp_price > 0:
                    if direction == "LONG":
                        tp_hit = high_price >= record.tp_price
                        sl_hit = low_price <= record.sl_price if record.sl_price > 0 else False
                    else:
                        tp_hit = low_price <= record.tp_price
                        sl_hit = high_price >= record.sl_price if record.sl_price > 0 else False
                else:
                    # No TP/SL — use percentage threshold
                    tp_hit = record.max_favorable >= EG.NEAR_MISS_TP_CHECK_PERCENT
                    sl_hit = record.max_adverse >= EG.NEAR_MISS_TP_CHECK_PERCENT

                if tp_hit and not sl_hit:
                    record.outcome_checked = True
                    record.would_have_won = True
                    record.would_have_lost = False
                    record.outcome_ts = now
                    self._outcomes_checked += 1
                    self._would_have_won += 1
                    logger.info(
                        f"📍 Near-miss WOULD HAVE WON: {symbol} {direction} | "
                        f"score was {record.execution_score:.0f} | "
                        f"max favorable {record.max_favorable:.2f}%"
                    )
                elif sl_hit and not tp_hit:
                    record.outcome_checked = True
                    record.would_have_won = False
                    record.would_have_lost = True
                    record.outcome_ts = now
                    self._outcomes_checked += 1
                    self._would_have_lost += 1

    def get_metrics(self) -> Dict:
        """
        Compute execution gate feedback metrics.

        Returns dict with:
          - near_miss_count: total near-misses recorded
          - near_miss_rate: near-misses / total blocks
          - would_have_won_count: near-misses that would have hit TP
          - would_have_won_rate: fraction of checked near-misses that won
          - false_negative_rate: near-misses that would have won / total blocks
            (how often the gate wrongly blocked a winner)
          - pass_win_rate: win rate of signals that passed the gate
          - gate_accuracy: overall accuracy of block/pass decisions
          - recent_near_misses: last 10 near-miss records as dicts
        """
        with self._lock:
            metrics: Dict = {}

            # Basic counts
            metrics["total_blocks"] = self._total_blocks
            metrics["total_passes"] = self._total_passes
            metrics["near_miss_count"] = self._total_near_misses
            metrics["outcomes_checked"] = self._outcomes_checked
            metrics["would_have_won"] = self._would_have_won
            metrics["would_have_lost"] = self._would_have_lost
            metrics["outcome_unknown"] = self._outcome_unknown

            # Rates
            if self._total_blocks > 0:
                metrics["near_miss_rate"] = round(
                    self._total_near_misses / self._total_blocks, 4
                )
            else:
                metrics["near_miss_rate"] = 0.0

            if self._outcomes_checked > 0:
                metrics["would_have_won_rate"] = round(
                    self._would_have_won / self._outcomes_checked, 4
                )
            else:
                metrics["would_have_won_rate"] = 0.0

            if self._total_blocks > 0:
                metrics["false_negative_rate"] = round(
                    self._would_have_won / self._total_blocks, 4
                )
            else:
                metrics["false_negative_rate"] = 0.0

            total_pass_outcomes = self._pass_wins + self._pass_losses
            if total_pass_outcomes > 0:
                metrics["pass_win_rate"] = round(
                    self._pass_wins / total_pass_outcomes, 4
                )
            else:
                metrics["pass_win_rate"] = 0.0

            correct_blocks = self._would_have_lost
            correct_passes = self._pass_wins
            total_known = (
                self._would_have_won + self._would_have_lost +
                self._pass_wins + self._pass_losses
            )
            if total_known >= EG.FEEDBACK_MIN_SAMPLE:
                metrics["gate_accuracy"] = round(
                    (correct_blocks + correct_passes) / total_known, 4
                )
            else:
                metrics["gate_accuracy"] = None

            factor_blame: Dict[str, int] = {}
            for record in self._near_misses:
                if record.would_have_won:
                    for f in record.bad_factors:
                        factor_blame[f] = factor_blame.get(f, 0) + 1
            metrics["factor_false_negative_blame"] = factor_blame

            recent = list(self._near_misses)[-10:]
            metrics["recent_near_misses"] = [
                {
                    "symbol": r.symbol,
                    "direction": r.direction,
                    "execution_score": round(r.execution_score, 1),
                    "block_threshold": r.block_threshold,
                    "bad_factors": r.bad_factors,
                    "kill_combo": r.kill_combo,
                    "grade": r.grade,
                    "is_checked": r.outcome_checked,
                    "would_have_won": r.would_have_won,
                    "max_favorable_pct": round(r.max_favorable, 2),
                    "max_adverse_pct": round(r.max_adverse, 2),
                    "age_seconds": round(time.time() - r.ts),
                }
                for r in recent
            ]

            return metrics

    # ── Signal Promotion ─────────────────────────────────────────

    def get_promotable_near_misses(self, max_age_secs: int = 0) -> List[NearMissRecord]:
        """Return near-misses that are young enough to be re-evaluated.

        These are records that were blocked, haven't been promoted yet,
        and are within the promotion age window.
        """
        max_age = max_age_secs or EG.PROMOTION_MAX_AGE_SECS
        now = time.time()
        with self._lock:
            candidates = []
            for record in self._near_misses:
                if record.outcome_checked:
                    continue
                if now - record.ts > max_age:
                    continue
                if record.promoted:
                    continue
                candidates.append(record)
            return candidates

    def mark_promoted(self, record: NearMissRecord):
        """Mark a near-miss record as promoted (re-activated)."""
        with self._lock:
            record.promoted = True
            record.outcome_checked = True
        logger.info(
            f"🔄 Near-miss PROMOTED: {record.symbol} {record.direction} | "
            f"original_score={record.execution_score:.0f}"
        )

    # ── Confidence Calibration (Feedback Loop) ───────────────────

    def compute_threshold_adjustment(self) -> float:
        """Compute how much to adjust the gate threshold based on recent outcomes.

        Returns a signed float: negative = loosen (too strict), positive = tighten.
        Returns 0.0 if insufficient data or calibration disabled.
        """
        if not EG.CALIBRATION_ENABLED:
            return 0.0

        with self._lock:
            total_decisions = (
                self._would_have_won + self._would_have_lost +
                self._pass_wins + self._pass_losses
            )
            if total_decisions < EG.FEEDBACK_MIN_SAMPLE:
                return 0.0

            fnr = 0.0
            if self._total_blocks > 0:
                fnr = self._would_have_won / self._total_blocks

            pass_total = self._pass_wins + self._pass_losses
            plr = 0.0
            if pass_total > 0:
                plr = self._pass_losses / pass_total

        # Decision: if FNR too high → gate is blocking too many winners → loosen
        if fnr > EG.CALIBRATION_FNR_HIGH:
            adj = EG.CALIBRATION_LOOSEN_STEP
            logger.info(
                f"📊 Calibration: FNR={fnr:.2%} > {EG.CALIBRATION_FNR_HIGH:.0%} "
                f"→ loosening by {adj}"
            )
            return adj

        # If FNR is very low but pass-loss-rate is high → gate is too loose → tighten
        if fnr < EG.CALIBRATION_FNR_LOW and plr > 0.60:
            adj = EG.CALIBRATION_TIGHTEN_STEP
            logger.info(
                f"📊 Calibration: FNR={fnr:.2%} low, PLR={plr:.2%} high "
                f"→ tightening by {adj}"
            )
            return adj

        return 0.0


# ── Singleton ──────────────────────────────────────────────
near_miss_tracker = NearMissTracker()
