"""
TitanBot Pro — Adaptive Ensemble Weights
==========================================
Self-learning signal weight system that closes the feedback loop:
outcome data → per-source scoring → regime-aware weight overrides.

Score formula (GPT-improved):
    score = (win_rate × avg_return) / max(drawdown, floor)

This avoids the trap of rewarding high win-rate but low-return sources
and penalises sources that contribute to drawdowns.

Rolling windows: 30d (60% blend) + 60d (40% blend) for stability.
Minimum sample threshold prevents overfitting on small data.

Integration:
    EnsembleVoter._get_weights() checks adaptive_weight_manager first.
    Falls back to static _WEIGHTS_BY_REGIME when data is insufficient.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.constants import AdaptiveWeights as AWC

logger = logging.getLogger(__name__)


# ── Vote source → ensemble voter source name mapping ──────────────
# These are the 9 sources the ensemble voter can weight
ALL_SOURCES = (
    "cvd", "smart_money", "oi_trend", "crowd", "whale_flow",
    "basis", "onchain", "stablecoin", "whale_intent",
    "mining_health", "network_demand", "vol_regime",
)


@dataclass
class SourceScore:
    """Performance score for a single vote source in a specific regime."""
    source: str
    regime: str
    win_rate: float = 0.0
    avg_return: float = 0.0        # Average R-multiple on completed trades
    max_drawdown: float = 0.0      # Worst R-multiple in the window
    sample_count: int = 0
    score: float = AWC.DEFAULT_SCORE
    adaptive_weight: float = 1.0


@dataclass
class AdaptiveWeightSnapshot:
    """Current adaptive weights for a regime."""
    regime: str
    weights: Dict[str, float] = field(default_factory=dict)
    scores: Dict[str, SourceScore] = field(default_factory=dict)
    last_update: float = 0.0
    sufficient_data: bool = False


class AdaptiveWeightManager:
    """
    Periodically queries outcome data and computes performance-based
    weights for each ensemble vote source, per regime.

    Usage:
        weights = adaptive_weight_manager.get_weights(regime)
        if weights:
            # Use adaptive weights
        else:
            # Fall back to static weights
    """

    def __init__(self):
        self._snapshots: Dict[str, AdaptiveWeightSnapshot] = {}
        self._last_recalc: float = 0.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start periodic recalculation loop.

        On start, load any persisted snapshots from DB so that adaptive
        weights are available immediately (not empty for the first 6h).
        """
        if self._running:
            return
        await self._load_persisted_snapshots()
        self._running = True
        self._task = asyncio.create_task(self._recalc_loop())
        logger.info("📊 AdaptiveWeightManager started (recalc every %dh)",
                     AWC.RECALC_INTERVAL_SECS // 3600)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _recalc_loop(self):
        """Periodically recalculate adaptive weights."""
        while self._running:
            try:
                await self.recalculate()
            except Exception as e:
                logger.warning(f"Adaptive weight recalc error: {e}")
            await asyncio.sleep(AWC.RECALC_INTERVAL_SECS)

    # ── Core calculation ──────────────────────────────────────────

    async def recalculate(self):
        """
        Query outcome data and compute adaptive weights for each regime.

        We query signals + outcomes, group by regime and ensemble vote
        alignment, and score each source.
        """
        try:
            from data.database import db
        except ImportError:
            logger.debug("Database not available for adaptive weights")
            return

        now = time.time()

        # Get signals with outcomes from both windows
        short_signals = await db.get_recent_signals(
            hours=AWC.SHORT_WINDOW_DAYS * 24, exclude_c_grade=True
        )
        long_signals = await db.get_recent_signals(
            hours=AWC.LONG_WINDOW_DAYS * 24, exclude_c_grade=True
        )

        # Group by regime
        regimes = set()
        for s in long_signals:
            regime = s.get("regime") or "UNKNOWN"
            if regime != "UNKNOWN":
                regimes.add(regime)

        for regime in regimes:
            # Filter signals for this regime that have outcomes
            short_regime = [
                s for s in short_signals
                if (s.get("regime") or "UNKNOWN") == regime
                and s.get("outcome") in ("WIN", "LOSS", "BREAKEVEN")
            ]
            long_regime = [
                s for s in long_signals
                if (s.get("regime") or "UNKNOWN") == regime
                and s.get("outcome") in ("WIN", "LOSS", "BREAKEVEN")
            ]

            # Compute blended scores per source
            scores: Dict[str, SourceScore] = {}
            weights: Dict[str, float] = {}
            sufficient = False

            for source in ALL_SOURCES:
                short_score = self._compute_source_score(source, regime, short_regime)
                long_score = self._compute_source_score(source, regime, long_regime)

                # Blend: 60% short + 40% long
                if short_score.sample_count >= AWC.MIN_SAMPLES:
                    blended_score = (
                        short_score.score * AWC.SHORT_WINDOW_BLEND +
                        long_score.score * AWC.LONG_WINDOW_BLEND
                    )
                    sufficient = True

                    # Stability factor: penalise when 30d vs 60d diverge
                    # (a source that worked briefly but is unstable gets docked)
                    if long_score.sample_count >= AWC.MIN_SAMPLES:
                        _max_s = max(short_score.score, long_score.score, 0.1)
                        _diff = abs(short_score.score - long_score.score)
                        stability = max(
                            AWC.STABILITY_FLOOR,
                            1.0 - _diff / _max_s,
                        )
                        blended_score *= stability
                elif long_score.sample_count >= AWC.MIN_SAMPLES:
                    blended_score = long_score.score
                    sufficient = True
                else:
                    blended_score = AWC.DEFAULT_SCORE

                # Normalise to weight (clamp to floor/ceil)
                weight = max(AWC.WEIGHT_FLOOR, min(AWC.WEIGHT_CEIL, blended_score))

                final_score = SourceScore(
                    source=source,
                    regime=regime,
                    win_rate=short_score.win_rate if short_score.sample_count > 0 else long_score.win_rate,
                    avg_return=short_score.avg_return if short_score.sample_count > 0 else long_score.avg_return,
                    max_drawdown=short_score.max_drawdown if short_score.sample_count > 0 else long_score.max_drawdown,
                    # FIX: short_signals is a subset of long_signals (30d ⊂ 60d), so
                    # short.sample_count + long.sample_count double-counts the overlap.
                    # Report the long-window count as the authoritative sample size.
                    sample_count=max(short_score.sample_count, long_score.sample_count),
                    score=blended_score,
                    adaptive_weight=weight,
                )
                scores[source] = final_score
                weights[source] = weight

            self._snapshots[regime] = AdaptiveWeightSnapshot(
                regime=regime,
                weights=weights,
                scores=scores,
                last_update=now,
                sufficient_data=sufficient,
            )

        self._last_recalc = now
        if regimes:
            logger.info(
                f"📊 Adaptive weights recalculated for {len(regimes)} regimes "
                f"({sum(s.sufficient_data for s in self._snapshots.values())} with sufficient data)"
            )
            # Persist snapshots to DB so they survive restarts
            await self._persist_snapshots()

    # ── Persistence ───────────────────────────────────────────────

    _PERSIST_KEY = "adaptive_weight_snapshots"

    async def _persist_snapshots(self) -> None:
        """Save current snapshots to DB via learning_state table."""
        try:
            from data.database import db
            payload: Dict = {}
            for regime, snap in self._snapshots.items():
                payload[regime] = {
                    "weights": snap.weights,
                    "last_update": snap.last_update,
                    "sufficient_data": snap.sufficient_data,
                }
            await db.save_learning_state(self._PERSIST_KEY, payload)
            logger.debug("📊 Adaptive weight snapshots persisted (%d regimes)", len(payload))
        except Exception as e:
            logger.debug(f"Adaptive weights persist failed (non-fatal): {e}")

    async def _load_persisted_snapshots(self) -> None:
        """Load previously persisted snapshots from DB on startup."""
        try:
            from data.database import db
            payload = await db.load_learning_state(self._PERSIST_KEY)
            if not payload:
                logger.debug("📊 No persisted adaptive weights found — starting fresh")
                return
            loaded = 0
            for regime, data in payload.items():
                if not isinstance(data, dict):
                    continue
                self._snapshots[regime] = AdaptiveWeightSnapshot(
                    regime=regime,
                    weights=data.get("weights", {}),
                    scores={},  # Scores aren't persisted — recalculated on next cycle
                    last_update=data.get("last_update", 0.0),
                    sufficient_data=data.get("sufficient_data", False),
                )
                loaded += 1
            if loaded:
                self._last_recalc = max(
                    s.last_update for s in self._snapshots.values()
                )
                logger.info(
                    f"📊 Loaded persisted adaptive weights for {loaded} regimes "
                    f"(age: {(time.time() - self._last_recalc) / 60:.0f}min)"
                )
        except Exception as e:
            logger.debug(f"Adaptive weights load failed (non-fatal): {e}")

    def _compute_source_score(
        self, source: str, regime: str, signals: List[Dict]
    ) -> SourceScore:
        """
        Compute performance score for a vote source in a regime.

        Prefer persisted per-source ensemble vote attributions when available.
        Backward-compatible fallback: if historical rows do not contain
        ensemble_votes, use overall signal outcomes as a proxy.

        Score = (win_rate × avg_return) / max(drawdown, floor)
        """
        if not signals:
            return SourceScore(source=source, regime=regime)

        # Sentinel: ensemble_votes blob exists but this source is absent —
        # the source did NOT vote on this signal and must not be credited.
        _SOURCE_ABSENT = object()

        def _get_source_vote_value(signal: Dict):
            """Return int vote value, None (legacy/missing blob), or _SOURCE_ABSENT."""
            votes = signal.get("ensemble_votes")
            if not votes:
                return None
            if isinstance(votes, str):
                try:
                    votes = json.loads(votes)
                except Exception:
                    return None
            if not isinstance(votes, dict):
                return None
            # FIX P4-1: blob exists — distinguish "source absent" from "legacy row".
            if source not in votes:
                return _SOURCE_ABSENT
            vote_data = votes.get(source)
            if not isinstance(vote_data, dict):
                return _SOURCE_ABSENT
            try:
                return int(vote_data.get("value", 0))
            except Exception:
                return _SOURCE_ABSENT

        source_signals: List[Dict] = []
        for s in signals:
            source_vote = _get_source_vote_value(s)
            outcome = s.get("outcome")
            pnl_r = s.get("pnl_r", 0.0) or 0.0

            # FIX P4-1: source was absent from ensemble_votes — it did not
            # vote on this trade.  Skip entirely so it is not credited/blamed.
            if source_vote is _SOURCE_ABSENT:
                continue

            # Historical fallback: if no per-source attribution exists
            # (legacy row without ensemble_votes blob), preserve the
            # "ensemble outcome as proxy" behaviour.
            if source_vote is None:
                source_signals.append({
                    "outcome": outcome,
                    "pnl_r": pnl_r,
                })
                continue

            # Neutral votes should not gain or lose credit.
            if source_vote == 0:
                continue

            effective_outcome = outcome
            effective_r = pnl_r

            # Opposing a trade that loses is a correct source call; invert both
            # outcome and R multiple so per-source scoring reflects that.
            if source_vote < 0:
                if outcome == "WIN":
                    effective_outcome = "LOSS"
                elif outcome == "LOSS":
                    effective_outcome = "WIN"
                effective_r = -pnl_r

            source_signals.append({
                "outcome": effective_outcome,
                "pnl_r": effective_r,
            })

        wins = sum(1 for s in source_signals if s.get("outcome") == "WIN")
        losses = sum(1 for s in source_signals if s.get("outcome") == "LOSS")
        total = wins + losses
        if total == 0:
            return SourceScore(source=source, regime=regime, sample_count=len(source_signals))

        win_rate = wins / total

        # Average R-multiple
        r_values = [
            s.get("pnl_r", 0.0) or 0.0
            for s in source_signals
            if s.get("outcome") in ("WIN", "LOSS")
        ]
        avg_return = sum(r_values) / len(r_values) if r_values else 0.0

        # Max drawdown — previously computed as `abs(min_r)`, which only
        # captures the *worst single trade*.  FIX Q4: use the peak-to-trough
        # cumulative-R drawdown of the ordered trade sequence, which is the
        # metric a trader actually cares about.  A source that grinds out
        # +0.5R wins but occasionally lets a −1R through is very different
        # from one that strings together five −1R in a row — the cumulative
        # curve highlights the latter, the old metric treated both equally.
        if r_values:
            # Preserve the source_signals ordering (chronological per the
            # input signals list) so "cumulative" has meaning.
            running = 0.0
            peak = 0.0
            worst_drawdown = 0.0
            for r in r_values:
                running += r
                peak = max(peak, running)
                worst_drawdown = max(worst_drawdown, peak - running)
            max_drawdown = worst_drawdown
        else:
            max_drawdown = 0.0

        # Score formula: (win_rate × avg_return) / max(drawdown, floor)
        denominator = max(max_drawdown, AWC.DRAWDOWN_FLOOR)
        raw_score = (win_rate * max(avg_return, 0.01)) / denominator

        # Normalise: scale so that score=1.0 represents baseline performance
        # A system with 55% WR, 0.5R avg, 1.0R max DD → score ≈ 0.275
        # We scale so that this baseline maps to 1.0
        baseline = (0.55 * 0.5) / 1.0  # = 0.275
        normalised_score = raw_score / baseline if baseline > 0 else AWC.DEFAULT_SCORE

        # Clamp to reasonable range
        normalised_score = max(0.1, min(5.0, normalised_score))

        return SourceScore(
            source=source,
            regime=regime,
            win_rate=win_rate,
            avg_return=avg_return,
            max_drawdown=max_drawdown,
            sample_count=total,
            score=normalised_score,
        )

    # ── Public API ────────────────────────────────────────────────

    def get_weights(self, regime: str) -> Optional[Dict[str, float]]:
        """
        Return adaptive weights for the given regime, or None if
        insufficient data (caller should fall back to static weights).
        """
        snapshot = self._snapshots.get(regime)
        if not snapshot or not snapshot.sufficient_data:
            return None

        # Check staleness (2× recalc interval)
        if time.time() - snapshot.last_update > AWC.RECALC_INTERVAL_SECS * 2:
            return None

        return snapshot.weights

    def get_snapshot(self, regime: str) -> Optional[AdaptiveWeightSnapshot]:
        """Return full snapshot for diagnostics."""
        return self._snapshots.get(regime)

    def get_all_snapshots(self) -> Dict[str, AdaptiveWeightSnapshot]:
        """Return all regime snapshots for /status display."""
        return dict(self._snapshots)

    def get_diagnostics(self) -> Dict:
        """Return summary for health checks."""
        return {
            "regimes_tracked": len(self._snapshots),
            "sufficient_data_regimes": sum(
                1 for s in self._snapshots.values() if s.sufficient_data
            ),
            "last_recalc": self._last_recalc,
            "recalc_age_min": (time.time() - self._last_recalc) / 60
            if self._last_recalc > 0 else -1,
        }


# ── Module-level singleton ────────────────────────────────────────
adaptive_weight_manager = AdaptiveWeightManager()
