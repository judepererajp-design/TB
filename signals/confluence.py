"""
TitanBot Pro — Confluence Scorer
===================================
Collects signals from ALL strategies for a single symbol/scan,
then scores based on cross-strategy agreement.

When multiple strategies agree on direction → much higher confidence.
When strategies disagree → reject entirely.
When only one strategy fires → penalize (lonely signal).

This is the #2 most impactful upgrade for accuracy.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from strategies.base import SignalResult, SignalDirection
from config.loader import cfg

logger = logging.getLogger(__name__)


@dataclass
class ConfluenceResult:
    """Result of cross-strategy confluence analysis"""
    best_signal: Optional[SignalResult] = None
    direction: Optional[str] = None
    agreeing_strategies: List[str] = field(default_factory=list)
    disagreeing_strategies: List[str] = field(default_factory=list)
    confluence_multiplier: float = 1.0
    confluence_notes: List[str] = field(default_factory=list)
    should_publish: bool = False


class ConfluenceScorer:
    """
    Evaluates cross-strategy agreement for a symbol.
    Uses strategy clustering to prevent correlated signals
    from inflating confidence artificially.
    """

    # Confluence multipliers
    # FIX CONFLUENCE-MULT: was 0.85. A 74-confidence signal × 0.85 = 62.9 — below
    # the 70 min floor. Most altcoins only trigger 1 strategy at a time, so this
    # effectively meant single-strategy signals could never pass. 0.92 still penalizes
    # (vs 1.0x for multi-strategy) but doesn't kill clean single-strategy setups.
    # FIX CONFLUENCE-MULT: was 0.85 → 0.95. 0.85 needed raw conf>=83 to pass 70 floor.
    # 0.95 needs raw conf>=74, which is achievable by well-scoring single-strategy signals.
    # Multi-strategy still gets TWO_CLUSTER_MULT=1.15 so the incentive to agree remains.
    SINGLE_STRATEGY_MULT = 0.95       # Penalize lonely signals (was 0.85 — killed everything)
    TWO_CLUSTER_MULT = 1.15           # Two independent clusters agree → boost
    THREE_CLUSTER_MULT = 1.30         # Three+ clusters agree → strong boost
    DISAGREEMENT_BLOCK = True         # Block if strategies disagree

    # Strategies that provide funding/sentiment context but should NOT
    # veto a technical structure signal by themselves. If the only
    # dissenter(s) belong to this set, we go with the technical side
    # and apply a confidence penalty instead of blocking entirely.
    # Rationale: FundingRateArb detects negative funding (shorts pay longs),
    # which is useful context but shouldn't override SMC/Ichimoku/EW structure.
    WEAK_VOTERS: set = {'FundingRateArb'}

    # Strategy clusters: strategies in the same cluster use similar
    # inputs (price structure, indicators) so their agreement is NOT
    # independent evidence. Only count 1 vote per cluster.
    STRATEGY_CLUSTERS = {
        # FIX #24: Added HarmonicDetector, GeometricPatterns, WyckoffStrategy, RangeScalper.
        # Previously these were unmapped — any two of them counted as "independent" clusters
        # and triggered TWO_CLUSTER_MULT=1.15 spuriously. Harmonic and Geometric both use
        # pivot-based price structure → same cluster as SMC. Wyckoff is structural.
        # RangeScalper is volatility/range-based → same as InstitutionalBreakout.
        'STRUCTURE':  {'SmartMoneyConcepts', 'WyckoffAccDist', 'Wyckoff', 'PriceAction',
                       'HarmonicDetector', 'HarmonicPattern', 'GeometricPatterns', 'GeometricPattern'},
        'TREND':      {'Momentum', 'MomentumContinuation', 'Ichimoku', 'IchimokuCloud'},
        'MEAN':       {'MeanReversion', 'ExtremeReversal'},
        'VOLATILITY': {'InstitutionalBreakout', 'RangeScalper'},
        'SENTIMENT':  {'FundingRateArb', 'ElliottWave'},
        # A+ Upgrade: on-chain data cluster (independent from technical signals)
        'ON_CHAIN':   {'OnChainValuation', 'StablecoinFlow', 'MiningSignal',
                       'NetworkActivity', 'WhaleBehavior'},
    }

    def _get_cluster(self, strategy_name: str) -> str:
        """Get the cluster for a strategy"""
        for cluster, members in self.STRATEGY_CLUSTERS.items():
            if strategy_name in members:
                return cluster
        return strategy_name  # Unknown strategy = its own cluster

    def _count_independent_clusters(self, strategies: List[str]) -> int:
        """Count how many INDEPENDENT clusters are represented"""
        clusters = set()
        for s in strategies:
            clusters.add(self._get_cluster(s))
        return len(clusters)

    def score(self, signals: List[SignalResult], whale_dominant_side: str = "") -> ConfluenceResult:
        """
        Score a list of signals from different strategies for the same symbol.

        BUG-NEW-9 FIX (v2.10 tagged item): FundingRateArb counter-crowd bonus.
        When FundingRateArb aligns with dominant whale flow, it is promoted from
        WEAK_VOTER to a full independent STRUCTURE vote for this call only.
        This captures the "funding extreme + institutional buying" setup that the
        veto system was incorrectly suppressing.

        Args:
            signals: List of SignalResult from different strategies (same symbol)
            whale_dominant_side: "LONG", "SHORT", or "" — from whale_aggregator

        Returns:
            ConfluenceResult with the best signal adjusted for confluence
        """
        result = ConfluenceResult()

        if not signals:
            return result

        # BUG-NEW-9 FIX: FundingRateArb + whale alignment promotion.
        # If FundingRateArb fires in the SAME direction as dominant whale flow,
        # temporarily remove it from WEAK_VOTERS for this score call — it becomes
        # a full independent signal. The +8 conf bonus applied below signals the upgrade.
        _active_weak_voters = set(self.WEAK_VOTERS)
        if whale_dominant_side:
            for sig in signals:
                if sig.strategy == 'FundingRateArb' and getattr(sig.direction, 'value', str(sig.direction)) == whale_dominant_side:
                    _active_weak_voters.discard('FundingRateArb')
                    sig.confidence = min(99, sig.confidence + 8)  # reward alignment
                    result.confluence_notes.append(
                        f"🐳 FundingRateArb promoted: aligns with dominant whale flow "
                        f"({whale_dominant_side}) — treated as independent signal (+8 conf)"
                    )
                    logger.info(
                        f"FundingRateArb promoted for {signals[0].symbol}: "
                        f"direction={whale_dominant_side} matches whale flow"
                    )
                    break

        # Count directions
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        short_signals = [s for s in signals if s.direction == SignalDirection.SHORT]

        n_long = len(long_signals)
        n_short = len(short_signals)

        # ── Disagreement check ────────────────────────────────
        if n_long > 0 and n_short > 0:
            # FIX #21: Use majority direction instead of blocking entirely.
            # If 3 strategies say SHORT and 1 says LONG, go with SHORT.
            # Only block if it's a true tie (equal count).
            if n_long == n_short:
                # FIX #22: Weak-voter tie resolution.
                # FundingRateArb fires on funding rate alone — it's sentiment context,
                # not structure. If the ONLY dissenter(s) on one side are weak voters,
                # don't deadlock: go with the technical side, apply -8 conf penalty.
                # Catches: SMC(SHORT) + FundingRateArb(LONG) → go SHORT -8 conf
                #          IchimokuCloud(SHORT) + FundingRateArb(LONG) → go SHORT -8 conf
                short_names = {s.strategy for s in short_signals}
                long_names  = {s.strategy for s in long_signals}

                if long_names.issubset(self.WEAK_VOTERS) and long_names:
                    # Long side is funding/sentiment only → go with technical SHORT
                    result.disagreeing_strategies = [s.strategy for s in long_signals]
                    result.confluence_notes.append(
                        "⚠️ Funding dissent (LONG) vs technical structure (SHORT)"
                        " — favouring structure, -8 conf"
                    )
                    best_short = max(short_signals, key=lambda s: s.confidence)
                    best_short.confidence = max(best_short.confidence - 8, 0)
                    signals = short_signals
                    n_long, n_short = 0, len(signals)
                    logger.info(
                        f"Confluence tie → SHORT kept ({signals[0].symbol}): "
                        f"FundingRateArb LONG dissent is weak-voter only, -8 conf"
                    )

                elif short_names.issubset(self.WEAK_VOTERS) and short_names:
                    # Short side is funding/sentiment only → go with technical LONG
                    result.disagreeing_strategies = [s.strategy for s in short_signals]
                    result.confluence_notes.append(
                        "⚠️ Funding dissent (SHORT) vs technical structure (LONG)"
                        " — favouring structure, -8 conf"
                    )
                    best_long = max(long_signals, key=lambda s: s.confidence)
                    best_long.confidence = max(best_long.confidence - 8, 0)
                    signals = long_signals
                    n_long, n_short = len(signals), 0
                    logger.info(
                        f"Confluence tie → LONG kept ({signals[0].symbol}): "
                        f"FundingRateArb SHORT dissent is weak-voter only, -8 conf"
                    )

                else:
                    # True tie between real technical strategies → block
                    result.disagreeing_strategies = (
                        [s.strategy for s in long_signals] +
                        [s.strategy for s in short_signals]
                    )
                    result.confluence_notes.append(
                        f"⚠️ Strategy tie: {n_long} LONG vs {n_short} SHORT — signal blocked"
                    )
                    result.should_publish = False
                    logger.debug(
                        f"Confluence blocked {signals[0].symbol}: "
                        f"{n_long} LONG vs {n_short} SHORT strategies (tie)"
                    )
                    return result
            else:
                # Go with majority — penalize confidence for disagreement
                majority_long = n_long > n_short
                if majority_long:
                    # Remove minority SHORT signals
                    result.disagreeing_strategies = [s.strategy for s in short_signals]
                    signals = long_signals  # Continue with LONG only
                    n_long = len(signals)
                    n_short = 0
                else:
                    result.disagreeing_strategies = [s.strategy for s in long_signals]
                    signals = short_signals
                    n_short = len(signals)
                    n_long = 0
                result.confluence_notes.append(
                    f"⚠️ {len(result.disagreeing_strategies)} dissenting strategies — "
                    f"going with majority (confidence penalized)"
                )

        # ── Agreement scoring ─────────────────────────────────
        # All signals agree on direction
        # FIX: was using the ORIGINAL long_signals/short_signals lists here even
        # after `signals` was reassigned above (majority path / weak-voter resolution).
        # After reassignment, `signals` IS the direction-filtered list. Use it directly.
        direction_signals = [s for s in signals if (
            s.direction == SignalDirection.LONG if n_long > 0 else s.direction == SignalDirection.SHORT
        )]
        direction = "LONG" if n_long > 0 else "SHORT"
        n_agreeing = len(direction_signals)

        if not direction_signals:
            return result  # No agreeing signals after filtering

        # Pick the best signal (highest confidence)
        best = max(direction_signals, key=lambda s: s.confidence)
        result.best_signal = best
        result.direction = direction
        result.agreeing_strategies = [s.strategy for s in direction_signals]

        # Apply confluence multiplier based on INDEPENDENT clusters
        # (not raw strategy count — prevents correlated inflation)
        n_clusters = self._count_independent_clusters(result.agreeing_strategies)

        if n_clusters >= 3:
            result.confluence_multiplier = self.THREE_CLUSTER_MULT
            result.confluence_notes.append(
                f"🔥 {n_agreeing} strategies ({n_clusters} independent clusters) "
                f"agree on {direction}: {', '.join(result.agreeing_strategies)}"
            )
        elif n_clusters >= 2:
            result.confluence_multiplier = self.TWO_CLUSTER_MULT
            result.confluence_notes.append(
                f"✅ {n_agreeing} strategies ({n_clusters} clusters) confirm {direction}: "
                f"{', '.join(result.agreeing_strategies)}"
            )
        else:
            # All agreeing strategies are from the same cluster.
            # V17 FIX: 3 intra-cluster strategies agreeing is still mildly
            # positive — just not as strong as cross-cluster. 1 strategy = 0.95x,
            # 2+ intra-cluster = 1.02x (slight boost, not penalty).
            if n_agreeing >= 2:
                result.confluence_multiplier = 1.02
            else:
                result.confluence_multiplier = self.SINGLE_STRATEGY_MULT
            cluster_name = self._get_cluster(result.agreeing_strategies[0])
            if n_agreeing > 1:
                result.confluence_notes.append(
                    f"📊 {n_agreeing} strategies agree but same cluster ({cluster_name}) "
                    f"— treated as single signal"
                )
            else:
                result.confluence_notes.append(
                    f"📊 Single strategy signal ({best.strategy})"
                )

        # Adjust confidence
        original_conf = best.confidence
        best.confidence = min(99, best.confidence * result.confluence_multiplier)
        # Store multiplier on signal so aggregator can copy to ScoredSignal
        best.confluence_multiplier = result.confluence_multiplier

        # Merge confluence notes from all agreeing strategies
        all_confluence = []
        seen = set()
        for sig in direction_signals:
            for note in sig.confluence:
                if note not in seen:
                    all_confluence.append(note)
                    seen.add(note)
        best.confluence = result.confluence_notes + all_confluence

        result.should_publish = True

        logger.debug(
            f"Confluence scored {best.symbol} {direction}: "
            f"{n_agreeing} strategies, mult={result.confluence_multiplier:.2f}, "
            f"conf {original_conf:.0f}→{best.confidence:.0f}"
        )

        return result


# ── Singleton ──────────────────────────────────────────────
confluence_scorer = ConfluenceScorer()
