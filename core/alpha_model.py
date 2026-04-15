"""
TitanBot Pro — Alpha Model (Pillar 2+3: EV Engine + Strategy Weighting)
=========================================================================
Professional trading is NOT accuracy-based. It is EV-based.

This module replaces the old "confidence > threshold" gate with:
  EV = P(win) × avg_win − P(loss) × avg_loss

A trade is taken only when:
  EV > transaction_cost + risk_buffer

This alone eliminates most bad trades.

Additionally, this module manages Bayesian strategy weighting:
  - Each strategy maintains a Beta distribution of (wins, losses)
  - Strategies are weighted by posterior confidence
  - Poorly performing strategies are automatically reduced
  - The system adapts to regime changes via exponential decay

The Alpha Model is the brain that decides:
  1. Is this signal worth taking? (EV gate)
  2. How much edge does it have? (alpha score)
  3. Which strategy should we trust right now? (Bayesian weights)
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlphaScore:
    """Complete alpha assessment for a potential trade"""
    # EV calculation
    expected_value_r: float       # Expected R multiple
    ev_per_dollar: float          # EV as fraction of risk capital
    ev_positive: bool             # Passes minimum EV threshold?

    # Probability
    p_win: float
    p_loss: float

    # Alpha components
    strategy_alpha: float         # Alpha from strategy edge
    timing_alpha: float           # Alpha from timing
    regime_alpha: float           # Alpha from regime alignment
    total_alpha: float            # Combined alpha score

    # Risk-adjusted metrics
    sharpe_estimate: float        # Single-trade Sharpe estimate
    kelly_fraction: float         # Optimal fraction of capital

    # Gate decision
    should_trade: bool            # Final go/no-go
    reject_reason: Optional[str] = None

    @property
    def grade(self) -> str:
        if self.total_alpha >= 0.80:
            return "A+"
        elif self.total_alpha >= 0.65:
            return "A"
        elif self.total_alpha >= 0.50:
            return "B+"
        elif self.total_alpha >= 0.35:
            return "B"
        else:
            return "C"


@dataclass
class StrategyBayesianWeight:
    """Bayesian weight for a single strategy"""
    name: str
    alpha: float = 2.0           # Beta dist successes + prior
    beta: float = 2.0            # Beta dist failures + prior
    avg_win_r: float = 2.0       # EWMA average R on wins
    avg_loss_r: float = 1.0      # EWMA average R on losses
    trades_in_regime: int = 0    # Trades since last regime change
    last_updated: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def ev(self) -> float:
        """Expected value per trade in R"""
        return self.win_rate * self.avg_win_r - (1 - self.win_rate) * self.avg_loss_r

    @property
    def weight(self) -> float:
        """Strategy weight: EV × confidence"""
        if self.count < 5:
            return 0.5  # Not enough data, use neutral
        # Weight = clipped EV × certainty factor
        certainty = 1 - self.uncertainty
        raw_weight = max(0.1, min(2.0, 0.5 + self.ev * certainty))
        return raw_weight

    @property
    def uncertainty(self) -> float:
        """Standard deviation of win rate estimate"""
        a, b = self.alpha, self.beta
        return math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))

    @property
    def count(self) -> int:
        # Prior mass = alpha_default(2.0) + beta_default(2.0) = 4.0
        # Clamp to 0 — after aggressive decay alpha+beta can dip below 4
        # and a negative trade count is nonsensical.
        return max(0, int(self.alpha + self.beta - 4))

    def update(self, won: bool, pnl_r: float, decay: float = 0.995):
        """Update with new observation"""
        # Decay old evidence
        self.alpha = max(1.5, self.alpha * decay)
        self.beta = max(1.5, self.beta * decay)

        if won:
            self.alpha += 1
            if pnl_r > 0:
                self.avg_win_r = 0.9 * self.avg_win_r + 0.1 * pnl_r
        else:
            self.beta += 1
            if pnl_r < 0:
                self.avg_loss_r = 0.9 * self.avg_loss_r + 0.1 * abs(pnl_r)

        self.trades_in_regime += 1
        self.last_updated = time.time()


class AlphaModel:
    """
    EV-based trade evaluation and strategy weighting.
    """

    # Named constants for gate thresholds — change here, takes effect everywhere.
    _MIN_P_WIN: float = 0.40      # Minimum win-probability floor
    _MIN_ALPHA: float = 0.30      # Minimum total alpha score

    def __init__(self):
        self._strategy_weights: Dict[str, StrategyBayesianWeight] = {}

        # EV thresholds
        self._min_ev_r = 0.10          # Minimum expected R per trade
        self._transaction_cost_r = 0.02 # Transaction costs in R terms
        self._risk_buffer_r = 0.05      # Additional buffer over costs

        # Alpha weights for components
        self._strategy_alpha_weight = 0.50
        self._timing_alpha_weight = 0.25
        self._regime_alpha_weight = 0.25

    def evaluate(
        self,
        strategy: str,
        p_win: float,
        rr_ratio: float,
        regime: str,
        direction: str,
        is_killzone: bool = False,
        confluence_count: int = 1,
        regime_alignment: float = 0.5,
        chop_strength: float = 0.0,   # Gap 1: continuous chop context for regime alpha
    ) -> AlphaScore:
        """
        Full alpha evaluation of a potential trade.

        Returns AlphaScore with go/no-go decision and all metrics.
        """
        # Guard against upstream probability engine or test code passing values
        # outside [0, 1].  Clamp rather than raise so the pipeline is robust.
        p_win = max(0.0, min(1.0, float(p_win)))
        sw = self._get_weight(strategy)

        # ── 1. EV Calculation ─────────────────────────────────
        p_loss = 1.0 - p_win

        # Use strategy-specific R averages if available
        avg_win_r = min(rr_ratio, sw.avg_win_r * 1.3) if sw.count > 10 else rr_ratio
        avg_loss_r = sw.avg_loss_r if sw.count > 10 else 1.0

        expected_value_r = p_win * avg_win_r - p_loss * avg_loss_r
        ev_per_dollar = expected_value_r  # In R terms, this IS per-dollar-risked

        # EV gate: must exceed costs + buffer
        ev_threshold = self._transaction_cost_r + self._risk_buffer_r + self._min_ev_r
        ev_positive = expected_value_r > ev_threshold

        # ── 2. Alpha Components ───────────────────────────────
        # Strategy alpha: how good is this strategy's edge?
        strategy_alpha = self._calc_strategy_alpha(sw, p_win)

        # Timing alpha: are we trading at the right time?
        timing_alpha = self._calc_timing_alpha(is_killzone, confluence_count)

        # Regime alpha: does this trade fit the regime?
        regime_alpha = self._calc_regime_alpha(regime_alignment, direction, regime, chop_strength)

        total_alpha = (
            strategy_alpha * self._strategy_alpha_weight +
            timing_alpha * self._timing_alpha_weight +
            regime_alpha * self._regime_alpha_weight
        )

        # ── 3. Risk-adjusted metrics ──────────────────────────
        # Single-trade Sharpe estimate
        trade_variance = p_win * (avg_win_r ** 2) + p_loss * (avg_loss_r ** 2) - expected_value_r ** 2
        trade_std = math.sqrt(max(0.001, trade_variance))
        sharpe_estimate = expected_value_r / trade_std if trade_std > 0 else 0

        # Kelly fraction: optimal bet size
        if avg_loss_r > 0 and rr_ratio > 0:
            kelly = (p_win * rr_ratio - p_loss) / rr_ratio
            kelly = max(0, min(0.25, kelly))  # Cap at 25%
            kelly *= 0.5  # Half-Kelly for safety
        else:
            kelly = 0.01

        # ── 4. Final decision ─────────────────────────────────
        # FIX H2: p_win gate runs FIRST — a signal with p_win=0.40 can produce positive
        # EV if RR is high enough, but is fundamentally unsafe with 40% win rate.
        reject_reason = None

        # FIX P-WIN GATE: lowered from 0.45 to 0.42. The cold-start prior is 0.52
        # but with minimal evidence the Bayesian update can still produce p_win around
        # 0.43-0.45. 0.45 was a hair's-breadth gate that randomly killed borderline
        # signals based on which evidence strings happened to match. 0.42 is the
        # genuine minimum for a positive-EV trade at RR=2.0 (EV = 0.42*2 - 0.58 = 0.26R).
        if p_win < self._MIN_P_WIN:
            should_trade = False
            reject_reason = f"Win probability too low: {p_win:.3f} (need ≥{self._MIN_P_WIN})"
        elif not ev_positive:
            should_trade = False
            reject_reason = f"EV too low: {expected_value_r:.3f}R (need >{ev_threshold:.3f}R)"
        elif total_alpha < self._MIN_ALPHA:
            should_trade = False
            reject_reason = f"Alpha too low: {total_alpha:.3f} (need >{self._MIN_ALPHA})"
        elif sw.count >= 15 and sw.ev < -0.1:
            should_trade = False
            reject_reason = f"Strategy {strategy} has negative EV: {sw.ev:.3f}R over {sw.count} trades"
        else:
            should_trade = ev_positive and total_alpha >= self._MIN_ALPHA

        return AlphaScore(
            expected_value_r=round(expected_value_r, 4),
            ev_per_dollar=round(ev_per_dollar, 4),
            ev_positive=ev_positive,
            p_win=round(p_win, 4),
            p_loss=round(p_loss, 4),
            strategy_alpha=round(strategy_alpha, 4),
            timing_alpha=round(timing_alpha, 4),
            regime_alpha=round(regime_alpha, 4),
            total_alpha=round(total_alpha, 4),
            sharpe_estimate=round(sharpe_estimate, 4),
            kelly_fraction=round(kelly, 4),
            should_trade=should_trade,
            reject_reason=reject_reason,
        )

    def record_outcome(
        self,
        strategy: str,
        won: bool,
        pnl_r: float,
    ):
        """
        Record a trade outcome and update strategy weights.

        FIX #4/#5: Strategy weights including avg_win_r/avg_loss_r are persisted
        by the learning_loop._save_state() → alpha_model.get_all_weights() path.
        On load: learning_loop.load_state() → alpha_model.load_weights() restores
        alpha/beta/avg_win_r/avg_loss_r. This method only needs to update the
        in-memory StrategyBayesianWeight; persistence is handled by LearningLoop.
        """
        sw = self._get_weight(strategy)
        sw.update(won, pnl_r)
        logger.debug(
            f"Alpha model updated: {strategy} → "
            f"WR={sw.win_rate:.3f}, EV={sw.ev:.3f}R, "
            f"weight={sw.weight:.3f}, "
            f"avg_win_r={sw.avg_win_r:.2f}, avg_loss_r={sw.avg_loss_r:.2f}, "
            f"count={sw.count}"
        )


    def reset_regime(self):
        """Reset regime-specific counters (call on regime change)"""
        for sw in self._strategy_weights.values():
            sw.trades_in_regime = 0

    def get_strategy_weight(self, strategy: str) -> float:
        """Get current Bayesian weight for a strategy"""
        return self._get_weight(strategy).weight

    def get_all_weights(self) -> Dict[str, dict]:
        """Export all strategy weights"""
        return {
            name: {
                'alpha': round(sw.alpha, 4),
                'beta': round(sw.beta, 4),
                'win_rate': round(sw.win_rate, 4),
                'ev': round(sw.ev, 4),
                'weight': round(sw.weight, 4),
                'count': sw.count,
                'avg_win_r': round(sw.avg_win_r, 3),
                'avg_loss_r': round(sw.avg_loss_r, 3),
            }
            for name, sw in self._strategy_weights.items()
        }

    def load_weights(self, data: Dict[str, dict]):
        """Load strategy weights from persistence"""
        for name, vals in data.items():
            sw = StrategyBayesianWeight(name=name)
            sw.alpha = vals.get('alpha', 2.0)
            sw.beta = vals.get('beta', 2.0)
            sw.avg_win_r = vals.get('avg_win_r', 2.0)
            sw.avg_loss_r = vals.get('avg_loss_r', 1.0)
            self._strategy_weights[name] = sw
        logger.info(f"Loaded {len(data)} strategy weights")

    def bootstrap_priors(self):
        """
        PHASE 3 FIX (KELLY-COLD): Informative strategy-specific priors at cold start.

        Instead of uniform Beta(2,2) = 50% for all strategies, we use weak priors
        derived from published SMC/crypto strategy win rates. These wash out after
        ~20-30 real trades but produce meaningfully differentiated Kelly fractions
        from trade 1, so early position sizing reflects signal quality.

        Prior format: (alpha, beta) → mean = alpha/(alpha+beta)
          Beta(7,5)  → 58% — trend-following in confirmed trend
          Beta(6,5)  → 55% — momentum / breakout
          Beta(5,5)  → 50% — symmetric (no directional edge assumed)
          Beta(5,6)  → 45% — counter-trend (historically harder)
        """
        # (strategy_name, alpha, beta, avg_win_r, avg_loss_r)
        # Win rates based on conservative published crypto strategy benchmarks
        strategy_priors = [
            ('SmartMoneyConcepts',    7.0, 5.0, 2.2, 1.0),  # 58% — structure-based, strong in trend
            ('InstitutionalBreakout', 6.0, 5.0, 2.0, 1.0),  # 55% — breakouts historically decent
            ('ExtremeReversal',       5.0, 6.0, 1.8, 1.0),  # 45% — reversals are harder
            ('MeanReversion',         5.0, 6.0, 1.6, 1.0),  # 45% — fades trend, needs range
            ('PriceAction',           6.0, 5.0, 1.9, 1.0),  # 55% — pure PA in crypto is solid
            ('Momentum',              6.0, 5.0, 2.0, 1.0),  # 55% — momentum works in trends
            ('MomentumContinuation',  6.0, 5.0, 2.0, 1.0),  # Legacy alias
            ('Ichimoku',              5.0, 5.0, 1.8, 1.0),  # 50% — neutral prior
            ('IchimokuCloud',         5.0, 5.0, 1.8, 1.0),  # Legacy alias
            ('ElliottWave',           5.0, 5.0, 2.5, 1.0),  # 50% — high RR but unpredictable
            ('FundingRateArb',        5.0, 5.0, 1.5, 1.0),  # 50% — regime-dependent
            ('RangeScalper',          5.0, 6.0, 1.4, 1.0),  # 45% — small RR, range-dependent
            ('WyckoffAccDist',        6.0, 5.0, 2.1, 1.0),  # 55% — institutional context
            ('Wyckoff',               6.0, 5.0, 2.1, 1.0),  # Legacy alias
            ('HarmonicPattern',       5.0, 5.0, 2.0, 1.0),  # 50% — neutral
            ('HarmonicDetector',      5.0, 5.0, 2.0, 1.0),  # Legacy alias
            ('GeometricPattern',      5.0, 5.0, 1.8, 1.0),  # 50% — neutral
            ('GeometricPatterns',     5.0, 5.0, 1.8, 1.0),  # Legacy alias
        ]
        for name, alpha, beta, avg_win_r, avg_loss_r in strategy_priors:
            if name not in self._strategy_weights:
                sw = StrategyBayesianWeight(name=name)
                sw.alpha = alpha
                sw.beta = beta
                sw.avg_win_r = avg_win_r
                sw.avg_loss_r = avg_loss_r
                self._strategy_weights[name] = sw
        logger.info(
            f"Bootstrapped {len(self._strategy_weights)} strategy weights "
            f"(informative priors: SMC=58%, breakout=55%, reversals=45%)"
        )

    # ── Alpha calculation helpers ─────────────────────────────

    def _calc_strategy_alpha(self, sw: StrategyBayesianWeight, p_win: float) -> float:
        """
        Strategy alpha: combination of edge and reliability.
        Range: 0.0 to 1.0

        Gap 6 fix: cold-start damping instead of a hard p_win >= 0.45 anchor.
        With < 30 trades the alpha contribution is scaled toward neutral (0.5)
        so early priors don't decide trade existence — they just nudge slightly.
        """
        if sw.count < 5:
            # No data: damp toward neutral, let other alpha components carry weight
            raw = min(1.0, max(0.0, (p_win - 0.45) * 3.0))
            confidence_weight = sw.count / 5.0  # 0 trades → 0.0, 5 trades → 1.0
            return 0.5 + (raw - 0.5) * confidence_weight

        # Edge component: EV relative to transaction costs
        edge_score = min(1.0, max(0.0, sw.ev / 0.5))

        # Reliability: how tight is the win rate estimate?
        reliability = max(0.0, 1.0 - sw.uncertainty * 5)

        # Recent performance: has the strategy been hot or cold?
        recent_wr = sw.win_rate
        recent_score = min(1.0, max(0.0, (recent_wr - 0.40) * 3.0))

        raw_alpha = 0.4 * edge_score + 0.3 * reliability + 0.3 * recent_score

        # Gap 6: damp influence proportionally until 30 trades of evidence
        if sw.count < 30:
            confidence_weight = min(1.0, sw.count / 30.0)
            return 0.5 + (raw_alpha - 0.5) * confidence_weight

        return raw_alpha

    def _calc_timing_alpha(self, is_killzone: bool, confluence_count: int) -> float:
        """
        Timing alpha: are conditions favorable right now?
        Range: 0.0 to 1.0

        FIX TIMING-ALPHA: base was 0.3. Outside London/NY killzone with a single
        strategy, timing_alpha = 0.3 → dragged total_alpha to ~0.40 even with
        perfect strategy + regime alpha. Raised base to 0.45 so a single clean
        signal in a neutral time window can still clear the 0.30 threshold without
        requiring killzone timing. Killzone + confluence still give full boosts.
        """
        alpha = 0.45  # FIX: was 0.3 — too punishing for off-session valid signals

        if is_killzone:
            alpha += 0.30
        if confluence_count >= 3:
            alpha += 0.25
        elif confluence_count >= 2:
            alpha += 0.12

        return min(1.0, alpha)

    def _calc_regime_alpha(
        self, regime_alignment: float, direction: str, regime: str,
        chop_strength: float = 0.0,
    ) -> float:
        """
        Regime alpha: does this trade fit the current market condition?
        Range: 0.0 to 1.0

        Gap 1 fix: CHOPPY/VOLATILE penalty now interpolates with chop_strength
        instead of a hard 0.6× multiplier for all chop intensities.
          chop_strength=0.0 → no penalty (pure trend)
          chop_strength=0.5 → 0.80× (moderate chop)
          chop_strength=1.0 → 0.60× (pure chop, same as before at max)
        """
        base = max(0.0, min(1.0, regime_alignment))

        regime_upper = regime.upper()
        if direction == "LONG" and "BULL" in regime_upper:
            base = min(1.0, base + 0.20)
        elif direction == "SHORT" and "BEAR" in regime_upper:
            base = min(1.0, base + 0.20)
        elif "CHOPPY" in regime_upper or "VOLATILE" in regime_upper:
            # Interpolate: 1.0× at chop=0, 0.60× at chop=1
            chop_mult = 1.0 - 0.40 * chop_strength
            base *= chop_mult

        return base

    def _get_weight(self, strategy: str) -> StrategyBayesianWeight:
        if strategy not in self._strategy_weights:
            self._strategy_weights[strategy] = StrategyBayesianWeight(name=strategy)
        return self._strategy_weights[strategy]


# ── Singleton ─────────────────────────────────────────────────
alpha_model = AlphaModel()
