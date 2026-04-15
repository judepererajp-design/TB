"""
TitanBot Pro — Probability Engine (Pillar 1: Probabilistic Core)
==================================================================
THE defining shift from trading bot to quant system.

Instead of asking "Is this a trade?" the bot now asks
"What is the probability of profit, and how confident am I?"

Every strategy becomes a Bayesian evidence source:
  P(profit | evidence) ∝ P(evidence | profit) × P(profit)

Evidence sources include:
  - Strategy signal (breakout, SMC, reversal, etc.)
  - Market regime (bull, bear, choppy, volatile)
  - Time of day / session
  - Volume context
  - Derivatives data (funding, OI)
  - Recent win rate of that strategy
  - Confluence count

Architecture:
  1. Prior: base rate for this strategy + regime combo
  2. Likelihood: how well does this evidence predict profit?
  3. Posterior: final probability after combining all evidence
  4. Calibration: compare predicted P(win) vs actual outcomes

This replaces the old deterministic confidence scoring with
a mathematically grounded probability estimate.
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_BETA_PRIOR_ALPHA = 2.6
DEFAULT_BETA_PRIOR_BETA = 2.4
DEFAULT_BETA_PRIOR_MASS = DEFAULT_BETA_PRIOR_ALPHA + DEFAULT_BETA_PRIOR_BETA


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class BetaPosterior:
    """
    Beta distribution posterior for a Bernoulli process.
    Models the probability of winning for a given context.

    alpha = successes + 1 (prior)
    beta  = failures + 1 (prior)
    """
    alpha: float = DEFAULT_BETA_PRIOR_ALPHA    # FIX PRIOR: was 1.8/2.2=0.45, now 2.6/2.4=0.52. The p_win gate
    beta: float = DEFAULT_BETA_PRIOR_BETA      # is 0.45, so a prior OF 0.45 means any absent-evidence penalty
    # immediately kills the signal. 0.52 gives the Bayesian update room to work.

    @property
    def mean(self) -> float:
        """Expected win probability"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Uncertainty in the estimate"""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def count(self) -> int:
        """Total observations (minus prior).
        BUG-13 FIX: Prior counts must track the current default prior mass.
        Hardcoding the subtraction makes this brittle if defaults ever change.
        Subtracting 4 (hardcoded) left a gap of 1.0 on a fresh posterior, and any
        posterior with alpha+beta below the prior mass (e.g. after aggressive decay) returned a
        negative count. Clamp to zero — a negative trade count is nonsensical and
        breaks calibration logic that divides by count.
        """
        return max(0, int(self.alpha + self.beta - DEFAULT_BETA_PRIOR_MASS))

    def update(self, won: bool):
        """Update posterior with new observation"""
        if won:
            self.alpha += 1
        else:
            self.beta += 1

    def update_with_decay(self, won: bool, decay: float = 0.995):
        """
        Update with exponential decay on old observations.
        This makes the system adapt faster to regime changes.
        """
        # Clamp decay to a valid range: 0 would zero-out all history;
        # > 1 would grow posteriors unboundedly.
        decay = max(0.900, min(0.999, decay))
        # Decay existing evidence (shrink toward prior)
        self.alpha = max(1.5, self.alpha * decay)
        self.beta = max(1.5, self.beta * decay)
        # Add new evidence
        if won:
            self.alpha += 1
        else:
            self.beta += 1

    @property
    def confidence_interval_95(self) -> Tuple[float, float]:
        """95% credible interval using normal approximation"""
        m = self.mean
        s = self.std
        return (max(0, m - 1.96 * s), min(1, m + 1.96 * s))

    def to_dict(self) -> dict:
        return {
            'alpha': round(self.alpha, 3),
            'beta': round(self.beta, 3),
            'mean': round(self.mean, 4),
            'std': round(self.std, 4),
            'count': self.count,
        }


@dataclass
class ProbabilityEstimate:
    """Complete probability estimate for a signal"""
    # Core probabilities
    p_win: float                # P(win | all evidence)
    p_loss: float               # P(loss | all evidence)
    uncertainty: float          # Width of credible interval

    # EV components
    expected_r: float           # Expected R multiple
    ev_positive: bool           # Is EV > threshold?

    # Evidence breakdown
    prior: float                # Base rate before evidence
    evidence_sources: Dict[str, float] = field(default_factory=dict)

    # Quality metrics
    sample_size: int = 0        # How many observations back this estimate
    calibration_error: float = 0.0  # How well-calibrated is this estimate

    @property
    def edge(self) -> float:
        """Trading edge = P(win) × avg_win_R - P(loss) × avg_loss_R"""
        return self.expected_r

    @property
    def confidence_grade(self) -> str:
        """Convert probability into A+/A/B/C grade"""
        if self.p_win >= 0.72 and self.uncertainty < 0.12 and self.ev_positive:
            return "A+"
        elif self.p_win >= 0.65 and self.uncertainty < 0.15 and self.ev_positive:
            return "A"
        elif self.p_win >= 0.58 and self.ev_positive:
            return "B+"
        elif self.p_win >= 0.52 and self.ev_positive:
            return "B"
        else:
            return "C"


# ── Probability Engine ────────────────────────────────────────

class ProbabilityEngine:
    """
    Bayesian probability engine.

    Maintains Beta-distribution posteriors for every
    (strategy, regime, direction) combination.

    When a signal arrives, it:
    1. Looks up the prior for this context
    2. Combines evidence from multiple sources
    3. Calculates posterior P(win)
    4. Estimates expected R value
    5. Returns a full ProbabilityEstimate
    """

    def __init__(self):
        # Posteriors indexed by context key
        # Key format: "strategy:regime:direction"
        self._posteriors: Dict[str, BetaPosterior] = {}

        # Evidence likelihood ratios (learned from data)
        # These represent how much each evidence type shifts the posterior
        self._likelihood_ratios: Dict[str, float] = {
            # Confluence evidence (log-odds adjustments)
            'htf_alignment': 0.15,       # HTF structure aligns
            'mtf_alignment': 0.10,       # MTF structure aligns
            'ob_fvg_confluence': 0.20,   # OB + FVG overlap
            'liquidity_sweep': 0.18,     # Recent sweep
            'liq_cluster_overlap': 0.22, # OB on liquidation cluster
            'volume_confirmation': 0.12, # Volume supports direction
            'funding_favorable': 0.08,   # Funding rate favorable
            'killzone_active': 0.10,     # In active killzone session
            'premium_discount_zone': 0.12, # Trading from right zone
            'multi_strategy_agree': 0.25,  # Multiple strategies agree
            'bar_confirmation': 0.08,    # Price action confirmed
            'vwap_alignment': 0.06,      # VWAP supports direction

            # Negative evidence
            'counter_regime': -0.15,     # Signal opposes regime
            'low_volume': -0.10,         # Low volume environment
            'dead_zone_time': -0.12,     # Trading in dead zone
            'weekend': -0.08,            # Weekend trading
            'high_correlation': -0.06,   # High BTC correlation (alt signal)
            'strategy_cold_streak': -0.15,  # Strategy on losing streak
        }
        # Baseline snapshot taken at init — used by the soft-restoration step in
        # _auto_tune_from_calibration() to prevent LRs from drifting to zero.
        # M2-FIX: without this, repeated calibration corrections compound
        # indefinitely toward the 0.01 floor, making all evidence meaningless.
        self._likelihood_ratios_baseline: Dict[str, float] = dict(self._likelihood_ratios)

        # Average R multiples by outcome (updated by learning loop)
        self._avg_win_r: float = 1.8     # Conservative prior — 2.2 was optimistic; real fills rarely capture full TP2
        self._avg_loss_r: float = 1.0    # Average R on losses (should be ~1.0 by design)

        # Calibration tracking
        self._calibration_bins: Dict[str, Dict] = {}  # bin -> {predicted, actual, count}

        # Decay factor for posterior updates
        self._decay = 0.995  # Slow decay to adapt to regime changes

        # Config-driven default Beta prior — read once at init so hot paths don't
        # call into the config layer on every new-key posterior creation.
        try:
            from config.loader import cfg as _cfg
            _pe_cfg = getattr(_cfg, 'probability_engine', None) or {}
            _pe_get = _pe_cfg.get if hasattr(_pe_cfg, 'get') else (lambda k, d: getattr(_pe_cfg, k, d))
            self._default_prior_alpha: float = float(_pe_get('default_prior_alpha', 2.6))
            self._default_prior_beta:  float = float(_pe_get('default_prior_beta',  2.4))
        except Exception:
            self._default_prior_alpha = 2.6
            self._default_prior_beta  = 2.4

    def estimate(
        self,
        strategy: str,
        regime: str,
        direction: str,
        confidence: float,
        rr_ratio: float,
        evidence: Dict[str, bool],
        confluence_count: int = 1,
    ) -> ProbabilityEstimate:
        """
        Generate a full probability estimate for a signal.

        Args:
            strategy: Strategy name (e.g., "SmartMoneyConcepts")
            regime: Current regime (e.g., "BULL_TREND")
            direction: "LONG" or "SHORT"
            confidence: Raw confidence score from strategy
            rr_ratio: Risk-reward ratio
            evidence: Dict of evidence flags (True/False)
            confluence_count: Number of agreeing strategies

        Returns:
            ProbabilityEstimate with all probability and EV data
        """
        # ── 1. Get prior from Beta posterior ────────────────────
        # Validate inputs before any computation to guard against upstream bugs
        confidence = max(0.0, min(100.0, float(confidence)))
        rr_ratio = max(0.0, float(rr_ratio))  # negative RR is physically meaningless
        context_key = self._context_key(strategy, regime, direction)
        posterior = self._get_posterior(context_key)
        prior = posterior.mean

        # ── 2. Combine evidence using log-odds ──────────────────
        # Convert prior to log-odds
        if math.isnan(prior) or not (0 < prior < 1):
            log_odds = 0.0
        else:
            log_odds = math.log(prior / (1 - prior))

        evidence_details = {}

        for evidence_name, is_present in evidence.items():
            if evidence_name in self._likelihood_ratios:
                lr = self._likelihood_ratios[evidence_name]
                if is_present:
                    # Evidence IS present: apply its likelihood ratio (positive or negative)
                    # e.g. htf_alignment=True → +0.15, counter_regime=True → -0.15
                    log_odds += lr
                    evidence_details[evidence_name] = lr
                # When is_present=False: NEUTRAL. Do nothing.
                # Absence of evidence is not evidence of absence.
                # The old code penalized when is_present=False AND lr<0, which meant
                # "counter_regime is absent" (a GOOD thing) was treated as a PENALTY.
                # This cascaded ~6 negative-LR absences into -0.66 total, killing p_win.

        # Confluence bonus (scales with number of agreeing strategies)
        if confluence_count >= 3:
            conf_bonus = 0.30
        elif confluence_count >= 2:
            conf_bonus = 0.15
        else:
            conf_bonus = 0.0
        log_odds += conf_bonus
        if conf_bonus > 0:
            evidence_details['confluence_bonus'] = conf_bonus

        # Confidence-based adjustment (map 0-100 → log-odds shift)
        # High confidence = slight positive evidence
        conf_shift = (confidence - 70) * 0.005  # e.g., conf=85 → +0.075
        log_odds += conf_shift
        evidence_details['confidence_shift'] = round(conf_shift, 4)

        # ── 3. Convert back to probability ──────────────────────
        # Guard against NaN propagation from upstream computation
        if math.isnan(log_odds) or math.isinf(log_odds):
            log_odds = 0.0
        p_win = 1.0 / (1.0 + math.exp(-log_odds))

        # R6-F1: Calibration auto-correction.  _calibration_bins stores
        # the signed error (predicted - actual) per bucket.  If the
        # 0.6 bucket says predicted=0.62, actual=0.54, error=0.08, then
        # we're systematically over-predicting in that range.  Shift
        # p_win toward actual reality instead of ignoring the measurement.
        # Uses 50% correction factor — conservative enough to avoid
        # over-shooting while still closing the calibration gap.
        _cal_bin_key = str(round(p_win, 1))
        if _cal_bin_key in self._calibration_bins:
            _cal_data = self._calibration_bins[_cal_bin_key]
            _cal_count = _cal_data.get('count', 0)
            if _cal_count >= 10:  # only correct with enough observations
                _predicted = _cal_data.get('predicted', p_win)
                _actual = _cal_data.get('actual', p_win)
                # Confidence weighting: Bayesian shrinkage estimator.
                # weight = count / (count + k), k=50. At 10 obs → 0.17, at 50 → 0.50,
                # at 200 → 0.80. Unlike linear count/100, this never reaches 1.0,
                # preserving a small uncertainty margin at any sample size and preventing
                # overconfidence at moderate counts (e.g. count=100 → linear gives 1.0,
                # shrinkage gives 0.67 — meaningfully different in a live system).
                _weight = _cal_count / (_cal_count + 50)
                _correction = (_actual - _predicted) * 0.5 * _weight
                p_win += _correction
                if abs(_correction) > 0.01:
                    evidence_details['calibration_correction'] = round(_correction, 4)

        p_win = max(0.05, min(0.95, p_win))  # Clip extremes

        # R6-F3: Log when clipping activates — silent bounds hide calibration issues.
        _pre_clip = 1.0 / (1.0 + math.exp(-log_odds))
        if _pre_clip < 0.06 or _pre_clip > 0.94:
            logger.debug(
                f"P_win clipped: raw={_pre_clip:.3f} → {p_win:.3f} "
                f"({strategy}/{regime}/{direction})"
            )

        p_loss = 1.0 - p_win

        # ── 4. Calculate expected R ─────────────────────────────
        # Use actual RR ratio for win estimate, 1.0R for loss
        win_r = min(rr_ratio, self._avg_win_r * 1.5)  # Cap optimistic RR
        loss_r = self._avg_loss_r

        expected_r = p_win * win_r - p_loss * loss_r

        # Is EV positive enough to trade? (account for costs)
        min_ev_threshold = 0.10  # Minimum 0.10R expected value
        ev_positive = expected_r > min_ev_threshold

        # ── 5. Estimate uncertainty ─────────────────────────────
        ci_low, ci_high = posterior.confidence_interval_95
        uncertainty = ci_high - ci_low

        return ProbabilityEstimate(
            p_win=round(p_win, 4),
            p_loss=round(p_loss, 4),
            uncertainty=round(uncertainty, 4),
            expected_r=round(expected_r, 4),
            ev_positive=ev_positive,
            prior=round(prior, 4),
            evidence_sources=evidence_details,
            sample_size=posterior.count,
            calibration_error=self._get_calibration_error(p_win),
        )

    def update(
        self,
        strategy: str,
        regime: str,
        direction: str,
        won: bool,
        pnl_r: float = 0.0,
        decay: Optional[float] = None,
    ):
        """
        Update posterior with trade outcome.
        Called by the learning loop after each trade resolves.

        M3-FIX: Accept an explicit per-call ``decay`` to prevent the shared
        ``self._decay`` instance variable from leaking between concurrent
        updates.  Callers that already set the rate via set_decay() are
        unaffected (decay=None falls back to self._decay).
        """
        context_key = self._context_key(strategy, regime, direction)
        posterior = self._get_posterior(context_key)
        _effective_decay = decay if decay is not None else self._decay
        posterior.update_with_decay(won, _effective_decay)

        # Update global average R values
        if won and pnl_r > 0:
            self._avg_win_r = 0.95 * self._avg_win_r + 0.05 * pnl_r
        elif not won and pnl_r < 0:
            self._avg_loss_r = 0.95 * self._avg_loss_r + 0.05 * abs(pnl_r)

        # Update calibration
        # (deferred to calibrate() method for batch processing)

        logger.debug(
            f"Posterior updated: {context_key} → "
            f"P(win)={posterior.mean:.3f} "
            f"(α={posterior.alpha:.1f}, β={posterior.beta:.1f})"
        )

    def get_strategy_prior(self, strategy: str, regime: str, direction: str) -> float:
        """Get current prior probability for a context"""
        key = self._context_key(strategy, regime, direction)
        return self._get_posterior(key).mean

    def get_all_posteriors(self) -> Dict[str, dict]:
        """Export all posteriors for persistence"""
        return {k: v.to_dict() for k, v in self._posteriors.items()}

    def get_full_state(self) -> Dict:
        """Export full learned state for persistence (posteriors + LRs + calibration).

        Phase 9-12 audit fix: previously only posteriors were saved, so
        likelihood_ratios, avg_win/loss_r, and calibration_bins were lost
        on restart — resetting the tuning memory each cycle.
        """
        return {
            'posteriors': self.get_all_posteriors(),
            'likelihood_ratios': dict(self._likelihood_ratios),
            'avg_win_r': self._avg_win_r,
            'avg_loss_r': self._avg_loss_r,
            'calibration_bins': {k: dict(v) for k, v in self._calibration_bins.items()},
        }

    def load_full_state(self, data: Dict):
        """Restore full learned state from persistence.

        Backwards-compatible: if data only contains posteriors keys
        (alpha/beta dicts) it falls through to load_posteriors().
        """
        if 'posteriors' in data:
            self.load_posteriors(data['posteriors'])
            # Restore likelihood ratios (keep baseline intact)
            if 'likelihood_ratios' in data:
                for k, v in data['likelihood_ratios'].items():
                    if k in self._likelihood_ratios:
                        self._likelihood_ratios[k] = float(v)
                logger.info(f"Restored {len(data['likelihood_ratios'])} likelihood ratios")
            if 'avg_win_r' in data:
                self._avg_win_r = float(data['avg_win_r'])
            if 'avg_loss_r' in data:
                self._avg_loss_r = float(data['avg_loss_r'])
            if 'calibration_bins' in data:
                self._calibration_bins = {
                    k: dict(v) for k, v in data['calibration_bins'].items()
                }
                logger.info(f"Restored {len(self._calibration_bins)} calibration bins")
        else:
            # Legacy format: data IS the posteriors dict directly
            self.load_posteriors(data)

    def load_posteriors(self, data: Dict[str, dict]):
        """Load posteriors from persistence"""
        for key, vals in data.items():
            self._posteriors[key] = BetaPosterior(
                alpha=vals.get('alpha', 2.0),
                beta=vals.get('beta', 2.0),
            )
        logger.info(f"Loaded {len(data)} posteriors from database")

    def bootstrap_priors(self):
        """
        Set informative priors for cold start.
        FIX #2: Uses REAL strategy class names and FULL regime names to match
        _context_key() format: "StrategyClassName:REGIME_NAME:DIRECTION"
        Previously used short names ('smc') and abbreviated regimes ('BULL')
        which never matched any real context key — cold start posteriors were
        created for keys that no signal ever queries. Every real signal started
        at uninformative Beta(2.6, 2.4) instead of the intended priors.

        Informative priors based on conservative published crypto win rates:
          SMC/PA/Wyckoff (structure): ~58% in trend, ~50% in chop
          Breakout/Momentum:          ~55% in trend, ~45% in chop
          Reversal/MeanReversion:     ~48% in trend, ~55% in chop
          FundingArb:                 ~52% all regimes (carry-based)
        """
        # Real strategy class names as used by engine._strat_key_map
        strategy_priors = {
            # (alpha, beta) → mean = alpha/(alpha+beta)
            'SmartMoneyConcepts':    (7.0, 5.0),   # 58% — structure-based
            'InstitutionalBreakout': (6.0, 5.0),   # 55% — breakouts
            'ExtremeReversal':       (5.0, 6.0),   # 45% — reversals harder
            'MeanReversion':         (5.0, 6.0),   # 45% — needs range
            'PriceAction':           (6.0, 5.0),   # 55% — PA solid
            'Momentum':              (6.0, 5.0),   # 55% — momentum in trends
            'MomentumContinuation':  (6.0, 5.0),   # Legacy alias
            'Ichimoku':              (5.0, 5.0),   # 50% — neutral prior
            'IchimokuCloud':         (5.0, 5.0),   # Legacy alias
            'ElliottWave':           (5.0, 5.0),   # 50% — high RR, uncertain
            'FundingRateArb':        (5.0, 5.0),   # 50% — regime-dependent
            'RangeScalper':          (5.0, 6.0),   # 45% — range-dependent
            'WyckoffAccDist':        (6.0, 5.0),   # 55% — institutional context
            'Wyckoff':               (6.0, 5.0),   # Legacy alias
            'HarmonicPattern':       (5.0, 5.0),   # 50% — neutral
            'HarmonicDetector':      (5.0, 5.0),   # Legacy alias
            'GeometricPattern':      (5.0, 5.0),   # 50% — neutral
            'GeometricPatterns':     (5.0, 5.0),   # Legacy alias
        }
        # Full regime names exactly as Regime enum values
        # BUG-NEW-3 FIX: was ['BULL_TREND','BEAR_TREND','CHOPPY','VOLATILE','NEUTRAL','VOLATILE']
        # — VOLATILE appeared twice (second overwrote first priors), NEUTRAL was correct but
        # sandwiched between two VOLATILEs. Fixed to a deduplicated, ordered list.
        regimes = ['BULL_TREND', 'BEAR_TREND', 'CHOPPY', 'VOLATILE', 'VOLATILE_PANIC', 'NEUTRAL']
        directions = ['LONG', 'SHORT']

        bootstrapped = 0
        for strat, (alpha, beta) in strategy_priors.items():
            for regime in regimes:
                for direction in directions:
                    # Adjust prior slightly for regime alignment
                    a, b = alpha, beta
                    if (direction == 'LONG' and 'BULL' in regime) or \
                       (direction == 'SHORT' and 'BEAR' in regime):
                        a += 0.5  # aligned with trend → slightly more optimistic
                    elif regime == 'CHOPPY':
                        # Trend strategies weaker in chop
                        if strat in ('InstitutionalBreakout', 'Momentum', 'MomentumContinuation', 'Ichimoku', 'IchimokuCloud'):
                            b += 0.5
                        # Mean reversion better in chop
                        elif strat in ('MeanReversion', 'ExtremeReversal', 'RangeScalper'):
                            a += 0.3

                    key = f"{strat}:{regime}:{direction}"
                    if key not in self._posteriors:
                        self._posteriors[key] = BetaPosterior(alpha=a, beta=b)
                        bootstrapped += 1

        logger.info(
            f"Bootstrapped {bootstrapped} probability priors "
            f"({len(strategy_priors)} strategies × {len(regimes)} regimes × 2 directions, "
            f"strategy-class names, full regime names)"
        )

    def calibrate(self, predictions: List[Tuple[float, bool]]):
        """
        Measure calibration: do our predicted probabilities match reality?

        M1-FIX: Bins are now merged with existing history using weighted
        running averages instead of replaced wholesale.  Replacing discarded
        all previously accumulated data (e.g. 500 trades replaced by 30),
        making calibration noisy and the 10-observation auto-correction gate
        almost always miss.  Cumulative merging gives the correction step
        stable, high-count observations to work with.

        Args:
            predictions: List of (predicted_p_win, actually_won) tuples
        """
        if len(predictions) < 20:
            return

        # Bin predictions into 10 buckets
        bins = {}
        for p_win, won in predictions:
            bin_key = round(p_win, 1)  # 0.0, 0.1, ..., 0.9, 1.0
            if bin_key not in bins:
                bins[bin_key] = {'predicted_sum': 0, 'actual_wins': 0, 'count': 0}
            bins[bin_key]['predicted_sum'] += p_win
            bins[bin_key]['actual_wins'] += int(won)
            bins[bin_key]['count'] += 1

        # M1-FIX: merge new batch into existing bins (weighted running average)
        # instead of replacing, so calibration history accumulates.
        total_error = 0
        total_count = 0
        for bin_key, data in bins.items():
            if data['count'] < 5:
                continue
            predicted_avg = data['predicted_sum'] / data['count']
            actual_rate = data['actual_wins'] / data['count']
            new_error = abs(predicted_avg - actual_rate)
            str_key = str(bin_key)

            if str_key in self._calibration_bins:
                old = self._calibration_bins[str_key]
                old_count = old['count']
                new_count = data['count']
                merged_count = old_count + new_count
                # Weighted average of predicted and actual rates
                merged_predicted = (old['predicted'] * old_count + predicted_avg * new_count) / merged_count
                merged_actual = (old['actual'] * old_count + actual_rate * new_count) / merged_count
                merged_error = abs(merged_predicted - merged_actual)
                self._calibration_bins[str_key] = {
                    'predicted': round(merged_predicted, 3),
                    'actual': round(merged_actual, 3),
                    'error': round(merged_error, 3),
                    'count': merged_count,
                }
                total_error += merged_error * merged_count
                total_count += merged_count
            else:
                self._calibration_bins[str_key] = {
                    'predicted': round(predicted_avg, 3),
                    'actual': round(actual_rate, 3),
                    'error': round(new_error, 3),
                    'count': data['count'],
                }
                total_error += new_error * data['count']
                total_count += data['count']

        avg_error = total_error / total_count if total_count > 0 else 0
        logger.info(f"Calibration: avg error = {avg_error:.3f} across {total_count} cumulative predictions")

        # R6-F1: Alert on poor calibration — don't just measure, surface the problem.
        if avg_error > 0.10 and total_count >= 30:
            logger.warning(
                f"⚠️  CALIBRATION WARNING: avg error {avg_error:.3f} exceeds 0.10 threshold "
                f"across {total_count} predictions. P_win estimates may be unreliable. "
                f"Auto-correction is active but manual review recommended."
            )
            # Log worst-calibrated bins for diagnostic detail
            for bk, bd in sorted(self._calibration_bins.items(),
                                 key=lambda x: x[1].get('error', 0), reverse=True)[:3]:
                logger.warning(
                    f"  Bin {bk}: predicted={bd['predicted']:.3f} "
                    f"actual={bd['actual']:.3f} error={bd['error']:.3f} "
                    f"(n={bd['count']})"
                )

        # R8-F10: Calibration-Driven Auto-Tuning — when calibration error
        # for a bin exceeds 10% with 50+ observations, automatically adjust
        # evidence likelihood ratios that contribute to that bin.
        self._auto_tune_from_calibration(predictions)

    def _auto_tune_from_calibration(self,
                                     predictions: List[Tuple[float, bool]]):
        """
        R8-F10: Auto-tune evidence likelihood ratios based on calibration errors.

        C1-FIX: The previous implementation looped over every qualifying bin and
        mutated ALL likelihood ratios for each bin independently.  When two bins
        had opposing calibration errors (one over-confident, one under-confident)
        their corrections partially cancelled — but also corrupted well-calibrated
        bins by dragging their LRs in directions unrelated to those bins.  This
        is a violation of statistical independence: local (per-bin) errors were
        applied to globally shared parameters.

        Correct approach: aggregate a single weighted composite bias signal
        across all qualifying bins (weight = count × error), then apply one
        correction pass.  This ensures:
          • each LR is adjusted at most once per calibration cycle
          • the direction reflects the net system-wide miscalibration, not
            noise from any individual bin
          • corrections from opposing bins are properly averaged out

        M2-FIX: After applying the composite correction, softly restore all LRs
        2% toward their baseline values.  This prevents unbounded decay toward
        the 0.01 floor when miscalibration is transient (e.g. a losing streak
        in a single unusual regime).  If calibration error is genuinely systematic
        the correction factor exceeds the 2% restoration and LRs still drift.
        """
        if not self._calibration_bins:
            return

        # ── Step 1: compute net weighted bias ────────────────────────────────
        # Positive net_bias = system over-predicts wins (need to dampen positive LRs).
        # Negative net_bias = system under-predicts wins (need to dampen negative LRs).
        total_weight = 0.0
        weighted_bias = 0.0

        for bin_data in self._calibration_bins.values():
            error = bin_data.get('error', 0)
            count = bin_data.get('count', 0)
            if error < 0.10 or count < 50:
                continue
            predicted = bin_data.get('predicted', 0.5)
            actual = bin_data.get('actual', 0.5)
            # predicted - actual > 0 → over-confident in this bin
            w = count * error
            weighted_bias += (predicted - actual) * w
            total_weight += w

        if total_weight == 0:
            # No qualifying bins — still apply soft restoration so LRs drift back
            for ev_name, baseline_lr in self._likelihood_ratios_baseline.items():
                current = self._likelihood_ratios.get(ev_name, baseline_lr)
                self._likelihood_ratios[ev_name] = current + 0.02 * (baseline_lr - current)
            return

        net_bias = weighted_bias / total_weight   # signed composite error
        correction_factor = abs(net_bias) * 0.3   # 30% of the composite error

        # ── Step 2: single correction pass ───────────────────────────────────
        tuned = 0
        for ev_name, lr in list(self._likelihood_ratios.items()):
            if net_bias > 0 and lr > 0:
                # Net over-confident → dampen positive LRs
                new_lr = lr * (1.0 - correction_factor)
                self._likelihood_ratios[ev_name] = max(0.01, new_lr)
                tuned += 1
            elif net_bias < 0 and lr < 0:
                # Net under-confident → dampen negative LRs (make them less negative)
                new_lr = lr * (1.0 - correction_factor)
                self._likelihood_ratios[ev_name] = min(-0.01, new_lr)
                tuned += 1

        # ── Step 3: soft restoration toward baseline (M2-FIX) ────────────────
        # 2% nudge per calibration cycle prevents indefinite LR decay when the
        # composite correction factor is smaller than the restoration rate.
        for ev_name, baseline_lr in self._likelihood_ratios_baseline.items():
            current = self._likelihood_ratios.get(ev_name, baseline_lr)
            self._likelihood_ratios[ev_name] = current + 0.02 * (baseline_lr - current)

        if tuned > 0:
            logger.info(
                f"🔧 Calibration auto-tuning: net_bias={net_bias:+.3f}, "
                f"correction={correction_factor:.3f}, {tuned} LR adjustments "
                f"({'dampened positives' if net_bias > 0 else 'dampened negatives'})"
            )

    def update_likelihood(self, evidence_name: str, new_ratio: float):
        """Update a likelihood ratio from empirical data"""
        if evidence_name in self._likelihood_ratios:
            # Blend with existing (don't change too fast)
            old = self._likelihood_ratios[evidence_name]
            self._likelihood_ratios[evidence_name] = 0.8 * old + 0.2 * new_ratio
            logger.debug(
                f"Likelihood updated: {evidence_name} "
                f"{old:.3f} → {self._likelihood_ratios[evidence_name]:.3f}"
            )

    # ── Internal helpers ──────────────────────────────────────

    def _context_key(self, strategy: str, regime: str, direction: str) -> str:
        return f"{strategy}:{regime}:{direction}"

    def _get_posterior(self, key: str) -> BetaPosterior:
        if key not in self._posteriors:
            # Use config-driven defaults so priors can be tuned without code changes
            self._posteriors[key] = BetaPosterior(
                alpha=self._default_prior_alpha,
                beta=self._default_prior_beta,
            )
        return self._posteriors[key]

    def _get_calibration_error(self, p_win: float) -> float:
        """Get calibration error for a given prediction level"""
        bin_key = str(round(p_win, 1))
        if bin_key in self._calibration_bins:
            return self._calibration_bins[bin_key].get('error', 0.0)
        return 0.0  # No data yet

    def set_decay(self, decay: float):
        """
        Set the posterior decay rate used on the next update() call.
        Called by the learning loop with get_chop_learning_decay() so that
        high chop slows learning (prevents noise from overwriting clean data).
          chop=0.0 → decay=0.995 (normal, adapts quickly)
          chop=1.0 → decay=0.999 (slow, preserves priors through noise)
        """
        self._decay = max(0.990, min(0.999, decay))



    def reset_regime(self):
        """
        Fix G7: Called on regime change to decay cross-regime penalty carry-over.
        Applies a moderate decay to all posteriors so penalties from the old regime
        fade faster rather than persisting until 10+ new trades wash them out.

        C2-FIX: Decay toward the configured default prior (self._default_prior_alpha /
        self._default_prior_beta = 2.6/2.4 → mean 0.52) rather than the hardcoded
        (2.0, 2.0) target which gives mean 0.50.  The 0.02 difference seems minor
        but compounds over many regime changes: every reset was nudging all priors
        below the p_win EV gate, biasing good trades toward systematic rejection.
        """
        decay = 0.97  # light decay — preserves signal while reducing stale penalty
        target_a = self._default_prior_alpha
        target_b = self._default_prior_beta
        for key, posterior in self._posteriors.items():
            posterior.alpha = target_a + (posterior.alpha - target_a) * decay
            posterior.beta  = target_b + (posterior.beta  - target_b) * decay
        logger.info(f"🔄 Probability engine posteriors softly decayed on regime change ({len(self._posteriors)} contexts)")


# ── Singleton ─────────────────────────────────────────────────
probability_engine = ProbabilityEngine()
