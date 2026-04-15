"""
TitanBot Pro — Learning Loop (Pillar 5: Feedback Learning System)
===================================================================
THE SECRET SAUCE. Grand quant systems are closed loops.

Pipeline:
  Trade → Outcome → Model Update → Strategy Weight Update → Better Trades

Every trade changes future decisions. Without this, the bot never improves.

The Learning Loop orchestrates updates to:
  1. Probability Engine (Bayesian posteriors per context)
  2. Alpha Model (strategy weights and EV estimates)
  3. Portfolio Engine (capital updates)
  4. Likelihood ratios (which evidence actually predicts wins?)
  5. Calibration (are our probability estimates accurate?)

It also performs periodic meta-analysis:
  - Which evidence factors are actually predictive?
  - Which strategies should be enabled/disabled?
  - Are there systematic biases in our predictions?
  - Should regime thresholds be adjusted?
"""

import asyncio
import logging
import math
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from config.loader import cfg
from data.database import db

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """A completed trade with all context"""
    signal_id: int
    symbol: str
    strategy: str
    direction: str
    regime: str
    confidence: float
    p_win_predicted: float
    rr_ratio: float

    # Outcome
    won: bool
    pnl_r: float
    outcome: str          # WIN | LOSS | BE | EXPIRED

    # Evidence that was present
    evidence: Dict[str, bool] = field(default_factory=dict)

    # Timing
    timestamp: float = 0.0
    bars_to_outcome: int = 0


class LearningLoop:
    """
    Closed-loop feedback system.
    Connects trade outcomes back to all prediction models.
    """

    def __init__(self):
        self._trade_history: List[TradeRecord] = []
        self._max_history = 500          # Keep last 500 trades
        self._analysis_interval = 3600   # Full analysis every hour
        self._last_analysis = 0
        self._last_save = 0.0            # EC7: throttle DB writes to once/minute
        self._save_interval = 60         # EC7: min seconds between DB saves
        self._pending_save = False       # EC7: dirty flag — save needed
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Evidence tracking: how often does each evidence type appear in wins vs losses?
        self._evidence_wins: Dict[str, int] = defaultdict(int)
        self._evidence_losses: Dict[str, int] = defaultdict(int)
        self._evidence_total: Dict[str, int] = defaultdict(int)

        # Calibration buffer
        self._calibration_buffer: List[Tuple[float, bool]] = []


    def start(self):
        """Start the learning loop background task"""
        self._running = True
        self._task = asyncio.create_task(self._analysis_loop())
        logger.info("🧠 Learning loop started")

    async def stop(self):
        """Stop the learning loop"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Persist state
        await self._save_state()
        logger.info("Learning loop stopped")

    def record_trade(
        self,
        signal_id: int,
        symbol: str,
        strategy: str,
        direction: str,
        regime: str,
        confidence: float,
        p_win_predicted: float,
        rr_ratio: float,
        won: bool,
        pnl_r: float,
        outcome: str,
        evidence: Dict[str, bool] = None,
        bars_to_outcome: int = 0,
    ):
        """
        Record a completed trade and immediately update all models.
        This is the core of the learning loop.
        """
        record = TradeRecord(
            signal_id=signal_id,
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            regime=regime,
            confidence=confidence,
            p_win_predicted=p_win_predicted,
            rr_ratio=rr_ratio,
            won=won,
            pnl_r=pnl_r,
            outcome=outcome,
            evidence=evidence or {},
            timestamp=time.time(),
            bars_to_outcome=bars_to_outcome,
        )
        self._trade_history.append(record)

        # Trim history — and decrement evidence counters for the trimmed record
        # FIX: previously evidence counters only ever incremented. After 500 trades
        # they reflected more history than existed, making likelihood ratios drift
        # upward indefinitely and overestimating every evidence factor's predictive power.
        if len(self._trade_history) > self._max_history:
            evicted = self._trade_history[0]
            self._trade_history = self._trade_history[-self._max_history:]
            # Undo the evicted record's contribution to evidence counters
            if evicted.evidence:
                _ev_won = evicted.won
                # FIX #7: BREAKEVEN only incremented _total (not _wins or _losses).
                # Old code treated BREAKEVEN same as EXPIRED, skipping both decrements.
                # But expired losses DID get counted in _losses at increment time.
                # Fix: separate BREAKEVEN (only undo _total) from EXPIRED/INVALIDATED.
                _ev_is_be  = evicted.outcome == "BREAKEVEN"
                _ev_is_exp = evicted.outcome in ("EXPIRED", "INVALIDATED")
                for ev_name, present in evicted.evidence.items():
                    if present:
                        self._evidence_total[ev_name] = max(0, self._evidence_total[ev_name] - 1)
                        if not _ev_is_be:  # BEs only increment _total, so only undo _total (done above)
                            if _ev_won and not _ev_is_exp:
                                self._evidence_wins[ev_name] = max(0, self._evidence_wins[ev_name] - 1)
                            elif not _ev_won:
                                # expired losses were counted in _losses, undo them too
                                self._evidence_losses[ev_name] = max(0, self._evidence_losses[ev_name] - 1)

        # ── Immediate updates ─────────────────────────────────

        # FIX P1-D: EXPIRED and INVALIDATED must feed learning loop as weak negatives.
        # FIX: BREAKEVEN is neutral — no posterior update.
        _is_expired = outcome in ("EXPIRED", "INVALIDATED")
        _is_breakeven = outcome == "BREAKEVEN"

        # 1. Update Probability Engine posteriors
        # Bug 5 fix: chop-aware decay prevents noise overwriting clean trend data.
        try:
            from core.probability_engine import probability_engine
            from signals.regime_thresholds import get_chop_learning_decay
            from analyzers.regime import regime_analyzer as _ra
            _chop = getattr(_ra, '_chop_strength', 0.0)
            probability_engine.set_decay(get_chop_learning_decay(_chop))
            if _is_breakeven:
                pass  # neutral — skip posterior update
            elif _is_expired:
                # FIX #24: Expired signals are truly neutral — setup was never tested.
                # Previously injected as 0.3-weight loss, which biased posteriors downward.
                pass  # neutral — skip posterior update for expired
            else:
                probability_engine.update(strategy, regime, direction, won, pnl_r)
        except Exception as e:
            logger.error(f"Probability engine update failed: {e}")

        # 2. Update Alpha Model strategy weights
        try:
            from core.alpha_model import alpha_model
            if _is_breakeven:
                pass  # neutral — skip alpha weight update
            elif _is_expired:
                # FIX #24: Expired = neutral for alpha model too (was -0.3 loss)
                pass
            else:
                alpha_model.record_outcome(strategy, won, pnl_r)
        except Exception as e:
            logger.error(f"Alpha model update failed: {e}")

        # 3. Update calibration buffer (only resolved trades, not expirations)
        # PHASE 3 FIX (CALIB-LAG): run calibration on every trade, not just hourly.
        # Calibration is a lightweight numpy op — no reason to batch it.
        if not _is_expired:
            self._calibration_buffer.append((p_win_predicted, won))
            if len(self._calibration_buffer) >= 15:
                try:
                    from core.probability_engine import probability_engine
                    probability_engine.calibrate(self._calibration_buffer)
                    # FIX 9: clear buffer after calibrate() — previously it was never
                    # cleared here, so once it hit 15 entries, calibrate() fired on
                    # every single trade call redundantly. Keep last 5 as rolling overlap.
                    self._calibration_buffer = self._calibration_buffer[-5:]
                except Exception:
                    pass

        # 4. Track evidence effectiveness
        if evidence:
            for ev_name, present in evidence.items():
                if present:
                    self._evidence_total[ev_name] += 1
                    if won and not _is_breakeven and not _is_expired:
                        self._evidence_wins[ev_name] += 1
                    elif not won and not _is_breakeven:
                        self._evidence_losses[ev_name] += 1

        logger.info(
            f"\U0001F9E0 Learning: {outcome} {strategy} {symbol} {direction} "
            f"({pnl_r:+.2f}R) | predicted P(win)={p_win_predicted:.2f}"
        )

        # FIX #6: Actually call _maybe_save() — it was defined but never invoked.
        # Without this, the only save path was _analysis_loop() every hour,
        # meaning any crash within an hour loses all trade learning data.
        self._pending_save = True
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._maybe_save())
        except RuntimeError:
            pass  # Not in async context — will save on next analysis cycle

        # ── Periodic analysis ─────────────────────────────────────

    async def _analysis_loop(self):
        """Periodic deep analysis of accumulated data"""
        while self._running:
            try:
                await asyncio.sleep(self._analysis_interval)

                if len(self._trade_history) < 20:
                    continue

                logger.info("🧠 Running learning loop analysis...")

                # 1. Calibration check
                self._run_calibration()

                # 2. Evidence effectiveness update
                self._update_evidence_likelihoods()

                # 3. Strategy meta-analysis
                self._strategy_meta_analysis()

                # 4. Persist state
                await self._save_state()

                self._last_analysis = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Learning loop analysis error: {e}")
                await asyncio.sleep(300)

    def _run_calibration(self):
        """Check how well-calibrated our probability estimates are"""
        if len(self._calibration_buffer) < 30:
            return

        try:
            from core.probability_engine import probability_engine
            probability_engine.calibrate(self._calibration_buffer)
        except Exception as e:
            logger.error(f"Calibration failed: {e}")

        # Keep only recent predictions for next calibration
        self._calibration_buffer = self._calibration_buffer[-200:]

    def _update_evidence_likelihoods(self):
        """
        Compute empirical likelihood ratios for each evidence type.
        Update the probability engine's likelihood table.

        FIX #1: Base rate is now computed from the FULL resolved trade set (not
        evidence-filtered subset). Previously: base_rate used len(resolved) from
        all trades, but wins/total used only trades WHERE evidence was present.
        This made factors that appeared in few trades look artificially strong.
        Fix: compare win-rate-given-evidence vs win-rate-overall properly.
        """
        try:
            from core.probability_engine import probability_engine

            # Compute global base win rate from ALL resolved trades once
            resolved = [t for t in self._trade_history
                        if t.outcome not in ("EXPIRED", "INVALIDATED", "BREAKEVEN")
                        and hasattr(t, 'won')]
            if len(resolved) < 20:
                return  # not enough data

            global_wins = sum(1 for t in resolved if t.won)
            global_base_rate = max(0.35, min(0.75, global_wins / len(resolved)))

            updated = 0
            for ev_name in list(self._evidence_total.keys()):
                total = self._evidence_total[ev_name]
                if total < 10:
                    continue  # Not enough data for this factor

                wins = self._evidence_wins[ev_name]
                # win_rate = P(win | evidence present)
                win_rate_given_ev = wins / total if total > 0 else global_base_rate

                # Log-odds lift: how much does this evidence shift the prior?
                # LR = log( P(win|ev)/P(loss|ev) ) - log( P(win)/P(loss) )
                if 0 < win_rate_given_ev < 1:
                    lr_ev = math.log(win_rate_given_ev / (1 - win_rate_given_ev))
                    lr_base = math.log(global_base_rate / (1 - global_base_rate))
                    empirical_lr = lr_ev - lr_base
                    # Clamp to ±0.5 to prevent single noisy factor dominating
                    empirical_lr = max(-0.5, min(0.5, empirical_lr))
                    probability_engine.update_likelihood(ev_name, empirical_lr)
                    updated += 1

            logger.info(
                f"Evidence likelihoods updated: {updated}/{len(self._evidence_total)} factors "
                f"(base_rate={global_base_rate:.2%}, {len(resolved)} resolved trades)"
            )
        except Exception as e:
            logger.error(f"Evidence likelihood update failed: {e}")

    def _strategy_meta_analysis(self):
        """
        PHASE 3 FIX (AUTO-DIS + LEARN-CAP): Time-weighted meta-analysis with
        two-stage auto-disable for persistently negative strategies.

        Stage 1 (avg_r < -0.15R, 15+ trades): Telegram alert + halve weight.
        Stage 2 (avg_r < -0.30R, 20+ trades): Auto-disable + alert requiring
        manual re-enable. Forces a human decision before recovery.

        Time weights: last 7 days = 1.0, 7-30 days = 0.5, 30+ days = 0.2.
        Old trades still count but carry much less influence.
        """
        import time as _time
        if len(self._trade_history) < 20:
            return

        now = _time.time()
        # Time-weighted stats
        strategy_stats = defaultdict(lambda: {
            'weighted_wins': 0.0, 'weighted_losses': 0.0, 'weighted_r': 0.0,
            'raw_count': 0
        })

        for trade in self._trade_history[-200:]:
            age_days = (now - trade.timestamp) / 86400 if trade.timestamp else 30
            if age_days <= 7:
                w = 1.0
            elif age_days <= 30:
                w = 0.5
            else:
                w = 0.2

            stats = strategy_stats[trade.strategy]
            stats['raw_count'] += 1
            if trade.won:
                stats['weighted_wins'] += w
            else:
                stats['weighted_losses'] += w
            stats['weighted_r'] += trade.pnl_r * w

        logger.info("=== Strategy Meta-Analysis (time-weighted) ===")
        for strat, stats in sorted(strategy_stats.items()):
            total_w = stats['weighted_wins'] + stats['weighted_losses']
            if total_w < 1.0 or stats['raw_count'] < 5:
                continue
            wr = stats['weighted_wins'] / total_w
            avg_r = stats['weighted_r'] / total_w

            status = "✅" if avg_r > 0 else "⚠️" if avg_r > -0.15 else "❌"
            logger.info(
                f"  {status} {strat}: {stats['raw_count']} trades | "
                f"WR={wr:.1%} | wtd avg R={avg_r:+.3f}"
            )

            # PHASE 3 FIX (AUTO-DIS): Two-stage auto-disable
            if stats['raw_count'] >= 20 and avg_r < -0.30:
                # Stage 2: auto-disable — requires human re-enable
                try:
                    from governance.performance_tracker import performance_tracker
                    performance_tracker.suppress_strategy(strat, duration_mins=10080)  # FIX 7: was disable_strategy() — method does not exist
                    logger.warning(
                        f"  🚫 AUTO-DISABLED {strat}: avg_r={avg_r:+.3f}R over "
                        f"{stats['raw_count']} trades (time-weighted). "
                        f"Manual re-enable required via /unstop or config."
                    )
                    # Fire Telegram alert
                    try:
                        import asyncio as _asyncio
                        from tg.bot import telegram_bot  # FIX 8: wrong module path
                        _asyncio.create_task(telegram_bot.send_signals_text(
                            text=(
                                f"🚫 <b>Strategy Auto-Disabled</b>\n"
                                f"<b>{strat}</b> has negative edge: "
                                f"<b>{avg_r:+.3f}R</b> over {stats['raw_count']} trades\n"
                                f"Manual re-enable required."
                            ),
                        ))
                    except Exception:
                        pass
                except Exception as e:
                    logger.error(f"Auto-disable failed for {strat}: {e}")

            elif stats['raw_count'] >= 15 and avg_r < -0.15:
                # Stage 1: halve weight + alert
                logger.warning(
                    f"  ⚠️ {strat} has negative edge ({avg_r:+.3f}R, {stats['raw_count']} trades) "
                    f"— halving strategy weight"
                )
                try:
                    from core.alpha_model import alpha_model
                    sw = alpha_model._strategy_weights.get(strat)
                    if sw and sw.alpha > 2.0:
                        sw.alpha = max(2.0, sw.alpha * 0.5)  # decay toward flat prior
                except Exception:
                    pass
                try:
                    import asyncio as _asyncio
                    from tg.bot import telegram_bot  # FIX 8: wrong module path
                    _asyncio.create_task(telegram_bot.send_signals_text(
                        text=(
                            f"⚠️ <b>Strategy Warning</b>\n"
                            f"<b>{strat}</b>: avg_r={avg_r:+.3f}R over {stats['raw_count']} trades\n"
                            f"Weight halved. If it continues, auto-disable will trigger."
                        ),
                    ))
                except Exception:
                    pass

    # ── R8-F4: Regime-Adaptive Strategy Rotation ────────────────

    def get_strategy_regime_status(self, strategy: str, regime: str) -> str:
        """
        Returns 'HOT', 'COLD', or 'NORMAL' for a strategy in the current regime.

        HOT: >60% win rate over rolling 20 trades → auto-boost confidence +5
        COLD: <40% win rate over rolling 20 trades → auto-disable temporarily
        NORMAL: between 40-60% → standard operation

        This creates emergent adaptation: in bear markets, reversal strategies
        naturally gain weight while breakout longs get cold-rotated out.
        """
        # Get recent trades for this strategy in this regime
        matching = [
            t for t in self._trade_history[-200:]
            if t.strategy == strategy
            and t.regime == regime
            and t.outcome not in ("EXPIRED", "INVALIDATED", "BREAKEVEN")
        ]

        if len(matching) < 10:
            return "NORMAL"  # Not enough data — don't penalize

        # Use last 20 trades (or fewer if not enough)
        window = matching[-20:]
        wins = sum(1 for t in window if t.won)
        win_rate = wins / len(window)

        if win_rate >= 0.60:
            return "HOT"
        elif win_rate <= 0.40:
            return "COLD"
        else:
            return "NORMAL"

    def get_rotation_confidence_adj(self, strategy: str, regime: str) -> int:
        """
        Get confidence adjustment based on strategy rotation status.

        Returns:
          +5  for HOT strategies (performing well in this regime)
          -15 for COLD strategies (underperforming — should be reduced)
           0  for NORMAL strategies
        """
        status = self.get_strategy_regime_status(strategy, regime)
        if status == "HOT":
            return +5
        elif status == "COLD":
            return -15
        return 0

    # ── Persistence ───────────────────────────────────────────

    async def _maybe_save(self):
        """
        EC7: Throttled save — writes to DB at most once per minute even if
        record_outcome fires many times (prevents SSD write amplification).
        """
        import time as _t
        self._pending_save = True
        if _t.time() - self._last_save >= self._save_interval:
            await self._save_state()
            self._last_save = _t.time()
            self._pending_save = False

    async def _save_state(self):
        """Save learning loop state to dedicated learning_state table (Fix L7: no more append-only log events)"""
        try:
            state = {
                'evidence_wins': dict(self._evidence_wins),
                'evidence_losses': dict(self._evidence_losses),
                'evidence_total': dict(self._evidence_total),
                'trade_count': len(self._trade_history),
                'last_analysis': self._last_analysis,
            }

            # Also save probability engine full state (posteriors + LRs + calibration)
            # Phase 9-12 audit fix: previously only posteriors were saved
            try:
                from core.probability_engine import probability_engine
                state['probability_state'] = probability_engine.get_full_state()
            except Exception:
                pass

            # Save alpha model weights
            try:
                from core.alpha_model import alpha_model
                state['strategy_weights'] = alpha_model.get_all_weights()
            except Exception:
                pass

            # Fix L7: upsert into dedicated table instead of appending to events
            await db.save_learning_state('main', state)
            logger.debug("Learning loop state saved to learning_state table")

        except Exception as e:
            logger.error(f"Failed to save learning loop state: {e}")

    async def load_state(self):
        """Load learning loop state from dedicated learning_state table (Fix L7)"""
        try:
            state = await db.load_learning_state('main')
            if not state:
                logger.info("No previous learning state found — bootstrapping priors")
                self._bootstrap_priors()
                return

            self._evidence_wins = defaultdict(int, state.get('evidence_wins', {}))
            self._evidence_losses = defaultdict(int, state.get('evidence_losses', {}))
            self._evidence_total = defaultdict(int, state.get('evidence_total', {}))

            # Restore probability engine full state (posteriors + LRs + calibration)
            # Phase 9-12 audit fix: supports both new full-state and legacy posteriors-only
            if 'probability_state' in state:
                try:
                    from core.probability_engine import probability_engine
                    probability_engine.load_full_state(state['probability_state'])
                except Exception:
                    pass
            elif 'posteriors' in state:
                try:
                    from core.probability_engine import probability_engine
                    probability_engine.load_posteriors(state['posteriors'])
                except Exception:
                    pass

            # Restore alpha model weights
            if 'strategy_weights' in state:
                try:
                    from core.alpha_model import alpha_model
                    alpha_model.load_weights(state['strategy_weights'])
                except Exception:
                    pass

            logger.info(
                f"Learning state loaded: {state.get('trade_count', 0)} historical trades"
            )
        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")

    def _bootstrap_priors(self):
        """
        Bootstrap priors when no historical data exists (cold start fix).

        Instead of starting from zero knowledge, we inject conservative
        prior beliefs based on general crypto trading statistics:
          - Evidence factors that are generally predictive get small head starts
          - Strategy base rates reflect typical published win rates
          - This is equivalent to Bayesian informative priors

        These priors are weak (pseudo-count of 5) so they get washed out
        quickly once real data comes in (after ~20-30 trades).
        """
        logger.info("🧠 Bootstrapping learning priors (cold start)")

        # Prior beliefs: evidence that generally helps in crypto
        # Format: (evidence_name, prior_win_count, prior_loss_count)
        # These represent "as if we'd seen 5 trades with this evidence"
        evidence_priors = [
            ('volume_spike',         3, 2),    # Volume helps a bit
            ('trend_aligned',        4, 1),    # Trend alignment is strong
            ('rejection_candle',     3, 2),    # Rejection candles are decent
            ('support_confluence',   3, 2),    # S/R confluence helps
            ('rsi_extreme',          3, 2),    # RSI extremes help for reversals
            ('killzone_session',     3, 2),    # Trading in killzone helps
            ('multi_tf_agreement',   4, 1),    # Multi-TF confluence is strong
            ('whale_flow_aligned',   3, 2),    # Whale alignment helps
            ('funding_extreme',      3, 2),    # Extreme funding = fade signal
        ]

        for ev_name, wins, losses in evidence_priors:
            self._evidence_wins[ev_name] = wins
            self._evidence_losses[ev_name] = losses
            self._evidence_total[ev_name] = wins + losses

        # Bootstrap probability engine with conservative priors
        try:
            from core.probability_engine import probability_engine
            probability_engine.bootstrap_priors()
        except Exception:
            pass

        # Bootstrap alpha model with equal strategy weights
        try:
            from core.alpha_model import alpha_model
            alpha_model.bootstrap_priors()
        except Exception:
            pass

        logger.info(
            f"  Priors set: {len(evidence_priors)} evidence factors bootstrapped"
        )

    # ── Public analytics ──────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get learning loop statistics"""
        total = len(self._trade_history)
        if total == 0:
            return {'total_trades': 0}

        resolved_trades = [t for t in self._trade_history if t.outcome not in ("EXPIRED", "INVALIDATED")]
        decisive_trades = [t for t in resolved_trades if t.outcome != "BREAKEVEN"]
        expired_count = sum(1 for t in self._trade_history if t.outcome == "EXPIRED")
        invalidated_count = sum(1 for t in self._trade_history if t.outcome == "INVALIDATED")
        breakeven_count = sum(1 for t in resolved_trades if t.outcome == "BREAKEVEN")
        wins = sum(1 for t in decisive_trades if t.won)
        losses = sum(1 for t in decisive_trades if not t.won)
        total_r = sum(t.pnl_r for t in self._trade_history)
        resolved_total = len(resolved_trades)
        decisive_total = len(decisive_trades)
        fill_denominator = resolved_total + expired_count + invalidated_count

        return {
            'total_trades': total,
            'real_trades': resolved_total,
            'expired_count': expired_count,
            'invalidated_count': invalidated_count,
            'wins': wins,
            'losses': losses,
            'breakevens': breakeven_count,
            # Legacy field retained for backward compatibility: decisive only.
            'win_rate': round(wins / decisive_total, 4) if decisive_total > 0 else 0,
            'decisive_win_rate': round(wins / decisive_total, 4) if decisive_total > 0 else 0,
            'execution_win_rate': round(wins / resolved_total, 4) if resolved_total > 0 else 0,
            'fill_rate': round(resolved_total / fill_denominator, 4) if fill_denominator > 0 else 0,
            'total_r': round(total_r, 2),
            'avg_r': round(total_r / resolved_total, 4) if resolved_total > 0 else 0,
            'evidence_factors_tracked': len(self._evidence_total),
            'calibration_samples': len(self._calibration_buffer),
        }


# ── Singleton ─────────────────────────────────────────────────
learning_loop = LearningLoop()
