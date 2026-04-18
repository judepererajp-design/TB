"""
TitanBot Pro — Performance Tracker & Adaptive Weights
======================================================
Tracks signal outcomes and adapts strategy weights over time.

Key feature: Strategies that perform well get more weight in the
confidence scoring. Strategies that underperform get penalised.

This prevents over-reliance on strategies that worked historically
but are no longer working in the current market regime.

EWMA (Exponentially Weighted Moving Average) gives more weight
to recent outcomes, so the system adapts quickly to regime changes.
"""

import json
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from config.loader import cfg
from config.constants import Portfolio as _PortfolioC
from data.database import db

logger = logging.getLogger(__name__)
governance_log = logging.getLogger("governance")

# Weight-smoothing factor (α): small value = slow, stable transitions.
# Prevents oscillation where a 0.01 EWMA crossing flips weight by ±0.15.
_WEIGHT_ALPHA = 0.15


@dataclass
class StrategyStats:
    name: str
    total_signals: int = 0
    trades_taken: int  = 0
    wins: int          = 0
    losses: int        = 0
    breakevens: int    = 0
    total_r: float     = 0.0        # Sum of R multiples achieved
    ewma_win_rate: float = 0.5      # EWMA win rate (starts at 50%)
    weight_mult: float   = 1.0      # Multiplier applied to this strategy's signals
    is_disabled: bool    = False
    disabled_at: float   = 0.0
    # C2 FIX: deadline timestamp replaces asyncio.sleep() timer so suppression
    # survives restarts. is_strategy_disabled() checks time.time() < disabled_until.
    disabled_until: float = 0.0

    # ── Advanced performance metrics (Medallion-grade) ────────────────
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    r_outcomes: list = None
    max_drawdown_r: float = 0.0

    # ── Execution-quality loss breakdown ──────────────────────────────
    # Tracks how many losses fell into each execution-quality class so
    # the system can distinguish "bad strategy" from "bad execution".
    # Keys: GOOD_LOSS | BORDERLINE | BAD_EXECUTION | SYSTEM_FAILURE
    loss_by_class: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.r_outcomes is None:
            self.r_outcomes = []
        if not self.loss_by_class:
            self.loss_by_class = {
                "GOOD_LOSS": 0,
                "BORDERLINE": 0,
                "BAD_EXECUTION": 0,
                "SYSTEM_FAILURE": 0,
            }


class PerformanceTracker:
    """
    Tracks signal outcomes and adjusts strategy weights.
    All data is persisted to SQLite and survives restarts.
    """

    def __init__(self):
        self._gov_cfg = cfg.governance
        self._adapt_cfg = cfg.aggregator.adaptation

        self._stats: Dict[str, StrategyStats] = {}
        self._ewma_decay = getattr(self._adapt_cfg, 'ewma_decay', 0.1)
        self._min_signals = getattr(self._adapt_cfg, 'min_signals_before_adapt', 10)
        self._max_boost   = getattr(self._adapt_cfg, 'max_weight_boost', 0.15)
        self._max_penalty = getattr(self._adapt_cfg, 'max_weight_penalty', -0.15)

    async def initialize(self):
        """Load historical performance from database.

        Two-pass restore:
          1. Load raw win/loss/BE counts from the outcomes table (30-day window)
             so summary displays are accurate.
          2. Override ewma_win_rate, weight_mult, is_disabled, disabled_until, and
             r_outcomes from strategy_persistence_v1 — these are the *learned* values
             that must survive restarts unchanged (C1 fix).

        The second pass is authoritative for any field it contains; the first pass
        only fills in counters that strategy_persistence_v1 doesn't store.
        """
        import asyncio
        try:
            rows = await asyncio.wait_for(
                db.get_strategy_performance(days=30),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("⚠️  Performance tracker DB query timed out — starting with empty stats (non-fatal)")
            rows = []
        except Exception as e:
            logger.warning(f"⚠️  Performance tracker DB query failed: {e} — starting empty")
            rows = []

        for row in rows:
            name = row.get('strategy', '')
            if name:
                stats = StrategyStats(name=name)
                stats.total_signals = row.get('signals', 0)
                stats.wins = row.get('wins', 0)
                stats.losses = row.get('losses', 0)
                stats.breakevens = row.get('breakevens', 0)
                stats.total_r = row.get('total_r', 0.0)
                stats.trades_taken = stats.wins + stats.losses + stats.breakevens

                if stats.trades_taken > 0:
                    decisive = stats.wins + stats.losses
                    raw_wr = stats.wins / decisive if decisive > 0 else _PortfolioC.PRIOR_WIN_RATE
                    # M2 FIX: use n/(n+50) shrinkage consistently (was min(1,n/10)
                    # in some paths). Lower-sample strategies get more conservative
                    # estimates; the formula converges to raw_wr with ~50+ decisive
                    # trades rather than immediately trusting 10 outcomes.
                    blend = decisive / (decisive + 50)
                    stats.ewma_win_rate = blend * raw_wr + (1 - blend) * _PortfolioC.PRIOR_WIN_RATE
                else:
                    stats.ewma_win_rate = _PortfolioC.PRIOR_WIN_RATE

                self._stats[name] = stats

        # ── C1 FIX: restore persisted adaptive state ──────────────────────
        # Load strategy_persistence_v1 and overlay learned values so restarts
        # never silently discard weight_mult, is_disabled, disabled_until, or
        # the r_outcomes buffer needed for Sharpe/Sortino/Calmar computation.
        try:
            persisted = await db.load_all_strategy_states()
        except Exception as e:
            logger.warning(f"⚠️  Could not load strategy persistence: {e}")
            persisted = []

        now = time.time()
        for row in persisted:
            name = row.get('strategy', '')
            if not name:
                continue
            if name not in self._stats:
                self._stats[name] = StrategyStats(name=name)
            stats = self._stats[name]

            stats.ewma_win_rate = float(row.get('ewma_win_rate', stats.ewma_win_rate))
            stats.weight_mult   = float(row.get('weight_mult',   stats.weight_mult))
            stats.is_disabled   = bool(row.get('is_disabled',  0))
            stats.disabled_until = float(row.get('disabled_until', 0.0))
            stats.disabled_at   = float(row.get('disabled_at', 0.0))

            # C2 FIX: if the suppression deadline has already passed, clear it
            # so the strategy is not left permanently disabled after a long outage.
            if stats.disabled_until > 0 and now >= stats.disabled_until:
                stats.is_disabled    = False
                stats.disabled_until = 0.0
                logger.info(f"▶️  Strategy {name} suppression expired during downtime — re-enabled")

            raw_json = row.get('r_outcomes_json', '[]')
            try:
                outcomes = json.loads(raw_json) if raw_json else []
                stats.r_outcomes = [x for x in outcomes if x is not None][-500:]
            except Exception as e:
                logger.debug(f"r_outcomes_json parse failed (non-fatal): {e}")
                stats.r_outcomes = []

        logger.info(
            f"Performance tracker loaded {len(self._stats)} strategy records "
            f"({len(persisted)} with persisted adaptive state)"
        )

    async def record_outcome(
        self, strategy: str, outcome: str, pnl_r: Optional[float] = None,
        mae_r: float = 0.0, mfe_r: float = 0.0,
        entry_status: Optional[str] = None, max_r: Optional[float] = None,
    ):
        """
        Record a trade outcome with MAE/MFE and update Sharpe/Sortino/Calmar.

        Args:
            strategy: Strategy name
            outcome: 'WIN' | 'LOSS' | 'BREAKEVEN' | 'SKIPPED'
            pnl_r: R multiple achieved (positive for win, negative for loss)
            mae_r: Max Adverse Excursion in R (worst intra-trade drawdown)
            mfe_r: Max Favourable Excursion in R (best intra-trade peak)
            entry_status: IN_ZONE | LATE | EXTENDED — captured at fill time
            max_r: Maximum favourable excursion reached during the trade (R)
        """
        if strategy not in self._stats:
            self._stats[strategy] = StrategyStats(name=strategy)

        stats = self._stats[strategy]
        stats.total_signals += 1

        if outcome == 'SKIPPED':
            return  # Don't count skipped signals in performance

        stats.trades_taken += 1

        if outcome == 'WIN':
            stats.wins += 1
            if pnl_r is not None:
                stats.total_r += pnl_r
                # C4 FIX: only append when pnl_r is not None; previously None was
                # appended and cast to NaN by numpy, silently corrupting all
                # Sharpe/Sortino/Calmar calculations for this strategy.
                stats.r_outcomes.append(pnl_r)
                if len(stats.r_outcomes) > 500:
                    stats.r_outcomes = stats.r_outcomes[-500:]
            self._update_advanced_metrics(stats)
            stats.ewma_win_rate = (
                (1 - self._ewma_decay) * stats.ewma_win_rate +
                self._ewma_decay * 1.0
            )
        elif outcome == 'LOSS':
            stats.losses += 1
            if pnl_r is not None:
                stats.total_r += pnl_r  # Negative R
            # R5-S5 FIX: Append LOSS R-outcomes and update Sharpe/Sortino/Calmar.
            # C4 FIX: use the actual pnl_r when available, fall back to -1.0.
            stats.r_outcomes.append(pnl_r if pnl_r is not None else -1.0)
            if len(stats.r_outcomes) > 500:
                stats.r_outcomes = stats.r_outcomes[-500:]
            self._update_advanced_metrics(stats)
            stats.ewma_win_rate = (
                (1 - self._ewma_decay) * stats.ewma_win_rate +
                self._ewma_decay * 0.0
            )
            # Classify the loss by execution quality and update the counter.
            try:
                from governance.execution_quality import classify_loss as _clf
                _eq = _clf(entry_status=entry_status, max_r=max_r)
                _cls = _eq["class"]
                stats.loss_by_class[_cls] = stats.loss_by_class.get(_cls, 0) + 1
                logger.info(
                    f"📊 Loss classification | {strategy} | {_cls} "
                    f"(score={_eq['score']:.2f}) | {_eq['reason']}"
                )
            except Exception as e:
                logger.debug(f"Loss classification failed (non-fatal): {e}")
        elif outcome == 'BREAKEVEN':
            stats.breakevens += 1
            stats.ewma_win_rate = (
                (1 - self._ewma_decay) * stats.ewma_win_rate +
                self._ewma_decay * 0.5
            )

        # Update adaptive weight
        if cfg.aggregator.adaptation.get('enabled', True):
            self._update_weight(stats)

        # Check for auto-disable
        await self._check_auto_disable(stats)

        # C1 FIX: persist updated state so restarts start from here, not zero.
        await self._save_strategy_state(stats)

        logger.debug(
            f"Performance update: {strategy} {outcome} | "
            f"EWMA WR: {stats.ewma_win_rate:.2f} | Weight: {stats.weight_mult:.2f}"
        )

    def _update_advanced_metrics(self, stats: "StrategyStats"):
        """
        Recompute Sharpe, Sortino, Calmar, MAE/MFE from R outcomes buffer.
        Called after each new trade outcome.

        Sharpe = (mean_R - risk_free) / std_R * sqrt(trades_per_year)
        Sortino = (mean_R - risk_free) / downside_std * sqrt(trades_per_year)
        Calmar = annualised_return / max_drawdown
        """
        if not stats.r_outcomes or len(stats.r_outcomes) < 3:
            return
        import numpy as _np

        r = _np.array(stats.r_outcomes, dtype=float)
        mean_r   = float(_np.mean(r))
        std_r    = float(_np.std(r))
        risk_free = 0.0  # 0 R hurdle rate

        # Annualisation factor — use the actual sample size to estimate
        # trades/year rather than the hardcoded 104 so fast scalpers and slow
        # swing traders are ranked on the same scale.
        n = len(r)
        # Use 104 as a reasonable default (2 trades/week); min 1 to avoid div-0.
        trades_per_year = max(1.0, min(104.0, n))

        # G-2 FIX: the old zero-vol fallback (mean_r * 10) turned 5 small wins
        # into a Sharpe of ~5, boosting the strategy's weight_mult artificially.
        # Replace with a capped constant (3.0) that is "good but not exceptional"
        # to avoid penalising steady low-vol strategies while preventing the
        # artifact from dominating rankings.
        if std_r > 0:
            stats.sharpe_ratio = round(
                (mean_r - risk_free) / std_r * (trades_per_year ** 0.5), 2
            )
        else:
            # No volatility yet — assign a neutral Sharpe, not a fabricated large one.
            stats.sharpe_ratio = round(mean_r * 3.0, 2) if mean_r > 0 else 0.0

        # G-3 FIX: initialize sortino_ratio to 0.0 explicitly before the
        # branches below so that when neither fires (e.g. len(neg_r) < 2 AND
        # std_r == 0) the field is deterministic rather than carrying over the
        # previous cycle's stale value.
        sortino: float = 0.0

        # Sortino (only penalise downside volatility)
        neg_r = r[r < 0]
        if len(neg_r) >= 2:
            downside_std = float(_np.std(neg_r))
            if downside_std > 0:
                sortino = round(
                    (mean_r - risk_free) / downside_std * (trades_per_year ** 0.5), 2
                )
            else:
                # All losses are identical — use Sharpe as a proxy (no extra inflation)
                sortino = stats.sharpe_ratio
        elif std_r > 0:
            # No losses yet — Sortino should be at least as good as Sharpe,
            # but cap the inflation multiplier to avoid the G-2-style artifact.
            sortino = round(stats.sharpe_ratio * 1.2, 2)

        stats.sortino_ratio = sortino

        # Max drawdown (worst peak-to-trough in R)
        cumulative = _np.cumsum(r)
        peak = _np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        stats.max_drawdown_r = round(float(_np.max(drawdown)), 2)

        # Calmar
        if stats.max_drawdown_r > 0:
            annualised_r = mean_r * trades_per_year
            stats.calmar_ratio = round(annualised_r / stats.max_drawdown_r, 2)

    def get_strategy_weight(self, strategy: str) -> float:
        """
        Get the weight multiplier for a strategy.
        Applied to that strategy's confidence score before aggregation.
        Returns 1.0 if no data yet.
        """
        if strategy not in self._stats:
            return 1.0
        stats = self._stats[strategy]
        if self.is_strategy_disabled(strategy):
            return 0.0
        return stats.weight_mult

    def is_strategy_disabled(self, strategy: str) -> bool:
        """Check if a strategy has been auto-disabled.

        C2 FIX: Also checks the disabled_until deadline so suppression
        survives restarts without relying on a running asyncio coroutine.
        When the deadline passes the strategy is automatically re-enabled
        and the DB state is NOT updated here (next record_outcome() or
        initialize() will clean it up) — this keeps the method side-effect-free.
        """
        if strategy not in self._stats:
            return False
        stats = self._stats[strategy]
        # Timestamp-based suppression (restart-safe)
        if stats.disabled_until > 0:
            if time.time() < stats.disabled_until:
                return True
            # Deadline expired — clear in-memory state.
            # The next persist call will write the cleared state to DB.
            stats.is_disabled    = False
            stats.disabled_until = 0.0
            logger.info(f"▶️  Strategy {strategy} suppression deadline passed — re-enabled")
        return stats.is_disabled

    def suppress_strategy(self, strategy: str, duration_mins: int = 120) -> None:
        """
        Temporarily disable a strategy for duration_mins.
        Called by diagnostic_engine on approved suppression changes.

        C2 FIX: Stores a disabled_until timestamp instead of scheduling an
        asyncio.sleep() coroutine. The coroutine was cancelled on every restart,
        making the suppression instant-reset. Now is_strategy_disabled() checks
        time.time() < disabled_until, which survives restarts correctly.
        """
        if strategy not in self._stats:
            logger.warning(
                f"suppress_strategy: '{strategy}' is not a tracked strategy — "
                f"suppression ignored. Known strategies: {list(self._stats.keys())}"
            )
            return
        now = time.time()
        new_until = now + duration_mins * 60
        stats_obj = self._stats[strategy]
        stats_obj.is_disabled = True
        # HIGH-FIX (suppress_strategy): take max() of the new deadline and any
        # existing deadline so consecutive suppressions do NOT shorten the total
        # cool-off.  Previously each call overwrote disabled_until, meaning a
        # second suppress while the first was still active would reduce the
        # remaining cool-off to duration_mins from *now* (potentially shorter).
        stats_obj.disabled_until = max(stats_obj.disabled_until, new_until)
        stats_obj.disabled_at    = now
        logger.info(f"⏸️  Strategy {strategy} suppressed for {duration_mins}min (until {stats_obj.disabled_until:.0f})")

        # Persist immediately so a restart within the window keeps the strategy
        # suppressed (the disabled_until timestamp is the source of truth).
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._save_strategy_state(self._stats[strategy]))
        except RuntimeError:
            logger.debug(f"suppress_strategy: no running loop — persistence deferred for {strategy}")

    def get_performance_summary(self) -> List[Dict]:
        """Get performance summary for all strategies (for /performance command)"""
        result = []
        for name, stats in self._stats.items():
            taken = stats.wins + stats.losses + stats.breakevens
            decisive = stats.wins + stats.losses
            win_rate = stats.wins / decisive if decisive > 0 else 0
            avg_r = stats.total_r / taken if taken > 0 else 0
            result.append({
                'strategy': name,
                'signals': stats.total_signals,
                'trades': taken,
                'wins': stats.wins,
                'losses': stats.losses,
                'win_rate': win_rate,
                'avg_r': avg_r,
                'ewma_wr': stats.ewma_win_rate,
                'weight': stats.weight_mult,
                'disabled': self.is_strategy_disabled(name),
            })
        return sorted(result, key=lambda x: x['wins'], reverse=True)

    def get_adaptation_notes(self) -> List[str]:
        """Get notes about adaptive weight changes for /performance"""
        notes = []
        for name, stats in self._stats.items():
            if stats.trades_taken < self._min_signals:
                continue
            if stats.weight_mult >= 1.1:
                notes.append(f"📈 {name}: +{(stats.weight_mult-1)*100:.0f}% weight (performing well)")
            elif stats.weight_mult <= 0.85:
                notes.append(f"📉 {name}: {(stats.weight_mult-1)*100:.0f}% weight (underperforming)")
            if self.is_strategy_disabled(name):
                notes.append(f"❌ {name}: AUTO-DISABLED (win rate too low)")
        return notes

    # ── Private methods ───────────────────────────────────────

    def _update_weight(self, stats: StrategyStats):
        """Update weight multiplier based on EWMA win rate.

        C5 FIX: Use EWMA smoothing (α=0.15) instead of hard assignment.
        Previously a strategy with EWMA=0.59 got weight=0.88 one update then
        EWMA=0.61 → weight=1.04 the next, causing constant oscillation.
        Now each update nudges the weight toward the target by 15% only,
        producing stable, smooth transitions.
        """
        if stats.trades_taken < self._min_signals:
            return  # Not enough data

        wr = stats.ewma_win_rate

        # Win rate above 60% = boost weight
        if wr > 0.60:
            boost = min(self._max_boost, (wr - 0.55) * 0.6)
            target = 1.0 + boost
        # Win rate below 45% = penalise
        elif wr < 0.45:
            penalty = max(self._max_penalty, -(0.50 - wr) * 0.8)
            target = 1.0 + penalty
        else:
            # Gradually return to 1.0
            target = 1.0

        # EWMA smooth: move 15% toward target each update
        stats.weight_mult = stats.weight_mult * (1 - _WEIGHT_ALPHA) + target * _WEIGHT_ALPHA

        # Clamp
        stats.weight_mult = max(
            1.0 + self._max_penalty,
            min(1.0 + self._max_boost, stats.weight_mult)
        )

    async def _check_auto_disable(self, stats: StrategyStats):
        """Auto-disable if win rate is too low for too long"""
        auto_cfg = self._gov_cfg.auto_disable
        if not getattr(auto_cfg, 'enabled', True):
            return

        min_signals = getattr(auto_cfg, 'min_signals', 20)
        min_wr      = getattr(auto_cfg, 'min_win_rate', 0.35)
        re_enable_d = getattr(auto_cfg, 're_enable_after_days', 7)

        taken = stats.wins + stats.losses

        if taken >= min_signals and stats.ewma_win_rate < min_wr and not stats.is_disabled:
            stats.is_disabled = True
            stats.disabled_at = time.time()
            # No fixed timeout for auto-disable — uses re_enable_after_days check below.
            # Set disabled_until=0 so the timestamp path doesn't interfere.
            stats.disabled_until = 0.0
            logger.warning(
                f"⛔ Strategy {stats.name} AUTO-DISABLED — "
                f"EWMA win rate {stats.ewma_win_rate:.1%} below {min_wr:.0%}"
            )
            await db.log_event(
                "STRATEGY_DISABLED",
                f"{stats.name} disabled",
                {'ewma_wr': stats.ewma_win_rate, 'trades': taken}
            )

        # Auto re-enable after cooldown
        if stats.is_disabled and stats.disabled_at > 0 and stats.disabled_until == 0:
            days_disabled = (time.time() - stats.disabled_at) / 86400
            if days_disabled >= re_enable_d:
                stats.is_disabled = False
                # HIGH-FIX: was hard-resetting ewma_win_rate = 0.5 regardless of
                # how badly the strategy performed. Now blend toward 0.5 from the
                # current value so a strategy with WR=0.20 starts at 0.35 (not 0.50)
                # and must prove itself before receiving a full-weight allocation.
                stats.ewma_win_rate = (stats.ewma_win_rate + 0.5) / 2.0
                stats.weight_mult = 0.8    # Start with slight penalty
                logger.info(f"✅ Strategy {stats.name} RE-ENABLED after {days_disabled:.0f} days (ewma_wr={stats.ewma_win_rate:.2f})")

    async def _save_strategy_state(self, stats: StrategyStats) -> None:
        """Persist one strategy's adaptive state to strategy_persistence_v1."""
        try:
            r_json = json.dumps(stats.r_outcomes[-500:])
            await db.save_strategy_state(
                strategy=stats.name,
                ewma_win_rate=stats.ewma_win_rate,
                weight_mult=stats.weight_mult,
                is_disabled=stats.is_disabled,
                disabled_until=stats.disabled_until,
                disabled_at=stats.disabled_at,
                r_outcomes_json=r_json,
            )
            disabled_until_str = (
                f"{stats.disabled_until:.0f}" if stats.disabled_until else "None"
            )
            governance_log.info(
                "WEIGHT_UPDATE | %s | ewma=%.3f | weight=%.3f | disabled=%s | disabled_until=%s",
                stats.name, stats.ewma_win_rate, stats.weight_mult,
                stats.is_disabled, disabled_until_str,
            )
        except Exception as e:
            logger.warning(f"Failed to persist strategy state for {stats.name}: {e}")

    def get_ranking(self) -> str:
        """
        Get strategy ranking dashboard formatted for Telegram/console.
        Ranks strategies by EWMA win rate * weight multiplier (composite score).
        """
        if not self._stats:
            return "📊 <b>Strategy Ranking</b>\n\nNo data yet — signals needed first."

        ranked = []
        for name, stats in self._stats.items():
            taken = stats.wins + stats.losses + stats.breakevens
            if taken == 0:
                avg_r = 0.0
                wr = 0.5
            else:
                avg_r = stats.total_r / taken
                wr = stats.ewma_win_rate
            composite = wr * stats.weight_mult
            ranked.append((name, stats, taken, wr, avg_r, composite))

        ranked.sort(key=lambda x: x[5], reverse=True)

        lines = ["📊 <b>Strategy Ranking</b>", "━━━━━━━━━━━━━━━━━━━━━━━━━"]
        for i, (name, stats, taken, wr, avg_r, composite) in enumerate(ranked):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
            status = "⛔" if self.is_strategy_disabled(name) else "✅"
            lines.append(
                f"{medal} <b>{name}</b> {status}\n"
                f"    WR: {wr:.0%} | Trades: {taken} | "
                f"Avg R: {avg_r:+.2f} | Weight: {stats.weight_mult:.2f}x"
            )

        return "\n".join(lines)

    def reset_regime(self, new_regime: str):
        """
        Fix G7: On regime change, reset EWMA slightly toward neutral so
        strategies penalized in old regime get a fresh evaluation window.
        """
        for name, stats in self._stats.items():
            stats.ewma_win_rate = 0.7 * stats.ewma_win_rate + 0.3 * 0.5
        logger.info(f"🔄 Performance tracker EWMA softened → {new_regime}")


# ── Singleton ──────────────────────────────────────────────
performance_tracker = PerformanceTracker()
