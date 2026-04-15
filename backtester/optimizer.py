"""
TitanBot Pro — Parameter Optimizer
=====================================
Grid search + walk-forward optimization over config parameters.

Answers questions like:
  - Is min_adx: 28 optimal, or should it be 22 or 35?
  - Should z_score_threshold be 2.0, 2.2, or 2.5?
  - What's the best min_confidence for choppy markets?
  - Should FVG min_size_pct be 0.002 or 0.005?

Optimization approach:
  1. Define parameter search space (grid)
  2. For each combination, run a backtest
  3. Score by chosen fitness metric (default: Sharpe * Profit Factor)
  4. Walk-forward validation to prevent overfitting
  5. Output ranked results with recommended config

Anti-overfitting measures:
  - Walk-forward splits (train 70% / test 30%)
  - Out-of-sample validation required
  - Penalty for excessive parameter sensitivity
  - Minimum trade count requirement
"""

import asyncio
import logging
import itertools
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpec:
    """Definition of a parameter to optimize"""
    name: str                     # Config key path (e.g., "strategies.breakout.min_adx")
    values: List[Any]             # Grid values to test
    description: str = ""


@dataclass
class OptimizationResult:
    """Result of a single parameter combination"""
    params: Dict[str, Any]        # Parameter values used
    train_result: Any = None      # BacktestResult on training data
    test_result: Any = None       # BacktestResult on test data
    fitness_score: float = 0.0    # Combined fitness metric
    is_overfit: bool = False      # Flagged if train >> test


@dataclass
class OptimizerReport:
    """Full optimization output"""
    parameter_specs: List[ParameterSpec]
    total_combinations: int
    total_time_seconds: float
    results: List[OptimizationResult] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_fitness: float = 0.0


class ParameterOptimizer:
    """
    Grid search optimizer with walk-forward validation.
    """

    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(__file__), "optimization")
        os.makedirs(self.output_dir, exist_ok=True)

    async def optimize(
        self,
        symbol: str,
        ohlcv_data: Dict[str, List],
        params: List[ParameterSpec],
        strategies: List = None,
        train_pct: float = 0.7,
        min_trades: int = 20,
        fitness_fn: str = "sharpe_pf",   # "sharpe_pf" | "total_r" | "win_rate" | "profit_factor"
        max_combinations: int = 500,
    ) -> OptimizerReport:
        """
        Run grid search optimization.

        Args:
            symbol: Trading pair
            ohlcv_data: Historical data dict
            params: List of ParameterSpec to search
            strategies: Strategy list (or None for all)
            train_pct: Training data percentage
            min_trades: Minimum trades to consider a result valid
            fitness_fn: How to rank parameter combinations
            max_combinations: Cap on total combinations to test

        Returns:
            OptimizerReport with ranked results
        """
        from backtester.engine import backtest_engine
        from backtester.data_loader import data_loader

        start_time = time.time()

        # Generate all parameter combinations
        combinations = self._generate_grid(params)
        total = len(combinations)

        if total > max_combinations:
            # Random sample if too many
            import random
            combinations = random.sample(combinations, max_combinations)
            logger.warning(
                f"Parameter space too large ({total}). "
                f"Sampling {max_combinations} combinations."
            )

        logger.info(
            f"🔍 Optimizer starting: {len(combinations)} combinations | "
            f"{len(params)} parameters | {symbol}"
        )

        # Split data into train/test
        primary_tf = '1h' if '1h' in ohlcv_data else list(ohlcv_data.keys())[0]
        candles = ohlcv_data[primary_tf]
        split_idx = int(len(candles) * train_pct)

        train_data = {}
        test_data = {}
        for tf, tf_candles in ohlcv_data.items():
            # Find split point by timestamp
            split_ts = candles[split_idx][0]
            train_data[tf] = [c for c in tf_candles if c[0] <= split_ts]
            test_data[tf] = [c for c in tf_candles if c[0] > split_ts]

        results = []

        for i, combo in enumerate(combinations):
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(combinations)}")

            param_dict = {
                params[j].name: combo[j]
                for j in range(len(params))
            }

            try:
                # Backtest on training data
                train_result = await backtest_engine.run(
                    symbol=symbol,
                    ohlcv_data=train_data,
                    strategies=strategies,
                    config_overrides=param_dict,
                )

                # Skip if not enough trades
                if train_result.total_trades < min_trades:
                    continue

                # Backtest on test data (out-of-sample)
                test_result = await backtest_engine.run(
                    symbol=symbol,
                    ohlcv_data=test_data,
                    strategies=strategies,
                    config_overrides=param_dict,
                )

                # Calculate fitness
                fitness = self._calculate_fitness(
                    train_result, test_result, fitness_fn
                )

                # Check for overfitting
                is_overfit = False
                if test_result.total_trades >= 5:
                    train_wr = train_result.win_rate
                    test_wr = test_result.win_rate
                    if train_wr > 0 and (train_wr - test_wr) / train_wr > 0.3:
                        is_overfit = True
                        fitness *= 0.5  # Heavy penalty

                results.append(OptimizationResult(
                    params=param_dict,
                    train_result=train_result,
                    test_result=test_result,
                    fitness_score=fitness,
                    is_overfit=is_overfit,
                ))

            except Exception as e:
                logger.debug(f"Combination {i} failed: {e}")
                continue

        # Sort by fitness (highest first)
        results.sort(key=lambda r: r.fitness_score, reverse=True)

        elapsed = time.time() - start_time

        report = OptimizerReport(
            parameter_specs=params,
            total_combinations=len(combinations),
            total_time_seconds=elapsed,
            results=results,
            best_params=results[0].params if results else {},
            best_fitness=results[0].fitness_score if results else 0,
        )

        logger.info(
            f"✅ Optimization complete in {elapsed:.0f}s | "
            f"{len(results)} valid results | "
            f"Best fitness: {report.best_fitness:.3f}"
        )

        return report

    def print_report(self, report: OptimizerReport, top_n: int = 10):
        """Print optimization results to console"""
        print("\n" + "=" * 80)
        print("  PARAMETER OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"  Combinations tested: {report.total_combinations}")
        print(f"  Valid results: {len(report.results)}")
        print(f"  Time: {report.total_time_seconds:.0f}s")
        print()

        if not report.results:
            print("  No valid results found.")
            return

        print(f"  TOP {min(top_n, len(report.results))} PARAMETER SETS:")
        print("-" * 80)

        for i, r in enumerate(report.results[:top_n]):
            overfit_flag = " ⚠️ OVERFIT" if r.is_overfit else ""
            print(f"\n  #{i+1} | Fitness: {r.fitness_score:.3f}{overfit_flag}")
            print(f"  Parameters:")
            for k, v in r.params.items():
                print(f"    {k}: {v}")

            if r.train_result:
                print(f"  Train: WR={r.train_result.win_rate:.1f}% | "
                      f"R={r.train_result.total_r:+.1f} | "
                      f"PF={r.train_result.profit_factor:.2f} | "
                      f"Trades={r.train_result.total_trades}")
            if r.test_result:
                print(f"  Test:  WR={r.test_result.win_rate:.1f}% | "
                      f"R={r.test_result.total_r:+.1f} | "
                      f"PF={r.test_result.profit_factor:.2f} | "
                      f"Trades={r.test_result.total_trades}")

        # Best parameters recommendation
        print("\n" + "=" * 80)
        print("  🏆 RECOMMENDED PARAMETERS (for settings.yaml):")
        print("-" * 80)
        best = report.results[0]
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print()

    def save_report(self, report: OptimizerReport, filename: str = None) -> str:
        """Save optimization results to JSON"""
        if not filename:
            filename = f"optimization_{int(time.time())}.json"

        path = os.path.join(self.output_dir, filename)

        data = {
            'total_combinations': report.total_combinations,
            'time_seconds': round(report.total_time_seconds, 1),
            'best_params': report.best_params,
            'best_fitness': round(report.best_fitness, 4),
            'parameters_searched': [
                {'name': p.name, 'values': p.values, 'description': p.description}
                for p in report.parameter_specs
            ],
            'top_results': [
                {
                    'rank': i + 1,
                    'params': r.params,
                    'fitness': round(r.fitness_score, 4),
                    'is_overfit': r.is_overfit,
                    'train_wr': round(r.train_result.win_rate, 2) if r.train_result else 0,
                    'test_wr': round(r.test_result.win_rate, 2) if r.test_result else 0,
                    'train_r': round(r.train_result.total_r, 2) if r.train_result else 0,
                    'test_r': round(r.test_result.total_r, 2) if r.test_result else 0,
                }
                for i, r in enumerate(report.results[:20])
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Optimization report saved: {path}")
        return path

    # ── Fitness functions ─────────────────────────────────────

    def _calculate_fitness(self, train_result, test_result, method: str) -> float:
        """Calculate fitness score for a parameter combination.

        FIX AUDIT-7: When OOS (test) results have fewer than 5 trades, return
        0 fitness instead of falling back to in-sample (train) results.  The
        previous fallback directly selected overfitted parameter sets because
        fitness was computed from the data the parameters were tuned on.
        """

        # Require minimum OOS trades — never fall back to in-sample
        r = test_result if test_result and test_result.total_trades >= 5 else None

        if r is None:
            return 0.0

        if method == "sharpe_pf":
            # Composite: Sharpe * sqrt(PF) * consistency bonus
            sharpe = max(0, r.sharpe_ratio)
            pf = max(0, r.profit_factor)
            consistency = 1.0

            # Bonus for consistency between train and test
            if test_result and test_result.total_trades >= 5:
                train_wr = train_result.win_rate
                test_wr = test_result.win_rate
                if train_wr > 0:
                    diff = abs(train_wr - test_wr) / train_wr
                    consistency = max(0.3, 1.0 - diff)

            return sharpe * np.sqrt(max(0.01, pf)) * consistency

        elif method == "total_r":
            return r.total_r

        elif method == "win_rate":
            return r.win_rate / 100

        elif method == "profit_factor":
            return r.profit_factor

        else:
            return r.total_r

    # ── Grid generation ───────────────────────────────────────

    def _generate_grid(self, params: List[ParameterSpec]) -> List[tuple]:
        """Generate all combinations of parameter values"""
        value_lists = [p.values for p in params]
        return list(itertools.product(*value_lists))


# ── Pre-built parameter spaces ────────────────────────────────

def default_optimization_params() -> List[ParameterSpec]:
    """
    Default parameter space covering the most impactful config values.
    Tests ~150 combinations (manageable in a few minutes).
    """
    return [
        ParameterSpec(
            name="aggregator.min_confidence",
            values=[72, 75, 78, 80],
            description="Minimum confidence to publish signal"
        ),
        ParameterSpec(
            name="risk.min_rr",
            values=[1.5, 2.0, 2.5],
            description="Minimum risk:reward ratio"
        ),
        ParameterSpec(
            name="strategies.breakout.min_adx",
            values=[22, 25, 28, 32],
            description="ADX threshold for breakout strategy"
        ),
        ParameterSpec(
            name="strategies.breakout.volume_confirmation_mult",
            values=[1.5, 1.8, 2.2],
            description="Volume spike multiplier for breakouts"
        ),
    ]


def smc_optimization_params() -> List[ParameterSpec]:
    """Parameter space focused on SMC strategy tuning"""
    return [
        ParameterSpec(
            name="strategies.smc.swing_lookback",
            values=[5, 8, 10, 15],
            description="Swing point lookback period"
        ),
        ParameterSpec(
            name="strategies.smc.ob_max_age",
            values=[30, 50, 75, 100],
            description="Max age (bars) for valid order block"
        ),
        ParameterSpec(
            name="strategies.smc.fvg_min_size_pct",
            values=[0.001, 0.002, 0.003, 0.005],
            description="Minimum FVG size as percentage of price"
        ),
    ]


def reversal_optimization_params() -> List[ParameterSpec]:
    """Parameter space focused on reversal strategy"""
    return [
        ParameterSpec(
            name="strategies.reversal.rsi_oversold",
            values=[22, 25, 28, 30],
            description="RSI oversold threshold"
        ),
        ParameterSpec(
            name="strategies.reversal.rsi_overbought",
            values=[70, 72, 75, 78],
            description="RSI overbought threshold"
        ),
        ParameterSpec(
            name="strategies.reversal.bb_std",
            values=[2.0, 2.2, 2.5, 3.0],
            description="Bollinger Band standard deviation"
        ),
    ]


# Singleton
optimizer = ParameterOptimizer()
