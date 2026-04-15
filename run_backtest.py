#!/usr/bin/env python3
"""
TitanBot Pro — Backtest Runner (CLI)
======================================
Run backtests from the command line to measure strategy performance.

Usage:
  # Backtest all strategies on BTC/USDT (last 90 days)
  python run_backtest.py --symbol BTC/USDT --days 90

  # Backtest specific strategy
  python run_backtest.py --symbol ETH/USDT --strategy smc --days 180

  # Run parameter optimization
  python run_backtest.py --symbol BTC/USDT --optimize --days 120

  # Backtest multiple symbols
  python run_backtest.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --days 60

  # Load from local CSV instead of Binance API
  python run_backtest.py --symbol BTC/USDT --source csv --csv-dir ./data

  # Output detailed report
  python run_backtest.py --symbol BTC/USDT --days 90 --report markdown
"""

import argparse
import asyncio
import logging
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.loader import cfg
from backtester.engine import backtest_engine, BacktestEngine
from backtester.data_loader import BacktestDataLoader
from backtester.reporter import BacktestReporter
from backtester.optimizer import ParameterOptimizer, ParameterSpec


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")


# ── Strategy name mapping ───────────────────────────────────

STRATEGY_MAP = {
    'smc': 'SmartMoneyConcepts',
    'breakout': 'Breakout',
    'reversal': 'Reversal',
    'mean_reversion': 'MeanReversion',
    'price_action': 'PriceAction',
    'momentum': 'Momentum',
    'ichimoku': 'Ichimoku',
    'funding_arb': 'FundingArb',
}


def get_strategy_instances(names=None):
    """Get strategy instances by name or all if None"""
    from strategies.smc import SMCStrategy
    from strategies.breakout import BreakoutStrategy
    from strategies.reversal import ReversalStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.price_action import PriceActionStrategy
    from strategies.momentum import MomentumStrategy
    from strategies.ichimoku import IchimokuStrategy
    from strategies.funding_arb import FundingArbStrategy

    all_strats = {
        'smc': SMCStrategy,
        'breakout': BreakoutStrategy,
        'reversal': ReversalStrategy,
        'mean_reversion': MeanReversionStrategy,
        'price_action': PriceActionStrategy,
        'momentum': MomentumStrategy,
        'ichimoku': IchimokuStrategy,
        'funding_arb': FundingArbStrategy,
    }

    if names:
        selected = []
        for name in names:
            key = name.lower().replace(' ', '_')
            if key in all_strats:
                selected.append(all_strats[key]())
            else:
                logger.warning(f"Unknown strategy: {name}. Available: {list(all_strats.keys())}")
        return selected
    return [cls() for cls in all_strats.values()]


# ── Default optimization parameter space ─────────────────────

DEFAULT_PARAM_SPACE = [
    ParameterSpec(
        name="aggregator.min_confidence",
        values=[72, 75, 78, 82],
        description="Global minimum confidence threshold"
    ),
    ParameterSpec(
        name="risk.min_rr",
        values=[1.5, 2.0, 2.5, 3.0],
        description="Minimum risk-reward ratio"
    ),
    ParameterSpec(
        name="strategies.smc.structure.min_swing_lookback",
        values=[5, 8, 10, 15],
        description="SMC swing point lookback period"
    ),
    ParameterSpec(
        name="strategies.breakout.min_adx",
        values=[22, 25, 28, 32],
        description="Breakout ADX threshold"
    ),
    ParameterSpec(
        name="strategies.mean_reversion.z_score_threshold",
        values=[1.8, 2.0, 2.2, 2.5],
        description="Mean reversion Z-score trigger"
    ),
]


async def run_single_backtest(
    symbol: str,
    days: int,
    strategies=None,
    source: str = "binance",
    csv_dir: str = None,
    window_size: int = 200,
    max_hold_bars: int = 48,
):
    """Run backtest for a single symbol"""
    loader = BacktestDataLoader(data_dir=csv_dir)

    # Calculate date range
    from datetime import datetime, timedelta
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Determine required timeframes
    timeframes = ['4h', '1h', '15m']

    logger.info(f"Loading {symbol} data: {start_date} → {end_date} ({days} days)")

    try:
        ohlcv_data = await loader.load(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )
    except Exception as e:
        logger.error(f"Failed to load data for {symbol}: {e}")
        return None

    if not ohlcv_data:
        logger.error(f"No data loaded for {symbol}")
        return None

    for tf, candles in ohlcv_data.items():
        logger.info(f"  {tf}: {len(candles)} candles loaded")

    # Run backtest
    strat_instances = strategies or get_strategy_instances()
    logger.info(f"Running backtest with {len(strat_instances)} strategies...")

    engine = BacktestEngine()
    result = await engine.run(
        symbol=symbol,
        ohlcv_data=ohlcv_data,
        strategies=strat_instances,
        window_size=window_size,
        max_hold_bars=max_hold_bars,
    )

    return result


async def run_optimization(
    symbol: str,
    days: int,
    param_specs: list = None,
    source: str = "binance",
    csv_dir: str = None,
):
    """Run parameter optimization"""
    loader = BacktestDataLoader(data_dir=csv_dir)

    from datetime import datetime, timedelta
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    timeframes = ['4h', '1h', '15m']

    logger.info(f"Loading {symbol} data for optimization: {start_date} → {end_date}")
    ohlcv_data = await loader.load(
        symbol=symbol, timeframes=timeframes,
        start_date=start_date, end_date=end_date,
        source=source,
    )

    if not ohlcv_data:
        logger.error(f"No data loaded for {symbol}")
        return None

    params = param_specs or DEFAULT_PARAM_SPACE
    optimizer = ParameterOptimizer()

    logger.info(f"Starting optimization with {len(params)} parameters...")
    report = await optimizer.optimize(
        symbol=symbol,
        ohlcv_data=ohlcv_data,
        params=params,
        train_pct=0.7,
    )

    return report


async def main():
    parser = argparse.ArgumentParser(
        description="TitanBot Pro Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --symbol BTC/USDT --days 90
  python run_backtest.py --symbols BTC/USDT,ETH/USDT --days 60
  python run_backtest.py --symbol BTC/USDT --strategy smc --days 180
  python run_backtest.py --symbol BTC/USDT --optimize --days 120
  python run_backtest.py --symbol BTC/USDT --days 90 --report markdown
        """
    )

    parser.add_argument('--symbol', type=str, help='Symbol to backtest (e.g., BTC/USDT)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--days', type=int, default=90, help='Days of history (default: 90)')
    parser.add_argument('--strategy', type=str, help='Single strategy to test (e.g., smc, breakout)')
    parser.add_argument('--strategies', type=str, help='Comma-separated strategies')
    parser.add_argument('--source', choices=['binance', 'csv'], default='binance',
                        help='Data source (default: binance)')
    parser.add_argument('--csv-dir', type=str, help='Directory for CSV data files')
    parser.add_argument('--window', type=int, default=200, help='Lookback window (default: 200)')
    parser.add_argument('--max-hold', type=int, default=48, help='Max bars to hold (default: 48)')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--report', choices=['console', 'markdown', 'csv', 'json', 'all'],
                        default='console', help='Report format (default: console)')
    parser.add_argument('--output-dir', type=str, help='Output directory for reports')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible results (optimizer sampling)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate --days
    if args.days <= 0:
        logger.error(f"--days must be a positive integer (got {args.days})")
        sys.exit(1)
    if args.days > 3650:
        logger.error(f"--days cannot exceed 3650 (10 years). Got {args.days}.")
        sys.exit(1)

    # Apply random seed for reproducibility
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Parse symbols
    symbols = []
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [sym for s in args.symbols.split(',') if (sym := s.strip())]
    else:
        symbols = ['BTC/USDT']  # Default

    # Parse strategies
    strat_names = None
    if args.strategy:
        strat_names = [args.strategy]
    elif args.strategies:
        strat_names = [s.strip() for s in args.strategies.split(',')]

    strat_instances = get_strategy_instances(strat_names) if strat_names else None

    reporter = BacktestReporter(output_dir=args.output_dir or 'backtester/reports')

    start_time = time.time()

    if args.optimize:
        # ── Optimization mode ──────────────────────────────────
        logger.info("=" * 60)
        logger.info("  PARAMETER OPTIMIZATION MODE")
        logger.info("=" * 60)

        report = await run_optimization(
            symbol=symbols[0],
            days=args.days,
            source=args.source,
            csv_dir=args.csv_dir,
        )

        if report:
            logger.info(f"\n{'='*60}")
            logger.info(f"  OPTIMIZATION COMPLETE")
            logger.info(f"  Combinations tested: {report.total_combinations}")
            logger.info(f"  Best fitness: {report.best_fitness:.4f}")
            logger.info(f"  Best parameters:")
            for k, v in report.best_params.items():
                logger.info(f"    {k}: {v}")
            logger.info(f"  Time: {report.total_time_seconds:.1f}s")
            logger.info(f"{'='*60}")

            # Save optimization report
            try:
                import json
                report_path = os.path.join(
                    reporter.output_dir,
                    f"optimization_{symbols[0].replace('/', '_')}_{args.days}d.json"
                )
                os.makedirs(reporter.output_dir, exist_ok=True)
                with open(report_path, 'w') as f:
                    json.dump({
                        'symbol': symbols[0],
                        'days': args.days,
                        'total_combinations': report.total_combinations,
                        'best_fitness': report.best_fitness,
                        'best_params': report.best_params,
                        'time_seconds': report.total_time_seconds,
                    }, f, indent=2)
                logger.info(f"Report saved to: {report_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")

    else:
        # ── Backtest mode ──────────────────────────────────────
        logger.info("=" * 60)
        logger.info("  TITANBOT PRO BACKTESTER")
        logger.info(f"  Symbols: {', '.join(symbols)}")
        logger.info(f"  Period: {args.days} days")
        if strat_names:
            logger.info(f"  Strategies: {', '.join(strat_names)}")
        logger.info("=" * 60)

        all_results = []

        async def _run_and_report(symbol):
            """Run backtest for one symbol and return the result (or None)."""
            result = await run_single_backtest(
                symbol=symbol,
                days=args.days,
                strategies=strat_instances,
                source=args.source,
                csv_dir=args.csv_dir,
                window_size=args.window,
                max_hold_bars=args.max_hold,
            )
            if result and result.total_trades > 0:
                return result
            logger.warning(f"No trades generated for {symbol}")
            return None

        # Run all symbols concurrently (parallel data fetch + backtesting)
        raw_results = await asyncio.gather(
            *[_run_and_report(s) for s in symbols],
            return_exceptions=True,
        )

        for symbol, res in zip(symbols, raw_results):
            if isinstance(res, Exception):
                logger.error(f"Backtest failed for {symbol}: {res}")
                continue
            if res is None:
                continue
            all_results.append(res)

            # Console report always shown
            report_text = reporter.console_report(res)
            print(report_text)

            # Additional report formats
            if args.report in ('markdown', 'all'):
                try:
                    md_path = reporter.save_markdown(res)
                    logger.info(f"Markdown report: {md_path}")
                except Exception as e:
                    logger.error(f"Markdown report failed: {e}")

            if args.report in ('csv', 'all'):
                try:
                    csv_path = reporter.save_csv(res)
                    logger.info(f"CSV report: {csv_path}")
                except Exception as e:
                    logger.error(f"CSV report failed: {e}")

            if args.report in ('json', 'all'):
                try:
                    json_path = reporter.save_json(res)
                    logger.info(f"JSON report: {json_path}")
                except Exception as e:
                    logger.error(f"JSON report failed: {e}")

        # Multi-symbol summary
        if len(all_results) > 1:
            print("\n" + "=" * 60)
            print("  MULTI-SYMBOL SUMMARY")
            print("=" * 60)
            total_trades = sum(r.total_trades for r in all_results)
            total_wins = sum(r.wins for r in all_results)
            total_r = sum(r.total_r for r in all_results)
            total_bah = sum(getattr(r, 'buy_and_hold_r', 0.0) for r in all_results)
            avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

            for r in all_results:
                bah = getattr(r, 'buy_and_hold_r', 0.0)
                print(
                    f"  {r.symbol:12s} | "
                    f"{r.total_trades:3d} trades | "
                    f"WR: {r.win_rate:5.1f}% | "
                    f"R: {r.total_r:+6.1f} | "
                    f"B&H: {bah:+6.1f} | "
                    f"Alpha: {r.total_r - bah:+6.1f} | "
                    f"PF: {r.profit_factor:.2f}"
                )

            print("-" * 60)
            print(
                f"  {'TOTAL':12s} | "
                f"{total_trades:3d} trades | "
                f"WR: {avg_wr:5.1f}% | "
                f"R: {total_r:+6.1f} | "
                f"B&H: {total_bah:+6.1f} | "
                f"Alpha: {total_r - total_bah:+6.1f}"
            )

    elapsed = time.time() - start_time
    logger.info(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
