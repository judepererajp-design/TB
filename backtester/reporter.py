"""
TitanBot Pro — Backtest Reporter
==================================
Generates detailed performance reports from backtest results.
Outputs to:
  - Console (pretty table)
  - CSV (for spreadsheet analysis)
  - JSON (for programmatic use)
  - Markdown (for documentation)

Key metrics:
  - Win rate, Profit Factor, Sharpe Ratio
  - Max drawdown (in R)
  - Per-strategy breakdown
  - Equity curve data
  - Time-of-day performance heatmap
"""

import os
import json
import csv
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


# Cross-platform filename sanitiser. Replaces every character that isn't
# alphanumeric / dash / underscore / dot with an underscore so trading pairs
# like ``BTC/USDT:USDT`` (which appear under CCXT's unified symbology) don't
# create directory traversal artefacts or fail on Windows where ``:`` and
# ``\`` are reserved.
_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(s: str) -> str:
    return _FILENAME_UNSAFE.sub("_", str(s)).strip("_") or "report"


class BacktestReporter:
    """Generate reports from backtest results"""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "reports"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def console_report(self, result) -> str:
        """Build a formatted console report"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"  BACKTEST REPORT: {result.symbol}")
        lines.append(f"  Period: {result.start_date} → {result.end_date}")
        lines.append(f"  Timeframe: {result.timeframe}")
        lines.append("=" * 70)
        lines.append("")

        # ── Overall Statistics ────────────────────────────────
        lines.append("📊 OVERALL PERFORMANCE")
        lines.append(f"  Total Signals:    {result.total_signals}")
        lines.append(f"  Traded:           {result.total_trades}")
        lines.append(f"  Wins:             {result.wins}")
        lines.append(f"  Losses:           {result.losses}")
        lines.append(f"  Breakeven:        {result.breakevens}")
        lines.append(f"  Expired:          {result.expired}")
        lines.append("")

        # Key metrics
        lines.append("📈 KEY METRICS")
        lines.append(f"  Win Rate:         {result.win_rate:.1f}%")
        lines.append(f"  Avg R per Trade:  {result.avg_r:+.2f}R")
        lines.append(f"  Total R Earned:   {result.total_r:+.1f}R")
        lines.append(f"  Profit Factor:    {result.profit_factor:.2f}")
        lines.append(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}  (freq-adjusted)")
        lines.append(f"  Sortino Ratio:    {result.sortino_ratio:.2f}  (downside-only std)")
        lines.append(f"  Calmar Ratio:     {result.calmar_ratio:.2f}  (total_r / max_dd)")
        lines.append(f"  Max Drawdown:     {result.max_drawdown_r:.1f}R")
        lines.append(f"  Avg MAE:          {result.avg_mae_r:.2f}R  (avg worst loss intra-trade)")
        lines.append(f"  Avg MFE:          {result.avg_mfe_r:.2f}R  (avg best gain intra-trade)")
        lines.append(f"  Best Trade:       {result.best_trade_r:+.1f}R")
        lines.append(f"  Worst Trade:      {result.worst_trade_r:+.1f}R")
        lines.append(f"  Avg Bars to Exit: {result.avg_bars_to_outcome:.0f}")
        lines.append(f"  Max Win Streak:   {result.max_consecutive_wins}")
        lines.append(f"  Max Loss Streak:  {result.max_consecutive_losses}")
        # Buy-and-hold benchmark comparison
        bah = getattr(result, 'buy_and_hold_r', 0.0)
        alpha = result.total_r - bah
        lines.append(f"  Buy & Hold:       {bah:+.1f}R  (passive benchmark)")
        lines.append(f"  Alpha vs B&H:     {alpha:+.1f}R  ({'✅ outperforming' if alpha >= 0 else '❌ underperforming'})")
        lines.append("")

        # Quality grade
        grade = self._calculate_grade(result)
        lines.append(f"  GRADE: {grade}")
        lines.append("")

        # ── Per-Strategy Breakdown ────────────────────────────
        if result.strategy_breakdown:
            lines.append("📋 PER-STRATEGY BREAKDOWN")
            lines.append(f"  {'Strategy':<25} {'Trades':>7} {'WR%':>7} "
                         f"{'Total R':>9} {'Avg R':>7}")
            lines.append("  " + "-" * 60)

            for strat, stats in sorted(
                result.strategy_breakdown.items(),
                key=lambda x: x[1]['total_r'],
                reverse=True
            ):
                lines.append(
                    f"  {strat:<25} {stats['total']:>7} "
                    f"{stats['win_rate']:>6.1f}% "
                    f"{stats['total_r']:>+8.1f}R "
                    f"{stats['avg_r']:>+6.2f}R"
                )
            lines.append("")

        # ── Equity Curve Summary ──────────────────────────────
        # FIX: previously included EXPIRED trades (filter ``outcome != "PENDING"``)
        # which used a mark-to-market pnl_r — that mismatched the headline
        # metrics block above (which excludes EXPIRED) and made the
        # sparkline tell a different story than the numbers next to it.
        if result.trades:
            r_values = [t.pnl_r for t in result.trades
                        if t.outcome in ("WIN", "LOSS", "BE")]
            if r_values:
                cumulative = np.cumsum(r_values)
                lines.append("📉 EQUITY CURVE (R)")
                # Simple ASCII sparkline
                if len(cumulative) > 1:
                    min_r = min(cumulative)
                    max_r = max(cumulative)
                    range_r = max_r - min_r
                    if range_r > 0:
                        spark = ""
                        chars = " ▁▂▃▄▅▆▇█"
                        step = max(1, len(cumulative) // 50)
                        for i in range(0, len(cumulative), step):
                            idx = int((cumulative[i] - min_r) / range_r * 8)
                            idx = min(max(idx, 0), 8)
                            spark += chars[idx]
                        lines.append(f"  {spark}")
                        lines.append(f"  Low: {min_r:+.1f}R | "
                                     f"High: {max_r:+.1f}R | "
                                     f"Final: {cumulative[-1]:+.1f}R")
                lines.append("")

        # ── Monte Carlo Confidence Intervals ──────────────────
        mc = getattr(result, 'monte_carlo', {})
        if mc:
            lines.append("🎲 MONTE CARLO (1000 resamples)")
            wr_mc = mc.get('win_rate', {})
            tr_mc = mc.get('total_r', {})
            dd_mc = mc.get('max_drawdown', {})
            lines.append(
                f"  Win Rate:   5th={wr_mc.get('p5', 0):.1f}% | "
                f"50th={wr_mc.get('p50', 0):.1f}% | "
                f"95th={wr_mc.get('p95', 0):.1f}%"
            )
            lines.append(
                f"  Total R:    5th={tr_mc.get('p5', 0):+.1f}R | "
                f"50th={tr_mc.get('p50', 0):+.1f}R | "
                f"95th={tr_mc.get('p95', 0):+.1f}R"
            )
            lines.append(
                f"  Max DD:     5th={dd_mc.get('p5', 0):.1f}R | "
                f"50th={dd_mc.get('p50', 0):.1f}R | "
                f"95th={dd_mc.get('p95', 0):.1f}R"
            )
            lines.append("")

        return "\n".join(lines)

    def save_csv(self, result, filename: str = None) -> str:
        """Save trade list to CSV"""
        if not filename:
            sym = _safe_name(result.symbol)
            sd  = _safe_name(result.start_date)
            ed  = _safe_name(result.end_date)
            filename = f"backtest_{sym}_{sd}_{ed}.csv"
        else:
            filename = _safe_name(filename)

        path = os.path.join(self.output_dir, filename)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'idx', 'bar', 'symbol', 'strategy', 'direction',
                'confidence', 'entry', 'stop_loss', 'tp1', 'tp2',
                'rr_ratio', 'outcome', 'exit_price', 'pnl_r',
                'bars_held', 'timestamp'
            ])
            for t in result.trades:
                writer.writerow([
                    t.signal_idx, t.bar_idx, t.symbol, t.strategy,
                    t.direction, f"{t.confidence:.0f}", f"{t.entry_price:.6f}",
                    f"{t.stop_loss:.6f}", f"{t.tp1:.6f}", f"{t.tp2:.6f}",
                    f"{t.rr_ratio:.2f}", t.outcome, f"{t.exit_price:.6f}",
                    f"{t.pnl_r:+.2f}", t.bars_to_outcome, t.timestamp
                ])

        logger.info(f"CSV saved: {path}")
        return path

    def save_json(self, result, filename: str = None) -> str:
        """Save full results to JSON"""
        if not filename:
            sym = _safe_name(result.symbol)
            sd  = _safe_name(result.start_date)
            ed  = _safe_name(result.end_date)
            filename = f"backtest_{sym}_{sd}_{ed}.json"
        else:
            filename = _safe_name(filename)

        path = os.path.join(self.output_dir, filename)

        # `profit_factor` may be inf when there are zero losses — JSON cannot
        # serialise inf, so convert to a sentinel string in that case.
        pf = result.profit_factor
        pf_serial = None if pf in (float('inf'), float('-inf')) else round(pf, 3)

        data = {
            'symbol': result.symbol,
            'strategy': result.strategy,
            'timeframe': result.timeframe,
            'period': f"{result.start_date} to {result.end_date}",
            'summary': {
                'total_signals': result.total_signals,
                'total_trades': result.total_trades,
                'invalidated_signals': getattr(result, 'invalidated_signals', 0),
                'wins': result.wins,
                'losses': result.losses,
                'breakevens': result.breakevens,
                'expired': result.expired,
                'win_rate': round(result.win_rate, 2),
                'avg_r': round(result.avg_r, 3),
                'total_r': round(result.total_r, 2),
                'profit_factor': pf_serial,
                'sharpe_ratio': round(result.sharpe_ratio, 3),
                # FIX: previously omitted from JSON summary even though
                # console / markdown reports already published them.
                'sortino_ratio': round(getattr(result, 'sortino_ratio', 0.0) or 0.0, 3),
                'calmar_ratio':  round(getattr(result, 'calmar_ratio',  0.0) or 0.0, 3),
                'avg_mae_r':     round(getattr(result, 'avg_mae_r', 0.0) or 0.0, 3),
                'avg_mfe_r':     round(getattr(result, 'avg_mfe_r', 0.0) or 0.0, 3),
                'avg_bars_to_outcome': round(getattr(result, 'avg_bars_to_outcome', 0.0) or 0.0, 2),
                'max_drawdown_r': round(result.max_drawdown_r, 2),
                'max_consecutive_wins': result.max_consecutive_wins,
                'max_consecutive_losses': result.max_consecutive_losses,
                'buy_and_hold_r': round(getattr(result, 'buy_and_hold_r', 0.0), 2),
                'alpha_vs_bah': round(result.total_r - getattr(result, 'buy_and_hold_r', 0.0), 2),
                'monte_carlo': getattr(result, 'monte_carlo', {}),
                'grade': self._calculate_grade(result),
            },
            'strategy_breakdown': {
                k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in result.strategy_breakdown.items()
            },
            'regime_breakdown': {
                k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in getattr(result, 'regime_breakdown', {}).items()
            },
            'trades': [
                {
                    'idx': t.signal_idx,
                    'strategy': t.strategy,
                    'direction': t.direction,
                    'confidence': t.confidence,
                    'entry': t.entry_price,
                    'outcome': t.outcome,
                    'pnl_r': round(t.pnl_r, 3),
                    'bars': t.bars_to_outcome,
                }
                for t in result.trades
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"JSON saved: {path}")
        return path

    def save_markdown(self, result, filename: str = None) -> str:
        """Save a Markdown report"""
        if not filename:
            sym = _safe_name(result.symbol)
            sd  = _safe_name(result.start_date)
            ed  = _safe_name(result.end_date)
            filename = f"backtest_{sym}_{sd}_{ed}.md"
        else:
            filename = _safe_name(filename)

        path = os.path.join(self.output_dir, filename)

        lines = []
        lines.append(f"# Backtest Report: {result.symbol}")
        lines.append(f"**Period:** {result.start_date} → {result.end_date} | **TF:** {result.timeframe}")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Signals | {result.total_signals} |")
        lines.append(f"| Traded | {result.total_trades} |")
        lines.append(f"| Win Rate | {result.win_rate:.1f}% |")
        lines.append(f"| Total R | {result.total_r:+.1f}R |")
        lines.append(f"| Profit Factor | {result.profit_factor:.2f} |")
        lines.append(f"| Sharpe | {result.sharpe_ratio:.2f} |")
        lines.append(f"| Sortino | {result.sortino_ratio:.2f} |")
        lines.append(f"| Calmar | {result.calmar_ratio:.2f} |")
        lines.append(f"| Max Drawdown | {result.max_drawdown_r:.1f}R |")
        bah = getattr(result, 'buy_and_hold_r', 0.0)
        lines.append(f"| Buy & Hold | {bah:+.1f}R |")
        lines.append(f"| Alpha vs B&H | {result.total_r - bah:+.1f}R |")
        lines.append(f"| Grade | {self._calculate_grade(result)} |")
        lines.append("")

        # Monte Carlo section
        mc = getattr(result, 'monte_carlo', {})
        if mc:
            lines.append("## Monte Carlo (1000 resamples)")
            lines.append("| Metric | 5th pct | 50th pct | 95th pct |")
            lines.append("|--------|---------|----------|----------|")
            wr_mc = mc.get('win_rate', {})
            tr_mc = mc.get('total_r', {})
            dd_mc = mc.get('max_drawdown', {})
            lines.append(
                f"| Win Rate | {wr_mc.get('p5', 0):.1f}% | "
                f"{wr_mc.get('p50', 0):.1f}% | {wr_mc.get('p95', 0):.1f}% |"
            )
            lines.append(
                f"| Total R | {tr_mc.get('p5', 0):+.1f}R | "
                f"{tr_mc.get('p50', 0):+.1f}R | {tr_mc.get('p95', 0):+.1f}R |"
            )
            lines.append(
                f"| Max Drawdown | {dd_mc.get('p5', 0):.1f}R | "
                f"{dd_mc.get('p50', 0):.1f}R | {dd_mc.get('p95', 0):.1f}R |"
            )
            lines.append("")

        if result.strategy_breakdown:
            lines.append("## Strategy Breakdown")
            lines.append("| Strategy | Trades | Win Rate | Total R | Avg R |")
            lines.append("|----------|--------|----------|---------|-------|")
            for strat, stats in sorted(
                result.strategy_breakdown.items(),
                key=lambda x: x[1]['total_r'], reverse=True
            ):
                lines.append(
                    f"| {strat} | {stats['total']} | "
                    f"{stats['win_rate']:.1f}% | "
                    f"{stats['total_r']:+.1f}R | "
                    f"{stats['avg_r']:+.2f}R |"
                )
            lines.append("")

        with open(path, 'w') as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown saved: {path}")
        return path

    def compare_runs(self, results: List, labels: List[str] = None,
                     print_output: bool = True) -> str:
        """
        Compare multiple backtest runs side by side.
        Useful for optimizer output.

        ``print_output`` defaults to True for backward compatibility; set to
        False to obtain the formatted comparison without writing to stdout
        (useful in tests / programmatic callers).
        """
        if not results:
            return ""

        if not labels:
            labels = [f"Run {i+1}" for i in range(len(results))]

        lines = []
        lines.append("=" * 80)
        lines.append("  BACKTEST COMPARISON")
        lines.append("=" * 80)
        lines.append("")

        header = f"{'Metric':<20}"
        for label in labels:
            header += f" {label:>12}"
        lines.append(header)
        lines.append("-" * (20 + 13 * len(labels)))

        metrics = [
            ('Win Rate', lambda r: f"{r.win_rate:.1f}%"),
            ('Total R', lambda r: f"{r.total_r:+.1f}R"),
            ('Avg R', lambda r: f"{r.avg_r:+.2f}R"),
            ('Profit Factor', lambda r: f"{r.profit_factor:.2f}"),
            ('Sharpe', lambda r: f"{r.sharpe_ratio:.2f}"),
            ('Sortino', lambda r: f"{getattr(r, 'sortino_ratio', 0.0):.2f}"),
            ('Calmar', lambda r: f"{getattr(r, 'calmar_ratio', 0.0):.2f}"),
            ('Max DD', lambda r: f"{r.max_drawdown_r:.1f}R"),
            ('Trades', lambda r: f"{r.total_trades}"),
            ('Max Win Streak', lambda r: f"{r.max_consecutive_wins}"),
            ('Max Loss Streak', lambda r: f"{r.max_consecutive_losses}"),
        ]

        for name, fn in metrics:
            row = f"{name:<20}"
            for r in results:
                row += f" {fn(r):>12}"
            lines.append(row)

        lines.append("")
        report = "\n".join(lines)
        if print_output:
            print(report)
        return report

    def _calculate_grade(self, result) -> str:
        """Grade the backtest result.

        FIX: the previous scoring scale could award A+ to a strategy with
        negative ``total_r`` (e.g. one good MC tail + high WR + high Sortino).
        We now require positive ``total_r`` AND positive ``avg_r`` for any
        positive grade, and cap A+/A on the size of the worst drawdown so
        a 30R-drawdown strategy can never grade A+ regardless of Sharpe.
        """
        # Hard floor — losing strategies cannot grade above D.
        if result.total_r <= 0 or result.avg_r <= 0:
            return "🔴 D (Poor — Do Not Trade)"

        score = 0

        # Win rate contribution
        if result.win_rate >= 70:
            score += 3
        elif result.win_rate >= 60:
            score += 2
        elif result.win_rate >= 50:
            score += 1

        # Profit factor (cap inf)
        pf = result.profit_factor if result.profit_factor != float('inf') else 5.0
        if pf >= 2.0:
            score += 3
        elif pf >= 1.5:
            score += 2
        elif pf >= 1.2:
            score += 1

        # Average R
        if result.avg_r >= 0.5:
            score += 2
        elif result.avg_r >= 0.2:
            score += 1

        # Drawdown penalty
        if result.max_drawdown_r > 10:
            score -= 2
        elif result.max_drawdown_r > 5:
            score -= 1

        # Sharpe — frequency-adjusted (reliable now)
        if result.sharpe_ratio >= 2.0:
            score += 2
        elif result.sharpe_ratio >= 1.0:
            score += 1

        # Sortino bonus — rewards strategies with low downside deviation
        sortino = getattr(result, 'sortino_ratio', 0.0) or 0.0
        if sortino >= 2.5:
            score += 2
        elif sortino >= 1.5:
            score += 1

        # Calmar penalty — penalise high-drawdown strategies regardless of Sharpe
        calmar = getattr(result, 'calmar_ratio', 0.0) or 0.0
        if 0 < calmar < 0.5:
            score -= 1  # Return not worth the drawdown

        # Hard caps so headline grade reflects true tradeability.
        if result.max_drawdown_r > 15 and score >= 9:
            score = 8        # cap at A
        if result.max_drawdown_r > 25 and score >= 7:
            score = 6        # cap at B+

        if score >= 9:
            return "⭐ A+ (Exceptional)"
        elif score >= 7:
            return "🟢 A (Excellent)"
        elif score >= 5:
            return "🟢 B+ (Good)"
        elif score >= 3:
            return "🟡 B (Acceptable)"
        elif score >= 1:
            return "🟠 C (Needs Work)"
        else:
            return "🔴 D (Poor — Do Not Trade)"


# Singleton
reporter = BacktestReporter()
