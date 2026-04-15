from types import SimpleNamespace

from backtester.reporter import BacktestReporter


def _result():
    return SimpleNamespace(
        symbol="BTC/USDT",
        start_date="2026-01-22",
        end_date="2026-04-14",
        timeframe="1h",
        total_signals=10,
        total_trades=8,
        wins=3,
        losses=3,
        breakevens=1,
        expired=1,
        win_rate=37.5,
        avg_r=0.12,
        total_r=1.0,
        profit_factor=1.2,
        sharpe_ratio=0.8,
        sortino_ratio=1.1,
        calmar_ratio=0.4,
        max_drawdown_r=2.0,
        avg_mae_r=0.5,
        avg_mfe_r=0.7,
        best_trade_r=2.5,
        worst_trade_r=-1.0,
        avg_bars_to_outcome=6,
        max_consecutive_wins=2,
        max_consecutive_losses=2,
        buy_and_hold_r=0.5,
        strategy_breakdown={},
        trades=[],
        monte_carlo={},
    )


def test_console_report_returns_text_without_printing(capsys):
    reporter = BacktestReporter(output_dir="/tmp/titanbot-backtest-reports")

    report = reporter.console_report(_result())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "BACKTEST REPORT: BTC/USDT" in report
    assert "Win Rate:         37.5%" in report
