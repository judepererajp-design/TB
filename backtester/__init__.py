"""
TitanBot Pro — Backtesting Framework
=======================================
Replays historical OHLCV data through the full strategy pipeline
to measure actual signal quality.

Modules:
  - data_loader: Load OHLCV from Binance or local CSV files
  - engine: Replay engine that simulates the scanning pipeline
  - reporter: Performance reports (win rate, PF, Sharpe, drawdown)
  - optimizer: Grid search over config parameters
"""
