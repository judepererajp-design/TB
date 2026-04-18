"""
TitanBot Pro — Backtest Engine
================================
Replays historical OHLCV data through each strategy exactly as the
live engine would, but without Telegram, database, or exchange connections.

The backtest engine:
  1. Slides a window across historical candles
  2. At each bar, feeds the window to strategy.analyze()
  3. Collects generated signals
  4. Simulates trade outcomes by checking future bars
  5. Records results for reporting

This is a SIGNAL backtest, not a portfolio backtest. It measures
whether the bot's signals (entry, SL, TP) would have been profitable,
not whether a specific position sizing strategy would have worked.

ACCURACY WARNING (v3.0.0):
  Backtest win rates will be HIGHER than live win rates because the
  following live-engine filters are not applied during backtesting:
    - time_filter (session/killzone gating)
    - signal_clarity scorer
    - no_trade_zones engine
    - equilibrium zone blocking
    - sector rotation weights
    - chop mid-range penalty
    - full probability + alpha model pipeline
    - HTF guardrail (partially applied in system_backtest=True mode only)

  The following signal quality gates ARE applied to prevent over-trading:
    - Per-bar signal limit (best 1 signal per bar by confidence)
    - Direction cooldown (min 6 bars between same-direction signals)
    - Minimum confluence (require ≥2 confluence factors)
    - Stricter confidence floor (75 vs live's 72)

  Use system_backtest=True for the most accurate comparison.
  Treat standard backtest numbers as an upper bound, not expected live performance.
"""

import asyncio
import logging
import time
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from config.loader import cfg
from config.constants import Backtester as BTC, STRATEGY_KEY_MAP

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A simulated trade from a backtested signal"""
    signal_idx: int
    bar_idx: int
    symbol: str
    strategy: str
    direction: str         # LONG | SHORT
    confidence: float

    entry_low: float
    entry_high: float
    entry_price: float     # Mid of entry zone (simulated)
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    rr_ratio: float

    # Outcome
    outcome: str = "PENDING"    # WIN | LOSS | BE | EXPIRED
    exit_price: float = 0.0
    pnl_r: float = 0.0         # P&L in R multiples (accounts for TP1 partial)
    bars_to_outcome: int = 0
    exit_bar_idx: int = 0
    confluence: List[str] = field(default_factory=list)

    # BUG-NEW-6 FIX: track partial TP1 close separately so reporter can
    # compute accurate avg_r that matches live trading behaviour.
    tp1_partial_r: float = 0.0   # R captured by 50% close at TP1 (0 if TP1 not hit)
    tp1_hit: bool = False         # Was TP1 reached at any point?

    # Timing
    timestamp: int = 0

    # BT-4: Regime at time of signal (inferred from price structure)
    regime_at_signal: str = 'UNKNOWN'  # BULL_TREND | BEAR_TREND | CHOPPY | VOLATILE


@dataclass
class BacktestResult:
    """Aggregated backtest results"""
    symbol: str
    strategy: str
    timeframe: str
    start_date: str
    end_date: str

    # Overall
    total_signals: int = 0      # Filled trades (any non-PENDING outcome)
    total_trades: int = 0       # Decided trades (WIN | LOSS | BE)
    invalidated_signals: int = 0  # Signals stopped out before entry was reached
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    expired: int = 0

    # Performance
    win_rate: float = 0.0
    avg_r: float = 0.0          # Average R per trade (TP1-partial-close aware)
    total_r: float = 0.0        # Total R earned
    profit_factor: float = 0.0
    max_drawdown_r: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0  # BUG-NEW-10 FIX: penalises downside deviation only
    calmar_ratio: float = 0.0   # BUG-NEW-10 FIX: annualised return / max drawdown
    avg_mae_r: float = 0.0      # BUG-NEW-10 FIX: avg Maximum Adverse Excursion per trade
    avg_mfe_r: float = 0.0      # BUG-NEW-10 FIX: avg Maximum Favourable Excursion per trade
    avg_bars_to_outcome: float = 0.0
    buy_and_hold_r: float = 0.0     # Equivalent R if price was bought at start and held to end

    # Best/Worst
    best_trade_r: float = 0.0
    worst_trade_r: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    # All trades for detailed analysis
    trades: List[BacktestTrade] = field(default_factory=list)

    # Per-strategy breakdown
    strategy_breakdown: Dict = field(default_factory=dict)

    # BT-4: Per-regime breakdown {regime: {trades, win_rate, total_r, avg_r}}
    regime_breakdown: Dict = field(default_factory=dict)

    # BT-1: Monte Carlo confidence intervals
    monte_carlo: Dict = field(default_factory=dict)

    # Accuracy warning surfaced with the result (populated by system_backtest mode)
    accuracy_note: str = ""


class BacktestEngine:
    """
    Replays historical data through strategies.

    FIX #10: Added realistic slippage + commission modeling.
    Without friction, backtest results overestimate net EV by ~15% at typical
    Binance taker fees (0.04% per side = 0.08% round-trip) plus market impact.
    Defaults mirror Binance Futures taker rates for a small account.
    """

    # Friction model defaults (override via constructor kwargs)
    DEFAULT_COMMISSION_PCT = BTC.DEFAULT_COMMISSION_PCT
    DEFAULT_SLIPPAGE_PCT   = BTC.DEFAULT_SLIPPAGE_PCT
    # Combined round-trip friction: 2 × (commission + slippage) = ~0.14%

    def __init__(
        self,
        commission_pct: float = DEFAULT_COMMISSION_PCT,
        slippage_pct:   float = DEFAULT_SLIPPAGE_PCT,
    ):
        self._strategies = []
        self.commission_pct = commission_pct
        self.slippage_pct   = slippage_pct
        # Round-trip cost in R-equivalent (applied per trade result)
        # Computed dynamically in _simulate_trade from actual stop distance

    async def run(
        self,
        symbol: str,
        ohlcv_data: Dict[str, List],
        strategies: List = None,
        window_size: int = BTC.CANDLE_WINDOW,
        max_hold_bars: int = BTC.MAX_HOLD_BARS,
        config_overrides: Dict = None,
        system_backtest: bool = False,
        oos_split: float = 0.0,
    ) -> "BacktestResult | Dict":
        """
        Run a full backtest on historical data.

        Args:
            symbol: Trading pair
            ohlcv_data: Dict of {timeframe: candles}
            strategies: List of strategy instances (or None for all)
            window_size: Number of bars visible to strategy at each step
            max_hold_bars: Max bars before signal expires
            config_overrides: Override config values for this run
            system_backtest: If True, run the full live filter stack
            oos_split: If > 0 (e.g. 0.20), reserve the LAST oos_split fraction of data
                       as a mandatory out-of-sample hold-out. The in-sample result is
                       used for development; the OOS result is the true unbiased estimate.
                       Returns {"in_sample": BacktestResult, "out_of_sample": BacktestResult}
                       instead of a single BacktestResult when oos_split > 0.
        """

        if strategies is None:
            strategies = self._get_default_strategies()

        # Apply config overrides for this run
        original_config = {}
        if config_overrides:
            original_config = self._apply_overrides(config_overrides)

        try:
            if oos_split > 0:
                # Split each timeframe by *timestamp* (not by per-TF bar count)
                # so IS/OOS boundaries align across timeframes. The previous
                # implementation cut every TF at ``len*(1-oos_split)`` which
                # let HTF data leak from OOS into IS whenever bar densities
                # differed (e.g. a 4h IS slice could contain bars that
                # closed *after* the 1h IS cutoff).
                primary_tf = '1h' if '1h' in ohlcv_data else next(iter(ohlcv_data))
                primary_candles = ohlcv_data[primary_tf]
                if len(primary_candles) < window_size + max_hold_bars + 10:
                    # Not enough data to OOS-split sensibly — fall through to
                    # a single-window run rather than producing a stub OOS
                    # slice that violates min-trade thresholds downstream.
                    logger.warning(
                        "OOS split requested but %s primary timeframe has only %d bars; "
                        "running without OOS hold-out.",
                        primary_tf, len(primary_candles),
                    )
                    result = await self._execute(
                        symbol, ohlcv_data, strategies,
                        window_size, max_hold_bars, system_backtest=system_backtest,
                    )
                else:
                    primary_split_idx = max(
                        window_size + max_hold_bars,
                        int(len(primary_candles) * (1 - oos_split)),
                    )
                    split_ts = primary_candles[primary_split_idx][0]

                    is_data: Dict[str, List] = {}
                    oos_data: Dict[str, List] = {}
                    for tf, candles in ohlcv_data.items():
                        is_data[tf]  = [c for c in candles if c[0] <  split_ts]
                        oos_data[tf] = [c for c in candles if c[0] >= split_ts]

                    is_result  = await self._execute(symbol, is_data,  strategies, window_size, max_hold_bars, system_backtest=system_backtest)
                    oos_result = await self._execute(symbol, oos_data, strategies, window_size, max_hold_bars, system_backtest=system_backtest)

                    logger.info(
                        f"IS/OOS split @ts={split_ts}: "
                        f"IS={len(is_data[primary_tf])} bars ({(1-oos_split)*100:.0f}%) | "
                        f"OOS={len(oos_data[primary_tf])} bars ({oos_split*100:.0f}%) | "
                        f"IS avg_r={is_result.avg_r:+.3f} OOS avg_r={oos_result.avg_r:+.3f}"
                    )

                    # Overfitting flag: compare Sharpe decay (more reliable than WR)
                    decay = None
                    if is_result.sharpe_ratio > 0 and oos_result.sharpe_ratio > 0:
                        decay = oos_result.sharpe_ratio / is_result.sharpe_ratio
                        if decay < 0.5:
                            logger.warning(
                                f"⚠️ Overfitting signal: IS Sharpe={is_result.sharpe_ratio:.2f} "
                                f"decays to OOS Sharpe={oos_result.sharpe_ratio:.2f} "
                                f"(decay={decay:.2f} — strategies may be curve-fit)"
                            )

                    result = {
                        "in_sample": is_result,
                        "out_of_sample": oos_result,
                        "oos_split": oos_split,
                        "overfitting_decay": round(decay, 3) if decay is not None else None,
                    }
            else:
                result = await self._execute(symbol, ohlcv_data, strategies,
                                             window_size, max_hold_bars, system_backtest=system_backtest)
        finally:
            if original_config:
                self._apply_overrides(original_config)

        return result

    async def _execute(
        self,
        symbol: str,
        ohlcv_data: Dict[str, List],
        strategies: List,
        window_size: int,
        max_hold_bars: int,
        system_backtest: bool = False,
    ) -> BacktestResult:
        """Core backtest execution. system_backtest=True enables full live filter stack."""

        # PHASE 2 FIX (BACKTEST-GAP): warm up HTF guardrail for system backtest mode
        _htf_ref = None
        if system_backtest:
            try:
                from analyzers.htf_guardrail import htf_guardrail as _htf_bt
                await _htf_bt.warm_up()
                _htf_ref = _htf_bt
            except Exception as _e:
                logger.warning(f"SystemBacktest: HTF warm_up failed: {_e}")

        # Use primary timeframe for bar-by-bar iteration
        primary_tf = '1h' if '1h' in ohlcv_data else list(ohlcv_data.keys())[0]
        candles = ohlcv_data[primary_tf]

        if len(candles) < window_size + max_hold_bars + 10:
            logger.warning(
                f"Not enough data for {symbol}: {len(candles)} candles, "
                f"need {window_size + max_hold_bars + 10}"
            )
            return BacktestResult(
                symbol=symbol, strategy="ALL",
                timeframe=primary_tf, start_date="", end_date=""
            )

        all_trades = []
        signal_idx = 0

        # ── Signal quality gates (prevent over-trading) ──────────────────
        # Track last bar index a signal was taken for each direction to
        # enforce a cooldown between same-direction signals.
        _last_signal_bar: Dict[str, int] = {}   # "LONG"/"SHORT" -> last bar_idx

        start_ts = candles[window_size][0] if len(candles) > window_size else candles[0][0]
        end_ts = candles[-1][0]

        logger.info(
            f"Backtesting {symbol} | {len(candles)} bars | "
            f"{len(strategies)} strategies | window={window_size}"
        )

        # Slide window across data
        for bar_idx in range(window_size, len(candles) - max_hold_bars):
            # Build the OHLCV dict as strategies would see it
            window_dict = {}
            for tf, tf_candles in ohlcv_data.items():
                # Find candles up to current timestamp
                current_ts = candles[bar_idx][0]
                visible = [c for c in tf_candles if c[0] <= current_ts]
                if len(visible) > window_size:
                    visible = visible[-window_size:]
                if visible:
                    window_dict[tf] = visible

            if not window_dict:
                continue

            # ── Collect all candidate signals for this bar ────────────────
            bar_candidates = []

            # Run each strategy
            for strategy in strategies:
                try:
                    strat_key = STRATEGY_KEY_MAP.get(
                        strategy.name,
                        strategy.name.lower().replace(' ', '_'),
                    )
                    if not cfg.is_strategy_enabled(strat_key):
                        continue

                    signal = await strategy.analyze(symbol, window_dict)
                    if not signal:
                        continue

                    # ── PHASE 2 FIX (BACKTEST-GAP): partial live filter stack ──────
                    # NOTE (AUDIT FIX #5): This applies 3 of ~8 live-engine gates:
                    #   ✅ HTF guardrail hard block
                    #   ✅ Equilibrium zone blocking
                    #   ✅ Regime filter (confidence adjustment + blocking)
                    # Still missing (see ACCURACY WARNING at top of file):
                    #   ❌ time_filter (session/killzone gating)
                    #   ❌ signal_clarity scorer
                    #   ❌ no_trade_zones engine
                    #   ❌ sector rotation weights
                    #   ❌ chop mid-range penalty
                    #   ❌ full probability + alpha model pipeline
                    # Backtest results with system_backtest=True are closer to live
                    # but still an upper bound. Treat as optimistic estimate.
                    if system_backtest:
                        from signals.signal_pipeline import (
                            apply_htf_hard_gate,
                            apply_equilibrium_gate,
                            apply_regime_filter_gate,
                            direction_str,
                        )
                        _filtered = False
                        _dir = direction_str(signal.direction)

                        # 1. HTF guardrail hard block
                        if _htf_ref is not None:
                            gate = apply_htf_hard_gate(
                                _htf_ref, _dir, signal.confidence, signal.strategy,
                            )
                            if gate.blocked:
                                logger.debug("BT HTF block: %s %s [%s]", symbol, _dir, signal.strategy)
                                _filtered = True

                        # 2. EQ zone check
                        if not _filtered:
                            try:
                                from analyzers.equilibrium import eq_analyzer
                                _entry_mid = (signal.entry_low + signal.entry_high) / 2
                                gate = apply_equilibrium_gate(eq_analyzer, _entry_mid, _dir)
                                if gate.blocked:
                                    logger.debug("BT EQ block: %s %s", symbol, _dir)
                                    _filtered = True
                                elif gate.adjusted_confidence is not None and gate.adjusted_confidence != 1.0:
                                    signal.confidence *= gate.adjusted_confidence
                            except Exception:
                                pass

                        # 3. Regime filter
                        if not _filtered:
                            try:
                                from analyzers.regime import regime_analyzer as _ra
                                gate = apply_regime_filter_gate(
                                    signal.confidence, _dir, signal.strategy,
                                    _ra.regime.value, chop_strength=_ra.chop_strength,
                                )
                                if gate.blocked:
                                    _filtered = True
                                elif gate.adjusted_confidence is not None:
                                    signal.confidence = gate.adjusted_confidence
                            except Exception:
                                pass

                        if _filtered:
                            continue

                    # 4. Confidence gate (stricter floor for backtest realism)
                    _min_conf = max(
                        BTC.MIN_CONFIDENCE_BACKTEST,
                        cfg.aggregator.get('min_confidence', 72),
                    )
                    if signal.confidence < _min_conf:
                        continue

                    # 5. Minimum confluence gate — reject single-factor signals
                    if len(getattr(signal, 'confluence', [])) < BTC.MIN_CONFLUENCE_COUNT:
                        continue

                    # 6. Direction cooldown — prevent same-direction signal spam
                    _dir_key = getattr(signal.direction, 'value', str(signal.direction))
                    _last_bar = _last_signal_bar.get(_dir_key, -BTC.SIGNAL_COOLDOWN_BARS - 1)
                    if bar_idx - _last_bar < BTC.SIGNAL_COOLDOWN_BARS:
                        continue

                    bar_candidates.append(signal)

                except Exception as e:
                    logger.debug(f"Strategy {strategy.name} error at bar {bar_idx}: {e}")
                    continue

            # ── Per-bar best-signal selection ──────────────────────────────
            # Sort by confidence descending, take top N (default 1)
            if len(bar_candidates) > BTC.MAX_SIGNALS_PER_BAR:
                bar_candidates.sort(key=lambda s: s.confidence, reverse=True)
                bar_candidates = bar_candidates[:BTC.MAX_SIGNALS_PER_BAR]

            for signal in bar_candidates:
                try:
                    # Simulate trade outcome using future bars
                    trade = self._simulate_trade(
                        signal, signal_idx, bar_idx, candles,
                        max_hold_bars
                    )

                    if trade:
                        # BT-4: Annotate trade with regime at time of signal
                        # for regime breakdown analysis
                        try:
                            from analyzers.regime import RegimeState
                            # Infer regime from recent bar pattern (simplified)
                            _w = window_dict.get(primary_tf, [])
                            if len(_w) >= BTC.MIN_SAMPLE_SIZE:
                                _cls = [c[4] for c in _w[-20:]]
                                _highs = [c[2] for c in _w[-20:]]
                                _lows  = [c[3] for c in _w[-20:]]
                                _sma20 = sum(_cls) / 20
                                _sma5 = sum(_cls[-5:]) / 5
                                # FIX: real ATR — mean of per-bar true ranges.
                                # The previous (max-min)/20 collapsed to one
                                # range-divided-by-20 figure that hugely under-
                                # stated volatility, so VOLATILE was almost
                                # never tagged in choppy crypto markets.
                                _trs = [
                                    max(
                                        _highs[k] - _lows[k],
                                        abs(_highs[k] - _cls[k - 1]) if k > 0 else 0.0,
                                        abs(_lows[k]  - _cls[k - 1]) if k > 0 else 0.0,
                                    )
                                    for k in range(len(_cls))
                                ]
                                _atr20 = sum(_trs) / max(1, len(_trs))
                                _chg = (_cls[-1] - _cls[0]) / _cls[0] if _cls[0] else 0
                                if _cls[-1] > 0 and _atr20 / _cls[-1] > BTC.HIGH_VOLATILITY_THRESHOLD:
                                    trade.regime_at_signal = 'VOLATILE'
                                elif _sma5 > _sma20 * 1.005 and _chg > BTC.MOMENTUM_THRESHOLD:
                                    trade.regime_at_signal = 'BULL_TREND'
                                elif _sma5 < _sma20 * 0.995 and _chg < -BTC.MOMENTUM_THRESHOLD:
                                    trade.regime_at_signal = 'BEAR_TREND'
                                else:
                                    trade.regime_at_signal = 'CHOPPY'
                            else:
                                trade.regime_at_signal = 'UNKNOWN'
                        except Exception:
                            trade.regime_at_signal = 'UNKNOWN'
                        all_trades.append(trade)
                        signal_idx += 1
                        # Update cooldown tracker for this direction
                        _dir_taken = getattr(signal.direction, 'value', str(signal.direction))
                        _last_signal_bar[_dir_taken] = bar_idx

                except Exception as e:
                    _sig_strat = getattr(signal, 'strategy', 'unknown')
                    logger.debug(f"Signal simulation error at bar {bar_idx} [{_sig_strat}]: {e}")
                    continue

        # Compile results
        result = self._compile_results(
            symbol, primary_tf, all_trades, start_ts, end_ts
        )

        # Buy-and-hold benchmark in R-multiples.
        #
        # FIX: previously the "no traded signal" and "no valid risk" branches
        # silently relabeled raw percent return as R (multiplying by 100),
        # producing wildly inflated B&H comparisons (e.g. a 5% market move
        # showed up as +500R alpha). When we cannot translate the move into
        # comparable R-units we now report 0.0 and document why, so the
        # console/markdown reports don't lie.
        try:
            start_close = float(candles[window_size][4])
            end_close = float(candles[-1][4])
            if start_close > 0:
                bah_pct = (end_close - start_close) / start_close
                valid_risk = [
                    abs(t.entry_price - t.stop_loss) / max(t.entry_price, 1e-10)
                    for t in result.trades if t.entry_price > 0 and t.stop_loss > 0
                ]
                if valid_risk:
                    avg_risk_pct = float(np.mean(valid_risk))
                    if avg_risk_pct > 0:
                        result.buy_and_hold_r = round(bah_pct / avg_risk_pct, 2)
        except Exception:
            pass

        if system_backtest:
            result.accuracy_note = (
                "system_backtest=True applies 3 of 8 live gates "
                "(HTF guardrail, equilibrium zone, regime filter). "
                "Missing: time_filter, signal_clarity, no_trade_zones, "
                "sector rotation weights, chop mid-range penalty. "
                "Results are an optimistic upper bound, not expected live performance."
            )

        logger.info(
            f"Backtest complete: {result.total_trades} trades | "
            f"Win rate: {result.win_rate:.1f}% | "
            f"Total R: {result.total_r:+.1f} | "
            f"PF: {result.profit_factor:.2f}"
        )

        return result

    def _simulate_trade(
        self,
        signal,
        signal_idx: int,
        bar_idx: int,
        candles: List,
        max_hold_bars: int,
    ) -> Optional[BacktestTrade]:
        """
        Simulate a trade by checking future bars.

        Logic:
          1. Wait for price to enter entry zone (PENDING → ACTIVE).
          2. Once active, check TP1 / TP2 / SL hits per bar.
          3. After TP1, move SL to breakeven (50% closed at TP1).
          4. Expire after ``max_hold_bars`` if neither TP nor SL is hit.

        P&L (FIXED — no slippage double-count, all paths use ``actual_entry``):
          * ``actual_entry`` already incorporates one leg of slippage in the
            adverse direction. ``friction_r`` is therefore *commission only*
            (round-trip) — slippage on the exit is folded into the exit price
            we book against. This corrects the previous behaviour which
            charged slippage in both ``actual_entry`` AND ``friction_r``,
            i.e. counted it twice on every winning leg and roughly twice on
            every loss as well.
          * LOSS pnl_r is computed from ``actual_entry`` rather than hard-
            coded to ``-1.0`` so the (small) extra slippage embedded in the
            fill is reflected. In LONG, ``actual_entry`` > ``entry_mid``, so
            an SL hit is slightly worse than -1R; reverse for SHORT.
          * Same-bar TP1+SL: when a single bar wicks through both the SL and
            TP1, we now favour the TP1-first-then-BE outcome (i.e. the trade
            books a partial at TP1 and stops out flat) rather than booking
            a full -1R loss. This reflects the realistic outcome under a
            tracking algorithm that monitors TP1 inside the bar.
        """
        direction = getattr(signal.direction, 'value', str(signal.direction))
        entry_low = signal.entry_low
        entry_high = signal.entry_high
        entry_mid = (entry_low + entry_high) / 2
        stop_loss = signal.stop_loss
        tp1 = signal.tp1
        tp2 = signal.tp2
        tp3 = signal.tp3

        # Risk uses the *actual* fill price so that losses correctly bake in
        # entry slippage rather than being normalised to a flat -1R.
        if direction == "LONG":
            actual_entry = entry_mid * (1.0 + self.slippage_pct)
        else:
            actual_entry = entry_mid * (1.0 - self.slippage_pct)

        risk = abs(actual_entry - stop_loss)
        if risk <= 0:
            return None

        # FIX #10 (revised): friction is *round-trip commission* expressed in
        # R-multiples. Slippage on the exit is applied directly to the exit
        # price we mark against, not added to friction_r as well.
        commission_r = (2.0 * self.commission_pct * actual_entry) / risk
        slip_pct = self.slippage_pct

        def _exit_price_long(intent: float, side: str) -> float:
            """Long exits suffer downside slippage on both wins and losses."""
            return intent * (1.0 - slip_pct) if side == "TP" else intent * (1.0 - slip_pct)

        def _exit_price_short(intent: float, side: str) -> float:
            """Short exits suffer upside slippage."""
            return intent * (1.0 + slip_pct)

        def _r_long(exit_px: float) -> float:
            return (exit_px - actual_entry) / risk - commission_r

        def _r_short(exit_px: float) -> float:
            return (actual_entry - exit_px) / risk - commission_r

        trade = BacktestTrade(
            signal_idx=signal_idx,
            bar_idx=bar_idx,
            symbol=signal.symbol,
            strategy=signal.strategy,
            direction=direction,
            confidence=signal.confidence,
            entry_low=entry_low,
            entry_high=entry_high,
            entry_price=entry_mid,
            stop_loss=stop_loss,
            tp1=tp1, tp2=tp2, tp3=tp3,
            rr_ratio=signal.rr_ratio,
            confluence=list(signal.confluence),
            timestamp=candles[bar_idx][0],
        )

        active = False
        be_stop = stop_loss
        tp1_hit = False
        entry_bar = bar_idx

        # Check future bars
        for i in range(1, max_hold_bars + 1):
            future_idx = bar_idx + i
            if future_idx >= len(candles):
                break

            high = float(candles[future_idx][2])
            low = float(candles[future_idx][3])

            # Phase 1: Wait for entry
            if not active:
                if low <= entry_high and high >= entry_low:
                    active = True
                    entry_bar = future_idx
                    continue
                # Check for invalidation (SL hit before entry)
                if direction == "LONG" and low <= stop_loss:
                    return None  # invalidated, never filled
                if direction == "SHORT" and high >= stop_loss:
                    return None
                continue

            bars_since_entry = future_idx - entry_bar

            # Phase 2: Trade is active.
            #
            # Order of checks within a single bar matters:
            #   * If the bar wicks through TP1 *and* SL we assume realistic
            #     behaviour: TP1 books a partial close, the stop is moved to
            #     breakeven, and the same bar's adverse wick then closes the
            #     remainder at BE (not -1R).
            #   * If only SL is touched (no TP1), we book a full loss using
            #     ``actual_entry`` so slippage shows up.
            if direction == "LONG":
                tp1_touched_now = (not tp1_hit) and (high >= tp1)
                tp2_touched_now = high >= tp2
                sl_touched_now = low <= be_stop

                # 1. Same-bar TP1 first if both TP1 and SL trip on this bar
                if tp1_touched_now and sl_touched_now and not tp2_touched_now:
                    tp1_hit = True
                    trade.tp1_hit = True
                    trade.tp1_partial_r = _r_long(_exit_price_long(tp1, "TP"))
                    # Remaining 50% closes at BE (= actual_entry) immediately.
                    trade.outcome = "BE"
                    trade.exit_price = actual_entry
                    # Net = 0.5 * TP1_R + 0.5 * 0 (BE) - commission already in tp1_r.
                    trade.pnl_r = 0.5 * trade.tp1_partial_r - 0.5 * commission_r
                    trade.bars_to_outcome = bars_since_entry
                    trade.exit_bar_idx = future_idx
                    return trade

                # 2. Outright SL (no TP1 protection yet)
                if sl_touched_now:
                    if tp1_hit:
                        trade.outcome = "BE"
                        exit_px = _exit_price_long(be_stop, "SL")
                        # 50% already booked at TP1. Remainder closes at BE.
                        trade.exit_price = exit_px
                        trade.pnl_r = 0.5 * trade.tp1_partial_r + 0.5 * _r_long(exit_px)
                    else:
                        trade.outcome = "LOSS"
                        exit_px = _exit_price_long(stop_loss, "SL")
                        trade.exit_price = exit_px
                        trade.pnl_r = _r_long(exit_px)
                    trade.bars_to_outcome = bars_since_entry
                    trade.exit_bar_idx = future_idx
                    return trade

                # 3. TP2 (full exit on remaining 50% if TP1 already hit, else 100%)
                if tp2_touched_now:
                    trade.outcome = "WIN"
                    exit_px = _exit_price_long(tp2, "TP")
                    trade.exit_price = exit_px
                    if tp1_hit:
                        trade.pnl_r = 0.5 * trade.tp1_partial_r + 0.5 * _r_long(exit_px)
                    else:
                        trade.pnl_r = _r_long(exit_px)
                    trade.bars_to_outcome = bars_since_entry
                    trade.exit_bar_idx = future_idx
                    return trade

                # 4. TP1-only this bar → record partial, move stop to BE
                if tp1_touched_now:
                    tp1_hit = True
                    trade.tp1_hit = True
                    trade.tp1_partial_r = _r_long(_exit_price_long(tp1, "TP"))
                    be_stop = actual_entry

            else:  # SHORT — mirror image
                tp1_touched_now = (not tp1_hit) and (low <= tp1)
                tp2_touched_now = low <= tp2
                sl_touched_now = high >= be_stop

                if tp1_touched_now and sl_touched_now and not tp2_touched_now:
                    tp1_hit = True
                    trade.tp1_hit = True
                    trade.tp1_partial_r = _r_short(_exit_price_short(tp1, "TP"))
                    trade.outcome = "BE"
                    trade.exit_price = actual_entry
                    trade.pnl_r = 0.5 * trade.tp1_partial_r - 0.5 * commission_r
                    trade.bars_to_outcome = bars_since_entry
                    trade.exit_bar_idx = future_idx
                    return trade

                if sl_touched_now:
                    if tp1_hit:
                        trade.outcome = "BE"
                        exit_px = _exit_price_short(be_stop, "SL")
                        trade.exit_price = exit_px
                        trade.pnl_r = 0.5 * trade.tp1_partial_r + 0.5 * _r_short(exit_px)
                    else:
                        trade.outcome = "LOSS"
                        exit_px = _exit_price_short(stop_loss, "SL")
                        trade.exit_price = exit_px
                        trade.pnl_r = _r_short(exit_px)
                    trade.bars_to_outcome = bars_since_entry
                    trade.exit_bar_idx = future_idx
                    return trade

                if tp2_touched_now:
                    trade.outcome = "WIN"
                    exit_px = _exit_price_short(tp2, "TP")
                    trade.exit_price = exit_px
                    if tp1_hit:
                        trade.pnl_r = 0.5 * trade.tp1_partial_r + 0.5 * _r_short(exit_px)
                    else:
                        trade.pnl_r = _r_short(exit_px)
                    trade.bars_to_outcome = bars_since_entry
                    trade.exit_bar_idx = future_idx
                    return trade

                if tp1_touched_now:
                    tp1_hit = True
                    trade.tp1_hit = True
                    trade.tp1_partial_r = _r_short(_exit_price_short(tp1, "TP"))
                    be_stop = actual_entry

        # Expired — mark to last close
        if active:
            last_close = float(candles[min(bar_idx + max_hold_bars, len(candles) - 1)][4])
            trade.outcome = "EXPIRED"
            trade.exit_price = last_close
            if direction == "LONG":
                pnl = _r_long(last_close * (1.0 - slip_pct))
            else:
                pnl = _r_short(last_close * (1.0 + slip_pct))
            # If TP1 partial already booked, blend with the still-open half.
            if tp1_hit:
                trade.pnl_r = 0.5 * trade.tp1_partial_r + 0.5 * pnl
            else:
                trade.pnl_r = pnl
            trade.bars_to_outcome = max_hold_bars
            trade.exit_bar_idx = bar_idx + max_hold_bars
            return trade

        return None  # Never entered

    def _compile_results(
        self,
        symbol: str,
        timeframe: str,
        trades: List[BacktestTrade],
        start_ts: int,
        end_ts: int,
    ) -> BacktestResult:
        """Compile all trades into a summary result"""

        from datetime import datetime, timezone

        start_date = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        end_date = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        result = BacktestResult(
            symbol=symbol,
            strategy="ALL",
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            trades=trades,
        )

        if not trades:
            return result

        # Semantics:
        #   total_signals  = anything that filled (any non-PENDING outcome).
        #                    PENDING is impossible here because _simulate_trade
        #                    only returns trades that have been categorised,
        #                    but we keep the filter to be defensive.
        #   total_trades   = decided trades whose R is in the headline stats
        #                    (WIN | LOSS | BE). EXPIRED is tracked separately
        #                    so reports never silently mix mark-to-market R
        #                    with realised R.
        result.total_signals = len([t for t in trades if t.outcome != "PENDING"])
        result.total_trades  = len([t for t in trades if t.outcome in ("WIN", "LOSS", "BE")])

        result.wins = len([t for t in trades if t.outcome == "WIN"])
        result.losses = len([t for t in trades if t.outcome == "LOSS"])
        result.breakevens = len([t for t in trades if t.outcome == "BE"])
        result.expired = len([t for t in trades if t.outcome == "EXPIRED"])

        decided = result.wins + result.losses
        result.win_rate = (result.wins / decided * 100) if decided > 0 else 0

        # BUG-NEW-8 FIX: exclude EXPIRED trades from performance metrics.
        # EXPIRED trades get a mark-to-market pnl_r that distorts Sharpe/profit_factor.
        # They are tracked separately (result.expired) but excluded from R statistics.
        r_values = [t.pnl_r for t in trades if t.outcome in ("WIN", "LOSS", "BE")]
        bars = [t.bars_to_outcome for t in trades
                if t.bars_to_outcome > 0 and t.outcome in ("WIN", "LOSS", "BE")]
        result.avg_bars_to_outcome = float(np.mean(bars)) if bars else 0
        if r_values:
            result.avg_r = float(np.mean(r_values))
            result.total_r = float(np.sum(r_values))
            result.best_trade_r = float(np.max(r_values))
            result.worst_trade_r = float(np.min(r_values))

            # Profit factor
            gross_profit = sum(r for r in r_values if r > 0)
            gross_loss = abs(sum(r for r in r_values if r < 0))
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Sharpe annualisation.
            #
            # FIX (BUG-NEW-5 v2): the previous version used sqrt(252 * trades_per_day)
            # which is wrong for crypto on two counts:
            #   1. crypto trades 24/7 — the calendar is 365 days, not 252.
            #   2. trades_per_day is already a *daily* trade count, so the
            #      annualisation factor is sqrt(trades_per_year), where
            #      trades_per_year = trades_per_day * 365.
            # Net effect of the old formula: Sharpe was about sqrt(252/365)
            # ≈ 0.83x of the correct value, biasing every grade downward and
            # under-rewarding genuinely high-frequency strategies.
            _bars_per_day = BTC.BARS_PER_DAY.get(timeframe, 24)
            if len(r_values) > 1:
                r_std = float(np.std(r_values, ddof=1))
                _avg_hold = result.avg_bars_to_outcome if result.avg_bars_to_outcome > 0 else _bars_per_day
                _trades_per_day = max(0.1, _bars_per_day / max(1.0, _avg_hold))
                _ann_factor = float(np.sqrt(365.0 * _trades_per_day))
                if r_std > 0:
                    result.sharpe_ratio = (result.avg_r / r_std) * _ann_factor

                # Sortino — downside deviation only.
                downside_returns = [r for r in r_values if r < 0]
                if len(downside_returns) >= 2:
                    downside_std = float(np.std(downside_returns, ddof=1))
                    if downside_std > 0:
                        result.sortino_ratio = (result.avg_r / downside_std) * _ann_factor

            # Max drawdown
            cumulative = np.cumsum(r_values)
            peak = np.maximum.accumulate(cumulative)
            drawdown = peak - cumulative
            result.max_drawdown_r = float(np.max(drawdown)) if len(drawdown) > 0 else 0

            # Calmar — total_r / max_drawdown.
            if result.max_drawdown_r > 0:
                result.calmar_ratio = round(result.total_r / result.max_drawdown_r, 3)

            # MAE / MFE per trade — bar-resolution approximations.
            maes = [abs(t.pnl_r) for t in trades if t.outcome == "LOSS"]
            mfes = [t.tp1_partial_r for t in trades if t.tp1_hit and t.tp1_partial_r > 0]
            result.avg_mae_r = float(np.mean(maes)) if maes else 0.0
            result.avg_mfe_r = float(np.mean(mfes)) if mfes else 0.0

        # Consecutive wins/losses
        result.max_consecutive_wins = self._max_consecutive(trades, "WIN")
        result.max_consecutive_losses = self._max_consecutive(trades, "LOSS")

        # Per-strategy breakdown — EXPIRED excluded from R sums to match
        # the headline metrics block above.
        strategies = set(t.strategy for t in trades)
        for strat in strategies:
            strat_trades = [t for t in trades if t.strategy == strat]
            strat_decided = [t for t in strat_trades if t.outcome in ("WIN", "LOSS")]
            strat_wins = len([t for t in strat_decided if t.outcome == "WIN"])
            strat_r = [t.pnl_r for t in strat_trades if t.outcome in ("WIN", "LOSS", "BE")]

            result.strategy_breakdown[strat] = {
                'total': len(strat_trades),
                'wins': strat_wins,
                'losses': len(strat_decided) - strat_wins,
                'win_rate': strat_wins / len(strat_decided) * 100 if strat_decided else 0,
                'total_r': float(np.sum(strat_r)) if strat_r else 0.0,
                'avg_r': float(np.mean(strat_r)) if strat_r else 0.0,
            }

        # ── BT-4: Per-regime breakdown — EXPIRED excluded for consistency ──
        regimes = set(getattr(t, 'regime_at_signal', 'UNKNOWN') for t in trades)
        for regime in regimes:
            rt = [t for t in trades if getattr(t, 'regime_at_signal', 'UNKNOWN') == regime]
            rd = [t for t in rt if t.outcome in ('WIN', 'LOSS')]
            rw = len([t for t in rd if t.outcome == 'WIN'])
            rr = [t.pnl_r for t in rt if t.outcome in ('WIN', 'LOSS', 'BE')]
            result.regime_breakdown[regime] = {
                'trades': len(rt),
                'wins': rw,
                'losses': len(rd) - rw,
                'win_rate': rw / len(rd) * 100 if rd else 0,
                'total_r': float(np.sum(rr)) if rr else 0,
                'avg_r': float(np.mean(rr)) if rr else 0,
            }

        # ── BT-1: Monte Carlo simulation (N=1000 resamples) ─────────────────
        # Resamples with replacement to build confidence intervals around
        # win rate, total R, and max drawdown. Shows best/worst/median outcomes.
        if r_values and len(r_values) >= 5:
            import random as _rnd
            _N = BTC.MONTE_CARLO_RESAMPLES
            _wrs, _trs, _dds = [], [], []
            for _ in range(_N):
                sample = [_rnd.choice(r_values) for _ in range(len(r_values))]
                _wins_s = sum(1 for r in sample if r > 0)
                _dec_s = sum(1 for r in sample if r != 0)
                _wrs.append(_wins_s / _dec_s * 100 if _dec_s else 0)
                _trs.append(float(np.sum(sample)))
                _cum = np.cumsum(sample)
                _pk = np.maximum.accumulate(_cum)
                _dd = _pk - _cum
                _dds.append(float(np.max(_dd)) if len(_dd) else 0)
            _wrs.sort(); _trs.sort(); _dds.sort()
            p5, p50, p95 = int(_N * 0.05), _N // 2, int(_N * 0.95)
            result.monte_carlo = {
                'n': _N,
                'samples': len(r_values),
                'win_rate': {
                    'p5':  round(_wrs[p5], 1),
                    'p50': round(_wrs[p50], 1),
                    'p95': round(_wrs[p95], 1),
                },
                'total_r': {
                    'p5':  round(_trs[p5], 2),
                    'p50': round(_trs[p50], 2),
                    'p95': round(_trs[p95], 2),
                },
                'max_drawdown': {
                    'p5':  round(_dds[p5], 2),
                    'p50': round(_dds[p50], 2),
                    'p95': round(_dds[p95], 2),
                },
            }

        return result

    def run_walk_forward(
        self,
        all_trades: list,
        window_trades: int = BTC.WALKFORWARD_WINDOW,
        step_trades: int = BTC.WALKFORWARD_STEP,
    ) -> list:
        """
        BT-2: Rolling walk-forward analysis across trade sequence.
        Slides a window of `window_trades` trades forward by `step_trades`
        to test whether edge is consistent over time (not a lucky streak).

        Returns list of window dicts: {start, end, trades, win_rate, total_r, avg_r}
        """
        if len(all_trades) < window_trades:
            return []
        windows = []
        for start in range(0, len(all_trades) - window_trades + 1, step_trades):
            w = all_trades[start:start + window_trades]
            decided = [t for t in w if t.outcome in ('WIN', 'LOSS')]
            wins = len([t for t in decided if t.outcome == 'WIN'])
            # EXPIRED excluded for consistency with headline metrics — see
            # _compile_results comments.
            r_vals = [t.pnl_r for t in w if t.outcome in ('WIN', 'LOSS', 'BE')]
            windows.append({
                'start_idx': start,
                'end_idx': start + len(w) - 1,
                'trades': len(w),
                'decided': len(decided),
                'wins': wins,
                'win_rate': round(wins / len(decided) * 100, 1) if decided else 0,
                'total_r': round(float(np.sum(r_vals)), 2) if r_vals else 0,
                'avg_r': round(float(np.mean(r_vals)), 3) if r_vals else 0,
            })
        return windows

    def _max_consecutive(self, trades, outcome_type) -> int:
        """Find max consecutive occurrences of an outcome"""
        max_streak = 0
        current = 0
        for t in trades:
            if t.outcome == outcome_type:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def _get_default_strategies(self):
        """Initialize all 13 strategies for backtesting.
        FIX: was only 8 strategies — missing ElliottWave, RangeScalper, Wyckoff,
        HarmonicDetector, GeometricPatterns. Backtest results were optimistically
        biased because 5 strategies that filter aggressively in production were absent.
        """
        from strategies.smc import SMCStrategy
        from strategies.breakout import BreakoutStrategy
        from strategies.reversal import ReversalStrategy
        from strategies.mean_reversion import MeanReversionStrategy
        from strategies.price_action import PriceActionStrategy
        from strategies.momentum import MomentumStrategy
        from strategies.ichimoku import IchimokuStrategy
        from strategies.elliott_wave import ElliottWaveStrategy
        from strategies.funding_arb import FundingArbStrategy
        from strategies.range_scalper import RangeScalperStrategy
        from strategies.wyckoff import WyckoffStrategy
        from patterns.harmonic import HarmonicDetector
        from patterns.geometric import GeometricPatterns

        return [
            SMCStrategy(),
            BreakoutStrategy(),
            ReversalStrategy(),
            MeanReversionStrategy(),
            PriceActionStrategy(),
            MomentumStrategy(),
            IchimokuStrategy(),
            ElliottWaveStrategy(),
            FundingArbStrategy(),
            RangeScalperStrategy(),
            WyckoffStrategy(),
            HarmonicDetector(),
            GeometricPatterns(),
        ]

    def _apply_overrides(self, overrides: Dict) -> Dict:
        """Apply config overrides and return original values.

        FIX: the previous traversal walked ``parts[:-1]`` but its dict
        fallback ``obj.get(p, obj)`` referenced the loop variable from the
        outer scope (already reassigned via ``getattr``), so nested keys like
        ``strategies.breakout.min_adx`` silently fell back to the root cfg
        and the override never took effect. Worse, the rollback on the
        ``finally`` branch then wrote the new value into the wrong slot,
        permanently mutating cfg between optimizer iterations.

        We now walk the path properly through both attributes and dict
        keys, recording the original value at the resolved leaf only.
        """
        original: Dict = {}
        for key_path, value in overrides.items():
            parts = key_path.split('.')
            obj = cfg
            ok = True
            for p in parts[:-1]:
                if isinstance(obj, dict) and p in obj:
                    obj = obj[p]
                elif hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    ok = False
                    break
            if not ok:
                logger.debug(f"Override path not found: {key_path}")
                continue
            final_key = parts[-1]
            if isinstance(obj, dict) and final_key in obj:
                original[key_path] = obj[final_key]
                obj[final_key] = value
            elif hasattr(obj, final_key):
                original[key_path] = getattr(obj, final_key)
                setattr(obj, final_key, value)
            else:
                logger.debug(f"Override leaf not found: {key_path}")
        return original


# Singleton
backtest_engine = BacktestEngine()
