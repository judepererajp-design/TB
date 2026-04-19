"""
TitanBot Pro — Portfolio Engine (Pillar 4: Portfolio Optimization)
===================================================================
The bot currently thinks in trades. Quants think in portfolios.

This module manages:
  1. Position sizing (Kelly-based, volatility-adjusted)
  2. Correlation control (limit correlated positions)
  3. Exposure limits (per-sector, per-direction, total)
  4. Volatility targeting (scale positions to target portfolio vol)

Key principle:
  position_size ∝ edge / volatility

The portfolio engine sits between signal approval and execution,
deciding not just "should we trade?" but "how much?"
"""

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from config.loader import cfg
from config.constants import Portfolio

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An active or pending position"""
    symbol: str
    direction: str          # LONG | SHORT
    strategy: str
    entry_price: float
    size_usdt: float        # Position size in USDT
    risk_usdt: float        # Risk amount in USDT
    risk_r: float           # Risk as fraction of capital
    stop_loss: float
    sector: str = ""
    correlation_to_btc: float = 0.7
    opened_at: float = 0.0
    signal_id: int = 0
    unrealized_pnl: float = 0.0
    # T1-FIX: funding rate at entry (percent per 8h period, e.g. 0.05 means 0.05%/8h).
    # Used to estimate accumulated funding fees on close so realized P&L is accurate.
    # A value of 0.0 means not set (non-FundingArb trades leave this at default).
    funding_rate_8h: float = 0.0

    @property
    def net_exposure(self) -> float:
        """Signed exposure: positive for long, negative for short"""
        return self.size_usdt if self.direction == "LONG" else -self.size_usdt


@dataclass
class PortfolioState:
    """Current portfolio snapshot"""
    total_capital: float
    total_risk_deployed: float     # Sum of risk across all positions
    total_long_exposure: float     # Total long notional
    total_short_exposure: float    # Total short notional
    net_exposure: float            # Long - Short
    gross_exposure: float          # Long + Short
    position_count: int
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    direction_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class SizingDecision:
    """Position sizing output"""
    approved: bool
    position_size_usdt: float
    risk_amount_usdt: float
    risk_pct: float               # Risk as % of capital
    kelly_fraction: float
    leverage_suggested: float
    reject_reason: Optional[str] = None

    # Adjustments applied
    vol_adjustment: float = 1.0   # Volatility scaling factor
    correlation_adjustment: float = 1.0  # Correlation penalty
    exposure_adjustment: float = 1.0     # Exposure limit scaling


class PortfolioEngine:
    """
    Portfolio-level position management and sizing.
    """

    def __init__(self):
        self._positions: Dict[int, Position] = {}   # signal_id -> Position  (FIX P1-A)
        self._symbol_to_signal: Dict[str, int] = {}  # symbol -> signal_id reverse index
        self._risk_cfg = cfg.risk

        # BUG-NEW-4 FIX: asyncio.Lock prevents concurrent _scan_symbol() tasks from
        # racing through the position-count / max-risk checks simultaneously and all
        # opening positions. Without this, 5 concurrent tasks each see position_count=0,
        # all pass the max_positions gate, and all call open_position() — bypassing every
        # portfolio limit. Lock must wrap the full size_position + open_position flow.
        self._lock = asyncio.Lock()

        # Portfolio limits
        self._max_total_risk_pct = Portfolio.MAX_TOTAL_RISK_PCT
        self._max_position_risk_pct = Portfolio.MAX_POSITION_RISK_PCT
        self._max_positions = Portfolio.MAX_POSITIONS
        self._max_sector_exposure_pct = Portfolio.MAX_SECTOR_EXPOSURE_PCT
        self._max_direction_imbalance = Portfolio.MAX_DIRECTION_IMBALANCE
        self._max_correlated_positions = Portfolio.MAX_CORRELATED_POSITIONS

        # V10: Correlation gating limits
        self.MAX_SAME_DIRECTION = Portfolio.MAX_SAME_DIRECTION
        self.MAX_SAME_SECTOR = Portfolio.MAX_SAME_SECTOR

        # Volatility targeting
        self._target_portfolio_vol = Portfolio.TARGET_PORTFOLIO_VOL
        self._vol_window = Portfolio.VOL_WINDOW_DAYS

        # Capital
        self._capital = float(self._risk_cfg.get('account_balance', 10000))
        # L2-FIX: Store the configured starting balance once so get_effective_capital()
        # can floor at a fraction of INITIAL capital rather than current capital.
        # NC2/NC4-FIX: _initial_capital now tracks the all-time peak capital (highest
        # balance ever reached), not just the configured starting balance. The name is
        # retained for backward compatibility with existing floor logic. It is updated
        # in update_capital() on new highs and persisted/restored in load_capital().
        self._initial_capital: float = self._capital

    async def size_position(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        entry_price: float,
        stop_loss: float,
        kelly_fraction: float,
        p_win: float,
        rr_ratio: float,
        sector: str = "",
        correlation_to_btc: float = -1.0,  # -1 = use cache lookup
        symbol_volatility: float = Portfolio.DEFAULT_SYMBOL_VOLATILITY,
        setup_class: str = "intraday",
    ) -> SizingDecision:
        """
        Determine optimal position size accounting for portfolio constraints.

        BUG-NEW-4 FIX: entire method body runs under self._lock so concurrent
        _scan_symbol() coroutines cannot race through position-count / max-risk
        checks and simultaneously open multiple positions past the limit.

        Args:
            symbol: Trading pair
            direction: LONG or SHORT
            strategy: Strategy name
            entry_price: Expected entry price
            stop_loss: Stop loss level
            kelly_fraction: Kelly fraction from alpha model
            p_win: Win probability
            rr_ratio: Risk-reward ratio
            sector: Asset sector (e.g., "L1", "DeFi", "Meme")
            correlation_to_btc: BTC correlation (0-1)
            symbol_volatility: Daily volatility of this symbol
            setup_class: Signal setup class (scalp/intraday/swing/positional)

        Returns:
            SizingDecision with position size and adjustments
        """
        async with self._lock:
            return self._size_position_locked(
                symbol=symbol, direction=direction, strategy=strategy,
                entry_price=entry_price, stop_loss=stop_loss,
                kelly_fraction=kelly_fraction, p_win=p_win, rr_ratio=rr_ratio,
                sector=sector, correlation_to_btc=correlation_to_btc,
                symbol_volatility=symbol_volatility, setup_class=setup_class,
            )

    def _size_position_locked(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        entry_price: float,
        stop_loss: float,
        kelly_fraction: float,
        p_win: float,
        rr_ratio: float,
        sector: str = "",
        correlation_to_btc: float = -1.0,
        symbol_volatility: float = Portfolio.DEFAULT_SYMBOL_VOLATILITY,
        setup_class: str = "intraday",
    ) -> SizingDecision:
        """Internal (unlocked) implementation — always called under self._lock."""
        # FIX 11: resolve the -1.0 sentinel immediately. The value was passed raw to
        # _count_correlated() and the BTC correlation gate (line ~258). _count_correlated
        # checks `new_corr > 0.6` — a -1.0 sentinel always fails that check, so the
        # BTC correlation gate NEVER fired regardless of actual correlation. Use a
        # conservative 0.7 default (most alt coins are highly correlated to BTC).
        if correlation_to_btc < 0:
            try:
                from analyzers.correlation import correlation_analyzer as _ca
                correlation_to_btc = _ca.get_cached_correlation(symbol)
            except Exception as _ca_err:
                logger.warning(
                    f"correlation lookup failed for {symbol}: {_ca_err}; "
                    f"using conservative 0.7 default (BTC-correlation gate may be inaccurate)"
                )
                correlation_to_btc = 0.7  # conservative crypto default

        # FIX 12: use effective capital (accounts for unrealized losses on open positions)
        # rather than raw _capital. get_effective_capital() is floored at 50% of capital
        # so this can't reduce sizing to zero, but does prevent oversizing when existing
        # positions are underwater — previously ignored entirely.
        effective_capital = self.get_effective_capital()

        # ── 1. Base risk from dynamic Kelly (live Bayesian posteriors) ──────
        # Override the caller-provided kelly_fraction with live posterior-based Kelly.
        # The caller's kelly_fraction comes from alpha_model which uses historical
        # priors; get_dynamic_kelly() uses the current posterior after actual trades.
        try:
            from analyzers.regime import regime_analyzer
            _cur_regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception:
            _cur_regime = ""
        _dynamic_kelly = self.get_dynamic_kelly(strategy, _cur_regime, direction)
        # Blend: 60% dynamic, 40% caller-provided (smooth transition for new strategies)
        blended_kelly = 0.6 * _dynamic_kelly + 0.4 * kelly_fraction

        # Setup-class Kelly multiplier — scalps are noisier and have higher rate-of-loss
        # than swing setups; this applies a consistent adjustment independent of regime.
        # Intraday (1.0×) is the baseline; scalp is reduced (0.80×), swing/positional
        # are slightly elevated to reward the wider structural confirmation they require.
        _setup_mult = {
            "scalp":       Portfolio.KELLY_MULT_SETUP_SCALP,
            "intraday":    Portfolio.KELLY_MULT_SETUP_INTRADAY,
            "swing":       Portfolio.KELLY_MULT_SETUP_SWING,
            "positional":  Portfolio.KELLY_MULT_SETUP_POSITIONAL,
        }.get(setup_class, Portfolio.KELLY_MULT_SETUP_INTRADAY)
        blended_kelly *= _setup_mult

        base_risk_pct = max(0.002, min(self._max_position_risk_pct, blended_kelly))

        # ── 2. Volatility adjustment ──────────────────────────
        # Scale risk inversely to volatility
        avg_vol = 0.03  # Baseline assumption for crypto
        if symbol_volatility > 0:
            vol_adjustment = avg_vol / symbol_volatility
            vol_adjustment = max(0.3, min(2.0, vol_adjustment))
        else:
            vol_adjustment = 1.0

        # ── 3. Correlation adjustment ─────────────────────────
        correlated_count = self._count_correlated(symbol, correlation_to_btc, sector, direction)
        if correlated_count >= self._max_correlated_positions:
            correlation_adjustment = 0.5
        elif correlated_count >= 2:
            correlation_adjustment = 0.75
        else:
            correlation_adjustment = 1.0

        # ── 4. Exposure checks ────────────────────────────────
        state = self.get_state()
        exposure_adjustment = 1.0
        reject_reason = None

        # ── Directional correlation cap (NEW) ────────────────────────
        # In BEAR_TREND or CHOPPY regime, cap same-direction positions at 2.
        # Prevents stacking 4 counter-trend longs that all lose when BTC drops.
        try:
            from analyzers.regime import regime_analyzer as _pe_ra
            _regime_val = _pe_ra.regime.value if hasattr(_pe_ra, 'regime') and _pe_ra.regime else "UNKNOWN"
            _is_risk_regime = _regime_val in ("BEAR_TREND", "CHOPPY", "VOLATILE")
        except Exception:
            _is_risk_regime = False

        if _is_risk_regime:
            _same_dir = sum(
                1 for p in self._positions.values()
                if p.direction == direction
            )
            # Cap COUNTER-TREND directions only. In BEAR_TREND, cap LONGs (counter-trend)
            # but NOT SHORTs (trend-aligned — they should flow freely).
            # In BULL_TREND, cap SHORTs. In CHOPPY, cap both directions.
            # Only cap the COUNTER-TREND direction:
            # BEAR_TREND: cap LONGs (counter-trend). SHORTs are free.
            # BULL_TREND: cap SHORTs. LONGs are free.
            # CHOPPY/VOLATILE: no directional cap — both directions valid.
            #   (cluster cap handles the correlation risk in choppy)
            _is_counter_direction = (
                (_regime_val == "BEAR_TREND" and direction == "LONG") or
                (_regime_val == "BULL_TREND" and direction == "SHORT")
            )
            _dir_cap = 2  # max 2 counter-trend trades in trending regimes
            if _is_counter_direction and _same_dir >= _dir_cap:
                return SizingDecision(
                    approved=False, position_size_usdt=0, risk_amount_usdt=0,
                    risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                    reject_reason=(
                        f"Directional cap: already {_same_dir} {direction} positions "
                        f"in {_regime_val} regime (max {_dir_cap})"
                    )
                )

        # ── Refined cluster guard (replaces hard cap) ───────────────────
        # Two FundingRateArb LONGs at -0.057% and -0.060% are independent edges
        # on different coins — correlation comes from price, not strategy name.
        # Hard cluster cap blocked real edge. Replaced with: only block if
        # existing same-strategy position is losing (failed thesis) OR if we
        # already have 2+ same-strategy same-direction positions open.
        _cluster_key = f"{strategy}:{direction}"
        _cluster_positions = [
            p for p in self._positions.values()
            if f"{p.strategy}:{p.direction}" == _cluster_key
        ]
        if len(_cluster_positions) >= 2:
            # 3+ same cluster = definitely one thesis, not diversification
            return SizingDecision(
                approved=False, position_size_usdt=0, risk_amount_usdt=0,
                risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                reject_reason=(
                    f"Cluster cap: already {len(_cluster_positions)} {_cluster_key} positions "
                    f"open (max 2 per cluster)"
                )
            )
        # If 1 exists and is significantly losing, it suggests thesis failure — block new one
        if len(_cluster_positions) == 1:
            _existing_pnl_pct = _cluster_positions[0].unrealized_pnl / max(1, _cluster_positions[0].size_usdt)
            if _existing_pnl_pct < -0.03:  # Existing position down >3% = thesis failing
                return SizingDecision(
                    approved=False, position_size_usdt=0, risk_amount_usdt=0,
                    risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                    reject_reason=(
                        f"Cluster guard: existing {_cluster_key} position losing "
                        f"({_existing_pnl_pct:.1%}) — thesis may be wrong"
                    )
                )

        # Max positions
        if state.position_count >= self._max_positions:
            return SizingDecision(
                approved=False, position_size_usdt=0, risk_amount_usdt=0,
                risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                reject_reason=f"Max positions reached ({self._max_positions})"
            )

        # Max total risk
        remaining_risk = self._max_total_risk_pct - (state.total_risk_deployed / effective_capital)
        if remaining_risk <= Portfolio.MIN_RISK_ALLOCATION:
            return SizingDecision(
                approved=False, position_size_usdt=0, risk_amount_usdt=0,
                risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                reject_reason="Total portfolio risk limit reached"
            )
        # Scale down if close to limit
        if base_risk_pct > remaining_risk:
            exposure_adjustment *= (remaining_risk / base_risk_pct)

        # Direction imbalance — BUG-5 FIX: estimate the actual position notional using
        # the real formula (risk_amount / stop_dist_pct) rather than a made-up
        # `effective_capital * base_risk_pct * 10` proxy. With tight stops the proxy
        # was 50-100× too small, so this gate effectively never fired.
        _imbal_stop_dist = max(0.001, abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0.01)
        _estimated_pos_size = (effective_capital * base_risk_pct) / _imbal_stop_dist
        if direction == "LONG":
            long_ratio = (state.total_long_exposure + _estimated_pos_size) / \
                         max(1, state.gross_exposure + _estimated_pos_size)
        else:
            long_ratio = state.total_long_exposure / \
                         max(1, state.gross_exposure + _estimated_pos_size)
        short_ratio = 1 - long_ratio

        if (direction == "LONG" and long_ratio > self._max_direction_imbalance) or \
           (direction == "SHORT" and short_ratio > self._max_direction_imbalance):
            exposure_adjustment *= 0.5

        # Sector concentration — gate on capital, not gross notional.
        # Using max(capital, gross_exposure) allowed sector notional to reach
        # 25% of gross (which at 5× leverage = 125% of capital) before firing.
        # Denominator is effective_capital so the cap means "sector notional > 25%
        # of actual account size", limiting sector risk to a true capital fraction.
        sector_exp = state.sector_exposure.get(sector, 0)
        sector_pct = sector_exp / max(effective_capital, 1.0)
        if sector and sector_pct > self._max_sector_exposure_pct:
            exposure_adjustment *= 0.5

        # Duplicate symbol check (uses reverse index, keyed by symbol)
        if symbol in self._symbol_to_signal:
            return SizingDecision(
                approved=False, position_size_usdt=0, risk_amount_usdt=0,
                risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                reject_reason=f"Already have position in {symbol}"
            )

        # ── 5. Final calculation ──────────────────────────────
        adjusted_risk_pct = base_risk_pct * vol_adjustment * correlation_adjustment * exposure_adjustment
        adjusted_risk_pct = max(Portfolio.MIN_RISK_ALLOCATION, min(self._max_position_risk_pct, adjusted_risk_pct))

        risk_amount = effective_capital * adjusted_risk_pct

        # Position size from stop distance
        stop_dist_pct = abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0.01
        stop_dist_pct = max(0.001, stop_dist_pct)

        position_size = risk_amount / stop_dist_pct

        # Max position cap
        max_pos = effective_capital * self._risk_cfg.get('max_position_pct', 0.10)
        position_size = min(position_size, max_pos)

        # Minimum viable order size — avoids approving "dead" positions that cannot
        # clear exchange minimum notionals and silently never fill in production.
        if position_size < Portfolio.MIN_POSITION_SIZE_USDT:
            return SizingDecision(
                approved=False, position_size_usdt=0, risk_amount_usdt=0,
                risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                reject_reason=(
                    f"Position too small (${position_size:.2f} < "
                    f"${Portfolio.MIN_POSITION_SIZE_USDT:.2f} minimum viable size)"
                ),
            )

        # Leverage calculation
        leverage = position_size / effective_capital if effective_capital > 0 else 1
        leverage = max(1, min(20, leverage))

        # V10: Correlation gating — limit concentrated directional bets
        state = self.get_state()
        _dir_count = state.direction_counts.get(direction, 0)
        if _dir_count >= self.MAX_SAME_DIRECTION:
            return SizingDecision(
                approved=False, position_size_usdt=0, risk_amount_usdt=0,
                risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                reject_reason=f"Max {direction} positions reached ({_dir_count}/{self.MAX_SAME_DIRECTION})",
            )

        # T3: BTC correlation gate — prevent stacking highly correlated altcoin bets.
        # 4 BTC-correlated SHORTs = 1 massive SHORT BTC in practice.
        # Cap at 2 highly-BTC-correlated positions per direction.
        MAX_BTC_CORRELATED = 2
        if correlation_to_btc >= 0.6:
            _btc_corr_count = sum(
                1 for p in self._positions.values()
                if p.direction == direction and p.correlation_to_btc >= 0.6
            )
            if _btc_corr_count >= MAX_BTC_CORRELATED:
                return SizingDecision(
                    approved=False, position_size_usdt=0, risk_amount_usdt=0,
                    risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                    reject_reason=(
                        f"T3: Too many BTC-correlated {direction}s "
                        f"({_btc_corr_count}/{MAX_BTC_CORRELATED}) — "
                        f"effectively shorting BTC with full size"
                    ),
                )

        if sector:
            _sector_exp = state.sector_exposure.get(sector, 0)
            if sum(1 for p in self._positions.values() if p.sector == sector) >= self.MAX_SAME_SECTOR:
                return SizingDecision(
                    approved=False, position_size_usdt=0, risk_amount_usdt=0,
                    risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                    reject_reason=f"Max positions in {sector} sector reached ({self.MAX_SAME_SECTOR})",
                )

        # Correlated-cluster stress test — reject positions that would make one BTC/sector
        # impulse large enough to consume most of the portfolio risk budget in one move.
        _cluster_risk = risk_amount + sum(
            p.risk_usdt for p in self._get_correlated_positions(
                symbol, correlation_to_btc, sector, direction
            )
        )
        _cluster_cap = effective_capital * self._max_total_risk_pct * Portfolio.MAX_CORRELATED_CLUSTER_SHARE
        if _cluster_risk > _cluster_cap:
            return SizingDecision(
                approved=False, position_size_usdt=0, risk_amount_usdt=0,
                risk_pct=0, kelly_fraction=kelly_fraction, leverage_suggested=1,
                reject_reason=(
                    f"Correlated cluster risk too high (${_cluster_risk:.2f} > "
                    f"${_cluster_cap:.2f} stress cap)"
                ),
            )

        # Pass 6 FIX: Apply regime-participation multiplier from risk_governor.
        # get_risk_adjustment() reads execution_journal.jsonl and returns 0.7–1.1x
        # based on how many historical trades occurred in the current regime.
        # This module was complete but never imported anywhere — dead code since v1.
        try:
            from governance.risk_governor import get_risk_adjustment
            from analyzers.regime import regime_analyzer as _ra_pe
            _regime_name = _ra_pe.regime.value if _ra_pe.regime else "UNKNOWN"
            _gov_mult = get_risk_adjustment(_regime_name)
            if _gov_mult != 1.0:
                position_size = min(position_size * _gov_mult, max_pos)
                risk_amount   = risk_amount * _gov_mult
                logger.debug(f"Portfolio: risk_governor mult={_gov_mult:.2f} ({_regime_name})")
        except Exception:
            pass  # Non-fatal — proceed with unscaled sizing

        return SizingDecision(
            approved=True,
            position_size_usdt=round(position_size, 2),
            risk_amount_usdt=round(risk_amount, 2),
            risk_pct=round(adjusted_risk_pct, 5),
            kelly_fraction=round(kelly_fraction, 4),
            leverage_suggested=round(leverage, 1),
            vol_adjustment=round(vol_adjustment, 3),
            correlation_adjustment=round(correlation_adjustment, 3),
            exposure_adjustment=round(exposure_adjustment, 3),
        )

    async def open_position(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        entry_price: float,
        size_usdt: float,
        risk_usdt: float,
        stop_loss: float,
        sector: str = "",
        correlation_to_btc: float = -1.0,  # -1 = use conservative default
        signal_id: int = 0,
        funding_rate_8h: float = 0.0,      # T1-FIX: funding rate at entry for P&L correction
    ):
        """
        Register an opened position.

        FIX #8: Resolve correlation_to_btc sentinel (-1.0) here, not just in
        size_position(). Previously the -1.0 was stored in the Position dataclass,
        so _count_correlated() always saw pos.correlation_to_btc = -1.0 for
        recovered positions, and the T3 BTC correlation gate (>= 0.6) never fired.
        """
        async with self._lock:
            await self._open_position_locked(
                symbol=symbol, direction=direction, strategy=strategy,
                entry_price=entry_price, size_usdt=size_usdt, risk_usdt=risk_usdt,
                stop_loss=stop_loss, sector=sector,
                correlation_to_btc=correlation_to_btc, signal_id=signal_id,
                funding_rate_8h=funding_rate_8h,
            )

    async def _open_position_locked(
        self,
        symbol: str, direction: str, strategy: str,
        entry_price: float, size_usdt: float, risk_usdt: float,
        stop_loss: float, sector: str = "", correlation_to_btc: float = -1.0,
        signal_id: int = 0, funding_rate_8h: float = 0.0,
    ):
        """Internal (unlocked) open_position — always called under self._lock."""
        # Resolve sentinel to conservative crypto default
        if correlation_to_btc < 0:
            try:
                from analyzers.correlation import correlation_analyzer as _ca
                correlation_to_btc = _ca.get_cached_correlation(symbol)
            except Exception:
                correlation_to_btc = 0.7  # conservative: most alts correlate to BTC

        pos = Position(
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            entry_price=entry_price,
            size_usdt=size_usdt,
            risk_usdt=risk_usdt,
            risk_r=risk_usdt / self._capital if self._capital > 0 else 0,
            stop_loss=stop_loss,
            sector=sector,
            correlation_to_btc=correlation_to_btc,
            opened_at=time.time(),
            signal_id=signal_id,
            funding_rate_8h=funding_rate_8h,
        )
        # FIX P1-A: key by signal_id, maintain reverse symbol index
        self._positions[signal_id] = pos
        self._symbol_to_signal[symbol] = signal_id
        logger.info(
            f"Portfolio: opened {direction} {symbol} | "
            f"size=${size_usdt:,.0f} risk=${risk_usdt:,.0f} ({pos.risk_r:.3%}) "
            f"corr_btc={correlation_to_btc:.2f}"
        )

    async def close_position(self, symbol_or_signal_id, pnl_usdt: float = 0.0):
        """Remove a closed position — accepts symbol str or signal_id int (Fix L3 + P1-A)

        R5-S1 FIX: Made async and acquires self._lock to prevent race condition.
        Previously synchronous — concurrent _scan_symbol() tasks could see stale
        position counts while close_position() was mutating _positions dict.
        """
        async with self._lock:
            # FIX P1-A: positions are now keyed by signal_id
            if isinstance(symbol_or_signal_id, int):
                signal_id = symbol_or_signal_id
            else:
                # Symbol string passed — look up via reverse index
                signal_id = self._symbol_to_signal.get(symbol_or_signal_id)
                if signal_id is None:
                    logger.debug(f"Portfolio: no position found for symbol={symbol_or_signal_id}")
                    return None

            pos = self._positions.pop(signal_id, None)
            if pos:
                # Clean up reverse index
                self._symbol_to_signal.pop(pos.symbol, None)
                logger.info(
                    f"Portfolio: closed {pos.symbol} (signal #{signal_id}) | PnL=${pnl_usdt:+,.2f}"
                )
            return pos

    def calculate_funding_cost(self, pos: "Position") -> float:
        """
        T1-FIX: Estimate the total funding fees paid/received while holding
        a perpetual-futures position.

        Binance and most perp exchanges settle funding every 8 hours.
        Funding is paid by the long side when the rate is positive (longs pay
        shorts) and by the short side when the rate is negative.

        Formula:
            periods_held = hold_time_seconds / (8 * 3600)
            funding_cost = abs(funding_rate_8h) * periods_held * size_usdt / 100

        Returns a *positive* USDT value representing money lost to funding.
        Returns 0.0 when funding_rate_8h is not set (non-FundingArb trades
        may not have a meaningful funding rate recorded).
        """
        if not pos or pos.funding_rate_8h == 0.0 or pos.size_usdt <= 0:
            return 0.0
        try:
            hold_seconds = max(0.0, time.time() - pos.opened_at)
            periods_held = hold_seconds / (8 * 3600)
            cost = abs(pos.funding_rate_8h) * periods_held * pos.size_usdt / 100
            return float(cost)
        except Exception:
            return 0.0

    def get_effective_capital(self) -> float:
        """Capital minus unrealized losses from open positions.
        Prevents over-sizing when existing positions are underwater.

        L2-FIX: Floor changed from 50% of current capital to 10% of initial
        configured capital. The old formula (current * 0.5) degraded alongside
        the account — at a 60% drawdown, current=$40k so floor=$20k, producing
        position sizes ~4× larger than actual equity supports. This recreates
        revenge-trade psychology in code (oversizing into a hole to "recover").
        The new floor is a fixed absolute minimum (10% of starting balance),
        ensuring that in a severe drawdown, sizing shrinks to near-zero rather
        than being propped up by a relative floor that itself has shrunk.

        BUG-FIX: The floor itself was previously allowed to exceed current
        capital (e.g. initial=$100k, current=$5k → floor=$10k > equity), which
        produced phantom sizing on capital that no longer exists. Cap the floor
        at current capital so sizing always tracks real equity from below.
        """
        unrealized_loss = sum(
            min(0, p.unrealized_pnl) for p in self._positions.values()
        )
        cur_capital = max(0.0, self._capital)
        initial_floor = min(self._initial_capital * 0.10, cur_capital)
        return max(initial_floor, cur_capital + unrealized_loss)

    async def update_unrealized(self, signal_id: int, pnl_usdt: float):
        """V13: Update unrealized P&L for an open position.

        Defensive clamp: a malformed exchange response (e.g. -1e9) would otherwise
        corrupt get_effective_capital() and freeze the entire sizing pipeline.
        """
        async with self._lock:
            pos = self._positions.get(signal_id)
            if pos:
                try:
                    pnl = float(pnl_usdt)
                except (TypeError, ValueError):
                    return
                # Clamp to ±10× position size — a real fill cannot lose more than
                # the notional, and even with leverage 10× notional is the absolute
                # ceiling for any realistic instrument.
                bound = max(1.0, pos.size_usdt) * 10.0
                pos.unrealized_pnl = max(-bound, min(bound, pnl))


    def get_all_positions(self) -> list:
        """Return all open positions as a list of dicts for the dashboard."""
        return [
            {
                "signal_id":   sig_id,
                "symbol":      pos.symbol,
                "direction":   pos.direction,
                "entry_price": pos.entry_price,
                "size_usdt":   pos.size_usdt,
                "risk_usdt":   pos.risk_usdt,
                "stop_loss":   pos.stop_loss,
                "sector":      getattr(pos, 'sector', ''),
            }
            for sig_id, pos in self._positions.items()
        ]

    def get_state(self) -> PortfolioState:
        """Get current portfolio snapshot"""
        total_long = sum(p.size_usdt for p in self._positions.values() if p.direction == "LONG")
        total_short = sum(p.size_usdt for p in self._positions.values() if p.direction == "SHORT")
        total_risk = sum(p.risk_usdt for p in self._positions.values())

        sector_exp = {}
        direction_counts = {"LONG": 0, "SHORT": 0}
        for p in self._positions.values():
            if p.sector:
                sector_exp[p.sector] = sector_exp.get(p.sector, 0) + p.size_usdt
            direction_counts[p.direction] = direction_counts.get(p.direction, 0) + 1

        return PortfolioState(
            total_capital=self._capital,
            total_risk_deployed=total_risk,
            total_long_exposure=total_long,
            total_short_exposure=total_short,
            net_exposure=total_long - total_short,
            gross_exposure=total_long + total_short,
            position_count=len(self._positions),
            sector_exposure=sector_exp,
            direction_counts=direction_counts,
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """Look up position by symbol via reverse index (FIX P1-A)"""
        signal_id = self._symbol_to_signal.get(symbol)
        if signal_id is not None:
            return self._positions.get(signal_id)
        return None

    def update_capital(self, new_capital: float):
        """Update capital after PnL and persist to DB so restarts resume correctly.

        NC2/NC4-FIX: Also updates _initial_capital (peak floor base) if capital
        has grown beyond its previous high, and persists it separately. This
        ensures the 10% floor tracks the account's all-time high, not just
        the current (possibly drawn-down) capital.
        """
        self._capital = new_capital
        # Track all-time high for floor calculation
        if new_capital > self._initial_capital:
            self._initial_capital = new_capital
        # Fire-and-forget persistence — a failed save is non-fatal (logged only).
        try:
            import asyncio as _asyncio
            loop = _asyncio.get_running_loop()
            loop.create_task(self._save_capital())
        except RuntimeError:
            pass  # No running loop (e.g., called from sync context) — skip persist

    async def _save_capital(self):
        """Persist current capital and peak capital to the DB learning_state table."""
        try:
            from data.database import db
            await db.save_learning_state('portfolio_capital', {'value': self._capital})
            # NC2/NC4-FIX: Persist peak capital separately so restarts don't lose it.
            await db.save_learning_state('portfolio_peak_capital', {'value': self._initial_capital})
        except Exception as e:
            logger.debug(f"Capital persist failed (non-fatal): {e}")

    async def load_capital(self):
        """
        Load the last persisted capital from DB on startup.
        Falls back to config account_balance if no record exists.
        Call this once after db.initialize() in engine.start().

        NC2/NC4-FIX: Loads separately persisted peak capital to set the floor
        base. This handles the case where:
          1. Account grew from $10k → $100k (peak=$100k persisted)
          2. Account drew down to $50k (capital=$50k persisted)
          3. Bot restarts → floor = $100k * 10% = $10k (correct)
        Without persisting peak separately, the floor would be $50k * 10% = $5k,
        which is economically meaningless for a $100k-peak account.
        """
        try:
            from data.database import db
            state = await db.load_learning_state('portfolio_capital')
            if state and isinstance(state.get('value'), (int, float)):
                persisted = float(state['value'])
                if persisted > 0:
                    self._capital = persisted
                    # Defensive baseline: if peak_capital record is missing/corrupt,
                    # at least use max(config, current capital) as the floor base.
                    self._initial_capital = max(self._initial_capital, persisted)

            # NC2/NC4-FIX: Load the persisted peak capital (all-time high).
            peak_state = await db.load_learning_state('portfolio_peak_capital')
            if peak_state and isinstance(peak_state.get('value'), (int, float)):
                persisted_peak = float(peak_state['value'])
                if persisted_peak > 0:
                    self._initial_capital = max(self._initial_capital, persisted_peak)

            logger.info(
                f"💰 Capital restored from DB: ${self._capital:,.2f} "
                f"(floor base: ${self._initial_capital:,.2f})"
            )
            return
        except Exception as e:
            logger.debug(f"Capital load from DB failed (non-fatal): {e}")
        logger.info(
            f"💰 Capital initialised from config: ${self._capital:,.2f}"
        )

    # ── Helpers ───────────────────────────────────────────────

    def _count_correlated(self, symbol: str, new_corr: float, new_sector: str = "", direction: str = "LONG") -> int:
        """
        Count existing positions highly correlated with this one.
        PHASE 2 FIX (CORR-PROXY): Uses actual computed BTC correlation
        passed from the correlation_analyzer, not a hardcoded 0.7 default.
        Also counts same-sector positions as correlated regardless of BTC beta —
        sector liquidation events hit whole groups simultaneously.
        """
        return len(self._get_correlated_positions(symbol, new_corr, new_sector, direction))

    def _get_correlated_positions(
        self,
        symbol: str,
        new_corr: float,
        new_sector: str = "",
        direction: str = "LONG",
    ) -> list[Position]:
        """Return currently open positions that belong to the same correlation cluster."""
        correlated: list[Position] = []
        for pos in self._positions.values():
            if pos.symbol == symbol:
                continue
            btc_correlated = pos.correlation_to_btc > 0.6 and new_corr > 0.6
            same_sector = (
                new_sector and pos.sector and
                new_sector == pos.sector and
                pos.direction == direction
            )
            if btc_correlated or same_sector:
                correlated.append(pos)
        return correlated


    def get_dynamic_kelly(self, strategy: str, regime: str = "",
                          direction: str = "LONG") -> float:
        """
        Dynamic Kelly fraction from live Bayesian posteriors.

        Classic Kelly: f* = (p * b - q) / b
        where b = avg_win / avg_loss, p = P(win), q = 1 - p

        Uses the probability engine's live posteriors rather than
        fixed priors, so the fraction adapts as the bot learns.
        Half-Kelly is applied as standard practice (Kelly is theoretically
        optimal but requires infinite bankroll; half-Kelly is the
        practitioner standard for finite accounts).

        R8-F8 FIX: Previously called probability_engine.get_posterior(strategy, regime)
        which doesn't exist as a public method. The correct API is
        get_strategy_prior(strategy, regime, direction) or
        _get_posterior(context_key) with format "Strategy:REGIME:DIRECTION".

        R8-F8 ENHANCEMENT: Regime-aware sizing multiplier.
        VOLATILE_PANIC caps Kelly at 25% of normal.
        Confirmed BULL_TREND with 65%+ win rate allows full half-Kelly.
        """
        try:
            from core.probability_engine import probability_engine

            # R8-F8 FIX: Use get_strategy_prior() with correct signature
            # (strategy, regime, direction) → returns posterior mean (p_win)
            p_win = probability_engine.get_strategy_prior(
                strategy, regime or "NEUTRAL", direction
            )
            # Clamp to reasonable bounds
            p_win = max(0.30, min(0.85, p_win))

            # Use strategy weights from alpha model for avg_win/loss
            from core.alpha_model import alpha_model
            sw = alpha_model._strategy_weights.get(strategy)
            if sw:
                avg_win  = max(0.5, sw.avg_win_r)
                avg_loss = max(0.5, sw.avg_loss_r)
            else:
                avg_win  = 2.0   # Conservative defaults
                avg_loss = 1.0

            p_loss = 1.0 - p_win
            b = avg_win / avg_loss  # Win/loss ratio
            kelly = (p_win * b - p_loss) / b

            # Half-Kelly with floor/ceiling
            half_kelly = max(0.002, min(self._max_position_risk_pct, kelly * 0.5))

            # R8-F8: Regime-aware sizing multiplier
            regime_mult = 1.0
            if regime == "VOLATILE_PANIC":
                regime_mult = Portfolio.KELLY_MULT_VOLATILE_PANIC
            elif regime == "VOLATILE":
                regime_mult = Portfolio.KELLY_MULT_VOLATILE
            elif regime == "CHOPPY":
                regime_mult = Portfolio.KELLY_MULT_CHOPPY
            elif regime == "BEAR_TREND" and direction == "LONG":
                # Counter-trend: halve Kelly — bear-market longs face strong headwind
                regime_mult = Portfolio.KELLY_MULT_COUNTER_TREND
            elif regime == "BULL_TREND" and direction == "SHORT":
                # Counter-trend: same 0.50× treatment as BEAR_TREND+LONG (symmetric).
                # Previously fell through to BULL_TREND → 0.85, meaning counter-trend
                # SHORTs were sized the same as trend-aligned LONGs. Fixed.
                regime_mult = Portfolio.KELLY_MULT_COUNTER_TREND
            elif regime == "BULL_TREND" and direction == "LONG" and p_win >= Portfolio.BULL_TREND_HIGH_CONF_THRESHOLD:
                regime_mult = 1.0   # Full half-Kelly for high-confidence trend trades
            elif regime == "BULL_TREND":
                regime_mult = Portfolio.KELLY_MULT_BULL_TREND

            adjusted_kelly = half_kelly * regime_mult

            logger.debug(
                f"Dynamic Kelly: {strategy}/{regime}/{direction} → "
                f"p_win={p_win:.3f}, kelly={kelly:.4f}, half={half_kelly:.4f}, "
                f"regime_mult={regime_mult:.2f}, final={adjusted_kelly:.4f}"
            )

            return max(0.002, adjusted_kelly)

        except Exception as e:
            logger.debug(f"Dynamic Kelly fallback for {strategy}: {e}")
            return self._max_position_risk_pct * 0.5  # Conservative fallback


# ── Singleton ─────────────────────────────────────────────────
portfolio_engine = PortfolioEngine()
