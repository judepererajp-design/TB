"""
TitanBot Pro — Correlation Analyzer
======================================
Prevents double-exposure to correlated assets.
Measures:
  - BTC beta (how much a coin moves per 1% BTC move)
  - Sector correlation (coins in same sector move together)
  - Portfolio correlation (active signals in same direction)

High correlation = reduce position size or skip if already exposed.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from config.loader import cfg
from data.api_client import api

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Measures correlation between assets to manage portfolio exposure.
    Runs lightweight calculations to stay Mac-friendly.
    """

    def __init__(self):
        self._corr_cfg   = cfg.analyzers.correlation
        self._window     = self._corr_cfg.get('btc_correlation_window', 48)
        self._high_corr  = self._corr_cfg.get('high_correlation_threshold', 0.85)
        self._size_pen   = self._corr_cfg.get('sector_correlation_penalty', 0.8)

        # Cache: symbol -> (btc_beta, correlation, timestamp, window)
        # The dynamic BTC-correlation window changes by regime, so the cached
        # value is only valid when reused with the same window length.
        self._cache: Dict[str, Tuple[float, float, float, int]] = {}
        self._cache_ttl = 1800   # 30 minutes

        # Active signals for portfolio tracking
        self._active_signals: Dict[str, str] = {}   # symbol -> direction

    async def get_btc_beta(self, symbol: str, ohlcv_symbol: List) -> Tuple[float, float]:
        """
        Calculate BTC beta and correlation with dynamic window by regime.

        Key insight: In trending markets, use longer window (stable correlation).
        In volatile/choppy markets, use shorter window (correlation breaks down).

        Beta (Cov(sym,btc)/Var(btc)) tells us directional sensitivity.
        Correlation (Pearson) tells us how reliably beta holds.
        We report both — high beta + low correlation = unreliable relationship.
        """
        import time

        # Dynamic window: shorter in volatile regimes (correlation is unstable)
        try:
            from analyzers.regime import regime_analyzer
            _reg = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
            dynamic_window = {
                "BULL_TREND": 72,   # 3 days — stable trend, use longer history
                "BEAR_TREND": 72,
                "CHOPPY":     24,   # 1 day — range-bound, correlation noisy
                "VOLATILE":   12,   # 12h — correlation breaks in panic
                "VOLATILE_PANIC":   24,
            }.get(_reg, self._window)
        except Exception:
            dynamic_window = self._window

        cached = self._cache.get(symbol)
        if cached:
            beta, corr, ts, cached_window = cached
            if cached_window == dynamic_window and time.time() - ts < self._cache_ttl:
                return beta, corr

        btc_ohlcv = await api.fetch_ohlcv("BTC/USDT", "1h", limit=dynamic_window + 1)
        if not btc_ohlcv or not ohlcv_symbol or len(ohlcv_symbol) < dynamic_window:
            return 1.0, 0.5

        # Align lengths
        n = min(len(btc_ohlcv), len(ohlcv_symbol), dynamic_window) - 1
        btc_closes = np.array([c[4] for c in btc_ohlcv[-n-1:]], dtype=float)
        sym_closes = np.array([c[4] for c in ohlcv_symbol[-n-1:]], dtype=float)

        btc_returns = np.diff(np.log(btc_closes + 1e-10))
        sym_returns = np.diff(np.log(sym_closes + 1e-10))

        if len(btc_returns) < 10 or np.std(btc_returns) == 0:
            return 1.0, 0.5

        # Beta = Cov(sym, btc) / Var(btc)
        cov  = np.cov(sym_returns, btc_returns)[0, 1]
        var  = np.var(btc_returns)
        beta = cov / var if var > 0 else 1.0

        # Pearson correlation
        corr_matrix = np.corrcoef(sym_returns, btc_returns)
        corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.5

        self._cache[symbol] = (beta, corr, time.time(), dynamic_window)
        return float(beta), float(corr)

    def assess_portfolio_exposure(self, symbol: str, direction: str) -> Tuple[float, List[str]]:
        """
        Check if adding this signal creates too much correlated exposure.
        Returns (confidence_adjustment, warning_notes).
        """
        notes        = []
        adjustment   = 0.0

        from analyzers.altcoin_rotation import rotation_tracker
        sector       = rotation_tracker.get_sector_for_symbol(symbol)

        # Count how many active signals are in the same sector/direction
        same_sector_count = 0
        for active_sym, active_dir in self._active_signals.items():
            if active_sym == symbol:
                continue
            active_sector = rotation_tracker.get_sector_for_symbol(active_sym)
            if active_sector == sector and active_dir == direction:
                same_sector_count += 1

        max_correlated = cfg.risk.get('max_correlated_positions', 2)

        if same_sector_count >= max_correlated:
            adjustment -= 15
            notes.append(
                f"⚠️ Already {same_sector_count} active {direction} signals in {sector} sector"
            )
        elif same_sector_count == 1:
            adjustment -= 5
            notes.append(f"ℹ️ 1 other {direction} signal in {sector} sector")

        return adjustment, notes

    def register_active_signal(self, symbol: str, direction: str):
        """Track that a signal was sent"""
        self._active_signals[symbol] = direction

    def unregister_signal(self, symbol: str):
        """Remove a signal from tracking (after outcome recorded)"""
        self._active_signals.pop(symbol, None)

    def get_size_multiplier(self, btc_beta: float) -> float:
        """
        Adjust position size based on BTC beta.
        High beta coins are riskier — reduce size.
        """
        if btc_beta > 2.5:
            return 0.5    # High beta: half size
        elif btc_beta > 1.8:
            return 0.7
        elif btc_beta < 0:
            return 0.6    # Inverse correlation: somewhat risky
        return 1.0


    def get_cached_correlation(self, symbol: str) -> float:
        """
        PHASE 2 FIX (CORR-PROXY): Return the cached BTC correlation for a symbol.
        Returns the actual computed Pearson correlation coefficient (0-1) rather
        than the hardcoded 0.7 default that portfolio_engine was using.
        Falls back to 0.65 (mid-range assumption) if not yet computed.

        Call get_btc_beta() (async) to populate the cache before using this.
        """
        cached = self._cache.get(symbol)
        if cached:
            _beta, corr, _ts, _window = cached
            import time
            if time.time() - _ts < self._cache_ttl:
                return float(corr)
        # Not cached yet — return a conservative mid-range assumption
        # (not 0.7 flat — 0.65 is slightly lower to avoid over-blocking new signals)
        return 0.65

# ── Singleton ──────────────────────────────────────────────
correlation_analyzer = CorrelationAnalyzer()
