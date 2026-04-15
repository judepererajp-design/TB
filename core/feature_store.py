"""
TitanBot Pro — Feature Store
===============================
Standardized features computed once and reused across strategies,
probability engine, and alpha model.

Features:
  - Volatility (ATR-based, realized, implied ratio)
  - Trend strength (ADX, linear regression slope)
  - Volume profile (relative volume, VWAP distance, OBV trend)
  - Momentum (RSI, MACD histogram, rate of change)
  - Orderflow (funding rate, OI change, long/short ratio)
  - Regime (probabilities, not categories)
  - Correlation (BTC beta, sector correlation)
  - Liquidity (spread, depth, liquidation proximity)

All features are computed in a single pass and cached per symbol+timeframe.
"""

import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Complete feature set for a symbol at a point in time"""
    symbol: str
    timeframe: str
    timestamp: float

    # ── Volatility ────────────────────────────────────────────
    atr_14: float = 0.0              # 14-period ATR
    atr_pct: float = 0.0             # ATR as % of price
    realized_vol_20: float = 0.0     # 20-bar realized volatility
    vol_percentile: float = 0.5      # Current vol vs recent range (0-1)
    is_expanding: bool = False       # Vol expanding?
    is_contracting: bool = False     # Vol contracting?

    # ── Trend ─────────────────────────────────────────────────
    adx: float = 0.0                 # Average Directional Index
    trend_slope: float = 0.0         # Linear regression slope (normalized)
    trend_r2: float = 0.0            # R² of trend fit (how clean)
    ema_20_dist: float = 0.0         # Distance from 20 EMA (% of price)
    ema_50_dist: float = 0.0         # Distance from 50 EMA
    ema_cross: int = 0               # +1 bullish cross, -1 bearish, 0 none

    # ── Volume ────────────────────────────────────────────────
    relative_volume: float = 1.0     # Current vol / 20-bar avg vol
    vwap_distance: float = 0.0       # Distance from VWAP (% of price)
    obv_trend: float = 0.0           # OBV slope (normalized)
    volume_spike: bool = False       # > 3x average volume

    # ── Momentum ──────────────────────────────────────────────
    rsi_14: float = 50.0             # RSI
    macd_histogram: float = 0.0      # MACD histogram
    roc_10: float = 0.0              # 10-bar rate of change (%)
    stoch_k: float = 50.0            # Stochastic %K
    is_overbought: bool = False      # RSI > 70 or Stoch > 80
    is_oversold: bool = False        # RSI < 30 or Stoch < 20

    # ── Orderflow (when available) ────────────────────────────
    funding_rate: float = 0.0        # Current funding rate
    funding_annualized: float = 0.0  # Annualized funding
    oi_change_24h: float = 0.0       # OI change % in 24h
    long_short_ratio: float = 1.0    # Long/short account ratio

    # ── Context ───────────────────────────────────────────────
    btc_correlation: float = 0.7     # Rolling 30-bar BTC correlation
    is_killzone: bool = False        # In active killzone session
    session: str = "OFF"             # LONDON | NY | ASIA | OVERLAP | OFF

    # ── BUG-NEW-2 FIX: engine-injectable fields for the 11 missing LR keys ──
    # These cannot be derived from OHLCV alone — they are set by engine.py
    # after strategy and HTF analysis runs, then passed into FeatureStore.
    # They were computed all along; they just weren't flowing into evidence_dict.
    htf_aligned: bool = False            # HTF structure confirms signal direction
    mtf_aligned: bool = False            # MTF (1h) structure confirms direction
    ob_fvg_confluence: bool = False      # Order Block + FVG overlap present
    liquidity_sweep_present: bool = False  # Liquidity sweep detected on this bar
    liq_cluster_overlap: bool = False    # OB sits on a liquidation cluster
    bar_confirmation: bool = False       # Entry candle confirms direction
    multi_strategy_agree: bool = False   # 2+ independent strategy clusters agree
    premium_discount_zone: bool = False  # Price in premium/discount zone for direction
    counter_regime: bool = False         # Signal direction opposes current regime
    is_dead_zone: bool = False           # Current time is a low-liquidity dead zone
    strategy_on_cold_streak: bool = False  # Issuing strategy has negative EV last 15 trades

    def to_evidence_dict(self) -> Dict[str, bool]:
        """
        Convert features into binary evidence for probability engine.

        BUG-NEW-2 FIX: The previous version populated 19 keys but missed the 11
        highest-impact likelihood ratio keys. Those keys existed in the LR table
        but were never present in evidence, so they contributed exactly 0 to every
        Bayesian log-odds calculation — making the engine blind to OB/FVG overlap,
        HTF alignment, liquidity sweeps, and cold streaks.

        The 11 missing keys are now populated from the engine-injectable fields
        added to FeatureSet (htf_aligned, ob_fvg_confluence, etc.). The engine
        sets these after strategy analysis runs, before calling probability_engine.
        """
        import datetime as _dt
        _now = _dt.datetime.now(_dt.timezone.utc)
        return {
            # ── Volume ───────────────────────────────────────────
            'volume_confirmation': self.relative_volume > 1.5,
            'low_volume':          self.relative_volume < 0.5,
            'volume_spike':        self.volume_spike,

            # ── Session / timing ─────────────────────────────────
            'killzone_active':     self.is_killzone,
            'dead_zone_time':      self.is_dead_zone,          # LR=-0.12  FIXED
            'weekend':             self.session == "OFF" and _now.weekday() >= 5,

            # ── Correlation risk ──────────────────────────────────
            'high_correlation':    self.btc_correlation > 0.85,

            # ── Funding / derivatives ─────────────────────────────
            'funding_favorable':   abs(self.funding_rate) > 0.0005,

            # ── Price location ────────────────────────────────────
            'vwap_alignment':      abs(self.vwap_distance) < 0.005,
            'premium_discount_zone': self.premium_discount_zone,  # LR=+0.12  FIXED

            # ── Momentum ─────────────────────────────────────────
            'rsi_extreme':         self.is_overbought or self.is_oversold,
            'strong_trend':        self.adx > 30,
            'weak_trend':          self.adx < 15,
            'ema_bullish_cross':   self.ema_cross == 1,
            'ema_bearish_cross':   self.ema_cross == -1,
            'above_ema50':         self.ema_50_dist > 0,
            'below_ema50':         self.ema_50_dist < 0,
            'obv_rising':          self.obv_trend > 0.01,
            'obv_falling':         self.obv_trend < -0.01,

            # ── Volatility context ────────────────────────────────
            'vol_expanding':       self.is_expanding,
            'vol_contracting':     self.is_contracting,

            # ── BUG-NEW-2 FIX: the 11 previously-missing LR keys ─
            # These are set by engine.py after strategy + HTF analysis:
            'htf_alignment':       self.htf_aligned,           # LR=+0.15  FIXED
            'mtf_alignment':       self.mtf_aligned,           # LR=+0.10  FIXED
            'ob_fvg_confluence':   self.ob_fvg_confluence,     # LR=+0.20  FIXED
            'liquidity_sweep':     self.liquidity_sweep_present, # LR=+0.18 FIXED
            'liq_cluster_overlap': self.liq_cluster_overlap,   # LR=+0.22  FIXED
            'bar_confirmation':    self.bar_confirmation,       # LR=+0.08  FIXED
            'multi_strategy_agree': self.multi_strategy_agree, # LR=+0.25  FIXED
            'counter_regime':      self.counter_regime,        # LR=-0.15  FIXED
            'strategy_cold_streak': self.strategy_on_cold_streak,  # LR=-0.15  FIXED
        }


class FeatureStore:
    """
    Computes and caches features for each symbol.
    Features are computed once per scan cycle and shared across all consumers.
    """

    def __init__(self):
        self._cache: Dict[str, FeatureSet] = {}  # "symbol:tf" -> FeatureSet
        self._cache_ttl = 120  # 2 minute cache

    def compute(
        self,
        symbol: str,
        timeframe: str,
        candles: list,
        funding_rate: float = 0.0,
        oi_change: float = 0.0,
        long_short_ratio: float = 1.0,
        btc_candles: list = None,
    ) -> FeatureSet:
        """
        Compute all features from raw OHLCV data.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            candles: List of [timestamp, open, high, low, close, volume]
            funding_rate: Current funding rate
            oi_change: OI change in 24h (%)
            long_short_ratio: Exchange long/short ratio
            btc_candles: BTC candles for correlation (optional)

        Returns:
            FeatureSet with all computed features
        """
        cache_key = f"{symbol}:{timeframe}"

        # Check cache
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached.timestamp < self._cache_ttl:
            return cached

        if len(candles) < 50:
            return FeatureSet(symbol=symbol, timeframe=timeframe, timestamp=time.time())

        # Extract arrays
        opens = np.array([float(c[1]) for c in candles])
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])

        current_price = closes[-1]

        fs = FeatureSet(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=time.time(),
        )

        # ── Volatility features ───────────────────────────────
        try:
            tr = np.maximum(highs[1:] - lows[1:],
                            np.maximum(abs(highs[1:] - closes[:-1]),
                                       abs(lows[1:] - closes[:-1])))
            atr_14 = self._wilder_smooth(tr, 14)
            fs.atr_14 = atr_14
            fs.atr_pct = atr_14 / current_price if current_price > 0 else 0

            # Guard against zero/negative close prices producing NaN/Inf
            close_ratios = closes[1:] / np.where(closes[:-1] > 0, closes[:-1], np.nan)
            log_returns = np.log(np.where(close_ratios > 0, close_ratios, np.nan))
            # Use nanstd so NaN/inf from zero or near-zero prices are ignored
            # without shortening the array (which would misalign the [-20:] slice).
            fs.realized_vol_20 = float(np.nanstd(log_returns[-20:])) if len(log_returns) >= 20 else 0

            # Vol percentile (where is current vol vs last 100 bars?)
            if len(tr) >= 100:
                recent_atr = float(np.mean(tr[-5:]))
                atr_100 = sorted(tr[-100:])
                rank = np.searchsorted(atr_100, recent_atr) / 100
                fs.vol_percentile = float(rank)

            # Expanding vs contracting
            if len(tr) >= 20:
                vol_short = float(np.mean(tr[-5:]))
                vol_long = float(np.mean(tr[-20:]))
                fs.is_expanding = vol_short > vol_long * 1.2
                fs.is_contracting = vol_short < vol_long * 0.8
        except Exception:
            pass

        # ── Trend features ────────────────────────────────────
        try:
            fs.adx = self._adx(highs, lows, closes, 14)

            # Linear regression slope (last 20 bars)
            if len(closes) >= 20:
                x = np.arange(20)
                y = closes[-20:]
                slope, intercept = np.polyfit(x, y, 1)
                fs.trend_slope = slope / current_price if current_price > 0 else 0

                # R-squared — guard against flat price series (ss_tot=0) and NaN
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                fs.trend_r2 = r2 if np.isfinite(r2) else 0.0

            # EMA distances
            ema_20 = self._ema(closes, 20)
            ema_50 = self._ema(closes, 50) if len(closes) >= 50 else ema_20
            fs.ema_20_dist = (current_price - ema_20) / current_price if current_price > 0 else 0
            fs.ema_50_dist = (current_price - ema_50) / current_price if current_price > 0 else 0

            # EMA cross detection
            if len(closes) >= 51:
                prev_ema20 = self._ema(closes[:-1], 20)
                prev_ema50 = self._ema(closes[:-1], 50)
                if prev_ema20 <= prev_ema50 and ema_20 > ema_50:
                    fs.ema_cross = 1
                elif prev_ema20 >= prev_ema50 and ema_20 < ema_50:
                    fs.ema_cross = -1
        except Exception:
            pass

        # ── Volume features ───────────────────────────────────
        try:
            avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
            fs.relative_volume = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
            fs.volume_spike = fs.relative_volume > 3.0

            # VWAP
            if len(closes) >= 20:
                cum_vol = np.cumsum(volumes[-20:])
                cum_pv = np.cumsum(closes[-20:] * volumes[-20:])
                vwap = cum_pv[-1] / cum_vol[-1] if cum_vol[-1] > 0 else current_price
                fs.vwap_distance = (current_price - vwap) / current_price if current_price > 0 else 0

            # OBV trend
            obv = np.zeros(len(closes))
            for i in range(1, len(closes)):
                if closes[i] > closes[i - 1]:
                    obv[i] = obv[i - 1] + volumes[i]
                elif closes[i] < closes[i - 1]:
                    obv[i] = obv[i - 1] - volumes[i]
                else:
                    obv[i] = obv[i - 1]
            if len(obv) >= 10:
                obv_slope = (obv[-1] - obv[-10]) / (abs(obv[-10]) + 1)
                fs.obv_trend = float(obv_slope)
        except Exception:
            pass

        # ── Momentum features ─────────────────────────────────
        try:
            fs.rsi_14 = self._rsi(closes, 14)
            fs.is_overbought = fs.rsi_14 > 70
            fs.is_oversold = fs.rsi_14 < 30

            # MACD histogram
            ema12 = self._ema(closes, 12)
            ema26 = self._ema(closes, 26)
            macd_line = ema12 - ema26
            # FIX #23: Normalise MACD by ATR (not price). MACD is in price units;
            # dividing by price gives ~0.0003 for BTC — useless for cross-asset comparison.
            # ATR normalisation makes the value comparable across symbols and timeframes.
            _atr_for_macd = fs.atr_14 if fs.atr_14 > 0 else (current_price * 0.01)
            fs.macd_histogram = float(macd_line) / _atr_for_macd if _atr_for_macd > 0 else 0

            # Rate of change
            if len(closes) >= 11:
                fs.roc_10 = (closes[-1] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0

            # Stochastic
            if len(closes) >= 14:
                low14 = float(np.min(lows[-14:]))
                high14 = float(np.max(highs[-14:]))
                if high14 > low14:
                    fs.stoch_k = (current_price - low14) / (high14 - low14) * 100
        except Exception:
            pass

        # ── Orderflow features ────────────────────────────────
        fs.funding_rate = funding_rate
        fs.funding_annualized = funding_rate * 3 * 365  # 3x per day
        fs.oi_change_24h = oi_change
        fs.long_short_ratio = long_short_ratio

        # ── Correlation ───────────────────────────────────────
        # FIX #24: Use 720-bar window (30 days on 1h) for BTC correlation.
        # 30 bars = 30 hours — a single volatile day can swing correlation ±0.5,
        # making the BTC correlation gate unreliable for the portfolio engine's T3 check.
        # 720 bars gives a stable, regime-representative correlation estimate.
        _corr_window = min(720, len(closes), len(btc_candles) if btc_candles else 0)
        if btc_candles and _corr_window >= 30:
            try:
                btc_closes_corr = np.array([float(c[4]) for c in btc_candles[-_corr_window:]])
                sym_closes_corr = closes[-_corr_window:]
                min_len = min(len(btc_closes_corr), len(sym_closes_corr))
                if min_len >= 10:
                    btc_ret = np.diff(np.log(btc_closes_corr[-min_len:]))
                    sym_ret = np.diff(np.log(sym_closes_corr[-min_len:]))
                    corr = np.corrcoef(btc_ret, sym_ret)[0, 1]
                    fs.btc_correlation = float(corr) if not np.isnan(corr) else 0.7
            except Exception:
                pass

        # ── Session detection ─────────────────────────────────
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            hour = now.hour
            if 7 <= hour < 9:
                fs.session = "LONDON"
                fs.is_killzone = True
            elif 12 <= hour < 14:
                fs.session = "NY"
                fs.is_killzone = True
            elif 13 <= hour < 16:
                fs.session = "OVERLAP"
                fs.is_killzone = True
            elif 0 <= hour < 3:
                fs.session = "ASIA"
            else:
                fs.session = "OFF"
        except Exception:
            pass

        # Cache result
        self._cache[cache_key] = fs
        return fs

    def get_cached(self, symbol: str, timeframe: str) -> Optional[FeatureSet]:
        """Get cached features if available"""
        key = f"{symbol}:{timeframe}"
        cached = self._cache.get(key)
        if cached and time.time() - cached.timestamp < self._cache_ttl:
            return cached
        return None

    def clear_cache(self):
        """Clear the feature cache"""
        self._cache.clear()

    # ── Technical indicator helpers ────────────────────────────

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Compute EMA and return the last value"""
        if len(data) < period:
            return float(np.mean(data))
        alpha = 2.0 / (period + 1)
        ema = float(data[0])
        for val in data[1:]:
            ema = alpha * float(val) + (1 - alpha) * ema
        return ema

    @staticmethod
    def _wilder_smooth(data: np.ndarray, period: int) -> float:
        """Wilder's smoothing (as used in ATR, ADX)"""
        if len(data) < period:
            return float(np.mean(data))
        smooth = float(np.mean(data[:period]))
        for i in range(period, len(data)):
            smooth = (smooth * (period - 1) + float(data[i])) / period
        return smooth

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        """RSI using Wilder's smoothing"""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _adx(highs, lows, closes, period=14) -> float:
        """ADX using Wilder's smoothing"""
        if len(highs) < period * 2:
            return 0.0

        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1]))
        )

        # +DM and -DM
        up = highs[1:] - highs[:-1]
        down = lows[:-1] - lows[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        # Wilder smooth
        atr = float(np.mean(tr[:period]))
        pdm = float(np.mean(plus_dm[:period]))
        mdm = float(np.mean(minus_dm[:period]))

        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + float(tr[i])) / period
            pdm = (pdm * (period - 1) + float(plus_dm[i])) / period
            mdm = (mdm * (period - 1) + float(minus_dm[i])) / period

        if atr == 0:
            return 0.0

        pdi = 100 * pdm / atr
        mdi = 100 * mdm / atr
        dx = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0

        return float(dx)


# ── Singleton ─────────────────────────────────────────────────

    def weighted_evidence_score(self) -> float:
        """
        Compute a single MI-weighted evidence score in [0, 1].

        This is the information-theoretic equivalent of confidence:
        it measures how much the current feature set predicts a winning trade,
        weighted by each feature's mutual information with outcomes.

        Returns a float in [0, 1] where:
          0.0 = no informative features present
          0.5 = neutral / mixed evidence
          1.0 = all high-MI features strongly present (theoretical max)
        """
        evidence = self.to_evidence_dict()
        total_mi = 0.0
        weighted_sum = 0.0

        for key, present in evidence.items():
            mi = self.MI_WEIGHTS.get(key, 0.10)  # Default MI for unmapped features
            total_mi += mi
            if present:
                weighted_sum += mi

        return weighted_sum / total_mi if total_mi > 0 else 0.5

    def regime_adjusted_score(self, regime: str) -> float:
        """
        Adjust the evidence score based on which features matter in this regime.
        Uses a simplified regime-feature interaction matrix.
        """
        base = self.weighted_evidence_score()
        evidence = self.to_evidence_dict()

        # Regime-specific feature bonuses (additive to MI weights)
        regime_bonuses = {
            "BULL_TREND":  {'htf_aligned': 0.15, 'trend_aligned': 0.12, 'oi_rising': 0.10},
            "BEAR_TREND":  {'htf_aligned': 0.15, 'trend_aligned': 0.12, 'oi_rising': 0.10},
            "CHOPPY":      {'vwap_alignment': 0.15, 'killzone_active': 0.12, 'rsi_not_extreme': 0.10},
            "VOLATILE":    {'volume_spike': 0.15, 'funding_favorable': 0.12},
            "VOLATILE_PANIC": {'killzone_active': 0.10, 'funding_favorable': 0.15},
        }
        bonuses = regime_bonuses.get(regime, {})
        bonus_sum = sum(v for k, v in bonuses.items() if evidence.get(k, False))

        return min(1.0, base + bonus_sum * 0.3)  # Cap at 1.0



feature_store = FeatureStore()
