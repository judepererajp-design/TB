"""
TitanBot Pro — Risk Parameters Helper
========================================
Centralized access to configurable risk/trade parameters.

All strategies should use these instead of hardcoding values.
Every parameter reads from settings.yaml with sensible defaults.

Usage in any strategy:
    from utils.risk_params import rp
    
    sl = current - atr * rp.sl_atr_mult
    entry_low = current - atr * rp.entry_zone_atr
    if rsi > rp.rsi_overbought: ...
    if adx > rp.adx_strong_trend: ...
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_vol_percentile(highs, lows, closes, lookback: int = 100) -> float:
    """Compute volatility percentile (0.0–1.0) from OHLCV arrays.

    Returns where the recent 5-bar ATR sits in the last ``lookback`` bars
    of true range.  Mirrors the logic in ``core/feature_store.py``.

    Strategies should call this with their numpy arrays to get a vol_percentile
    suitable for passing to ``rp.volatility_scaled_tp*()`` methods.
    """
    try:
        if len(closes) < lookback:
            return 0.5  # insufficient data → neutral
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
        )
        if len(tr) < lookback:
            return 0.5
        recent_atr = float(np.mean(tr[-5:]))
        atr_sorted = np.sort(tr[-lookback:])
        rank = float(np.searchsorted(atr_sorted, recent_atr)) / lookback
        return max(0.0, min(1.0, rank))
    except Exception:
        return 0.5


class RiskParams:
    """
    Reads all risk/trade parameters from config with fallback defaults.
    Single instance shared by all strategies.
    """

    @property
    def _risk_cfg(self):
        try:
            from config.loader import cfg
            return cfg.risk
        except (ImportError, AttributeError):
            return None

    def _get(self, key: str, default):
        """Safe config access"""
        c = self._risk_cfg
        if c is None:
            return default
        return getattr(c, key, default) if hasattr(c, key) else c.get(key, default) if hasattr(c, 'get') else default

    # ── ATR Multipliers ───────────────────────────────────
    # Timeframe scaling: scalp (5m/15m) uses tighter ATR,
    # swing (4h/1d) uses wider ATR. This is how prop desks
    # prevent scalp stops from being too wide and swing
    # stops from being too tight.
    TIMEFRAME_ATR_SCALE = {
        '5m':  0.50,    # Scalp: half the ATR multiplier
        '15m': 0.65,    # Scalp/intraday
        '30m': 0.80,    # Intraday
        '1h':  1.00,    # Base (default)
        '2h':  1.15,    # Swing
        '4h':  1.40,    # Swing
        '1d':  2.00,    # Position
    }

    def atr_scale(self, timeframe: str = '1h') -> float:
        """
        Get ATR scaling factor for the timeframe.
        Scalp timeframes (5m/15m) → tighter stops/entries.
        Swing timeframes (4h/1d) → wider stops/entries.
        """
        return self.TIMEFRAME_ATR_SCALE.get(timeframe, 1.0)

    def scaled_sl(self, timeframe: str = '1h') -> float:
        """SL ATR multiplier scaled for timeframe"""
        return self.sl_atr_mult * self.atr_scale(timeframe)

    def scaled_tp1(self, timeframe: str = '1h') -> float:
        return self.tp1_atr_mult * self.atr_scale(timeframe)

    def scaled_tp2(self, timeframe: str = '1h') -> float:
        return self.tp2_atr_mult * self.atr_scale(timeframe)

    def scaled_tp3(self, timeframe: str = '1h') -> float:
        return self.tp3_atr_mult * self.atr_scale(timeframe)

    def scaled_entry_zone(self, timeframe: str = '1h') -> float:
        return self.entry_zone_atr * self.atr_scale(timeframe)

    @property
    def sl_atr_mult(self) -> float:
        """Stop loss ATR multiplier"""
        return float(self._get('sl_atr_mult', 1.2))

    @property
    def tp1_atr_mult(self) -> float:
        return float(self._get('tp1_atr_mult', 1.5))

    @property
    def tp2_atr_mult(self) -> float:
        return float(self._get('tp2_atr_mult', 2.2))

    @property
    def tp3_atr_mult(self) -> float:
        return float(self._get('tp3_atr_mult', 4.0))

    @property
    def entry_zone_atr(self) -> float:
        """Entry zone width (ATR multiplier)"""
        return float(self._get('entry_zone_atr_mult', 0.20))

    def adaptive_entry_zone(self, vol_percentile: float = 0.5) -> float:
        """V10: Entry zone width scaled by current volatility.
        Tighter in squeezes (low vol), wider in expansion (high vol).
        vol_percentile: 0.0 (lowest vol) to 1.0 (highest vol)."""
        base = self.entry_zone_atr
        return base * (0.7 + 0.6 * max(0.0, min(1.0, vol_percentile)))

    @property
    def entry_zone_tight(self) -> float:
        """Tight entry zone for scalps"""
        return float(self._get('entry_zone_tight_mult', 0.10))

    # ── R8: Adaptive TP/SL Volatility Scaling ────────────────
    def volatility_scaled_sl(self, timeframe: str = '1h',
                             vol_percentile: float = 0.5) -> float:
        """
        SL ATR multiplier dynamically scaled by current volatility percentile.

        In high-volatility environments (90th+ percentile), SLs are widened
        by up to 1.15× to avoid premature stop-outs from noise.
        In low-volatility environments (below 20th percentile), SLs are
        tightened by 0.85× to keep risk proportional to actual movement.

        vol_percentile: 0.0 (lowest vol) → 1.0 (highest vol)
        """
        vp = max(0.0, min(1.0, vol_percentile))
        # Linear scale: 0.85 at vp=0, 1.0 at vp=0.5, 1.15 at vp=1.0
        vol_scale = 0.85 + 0.30 * vp
        return self.sl_atr_mult * self.atr_scale(timeframe) * vol_scale

    def volatility_scaled_tp1(self, timeframe: str = '1h',
                              vol_percentile: float = 0.5) -> float:
        """TP1 ATR multiplier scaled by volatility — wider in high vol."""
        vp = max(0.0, min(1.0, vol_percentile))
        vol_scale = 0.85 + 0.45 * vp  # 0.85 → 1.30
        return self.tp1_atr_mult * self.atr_scale(timeframe) * vol_scale

    def volatility_scaled_tp2(self, timeframe: str = '1h',
                              vol_percentile: float = 0.5) -> float:
        """TP2 ATR multiplier scaled by volatility."""
        vp = max(0.0, min(1.0, vol_percentile))
        vol_scale = 0.85 + 0.45 * vp
        return self.tp2_atr_mult * self.atr_scale(timeframe) * vol_scale

    def volatility_scaled_tp3(self, timeframe: str = '1h',
                              vol_percentile: float = 0.5) -> float:
        """TP3 ATR multiplier scaled by volatility."""
        vp = max(0.0, min(1.0, vol_percentile))
        vol_scale = 0.85 + 0.45 * vp
        return self.tp3_atr_mult * self.atr_scale(timeframe) * vol_scale

    # ── RSI Thresholds ────────────────────────────────────
    @property
    def rsi_overbought(self) -> float:
        return float(self._get('rsi_overbought', 70))

    @property
    def rsi_oversold(self) -> float:
        return float(self._get('rsi_oversold', 30))

    @property
    def rsi_extreme_ob(self) -> float:
        return float(self._get('rsi_extreme_ob', 80))

    @property
    def rsi_extreme_os(self) -> float:
        return float(self._get('rsi_extreme_os', 20))

    # ── ADX Thresholds ────────────────────────────────────
    @property
    def adx_trend(self) -> float:
        """ADX above this = trending"""
        return float(self._get('adx_trend_threshold', 25))

    @property
    def adx_strong_trend(self) -> float:
        """ADX above this = very strong trend"""
        return float(self._get('adx_strong_trend', 40))

    @property
    def adx_no_trend(self) -> float:
        """ADX below this = ranging"""
        return float(self._get('adx_no_trend', 20))

    # ── Volume ────────────────────────────────────────────
    @property
    def volume_spike_mult(self) -> float:
        return float(self._get('volume_spike_mult', 2.0))

    # ── Confidence bonuses (strategy-agnostic) ────────────
    @property
    def conf_rejection_candle(self) -> int:
        """Confidence bonus for rejection candle"""
        return int(self._get('conf_rejection_candle', 8))

    @property
    def conf_volume_spike(self) -> int:
        """Confidence bonus for volume spike"""
        return int(self._get('conf_volume_spike', 5))

    @property
    def conf_rsi_extreme(self) -> int:
        """Confidence bonus for extreme RSI"""
        return int(self._get('conf_rsi_extreme', 7))

    @property
    def conf_multi_tf(self) -> int:
        """Confidence bonus for multi-timeframe agreement"""
        return int(self._get('conf_multi_tf', 10))


# ── Singleton ──────────────────────────────────────────────
rp = RiskParams()
