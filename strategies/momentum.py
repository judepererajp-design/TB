"""
TitanBot Pro — Momentum Strategy
==================================
Detects momentum setups using MACD crossover with trend confirmation.

Requirements:
  - MACD line crosses signal line (crossover, not just separation)
  - Histogram is expanding (momentum building)
  - ADX > min_adx (trend established)
  - Volume surge >= 2x average at signal bar

Directional regime gate:
  - BULL_TREND: LONG only
  - BEAR_TREND: SHORT only
  - VOLATILE: either direction
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.loader import cfg
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp, compute_vol_percentile

logger = logging.getLogger(__name__)


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    result = np.zeros_like(arr, dtype=float)
    k = 2.0 / (period + 1)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result


def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram) as 1D arrays."""
    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


class Momentum(BaseStrategy):

    name = "Momentum"
    description = "MACD crossover with ADX trend filter and volume surge"

    VALID_REGIMES = {"BULL_TREND", "BEAR_TREND", "VOLATILE"}

    _REGIME_DIR_CONSTRAINT = {
        "BULL_TREND": "LONG",
        "BEAR_TREND": "SHORT",
        "VOLATILE":   None,    # Either direction
    }

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.momentum

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            logger.debug(f"Momentum.analyze {symbol}: {e}")
            return None

    async def _analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime gate ───────────────────────────────────────────────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception:
            regime = "UNKNOWN"

        if regime not in self.VALID_REGIMES:
            return None

        regime_dir = self._REGIME_DIR_CONSTRAINT.get(regime)

        # ── Market state awareness ──────────────────────────────────
        _ms_bonus = 0
        try:
            from analyzers.market_state_engine import market_state_engine, MarketState
            _ms_result = await market_state_engine.get_state()
            _ms_state = _ms_result.state
            if _ms_state == MarketState.EXPANSION:
                _ms_bonus = 10  # Momentum thrives in expanding volatility
            elif _ms_state == MarketState.TRENDING:
                _ms_bonus = 6   # Confirmed trend environment
            elif _ms_state == MarketState.COMPRESSION:
                _ms_bonus = -15  # Momentum signals in compression are noise
            elif _ms_state == MarketState.LIQUIDITY_HUNT:
                _ms_bonus = -10  # Stop hunts create false momentum readings
        except Exception:
            _ms_state = None

        tf = getattr(self._cfg, "timeframe", "1h")
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 60:
            return None

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

        opens   = df["open"].values
        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values
        volumes = df["volume"].values

        atr = self.calculate_atr(highs, lows, closes, period=14)
        if atr == 0:
            return None

        fast_period  = getattr(self._cfg, "macd_fast", 12)
        slow_period  = getattr(self._cfg, "macd_slow", 26)
        sig_period   = getattr(self._cfg, "macd_signal", 9)
        min_adx      = getattr(self._cfg, "min_adx", 30)
        vol_surge    = getattr(self._cfg, "volume_surge_mult", 2.0)
        confidence_base = getattr(self._cfg, "confidence_base", 70)

        # ── MACD calculation ──────────────────────────────────────────────
        macd_line, signal_line, histogram = _macd(closes, fast_period, slow_period, sig_period)

        if len(macd_line) < 3:
            return None

        # Detect crossover on the most recent completed bar ([-2] vs [-1])
        prev_macd_diff  = macd_line[-2] - signal_line[-2]
        curr_macd_diff  = macd_line[-1] - signal_line[-1]

        bullish_cross = prev_macd_diff <= 0 < curr_macd_diff
        bearish_cross = prev_macd_diff >= 0 > curr_macd_diff

        if not bullish_cross and not bearish_cross:
            return None

        direction = "LONG" if bullish_cross else "SHORT"

        # ── Regime direction gate ─────────────────────────────────────────
        if regime_dir and direction != regime_dir:
            return None

        # ── Histogram expansion ───────────────────────────────────────────
        histogram_expanding = abs(histogram[-1]) > abs(histogram[-2])
        require_expansion   = getattr(self._cfg, "histogram_expansion", True)
        if require_expansion and not histogram_expanding:
            return None

        # ── ADX filter ─────────────────────────────────────────────────────
        adx = self.calculate_adx(highs, lows, closes, period=14)
        if adx < min_adx:
            return None

        # ── Volume surge ──────────────────────────────────────────────────
        avg_vol   = float(np.mean(volumes[-20:]))
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio < vol_surge:
            return None

        # ── Confidence ────────────────────────────────────────────────────
        confidence = float(confidence_base)
        confidence += _ms_bonus
        confidence += 5   # MACD crossover
        if histogram_expanding:
            confidence += 5
        adx_bonus   = min(10, (adx - min_adx) * 1.0)
        confidence  += adx_bonus
        if vol_ratio >= 3.0:
            confidence += 5

        # ── Entry / SL / TP ───────────────────────────────────────────────
        current_close = closes[-1]
        buf = atr * 0.5
        vp = compute_vol_percentile(highs, lows, closes)

        if direction == "LONG":
            entry_low   = current_close
            entry_high  = current_close + buf
            swing_low_3 = float(np.min(lows[-4:-1]))
            stop_loss   = swing_low_3 - atr * 0.5
            tp1         = entry_high + atr * rp.volatility_scaled_tp1(tf, vp)
            tp2         = entry_high + atr * rp.volatility_scaled_tp2(tf, vp)
            tp3         = entry_high + atr * rp.volatility_scaled_tp3(tf, vp)
        else:
            entry_high  = current_close
            entry_low   = current_close - buf
            swing_high_3 = float(np.max(highs[-4:-1]))
            stop_loss   = swing_high_3 + atr * 0.5
            tp1         = entry_low - atr * rp.volatility_scaled_tp1(tf, vp)
            tp2         = entry_low - atr * rp.volatility_scaled_tp2(tf, vp)
            tp3         = entry_low - atr * rp.volatility_scaled_tp3(tf, vp)

        # BUG-6 FIX: use entry_mid for risk calculation to match the aggregator's
        # geometry-validation formula (entry_mid - stop_loss / tp2 - entry_mid).
        # Previously used entry_low (LONG) / entry_high (SHORT) which inflated
        # internal R:R by 10-30% and passed signals the aggregator would later correct.
        entry_mid = (entry_low + entry_high) / 2.0
        risk = (entry_mid - stop_loss) if direction == "LONG" else (stop_loss - entry_mid)
        if risk <= 0:
            return None
        rr_ratio = abs(tp2 - entry_mid) / risk

        confluence: List[str] = [
            f"✅ MACD {'bullish' if direction == 'LONG' else 'bearish'} crossover",
            f"   MACD: {macd_line[-1]:.4f} | Signal: {signal_line[-1]:.4f}",
        ]
        if histogram_expanding:
            confluence.append(f"✅ Histogram expanding: {histogram[-1]:.4f} > {histogram[-2]:.4f}")
        confluence.append(f"✅ ADX: {adx:.1f} > {min_adx} — trend established")
        confluence.append(f"✅ Volume surge: {vol_ratio:.1f}x average")
        confluence.append(f"📊 Regime: {regime} | TF: {tf}")
        confluence.append(f"🎯 R:R {rr_ratio:.2f} | ATR: {fmt_price(atr)}")

        if _ms_bonus != 0:
            confluence.append(f"🧠 Market State: {getattr(_ms_state, 'value', 'N/A')} ({'+' if _ms_bonus >= 0 else ''}{_ms_bonus})")

        confidence = min(93, max(40, confidence))

        candidate = SignalResult(
            symbol=symbol,
            direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
            strategy=self.name,
            confidence=confidence,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            tp1=tp1, tp2=tp2, tp3=tp3,
            rr_ratio=rr_ratio,
            atr=atr,
            setup_class="intraday",
            timeframe=tf,
            analysis_timeframes=[tf],
            confluence=confluence,
            raw_data={
                "macd": float(macd_line[-1]),
                "macd_signal": float(signal_line[-1]),
                "histogram": float(histogram[-1]),
                "adx": adx,
                "vol_ratio": vol_ratio,
                "regime": regime,
                "market_state": getattr(_ms_state, 'value', None),
            },
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
MomentumStrategy = Momentum
