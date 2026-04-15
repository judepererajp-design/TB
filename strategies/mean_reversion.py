"""
TitanBot Pro — Mean Reversion Strategy
========================================
Detects statistical overextensions using z-score analysis.
Fires ONLY in CHOPPY regimes where price oscillates around a mean.

Requirements:
  - Z-score > threshold (price significantly above/below rolling mean)
  - ADX < max_adx (not trending — ensures range-bound conditions)
  - Rejection candle at the extreme
  - Volume confirmation (1.5x average)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.loader import cfg
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp

logger = logging.getLogger(__name__)


class MeanReversion(BaseStrategy):

    name = "MeanReversion"
    description = "Z-score mean reversion in choppy/ranging markets"

    VALID_REGIMES = {"CHOPPY"}

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.mean_reversion

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            logger.debug(f"MeanReversion.analyze {symbol}: {e}")
            return None

    async def _analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime gate — mean reversion only in choppy markets ───────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception:
            regime = "UNKNOWN"

        if regime not in self.VALID_REGIMES:
            return None

        tf = "1h"
        for candidate_tf in ("1h", "15m"):
            if candidate_tf in ohlcv_dict and len(ohlcv_dict[candidate_tf]) >= 60:
                tf = candidate_tf
                break
        else:
            return None

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

        z_period    = getattr(self._cfg, "z_score_period", 48)
        z_threshold = getattr(self._cfg, "z_score_threshold", 2.2)
        max_adx     = getattr(self._cfg, "max_adx", 35)
        vol_mult    = getattr(self._cfg, "volume_confirmation_mult", 1.5)
        confidence_base = getattr(self._cfg, "confidence_base", 65)

        if len(closes) < z_period + 1:
            return None

        # ── Z-score calculation ───────────────────────────────────────────
        window    = closes[-z_period:]
        mean      = float(np.mean(window))
        std       = float(np.std(window, ddof=1))
        if std < 1e-12:
            return None

        current_price = closes[-1]
        z_score       = (current_price - mean) / std

        if abs(z_score) < z_threshold:
            return None

        # ── ADX filter — ensure not trending ─────────────────────────────
        adx = self.calculate_adx(highs, lows, closes, period=14)
        if adx >= max_adx:
            return None

        # ── Volume confirmation ────────────────────────────────────────────
        avg_vol = float(np.mean(volumes[-z_period:]))
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio < vol_mult:
            return None

        # ── Rejection candle check ────────────────────────────────────────
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        body_size  = abs(c - o)
        total_size = h - l
        is_rejection = False
        if total_size > 0:
            if z_score < 0:  # Looking for LONG — need bullish rejection (hammer)
                lower_wick = min(o, c) - l
                is_rejection = lower_wick > body_size * 1.5 and c > o
            else:  # Looking for SHORT — need bearish rejection (shooting star)
                upper_wick = h - max(o, c)
                is_rejection = upper_wick > body_size * 1.5 and c < o

        require_rejection = getattr(self._cfg, "require_rejection", True)
        if require_rejection and not is_rejection:
            return None

        # ── Direction ─────────────────────────────────────────────────────
        direction = "LONG" if z_score < -z_threshold else "SHORT"

        # ── Confidence ────────────────────────────────────────────────────
        confidence = float(confidence_base)
        z_excess   = abs(z_score) - z_threshold
        confidence += min(12, z_excess * 4)           # z-score excess bonus
        if vol_ratio >= 2.0:
            confidence += 5
        if is_rejection:
            confidence += 8
        if adx < 20:
            confidence += 5   # Very rangy = higher reversion probability

        # ── Entry zone ────────────────────────────────────────────────────
        buf = atr * rp.entry_zone_tight

        if direction == "LONG":
            entry_low  = current_price - buf
            entry_high = current_price + buf
            # SL: 0.8x ATR past the extreme (further from mean)
            stop_loss  = l - atr * 0.8
            # TP1: 25% reversion toward mean
            tp1 = current_price + (mean - current_price) * 0.25
            # TP2: full reversion to mean
            tp2 = mean
            # TP3: opposite z-score level
            tp3 = mean + std * z_threshold
        else:
            entry_low  = current_price - buf
            entry_high = current_price + buf
            stop_loss  = h + atr * 0.8
            tp1 = current_price - (current_price - mean) * 0.25
            tp2 = mean
            tp3 = mean - std * z_threshold

        # Sanity: tp1 must be strictly in the right direction
        if direction == "LONG":
            tp1 = max(tp1, entry_high + atr * 0.5)
            tp2 = max(tp2, tp1 + atr * 0.3)
            tp3 = max(tp3, tp2 + atr * 0.3)
        else:
            tp1 = min(tp1, entry_low - atr * 0.5)
            tp2 = min(tp2, tp1 - atr * 0.3)
            tp3 = min(tp3, tp2 - atr * 0.3)

        risk = (entry_low - stop_loss) if direction == "LONG" else (stop_loss - entry_high)
        if risk <= 0:
            return None
        rr_ratio = abs(tp2 - current_price) / risk

        confluence: List[str] = [
            f"✅ Z-score: {z_score:.2f} (threshold: ±{z_threshold})",
            f"✅ Rolling mean ({z_period}): {fmt_price(mean)} | Std: {fmt_price(std)}",
            f"✅ ADX: {adx:.1f} < {max_adx} — range confirmed",
            f"✅ Volume: {vol_ratio:.1f}x average",
        ]
        if is_rejection:
            confluence.append("✅ Rejection candle at extreme")
        confluence.append(f"📊 Regime: {regime} | TF: {tf}")
        confluence.append(f"🎯 R:R {rr_ratio:.2f} | ATR: {fmt_price(atr)}")

        confidence = min(92, max(40, confidence))

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
            analysis_timeframes=["1h", "15m"],
            confluence=confluence,
            raw_data={
                "z_score": z_score,
                "rolling_mean": mean,
                "rolling_std": std,
                "adx": adx,
                "vol_ratio": vol_ratio,
                "regime": regime,
            },
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
MeanReversionStrategy = MeanReversion
