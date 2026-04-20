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
from config.constants import STRATEGY_VALID_REGIMES
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp

logger = logging.getLogger(__name__)


class MeanReversion(BaseStrategy):

    name = "MeanReversion"
    description = "Z-score mean reversion in choppy/ranging markets"

    VALID_REGIMES = STRATEGY_VALID_REGIMES["MeanReversion"]

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.mean_reversion

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            self._record_analyze_error(self.name, e, symbol)
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

        # BTC trend guard — alts move with ~80% BTC correlation in trending markets.
        # If BTC's own ADX indicates it is trending strongly, mean-reverting an alt
        # is likely to get swept by the macro move regardless of local CHOPPY regime.
        # Graduated: soft −10 confidence at btc_adx > 30; hard block at btc_adx > 40.
        _btc_adx = 0.0
        _btc_bearish = False
        try:
            from analyzers.regime import regime_analyzer as _ra
            _btc_adx      = float(getattr(_ra, '_btc_adx', 0.0))
            # _btc_ma_bullish is True when fast MA > slow MA (BTC uptrend)
            _btc_ma_bull  = bool(getattr(_ra, '_btc_ma_bullish', True))
            _btc_bearish  = not _btc_ma_bull
        except Exception:
            pass
        _btc_trend_penalty = 0
        if _btc_adx > 40:
            return None
        elif _btc_adx > 30:
            _btc_trend_penalty = -10

        tf = "1h"
        for candidate_tf in ("1h", "15m"):
            if candidate_tf in ohlcv_dict and len(ohlcv_dict[candidate_tf]) >= 60:
                tf = candidate_tf
                break
        else:
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

        # ATR expansion guard — if volatility is expanding (range breaking), fading
        # via mean-reversion is exactly wrong.  Use the 50-bar ATR as a baseline; if
        # the current 14-bar ATR exceeds it by 50%, the market is no longer ranging.
        if len(closes) >= 50:
            atr_50 = self.calculate_atr(highs, lows, closes, period=50)
            if atr_50 > 0 and atr > atr_50 * 1.5:
                return None

        z_period    = getattr(self._cfg, "z_score_period", 48)
        z_threshold = getattr(self._cfg, "z_score_threshold", 2.2)
        max_adx     = getattr(self._cfg, "max_adx", 25)
        vol_mult    = getattr(self._cfg, "volume_confirmation_mult", 1.5)
        confidence_base = getattr(self._cfg, "confidence_base", 65)

        if len(closes) < z_period + 1:
            return None

        # ── Z-score calculation ───────────────────────────────────────────
        window    = closes[-z_period:]
        mean      = float(np.mean(window))
        std       = float(np.std(window, ddof=1))
        # MR-3: Coefficient-of-variation guard — a near-zero CV means the window
        # is structurally flat (dead market), which would produce astronomically
        # large z-scores on any tiny price tick.  Also guard the numeric floor.
        if std < 1e-12 or (abs(mean) > 0 and std / abs(mean) < 0.001):
            return None

        current_price = closes[-1]
        # Clamp z-score to ±4.0 — crypto fat tails can produce z-scores of 6+
        # which are mathematically valid but practically indistinguishable from 4.
        # Clamping prevents extreme values from inflating the confidence bonus.
        z_score = float(max(-4.0, min(4.0, (current_price - mean) / std)))

        if abs(z_score) < z_threshold:
            return None

        # MR-Q3: Time-in-range confirmation — a genuine ranging market oscillates
        # around its mean multiple times within the z_period window.  Count how
        # many times the raw deviation changes sign; fewer than 4 crossings
        # suggests a trending regime that has only recently entered "choppy"
        # classification (regime-detection lag) rather than established ranging.
        _deviation    = closes[-z_period:] - mean
        _sign_changes = int(np.sum(np.diff(np.sign(_deviation)) != 0))
        if _sign_changes < 4:
            return None

        # Range stability guard — if recent volatility is expanding relative to
        # the earlier part of the window, the range is likely breaking out.
        # Split the window into a baseline (first half) and recent (second half).
        _half = max(4, z_period // 2)
        _std_baseline = float(np.std(closes[-z_period:-_half], ddof=1)) if z_period > _half else std
        _std_recent   = float(np.std(closes[-_half:], ddof=1))
        if _std_baseline > 1e-12 and _std_recent > _std_baseline * 1.3:
            return None

        # Phase-2 MR-Q5: z-acceleration filter.  Even when |z| exceeds threshold,
        # if |z| is still expanding bar-over-bar the move may keep running into
        # a breakout rather than reverting.  We compute |z| one bar back and
        # skip the signal when the most recent bar widened the deviation.
        # The guard only fires on clear expansion (>10% growth) so normal noise
        # around the threshold does not over-filter.
        if len(closes) >= z_period + 1:
            _prev_window = closes[-z_period - 1:-1]
            _prev_mean = float(np.mean(_prev_window))
            _prev_std  = float(np.std(_prev_window, ddof=1))
            if _prev_std > 1e-12:
                _prev_z = abs((closes[-2] - _prev_mean) / _prev_std)
                if abs(z_score) > _prev_z * 1.10 and abs(z_score) > z_threshold * 1.05:
                    return None

        # ── ADX filter — ensure not trending ─────────────────────────────
        adx = self.calculate_adx(highs, lows, closes, period=14)
        if adx >= max_adx:
            return None

        # ── Volume confirmation ────────────────────────────────────────────
        # MR-2: Use a fixed 20-bar window for average volume so that a structurally
        # higher-volume regime today does not suppress vol_ratio by including stale
        # high-volume bars from z_period ago.
        avg_vol = float(np.mean(volumes[-20:]))
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
        confidence += _btc_trend_penalty
        # MR-1: Directional asymmetry — alts drop harder than they bounce in BTC
        # downtrends, so LONGs are more likely to get swept.  Apply an extra −6
        # when BTC's own trend is bearish (fast MA below slow MA).
        _btc_dir_penalty = 0
        if _btc_bearish and direction == "LONG":
            _btc_dir_penalty = -6
        confidence += _btc_dir_penalty

        # ── Entry zone ────────────────────────────────────────────────────
        buf = atr * rp.entry_zone_tight

        # Phase-2 MR-Q6: dynamic TP2 reversion fraction scaled by |z|.
        # Default 0.70 for z around threshold; up to 0.85 when z is very
        # extreme (≥ 3.5σ — capitulation/euphoria where full mean-return
        # is more likely).  Capped so small threshold crosses stay tactical.
        _abs_z = abs(z_score)
        if _abs_z >= 3.5:
            tp2_frac = 0.85
        elif _abs_z >= 2.5:
            tp2_frac = 0.78
        else:
            tp2_frac = 0.70

        if direction == "LONG":
            entry_low  = current_price - buf
            entry_high = current_price + buf
            # MR-4: SL at 1.5×ATR below/above the bar extreme.
            # Using only 0.8×ATR was too tight — on the bar where reversion starts,
            # the current bar's extreme is the likely future swing level and gets
            # retested on the first pullback; 1.5×ATR gives the trade room to breathe.
            stop_loss  = l - atr * 1.5
            # TP1: 25% reversion toward mean
            tp1 = current_price + (mean - current_price) * 0.25
            # MR-Q4 (Phase-2: scaled by |z|): TP2 at 70–85% of the reversion.
            tp2 = current_price + (mean - current_price) * tp2_frac
            # TP3: opposite z-score level
            tp3 = mean + std * z_threshold
        else:
            entry_low  = current_price - buf
            entry_high = current_price + buf
            stop_loss  = h + atr * 1.5
            tp1 = current_price - (current_price - mean) * 0.25
            # MR-Q4 (Phase-2: scaled by |z|): 70-85% reversion.
            tp2 = current_price - (current_price - mean) * tp2_frac
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
        # X3: use calculate_effective_rr for consistent worst-case fill
        rr_ratio = self.calculate_effective_rr(direction, entry_low, entry_high, stop_loss, tp2)

        confluence: List[str] = [
            f"✅ Z-score: {z_score:.2f} (threshold: ±{z_threshold})",
            f"✅ Rolling mean ({z_period}): {fmt_price(mean)} | Std: {fmt_price(std)}",
            f"✅ ADX: {adx:.1f} < {max_adx} — range confirmed",
            f"✅ Volume: {vol_ratio:.1f}x average",
        ]
        if is_rejection:
            confluence.append("✅ Rejection candle at extreme")
        if _btc_trend_penalty != 0:
            confluence.append(f"⚠️ BTC trending (ADX {_btc_adx:.0f}) — macro risk ({_btc_trend_penalty})")
        if _btc_dir_penalty != 0:
            confluence.append(f"⚠️ BTC bearish — LONG alt correlation risk ({_btc_dir_penalty})")
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
