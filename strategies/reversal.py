"""
TitanBot Pro — Reversal Strategy
==================================
Detects extreme reversals using:
  - RSI oversold/overbought at 2.5-sigma Bollinger extremes
  - RSI divergence confirmation (price makes new extreme, RSI doesn't)
  - Rejection candle (wick-heavy candle at extreme)
  - Volume confirmation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from config.loader import cfg
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp, compute_vol_percentile

logger = logging.getLogger(__name__)


class ReversalStrategy(BaseStrategy):

    name = "ExtremeReversal"
    description = "RSI extreme + Bollinger Band reversal with divergence"

    # Reversals work at EXTREMES — which only happen after clear moves.
    # Valid in CHOPPY (fading range boundaries) and VOLATILE (panic reversals).
    # NEVER in a clean trend where "overbought" stays overbought.
    VALID_REGIMES = {"CHOPPY", "VOLATILE"}

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.reversal

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime gate ───────────────────────────────────────────────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
            if regime not in self.VALID_REGIMES:
                return None  # RSI stays extreme in trends — don't fade
            # Volatile regime: increase thresholds (extremes are more extreme)
            _vol_adj = 5 if regime == "VOLATILE" else 0
        except Exception:
            _vol_adj = 0

        tf = getattr(self._cfg, 'timeframe', '15m')
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 50:
            return None

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

        closes  = df['close'].values
        highs   = df['high'].values
        lows    = df['low'].values
        opens   = df['open'].values
        volumes = df['volume'].values

        rsi_period   = getattr(self._cfg, 'rsi_period', 14)
        rsi_os       = getattr(self._cfg, 'rsi_oversold', 28) - _vol_adj
        rsi_ob       = getattr(self._cfg, 'rsi_overbought', 72) + _vol_adj
        bb_period    = getattr(self._cfg, 'bb_period', 20)
        bb_std       = getattr(self._cfg, 'bb_std', 2.5)
        req_div      = getattr(self._cfg, 'require_divergence', True)
        req_rej      = getattr(self._cfg, 'require_rejection_candle', True)
        conf_base    = getattr(self._cfg, 'confidence_base', 72)

        rsi    = self.calculate_rsi(closes, rsi_period)
        bb_mid, bb_up, bb_lo = self.calculate_bollinger(closes, bb_period, bb_std)
        atr    = self.calculate_atr(highs, lows, closes, 14)
        if atr == 0:
            return None

        current = closes[-1]
        avg_vol = np.mean(volumes[-21:-1])  # exclude live candle from average
        cur_vol = volumes[-2]               # last CLOSED candle ([-1] is live/unconfirmed)

        direction  = None
        confluence = []
        confidence = conf_base

        # ── Bullish reversal: RSI oversold + price at lower BB ──
        if rsi < rsi_os and current <= bb_lo * 1.005:
            direction = "LONG"
            confluence.append(f"✅ RSI oversold: {rsi:.1f} (< {rsi_os})")
            confluence.append(f"✅ Price at lower BB (2.5σ): {fmt_price(bb_lo)}")
            confidence += (rsi_os - rsi) * 0.5  # more oversold = more confident

        # ── Bearish reversal: RSI overbought + price at upper BB ──
        elif rsi > rsi_ob and current >= bb_up * 0.995:
            direction = "SHORT"
            confluence.append(f"✅ RSI overbought: {rsi:.1f} (> {rsi_ob})")
            confluence.append(f"✅ Price at upper BB (2.5σ): {fmt_price(bb_up)}")
            confidence += (rsi - rsi_ob) * 0.5
        else:
            return None

        # ── RSI divergence check ──────────────────────────────
        if req_div:
            divergence = self._check_divergence(closes, highs, lows, direction, lookback=20)
            if divergence:
                confluence.append(f"✅ RSI divergence confirmed")
                confidence += 10
            else:
                confidence -= 5  # No divergence = weaker setup

        # ── Rejection candle check ────────────────────────────
        if req_rej:
            rejection = self._check_rejection_candle(opens, highs, lows, closes, direction)
            if rejection:
                confluence.append("✅ Rejection candle (wick at extreme)")
                confidence += 8

        # ── Volume confirmation ───────────────────────────────
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio >= 1.5:
            confluence.append(f"✅ Volume: {vol_ratio:.1f}x average")
            confidence += 5

        # ── Targets ───────────────────────────────────────────
        # Wire timeframe+volatility-scaled TPs for realistic targets per setup class
        vp = compute_vol_percentile(highs, lows, closes)
        tp1_m = rp.volatility_scaled_tp1(tf, vp)
        tp2_m = rp.volatility_scaled_tp2(tf, vp)
        tp3_m = rp.volatility_scaled_tp3(tf, vp)

        if direction == "LONG":
            entry_low  = current - atr * rp.entry_zone_tight
            entry_high = current + atr * rp.entry_zone_atr
            stop_loss  = lows[-3:].min() - atr * rp.sl_atr_mult * 0.25
            tp1 = current + atr * tp1_m
            tp2 = bb_mid  # Mean reversion target
            tp3 = current + atr * tp3_m
            # FIX #15: Ensure TP ordering for LONG
            tp1 = max(tp1, entry_high + atr * 0.3)
            tp2 = max(tp2, tp1 + atr * 0.3)
            tp3 = max(tp3, tp2 + atr * 0.3)
        else:
            entry_high = current + atr * rp.entry_zone_tight
            entry_low  = current - atr * rp.entry_zone_atr
            stop_loss  = highs[-3:].max() + atr * rp.sl_atr_mult * 0.25
            tp1 = current - atr * tp1_m
            tp2 = bb_mid
            tp3 = current - atr * tp3_m
            # FIX #15: Ensure TP ordering for SHORT
            tp1 = min(tp1, entry_low - atr * 0.3)
            tp2 = min(tp2, tp1 - atr * 0.3)
            tp3 = min(tp3, tp2 - atr * 0.3)

        risk = abs(current - stop_loss)
        rr   = abs(tp2 - current) / risk if risk > 0 else 0
        if rr < cfg_min_rr("intraday"):
            return None

        confidence = min(95, max(40, confidence))

        # FIX: validate geometry and minimum RR before returning.
        # validate_signal() checks confidence, rr_ratio, entry zone, SL/TP sides.
        _candidate = SignalResult(
            symbol=symbol,
            direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
            strategy=self.name,
            confidence=confidence,
            entry_low=entry_low, entry_high=entry_high,
            stop_loss=stop_loss,
            tp1=tp1, tp2=tp2, tp3=tp3,
            rr_ratio=rr, atr=atr, setup_class="intraday", analysis_timeframes=[tf],
            confluence=confluence,
            raw_data={'rsi': rsi, 'bb_upper': bb_up, 'bb_lower': bb_lo, 'bb_mid': bb_mid}
        )
        if not self.validate_signal(_candidate):
            return None
        return _candidate

    def _check_divergence(self, closes, highs, lows, direction, lookback=20) -> bool:
        """
        Detect RSI divergence over last N bars.

        Requires two genuine swing extremes within the lookback window:
        - A swing low/high is a local extreme where both adjacent bars are higher/lower.
        - The MOST RECENT swing extreme must show price making a new extreme while
          RSI fails to confirm (diverges).
        - Minimum RSI separation of 5 points required to filter noise-level drift.

        NC3-FIX: Previous implementation used argmin/argmax which finds the global
        minimum/maximum of the window, not a true swing. In a monotonically rising
        window, argmin returns bar 0 (the oldest bar), producing degenerate
        comparisons. Now searches from the right (most recent bar) backward,
        identifying local swing points where the bars immediately before AND after
        are both on the opposite side — the standard definition of a swing point
        in technical analysis.

        S4-FIX: Minimum RSI separation threshold raised to 5.0 points.
        Institutional divergence traders require 5-8 RSI points of separation;
        the previous suggestion of 3.0 is noise-level on a 0-100 scale.
        """
        MIN_RSI_DELTA = 5.0  # S4: minimum RSI separation to qualify as divergence

        try:
            closes = np.array(closes, dtype=float)
            highs  = np.array(highs,  dtype=float)
            lows   = np.array(lows,   dtype=float)

            if len(closes) < lookback + 14:
                return False

            window_highs = highs[-lookback:]
            window_lows  = lows[-lookback:]

            if direction == "LONG":
                # Bullish divergence: price makes lower low, RSI makes higher low
                # Find the two most recent swing lows (local minima) in the window
                swing_lows = self._find_swing_points_in_window(window_lows, is_high=False)
                if len(swing_lows) < 2:
                    return False

                # Most recent swing low and the one before it
                recent_idx, recent_price = swing_lows[-1]
                prior_idx, prior_price   = swing_lows[-2]

                # Price must make a lower low
                if recent_price >= prior_price:
                    return False

                # RSI at both swing lows
                abs_recent = len(closes) - lookback + recent_idx
                abs_prior  = len(closes) - lookback + prior_idx
                rsi_recent = self.calculate_rsi(closes[:abs_recent + 1], 14)
                rsi_prior  = self.calculate_rsi(closes[:abs_prior + 1],  14)

                # RSI must be higher at lower price, by at least MIN_RSI_DELTA
                return (rsi_recent - rsi_prior) >= MIN_RSI_DELTA

            else:
                # Bearish divergence: price makes higher high, RSI makes lower high
                swing_highs = self._find_swing_points_in_window(window_highs, is_high=True)
                if len(swing_highs) < 2:
                    return False

                recent_idx, recent_price = swing_highs[-1]
                prior_idx, prior_price   = swing_highs[-2]

                if recent_price <= prior_price:
                    return False

                abs_recent = len(closes) - lookback + recent_idx
                abs_prior  = len(closes) - lookback + prior_idx
                rsi_recent = self.calculate_rsi(closes[:abs_recent + 1], 14)
                rsi_prior  = self.calculate_rsi(closes[:abs_prior + 1],  14)

                # RSI must be lower at higher price, by at least MIN_RSI_DELTA
                return (rsi_prior - rsi_recent) >= MIN_RSI_DELTA

        except Exception:
            return False

    @staticmethod
    def _find_swing_points_in_window(values, is_high: bool):
        """
        Find local swing points in a 1D array.
        A swing high: values[i] > values[i-1] AND values[i] > values[i+1]
        A swing low:  values[i] < values[i-1] AND values[i] < values[i+1]
        Returns list of (index, value) tuples, ordered by index (oldest first).
        """
        points = []
        for i in range(1, len(values) - 1):
            if is_high:
                if values[i] > values[i - 1] and values[i] > values[i + 1]:
                    points.append((i, float(values[i])))
            else:
                if values[i] < values[i - 1] and values[i] < values[i + 1]:
                    points.append((i, float(values[i])))
        return points

    def _check_rejection_candle(self, opens, highs, lows, closes, direction) -> bool:
        """Check if current candle is a rejection (pin bar)"""
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        total_range = h - l
        if total_range == 0:
            return False
        body = abs(c - o)
        if direction == "LONG":
            lower_wick = min(o, c) - l
            return lower_wick > body * 1.5 and lower_wick / total_range > 0.5
        else:
            upper_wick = h - max(o, c)
            return upper_wick > body * 1.5 and upper_wick / total_range > 0.5
