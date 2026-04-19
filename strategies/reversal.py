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
from config.constants import STRATEGY_VALID_REGIMES
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp, compute_vol_percentile

logger = logging.getLogger(__name__)


class ReversalStrategy(BaseStrategy):

    name = "ExtremeReversal"
    description = "RSI extreme + Bollinger Band reversal with divergence"

    # Reversals work at EXTREMES — which happen after clear moves in tight conditions.
    # Valid in CHOPPY (fading range boundaries).
    # VOLATILE is excluded by default: extreme directional moves can extend 3-5×
    # before reverting, making early fades high-risk. Set allow_volatile: true in
    # config to opt back in (e.g., if scanning for panic capitulation setups).
    VALID_REGIMES = STRATEGY_VALID_REGIMES["ExtremeReversal"]

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.reversal

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            self._record_analyze_error(self.name, e, symbol)
            return None

    async def _analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime gate ───────────────────────────────────────────────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
            allow_volatile = bool(getattr(self._cfg, 'allow_volatile', False))
            _valid = set(self.VALID_REGIMES)
            if allow_volatile:
                _valid = _valid | {"VOLATILE"}
            if regime not in _valid:
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
        if atr == 0 or not np.isfinite(bb_mid) or not np.isfinite(bb_up) or not np.isfinite(bb_lo):
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
        div_strength = 0.0
        if req_div:
            div_strength = self._check_divergence(closes, highs, lows, direction, lookback=20)
            if div_strength > 0:
                # Scale bonus: +10 at minimum qualifying divergence, up to +18 for strong
                _div_bonus = int(min(18, 10 + (div_strength - 1.0) * 8))
                confluence.append(f"✅ RSI divergence confirmed (strength: {div_strength:.1f}x, +{_div_bonus})")
                confidence += _div_bonus
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
        vol_conf_mult = float(getattr(self._cfg, 'vol_confirmation_mult', 2.0))
        if vol_ratio >= vol_conf_mult:
            confluence.append(f"✅ Volume: {vol_ratio:.1f}x average")
            confidence += 5
        # Climactic volume: genuine capitulation/euphoria spikes are 3–5× normal.
        if vol_ratio > 3.0:
            confluence.append(f"🔥 Climactic volume: {vol_ratio:.1f}x — capitulation signal (+4)")
            confidence += 4

        # ── Late-divergence guard ─────────────────────────────
        # If price has already bounced more than 1.5 ATR from the BB extreme,
        # the reversal is underway and entering now is chasing.
        _signal_extreme = bb_lo if direction == "LONG" else bb_up
        if abs(current - _signal_extreme) > 1.5 * atr:
            confluence.append(
                f"⚠️ Late entry: {abs(current - _signal_extreme) / atr:.1f}× ATR from BB extreme (-4)"
            )
            confidence -= 4

        # ── Divergence + volume coupling ─────────────────────
        # Strong divergence without participation is a weaker signal: smart-money
        # divergence must be accompanied by at least baseline volume to be credible.
        vol_conf_mult_for_div = float(getattr(self._cfg, 'vol_confirmation_mult', 2.0))
        if div_strength > 1.0 and vol_ratio < vol_conf_mult_for_div:
            confluence.append("⚠️ Strong divergence but low volume — signal weaker (-3)")
            confidence -= 3

        # ── Targets ───────────────────────────────────────────
        # Wire timeframe+volatility-scaled TPs for realistic targets per setup class
        vp = compute_vol_percentile(highs, lows, closes)
        tp1_m = rp.volatility_scaled_tp1(tf, vp)
        tp2_m = rp.volatility_scaled_tp2(tf, vp)
        tp3_m = rp.volatility_scaled_tp3(tf, vp)

        # ── SL lookback: wider window for volatile extremes ──────
        sl_lookback = max(3, int(getattr(self._cfg, 'sl_bar_lookback', 5)))
        if direction == "LONG":
            entry_low  = current - atr * rp.entry_zone_tight
            entry_high = current + atr * rp.entry_zone_atr
            stop_loss  = lows[-sl_lookback:].min() - atr * rp.sl_atr_mult
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
            stop_loss  = highs[-sl_lookback:].max() + atr * rp.sl_atr_mult
            tp1 = current - atr * tp1_m
            tp2 = bb_mid
            tp3 = current - atr * tp3_m
            # FIX #15: Ensure TP ordering for SHORT
            tp1 = min(tp1, entry_low - atr * 0.3)
            tp2 = min(tp2, tp1 - atr * 0.3)
            tp3 = min(tp3, tp2 - atr * 0.3)

        risk = abs(current - stop_loss)
        if risk <= 0:
            return None
        # X3: use calculate_effective_rr for consistent worst-case fill
        rr   = self.calculate_effective_rr(direction, entry_low, entry_high, stop_loss, tp2)
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

    def _check_divergence(self, closes, highs, lows, direction, lookback=20) -> float:
        """
        Detect RSI divergence over last N bars and return a strength score.

        Returns:
            0.0  — no qualifying divergence
            ≥1.0 — divergence detected; score = rsi_delta / MIN_RSI_DELTA so that
                   1.0 = minimum qualifying (delta=5), 2.0 = double minimum (delta=10).
                   Higher scores indicate stronger divergences that warrant more
                   confidence. Callers should cap total bonus (e.g. min(18, …)).

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
                return 0.0

            window_highs = highs[-lookback:]
            window_lows  = lows[-lookback:]

            if direction == "LONG":
                # Bullish divergence: price makes lower low, RSI makes higher low
                # Find the two most recent swing lows (local minima) in the window
                swing_lows = self._find_swing_points_in_window(window_lows, is_high=False)
                if len(swing_lows) < 2:
                    return 0.0

                # Most recent swing low and the one before it
                recent_idx, recent_price = swing_lows[-1]
                prior_idx, prior_price   = swing_lows[-2]

                # Price must make a lower low
                if recent_price >= prior_price:
                    return 0.0

                # RSI at both swing lows
                abs_recent = len(closes) - lookback + recent_idx
                abs_prior  = len(closes) - lookback + prior_idx
                rsi_series = self._calculate_rsi_series(closes, 14)
                if abs_recent >= len(rsi_series) or abs_prior >= len(rsi_series):
                    return 0.0
                rsi_recent = rsi_series[abs_recent]
                rsi_prior  = rsi_series[abs_prior]
                if not np.isfinite(rsi_recent) or not np.isfinite(rsi_prior):
                    return 0.0

                # RSI must be higher at lower price, by at least MIN_RSI_DELTA
                rsi_delta = rsi_recent - rsi_prior
                if rsi_delta < MIN_RSI_DELTA:
                    return 0.0
                return rsi_delta / MIN_RSI_DELTA

            else:
                # Bearish divergence: price makes higher high, RSI makes lower high
                swing_highs = self._find_swing_points_in_window(window_highs, is_high=True)
                if len(swing_highs) < 2:
                    return 0.0

                recent_idx, recent_price = swing_highs[-1]
                prior_idx, prior_price   = swing_highs[-2]

                if recent_price <= prior_price:
                    return 0.0

                abs_recent = len(closes) - lookback + recent_idx
                abs_prior  = len(closes) - lookback + prior_idx
                rsi_series = self._calculate_rsi_series(closes, 14)
                if abs_recent >= len(rsi_series) or abs_prior >= len(rsi_series):
                    return 0.0
                rsi_recent = rsi_series[abs_recent]
                rsi_prior  = rsi_series[abs_prior]
                if not np.isfinite(rsi_recent) or not np.isfinite(rsi_prior):
                    return 0.0

                # RSI must be lower at higher price, by at least MIN_RSI_DELTA
                rsi_delta = rsi_prior - rsi_recent
                if rsi_delta < MIN_RSI_DELTA:
                    return 0.0
                return rsi_delta / MIN_RSI_DELTA

        except Exception:
            return 0.0

    @staticmethod
    def _calculate_rsi_series(closes, period: int = 14):
        def _to_rsi(avg_gain: float, avg_loss: float) -> float:
            if avg_loss == 0:
                return 100.0
            return 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))

        closes = np.asarray(closes, dtype=float)
        rsi = np.full(len(closes), np.nan, dtype=float)
        if len(closes) < period + 1:
            return rsi
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        rsi[period] = _to_rsi(avg_gain, avg_loss)
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            idx = i + 1
            rsi[idx] = _to_rsi(avg_gain, avg_loss)
        return rsi

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
