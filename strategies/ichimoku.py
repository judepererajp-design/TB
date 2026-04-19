"""
TitanBot Pro — Ichimoku Cloud Strategy
========================================
Detects high-probability Ichimoku setups requiring all three conditions:
  1. TK Cross — Tenkan (9) crosses Kijun (26)
  2. Price above/below the Kumo (cloud)
  3. Chikou clear — lagging span confirms direction

Entry around the TK cross level, SL at Kijun ± ATR buffer.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.loader import cfg
from config.constants import STRATEGY_VALID_REGIMES
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp, compute_vol_percentile

logger = logging.getLogger(__name__)


def _period_midline(highs: np.ndarray, lows: np.ndarray, period: int) -> np.ndarray:
    """(highest high + lowest low) / 2 over rolling period."""
    result = np.zeros(len(highs))
    for i in range(len(highs)):
        if i < period - 1:
            result[i] = (highs[: i + 1].max() + lows[: i + 1].min()) / 2
        else:
            result[i] = (highs[i - period + 1: i + 1].max() + lows[i - period + 1: i + 1].min()) / 2
    return result


class Ichimoku(BaseStrategy):

    name = "Ichimoku"
    description = "Full Ichimoku cloud strategy: TK cross + Kumo + Chikou"

    VALID_REGIMES = STRATEGY_VALID_REGIMES["Ichimoku"]

    # Direction-aware regime confidence: Ichimoku is a trend-following indicator.
    # Counter-trend signals (LONG in BEAR, SHORT in BULL) get penalized.
    _REGIME_CONF_WITH_TREND = {
        "BULL_TREND":  +6,
        "BEAR_TREND":  +6,
        "VOLATILE":    0,
    }
    _REGIME_CONF_COUNTER_TREND = {
        "BULL_TREND":  -10,   # SHORT in bull
        "BEAR_TREND":  -10,   # LONG in bear
        "VOLATILE":    0,
    }

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.ichimoku

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
        except Exception:
            regime = "UNKNOWN"

        if regime not in self.VALID_REGIMES:
            return None

        tf = getattr(self._cfg, "timeframe", "4h")
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 100:
            return None

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values

        atr = self.calculate_atr(highs, lows, closes, period=14)
        if atr == 0:
            return None

        tenkan_period  = getattr(self._cfg, "tenkan", 9)
        kijun_period   = getattr(self._cfg, "kijun", 26)
        senkou_b_period = getattr(self._cfg, "senkou_b", 52)
        displacement   = getattr(self._cfg, "displacement", 26)
        confidence_base = getattr(self._cfg, "confidence_base", 73)

        # ── Calculate Ichimoku components ──────────────────────────────────
        tenkan  = _period_midline(highs, lows, tenkan_period)
        kijun   = _period_midline(highs, lows, kijun_period)

        # Senkou Span A = (Tenkan + Kijun) / 2, shifted forward 26
        senkou_a_raw = (tenkan + kijun) / 2
        # We access the value from `displacement` bars ago as the current cloud value
        # The ≥100-bar guard above means len(senkou_a_raw) > displacement (26) always holds.
        senkou_a_current = senkou_a_raw[-displacement]

        # Senkou Span B = (52-period H+L) / 2, shifted forward 26
        senkou_b_raw = _period_midline(highs, lows, senkou_b_period)
        senkou_b_current = senkou_b_raw[-displacement]

        kumo_top    = max(senkou_a_current, senkou_b_current)
        kumo_bottom = min(senkou_a_current, senkou_b_current)
        kumo_size   = kumo_top - kumo_bottom
        current_price = closes[-1]

        # IC-Q1: Thin cloud → weak S/R zone → high failure rate.
        # Use the larger of 1×ATR or 0.2% of price so the gate scales correctly across
        # assets where ATR is very small relative to price (e.g. low-vol alts) or very
        # large relative to price (e.g. high-vol micro-caps).
        if kumo_size < max(atr * 1.0, current_price * 0.002):
            return None

        # Chikou span = close shifted back `displacement` bars
        chikou_current_price = closes[-1]
        chikou_compare_price = closes[-displacement - 1] if len(closes) > displacement + 1 else closes[0]

        current_tenkan = tenkan[-1]
        current_kijun  = kijun[-1]
        prev_tenkan    = tenkan[-2]
        prev_kijun     = kijun[-2]

        # ── Condition 1: TK Cross ─────────────────────────────────────────
        bullish_tk_cross = (prev_tenkan <= prev_kijun) and (current_tenkan > current_kijun)
        bearish_tk_cross = (prev_tenkan >= prev_kijun) and (current_tenkan < current_kijun)

        require_tk_cross = getattr(self._cfg, "require_tk_cross", True)
        if require_tk_cross and not bullish_tk_cross and not bearish_tk_cross:
            return None

        if bullish_tk_cross:
            direction = "LONG"
        elif bearish_tk_cross:
            direction = "SHORT"
        else:
            direction = "LONG" if current_tenkan > current_kijun else "SHORT"

        # ── Condition 2: Price vs Kumo ────────────────────────────────────
        require_cloud = getattr(self._cfg, "require_above_cloud", True)
        _bars_above_cloud = 0
        _late_strong_entry = False  # set below when bars > 5 but trend is still intact
        if require_cloud:
            if direction == "LONG" and current_price <= kumo_top:
                return None
            if direction == "SHORT" and current_price >= kumo_bottom:
                return None
            # IC-Q2: Breakout recency — price that has been above/below the cloud
            # for many bars signals trend continuation, not a fresh breakout entry.
            # Count consecutive bars price has cleared the cloud.
            _bars_above_cloud = 0
            for _i in range(1, min(len(closes), 7)):
                if direction == "LONG" and closes[-_i] > kumo_top:
                    _bars_above_cloud += 1
                elif direction == "SHORT" and closes[-_i] < kumo_bottom:
                    _bars_above_cloud += 1
                else:
                    break
            if _bars_above_cloud > 5:
                # Distinguish two sub-cases:
                #   • Cloud-hugging (< 0.5×ATR away): price is pulling back toward the cloud
                #     edge — momentum is fading, breakout is at risk of reverting.  Hard-reject.
                #   • Strong-but-late (≥ 0.5×ATR away): trend is clearly intact but entry is
                #     extended.  Allow through; the confidence path applies a −6 penalty below.
                _close_distance_cloud = abs(current_price - (kumo_top if direction == "LONG" else kumo_bottom))
                if _close_distance_cloud < 0.5 * atr:
                    return None  # Cloud-hugging: breakout losing momentum, likely to fade back
                _late_strong_entry = True  # Extended — penalised in confidence, not hard-rejected

        # ── Condition 3: Chikou clear ──────────────────────────────────────
        require_chikou = getattr(self._cfg, "require_chikou_clear", True)
        chikou_long_clear  = chikou_current_price > chikou_compare_price
        chikou_short_clear = chikou_current_price < chikou_compare_price

        if require_chikou:
            if direction == "LONG" and not chikou_long_clear:
                return None
            if direction == "SHORT" and not chikou_short_clear:
                return None

        # IC-Q4: TK cross recency — signals are strongest when the cross is fresh.
        # Stale crosses produce "Frankenstein setups" where TK, cloud, and chikou
        # conditions are temporally disconnected.  Scan back up to 15 bars for the
        # most recent directional TK cross; reject if it is more than 3 bars old.
        _bars_since_tk_cross = 20  # default: no recent cross found in scan window
        for _i in range(min(len(tenkan) - 1, 15)):
            _idx_after  = -1 - _i
            _idx_before = -2 - _i
            if direction == "LONG":
                if tenkan[_idx_before] <= kijun[_idx_before] and tenkan[_idx_after] > kijun[_idx_after]:
                    _bars_since_tk_cross = _i
                    break
            else:
                if tenkan[_idx_before] >= kijun[_idx_before] and tenkan[_idx_after] < kijun[_idx_after]:
                    _bars_since_tk_cross = _i
                    break
        if _bars_since_tk_cross > 3:
            return None

        # TK cross quality: compute Tenkan 3-bar slope for use in confidence scoring.
        # A slope below 5% of ATR indicates a nearly-flat cross (noise tick, not momentum).
        _tenkan_slope = tenkan[-1] - tenkan[-3] if len(tenkan) >= 3 else 0.0

        # Direction-aware regime alignment (used for kijun bonus and regime confidence)
        _is_with_trend = (
            (direction == "LONG" and regime == "BULL_TREND") or
            (direction == "SHORT" and regime == "BEAR_TREND")
        )

        # ── Condition 4: Kijun pullback detection ────────────────────────
        # After TK cross, best entry is on a pullback to flat Kijun.
        # Kijun is "flat" when it hasn't moved for 3+ bars (acting as S/R).
        _kijun_flat = False
        _kijun_pullback = False
        _kijun_pullback_bonus = 0
        if len(kijun) >= 5:
            _kijun_range = abs(kijun[-1] - kijun[-4])
            _kijun_flat = _kijun_range < atr * 0.15  # Kijun barely moved in 4 bars
            if _kijun_flat:
                # Check if price is near Kijun (pullback in progress)
                _dist_to_kijun = abs(current_price - current_kijun) / atr
                if _dist_to_kijun < 0.8:
                    _kijun_pullback = True
                    # IC-Q3: Only give full bonus in trending regime — flat Kijun
                    # fires most often in sideways/low-vol markets where Ichimoku
                    # is least reliable.  Halve the reward outside trend context.
                    _kijun_pullback_bonus = 6 if _is_with_trend else 2

        # ── Confidence ────────────────────────────────────────────────────
        # Direction-aware regime bonus
        if _is_with_trend or regime not in ("BULL_TREND", "BEAR_TREND"):
            _regime_bonus = self._REGIME_CONF_WITH_TREND.get(regime, 0)
        else:
            _regime_bonus = self._REGIME_CONF_COUNTER_TREND.get(regime, 0)

        confidence = float(confidence_base) + _regime_bonus + _kijun_pullback_bonus

        # TK cross in same direction as cloud
        if direction == "LONG" and current_price > kumo_top:
            confidence += 8
        elif direction == "SHORT" and current_price < kumo_bottom:
            confidence += 8

        # Price clearance from cloud — three-zone model:
        #   <1×ATR : borderline break (no adjustment — too close to cloud)
        #   1–2×ATR: optimal clearance (+5 — confirmed break, not overextended)
        #   >2×ATR : overextended entry (-5 — trend is real but entry is late/stretched)
        cloud_distance = abs(current_price - (kumo_top if direction == "LONG" else kumo_bottom))
        if atr <= cloud_distance < 2 * atr:
            confidence += 5  # Optimal clearance
        elif cloud_distance >= 2 * atr:
            confidence -= 5  # Overextended

        # Chikou strength
        chikou_margin = abs(chikou_current_price - chikou_compare_price)
        if chikou_margin > atr:
            confidence += 5

        # TK cross slope quality: a near-flat cross is a noise tick, not a momentum signal.
        # Threshold: 5% of ATR over 3 bars — anything below is considered structurally flat.
        if abs(_tenkan_slope) < atr * 0.05:
            confidence -= 3

        # Late-but-strong-trend penalty: price has been above/below the cloud for 6+ bars
        # but trend is intact (passed the cloud-hugging hard gate above).  Entry is extended.
        if _late_strong_entry:
            confidence -= 6

        # ── Entry around TK cross level (or Kijun on pullback) ──────────
        # When flat-Kijun pullback is detected, enter near Kijun level
        # for a tighter stop and better R:R.
        if _kijun_pullback:
            tk_cross_level = current_kijun  # Use Kijun as entry anchor
        else:
            tk_cross_level = (current_tenkan + current_kijun) / 2

        vp = compute_vol_percentile(highs, lows, closes)
        if direction == "LONG":
            entry_low   = tk_cross_level - atr * 0.3
            entry_high  = tk_cross_level + atr * 0.3
            # SL: Kijun level - ATR buffer
            stop_loss   = current_kijun - atr * rp.sl_atr_mult
            # TP1: structural target (cloud top) when cloud is above entry; otherwise ATR-scaled.
            if kumo_top > entry_high:
                tp1 = kumo_top + atr * 0.5 * rp.atr_scale(tf)
            else:
                tp1 = entry_high + atr * rp.volatility_scaled_tp1(tf, vp)
            # Wire timeframe+volatility-scaled TPs
            tp2         = entry_high + atr * rp.volatility_scaled_tp2(tf, vp)
            tp3         = entry_high + atr * rp.volatility_scaled_tp3(tf, vp)
        else:
            entry_high  = tk_cross_level + atr * 0.3
            entry_low   = tk_cross_level - atr * 0.3
            stop_loss   = current_kijun + atr * rp.sl_atr_mult
            # TP1: structural target (cloud bottom) when cloud is below entry; otherwise ATR-scaled.
            if kumo_bottom < entry_low:
                tp1 = kumo_bottom - atr * 0.5 * rp.atr_scale(tf)
            else:
                tp1 = entry_low - atr * rp.volatility_scaled_tp1(tf, vp)
            tp2         = entry_low - atr * rp.volatility_scaled_tp2(tf, vp)
            tp3         = entry_low - atr * rp.volatility_scaled_tp3(tf, vp)

        risk = (entry_low - stop_loss) if direction == "LONG" else (stop_loss - entry_high)
        if risk <= 0:
            return None
        rr_ratio = abs(tp2 - tk_cross_level) / risk

        confluence: List[str] = [
            f"✅ {'Bullish' if direction == 'LONG' else 'Bearish'} TK cross",
            f"   Tenkan: {fmt_price(current_tenkan)} | Kijun: {fmt_price(current_kijun)}",
            f"✅ Price {'above' if direction == 'LONG' else 'below'} Kumo: {fmt_price(kumo_bottom)}-{fmt_price(kumo_top)}",
            f"   Cloud thickness: {fmt_price(kumo_size)} ({kumo_size / atr:.1f}x ATR)",
        ]
        if (direction == "LONG" and chikou_long_clear) or (direction == "SHORT" and chikou_short_clear):
            confluence.append(f"✅ Chikou clear — lagging span confirms direction")
        if _kijun_pullback:
            confluence.append(f"✅ Flat Kijun pullback — high-probability entry at {fmt_price(current_kijun)}")
        elif _kijun_flat:
            confluence.append(f"📊 Kijun flat — awaiting pullback for optimal entry")
        confluence.append(f"📊 Regime: {regime} | TF: {tf}")
        confluence.append(f"🎯 R:R {rr_ratio:.2f} | ATR: {fmt_price(atr)}")

        confidence = min(94, max(40, confidence))

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
            setup_class="swing",
            timeframe=tf,
            analysis_timeframes=[tf],
            confluence=confluence,
            raw_data={
                "tenkan": float(current_tenkan),
                "kijun": float(current_kijun),
                "kumo_top": float(kumo_top),
                "kumo_bottom": float(kumo_bottom),
                "kumo_size": float(kumo_size),
                "senkou_a": float(senkou_a_current),
                "senkou_b": float(senkou_b_current),
                "chikou_compare_price": float(chikou_compare_price),
                "atr": atr,
                "regime": regime,
                "bars_since_tk_cross": _bars_since_tk_cross,
                "bars_above_cloud": _bars_above_cloud if require_cloud else None,
            },
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
IchimokuStrategy = Ichimoku
