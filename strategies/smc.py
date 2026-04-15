"""
TitanBot Pro — Smart Money Concepts Strategy
=============================================
Detects institutional price action setups using SMC methodology:
  - Liquidity Sweep + Reversal (stop hunt detection)
  - Fair Value Gap (FVG / imbalance zones)
  - Order Block (last candle before impulse)
  - Break of Structure (BOS with momentum)

Killzone bonus applied during high-probability trading sessions.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.loader import cfg
from config.constants import Penalties
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp, compute_vol_percentile

logger = logging.getLogger(__name__)


class SmartMoneyConcepts(BaseStrategy):

    name = "SmartMoneyConcepts"
    description = "Liquidity sweeps, FVG, order blocks, and BOS detection"

    # SMC works in all regimes — VOLATILE_PANIC is allowed here because
    # regime_thresholds already gates it to only SMC/Wyckoff/FundingRateArb
    # at max 2 signals/hr and 90 min_confidence.  Previously VOLATILE_PANIC
    # was missing, so the strategy ran fully then got blocked downstream
    # instead of short-circuiting early via the VALID_REGIMES check.
    VALID_REGIMES = {"BULL_TREND", "BEAR_TREND", "VOLATILE", "VOLATILE_PANIC", "CHOPPY", "UNKNOWN"}

    # Direction-aware regime confidence: with-trend gets a boost, counter-trend
    # gets a penalty.  Previously flat +8 for both BULL/BEAR regardless of
    # direction — that inflated counter-trend LONGs in BEAR_TREND (production
    # evidence: SMC firing 2 LONGs, 0 SHORTs in BEAR_TREND, all killed).
    _REGIME_CONF_WITH_TREND = {
        "BULL_TREND":  +8,
        "BEAR_TREND":  +8,
        "VOLATILE":    +3,
        "VOLATILE_PANIC": +3,
        "CHOPPY":      -10,
        "UNKNOWN":     0,
    }
    _REGIME_CONF_COUNTER_TREND = {
        "BULL_TREND":  -12,   # SHORT in bull — high risk
        "BEAR_TREND":  -12,   # LONG in bear — high risk
        "VOLATILE":    +3,    # No trend to counter
        "VOLATILE_PANIC": +3,
        "CHOPPY":      -10,
        "UNKNOWN":     0,
    }
    _COUNTER_TREND_REVERSAL_SETUPS = {"LiquiditySweep"}

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.smc

    @classmethod
    def _allows_counter_trend_setup(
        cls,
        setup_type: str,
        has_choch: bool,
        choch_direction: str,
        direction: str,
    ) -> bool:
        """Allow only genuine reversal-style counter-trend SMC setups upstream."""
        if setup_type in cls._COUNTER_TREND_REVERSAL_SETUPS:
            return True
        expected_choch = "BULLISH" if direction == "LONG" else "BEARISH"
        return has_choch and choch_direction == expected_choch

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            logger.debug(f"SMC.analyze {symbol}: {e}")
            return None

    async def _analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime gate ───────────────────────────────────────────────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception:
            regime = "UNKNOWN"

        regime_bonus = 0  # deferred — computed after direction is known

        # Primary timeframe for structure analysis
        tf = "4h"
        for candidate_tf in ("4h", "1h", "15m"):
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

        confidence_base = getattr(self._cfg, "confidence_base", 75)
        lookback_swings = getattr(getattr(self._cfg, "structure", self._cfg), "lookback_swings", 20)
        sweep_lookback  = getattr(getattr(self._cfg, "sweeps", self._cfg), "lookback", 30)

        # ── Killzone bonus ────────────────────────────────────────────────
        # FIX AUDIT-5: Use the last candle's timestamp instead of wall-clock
        # so backtests get deterministic killzone bonuses based on when the
        # candle formed, not when the backtest runs.
        _last_ts = df["ts"].iloc[-1]
        try:
            _candle_dt = datetime.fromtimestamp(float(_last_ts) / 1000, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            logger.warning(f"SMC {symbol}: failed to parse candle timestamp {_last_ts!r}, using wall-clock fallback")
            _candle_dt = datetime.now(timezone.utc)
        current_hour = _candle_dt.hour
        _killzone_hours = {0, 1, 2, 3, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}
        killzone_bonus = 10 if current_hour in _killzone_hours else 0

        # ── Identify swing highs/lows ─────────────────────────────────────
        swing_highs: List[float] = []
        swing_lows:  List[float] = []
        swing_high_points: List[Tuple[int, float]] = []
        swing_low_points: List[Tuple[int, float]] = []
        lb = min(lookback_swings, len(highs) - 3)
        for i in range(1, lb):
            idx = -(i + 1)
            if highs[idx] > highs[idx - 1] and highs[idx] > highs[idx + 1]:
                swing_highs.append(highs[idx])
                swing_high_points.append((idx, highs[idx]))
            if lows[idx] < lows[idx - 1] and lows[idx] < lows[idx + 1]:
                swing_lows.append(lows[idx])
                swing_low_points.append((idx, lows[idx]))

        if not swing_highs or not swing_lows:
            return None

        key_swing_high = max(swing_highs[:5]) if swing_highs else highs[-20:-1].max()
        key_swing_low  = min(swing_lows[:5])  if swing_lows  else lows[-20:-1].min()
        # BOS SL FIX: for BOS stop-loss placement, use the NEAREST (highest) recent swing
        # low — the structural support just before the breakout.  min(swing_lows[:5]) gives
        # the deepest low, which produces excessively wide stops that kill RR on BOS entries.
        # We keep key_swing_low (deep) for liquidity sweep detection where we need the true
        # liquidity pool level, and add key_swing_low_near for BOS SL placement.
        key_swing_low_near  = max(swing_lows[:5])  if swing_lows  else lows[-20:-1].max()

        current_close = closes[-1]
        current_high  = highs[-1]
        current_low   = lows[-1]

        min_break_pct = getattr(getattr(self._cfg, "structure", self._cfg), "min_break_pct", 0.002)
        structure_trend = "UNKNOWN"
        has_choch = False
        choch_direction = ""
        if len(swing_high_points) >= 2 and len(swing_low_points) >= 2:
            _recent_high, _prev_high = swing_high_points[0][1], swing_high_points[1][1]
            _recent_low, _prev_low = swing_low_points[0][1], swing_low_points[1][1]
            if _recent_high > _prev_high and _recent_low > _prev_low:
                structure_trend = "BULLISH"
                if current_close < _recent_low * (1.0 - min_break_pct):
                    has_choch = True
                    choch_direction = "BEARISH"
            elif _recent_high < _prev_high and _recent_low < _prev_low:
                structure_trend = "BEARISH"
                if current_close > _recent_high * (1.0 + min_break_pct):
                    has_choch = True
                    choch_direction = "BULLISH"

        direction:  Optional[str] = None
        setup_type: str = ""
        confidence: float = confidence_base + regime_bonus + killzone_bonus
        entry_ref:  float = current_close
        sl_level:   float = 0.0
        confluence: List[str] = []
        raw_data:   Dict = {}

        # ─────────────────────────────────────────────────────────────────
        # Priority 1: Liquidity Sweep + Reversal
        # ─────────────────────────────────────────────────────────────────
        sweep_lows  = lows[-sweep_lookback:-1]
        sweep_highs = highs[-sweep_lookback:-1]
        min_wick_pct = getattr(getattr(self._cfg, "sweeps", self._cfg), "min_wick_pct", 0.003)

        # Bullish sweep: wick below key swing low then close above it
        if (current_low < key_swing_low
                and current_close > key_swing_low
                and (key_swing_low - current_low) / key_swing_low >= min_wick_pct):
            direction  = "LONG"
            setup_type = "LiquiditySweep"
            confidence += 15
            sl_level   = current_low - atr * 0.3
            entry_ref  = current_close
            confluence.append(f"✅ Bullish liquidity sweep below {fmt_price(key_swing_low)} — stop hunt confirmed")
            confluence.append(f"✅ Wick below swing low by {fmt_price(key_swing_low - current_low)}")
            raw_data["sweep_low"] = current_low
            raw_data["sweep_level"] = key_swing_low

        # Bearish sweep: wick above key swing high then close below it
        elif (current_high > key_swing_high
                and current_close < key_swing_high
                and (current_high - key_swing_high) / key_swing_high >= min_wick_pct):
            direction  = "SHORT"
            setup_type = "LiquiditySweep"
            confidence += 15
            sl_level   = current_high + atr * 0.3
            entry_ref  = current_close
            confluence.append(f"✅ Bearish liquidity sweep above {fmt_price(key_swing_high)} — stop hunt confirmed")
            confluence.append(f"✅ Wick above swing high by {fmt_price(current_high - key_swing_high)}")
            raw_data["sweep_high"] = current_high
            raw_data["sweep_level"] = key_swing_high

        # ─────────────────────────────────────────────────────────────────
        # Priority 2: Fair Value Gap
        # ─────────────────────────────────────────────────────────────────
        if direction is None and len(closes) >= 3:
            fvg_min_pct = getattr(getattr(self._cfg, "fvg", self._cfg), "min_size_pct", 0.002)
            c1_high, c3_low   = highs[-3], lows[-1]   # bullish FVG
            c1_low,  c3_high  = lows[-3],  highs[-1]  # bearish FVG

            # Bullish FVG: candle-1 high < candle-3 low (gap between c1 top and c3 bottom)
            if c1_high < c3_low and current_close > c1_high:
                fvg_mid = (c1_high + c3_low) / 2
                fvg_size = c3_low - c1_high
                # FVG RETURN FIX: require price to be INSIDE or approaching the gap from above,
                # not still above it. A valid FVG trade needs price to return toward c1_high.
                # Old check: current_close <= c3_low + 0.5 ATR — allowed entries above the gap.
                # New check: current_close <= fvg_mid + 0.3 ATR — price must be in the lower
                # half of the gap or approaching it, confirming the retest is in progress.
                _in_gap = c1_high - atr * 0.3 <= current_close <= fvg_mid + atr * 0.3
                if fvg_size / c1_high >= fvg_min_pct and _in_gap:
                    direction  = "LONG"
                    setup_type = "FairValueGap"
                    entry_ref  = fvg_mid
                    sl_level   = c1_high - atr * 0.3
                    confluence.append(f"✅ Bullish FVG: {fmt_price(c1_high)} → {fmt_price(c3_low)}")
                    confluence.append(f"   FVG midpoint: {fmt_price(fvg_mid)} | Price returned to gap")
                    raw_data["fvg_low"] = c1_high
                    raw_data["fvg_high"] = c3_low

            # Bearish FVG: candle-1 low > candle-3 high
            elif c1_low > c3_high and current_close < c1_low:
                fvg_mid  = (c1_low + c3_high) / 2
                fvg_size = c1_low - c3_high
                # FVG RETURN FIX: price must be in upper half of gap or approaching from below
                _in_gap = fvg_mid - atr * 0.3 <= current_close <= c1_low + atr * 0.3
                if fvg_size / c1_low >= fvg_min_pct and _in_gap:
                    direction  = "SHORT"
                    setup_type = "FairValueGap"
                    entry_ref  = fvg_mid
                    sl_level   = c1_low + atr * 0.3
                    confluence.append(f"✅ Bearish FVG: {fmt_price(c3_high)} → {fmt_price(c1_low)}")
                    confluence.append(f"   FVG midpoint: {fmt_price(fvg_mid)} | Price returned to gap")
                    raw_data["fvg_high"] = c1_low
                    raw_data["fvg_low"] = c3_high

        # ─────────────────────────────────────────────────────────────────
        # Priority 3: Order Block
        # ─────────────────────────────────────────────────────────────────
        if direction is None:
            ob_min_impulse = getattr(getattr(self._cfg, "order_blocks", self._cfg), "min_impulse_pct", 0.008)
            ob_max_age     = getattr(getattr(self._cfg, "order_blocks", self._cfg), "max_ob_age_bars", 50)
            search_bars    = min(ob_max_age, len(closes) - 3)

            for i in range(2, search_bars):
                # Bullish OB: last bearish candle before a bullish impulse
                # BUG-9 FIX: measure impulse size relative to the OB body, not the
                # impulse candle's own open price. On near-zero-price tokens, dividing
                # by open price produces extreme percentages that fire on noise.
                _bull_ob_body = opens[-i] - closes[-i]  # bearish OB body
                if (_bull_ob_body > 0
                        and closes[-i + 1] > opens[-i + 1]   # impulse candle is bullish
                        and (closes[-i + 1] - opens[-i + 1]) / _bull_ob_body >= ob_min_impulse):
                    ob_high = opens[-i]
                    ob_low  = closes[-i]
                    if ob_low <= current_close <= ob_high + atr * 0.5:
                        direction  = "LONG"
                        setup_type = "OrderBlock"
                        entry_ref  = (ob_high + ob_low) / 2
                        sl_level   = ob_low - atr * 0.3
                        confluence.append(f"✅ Bullish OB at {fmt_price(ob_low)}-{fmt_price(ob_high)} ({i} bars ago)")
                        raw_data["ob_low"] = ob_low
                        raw_data["ob_high"] = ob_high
                        break

                # Bearish OB: last bullish candle before a bearish impulse
                _bear_ob_body = closes[-i] - opens[-i]  # bullish OB body
                if (_bear_ob_body > 0
                        and closes[-i + 1] < opens[-i + 1]  # impulse is bearish
                        and (opens[-i + 1] - closes[-i + 1]) / _bear_ob_body >= ob_min_impulse):
                    ob_high = closes[-i]
                    ob_low  = opens[-i]
                    if ob_low - atr * 0.5 <= current_close <= ob_high:
                        direction  = "SHORT"
                        setup_type = "OrderBlock"
                        entry_ref  = (ob_high + ob_low) / 2
                        sl_level   = ob_high + atr * 0.3
                        confluence.append(f"✅ Bearish OB at {fmt_price(ob_low)}-{fmt_price(ob_high)} ({i} bars ago)")
                        raw_data["ob_low"] = ob_low
                        raw_data["ob_high"] = ob_high
                        break

        # ─────────────────────────────────────────────────────────────────
        # Priority 4: Break of Structure
        # ─────────────────────────────────────────────────────────────────
        if direction is None:
            if (current_close > key_swing_high
                    and (current_close - key_swing_high) / key_swing_high >= min_break_pct):
                direction  = "LONG"
                setup_type = "BreakOfStructure"
                entry_ref  = current_close
                sl_level   = key_swing_low_near - atr * 0.3
                confluence.append(f"✅ BOS: Price broke above swing high {fmt_price(key_swing_high)}")
                raw_data["bos_level"] = key_swing_high

            elif (current_close < key_swing_low
                    and (key_swing_low - current_close) / key_swing_low >= min_break_pct):
                direction  = "SHORT"
                setup_type = "BreakOfStructure"
                entry_ref  = current_close
                sl_level   = key_swing_high + atr * 0.3
                confluence.append(f"✅ BOS: Price broke below swing low {fmt_price(key_swing_low)}")
                raw_data["bos_level"] = key_swing_low

        if direction is None:
            return None

        # ── Direction-aware regime confidence ─────────────────────────────
        # With-trend (LONG in BULL, SHORT in BEAR) gets +8;
        # Counter-trend (LONG in BEAR, SHORT in BULL) gets -12.
        _is_with_trend = (
            (direction == "LONG" and regime == "BULL_TREND") or
            (direction == "SHORT" and regime == "BEAR_TREND")
        )
        if _is_with_trend or regime not in ("BULL_TREND", "BEAR_TREND"):
            regime_bonus = self._REGIME_CONF_WITH_TREND.get(regime, 0)
        else:
            if not self._allows_counter_trend_setup(setup_type, has_choch, choch_direction, direction):
                return None
            regime_bonus = self._REGIME_CONF_COUNTER_TREND.get(regime, 0)
        confidence += regime_bonus

        if sl_level == 0.0:
            return None

        # ── Build entry zone around entry_ref ─────────────────────────────
        vp = compute_vol_percentile(highs, lows, closes)
        if direction == "LONG":
            entry_low  = entry_ref - atr * rp.entry_zone_tight
            entry_high = entry_ref + atr * rp.entry_zone_atr
            stop_loss  = min(sl_level, entry_low - atr * 0.1)
            risk       = entry_low - stop_loss
            # FIX: TPs were ATR-proportional from entry_high, but aggregator
            # verifies R/R from entry_mid — mismatch inflated R/R to 9-13R.
            # Now risk-proportional from entry_ref: tp_mult=2.8 → exactly 2.8R.
            risk_dist  = entry_ref - stop_loss
            tp1_calc   = entry_ref + risk_dist * rp.volatility_scaled_tp1(tf, vp)
            # Anchor TP1 to nearest swing high between entry and calculated TP1.
            _dz = atr * Penalties.TP1_SWING_DEAD_ZONE_ATR
            _n  = Penalties.TP1_MAX_SWING_LEVELS
            _tp1_resist = [sh for sh in swing_highs[:_n]
                          if entry_high + _dz < sh < tp1_calc]
            tp1        = min(_tp1_resist) if _tp1_resist else tp1_calc
            tp2        = entry_ref + risk_dist * rp.volatility_scaled_tp2(tf, vp)
            tp3        = entry_ref + risk_dist * rp.volatility_scaled_tp3(tf, vp)
        else:
            entry_high = entry_ref + atr * rp.entry_zone_tight
            entry_low  = entry_ref - atr * rp.entry_zone_atr
            stop_loss  = max(sl_level, entry_high + atr * 0.1)
            risk       = stop_loss - entry_high
            risk_dist  = stop_loss - entry_ref
            tp1_calc   = entry_ref - risk_dist * rp.volatility_scaled_tp1(tf, vp)
            # Anchor TP1 to nearest swing low between entry and calculated TP1.
            _dz = atr * Penalties.TP1_SWING_DEAD_ZONE_ATR
            _n  = Penalties.TP1_MAX_SWING_LEVELS
            _tp1_support = [sl for sl in swing_lows[:_n]
                           if tp1_calc < sl < entry_low - _dz]
            tp1        = max(_tp1_support) if _tp1_support else tp1_calc
            tp2        = entry_ref - risk_dist * rp.volatility_scaled_tp2(tf, vp)
            tp3        = entry_ref - risk_dist * rp.volatility_scaled_tp3(tf, vp)

        if risk <= 0:
            return None

        rr_ratio = self.calculate_effective_rr(
            direction=direction,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            tp2=tp2,
        )

        # Confluence additions
        confluence.append(f"📊 Setup: {setup_type} | TF: {tf}")
        if has_choch:
            confluence.append(f"🧭 CHoCH: {choch_direction.title()} structure shift detected")
        if killzone_bonus:
            confluence.append(f"⏰ Killzone active (UTC {current_hour}h): +{killzone_bonus} confidence")
        confluence.append(f"📈 Regime: {regime} ({'+' if regime_bonus >= 0 else ''}{regime_bonus})")
        confluence.append(f"🎯 R:R {rr_ratio:.2f} | ATR: {fmt_price(atr)}")

        raw_data.update({
            "setup_type": setup_type,
            "atr": atr,
            "regime": regime,
            "key_swing_high": key_swing_high,
            "key_swing_low": key_swing_low,
            "key_swing_low_near": key_swing_low_near,
            "structure_trend": structure_trend,
            "has_choch": has_choch,
            "choch_direction": choch_direction,
            "has_bos": setup_type == "BreakOfStructure",
            "has_ob": setup_type == "OrderBlock",
            "has_fvg": setup_type == "FairValueGap",
            "has_sweep": setup_type == "LiquiditySweep",
        })

        confidence = min(95, max(40, confidence))

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
            analysis_timeframes=["4h", "1h", "15m"],
            confluence=confluence,
            raw_data=raw_data,
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
SMCStrategy = SmartMoneyConcepts
