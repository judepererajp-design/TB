"""
TitanBot Pro — Price Action Strategy
======================================
Detects high-probability candlestick patterns at key structural levels.

Patterns scanned (in priority order):
  - PinBar (hammer / shooting star)    — highest weight
  - Engulfing (bullish / bearish)
  - MorningStar / EveningStar
  - ThreeWhiteSoldiers / ThreeBlackCrows
  - Doji (only at key levels)

Requires pattern to form at a key structural level:
  - Swing high / swing low (last 50 bars)
  - Bollinger Band extremes
  - Round-number proximity (0.5% tolerance)
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.loader import cfg
from config.constants import Penalties, STRATEGY_VALID_REGIMES
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp, compute_vol_percentile

logger = logging.getLogger(__name__)

# Pattern priority weights for confidence calculation
_PATTERN_WEIGHTS = {
    "PinBar":             1.0,
    "Engulfing":          0.85,
    "MorningStar":        0.80,
    "EveningStar":        0.80,
    "ThreeWhiteSoldiers": 0.75,
    "ThreeBlackCrows":    0.75,
    "InsideBar":          0.55,
    "Doji":               0.50,
}

# SL buffer as a fraction of sl_atr_mult.
# 0.35 × 1.2 ATR ≈ 0.42 ATR — tight enough to respect pattern invalidation,
# wide enough to survive market noise (spreads, slippage, minor wicks).
_SL_BUFFER_FACTOR = 0.35
_TP_MIN_GAP_ATR = 0.2


class PriceAction(BaseStrategy):

    name = "PriceAction"
    description = "Candlestick pattern detection at key structural levels"

    # Price action patterns work in any regime — regime bonus/penalty applied
    VALID_REGIMES = STRATEGY_VALID_REGIMES["PriceAction"]

    # Direction-aware regime confidence: with-trend gets a boost, counter-trend
    # gets a penalty.  Same fix as SMC — prevents inflating counter-trend signals.
    _REGIME_CONF_WITH_TREND = {
        "BULL_TREND":  +5,
        "BEAR_TREND":  +5,
        "VOLATILE":    +3,
        "CHOPPY":      +2,
        "UNKNOWN":     0,
    }
    _REGIME_CONF_COUNTER_TREND = {
        "BULL_TREND":  -8,    # SHORT in bull
        "BEAR_TREND":  -8,    # LONG in bear
        "VOLATILE":    +3,
        "CHOPPY":      +2,
        "UNKNOWN":     0,
    }

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.price_action

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

        regime_bonus = 0  # deferred — computed after direction is known

        # ── Multi-Timeframe Analysis: 4h bias → 1h zone → 15m trigger ────
        _htf_bias = self.mtf_get_bias(ohlcv_dict, tf="4h")
        _mtf_zone = None
        _mtf_trigger = {"triggered": False, "trigger_type": "none", "quality": 0.0}
        _mtf_bonus = 0

        if _htf_bias["bias"] != "NEUTRAL" and _htf_bias["confidence"] > 55:
            _mtf_zone = self.mtf_find_zone(ohlcv_dict, tf="1h", bias=_htf_bias["bias"])
            if _mtf_zone:
                _trigger_dir = "LONG" if _htf_bias["bias"] == "BULLISH" else "SHORT"
                _mtf_trigger = self.mtf_check_trigger(
                    ohlcv_dict, tf="15m", direction=_trigger_dir, zone=_mtf_zone
                )
                if _mtf_trigger["triggered"]:
                    _mtf_bonus = int(_mtf_trigger["quality"] * 10)  # Up to +10

        tf = None
        for candidate_tf in ("4h", "1h"):
            if candidate_tf in ohlcv_dict and len(ohlcv_dict[candidate_tf]) >= 55:
                tf = candidate_tf
                break
        if tf is None:
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

        confidence_base = getattr(self._cfg, "confidence_base", 68)

        # ── Detect patterns on the last signal candle ─────────────────────
        try:
            from patterns.candlestick import detect_all
            patterns = detect_all(opens, highs, lows, closes, atr=atr)
        except Exception as e:
            logger.debug(f"PriceAction pattern detection error: {e}")
            return None

        if not patterns:
            return None

        # ── Filter to configured patterns and determine dominant direction ─
        cfg_patterns = getattr(self._cfg, "patterns", None)
        long_patterns  = [p for p in patterns if p["direction"] == "LONG"]
        short_patterns = [p for p in patterns if p["direction"] == "SHORT"]

        # Pick dominant direction by highest-weight pattern
        def _score(p_list):
            if not p_list:
                return 0.0
            return max(_PATTERN_WEIGHTS.get(p["pattern"], 0.5) * p["strength"] for p in p_list)

        long_score  = _score(long_patterns)
        short_score = _score(short_patterns)

        if long_score == 0.0 and short_score == 0.0:
            return None

        if long_score >= short_score:
            direction = "LONG"
            signal_patterns = long_patterns
        else:
            direction = "SHORT"
            signal_patterns = short_patterns

        # ── Direction-aware regime confidence ─────────────────────────────
        _is_with_trend = (
            (direction == "LONG" and regime == "BULL_TREND") or
            (direction == "SHORT" and regime == "BEAR_TREND")
        )
        if _is_with_trend or regime not in ("BULL_TREND", "BEAR_TREND"):
            regime_bonus = self._REGIME_CONF_WITH_TREND.get(regime, 0)
        else:
            regime_bonus = self._REGIME_CONF_COUNTER_TREND.get(regime, 0)

        # Pick the strongest pattern as primary
        primary_pattern = max(signal_patterns, key=lambda p: _PATTERN_WEIGHTS.get(p["pattern"], 0.5) * p["strength"])

        # ── Key level detection ────────────────────────────────────────────
        # Audit P1: enforce a minimum 30-bar swing-scan floor regardless of
        # config, so major structural swings aren't missed when a user sets
        # an unusually small key_level_lookback.
        lookback_kl = int(getattr(self._cfg, "key_level_lookback", 50))
        lookback_kl = max(30, lookback_kl)
        swing_highs = []
        swing_lows  = []
        lb = min(lookback_kl, len(highs) - 2)
        for i in range(1, lb):
            idx = -(i + 1)
            if highs[idx] > highs[idx - 1] and highs[idx] > highs[idx + 1]:
                swing_highs.append(highs[idx])
            if lows[idx] < lows[idx - 1] and lows[idx] < lows[idx + 1]:
                swing_lows.append(lows[idx])

        current_price = closes[-1]
        current_high  = highs[-1]
        current_low   = lows[-1]

        # Bollinger extremes
        _, bb_upper, bb_lower = self.calculate_bollinger(closes, period=20, std_mult=2.0)
        if not np.isfinite(bb_upper) or not np.isfinite(bb_lower):
            return None

        # Round numbers: dynamic magnitude scaled to instrument price.
        # Keep coarse levels for high-priced assets and usable decimals for sub-$1.
        if current_price <= 0:
            logger.warning(f"PriceAction: invalid current price for {symbol}: {current_price}")
            return None
        _log10 = math.floor(math.log10(current_price))
        if current_price >= 1.0:
            _exp = max(0, _log10 - 1)
        else:
            _exp = _log10 - 1
        magnitude = 10 ** _exp
        nearest_round = round(current_price / magnitude) * magnitude
        round_proximity = abs(current_price - nearest_round) / current_price

        at_key_level = False
        key_level_desc = ""
        kl_tolerance = atr * float(getattr(self._cfg, "key_level_tolerance_atr", 0.5))

        if direction == "LONG":
            # At swing low, BB lower, or round number below
            for sl in swing_lows[:5]:
                if abs(current_low - sl) <= kl_tolerance:
                    at_key_level = True
                    key_level_desc = f"swing low {fmt_price(sl)}"
                    break
            if not at_key_level and abs(current_low - bb_lower) <= kl_tolerance:
                at_key_level = True
                key_level_desc = f"BB lower {fmt_price(bb_lower)}"
            if not at_key_level and round_proximity < 0.005:
                at_key_level = True
                key_level_desc = f"round number {fmt_price(nearest_round)}"
        else:
            for sh in swing_highs[:5]:
                if abs(current_high - sh) <= kl_tolerance:
                    at_key_level = True
                    key_level_desc = f"swing high {fmt_price(sh)}"
                    break
            if not at_key_level and abs(current_high - bb_upper) <= kl_tolerance:
                at_key_level = True
                key_level_desc = f"BB upper {fmt_price(bb_upper)}"
            if not at_key_level and round_proximity < 0.005:
                at_key_level = True
                key_level_desc = f"round number {fmt_price(nearest_round)}"

        require_key_level = getattr(self._cfg, "require_key_level", True)
        if require_key_level and not at_key_level:
            return None

        # ── Volume confirmation ────────────────────────────────────────────
        avg_vol    = float(np.mean(volumes[-20:]))
        vol_ratio  = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        vol_mult = float(getattr(self._cfg, "vol_confirmation_mult", 1.5))
        vol_confirmed = vol_ratio >= vol_mult

        # Rising volume sequence: 3-bar momentum toward the reversal bar
        _vol_rising = (
            len(volumes) >= 4
            and volumes[-2] > volumes[-3] > volumes[-4]
        )

        # ── HTF alignment gate/penalty ───────────────────────────────────────
        _mtf_mismatch = False
        _mtf_penalty = 0
        if _htf_bias["bias"] in ("BULLISH", "BEARISH") and _htf_bias["confidence"] > 55:
            _htf_dir = "LONG" if _htf_bias["bias"] == "BULLISH" else "SHORT"
            if direction != _htf_dir:
                _mtf_mismatch = True
                if bool(getattr(self._cfg, "reject_mtf_mismatch", False)):
                    return None
                _mtf_penalty = int(getattr(self._cfg, "mtf_mismatch_penalty", 8))
                _mtf_bonus = 0

        # ── Confidence ────────────────────────────────────────────────────
        confidence = float(confidence_base) + regime_bonus + _mtf_bonus - _mtf_penalty
        confidence += primary_pattern["strength"] * 15
        if at_key_level:
            confidence += 12
        if vol_confirmed:
            confidence += 8
        if len(signal_patterns) > 1:
            confidence += 5 * (len(signal_patterns) - 1)

        # ── Prior-bar context for single-bar patterns ─────────────────────
        # For PinBar and Doji — which are single-bar signals — check that the
        # preceding bar's direction is consistent with the expected reversal.
        # Strength is ATR-normalized: a large prior body = stronger context signal.
        # LONG setup: prior bar should be bearish (downmove to fade)
        # SHORT setup: prior bar should be bullish (upmove to fade)
        _single_bar_patterns = {"PinBar", "Doji"}
        _is_single_bar = primary_pattern["pattern"] in _single_bar_patterns
        _prior_bar_confirms = False
        _prior_bar_conflicts = False
        _context_adj = 0.0
        _context_bonus = 0.0
        if _is_single_bar and len(opens) >= 2:
            _prior_close_vs_open = closes[-2] - opens[-2]
            _context_strength = abs(_prior_close_vs_open) / atr if atr > 0 else 0.0
            _context_adj = min(5, _context_strength * 5)
            if direction == "LONG" and _prior_close_vs_open < 0:
                _prior_bar_confirms = True   # prior sell-off confirms reversal context
                _context_bonus = _context_adj
            elif direction == "SHORT" and _prior_close_vs_open > 0:
                _prior_bar_confirms = True
                _context_bonus = _context_adj
            elif direction == "LONG" and _prior_close_vs_open > 0:
                _prior_bar_conflicts = True  # prior bar moving with us = chasing
                confidence -= _context_adj   # penalty applies in full, not capped
            elif direction == "SHORT" and _prior_close_vs_open < 0:
                _prior_bar_conflicts = True
                confidence -= _context_adj   # penalty applies in full, not capped

        # Vol-sequence bonuses: rising sequence (+4) + above-average magnitude (+2).
        # Context and volume are correlated (a strong prior bar drives volume), so cap
        # the combined upside at +8 to prevent correlated signals inflating confidence.
        _vol_seq_bonus = 4.0 if _vol_rising else 0.0
        _vol_mag_bonus = (
            2.0 if (_vol_rising and avg_vol > 0 and volumes[-2] > 1.5 * avg_vol) else 0.0
        )
        _pa_micro_bonus = min(8, _context_bonus + _vol_seq_bonus + _vol_mag_bonus)
        confidence += _pa_micro_bonus

        # ── Entry / SL / TP levels ────────────────────────────────────────
        # FIX: SL buffer used entry_zone_tight (0.10 ATR) — far too thin.
        # A 0.10 ATR buffer above/below the pattern extreme gets hit by noise
        # and inflates R:R to unrealistic levels (5-7R on small candles).
        # Use sl_atr_mult × _SL_BUFFER_FACTOR — still tight enough to respect
        # the pattern invalidation level, but wide enough to survive noise.
        _sl_buf = atr * rp.sl_atr_mult * _SL_BUFFER_FACTOR
        vp = compute_vol_percentile(highs, lows, closes)
        if direction == "LONG":
            pattern_low  = current_low
            entry_low    = current_price - atr * rp.entry_zone_tight
            entry_high   = current_price + atr * rp.entry_zone_atr
            stop_loss    = pattern_low - _sl_buf
            tp1_atr      = entry_high + atr * rp.volatility_scaled_tp1(tf, vp)
            # Anchor TP1 to nearest swing high between entry and ATR-based TP1.
            # Prevents setting TP1 above untested resistance (e.g. PDH).
            _dz = atr * Penalties.TP1_SWING_DEAD_ZONE_ATR
            _n  = Penalties.TP1_MAX_SWING_LEVELS
            _tp1_resist  = [sh for sh in swing_highs[:_n]
                           if entry_high + _dz < sh < tp1_atr]
            tp1          = min(_tp1_resist) if _tp1_resist else tp1_atr
            tp2_candidates = [sh for sh in swing_highs[:3] if sh > entry_high + atr]
            tp2 = min(tp2_candidates) if tp2_candidates else (entry_high + atr * rp.volatility_scaled_tp2(tf, vp))
            tp3 = entry_high + atr * rp.volatility_scaled_tp3(tf, vp)
            _tp_gap = atr * float(getattr(self._cfg, "tp_min_gap_atr", _TP_MIN_GAP_ATR))
            tp2 = max(tp2, tp1 + _tp_gap)
            tp3 = max(tp3, tp2 + _tp_gap)
        else:
            pattern_high = current_high
            entry_high   = current_price + atr * rp.entry_zone_tight
            entry_low    = current_price - atr * rp.entry_zone_atr
            stop_loss    = pattern_high + _sl_buf
            tp1_atr      = entry_low - atr * rp.volatility_scaled_tp1(tf, vp)
            # Anchor TP1 to nearest swing low between entry and ATR-based TP1.
            _dz = atr * Penalties.TP1_SWING_DEAD_ZONE_ATR
            _n  = Penalties.TP1_MAX_SWING_LEVELS
            _tp1_support = [sl for sl in swing_lows[:_n]
                           if tp1_atr < sl < entry_low - _dz]
            tp1          = max(_tp1_support) if _tp1_support else tp1_atr
            tp2_candidates = [sl for sl in swing_lows[:3] if sl < entry_low - atr]
            tp2 = max(tp2_candidates) if tp2_candidates else (entry_low - atr * rp.volatility_scaled_tp2(tf, vp))
            tp3 = entry_low - atr * rp.volatility_scaled_tp3(tf, vp)
            _tp_gap = atr * float(getattr(self._cfg, "tp_min_gap_atr", _TP_MIN_GAP_ATR))
            tp2 = min(tp2, tp1 - _tp_gap)
            tp3 = min(tp3, tp2 - _tp_gap)

        risk = (entry_low - stop_loss) if direction == "LONG" else (stop_loss - entry_high)
        if risk <= 0:
            return None
        rr_ratio = abs(tp2 - current_price) / risk

        # Setup class depends on timeframe
        setup_class = "swing" if tf == "4h" else "intraday"

        confluence: List[str] = [
            f"✅ {primary_pattern['pattern']} ({direction}) — strength {primary_pattern['strength']:.2f}",
        ]
        if len(signal_patterns) > 1:
            extras = [p["pattern"] for p in signal_patterns if p != primary_pattern]
            confluence.append(f"✅ Additional patterns: {', '.join(extras)}")
        if at_key_level:
            confluence.append(f"✅ Key level: {key_level_desc}")
        if vol_confirmed:
            confluence.append(f"✅ Volume: {vol_ratio:.1f}x average")
        if _vol_rising:
            confluence.append(
                "✅ Volume trend: rising sequence (x3 bars)"
                + (" + magnitude" if _vol_mag_bonus > 0 else "")
                + f" — micro-bonus +{_pa_micro_bonus:.0f} (cap 8)"
            )
        if _is_single_bar:
            if _prior_bar_confirms:
                confluence.append(f"✅ Prior bar: confirms reversal context (+{_context_bonus:.1f})")
            elif _prior_bar_conflicts:
                confluence.append(f"⚠️ Prior bar: conflicts with signal direction (-{_context_adj:.1f})")
        if _mtf_mismatch:
            confluence.append(f"⚠️ HTF mismatch: 4h {_htf_bias['bias']} vs signal {direction} (-{_mtf_penalty})")
        confluence.append(f"📊 Regime: {regime} | TF: {tf}")
        if _mtf_bonus > 0:
            confluence.append(
                f"🔄 MTF confirmed: 4h {_htf_bias['bias']} → "
                f"1h {_mtf_zone['zone_type'] if _mtf_zone else 'N/A'} → "
                f"15m {_mtf_trigger['trigger_type']} (+{_mtf_bonus})"
            )
        confluence.append(f"🎯 R:R {rr_ratio:.2f} | ATR: {fmt_price(atr)}")

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
            setup_class=setup_class,
            timeframe=tf,
            analysis_timeframes=["4h", "1h"],
            confluence=confluence,
            raw_data={
                "primary_pattern": primary_pattern["pattern"],
                "all_patterns": signal_patterns,
                "at_key_level": at_key_level,
                "key_level_desc": key_level_desc,
                "vol_ratio": vol_ratio,
                "atr": atr,
                "regime": regime,
            },
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
PriceActionStrategy = PriceAction
