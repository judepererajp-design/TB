"""
TitanBot Pro — Wyckoff Accumulation/Distribution Strategy
===========================================================
Detects Wyckoff Phase C (Spring/UTAD) and Phase D (SOS/SOW) setups.

Phase C entries (Spring/UTAD) are the highest conviction:
  - Spring: price dips below trading range support, then immediately recovers
    (last manipulation shaking out weak longs before the markup phase)
  - UTAD: price spikes above trading range resistance, then fails
    (last manipulation trapping weak shorts before the markdown phase)

Phase D entries (SOS/SOW) are more common but lower conviction.

Uses patterns/wyckoff.py WyckoffAnalyzer for phase detection.
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


class WyckoffAccDist(BaseStrategy):

    name = "WyckoffAccDist"
    description = "Wyckoff Spring/UTAD and SOS/SOW phase entries"

    # Wyckoff accumulation/distribution works in all regimes, but direction must
    # align with trend to receive full confidence.  Counter-trend setups (LONG in
    # BEAR_TREND, SHORT in BULL_TREND) receive a penalty; neutral regimes are flat.
    VALID_REGIMES = {"BULL_TREND", "BEAR_TREND", "VOLATILE", "CHOPPY", "UNKNOWN"}

    _REGIME_CONF_WITH_TREND = {
        "BULL_TREND":  +8,
        "BEAR_TREND":  +8,
        "VOLATILE":    +3,
        "CHOPPY":      -5,
        "UNKNOWN":     0,
    }
    _REGIME_CONF_COUNTER_TREND = {
        "BULL_TREND":  -12,   # SHORT in bull — counter-trend risk
        "BEAR_TREND":  -12,   # LONG in bear — counter-trend risk
        "VOLATILE":    +3,    # No dominant trend to counter
        "CHOPPY":      -5,
        "UNKNOWN":     0,
    }

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.wyckoff

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            logger.debug(f"WyckoffAccDist.analyze {symbol}: {e}")
            return None

    async def _analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime context ────────────────────────────────────────────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception:
            regime = "UNKNOWN"

        tf = getattr(self._cfg, "timeframe", "4h")
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 60:
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

        confidence_base = getattr(self._cfg, "confidence_base", 76)

        # ── Run Wyckoff analysis ───────────────────────────────────────────
        try:
            from patterns.wyckoff import WyckoffAnalyzer, WyckoffPhase
            analyzer = WyckoffAnalyzer()
            result   = analyzer.analyze(ohlcv, timeframe=tf)
        except Exception as e:
            logger.warning(f"WyckoffAccDist: WyckoffAnalyzer error for {symbol}: {e}")
            return None

        if result is None:
            return None

        phase = result.phase

        # ── Phase gate — only act on Phase C and D ─────────────────────────
        from patterns.wyckoff import WyckoffPhase as WP

        actionable_phases = {
            WP.ACCUMULATION_C,  # Spring detected → LONG
            WP.DISTRIBUTION_C,  # UTAD detected   → SHORT
            WP.ACCUMULATION_D,  # SOS              → LONG
            WP.DISTRIBUTION_D,  # SOW              → SHORT
        }

        if phase not in actionable_phases:
            return None

        # ── Direction and confidence from phase ───────────────────────────
        if phase in (WP.ACCUMULATION_C, WP.ACCUMULATION_D):
            direction = "LONG"
        else:
            direction = "SHORT"

        confidence = float(confidence_base)

        if phase in (WP.ACCUMULATION_C, WP.DISTRIBUTION_C):
            confidence += 15   # Phase C = highest edge (Spring/UTAD)
        else:
            confidence += 5    # Phase D = lower edge (SOS/SOW)

        if result.volume_confirms:
            confidence += 10
        if result.spring_detected:
            confidence += 8
        if result.utad_detected:
            confidence += 8

        # ── W-1: Direction-aware regime bonus/penalty ─────────────────────
        _is_with_trend = (
            (direction == "LONG"  and regime == "BULL_TREND") or
            (direction == "SHORT" and regime == "BEAR_TREND")
        )
        if _is_with_trend or regime not in ("BULL_TREND", "BEAR_TREND"):
            regime_bonus = self._REGIME_CONF_WITH_TREND.get(regime, 0)
        else:
            regime_bonus = self._REGIME_CONF_COUNTER_TREND.get(regime, 0)
        confidence += regime_bonus

        # ── Entry zone from WyckoffResult ──────────────────────────────────
        entry_zone = result.entry_zone
        if not (isinstance(entry_zone, (tuple, list)) and len(entry_zone) == 2):
            return None
        wy_entry_low, wy_entry_high = float(entry_zone[0]), float(entry_zone[1])
        wy_stop_loss = result.stop_loss
        current_price = closes[-1]

        # W-2: Tighten staleness check — entry zone must be reachable
        _staleness_atr = getattr(self._cfg, "entry_staleness_atr", 2)
        if abs(current_price - (wy_entry_low + wy_entry_high) / 2) > atr * _staleness_atr:
            return None

        # W-Q4: Phase D requires Phase E onset — price must break the range boundary
        # with volume confirmation before entry (avoid buying "hope it works" zones)
        if phase in (WP.ACCUMULATION_D, WP.DISTRIBUTION_D):
            key = result.key_level
            phase_e_confirmed = (
                (direction == "LONG"  and current_price > key and result.volume_confirms) or
                (direction == "SHORT" and current_price < key and result.volume_confirms)
            )
            if not phase_e_confirmed:
                return None

        _sl_atr_max = getattr(self._cfg, "sl_atr_max", 3)
        _swing_lookback = getattr(self._cfg, "swing_lookback_bars", 50)
        vp = compute_vol_percentile(highs, lows, closes)
        if direction == "LONG":
            entry_low   = wy_entry_low
            entry_high  = wy_entry_high
            # W-4: clamp SL so it never trails more than sl_atr_max below entry
            stop_loss   = min(wy_stop_loss, entry_low - atr * 0.3)
            stop_loss   = max(stop_loss,    entry_low - atr * _sl_atr_max)
            tp1         = entry_high + atr * rp.volatility_scaled_tp1(tf, vp)
            # W-Q3: TP2 prefers structurally validated swing highs (multi-touch)
            swing_highs = []
            _tol = atr * 0.5
            for i in range(1, min(_swing_lookback, len(highs) - 1)):
                idx = -(i + 1)
                if highs[idx] > highs[idx - 1] and highs[idx] > highs[idx + 1]:
                    if highs[idx] > entry_high + atr:
                        level = highs[idx]
                        touches = sum(1 for h in highs[-_swing_lookback:] if abs(h - level) <= _tol)
                        swing_highs.append((touches, level))
            # Prefer levels tested 2+ times; fall back to any pivot if none qualify
            validated = [lvl for (cnt, lvl) in swing_highs if cnt >= 2]
            tp2 = min(validated) if validated else (
                min(lvl for _, lvl in swing_highs) if swing_highs
                else entry_high + atr * rp.volatility_scaled_tp2(tf, vp)
            )
            tp3 = entry_high + atr * rp.volatility_scaled_tp3(tf, vp)
        else:
            entry_low   = wy_entry_low
            entry_high  = wy_entry_high
            # W-4: clamp SL so it never exceeds sl_atr_max above entry
            stop_loss   = max(wy_stop_loss, entry_high + atr * 0.3)
            stop_loss   = min(stop_loss,    entry_high + atr * _sl_atr_max)
            tp1         = entry_low - atr * rp.volatility_scaled_tp1(tf, vp)
            # W-Q3: TP2 prefers structurally validated swing lows (multi-touch)
            swing_lows  = []
            _tol = atr * 0.5
            for i in range(1, min(_swing_lookback, len(lows) - 1)):
                idx = -(i + 1)
                if lows[idx] < lows[idx - 1] and lows[idx] < lows[idx + 1]:
                    if lows[idx] < entry_low - atr:
                        level = lows[idx]
                        touches = sum(1 for l in lows[-_swing_lookback:] if abs(l - level) <= _tol)
                        swing_lows.append((touches, level))
            validated = [lvl for (cnt, lvl) in swing_lows if cnt >= 2]
            tp2 = max(validated) if validated else (
                max(lvl for _, lvl in swing_lows) if swing_lows
                else entry_low - atr * rp.volatility_scaled_tp2(tf, vp)
            )
            tp3 = entry_low - atr * rp.volatility_scaled_tp3(tf, vp)

        risk = (entry_low - stop_loss) if direction == "LONG" else (stop_loss - entry_high)
        if risk <= 0:
            return None
        rr_ratio = abs(tp2 - current_price) / risk

        # ── Confluence ────────────────────────────────────────────────────
        phase_labels = {
            WP.ACCUMULATION_C: "Phase C — Spring (last shake-out before markup)",
            WP.DISTRIBUTION_C: "Phase C — UTAD (last pump before markdown)",
            WP.ACCUMULATION_D: "Phase D — Sign of Strength (demand dominant)",
            WP.DISTRIBUTION_D: "Phase D — Sign of Weakness (supply dominant)",
        }
        confluence: List[str] = [
            f"✅ Wyckoff {phase_labels.get(phase, phase.value)}",
        ]
        if result.notes:
            for note in result.notes[:3]:
                confluence.append(f"   {note}")
        if result.volume_confirms:
            confluence.append("✅ Volume confirms the Wyckoff phase")
        if result.spring_detected:
            confluence.append("✅ Spring detected — stop hunt below range support")
        if result.utad_detected:
            confluence.append("✅ UTAD detected — trap above range resistance")
        confluence.append(f"📊 Regime: {regime} ({'+' if regime_bonus >= 0 else ''}{regime_bonus}) | TF: {tf}")
        confluence.append(f"🎯 R:R {rr_ratio:.2f} | ATR: {fmt_price(atr)}")

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
            analysis_timeframes=[tf],
            confluence=confluence,
            raw_data={
                "wyckoff_phase": phase.value,
                "spring_detected": result.spring_detected,
                "utad_detected": result.utad_detected,
                "volume_confirms": result.volume_confirms,
                "wyckoff_confidence": result.confidence,
                "key_level": result.key_level,
                "atr": atr,
                "regime": regime,
            },
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
WyckoffStrategy = WyckoffAccDist
