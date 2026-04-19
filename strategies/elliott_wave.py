"""
TitanBot Pro — Elliott Wave Strategy
=======================================
Simplified 5-wave Elliott count using local swing pivots.

Targets:
  - Wave 3 entry (after Wave 2 retracement completes)
  - Wave 5 entry (after Wave 4 retracement completes)

Fibonacci validation:
  - Wave 2: 38.2%, 50%, or 61.8% retracement of Wave 1
  - Wave 3: 161.8%, 200%, or 261.8% extension of Wave 1
  - Wave 4: 23.6% or 38.2% retracement of Wave 3
  - Wave 5: 61.8%, 100%, or 161.8% of Wave 1

SL at Wave 1 start (LONG) / Wave 1 high (SHORT) — wave invalidation level.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.loader import cfg
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp

logger = logging.getLogger(__name__)

_WAVE2_FIBS  = [0.382, 0.500, 0.618]
_WAVE3_FIBS  = [1.618, 2.000, 2.618]
_WAVE4_FIBS  = [0.236, 0.382]
_WAVE5_FIBS  = [0.618, 1.000, 1.618]


def _closest_fib(ratio: float, fibs: List[float]) -> Tuple[float, float]:
    """Return (closest_fib, distance_from_fib) for a given ratio."""
    best = min(fibs, key=lambda f: abs(ratio - f))
    return best, abs(ratio - best)


def _find_swing_pivots(highs: np.ndarray, lows: np.ndarray,
                        min_wave_pct: float = 0.03,
                        max_bars: int = 200) -> List[Tuple[int, float, str]]:
    """
    Identify significant swing highs/lows as wave pivot candidates.
    Returns list of (index, price, 'H'/'L') sorted by index ascending.
    """
    n = len(highs)
    limit = min(n, max_bars)
    start = n - limit

    pivots = []
    window = 3  # minimum bars on each side for a pivot

    for i in range(start + window, n - window):
        # Swing high — strict >: flat tops do NOT qualify as pivot highs.
        if all(highs[i] > highs[i - j] for j in range(1, window + 1)) \
                and all(highs[i] > highs[i + j] for j in range(1, window + 1)):
            pivots.append((i, float(highs[i]), "H"))
        # Swing low — strict <: flat bottoms do NOT qualify as pivot lows.
        elif all(lows[i] < lows[i - j] for j in range(1, window + 1)) \
                and all(lows[i] < lows[i + j] for j in range(1, window + 1)):
            pivots.append((i, float(lows[i]), "L"))

    # Filter: minimum wave size
    filtered = []
    for i in range(len(pivots)):
        if i == 0:
            filtered.append(pivots[i])
            continue
        prev_price = filtered[-1][1]
        curr_price = pivots[i][1]
        if abs(curr_price - prev_price) / prev_price >= min_wave_pct:
            filtered.append(pivots[i])

    return filtered


class ElliottWave(BaseStrategy):

    name = "ElliottWave"
    description = "Simplified 5-wave Elliott count targeting Wave 3 and Wave 5 entries"

    VALID_REGIMES = {"BULL_TREND", "BEAR_TREND"}

    # Direction-aware regime confidence: Elliott Wave 3/5 entries should
    # follow the macro trend.  Counter-trend wave counts are lower probability.
    _REGIME_CONF_WITH_TREND = {
        "BULL_TREND":  +6,
        "BEAR_TREND":  +6,
    }
    _REGIME_CONF_COUNTER_TREND = {
        "BULL_TREND":  -10,
        "BEAR_TREND":  -10,
    }

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.elliott_wave

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            # EW-6: warn (not debug) so unexpected exceptions surface in production logs.
            logger.warning(f"ElliottWave.analyze {symbol}: {e}")
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
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 60:
            return None

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values

        atr = self.calculate_atr(highs, lows, closes, period=14)
        if atr == 0:
            return None

        min_wave_pct  = getattr(self._cfg, "min_wave_pct", 0.03)
        fib_tolerance = getattr(self._cfg, "fibonacci_tolerance", 0.08)
        # EW-4: Wave 4 retracements are structurally tighter than Wave 2 — use a
        # stricter tolerance so the 23.6%/38.2% check is actually selective
        # (at 0.08 tolerance any retrace 15.6%–46.2% passes, which is too wide).
        fib_tolerance_w4 = getattr(self._cfg, "fibonacci_tolerance_w4", min(fib_tolerance, 0.04))
        max_wave_bars = getattr(self._cfg, "max_wave_bars", 200)
        confidence_base = getattr(self._cfg, "confidence_base", 72)

        pivots = _find_swing_pivots(highs, lows, min_wave_pct, max_wave_bars)

        if len(pivots) < 5:
            return None

        current_price = closes[-1]
        direction:   Optional[str] = None
        target_wave: Optional[int] = None
        entry_level: float = 0.0
        sl_level:    float = 0.0
        tp_proj:     float = 0.0
        confidence:  float = float(confidence_base)
        confluence:  List[str] = []
        raw_data:    Dict = {}

        # ── Try to identify Wave 3 entry (best setup) ─────────────────────
        # Pattern for LONG: L → H → L (W1 up, W2 retrace, enter Wave 3 up)
        # Last 3 pivots: p0=W1start(L), p1=W1end(H), p2=W2end(L)
        for i in range(len(pivots) - 3, -1, -1):
            if i + 2 >= len(pivots):
                continue
            p0, p1, p2 = pivots[i], pivots[i + 1], pivots[i + 2]

            # Bullish Wave 3 setup: L - H - L pattern (W1 up, W2 retrace down)
            if p0[2] == "L" and p1[2] == "H" and p2[2] == "L":
                w1_size    = p1[1] - p0[1]
                w2_retrace = p1[1] - p2[1]
                if w1_size <= 0:
                    continue
                w2_ratio = w2_retrace / w1_size
                best_fib, fib_dist = _closest_fib(w2_ratio, _WAVE2_FIBS)

                if fib_dist > fib_tolerance:
                    continue

                # EW-Q2: tighten proximity — 3 ATRs lets a trade be taken well
                # into Wave 3 already; 1.5 ATRs keeps entries near the W2 end.
                if abs(current_price - p2[1]) <= atr * 1.5:
                    direction   = "LONG"
                    target_wave = 3
                    entry_level = p2[1]
                    sl_level    = p0[1]  # Below Wave 1 start
                    # EW-Q5: most W3 waves extend to 200–261.8% of W1; use 200% as
                    # the primary target (tp2) so we're not capped at 161.8%.
                    # Store the 161.8% level for reference in raw_data.
                    tp_proj     = p2[1] + w1_size * 2.000

                    confidence_bonus = 15 if fib_dist < 0.03 else 8 if fib_dist < 0.05 else 0
                    confluence.append(f"✅ Bullish Wave 2 at {fmt_price(p2[1])} — retraced {w2_ratio:.1%} of W1")
                    confluence.append(f"   Closest Fib: {best_fib:.1%} | Distance: {fib_dist:.1%}")
                    confluence.append(f"   W1 size: {fmt_price(w1_size)} | W3 proj (200%): {fmt_price(tp_proj)}")
                    confidence += confidence_bonus
                    raw_data["wave1_start"] = p0[1]
                    raw_data["wave1_end"] = p1[1]
                    raw_data["wave2_end"] = p2[1]
                    raw_data["w2_fib"] = best_fib
                    raw_data["w3_proj"] = tp_proj
                    raw_data["w3_proj_162"] = p2[1] + w1_size * 1.618  # reference: confirmed W3 start level
                    break

            # Bearish Wave 3 setup: H - L - H pattern (W1 down, W2 retrace up)
            if p0[2] == "H" and p1[2] == "L" and p2[2] == "H":
                w1_size    = p0[1] - p1[1]
                w2_retrace = p2[1] - p1[1]
                if w1_size <= 0:
                    continue
                w2_ratio = w2_retrace / w1_size
                best_fib, fib_dist = _closest_fib(w2_ratio, _WAVE2_FIBS)

                if fib_dist > fib_tolerance:
                    continue

                if abs(current_price - p2[1]) <= atr * 1.5:
                    direction   = "SHORT"
                    target_wave = 3
                    entry_level = p2[1]
                    sl_level    = p0[1]

                    # EW-Q5: 200% extension as primary target; store 161.8% reference.
                    tp_proj = p2[1] - w1_size * 2.000

                    confidence_bonus = 15 if fib_dist < 0.03 else 8 if fib_dist < 0.05 else 0
                    confluence.append(f"✅ Bearish Wave 2 at {fmt_price(p2[1])} — retraced {w2_ratio:.1%} of W1")
                    confluence.append(f"   Closest Fib: {best_fib:.1%} | Distance: {fib_dist:.1%}")
                    confluence.append(f"   W1 size: {fmt_price(w1_size)} | W3 proj (200%): {fmt_price(tp_proj)}")
                    confidence += confidence_bonus
                    raw_data["wave1_start"] = p0[1]
                    raw_data["wave1_end"] = p1[1]
                    raw_data["wave2_end"] = p2[1]
                    raw_data["w2_fib"] = best_fib
                    # EW-1: SHORT was missing w3_proj — analytics reading it got None.
                    raw_data["w3_proj"] = tp_proj
                    raw_data["w3_proj_162"] = p2[1] - w1_size * 1.618
                    break

        # ── Try Wave 5 entry if no Wave 3 found ───────────────────────────
        if direction is None and len(pivots) >= 5:
            for i in range(len(pivots) - 5, -1, -1):
                if i + 4 >= len(pivots):
                    continue
                p0, p1, p2, p3, p4 = pivots[i:i + 5]

                # Bullish 5-wave: L H L H L (looking to enter at p4 for W5 up)
                if (p0[2] == "L" and p1[2] == "H" and p2[2] == "L"
                        and p3[2] == "H" and p4[2] == "L"):
                    w1_size    = p1[1] - p0[1]
                    w3_size    = p3[1] - p2[1]
                    w4_retrace = p3[1] - p4[1]
                    if w3_size <= 0:
                        continue
                    w4_ratio = w4_retrace / w3_size
                    best_fib, fib_dist = _closest_fib(w4_ratio, _WAVE4_FIBS)

                    if fib_dist > fib_tolerance_w4:
                        continue

                    # EW-Q4: Wave 4 must not enter Wave 1 territory (classic EW rule).
                    # For bullish W5: W4 low (p4[1]) must stay above W1 high (p1[1]).
                    if p4[1] < p1[1]:
                        continue

                    if abs(current_price - p4[1]) <= atr * 1.5:
                        direction   = "LONG"
                        target_wave = 5
                        entry_level = p4[1]
                        sl_level    = p0[1]
                        tp_proj     = p4[1] + w1_size * 1.0

                        conf_bonus = 10 if fib_dist < 0.03 else 5
                        confluence.append(f"✅ Wave 4 end at {fmt_price(p4[1])} — retraced {w4_ratio:.1%} of W3")
                        confluence.append(f"   W5 proj (100% of W1): {fmt_price(tp_proj)}")
                        confidence += conf_bonus
                        raw_data.update({"wave4_end": p4[1], "wave4_fib": best_fib,
                                         "w1_size": w1_size, "elliott_invalidation_level": p0[1]})
                        break

                # Bearish 5-wave: H L H L H
                if (p0[2] == "H" and p1[2] == "L" and p2[2] == "H"
                        and p3[2] == "L" and p4[2] == "H"):
                    w1_size    = p0[1] - p1[1]
                    w3_size    = p2[1] - p3[1]
                    w4_retrace = p4[1] - p3[1]
                    if w3_size <= 0:
                        continue
                    w4_ratio = w4_retrace / w3_size
                    best_fib, fib_dist = _closest_fib(w4_ratio, _WAVE4_FIBS)

                    if fib_dist > fib_tolerance_w4:
                        continue

                    # EW-Q4: Wave 4 must not enter Wave 1 territory (classic EW rule).
                    # For bearish W5: W4 high (p4[1]) must stay below W1 low (p1[1]).
                    if p4[1] > p1[1]:
                        continue

                    if abs(current_price - p4[1]) <= atr * 1.5:
                        direction   = "SHORT"
                        target_wave = 5
                        entry_level = p4[1]
                        sl_level    = p0[1]
                        tp_proj     = p4[1] - w1_size * 1.0

                        conf_bonus = 10 if fib_dist < 0.03 else 5
                        confluence.append(f"✅ Wave 4 end at {fmt_price(p4[1])} — retraced {w4_ratio:.1%} of W3")
                        confluence.append(f"   W5 proj (100% of W1): {fmt_price(tp_proj)}")
                        confidence += conf_bonus
                        raw_data.update({"wave4_end": p4[1], "wave4_fib": best_fib,
                                         "w1_size": w1_size, "elliott_invalidation_level": p0[1]})
                        break

        if direction is None:
            return None

        # ── Direction-aware regime confidence ─────────────────────────────
        _is_with_trend = (
            (direction == "LONG" and regime == "BULL_TREND") or
            (direction == "SHORT" and regime == "BEAR_TREND")
        )
        if _is_with_trend:
            confidence += self._REGIME_CONF_WITH_TREND.get(regime, 0)
        elif regime in ("BULL_TREND", "BEAR_TREND"):
            confidence += self._REGIME_CONF_COUNTER_TREND.get(regime, 0)

        # ── Entry zone / SL / TP ──────────────────────────────────────────
        if direction == "LONG":
            entry_low   = entry_level - atr * rp.entry_zone_tight
            entry_high  = entry_level + atr * rp.entry_zone_atr
            stop_loss   = sl_level - atr * 0.3
            tp1         = entry_high + (tp_proj - entry_level) * 0.5
            tp2         = tp_proj
            tp3         = entry_high + (tp_proj - entry_level) * 1.618
        else:
            entry_high  = entry_level + atr * rp.entry_zone_tight
            entry_low   = entry_level - atr * rp.entry_zone_atr
            stop_loss   = sl_level + atr * 0.3
            tp1         = entry_low - (entry_level - tp_proj) * 0.5
            tp2         = tp_proj
            tp3         = entry_low - (entry_level - tp_proj) * 1.618

        # Ensure TP ordering
        if direction == "LONG":
            tp1 = max(tp1, entry_high + atr * 0.5)
            tp2 = max(tp2, tp1 + atr * 0.3)
            tp3 = max(tp3, tp2 + atr * 0.3)
        else:
            tp1 = min(tp1, entry_low - atr * 0.5)
            tp2 = min(tp2, tp1 - atr * 0.3)
            tp3 = min(tp3, tp2 - atr * 0.3)

        rr_ratio = self.calculate_effective_rr(
            direction=direction,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            tp2=tp2,
        )
        if rr_ratio <= 0:
            return None

        raw_data["elliott_invalidation_level"] = sl_level
        raw_data.update({"target_wave": target_wave, "tp_proj": tp_proj, "atr": atr, "regime": regime})

        confluence.append(f"📊 Target: Wave {target_wave} | Regime: {regime} | TF: {tf}")
        confluence.append(f"🎯 R:R {rr_ratio:.2f} | Invalidation: {fmt_price(sl_level)}")

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
            raw_data=raw_data,
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
ElliottWaveStrategy = ElliottWave
