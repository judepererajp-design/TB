"""
TitanBot Pro — Harmonic Pattern Detector
==========================================
Detects XABCD harmonic patterns using Fibonacci ratios.
Patterns: Gartley, Bat, Crab, Butterfly, Shark, Cypher.

Harmonic patterns are precise Fibonacci-based reversal setups.
They have very specific ratio requirements — this makes them
high-probability when they meet all criteria.

Pattern completion (D point) = entry zone.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config.loader import cfg
from strategies.base import cfg_min_rr
from strategies.base import BaseStrategy, SignalResult, SignalDirection
from patterns._common import (
    find_alternating_pivots,
    regime_allows_structural,
    regime_penalty_for_pattern,
)

logger = logging.getLogger(__name__)


# Fibonacci ratios for each pattern
PATTERN_RATIOS = {
    'Gartley': {
        'XAB': (0.618,),
        'ABC': (0.382, 0.886),
        'BCD': (1.272, 1.618),
        'XAD': (0.786,),
    },
    'Bat': {
        'XAB': (0.382, 0.500),
        'ABC': (0.382, 0.886),
        'BCD': (1.618, 2.618),
        'XAD': (0.886,),
    },
    'Crab': {
        'XAB': (0.382, 0.618),
        'ABC': (0.382, 0.886),
        'BCD': (2.240, 3.618),
        'XAD': (1.618,),
    },
    'Butterfly': {
        'XAB': (0.786,),
        'ABC': (0.382, 0.886),
        'BCD': (1.618, 2.618),
        'XAD': (1.272, 1.618),
    },
    'Shark': {
        'XAB': (0.446, 0.618),
        'ABC': (1.130, 1.618),
        'BCD': (1.618, 2.240),
        'XAD': (0.886, 1.130),
    },
    # FIX HARMONIC-3: Cypher was enabled in config but missing from PATTERN_RATIOS
    # FIX AUDIT P1-H1: Prior BCD was (0.786,) which is not a Cypher at all — it
    # produced false matches on every shallow retracement. Classic Cypher has
    # BCD in (1.272, 2.000) and XAD ≈ 0.786; keep ABC at (1.130, 1.414).
    'Cypher': {
        'XAB': (0.382, 0.618),
        'ABC': (1.130, 1.414),
        'BCD': (1.272, 2.000),
        'XAD': (0.786,),
    },
}

# Per-pattern absolute tolerance (in Fib-ratio units). Relative tolerance is
# asymmetric — ±10% on 0.382 is ±0.038 (tight) but ±10% on 2.618 is ±0.262
# (huge), which over-accepts deep extensions. Absolute tolerance is uniform.
_PATTERN_ABS_TOL = {
    'Gartley':   0.05,
    'Bat':       0.05,
    'Crab':      0.08,   # larger extension → slightly looser
    'Butterfly': 0.06,
    'Shark':     0.06,
    'Cypher':    0.06,
}


@dataclass
class HarmonicResult:
    pattern:   str
    direction: str          # LONG | SHORT (bullish = long, bearish = short)
    x: float; a: float; b: float; c: float; d: float
    confidence: float
    entry_zone: Tuple[float, float]
    stop_loss:  float
    tp1: float; tp2: float


class HarmonicDetector(BaseStrategy):
    """
    Detects harmonic patterns in recent price data.
    Scans the last N bars for XABCD pivot formations.
    """

    name = "HarmonicPattern"
    description = "XABCD harmonic patterns: Gartley, Bat, Crab, Butterfly"

    # Harmonic patterns are structural — valid in trending and choppy regimes
    # but not in extreme panic where price structure is unreliable.
    VALID_REGIMES = {"BULL_TREND", "BEAR_TREND", "VOLATILE", "CHOPPY", "UNKNOWN"}

    def __init__(self):
        super().__init__()
        self._cfg       = cfg.patterns.harmonic
        self._tolerance = getattr(self._cfg, 'tolerance', 0.10)
        self._min_swing = getattr(self._cfg, 'min_swing_pct', 0.03)
        self._pivot_ord = getattr(self._cfg, 'pivot_order', 5)
        # P1-H5: minimum bar spacing between consecutive pivots — prevents
        # XABCD patterns that are just 3 bars apart from being treated as
        # equivalent to multi-week structures. Per-TF in analyze().
        self._min_pivot_spacing = getattr(self._cfg, 'min_pivot_spacing', 3)
        # P2-H1: PRZ-cluster bonus — when multiple Fib ratios from different
        # legs land in the same tight zone the reversal probability rises.
        self._prz_cluster_bonus = getattr(self._cfg, 'prz_cluster_bonus', 5.0)
        # P2-H2: confidence bump when a bullish/bearish candle prints AT D
        self._d_reversal_bonus = getattr(self._cfg, 'd_reversal_bonus', 4.0)

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime gate ───────────────────────────────────────────────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception:
            regime = "UNKNOWN"
        if regime not in self.VALID_REGIMES:
            return None

        # Use 4H for harmonic patterns
        tf = '4h'
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 80:
            # Fall back to 1H
            tf = '1h'
            if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 80:
                return None

        import pandas as pd
        ohlcv  = ohlcv_dict[tf]
        df     = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
        df     = df.astype(float)
        opens  = df['open'].values
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values

        atr    = self.calculate_atr(highs, lows, closes, 14)
        if atr == 0:
            return None

        # Find pivot points (returns list of (bar_idx, price, kind))
        # P2: delegate to shared helper — identical semantics, one place to fix bugs.
        pivots = find_alternating_pivots(highs, lows, self._pivot_ord)
        if len(pivots) < 5:
            return None

        # Scan recent pivots for harmonic patterns
        recent_pivots = pivots[-8:]
        current_price = float(closes[-1])
        # P1-H6: D-proximity is ATR-scaled, not a fixed 2%.
        # On a tight-ATR asset 2% is huge; on a wide-ATR asset 2% is tiny.
        d_proximity = max(atr * 0.5, current_price * 0.005)

        for i in range(len(recent_pivots) - 4):
            window = recent_pivots[i:i+5]
            prices = [p[1] for p in window]
            bars   = [p[0] for p in window]
            # P1-H5: require minimum bar spacing between each consecutive pivot
            if any(bars[j+1] - bars[j] < self._min_pivot_spacing
                   for j in range(len(bars) - 1)):
                continue
            x, a, b, c, d_pivot = prices
            result = self._check_all_patterns(x, a, b, c, d_pivot, atr)
            if result:
                # D point should be near current price (ATR-scaled proximity)
                if abs(current_price - result.d) <= d_proximity:
                    # P2-H1: PRZ-cluster bonus. When the XAD and BCD projection
                    # zones overlap within 0.5*ATR of D the confluence is
                    # stronger than either ratio alone.
                    prz_cluster = self._check_prz_cluster(
                        result.x, result.a, result.b, result.c, result.d, atr
                    )
                    # P2-H2: reversal-candle confirmation AT D
                    d_candle_ok = self._has_reversal_candle_at_d(
                        opens, highs, lows, closes, result.direction
                    )
                    if prz_cluster:
                        result.confidence = min(
                            95.0, float(result.confidence) + self._prz_cluster_bonus
                        )
                    if d_candle_ok:
                        result.confidence = min(
                            95.0, float(result.confidence) + self._d_reversal_bonus
                        )
                    # P2-R1: apply regime penalty; hard gate already above.
                    penalty = regime_penalty_for_pattern("harmonic", regime)
                    if penalty:
                        result.confidence = max(50.0, float(result.confidence) - penalty)

                    return self._build_signal(
                        symbol, result, atr, tf,
                        prz_cluster=prz_cluster,
                        d_candle_ok=d_candle_ok,
                        regime=regime,
                    )

        return None

    def _check_prz_cluster(self, x, a, b, c, d, atr: float) -> bool:
        """
        P2-H1: PRZ (Potential Reversal Zone) cluster check.

        Harmonic textbooks define the PRZ as the overlap of:
          * XAD Fib projection (completion level)
          * BCD Fib projection
          * Previous swing levels (structural S/R)

        When the XAD-projected D and BCD-projected D land within a 0.5-ATR
        window of each other (and of the actual pivot D), the reversal is
        more reliable.  We approximate this by computing both projection
        targets and comparing their distance to the actual D.
        """
        if atr <= 0:
            return False
        xa = abs(a - x)
        bc = abs(c - b)
        if xa == 0 or bc == 0:
            return False

        # Use 0.786 (Gartley/Cypher) and 1.272 (Butterfly) as representative
        # XAD projections; 1.618 (Bat/Crab) as representative BCD.  A real
        # cluster shows two of these in the same zone.
        direction_is_down = a < x  # bullish completion (X high → A low)
        sign = -1 if direction_is_down else 1

        xad_candidates = [x + sign * xa * 0.786, x + sign * xa * 1.272]
        bcd_candidates = [c + sign * bc * 1.272, c + sign * bc * 1.618]

        tol = atr * 0.5
        # Cluster exists if any xad candidate is within tolerance of any bcd
        # candidate AND both are within tolerance of the actual pivot D.
        for xa_t in xad_candidates:
            for bc_t in bcd_candidates:
                if abs(xa_t - bc_t) <= tol and abs(xa_t - d) <= tol:
                    return True
        return False

    def _has_reversal_candle_at_d(
        self, opens, highs, lows, closes, direction: str
    ) -> bool:
        """
        P2-H2: require a confirming candle at D.

        LONG harmonics complete on a bullish hammer/engulfing/morning-star
        signal; SHORT harmonics mirror.  Use patterns.candlestick to check.
        """
        try:
            from patterns.candlestick import (
                pin_bar, engulfing, morning_star, evening_star,
                piercing_line, dark_cloud_cover,
            )
        except Exception:  # pragma: no cover - defensive
            return False

        detectors_long = (pin_bar, engulfing, morning_star, piercing_line)
        detectors_short = (pin_bar, engulfing, evening_star, dark_cloud_cover)
        detectors = detectors_long if direction == "LONG" else detectors_short

        for fn in detectors:
            try:
                detected, d_dir, strength = fn(opens, highs, lows, closes)
            except Exception:
                continue
            if detected and strength > 0.25 and (
                d_dir == direction or d_dir == "NEUTRAL"
            ):
                return True
        return False

    # Legacy name kept for backward-compat — delegates to shared impl.
    def _find_pivots(self, highs, lows, order: int):
        """
        Deprecated wrapper; retained so external imports don't break.
        Use patterns._common.find_alternating_pivots directly.
        """
        return find_alternating_pivots(highs, lows, order)

    def _check_all_patterns(self, x, a, b, c, d, atr: float = 0) -> Optional[HarmonicResult]:
        """
        Check all enabled patterns and return the BEST match (lowest ratio error).
        Previously returned the first match, which biased toward whichever pattern
        came first in the name list when XAB windows overlapped (Gartley vs Bat).
        """
        enabled_patterns = []
        patterns_cfg = getattr(self._cfg, 'patterns', None)

        for name in ['Gartley', 'Bat', 'Crab', 'Butterfly', 'Shark', 'Cypher']:
            cfg_key = name.lower()
            # ConfigNode or dict-like; default True if absent
            if patterns_cfg is None:
                enabled_patterns.append(name)
            else:
                val = getattr(patterns_cfg, cfg_key, None)
                if val is None and hasattr(patterns_cfg, 'get'):
                    val = patterns_cfg.get(cfg_key, True)
                if val is None:
                    val = True
                if val:
                    enabled_patterns.append(name)

        best: Optional[HarmonicResult] = None
        best_err = float('inf')
        for pattern_name in enabled_patterns:
            result = self._check_pattern(pattern_name, x, a, b, c, d, atr)
            if result is None:
                continue
            # Higher confidence ⇔ lower error (see confidence formula). Use
            # (100 - confidence) as the error proxy for consistent comparison.
            err = 100.0 - float(result.confidence)
            if err < best_err:
                best_err = err
                best = result
        return best

    def _check_pattern(self, name, x, a, b, c, d, atr: float = 0) -> Optional[HarmonicResult]:
        """Validate XABCD against pattern ratios"""
        ratios = PATTERN_RATIOS.get(name)
        if not ratios:
            return None

        # Determine direction from X->A move
        is_bullish = a < x   # X high, A low = bullish (long at D)
        is_bearish = a > x

        if not (is_bullish or is_bearish):
            return None

        # P1-H7: monotonicity sanity check — with strict H/L alternation of
        # pivots enforced, bullish XABCD must satisfy x>a<b>c<d (highs and
        # lows alternate); bearish is the mirror. Reject silently otherwise.
        if is_bullish and not (x > a and a < b and b > c and c < d):
            return None
        if is_bearish and not (x < a and a > b and b < c and c > d):
            return None

        # P1-H4: prefer absolute per-pattern tolerance (relative tol over-accepts
        # large targets). Fall back to legacy relative tolerance if not listed.
        abs_tol = _PATTERN_ABS_TOL.get(name)
        rel_tol = self._tolerance

        def in_range(val, targets):
            # P-3 FIX: guard t <= 0. Absolute tolerance path preferred.
            for t in targets:
                if t <= 0:
                    continue
                if abs_tol is not None:
                    if abs(val - t) <= abs_tol:
                        return True
                else:
                    if abs(val - t) / t < rel_tol:
                        return True
            return False

        # Calculate ratios
        xa = abs(a - x)
        ab = abs(b - a)
        bc = abs(c - b)
        cd = abs(d - c)
        xd = abs(d - x)

        if xa == 0 or ab == 0 or bc == 0 or cd == 0:
            return None

        xab_ratio = ab / xa
        abc_ratio = bc / ab
        bcd_ratio = cd / bc
        xad_ratio = xd / xa

        checks = [
            in_range(xab_ratio, ratios['XAB']),
            in_range(abc_ratio, ratios['ABC']),
            in_range(bcd_ratio, ratios['BCD']),
            in_range(xad_ratio, ratios['XAD']),
        ]

        if not all(checks):
            return None

        # Calculate confidence based on how close ratios are to ideal
        errors    = []
        for ratio, targets in [
            (xab_ratio, ratios['XAB']),
            (abc_ratio, ratios['ABC']),
            (bcd_ratio, ratios['BCD']),
            (xad_ratio, ratios['XAD']),
        ]:
            positive_targets = [t for t in targets if t > 0]
            if not positive_targets:
                errors.append(0.0)
                continue
            best_target = min(
                positive_targets,
                key=lambda t: abs(ratio - t) / t,
            )
            errors.append(abs(ratio - best_target) / best_target if best_target > 0 else 0.0)

        avg_error   = float(np.mean(errors))
        # P1-H3: previously capped at 85 via `max(55, ...)` then `min(88, ...)`
        # at the caller — the 88 cap was dead. Widen the range to 60-92 so
        # textbook patterns with near-zero error can actually score highly.
        confidence  = max(60.0, min(92.0, 92.0 - avg_error * 250.0))

        direction   = "LONG" if is_bullish else "SHORT"
        # ATR-based SL. For shallow patterns (Gartley/Bat) X is just barely
        # past D, so SL = D ± k*ATR is sufficient. For extension patterns
        # (Crab/Butterfly) X is further away and SL should sit beyond X.
        _atr_val = atr if (atr is not None and atr > 0) else 0.0
        is_extension = name in ('Crab', 'Butterfly')
        if is_extension:
            # Place SL beyond X with a 0.2*ATR buffer (or 0.5% of price fallback).
            _buffer = _atr_val * 0.2 if _atr_val > 0 else d * 0.005
            stop_loss = (x - _buffer) if direction == "LONG" else (x + _buffer)
        else:
            _atr_sl_mult = 0.8 if name in ('Gartley', 'Bat') else 1.0
            if _atr_val > 0:
                _sl_dist = _atr_val * _atr_sl_mult
            else:
                _sl_dist = d * (0.03 if name in ('Gartley', 'Bat') else 0.04)
            stop_loss = (d - _sl_dist) if direction == "LONG" else (d + _sl_dist)

        # Entry zone aligned with D-proximity (ATR-scaled band ≈ 0.3 ATR)
        _ez_half = (_atr_val * 0.3) if _atr_val > 0 else d * 0.005
        entry_zone = (d - _ez_half, d + _ez_half)
        # Targets: classic harmonic uses Fib retracements of AD leg for more
        # realistic TPs. TP1 = 0.382 of AD retrace, TP2 = 0.618 of AD retrace.
        ad_leg = abs(d - a)
        if direction == "LONG":
            tp1 = d + ad_leg * 0.382
            tp2 = d + ad_leg * 0.618
        else:
            tp1 = d - ad_leg * 0.382
            tp2 = d - ad_leg * 0.618

        # Safety: guarantee TPs stay on the correct side of entry
        if direction == "LONG" and (tp1 <= d or tp2 <= d):
            return None
        if direction == "SHORT" and (tp1 >= d or tp2 >= d):
            return None

        return HarmonicResult(
            pattern=name, direction=direction,
            x=x, a=a, b=b, c=c, d=d,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            tp1=tp1, tp2=tp2
        )

    def _build_signal(
        self,
        symbol,
        result: HarmonicResult,
        atr,
        tf,
        *,
        prz_cluster: bool = False,
        d_candle_ok: bool = False,
        regime: Optional[str] = None,
    ) -> Optional[SignalResult]:
        current = result.d
        risk    = abs(current - result.stop_loss)
        if risk == 0:
            return None

        tp3     = result.tp2 + (result.tp2 - result.tp1) if result.direction == "LONG" \
                  else result.tp2 - (result.tp1 - result.tp2)
        rr      = abs(result.tp2 - current) / risk

        if rr < cfg_min_rr("swing"):
            return None

        confluence = [
            f"✅ {result.pattern} harmonic pattern at D point",
            f"✅ All 4 Fibonacci ratios validated",
            f"📐 XABCD: {result.x:.4f} → {result.a:.4f} → {result.b:.4f} → {result.c:.4f} → {result.d:.4f}",
        ]
        if prz_cluster:
            confluence.append("🎯 PRZ cluster — XAD/BCD projections overlap at D")
        if d_candle_ok:
            confluence.append("🕯 Confirming reversal candle at D")

        # P1-H3: confidence cap removed (was min(88, ...)). Use raw result value.
        return SignalResult(
            symbol=symbol,
            direction=SignalDirection.LONG if result.direction == "LONG" else SignalDirection.SHORT,
            strategy=f"{self.name}:{result.pattern}",
            confidence=float(result.confidence),
            entry_low=result.entry_zone[0], entry_high=result.entry_zone[1],
            stop_loss=result.stop_loss,
            tp1=result.tp1, tp2=result.tp2, tp3=tp3,
            rr_ratio=rr, setup_class="swing", analysis_timeframes=[tf],
            confluence=confluence,
            raw_data={
                'pattern': result.pattern,
                'harmonic_pattern': result.pattern,   # governance lineage
                'harmonic_tf': tf,
                'harmonic_direction': result.direction,
                'prz_cluster': bool(prz_cluster),
                'd_reversal_candle': bool(d_candle_ok),
                'regime_at_detection': regime,
                'x': result.x, 'a': result.a,
                'b': result.b, 'c': result.c, 'd': result.d,
            }
        )
