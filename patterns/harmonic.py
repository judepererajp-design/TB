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
    # Cypher: more relaxed ratios, common in crypto trending markets
    'Cypher': {
        'XAB': (0.382, 0.618),
        'ABC': (1.130, 1.414),
        'BCD': (0.786,),
        'XAD': (0.786,),
    },
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
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values

        atr    = self.calculate_atr(highs, lows, closes, 14)
        if atr == 0:
            return None

        # Find pivot points
        pivots = self._find_pivots(highs, lows, self._pivot_ord)
        if len(pivots) < 5:
            return None

        # Scan recent pivots for harmonic patterns
        recent_pivots = pivots[-8:]
        for i in range(len(recent_pivots) - 4):
            x, a, b, c, d_pivot = recent_pivots[i:i+5]
            # FIX AUDIT-6: Pass atr to _check_all_patterns so _check_pattern
            # uses the current symbol's ATR, not a stale value from a previous call.
            result = self._check_all_patterns(x, a, b, c, d_pivot, atr)
            if result:
                current_price = closes[-1]
                # D point should be close to current price
                if abs(current_price - result.d) / max(result.d, 1e-10) < 0.02:
                    return self._build_signal(symbol, result, atr, tf)

        return None

    def _find_pivots(self, highs, lows, order: int) -> List[float]:
        """
        Find alternating high/low pivot points.
        Returns list of pivot prices (strictly alternating H/L).

        FIX HARMONIC-1: Old version appended raw floats with no H/L tag.
        Non-alternating sequences (e.g. two consecutive highs in a plateau)
        caused _check_pattern to misidentify direction: is_bullish = a < x
        assumed X=high,A=low but could get X=high,A=high instead.

        Fix: tag each pivot as ('H', price) or ('L', price), enforce strict
        alternation by only keeping a new pivot if its type differs from the
        last appended, then strip tags before returning.
        """
        tagged = []   # List of ('H'|'L', price)
        n = len(highs)

        for i in range(order, n - order):
            is_swing_high = (
                all(highs[i] > highs[i-j] for j in range(1, order+1)) and
                all(highs[i] > highs[i+j] for j in range(1, order+1))
            )
            is_swing_low = (
                all(lows[i] < lows[i-j] for j in range(1, order+1)) and
                all(lows[i] < lows[i+j] for j in range(1, order+1))
            )

            if is_swing_high:
                if not tagged or tagged[-1][0] != 'H':
                    tagged.append(('H', float(highs[i])))
                elif highs[i] > tagged[-1][1]:
                    # Replace previous high with the more extreme one (cluster merge)
                    tagged[-1] = ('H', float(highs[i]))
            elif is_swing_low:
                if not tagged or tagged[-1][0] != 'L':
                    tagged.append(('L', float(lows[i])))
                elif lows[i] < tagged[-1][1]:
                    tagged[-1] = ('L', float(lows[i]))

        # Return raw prices — alternation is now guaranteed by construction
        return [price for _, price in tagged]

    def _check_all_patterns(self, x, a, b, c, d, atr: float = 0) -> Optional[HarmonicResult]:
        """Check all enabled patterns for XABCD points"""
        enabled_patterns = []
        patterns_cfg = getattr(self._cfg, 'patterns', {})

        for name in ['Gartley', 'Bat', 'Crab', 'Butterfly', 'Shark', 'Cypher']:
            cfg_key = name.lower()
            if getattr(patterns_cfg, cfg_key, True):
                enabled_patterns.append(name)

        for pattern_name in enabled_patterns:
            result = self._check_pattern(pattern_name, x, a, b, c, d, atr)
            if result:
                return result

        return None

    def _check_pattern(self, name, x, a, b, c, d, atr: float = 0) -> Optional[HarmonicResult]:
        """Validate XABCD against pattern ratios"""
        ratios = PATTERN_RATIOS.get(name)
        if not ratios:
            return None

        tol = self._tolerance

        # Determine direction from X->A move
        is_bullish = a < x   # X high, A low = bullish (long at D)
        is_bearish = a > x

        if not (is_bullish or is_bearish):
            return None

        def in_range(val, targets):
            # P-3 FIX: guard against t <= 0 before dividing; a zero or negative
            # target ratio (possible on near-zero price assets or data corruption)
            # would produce ZeroDivisionError or a negative comparison that always
            # passes, silently injecting a false harmonic match into trade decisions.
            return any(t > 0 and abs(val - t) / t < tol for t in targets)

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
            best_target = min(
                (t for t in targets if t > 0),
                key=lambda t: abs(ratio - t) / t,
                default=targets[0] if targets else 1.0,
            )
            errors.append(abs(ratio - best_target) / best_target if best_target > 0 else 0.0)

        avg_error   = np.mean(errors)
        confidence  = max(55, 85 - avg_error * 200)

        direction   = "LONG" if is_bullish else "SHORT"
        # ATR-based SL instead of flat percentage.
        # Flat % (3-4%) was too wide on BTC ($2400-3200 on $80k) and
        # too tight on volatile alts. ATR respects actual volatility.
        # Gartley/Bat complete at tighter Fib levels → smaller ATR buffer.
        # Crab/Butterfly/Shark/Cypher extend further → wider buffer needed.
        # These are set by the caller (_build_signal passes atr).
        # Fallback to flat % if atr not available (shouldn't happen in practice).
        _atr_sl_mult = 0.8 if name in ('Gartley', 'Bat') else 1.2
        # FIX AUDIT-6: Use the atr parameter passed from the caller instead of
        # stale self._last_atr (which was only set in _build_signal, AFTER this
        # method returns — meaning _check_pattern always used ATR from the
        # previous symbol/call, or 0 on the first call).
        _atr_val = atr
        if _atr_val > 0:
            _sl_dist = _atr_val * _atr_sl_mult
        else:
            _stop_pct = 0.03 if name in ('Gartley', 'Bat') else 0.04
            _sl_dist = d * _stop_pct

        if direction == "LONG":
            entry_zone = (d * (1 - 0.005), d * (1 + 0.005))
            stop_loss  = d - _sl_dist
            tp1        = c
            tp2        = a
        else:
            entry_zone = (d * (1 - 0.005), d * (1 + 0.005))
            stop_loss  = d + _sl_dist
            tp1        = c
            tp2        = a

        return HarmonicResult(
            pattern=name, direction=direction,
            x=x, a=a, b=b, c=c, d=d,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            tp1=tp1, tp2=tp2
        )

    def _build_signal(self, symbol, result: HarmonicResult, atr, tf) -> Optional[SignalResult]:
        # Store ATR so _check_pattern can use it for ATR-based SL.
        # This is set before the result is used so _last_atr is always fresh.
        self._last_atr = atr
        current = result.d
        risk    = abs(current - result.stop_loss)
        if risk == 0:
            return None

        tp3     = result.tp2 + (result.tp2 - result.tp1) if result.direction == "LONG" \
                  else result.tp2 - (result.tp1 - result.tp2)
        rr      = abs(result.tp2 - current) / risk

        if rr < cfg_min_rr("swing"):
            return None

        return SignalResult(
            symbol=symbol,
            direction=SignalDirection.LONG if result.direction == "LONG" else SignalDirection.SHORT,
            strategy=f"{self.name}:{result.pattern}",
            confidence=min(88, result.confidence),
            entry_low=result.entry_zone[0], entry_high=result.entry_zone[1],
            stop_loss=result.stop_loss,
            tp1=result.tp1, tp2=result.tp2, tp3=tp3,
            rr_ratio=rr, setup_class="swing", analysis_timeframes=[tf],
            confluence=[
                f"✅ {result.pattern} harmonic pattern at D point",
                f"✅ All 4 Fibonacci ratios validated",
                f"📐 XABCD: {result.x:.4f} → {result.a:.4f} → {result.b:.4f} → {result.c:.4f} → {result.d:.4f}",
            ],
            raw_data={'pattern': result.pattern, 'x': result.x, 'a': result.a,
                      'b': result.b, 'c': result.c, 'd': result.d}
        )
