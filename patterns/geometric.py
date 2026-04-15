"""
TitanBot Pro — Geometric Pattern Detector
==========================================
Detects chart patterns using swing point analysis:
  - Bull/Bear Flags (continuation after impulse)
  - Ascending/Descending/Symmetrical Triangles
  - Head & Shoulders / Inverse H&S
  - Rising/Falling Wedges
  - Double Top / Double Bottom

These are higher-timeframe patterns (4H/1D) used for directional bias
and confluence stacking. Signals fire at breakout confirmation.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from config.loader import cfg
from strategies.base import cfg_min_rr
from strategies.base import BaseStrategy, SignalResult, SignalDirection
from utils.formatting import fmt_price
from utils.risk_params import rp

logger = logging.getLogger(__name__)


class GeometricPatterns(BaseStrategy):
    """Detects geometric chart patterns for confluence"""

    name = "GeometricPattern"
    description = "Flags, triangles, H&S, wedges"

    # Geometric patterns are structural — valid in trending and choppy regimes
    # but not in extreme panic where price structure breaks down.
    VALID_REGIMES = {"BULL_TREND", "BEAR_TREND", "VOLATILE", "CHOPPY", "UNKNOWN"}

    def __init__(self):
        super().__init__()
        self._cfg       = cfg.patterns.geometric
        self._threshold = getattr(self._cfg, 'confidence_threshold', 65)
        self._lookback  = getattr(self._cfg, 'lookback', 50)

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime gate ───────────────────────────────────────────────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception:
            regime = "UNKNOWN"
        if regime not in self.VALID_REGIMES:
            return None

        import pandas as pd

        tf = '4h'
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < self._lookback:
            tf = '1h'
            if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < self._lookback:
                return None

        ohlcv  = ohlcv_dict[tf]
        df     = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
        df     = df.astype(float)
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values

        atr    = self.calculate_atr(highs, lows, closes, 14)
        if atr == 0:
            return None

        # Stamp current timeframe so detectors can scale their windows
        self._current_tf = tf

        # Run all pattern detectors (ordered by reward potential)
        detectors = [
            self._detect_cup_and_handle,       # High reward: 3-5 R:R
            self._detect_head_and_shoulders,   # High reward: 2-4 R:R
            self._detect_wedge,                # High reward: 2-4 R:R
            self._detect_flag,                 # Good: 2-3 R:R
            self._detect_triangle,             # Good: 2-3 R:R
            self._detect_double_top_bottom,    # Good: 2-3 R:R
        ]

        for detector in detectors:
            result = detector(highs, lows, closes, atr, volumes)
            if result:
                direction, pattern_name, confidence, notes, entry, sl, tp1, tp2 = result
                if confidence < self._threshold:
                    continue

                risk = abs(entry - sl)
                rr   = abs(tp2 - entry) / risk if risk > 0 else 0

                if rr < cfg_min_rr("swing"):
                    continue

                # BUG-2 FIX: Clamp all price levels to a minimum of 0.1% of entry price.
                # On very low-price tokens (e.g. BONK at $0.00002), wedge_range can
                # exceed the current price itself, producing negative TP1/TP2/SL values.
                # Negative prices are physically impossible and cause absurd TP3 percentages
                # downstream (1,902,729,226% → 50% cap logs). The signal is invalid if any
                # computed level is non-positive; discard it rather than emit garbage geometry.
                _price_floor = entry * 0.001  # 0.1% of entry as absolute minimum
                if tp1 <= _price_floor or tp2 <= _price_floor or sl <= _price_floor:
                    continue  # Discard this signal — geometry is invalid for this token

                _tp3 = tp2 + (tp2 - tp1) if direction == "LONG" else tp2 - (tp1 - tp2)
                if _tp3 <= _price_floor:
                    _tp3 = None  # tp3 invalid — set to None rather than negative

                return SignalResult(
                    symbol=symbol,
                    direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
                    strategy=f"{self.name}:{pattern_name}",
                    confidence=min(88, confidence),
                    entry_low=entry - atr * rp.entry_zone_atr,
                    entry_high=entry + atr * rp.entry_zone_atr,
                    stop_loss=sl,
                    tp1=tp1, tp2=tp2, tp3=_tp3,
                    rr_ratio=rr, setup_class="swing", analysis_timeframes=[tf],
                    confluence=notes,
                    raw_data={'pattern': pattern_name}
                )

        return None

    def _detect_flag(self, highs, lows, closes, atr, volumes=None) -> Optional[tuple]:
        """
        Bull/Bear flag: sharp impulse followed by tight consolidation channel,
        then breakout in the direction of the impulse.
        """
        # Scale windows to timeframe: 4h needs fewer bars than 1h for same duration.
        # _tf is set by analyze() before calling detectors.
        _tf = getattr(self, '_current_tf', '4h')
        _tf_scale = {'1m':12,'3m':8,'5m':6,'15m':4,'30m':3,'1h':2,'2h':1,'4h':1,'1d':1}.get(_tf, 1)
        # Base: 25 bars impulse window, 10 bars consolidation (calibrated for 4h)
        # On 1h (scale=2): 50 bars impulse, 20 bars consolidation
        _impulse_bars = 25 * _tf_scale
        _consol_bars  = 10 * _tf_scale
        _min_bars     = _impulse_bars + 5

        if len(closes) < _min_bars:
            return None

        # Look for impulse in bars -_impulse_bars to -_consol_bars
        impulse_window = closes[-_impulse_bars:-_consol_bars]
        consol_window  = closes[-_consol_bars:]

        impulse_range  = max(impulse_window) - min(impulse_window)
        consol_range   = max(consol_window)  - min(consol_window)

        if impulse_range == 0:
            return None

        # Flag consolidation should be tight (<40% of impulse)
        if consol_range > impulse_range * 0.4:
            return None

        current = closes[-1]
        consol_high = max(highs[-10:])
        consol_low  = min(lows[-10:])

        # Bull flag: impulse was up, consolidation is sideways/slight down
        if closes[-10] > closes[-25]:  # Upward impulse
            # Breakout confirmation: current close above consolidation high
            # FIX GEO-1: was `current > consol_high - atr*0.2` which fired INSIDE the
            # consolidation (up to 0.2 ATR below the high). Changed to `+ atr*0.1`
            # so the breakout bar must actually close ABOVE the consolidation high.
            if current > consol_high + atr * 0.1:
                entry      = current
                stop_loss  = consol_low - atr * rp.sl_atr_mult * 0.4
                tp1        = entry + impulse_range * 0.6
                tp2        = entry + impulse_range
                notes      = [
                    f"✅ Bull flag breakout — impulse range {fmt_price(impulse_range)}",
                    f"✅ Tight consolidation ({consol_range/impulse_range*100:.0f}% of impulse)",
                ]
                return ("LONG", "BullFlag", 68, notes, entry, stop_loss, tp1, tp2)

        # Bear flag: impulse was down, consolidation sideways/slight up
        elif closes[-10] < closes[-25]:
            # FIX GEO-1: mirror fix for bear flag — must close BELOW consolidation low
            if current < consol_low - atr * 0.1:
                entry      = current
                stop_loss  = consol_high + atr * rp.sl_atr_mult * 0.4
                tp1        = entry - impulse_range * 0.6
                tp2        = entry - impulse_range
                notes      = [
                    f"✅ Bear flag breakdown — impulse range {fmt_price(impulse_range)}",
                    f"✅ Tight consolidation ({consol_range/impulse_range*100:.0f}% of impulse)",
                ]
                return ("SHORT", "BearFlag", 68, notes, entry, stop_loss, tp1, tp2)

        return None

    def _detect_triangle(self, highs, lows, closes, atr, volumes=None) -> Optional[tuple]:
        """
        Symmetrical triangle: converging highs and lows toward an apex.
        Breakout direction = trade direction.
        """
        lookback = min(self._lookback, len(closes))
        h        = highs[-lookback:]
        l        = lows[-lookback:]

        if len(h) < 20:
            return None

        # Linear regression on highs (descending) and lows (ascending) = symmetrical
        x       = np.arange(len(h))
        h_slope = np.polyfit(x, h, 1)[0]
        l_slope = np.polyfit(x, l, 1)[0]

        # FIX GEO-2: was `h_slope < -atr * 0.01` — ATR-based threshold breaks on
        # low-cap altcoins with tiny ATR (e.g. $0.00005 → threshold = 0.0000005,
        # nearly zero, triggers on any noise). Use percentage of current price instead.
        current_price = h[-1]
        slope_pct_threshold = current_price * 0.001 / max(1, len(h) // 4)  # 0.1% per quarter of lookback
        is_symmetrical = h_slope < -slope_pct_threshold and l_slope > slope_pct_threshold

        if not is_symmetrical:
            return None

        # Project apex
        h_line = np.polyfit(x, h, 1)
        l_line = np.polyfit(x, l, 1)

        # Current price relative to the triangle
        proj_high = np.polyval(h_line, len(h))
        proj_low  = np.polyval(l_line, len(h))

        current  = closes[-1]
        width    = proj_high - proj_low

        if width <= 0 or width > atr * 20:
            return None  # Not a valid triangle

        # Breakout: price outside triangle
        # SL: just beyond the opposite triangle boundary + 1 ATR buffer.
        # Previously used full proj_low/proj_high which locked RR at exactly 2.0
        # regardless of triangle size. ATR buffer makes SL proportional to volatility.
        if current > proj_high:
            entry     = current
            stop_loss = proj_low - atr * 0.5   # Below triangle low + buffer
            tp1       = current + width
            tp2       = current + width * 2
            notes     = [f"✅ Symmetrical triangle breakout (width: {fmt_price(width)})"]
            return ("LONG", "SymTriangle", 66, notes, entry, stop_loss, tp1, tp2)

        elif current < proj_low:
            entry     = current
            stop_loss = proj_high + atr * 0.5  # Above triangle high + buffer
            tp1       = current - width
            tp2       = current - width * 2
            notes     = [f"✅ Symmetrical triangle breakdown (width: {fmt_price(width)})"]
            return ("SHORT", "SymTriangle", 66, notes, entry, stop_loss, tp1, tp2)

        return None

    def _detect_double_top_bottom(self, highs, lows, closes, atr, volumes=None) -> Optional[tuple]:
        """
        Double top (bearish) / Double bottom (bullish).
        Two equal highs or lows with a neckline.
        """
        if len(closes) < 30:
            return None

        tolerance = atr * 1.5   # Tops/bottoms must be within 1.5 ATR of each other

        # Find swing highs for double top
        swing_highs = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append((i, highs[i]))

        # Double top: last two swing highs are approximately equal
        if len(swing_highs) >= 2:
            h1_idx, h1_val = swing_highs[-2]
            h2_idx, h2_val = swing_highs[-1]
            if abs(h1_val - h2_val) < tolerance and h2_idx - h1_idx > 5:
                # Neckline = low between the two tops
                neckline = min(lows[h1_idx:h2_idx])
                current  = closes[-1]

                if current < neckline:  # Breakdown below neckline
                    entry     = current
                    stop_loss = h2_val + atr * rp.sl_atr_mult * 0.25
                    height    = (h1_val + h2_val) / 2 - neckline
                    tp1       = neckline - height * 0.5
                    tp2       = neckline - height
                    notes     = [
                        f"✅ Double Top breakdown below neckline {fmt_price(neckline)}",
                        f"✅ Both tops at ~{fmt_price((h1_val+h2_val)/2)} (within {tolerance:,.4f})",
                    ]
                    return ("SHORT", "DoubleTop", 72, notes, entry, stop_loss, tp1, tp2)

        # Find swing lows for double bottom
        swing_lows = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append((i, lows[i]))

        if len(swing_lows) >= 2:
            l1_idx, l1_val = swing_lows[-2]
            l2_idx, l2_val = swing_lows[-1]
            if abs(l1_val - l2_val) < tolerance and l2_idx - l1_idx > 5:
                neckline = max(highs[l1_idx:l2_idx])
                current  = closes[-1]

                if current > neckline:  # Breakout above neckline
                    entry     = current
                    stop_loss = l2_val - atr * rp.sl_atr_mult * 0.25
                    height    = neckline - (l1_val + l2_val) / 2
                    tp1       = neckline + height * 0.5
                    tp2       = neckline + height
                    notes     = [
                        f"✅ Double Bottom breakout above neckline {fmt_price(neckline)}",
                        f"✅ Both bottoms at ~{fmt_price((l1_val+l2_val)/2)}",
                    ]
                    return ("LONG", "DoubleBottom", 72, notes, entry, stop_loss, tp1, tp2)

        return None

    # ── HIGH REWARD PATTERNS ───────────────────────────────

    def _detect_cup_and_handle(self, highs, lows, closes, atr, volumes=None) -> Optional[tuple]:
        """
        Cup and Handle — one of the highest reward patterns (3-5 R:R).
        
        Structure:
          1. Left rim: local high
          2. Cup: U-shaped rounded bottom (20-40 bars)
          3. Right rim: price returns to left rim level
          4. Handle: small pullback (5-15 bars, <50% of cup depth)
          5. Breakout above rim = entry
          
        Volume: decreasing during cup, increasing on breakout
        """
        if len(closes) < 40:
            return None

        # Find potential left rim (highest point in first half)
        mid = len(closes) // 2
        left_rim_idx = np.argmax(highs[:mid])
        left_rim = highs[left_rim_idx]

        # Cup bottom: lowest point between left rim and recent bars
        cup_zone = lows[left_rim_idx + 3:-8]
        if len(cup_zone) < 10:
            return None
        cup_bottom_rel = np.argmin(cup_zone)
        cup_bottom = cup_zone[cup_bottom_rel]
        cup_depth = left_rim - cup_bottom

        if cup_depth < atr * 2:
            return None  # Cup too shallow

        # Right rim: price should recover to near left rim level
        recent_high = max(highs[-15:-5])
        rim_tolerance = cup_depth * 0.15
        if abs(recent_high - left_rim) > rim_tolerance:
            return None  # Right rim doesn't match left rim

        # Handle: small pullback in last 5-12 bars
        handle_low = min(lows[-10:])
        handle_depth = recent_high - handle_low
        if handle_depth > cup_depth * 0.50:
            return None  # Handle too deep (>50% of cup)
        if handle_depth < atr * 0.3:
            return None  # Handle too shallow

        # Breakout confirmation
        current = closes[-1]
        rim_level = (left_rim + recent_high) / 2

        if current < rim_level - atr * 0.3:
            return None  # Not breaking out yet

        # Volume confirmation: breakout bar should have higher volume
        confidence = 75
        if volumes is not None and len(volumes) > 20:
            avg_vol = np.mean(volumes[-20:-1])
            if volumes[-1] > avg_vol * 1.5:
                confidence += 5  # Volume confirms breakout

        entry = current
        stop_loss = handle_low - atr * rp.sl_atr_mult * 0.3
        # Cup & handle target = cup depth projected above rim
        tp1 = rim_level + cup_depth * 0.6
        tp2 = rim_level + cup_depth       # Full measured move
        notes = [
            f"🏆 Cup & Handle breakout — cup depth {fmt_price(cup_depth)}",
            f"✅ Handle pullback {handle_depth/cup_depth*100:.0f}% of cup (healthy)",
            f"✅ Rim level: {fmt_price(rim_level)}",
        ]
        return ("LONG", "CupAndHandle", confidence, notes, entry, stop_loss, tp1, tp2)

    def _detect_head_and_shoulders(self, highs, lows, closes, atr, volumes=None) -> Optional[tuple]:
        """
        Head & Shoulders (bearish) + Inverse H&S (bullish).
        High reward: neckline-to-head distance projected as target.
        
        Structure:
          Left shoulder → Head (higher high) → Right shoulder (lower high) → Neckline break
        """
        if len(closes) < 35:
            return None

        # Find 3 major swing highs for regular H&S
        # Use a simple approach: divide into 3 zones
        third = len(closes) // 3
        z1_highs = highs[:third]
        z2_highs = highs[third:third*2]
        z3_highs = highs[third*2:]

        ls_val = max(z1_highs)  # Left shoulder
        head_val = max(z2_highs)  # Head
        rs_val = max(z3_highs)  # Right shoulder

        # Regular H&S: Head must be highest, shoulders roughly equal
        if head_val > ls_val and head_val > rs_val:
            shoulder_diff = abs(ls_val - rs_val)
            if shoulder_diff < atr * 2:  # Shoulders roughly same height
                # Neckline: connect lows between shoulders
                z1_low = min(lows[third-5:third+5]) if third > 5 else min(lows[:third])
                z2_low = min(lows[third*2-5:third*2+5]) if third*2 > 5 else min(lows[third:third*2])
                neckline = (z1_low + z2_low) / 2

                current = closes[-1]
                head_to_neckline = head_val - neckline

                if head_to_neckline < atr * 2:
                    pass  # Pattern too small
                elif current < neckline + atr * 0.3:
                    # Breaking or below neckline
                    confidence = 73
                    if volumes is not None and len(volumes) > 10:
                        if volumes[-1] > np.mean(volumes[-10:]) * 1.3:
                            confidence += 5

                    entry = current
                    stop_loss = rs_val + atr * rp.sl_atr_mult * 0.3
                    tp1 = neckline - head_to_neckline * 0.5
                    tp2 = neckline - head_to_neckline  # Full measured move
                    notes = [
                        f"🏆 Head & Shoulders — neckline {fmt_price(neckline)}",
                        f"✅ Head: {fmt_price(head_val)} | Shoulders: ~{fmt_price((ls_val+rs_val)/2)}",
                        f"✅ Measured target: {fmt_price(tp2)}",
                    ]
                    return ("SHORT", "HeadAndShoulders", confidence, notes, entry, stop_loss, tp1, tp2)

        # Inverse H&S (bullish): find 3 swing lows
        z1_lows = lows[:third]
        z2_lows = lows[third:third*2]
        z3_lows = lows[third*2:]

        ls_low = min(z1_lows)
        head_low = min(z2_lows)
        rs_low = min(z3_lows)

        if head_low < ls_low and head_low < rs_low:
            shoulder_diff = abs(ls_low - rs_low)
            if shoulder_diff < atr * 2:
                z1_high = max(highs[third-5:third+5]) if third > 5 else max(highs[:third])
                z2_high = max(highs[third*2-5:third*2+5]) if third*2 > 5 else max(highs[third:third*2])
                neckline = (z1_high + z2_high) / 2

                current = closes[-1]
                neckline_to_head = neckline - head_low

                if neckline_to_head < atr * 2:
                    pass
                elif current > neckline - atr * 0.3:
                    confidence = 73
                    if volumes is not None and len(volumes) > 10:
                        if volumes[-1] > np.mean(volumes[-10:]) * 1.3:
                            confidence += 5

                    entry = current
                    stop_loss = rs_low - atr * rp.sl_atr_mult * 0.3
                    tp1 = neckline + neckline_to_head * 0.5
                    tp2 = neckline + neckline_to_head
                    notes = [
                        f"🏆 Inverse Head & Shoulders — neckline {fmt_price(neckline)}",
                        f"✅ Head: {fmt_price(head_low)} | Shoulders: ~{fmt_price((ls_low+rs_low)/2)}",
                        f"✅ Measured target: {fmt_price(tp2)}",
                    ]
                    return ("LONG", "InverseHS", confidence, notes, entry, stop_loss, tp1, tp2)

        return None

    def _detect_wedge(self, highs, lows, closes, atr, volumes=None) -> Optional[tuple]:
        """
        Rising Wedge (bearish) + Falling Wedge (bullish).
        Both trendlines converge — breakout opposite to wedge direction.
        High reward because wedges often produce sharp reversals.
        """
        if len(closes) < 25:
            return None

        # Fit trendlines to recent highs and lows
        n = min(25, len(closes))
        x = np.arange(n)
        recent_highs = highs[-n:]
        recent_lows = lows[-n:]

        try:
            high_slope = np.polyfit(x, recent_highs, 1)[0]
            low_slope = np.polyfit(x, recent_lows, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Both slopes must be same direction (converging)
        # Rising wedge: both up, but lows rising faster (converging)
        # Falling wedge: both down, but highs falling faster

        current = closes[-1]
        wedge_range = max(recent_highs) - min(recent_lows)

        if wedge_range < atr * 2:
            return None  # Too tight

        # Rising wedge (bearish): both slopes positive, converging
        if high_slope > 0 and low_slope > 0 and low_slope > high_slope * 0.5:
            # Check convergence: range is narrowing
            first_range = recent_highs[0] - recent_lows[0]
            last_range = recent_highs[-1] - recent_lows[-1]
            if last_range < first_range * 0.75:  # Converging
                # FIX GEO-3: was `current < lower_tl + atr * 0.2` which fired when price
                # was INSIDE the wedge (up to 0.2 ATR ABOVE the lower trendline). This
                # produced 121 false SHORT signals in a single session — every coin
                # consolidating near the bottom of an upward channel triggered it.
                # The pattern is only valid on CONFIRMED BREAKDOWN: close must be
                # BELOW the trendline by at least 0.3 ATR. This matches the same
                # breakout confirmation logic used by BullFlag (GEO-1 fix).
                lower_tl = recent_lows[-1]
                if current < lower_tl - atr * 0.3:
                    confidence = 70
                    entry = current
                    stop_loss = max(recent_highs[-5:]) + atr * rp.sl_atr_mult * 0.25
                    tp1 = entry - wedge_range * 0.5
                    tp2 = entry - wedge_range * 0.8
                    notes = [
                        f"🏆 Rising Wedge breakdown — bearish reversal",
                        f"✅ Converging trendlines (range narrowed {last_range/first_range*100:.0f}%)",
                        f"✅ Confirmed close {fmt_price(lower_tl - current):.6f} below lower TL",
                    ]
                    return ("SHORT", "RisingWedge", confidence, notes, entry, stop_loss, tp1, tp2)

        # Falling wedge (bullish): both slopes negative, converging
        if high_slope < 0 and low_slope < 0 and high_slope < low_slope * 0.5:
            first_range = recent_highs[0] - recent_lows[0]
            last_range = recent_highs[-1] - recent_lows[-1]
            if last_range < first_range * 0.75:
                # FIX GEO-3 (mirror): was `current > upper_tl - atr * 0.2` which fired
                # when price was still INSIDE the wedge. Match the confirmed-breakout
                # standard: close must be ABOVE upper trendline by at least 0.3 ATR.
                upper_tl = recent_highs[-1]
                if current > upper_tl + atr * 0.3:
                    confidence = 70
                    entry = current
                    stop_loss = min(recent_lows[-5:]) - atr * rp.sl_atr_mult * 0.25
                    tp1 = entry + wedge_range * 0.5
                    tp2 = entry + wedge_range * 0.8
                    notes = [
                        f"🏆 Falling Wedge breakout — bullish reversal",
                        f"✅ Converging trendlines (range narrowed {last_range/first_range*100:.0f}%)",
                        f"✅ Confirmed close {fmt_price(current - upper_tl):.6f} above upper TL",
                    ]
                    return ("LONG", "FallingWedge", confidence, notes, entry, stop_loss, tp1, tp2)

        return None
