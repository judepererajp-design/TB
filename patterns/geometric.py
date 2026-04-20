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
        # P1-G13: sub-pattern enable/disable booleans. Previously these YAML
        # keys existed but nothing in code honoured them.
        _sub = getattr(self._cfg, 'patterns', None)
        self._enabled_subs = {
            'flags':             self._cfg_bool(_sub, 'flags', True),
            'triangles':         self._cfg_bool(_sub, 'triangles', True),
            'head_shoulders':    self._cfg_bool(_sub, 'head_shoulders', True),
            'double_top_bottom': self._cfg_bool(_sub, 'double_top_bottom', True),
            'wedges':            self._cfg_bool(_sub, 'wedges', True),
            'cup_and_handle':    self._cfg_bool(_sub, 'cup_and_handle', True),
        }
        # P1-G15: flag consolidation tightness knob (default 0.30, was hard 0.40)
        self._flag_consol_ratio = getattr(self._cfg, 'flag_consol_ratio', 0.30)

    @staticmethod
    def _cfg_bool(sub, key: str, default: bool) -> bool:
        if sub is None:
            return default
        val = getattr(sub, key, None)
        if val is None and hasattr(sub, 'get'):
            val = sub.get(key, default)
        return bool(val) if val is not None else default

    @staticmethod
    def _clamp_projection(raw: float, entry: float, atr: float) -> float:
        """
        BUG-2 ROOT-CAUSE FIX: bound pattern-projected distances (wedge/flag/
        triangle measured moves) so that when wedge_range > entry (low-priced
        tokens) projections don't push TP1/TP2/SL into negative price territory.
        Cap at the smaller of the raw projection or half the entry price, with
        an ATR floor to keep very low-vol/low-price assets meaningful.
        """
        if entry <= 0:
            return max(raw, 0.0)
        cap = max(entry * 0.5, atr * 3.0 if atr > 0 else entry * 0.1)
        return min(max(raw, 0.0), cap)

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

        # P1-G1: pass `tf` explicitly to every detector — no instance-state.
        # Run all pattern detectors (ordered by reward potential), skipping
        # any disabled in config.
        detectors = []
        if self._enabled_subs['cup_and_handle']:
            detectors.append(('cup_and_handle',    self._detect_cup_and_handle))
        if self._enabled_subs['head_shoulders']:
            detectors.append(('head_shoulders',    self._detect_head_and_shoulders))
        if self._enabled_subs['wedges']:
            detectors.append(('wedges',            self._detect_wedge))
        if self._enabled_subs['flags']:
            detectors.append(('flags',             self._detect_flag))
        if self._enabled_subs['triangles']:
            detectors.append(('triangles',         self._detect_triangle))
        if self._enabled_subs['double_top_bottom']:
            detectors.append(('double_top_bottom', self._detect_double_top_bottom))

        for sub_name, detector in detectors:
            result = detector(highs, lows, closes, atr, volumes, tf)
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
                # Root-cause fix is _clamp_projection used inside detectors; this
                # remains as a defense-in-depth guard.
                _price_floor = entry * 0.001  # 0.1% of entry as absolute minimum
                if tp1 <= _price_floor or tp2 <= _price_floor or sl <= _price_floor:
                    continue  # Discard this signal — geometry is invalid for this token

                _tp3 = tp2 + (tp2 - tp1) if direction == "LONG" else tp2 - (tp1 - tp2)
                if _tp3 <= _price_floor:
                    _tp3 = None  # tp3 invalid — set to None rather than negative

                # P1-G16: raise the confidence ceiling from 88 to 95 so strong
                # confluence (cup+volume+MTF) can actually outrank marginal setups.
                _conf = min(95.0, float(confidence))

                return SignalResult(
                    symbol=symbol,
                    direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
                    strategy=f"{self.name}:{pattern_name}",
                    confidence=_conf,
                    entry_low=entry - atr * rp.entry_zone_atr,
                    entry_high=entry + atr * rp.entry_zone_atr,
                    stop_loss=sl,
                    tp1=tp1, tp2=tp2, tp3=_tp3,
                    rr_ratio=rr, setup_class="swing", analysis_timeframes=[tf],
                    confluence=notes,
                    raw_data={
                        'pattern': pattern_name,
                        'geometric_pattern': pattern_name,   # governance lineage
                        'geometric_sub': sub_name,
                        'geometric_tf': tf,
                    }
                )

        return None

    def _detect_flag(self, highs, lows, closes, atr, volumes=None, tf='4h') -> Optional[tuple]:
        """
        Bull/Bear flag: sharp impulse followed by tight consolidation channel,
        then breakout in the direction of the impulse.
        """
        # Scale windows to timeframe: 4h needs fewer bars than 1h for same duration.
        # P1-G14: trimmed _tf_scale to the TFs analyze() actually passes (4h, 1h).
        _tf_scale = {'1h': 2, '4h': 1}.get(tf, 1)
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

        # P1-G15: configurable consolidation tightness (default 0.30, was 0.40)
        if consol_range > impulse_range * self._flag_consol_ratio:
            return None

        current = closes[-1]
        # P-2 FIX: when the series is shorter than 10 bars, highs[-10:] / lows[-10:]
        # silently use the entire series (Python negative-index slicing behaviour),
        # producing a much wider consolidation range than intended and generating
        # false breakout signals during warmup.  Guard explicitly.
        _consol_lookback = min(_consol_bars, len(highs))
        consol_high = max(highs[-_consol_lookback:])
        consol_low  = min(lows[-_consol_lookback:])
        _impulse_lookback = min(_impulse_bars, len(closes) - 1)
        if _impulse_lookback < 10:
            return None  # Not enough history to determine impulse direction

        # P1-G16: use means of the two windows, not two single bars, for impulse
        # direction — a single wick no longer flips the flag direction.
        impulse_mean = float(np.mean(impulse_window))
        consol_mean  = float(np.mean(consol_window))

        # Bull flag: consolidation mean above impulse mean + 0.5 ATR = upward impulse
        if consol_mean > impulse_mean + atr * 0.5:
            # Breakout confirmation: close above consolidation high + 0.1 ATR
            if current > consol_high + atr * 0.1:
                entry      = current
                stop_loss  = consol_low - atr * rp.sl_atr_mult * 0.4
                # P1-G9: clamp projections so low-price tokens don't produce
                # negative TPs; preserve original measured-move behavior otherwise.
                tp1_dist   = self._clamp_projection(impulse_range * 0.6, entry, atr)
                tp2_dist   = self._clamp_projection(impulse_range,       entry, atr)
                tp1        = entry + tp1_dist
                tp2        = entry + tp2_dist
                notes      = [
                    f"✅ Bull flag breakout — impulse range {fmt_price(impulse_range)}",
                    f"✅ Tight consolidation ({consol_range/impulse_range*100:.0f}% of impulse)",
                ]
                return ("LONG", "BullFlag", 68, notes, entry, stop_loss, tp1, tp2)

        # Bear flag: consolidation mean below impulse mean - 0.5 ATR
        elif consol_mean < impulse_mean - atr * 0.5:
            if current < consol_low - atr * 0.1:
                entry      = current
                stop_loss  = consol_high + atr * rp.sl_atr_mult * 0.4
                tp1_dist   = self._clamp_projection(impulse_range * 0.6, entry, atr)
                tp2_dist   = self._clamp_projection(impulse_range,       entry, atr)
                tp1        = entry - tp1_dist
                tp2        = entry - tp2_dist
                notes      = [
                    f"✅ Bear flag breakdown — impulse range {fmt_price(impulse_range)}",
                    f"✅ Tight consolidation ({consol_range/impulse_range*100:.0f}% of impulse)",
                ]
                return ("SHORT", "BearFlag", 68, notes, entry, stop_loss, tp1, tp2)

        return None

    def _detect_triangle(self, highs, lows, closes, atr, volumes=None, tf='4h') -> Optional[tuple]:
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

        # P1-G3: triangle slope threshold — previous formula divided by lookback/4
        # making the threshold 0.1% per quarter of lookback, essentially 0 over
        # 50 bars. Use the minimum slope that produces a ≥2% total convergence
        # over the lookback window as the threshold.
        current_price = float(h[-1])
        min_total_convergence = 0.02  # 2% cumulative over the window
        slope_threshold = (current_price * min_total_convergence) / max(len(h), 1)
        is_symmetrical = h_slope < -slope_threshold and l_slope > slope_threshold

        if not is_symmetrical:
            return None

        # Project apex
        h_line = np.polyfit(x, h, 1)
        l_line = np.polyfit(x, l, 1)

        # Current price relative to the triangle
        proj_high = float(np.polyval(h_line, len(h)))
        proj_low  = float(np.polyval(l_line, len(h)))

        current  = float(closes[-1])
        width    = proj_high - proj_low

        if width <= 0 or width > atr * 20:
            return None  # Not a valid triangle

        # P1-G7: ATR-normalized TPs so small triangles don't produce tiny TPs.
        # Breakout: price outside triangle
        if current > proj_high:
            entry     = current
            stop_loss = proj_low - atr * 0.5
            tp1_raw   = max(width, atr * 1.5)
            tp2_raw   = max(width * 2, atr * 3.0)
            tp1_dist  = self._clamp_projection(tp1_raw, entry, atr)
            tp2_dist  = self._clamp_projection(tp2_raw, entry, atr)
            tp1       = entry + tp1_dist
            tp2       = entry + tp2_dist
            notes     = [f"✅ Symmetrical triangle breakout (width: {fmt_price(width)})"]
            return ("LONG", "SymTriangle", 66, notes, entry, stop_loss, tp1, tp2)

        elif current < proj_low:
            entry     = current
            stop_loss = proj_high + atr * 0.5
            tp1_raw   = max(width, atr * 1.5)
            tp2_raw   = max(width * 2, atr * 3.0)
            tp1_dist  = self._clamp_projection(tp1_raw, entry, atr)
            tp2_dist  = self._clamp_projection(tp2_raw, entry, atr)
            tp1       = entry - tp1_dist
            tp2       = entry - tp2_dist
            notes     = [f"✅ Symmetrical triangle breakdown (width: {fmt_price(width)})"]
            return ("SHORT", "SymTriangle", 66, notes, entry, stop_loss, tp1, tp2)

        return None

    def _detect_double_top_bottom(self, highs, lows, closes, atr, volumes=None, tf='4h') -> Optional[tuple]:
        """
        Double top (bearish) / Double bottom (bullish).
        Two equal highs or lows with a neckline.
        P1-G6: prominence filter — micro-swings at width=2 used to fire on noise.
        Require the swing high/low to be at least 0.5 ATR more extreme than the
        outer 2-bar neighbors.
        """
        if len(closes) < 30:
            return None

        tolerance = atr * 1.5   # Tops/bottoms must be within 1.5 ATR of each other
        prominence = atr * 0.5  # P1-G6: swing must stick out by ≥0.5 ATR

        # Find swing highs for double top with prominence
        swing_highs = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                nbr_max = max(highs[i-2], highs[i-1], highs[i+1], highs[i+2])
                if highs[i] - nbr_max >= prominence:
                    swing_highs.append((i, float(highs[i])))

        # Double top: last two swing highs are approximately equal
        if len(swing_highs) >= 2:
            h1_idx, h1_val = swing_highs[-2]
            h2_idx, h2_val = swing_highs[-1]
            if abs(h1_val - h2_val) < tolerance and h2_idx - h1_idx > 5:
                # Neckline = low between the two tops
                neckline = float(min(lows[h1_idx:h2_idx]))
                current  = float(closes[-1])

                if current < neckline:  # Breakdown below neckline
                    entry     = current
                    stop_loss = h2_val + atr * rp.sl_atr_mult * 0.25
                    height    = (h1_val + h2_val) / 2 - neckline
                    tp1_dist  = self._clamp_projection(height * 0.5, entry, atr)
                    tp2_dist  = self._clamp_projection(height,       entry, atr)
                    tp1       = neckline - tp1_dist
                    tp2       = neckline - tp2_dist
                    notes     = [
                        f"✅ Double Top breakdown below neckline {fmt_price(neckline)}",
                        f"✅ Both tops at ~{fmt_price((h1_val+h2_val)/2)} (within {tolerance:,.4f})",
                    ]
                    return ("SHORT", "DoubleTop", 72, notes, entry, stop_loss, tp1, tp2)

        # Find swing lows for double bottom with prominence
        swing_lows = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                nbr_min = min(lows[i-2], lows[i-1], lows[i+1], lows[i+2])
                if nbr_min - lows[i] >= prominence:
                    swing_lows.append((i, float(lows[i])))

        if len(swing_lows) >= 2:
            l1_idx, l1_val = swing_lows[-2]
            l2_idx, l2_val = swing_lows[-1]
            if abs(l1_val - l2_val) < tolerance and l2_idx - l1_idx > 5:
                neckline = float(max(highs[l1_idx:l2_idx]))
                current  = float(closes[-1])

                if current > neckline:  # Breakout above neckline
                    entry     = current
                    stop_loss = l2_val - atr * rp.sl_atr_mult * 0.25
                    height    = neckline - (l1_val + l2_val) / 2
                    tp1_dist  = self._clamp_projection(height * 0.5, entry, atr)
                    tp2_dist  = self._clamp_projection(height,       entry, atr)
                    tp1       = neckline + tp1_dist
                    tp2       = neckline + tp2_dist
                    notes     = [
                        f"✅ Double Bottom breakout above neckline {fmt_price(neckline)}",
                        f"✅ Both bottoms at ~{fmt_price((l1_val+l2_val)/2)}",
                    ]
                    return ("LONG", "DoubleBottom", 72, notes, entry, stop_loss, tp1, tp2)

        return None

    # ── HIGH REWARD PATTERNS ───────────────────────────────

    def _detect_cup_and_handle(self, highs, lows, closes, atr, volumes=None, tf='4h') -> Optional[tuple]:
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
        left_rim_idx = int(np.argmax(highs[:mid]))
        left_rim = float(highs[left_rim_idx])

        # P1-G8: require isolated swing high (5/5). A pump-dump at the series
        # start should not pass as a rim.
        _iso_k = 5
        if left_rim_idx < _iso_k or left_rim_idx + _iso_k >= len(highs):
            return None
        if not (all(highs[left_rim_idx] > highs[left_rim_idx - j] for j in range(1, _iso_k + 1)) and
                all(highs[left_rim_idx] > highs[left_rim_idx + j] for j in range(1, _iso_k + 1))):
            return None

        # Cup bottom: lowest point between left rim and recent bars
        cup_zone = lows[left_rim_idx + 3:-8]
        if len(cup_zone) < 10:
            return None
        cup_bottom_rel = int(np.argmin(cup_zone))
        cup_bottom = float(cup_zone[cup_bottom_rel])
        cup_depth = left_rim - cup_bottom

        if cup_depth < atr * 2:
            return None  # Cup too shallow

        # Right rim: price should recover to near left rim level
        recent_high = float(max(highs[-15:-5]))
        # P1-G8: tighten rim tolerance from 15% to 8% of cup depth (textbook)
        rim_tolerance = cup_depth * 0.08
        if abs(recent_high - left_rim) > rim_tolerance:
            return None  # Right rim doesn't match left rim

        # Handle: small pullback in last 5-12 bars
        handle_low = float(min(lows[-10:]))
        handle_depth = recent_high - handle_low
        if handle_depth > cup_depth * 0.50:
            return None  # Handle too deep (>50% of cup)
        if handle_depth < atr * 0.3:
            return None  # Handle too shallow

        # Breakout confirmation
        current = float(closes[-1])
        rim_level = (left_rim + recent_high) / 2

        if current < rim_level - atr * 0.3:
            return None  # Not breaking out yet

        # Volume confirmation: breakout bar should have higher volume
        confidence = 75.0
        if volumes is not None and len(volumes) > 20:
            avg_vol = float(np.mean(volumes[-20:-1]))
            if avg_vol > 0 and volumes[-1] > avg_vol * 1.5:
                confidence += 5.0  # Volume confirms breakout

        entry = current
        stop_loss = handle_low - atr * rp.sl_atr_mult * 0.3
        # Cup & handle target = cup depth projected above rim (clamped)
        tp1_dist = self._clamp_projection(cup_depth * 0.6, entry, atr)
        tp2_dist = self._clamp_projection(cup_depth,       entry, atr)
        tp1 = rim_level + tp1_dist
        tp2 = rim_level + tp2_dist
        notes = [
            f"🏆 Cup & Handle breakout — cup depth {fmt_price(cup_depth)}",
            f"✅ Handle pullback {handle_depth/cup_depth*100:.0f}% of cup (healthy)",
            f"✅ Rim level: {fmt_price(rim_level)}",
        ]
        return ("LONG", "CupAndHandle", confidence, notes, entry, stop_loss, tp1, tp2)

    def _detect_head_and_shoulders(self, highs, lows, closes, atr, volumes=None, tf='4h') -> Optional[tuple]:
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

        ls_val = float(max(z1_highs))   # Left shoulder
        head_val = float(max(z2_highs)) # Head
        rs_val = float(max(z3_highs))   # Right shoulder

        # P1-G5: right shoulder must NOT exceed left shoulder by more than 0.5 ATR.
        # A rising right shoulder invalidates the pattern's momentum-decay premise.
        if head_val > ls_val and head_val > rs_val and rs_val <= ls_val + atr * 0.5:
            shoulder_diff = abs(ls_val - rs_val)
            if shoulder_diff < atr * 2:  # Shoulders roughly same height
                # Neckline: connect lows between shoulders
                z1_low = float(min(lows[third-5:third+5])) if third > 5 else float(min(lows[:third]))
                z2_low = float(min(lows[third*2-5:third*2+5])) if third*2 > 5 else float(min(lows[third:third*2]))
                neckline = (z1_low + z2_low) / 2

                current = float(closes[-1])
                head_to_neckline = head_val - neckline

                if head_to_neckline >= atr * 2 and current < neckline + atr * 0.3:
                    # Breaking or below neckline
                    confidence = 73.0
                    if volumes is not None and len(volumes) > 10:
                        _bl = float(np.mean(volumes[-10:]))
                        if _bl > 0 and volumes[-1] > _bl * 1.3:
                            confidence += 5.0

                    entry = current
                    stop_loss = rs_val + atr * rp.sl_atr_mult * 0.3
                    tp1_dist = self._clamp_projection(head_to_neckline * 0.5, entry, atr)
                    tp2_dist = self._clamp_projection(head_to_neckline,       entry, atr)
                    tp1 = neckline - tp1_dist
                    tp2 = neckline - tp2_dist
                    notes = [
                        f"🏆 Head & Shoulders — neckline {fmt_price(neckline)}",
                        f"✅ Head: {fmt_price(head_val)} | Shoulders: ~{fmt_price((ls_val+rs_val)/2)}",
                        f"✅ Measured target: {fmt_price(tp2)}",
                    ]
                    return ("SHORT", "HeadAndShoulders", confidence, notes, entry, stop_loss, tp1, tp2)

        # Inverse H&S (bullish): find 3 swing lows — mirror prominence rule
        z1_lows = lows[:third]
        z2_lows = lows[third:third*2]
        z3_lows = lows[third*2:]

        ls_low = float(min(z1_lows))
        head_low = float(min(z2_lows))
        rs_low = float(min(z3_lows))

        # Right shoulder low must not be lower than left shoulder by more than 0.5 ATR
        if head_low < ls_low and head_low < rs_low and rs_low >= ls_low - atr * 0.5:
            shoulder_diff = abs(ls_low - rs_low)
            if shoulder_diff < atr * 2:
                z1_high = float(max(highs[third-5:third+5])) if third > 5 else float(max(highs[:third]))
                z2_high = float(max(highs[third*2-5:third*2+5])) if third*2 > 5 else float(max(highs[third:third*2]))
                neckline = (z1_high + z2_high) / 2

                current = float(closes[-1])
                neckline_to_head = neckline - head_low

                if neckline_to_head >= atr * 2 and current > neckline - atr * 0.3:
                    confidence = 73.0
                    if volumes is not None and len(volumes) > 10:
                        _bl = float(np.mean(volumes[-10:]))
                        if _bl > 0 and volumes[-1] > _bl * 1.3:
                            confidence += 5.0

                    entry = current
                    stop_loss = rs_low - atr * rp.sl_atr_mult * 0.3
                    tp1_dist = self._clamp_projection(neckline_to_head * 0.5, entry, atr)
                    tp2_dist = self._clamp_projection(neckline_to_head,       entry, atr)
                    tp1 = neckline + tp1_dist
                    tp2 = neckline + tp2_dist
                    notes = [
                        f"🏆 Inverse Head & Shoulders — neckline {fmt_price(neckline)}",
                        f"✅ Head: {fmt_price(head_low)} | Shoulders: ~{fmt_price((ls_low+rs_low)/2)}",
                        f"✅ Measured target: {fmt_price(tp2)}",
                    ]
                    return ("LONG", "InverseHS", confidence, notes, entry, stop_loss, tp1, tp2)

        return None

    def _detect_wedge(self, highs, lows, closes, atr, volumes=None, tf='4h') -> Optional[tuple]:
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
            high_fit  = np.polyfit(x, recent_highs, 1)
            low_fit   = np.polyfit(x, recent_lows, 1)
            high_slope = float(high_fit[0])
            low_slope  = float(low_fit[0])
        except (np.linalg.LinAlgError, ValueError):
            return None

        current = float(closes[-1])
        wedge_range = float(max(recent_highs) - min(recent_lows))

        if wedge_range < atr * 2:
            return None  # Too tight

        # P1-G5: use projected trendline values at x=n-1 (the current bar),
        # not single-bar extremes (`recent_lows[-1]`). A single wick no longer
        # satisfies the breakdown gate.
        lower_tl_now = float(np.polyval(low_fit,  n - 1))
        upper_tl_now = float(np.polyval(high_fit, n - 1))

        # Rising wedge (bearish): both slopes positive, converging
        if high_slope > 0 and low_slope > 0 and low_slope > high_slope * 0.5:
            # Check convergence: range is narrowing
            first_range = recent_highs[0] - recent_lows[0]
            last_range = recent_highs[-1] - recent_lows[-1]
            if last_range < first_range * 0.75:  # Converging
                # FIX GEO-3: was `current < lower_tl + atr * 0.2` which fired when price
                # was INSIDE the wedge.  Must close BELOW the trendline by 0.3 ATR.
                if current < lower_tl_now - atr * 0.3:
                    confidence = 70.0
                    entry = current
                    stop_loss = float(max(recent_highs[-5:])) + atr * rp.sl_atr_mult * 0.25
                    tp1_dist = self._clamp_projection(wedge_range * 0.5, entry, atr)
                    tp2_dist = self._clamp_projection(wedge_range * 0.8, entry, atr)
                    tp1 = entry - tp1_dist
                    tp2 = entry - tp2_dist
                    # P1-G2: previous version applied ':.6f' to the str returned by
                    # fmt_price() → TypeError on every wedge breakdown. Now the
                    # numeric distance is formatted directly.
                    _break_dist = abs(lower_tl_now - current)
                    notes = [
                        f"🏆 Rising Wedge breakdown — bearish reversal",
                        f"✅ Converging trendlines (range narrowed {last_range/first_range*100:.0f}%)",
                        f"✅ Confirmed close {_break_dist:.6f} below lower TL",
                    ]
                    return ("SHORT", "RisingWedge", confidence, notes, entry, stop_loss, tp1, tp2)

        # Falling wedge (bullish): both slopes negative, converging
        if high_slope < 0 and low_slope < 0 and high_slope < low_slope * 0.5:
            first_range = recent_highs[0] - recent_lows[0]
            last_range = recent_highs[-1] - recent_lows[-1]
            if last_range < first_range * 0.75:
                if current > upper_tl_now + atr * 0.3:
                    confidence = 70.0
                    entry = current
                    stop_loss = float(min(recent_lows[-5:])) - atr * rp.sl_atr_mult * 0.25
                    tp1_dist = self._clamp_projection(wedge_range * 0.5, entry, atr)
                    tp2_dist = self._clamp_projection(wedge_range * 0.8, entry, atr)
                    tp1 = entry + tp1_dist
                    tp2 = entry + tp2_dist
                    _break_dist = abs(current - upper_tl_now)
                    notes = [
                        f"🏆 Falling Wedge breakout — bullish reversal",
                        f"✅ Converging trendlines (range narrowed {last_range/first_range*100:.0f}%)",
                        f"✅ Confirmed close {_break_dist:.6f} above upper TL",
                    ]
                    return ("LONG", "FallingWedge", confidence, notes, entry, stop_loss, tp1, tp2)

        return None
