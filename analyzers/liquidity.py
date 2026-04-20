"""
TitanBot Pro — Liquidity Analyzer
=====================================
Detects liquidity levels and reactions using:
  - Order book depth (bid/ask clusters)
  - Recent swing highs/lows (equal highs/lows = liquidity pools)
  - Sweep detection (price takes liquidity then reverses)

Used by execution engine as 4th entry trigger:
  liquidity_below_swept + strong_close_up → bullish reversal trigger
  liquidity_above_swept + rejection candle → bearish reversal trigger

Professional cycle: Seek liquidity → Take liquidity → Reverse
This module detects the "take" and "reverse" phases.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config.constants import LiqSweep as LiqSweepConst
from data.api_client import api

logger = logging.getLogger(__name__)

# ── Sweep-detection thresholds (PR-2 tier-1 fix targets) ────────────
# These replace prior magic numbers inline in detect_reaction().  A
# future PR can promote them to config/constants.py; kept local here
# to keep this surgical bug-fix small.
_SWEEP_ATR_EPS          = 1e-9    # Below this ATR is treated as unusable
_SWEEP_VOLUME_MIN_RATIO = 1.2     # Sweep bar volume must exceed 20-bar avg × this
_SWEEP_DWELL_MIN_BARS   = 2       # Sweep must hold for ≥ N bars (not a single-wick tick)


@dataclass
class LiquidityLevel:
    """A detected liquidity cluster"""
    price: float
    side: str          # 'above' (sell-side) or 'below' (buy-side)
    strength: float    # 0-1 strength score
    source: str        # 'equal_highs', 'equal_lows', 'order_book', 'swing'


@dataclass 
class LiquidityReaction:
    """Result of checking if liquidity was swept and price reacted"""
    swept: bool = False
    direction: str = ""     # LONG or SHORT reversal
    swept_level: float = 0
    reaction_strength: float = 0   # 0-1 how strong the reaction
    notes: str = ""


class LiquidityAnalyzer:
    """
    Analyzes liquidity conditions for execution trigger decisions.
    """

    def find_levels(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray,
        current_price: float,
        atr: float,
    ) -> List[LiquidityLevel]:
        """
        Find nearby liquidity levels from price action.
        
        Sources:
          - Equal highs/lows (retail stop clusters)
          - Recent swing points (obvious S/R = liquidity)
        """
        levels = []
        n = len(closes)
        if n < 20:
            return levels

        # AUDIT FIX (ATR=0 guard): when ATR is zero or negative (flat
        # candles, padded array) `tolerance` becomes 0 and the dedup
        # key below (`round(price / tolerance)`) raises ZeroDivisionError.
        # A non-positive ATR is also meaningless for "within 0.15 ATR"
        # equivalence checks, so bail out early with no levels.
        if not atr or atr <= _SWEEP_ATR_EPS or not np.isfinite(atr):
            return levels

        tolerance = atr * 0.15  # Prices within 0.15 ATR are "equal"

        # Find equal highs (sell-side liquidity above)
        recent_highs = highs[-20:]
        for i in range(len(recent_highs)):
            for j in range(i + 2, min(i + 10, len(recent_highs))):
                if abs(recent_highs[i] - recent_highs[j]) < tolerance:
                    level = (recent_highs[i] + recent_highs[j]) / 2
                    if level > current_price:
                        levels.append(LiquidityLevel(
                            price=level, side='above', strength=0.7,
                            source='equal_highs'
                        ))

        # Find equal lows (buy-side liquidity below)
        recent_lows = lows[-20:]
        for i in range(len(recent_lows)):
            for j in range(i + 2, min(i + 10, len(recent_lows))):
                if abs(recent_lows[i] - recent_lows[j]) < tolerance:
                    level = (recent_lows[i] + recent_lows[j]) / 2
                    if level < current_price:
                        levels.append(LiquidityLevel(
                            price=level, side='below', strength=0.7,
                            source='equal_lows'
                        ))

        # Find swing highs/lows (3-bar pivots)
        for i in range(2, n - 2):
            # Swing high
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] >= highs[min(i+2, n-1)]:
                if highs[i] > current_price:
                    levels.append(LiquidityLevel(
                        price=highs[i], side='above', strength=0.5,
                        source='swing'
                    ))
            # Swing low
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] <= lows[min(i+2, n-1)]:
                if lows[i] < current_price:
                    levels.append(LiquidityLevel(
                        price=lows[i], side='below', strength=0.5,
                        source='swing'
                    ))

        # Deduplicate levels within tolerance
        deduped = []
        used = set()
        for lv in sorted(levels, key=lambda x: x.strength, reverse=True):
            key = round(lv.price / tolerance)
            if key not in used:
                deduped.append(lv)
                used.add(key)
        
        return deduped[:10]  # Top 10 levels

    def detect_reaction(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
        current_price: float,
        direction: str,
        atr: float,
        volumes: Optional[np.ndarray] = None,
    ) -> LiquidityReaction:
        """
        Detect if liquidity was recently swept and price reacted.
        
        For LONG: check if price swept below a low then closed back up
        For SHORT: check if price swept above a high then closed back down
        
        This is the professional "take liquidity → reverse" pattern.

        AUDIT FIX (PR-2): the sweep pattern now requires:
          * ATR > 0 (see guards at L170/L202)
          * If ``volumes`` is supplied, the sweep bar volume must exceed
            the 20-bar average volume × ``_SWEEP_VOLUME_MIN_RATIO``.
            A low-volume probe-and-retrace is almost always noise, not
            a real liquidity take.
          * Dwell time ≥ ``_SWEEP_DWELL_MIN_BARS``: the recent-window
            extreme must have broken the prior level on at least N bars,
            so a single wick tagging a low before snapping back is
            rejected as stop-hunt-noise rather than accepted as a sweep.
        """
        n = len(closes)
        if n < 10:
            return LiquidityReaction()

        tolerance = atr * 0.15
        lookback = min(10, n - 2)

        def _volume_gate(bar_indices: List[int]) -> bool:
            """True when at least one of ``bar_indices`` had volume
            above the volume-confirmation threshold.  Fails open
            (returns True) when no volume array is supplied so existing
            callers that never pass volumes retain their prior
            behaviour."""
            if volumes is None or len(volumes) < 20:
                return True
            try:
                avg_vol = float(np.mean(volumes[-20:]))
            except (ValueError, TypeError):
                return True
            if avg_vol <= 0 or not np.isfinite(avg_vol):
                return True
            threshold = avg_vol * _SWEEP_VOLUME_MIN_RATIO
            for idx in bar_indices:
                try:
                    v = float(volumes[idx])
                except (IndexError, ValueError, TypeError):
                    continue
                if v > threshold:
                    return True
            return False

        if direction == "LONG":
            # Look for sweep below recent lows
            # Find the lowest low in recent history (potential liquidity)
            _prev_slice = lows[-lookback-5:-lookback] if n > lookback + 5 else lows[:-lookback]
            if len(_prev_slice) == 0:
                return LiquidityReaction()
            prev_low = min(_prev_slice)
            recent_low = min(lows[-lookback:])
            
            # Was there a sweep? (recent low went below previous low)
            if recent_low < prev_low - tolerance * 0.5:
                # AUDIT FIX (dwell-time): require the break-below to
                # persist for ≥ _SWEEP_DWELL_MIN_BARS, not a single-tick
                # wick.  Count bars in the lookback window whose low
                # dipped below the prior level.
                break_bars = [
                    -lookback + i
                    for i in range(lookback)
                    if lows[-lookback + i] < prev_low - tolerance * 0.5
                ]
                if len(break_bars) < _SWEEP_DWELL_MIN_BARS:
                    return LiquidityReaction()
                # Did price recover? (current close back above previous low)
                if current_price > prev_low:
                    # Check last candle for strong close
                    last_close = closes[-1]
                    last_open = opens[-1]
                    bullish_close = last_close > last_open
                    
                    # AUDIT FIX: guard against zero ATR (from flat/padded
                    # candles) before dividing.
                    if atr <= 0:
                        return LiquidityReaction()
                    recovery = (current_price - recent_low) / atr
                    reaction_strength = min(1.0, recovery / 1.5)
                    
                    if (
                        bullish_close
                        and reaction_strength > 0.3
                        and _volume_gate(break_bars)
                    ):
                        return LiquidityReaction(
                            swept=True,
                            direction="LONG",
                            swept_level=prev_low,
                            reaction_strength=reaction_strength,
                            notes=f"Buy-side liquidity swept at {prev_low:.4g}, "
                                  f"strong recovery ({recovery:.1f} ATR)"
                        )

        elif direction == "SHORT":
            # Look for sweep above recent highs
            _prev_slice_h = highs[-lookback-5:-lookback] if n > lookback + 5 else highs[:-lookback]
            if len(_prev_slice_h) == 0:
                return LiquidityReaction()
            prev_high = max(_prev_slice_h)
            recent_high = max(highs[-lookback:])
            
            if recent_high > prev_high + tolerance * 0.5:
                # Dwell-time gate for SHORT (break above prev_high)
                break_bars = [
                    -lookback + i
                    for i in range(lookback)
                    if highs[-lookback + i] > prev_high + tolerance * 0.5
                ]
                if len(break_bars) < _SWEEP_DWELL_MIN_BARS:
                    return LiquidityReaction()
                if current_price < prev_high:
                    last_close = closes[-1]
                    last_open = opens[-1]
                    bearish_close = last_close < last_open
                    
                    # AUDIT FIX: guard against zero ATR before dividing.
                    if atr <= 0:
                        return LiquidityReaction()
                    recovery = (recent_high - current_price) / atr
                    reaction_strength = min(1.0, recovery / 1.5)
                    
                    if (
                        bearish_close
                        and reaction_strength > 0.3
                        and _volume_gate(break_bars)
                    ):
                        return LiquidityReaction(
                            swept=True,
                            direction="SHORT",
                            swept_level=prev_high,
                            reaction_strength=reaction_strength,
                            notes=f"Sell-side liquidity swept at {prev_high:.4g}, "
                                  f"strong rejection ({recovery:.1f} ATR)"
                        )

        return LiquidityReaction()

    async def check_orderbook_imbalance(
        self,
        symbol: str,
        direction: str,
    ) -> Tuple[float, str]:
        """
        Check order book for directional imbalance.
        
        Returns:
            (imbalance_score, description)
            imbalance_score: -1 to +1 (positive = bullish)
        """
        try:
            import asyncio
            book = await asyncio.wait_for(
                api.fetch_order_book(symbol, limit=20), timeout=5.0
            )
            if not book:
                return 0.0, "No order book data"

            bids = book.get('bids', [])
            asks = book.get('asks', [])
            if not bids or not asks:
                return 0.0, "Empty order book"

            # AUDIT FIX: validate row shape before unpacking — a
            # malformed exchange response (e.g. list of dicts) would
            # otherwise raise IndexError inside the generator and get
            # swallowed by the broad except below without explanation.
            def _ok(row) -> bool:
                return isinstance(row, (list, tuple)) and len(row) >= 2

            bids = [b for b in bids[:10] if _ok(b)]
            asks = [a for a in asks[:10] if _ok(a)]
            if not bids or not asks:
                return 0.0, "Malformed order book rows"

            try:
                bid_depth = sum(float(b[1]) * float(b[0]) for b in bids)
                ask_depth = sum(float(a[1]) * float(a[0]) for a in asks)
            except (ValueError, TypeError) as _e:
                return 0.0, f"Order book parse error: {_e}"
            total = bid_depth + ask_depth

            if total == 0:
                return 0.0, "Zero depth"

            imbalance = (bid_depth - ask_depth) / total  # -1 to +1

            if direction == "LONG" and imbalance > 0.15:
                return imbalance, f"Order book bullish ({imbalance:+.2f})"
            elif direction == "SHORT" and imbalance < -0.15:
                return imbalance, f"Order book bearish ({imbalance:+.2f})"
            else:
                return imbalance, f"Order book neutral ({imbalance:+.2f})"

        except Exception as e:
            return 0.0, f"Order book error: {e}"


# ── Singleton ──────────────────────────────────────────────
liquidity_analyzer = LiquidityAnalyzer()
