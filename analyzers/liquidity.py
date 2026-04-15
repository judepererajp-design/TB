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

from data.api_client import api

logger = logging.getLogger(__name__)


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
    ) -> LiquidityReaction:
        """
        Detect if liquidity was recently swept and price reacted.
        
        For LONG: check if price swept below a low then closed back up
        For SHORT: check if price swept above a high then closed back down
        
        This is the professional "take liquidity → reverse" pattern.
        """
        n = len(closes)
        if n < 10:
            return LiquidityReaction()

        tolerance = atr * 0.15
        lookback = min(10, n - 2)

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
                # Did price recover? (current close back above previous low)
                if current_price > prev_low:
                    # Check last candle for strong close
                    last_close = closes[-1]
                    last_open = opens[-1]
                    bullish_close = last_close > last_open
                    
                    recovery = (current_price - recent_low) / atr
                    reaction_strength = min(1.0, recovery / 1.5)
                    
                    if bullish_close and reaction_strength > 0.3:
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
                if current_price < prev_high:
                    last_close = closes[-1]
                    last_open = opens[-1]
                    bearish_close = last_close < last_open
                    
                    recovery = (recent_high - current_price) / atr
                    reaction_strength = min(1.0, recovery / 1.5)
                    
                    if bearish_close and reaction_strength > 0.3:
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

            bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:10])
            ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:10])
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
