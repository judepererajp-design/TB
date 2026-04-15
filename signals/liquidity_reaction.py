
"""
Liquidity Reaction Trigger
--------------------------
Adds execution confirmation based on liquidity sweep + reaction.
This module is intentionally lightweight and async-safe.

Logic:
LONG  -> liquidity below swept + strong bullish close
SHORT -> liquidity above swept + rejection candle
"""

from typing import Dict

def detect_liquidity_reaction(signal: Dict, market_state: Dict) -> bool:
    """Return True if liquidity reaction confirmation exists."""

    liquidity = market_state.get("liquidity", {})
    candle = market_state.get("last_candle", {})

    swept_below = liquidity.get("below_swept", False)
    swept_above = liquidity.get("above_swept", False)

    strong_close_up = candle.get("close_strength_up", False)
    rejection = candle.get("rejection", False)

    direction = signal.get("direction", "").upper()

    if direction == "LONG":
        if swept_below and strong_close_up:
            return True

    if direction == "SHORT":
        if swept_above and rejection:
            return True

    return False
