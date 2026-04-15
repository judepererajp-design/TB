"""
TitanBot Pro — Candlestick Pattern Library
============================================
Pure functions for detecting classic candlestick patterns.
Used by PriceActionStrategy and other modules.
All functions take (opens, highs, lows, closes) numpy arrays
and return (detected: bool, strength: float 0-1).
"""

import numpy as np
from typing import Tuple


def pin_bar(opens, highs, lows, closes, wick_ratio: float = 2.5) -> Tuple[bool, str, float]:
    """
    Pin bar (hammer / shooting star).
    Returns (detected, direction, strength)
    """
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    body        = abs(c - o)
    total       = h - l
    if total == 0:
        return False, "", 0.0

    lower_wick  = min(o, c) - l
    upper_wick  = h - max(o, c)

    # Bullish pin bar (hammer): long lower wick
    if lower_wick > body * wick_ratio and lower_wick > upper_wick * 2:
        strength = min(1.0, lower_wick / total)
        return True, "LONG", strength

    # Bearish pin bar (shooting star): long upper wick
    if upper_wick > body * wick_ratio and upper_wick > lower_wick * 2:
        strength = min(1.0, upper_wick / total)
        return True, "SHORT", strength

    return False, "", 0.0


def engulfing(opens, highs, lows, closes, min_size: float = 1.2) -> Tuple[bool, str, float]:
    """
    Bullish/Bearish Engulfing candle.
    """
    if len(closes) < 2:
        return False, "", 0.0

    o1, h1, l1, c1 = opens[-1], highs[-1], lows[-1], closes[-1]
    o2, c2          = opens[-2], closes[-2]
    body1           = abs(c1 - o1)
    body2           = abs(c2 - o2)

    if body2 == 0:
        return False, "", 0.0

    # Bullish engulfing: prev bearish, current bullish, current body > prev body
    if c2 < o2 and c1 > o1 and body1 >= body2 * min_size:
        strength = min(1.0, body1 / body2 / 2)
        return True, "LONG", strength

    # Bearish engulfing: prev bullish, current bearish, current body > prev body
    if c2 > o2 and c1 < o1 and body1 >= body2 * min_size:
        strength = min(1.0, body1 / body2 / 2)
        return True, "SHORT", strength

    return False, "", 0.0


def inside_bar(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """
    Inside bar: current bar's range is completely within previous bar's range.
    Breakout direction determines trade direction.
    """
    if len(highs) < 2:
        return False, "", 0.0

    if highs[-1] < highs[-2] and lows[-1] > lows[-2]:
        # Inside bar detected — direction depends on context (neutral here)
        strength = 1.0 - (highs[-1] - lows[-1]) / (highs[-2] - lows[-2] + 1e-10)
        return True, "NEUTRAL", min(1.0, strength)

    return False, "", 0.0


def morning_star(opens, highs, lows, closes, atr: float = 0.0) -> Tuple[bool, str, float]:
    """
    Morning Star: 3-candle bullish reversal.
    Large bearish → small body doji/star → large bullish
    """
    if len(closes) < 3:
        return False, "", 0.0

    o1, c1 = opens[-3], closes[-3]   # First candle (bearish)
    o2, c2 = opens[-2], closes[-2]   # Middle candle (small)
    o3, c3 = opens[-1], closes[-1]   # Third candle (bullish)

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    body3 = abs(c3 - o3)

    min_body = atr * 0.3 if atr > 0 else body1 * 0.3

    if (c1 < o1 and                    # First = bearish
            body1 > min_body and       # First has decent body
            body2 < body1 * 0.4 and    # Middle = small body
            c3 > o3 and                # Third = bullish
            body3 > min_body and       # Third has decent body
            c3 > (o1 + c1) / 2):      # Third closes above midpoint of first
        strength = min(1.0, body3 / body1)
        return True, "LONG", strength

    return False, "", 0.0


def evening_star(opens, highs, lows, closes, atr: float = 0.0) -> Tuple[bool, str, float]:
    """
    Evening Star: 3-candle bearish reversal. Mirror of Morning Star.
    """
    if len(closes) < 3:
        return False, "", 0.0

    o1, c1 = opens[-3], closes[-3]
    o2, c2 = opens[-2], closes[-2]
    o3, c3 = opens[-1], closes[-1]

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    body3 = abs(c3 - o3)

    min_body = atr * 0.3 if atr > 0 else body1 * 0.3

    if (c1 > o1 and
            body1 > min_body and
            body2 < body1 * 0.4 and
            c3 < o3 and
            body3 > min_body and
            c3 < (o1 + c1) / 2):
        strength = min(1.0, body3 / body1)
        return True, "SHORT", strength

    return False, "", 0.0


def doji(opens, highs, lows, closes, doji_pct: float = 0.1) -> Tuple[bool, str, float]:
    """
    Doji: open ≈ close (indecision candle).
    Context (at support vs resistance) determines direction.
    """
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    total       = h - l
    body        = abs(c - o)

    if total == 0:
        return False, "", 0.0

    if body / total < doji_pct:
        return True, "NEUTRAL", 1.0 - (body / total / doji_pct)

    return False, "", 0.0


def three_white_soldiers(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """Three consecutive bullish candles with higher closes"""
    if len(closes) < 3:
        return False, "", 0.0

    c1, c2, c3 = closes[-3], closes[-2], closes[-1]
    o1, o2, o3 = opens[-3], opens[-2], opens[-1]

    if (c1 > o1 and c2 > o2 and c3 > o3 and
            c3 > c2 > c1 and
            o2 > o1 and o3 > o2):
        return True, "LONG", 0.8

    return False, "", 0.0


def three_black_crows(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """Three consecutive bearish candles with lower closes"""
    if len(closes) < 3:
        return False, "", 0.0

    c1, c2, c3 = closes[-3], closes[-2], closes[-1]
    o1, o2, o3 = opens[-3], opens[-2], opens[-1]

    if (c1 < o1 and c2 < o2 and c3 < o3 and
            c3 < c2 < c1 and
            o2 < o1 and o3 < o2):
        return True, "SHORT", 0.8

    return False, "", 0.0


def detect_all(opens, highs, lows, closes, atr: float = 0.0) -> list:
    """
    Run all pattern detectors and return list of detected patterns.
    Returns: [{'pattern': str, 'direction': str, 'strength': float}]
    """
    detectors = [
        ('PinBar',              lambda: pin_bar(opens, highs, lows, closes)),
        ('Engulfing',           lambda: engulfing(opens, highs, lows, closes)),
        ('InsideBar',           lambda: inside_bar(opens, highs, lows, closes)),
        ('MorningStar',         lambda: morning_star(opens, highs, lows, closes, atr)),
        ('EveningStar',         lambda: evening_star(opens, highs, lows, closes, atr)),
        ('Doji',                lambda: doji(opens, highs, lows, closes)),
        ('ThreeWhiteSoldiers',  lambda: three_white_soldiers(opens, highs, lows, closes)),
        ('ThreeBlackCrows',     lambda: three_black_crows(opens, highs, lows, closes)),
    ]

    results = []
    for name, fn in detectors:
        try:
            detected, direction, strength = fn()
            if detected:
                results.append({'pattern': name, 'direction': direction, 'strength': strength})
        except Exception:
            continue

    return results
