"""
TitanBot Pro — Candlestick Pattern Library
============================================
Pure functions for detecting classic candlestick patterns.
Used by PriceActionStrategy and the signals aggregator as confluence.

All functions accept (opens, highs, lows, closes) array-likes (optionally volume
and atr) and return (detected: bool, direction: str, strength: float in 0-1).

Design rules (Phase-1 audit fixes):
  * Inputs are coerced to float numpy arrays and screened for NaN / inf.
  * Bodies of zero are handled explicitly — no ZeroDivisionError.
  * Reversal strengths penalize dominant-body candles (e.g. a "pin bar"
    with a large body gets a reduced strength, not the full wick ratio).
  * Engulfing requires body-level containment (not just larger-body), matching
    the textbook definition.
  * Optional volume and ATR inputs tighten detection when available.
  * detect_all logs exceptions instead of swallowing them silently.
"""

import logging
import numpy as np
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────

def _prep(arrays: Sequence[Sequence[float]]) -> Optional[Tuple[np.ndarray, ...]]:
    """
    Coerce each input to a 1-D float ndarray and verify finiteness on the
    trailing bars every detector actually inspects (last 3 by convention).
    Returns None if any input is empty or contains non-finite tail values.
    """
    out = []
    for a in arrays:
        try:
            arr = np.asarray(a, dtype=float)
        except Exception:
            return None
        if arr.ndim != 1 or arr.size == 0:
            return None
        # Only the tail (last 3 bars) is read by any detector.
        tail = arr[-3:] if arr.size >= 3 else arr
        if not np.all(np.isfinite(tail)):
            return None
        out.append(arr)
    return tuple(out)


def _body(o: float, c: float) -> float:
    return abs(c - o)


# ── Single-bar patterns ────────────────────────────────────────────────

def pin_bar(
    opens, highs, lows, closes,
    wick_ratio: float = 2.5,
    volume: Optional[Sequence[float]] = None,
) -> Tuple[bool, str, float]:
    """
    Pin bar (hammer / shooting star).

    Strength is wick/range scaled down by body share so candles with a large
    body (directional candles) can't masquerade as exhaustion pins.
    """
    prep = _prep((opens, highs, lows, closes))
    if prep is None:
        return False, "", 0.0
    opens_a, highs_a, lows_a, closes_a = prep

    o, h, l, c = opens_a[-1], highs_a[-1], lows_a[-1], closes_a[-1]
    total = h - l
    if total <= 0:
        return False, "", 0.0

    body = _body(o, c)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    body_share = body / total  # 0 = doji, 1 = marubozu

    # Bullish pin (hammer-like): long lower wick, small body, small upper wick
    if lower_wick > body * wick_ratio and lower_wick > upper_wick * 2 and lower_wick > 0:
        # Strength: wick dominance × (1 - body_share) to deflate big-body pins
        raw = (lower_wick / total) * max(0.0, 1.0 - body_share)
        return True, "LONG", min(1.0, raw)

    # Bearish pin (shooting-star / hanging-man): long upper wick
    if upper_wick > body * wick_ratio and upper_wick > lower_wick * 2 and upper_wick > 0:
        raw = (upper_wick / total) * max(0.0, 1.0 - body_share)
        return True, "SHORT", min(1.0, raw)

    return False, "", 0.0


def doji(opens, highs, lows, closes, doji_pct: float = 0.1) -> Tuple[bool, str, float]:
    """Doji: body very small relative to range. Neutral — caller provides context."""
    prep = _prep((opens, highs, lows, closes))
    if prep is None:
        return False, "", 0.0
    o, h, l, c = prep[0][-1], prep[1][-1], prep[2][-1], prep[3][-1]
    total = h - l
    if total <= 0:
        return False, "", 0.0
    body = _body(o, c)
    ratio = body / total
    if ratio < doji_pct:
        return True, "NEUTRAL", max(0.0, min(1.0, 1.0 - ratio / doji_pct))
    return False, "", 0.0


# ── Two-bar patterns ───────────────────────────────────────────────────

def engulfing(
    opens, highs, lows, closes, min_size: float = 1.2
) -> Tuple[bool, str, float]:
    """
    Bullish / Bearish engulfing — textbook form:
      bullish: prev bar bearish AND current bullish AND c_curr >= o_prev
               AND o_curr <= c_prev AND body_curr >= body_prev * min_size
    """
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 2:
        return False, "", 0.0
    opens_a, _, _, closes_a = prep
    o_prev, c_prev = opens_a[-2], closes_a[-2]
    o_cur, c_cur = opens_a[-1], closes_a[-1]
    body_prev = _body(o_prev, c_prev)
    body_cur = _body(o_cur, c_cur)
    if body_prev <= 0 or body_cur <= 0:
        return False, "", 0.0

    # Bullish engulfing
    if (c_prev < o_prev and c_cur > o_cur
            and body_cur >= body_prev * min_size
            and o_cur <= c_prev
            and c_cur >= o_prev):
        return True, "LONG", min(1.0, body_cur / (body_prev * 2))

    # Bearish engulfing
    if (c_prev > o_prev and c_cur < o_cur
            and body_cur >= body_prev * min_size
            and o_cur >= c_prev
            and c_cur <= o_prev):
        return True, "SHORT", min(1.0, body_cur / (body_prev * 2))

    return False, "", 0.0


def inside_bar(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """Inside bar: current range fully inside previous range. Direction = NEUTRAL."""
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[1].size < 2:
        return False, "", 0.0
    _, highs_a, lows_a, _ = prep
    if highs_a[-1] < highs_a[-2] and lows_a[-1] > lows_a[-2]:
        prev_range = highs_a[-2] - lows_a[-2]
        cur_range = highs_a[-1] - lows_a[-1]
        if prev_range <= 0:
            return False, "", 0.0
        strength = 1.0 - cur_range / prev_range
        return True, "NEUTRAL", max(0.0, min(1.0, strength))
    return False, "", 0.0


def tweezer_top(opens, highs, lows, closes, tol: float = 0.001) -> Tuple[bool, str, float]:
    """
    Tweezer top (bearish): two consecutive bars with nearly identical highs,
    the second bar bearish. Common double-rejection at local resistance.
    """
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 2:
        return False, "", 0.0
    o_a, h_a, _, c_a = prep
    h1, h2 = h_a[-2], h_a[-1]
    if h1 <= 0:
        return False, "", 0.0
    if abs(h1 - h2) / h1 > tol:
        return False, "", 0.0
    # Second bar must show rejection (bearish or long upper wick)
    body2 = _body(o_a[-1], c_a[-1])
    upper2 = h2 - max(o_a[-1], c_a[-1])
    total2 = h2 - prep[2][-1]
    if total2 <= 0:
        return False, "", 0.0
    if c_a[-1] < o_a[-1] or upper2 > body2:
        strength = min(1.0, (upper2 + body2) / total2)
        return True, "SHORT", strength
    return False, "", 0.0


def tweezer_bottom(opens, highs, lows, closes, tol: float = 0.001) -> Tuple[bool, str, float]:
    """Tweezer bottom (bullish): mirror of tweezer_top."""
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 2:
        return False, "", 0.0
    o_a, _, l_a, c_a = prep
    l1, l2 = l_a[-2], l_a[-1]
    if l1 <= 0:
        return False, "", 0.0
    if abs(l1 - l2) / l1 > tol:
        return False, "", 0.0
    body2 = _body(o_a[-1], c_a[-1])
    lower2 = min(o_a[-1], c_a[-1]) - l2
    total2 = prep[1][-1] - l2
    if total2 <= 0:
        return False, "", 0.0
    if c_a[-1] > o_a[-1] or lower2 > body2:
        strength = min(1.0, (lower2 + body2) / total2)
        return True, "LONG", strength
    return False, "", 0.0


def piercing_line(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """
    Piercing line (bullish): prev bearish, current opens below prev low
    and closes above midpoint of prev body (but not fully engulfing).
    """
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 2:
        return False, "", 0.0
    o_a, _, l_a, c_a = prep
    o1, c1 = o_a[-2], c_a[-2]
    o2, c2 = o_a[-1], c_a[-1]
    if c1 >= o1:
        return False, "", 0.0
    body1 = _body(o1, c1)
    if body1 <= 0:
        return False, "", 0.0
    mid1 = (o1 + c1) / 2.0
    # Open below prev low, close above midpoint, but not above prev open (not engulfing)
    if o2 < l_a[-2] and mid1 < c2 < o1:
        strength = min(1.0, (c2 - mid1) / (o1 - mid1))
        return True, "LONG", strength
    return False, "", 0.0


def dark_cloud_cover(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """Dark cloud cover (bearish): mirror of piercing line."""
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 2:
        return False, "", 0.0
    o_a, h_a, _, c_a = prep
    o1, c1 = o_a[-2], c_a[-2]
    o2, c2 = o_a[-1], c_a[-1]
    if c1 <= o1:
        return False, "", 0.0
    body1 = _body(o1, c1)
    if body1 <= 0:
        return False, "", 0.0
    mid1 = (o1 + c1) / 2.0
    if o2 > h_a[-2] and o1 < c2 < mid1:
        strength = min(1.0, (mid1 - c2) / (mid1 - o1))
        return True, "SHORT", strength
    return False, "", 0.0


# ── Three-bar patterns ─────────────────────────────────────────────────

def morning_star(opens, highs, lows, closes, atr: float = 0.0) -> Tuple[bool, str, float]:
    """
    Morning Star: bearish → small body → bullish.
    `min_body` uses ATR when available; otherwise a percent-of-price floor so the
    gate is never a tautology (the old body1*0.3 gate against itself was dead).
    """
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 3:
        return False, "", 0.0
    o_a, _, _, c_a = prep
    o1, c1 = o_a[-3], c_a[-3]
    o2, c2 = o_a[-2], c_a[-2]
    o3, c3 = o_a[-1], c_a[-1]
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    body3 = _body(o3, c3)
    if body1 <= 0 or body3 <= 0:
        return False, "", 0.0
    # ATR-based gate or 0.2% of price fallback (real gate, not tautology).
    ref_price = max(abs(c1), 1e-12)
    min_body = atr * 0.3 if atr > 0 else ref_price * 0.002
    if (c1 < o1 and body1 > min_body
            and body2 < body1 * 0.4
            and c3 > o3 and body3 > min_body
            and c3 > (o1 + c1) / 2):
        return True, "LONG", min(1.0, body3 / body1)
    return False, "", 0.0


def evening_star(opens, highs, lows, closes, atr: float = 0.0) -> Tuple[bool, str, float]:
    """Evening Star: mirror of Morning Star."""
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 3:
        return False, "", 0.0
    o_a, _, _, c_a = prep
    o1, c1 = o_a[-3], c_a[-3]
    o2, c2 = o_a[-2], c_a[-2]
    o3, c3 = o_a[-1], c_a[-1]
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    body3 = _body(o3, c3)
    if body1 <= 0 or body3 <= 0:
        return False, "", 0.0
    ref_price = max(abs(c1), 1e-12)
    min_body = atr * 0.3 if atr > 0 else ref_price * 0.002
    if (c1 > o1 and body1 > min_body
            and body2 < body1 * 0.4
            and c3 < o3 and body3 > min_body
            and c3 < (o1 + c1) / 2):
        return True, "SHORT", min(1.0, body3 / body1)
    return False, "", 0.0


def three_white_soldiers(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """Three consecutive bullish candles with higher closes and rising opens."""
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 3:
        return False, "", 0.0
    o_a, _, _, c_a = prep
    o1, o2, o3 = o_a[-3], o_a[-2], o_a[-1]
    c1, c2, c3 = c_a[-3], c_a[-2], c_a[-1]
    if (c1 > o1 and c2 > o2 and c3 > o3
            and c3 > c2 > c1 and o2 > o1 and o3 > o2):
        # Bonus if bodies are non-decreasing (acceleration)
        body1 = _body(o1, c1); body3 = _body(o3, c3)
        if body1 <= 0:
            return True, "LONG", 0.8
        strength = min(1.0, 0.6 + 0.4 * min(1.0, body3 / max(body1, 1e-12)))
        return True, "LONG", strength
    return False, "", 0.0


def three_black_crows(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """Three consecutive bearish candles with lower closes and falling opens."""
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 3:
        return False, "", 0.0
    o_a, _, _, c_a = prep
    o1, o2, o3 = o_a[-3], o_a[-2], o_a[-1]
    c1, c2, c3 = c_a[-3], c_a[-2], c_a[-1]
    if (c1 < o1 and c2 < o2 and c3 < o3
            and c3 < c2 < c1 and o2 < o1 and o3 < o2):
        body1 = _body(o1, c1); body3 = _body(o3, c3)
        if body1 <= 0:
            return True, "SHORT", 0.8
        strength = min(1.0, 0.6 + 0.4 * min(1.0, body3 / max(body1, 1e-12)))
        return True, "SHORT", strength
    return False, "", 0.0


def three_inside_up(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """
    Three-inside-up (bullish): harami then confirmation close above prev bar high.
    Higher-probability than raw engulfing because it waits for a third-bar confirm.
    """
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 3:
        return False, "", 0.0
    o_a, h_a, _, c_a = prep
    o1, c1 = o_a[-3], c_a[-3]
    o2, c2 = o_a[-2], c_a[-2]
    o3, c3 = o_a[-1], c_a[-1]
    body1 = _body(o1, c1)
    if body1 <= 0 or c1 >= o1:
        return False, "", 0.0
    # Inside harami: second bar bullish, body inside first body
    if not (c2 > o2 and max(o2, c2) < o1 and min(o2, c2) > c1):
        return False, "", 0.0
    # Confirmation: third closes above prev high
    if c3 > h_a[-2] and c3 > o3:
        return True, "LONG", 0.75
    return False, "", 0.0


def three_inside_down(opens, highs, lows, closes) -> Tuple[bool, str, float]:
    """Three-inside-down (bearish): mirror of three_inside_up."""
    prep = _prep((opens, highs, lows, closes))
    if prep is None or prep[0].size < 3:
        return False, "", 0.0
    o_a, _, l_a, c_a = prep
    o1, c1 = o_a[-3], c_a[-3]
    o2, c2 = o_a[-2], c_a[-2]
    o3, c3 = o_a[-1], c_a[-1]
    body1 = _body(o1, c1)
    if body1 <= 0 or c1 <= o1:
        return False, "", 0.0
    if not (c2 < o2 and max(o2, c2) < c1 and min(o2, c2) > o1):
        return False, "", 0.0
    if c3 < l_a[-2] and c3 < o3:
        return True, "SHORT", 0.75
    return False, "", 0.0


# ── Dispatcher ─────────────────────────────────────────────────────────

def detect_all(
    opens, highs, lows, closes,
    atr: float = 0.0,
    volumes: Optional[Sequence[float]] = None,
) -> list:
    """
    Run all detectors and return a list of
      {'pattern': str, 'direction': str, 'strength': float, 'volume_confirmed': bool}

    `volume_confirmed` is True when the signal bar volume exceeds 1.3 × the
    20-bar mean. It's a hint for consumers (aggregator / PriceAction); existing
    callers that only read pattern/direction/strength remain compatible.
    """
    # Volume confirmation (once, reused for every pattern)
    vol_confirmed = False
    try:
        if volumes is not None:
            v = np.asarray(volumes, dtype=float)
            if v.size >= 20 and np.all(np.isfinite(v[-20:])):
                baseline = float(np.median(v[-20:-1]))
                if baseline > 0 and v[-1] > baseline * 1.3:
                    vol_confirmed = True
    except Exception:
        vol_confirmed = False

    detectors = [
        ('PinBar',              lambda: pin_bar(opens, highs, lows, closes)),
        ('Engulfing',           lambda: engulfing(opens, highs, lows, closes)),
        ('InsideBar',           lambda: inside_bar(opens, highs, lows, closes)),
        ('PiercingLine',        lambda: piercing_line(opens, highs, lows, closes)),
        ('DarkCloudCover',      lambda: dark_cloud_cover(opens, highs, lows, closes)),
        ('TweezerTop',          lambda: tweezer_top(opens, highs, lows, closes)),
        ('TweezerBottom',       lambda: tweezer_bottom(opens, highs, lows, closes)),
        ('MorningStar',         lambda: morning_star(opens, highs, lows, closes, atr)),
        ('EveningStar',         lambda: evening_star(opens, highs, lows, closes, atr)),
        ('Doji',                lambda: doji(opens, highs, lows, closes)),
        ('ThreeWhiteSoldiers',  lambda: three_white_soldiers(opens, highs, lows, closes)),
        ('ThreeBlackCrows',     lambda: three_black_crows(opens, highs, lows, closes)),
        ('ThreeInsideUp',       lambda: three_inside_up(opens, highs, lows, closes)),
        ('ThreeInsideDown',     lambda: three_inside_down(opens, highs, lows, closes)),
    ]

    results = []
    for name, fn in detectors:
        try:
            detected, direction, strength = fn()
        except Exception:
            logger.debug("candlestick.%s failed", name, exc_info=True)
            continue
        if detected:
            results.append({
                'pattern': name,
                'direction': direction,
                'strength': float(strength),
                'volume_confirmed': vol_confirmed,
            })
    return results
