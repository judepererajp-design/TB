"""
Projection / price-floor helpers.

Rationale (BUG-2, Phase-1):
  On very low-priced tokens (e.g. BONK @ $0.00002) a raw measured-move
  projection — cup depth, wedge range, impulse range — can exceed the
  entry price itself, producing negative target or stop prices and absurd
  downstream percentage calculations.

Phase-2 moves the clamp helper out of GeometricPatterns so harmonic and
wyckoff can reuse it for their own projected targets (cause-effect, etc.).
"""

from typing import Optional


def clamp_projection(raw: float, entry: float, atr: float = 0.0) -> float:
    """
    Bound a measured-move distance to something price-sane.

    cap = max(entry * 0.5, 3 * ATR)   when ATR > 0
    cap = entry * 0.5                 otherwise

    Returns max(0, min(raw, cap)). Negative or NaN inputs are clamped to 0.
    """
    try:
        r = float(raw)
        e = float(entry)
    except (TypeError, ValueError):
        return 0.0

    if e <= 0:
        return max(r, 0.0)

    # NaN check (NaN != NaN)
    if r != r:
        return 0.0

    cap = e * 0.5
    if atr and atr > 0:
        cap = max(cap, float(atr) * 3.0)

    return min(max(r, 0.0), cap)


def price_floor_valid(
    entry: float,
    *levels: float,
    floor_pct: float = 0.001,
) -> bool:
    """
    Returns True iff every target level is at least ``floor_pct`` of entry.

    Used as a defense-in-depth gate after a signal is constructed: even when
    individual projections are clamped, combinations (TP2 + negative delta
    from TP1) could produce below-floor values. Rejecting the whole signal
    is safer than emitting one with suspect geometry.
    """
    if entry <= 0:
        return False
    threshold = entry * floor_pct
    for lvl in levels:
        if lvl is None:
            continue
        try:
            if float(lvl) <= threshold:
                return False
        except (TypeError, ValueError):
            return False
    return True
