"""
Shared pivot-point detection.

A single, tested implementation of the alternating swing-high / swing-low
detector. Previously this logic existed in-line in patterns/harmonic.py; it is
now reused by the geometric H&S and double-top/bottom detectors.

All three consumers want:
  * strict H/L alternation (never two consecutive same-kind pivots)
  * fractal isolation (N bars on each side strictly exceeded/undercut)
  * bar-index bookkeeping (for min-spacing or recency filters)

Returns ``list[(bar_index, price, kind)]`` with kind ∈ {'H', 'L'}.

The function is tolerant: it accepts plain Python lists or numpy arrays —
any indexable sequence of finite floats works.
"""

from typing import List, Sequence, Tuple


Pivot = Tuple[int, float, str]


def find_alternating_pivots(
    highs: Sequence[float],
    lows: Sequence[float],
    order: int = 5,
) -> List[Pivot]:
    """
    Detect alternating H/L pivots using a symmetric N-bars-each-side fractal.

    If two same-kind pivots occur back-to-back (possible in noisy plateau data),
    the more extreme one replaces the earlier one — this guarantees strict
    alternation in the returned list.
    """
    n = len(highs)
    if len(lows) != n or n == 0 or order < 1:
        return []

    tagged: List[Pivot] = []
    rng = range(1, order + 1)

    for i in range(order, n - order):
        hi = highs[i]
        lo = lows[i]

        is_swing_high = (
            all(hi > highs[i - j] for j in rng)
            and all(hi > highs[i + j] for j in rng)
        )
        is_swing_low = (
            all(lo < lows[i - j] for j in rng)
            and all(lo < lows[i + j] for j in rng)
        )

        if is_swing_high:
            if not tagged or tagged[-1][2] != 'H':
                tagged.append((i, float(hi), 'H'))
            elif hi > tagged[-1][1]:
                tagged[-1] = (i, float(hi), 'H')
        elif is_swing_low:
            if not tagged or tagged[-1][2] != 'L':
                tagged.append((i, float(lo), 'L'))
            elif lo < tagged[-1][1]:
                tagged[-1] = (i, float(lo), 'L')

    return tagged


def is_isolated_swing(
    series: Sequence[float],
    idx: int,
    order: int,
    direction: str = 'high',
) -> bool:
    """
    Standalone isolated-swing check — used by cup-rim validation etc.

    direction='high': series[idx] must strictly exceed the `order` bars on
                      each side.
    direction='low':  series[idx] must be strictly below.

    Returns False for out-of-range indices instead of raising — callers
    can use it as a gate.
    """
    n = len(series)
    if idx < order or idx + order >= n or order < 1:
        return False

    val = series[idx]
    if direction == 'high':
        return (
            all(val > series[idx - j] for j in range(1, order + 1))
            and all(val > series[idx + j] for j in range(1, order + 1))
        )
    if direction == 'low':
        return (
            all(val < series[idx - j] for j in range(1, order + 1))
            and all(val < series[idx + j] for j in range(1, order + 1))
        )
    return False
