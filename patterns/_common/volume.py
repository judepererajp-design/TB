"""
Volume-confirmation helpers — a single source of truth for 'was the event
bar on expanded volume?'

Phase-2 wires these into geometric flag/triangle (where volume was previously
ignored) and keeps the cup-and-handle / H&S usage in sync.

Stdlib-only — no numpy dependency so tests using the conftest numpy mock can
still exercise the helpers.
"""

from typing import Optional, Sequence, Tuple


def compute_volume_stats(
    volumes: Optional[Sequence[float]],
    lookback: int = 20,
) -> Optional[Tuple[float, float]]:
    """
    Returns ``(current_vol, baseline_median)`` for the last ``lookback`` bars,
    or ``None`` when there aren't enough bars / the values aren't finite.

    Median (not mean) keeps a single outlier from pushing the baseline above
    the current bar and silently defeating the confirmation gate.
    """
    # Guard against numpy-ndarray truthiness: ``if not volumes`` would raise
    # ``ValueError: truth value of an array is ambiguous`` here, because
    # callers in patterns/geometric.py pass ``df['volume'].values`` (an
    # ndarray) directly. Handle None explicitly and let len() do the rest.
    if volumes is None:
        return None
    try:
        vols = list(volumes)
    except TypeError:
        return None
    if len(vols) < max(5, lookback // 2):
        return None

    window = vols[-lookback:] if len(vols) >= lookback else vols
    # Filter finite values
    finite = [v for v in window if _is_finite(v)]
    if len(finite) < max(5, lookback // 2):
        return None

    current = finite[-1]
    # Median of the preceding bars (exclude the current bar so the current
    # bar's volume is compared against a neutral baseline).
    baseline_pool = finite[:-1] if len(finite) > 1 else []
    if not baseline_pool:
        # Only a single finite bar — no baseline to compare against.
        return None
    baseline = _median(baseline_pool)
    if baseline <= 0:
        return None
    return float(current), float(baseline)


def volume_confirmed(
    volumes: Optional[Sequence[float]],
    mult: float = 1.3,
    lookback: int = 20,
) -> bool:
    """True iff last-bar volume > ``mult`` × median of trailing lookback."""
    stats = compute_volume_stats(volumes, lookback=lookback)
    if stats is None:
        return False
    current, baseline = stats
    return current > baseline * mult


# ── internals ─────────────────────────────────────────────────────────

def _is_finite(x) -> bool:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    # NaN != NaN; inf guards
    if f != f:
        return False
    if f == float('inf') or f == float('-inf'):
        return False
    return True


def _median(xs) -> float:
    arr = sorted(float(x) for x in xs)
    n = len(arr)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return arr[n // 2]
    return (arr[n // 2 - 1] + arr[n // 2]) / 2.0
