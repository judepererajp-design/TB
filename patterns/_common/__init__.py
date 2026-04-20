"""
Patterns shared helpers — Phase-2 audit.

Small, pure-function utilities used by candlestick.py / geometric.py /
harmonic.py / wyckoff.py.  Kept stdlib-only where possible so tests under
tests/conftest.py (which mocks numpy) continue to work for the helpers
that don't actually need numeric operations.
"""

from .pivots import find_alternating_pivots, is_isolated_swing
from .projection import clamp_projection, price_floor_valid
from .regime import regime_allows_structural, regime_penalty_for_pattern
from .volume import volume_confirmed, compute_volume_stats

__all__ = [
    "find_alternating_pivots",
    "is_isolated_swing",
    "clamp_projection",
    "price_floor_valid",
    "regime_allows_structural",
    "regime_penalty_for_pattern",
    "volume_confirmed",
    "compute_volume_stats",
]
