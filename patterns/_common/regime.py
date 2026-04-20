"""
Regime-gating helpers — used by patterns/* to down-weight or reject
signals in regimes that historically fade for that pattern family.

Previously each detector declared its own ``VALID_REGIMES`` set but treated
every allowed regime identically.  Phase-2 splits this into:

  * ``regime_allows_structural(pattern_type, regime)`` — hard gate.
  * ``regime_penalty_for_pattern(pattern_type, regime)`` — soft
    confidence subtraction for historically weaker regimes.

``pattern_type`` is one of the literal strings below; unknown types default
to allow with zero penalty.
"""

from typing import Optional


# Patterns that are reversal-at-range-edge setups — they work best in
# non-trending regimes (CHOPPY) and fade inside strong directional regimes.
_REVERSAL = {"harmonic", "wyckoff_spring", "wyckoff_utad",
             "double_top", "double_bottom", "head_shoulders", "inverse_hs",
             "rising_wedge", "falling_wedge"}

# Patterns that are continuation setups — they work best inside a trend.
_CONTINUATION = {"bull_flag", "bear_flag", "sym_triangle",
                 "cup_and_handle"}


def regime_allows_structural(pattern_type: str, regime: Optional[str]) -> bool:
    """
    Hard gate.  Returns False for the rare cases where a pattern is
    structurally incompatible with a regime (e.g. bull flag during
    BEAR_TREND).  Returns True otherwise — including UNKNOWN regimes.
    """
    if regime is None:
        return True
    r = str(regime).upper()
    pt = pattern_type.lower()

    # Continuation: don't trade counter-trend continuation setups.
    if pt == "bull_flag" and r == "BEAR_TREND":
        return False
    if pt == "bear_flag" and r == "BULL_TREND":
        return False
    if pt == "cup_and_handle" and r == "BEAR_TREND":
        return False

    # Reversal patterns: short setups in strong BULL_TREND (and vice versa)
    # rarely work — gate them.
    if pt in ("head_shoulders", "double_top", "rising_wedge") and r == "BULL_TREND":
        return False
    if pt in ("inverse_hs", "double_bottom", "falling_wedge") and r == "BEAR_TREND":
        return False

    return True


def regime_penalty_for_pattern(pattern_type: str, regime: Optional[str]) -> float:
    """
    Soft down-weight (in confidence points) for historically weaker
    regime/pattern combos. Returns 0 when neutral.

    Callers may subtract the returned value from the raw detector confidence,
    or ignore it.  Magnitudes are intentionally modest (≤8) so they don't
    overwhelm the primary confluence scoring.
    """
    if regime is None:
        return 0.0
    r = str(regime).upper()
    pt = pattern_type.lower()

    # Choppy / unknown environments hurt continuation quality
    if pt in _CONTINUATION and r in ("CHOPPY", "VOLATILE", "UNKNOWN"):
        return 5.0

    # Trending environments hurt reversals (even when not gated outright)
    if pt in _REVERSAL and r in ("BULL_TREND", "BEAR_TREND"):
        return 4.0

    return 0.0
