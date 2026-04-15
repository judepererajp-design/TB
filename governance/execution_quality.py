"""
TitanBot Pro — Execution Quality Classifier
============================================
Classifies a closed loss by *how* it occurred, separating strategy
failures from execution failures.

Classification logic (from the four-tier model):

  SYSTEM_FAILURE   — entry was IN_ZONE but max_r ≈ 0; price moved
                     instantly against the position. Either the entry
                     zone was too wide, the signal was wrong, or there
                     was a sudden structural break.

  BAD_EXECUTION    — entry was LATE or EXTENDED (price already beyond
                     the intended zone) OR max_r < 0.5R regardless of
                     entry quality. The trade had limited upside from
                     the moment it was taken.

  BORDERLINE       — IN_ZONE entry with moderate max_r (0.5–0.8R).
                     The setup had some edge but market conditions or
                     timing cut the move short.

  GOOD_LOSS        — IN_ZONE entry with max_r ≥ 0.8R. The trade showed
                     real momentum before reversing; this is an
                     acceptable loss — execution was not the problem.

Usage
-----
    from governance.execution_quality import classify_loss

    result = classify_loss(entry_status="LATE", max_r=0.3)
    # result = {"class": "BAD_EXECUTION", "score": 0.25, "reason": "..."}
"""

from __future__ import annotations

from typing import Optional

# ─── Thresholds ────────────────────────────────────────────────────────────────
# max_r bands that define whether the trade showed genuine edge.
_MAX_R_SYSTEM_FAIL = 0.1   # price barely moved in our favour
_MAX_R_BAD_EXEC    = 0.5   # insufficient upside; trade had limited edge from entry
_MAX_R_BORDERLINE  = 0.8   # some edge but not enough to secure profit

# Entry status values that indicate sub-optimal execution
_LATE_STATUSES = {"LATE", "EXTENDED"}


def classify_loss(
    entry_status: Optional[str],
    max_r: Optional[float],
) -> dict:
    """
    Classify a loss outcome into one of four execution-quality tiers.

    Parameters
    ----------
    entry_status : str | None
        "IN_ZONE" | "LATE" | "EXTENDED" | None
    max_r : float | None
        Maximum favourable excursion reached during the trade (in R).

    Returns
    -------
    dict with keys:
        "class"  — str  : tier label
        "score"  — float: 0.0 (worst) … 1.0 (best execution quality)
        "reason" — str  : human-readable one-liner
    """
    es = (entry_status or "UNKNOWN").upper()
    mr = max_r if max_r is not None else 0.0

    # 1. SYSTEM_FAILURE — in-zone entry but price immediately reversed
    if es == "IN_ZONE" and mr <= _MAX_R_SYSTEM_FAIL:
        return {
            "class":  "SYSTEM_FAILURE",
            "score":  0.0,
            "reason": f"IN_ZONE entry but max_r={mr:.2f}R — instant reversal or signal failure",
        }

    # 2. BAD_EXECUTION — late/extended entry OR insufficient max_r
    if es in _LATE_STATUSES or mr < _MAX_R_BAD_EXEC:
        score = round(mr / _MAX_R_BAD_EXEC * 0.25, 3)   # 0.0 – 0.25
        if es in _LATE_STATUSES:
            reason = f"entry_status={es} — price was already beyond intended zone"
        else:
            reason = f"max_r={mr:.2f}R < {_MAX_R_BAD_EXEC}R — trade had limited edge from entry"
        return {"class": "BAD_EXECUTION", "score": score, "reason": reason}

    # 3. BORDERLINE — in-zone but moderate max_r
    if mr < _MAX_R_BORDERLINE:
        score = round(0.25 + (mr - _MAX_R_BAD_EXEC) / (_MAX_R_BORDERLINE - _MAX_R_BAD_EXEC) * 0.25, 3)
        return {
            "class":  "BORDERLINE",
            "score":  score,
            "reason": f"IN_ZONE entry, max_r={mr:.2f}R — some edge but move cut short",
        }

    # 4. GOOD_LOSS — in-zone entry and meaningful max_r; execution was fine
    score = round(min(0.5 + (mr - _MAX_R_BORDERLINE) * 0.5, 1.0), 3)
    return {
        "class":  "GOOD_LOSS",
        "score":  score,
        "reason": f"IN_ZONE entry, max_r={mr:.2f}R — acceptable loss; execution not the problem",
    }
