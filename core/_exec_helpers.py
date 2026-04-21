"""
TitanBot Pro — Execution Lifecycle Helpers
===========================================
Pure helper functions extracted from execution_engine.py so they can be
unit-tested under tests/conftest.py's lightweight numpy mock and composed
by callers outside the execution loop.

Every function here is stdlib-only (no numpy) and has no side effects.
"""

from __future__ import annotations

from typing import Tuple


# ── Gap 1 — Volatility-aware trigger window ───────────────────────────
def vol_scale_factor(atr_pct_recent: float, atr_pct_baseline: float,
                     floor: float = 0.5, cap: float = 2.0) -> float:
    """Ratio of recent ATR % to baseline ATR %, clamped to [floor, cap].

    Used to stretch or shorten the trigger-accumulation window based on
    how "fast" the market is currently moving relative to its own recent
    history.  The floor (0.5×) ensures vol-expansion days still get a
    minimum window; the cap (2.0×) prevents low-vol creep from stretching
    the window indefinitely.
    """
    try:
        r = float(atr_pct_recent)
        b = float(atr_pct_baseline)
    except (TypeError, ValueError):
        return 1.0
    if not (r > 0 and b > 0):
        return 1.0
    return max(floor, min(cap, r / b))


def scale_trigger_window(base_secs: int, atr_pct_recent: float,
                         atr_pct_baseline: float,
                         min_secs: int = 20 * 60,
                         max_secs: int = 48 * 3600) -> int:
    """Apply vol_scale_factor to a base window, respecting hard [min, max].

    When realized volatility is high (recent > baseline), the scale factor
    is > 1 — but the window is *shortened*, not stretched, because triggers
    should confluence quicker in fast markets.  Symmetrically, slow markets
    get more time.  This matches what a human discretionary trader does:
    raise attention in vol, relax attention in chop.
    """
    s = vol_scale_factor(atr_pct_recent, atr_pct_baseline)
    if s <= 0:
        return int(base_secs)
    # High vol (s > 1) shortens window; low vol (s < 1) stretches it.
    # Inverse scaling preserves the "more time when slow" intuition.
    scaled = int(base_secs / s)
    return max(int(min_secs), min(int(max_secs), scaled))


# ── Symmetric trigger adaptation ──────────────────────────────────────
def trigger_adjustment_for_regime(
    direction: str,
    regime: str,
    weekly_bias: str,
    weekly_adx: float,
    *,
    base_min_triggers: int = 2,
    floor: int = 1,
) -> Tuple[int, str]:
    """Return (adjusted_min_triggers, note) based on (regime × direction) fit.

    Mirrors the counter-trend ratchet already in execution_engine.track()
    with a with-trend *discount* — a LONG in a confirmed BULL_TREND+BULLISH
    weekly bias can fire on one fewer trigger (floored at 1).  This is the
    symmetry the audit called out: the engine previously only tightened for
    counter-trend and never loosened for with-trend continuation.
    """
    try:
        adx = float(weekly_adx or 0.0)
    except (TypeError, ValueError):
        adx = 0.0
    d = str(direction or "").upper()
    r = str(regime or "").upper()
    b = str(weekly_bias or "").upper()

    is_counter = (
        (d == "LONG" and b == "BEARISH") or
        (d == "SHORT" and b == "BULLISH")
    )
    is_with_trend = (
        (d == "LONG" and b == "BULLISH" and r == "BULL_TREND") or
        (d == "SHORT" and b == "BEARISH" and r == "BEAR_TREND")
    )

    # Counter-trend should tighten in *any* confirmed trending or choppy
    # regime — both BEAR_TREND (original LONG vs bearish week) and
    # BULL_TREND (SHORT vs bullish week) qualify.  Choppy still counts
    # because counter-trend in chop tends to chop-out the stop.
    if is_counter and r in ("BEAR_TREND", "BULL_TREND", "CHOPPY") and adx >= 25:
        return (max(base_min_triggers, 2),
                f"counter-trend {d}/{r}/{b} adx={adx:.0f} → ratchet up")
    if is_with_trend and adx >= 25:
        adjusted = max(int(floor), int(base_min_triggers) - 1)
        if adjusted < base_min_triggers:
            return (adjusted,
                    f"with-trend {d}/{r}/{b} adx={adx:.0f} → discount "
                    f"{base_min_triggers}→{adjusted}")
    return (int(base_min_triggers), "")


# ── Gap 5 — Magnitude-weighted trigger quality ────────────────────────
def magnitude_multiplier(observed: float, baseline: float,
                         floor: float = 0.5, cap: float = 2.0) -> float:
    """Scale a trigger's nominal weight by how much it exceeds its baseline.

    Example: nominal liq-reaction weight is 1.0, and the observed sweep is
    $5M vs a median sweep of $500K on this symbol.  magnitude_multiplier
    returns 2.0 (capped), so the weight becomes 2.0 instead of 1.0 — a
    real whale event gets the credit it deserves.  A $200K sweep on the
    same symbol returns 0.5 and counts half — it's valid but not decisive.
    """
    try:
        o = float(observed)
        b = float(baseline)
    except (TypeError, ValueError):
        return 1.0
    if not (b > 0):
        return 1.0
    return max(floor, min(cap, o / b))


# ── Gap 3 — RR-decay expire reason helpers ────────────────────────────
def compute_rr(direction: str, entry: float, stop_loss: float,
               take_profit: float) -> float:
    """Reward-to-risk given directional entry/SL/TP.

    Returns 0.0 on malformed inputs (division-by-zero, wrong-side levels).
    The execution engine uses this to detect RR decay as new swing
    pivots form while the signal sits in WATCHING/ENTRY_ZONE.
    """
    try:
        e = float(entry)
        sl = float(stop_loss)
        tp = float(take_profit)
    except (TypeError, ValueError):
        return 0.0
    d = str(direction or "").upper()
    if d == "LONG":
        risk = e - sl
        reward = tp - e
    elif d == "SHORT":
        risk = sl - e
        reward = e - tp
    else:
        return 0.0
    if risk <= 0 or reward <= 0:
        return 0.0
    return reward / risk


#: Buffer (as a fraction) used when tightening an SL past a newly formed
#: swing pivot.  15 bps keeps us just below (LONG) or just above (SHORT) the
#: pivot — close enough to preserve the risk tightening but with margin for
#: wick noise that would otherwise stop us out prematurely.
SL_TIGHTENING_BUFFER_PCT: float = 0.0015


def tightened_sl(direction: str, original_sl: float, swing_pivot: float,
                 buffer_pct: float = SL_TIGHTENING_BUFFER_PCT) -> float:
    """Return the *tighter* of (original_sl, swing_pivot ± buffer).

    For a LONG, tighter means higher (closer to entry).  For a SHORT,
    tighter means lower.  Never loosens an SL — that would violate the
    "no lower standards" rule.  Returns original_sl on malformed input.
    """
    try:
        o = float(original_sl)
        p = float(swing_pivot)
        buf = float(buffer_pct)
    except (TypeError, ValueError):
        return float(original_sl)
    d = str(direction or "").upper()
    if d == "LONG":
        # Candidate SL just below the swing low
        candidate = p * (1.0 - buf)
        return max(o, candidate)  # tighter = higher
    if d == "SHORT":
        candidate = p * (1.0 + buf)
        return min(o, candidate)  # tighter = lower
    return o


# ── Option B tiered auto-approval window ──────────────────────────────
#: Auto-apply LOW-risk proposals after this many seconds without a veto.
AUTO_APPLY_LOW_RISK_VETO_SECS: int = 30 * 60


def auto_apply_at_for_risk(risk_level: str, now: float) -> float:
    """Return the timestamp after which a proposal auto-applies, or 0.

    0 means "always requires manual approval".  MEDIUM and HIGH both return
    0 — only LOW is auto-applied after the veto window.
    """
    r = str(risk_level or "").upper()
    if r == "LOW":
        return float(now) + float(AUTO_APPLY_LOW_RISK_VETO_SECS)
    return 0.0
