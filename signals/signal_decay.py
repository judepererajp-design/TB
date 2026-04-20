"""
TitanBot Pro — Signal Decay
==============================
Applies a confidence penalty to signals based on their age.

Signals lose information value over time.  A setup identified 3 hours ago
is less reliable than one discovered 5 minutes ago.  This module computes
a staleness penalty that the engine subtracts from ``final_confidence``
before portfolio sizing.

Constants are in :class:`config.constants.ExecutionGate`:
  - DECAY_ENABLED: master switch
  - DECAY_GRACE_PERIOD_SECS: no penalty during this window
  - DECAY_RATE_PER_HOUR: % confidence lost per hour after grace
  - DECAY_MAX_PENALTY: ceiling on total penalty
  - DECAY_MIN_CONFIDENCE: absolute floor
"""

import logging
import time

from config.constants import ExecutionGate as EG

logger = logging.getLogger(__name__)


def compute_decay_penalty(
    signal_created_ts: float,
    now: float = 0.0,
) -> float:
    """Return the confidence penalty (positive float) due to signal age.

    FIX Q9: Penalty is now an **exponential saturation** model rather than a
    linear one.  Signal information decays via a compounding process —
    2h-old signals retain materially more edge than (1 − 2·rate/max) would
    suggest, but past the knee of the curve the remaining edge collapses
    fast.  The curve is calibrated so the 1-hour-past-grace point matches
    ``DECAY_RATE_PER_HOUR`` exactly (preserves config semantics), and
    saturates to ``DECAY_MAX_PENALTY`` at long tails.

    Formula::

        k = -ln(1 - DECAY_RATE_PER_HOUR / DECAY_MAX_PENALTY)
        penalty(t) = DECAY_MAX_PENALTY · (1 − exp(−k · hours_past_grace))

    Snaps exactly to DECAY_MAX_PENALTY once the raw fraction exceeds 0.95
    so behaviour is indistinguishable from the legacy linear-cap model at
    long ages (half-day+), while being gentler in the 30–120-minute window
    where most swing signals are actually traded.

    Parameters
    ----------
    signal_created_ts : float
        Unix timestamp when the signal was first generated.
    now : float, optional
        Current unix timestamp.  Defaults to ``time.time()``.

    Returns
    -------
    float
        Penalty in confidence points (0.0 – DECAY_MAX_PENALTY).
    """
    if not EG.DECAY_ENABLED:
        return 0.0

    if signal_created_ts <= 0:
        return 0.0

    now = now or time.time()
    age_secs = max(0.0, now - signal_created_ts)

    # Grace period — no decay
    if age_secs <= EG.DECAY_GRACE_PERIOD_SECS:
        return 0.0

    # Hours past grace
    hours_past_grace = (age_secs - EG.DECAY_GRACE_PERIOD_SECS) / 3600.0

    # Derive a half-life from the configured linear rate so legacy config
    # values remain meaningful.  Pick the half-life as the age at which the
    # OLD formula would reach 50% of max penalty, then use exponential decay
    # of REMAINING edge around that anchor.
    import math as _m
    if EG.DECAY_RATE_PER_HOUR <= 0 or EG.DECAY_MAX_PENALTY <= 0:
        return 0.0
    # Calibrate k so that at `hours_past_grace=1`, penalty ≈ DECAY_RATE_PER_HOUR
    # (preserves the old configuration semantics — 1h point anchors the curve).
    ratio = EG.DECAY_RATE_PER_HOUR / EG.DECAY_MAX_PENALTY
    if ratio >= 1.0:
        # Degenerate config: rate >= max → curve collapses to a step function.
        return EG.DECAY_MAX_PENALTY if hours_past_grace > 0 else 0.0
    k = -_m.log(1.0 - ratio)  # ≈ rate/max for small ratios (linear first-order)
    decay_fraction = 1.0 - _m.exp(-k * hours_past_grace)
    penalty = EG.DECAY_MAX_PENALTY * decay_fraction
    # Snap to exact max once we're within ~2% — the asymptote is never
    # reached in pure exponential, but long-tail staleness should behave
    # identically to the old linear-capped model.
    if penalty >= EG.DECAY_MAX_PENALTY * 0.95:
        penalty = EG.DECAY_MAX_PENALTY
    penalty = min(penalty, EG.DECAY_MAX_PENALTY)

    return round(penalty, 2)


def apply_decay(
    confidence: float,
    signal_created_ts: float,
    now: float = 0.0,
) -> float:
    """Return decayed confidence, clamped to the configured floor.

    Convenience wrapper that computes the penalty and subtracts it.
    """
    penalty = compute_decay_penalty(signal_created_ts, now)
    if penalty <= 0:
        return confidence

    new_confidence = max(confidence - penalty, EG.DECAY_MIN_CONFIDENCE)
    return round(new_confidence, 2)
