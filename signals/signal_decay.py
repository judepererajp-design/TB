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
        The caller should subtract this from ``final_confidence`` and
        clamp to ``DECAY_MIN_CONFIDENCE``.
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

    penalty = hours_past_grace * EG.DECAY_RATE_PER_HOUR
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
