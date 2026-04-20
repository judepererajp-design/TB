"""
analyzers._common.freshness
===========================
Every analyzer public return should carry a *freshness tag* so that
downstream consumers (aggregator, execution_gate, strategies) can
decide whether to trust, down-weight, or reject the signal based on
how old the underlying data is.

Using a dedicated dataclass rather than an ad-hoc ``age_seconds`` float
prevents the common mistakes of:

* forgetting to propagate the data source (so you can't tell whether
  "funding" came from the primary exchange or a fallback),
* comparing ``time.time() - ts`` inconsistently (some callers used
  wall-clock, others monotonic),
* ignoring the ``stale`` decision entirely because every consumer had
  its own staleness threshold.

This module is intentionally dependency-free — stdlib only — so it can
be imported at the very top of every analyzer module without incurring
numpy/pandas import cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FreshnessStatus(str, Enum):
    """Coarse freshness label. Use ``enum identity`` (``status is FRESH``)
    rather than string comparison to avoid typo-silent-match bugs."""

    FRESH = "FRESH"        # age ≤ ttl_seconds
    AGING = "AGING"        # ttl < age ≤ 2 × ttl (soft-degraded; use but down-weight)
    STALE = "STALE"        # age > 2 × ttl OR explicitly marked stale (do not use)
    UNKNOWN = "UNKNOWN"    # age_seconds could not be computed (no timestamp)


def now_ts() -> float:
    """Wall-clock seconds-since-epoch. Centralised so tests can monkey-patch."""
    return time.time()


def staleness(age_seconds: Optional[float], ttl_seconds: float) -> FreshnessStatus:
    """Map an age + ttl to a three-state freshness label.

    * ``None`` age → ``UNKNOWN`` (caller forgot to stamp the value)
    * negative age → ``UNKNOWN`` (clock skew — don't lie about being fresh)
    * age ≤ ttl → ``FRESH``
    * ttl < age ≤ 2·ttl → ``AGING``
    * age > 2·ttl → ``STALE``
    """
    if age_seconds is None:
        return FreshnessStatus.UNKNOWN
    try:
        a = float(age_seconds)
    except (TypeError, ValueError):
        return FreshnessStatus.UNKNOWN
    if a != a:  # NaN
        return FreshnessStatus.UNKNOWN
    if a < 0:
        return FreshnessStatus.UNKNOWN
    if ttl_seconds <= 0:
        # Degenerate ttl — treat anything > 0 as stale
        return FreshnessStatus.STALE if a > 0 else FreshnessStatus.FRESH
    if a <= ttl_seconds:
        return FreshnessStatus.FRESH
    if a <= 2.0 * ttl_seconds:
        return FreshnessStatus.AGING
    return FreshnessStatus.STALE


@dataclass(frozen=True)
class Freshness:
    """Immutable freshness tag attached to every analyzer return.

    Attributes
    ----------
    age_seconds
        Seconds since the underlying data was produced. ``None`` when the
        analyzer could not compute one (e.g. first call after boot).
    source
        Short machine-readable tag identifying where the data came from
        (``"binance"``, ``"coingecko"``, ``"local-cache"``, …). Never put
        a human-readable sentence here — consumers may pattern-match.
    stale
        Explicit staleness flag. ``True`` means the producer itself knows
        the value is too old to act on (e.g. websocket timed out). This
        is *independent* of the ``age_seconds`` / ``ttl`` check — a
        producer can force ``stale=True`` even if age is low.
    ttl_seconds
        The TTL the producer considers "fresh". Used by ``status()`` to
        derive a three-state label without asking the consumer to know
        the right number.
    """

    age_seconds: Optional[float] = None
    source: str = "unknown"
    stale: bool = False
    ttl_seconds: float = 60.0

    def status(self) -> FreshnessStatus:
        """Return the three-state freshness label, honoring ``self.stale``."""
        if self.stale:
            return FreshnessStatus.STALE
        return staleness(self.age_seconds, self.ttl_seconds)

    def is_fresh(self) -> bool:
        return self.status() is FreshnessStatus.FRESH

    def is_actionable(self) -> bool:
        """True when the value is fresh *or* aging (usable with caution).
        ``STALE`` and ``UNKNOWN`` are both rejected here — callers that
        want to accept ``UNKNOWN`` should check ``status()`` directly."""
        s = self.status()
        return s is FreshnessStatus.FRESH or s is FreshnessStatus.AGING

    @classmethod
    def fresh(cls, source: str, ttl_seconds: float = 60.0) -> "Freshness":
        """Build a Freshness for a value just produced (age 0)."""
        return cls(age_seconds=0.0, source=source, stale=False, ttl_seconds=ttl_seconds)

    @classmethod
    def from_timestamp(
        cls,
        produced_at: Optional[float],
        source: str,
        ttl_seconds: float = 60.0,
        now: Optional[float] = None,
    ) -> "Freshness":
        """Build a Freshness from the wall-clock time the data was produced.

        ``produced_at`` ``None`` yields an ``UNKNOWN``-age freshness with
        ``stale=True`` (no timestamp => can't trust).
        """
        if produced_at is None:
            return cls(age_seconds=None, source=source, stale=True, ttl_seconds=ttl_seconds)
        t = now if now is not None else now_ts()
        age = t - float(produced_at)
        if age < 0:
            # Clock skew: don't pretend we got a value from the future.
            age = 0.0
        return cls(age_seconds=age, source=source, stale=False, ttl_seconds=ttl_seconds)

    @classmethod
    def unknown(cls, source: str = "unknown") -> "Freshness":
        """Sentinel for analyzers that have no data at all (not even a timestamp)."""
        return cls(age_seconds=None, source=source, stale=True, ttl_seconds=60.0)

    def to_dict(self) -> dict:
        """JSON-serialisable representation for logging / metrics."""
        return {
            "age_seconds": self.age_seconds,
            "source": self.source,
            "stale": self.stale,
            "ttl_seconds": self.ttl_seconds,
            "status": self.status().value,
        }


__all__ = ["Freshness", "FreshnessStatus", "now_ts", "staleness"]
