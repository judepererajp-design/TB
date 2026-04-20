"""
analyzers._common — cross-cutting infrastructure for analyzer modules.

Re-exports the small building blocks every analyzer should use instead
of re-implementing its own (``Freshness``, ``TTLCache``, safe stats).
"""

from analyzers._common.freshness import (
    Freshness,
    FreshnessStatus,
    now_ts,
    staleness,
)
from analyzers._common.cache import (
    CacheEntry,
    CacheState,
    EMPTY,
    MISS,
    TTLCache,
)
from analyzers._common.stats import (
    blom_percentile,
    clamp,
    is_finite,
    log_zscore,
    mad,
    mean,
    nan_to_num,
    percentile_rank,
    safe_div,
    stdev,
    zscore,
)

__all__ = [
    # freshness
    "Freshness",
    "FreshnessStatus",
    "now_ts",
    "staleness",
    # cache
    "TTLCache",
    "CacheState",
    "CacheEntry",
    "MISS",
    "EMPTY",
    # stats
    "safe_div",
    "clamp",
    "nan_to_num",
    "mean",
    "stdev",
    "zscore",
    "log_zscore",
    "mad",
    "blom_percentile",
    "percentile_rank",
    "is_finite",
]
