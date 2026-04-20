"""
analyzers._common.stats
=======================
Numerically-safe statistical primitives used across the analyzer folder.

Why this module exists
----------------------
The audit flagged dozens of sites where inline stats were subtly wrong:
* ``x / y`` without ``y == 0`` guard
* ``(x - mean) / stdev`` with ``stdev`` computed from a 1-sample window
* Blom-style percentile formulas off by one sample
* percentile-rank returning ``nan`` for a single-element distribution
* volume / funding z-scores computed on the raw value (power-law) rather
  than on log-volume

Rather than fix each site individually, every analyzer should call into
these helpers. Each function:

* never raises on pathological input (empty list, all-NaN, σ≈0). Instead
  returns a documented fallback (``0.0``, ``None``, or the caller's
  supplied default).
* works on plain Python sequences so it runs under the test-suite's
  numpy mock (see ``tests/conftest.py``).
* uses ``float('nan')`` / ``float('inf')`` checks via :func:`math.isfinite`
  not identity comparisons.

All functions are pure and side-effect-free.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

_TWO: float = 2.0


# ──────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────


def is_finite(x: object) -> bool:
    """Like :func:`math.isfinite` but tolerant of non-numeric input.

    Returns ``False`` for ``None``, strings, NaN, and ±Inf.
    """
    if x is None:
        return False
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def safe_div(
    numerator: float,
    denominator: float,
    default: float = 0.0,
    *,
    min_denominator: float = 1e-12,
) -> float:
    """Divide, returning ``default`` on zero/near-zero/NaN denominator.

    Use ``min_denominator`` to guard against catastrophic blow-up when
    the denominator is technically non-zero but so small the quotient is
    meaningless. The default (``1e-12``) protects float64 precision.
    """
    if not is_finite(numerator) or not is_finite(denominator):
        return default
    if abs(denominator) < min_denominator:
        return default
    return numerator / denominator


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp ``x`` into ``[lo, hi]``. NaN input returns ``lo`` (not NaN)."""
    if not is_finite(x):
        return lo
    if lo > hi:
        lo, hi = hi, lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def nan_to_num(x: object, default: float = 0.0) -> float:
    """Coerce any non-finite value to ``default``. Accepts None / strings."""
    if is_finite(x):
        return float(x)  # type: ignore[arg-type]
    return default


# ──────────────────────────────────────────────────────────────────
# Aggregates
# ──────────────────────────────────────────────────────────────────


def _finite_values(values: Iterable[object]) -> List[float]:
    """Keep only finite numeric entries (drops NaN, Inf, None, strings)."""
    out: List[float] = []
    for v in values:
        if is_finite(v):
            out.append(float(v))  # type: ignore[arg-type]
    return out


def mean(values: Sequence[object], default: float = 0.0) -> float:
    """Arithmetic mean of finite entries. Empty → ``default``."""
    v = _finite_values(values)
    if not v:
        return default
    return sum(v) / len(v)


def stdev(
    values: Sequence[object],
    default: float = 0.0,
    *,
    ddof: int = 1,
) -> float:
    """Sample (``ddof=1``) or population (``ddof=0``) standard deviation.

    Returns ``default`` when fewer than ``ddof + 1`` finite samples are
    available. Never raises.
    """
    v = _finite_values(values)
    n = len(v)
    if n <= ddof:
        return default
    m = sum(v) / n
    ss = sum((x - m) ** 2 for x in v)
    denom = n - ddof
    if denom <= 0:
        return default
    var = ss / denom
    if var < 0 or not math.isfinite(var):
        return default
    return math.sqrt(var)


def mad(
    values: Sequence[object],
    default: float = 0.0,
    *,
    scale: bool = True,
) -> float:
    """Median Absolute Deviation.

    ``scale=True`` multiplies by the normal-consistency constant
    ``1/Φ⁻¹(0.75) ≈ 1.4826`` so that, for Gaussian data, ``mad ≈ σ``.
    This is the preferred fallback when sample σ is degenerate
    (≈0 or inflated by a single outlier).
    """
    v = _finite_values(values)
    n = len(v)
    if n == 0:
        return default
    v_sorted = sorted(v)
    med = _median_sorted(v_sorted)
    abs_dev = sorted(abs(x - med) for x in v)
    med_abs = _median_sorted(abs_dev)
    if scale:
        return 1.4826 * med_abs
    return med_abs


def _median_sorted(v_sorted: Sequence[float]) -> float:
    n = len(v_sorted)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(v_sorted[mid])
    return (float(v_sorted[mid - 1]) + float(v_sorted[mid])) / 2.0


# ──────────────────────────────────────────────────────────────────
# Z-scores
# ──────────────────────────────────────────────────────────────────


def zscore(
    x: float,
    population: Sequence[object],
    *,
    default: float = 0.0,
    min_stdev: float = 1e-9,
    use_mad_fallback: bool = True,
) -> float:
    """Compute ``(x - mean) / σ`` with graceful degradation.

    * ``x`` non-finite → ``default``
    * ``population`` has <2 finite samples → ``default``
    * ``σ < min_stdev`` → if ``use_mad_fallback``, fall back to MAD;
      otherwise return ``default``.

    Result is clamped to ``[-10, +10]`` to prevent numeric blow-up from
    contaminating downstream weighted sums. Callers that legitimately
    need unclamped z-scores should compute manually.
    """
    if not is_finite(x):
        return default
    pop = _finite_values(population)
    if len(pop) < 2:
        return default
    m = sum(pop) / len(pop)
    s = stdev(pop, default=0.0, ddof=1)
    if s < min_stdev:
        if use_mad_fallback:
            s = mad(pop, default=0.0)
        if s < min_stdev:
            return default
    z = (float(x) - m) / s
    return clamp(z, -10.0, 10.0)


def log_zscore(
    x: float,
    population: Sequence[object],
    *,
    default: float = 0.0,
    floor: float = 1e-9,
) -> float:
    """Z-score on ``log(x)`` against ``log(population)``.

    Use this for values that are heavy-tailed / power-law distributed
    (volume, open-interest, funding basis) where a raw z-score is
    dominated by the fattest outliers.

    Non-positive values (``≤ floor``) are clamped to ``floor`` before
    taking ``log``, which is the standard treatment for log-volume.
    """
    if not is_finite(x):
        return default
    try:
        lx = math.log(max(float(x), floor))
    except (ValueError, OverflowError):
        return default
    logs: List[float] = []
    for v in population:
        if is_finite(v):
            fv = float(v)  # type: ignore[arg-type]
            try:
                logs.append(math.log(max(fv, floor)))
            except (ValueError, OverflowError):
                continue
    if len(logs) < 2:
        return default
    return zscore(lx, logs, default=default)


# ──────────────────────────────────────────────────────────────────
# Percentiles
# ──────────────────────────────────────────────────────────────────


def blom_percentile(
    x: float,
    population: Sequence[object],
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    """Blom-adjusted percentile rank of ``x`` in ``population``, in ``[0, 1]``.

    Uses the plotting-position formula ``(rank - 3/8) / (n + 1/4)`` which
    is unbiased for Gaussian populations and well-behaved for tiny ``n``.
    Ties are handled by midrank (average rank of equal values).

    Returns ``default`` (``None`` unless overridden) when the population
    is empty or ``x`` is non-finite — this is deliberate; callers must
    decide whether "unknown percentile" means 0.5, 0.0, or "reject".
    """
    if not is_finite(x):
        return default
    pop = _finite_values(population)
    n = len(pop)
    if n == 0:
        return default
    fx = float(x)
    below = sum(1 for v in pop if v < fx)
    equal = sum(1 for v in pop if v == fx)
    # Midrank: count half of the ties.
    rank = below + 0.5 * equal + 0.5  # +0.5 to place at midrank of its own cell
    blom = (rank - 3.0 / 8.0) / (n + 1.0 / 4.0)
    return clamp(blom, 0.0, 1.0)


def percentile_rank(
    x: float,
    population: Sequence[object],
    *,
    default: float = 0.5,
) -> float:
    """Ordinary percentile rank (``fraction below or equal``) in ``[0, 1]``.

    Unlike :func:`blom_percentile`, this is the naive definition and is
    bounded ``[0, 1]`` inclusive. Use :func:`blom_percentile` when the
    result will feed statistical thresholds; use this when you just want
    "how extreme is this value".
    """
    if not is_finite(x):
        return default
    pop = _finite_values(population)
    n = len(pop)
    if n == 0:
        return default
    fx = float(x)
    leq = sum(1 for v in pop if v <= fx)
    return clamp(leq / n, 0.0, 1.0)


__all__ = [
    "is_finite",
    "safe_div",
    "clamp",
    "nan_to_num",
    "mean",
    "stdev",
    "mad",
    "zscore",
    "log_zscore",
    "blom_percentile",
    "percentile_rank",
]
