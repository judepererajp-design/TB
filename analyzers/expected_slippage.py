"""
TitanBot Pro — Predictive Slippage Model
==========================================
Estimates expected execution slippage *before* trade entry, so the
fee-adjusted RR gate can reject setups whose predicted (rather than
historical) friction would eat the edge.

Inputs (all optional — model degrades gracefully):
  * spread_bps           — observed bid-ask spread in basis points
  * atr_pct              — recent volatility as % of price (e.g. ATR / price)
  * size_usd             — intended trade notional
  * top_book_depth_usd   — sum of bid+ask USD volume in the top N levels
                            within ~25 bps of mid

Output:
  expected_slippage_pct (decimal, e.g. 0.0008 = 8 bps)

Formula (simple, transparent, defensible):

  slip = base_half_spread + impact_term + vol_term

where
  base_half_spread = spread_bps / 2 / 10_000        (you cross half the spread)
  impact_term      = K_IMPACT * sqrt(size_usd / depth_usd)  (square-root market impact)
  vol_term         = K_VOL * atr_pct                 (extra slip in fast tape)

The K_* constants are exposed in `config.constants.ExpectedSlippage` so they
can be calibrated against `core.slippage_tracker` history without code
changes. When inputs are missing the function falls back to the existing
`Backtester.DEFAULT_SLIPPAGE_PCT` so callers always get a sane number.

This is NOT a learned model — by design, it's a transparent first-principles
estimator. Future work can layer a regression on top fed by SlippageTracker.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Optional

from config.constants import Backtester, ExpectedSlippage as _ES

logger = logging.getLogger(__name__)


@dataclass
class SlippageEstimate:
    """Result of a slippage prediction."""
    expected_pct: float          # decimal, e.g. 0.0008 = 8 bps
    half_spread_pct: float
    impact_pct: float
    vol_pct: float
    inputs_known: int            # 0..3 — how many of {spread, depth, vol} were supplied
    fallback: bool               # True if model fell back to Backtester default


def estimate_slippage(
    *,
    spread_bps: Optional[float] = None,
    atr_pct: Optional[float] = None,
    size_usd: Optional[float] = None,
    top_book_depth_usd: Optional[float] = None,
) -> SlippageEstimate:
    """Predict execution slippage as a fraction of price.

    All inputs are optional. The estimator returns a fallback value
    (Backtester.DEFAULT_SLIPPAGE_PCT) if no useful inputs are provided so
    downstream RR math never crashes on a None.
    """
    inputs_known = 0

    # ── Half-spread contribution ───────────────────────────────────
    half_spread_pct = 0.0
    if spread_bps is not None and spread_bps > 0:
        half_spread_pct = (float(spread_bps) / 2.0) / 10_000.0
        inputs_known += 1

    # ── Square-root market impact ──────────────────────────────────
    impact_pct = 0.0
    if (size_usd is not None and size_usd > 0
            and top_book_depth_usd is not None and top_book_depth_usd > 0):
        ratio = float(size_usd) / float(top_book_depth_usd)
        # Cap ratio so a thin book doesn't produce nonsensical 50% slippage.
        ratio = min(ratio, _ES.MAX_DEPTH_RATIO)
        impact_pct = _ES.K_IMPACT * math.sqrt(ratio)
        inputs_known += 1

    # ── Volatility surcharge ───────────────────────────────────────
    vol_pct = 0.0
    if atr_pct is not None and atr_pct > 0:
        vol_pct = _ES.K_VOL * float(atr_pct)
        inputs_known += 1

    if inputs_known == 0:
        # Nothing usable — use the static backtester default so callers
        # always have a finite cost to subtract from RR math.
        return SlippageEstimate(
            expected_pct=Backtester.DEFAULT_SLIPPAGE_PCT,
            half_spread_pct=0.0,
            impact_pct=0.0,
            vol_pct=0.0,
            inputs_known=0,
            fallback=True,
        )

    total = half_spread_pct + impact_pct + vol_pct
    # Floor at the static default so we never *under*-estimate slippage to
    # below what the backtester assumes (avoids the model fooling the gate
    # into accepting pathological setups).
    floored = max(total, _ES.FLOOR_PCT)
    # Cap at a sensible upper bound so a single bad input can't gate every
    # trade as "too costly".
    capped = min(floored, _ES.CEILING_PCT)

    return SlippageEstimate(
        expected_pct=capped,
        half_spread_pct=half_spread_pct,
        impact_pct=impact_pct,
        vol_pct=vol_pct,
        inputs_known=inputs_known,
        fallback=False,
    )


def estimate_slippage_pct_from_signal(signal) -> float:
    """Convenience: pull commonly-available fields off a signal/raw_data dict
    and return just the expected slippage as a fraction of price.

    Looks for, in `signal.raw_data`:
      * spread_bps
      * atr_pct  (or derives from atr_proxy / entry_mid)
      * top_book_depth_usd

    Always returns a number (never None) — uses Backtester default on miss.
    """
    if isinstance(signal, Mapping):
        raw = signal.get("raw_data", None) or {}
    else:
        raw = getattr(signal, "raw_data", None) or {}
    spread_bps = raw.get("spread_bps")
    atr_pct = raw.get("atr_pct")
    if atr_pct is None:
        atr = raw.get("atr_proxy") or raw.get("atr")
        entry_low = getattr(signal, "entry_low", None)
        entry_high = getattr(signal, "entry_high", None)
        if atr and entry_low and entry_high:
            mid = (float(entry_low) + float(entry_high)) / 2.0
            if mid > 0:
                atr_pct = float(atr) / mid
    top_book_depth_usd = raw.get("top_book_depth_usd")
    top_level = signal if isinstance(signal, Mapping) else {}
    size_usd = (
        raw.get("intended_size_usd")
        or raw.get("position_size")
        or raw.get("position_size_usdt")
        or top_level.get("position_size")
        or top_level.get("position_size_usdt")
        or getattr(signal, "position_size", None)
        or getattr(signal, "position_size_usdt", None)
        or _ES.DEFAULT_SIZE_USD
    )

    est = estimate_slippage(
        spread_bps=spread_bps,
        atr_pct=atr_pct,
        size_usd=size_usd,
        top_book_depth_usd=top_book_depth_usd,
    )
    return est.expected_pct
