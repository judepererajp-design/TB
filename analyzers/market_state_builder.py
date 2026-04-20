
"""
Market State Builder — Extended (Live Intelligence Bus)
-------------------------------------------------------
Aggregates multiple analyzers into a unified execution-ready
market_state object.

Sources:
- Liquidity sweeps
- Orderbook imbalance
- Orderflow pressure
- Liquidation heatmap (if available)

NO MOCK DATA.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Tuple

from data.api_client import api
from analyzers.liquidity import LiquidityAnalyzer
from analyzers.energy_model import compute_energy

logger = logging.getLogger(__name__)

# Optional imports (safe fallback if modules unavailable)
# AUDIT FIX: previously the failure was swallowed silently, leaving
# future operators with no clue why `get_liquidation_state` was None.
try:
    from signals.liquidation_heatmap import get_liquidation_state
except ImportError as _e:
    logger.warning(
        "signals.liquidation_heatmap unavailable — liquidation "
        "cluster detection disabled: %s", _e,
    )
    get_liquidation_state = None
except Exception as _e:  # pragma: no cover — unexpected import failure
    logger.warning(
        "Unexpected error importing signals.liquidation_heatmap — "
        "liquidation cluster detection disabled: %s", _e,
    )
    get_liquidation_state = None

liq_analyzer = LiquidityAnalyzer()

# ── Shared market-state cache (Performance Fix — API Spam) ───────────────────
# build_market_state() is called once per tracked signal per cycle
# (every 15 s) and costs 2 API round-trips each time:
#   • api.fetch_ohlcv(symbol, "5m", limit=60)
#   • liq_analyzer.check_orderbook_imbalance(symbol, ...)
# With 10 signals on the same symbol that is 20 API calls every 15 s.
# The cache key is (symbol, direction); TTL is 15 s so data stays
# fresh for the execution cycle but is only fetched once per cycle.
#
# AUDIT FIX: the cache is now guarded by an asyncio.Lock so N concurrent
# callers for the same (symbol, direction) collapse into a single
# upstream fetch rather than stampeding the API.  We also track a
# dedicated "empty sentinel" so a legitimately-empty fetch (insufficient
# bars, symbol delisted) is distinguishable from "nothing has been
# fetched yet" in downstream logging.
_MARKET_STATE_CACHE: Dict[Tuple[str, str], Tuple[float, Dict]] = {}
_MARKET_STATE_TTL = 15.0  # seconds
_MARKET_STATE_LOCK = asyncio.Lock()
# Sentinel value stored in the cache when a fetch legitimately yielded
# no usable data (not enough bars, exchange returned empty).  Downstream
# logging can distinguish this from a fresh successful result using an
# identity check (`is _EMPTY_MARKET_STATE`).  Kept as a frozen, empty
# dict so normal dict-access on the returned value is still safe.
_EMPTY_MARKET_STATE: Dict = {}


async def build_market_state(symbol: str, direction: str) -> Dict:
    """Build real-time market state used by execution engine.

    Results are cached per (symbol, direction) for _MARKET_STATE_TTL seconds.
    Multiple signals on the same symbol share a single API round-trip per
    execution cycle, cutting API load from O(signals) to O(unique symbols).
    """
    # AUDIT FIX: assert the direction contract up-front.  Downstream
    # liquidity / energy code branches on direction == "LONG"/"SHORT";
    # a caller passing e.g. "long" or "" would silently take the SHORT
    # branch and produce a nonsense reaction.
    assert direction in ("LONG", "SHORT"), (
        f"build_market_state: direction must be 'LONG' or 'SHORT', "
        f"got {direction!r}"
    )

    cache_key = (symbol, direction)
    _now = time.time()
    _cached = _MARKET_STATE_CACHE.get(cache_key)
    if _cached is not None:
        _ts, _state = _cached
        if _now - _ts < _MARKET_STATE_TTL:
            return _state

    # Collapse concurrent callers: if two tasks miss the cache for the
    # same key at the same instant, only the first should fetch upstream
    # while the second waits and re-reads the cache.
    async with _MARKET_STATE_LOCK:
        _cached = _MARKET_STATE_CACHE.get(cache_key)
        if _cached is not None:
            _ts, _state = _cached
            if _now - _ts < _MARKET_STATE_TTL:
                return _state

        ohlcv = await api.fetch_ohlcv(symbol, "5m", limit=60)
        if not ohlcv or len(ohlcv) < 20:
            # Cache the EMPTY sentinel briefly so we don't retry
            # immediately — distinct from "never fetched".
            _MARKET_STATE_CACHE[cache_key] = (_now, _EMPTY_MARKET_STATE)
            return _EMPTY_MARKET_STATE

        opens = np.array([float(c[1]) for c in ohlcv])
        highs = np.array([float(c[2]) for c in ohlcv])
        lows = np.array([float(c[3]) for c in ohlcv])
        closes = np.array([float(c[4]) for c in ohlcv])

        # ATR approximation
        # AUDIT FIX: the old code branched on `len(tr) >= 14` and used
        # `np.mean(tr)` on the short side, so ATR jumped discontinuously
        # at the 14th bar (the full-history mean is typically ≠ the
        # 14-bar trailing mean).  Use a consistent "min(14, len(tr))"
        # trailing window on both branches so the ATR curve is smooth.
        tr = np.maximum(highs[1:] - lows[1:],
             np.maximum(abs(highs[1:] - closes[:-1]),
                        abs(lows[1:] - closes[:-1])))
        if len(tr) == 0:
            _MARKET_STATE_CACHE[cache_key] = (_now, _EMPTY_MARKET_STATE)
            return _EMPTY_MARKET_STATE
        _atr_window = min(14, len(tr))
        atr = float(np.mean(tr[-_atr_window:]))

        current_price = closes[-1]

        # --- Liquidity Reaction ---
        reaction = liq_analyzer.detect_reaction(
            highs=highs,
            lows=lows,
            closes=closes,
            opens=opens,
            current_price=current_price,
            direction=direction,
            atr=atr,
        )

        # --- Orderbook Imbalance ---
        imbalance, imbalance_note = await liq_analyzer.check_orderbook_imbalance(
            symbol, direction
        )

        # --- Energy Model ---
        volumes = np.array([float(c[5]) for c in ohlcv])
        energy = compute_energy(highs, lows, closes, volumes)

        # --- Liquidation Heatmap (optional) ---
        liquidation_cluster_hit = False
        if get_liquidation_state:
            try:
                liq_state = await get_liquidation_state(symbol)
                liquidation_cluster_hit = liq_state.get("cluster_hit", False)
            except Exception as _e:
                # Narrow log — full stack at DEBUG so operators can opt
                # in without spamming live logs on transient upstream
                # failures.
                logger.debug(
                    "get_liquidation_state(%s) failed — treating as "
                    "no-cluster: %s", symbol, _e, exc_info=True,
                )
                liquidation_cluster_hit = False

        market_state = {
            "liquidity": {
                "below_swept": reaction.swept and reaction.direction == "LONG",
                "above_swept": reaction.swept and reaction.direction == "SHORT",
                "reaction_strength": reaction.reaction_strength,
            },

            "orderbook": {
                "imbalance": imbalance,
                "bullish_pressure": imbalance > 0.15,
                "bearish_pressure": imbalance < -0.15,
                "note": imbalance_note,
            },

            "orderflow": {
                # proxy until deeper delta integration
                "buyer_aggression": max(0.0, imbalance),
                "seller_aggression": abs(min(0.0, imbalance)),
            },

            "liquidations": {
                "cluster_hit": liquidation_cluster_hit,
            },

            "energy": energy,
            "last_candle": {
                # close_strength_up: close must be in the upper half of the candle's range.
                # OLD: closes[-1] > opens[-1]  — any green candle qualifies (even +0.01%).
                # NEW: close > midpoint of (low, high) for LONG; < midpoint for SHORT.
                #   This ensures the close represents real upward/downward conviction,
                #   not just a 1-tick green candle at the bottom of a bearish range.
                #   Strict inequality (> / <) so they are mutually exclusive at midpoint.
                "close_strength_up":  closes[-1] > (highs[-1] + lows[-1]) / 2,
                "rejection":          closes[-1] < (highs[-1] + lows[-1]) / 2,
            },
        }

        _MARKET_STATE_CACHE[cache_key] = (_now, market_state)
        return market_state
