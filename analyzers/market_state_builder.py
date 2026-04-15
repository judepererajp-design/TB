
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

import time
import numpy as np
from typing import Dict, Tuple

from data.api_client import api
from analyzers.liquidity import LiquidityAnalyzer
from analyzers.energy_model import compute_energy

# Optional imports (safe fallback if modules unavailable)
try:
    from signals.liquidation_heatmap import get_liquidation_state
except Exception:
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
_MARKET_STATE_CACHE: Dict[Tuple[str, str], Tuple[float, Dict]] = {}
_MARKET_STATE_TTL = 15.0  # seconds


async def build_market_state(symbol: str, direction: str) -> Dict:
    """Build real-time market state used by execution engine.

    Results are cached per (symbol, direction) for _MARKET_STATE_TTL seconds.
    Multiple signals on the same symbol share a single API round-trip per
    execution cycle, cutting API load from O(signals) to O(unique symbols).
    """
    cache_key = (symbol, direction)
    _now = time.time()
    _cached = _MARKET_STATE_CACHE.get(cache_key)
    if _cached is not None:
        _ts, _state = _cached
        if _now - _ts < _MARKET_STATE_TTL:
            return _state

    ohlcv = await api.fetch_ohlcv(symbol, "5m", limit=60)
    if not ohlcv or len(ohlcv) < 20:
        # Cache the empty result briefly so we don't retry immediately
        _MARKET_STATE_CACHE[cache_key] = (_now, {})
        return {}

    opens = np.array([float(c[1]) for c in ohlcv])
    highs = np.array([float(c[2]) for c in ohlcv])
    lows = np.array([float(c[3]) for c in ohlcv])
    closes = np.array([float(c[4]) for c in ohlcv])

    # ATR approximation
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(abs(highs[1:] - closes[:-1]),
                    abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

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
        except Exception:
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
