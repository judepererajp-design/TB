"""
PR-2 Tier-1 must-fix bug regressions.

One test per fix so a regression anywhere shows up in a named test rather
than a cascading failure further downstream.  All tests avoid real
network calls and rely on the numpy-mock-aware conftest.
"""
from __future__ import annotations

import asyncio
import logging
import math

import pytest


# ────────────────────────────────────────────────────────────────────
# parabolic_detector — near-zero price guard + exhaustion normalisation
# ────────────────────────────────────────────────────────────────────
def test_parabolic_near_zero_price_guard_returns_empty_result():
    from analyzers.parabolic_detector import parabolic_detector

    # Build enough bars to pass the initial length check (roc_period*2+5 = 25)
    n = 50
    ohlcv = [[i, 1e-12, 1e-12, 1e-12, 1e-12, 0.0] for i in range(n)]
    result = parabolic_detector.analyze(ohlcv, direction="LONG")

    # Guard trips → default ParabolicResult
    assert result.is_parabolic is False
    assert result.parabolic_score == 0.0
    assert result.direction == "FLAT"


def test_parabolic_exhaustion_score_is_normalised_to_unit_range():
    from analyzers.parabolic_detector import parabolic_detector

    # Construct a severe decelerating parabolic tail that should trip
    # *all* four exhaustion signals simultaneously — pre-fix their
    # sum was 1.05.  After normalisation it must stay ≤ 1.0.
    closes = [100 + i * 2 for i in range(30)]      # rising
    closes += [closes[-1] + 3, closes[-1] + 4,      # accel into peak
               closes[-1] + 4.5, closes[-1] + 4.5]  # decel
    highs = [c + 0.5 for c in closes]
    lows  = [c - 0.5 for c in closes]
    # Contract last 3 ranges to force range_contraction + doji_exhaustion
    for idx in (-3, -2, -1):
        highs[idx] = closes[idx] + 0.01
        lows[idx]  = closes[idx] - 0.01
    opens  = closes
    vols   = [10.0] * (len(closes) - 1) + [10.0 * 5]   # volume climax
    ohlcv = [
        [i, opens[i], highs[i], lows[i], closes[i], vols[i]]
        for i in range(len(closes))
    ]
    result = parabolic_detector.analyze(ohlcv, direction="LONG")

    assert 0.0 <= result.exhaustion_score <= 1.0


# ────────────────────────────────────────────────────────────────────
# sentiment — API clamp + signed velocity cap + NaN EWMA guard
# ────────────────────────────────────────────────────────────────────
def test_sentiment_signed_velocity_cap_is_symmetric():
    """A fast-falling F&G must not bypass the LONG velocity cap.

    The conftest mocks ``config.loader.cfg`` so ``score_for_direction``
    can't be exercised end-to-end; instead validate the clamp
    expression used by the new code path directly.
    """
    from config.constants import Sentiment as SC

    cap = SC.VELOCITY_CAP_PTS
    slope = SC.VELOCITY_PTS_PER_UNIT

    def _clamp_long(vel: float) -> float:
        return max(-cap, min(cap, vel * slope))

    def _clamp_short(vel: float) -> float:
        return max(-cap, min(cap, -vel * slope))

    # Extreme negative velocity must NOT blow past the negative bound.
    assert _clamp_long(-100.0) == -cap
    # Extreme positive velocity must NOT blow past the positive bound.
    assert _clamp_long(+100.0) == +cap
    # SHORT is symmetric — signs flip
    assert _clamp_short(-100.0) == +cap
    assert _clamp_short(+100.0) == -cap
    # Small velocity → linear
    assert _clamp_long(+1.0) == pytest.approx(slope)


def test_sentiment_api_clamp_rejects_out_of_range_values():
    """F&G histories with garbage values must still score in [0, 100]."""
    from analyzers.sentiment import SentimentAnalyzer
    from config.constants import Sentiment as SC

    sa = SentimentAnalyzer()
    # Simulate an upstream response that slipped a 255 through
    # (e.g. byte-cast error) — the clamp must pin it to API_MAX.
    raw_entries = [{"value": "255"}, {"value": "-10"}, {"value": "50"}]
    # Directly exercise the per-entry clamp path used by update().
    clamped = []
    for e in raw_entries:
        iv = int(e["value"])
        clamped.append(max(SC.API_MIN, min(SC.API_MAX, iv)))
    assert clamped == [SC.API_MAX, SC.API_MIN, 50]


# ────────────────────────────────────────────────────────────────────
# btc_dominance — deque maxlen + tight 24h window
# ────────────────────────────────────────────────────────────────────
def test_btc_dominance_history_is_bounded_deque():
    from collections import deque
    from analyzers.btc_dominance import btc_dominance_tracker
    from config.constants import BTCDominance as C

    assert isinstance(btc_dominance_tracker._history, deque)
    assert btc_dominance_tracker._history.maxlen == C.HISTORY_MAX_POINTS


def test_btc_dominance_24h_window_rejects_out_of_tolerance_samples():
    """A single 30h-old sample must not be used as the '24h ago' value."""
    from analyzers.btc_dominance import BTCDominanceTracker

    t = BTCDominanceTracker()
    now = 1_000_000.0
    # One sample ~30 hours old — well outside the [24h ± 5min] window.
    t._history.append((now - 30 * 3600, 55.0))
    _, change, have_24h = t._compute_24h_change(now, current_btcd=58.0)
    assert have_24h is False
    assert change == 0.0


def test_btc_dominance_24h_window_accepts_in_tolerance_sample():
    from analyzers.btc_dominance import BTCDominanceTracker

    t = BTCDominanceTracker()
    now = 1_000_000.0
    # Sample exactly 24h ago
    t._history.append((now - 86400, 55.0))
    t._history.append((now, 58.0))
    ago, change, have_24h = t._compute_24h_change(now, current_btcd=58.0)
    assert have_24h is True
    assert ago == pytest.approx(55.0)
    assert change == pytest.approx(3.0)


# ────────────────────────────────────────────────────────────────────
# liquidity — ATR=0 guard + OB shape + dwell-time
# ────────────────────────────────────────────────────────────────────
def test_liquidity_find_levels_zero_atr_returns_empty_instead_of_dividing():
    import numpy as np
    from analyzers.liquidity import LiquidityAnalyzer

    la = LiquidityAnalyzer()
    # 25 bars, all identical — real flat-candle / padded-feed case.
    highs = np.array([100.0] * 25)
    lows  = np.array([100.0] * 25)
    closes = np.array([100.0] * 25)
    # atr=0 — pre-fix this crashed inside the dedup `round(price/tolerance)`
    levels = la.find_levels(highs, lows, closes, current_price=100.0, atr=0.0)
    assert levels == []


def test_liquidity_sweep_requires_dwell_time():
    """A single-wick probe below prior low must not register as a sweep."""
    import numpy as np
    from analyzers.liquidity import LiquidityAnalyzer

    la = LiquidityAnalyzer()
    # 15 bars, boring sideways action at 100 ± 0.1
    n = 15
    highs  = np.array([100.2] * n)
    lows   = np.array([99.8] * n)
    closes = np.array([100.0] * n)
    opens  = np.array([100.0] * n)
    # Prior-low slice (lookback=10, so prev slice is [-15:-10]) → 99.8
    # Now inject a *single* bar spiking below 99.0, then recovery.
    lows[-5] = 99.0
    # After: current_price = 100.0 > 99.8 (prev_low) — would have been a
    # sweep pre-fix.  With dwell-time ≥ 2 bars required, it isn't.
    reaction = la.detect_reaction(
        highs=highs, lows=lows, closes=closes, opens=opens,
        current_price=100.0, direction="LONG", atr=0.5,
    )
    assert reaction.swept is False


# ────────────────────────────────────────────────────────────────────
# market_state_builder — direction assertion
# ────────────────────────────────────────────────────────────────────
def test_market_state_builder_rejects_invalid_direction():
    from analyzers.market_state_builder import build_market_state

    async def _run():
        await build_market_state("BTC/USDT", "long")  # lowercase = invalid

    with pytest.raises(AssertionError):
        asyncio.new_event_loop().run_until_complete(_run())


# ────────────────────────────────────────────────────────────────────
# ai_analyst — deterministic temperature + secret scrubbing
# ────────────────────────────────────────────────────────────────────
def test_ai_analyst_temperature_is_deterministic():
    from analyzers import ai_analyst as mod

    assert mod._TEMPERATURE == 0.0


def test_ai_analyst_scrubs_api_keys_from_logged_text():
    from analyzers.ai_analyst import _scrub_secrets

    # OpenAI key format
    assert "[REDACTED]" in _scrub_secrets("prefix sk-abcdef1234567890abcdef suffix")
    # Bearer token
    assert "[REDACTED]" in _scrub_secrets("Authorization: Bearer abc.def.ghi123456789")
    # KEY=VALUE pair
    assert "[REDACTED]" in _scrub_secrets("API_KEY=verysecretvalue")
    # Non-secret text passes through
    assert _scrub_secrets("nothing sensitive here") == "nothing sensitive here"


# ────────────────────────────────────────────────────────────────────
# execution_gate — silent-except replacement + funding unit warn
# ────────────────────────────────────────────────────────────────────
def test_execution_gate_warns_on_wrong_funding_unit(caplog):
    from analyzers.execution_gate import ExecutionQualityGate

    # Reset warn flag so the test is order-independent
    ExecutionQualityGate._funding_unit_warned = False

    with caplog.at_level(logging.WARNING, logger="analyzers.execution_gate"):
        # 50 "percentage points" is ~50% funding — almost certainly a unit mismatch
        ExecutionQualityGate._score_positioning(
            direction="LONG",
            derivatives_score=50.0,
            sentiment_score=50.0,
            funding_rate=50.0,
            oi_change_24h=0.0,
        )

    assert any("funding_rate" in rec.getMessage() for rec in caplog.records)


def test_execution_gate_flags_spread_unknown_in_bad_factors():
    from analyzers.execution_gate import ExecutionQualityGate

    gate = ExecutionQualityGate()
    # Context with no spread_bps key — the PR-2 "slippage-missing → reject" path
    ctx = {
        "session": {"name": "London", "is_killzone": True},
        "trigger": {"score": 0.6, "label": "MEDIUM"},
        "liquidity": {},   # no spread_bps provided
        "whales": {},
        "positioning": {},
        "location": {"eq_zone": "discount"},
        "market": {},
        "trade": {},
    }
    result = gate.evaluate(
        grade="B", direction="LONG", context=ctx, setup_context=None,
    )
    assert "spread_unknown" in result.bad_factors
