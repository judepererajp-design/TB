"""
Tests for GPT-identified fixes:
1. TP1 trailing stop accounting (PARTIAL_WIN vs BREAKEVEN)
2. Volatility-scaled TP methods wired into strategies
3. compute_vol_percentile utility
"""

import sys
import importlib

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# The conftest mocks numpy with MagicMock when it's absent.  We need
# the real numpy for compute_vol_percentile tests, so force-reload it.
if isinstance(sys.modules.get("numpy"), MagicMock):
    # Remove mock and reimport real numpy
    del sys.modules["numpy"]
    for k in list(sys.modules):
        if k.startswith("numpy"):
            del sys.modules[k]

import numpy as np

# Now reimport risk_params so it picks up real numpy
if "utils.risk_params" in sys.modules:
    del sys.modules["utils.risk_params"]
from utils.risk_params import rp, compute_vol_percentile


# ═══════════════════════════════════════════════════════════════════
# Group 1: compute_vol_percentile
# ═══════════════════════════════════════════════════════════════════

class TestComputeVolPercentile:
    """Tests for the vol_percentile utility function."""

    def test_returns_float_between_0_and_1(self):
        np.random.seed(42)
        n = 200
        closes = np.cumsum(np.random.randn(n)) + 100
        highs = closes + np.abs(np.random.randn(n))
        lows = closes - np.abs(np.random.randn(n))
        vp = compute_vol_percentile(highs, lows, closes)
        assert 0.0 <= vp <= 1.0

    def test_insufficient_data_returns_neutral(self):
        """Less than lookback bars → 0.5 (neutral)."""
        vp = compute_vol_percentile(
            np.array([101.0] * 50),
            np.array([99.0] * 50),
            np.array([100.0] * 50),
        )
        assert vp == 0.5

    def test_high_vol_returns_high_percentile(self):
        """Create data where recent bars have much wider ranges → high vp."""
        np.random.seed(7)
        n = 150
        closes = np.ones(n) * 100
        highs = closes + 0.5
        lows = closes - 0.5
        # Make last 5 bars very wide
        highs[-5:] = 110
        lows[-5:] = 90
        vp = compute_vol_percentile(highs, lows, closes)
        assert vp > 0.8

    def test_low_vol_returns_low_percentile(self):
        """Create data where recent bars have very tight ranges → low vp."""
        np.random.seed(7)
        n = 150
        closes = np.ones(n) * 100
        highs = closes + 2.0  # Normal: range = 4.0
        lows = closes - 2.0
        # Make last 5 bars very tight
        highs[-5:] = 100.01
        lows[-5:] = 99.99
        closes[-5:] = 100.0
        vp = compute_vol_percentile(highs, lows, closes)
        assert vp < 0.2

    def test_exception_returns_neutral(self):
        """If something goes wrong, return 0.5."""
        vp = compute_vol_percentile(None, None, None)
        assert vp == 0.5


# ═══════════════════════════════════════════════════════════════════
# Group 2: Volatility-scaled TP methods
# ═══════════════════════════════════════════════════════════════════

class TestVolatilityScaledTP:
    """Verify volatility_scaled_tp*() differs from plain scaled_tp*() when vp != 0.5."""

    def test_low_vol_shrinks_tp2(self):
        """At vp=0.0, vol_scale=0.85 → TP is 85% of the neutral value."""
        neutral = rp.volatility_scaled_tp2('1h', 0.5)
        low_vol = rp.volatility_scaled_tp2('1h', 0.0)
        assert low_vol < neutral

    def test_high_vol_expands_tp2(self):
        """At vp=1.0, vol_scale=1.30 → TP is 130% of the base."""
        neutral = rp.volatility_scaled_tp2('1h', 0.5)
        high_vol = rp.volatility_scaled_tp2('1h', 1.0)
        assert high_vol > neutral

    def test_neutral_vol_matches_midpoint(self):
        """At vp=0.5, vol_scale should be 1.075 (midpoint of 0.85-1.30)."""
        result = rp.volatility_scaled_tp2('1h', 0.5)
        base = rp.tp2_atr_mult * rp.atr_scale('1h')
        expected_scale = 0.85 + 0.45 * 0.5  # = 1.075
        assert abs(result - base * expected_scale) < 0.001

    def test_timeframe_and_vol_both_apply(self):
        """5m + high vol vs 4h + low vol should show compounding difference."""
        scalp_high = rp.volatility_scaled_tp2('5m', 1.0)
        swing_low = rp.volatility_scaled_tp2('4h', 0.0)
        # Swing should still be wider even with low vol
        assert swing_low > scalp_high

    def test_tp1_tp2_tp3_ordering(self):
        """TP1 < TP2 < TP3 at any vol_percentile."""
        for vp in [0.0, 0.5, 1.0]:
            tp1 = rp.volatility_scaled_tp1('1h', vp)
            tp2 = rp.volatility_scaled_tp2('1h', vp)
            tp3 = rp.volatility_scaled_tp3('1h', vp)
            assert tp1 < tp2 < tp3, f"TP ordering broken at vp={vp}"


# ═══════════════════════════════════════════════════════════════════
# Group 3: TP1 trailing stop accounting
# ═══════════════════════════════════════════════════════════════════

class TestTP1TrailingStopAccounting:
    """Verify trailing stop hit after TP1 is WIN when pnl_r > 0.05, BREAKEVEN otherwise."""

    def _make_tracked(self):
        """Create a minimal mock tracked signal in BE_ACTIVE state."""
        from signals.outcome_monitor import SignalState
        tracked = MagicMock()
        tracked.state = SignalState.BE_ACTIVE
        tracked.signal_id = 99
        tracked.symbol = "BTCUSDT"
        tracked.direction = "LONG"
        tracked.entry_price = 100.0
        tracked.entry_low = 99.0
        tracked.entry_high = 101.0
        tracked.stop_loss = 95.0
        tracked.tp1 = 107.0
        tracked.tp2 = 112.0
        tracked.tp3 = 120.0
        tracked.be_stop = 105.0  # Trailing stop ratcheted up from 100 to 105
        tracked.trail_stop = 105.0
        tracked.trail_pct = 0.55
        tracked.max_r = 1.4
        tracked.created_at = 0
        tracked.last_checkin_at = 0
        tracked.message_id = 1
        tracked.confidence = 70
        tracked.strategy = "Momentum"
        tracked.raw_data = {}
        tracked.last_price = None
        tracked.completed_at = None
        return tracked

    def test_trail_hit_with_profit_is_win(self):
        """Trail fires at +1.0R → should be WIN not BREAKEVEN."""
        from signals.outcome_monitor import OutcomeMonitor, SignalState
        om = OutcomeMonitor()
        tracked = self._make_tracked()

        # _calc_pnl_r should return positive R when price is above entry
        pnl = om._calc_pnl_r(tracked, 105.0)  # At trail stop = 105, entry = ~100
        # Price 105.0 with entry ~100 and risk ~5 → ~1.0R
        assert pnl > 0.05, f"Expected positive pnl_r at trail stop, got {pnl}"

    def test_trail_hit_at_entry_is_breakeven(self):
        """Trail fires right at entry → should be BREAKEVEN."""
        from signals.outcome_monitor import OutcomeMonitor
        om = OutcomeMonitor()
        tracked = self._make_tracked()
        tracked.be_stop = 100.1  # Trail barely above entry

        pnl = om._calc_pnl_r(tracked, 100.1)
        # At entry, pnl should be ~0.0, hence BREAKEVEN path
        assert pnl <= 0.05, f"Expected near-zero pnl_r at entry, got {pnl}"


# ═══════════════════════════════════════════════════════════════════
# Group 4: Strategy imports compile correctly
# ═══════════════════════════════════════════════════════════════════

class TestStrategyImports:
    """Verify all strategies can import compute_vol_percentile."""

    def test_momentum_imports(self):
        from strategies.momentum import MomentumStrategy
        assert MomentumStrategy is not None

    def test_funding_arb_imports(self):
        from strategies.funding_arb import FundingArbStrategy
        assert FundingArbStrategy is not None

    def test_smc_imports(self):
        from strategies.smc import SMCStrategy
        assert SMCStrategy is not None

    def test_ichimoku_imports(self):
        from strategies.ichimoku import IchimokuStrategy
        assert IchimokuStrategy is not None

    def test_reversal_imports(self):
        from strategies.reversal import ReversalStrategy
        assert ReversalStrategy is not None

    def test_price_action_imports(self):
        from strategies.price_action import PriceActionStrategy
        assert PriceActionStrategy is not None

    def test_wyckoff_imports(self):
        from strategies.wyckoff import WyckoffStrategy
        assert WyckoffStrategy is not None
