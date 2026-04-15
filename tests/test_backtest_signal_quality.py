"""
Tests for backtest signal quality gates that prevent over-trading.

These gates were added to address:
  - 2187 trades on 2161 bars (> 1 trade/bar)
  - 27.9% win rate with -246.3R total loss
  - No deduplication, no per-bar limits, no confluence requirement

The gates tested:
  1. MAX_SIGNALS_PER_BAR: Only best signal per bar (by confidence)
  2. SIGNAL_COOLDOWN_BARS: Min bars between same-direction signals
  3. MIN_CONFLUENCE_COUNT: Require ≥2 confluence factors
  4. MIN_CONFIDENCE_BACKTEST: Stricter confidence floor (75)
"""

import pytest

from config.constants import Backtester as BTC


# ── Constants exist and have sane defaults ────────────────────────────────

class TestBacktestSignalQualityConstants:
    """Verify the new Backtester constants are configured correctly."""

    def test_max_signals_per_bar_exists(self):
        assert hasattr(BTC, 'MAX_SIGNALS_PER_BAR')
        assert isinstance(BTC.MAX_SIGNALS_PER_BAR, int)

    def test_max_signals_per_bar_is_restrictive(self):
        """Should allow at most a few signals per bar to prevent over-trading."""
        assert 1 <= BTC.MAX_SIGNALS_PER_BAR <= 3

    def test_signal_cooldown_bars_exists(self):
        assert hasattr(BTC, 'SIGNAL_COOLDOWN_BARS')
        assert isinstance(BTC.SIGNAL_COOLDOWN_BARS, int)

    def test_signal_cooldown_bars_reasonable(self):
        """Should enforce at least a few bars between same-direction signals."""
        assert BTC.SIGNAL_COOLDOWN_BARS >= 3

    def test_min_confluence_count_exists(self):
        assert hasattr(BTC, 'MIN_CONFLUENCE_COUNT')
        assert isinstance(BTC.MIN_CONFLUENCE_COUNT, int)

    def test_min_confluence_count_requires_multi_factor(self):
        """Should require at least 2 confluence factors."""
        assert BTC.MIN_CONFLUENCE_COUNT >= 2

    def test_min_confidence_backtest_exists(self):
        assert hasattr(BTC, 'MIN_CONFIDENCE_BACKTEST')
        assert isinstance(BTC.MIN_CONFIDENCE_BACKTEST, int)

    def test_min_confidence_backtest_stricter_than_live(self):
        """Backtest floor should be at least as strict as live default (72)."""
        assert BTC.MIN_CONFIDENCE_BACKTEST >= 72

    def test_min_confidence_backtest_not_too_high(self):
        """Should not be so high that it blocks everything (< 85 = A grade)."""
        assert BTC.MIN_CONFIDENCE_BACKTEST < 85


# ── Constants are consistent with each other ──────────────────────────────

class TestBacktestConstantsConsistency:
    """Verify the constants work together sensibly."""

    def test_cooldown_less_than_max_hold(self):
        """Cooldown should be shorter than max hold bars to allow re-entry."""
        assert BTC.SIGNAL_COOLDOWN_BARS < BTC.MAX_HOLD_BARS

    def test_max_daily_trades_reasonable(self):
        """With 1h bars and cooldown, max daily trades should be manageable.
        24 bars/day ÷ cooldown = max trades per day per direction.
        With both LONG and SHORT, that's 2× max.
        """
        bars_per_day = BTC.BARS_PER_DAY.get('1h', 24)
        max_per_direction = bars_per_day / BTC.SIGNAL_COOLDOWN_BARS
        max_both = max_per_direction * 2  # LONG + SHORT
        # Should not exceed ~16 trades per day (8 per direction max)
        assert max_both <= 16, (
            f"Too many possible trades/day: {max_both:.0f} "
            f"(cooldown={BTC.SIGNAL_COOLDOWN_BARS} bars)"
        )

    def test_confluence_gate_is_achievable(self):
        """MIN_CONFLUENCE_COUNT should be low enough that strategies
        can actually pass it (most strategies emit 2-5 confluence factors)."""
        assert BTC.MIN_CONFLUENCE_COUNT <= 4


# ── BacktestEngine references the constants correctly ─────────────────────

class TestBacktestEngineUsesGates:
    """Verify the engine module references the new constants."""

    def test_engine_module_imports_constants(self):
        """The engine should import and use BTC constants."""
        import importlib
        import backtester.engine as mod
        importlib.reload(mod)  # ensure fresh import
        with open(mod.__file__) as f:
            src = f.read()
        assert 'BTC.MAX_SIGNALS_PER_BAR' in src, (
            "Engine must reference MAX_SIGNALS_PER_BAR"
        )
        assert 'BTC.SIGNAL_COOLDOWN_BARS' in src, (
            "Engine must reference SIGNAL_COOLDOWN_BARS"
        )
        assert 'BTC.MIN_CONFLUENCE_COUNT' in src, (
            "Engine must reference MIN_CONFLUENCE_COUNT"
        )
        assert 'BTC.MIN_CONFIDENCE_BACKTEST' in src, (
            "Engine must reference MIN_CONFIDENCE_BACKTEST"
        )

    def test_engine_has_per_bar_selection(self):
        """Engine should sort candidates by confidence and limit per bar."""
        with open('backtester/engine.py') as f:
            src = f.read()
        assert 'bar_candidates' in src, (
            "Engine must collect candidates per bar before selecting best"
        )
        assert 'bar_candidates.sort' in src or 'sorted' in src, (
            "Engine must sort candidates by confidence"
        )

    def test_engine_has_cooldown_tracking(self):
        """Engine should track last signal bar per direction."""
        with open('backtester/engine.py') as f:
            src = f.read()
        assert '_last_signal_bar' in src, (
            "Engine must track cooldown per direction"
        )

    def test_engine_accuracy_warning_documents_gates(self):
        """The ACCURACY WARNING should mention the new gates."""
        with open('backtester/engine.py') as f:
            src = f.read()
        assert 'Per-bar signal limit' in src
        assert 'Direction cooldown' in src
        assert 'Minimum confluence' in src
        assert 'confidence floor' in src
