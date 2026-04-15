"""
Tests for Phase 9-12 audit findings.

Covers:
  - ProbabilityEngine: full state save/load round-trip
  - NoTradeZoneEngine: session dead-zone / weekend penalty
  - MarketStateEngine: liquidity-hunt volume confirmation
  - Ensemble vote persistence structure
"""

import numpy as np
import pytest


# ── ProbabilityEngine full state persistence ─────────────────────────────

class TestProbabilityEngineState:
    """Verify the full-state save/load round-trip preserves learned params."""

    def _make_engine(self):
        from core.probability_engine import ProbabilityEngine
        return ProbabilityEngine()

    def test_get_full_state_keys(self):
        """get_full_state() returns posteriors + LRs + avg_R + calibration."""
        pe = self._make_engine()
        state = pe.get_full_state()
        assert 'posteriors' in state
        assert 'likelihood_ratios' in state
        assert 'avg_win_r' in state
        assert 'avg_loss_r' in state
        assert 'calibration_bins' in state

    def test_full_state_round_trip(self):
        """Mutate LRs and avg_R, save, reload into fresh engine → values match."""
        pe = self._make_engine()
        pe._likelihood_ratios['htf_alignment'] = 0.42
        pe._avg_win_r = 2.5
        pe._avg_loss_r = 0.8
        pe._calibration_bins['bin_50'] = {'predicted': 0.5, 'actual': 0.48, 'count': 20}

        state = pe.get_full_state()

        pe2 = self._make_engine()
        pe2.load_full_state(state)

        assert pe2._likelihood_ratios['htf_alignment'] == 0.42
        assert pe2._avg_win_r == 2.5
        assert pe2._avg_loss_r == 0.8
        assert pe2._calibration_bins['bin_50']['count'] == 20

    def test_load_full_state_backwards_compat(self):
        """Legacy format (flat posteriors dict) still loads via load_full_state."""
        pe = self._make_engine()
        legacy = {'smc:BULL_TREND:LONG': {'alpha': 8.0, 'beta': 4.0}}
        pe.load_full_state(legacy)
        assert 'smc:BULL_TREND:LONG' in pe._posteriors

    def test_get_all_posteriors_still_works(self):
        """Original get_all_posteriors() still returns dict of alpha/beta."""
        pe = self._make_engine()
        out = pe.get_all_posteriors()
        assert isinstance(out, dict)


# ── NoTradeZoneEngine session rules ──────────────────────────────────────

class TestNoTradeSessionRules:
    """Verify the dead-zone and weekend penalty rules."""

    def _engine(self):
        from analyzers.no_trade_zones import NoTradeZoneEngine
        return NoTradeZoneEngine()

    def test_dead_zone_penalty(self):
        """Session=DEAD_ZONE should add an 8-point penalty."""
        eng = self._engine()
        result = eng.evaluate(
            symbol='ETHUSDT', strategy='SmartMoneyConcepts',
            direction='LONG', session='DEAD_ZONE',
        )
        assert result.confidence_penalty >= 8
        assert any('dead' in r.lower() or 'Dead' in r for r in result.reasons)

    def test_weekend_penalty(self):
        """Session=WEEKEND should add a 5-point penalty."""
        eng = self._engine()
        result = eng.evaluate(
            symbol='ETHUSDT', strategy='SmartMoneyConcepts',
            direction='LONG', session='WEEKEND',
        )
        assert result.confidence_penalty >= 5
        assert any('weekend' in r.lower() or 'Weekend' in r for r in result.reasons)

    def test_neutral_session_no_penalty(self):
        """Non-dead-zone sessions should not trigger session penalty."""
        eng = self._engine()
        result = eng.evaluate(
            symbol='ETHUSDT', strategy='SmartMoneyConcepts',
            direction='LONG', session='LONDON_OPEN',
        )
        # No session-related reasons
        session_reasons = [r for r in result.reasons if 'session' in r.lower() or 'dead' in r.lower() or 'weekend' in r.lower()]
        assert len(session_reasons) == 0

    def test_default_session_no_penalty(self):
        """Default session param ('NEUTRAL') should not trigger penalty."""
        eng = self._engine()
        result = eng.evaluate(
            symbol='ETHUSDT', strategy='SmartMoneyConcepts',
            direction='LONG',
        )
        session_reasons = [r for r in result.reasons if 'dead' in r.lower() or 'weekend' in r.lower()]
        assert len(session_reasons) == 0


# ── MarketStateEngine liquidity-hunt volume confirmation ─────────────────

class TestLiquidityHuntVolume:
    """Verify volume confirmation in _detect_liquidity_hunt."""

    def _engine(self):
        from analyzers.market_state_engine import MarketStateEngine
        return MarketStateEngine()

    def test_with_volume_spike_detects(self):
        """Sweep + volume spike → should detect liquidity hunt."""
        from unittest.mock import patch, MagicMock
        import sys
        eng = self._engine()

        # numpy is mocked in conftest; patch the np calls used inside
        mock_np = sys.modules['numpy']
        # np.median → returns 1000.0; np.max → 101.0; np.min → 99.0
        mock_np.median.return_value = 1000.0
        mock_np.max.return_value = 101.0
        mock_np.min.return_value = 99.0

        n = 20
        closes = [100.0] * n
        highs = [101.0] * n
        lows = [99.0] * n
        volumes = [1000.0] * n

        # Last candle: big upper wick sweep beyond recent high
        closes[-1] = 100.5
        closes[-2] = 100.0
        highs[-1] = 103.0
        volumes[-1] = 2000.0  # 2x median → above 1.3× threshold

        result = eng._detect_liquidity_hunt(closes, highs, lows, None, volumes=volumes)
        assert result is True

    def test_without_volume_spike_rejects(self):
        """Sweep without volume spike → should NOT detect (false positive filter)."""
        import sys
        eng = self._engine()

        mock_np = sys.modules['numpy']
        mock_np.median.return_value = 1000.0
        mock_np.max.return_value = 101.0
        mock_np.min.return_value = 99.0

        n = 20
        closes = [100.0] * n
        highs = [101.0] * n
        lows = [99.0] * n
        volumes = [1000.0] * n

        closes[-1] = 100.5
        closes[-2] = 100.0
        highs[-1] = 103.0
        volumes[-1] = 800.0  # below 1.3× median → no spike

        result = eng._detect_liquidity_hunt(closes, highs, lows, None, volumes=volumes)
        assert result is False

    def test_no_volumes_backward_compat(self):
        """When volumes=None (backward compat), sweep should still trigger."""
        import sys
        eng = self._engine()

        mock_np = sys.modules['numpy']
        mock_np.median.return_value = 1000.0
        mock_np.max.return_value = 101.0
        mock_np.min.return_value = 99.0

        n = 20
        closes = [100.0] * n
        highs = [101.0] * n
        lows = [99.0] * n

        closes[-1] = 100.5
        closes[-2] = 100.0
        highs[-1] = 103.0

        result = eng._detect_liquidity_hunt(closes, highs, lows, None, volumes=None)
        assert result is True


# ── Ensemble vote structure ──────────────────────────────────────────────

class TestEnsembleVoteStructure:
    """Verify that EnsembleVerdict.votes can be serialized for persistence."""

    def test_votes_serializable(self):
        """Vote objects can be converted to the dict format used in sig_data."""
        from signals.ensemble_voter import Vote, VoteValue
        votes = [
            Vote(source='cvd', value=VoteValue.SUPPORT, reason='bullish', weight=1.8),
            Vote(source='smart_money', value=VoteValue.OPPOSE, reason='bearish', weight=2.0),
        ]
        serialized = {
            v.source: {'value': int(v.value), 'weight': v.weight}
            for v in votes
        }
        assert serialized['cvd']['value'] == 1
        assert serialized['smart_money']['value'] == -1
        assert serialized['smart_money']['weight'] == 2.0
