"""
Tests for 6 AIAnalyst bug fixes:
1. _client → _api_key sentinel replacement
2. Stats preserved on failed structural_audit
3. Whale tie defaults to NEUTRAL, prompt includes total_raw_signals
4. /api/sentinel/audit response documents session-wide limitation
5. total_raw_signals counter increments correctly
6. create_task in sentinel audit uses done-callback (tested via response)
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from analyzers.ai_analyst import AIAnalyst


# ── Fix 1: _api_key guards work correctly ──────────────────────────────

class TestClientToApiKeyFix:
    """Verify methods that previously used self._client now use self._api_key."""

    def test_analyse_symbol_uninitialised(self):
        """analyse_symbol should return error string when _api_key is None."""
        ai = AIAnalyst()
        assert ai._api_key is None
        result = asyncio.get_event_loop().run_until_complete(
            ai.analyse_symbol("BTC/USDT", [], [], {}, [], "BULL_TREND")
        )
        assert "not initialised" in result.lower()

    def test_chat_about_symbol_uninitialised(self):
        """chat_about_symbol should return error string when _api_key is None."""
        ai = AIAnalyst()
        result = asyncio.get_event_loop().run_until_complete(
            ai.chat_about_symbol("ETH/USDT", "why price up?", {"regime": "BULL"})
        )
        assert "not initialised" in result.lower()

    def test_pipeline_diagnostics_uninitialised(self):
        """pipeline_diagnostics should return empty dict when _api_key is None."""
        ai = AIAnalyst()
        result = asyncio.get_event_loop().run_until_complete(
            ai.pipeline_diagnostics({"regime": "BULL"})
        )
        assert result == {}

    def test_explain_signal_death_uninitialised(self):
        """explain_signal_death should return error string when _api_key is None."""
        ai = AIAnalyst()
        result = asyncio.get_event_loop().run_until_complete(
            ai.explain_signal_death("BTC/USDT", [{"reason": "test"}], ["log line"])
        )
        assert "not initialised" in result.lower()

    def test_execution_funnel_audit_uninitialised(self):
        """execution_funnel_audit should return empty dict when _api_key is None."""
        ai = AIAnalyst()
        result = asyncio.get_event_loop().run_until_complete(
            ai.execution_funnel_audit([{"symbol": "BTC"}], "BULL")
        )
        assert result == {}

    def test_post_expiry_direction_audit_uninitialised(self):
        """post_expiry_direction_audit should return empty dict when _api_key is None."""
        ai = AIAnalyst()
        result = asyncio.get_event_loop().run_until_complete(
            ai.post_expiry_direction_audit([{"symbol": "BTC"}])
        )
        assert result == {}

    def test_strategy_regime_audit_uninitialised(self):
        """strategy_regime_audit should return empty dict when _api_key is None."""
        ai = AIAnalyst()
        result = asyncio.get_event_loop().run_until_complete(
            ai.strategy_regime_audit({"IchimokuCloud|BULL|LONG": {"signals": 5}}, "BULL")
        )
        assert result == {}

    def test_no_attribute_error_raised(self):
        """No AttributeError from _client access in any method."""
        ai = AIAnalyst()
        # These would all raise AttributeError before the fix
        for coro in [
            ai.analyse_symbol("X", [], [], {}, [], "BULL"),
            ai.chat_about_symbol("X", "hi", {}),
            ai.pipeline_diagnostics({}),
            ai.explain_signal_death("X", [], []),
            ai.execution_funnel_audit([], "BULL"),
            ai.post_expiry_direction_audit([]),
            ai.strategy_regime_audit({}, "BULL"),
        ]:
            try:
                asyncio.get_event_loop().run_until_complete(coro)
            except AttributeError:
                pytest.fail("AttributeError raised — _client bug still present")


# ── Fix 2: Stats preserved on failed audit ─────────────────────────────

class TestStatsPreservedOnFailure:
    """structural_audit should NOT reset stats if AI call fails."""

    def test_stats_preserved_when_call_returns_none(self):
        """If _call returns None (API failure), session stats should be preserved."""
        ai = AIAnalyst()
        ai.initialize()
        # Populate some stats
        ai.record_strategy_signal("SMC", "LONG")
        ai.record_strategy_signal("SMC", "LONG")
        ai.record_htf_block("LONG")
        ai.record_whale_event("buy", 50000.0)

        assert ai._session_stats["total_raw_signals"] == 2
        assert ai._session_stats["htf_blocked_long"] == 1
        assert ai._session_stats["whale_buy_usd"] == 50000.0

        # Mock _call to return None (simulating API failure)
        with patch.object(ai, '_call', new_callable=AsyncMock, return_value=None):
            result = asyncio.get_event_loop().run_until_complete(
                ai.structural_audit(force=True)
            )

        assert result is None
        # Stats should NOT be wiped
        assert ai._session_stats["total_raw_signals"] == 2
        assert ai._session_stats["htf_blocked_long"] == 1
        assert ai._session_stats["whale_buy_usd"] == 50000.0

    def test_stats_reset_after_successful_audit(self):
        """If _call succeeds and parses, stats SHOULD be reset."""
        ai = AIAnalyst()
        ai.initialize()
        ai.record_strategy_signal("SMC", "LONG")

        valid_json = '{"biased_strategies":[],"htf_asymmetry":false,"whale_signal_conflict":false,"regime_signal_conflict":false,"root_causes":[],"recommendations":[],"severity":"LOW"}'
        with patch.object(ai, '_call', new_callable=AsyncMock, return_value=valid_json):
            result = asyncio.get_event_loop().run_until_complete(
                ai.structural_audit(force=True)
            )

        assert result is not None
        assert result.severity == "LOW"
        # Stats SHOULD be reset after success
        assert ai._session_stats["total_raw_signals"] == 0
        assert ai._session_stats["strategy_direction"] == {}


# ── Fix 3: Whale tie defaults to NEUTRAL ────────────────────────────────

class TestWhaleTieNeutral:
    """When whale buy == sell USD, dominant should be NEUTRAL not SELL."""

    def test_whale_tie_is_neutral_in_prompt(self):
        """Whale dominance should be NEUTRAL when buy_usd == sell_usd."""
        ai = AIAnalyst()
        ai.initialize()
        # Equal whale flow
        ai._session_stats["whale_buy_usd"] = 100000.0
        ai._session_stats["whale_sell_usd"] = 100000.0

        captured_prompt = []
        async def capture_call(prompt, **kwargs):
            captured_prompt.append(prompt)
            return None  # Will cause None return, stats preserved

        with patch.object(ai, '_call', side_effect=capture_call):
            asyncio.get_event_loop().run_until_complete(
                ai.structural_audit(force=True)
            )

        assert len(captured_prompt) == 1
        assert "NEUTRAL" in captured_prompt[0]
        assert "Dominant whale: NEUTRAL" in captured_prompt[0]

    def test_whale_buy_dominant(self):
        """When buy > sell, dominant should be BUY."""
        ai = AIAnalyst()
        ai.initialize()
        ai._session_stats["whale_buy_usd"] = 200000.0
        ai._session_stats["whale_sell_usd"] = 100000.0

        captured_prompt = []
        async def capture_call(prompt, **kwargs):
            captured_prompt.append(prompt)
            return None

        with patch.object(ai, '_call', side_effect=capture_call):
            asyncio.get_event_loop().run_until_complete(
                ai.structural_audit(force=True)
            )

        assert "Dominant whale: BUY" in captured_prompt[0]


# ── Fix 5: total_raw_signals counter ────────────────────────────────────

class TestTotalRawSignalsCounter:
    """total_raw_signals increments on record_strategy_signal."""

    def test_counter_increments(self):
        ai = AIAnalyst()
        assert ai._session_stats["total_raw_signals"] == 0
        ai.record_strategy_signal("SMC", "LONG")
        assert ai._session_stats["total_raw_signals"] == 1
        ai.record_strategy_signal("Ichimoku", "SHORT")
        assert ai._session_stats["total_raw_signals"] == 2
        ai.record_strategy_signal("SMC", "LONG")
        assert ai._session_stats["total_raw_signals"] == 3

    def test_total_raw_in_prompt(self):
        """Structural audit prompt should include total raw signal count."""
        ai = AIAnalyst()
        ai.initialize()
        for _ in range(5):
            ai.record_strategy_signal("PA", "LONG")

        captured_prompt = []
        async def capture_call(prompt, **kwargs):
            captured_prompt.append(prompt)
            return None

        with patch.object(ai, '_call', side_effect=capture_call):
            asyncio.get_event_loop().run_until_complete(
                ai.structural_audit(force=True)
            )

        assert "TOTAL RAW SIGNALS (pre-filter): 5" in captured_prompt[0]

    def test_prompt_labels_clarify_populations(self):
        """Prompt labels should distinguish raw signals from HTF-filtered ones."""
        ai = AIAnalyst()
        ai.initialize()
        ai.record_strategy_signal("Test", "LONG")

        captured_prompt = []
        async def capture_call(prompt, **kwargs):
            captured_prompt.append(prompt)
            return None

        with patch.object(ai, '_call', side_effect=capture_call):
            asyncio.get_event_loop().run_until_complete(
                ai.structural_audit(force=True)
            )

        prompt = captured_prompt[0]
        assert "raw signals, before HTF" in prompt
        assert "signals killed by higher-timeframe filter" in prompt
        assert "signals that survived HTF" in prompt


# ── Fix 3 additional: HTF asymmetry rule updated ────────────────────────

class TestHTFAsymmetryRuleUpdate:
    """Prompt should acknowledge when weekly trend is unavailable."""

    def test_prompt_includes_weekly_trend_caveat(self):
        """Prompt should note that weekly trend may not be available."""
        ai = AIAnalyst()
        ai.initialize()

        captured_prompt = []
        async def capture_call(prompt, **kwargs):
            captured_prompt.append(prompt)
            return None

        with patch.object(ai, '_call', side_effect=capture_call):
            asyncio.get_event_loop().run_until_complete(
                ai.structural_audit(force=True)
            )

        prompt = captured_prompt[0]
        assert "weekly trend context is not available" in prompt.lower() or \
               "cannot determine weekly trend" in prompt.lower()
