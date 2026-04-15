"""
Tests for TitanBot Pro — Signal Data Validator ("Skeptical LLM Mode")
=====================================================================
Covers:
  - Layer A: programmatic hard checks (RSI bounds, price positivity,
    volume non-negative, entry zone geometry, direction/SL consistency)
  - Layer B: LLM anomaly detection (mocked — verifies prompt building
    and response parsing)
  - ValidationResult data quality scoring
  - Integration: feature flag gating, throttling, stats tracking
"""

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on sys.path (conftest.py handles mocking)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from signals.signal_validator import (
    SignalValidator, ValidationResult, HARD_RULES, _summarise_data,
    _calc_dynamic_penalty, TREND_STRATEGIES, REVERSAL_STRATEGIES,
    KILL_SWITCH_DATA_CONFIDENCE, KILL_SWITCH_CRITICAL_ISSUES,
    DRIFT_WINDOW_SIZE, DRIFT_ALERT_THRESHOLD,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def validator():
    """Fresh validator instance for each test."""
    return SignalValidator()


def _base_signal(**overrides) -> dict:
    """Minimal valid signal data dict."""
    data = {
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "strategy": "Momentum",
        "confidence": 75,
        "regime": "BULL_TREND",
        "entry_low": 63000.0,
        "entry_high": 63500.0,
        "stop_loss": 62000.0,
        "price": 63250.0,
        "rr_ratio": 2.5,
        "rsi": 55.0,
        "adx": 30.0,
    }
    data.update(overrides)
    return data


# ══════════════════════════════════════════════════════════════════════
# LAYER A — PROGRAMMATIC HARD CHECKS
# ══════════════════════════════════════════════════════════════════════

class TestLayerAHardChecks:
    """Test the programmatic validation layer (instant, no LLM)."""

    async def test_valid_signal_passes(self, validator):
        """Clean signal data should pass with OK status."""
        result = await validator.validate(_base_signal(), run_llm=False)
        assert result.status == "OK"
        assert result.data_quality == "HIGH"
        assert result.hard_errors == []

    async def test_rsi_above_100_is_error(self, validator):
        """RSI > 100 is mathematically impossible."""
        result = await validator.validate(
            _base_signal(rsi=150), run_llm=False
        )
        assert result.status == "ERROR"
        assert result.data_quality == "LOW"
        assert any("RSI" in e and "150" in e for e in result.hard_errors)

    async def test_rsi_below_0_is_error(self, validator):
        """RSI < 0 is mathematically impossible."""
        result = await validator.validate(
            _base_signal(rsi=-5), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("RSI" in e for e in result.hard_errors)

    async def test_rsi_at_boundaries_passes(self, validator):
        """RSI exactly 0 or 100 should be valid."""
        for rsi_val in (0, 100):
            result = await validator.validate(
                _base_signal(rsi=rsi_val), run_llm=False
            )
            assert result.status == "OK", f"RSI={rsi_val} should be valid"

    async def test_adx_above_100_is_error(self, validator):
        """ADX > 100 is impossible."""
        result = await validator.validate(
            _base_signal(adx=120), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("ADX" in e for e in result.hard_errors)

    async def test_confidence_above_100_is_error(self, validator):
        """Confidence > 100 is impossible."""
        result = await validator.validate(
            _base_signal(confidence=105), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("Confidence" in e for e in result.hard_errors)

    async def test_negative_price_is_error(self, validator):
        """Price must be > 0."""
        result = await validator.validate(
            _base_signal(price=-100), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("Price" in e for e in result.hard_errors)

    async def test_zero_price_is_error(self, validator):
        """Price = 0 is invalid."""
        result = await validator.validate(
            _base_signal(price=0), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("Price" in e for e in result.hard_errors)

    async def test_negative_volume_is_error(self, validator):
        """Volume cannot be negative."""
        result = await validator.validate(
            _base_signal(volume=-500), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("Volume" in e for e in result.hard_errors)

    async def test_zero_volume_passes(self, validator):
        """Volume = 0 is valid (low liquidity)."""
        result = await validator.validate(
            _base_signal(volume=0), run_llm=False
        )
        assert result.status == "OK"

    async def test_entry_zone_inverted_is_error(self, validator):
        """entry_low > entry_high is a geometry bug."""
        result = await validator.validate(
            _base_signal(entry_low=65000, entry_high=63000), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("inverted" in e.lower() for e in result.hard_errors)

    async def test_long_sl_above_entry_is_error(self, validator):
        """LONG signal with SL >= entry_low is a wiring bug."""
        result = await validator.validate(
            _base_signal(direction="LONG", stop_loss=64000, entry_low=63000),
            run_llm=False,
        )
        assert result.status == "ERROR"
        assert any("LONG" in e and "SL" in e for e in result.hard_errors)

    async def test_short_sl_below_entry_is_error(self, validator):
        """SHORT signal with SL <= entry_high is a wiring bug."""
        result = await validator.validate(
            _base_signal(
                direction="SHORT",
                entry_low=63000, entry_high=63500,
                stop_loss=63000,
            ),
            run_llm=False,
        )
        assert result.status == "ERROR"
        assert any("SHORT" in e and "SL" in e for e in result.hard_errors)

    async def test_negative_atr_is_error(self, validator):
        """ATR cannot be negative."""
        result = await validator.validate(
            _base_signal(atr=-0.5), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("ATR" in e for e in result.hard_errors)

    async def test_negative_rr_ratio_is_error(self, validator):
        """R:R ratio cannot be negative."""
        result = await validator.validate(
            _base_signal(rr_ratio=-1.5), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("R:R" in e for e in result.hard_errors)

    async def test_multiple_errors_all_captured(self, validator):
        """Multiple violations should all be reported."""
        result = await validator.validate(
            _base_signal(rsi=200, price=-10, volume=-50),
            run_llm=False,
        )
        assert result.status == "ERROR"
        assert len(result.hard_errors) >= 3

    async def test_non_numeric_rsi_is_error(self, validator):
        """Non-numeric RSI should be caught."""
        result = await validator.validate(
            _base_signal(rsi="not_a_number"), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("not a number" in e for e in result.hard_errors)

    async def test_missing_optional_fields_passes(self, validator):
        """Minimal data without optional fields should pass."""
        data = {"symbol": "ETH/USDT", "direction": "LONG", "confidence": 70}
        result = await validator.validate(data, run_llm=False)
        assert result.status == "OK"

    async def test_entry_mid_used_when_price_missing(self, validator):
        """entry_mid should be checked if price is not present."""
        data = _base_signal()
        del data["price"]
        data["entry_mid"] = -100
        result = await validator.validate(data, run_llm=False)
        assert result.status == "ERROR"
        assert any("Price" in e for e in result.hard_errors)


# ══════════════════════════════════════════════════════════════════════
# LAYER B — LLM ANOMALY DETECTION (mocked)
# ══════════════════════════════════════════════════════════════════════

class TestLayerBLLMCheck:
    """Test the LLM-based anomaly detection layer (mocked)."""

    async def test_llm_warning_detected(self, validator):
        """LLM returning a WARNING should populate llm_warnings."""
        llm_parsed = {
            "status": "WARNING",
            "data_confidence": 35,
            "issues": ["RSI is 80 but signal is BUY — overbought condition"],
        }
        # call_llm returns raw text; parse_json_response converts it to dict
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value='{"status":"WARNING"}'):
            with patch("utils.free_llm.parse_json_response", return_value=llm_parsed):
                # Force LLM to run (reset cooldown)
                validator._last_llm_validation = 0
                result = await validator.validate(
                    _base_signal(rsi=80), run_llm=True
                )
                assert result.status == "WARNING"
                assert result.llm_confidence_in_data == 35
                assert len(result.llm_warnings) >= 1
                assert result.data_quality in ("LOW", "MEDIUM")

    async def test_llm_ok_passes_clean(self, validator):
        """LLM returning OK should not add warnings."""
        # call_llm raw text is irrelevant — parse_json_response mock controls the parsed result
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value='{"status":"OK"}'):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "OK",
                "data_confidence": 95,
                "issues": [],
            }):
                validator._last_llm_validation = 0
                result = await validator.validate(
                    _base_signal(), run_llm=True
                )
                assert result.status == "OK"
                assert result.data_quality == "HIGH"
                assert result.llm_warnings == []

    async def test_llm_failure_graceful(self, validator):
        """LLM returning None should not crash — just skip Layer B."""
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value=None):
            validator._last_llm_validation = 0
            result = await validator.validate(
                _base_signal(), run_llm=True
            )
            assert result.status == "OK"  # Layer A passes, Layer B skipped

    async def test_llm_parse_failure_graceful(self, validator):
        """Unparseable LLM response should not crash."""
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value="not json"):
            with patch("utils.free_llm.parse_json_response", return_value=None):
                validator._last_llm_validation = 0
                result = await validator.validate(
                    _base_signal(), run_llm=True
                )
                assert result.status == "OK"

    async def test_llm_throttled_by_cooldown(self, validator):
        """LLM should not run if cooldown hasn't elapsed."""
        import time
        validator._last_llm_validation = time.time()  # just ran
        mock_llm = AsyncMock(return_value='{"status":"OK","data_confidence":90,"issues":[]}')
        with patch("utils.free_llm.call_llm", mock_llm):
            result = await validator.validate(_base_signal(), run_llm=True)
            # LLM should NOT have been called
            mock_llm.assert_not_called()
            assert result.status == "OK"

    async def test_llm_run_false_skips_layer_b(self, validator):
        """run_llm=False should skip Layer B entirely."""
        mock_llm = AsyncMock()
        with patch("utils.free_llm.call_llm", mock_llm):
            validator._last_llm_validation = 0
            result = await validator.validate(_base_signal(), run_llm=False)
            mock_llm.assert_not_called()

    async def test_llm_low_confidence_without_issues(self, validator):
        """Low data_confidence even without explicit issues should warn."""
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value='{}'):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "OK",
                "data_confidence": 30,
                "issues": [],
            }):
                validator._last_llm_validation = 0
                result = await validator.validate(
                    _base_signal(), run_llm=True
                )
                assert result.status == "WARNING"
                assert result.llm_confidence_in_data == 30
                assert any("low" in w.lower() for w in result.llm_warnings)


# ══════════════════════════════════════════════════════════════════════
# DATA QUALITY SCORING
# ══════════════════════════════════════════════════════════════════════

class TestDataQuality:
    """Test the data quality classification."""

    async def test_clean_data_is_high_quality(self, validator):
        result = await validator.validate(_base_signal(), run_llm=False)
        assert result.data_quality == "HIGH"

    async def test_hard_error_is_low_quality(self, validator):
        result = await validator.validate(
            _base_signal(rsi=200), run_llm=False
        )
        assert result.data_quality == "LOW"

    async def test_llm_very_low_confidence_is_low_quality(self, validator):
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value='{}'):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "WARNING",
                "data_confidence": 20,
                "issues": ["Multiple indicators conflict"],
            }):
                validator._last_llm_validation = 0
                result = await validator.validate(
                    _base_signal(), run_llm=True
                )
                assert result.data_quality == "LOW"

    async def test_llm_moderate_confidence_is_medium_quality(self, validator):
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value='{}'):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "WARNING",
                "data_confidence": 55,
                "issues": ["Mild trend/signal mismatch"],
            }):
                validator._last_llm_validation = 0
                result = await validator.validate(
                    _base_signal(), run_llm=True
                )
                assert result.data_quality == "MEDIUM"


# ══════════════════════════════════════════════════════════════════════
# STATS TRACKING
# ══════════════════════════════════════════════════════════════════════

class TestStats:
    """Test validation statistics tracking."""

    async def test_stats_increment_on_clean(self, validator):
        await validator.validate(_base_signal(), run_llm=False)
        stats = validator.get_stats()
        assert stats["total_checked"] == 1
        assert stats["passed_clean"] == 1
        assert stats["hard_errors"] == 0

    async def test_stats_increment_on_error(self, validator):
        await validator.validate(_base_signal(rsi=200), run_llm=False)
        stats = validator.get_stats()
        assert stats["total_checked"] == 1
        assert stats["hard_errors"] == 1
        assert stats["passed_clean"] == 0

    async def test_stats_multiple_calls(self, validator):
        await validator.validate(_base_signal(), run_llm=False)
        await validator.validate(_base_signal(rsi=200), run_llm=False)
        await validator.validate(_base_signal(), run_llm=False)
        stats = validator.get_stats()
        assert stats["total_checked"] == 3
        assert stats["hard_errors"] == 1
        assert stats["passed_clean"] == 2


# ══════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ══════════════════════════════════════════════════════════════════════

class TestPromptBuilding:
    """Test the LLM prompt generation."""

    def test_prompt_includes_direction(self):
        prompt = SignalValidator._build_llm_prompt(
            _base_signal(direction="SHORT")
        )
        assert "SHORT" in prompt
        assert "Direction:" in prompt

    def test_prompt_includes_indicators(self):
        prompt = SignalValidator._build_llm_prompt(
            _base_signal(rsi=80, adx=45)
        )
        assert "rsi: 80" in prompt
        assert "adx: 45" in prompt

    def test_prompt_includes_validation_rules(self):
        prompt = SignalValidator._build_llm_prompt(_base_signal())
        assert "RSI > 70" in prompt
        assert "overbought" in prompt
        assert "oversold" in prompt

    def test_prompt_handles_missing_indicators(self):
        data = {"direction": "LONG", "strategy": "Test", "confidence": 70}
        prompt = SignalValidator._build_llm_prompt(data)
        assert "no indicators provided" in prompt

    def test_prompt_mentions_reversal_exception(self):
        prompt = SignalValidator._build_llm_prompt(_base_signal())
        assert "Reversal" in prompt or "reversal" in prompt


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

class TestHelpers:
    """Test helper functions."""

    def test_summarise_data_basic(self):
        summary = _summarise_data({"symbol": "BTC/USDT", "rsi": 55})
        assert "BTC/USDT" in summary
        assert "rsi=55" in summary

    def test_summarise_data_empty(self):
        assert _summarise_data({}) == "(empty)"

    def test_validation_result_defaults(self):
        r = ValidationResult()
        assert r.status == "OK"
        assert r.data_quality == "HIGH"
        assert r.hard_errors == []
        assert r.llm_warnings == []
        assert r.llm_confidence_in_data == 100
        assert r.kill_switch is False
        assert r.dynamic_penalty == 0

    def test_dynamic_penalty_formula(self):
        """penalty = round((100 - data_confidence) * 0.1), clamped [0, 10]"""
        assert _calc_dynamic_penalty(30) == 7   # (100-30)*0.1 = 7
        assert _calc_dynamic_penalty(10) == 9   # (100-10)*0.1 = 9
        assert _calc_dynamic_penalty(60) == 4   # (100-60)*0.1 = 4
        assert _calc_dynamic_penalty(85) == 2   # (100-85)*0.1 = 1.5 → round=2
        assert _calc_dynamic_penalty(100) == 0  # no penalty
        assert _calc_dynamic_penalty(0) == 10   # max penalty

    def test_dynamic_penalty_clamps(self):
        """Extreme values should clamp to [0, 10]."""
        assert _calc_dynamic_penalty(-50) == 10  # below 0 → clamp to 0 → penalty=10
        assert _calc_dynamic_penalty(200) == 0   # above 100 → clamp to 100 → penalty=0


# ══════════════════════════════════════════════════════════════════════
# HARD_RULES CONFIG
# ══════════════════════════════════════════════════════════════════════

class TestHardRulesConfig:
    """Verify the hard rules configuration."""

    def test_rsi_bounds(self):
        assert HARD_RULES["rsi"]["min"] == 0
        assert HARD_RULES["rsi"]["max"] == 100

    def test_adx_bounds(self):
        assert HARD_RULES["adx"]["min"] == 0
        assert HARD_RULES["adx"]["max"] == 100

    def test_confidence_bounds(self):
        assert HARD_RULES["confidence"]["min"] == 0
        assert HARD_RULES["confidence"]["max"] == 100


# ══════════════════════════════════════════════════════════════════════
# CROSS-FIELD CONSISTENCY CHECKS (Layer A extension)
# ══════════════════════════════════════════════════════════════════════

class TestCrossFieldChecks:
    """Test cross-field consistency validation."""

    async def test_trend_strategy_low_adx_is_error(self, validator):
        """Trend strategy with ADX < 20 should flag."""
        result = await validator.validate(
            _base_signal(strategy="Momentum", adx=15), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("ADX" in e and "trend" in e.lower() for e in result.hard_errors)

    async def test_trend_strategy_good_adx_passes(self, validator):
        """Trend strategy with ADX >= 20 should be fine."""
        result = await validator.validate(
            _base_signal(strategy="Momentum", adx=25), run_llm=False
        )
        assert result.status == "OK"

    async def test_reversal_strategy_skips_cross_field(self, validator):
        """Reversal strategies should NOT be flagged for cross-field issues."""
        result = await validator.validate(
            _base_signal(strategy="ExtremeReversal", adx=10, regime="BULL_TREND",
                         direction="SHORT", stop_loss=64000,
                         entry_low=63000, entry_high=63500), run_llm=False
        )
        # Should pass — reversal exemption skips cross-field checks
        assert result.status == "OK"

    async def test_mean_reversion_skips_cross_field(self, validator):
        """MeanReversion should also be exempt."""
        result = await validator.validate(
            _base_signal(strategy="MeanReversion", adx=10), run_llm=False
        )
        assert result.status == "OK"

    async def test_short_in_bull_trend_is_error(self, validator):
        """Non-reversal SHORT in BULL_TREND should flag."""
        result = await validator.validate(
            _base_signal(direction="SHORT", regime="BULL_TREND",
                         strategy="Momentum", stop_loss=64000), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("SHORT" in e and "BULL" in e for e in result.hard_errors)

    async def test_long_in_bear_trend_is_error(self, validator):
        """Non-reversal LONG in BEAR_TREND should flag."""
        result = await validator.validate(
            _base_signal(direction="LONG", regime="BEAR_TREND",
                         strategy="PriceAction"), run_llm=False
        )
        assert result.status == "ERROR"
        assert any("LONG" in e and "BEAR" in e for e in result.hard_errors)

    async def test_neutral_regime_passes(self, validator):
        """Neutral/ranging regime should not trigger direction checks."""
        result = await validator.validate(
            _base_signal(direction="LONG", regime="RANGING"), run_llm=False
        )
        assert result.status == "OK"

    async def test_breakout_low_volume_is_error(self, validator):
        """InstitutionalBreakout with low volume ratio should flag."""
        result = await validator.validate(
            _base_signal(strategy="InstitutionalBreakout", volume_ratio=0.5),
            run_llm=False,
        )
        assert result.status == "ERROR"
        assert any("volume" in e.lower() for e in result.hard_errors)

    async def test_breakout_normal_volume_passes(self, validator):
        """InstitutionalBreakout with good volume should pass."""
        result = await validator.validate(
            _base_signal(strategy="InstitutionalBreakout", volume_ratio=1.5),
            run_llm=False,
        )
        assert result.status == "OK"

    async def test_non_breakout_strategy_ignores_volume(self, validator):
        """Non-breakout strategy should not flag low volume."""
        result = await validator.validate(
            _base_signal(strategy="Momentum", volume_ratio=0.3),
            run_llm=False,
        )
        assert result.status == "OK"


# ══════════════════════════════════════════════════════════════════════
# KILL SWITCH
# ══════════════════════════════════════════════════════════════════════

class TestKillSwitch:
    """Test the kill-switch mechanism."""

    async def test_very_low_llm_confidence_triggers_kill(self, validator):
        """data_confidence < 20 should trigger kill switch."""
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value='{}'):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "WARNING",
                "data_confidence": 10,
                "issues": ["Data looks completely wrong"],
            }):
                validator._last_llm_validation = 0
                result = await validator.validate(_base_signal(), run_llm=True)
                assert result.kill_switch is True
                assert result.status == "ERROR"
                assert result.data_quality == "LOW"

    async def test_moderate_confidence_no_kill(self, validator):
        """data_confidence >= 20 should NOT trigger kill switch."""
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value='{}'):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "WARNING",
                "data_confidence": 35,
                "issues": ["Mild inconsistency"],
            }):
                validator._last_llm_validation = 0
                result = await validator.validate(_base_signal(), run_llm=True)
                assert result.kill_switch is False
                assert result.status == "WARNING"

    async def test_multiple_hard_errors_triggers_kill(self, validator):
        """3+ hard errors should trigger kill switch."""
        result = await validator.validate(
            _base_signal(rsi=200, price=-10, volume=-50), run_llm=False
        )
        assert result.kill_switch is True
        assert result.status == "ERROR"

    async def test_single_hard_error_no_kill(self, validator):
        """Single hard error should NOT trigger kill switch."""
        result = await validator.validate(
            _base_signal(rsi=200), run_llm=False
        )
        assert result.kill_switch is False
        assert result.status == "ERROR"

    async def test_kill_switch_stats(self, validator):
        """Kill switches should be counted in stats."""
        result = await validator.validate(
            _base_signal(rsi=200, price=-10, volume=-50), run_llm=False
        )
        assert validator.get_stats()["kill_switches"] == 1


# ══════════════════════════════════════════════════════════════════════
# DYNAMIC PENALTY
# ══════════════════════════════════════════════════════════════════════

class TestDynamicPenalty:
    """Test dynamic penalty calculation on ValidationResult."""

    async def test_low_confidence_high_penalty(self, validator):
        """Low LLM confidence should produce higher penalty."""
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value='{}'):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "WARNING",
                "data_confidence": 30,
                "issues": ["Conflicting signals"],
            }):
                validator._last_llm_validation = 0
                result = await validator.validate(_base_signal(), run_llm=True)
                assert result.dynamic_penalty == 7  # (100-30)*0.1

    async def test_high_confidence_low_penalty(self, validator):
        """High LLM confidence should produce lower penalty."""
        with patch("utils.free_llm.call_llm", new_callable=AsyncMock, return_value='{}'):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "WARNING",
                "data_confidence": 60,
                "issues": ["Minor concern"],
            }):
                validator._last_llm_validation = 0
                result = await validator.validate(_base_signal(), run_llm=True)
                assert result.dynamic_penalty == 4  # (100-60)*0.1


# ══════════════════════════════════════════════════════════════════════
# PRIORITY-BASED LLM COOLDOWN BYPASS
# ══════════════════════════════════════════════════════════════════════

class TestPriorityCooldownBypass:
    """Test that high-risk signals bypass LLM cooldown."""

    async def test_low_confidence_bypasses_cooldown(self, validator):
        """Signals with confidence < 60 should always get LLM validation."""
        import time
        validator._last_llm_validation = time.time()  # just ran — normally blocked
        mock_llm = AsyncMock(return_value='{"status":"OK","data_confidence":90,"issues":[]}')
        with patch("utils.free_llm.call_llm", mock_llm):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "OK", "data_confidence": 90, "issues": [],
            }):
                await validator.validate(
                    _base_signal(confidence=50), run_llm=True
                )
                # LLM SHOULD have been called despite cooldown
                mock_llm.assert_called_once()

    async def test_high_rr_bypasses_cooldown(self, validator):
        """Signals with R:R > 5.0 should always get LLM validation."""
        import time
        validator._last_llm_validation = time.time()
        mock_llm = AsyncMock(return_value='{"status":"OK","data_confidence":90,"issues":[]}')
        with patch("utils.free_llm.call_llm", mock_llm):
            with patch("utils.free_llm.parse_json_response", return_value={
                "status": "OK", "data_confidence": 90, "issues": [],
            }):
                await validator.validate(
                    _base_signal(rr_ratio=6.0), run_llm=True
                )
                mock_llm.assert_called_once()

    async def test_normal_signal_respects_cooldown(self, validator):
        """Normal signals should respect cooldown."""
        import time
        validator._last_llm_validation = time.time()
        mock_llm = AsyncMock(return_value='{"status":"OK","data_confidence":90,"issues":[]}')
        with patch("utils.free_llm.call_llm", mock_llm):
            await validator.validate(
                _base_signal(confidence=75, rr_ratio=2.5), run_llm=True
            )
            mock_llm.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════

class TestDriftDetection:
    """Test the rolling drift detection system."""

    async def test_drift_rate_zero_when_clean(self, validator):
        """No flags → 0% drift rate."""
        for _ in range(10):
            await validator.validate(_base_signal(), run_llm=False)
        assert validator.get_drift_rate() == 0.0

    async def test_drift_rate_tracks_flags(self, validator):
        """Mix of clean and flagged signals should give correct rate."""
        for _ in range(8):
            await validator.validate(_base_signal(), run_llm=False)
        for _ in range(2):
            await validator.validate(_base_signal(rsi=200), run_llm=False)
        # 2 out of 10 → 20%
        assert abs(validator.get_drift_rate() - 0.2) < 0.01

    async def test_drift_alert_on_high_flag_rate(self, validator):
        """If >30% of signals are flagged, drift_alerts should increment."""
        # Fill the window with mostly flagged signals
        for i in range(DRIFT_WINDOW_SIZE):
            if i % 2 == 0:  # 50% flagged → well above 30%
                await validator.validate(_base_signal(rsi=200), run_llm=False)
            else:
                await validator.validate(_base_signal(), run_llm=False)
        assert validator.get_stats()["drift_alerts"] > 0

    async def test_no_drift_alert_below_threshold(self, validator):
        """If flag rate is below 30%, no drift alert."""
        # Only 10% flagged
        for i in range(DRIFT_WINDOW_SIZE):
            if i % 10 == 0:  # 10% flagged
                await validator.validate(_base_signal(rsi=200), run_llm=False)
            else:
                await validator.validate(_base_signal(), run_llm=False)
        assert validator.get_stats()["drift_alerts"] == 0

    def test_drift_rate_empty_history(self, validator):
        """Empty history → 0 drift rate."""
        assert validator.get_drift_rate() == 0.0

    async def test_drift_stats_in_get_stats(self, validator):
        """get_stats() should include drift_flag_rate."""
        await validator.validate(_base_signal(), run_llm=False)
        stats = validator.get_stats()
        assert "drift_flag_rate" in stats
        assert "drift_alerts" in stats


# ══════════════════════════════════════════════════════════════════════
# STRATEGY CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

class TestStrategyClassification:
    """Verify strategy classification sets."""

    def test_trend_strategies_are_frozenset(self):
        assert isinstance(TREND_STRATEGIES, frozenset)
        assert "Momentum" in TREND_STRATEGIES
        assert "Ichimoku" in TREND_STRATEGIES

    def test_reversal_strategies_are_frozenset(self):
        assert isinstance(REVERSAL_STRATEGIES, frozenset)
        assert "ExtremeReversal" in REVERSAL_STRATEGIES
        assert "MeanReversion" in REVERSAL_STRATEGIES

    def test_no_overlap(self):
        """Trend and reversal sets must not overlap."""
        assert TREND_STRATEGIES & REVERSAL_STRATEGIES == frozenset()
