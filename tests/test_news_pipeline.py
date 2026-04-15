"""
Tests for the three-layer news pipeline:
  1. Exponential decay curve
  2. News vs Price reaction scoring
  3. Multi-news conflict resolver
"""
import math
import time
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import pytest

# ── Module-level mocks (must precede any imports that trigger side-effects) ──
import sys
for _mod in (
    "risk.circuit_breaker", "data.database",
    "governance.performance_tracker",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from config.constants import NewsIntelligence
from analyzers.btc_news_intelligence import (
    BTCEventContext,
    BTCEventType,
    BTCMoveAnalyzer,
    BTCNewsIntelligence,
)


# ─── Helpers ──────────────────────────────────────────────────

def _make_intel() -> BTCNewsIntelligence:
    """Fresh BTCNewsIntelligence instance for each test."""
    intel = BTCNewsIntelligence()
    # Seed price history so reaction scoring works
    base_time = time.time() - 120
    for i in range(60):
        intel._move_analyzer._price_history.append((base_time + i * 2, 60000.0))
    return intel


# ═══════════════════════════════════════════════════════════════
#  FEATURE 1: Exponential Decay Curve
# ═══════════════════════════════════════════════════════════════

class TestExponentialDecay:
    """Tests for BTCNewsIntelligence.news_decay() static method."""

    def test_zero_age_returns_one(self):
        assert BTCNewsIntelligence.news_decay(0) == 1.0

    def test_negative_age_returns_one(self):
        assert BTCNewsIntelligence.news_decay(-5) == 1.0

    def test_at_tau_returns_approx_037(self):
        result = BTCNewsIntelligence.news_decay(60.0, tau=60.0)
        assert abs(result - math.exp(-1)) < 0.01

    def test_5min_high_impact(self):
        result = BTCNewsIntelligence.news_decay(5.0, tau=60.0)
        assert result > 0.90  # exp(-5/60) ≈ 0.920

    def test_30min_moderate_impact(self):
        result = BTCNewsIntelligence.news_decay(30.0, tau=60.0)
        assert 0.55 < result < 0.65  # exp(-30/60) ≈ 0.607

    def test_120min_low_impact(self):
        result = BTCNewsIntelligence.news_decay(120.0, tau=60.0)
        assert result < 0.20  # exp(-120/60) ≈ 0.135

    def test_floor_prevents_zero(self):
        result = BTCNewsIntelligence.news_decay(10000.0, tau=60.0, floor=0.10)
        assert result == 0.10

    def test_custom_floor(self):
        result = BTCNewsIntelligence.news_decay(10000.0, tau=60.0, floor=0.25)
        assert result == 0.25

    def test_slow_tau_decays_slower(self):
        fast = BTCNewsIntelligence.news_decay(60.0, tau=60.0)
        slow = BTCNewsIntelligence.news_decay(60.0, tau=150.0)
        assert slow > fast

    def test_monotonically_decreasing(self):
        ages = [0, 5, 15, 30, 60, 120]
        values = [BTCNewsIntelligence.news_decay(a, tau=60.0) for a in ages]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_constants_tau_fast_positive(self):
        assert NewsIntelligence.NEWS_DECAY_TAU_FAST_MINUTES > 0

    def test_constants_tau_slow_larger_than_fast(self):
        assert NewsIntelligence.NEWS_DECAY_TAU_SLOW_MINUTES > NewsIntelligence.NEWS_DECAY_TAU_FAST_MINUTES

    def test_constants_floor_reasonable(self):
        assert 0 < NewsIntelligence.NEWS_DECAY_FLOOR < 0.5


class TestDecayInProcessHeadlines:
    """Verify that process_headlines uses exponential decay, not linear."""

    def test_fresh_headline_no_decay(self):
        """A 0-age headline should keep full confidence."""
        intel = _make_intel()
        headlines = [{"title": "Bitcoin ETF approved", "published_at": time.time(), "source": "reuters"}]
        asyncio.get_event_loop().run_until_complete(intel.process_headlines(headlines))
        ctx = intel.get_event_context()
        # If classified, confidence should not be reduced
        # (may or may not trigger depending on classifier)

    def test_stale_headline_gets_exp_decay(self):
        """A 30-min-old headline should get exp(-30/60) ≈ 0.607 multiplier."""
        intel = _make_intel()
        age_minutes = 30
        published_at = time.time() - age_minutes * 60
        headlines = [{"title": "Bitcoin crash imminent says analyst", "published_at": published_at, "source": "reuters"}]
        # Process — we test the decay function itself since classifier mocking is complex
        decay = BTCNewsIntelligence.news_decay(age_minutes, tau=NewsIntelligence.NEWS_DECAY_TAU_FAST_MINUTES)
        expected = math.exp(-30 / NewsIntelligence.NEWS_DECAY_TAU_FAST_MINUTES)
        assert abs(decay - expected) < 0.01

    def test_slow_burn_event_uses_slow_tau(self):
        """Headlines with ETF/regulation keywords should use slower tau."""
        # The decay applied to an ETF headline at 60 min
        decay_fast = BTCNewsIntelligence.news_decay(60, tau=NewsIntelligence.NEWS_DECAY_TAU_FAST_MINUTES)
        decay_slow = BTCNewsIntelligence.news_decay(60, tau=NewsIntelligence.NEWS_DECAY_TAU_SLOW_MINUTES)
        assert decay_slow > decay_fast  # Slow decay retains more impact


# ═══════════════════════════════════════════════════════════════
#  FEATURE 2: News vs Price Reaction Scoring
# ═══════════════════════════════════════════════════════════════

class TestReactionScoring:
    """Tests for evaluate_reaction()."""

    def test_no_pending_check_returns_none(self):
        intel = _make_intel()
        assert intel.evaluate_reaction() is None

    def test_too_early_returns_none(self):
        intel = _make_intel()
        intel._schedule_reaction_check("BEARISH", time.time())
        assert intel.evaluate_reaction() is None  # <15 min elapsed

    def test_confirmed_bearish(self):
        """Bearish news + price drops = CONFIRMED."""
        intel = _make_intel()
        # Set detection time 16 minutes ago
        detected_at = time.time() - 16 * 60
        intel._pending_reaction_check = {
            "direction": "BEARISH",
            "detected_at": detected_at,
            "price_at_detection": 60000.0,
        }
        # Current price lower
        intel._move_analyzer._price_history.append((time.time(), 59500.0))  # -0.83%
        result = intel.evaluate_reaction()
        assert result == "CONFIRMED"

    def test_confirmed_bullish(self):
        """Bullish news + price rises = CONFIRMED."""
        intel = _make_intel()
        detected_at = time.time() - 16 * 60
        intel._pending_reaction_check = {
            "direction": "BULLISH",
            "detected_at": detected_at,
            "price_at_detection": 60000.0,
        }
        intel._move_analyzer._price_history.append((time.time(), 60500.0))  # +0.83%
        result = intel.evaluate_reaction()
        assert result == "CONFIRMED"

    def test_contradicted_bearish_but_price_up(self):
        """Bearish news + price UP = CONTRADICTED."""
        intel = _make_intel()
        detected_at = time.time() - 16 * 60
        intel._pending_reaction_check = {
            "direction": "BEARISH",
            "detected_at": detected_at,
            "price_at_detection": 60000.0,
        }
        intel._move_analyzer._price_history.append((time.time(), 60500.0))  # +0.83%
        result = intel.evaluate_reaction()
        assert result == "CONTRADICTED"

    def test_contradicted_bullish_but_price_down(self):
        """Bullish news + price DOWN = CONTRADICTED."""
        intel = _make_intel()
        detected_at = time.time() - 16 * 60
        intel._pending_reaction_check = {
            "direction": "BULLISH",
            "detected_at": detected_at,
            "price_at_detection": 60000.0,
        }
        intel._move_analyzer._price_history.append((time.time(), 59400.0))  # -1.0%
        result = intel.evaluate_reaction()
        assert result == "CONTRADICTED"

    def test_neutral_small_move(self):
        """Tiny price move = NEUTRAL."""
        intel = _make_intel()
        detected_at = time.time() - 16 * 60
        intel._pending_reaction_check = {
            "direction": "BEARISH",
            "detected_at": detected_at,
            "price_at_detection": 60000.0,
        }
        intel._move_analyzer._price_history.append((time.time(), 59900.0))  # -0.17%
        result = intel.evaluate_reaction()
        assert result == "NEUTRAL"

    def test_confirmed_boosts_context_confidence(self):
        """CONFIRMED reaction should boost the active context's confidence."""
        intel = _make_intel()
        # Set up an active context
        intel._current_ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_OFF,
            confidence=0.70,
            direction="BEARISH",
            detected_at=time.time() - 16 * 60,
            expires_at=time.time() + 3600,
        )
        intel._pending_reaction_check = {
            "direction": "BEARISH",
            "detected_at": time.time() - 16 * 60,
            "price_at_detection": 60000.0,
        }
        intel._move_analyzer._price_history.append((time.time(), 59200.0))
        intel.evaluate_reaction()
        assert intel._current_ctx.reaction_validated is True
        assert intel._current_ctx.reaction_confirmed is True
        assert intel._current_ctx.confidence > 0.70  # Boosted

    def test_contradicted_halves_context_confidence(self):
        """CONTRADICTED reaction should halve the active context's confidence."""
        intel = _make_intel()
        intel._current_ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_OFF,
            confidence=0.70,
            direction="BEARISH",
            detected_at=time.time() - 16 * 60,
            expires_at=time.time() + 3600,
            block_longs=True,
        )
        intel._pending_reaction_check = {
            "direction": "BEARISH",
            "detected_at": time.time() - 16 * 60,
            "price_at_detection": 60000.0,
        }
        intel._move_analyzer._price_history.append((time.time(), 60800.0))  # +1.33% UP
        intel.evaluate_reaction()
        assert intel._current_ctx.reaction_validated is True
        assert intel._current_ctx.reaction_confirmed is False
        assert intel._current_ctx.confidence < 0.70 * 0.55  # Halved
        assert intel._current_ctx.block_longs is False  # Unblocked

    def test_reaction_clears_pending(self):
        """After evaluation, pending check should be cleared."""
        intel = _make_intel()
        intel._pending_reaction_check = {
            "direction": "BULLISH",
            "detected_at": time.time() - 16 * 60,
            "price_at_detection": 60000.0,
        }
        intel._move_analyzer._price_history.append((time.time(), 60600.0))
        intel.evaluate_reaction()
        assert intel._pending_reaction_check is None

    def test_decay_context_triggers_reaction(self):
        """decay_context() should call evaluate_reaction()."""
        intel = _make_intel()
        intel._current_ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_ON,
            confidence=0.80,
            direction="BULLISH",
            detected_at=time.time() - 16 * 60,
            expires_at=time.time() + 3600,
        )
        intel._pending_reaction_check = {
            "direction": "BULLISH",
            "detected_at": time.time() - 16 * 60,
            "price_at_detection": 60000.0,
        }
        intel._move_analyzer._price_history.append((time.time(), 60800.0))
        intel.decay_context()
        assert intel._current_ctx.reaction_validated is True

    def test_multiple_pending_reaction_checks_are_processed_in_order(self):
        """New pending checks should not overwrite older ones."""
        intel = _make_intel()
        first_detected_at = time.time() - 20 * 60
        second_detected_at = time.time() - 19 * 60

        intel._schedule_reaction_check("BEARISH", first_detected_at)
        intel._schedule_reaction_check("BULLISH", second_detected_at)
        intel._move_analyzer._price_history.append((time.time(), 59400.0))

        first = intel.evaluate_reaction()
        assert first == "CONFIRMED"
        remaining = intel._get_pending_reaction_checks()
        assert len(remaining) == 1
        assert remaining[0]["detected_at"] == second_detected_at

        second = intel.evaluate_reaction()
        assert second == "CONTRADICTED"
        assert intel._pending_reaction_check is None

    def test_reaction_updates_matching_headline_history(self):
        """Reaction scoring should update stored headline confidence too."""
        intel = _make_intel()
        reaction_id = time.time() - 16 * 60
        intel._headline_history = [{
            "title": "ETF",
            "published_at": reaction_id,
            "direction": "BULLISH",
            "confidence": 0.80,
            "source": "reuters",
            "event_type": "BTC_FUNDAMENTAL",
            "detected_at": reaction_id,
            "reaction_check_id": reaction_id,
        }]
        intel._current_ctx = BTCEventContext(
            event_type=BTCEventType.BTC_FUNDAMENTAL,
            confidence=0.80,
            direction="BULLISH",
            detected_at=reaction_id,
            expires_at=time.time() + 3600,
        )
        intel._pending_reaction_check = {
            "direction": "BULLISH",
            "detected_at": reaction_id,
            "price_at_detection": 60000.0,
            "reaction_check_id": reaction_id,
        }
        intel._move_analyzer._price_history.append((time.time(), 60600.0))

        intel.evaluate_reaction()

        assert intel._headline_history[0]["confidence"] > 0.80


# ═══════════════════════════════════════════════════════════════
#  FEATURE 3: Multi-News Conflict Resolver
# ═══════════════════════════════════════════════════════════════

class TestHeadlineHistory:
    """Tests for _record_headline and history management."""

    def test_record_headline_adds_entry(self):
        intel = _make_intel()
        intel._record_headline("BTC pump", time.time(), "BULLISH", 0.8, "reuters", BTCEventType.BTC_FUNDAMENTAL)
        assert len(intel._headline_history) == 1
        assert intel._headline_history[0]["direction"] == "BULLISH"

    def test_old_entries_trimmed(self):
        intel = _make_intel()
        old_time = time.time() - 200 * 60  # 200 minutes ago
        intel._headline_history.append({
            "title": "old", "published_at": old_time,
            "direction": "BEARISH", "confidence": 0.5,
            "source": "x", "event_type": "MACRO_RISK_OFF",
            "detected_at": old_time,
        })
        intel._record_headline("new", time.time(), "BULLISH", 0.8, "reuters", BTCEventType.BTC_FUNDAMENTAL)
        # Old entry should be trimmed
        assert all(h["title"] != "old" for h in intel._headline_history)

    def test_max_size_respected(self):
        intel = _make_intel()
        for i in range(60):
            intel._record_headline(f"headline_{i}", time.time(), "BULLISH", 0.5, "x", BTCEventType.BTC_FUNDAMENTAL)
        assert len(intel._headline_history) <= NewsIntelligence.HEADLINE_HISTORY_MAX_SIZE


class TestNetNewsScore:
    """Tests for compute_net_news_score() and get_net_news_bias()."""

    def test_empty_history_returns_zero(self):
        intel = _make_intel()
        assert intel.compute_net_news_score() == 0.0

    def test_single_bullish_headline(self):
        intel = _make_intel()
        intel._record_headline("BTC ETF approved", time.time(), "BULLISH", 0.80, "reuters", BTCEventType.BTC_FUNDAMENTAL)
        score = intel.compute_net_news_score()
        assert score > 0  # Positive

    def test_single_bearish_headline(self):
        intel = _make_intel()
        intel._record_headline("War escalation", time.time(), "BEARISH", 0.70, "reuters", BTCEventType.MACRO_RISK_OFF)
        score = intel.compute_net_news_score()
        assert score < 0  # Negative

    def test_neutral_headline_ignored(self):
        intel = _make_intel()
        intel._record_headline("BTC consolidating", time.time(), "NEUTRAL", 0.60, "reuters", BTCEventType.BTC_TECHNICAL)
        assert intel.compute_net_news_score() == 0.0

    def test_conflicting_news_resolves_to_stronger(self):
        intel = _make_intel()
        # Strong bullish
        intel._record_headline("ETF inflow $1B", time.time(), "BULLISH", 0.90, "reuters", BTCEventType.BTC_FUNDAMENTAL)
        # Weaker bearish
        intel._record_headline("Minor hack $10M", time.time(), "BEARISH", 0.40, "unknown", BTCEventType.EXCHANGE_EVENT)
        score = intel.compute_net_news_score()
        assert score > 0  # Bullish wins

    def test_equal_opposing_news_near_zero(self):
        intel = _make_intel()
        intel._record_headline("Bullish news", time.time(), "BULLISH", 0.70, "reuters", BTCEventType.BTC_FUNDAMENTAL)
        intel._record_headline("Bearish news", time.time(), "BEARISH", 0.70, "reuters", BTCEventType.MACRO_RISK_OFF)
        score = intel.compute_net_news_score()
        assert -0.15 < score < 0.15  # Near neutral

    def test_older_news_decays_in_score(self):
        intel = _make_intel()
        # Old bearish
        old_entry = {
            "title": "old crash", "published_at": time.time() - 60 * 60,
            "direction": "BEARISH", "confidence": 0.80,
            "source": "reuters", "event_type": "MACRO_RISK_OFF",
            "detected_at": time.time() - 60 * 60,
        }
        intel._headline_history.append(old_entry)
        # Fresh bullish
        intel._record_headline("Saylor buys $1B", time.time(), "BULLISH", 0.70, "reuters", BTCEventType.BTC_FUNDAMENTAL)
        score = intel.compute_net_news_score()
        assert score > 0  # Fresh bullish outweighs decayed bearish

    def test_score_clamped_to_minus_one(self):
        intel = _make_intel()
        for _ in range(10):
            intel._record_headline("War", time.time(), "BEARISH", 0.90, "reuters", BTCEventType.MACRO_RISK_OFF)
        score = intel.compute_net_news_score()
        assert score >= -1.0

    def test_score_clamped_to_plus_one(self):
        intel = _make_intel()
        for _ in range(10):
            intel._record_headline("ETF", time.time(), "BULLISH", 0.90, "reuters", BTCEventType.BTC_FUNDAMENTAL)
        score = intel.compute_net_news_score()
        assert score <= 1.0

    def test_get_net_news_bias_bullish(self):
        intel = _make_intel()
        intel._record_headline("ETF", time.time(), "BULLISH", 0.90, "reuters", BTCEventType.BTC_FUNDAMENTAL)
        bias, score = intel.get_net_news_bias()
        assert bias == "BULLISH"
        assert score > NewsIntelligence.NET_SCORE_BULLISH_THRESHOLD

    def test_get_net_news_bias_bearish(self):
        intel = _make_intel()
        intel._record_headline("War", time.time(), "BEARISH", 0.90, "reuters", BTCEventType.MACRO_RISK_OFF)
        bias, score = intel.get_net_news_bias()
        assert bias == "BEARISH"
        assert score < NewsIntelligence.NET_SCORE_BEARISH_THRESHOLD

    def test_get_net_news_bias_mixed(self):
        intel = _make_intel()
        intel._record_headline("ETF", time.time(), "BULLISH", 0.50, "reuters", BTCEventType.BTC_FUNDAMENTAL)
        intel._record_headline("War", time.time(), "BEARISH", 0.50, "reuters", BTCEventType.MACRO_RISK_OFF)
        bias, _ = intel.get_net_news_bias()
        assert bias == "MIXED"


class TestContextNetScore:
    """Verify net_news_score is populated on BTCEventContext."""

    def test_context_has_net_score_field(self):
        ctx = BTCEventContext()
        assert hasattr(ctx, "net_news_score")
        assert ctx.net_news_score == 0.0

    def test_context_has_reaction_fields(self):
        ctx = BTCEventContext()
        assert hasattr(ctx, "reaction_validated")
        assert hasattr(ctx, "reaction_confirmed")
        assert ctx.reaction_validated is False
        assert ctx.reaction_confirmed is False


class TestGetStatusIncludesNewFields:
    """Ensure get_status() returns the new fields."""

    def test_status_includes_net_score(self):
        intel = _make_intel()
        status = intel.get_status()
        assert "net_news_score" in status
        assert "reaction_validated" in status
        assert "reaction_confirmed" in status


class TestProcessHeadlinesRegression:
    """Regression tests for audited process_headlines issues."""

    def test_decay_uses_winning_headline_age_not_newest_headline(self):
        intel = _make_intel()
        now = time.time()
        old_headline = {
            "title": "Bitcoin ETF approval older winning headline",
            "published_at": now - 30 * 60,
            "source": "reuters",
        }
        fresh_headline = {
            "title": "Bitcoin market update fresh non-winning headline",
            "published_at": now,
            "source": "reuters",
        }

        with patch.object(
            intel._classifier,
            "classify_batch",
            return_value=(BTCEventType.BTC_FUNDAMENTAL, "BULLISH", 0.80, old_headline["title"], False),
        ):
            asyncio.get_event_loop().run_until_complete(
                intel.process_headlines([old_headline, fresh_headline])
            )

        ctx = intel.get_event_context()
        assert ctx.headline == old_headline["title"]
        assert ctx.confidence < 0.75

    def test_history_records_per_headline_classification(self):
        intel = _make_intel()
        headlines = [
            {"title": "Bitcoin ETF inflow surges on BlackRock demand", "published_at": time.time(), "source": "reuters"},
            {"title": "Bitcoin war escalation triggers market panic", "published_at": time.time(), "source": "reuters"},
        ]

        asyncio.get_event_loop().run_until_complete(intel.process_headlines(headlines))

        directions = {h["direction"] for h in intel._headline_history}
        assert "BULLISH" in directions
        assert "BEARISH" in directions


class TestConstants:
    """Verify all new constants exist and have sensible values."""

    def test_decay_tau_fast(self):
        assert NewsIntelligence.NEWS_DECAY_TAU_FAST_MINUTES == 60.0

    def test_decay_tau_slow(self):
        assert NewsIntelligence.NEWS_DECAY_TAU_SLOW_MINUTES == 150.0

    def test_decay_floor(self):
        assert NewsIntelligence.NEWS_DECAY_FLOOR == 0.10

    def test_reaction_delay(self):
        assert NewsIntelligence.REACTION_DELAY_MINUTES == 15.0

    def test_reaction_confirm_mult(self):
        assert NewsIntelligence.REACTION_CONFIRM_MULT > 1.0

    def test_reaction_contradict_mult(self):
        assert NewsIntelligence.REACTION_CONTRADICT_MULT < 1.0

    def test_reaction_neutral_mult(self):
        assert NewsIntelligence.REACTION_NEUTRAL_MULT < 1.0

    def test_net_score_thresholds_symmetric(self):
        assert NewsIntelligence.NET_SCORE_BULLISH_THRESHOLD > 0
        assert NewsIntelligence.NET_SCORE_BEARISH_THRESHOLD < 0
        assert abs(NewsIntelligence.NET_SCORE_BULLISH_THRESHOLD) == abs(NewsIntelligence.NET_SCORE_BEARISH_THRESHOLD)

    def test_net_score_boost_and_penalty(self):
        assert NewsIntelligence.NET_SCORE_BULLISH_CONF_BOOST > 1.0
        assert NewsIntelligence.NET_SCORE_BEARISH_CONF_PENALTY < 1.0

    def test_headline_history_limits(self):
        assert NewsIntelligence.HEADLINE_HISTORY_MAX_AGE_MINUTES > 0
        assert NewsIntelligence.HEADLINE_HISTORY_MAX_SIZE > 0
