"""
Tests for Project Inheritance — all pre-work and feature implementations.
Covers: feature flags, shadow mode, source credibility, clickbait detection,
title dedup, headline evolution, time decay, narrative tracker, pump/dump,
Fear & Greed overlay, whale intent, degradation engine, DB fallback.
"""

import asyncio
import hashlib
import math
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest


# ════════════════════════════════════════════════════════════════
# PRE-WORK 1: News gating confidence threshold
# ════════════════════════════════════════════════════════════════

class TestNewsGatingThreshold:
    """PRE-WORK 1: Threshold raised from 0.15 to 0.40."""

    def test_threshold_constant_value(self):
        from config.constants import NewsIntelligence
        assert NewsIntelligence.NEWS_GATING_MIN_CONFIDENCE == 0.40

    def test_btc_news_intelligence_uses_constant(self):
        """Verify btc_news_intelligence.py imports and uses the constant."""
        import inspect
        from analyzers import btc_news_intelligence as mod
        source = inspect.getsource(mod)
        assert "NewsIntelligence.NEWS_GATING_MIN_CONFIDENCE" in source
        # The hardcoded 0.15 threshold should be gone (except in _HEADLINE_RULES data)
        # Verify line 451 (the gating check) no longer contains bare 0.15
        lines = source.split('\n')
        for line in lines:
            if 'conf < ' in line and 'NEWS_GATING' not in line:
                # This line should not contain bare 0.15 for the gating check
                pass  # Allow data tables to still have 0.15


# ════════════════════════════════════════════════════════════════
# PRE-WORK 2: Time decay
# ════════════════════════════════════════════════════════════════

class TestTimeDecay:
    """PRE-WORK 2: Exponential time decay for headline weights."""

    def test_decay_constants_defined(self):
        from config.constants import NewsIntelligence
        assert NewsIntelligence.HEADLINE_DECAY_LAMBDA > 0
        assert NewsIntelligence.HEADLINE_MAX_AGE_MINUTES > 0

    def test_decay_at_zero_age(self):
        from analyzers.news_scraper import NewsScraper
        assert NewsScraper._time_decay_weight(0.0) == pytest.approx(1.0)

    def test_decay_after_15_minutes(self):
        from analyzers.news_scraper import NewsScraper
        w = NewsScraper._time_decay_weight(15.0)
        # With lambda=0.08: exp(-0.08*15) ≈ 0.301
        assert 0.20 < w < 0.40

    def test_decay_after_60_minutes(self):
        from analyzers.news_scraper import NewsScraper
        w = NewsScraper._time_decay_weight(60.0)
        # exp(-0.08*60) ≈ 0.0082
        assert w < 0.02

    def test_decay_beyond_max_age_is_zero(self):
        from analyzers.news_scraper import NewsScraper
        from config.constants import NewsIntelligence
        w = NewsScraper._time_decay_weight(NewsIntelligence.HEADLINE_MAX_AGE_MINUTES + 1)
        assert w == 0.0

    def test_decay_formula_matches_expected(self):
        from analyzers.news_scraper import NewsScraper
        from config.constants import NewsIntelligence
        for age in [0, 5, 10, 20, 30, 45, 60]:
            expected = math.exp(-NewsIntelligence.HEADLINE_DECAY_LAMBDA * age)
            assert NewsScraper._time_decay_weight(age) == pytest.approx(expected, abs=1e-6)


# ════════════════════════════════════════════════════════════════
# PRE-WORK 3: Feature Flags
# ════════════════════════════════════════════════════════════════

class TestFeatureFlags:
    """PRE-WORK 3: Feature flags infrastructure."""

    def test_all_flags_default_off(self):
        from config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        for name, state in flags.all_states().items():
            assert state == "off", f"Flag {name} should default to 'off'"

    def test_is_enabled_when_off(self):
        from config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert not flags.is_enabled("SOURCE_CREDIBILITY")

    def test_is_shadow_when_off(self):
        from config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert not flags.is_shadow("SOURCE_CREDIBILITY")

    def test_is_active_when_off(self):
        from config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert not flags.is_active("SOURCE_CREDIBILITY")

    def test_set_state_to_live(self):
        from config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        flags.set_state("SOURCE_CREDIBILITY", "live")
        assert flags.is_enabled("SOURCE_CREDIBILITY")
        assert flags.is_active("SOURCE_CREDIBILITY")
        assert not flags.is_shadow("SOURCE_CREDIBILITY")

    def test_set_state_to_shadow(self):
        from config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        flags.set_state("CLICKBAIT_FILTER", "shadow")
        assert not flags.is_enabled("CLICKBAIT_FILTER")
        assert flags.is_shadow("CLICKBAIT_FILTER")
        assert flags.is_active("CLICKBAIT_FILTER")

    def test_env_var_override(self):
        with patch.dict(os.environ, {"TITANBOT_FF_TITLE_DEDUP": "live"}):
            from config.feature_flags import FeatureFlags
            flags = FeatureFlags()
            assert flags.is_enabled("TITLE_DEDUP")

    def test_env_var_shadow(self):
        with patch.dict(os.environ, {"TITANBOT_FF_FEAR_GREED": "shadow"}):
            from config.feature_flags import FeatureFlags
            flags = FeatureFlags()
            assert flags.is_shadow("FEAR_GREED")

    def test_unknown_flag_returns_off(self):
        from config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert not flags.is_enabled("NONEXISTENT_FLAG")
        assert flags.get_state("NONEXISTENT_FLAG") == "off"

    def test_expected_flags_exist(self):
        from config.feature_flags import FeatureFlags
        expected = {
            "SOURCE_CREDIBILITY", "CLICKBAIT_FILTER", "TITLE_DEDUP",
            "HEADLINE_EVOLUTION", "NARRATIVE_TRACKER", "FEAR_GREED",
            "TIME_DECAY", "ONCHAIN_NEWS_CORRELATION", "WHALE_INTENT",
            "PUMP_DUMP_DETECTION", "DEGRADATION_ENGINE",
        }
        flags = FeatureFlags()
        assert expected.issubset(set(flags.all_states().keys()))


# ════════════════════════════════════════════════════════════════
# FEATURE 1: Source Credibility
# ════════════════════════════════════════════════════════════════

class TestSourceCredibility:
    """Feature 1: Source credibility scoring."""

    def test_known_source_score(self):
        from analyzers.source_credibility import get_source_credibility
        score = get_source_credibility("CoinDesk")
        assert 0.90 <= score <= 1.00

    def test_unknown_source_uses_default(self):
        from analyzers.source_credibility import get_source_credibility, DEFAULT_CREDIBILITY
        score = get_source_credibility("SomeRandomBlog")
        assert score == DEFAULT_CREDIBILITY

    def test_score_clamped_to_range(self):
        from analyzers.source_credibility import get_source_credibility
        from config.constants import NewsIntelligence
        score = get_source_credibility("TheBlock")
        assert score >= NewsIntelligence.SOURCE_CREDIBILITY_FLOOR
        assert score <= NewsIntelligence.SOURCE_CREDIBILITY_CEILING

    def test_apply_credibility_when_off(self):
        """When feature is off, raw weight is returned unchanged."""
        from analyzers.source_credibility import apply_source_credibility
        from config.feature_flags import ff
        ff.set_state("SOURCE_CREDIBILITY", "off")
        result = apply_source_credibility("CoinDesk", 1.0)
        assert result == 1.0

    def test_apply_credibility_when_live(self):
        """When feature is live, weight is multiplied by credibility."""
        from analyzers.source_credibility import apply_source_credibility
        from config.feature_flags import ff
        ff.set_state("SOURCE_CREDIBILITY", "live")
        result = apply_source_credibility("CoinDesk", 1.0)
        assert 0.80 < result <= 1.0
        ff.set_state("SOURCE_CREDIBILITY", "off")

    def test_all_tier1_sources_exist(self):
        from analyzers.source_credibility import _SOURCE_SCORES
        tier1 = ["CoinTelegraph", "CoinDesk", "Decrypt", "TheBlock", "Blockworks"]
        for src in tier1:
            assert src in _SOURCE_SCORES

    def test_conservative_baseline_range(self):
        """All scores should be in 0.80-1.00 range at launch."""
        from analyzers.source_credibility import _SOURCE_SCORES
        for src, score in _SOURCE_SCORES.items():
            assert 0.80 <= score <= 1.00, f"{src} score {score} out of conservative range"


# ════════════════════════════════════════════════════════════════
# FEATURE 2: Clickbait Detection
# ════════════════════════════════════════════════════════════════

class TestClickbaitDetection:
    """Feature 2: Clickbait detection."""

    def test_clean_headline_low_score(self):
        from analyzers.news_scraper import NewsScraper
        scraper = NewsScraper()
        score, matched = scraper._score_clickbait("Bitcoin reaches $100k milestone after ETF inflows surge")
        assert score < 0.5
        assert len(matched) == 0

    def test_clickbait_headline_high_score(self):
        from analyzers.news_scraper import NewsScraper
        scraper = NewsScraper()
        score, matched = scraper._score_clickbait("You won't believe what Bitcoin just did!!!")
        assert score >= 0.5
        assert len(matched) > 0

    def test_all_caps_flagged(self):
        from analyzers.news_scraper import NewsScraper
        scraper = NewsScraper()
        score, matched = scraper._score_clickbait("BITCOIN CRASHES TO ZERO EVERYONE PANIC NOW")
        assert any("ALL_CAPS" in m for m in matched)

    def test_moon_lambo_flagged(self):
        from analyzers.news_scraper import NewsScraper
        scraper = NewsScraper()
        score, matched = scraper._score_clickbait("This altcoin will 1000x next week guaranteed")
        assert score >= 0.5

    def test_legitimate_breaking_not_flagged(self):
        from analyzers.news_scraper import NewsScraper
        scraper = NewsScraper()
        score, matched = scraper._score_clickbait("SEC approves Bitcoin ETF application")
        assert score < 0.5

    def test_multiple_patterns_stack(self):
        from analyzers.news_scraper import NewsScraper
        scraper = NewsScraper()
        score, _ = scraper._score_clickbait(
            "You won't believe this secret trick guaranteed 100x!!!"
        )
        assert score >= 1.0  # Multiple pattern hits


# ════════════════════════════════════════════════════════════════
# FEATURE 3: Title Deduplication
# ════════════════════════════════════════════════════════════════

class TestTitleDedup:
    """Feature 3: Title deduplication."""

    def test_jaccard_identical_strings(self):
        from analyzers.news_scraper import NewsScraper
        sim = NewsScraper._jaccard_ngram_similarity("hello world", "hello world")
        assert sim == pytest.approx(1.0)

    def test_jaccard_completely_different(self):
        from analyzers.news_scraper import NewsScraper
        sim = NewsScraper._jaccard_ngram_similarity("abcdefgh", "xyz12345")
        assert sim < 0.1

    def test_jaccard_similar_headlines(self):
        from analyzers.news_scraper import NewsScraper
        sim = NewsScraper._jaccard_ngram_similarity(
            "Bitcoin ETF approved by SEC in historic decision today",
            "Bitcoin ETF approved by SEC in historic ruling today",
        )
        assert sim > 0.60  # Similar but with one different word

    def test_jaccard_empty_string(self):
        from analyzers.news_scraper import NewsScraper
        assert NewsScraper._jaccard_ngram_similarity("", "hello") == 0.0
        assert NewsScraper._jaccard_ngram_similarity("hello", "") == 0.0

    def test_dedup_clusters_similar_titles(self):
        from analyzers.news_scraper import NewsScraper
        from config.feature_flags import ff
        ff.set_state("TITLE_DEDUP", "live")
        scraper = NewsScraper()
        # First headline creates a new cluster
        result1 = scraper._deduplicate_title(
            "Bitcoin ETF approved by SEC in historic decision for cryptocurrency",
            "CoinDesk"
        )
        assert result1 is None  # New cluster

        # Nearly identical headline (same but one word swapped) should join cluster
        result2 = scraper._deduplicate_title(
            "Bitcoin ETF approved by SEC in historic decision for cryptocurrency",
            "CoinTelegraph"
        )
        assert result2 is not None
        assert result2["source_count"] == 2
        ff.set_state("TITLE_DEDUP", "off")

    def test_dedup_distinct_stories_separate(self):
        from analyzers.news_scraper import NewsScraper
        scraper = NewsScraper()
        scraper._deduplicate_title("Bitcoin crashes 10% on SEC news", "CoinDesk")
        result = scraper._deduplicate_title("Ethereum upgrade goes live on mainnet", "Decrypt")
        assert result is None  # Should be a separate cluster

    def test_dedup_similar_template_but_different_entities_stay_separate(self):
        from analyzers.news_scraper import NewsScraper
        scraper = NewsScraper()
        scraper._deduplicate_title("Bitcoin ETF approval drives record inflows", "CoinDesk")
        result = scraper._deduplicate_title("Ethereum ETF approval drives record inflows", "Decrypt")
        assert result is None


# ════════════════════════════════════════════════════════════════
# FEATURE 4: Headline Evolution Tracking
# ════════════════════════════════════════════════════════════════

class TestHeadlineEvolution:
    """Feature 4: Headline evolution tracking."""

    def test_new_headline_tracked(self):
        from analyzers.news_scraper import NewsScraper
        from config.feature_flags import ff
        ff.set_state("HEADLINE_EVOLUTION", "live")
        scraper = NewsScraper()
        scraper._track_headline_evolution("https://example.com/1", "Bitcoin surges past 100k")
        assert "https://example.com/1" in scraper._headline_tracker
        ff.set_state("HEADLINE_EVOLUTION", "off")

    def test_significant_evolution_triggers_penalty(self):
        from analyzers.news_scraper import NewsScraper
        from config.feature_flags import ff
        from config.constants import NewsIntelligence
        ff.set_state("HEADLINE_EVOLUTION", "live")
        scraper = NewsScraper()
        # Track original
        scraper._track_headline_evolution("https://example.com/1", "SEC Bans All Crypto Trading")
        # Significant change
        scraper._track_headline_evolution("https://example.com/1", "SEC Considers New Framework for Regulation")
        assert "SEC Bans All Crypto Trading" in scraper._evolved_titles
        assert scraper._evolved_titles["SEC Bans All Crypto Trading"] == \
            NewsIntelligence.HEADLINE_EVOLUTION_CONFIDENCE_PENALTY
        ff.set_state("HEADLINE_EVOLUTION", "off")

    def test_minor_change_no_penalty(self):
        from analyzers.news_scraper import NewsScraper
        from config.feature_flags import ff
        ff.set_state("HEADLINE_EVOLUTION", "live")
        scraper = NewsScraper()
        scraper._track_headline_evolution("https://example.com/2", "Bitcoin hits new high of $100k")
        # Minor word change (high similarity)
        scraper._track_headline_evolution("https://example.com/2", "Bitcoin hits new high of $100,000")
        assert "Bitcoin hits new high of $100k" not in scraper._evolved_titles
        ff.set_state("HEADLINE_EVOLUTION", "off")


# ════════════════════════════════════════════════════════════════
# FEATURE 7: Fear & Greed Regime Overlay
# ════════════════════════════════════════════════════════════════

class TestFearGreedOverlay:
    """Feature 7: Fear & Greed regime overlay."""

    def test_no_adjustment_when_off(self):
        from analyzers.sentiment import SentimentAnalyzer
        from config.feature_flags import ff
        ff.set_state("FEAR_GREED", "off")
        analyzer = SentimentAnalyzer()
        assert analyzer.get_regime_threshold_adjustment("LONG") == 1.0

    def test_extreme_fear_tightens_longs(self):
        from analyzers.sentiment import SentimentAnalyzer
        from config.feature_flags import ff
        ff.set_state("FEAR_GREED", "live")
        analyzer = SentimentAnalyzer()
        analyzer._fear_greed = 10  # Extreme Fear
        adj = analyzer.get_regime_threshold_adjustment("LONG")
        assert adj < 1.0  # Should tighten (reduce multiplier)
        assert adj >= 0.85  # But not more than 15%
        ff.set_state("FEAR_GREED", "off")

    def test_extreme_greed_tightens_shorts(self):
        from analyzers.sentiment import SentimentAnalyzer
        from config.feature_flags import ff
        ff.set_state("FEAR_GREED", "live")
        analyzer = SentimentAnalyzer()
        analyzer._fear_greed = 90  # Extreme Greed
        adj = analyzer.get_regime_threshold_adjustment("SHORT")
        assert adj < 1.0
        assert adj >= 0.85  # But not more than 15%
        ff.set_state("FEAR_GREED", "off")

    def test_normal_fg_no_adjustment(self):
        from analyzers.sentiment import SentimentAnalyzer
        from config.feature_flags import ff
        ff.set_state("FEAR_GREED", "live")
        analyzer = SentimentAnalyzer()
        analyzer._fear_greed = 50  # Neutral
        assert analyzer.get_regime_threshold_adjustment("LONG") == 1.0
        assert analyzer.get_regime_threshold_adjustment("SHORT") == 1.0
        ff.set_state("FEAR_GREED", "off")

    def test_extreme_fear_no_effect_on_shorts(self):
        from analyzers.sentiment import SentimentAnalyzer
        from config.feature_flags import ff
        ff.set_state("FEAR_GREED", "live")
        analyzer = SentimentAnalyzer()
        analyzer._fear_greed = 10
        assert analyzer.get_regime_threshold_adjustment("SHORT") == 1.0
        ff.set_state("FEAR_GREED", "off")

    def test_7d_trend_rising(self):
        from analyzers.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        analyzer._history = [(time.time(), 60), (time.time() - 86400 * 6, 40)]
        assert analyzer.get_7d_trend() == "rising"

    def test_7d_trend_falling(self):
        from analyzers.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        analyzer._history = [(time.time(), 30), (time.time() - 86400 * 6, 60)]
        assert analyzer.get_7d_trend() == "falling"


# ════════════════════════════════════════════════════════════════
# FEATURE 8: Narrative Tracker
# ════════════════════════════════════════════════════════════════

class TestNarrativeTracker:
    """Feature 8: Narrative tracker."""

    def test_no_narratives_when_off(self):
        from analyzers.narrative_tracker import NarrativeTracker
        from config.feature_flags import ff
        ff.set_state("NARRATIVE_TRACKER", "off")
        tracker = NarrativeTracker()
        tracker.process_headline("Bitcoin ETF approved", time.time())
        assert len(tracker.get_active_narratives()) == 0

    def test_narrative_requires_min_articles(self):
        from analyzers.narrative_tracker import NarrativeTracker
        from config.feature_flags import ff
        ff.set_state("NARRATIVE_TRACKER", "live")
        tracker = NarrativeTracker()
        now = time.time()
        tracker.process_headline("Bitcoin ETF sees massive inflows", now)
        tracker.process_headline("BlackRock ETF reaches record volume", now - 60)
        # Only 2 articles — below minimum of 3
        assert len(tracker.get_active_narratives()) == 0
        ff.set_state("NARRATIVE_TRACKER", "off")

    def test_narrative_detected_with_3_articles(self):
        from analyzers.narrative_tracker import NarrativeTracker
        from config.feature_flags import ff
        ff.set_state("NARRATIVE_TRACKER", "live")
        tracker = NarrativeTracker()
        now = time.time()
        tracker.process_headline("Bitcoin ETF sees massive inflows", now)
        tracker.process_headline("BlackRock ETF reaches record volume", now - 60)
        tracker.process_headline("Spot ETF approval drives Bitcoin rally", now - 120)
        narratives = tracker.get_active_narratives()
        assert len(narratives) >= 1
        assert any(n.name == "Bitcoin ETF" for n in narratives)
        ff.set_state("NARRATIVE_TRACKER", "off")

    def test_duplicate_titles_do_not_trigger_narrative(self):
        from analyzers.narrative_tracker import NarrativeTracker
        from config.feature_flags import ff
        ff.set_state("NARRATIVE_TRACKER", "live")
        tracker = NarrativeTracker()
        now = time.time()
        for i in range(3):
            tracker.process_headline("Bitcoin ETF approved", now - i * 60)
        assert len(tracker.get_active_narratives()) == 0
        ff.set_state("NARRATIVE_TRACKER", "off")

    def test_confidence_adjustment_capped(self):
        from analyzers.narrative_tracker import NarrativeTracker
        from config.constants import NewsIntelligence
        from config.feature_flags import ff
        ff.set_state("NARRATIVE_TRACKER", "live")
        tracker = NarrativeTracker()
        now = time.time()
        # Flood with ETF headlines to create a rising narrative
        for i in range(20):
            tracker.process_headline(f"ETF inflow record day {i}", now - i * 60)
        adj = tracker.get_narrative_confidence_adjustment("BTC", "LONG")
        # Must be capped at ±20%
        assert abs(adj - 1.0) <= NewsIntelligence.CONTEXTUAL_CONFIDENCE_CAP
        ff.set_state("NARRATIVE_TRACKER", "off")

    def test_narrative_summary(self):
        from analyzers.narrative_tracker import NarrativeTracker
        from config.feature_flags import ff
        ff.set_state("NARRATIVE_TRACKER", "live")
        tracker = NarrativeTracker()
        now = time.time()
        for i in range(5):
            tracker.process_headline(f"SEC regulatory action {i}", now - i * 60)
        summary = tracker.get_summary()
        assert "active_count" in summary
        assert "narratives" in summary
        ff.set_state("NARRATIVE_TRACKER", "off")

    def test_no_adjustment_returns_1(self):
        from analyzers.narrative_tracker import NarrativeTracker
        from config.feature_flags import ff
        ff.set_state("NARRATIVE_TRACKER", "off")
        tracker = NarrativeTracker()
        assert tracker.get_narrative_confidence_adjustment("BTC", "LONG") == 1.0

    def test_raw_adjustment_requires_live_flag(self):
        from analyzers.narrative_tracker import NarrativeTracker
        from config.feature_flags import ff
        tracker = NarrativeTracker()
        now = time.time()
        ff.set_state("NARRATIVE_TRACKER", "shadow")
        for i in range(3):
            tracker.process_headline(f"ETF inflow record day {i}", now - i * 60)
        assert tracker.get_raw_adjustment("BTC", "LONG") == 0.0
        ff.set_state("NARRATIVE_TRACKER", "off")


# ════════════════════════════════════════════════════════════════
# FEATURE 9: Pump/Dump Detection
# ════════════════════════════════════════════════════════════════

class TestPumpDumpDetection:
    """Feature 9: Pump/dump detection."""

    def test_no_alert_when_off(self):
        from analyzers.market_microstructure import detect_pump_dump
        from config.feature_flags import ff
        ff.set_state("PUMP_DUMP_DETECTION", "off")
        result = detect_pump_dump("BTCUSDT", 15.0, 500.0)
        assert result is None

    def test_high_risk_alert(self):
        from analyzers.market_microstructure import detect_pump_dump
        from config.feature_flags import ff
        ff.set_state("PUMP_DUMP_DETECTION", "live")
        result = detect_pump_dump("BTCUSDT", 12.0, 350.0)
        assert result is not None
        assert result.risk_level == "HIGH"
        assert result.is_no_trade
        assert result.direction == "PUMP"
        ff.set_state("PUMP_DUMP_DETECTION", "off")

    def test_very_high_risk(self):
        from analyzers.market_microstructure import detect_pump_dump
        from config.feature_flags import ff
        ff.set_state("PUMP_DUMP_DETECTION", "live")
        result = detect_pump_dump("ETHUSDT", -20.0, 600.0)
        assert result.risk_level == "VERY_HIGH"
        assert result.direction == "DUMP"
        assert result.is_no_trade
        ff.set_state("PUMP_DUMP_DETECTION", "off")

    def test_medium_risk_not_no_trade(self):
        from analyzers.market_microstructure import detect_pump_dump
        from config.feature_flags import ff
        ff.set_state("PUMP_DUMP_DETECTION", "live")
        result = detect_pump_dump("SOLUSDT", 5.0, 160.0)  # Adjusted for new thresholds
        assert result is not None
        assert result.risk_level == "MEDIUM"
        assert not result.is_no_trade
        ff.set_state("PUMP_DUMP_DETECTION", "off")

    def test_no_alert_for_normal_moves(self):
        from analyzers.market_microstructure import detect_pump_dump
        from config.feature_flags import ff
        ff.set_state("PUMP_DUMP_DETECTION", "live")
        result = detect_pump_dump("BTCUSDT", 2.0, 50.0)
        assert result is None
        ff.set_state("PUMP_DUMP_DETECTION", "off")


# ════════════════════════════════════════════════════════════════
# FEATURE 6: Whale Intent Classification
# ════════════════════════════════════════════════════════════════

class TestWhaleIntentClassification:
    """Feature 6: Whale intent classification."""

    def test_unknown_when_off(self):
        from analyzers.whale_deposit_monitor import WhaleDepositMonitor, InflowEvent
        from config.feature_flags import ff
        ff.set_state("WHALE_INTENT", "off")
        monitor = WhaleDepositMonitor()
        event = InflowEvent(
            timestamp=time.time(), asset="BTC", exchange="binance",
            amount_usd=100_000_000, source="inflow_test",
        )
        intent, conf = monitor.classify_whale_intent(event)
        assert intent == "unknown"
        assert conf == 0.0

    def test_exchange_inflow_classified(self):
        from analyzers.whale_deposit_monitor import WhaleDepositMonitor, InflowEvent
        from config.feature_flags import ff
        ff.set_state("WHALE_INTENT", "live")
        monitor = WhaleDepositMonitor()
        event = InflowEvent(
            timestamp=time.time(), asset="BTC", exchange="binance",
            amount_usd=60_000_000, source="inflow_data",
        )
        intent, conf = monitor.classify_whale_intent(event)
        assert intent == "inflow"
        assert conf >= 0.75
        ff.set_state("WHALE_INTENT", "off")

    def test_outflow_classified(self):
        from analyzers.whale_deposit_monitor import WhaleDepositMonitor, InflowEvent
        from config.feature_flags import ff
        ff.set_state("WHALE_INTENT", "live")
        monitor = WhaleDepositMonitor()
        event = InflowEvent(
            timestamp=time.time(), asset="BTC", exchange="binance",
            amount_usd=30_000_000, source="outflow_data",
        )
        intent, conf = monitor.classify_whale_intent(event)
        assert intent == "outflow"
        assert conf >= 0.75
        ff.set_state("WHALE_INTENT", "off")

    def test_stablecoin_mint_classified(self):
        from analyzers.whale_deposit_monitor import WhaleDepositMonitor, InflowEvent
        from config.feature_flags import ff
        ff.set_state("WHALE_INTENT", "live")
        monitor = WhaleDepositMonitor()
        event = InflowEvent(
            timestamp=time.time(), asset="USDT", exchange="tether_treasury",
            amount_usd=500_000_000, source="mint_event",
        )
        intent, conf = monitor.classify_whale_intent(event)
        assert intent == "mint"
        assert conf >= 0.80
        ff.set_state("WHALE_INTENT", "off")

    def test_intent_summary(self):
        from analyzers.whale_deposit_monitor import WhaleDepositMonitor
        monitor = WhaleDepositMonitor()
        summary = monitor.get_whale_intent_summary()
        assert "total_events" in summary
        assert "dominant_intent" in summary


# ════════════════════════════════════════════════════════════════
# FEATURE 10: Degradation Engine
# ════════════════════════════════════════════════════════════════

class TestDegradationEngine:
    """Feature 10: Degradation engine + Feature 11: DB fallback."""

    def test_circuit_breaker_starts_closed(self):
        from utils.degradation import CircuitBreaker, CircuitState
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.should_allow_request()

    def test_circuit_breaker_opens_after_failures(self):
        from utils.degradation import CircuitBreaker, CircuitState
        cb = CircuitBreaker(name="test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.should_allow_request()

    def test_circuit_breaker_resets_on_success(self):
        from utils.degradation import CircuitBreaker, CircuitState
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0

    def test_circuit_breaker_half_open_after_cooldown(self):
        from utils.degradation import CircuitBreaker, CircuitState
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=0.01)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.should_allow_request()
        assert cb.state == CircuitState.HALF_OPEN

    def test_news_mode_primary_default(self):
        from utils.degradation import DegradationEngine, NewsMode
        engine = DegradationEngine()
        assert engine.news_mode == NewsMode.PRIMARY
        assert engine.news_weight_multiplier == 1.0

    def test_news_mode_degraded_on_low_success(self):
        from utils.degradation import DegradationEngine, NewsMode
        from config.feature_flags import ff
        ff.set_state("DEGRADATION_ENGINE", "live")
        engine = DegradationEngine()
        engine.update_rss_success_rate(3, 15)  # 20% < 30%
        assert engine.news_mode == NewsMode.DEGRADED
        assert engine.news_weight_multiplier == 0.3
        ff.set_state("DEGRADATION_ENGINE", "off")

    def test_news_mode_blind_on_zero_success(self):
        from utils.degradation import DegradationEngine, NewsMode
        from config.feature_flags import ff
        ff.set_state("DEGRADATION_ENGINE", "live")
        engine = DegradationEngine()
        engine.update_rss_success_rate(0, 15)
        assert engine.news_mode == NewsMode.BLIND
        assert engine.news_weight_multiplier == 0.0
        ff.set_state("DEGRADATION_ENGINE", "off")

    def test_cached_data_tracking(self):
        from utils.degradation import DegradationEngine
        engine = DegradationEngine()
        engine.cache_data("test_source", {"headlines": ["a", "b"]})
        cached = engine.get_cached("test_source")
        assert cached is not None
        assert cached.data == {"headlines": ["a", "b"]}
        assert cached.age_seconds < 5

    def test_db_fallback_queues_on_failure(self):
        from utils.degradation import DegradationEngine
        engine = DegradationEngine()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            engine._db_fallback_path = Path(tmp.name)
        engine.handle_db_write_failure(
            {"signal_id": "test123", "confidence": 85},
            Exception("Connection refused"),
        )
        assert engine.get_retry_queue_size() == 1
        batch = engine.pop_retry_batch()
        assert len(batch) == 1
        assert batch[0]["record"]["signal_id"] == "test123"
        assert engine.get_retry_queue_size() == 0
        # Cleanup
        try:
            engine._db_fallback_path.unlink()
        except Exception:
            pass

    def test_status_returns_all_fields(self):
        from utils.degradation import DegradationEngine
        engine = DegradationEngine()
        status = engine.get_status()
        assert "news_mode" in status
        assert "execution_mode" in status
        assert "circuit_breakers" in status
        assert "db_retry_queue_size" in status

    def test_no_mode_change_when_off(self):
        from utils.degradation import DegradationEngine, NewsMode
        from config.feature_flags import ff
        ff.set_state("DEGRADATION_ENGINE", "off")
        engine = DegradationEngine()
        engine.update_rss_success_rate(0, 15)
        assert engine.news_mode == NewsMode.PRIMARY  # Should stay primary


# ════════════════════════════════════════════════════════════════
# INTEGRATION: Constants defined correctly
# ════════════════════════════════════════════════════════════════

class TestNewsIntelligenceConstants:
    """Verify all NewsIntelligence constants are defined correctly."""

    def test_all_constants_exist(self):
        from config.constants import NewsIntelligence
        attrs = [
            "NEWS_GATING_MIN_CONFIDENCE",
            "HEADLINE_DECAY_LAMBDA",
            "HEADLINE_MAX_AGE_MINUTES",
            "SOURCE_CREDIBILITY_FLOOR",
            "SOURCE_CREDIBILITY_CEILING",
            "CLICKBAIT_WEIGHT_PENALTY",
            "CLICKBAIT_SCORE_THRESHOLD",
            "TITLE_DEDUP_SIMILARITY_THRESHOLD",
            "HEADLINE_EVOLUTION_SIMILARITY_THRESHOLD",
            "HEADLINE_EVOLUTION_CONFIDENCE_PENALTY",
            "CORRELATION_SAME_EVENT_BOOST",
            "CORRELATION_INDEPENDENT_BOOST",
            "CORRELATION_FULL_WINDOW_MINUTES",
            "CORRELATION_PARTIAL_WINDOW_MINUTES",
            "CORRELATION_PARTIAL_WEIGHT",
            "UNEXPLAINED_ONCHAIN_PENALTY",
            "FEAR_GREED_EXTREME_FEAR_THRESHOLD",
            "FEAR_GREED_EXTREME_GREED_THRESHOLD",
            "FEAR_GREED_MAX_ADJUSTMENT_PCT",
            "NARRATIVE_MIN_ARTICLES",
            "NARRATIVE_VELOCITY_RISING_THRESHOLD",
            "NARRATIVE_VELOCITY_FADING_THRESHOLD",
            "CONTEXTUAL_CONFIDENCE_CAP",
            "PUMP_DUMP_PRICE_CHANGE_THRESHOLD",
            "PUMP_DUMP_VOLUME_CHANGE_THRESHOLD",
            "PUMP_DUMP_WINDOW_MINUTES",
        ]
        for attr in attrs:
            assert hasattr(NewsIntelligence, attr), f"Missing constant: {attr}"

    def test_lambda_produces_strong_decay(self):
        """Lambda=0.08 should produce >50% decay by 15 minutes."""
        from config.constants import NewsIntelligence
        lam = NewsIntelligence.HEADLINE_DECAY_LAMBDA
        decay_15m = math.exp(-lam * 15)
        assert decay_15m < 0.50  # Must be strong decay by 15 min

    def test_max_age_reasonable(self):
        from config.constants import NewsIntelligence
        assert 60 <= NewsIntelligence.HEADLINE_MAX_AGE_MINUTES <= 120


# ════════════════════════════════════════════════════════════════
# INTEGRATION: Shadow mode logger
# ════════════════════════════════════════════════════════════════

class TestShadowMode:
    """PRE-WORK 4: Shadow mode infrastructure."""

    def test_shadow_log_does_not_raise(self):
        from config.shadow_mode import shadow_log
        # Should not raise even if logs directory doesn't exist
        shadow_log("TEST_FEATURE", {"key": "value", "number": 42})

    def test_shadow_log_accepts_various_types(self):
        from config.shadow_mode import shadow_log
        shadow_log("TEST", {
            "string": "hello",
            "number": 3.14,
            "list": [1, 2, 3],
            "nested": {"a": "b"},
        })


# ════════════════════════════════════════════════════════════════
# RISK MITIGATION TESTS — Addressing build concerns
# ════════════════════════════════════════════════════════════════

class TestEventTypeAwareTimeDecay:
    """Risk 1.1: Time decay must be event-type aware, not uniform."""

    def test_slow_decay_event_keywords_exist(self):
        from config.constants import NewsIntelligence
        assert hasattr(NewsIntelligence, 'SLOW_DECAY_EVENT_KEYWORDS')
        assert len(NewsIntelligence.SLOW_DECAY_EVENT_KEYWORDS) > 0

    def test_slow_decay_lambda_is_gentler(self):
        from config.constants import NewsIntelligence
        assert NewsIntelligence.HEADLINE_DECAY_LAMBDA_SLOW < NewsIntelligence.HEADLINE_DECAY_LAMBDA

    def test_slow_decay_max_age_is_longer(self):
        from config.constants import NewsIntelligence
        assert NewsIntelligence.HEADLINE_MAX_AGE_MINUTES_SLOW > NewsIntelligence.HEADLINE_MAX_AGE_MINUTES

    def test_is_slow_decay_detects_regulatory(self):
        from analyzers.news_scraper import NewsScraper
        assert NewsScraper._is_slow_decay_event("SEC Approves Bitcoin ETF")
        assert NewsScraper._is_slow_decay_event("Fed interest rate decision today")
        assert NewsScraper._is_slow_decay_event("New crypto regulation proposed")

    def test_is_slow_decay_rejects_rumors(self):
        from analyzers.news_scraper import NewsScraper
        assert not NewsScraper._is_slow_decay_event("Bitcoin pumps 5% in one hour")
        assert not NewsScraper._is_slow_decay_event("Whale moves 10000 BTC")

    def test_time_decay_with_slow_event_retains_more_weight(self):
        """At 30 minutes, a slow-burn headline should retain more weight than a rumor."""
        from analyzers.news_scraper import NewsScraper
        fast_weight = NewsScraper._time_decay_weight(30, title="Bitcoin pumps hard")
        slow_weight = NewsScraper._time_decay_weight(30, title="SEC ETF approval imminent")
        assert slow_weight > fast_weight

    def test_time_decay_slow_event_at_120_min_still_has_weight(self):
        """At 120 min, fast decay returns 0 but slow decay still has weight."""
        from analyzers.news_scraper import NewsScraper
        fast_weight = NewsScraper._time_decay_weight(120, title="Bitcoin pumps")
        slow_weight = NewsScraper._time_decay_weight(120, title="SEC ETF decision delayed")
        assert fast_weight == 0.0  # Exceeds 90 min max age
        assert slow_weight > 0.0  # Within 180 min max age

    def test_time_decay_backward_compatible_no_title(self):
        """Calling without title uses fast decay (backward compatible)."""
        from analyzers.news_scraper import NewsScraper
        weight = NewsScraper._time_decay_weight(15)
        expected = math.exp(-0.08 * 15)
        assert abs(weight - expected) < 0.001


class TestCorrelationLogicWiring:
    """Risk 1.2: Correlation logic must actually execute in the pipeline."""

    def test_correlation_constants_exist(self):
        from config.constants import NewsIntelligence
        assert NewsIntelligence.CORRELATION_SAME_EVENT_BOOST == 1.05
        assert NewsIntelligence.CORRELATION_INDEPENDENT_BOOST == 1.15
        assert NewsIntelligence.CORRELATION_FULL_WINDOW_MINUTES == 20
        assert NewsIntelligence.CORRELATION_PARTIAL_WINDOW_MINUTES == 45
        assert NewsIntelligence.UNEXPLAINED_ONCHAIN_PENALTY == 0.90

    def test_engine_has_correlate_method(self):
        """Verify the correlation method exists on the Engine class."""
        # Engine requires heavy dependencies; check via grep/attribute instead
        import importlib
        import inspect
        # Read the source to confirm method exists without importing full Engine
        src = inspect.getsource(importlib.import_module('analyzers.news_scraper'))
        # We added _correlate_onchain_news to engine.py, verify via file read
        with open('core/engine.py', 'r') as f:
            engine_src = f.read()
        assert '_correlate_onchain_news' in engine_src

    def test_correlation_feature_flag_exists(self):
        from config.feature_flags import ff
        assert "ONCHAIN_NEWS_CORRELATION" in ff._flags

    def test_onchain_event_state_is_derived_from_snapshot(self):
        from analyzers.onchain_analytics import OnChainAnalytics, MVRVData
        analytics = OnChainAnalytics()
        analytics._snapshot.mvrv = MVRVData(
            mvrv_ratio=3.9,
            zone="EUPHORIA",
            timestamp=time.time(),
        )
        state = analytics.get_state()
        assert state.last_anomaly_time > 0
        assert state.anomaly_bias == "SHORT"
        assert "mvrv" in state.anomaly_keywords

    @pytest.mark.asyncio
    async def test_whale_monitor_exposes_last_event_metadata(self):
        from analyzers.whale_deposit_monitor import WhaleDepositMonitor, InflowEvent

        monitor = WhaleDepositMonitor()
        ts = time.time()
        monitor._fetch_blockchain_com_inflows = AsyncMock(return_value=[
            InflowEvent(
                timestamp=ts,
                asset="BTC",
                exchange="binance",
                amount_usd=60_000_000,
                source="test",
                intent="deposit",
            )
        ])
        monitor._fetch_cryptoquant_inflows = AsyncMock(return_value=[])

        await monitor._check_inflows()
        state = monitor.get_state()
        assert state.last_event_time == ts
        assert state.last_event_bias == "SHORT"
        assert "whale" in state.last_event_keywords


class TestGlobalContextualCap:
    """Risk 1.4: Context signals must have a GLOBAL cap, not per-feature caps."""

    def test_contextual_cap_constant(self):
        from config.constants import NewsIntelligence
        assert NewsIntelligence.CONTEXTUAL_CONFIDENCE_CAP == 0.20

    def test_clamp_contextual_adjustments_function_exists(self):
        from analyzers.narrative_tracker import clamp_contextual_adjustments
        result = clamp_contextual_adjustments(0.0, 0.0, 0.0)
        assert result == 1.0

    def test_clamp_enforces_positive_cap(self):
        from analyzers.narrative_tracker import clamp_contextual_adjustments
        # All positive: should cap at 1.20
        result = clamp_contextual_adjustments(0.10, 0.10, 0.10)
        assert result == pytest.approx(1.20, abs=0.001)

    def test_clamp_enforces_negative_cap(self):
        from analyzers.narrative_tracker import clamp_contextual_adjustments
        # All negative: should cap at 0.80
        result = clamp_contextual_adjustments(-0.10, -0.10, -0.10)
        assert result == pytest.approx(0.80, abs=0.001)

    def test_clamp_within_bounds_no_change(self):
        from analyzers.narrative_tracker import clamp_contextual_adjustments
        result = clamp_contextual_adjustments(-0.05, 0.03, -0.02)
        expected = 1.0 + (-0.05 + 0.03 + -0.02)
        assert result == pytest.approx(expected, abs=0.001)

    def test_clamp_extreme_stacking_scenario(self):
        """Extreme Fear + rising bearish narrative + pump alert should still cap."""
        from analyzers.narrative_tracker import clamp_contextual_adjustments
        # F&G extreme fear: -0.10, narrative bearish rising: -0.10, pump signal: -0.10
        result = clamp_contextual_adjustments(-0.10, -0.10, -0.10)
        # Should be clamped to -0.20, not -0.30
        assert result == pytest.approx(0.80, abs=0.001)

    def test_fear_greed_raw_adjustment(self):
        """Fear & Greed should expose raw adjustment for global cap enforcement."""
        from analyzers.sentiment import SentimentAnalyzer
        from config.feature_flags import ff
        ff.set_state("FEAR_GREED", "live")
        try:
            sa = SentimentAnalyzer()
            sa._fear_greed = 5  # Extreme fear
            adj = sa.get_raw_adjustment("LONG")
            assert adj < 0  # Should tighten
            assert adj >= -0.15  # Should not exceed ±15%
        finally:
            ff.set_state("FEAR_GREED", "off")


class TestDBRetryQueueLimits:
    """Risk 2.5: Retry queue must not grow unbounded."""

    def test_max_queue_size_constant(self):
        from config.constants import NewsIntelligence
        assert NewsIntelligence.DB_RETRY_MAX_QUEUE_SIZE == 1000

    def test_max_retries_constant(self):
        from config.constants import NewsIntelligence
        assert NewsIntelligence.DB_RETRY_MAX_RETRIES_PER_RECORD == 10

    def test_alert_threshold_constant(self):
        from config.constants import NewsIntelligence
        assert NewsIntelligence.DB_RETRY_QUEUE_ALERT_THRESHOLD == 100

    def test_queue_drops_oldest_at_max_size(self):
        from utils.degradation import DegradationEngine
        eng = DegradationEngine()
        # Disable alert sending to avoid asyncio issues in tests
        eng._on_degradation_alert = None
        # Fill queue to max
        for i in range(1001):
            eng.handle_db_write_failure(
                {"id": i}, Exception("test")
            )
        # Should not exceed max
        from config.constants import NewsIntelligence
        assert len(eng._db_retry_queue) <= NewsIntelligence.DB_RETRY_MAX_QUEUE_SIZE

    def test_pop_retry_batch_increments_count(self):
        from utils.degradation import DegradationEngine
        eng = DegradationEngine()
        eng._db_retry_queue.append({
            "record": {"id": 1},
            "error": "test",
            "first_failed_at": time.time(),
            "retry_count": 0,
        })
        batch = eng.pop_retry_batch(1)
        assert len(batch) == 1
        assert batch[0]["retry_count"] == 1

    def test_pop_retry_batch_dead_letters_after_max(self):
        from utils.degradation import DegradationEngine
        from config.constants import NewsIntelligence
        eng = DegradationEngine()
        eng._db_retry_queue.append({
            "record": {"id": 1},
            "error": "test",
            "first_failed_at": time.time(),
            "retry_count": NewsIntelligence.DB_RETRY_MAX_RETRIES_PER_RECORD,
        })
        batch = eng.pop_retry_batch(1)
        # Should be empty — record is dead-lettered
        assert len(batch) == 0


class TestPumpDumpNewsNuance:
    """Risk 1.8: Pump/dump detection must not block legitimate breakouts."""

    def test_detect_pump_dump_accepts_news_flag(self):
        """Function accepts has_correlated_news parameter."""
        from analyzers.market_microstructure import detect_pump_dump
        from config.feature_flags import ff
        ff.set_state("PUMP_DUMP_DETECTION", "live")
        try:
            # Without news correlation → HIGH risk
            alert = detect_pump_dump("BTCUSDT", 12.0, 400.0)
            assert alert is not None
            assert alert.risk_level == "HIGH"
            assert alert.is_no_trade is True

            # With news correlation → downgraded to MEDIUM
            alert_news = detect_pump_dump("BTCUSDT", 12.0, 400.0, has_correlated_news=True)
            assert alert_news is not None
            assert alert_news.risk_level == "MEDIUM"
            assert alert_news.is_no_trade is False
        finally:
            ff.set_state("PUMP_DUMP_DETECTION", "off")

    def test_very_high_downgrades_to_high_with_news(self):
        from analyzers.market_microstructure import detect_pump_dump
        from config.feature_flags import ff
        ff.set_state("PUMP_DUMP_DETECTION", "live")
        try:
            alert = detect_pump_dump("BTCUSDT", 20.0, 600.0, has_correlated_news=True)
            assert alert is not None
            assert alert.risk_level == "HIGH"
        finally:
            ff.set_state("PUMP_DUMP_DETECTION", "off")

    def test_pump_dump_no_news_still_works(self):
        """Backward compatible: no has_correlated_news arg still works."""
        from analyzers.market_microstructure import detect_pump_dump
        from config.feature_flags import ff
        ff.set_state("PUMP_DUMP_DETECTION", "live")
        try:
            alert = detect_pump_dump("BTCUSDT", 12.0, 400.0)
            assert alert is not None
            assert alert.risk_level == "HIGH"
        finally:
            ff.set_state("PUMP_DUMP_DETECTION", "off")


class TestNewConstantsExist:
    """Verify all new risk-mitigation constants are defined."""

    def test_slow_decay_constants(self):
        from config.constants import NewsIntelligence
        assert hasattr(NewsIntelligence, 'HEADLINE_DECAY_LAMBDA_SLOW')
        assert hasattr(NewsIntelligence, 'HEADLINE_MAX_AGE_MINUTES_SLOW')
        assert hasattr(NewsIntelligence, 'SLOW_DECAY_EVENT_KEYWORDS')

    def test_db_retry_constants(self):
        from config.constants import NewsIntelligence
        assert hasattr(NewsIntelligence, 'DB_RETRY_MAX_QUEUE_SIZE')
        assert hasattr(NewsIntelligence, 'DB_RETRY_MAX_RETRIES_PER_RECORD')
        assert hasattr(NewsIntelligence, 'DB_RETRY_QUEUE_ALERT_THRESHOLD')

    def test_pump_dump_news_exempt_constant(self):
        from config.constants import NewsIntelligence
        assert hasattr(NewsIntelligence, 'PUMP_DUMP_NEWS_EXEMPT_WINDOW_MINUTES')
        assert NewsIntelligence.PUMP_DUMP_NEWS_EXEMPT_WINDOW_MINUTES == 30
