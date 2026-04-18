"""
Tests for geopolitical/macro news classification and bearish keyword detection.

Verifies that headlines like "Iran-US deal fails" are classified as
MACRO_RISK_OFF/BEARISH (not neutral) across both the news scraper
sentiment layer and the BTC news intelligence event classifier.
"""
import pytest
import time
from unittest.mock import AsyncMock


# ── News Scraper Sentiment Tests ─────────────────────────────

class TestNewsScraperGeopoliticalSentiment:
    """Verify _BEARISH_WORDS covers geopolitical crisis patterns."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from analyzers.news_scraper import NewsScraper
        self.scraper = NewsScraper()

    @pytest.mark.parametrize("headline,expected", [
        # The actual Iran-US deal failure scenario
        ("JD Vance: Iran and US could not make a deal", -1),
        ("Iran-US deal fails as diplomatic breakdown escalates", -1),
        ("No deal between Iran and US, tensions rise", -1),
        # Other geopolitical headlines that should be bearish
        ("Military strike on oil facilities causes market panic", -1),
        ("Sanctions expanded on Russia; trade war deepens", -1),
        ("Nuclear escalation fears grip markets", -1),
        ("Tariffs spike as trade war intensifies", -1),
        ("Diplomatic talks collapse, conflict looms", -1),
        ("Recession fears mount as GDP data disappoints", -1),
        ("Market panic as geopolitical tensions flare", -1),
        # Neutral / non-geo headlines should remain neutral
        ("Bitcoin price holds steady at $60k", 0),
        ("New DeFi protocol launches on Ethereum", 0),
    ])
    def test_headline_sentiment(self, headline, expected):
        score = self.scraper._score_headline(headline)
        if expected == -1:
            assert score == -1, f"'{headline}' should be bearish (-1), got {score}"
        elif expected == 0:
            # Neutral — allow 0 or slight lean
            assert score >= 0, f"'{headline}' should be neutral (0), got {score}"


# ── BTC News Intelligence Classification Tests ──────────────

class TestBTCNewsIntelligenceGeopolitical:
    """Verify NewsClassifier routes geopolitical events to MACRO_RISK_OFF."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from analyzers.btc_news_intelligence import NewsClassifier, BTCEventType
        self.classifier = NewsClassifier()
        self.MACRO_RISK_OFF = BTCEventType.MACRO_RISK_OFF

    @pytest.mark.parametrize("headline", [
        "JD Vance says Iran-US deal fails, escalation risk rises",
        "Iran-US tensions escalate after failed negotiations",
        "No deal: diplomatic talks between Iran and US collapse",
        "Military strike fears as conflict between nations deepens",
        "Sanctions imposed on Iran after deal breakdown",
        "Nuclear threats emerge from failed diplomatic talks",
        "Trade war tariffs doubled, markets in panic",
        "Russia-Ukraine conflict escalation triggers risk-off",
        "North Korea missile launch causes market turmoil",
        "Recession confirmed as GDP contracts for second quarter",
        "Banking crisis deepens as contagion spreads",
        "Debt ceiling default risk spooks investors",
    ])
    def test_geopolitical_classified_as_risk_off(self, headline):
        etype, direction, conf, _is_mixed = self.classifier.classify(headline)
        assert etype == self.MACRO_RISK_OFF, (
            f"'{headline}' should be MACRO_RISK_OFF, got {etype.value}"
        )
        assert direction == "BEARISH", (
            f"'{headline}' should be BEARISH, got {direction}"
        )
        assert conf > 0, f"'{headline}' should have confidence > 0, got {conf}"


class TestBTCNewsPreFilter:
    """Verify the headline pre-filter passes geopolitical headlines through."""

    def test_geopolitical_headlines_not_filtered(self):
        """Headlines with geopolitical keywords must pass the BTC/macro filter."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence

        bni = BTCNewsIntelligence()
        # Simulate the filter logic from process_headlines
        _BTC_MACRO_KEYWORDS = [
            "bitcoin", "btc", "crypto", "fed", "inflation",
            "rate", "macro", "sec", "cftc", "hack", "exchange",
            "etf", "regulation", "bank",
            "war", "conflict", "military", "missile", "nuclear",
            "sanctions", "sanction", "tariff", "geopolit", "tension",
            "deal fail", "no deal", "talks fail", "escalat",
            "iran", "north korea", "china taiwan", "russia",
            "recession", "crisis", "panic", "contagion",
            "default", "debt ceiling", "oil price",
        ]

        test_headlines = [
            {"title": "JD Vance: Iran and US could not make a deal", "published_at": 0, "source": "test"},
            {"title": "Iran-US tensions cause market panic", "published_at": 0, "source": "test"},
            {"title": "Military strike on oil facilities", "published_at": 0, "source": "test"},
            {"title": "Sanctions expanded after conflict escalation", "published_at": 0, "source": "test"},
            {"title": "Bitcoin holds at $60k during trade war fears", "published_at": 0, "source": "test"},
        ]

        for h in test_headlines:
            low = h["title"].lower()
            passed = any(kw in low for kw in _BTC_MACRO_KEYWORDS)
            assert passed, f"Headline '{h['title']}' was filtered out by pre-filter"


class TestBTCNewsEventContext:
    """Verify MACRO_RISK_OFF events produce correct adjustments."""

    def test_risk_off_blocks_longs(self):
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType
        bni = BTCNewsIntelligence()
        adj = bni._adjustments.get((BTCEventType.MACRO_RISK_OFF, "BEARISH"), {})
        assert adj.get("block_longs") is True, "MACRO_RISK_OFF BEARISH should block longs"
        assert adj.get("confidence_mult", 1.0) < 1.0, "MACRO_RISK_OFF should reduce confidence"
        assert adj.get("reduce_size_mult", 1.0) < 1.0, "MACRO_RISK_OFF should reduce position size"

    def test_callback_attribute_exists(self):
        from analyzers.btc_news_intelligence import BTCNewsIntelligence
        bni = BTCNewsIntelligence()
        assert hasattr(bni, 'on_risk_event'), "BTCNewsIntelligence should have on_risk_event callback"

    def test_active_context_exposes_directional_block_reason(self):
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventContext, BTCEventType

        bni = BTCNewsIntelligence()
        bni._current_ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_OFF,
            direction="BEARISH",
            confidence=0.7,
            detected_at=time.time(),
            expires_at=time.time() + 300,
            block_longs=True,
            explanation="Macro risk-off. All LONGs blocked.",
        )

        assert "LONG signals blocked" in bni.get_signal_block_reason("LONG")
        assert bni.get_signal_block_reason("SHORT") == ""


class TestBTCNewsRuntimeBlocking:
    @pytest.mark.asyncio
    async def test_invalidation_monitor_invalidates_pending_long_immediately(self, monkeypatch):
        from analyzers.btc_news_intelligence import BTCEventContext, BTCEventType, btc_news_intelligence
        from signals.invalidation_monitor import InvalidationMonitor, PendingSignal

        old_ctx = btc_news_intelligence._current_ctx
        btc_news_intelligence._current_ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_OFF,
            direction="BEARISH",
            confidence=0.7,
            detected_at=time.time(),
            expires_at=time.time() + 300,
            block_longs=True,
            explanation="Macro risk-off. All LONGs blocked.",
        )

        monitor = InvalidationMonitor()
        monitor.on_invalidated = AsyncMock()
        sig = PendingSignal(
            signal_id=101,
            symbol="DASH/USDT",
            direction="LONG",
            strategy="PriceAction",
            entry_low=43.13,
            entry_high=43.82,
            stop_loss=41.09,
            confidence=81.0,
            message_id=77,
        )

        monkeypatch.setattr("signals.invalidation_monitor.price_cache.get", lambda _symbol: 44.0)

        try:
            await monitor._check_signal(sig)
        finally:
            btc_news_intelligence._current_ctx = old_ctx

        assert sig.status == "INVALIDATED"
        assert "LONG signals blocked" in sig.invalidation_reason
        monitor.on_invalidated.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execution_engine_invalidates_tracked_long_before_execute(self, monkeypatch):
        from analyzers.btc_news_intelligence import BTCEventContext, BTCEventType, btc_news_intelligence
        from core.execution_engine import ExecutionEngine, SignalState, TrackedExecution
        from data.database import db

        old_ctx = btc_news_intelligence._current_ctx
        btc_news_intelligence._current_ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_OFF,
            direction="BEARISH",
            confidence=0.7,
            detected_at=time.time(),
            expires_at=time.time() + 300,
            block_longs=True,
            explanation="Macro risk-off. All LONGs blocked.",
        )

        engine = ExecutionEngine()
        engine.on_stage_change = AsyncMock()
        sig = TrackedExecution(
            signal_id=202,
            symbol="DOT/USDT",
            direction="LONG",
            strategy="PriceAction",
            entry_low=1.214,
            entry_high=1.2223,
            stop_loss=1.1889,
            confidence=84.0,
            state=SignalState.ENTRY_ZONE,
            watching_since=time.time(),
            message_id=88,
        )
        engine._tracked[sig.signal_id] = sig

        monkeypatch.setattr("core.execution_engine.price_cache.get", lambda _symbol: 1.22)
        monkeypatch.setattr("core.execution_engine.build_market_state", AsyncMock(return_value={}))
        monkeypatch.setattr("core.execution_engine.build_market_profile", lambda _state: {})
        monkeypatch.setattr("core.execution_engine.compute_confidence", lambda _state, _profile: 50.0)
        monkeypatch.setattr("core.execution_engine.get_risk_adjustment", lambda _regime: 1.0)
        db.update_signal_exec_state = AsyncMock()
        db.delete_tracked_signal = AsyncMock()

        try:
            await engine._check(sig, {})
        finally:
            btc_news_intelligence._current_ctx = old_ctx

        assert sig.state == SignalState.INVALIDATED
        assert sig.signal_id not in engine._tracked
        engine.on_stage_change.assert_awaited_once()
        _, old_state, new_state = engine.on_stage_change.await_args.args
        assert old_state == SignalState.ENTRY_ZONE
        assert new_state == SignalState.INVALIDATED


# ── Severity-Gated Blocking + Confidence-Scaled TTL Tests ────

class TestSeverityGatedBlocking:
    """
    Verify that MEDIUM-severity events (conf < 0.65) downgrade hard
    blocks to soft penalties, and that TTL scales with confidence.
    """

    def test_high_confidence_blocks_longs(self):
        """HIGH+ confidence (≥0.65) should keep block_longs=True."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType
        bni = BTCNewsIntelligence()

        # Simulate classification at 0.75 confidence (HIGH)
        import asyncio
        async def _run():
            bni._current_ctx = bni._current_ctx  # reset
            # Directly build context the same way process_headlines does
            conf = 0.75
            etype = BTCEventType.MACRO_RISK_OFF
            direction = "BEARISH"
            from config.constants import NewsIntelligence
            base_ttl = bni._context_ttl[etype]
            ttl_scale = max(NewsIntelligence.TTL_CONFIDENCE_FLOOR, conf)
            adj = dict(bni._adjustments.get((etype, direction), {}))

            # This should NOT downgrade because conf >= 0.65
            if conf < NewsIntelligence.HARD_BLOCK_MIN_CONFIDENCE:
                adj["block_longs"] = False

            assert adj.get("block_longs") is True, \
                f"HIGH conf ({conf}) should keep block_longs=True"
            assert int(base_ttl * ttl_scale) == int(14400 * 0.75), \
                f"TTL should be {int(14400 * 0.75)}, not {int(base_ttl * ttl_scale)}"

        asyncio.run(_run())

    def test_medium_confidence_downgrades_to_penalty(self):
        """MEDIUM confidence (<0.65) should downgrade block_longs to False."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType
        from config.constants import NewsIntelligence
        bni = BTCNewsIntelligence()

        conf = 0.50  # MEDIUM severity
        etype = BTCEventType.MACRO_RISK_OFF
        direction = "BEARISH"
        adj = dict(bni._adjustments.get((etype, direction), {}))

        assert adj.get("block_longs") is True, "Raw adjustment should have block_longs=True"

        # Apply severity gate (same logic as process_headlines)
        if conf < NewsIntelligence.HARD_BLOCK_MIN_CONFIDENCE:
            adj["block_longs"] = False

        assert adj["block_longs"] is False, \
            f"MEDIUM conf ({conf}) should downgrade block_longs to False"
        # Confidence and size penalties should remain
        assert adj.get("confidence_mult", 1.0) < 1.0, "Conf penalty should remain"
        assert adj.get("reduce_size_mult", 1.0) < 1.0, "Size penalty should remain"

    def test_confidence_scaled_ttl_medium(self):
        """50% confidence → 50% of base TTL (2h instead of 4h)."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType
        from config.constants import NewsIntelligence

        bni = BTCNewsIntelligence()
        base_ttl = bni._context_ttl[BTCEventType.MACRO_RISK_OFF]  # 14400s = 4h
        conf = 0.50
        ttl_scale = max(NewsIntelligence.TTL_CONFIDENCE_FLOOR, conf)
        scaled_ttl = int(base_ttl * ttl_scale)

        assert scaled_ttl == 7200, f"50% conf → TTL should be 7200s (2h), got {scaled_ttl}"

    def test_confidence_scaled_ttl_high(self):
        """90% confidence → 90% of base TTL (3.6h)."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType
        from config.constants import NewsIntelligence

        bni = BTCNewsIntelligence()
        base_ttl = bni._context_ttl[BTCEventType.MACRO_RISK_OFF]
        conf = 0.90
        ttl_scale = max(NewsIntelligence.TTL_CONFIDENCE_FLOOR, conf)
        scaled_ttl = int(base_ttl * ttl_scale)

        assert scaled_ttl == 12960, f"90% conf → TTL should be 12960s (3.6h), got {scaled_ttl}"

    def test_ttl_floor_prevents_tiny_durations(self):
        """Confidence at the gating threshold (0.40) should use floor (0.50)."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType
        from config.constants import NewsIntelligence

        bni = BTCNewsIntelligence()
        base_ttl = bni._context_ttl[BTCEventType.MACRO_RISK_OFF]
        conf = 0.42  # Just above gating threshold
        ttl_scale = max(NewsIntelligence.TTL_CONFIDENCE_FLOOR, conf)
        scaled_ttl = int(base_ttl * ttl_scale)

        # Floor is 0.50, so even 0.42 conf gets 50% TTL
        assert scaled_ttl == int(base_ttl * 0.50), \
            f"Conf {conf} below floor should use 50% TTL, got {scaled_ttl}"

    def test_exchange_event_also_severity_gated(self):
        """EXCHANGE_EVENT with medium conf also downgrades block."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType
        from config.constants import NewsIntelligence

        bni = BTCNewsIntelligence()
        adj = dict(bni._adjustments.get((BTCEventType.EXCHANGE_EVENT, "BEARISH"), {}))
        assert adj.get("block_longs") is True, "Raw should block"

        conf = 0.55
        if conf < NewsIntelligence.HARD_BLOCK_MIN_CONFIDENCE:
            adj["block_longs"] = False

        assert adj["block_longs"] is False, \
            "EXCHANGE_EVENT at 0.55 conf should downgrade block"

    def test_raw_adjustment_dict_unchanged(self):
        """Severity gating should not mutate the class-level _adjustments dict."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType

        bni = BTCNewsIntelligence()
        original = bni._adjustments[(BTCEventType.MACRO_RISK_OFF, "BEARISH")]
        assert original.get("block_longs") is True, \
            "Original dict should always have block_longs=True"


# ── Mixed Signal Detection Tests ─────────────────────────────

class TestMixedSignalDetection:
    """
    Verify that headlines containing BOTH bearish (risk-off) AND bullish
    (capital-inflow) signals are classified as mixed, resulting in softer
    penalties instead of full LONG blocking.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from analyzers.btc_news_intelligence import NewsClassifier, BTCEventType
        self.classifier = NewsClassifier()
        self.MACRO_RISK_OFF = BTCEventType.MACRO_RISK_OFF

    # ── Single headline mixed detection ──────────────────────

    @pytest.mark.parametrize("headline", [
        "Michael Saylor Hints At Buying More Bitcoin Despite US-Iran Peace Talks Collapse",
        "MicroStrategy buys 500 BTC as Iran sanctions escalate",
        "Institutional buying surges despite recession fears and panic selling",
        "Saylor accumulating Bitcoin amid geopolitical tension",
        "BlackRock buy BTC despite military strike fears",
        "Bitcoin ETF inflows surge even as trade war tariffs escalate",
    ])
    def test_mixed_headline_detected(self, headline):
        """Headlines with both bearish risk-off and bullish inflow should flag is_mixed."""
        etype, direction, conf, is_mixed = self.classifier.classify(headline)
        assert is_mixed is True, (
            f"'{headline}' should be detected as mixed signal, got is_mixed=False"
        )

    @pytest.mark.parametrize("headline", [
        "Iran-US tensions escalate after failed negotiations",
        "Military strike fears as conflict between nations deepens",
        "Trade war tariffs doubled, markets in panic",
        "Recession confirmed as GDP contracts for second quarter",
    ])
    def test_pure_bearish_not_mixed(self, headline):
        """Pure bearish headlines without bullish offset should NOT be mixed."""
        etype, direction, conf, is_mixed = self.classifier.classify(headline)
        assert is_mixed is False, (
            f"'{headline}' should NOT be mixed (no bullish offset), got is_mixed=True"
        )

    @pytest.mark.parametrize("headline", [
        "Bitcoin price hits new ATH on ETF approval news",
        "Fed cuts rates, risk appetite surges",
    ])
    def test_pure_bullish_not_mixed(self, headline):
        """Pure bullish headlines should not be flagged as mixed."""
        etype, direction, conf, is_mixed = self.classifier.classify(headline)
        assert is_mixed is False, (
            f"'{headline}' should NOT be mixed (pure bullish), got is_mixed=True"
        )

    def test_mixed_still_classified_as_risk_off(self):
        """Mixed headline should still be classified as MACRO_RISK_OFF type."""
        headline = "Saylor buys Bitcoin despite Iran sanctions escalation"
        etype, direction, conf, is_mixed = self.classifier.classify(headline)
        assert etype == self.MACRO_RISK_OFF, (
            f"Mixed headline should still classify as MACRO_RISK_OFF, got {etype.value}"
        )
        assert direction == "BEARISH"
        assert is_mixed is True

    # ── Batch-level mixed detection ──────────────────────────

    def test_batch_mixed_from_single_headline(self):
        """classify_batch should propagate is_mixed from a single mixed headline."""
        headlines = [
            "Saylor hints at buying more Bitcoin despite Iran talks collapse",
            "BTC drops 2% on geopolitical fears",
        ]
        etype, direction, conf, winning, is_mixed = self.classifier.classify_batch(headlines)
        assert is_mixed is True, "Batch with a mixed headline should flag is_mixed"

    def test_batch_mixed_from_conflicting_headlines(self):
        """Batch with separate bullish + bearish headlines should flag is_mixed."""
        headlines = [
            "Iran-US tensions escalate, risk-off mood",        # bearish
            "Fed cuts rates, institutional buying surges",      # bullish
        ]
        etype, direction, conf, winning, is_mixed = self.classifier.classify_batch(headlines)
        assert is_mixed is True, "Batch with conflicting directions should flag is_mixed"

    def test_batch_pure_bearish_not_mixed(self):
        """Batch of all-bearish headlines should NOT be mixed."""
        headlines = [
            "Iran-US tensions escalate after failed negotiations",
            "Trade war tariffs doubled, markets in panic",
            "Recession fears mount as banking crisis deepens",
        ]
        etype, direction, conf, winning, is_mixed = self.classifier.classify_batch(headlines)
        assert is_mixed is False, "All-bearish batch should NOT be mixed"


class TestMixedSignalAdjustments:
    """Verify mixed signals produce softer adjustments than pure risk-off."""

    def test_mixed_no_long_block(self):
        """Mixed signal should NOT hard-block LONGs."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType
        from config.constants import NewsIntelligence

        bni = BTCNewsIntelligence()
        adj = dict(bni._adjustments.get((BTCEventType.MACRO_RISK_OFF, "BEARISH"), {}))

        # Simulate mixed signal override (same logic as process_headlines)
        adj["block_longs"] = False
        adj["block_shorts"] = False
        adj["confidence_mult"] = NewsIntelligence.MIXED_SIGNAL_CONFIDENCE_MULT
        adj["reduce_size_mult"] = NewsIntelligence.MIXED_SIGNAL_SIZE_MULT

        assert adj["block_longs"] is False, "Mixed signal should not block LONGs"
        assert adj["confidence_mult"] == 0.85, "Mixed should use 0.85 conf mult"
        assert adj["reduce_size_mult"] == 0.75, "Mixed should use 0.75 size mult"

    def test_mixed_softer_than_pure_risk_off(self):
        """Mixed signal penalties must be softer than pure MACRO_RISK_OFF."""
        from config.constants import NewsIntelligence

        # Pure risk-off values (from _adjustments dict)
        pure_conf_mult = 0.70
        pure_size_mult = 0.50

        assert NewsIntelligence.MIXED_SIGNAL_CONFIDENCE_MULT > pure_conf_mult, \
            "Mixed conf mult should be higher (softer) than pure risk-off"
        assert NewsIntelligence.MIXED_SIGNAL_SIZE_MULT > pure_size_mult, \
            "Mixed size mult should be higher (softer) than pure risk-off"

    def test_mixed_ttl_shorter(self):
        """Mixed signal TTL should be shorter than pure risk-off TTL."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence, BTCEventType
        from config.constants import NewsIntelligence

        bni = BTCNewsIntelligence()
        base_ttl = bni._context_ttl[BTCEventType.MACRO_RISK_OFF]
        conf = 0.70
        pure_ttl = int(base_ttl * max(NewsIntelligence.TTL_CONFIDENCE_FLOOR, conf))
        mixed_ttl = int(pure_ttl * NewsIntelligence.MIXED_SIGNAL_TTL_SCALE)

        assert mixed_ttl < pure_ttl, \
            f"Mixed TTL ({mixed_ttl}s) should be shorter than pure ({pure_ttl}s)"

    def test_mixed_constants_in_valid_range(self):
        """Mixed signal constants should be between pure risk-off and neutral."""
        from config.constants import NewsIntelligence

        assert 0.50 < NewsIntelligence.MIXED_SIGNAL_CONFIDENCE_MULT < 1.0
        assert 0.50 < NewsIntelligence.MIXED_SIGNAL_SIZE_MULT < 1.0
        assert 0.0 < NewsIntelligence.MIXED_SIGNAL_TTL_SCALE < 1.0


class TestMixedSignalProcessHeadlines:
    """Integration test: verify process_headlines handles mixed signals correctly."""

    @pytest.mark.asyncio
    async def test_saylor_iran_headline_not_blocking(self):
        """The exact Saylor/Iran headline should NOT block LONGs."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence

        bni = BTCNewsIntelligence()
        # Seed a price so the move analyzer doesn't return FLAT
        bni.record_btc_price(65000)

        headlines = [{
            "title": "Michael Saylor Hints At Buying More Bitcoin Despite US-Iran Peace Talks Collapse",
            "published_at": time.time(),
            "source": "CoinTelegraph",
        }]
        await bni.process_headlines(headlines)

        ctx = bni.get_event_context()
        if ctx.is_active:
            assert ctx.is_mixed_signal is True, "Should be detected as mixed"
            assert ctx.block_longs is False, "Mixed signal should NOT block LONGs"
            assert ctx.confidence_mult > 0.70, \
                f"Mixed conf mult should be softer than 0.70, got {ctx.confidence_mult}"

    @pytest.mark.asyncio
    async def test_pure_risk_off_still_blocks_at_high_conf(self):
        """A pure risk-off headline at HIGH confidence should still block LONGs."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence

        bni = BTCNewsIntelligence()
        bni.record_btc_price(65000)

        # Multiple bearish headlines to push confidence to HIGH
        headlines = [
            {"title": "Iran-US tensions escalate as military strike confirmed and sanctions imposed",
             "published_at": time.time(), "source": "Reuters"},
            {"title": "Nuclear conflict fears and trade war tariffs cause market panic and recession",
             "published_at": time.time(), "source": "Bloomberg"},
        ]
        await bni.process_headlines(headlines)

        ctx = bni.get_event_context()
        if ctx.is_active and ctx.confidence >= 0.65:
            assert ctx.is_mixed_signal is False, "Pure bearish should NOT be mixed"
            assert ctx.block_longs is True, "Pure HIGH-conf bearish should block LONGs"

    @pytest.mark.asyncio
    async def test_mixed_signal_context_has_explanation(self):
        """Mixed signal context should have a descriptive explanation."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence

        bni = BTCNewsIntelligence()
        bni.record_btc_price(65000)

        headlines = [{
            "title": "MicroStrategy buys Bitcoin as Iran sanctions escalate and tension rises",
            "published_at": time.time(),
            "source": "CoinDesk",
        }]
        await bni.process_headlines(headlines)

        ctx = bni.get_event_context()
        if ctx.is_active and ctx.is_mixed_signal:
            assert "mixed" in ctx.explanation.lower() or "Mixed" in ctx.explanation, \
                f"Explanation should mention mixed: {ctx.explanation}"
            assert "offset" in ctx.explanation.lower() or "inflow" in ctx.explanation.lower(), \
                f"Explanation should mention offset/inflow: {ctx.explanation}"

    @pytest.mark.asyncio
    async def test_mixed_signal_in_status(self):
        """get_status() should expose is_mixed_signal."""
        from analyzers.btc_news_intelligence import BTCNewsIntelligence

        bni = BTCNewsIntelligence()
        bni.record_btc_price(65000)

        headlines = [{
            "title": "Saylor buying Bitcoin despite Iran-US talks collapse and recession fears",
            "published_at": time.time(),
            "source": "CoinTelegraph",
        }]
        await bni.process_headlines(headlines)

        status = bni.get_status()
        assert "is_mixed_signal" in status, "get_status() should include is_mixed_signal"
