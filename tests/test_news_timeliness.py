"""
Tests for news timeliness / staleness handling.

The bot should:
1. Gate out stale headlines (>45 min) from creating event contexts / alerts
2. Decay confidence for headlines that pass the gate but aren't brand-new
3. Carry headline_published_at in BTCEventContext so alerts show age
4. Display headline age in Telegram alerts and dashboard API
"""

import asyncio
import sys
import time
import types
from unittest.mock import AsyncMock, patch

import pytest

from analyzers.btc_news_intelligence import (
    BTCEventContext,
    BTCEventType,
    BTCNewsIntelligence,
)
from config.constants import NewsIntelligence


# ─── BTCEventContext tests ────────────────────────────────────────

class TestBTCEventContextHeadlineAge:
    """Test the headline_published_at field and headline_age_minutes property."""

    def test_headline_age_minutes_uses_published_at(self):
        """headline_age_minutes should use headline_published_at, not detected_at."""
        now = time.time()
        ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_OFF,
            detected_at=now,
            headline_published_at=now - 1800,  # 30 minutes ago
        )
        # headline_age should be ~30 min, not ~0
        assert 29 < ctx.headline_age_minutes < 31

    def test_headline_age_falls_back_to_detection_age(self):
        """When headline_published_at is 0, fall back to age_minutes."""
        now = time.time()
        ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_OFF,
            detected_at=now - 600,  # 10 min ago
            headline_published_at=0.0,
        )
        assert 9 < ctx.headline_age_minutes < 11

    def test_headline_age_zero_when_just_published(self):
        """Brand-new headline should have ~0 age."""
        now = time.time()
        ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_OFF,
            detected_at=now,
            headline_published_at=now,
        )
        assert ctx.headline_age_minutes < 1

    def test_headline_published_at_default_is_zero(self):
        """Default headline_published_at should be 0.0."""
        ctx = BTCEventContext()
        assert ctx.headline_published_at == 0.0


# ─── Staleness gate tests ────────────────────────────────────────

class TestStalenessGate:
    """Test that process_headlines gates out stale headlines."""

    def _make_headline(self, title, age_minutes):
        return {
            "title": title,
            "published_at": time.time() - (age_minutes * 60),
            "source": "test",
        }

    @pytest.mark.asyncio
    async def test_stale_headlines_are_gated(self):
        """Headlines older than STALE_HEADLINE_ALERT_GATE_MINUTES should not create context."""
        bni = BTCNewsIntelligence()
        gate = NewsIntelligence.STALE_HEADLINE_ALERT_GATE_MINUTES

        headlines = [
            self._make_headline("Fed raises rates, recession fears grow — bitcoin plummets", gate + 10),
        ]
        await bni.process_headlines(headlines)
        ctx = bni.get_event_context()
        assert not ctx.is_active, "Stale headline should not create active context"

    @pytest.mark.asyncio
    async def test_fresh_headlines_pass_gate(self):
        """Headlines within the gate should create context normally."""
        bni = BTCNewsIntelligence()

        # Use a headline that independently scores above NEWS_GATING_MIN_CONFIDENCE
        # after the news-batch-dilution fix (which removed the old 2× amplification
        # of single-headline batches).  Multiple rule keywords ensure this.
        headlines = [
            self._make_headline(
                "Recession looms as hawkish Fed rate hike triggers market panic",
                2,
            ),
        ]
        await bni.process_headlines(headlines)
        ctx = bni.get_event_context()
        assert ctx.is_active, "Fresh headline should create active context"

    @pytest.mark.asyncio
    async def test_mixed_batch_fresh_and_stale(self):
        """When batch has both fresh and stale, only fresh are processed."""
        bni = BTCNewsIntelligence()
        gate = NewsIntelligence.STALE_HEADLINE_ALERT_GATE_MINUTES

        headlines = [
            self._make_headline("Old: Fed raises rates, recession fears", gate + 30),
            self._make_headline(
                "Banking crisis deepens: bank run triggers contagion, capital flight from risk assets",
                5,
            ),
        ]
        await bni.process_headlines(headlines)
        ctx = bni.get_event_context()
        assert ctx.is_active, "Should process the fresh headline"

    @pytest.mark.asyncio
    async def test_headline_published_at_is_carried(self):
        """BTCEventContext should carry the winning headline's published_at."""
        bni = BTCNewsIntelligence()
        pub_time = time.time() - 300  # 5 min ago

        headlines = [{
            "title": "Fed raises rates, recession fears grow — bitcoin crashes sharply",
            "published_at": pub_time,
            "source": "CoinDesk",
        }]
        await bni.process_headlines(headlines)
        ctx = bni.get_event_context()
        if ctx.is_active:
            assert ctx.headline_published_at == pub_time

    @pytest.mark.asyncio
    async def test_gate_is_configurable(self):
        """Verify the gate uses the constant value."""
        assert NewsIntelligence.STALE_HEADLINE_ALERT_GATE_MINUTES == 45


class TestPublicationAnchoredExpiry:
    """Ensure expires_at is anchored to the headline's publication time,
    not the bot's detection time, so restarting the bot after a headline
    has partially aged does not re-arm a full-duration penalty window."""

    def _make_headline(self, title, age_minutes):
        return {
            "title": title,
            "published_at": time.time() - (age_minutes * 60),
            "source": "test",
        }

    @pytest.mark.asyncio
    async def test_expires_at_anchored_to_publication_time(self):
        """A 30-minute-old fresh headline must expire ~30 min earlier
        than a brand-new one of the same event type."""
        bni = BTCNewsIntelligence()
        headline_age_min = 30
        headlines = [self._make_headline(
            "Recession looms as hawkish Fed rate hike triggers market panic",
            headline_age_min,
        )]
        pub_time = headlines[0]["published_at"]

        await bni.process_headlines(headlines)
        ctx = bni.get_event_context()
        if not ctx.is_active:
            pytest.skip("classifier did not produce active context for this fixture")

        now = time.time()
        base_ttl = bni._context_ttl.get(ctx.event_type, 3600)
        # Remaining life must be strictly less than the full base TTL
        # (headline is already 30 min old so we've burnt ~30 min of TTL).
        remaining = ctx.expires_at - now
        assert remaining < base_ttl, (
            f"expires_at should account for headline age: "
            f"remaining={remaining:.0f}s, base_ttl={base_ttl}s"
        )
        # expires_at should equal publication_time + ttl_scaled, where
        # ttl_scaled is base_ttl * max(TTL_CONFIDENCE_FLOOR, conf). Since
        # we don't know the exact scaled TTL here, verify instead that:
        #   (a) expires_at > publication_time (always true for a live ctx)
        #   (b) expires_at - publication_time <= base_ttl (scaling only
        #       shrinks), and
        #   (c) (detected_at - publication_time) + remaining ≈ ttl_scaled,
        #       i.e. total life == ttl_scaled regardless of when we
        #       detected the article — this is the publication-anchor
        #       property we care about.
        ttl_scaled_from_pub = ctx.expires_at - pub_time
        assert ttl_scaled_from_pub > 0
        assert ttl_scaled_from_pub <= base_ttl + 5, (
            f"ttl anchored from publication ({ttl_scaled_from_pub:.0f}s) "
            f"must not exceed base_ttl ({base_ttl}s)"
        )
        # Sanity: total life from detection should be less than full
        # ttl_scaled by roughly the headline age at detection.
        life_from_detection = ctx.expires_at - ctx.detected_at
        age_at_detection = ctx.detected_at - pub_time
        assert abs((life_from_detection + age_at_detection) - ttl_scaled_from_pub) < 5

    @pytest.mark.asyncio
    async def test_article_past_ttl_is_rejected(self):
        """An article older than its effective TTL must not create an
        active context even when it passes the freshness gate. To keep
        this test hermetic we patch the TTL down so we stay below the
        STALE_HEADLINE_ALERT_GATE threshold."""
        bni = BTCNewsIntelligence()
        # 40-minute-old headline (below 45-min freshness gate)
        headlines = [self._make_headline(
            "Recession looms as hawkish Fed rate hike triggers market panic",
            40,
        )]
        # Shrink TTL for this event type so the article is past-TTL.
        for k in list(bni._context_ttl.keys()):
            bni._context_ttl[k] = 600  # 10 min

        await bni.process_headlines(headlines)
        ctx = bni.get_event_context()
        assert not ctx.is_active, (
            "Article older than effective TTL must not activate context "
            "— otherwise a bot restart re-arms a full fresh penalty window."
        )


@pytest.mark.asyncio
async def test_news_scraper_stop_cancels_tracked_bni_tasks():
    """Tracked background news-intelligence tasks should be cancelled on stop."""
    from analyzers.news_scraper import NewsScraper

    scraper = NewsScraper()
    sleeper = asyncio.create_task(asyncio.sleep(60))
    scraper._bni_tasks.add(sleeper)

    await scraper.stop()

    assert sleeper.cancelled()
    assert scraper._bni_tasks == set()


@pytest.mark.asyncio
async def test_news_scraper_fetch_news_tracks_real_bni_dispatch_path():
    """Fresh headlines should create and track the background BNI task via
    the real _fetch_news dispatch path."""
    from analyzers.news_scraper import NewsItem, NewsScraper
    import analyzers.news_scraper as news_scraper_mod

    class _Resp:
        status = 200

        async def text(self):
            return "<rss />"

    class _RespCtx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _Session:
        def get(self, _url):
            return _RespCtx()

    blocker = asyncio.Event()
    seen_batches = []

    async def _process_headlines(batch):
        seen_batches.append(batch)
        await blocker.wait()

    fake_bni_mod = types.SimpleNamespace(
        btc_news_intelligence=types.SimpleNamespace(process_headlines=_process_headlines)
    )
    fake_narrative_mod = types.SimpleNamespace(
        narrative_tracker=types.SimpleNamespace(process_headline=lambda *_args, **_kwargs: None)
    )

    scraper = NewsScraper()
    scraper._news_ttl = 0.0

    with patch.object(news_scraper_mod, "RSS_FEEDS", [("TestFeed", "https://example.test/rss")]), \
         patch.object(scraper, "_get_session", AsyncMock(return_value=_Session())), \
         patch.object(scraper, "_parse_rss", return_value=[
             NewsItem(
                 title="Fresh headline",
                 source="TestFeed",
                 url="https://example.test/story",
                 published_at=time.time(),
             )
         ]), \
         patch.object(scraper, "_fetch_free_crypto_news", AsyncMock(return_value=[])), \
         patch.dict(sys.modules, {
             "analyzers.btc_news_intelligence": fake_bni_mod,
             "analyzers.narrative_tracker": fake_narrative_mod,
         }):
        await scraper._fetch_news()
        await asyncio.sleep(0)
        assert len(scraper._bni_tasks) == 1
        assert seen_batches and seen_batches[0][0]["title"] == "Fresh headline"
        await scraper.stop()

    assert scraper._bni_tasks == set()


# ─── Confidence decay tests ──────────────────────────────────────

class TestStalenessConfidenceDecay:
    """Test that stale-but-gated headlines get reduced confidence."""

    def _make_headline(self, title, age_minutes):
        return {
            "title": title,
            "published_at": time.time() - (age_minutes * 60),
            "source": "test",
        }

    @pytest.mark.asyncio
    async def test_brand_new_headline_full_confidence(self):
        """A brand-new headline should get full (undecayed) confidence."""
        bni = BTCNewsIntelligence()
        headlines = [
            self._make_headline("Fed raises rates, recession fears grow — bitcoin plummets sharply", 0),
        ]
        await bni.process_headlines(headlines)
        ctx_fresh = bni.get_event_context()

        if ctx_fresh.is_active:
            # Fresh should have higher confidence than stale
            bni2 = BTCNewsIntelligence()
            headlines_stale = [
                self._make_headline("Fed raises rates, recession fears grow — bitcoin plummets sharply", 30),
            ]
            await bni2.process_headlines(headlines_stale)
            ctx_stale = bni2.get_event_context()
            if ctx_stale.is_active:
                assert ctx_fresh.confidence > ctx_stale.confidence, \
                    f"Fresh ({ctx_fresh.confidence:.3f}) should be higher than stale ({ctx_stale.confidence:.3f})"

    @pytest.mark.asyncio
    async def test_confidence_floor_respected(self):
        """Even at max gate age, confidence should not go below the floor × original."""
        bni = BTCNewsIntelligence()
        gate = NewsIntelligence.STALE_HEADLINE_ALERT_GATE_MINUTES
        floor = NewsIntelligence.STALE_HEADLINE_CONFIDENCE_FLOOR

        # Get baseline confidence with a fresh headline
        bni_fresh = BTCNewsIntelligence()
        headlines_fresh = [
            self._make_headline("Fed raises rates, recession fears grow — bitcoin plummets sharply", 0),
        ]
        await bni_fresh.process_headlines(headlines_fresh)
        ctx_fresh = bni_fresh.get_event_context()
        if not ctx_fresh.is_active:
            pytest.skip("Headline didn't create context (classification too weak)")

        # Now test with near-gate-age headline
        headlines_old = [
            self._make_headline("Fed raises rates, recession fears grow — bitcoin plummets sharply", gate - 1),
        ]
        await bni.process_headlines(headlines_old)
        ctx = bni.get_event_context()
        if ctx.is_active:
            # Confidence should be >= floor × original (not 0)
            assert ctx.confidence >= ctx_fresh.confidence * floor * 0.9, \
                f"Confidence {ctx.confidence:.3f} below floor ({ctx_fresh.confidence:.3f} × {floor} = {ctx_fresh.confidence * floor:.3f})"


# ─── Telegram alert age display tests ────────────────────────────

class TestAlertAgeDisplay:
    """Test the age string logic via BTCEventContext.format_headline_age()."""

    def test_age_string_just_now(self):
        """Headlines < 2 min old should show 'Just now'."""
        ctx = BTCEventContext(
            detected_at=time.time(),
            headline_published_at=time.time() - 60,  # 1 min ago
        )
        assert ctx.format_headline_age() == "🕐 Just now"

    def test_age_string_minutes(self):
        """Headlines 2-120 min old should show minutes."""
        ctx = BTCEventContext(
            detected_at=time.time(),
            headline_published_at=time.time() - 1500,  # 25 min ago
        )
        assert ctx.format_headline_age() == "🕐 Published 25m ago"

    def test_age_string_hours(self):
        """Headlines ≥120 min old should show hours."""
        ctx = BTCEventContext(
            detected_at=time.time(),
            headline_published_at=time.time() - 10800,  # 3 hours ago
        )
        assert ctx.format_headline_age() == "🕐 Published 3.0h ago"


# ─── Dashboard API age field tests ────────────────────────────────

class TestDashboardAgeFields:
    """Test that get_status() includes headline age fields."""

    def test_status_includes_headline_age(self):
        """get_status() should include headline_age_minutes and headline_published_at."""
        bni = BTCNewsIntelligence()
        status = bni.get_status()
        assert "headline_age_minutes" in status
        assert "headline_published_at" in status

    def test_status_headline_age_reflects_published_at(self):
        """When context has headline_published_at, status should reflect it."""
        bni = BTCNewsIntelligence()
        now = time.time()
        bni._current_ctx = BTCEventContext(
            event_type=BTCEventType.MACRO_RISK_OFF,
            confidence=0.7,
            direction="BEARISH",
            detected_at=now,
            expires_at=now + 3600,
            headline="Test headline",
            headline_published_at=now - 1200,  # 20 min ago
        )
        status = bni.get_status()
        assert 19 < status["headline_age_minutes"] < 21
        assert status["headline_published_at"] == now - 1200


# ─── Constants sanity checks ─────────────────────────────────────

class TestTimelynessConstants:
    """Verify timeliness constants are reasonable."""

    def test_gate_is_positive(self):
        assert NewsIntelligence.STALE_HEADLINE_ALERT_GATE_MINUTES > 0

    def test_confidence_floor_is_between_zero_and_one(self):
        f = NewsIntelligence.STALE_HEADLINE_CONFIDENCE_FLOOR
        assert 0 < f < 1

    def test_gate_less_than_max_headline_age(self):
        """Alert gate should be ≤ headline max age (otherwise we'd alert on discarded headlines)."""
        assert NewsIntelligence.STALE_HEADLINE_ALERT_GATE_MINUTES <= \
               NewsIntelligence.HEADLINE_MAX_AGE_MINUTES
