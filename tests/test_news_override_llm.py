"""
Tests for the LLM-assisted news re-ranker, override system, and
ambiguity / de-escalation detection in NewsClassifier.

All tests are deterministic and avoid real network / LLM calls.  LLM
behaviour is exercised via a stub LLMClient.
"""

from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import pytest


# ══════════════════════════════════════════════════════════════
# Phase 1 — Ambiguity detector + de-escalation inverter
# ══════════════════════════════════════════════════════════════

class TestAmbiguityDetector:
    """Verify the new de-escalation + ambiguity heuristics."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from analyzers.btc_news_intelligence import NewsClassifier, BTCEventType
        self.classifier = NewsClassifier()
        self.BTCEventType = BTCEventType

    def test_iran_delegation_flagged_as_mixed(self):
        """The motivating case — diplomatic engagement headline must not
        produce a pure risk-off hard-block."""
        title = "Iran plans delegation to Pakistan amid military threats"
        etype, direction, conf, is_mixed = self.classifier.classify(title)
        assert etype == self.BTCEventType.MACRO_RISK_OFF
        assert direction == "BEARISH"
        assert is_mixed is True

    @pytest.mark.parametrize("title", [
        "Iran and US agree to resume diplomatic talks",
        "Ceasefire announced as military operations pause",
        "North Korea summit plans dialogue amid nuclear concerns",
    ])
    def test_de_escalation_cues_flip_to_mixed(self, title):
        etype, direction, conf, is_mixed = self.classifier.classify(title)
        # Still MACRO_RISK_OFF (keyword stays authoritative), but marked mixed.
        assert is_mixed is True, f"Expected is_mixed=True for: {title!r}"

    def test_pure_escalation_still_not_mixed(self):
        title = "Military strike on oil facilities causes market panic"
        etype, direction, conf, is_mixed = self.classifier.classify(title)
        assert etype == self.BTCEventType.MACRO_RISK_OFF
        assert is_mixed is False

    def test_diplomatic_breakdown_is_not_de_escalation(self):
        """Escalation-veto: 'diplomatic breakdown' contains diplomacy-flavoured
        language but is unambiguously escalation — must not be flagged mixed."""
        title = "Diplomatic breakdown between Iran and Israel after failed talks"
        etype, direction, conf, is_mixed = self.classifier.classify(title)
        assert etype == self.BTCEventType.MACRO_RISK_OFF
        assert is_mixed is False, (
            "Headlines containing escalation-veto phrases (breakdown/fail/…) "
            "must NOT be softened via the de-escalation path."
        )

    def test_peace_talks_collapse_is_not_de_escalation(self):
        title = "Peace talks collapse as North Korea walks out of summit"
        etype, direction, conf, is_mixed = self.classifier.classify(title)
        # 'collapse' + 'walk out' veto the de-escalation inversion even though
        # "peace talks" and "summit" are in the de-escalation list.
        assert is_mixed is False

    def test_is_ambiguous_flags_mixed(self):
        title = "Iran plans delegation to Pakistan amid military threats"
        etype, direction, conf, is_mixed = self.classifier.classify(title)
        amb, reason = self.classifier.is_ambiguous(title, etype, conf, is_mixed)
        assert amb is True
        assert reason in ("mixed_signal", "de_escalation_cue", "high_severity", "low_confidence_high_impact")

    def test_is_ambiguous_not_flagged_for_benign_headline(self):
        title = "Bitcoin price holds steady at 60k after quiet weekend"
        etype, direction, conf, is_mixed = self.classifier.classify(title)
        amb, reason = self.classifier.is_ambiguous(title, etype, conf, is_mixed)
        # UNKNOWN event type with zero confidence is never ambiguous.
        assert amb is False

    def test_is_ambiguous_high_severity_hard_hitter(self):
        # Strong exchange-event headline — high severity even without mixed.
        title = "Major exchange hack: withdrawals suspended amid insolvency fears"
        etype, direction, conf, is_mixed = self.classifier.classify(title)
        assert etype == self.BTCEventType.EXCHANGE_EVENT
        amb, reason = self.classifier.is_ambiguous(title, etype, conf, is_mixed)
        # We want high-severity risk-off classes to be flagged (worth $0.0002).
        if conf >= 0.65:
            assert amb is True
            assert reason == "high_severity"


# ══════════════════════════════════════════════════════════════
# Phase 2 — Manual override system
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def override_store(monkeypatch):
    """Fresh NewsOverrideStore with a stubbed DB."""
    from analyzers import news_override
    # Stub db.save_learning_state / load_learning_state
    fake_db = MagicMock()
    fake_db.save_learning_state = AsyncMock()
    fake_db.load_learning_state = AsyncMock(return_value=None)
    fake_module = MagicMock(db=fake_db)
    monkeypatch.setitem(sys.modules, "data.database", fake_module)
    store = news_override.NewsOverrideStore()
    return store, fake_db


class TestNewsOverrideStore:
    @pytest.mark.asyncio
    async def test_set_and_get_override(self, override_store):
        store, _ = override_store
        ov = await store.set_override(
            event_type="MACRO_RISK_OFF",
            direction="BEARISH",
            confidence_mult=0.9,
            size_mult=0.9,
            reason="Iran delegation is de-escalation",
            set_by="alice",
            ttl_minutes=30,
        )
        assert ov.event_type == "MACRO_RISK_OFF"
        active = store.get_active()
        assert active is not None
        assert active.reason.startswith("Iran delegation")

    @pytest.mark.asyncio
    async def test_override_clamps_out_of_range(self, override_store):
        store, _ = override_store
        ov = await store.set_override(
            event_type="MACRO_RISK_ON",
            direction="BULLISH",
            confidence_mult=99.0,       # should clamp to MAX_CONF_MULT
            size_mult=-5.0,             # should clamp to MIN_SIZE_MULT
            ttl_minutes=10_000,         # should clamp to MAX_TTL_MINUTES
        )
        from config.constants import NewsOverrideDefaults
        assert ov.confidence_mult == NewsOverrideDefaults.MAX_CONF_MULT
        assert ov.size_mult == NewsOverrideDefaults.MIN_SIZE_MULT
        ttl_m = (ov.expires_at_utc - ov.set_at_utc) / 60
        assert ttl_m <= NewsOverrideDefaults.MAX_TTL_MINUTES + 1

    @pytest.mark.asyncio
    async def test_set_override_rejects_invalid_event_type(self, override_store):
        """Typos like FOOBAR must raise at set time instead of silently
        installing a no-op override."""
        store, _ = override_store
        with pytest.raises(ValueError, match="Unknown event_type"):
            await store.set_override(
                event_type="FOOBAR",
                direction="BEARISH",
                confidence_mult=0.9,
                size_mult=0.9,
            )
        assert store.get_active() is None

    @pytest.mark.asyncio
    async def test_load_is_idempotent_and_hydrates_persisted_override(
        self, monkeypatch
    ):
        """After a 'restart' (fresh store, DB has a payload), load() must
        hydrate the in-memory override exactly once."""
        from analyzers import news_override
        now = time.time()
        persisted = {
            "active": {
                "event_type": "MACRO_RISK_OFF",
                "direction": "BEARISH",
                "confidence_mult": 0.8,
                "size_mult": 0.8,
                "set_by": "alice",
                "set_at_utc": now,
                "expires_at_utc": now + 3600,
                "consume_on_next_event": True,
                "declared_severity": "MEDIUM",
                "reason": "persisted before restart",
                "source_headline": "",
                "block_longs": False,
                "block_shorts": False,
            },
            "trust": {"alice": 0.1},
            "history": [],
        }
        fake_db = MagicMock()
        fake_db.save_learning_state = AsyncMock()
        fake_db.load_learning_state = AsyncMock(return_value=persisted)
        monkeypatch.setitem(
            sys.modules, "data.database", MagicMock(db=fake_db)
        )
        store = news_override.NewsOverrideStore()
        # Before load: sync get_active sees nothing.
        assert store.get_active() is None
        # After load: override is hydrated.
        await store.load()
        ov = store.get_active()
        assert ov is not None
        assert ov.event_type == "MACRO_RISK_OFF"
        assert ov.set_by == "alice"
        assert store.trust_score("alice") == pytest.approx(0.1)
        # Idempotent second call must not re-issue the DB read.
        fake_db.load_learning_state.reset_mock()
        await store.load()
        fake_db.load_learning_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_override_ttl_expiry(self, override_store):
        store, _ = override_store
        ov = await store.set_override(
            event_type="MACRO_RISK_OFF",
            direction="BEARISH",
            confidence_mult=0.8,
            size_mult=0.8,
            ttl_minutes=1,
        )
        # Manually fast-forward the expiry.
        ov.expires_at_utc = time.time() - 1
        assert store.get_active() is None

    @pytest.mark.asyncio
    async def test_consume_on_agree(self, override_store):
        store, _ = override_store
        await store.set_override(
            event_type="MACRO_RISK_OFF",
            direction="BEARISH",
            confidence_mult=0.9,
            size_mult=0.9,
            set_by="alice",
            consume_on_next_event=True,
            declared_severity="MEDIUM",
        )
        result = await store.consume_if_applicable(
            new_event_type="MACRO_RISK_OFF",
            new_direction="BEARISH",
            new_severity="HIGH",
            new_headline="Iran missile strike escalates crisis",
        )
        assert result is not None
        outcome, prev = result
        assert outcome == "agree"
        assert store.get_active() is None
        assert store.trust_score("alice") > 0

    @pytest.mark.asyncio
    async def test_consume_on_disagree(self, override_store):
        store, _ = override_store
        await store.set_override(
            event_type="MACRO_RISK_OFF",
            direction="BEARISH",
            confidence_mult=0.9,
            size_mult=0.9,
            set_by="alice",
            consume_on_next_event=True,
            declared_severity="MEDIUM",
        )
        result = await store.consume_if_applicable(
            new_event_type="MACRO_RISK_ON",
            new_direction="BULLISH",
            new_severity="HIGH",
        )
        assert result is not None
        outcome, _ = result
        assert outcome == "disagree"
        assert store.trust_score("alice") < 0

    @pytest.mark.asyncio
    async def test_consume_skipped_if_below_severity(self, override_store):
        store, _ = override_store
        await store.set_override(
            event_type="MACRO_RISK_OFF",
            direction="BEARISH",
            confidence_mult=0.9,
            size_mult=0.9,
            consume_on_next_event=True,
            declared_severity="HIGH",
        )
        result = await store.consume_if_applicable(
            new_event_type="BTC_TECHNICAL",
            new_direction="NEUTRAL",
            new_severity="LOW",
        )
        assert result is None
        assert store.get_active() is not None, "trivial event must not consume override"

    @pytest.mark.asyncio
    async def test_ttl_only_mode_not_consumed(self, override_store):
        store, _ = override_store
        await store.set_override(
            event_type="MACRO_RISK_OFF",
            direction="BEARISH",
            confidence_mult=0.9,
            size_mult=0.9,
            consume_on_next_event=False,     # pure TTL
        )
        result = await store.consume_if_applicable(
            new_event_type="MACRO_RISK_OFF",
            new_direction="BEARISH",
            new_severity="HIGH",
        )
        assert result is None
        assert store.get_active() is not None


# ══════════════════════════════════════════════════════════════
# Phase 3 — LLM re-ranker: schema veto, sanity veto, circuit breaker
# ══════════════════════════════════════════════════════════════

class _StubClient:
    """Injectable test double implementing the LLMClient protocol."""

    def __init__(self, responses=None, behaviour="ok"):
        self._responses = list(responses or [])
        self._behaviour = behaviour
        self.calls = 0

    async def classify(self, title, body, keyword_verdict):
        self.calls += 1
        if self._behaviour == "timeout":
            await asyncio.sleep(10)  # will trip the re-ranker timeout
            return {}
        if self._behaviour == "exception":
            raise RuntimeError("boom")
        if self._responses:
            return self._responses.pop(0)
        return {
            "event_type": "MACRO_RISK_OFF",
            "direction": "BEARISH",
            "confidence": 0.6,
            "is_escalation": True,
            "reasoning": "stub",
        }


@pytest.fixture
def enabled_llm(monkeypatch):
    """Enable the LLM with a fresh re-ranker; reset flags after test."""
    from config.constants import NewsLLM
    monkeypatch.setattr(NewsLLM, "ENABLED", True)
    monkeypatch.setattr(NewsLLM, "SHADOW_MODE", False)
    # Tight timeout so the timeout test doesn't hang tests.
    monkeypatch.setattr(NewsLLM, "CALL_TIMEOUT_SEC", 0.1)
    from analyzers.news_llm_classifier import LLMReRanker
    return LLMReRanker()


class TestLLMReRankerVeto:
    @pytest.mark.asyncio
    async def test_disabled_returns_keyword(self, monkeypatch):
        from config.constants import NewsLLM
        monkeypatch.setattr(NewsLLM, "ENABLED", False)
        from analyzers.news_llm_classifier import LLMReRanker
        from analyzers.btc_news_intelligence import BTCEventType
        rr = LLMReRanker(client=_StubClient())
        result = await rr.rerank(
            title="test",
            body="",
            keyword_event_type=BTCEventType.MACRO_RISK_OFF,
            keyword_direction="BEARISH",
            keyword_confidence=0.5,
            keyword_is_mixed=False,
        )
        assert result.source == "keyword"
        assert result.direction == "BEARISH"

    @pytest.mark.asyncio
    async def test_schema_veto_on_bad_event_type(self, enabled_llm):
        from analyzers.btc_news_intelligence import BTCEventType
        enabled_llm.set_client(_StubClient([
            {"event_type": "INVENTED_CATEGORY", "direction": "BULLISH", "confidence": 0.9}
        ]))
        result = await enabled_llm.rerank(
            title="odd headline",
            body="",
            keyword_event_type=BTCEventType.MACRO_RISK_OFF,
            keyword_direction="BEARISH",
            keyword_confidence=0.5,
            keyword_is_mixed=False,
        )
        assert result.source == "keyword"
        assert result.direction == "BEARISH"
        assert result.llm_verdict is not None
        assert not result.llm_verdict.ok
        assert "unknown_event_type" in result.llm_verdict.reject_reason

    @pytest.mark.asyncio
    async def test_schema_veto_on_out_of_range_confidence(self, enabled_llm):
        from analyzers.btc_news_intelligence import BTCEventType
        enabled_llm.set_client(_StubClient([
            {"event_type": "MACRO_RISK_OFF", "direction": "BEARISH", "confidence": 99.0}
        ]))
        result = await enabled_llm.rerank(
            title="x", body="",
            keyword_event_type=BTCEventType.MACRO_RISK_OFF,
            keyword_direction="BEARISH",
            keyword_confidence=0.5,
            keyword_is_mixed=False,
        )
        assert result.source == "keyword"

    @pytest.mark.asyncio
    async def test_sanity_veto_on_low_confidence_flip(self, enabled_llm):
        """LLM wants to flip BEARISH→BULLISH but keyword is high-conviction
        and LLM confidence < MIN_FLIP_CONFIDENCE → downgrade to mixed, don't flip."""
        from analyzers.btc_news_intelligence import BTCEventType
        enabled_llm.set_client(_StubClient([
            {"event_type": "MACRO_RISK_ON", "direction": "BULLISH", "confidence": 0.4}
        ]))
        result = await enabled_llm.rerank(
            title="x", body="",
            keyword_event_type=BTCEventType.MACRO_RISK_OFF,
            keyword_direction="BEARISH",
            keyword_confidence=0.8,     # high conviction
            keyword_is_mixed=False,
            keyword_hit_count=3,        # multiple distinct hits
        )
        assert result.source == "keyword_veto"
        assert result.direction == "BEARISH"
        assert result.is_mixed is True

    @pytest.mark.asyncio
    async def test_honours_high_confidence_flip(self, enabled_llm):
        from analyzers.btc_news_intelligence import BTCEventType
        enabled_llm.set_client(_StubClient([
            {"event_type": "MACRO_RISK_ON", "direction": "BULLISH", "confidence": 0.85}
        ]))
        result = await enabled_llm.rerank(
            title="x", body="",
            keyword_event_type=BTCEventType.MACRO_RISK_OFF,
            keyword_direction="BEARISH",
            keyword_confidence=0.55,
            keyword_is_mixed=False,
            keyword_hit_count=2,
        )
        assert result.source == "llm"
        assert result.direction == "BULLISH"

    @pytest.mark.asyncio
    async def test_timeout_trips_circuit_breaker(self, enabled_llm, monkeypatch):
        from config.constants import NewsLLM
        from analyzers.btc_news_intelligence import BTCEventType
        monkeypatch.setattr(NewsLLM, "CIRCUIT_BREAKER_FAILS", 2)
        enabled_llm.set_client(_StubClient(behaviour="timeout"))
        kwargs = dict(
            title="x", body="",
            keyword_event_type=BTCEventType.MACRO_RISK_OFF,
            keyword_direction="BEARISH",
            keyword_confidence=0.5,
            keyword_is_mixed=False,
        )
        r1 = await enabled_llm.rerank(**kwargs)
        r2 = await enabled_llm.rerank(**kwargs)
        # Both are timeouts, both return keyword.
        assert r1.source == "keyword"
        assert r2.source == "keyword"
        # Now circuit should be tripped — next call short-circuits.
        r3 = await enabled_llm.rerank(**kwargs)
        assert r3.source == "keyword"
        assert "llm_circuit_open" in r3.notes
        # And the client should NOT have been called a 3rd time.
        assert enabled_llm._client.calls == 2

    @pytest.mark.asyncio
    async def test_shadow_mode_returns_keyword(self, monkeypatch):
        from config.constants import NewsLLM
        from analyzers.btc_news_intelligence import BTCEventType
        monkeypatch.setattr(NewsLLM, "ENABLED", True)
        monkeypatch.setattr(NewsLLM, "SHADOW_MODE", True)
        monkeypatch.setattr(NewsLLM, "CALL_TIMEOUT_SEC", 1.0)
        from analyzers.news_llm_classifier import LLMReRanker
        rr = LLMReRanker(client=_StubClient([
            {"event_type": "MACRO_RISK_ON", "direction": "BULLISH", "confidence": 0.95}
        ]))
        result = await rr.rerank(
            title="x", body="",
            keyword_event_type=BTCEventType.MACRO_RISK_OFF,
            keyword_direction="BEARISH",
            keyword_confidence=0.5,
            keyword_is_mixed=False,
        )
        assert result.source == "llm_shadow"
        # Keyword verdict is returned even though LLM disagreed strongly.
        assert result.direction == "BEARISH"


# ══════════════════════════════════════════════════════════════
# Config schema validators
# ══════════════════════════════════════════════════════════════

class TestConfigSchemaValidators:
    def test_news_llm_validators_clean(self):
        from config.schema import validate_analyzer_constants
        errors = validate_analyzer_constants()
        # The module should at least not fail on our new invariants
        # with default values.  Filter to just our classes for clarity.
        ours = [e for e in errors if e.startswith(("NewsLLM:", "NewsOverrideDefaults:"))]
        assert ours == [], f"Unexpected NewsLLM/NewsOverrideDefaults errors: {ours}"

    def test_news_llm_validator_catches_bad_timeout(self, monkeypatch):
        from config.constants import NewsLLM
        from config.schema import validate_analyzer_constants
        monkeypatch.setattr(NewsLLM, "CALL_TIMEOUT_SEC", 0.0)
        errors = validate_analyzer_constants()
        assert any("CALL_TIMEOUT_SEC" in e for e in errors)
