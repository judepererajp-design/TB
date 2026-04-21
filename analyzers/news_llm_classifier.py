"""
TitanBot Pro — LLM-Assisted News Re-Ranker
============================================

Thin wrapper around an injectable LLM client.  Consulted only for headlines
flagged as ambiguous by :meth:`analyzers.btc_news_intelligence.NewsClassifier.is_ambiguous`,
and even then every verdict passes through a strict veto pipeline before
it can modify the keyword result:

1. **Schema validation** — the LLM must return a known
   :class:`BTCEventType`, a valid direction, and a parseable confidence.
   Any deviation → discard, keyword verdict wins.
2. **Sanity veto** — if the LLM wants to flip direction but the keyword
   hit count + confidence is high, require ``MIN_FLIP_CONFIDENCE`` or we
   downgrade to ``is_mixed=True`` (soft path) instead of honouring the flip.
3. **Circuit breaker** — consecutive failures trip a cooldown; during the
   cooldown every call short-circuits to the keyword verdict.
4. **Disagreement log** — every keyword/LLM disagreement is appended to
   persistent learning state for later review + keyword-rule evolution.

When :attr:`NewsLLM.SHADOW_MODE` is True the LLM is called and logged but
its verdict is **not** honoured — the keyword verdict is always returned.
This is the rollout posture for the first week after enablement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple

from config.constants import NewsLLM

logger = logging.getLogger(__name__)


# Local import deferred — avoid circular dep with btc_news_intelligence.
_BTCEventType = None  # type: ignore[assignment]


def _get_btc_event_type():
    global _BTCEventType
    if _BTCEventType is None:
        from analyzers.btc_news_intelligence import BTCEventType as _ET
        _BTCEventType = _ET
    return _BTCEventType


# ── LLM client protocol ──────────────────────────────────────

class LLMClient(Protocol):
    """Minimal protocol the re-ranker expects from any LLM backend.

    Implementations return a JSON-like dict with keys:
      * ``event_type`` — one of the BTCEventType string values
      * ``direction``  — "BULLISH" | "BEARISH" | "NEUTRAL"
      * ``confidence`` — float in [0, 1]
      * ``is_escalation`` — bool (geopolitical escalation or de-escalation)
      * ``reasoning`` — short free-form string

    The protocol is intentionally async so real clients (OpenAI, Anthropic,
    local llama.cpp) can await their HTTP/IPC layer.  Tests supply a plain
    dict-returning stub.
    """

    async def classify(self, title: str, body: str, keyword_verdict: Dict[str, Any]) -> Dict[str, Any]:
        ...  # pragma: no cover


# ── Data records ─────────────────────────────────────────────

@dataclass
class LLMVerdict:
    """A validated LLM response, or a rejection explanation."""

    ok: bool = False
    event_type: str = "UNKNOWN"
    direction: str = "NEUTRAL"
    confidence: float = 0.0
    is_escalation: Optional[bool] = None
    reasoning: str = ""
    reject_reason: str = ""              # non-empty when ok=False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "event_type": self.event_type,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "is_escalation": self.is_escalation,
            "reasoning": self.reasoning[:240],
            "reject_reason": self.reject_reason[:120],
        }


@dataclass
class ReRankResult:
    """Final result returned to the news pipeline."""

    event_type: str
    direction: str
    confidence: float
    is_mixed: bool
    source: str                          # "keyword", "llm", "llm_shadow", "keyword_veto"
    llm_verdict: Optional[LLMVerdict] = None
    notes: List[str] = field(default_factory=list)


# ── Circuit breaker state ────────────────────────────────────

@dataclass
class _CircuitState:
    recent_failures: List[float] = field(default_factory=list)
    tripped_until: float = 0.0

    def record_failure(self, now: float, window_sec: float, fail_threshold: int, cooldown_sec: float) -> None:
        cutoff = now - window_sec
        self.recent_failures = [t for t in self.recent_failures if t >= cutoff]
        self.recent_failures.append(now)
        if len(self.recent_failures) >= fail_threshold:
            self.tripped_until = now + cooldown_sec
            self.recent_failures.clear()

    def record_success(self) -> None:
        self.recent_failures.clear()

    def is_tripped(self, now: float) -> bool:
        return now < self.tripped_until

    def status(self, now: float) -> str:
        if self.is_tripped(now):
            remaining = max(0.0, self.tripped_until - now)
            return f"DEGRADED ({remaining:.0f}s)"
        if self.recent_failures:
            return f"WARN ({len(self.recent_failures)} recent fails)"
        return "OK"


# ── Re-ranker ────────────────────────────────────────────────

class LLMReRanker:
    """
    Applies the LLM re-rank + veto pipeline to a single headline.

    The re-ranker is stateless w.r.t. the news pipeline and only holds a
    small circuit-breaker state plus a pointer to the injected client.
    """

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self._client = client
        self._circuit = _CircuitState()
        # In-memory disagreement buffer; flushed to DB via
        # ``_persist_disagreement``.  We keep a local copy so tests can
        # inspect without a round-trip.
        self._disagreement_buffer: List[Dict[str, Any]] = []

    def set_client(self, client: Optional[LLMClient]) -> None:
        self._client = client

    # ── Main entry point ──────────────────────────────────────
    async def rerank(
        self,
        *,
        title: str,
        body: str,
        keyword_event_type,
        keyword_direction: str,
        keyword_confidence: float,
        keyword_is_mixed: bool,
        keyword_hit_count: int = 1,
    ) -> ReRankResult:
        """
        Run the re-rank + veto pipeline.  Always returns a :class:`ReRankResult`
        — callers never see an exception from this method.
        """
        BTCEventType = _get_btc_event_type()
        kw_type_val = getattr(keyword_event_type, "value", str(keyword_event_type))
        now = time.time()
        notes: List[str] = []

        # ── Fast fail: feature flag / circuit breaker / no client ──
        if not NewsLLM.ENABLED:
            notes.append("llm_disabled")
            return ReRankResult(
                event_type=kw_type_val,
                direction=keyword_direction,
                confidence=keyword_confidence,
                is_mixed=keyword_is_mixed,
                source="keyword",
                notes=notes,
            )
        if self._client is None:
            notes.append("llm_no_client")
            return ReRankResult(
                event_type=kw_type_val,
                direction=keyword_direction,
                confidence=keyword_confidence,
                is_mixed=keyword_is_mixed,
                source="keyword",
                notes=notes,
            )
        if self._circuit.is_tripped(now):
            notes.append("llm_circuit_open")
            return ReRankResult(
                event_type=kw_type_val,
                direction=keyword_direction,
                confidence=keyword_confidence,
                is_mixed=keyword_is_mixed,
                source="keyword",
                notes=notes,
            )

        # ── Call the LLM with timeout ────────────────────────────
        keyword_payload = {
            "event_type": kw_type_val,
            "direction": keyword_direction,
            "confidence": round(float(keyword_confidence), 3),
            "is_mixed": bool(keyword_is_mixed),
        }
        body_snippet = (body or "")[: max(0, int(NewsLLM.MAX_BODY_CHARS))]

        try:
            raw = await asyncio.wait_for(
                self._client.classify(title, body_snippet, keyword_payload),
                timeout=float(NewsLLM.CALL_TIMEOUT_SEC),
            )
        except asyncio.TimeoutError:
            self._circuit.record_failure(
                now,
                NewsLLM.CIRCUIT_BREAKER_WINDOW_SEC,
                NewsLLM.CIRCUIT_BREAKER_FAILS,
                NewsLLM.CIRCUIT_BREAKER_COOLDOWN_SEC,
            )
            notes.append("llm_timeout")
            logger.warning(f"LLM re-rank timeout on headline: {title[:80]!r}")
            return ReRankResult(
                event_type=kw_type_val,
                direction=keyword_direction,
                confidence=keyword_confidence,
                is_mixed=keyword_is_mixed,
                source="keyword",
                notes=notes,
            )
        except Exception as exc:
            self._circuit.record_failure(
                now,
                NewsLLM.CIRCUIT_BREAKER_WINDOW_SEC,
                NewsLLM.CIRCUIT_BREAKER_FAILS,
                NewsLLM.CIRCUIT_BREAKER_COOLDOWN_SEC,
            )
            notes.append(f"llm_error:{type(exc).__name__}")
            logger.warning(f"LLM re-rank error ({type(exc).__name__}: {exc}) on headline: {title[:80]!r}")
            return ReRankResult(
                event_type=kw_type_val,
                direction=keyword_direction,
                confidence=keyword_confidence,
                is_mixed=keyword_is_mixed,
                source="keyword",
                notes=notes,
            )

        # ── Validate schema ──────────────────────────────────────
        verdict = self._validate_schema(raw)
        if not verdict.ok:
            # Schema failure is NOT a transport failure — don't trip the
            # circuit breaker on repeated malformed outputs (otherwise a
            # model change could silently knock us offline).  We log and
            # fall back to keyword.
            notes.append(f"llm_schema_fail:{verdict.reject_reason}")
            self._log_disagreement(
                title=title,
                keyword=keyword_payload,
                verdict=verdict,
                outcome="schema_fail",
            )
            return ReRankResult(
                event_type=kw_type_val,
                direction=keyword_direction,
                confidence=keyword_confidence,
                is_mixed=keyword_is_mixed,
                source="keyword",
                llm_verdict=verdict,
                notes=notes,
            )
        self._circuit.record_success()

        # ── Sanity veto on direction flip ────────────────────────
        kw_dir_u = (keyword_direction or "").upper()
        llm_dir_u = (verdict.direction or "").upper()
        same_direction = kw_dir_u == llm_dir_u
        flipping = (
            not same_direction
            and kw_dir_u in ("BULLISH", "BEARISH")
            and llm_dir_u in ("BULLISH", "BEARISH")
        )
        veto_flip = False
        if flipping:
            high_conviction_keyword = (
                float(keyword_confidence) >= float(NewsLLM.FLIP_GUARD_KEYWORD_CONF)
                and int(keyword_hit_count) >= int(NewsLLM.FLIP_GUARD_MIN_KEYWORD_HITS)
            )
            if high_conviction_keyword and verdict.confidence < float(NewsLLM.MIN_FLIP_CONFIDENCE):
                veto_flip = True
                notes.append("sanity_veto_flip")

        disagrees = (
            verdict.event_type != kw_type_val
            or not same_direction
        )
        if disagrees:
            self._log_disagreement(
                title=title,
                keyword=keyword_payload,
                verdict=verdict,
                outcome="honoured" if (not veto_flip and not NewsLLM.SHADOW_MODE) else "shadow_or_vetoed",
            )

        # ── Shadow mode: always return keyword verdict ──────────
        if NewsLLM.SHADOW_MODE:
            notes.append("shadow_mode")
            return ReRankResult(
                event_type=kw_type_val,
                direction=keyword_direction,
                confidence=keyword_confidence,
                is_mixed=keyword_is_mixed,
                source="llm_shadow",
                llm_verdict=verdict,
                notes=notes,
            )

        # ── Apply veto: flipping but not confident enough → mixed ──
        if veto_flip:
            return ReRankResult(
                event_type=kw_type_val,
                direction=keyword_direction,
                confidence=keyword_confidence,
                is_mixed=True,      # soft path — let existing mixed handling dampen
                source="keyword_veto",
                llm_verdict=verdict,
                notes=notes,
            )

        # ── Honour the LLM ──────────────────────────────────────
        return ReRankResult(
            event_type=verdict.event_type,
            direction=verdict.direction,
            confidence=float(verdict.confidence),
            is_mixed=keyword_is_mixed,
            source="llm",
            llm_verdict=verdict,
            notes=notes,
        )

    # ── Schema validation ────────────────────────────────────
    def _validate_schema(self, raw: Any) -> LLMVerdict:
        BTCEventType = _get_btc_event_type()
        if not isinstance(raw, dict):
            return LLMVerdict(ok=False, reject_reason="not_a_dict")

        et = raw.get("event_type")
        if not isinstance(et, str):
            return LLMVerdict(ok=False, reject_reason="missing_event_type")
        et_upper = et.strip().upper()
        try:
            BTCEventType(et_upper)
        except ValueError:
            return LLMVerdict(ok=False, reject_reason=f"unknown_event_type:{et_upper[:32]}")

        direction = raw.get("direction")
        if not isinstance(direction, str):
            return LLMVerdict(ok=False, reject_reason="missing_direction")
        dir_upper = direction.strip().upper()
        if dir_upper not in ("BULLISH", "BEARISH", "NEUTRAL"):
            return LLMVerdict(ok=False, reject_reason=f"invalid_direction:{dir_upper[:16]}")

        conf_raw = raw.get("confidence")
        try:
            confidence = float(conf_raw)
        except (TypeError, ValueError):
            return LLMVerdict(ok=False, reject_reason="invalid_confidence")
        if not (0.0 <= confidence <= 1.0):
            return LLMVerdict(ok=False, reject_reason=f"out_of_range_confidence:{confidence}")

        is_esc = raw.get("is_escalation")
        if is_esc is not None and not isinstance(is_esc, bool):
            is_esc = None  # tolerate missing or soft-type

        reasoning = raw.get("reasoning")
        if not isinstance(reasoning, str):
            reasoning = ""

        return LLMVerdict(
            ok=True,
            event_type=et_upper,
            direction=dir_upper,
            confidence=confidence,
            is_escalation=is_esc,
            reasoning=reasoning,
        )

    # ── Disagreement log ────────────────────────────────────
    def _log_disagreement(
        self,
        *,
        title: str,
        keyword: Dict[str, Any],
        verdict: LLMVerdict,
        outcome: str,
    ) -> None:
        entry = {
            "ts": time.time(),
            "title": (title or "")[:240],
            "keyword": keyword,
            "llm": verdict.to_dict(),
            "outcome": outcome,
        }
        self._disagreement_buffer.append(entry)
        cap = int(NewsLLM.DISAGREEMENT_LOG_CAP)
        if len(self._disagreement_buffer) > cap:
            self._disagreement_buffer = self._disagreement_buffer[-cap:]
        # Fire-and-forget persistence; tests supply a MagicMock db.
        try:
            asyncio.get_event_loop().create_task(self._persist_disagreement())
        except RuntimeError:
            # No running loop (sync test) — skip; buffer still readable.
            pass

    async def _persist_disagreement(self) -> None:
        try:
            from data.database import db
        except Exception:  # pragma: no cover
            return
        try:
            await db.save_learning_state(
                NewsLLM.DISAGREEMENT_STATE_KEY,
                {"entries": list(self._disagreement_buffer)},
            )
        except Exception as exc:
            logger.debug(f"persist_disagreement failed: {exc}")

    # ── Observability ───────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "enabled": bool(NewsLLM.ENABLED),
            "shadow_mode": bool(NewsLLM.SHADOW_MODE),
            "has_client": self._client is not None,
            "circuit": self._circuit.status(now),
            "disagreement_buffered": len(self._disagreement_buffer),
        }

    def recent_disagreements(self, n: int = 20) -> List[Dict[str, Any]]:
        return list(self._disagreement_buffer[-n:])


# ── Module-level singleton ───────────────────────────────────
llm_reranker = LLMReRanker()
