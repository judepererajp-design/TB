"""
TitanBot Pro — Manual News Classification Override
===================================================

Operator-set overrides that replace or modify the automated BTC news event
context produced by :mod:`analyzers.btc_news_intelligence`.  Used when the
deterministic keyword classifier (and optional LLM re-ranker) misreads a
headline — e.g. treating a diplomatic-delegation story as pure risk-off.

Lifecycle
---------
An override can expire in two complementary ways:

* **TTL-based** — straight clock expiry at ``expires_at_utc``.  Good for
  "override until the Fed presser ends".

* **Next-news-based** — when ``consume_on_next_event=True`` the override
  sticks until the news pipeline emits a new classification of at least
  the same severity.  At that moment:
    - the new classification *agrees* with the override → override is
      **confirmed** (boost the operator's trust score), cleared, and a
      record is persisted.
    - the new classification *disagrees* → override is **penalised**
      (reduce trust score), cleared, and the miss is logged for review.

Both modes coexist — the override is retired as soon as *either* trigger
fires.

Persistence
-----------
The active override and a small operator-trust map are persisted via
``data.database.db.save_learning_state`` under
``NewsLLM.OVERRIDE_STATE_KEY`` so the override survives restarts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.constants import NewsLLM, NewsOverrideDefaults

logger = logging.getLogger(__name__)


# Severity ranking used by next-news consumption.  A new classification must
# meet or exceed the override's severity rank to consume the override —
# otherwise a trivial BTC_TECHNICAL headline could clear a deliberate
# MACRO_RISK_OFF override.
_SEVERITY_RANK: Dict[str, int] = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "EXTREME": 3,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class NewsOverride:
    """A single active manual override."""

    event_type: str = "MACRO_RISK_OFF"
    direction: str = "NEUTRAL"                   # BULLISH / BEARISH / NEUTRAL
    confidence_mult: float = 1.0
    size_mult: float = 1.0

    # Block flags (operator can explicitly unblock by setting to False).
    block_longs: bool = False
    block_shorts: bool = False

    reason: str = ""
    set_by: str = "operator"
    set_at_utc: float = 0.0
    expires_at_utc: float = 0.0

    # When True, the first classification event with severity ≥ the
    # override's declared severity will retire the override (see
    # ``consume_if_applicable``).  When False, only the TTL clock applies.
    consume_on_next_event: bool = True

    # Declared severity used for next-news consumption comparisons.
    declared_severity: str = "MEDIUM"

    # Snapshot of the headline that prompted the override (free-form).
    source_headline: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NewsOverride":
        # Filter to known fields so legacy / newer payloads don't crash.
        known = {f for f in cls.__dataclass_fields__.keys()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    # ── Lifecycle predicates ──────────────────────────────────
    def is_expired(self, now: Optional[float] = None) -> bool:
        _now = now if now is not None else time.time()
        return self.expires_at_utc > 0 and _now >= self.expires_at_utc

    def severity_rank(self) -> int:
        return _SEVERITY_RANK.get((self.declared_severity or "MEDIUM").upper(), 1)


class NewsOverrideStore:
    """
    In-memory + DB-backed store for the single active override plus a trust
    map of ``set_by → score``.  Thread-safe for a single asyncio event loop
    via the surrounding code paths (callers are async).
    """

    def __init__(self) -> None:
        self._active: Optional[NewsOverride] = None
        self._trust: Dict[str, float] = {}
        # Small ring buffer of consumed overrides for observability.
        self._history: List[Dict[str, Any]] = []
        self._history_cap: int = 50
        self._loaded: bool = False

    # ── Persistence ───────────────────────────────────────────
    def _snapshot(self) -> Dict[str, Any]:
        return {
            "active": self._active.to_dict() if self._active else None,
            "trust": dict(self._trust),
            "history": list(self._history[-self._history_cap:]),
        }

    async def load(self) -> None:
        """Restore override + trust map from DB.  Safe to call multiple times."""
        if self._loaded:
            return
        try:
            from data.database import db
        except Exception as exc:  # pragma: no cover — tests stub DB
            logger.debug(f"NewsOverrideStore.load: DB unavailable ({exc})")
            self._loaded = True
            return
        try:
            payload = await db.load_learning_state(NewsLLM.OVERRIDE_STATE_KEY)
        except Exception as exc:
            logger.debug(f"NewsOverrideStore.load failed: {exc}")
            payload = None
        if isinstance(payload, dict):
            active = payload.get("active")
            if isinstance(active, dict):
                try:
                    self._active = NewsOverride.from_dict(active)
                    if self._active.is_expired():
                        self._active = None
                except Exception as exc:
                    logger.warning(f"Dropping malformed override on load: {exc}")
                    self._active = None
            trust = payload.get("trust")
            if isinstance(trust, dict):
                self._trust = {
                    str(k): float(v) for k, v in trust.items()
                    if isinstance(v, (int, float))
                }
            history = payload.get("history")
            if isinstance(history, list):
                self._history = [h for h in history if isinstance(h, dict)][-self._history_cap:]
        self._loaded = True

    async def _persist(self) -> None:
        try:
            from data.database import db
        except Exception as exc:  # pragma: no cover
            logger.debug(f"NewsOverrideStore._persist: DB unavailable ({exc})")
            return
        try:
            await db.save_learning_state(NewsLLM.OVERRIDE_STATE_KEY, self._snapshot())
        except Exception as exc:
            logger.debug(f"NewsOverrideStore._persist failed: {exc}")

    # ── Public API ────────────────────────────────────────────
    def get_active(self) -> Optional[NewsOverride]:
        if self._active is None:
            return None
        if self._active.is_expired():
            # Retire silently on read — persistence happens on next mutation
            # or explicit clear().  Callers see no override.
            return None
        return self._active

    def trust_score(self, user: str) -> float:
        return float(self._trust.get(user, 0.0))

    async def set_override(
        self,
        *,
        event_type: str,
        direction: str,
        confidence_mult: float = 1.0,
        size_mult: float = 1.0,
        reason: str = "",
        set_by: str = "operator",
        ttl_minutes: Optional[int] = None,
        consume_on_next_event: bool = True,
        declared_severity: str = "MEDIUM",
        source_headline: str = "",
        block_longs: bool = False,
        block_shorts: bool = False,
    ) -> NewsOverride:
        """
        Install a new override, replacing any prior one.  Bounds are clamped
        to :class:`NewsOverrideDefaults` limits.
        """
        await self.load()

        ttl = ttl_minutes if ttl_minutes is not None else NewsOverrideDefaults.DEFAULT_TTL_MINUTES
        ttl = int(_clamp(
            ttl,
            NewsOverrideDefaults.MIN_TTL_MINUTES,
            NewsOverrideDefaults.MAX_TTL_MINUTES,
        ))

        now = time.time()
        ov = NewsOverride(
            event_type=str(event_type).upper(),
            direction=str(direction).upper(),
            confidence_mult=_clamp(
                float(confidence_mult),
                NewsOverrideDefaults.MIN_CONF_MULT,
                NewsOverrideDefaults.MAX_CONF_MULT,
            ),
            size_mult=_clamp(
                float(size_mult),
                NewsOverrideDefaults.MIN_SIZE_MULT,
                NewsOverrideDefaults.MAX_SIZE_MULT,
            ),
            block_longs=bool(block_longs),
            block_shorts=bool(block_shorts),
            reason=str(reason or "")[:500],
            set_by=str(set_by or "operator")[:64],
            set_at_utc=now,
            expires_at_utc=now + ttl * 60,
            consume_on_next_event=bool(consume_on_next_event),
            declared_severity=(str(declared_severity) or "MEDIUM").upper(),
            source_headline=str(source_headline or "")[:240],
        )
        self._active = ov
        await self._persist()
        logger.info(
            f"📝 News override installed by {ov.set_by}: "
            f"{ov.event_type}/{ov.direction} conf×{ov.confidence_mult:.2f} "
            f"size×{ov.size_mult:.2f} ttl={ttl}m "
            f"consume_on_next={ov.consume_on_next_event}"
        )
        return ov

    async def clear(self, reason: str = "manual") -> Optional[NewsOverride]:
        await self.load()
        prev = self._active
        self._active = None
        if prev is not None:
            self._record_history(prev, outcome="cleared", note=reason)
            await self._persist()
            logger.info(f"📝 News override cleared ({reason}): {prev.event_type}/{prev.direction}")
        return prev

    async def consume_if_applicable(
        self,
        *,
        new_event_type: str,
        new_direction: str,
        new_severity: str,
        new_headline: str = "",
    ) -> Optional[Tuple[str, NewsOverride]]:
        """
        Called by the news pipeline when a new classification is about to
        activate.  If the currently-active override is in "consume on next
        event" mode and the incoming event meets the severity threshold,
        decide whether it *agrees* or *disagrees* with the override, apply
        the trust delta, clear the override, and return
        ``(outcome, prior_override)`` where outcome is "agree"/"disagree".

        Returns ``None`` when no action was taken (no active override, TTL
        mode only, or severity too low to consume).
        """
        await self.load()
        if self._active is None:
            return None
        if self._active.is_expired():
            prev = self._active
            self._active = None
            self._record_history(prev, outcome="expired", note="ttl")
            await self._persist()
            return None
        if not self._active.consume_on_next_event:
            return None

        new_rank = _SEVERITY_RANK.get((new_severity or "").upper(), 0)
        if new_rank < self._active.severity_rank():
            return None  # not a meaningful supersede event

        agrees = (
            str(new_event_type).upper() == self._active.event_type
            and str(new_direction).upper() == self._active.direction
        )
        outcome = "agree" if agrees else "disagree"
        delta = (
            NewsOverrideDefaults.TRUST_AGREE_BOOST
            if agrees
            else -NewsOverrideDefaults.TRUST_DISAGREE_PENALTY
        )
        user = self._active.set_by or "operator"
        self._trust[user] = _clamp(
            self._trust.get(user, 0.0) + delta,
            NewsOverrideDefaults.TRUST_SCORE_FLOOR,
            NewsOverrideDefaults.TRUST_SCORE_CEILING,
        )
        prev = self._active
        self._active = None
        self._record_history(
            prev,
            outcome=outcome,
            note=(new_headline or "")[:240],
            meta={"new_event_type": str(new_event_type), "new_direction": str(new_direction)},
        )
        await self._persist()
        logger.info(
            f"📝 Override consumed ({outcome}) by new event "
            f"{new_event_type}/{new_direction}: {user} trust={self._trust[user]:.2f}"
        )
        return outcome, prev

    def _record_history(
        self,
        override: NewsOverride,
        *,
        outcome: str,
        note: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "set_by": override.set_by,
            "event_type": override.event_type,
            "direction": override.direction,
            "set_at_utc": override.set_at_utc,
            "outcome": outcome,
            "note": note,
            "meta": dict(meta) if meta else {},
        }
        self._history.append(entry)
        self._history = self._history[-self._history_cap:]

    def status(self) -> Dict[str, Any]:
        ov = self.get_active()
        remaining = 0.0
        if ov is not None and ov.expires_at_utc > 0:
            remaining = max(0.0, (ov.expires_at_utc - time.time()) / 60.0)
        return {
            "active": ov.to_dict() if ov else None,
            "expires_in_minutes": round(remaining, 2),
            "trust": dict(self._trust),
            "history_tail": list(self._history[-5:]),
        }


# ── Module-level singleton ───────────────────────────────────
news_override_store = NewsOverrideStore()
