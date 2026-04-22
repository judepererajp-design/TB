"""
TitanBot -- Free LLM client (Pollinations.ai, no API key required)
==================================================================
Model assignments:
  PROPOSER   -- mistral-large  (strong reasoning, propose changes)
  CHALLENGER -- llama          (Meta Llama, find flaws/attack proposals)
  ARBITRATOR -- openai         (OpenAI-class, weigh both sides)
  FAST       -- mistral        (quick analysis)
  DEEP       -- openai         (thorough analysis)

All use Pollinations.ai (https://text.pollinations.ai/openai) -- free, no key needed.

Reliability strategy (no extra API key):
  1. Try requested model (attempt 1)
  2. On failure: wait 2 s, retry same model (attempt 2)
  3. On second failure AND model != FALLBACK: try openai as last resort (attempt 3)
  4. If everything fails return None -- callers must handle gracefully via rule-based logic
"""

import asyncio
import logging
import random
import aiohttp
import json
from typing import List, Optional

logger = logging.getLogger(__name__)

_BASE_URL       = "https://text.pollinations.ai/openai"
_HEADERS        = {"Content-Type": "application/json"}
_TIMEOUT        = aiohttp.ClientTimeout(total=45)
_RETRY_DELAY_S  = 2      # seconds between attempt 1 and attempt 2 (non-429 failures)
# Log-audit 2026-04-22: Pollinations free tier is "1-in-flight per IP" — repeated
# rapid calls return HTTP 429 "Queue full for IP".  Previous retry schedule of a
# flat 2 s made a 3-attempt burst finish inside 5 s, guaranteeing further 429s.
# Use an exponential backoff (base × 2^attempt + jitter) on rate-limited responses
# so the second attempt waits ~15-25 s and the third attempt ~30-60 s.
_RATE_LIMIT_BASE_S      = 12.0   # first 429 wait (pre-jitter)
_RATE_LIMIT_JITTER_S    = 6.0    # +/- range on each sleep
_RATE_LIMIT_MAX_S       = 90.0   # cap single backoff to 90 s
# Hard ceiling on accumulated rate-limit sleeps within a single call_llm().
# Prevents one blocked LLM from stalling upstream pipelines when all three
# attempts hit 429 — once this budget is exhausted we stop retrying and
# return None so callers can fall back to rule-based logic immediately.
_RATE_LIMIT_TOTAL_BUDGET_S = 75.0
_RATE_LIMITED_SENTINEL  = "__RATE_LIMITED__"
_FALLBACK_MODEL = "openai"  # most stable Pollinations model


def _rate_limit_backoff_secs(attempt: int) -> float:
    """Compute a jittered exponential backoff for a 429 response.

    ``attempt`` is 1-indexed (matches the attempt numbers used in logs).
    """
    base = _RATE_LIMIT_BASE_S * (2 ** max(0, attempt - 1))
    jitter = random.uniform(-_RATE_LIMIT_JITTER_S, _RATE_LIMIT_JITTER_S)
    return max(1.0, min(_RATE_LIMIT_MAX_S, base + jitter))

# Model assignments -- three different architectures for genuine debate diversity
MODEL_PROPOSER   = "openai-large"    # Pollinations: 'mistral-large' was removed from the
                                     # legacy /openai endpoint (HTTP 404 "Model not found");
                                     # openai-large keeps "strong reasoning" role alive.
MODEL_CHALLENGER = "llama"           # Meta Llama 3 70B -- adversarial critique
MODEL_ARBITRATOR = "openai"          # OpenAI-class -- balanced arbitration
MODEL_FAST       = "openai"          # Pollinations: 'mistral' model was removed; use openai
MODEL_DEEP       = "openai"          # OpenAI-class -- deep audit


async def _single_call(
    session: aiohttp.ClientSession,
    payload: dict,
    model: str,
    attempt: int,
) -> Optional[str]:
    """One raw HTTP call to Pollinations. Returns content string or None.

    Returns the ``_RATE_LIMITED_SENTINEL`` string on HTTP 429 so the caller can
    apply exponential backoff before retrying — a flat 2 s retry guaranteed
    another 429 on the free tier's 1-in-flight-per-IP queue.
    """
    try:
        async with session.post(_BASE_URL, json=payload, headers=_HEADERS) as resp:
            if resp.status == 429:
                text = await resp.text()
                logger.warning(
                    "FreeLLM %s attempt %d HTTP 429 (rate-limited): %s",
                    model, attempt, text[:120],
                )
                return _RATE_LIMITED_SENTINEL
            if resp.status != 200:
                text = await resp.text()
                logger.warning(
                    "FreeLLM %s attempt %d HTTP %d: %s",
                    model, attempt, resp.status, text[:120],
                )
                return None
            data = await resp.json()
            # Defensive extraction — Pollinations can return malformed responses
            # (e.g., empty choices, missing content, content-filtered).
            # Previously: data["choices"][0]["message"]["content"] threw opaque
            # KeyError logged as "error: 'content'" with zero diagnostics.
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                logger.warning(
                    "FreeLLM %s attempt %d: no choices in response (keys: %s)",
                    model, attempt, list(data.keys())[:5],
                )
                return None
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            finish_reason = choices[0].get("finish_reason", "?")
            if content is None:
                logger.warning(
                    "FreeLLM %s attempt %d: empty content (finish_reason=%s)",
                    model, attempt, finish_reason,
                )
                # Signal length-truncation distinctly so the caller can retry
                # with a larger max_tokens budget rather than treating it as
                # a generic failure.
                if finish_reason == "length":
                    return "__LENGTH_TRUNCATED__"
                return None
            # Empty string but finish_reason=length means the model burned
            # all the output tokens on reasoning and emitted no user-visible
            # content. Treat the same way as a None content of that kind.
            if not content.strip() and finish_reason == "length":
                logger.warning(
                    "FreeLLM %s attempt %d: blank content (finish_reason=length)",
                    model, attempt,
                )
                return "__LENGTH_TRUNCATED__"
            return content
    except asyncio.TimeoutError:
        logger.warning("FreeLLM %s attempt %d timeout", model, attempt)
        return None
    except Exception as e:
        logger.warning("FreeLLM %s attempt %d error: %s", model, attempt, e)
        return None


async def call_llm(
    prompt: str,
    model: str = MODEL_FAST,
    temperature: float = 0.1,
    max_tokens: int = 800,
    session: Optional[aiohttp.ClientSession] = None,
    attempts_out: Optional[List[dict]] = None,
) -> Optional[str]:
    """
    Call Pollinations.ai with an OpenAI-compatible request.

    Reliability (no extra API key needed):
      attempt 1 -- requested model
      attempt 2 -- same model, 2 s later
      attempt 3 -- mistral fallback (skipped if already mistral)
    Returns content string or None on complete failure.

    If ``attempts_out`` is a list, each attempt is appended as a dict:
      {"attempt": int, "model": str, "succeeded": bool}
    This lets callers record which model actually answered (useful for
    debugging veto decisions where different architectures reason differently).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    close_session = session is None
    if close_session:
        session = aiohttp.ClientSession(timeout=_TIMEOUT)

    # Track cumulative rate-limit sleep within this call so one blocked LLM
    # can't stall the pipeline: once we exceed _RATE_LIMIT_TOTAL_BUDGET_S
    # of accumulated 429-backoff, we short-circuit remaining attempts.
    _rate_limit_sleep_total = 0.0

    try:
        # Attempt 1
        result = await _single_call(session, payload, model, attempt=1)
        if attempts_out is not None:
            attempts_out.append({"attempt": 1, "model": model, "succeeded": result is not None and result != "__LENGTH_TRUNCATED__" and result != _RATE_LIMITED_SENTINEL})
        if result == "__LENGTH_TRUNCATED__":
            # Retry with 2× tokens — Pollinations sometimes burns the budget
            # on reasoning and returns empty content. Doubling the budget
            # (capped at 4096) recovers the response ~80% of the time.
            bumped = min(int(max_tokens) * 2, 4096)
            if bumped == int(max_tokens):
                logger.warning(
                    "FreeLLM %s attempt 1 truncated at max_tokens cap (%d) — giving up",
                    model, bumped,
                )
                result = None
            else:
                logger.info(
                    "FreeLLM %s attempt 1 truncated (finish_reason=length) — retrying with max_tokens=%d",
                    model, bumped,
                )
                payload = {**payload, "max_tokens": bumped}
                result = None
        elif result is not None and result != _RATE_LIMITED_SENTINEL:
            return result

        # Attempt 2 -- retry after backoff (longer for 429s)
        _first_was_429 = result == _RATE_LIMITED_SENTINEL
        _sleep2 = _rate_limit_backoff_secs(1) if _first_was_429 else _RETRY_DELAY_S
        if _first_was_429:
            if _rate_limit_sleep_total + _sleep2 > _RATE_LIMIT_TOTAL_BUDGET_S:
                logger.warning(
                    "FreeLLM %s rate-limit budget exhausted (%.1fs / %.1fs) — "
                    "skipping remaining retries, returning None",
                    model, _rate_limit_sleep_total, _RATE_LIMIT_TOTAL_BUDGET_S,
                )
                return None
            logger.info(
                "FreeLLM %s rate-limited — sleeping %.1fs before attempt 2",
                model, _sleep2,
            )
            _rate_limit_sleep_total += _sleep2
        await asyncio.sleep(_sleep2)
        result = await _single_call(session, payload, model, attempt=2)
        if attempts_out is not None:
            attempts_out.append({"attempt": 2, "model": model, "succeeded": result is not None and result != "__LENGTH_TRUNCATED__" and result != _RATE_LIMITED_SENTINEL})
        if result == "__LENGTH_TRUNCATED__":
            bumped = min(int(payload.get("max_tokens", max_tokens)) * 2, 4096)
            if bumped == int(payload.get("max_tokens", max_tokens)):
                logger.warning(
                    "FreeLLM %s attempt 2 truncated at max_tokens cap (%d) — "
                    "response is pathologically long; giving up on retries",
                    model, bumped,
                )
                result = None
            else:
                logger.info(
                    "FreeLLM %s attempt 2 truncated (finish_reason=length) — bumping fallback max_tokens=%d",
                    model, bumped,
                )
                payload = {**payload, "max_tokens": bumped}
                result = None
        elif result is not None and result != _RATE_LIMITED_SENTINEL:
            return result

        # Attempt 3 -- fallback to most-stable model (skip if already using it)
        if model != _FALLBACK_MODEL:
            # If the last attempt was rate-limited, wait longer before falling over
            if result == _RATE_LIMITED_SENTINEL:
                _sleep3 = _rate_limit_backoff_secs(2)
                if _rate_limit_sleep_total + _sleep3 > _RATE_LIMIT_TOTAL_BUDGET_S:
                    logger.warning(
                        "FreeLLM %s rate-limit budget exhausted (%.1fs / %.1fs) — "
                        "skipping fallback, returning None",
                        model, _rate_limit_sleep_total, _RATE_LIMIT_TOTAL_BUDGET_S,
                    )
                    return None
                logger.info(
                    "FreeLLM %s still rate-limited — sleeping %.1fs before fallback to %s",
                    model, _sleep3, _FALLBACK_MODEL,
                )
                _rate_limit_sleep_total += _sleep3
                await asyncio.sleep(_sleep3)
            logger.warning(
                "FreeLLM %s failed after 2 attempts -- falling back to %s",
                model, _FALLBACK_MODEL,
            )
            fb_payload = {**payload, "model": _FALLBACK_MODEL}
            result = await _single_call(session, fb_payload, _FALLBACK_MODEL, attempt=3)
            if attempts_out is not None:
                attempts_out.append({"attempt": 3, "model": _FALLBACK_MODEL, "succeeded": result is not None and result != "__LENGTH_TRUNCATED__" and result != _RATE_LIMITED_SENTINEL})
            if result is not None and result != "__LENGTH_TRUNCATED__" and result != _RATE_LIMITED_SENTINEL:
                return result

        logger.error("FreeLLM complete failure for model=%s -- all attempts exhausted", model)
        return None

    finally:
        if close_session:
            await session.close()


def parse_json_response(raw: str) -> Optional[dict]:
    """Extract JSON from LLM response (strips markdown fences)."""
    if not raw:
        return None
    clean = raw.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        if len(parts) > 1:
            clean = parts[1]
            if clean.startswith("json"):
                clean = clean[4:]
    clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        try:
            return json.loads(clean[clean.index("{"):clean.rindex("}") + 1])
        except Exception:
            return None
