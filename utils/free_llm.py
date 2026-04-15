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
import aiohttp
import json
from typing import List, Optional

logger = logging.getLogger(__name__)

_BASE_URL       = "https://text.pollinations.ai/openai"
_HEADERS        = {"Content-Type": "application/json"}
_TIMEOUT        = aiohttp.ClientTimeout(total=45)
_RETRY_DELAY_S  = 2      # seconds between attempt 1 and attempt 2
_FALLBACK_MODEL = "openai"  # most stable Pollinations model

# Model assignments -- three different architectures for genuine debate diversity
MODEL_PROPOSER   = "mistral-large"   # Mistral Large -- structured reasoning
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
    """One raw HTTP call to Pollinations. Returns content string or None."""
    try:
        async with session.post(_BASE_URL, json=payload, headers=_HEADERS) as resp:
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
            if content is None:
                logger.warning(
                    "FreeLLM %s attempt %d: empty content (finish_reason=%s)",
                    model, attempt, choices[0].get("finish_reason", "?"),
                )
                return None
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

    try:
        # Attempt 1
        result = await _single_call(session, payload, model, attempt=1)
        if attempts_out is not None:
            attempts_out.append({"attempt": 1, "model": model, "succeeded": result is not None})
        if result is not None:
            return result

        # Attempt 2 -- retry after short pause
        await asyncio.sleep(_RETRY_DELAY_S)
        result = await _single_call(session, payload, model, attempt=2)
        if attempts_out is not None:
            attempts_out.append({"attempt": 2, "model": model, "succeeded": result is not None})
        if result is not None:
            return result

        # Attempt 3 -- fallback to most-stable model (skip if already using it)
        if model != _FALLBACK_MODEL:
            logger.warning(
                "FreeLLM %s failed after 2 attempts -- falling back to %s",
                model, _FALLBACK_MODEL,
            )
            fb_payload = {**payload, "model": _FALLBACK_MODEL}
            result = await _single_call(session, fb_payload, _FALLBACK_MODEL, attempt=3)
            if attempts_out is not None:
                attempts_out.append({"attempt": 3, "model": _FALLBACK_MODEL, "succeeded": result is not None})
            if result is not None:
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
