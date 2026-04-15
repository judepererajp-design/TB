"""
TitanBot Pro — AI Analyst (Multi-Model via OpenRouter)
=======================================================
V2.09 CHANGES:
  - Dedicated ai_analyst.log: every prompt sent and response received is written
    to logs/ai_analyst.log separately from titanbot.log.

  - Smart model routing (two tiers):
      FAST tier  (meta-llama/llama-3.3-70b-instruct:free) — dead signal batches,
                  market state, news sentiment. ~24 calls/day.
      DEEP tier  (nvidia/nemotron-3-super-120b-a12b:free) — structural audit only.
                  1 call/day. 262K context window.

  - Signal narrative generation REMOVED. Was burning ~20 calls/day on cosmetic
    text with zero edge. Keeps bot under 50 calls/day on free tier.

  - New structural_audit() function — sends AI the data needed to catch bugs:
      * Per-strategy direction breakdown (SHORT% vs LONG%)
      * HTF guardrail block counts by direction
      * Whale flow vs signal direction alignment
      * Regime vs signal direction conflicts

  - Dead signal batch interval extended 30min → 2h. Saves ~36 calls/day.

Call budget: ~25-30/day default (free = 50 limit). Add $10 to OpenRouter → 1000/day.

Functions:
  1. evaluate_market_state()   — go/no-go + confidence multiplier (fast tier)
  2. score_news_sentiment()    — per-symbol news sentiment (fast tier)
  3. analyse_dead_signals()    — why signals die, 2h batch (fast tier)
  4. structural_audit()        — NEW: direction bias bug detection (deep tier, 1/day)
  5. explain_signal_death()    — /ai why SYMBOL answer (fast tier)
  6. review_narrative()        — self-critique, kept for compat (fast tier)
"""

import asyncio
import json
import logging
import logging.handlers
import os
import time
from collections import Counter, deque
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from utils.free_llm import call_llm as _free_llm_call, MODEL_FAST as _FREE_MODEL_FAST, MODEL_DEEP as _FREE_MODEL_DEEP

logger = logging.getLogger(__name__)

# ── Model routing (Pollinations.ai — free, no key needed) ───
_MODEL_FAST    = _FREE_MODEL_FAST   # mistral — fast analysis
_MODEL_BATCH   = _FREE_MODEL_FAST   # mistral — batch audits
_MODEL_ANALYST = _FREE_MODEL_FAST   # mistral — scan notebook
_MODEL_DEEP    = _FREE_MODEL_DEEP   # openai-class — deep audit
_TEMPERATURE   = 0.1
_MAX_TOKENS    = 1000


# ── Dedicated AI log ──────────────────────────────────────────
def _build_ai_logger() -> logging.Logger:
    ai_log = logging.getLogger("titanbot.ai_analyst")
    if ai_log.handlers:
        return ai_log
    os.makedirs("logs", exist_ok=True)
    h = logging.handlers.RotatingFileHandler(
        "logs/ai_analyst.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    ai_log.addHandler(h)
    ai_log.setLevel(logging.DEBUG)
    ai_log.propagate = False
    return ai_log


_ai_log = _build_ai_logger()


# ── Data classes ──────────────────────────────────────────────

@dataclass
class MarketStateResult:
    market_quality: float = 50.0
    confidence_multiplier: float = 1.0
    hard_block: bool = False
    block_reason: Optional[str] = None
    favored_direction: str = "BOTH"
    session_note: str = ""


@dataclass
class SentimentResult:
    sentiment_score: float = 50.0
    confidence: float = 50.0
    direction_bias: str = "NEUTRAL"
    urgency: str = "LOW"
    key_reason: str = ""
    price_impact_window: str = "hours"
    no_trade: bool = False


@dataclass
class DeadSignalAnalysis:
    dominant_kill_reason: str = ""
    regime_mismatch: bool = False
    strategy_to_suppress: Optional[str] = None
    rr_issue: bool = False
    recommendation: str = ""


@dataclass
class StructuralAuditResult:
    biased_strategies: List[str] = field(default_factory=list)
    htf_asymmetry: bool = False
    whale_signal_conflict: bool = False
    regime_signal_conflict: bool = False
    root_causes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    severity: str = "LOW"


class AIAnalyst:

    # ── AI Safety constants ───────────────────────────────────
    # Hard limits that cannot be overridden by config — prevents runaway
    # API spend and prompt-injection attacks from market data.
    _HARD_DAILY_CALL_LIMIT   = 120     # Absolute ceiling regardless of paid tier
    _HARD_MAX_PROMPT_CHARS   = 12_000  # Truncate prompts exceeding this length
    _HARD_MAX_RESPONSE_CHARS = 8_000   # Discard responses suspiciously long
    _HARD_CALL_TIMEOUT_SECS  = 30      # Per-call timeout (prevents hung coroutines; 30s for slow free models)

    # Prompt injection patterns — if any appear in AI inputs from market data,
    # the relevant field is stripped before sending to the model.
    # FIX: Bare "root" caused false positives on market data containing "root",
    # "rooted", etc. Replaced with specific attack-vector phrases.
    # "sudo" and "admin" (both kept) already cover privilege-escalation prompts.
    _INJECTION_PATTERNS = (
        "ignore previous", "ignore all", "disregard", "new instructions",
        "system:", "assistant:", "forget", "jailbreak", "act as",
        "you are now", "override", "bypass", "sudo", "admin",
        "as root", "run as root", "root access", "root privilege",
        "execute", "eval(", "import os", "__import__",
    )

    def __init__(self):
        self._mode = "full"
        self._api_key = None  # set to True by initialize(); used as init sentinel

        self._calls_fast_today = 0
        self._calls_deep_today = 0
        self._total_latency = 0.0
        self._call_count_for_avg = 0
        self._day_reset = time.time()

        # AI Safety: budget circuit breaker state
        self._budget_cb_tripped  = False
        self._budget_cb_reason   = ""
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5  # Trip CB after 5 consecutive API failures

        self._market_state_cache: Optional[MarketStateResult] = None
        self._market_state_ts: float = 0.0
        self._market_state_ttl: float = 90.0

        self._dead_signal_buffer: List[dict] = []
        self._last_dead_analysis: float = 0.0
        self._dead_analysis_interval: float = 7200.0  # 2h

        # Session stats for structural audit
        self._session_stats: dict = {
            "strategy_direction": {},
            "htf_blocked_long": 0,
            "htf_blocked_short": 0,
            "whale_buy_events": 0,
            "whale_sell_events": 0,
            "whale_buy_usd": 0.0,
            "whale_sell_usd": 0.0,
            "signals_vs_whale_conflict": 0,
            "regime_at_signal": {},
            "total_raw_signals": 0,
        }
        self._last_structural_audit: float = 0.0
        self._structural_audit_interval: float = 7200.0
        self._last_deep_audit: float = 0.0
        self._deep_audit_interval: float = 86400.0

        # Structured decision history: last 50 AI decisions with outcome
        # Each entry: {ts, label, model, latency_ms, outcome, summary}
        self._decision_history: deque = deque(maxlen=50)

    # ── Init ──────────────────────────────────────────────────

    def initialize(self, api_key: str = ""):
        self._api_key = True  # sentinel: marks as initialised (no real key needed)
        logger.info(
            f"✅ AI Analyst initialised — fast={_MODEL_FAST} | deep={_MODEL_DEEP} "
            f"| mode={self._mode} | ai log: logs/ai_analyst.log (Pollinations.ai, no key)"
        )
        _ai_log.info(f"=== AI Analyst initialised | fast={_MODEL_FAST} | deep={_MODEL_DEEP} ===")

    def set_mode(self, mode: str):
        valid = ("full", "off", "diagnosis_only")
        if mode not in valid:
            logger.warning(f"Invalid AI mode '{mode}'. Valid: {valid}")
            return
        self._mode = mode
        logger.info(f"🤖 AI mode: {mode}")

    def get_status(self) -> dict:
        now = time.time()
        if now - self._day_reset > 86400:
            self._calls_fast_today = 0
            self._calls_deep_today = 0
            self._day_reset = now
        avg_lat = self._total_latency / self._call_count_for_avg if self._call_count_for_avg else 0.0
        return {
            "mode": self._mode,
            "calls_fast_today": self._calls_fast_today,
            "calls_deep_today": self._calls_deep_today,
            "calls_total_today": self._calls_fast_today + self._calls_deep_today,
            "avg_latency_ms": round(avg_lat * 1000),
            "model_fast": _MODEL_FAST,
            "model_batch": _MODEL_BATCH,
            "model_analyst": _MODEL_ANALYST,
            "model_deep": _MODEL_DEEP,
            "initialized": self._api_key is not None,
        }

    def get_budget_status(self) -> dict:
        """
        Returns current AI budget and safety state for Telegram /ai status and dashboard.
        Surfaces if the budget circuit breaker is tripped and why.
        """
        now = time.time()
        if now - self._day_reset > 86400:
            self._calls_fast_today = 0
            self._calls_deep_today = 0
            self._day_reset = now
        avg_lat = self._total_latency / self._call_count_for_avg if self._call_count_for_avg else 0.0
        return {
            "mode":               self._mode,
            "calls_fast_today":   self._calls_fast_today,
            "calls_deep_today":   self._calls_deep_today,
            "calls_total_today":  self._calls_fast_today + self._calls_deep_today,
            "hard_daily_limit":   self._HARD_DAILY_CALL_LIMIT,
            "calls_remaining":    max(0, self._HARD_DAILY_CALL_LIMIT
                                      - self._calls_fast_today - self._calls_deep_today),
            "budget_cb_tripped":  self._budget_cb_tripped,
            "budget_cb_reason":   self._budget_cb_reason,
            "consecutive_errors": self._consecutive_errors,
            "avg_latency_ms":     round(avg_lat * 1000),
            "model_fast":         _MODEL_FAST,
            "model_deep":         _MODEL_DEEP,
            "initialized":        self._api_key is not None,
        }

    def reset_budget_cb(self):
        """Manually clear the budget circuit breaker (via /ai reset Telegram command)."""
        self._budget_cb_tripped  = False
        self._budget_cb_reason   = ""
        self._consecutive_errors = 0
        logger.warning("AI budget circuit breaker manually reset")

    def get_audit_status(self) -> dict:
        """Returns current audit state for Telegram /audit status display."""
        now = time.time()
        fast_age = now - self._last_structural_audit
        deep_age = now - self._last_deep_audit
        fast_next = max(0, self._structural_audit_interval - fast_age)
        deep_next = max(0, self._deep_audit_interval - deep_age)
        return {
            "fast_last_ran": int(fast_age),             # seconds ago (0 = never)
            "fast_next_in": int(fast_next),             # seconds until next auto run
            "deep_last_ran": int(deep_age),             # seconds ago
            "deep_next_in": int(deep_next),             # seconds until next deep allowed
            "session_stats": dict(self._session_stats), # current accumulation
        }

    def get_decision_history(self) -> list:
        """Return last 50 AI decisions as a list of dicts (most recent first)."""
        return list(reversed(self._decision_history))

    # ── Session stat recorders (called from engine.py) ────────

    def record_strategy_signal(self, strategy: str, direction: str):
        s = self._session_stats["strategy_direction"]
        if strategy not in s:
            s[strategy] = {"LONG": 0, "SHORT": 0}
        s[strategy][direction] = s[strategy].get(direction, 0) + 1
        self._session_stats["total_raw_signals"] += 1

    def record_htf_block(self, direction: str):
        if direction == "LONG":
            self._session_stats["htf_blocked_long"] += 1
        else:
            self._session_stats["htf_blocked_short"] += 1

    def record_whale_event(self, side: str, usd: float):
        if side == "buy":
            self._session_stats["whale_buy_events"] += 1
            self._session_stats["whale_buy_usd"] += usd
        else:
            self._session_stats["whale_sell_events"] += 1
            self._session_stats["whale_sell_usd"] += usd

    def record_signal_vs_whale_conflict(self):
        self._session_stats["signals_vs_whale_conflict"] += 1

    def record_regime_signal(self, regime: str, direction: str):
        r = self._session_stats["regime_at_signal"]
        if regime not in r:
            r[regime] = {"LONG": 0, "SHORT": 0}
        r[regime][direction] = r[regime].get(direction, 0) + 1

    # ── Core call ─────────────────────────────────────────────

    async def _call(self, prompt: str, context: str = "", model: str = None, call_label: str = "unnamed") -> Optional[str]:
        if self._api_key is None:
            logger.warning("AI Analyst not initialised")
            return None

        # ── Safety gate 1: budget circuit breaker ─────────────
        if self._budget_cb_tripped:
            logger.warning(f"AI call blocked — budget CB active: {self._budget_cb_reason}")
            return None

        now = time.time()
        if now - self._day_reset > 86400:
            self._calls_fast_today  = 0
            self._calls_deep_today  = 0
            self._day_reset         = now
            self._consecutive_errors = 0
            if self._budget_cb_tripped and "daily" in self._budget_cb_reason:
                self._budget_cb_tripped = False
                self._budget_cb_reason  = ""
                logger.info("AI budget CB cleared on daily reset")

        total_today = self._calls_fast_today + self._calls_deep_today
        if total_today >= self._HARD_DAILY_CALL_LIMIT:
            reason = f"Hard daily limit ({self._HARD_DAILY_CALL_LIMIT}) reached"
            self._budget_cb_tripped = True
            self._budget_cb_reason  = reason
            logger.error(f"🛑 AI budget CB tripped: {reason}")
            return None

        # ── Safety gate 2: prompt injection sanitisation ──────
        # Market data (symbol names, news headlines, confluence notes) flows into
        # prompts. A maliciously crafted symbol name or news headline could attempt
        # prompt injection. Strip known patterns before sending to the model.
        prompt  = self._sanitise_input(prompt)
        context = self._sanitise_input(context)

        # ── Safety gate 3: prompt length cap ─────────────────
        if len(prompt) > self._HARD_MAX_PROMPT_CHARS:
            prompt = prompt[:self._HARD_MAX_PROMPT_CHARS] + "\n[TRUNCATED — prompt exceeded safe length]"
            logger.warning(f"AI prompt truncated for {call_label} ({len(prompt)} chars → {self._HARD_MAX_PROMPT_CHARS})")

        # Prepend system context to prompt if provided (Pollinations uses single user turn)
        if context:
            full_prompt = context + "\n\n" + prompt
        else:
            full_prompt = prompt

        use_model = model or _MODEL_FAST
        t0 = time.time()
        _ai_log.debug(
            f"[CALL:{call_label}] model={use_model}\n--- PROMPT ---\n"
            f"{full_prompt}\n--- END PROMPT ---"
        )
        try:
            # ── Safety gate 4: per-call timeout ───────────────
            raw = await asyncio.wait_for(
                _free_llm_call(
                    full_prompt,
                    model=use_model,
                    temperature=_TEMPERATURE,
                    max_tokens=_MAX_TOKENS,
                ),
                timeout=self._HARD_CALL_TIMEOUT_SECS,
            )

            elapsed = time.time() - t0
            self._total_latency      += elapsed
            self._call_count_for_avg += 1
            self._consecutive_errors  = 0  # reset on success

            if use_model == _MODEL_DEEP:
                self._calls_deep_today += 1
            else:
                self._calls_fast_today += 1

            if raw is None:
                return None

            raw = raw or ""

            # ── Safety gate 5: response length / sanity check ─
            if len(raw) > self._HARD_MAX_RESPONSE_CHARS:
                logger.warning(
                    f"AI response suspiciously long for {call_label} "
                    f"({len(raw)} chars) — truncating to {self._HARD_MAX_RESPONSE_CHARS}"
                )
                raw = raw[:self._HARD_MAX_RESPONSE_CHARS]

            _ai_log.debug(f"[RESP:{call_label}] latency={elapsed*1000:.0f}ms\n--- RESPONSE ---\n{raw}\n--- END RESPONSE ---")
            # Record to structured decision history
            self._decision_history.append({
                "ts": time.time(),
                "label": call_label,
                "model": use_model,
                "latency_ms": int(elapsed * 1000),
                "outcome": "ok",
                "summary": (raw or "")[:120],
            })
            return raw

        except asyncio.TimeoutError:
            self._consecutive_errors += 1
            logger.error(f"AI call timeout ({self._HARD_CALL_TIMEOUT_SECS}s) for {call_label}")
            self._check_error_cb(call_label)
            self._decision_history.append({
                "ts": time.time(),
                "label": call_label,
                "model": use_model,
                "latency_ms": int((time.time() - t0) * 1000),
                "outcome": "timeout",
                "summary": f"Timed out after {self._HARD_CALL_TIMEOUT_SECS}s",
            })
            return None
        except Exception as e:
            self._consecutive_errors += 1
            _ai_log.error(f"[ERROR:{call_label}] {e}")
            logger.error(f"AI Analyst API error ({call_label}): {e}")
            self._check_error_cb(call_label)
            self._decision_history.append({
                "ts": time.time(),
                "label": call_label,
                "model": use_model,
                "latency_ms": int((time.time() - t0) * 1000),
                "outcome": "error",
                "summary": str(e)[:120],
            })
            return None

    def _sanitise_input(self, text: str) -> str:
        """
        Strip prompt injection patterns from market-data-derived text.
        Patterns are lowercased for matching but original casing is preserved
        for legitimate content — only the matching segments are removed.
        """
        if not text:
            return text
        lower = text.lower()
        for pattern in self._INJECTION_PATTERNS:
            if pattern in lower:
                # Find and redact the offending line/segment
                lines = text.split("\n")
                cleaned = []
                for line in lines:
                    if pattern in line.lower():
                        cleaned.append(f"[REDACTED — injection pattern: '{pattern}']")
                        _ai_log.warning(f"Prompt injection pattern '{pattern}' detected and redacted")
                    else:
                        cleaned.append(line)
                text = "\n".join(cleaned)
                lower = text.lower()  # recompute after edit
        return text

    def _check_error_cb(self, call_label: str):
        """Trip the budget circuit breaker after too many consecutive errors."""
        if self._consecutive_errors >= self._max_consecutive_errors:
            reason = f"{self._consecutive_errors} consecutive API errors (last: {call_label})"
            self._budget_cb_tripped = True
            self._budget_cb_reason  = reason
            logger.error(f"🛑 AI budget CB tripped (error threshold): {reason}")

    def _parse_json(self, raw: str) -> Optional[dict]:
        if not raw:
            return None
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                parts = clean.split("```")
                if len(parts) > 1:
                    clean = parts[1]
                    if clean.startswith("json"):
                        clean = clean[4:]
            result = json.loads(clean.strip())
        except json.JSONDecodeError:
            try:
                start, end = raw.index("{"), raw.rindex("}") + 1
                result = json.loads(raw[start:end])
            except Exception:
                logger.debug(f"AI JSON parse failed: {raw[:200]}")
                return None

        # AI Safety: validate that the parsed result is a dict (not a list/scalar)
        # and that no value is a callable or contains executable code — hallucinated
        # fields with "import", "exec", "eval" are stripped before use.
        if not isinstance(result, dict):
            logger.warning(f"AI returned non-dict JSON ({type(result).__name__}) — discarding")
            return None

        safe = {}
        for k, v in result.items():
            str_v = str(v).lower()
            if any(p in str_v for p in ("import ", "exec(", "eval(", "__import__", "os.system")):
                logger.warning(f"AI response field '{k}' contains suspicious content — stripped")
                continue
            safe[k] = v
        return safe if safe else None

    # ── 1: Market State ───────────────────────────────────────

    async def evaluate_market_state(self, context: dict) -> MarketStateResult:
        if self._mode == "off":
            return MarketStateResult()
        if self._market_state_cache and (time.time() - self._market_state_ts) < self._market_state_ttl:
            return self._market_state_cache
        # v2.10: Don't call AI in first 90s of startup — let the bot stabilize
        # and avoid hitting provider rate limits on cold start.
        if not self._market_state_cache and (time.time() - self._day_reset < 90):
            return MarketStateResult()

        prompt = f"""You are the chief market analyst for a crypto futures trading bot.
Market data:
- Regime: {context.get('regime','UNKNOWN')} | Chop: {context.get('chop',0):.2f} | ADX: {context.get('adx',0):.1f}
- Fear & Greed: {context.get('fear_greed',50)}/100
- BTC: ${context.get('btc_price',0):,.2f} | 24h: {context.get('btc_change',0):+.2f}%
- BTC dominance: {context.get('btc_dominance',0):.1f}%
- Market cap 24h change: {context.get('market_cap_change',0):+.2f}%
- Liquidations 1h: ${context.get('liquidations_usd',0):,.0f}
- Stablecoin supply 24h: {context.get('stablecoin_change',0):+.2f}%
- Events next 24h: {json.dumps(context.get('events',[]))}
- Recent news: {json.dumps(context.get('news',[])[:5])}
- Whale events 1h: {context.get('whale_count',0)} (${context.get('whale_volume_usd',0):,.0f})

Return ONLY valid JSON:
{{"market_quality":<0-100>,"confidence_multiplier":<0.70-1.30>,"hard_block":<bool>,"block_reason":"<str|null>","favored_direction":"<LONG|SHORT|BOTH|NEITHER>","session_note":"<120 chars max>"}}"""

        data = self._parse_json(await self._call(prompt, model=_MODEL_BATCH, call_label="market_state"))
        if not data:
            return MarketStateResult()

        result = MarketStateResult(
            market_quality=float(data.get("market_quality", 50)),
            confidence_multiplier=max(0.70, min(1.30, float(data.get("confidence_multiplier", 1.0)))),
            hard_block=bool(data.get("hard_block", False)),
            block_reason=data.get("block_reason"),
            favored_direction=data.get("favored_direction", "BOTH"),
            session_note=data.get("session_note", ""),
        )
        self._market_state_cache = result
        self._market_state_ts = time.time()
        if result.session_note:
            logger.info(f"🤖 Market state: q={result.market_quality:.0f} mult={result.confidence_multiplier:.2f} dir={result.favored_direction} | {result.session_note}")
        if result.hard_block:
            logger.info(f"🤖 AI HARD BLOCK: {result.block_reason}")
        return result

    # ── 2: News Sentiment ─────────────────────────────────────

    async def score_news_sentiment(self, symbol: str, headlines: List[dict], signal_direction: str = "NEUTRAL") -> Optional[SentimentResult]:
        if self._mode == "off" or not headlines:
            return None
        formatted = [f"[{h.get('source','?')}] {h.get('title','')}" for h in headlines[:10]]
        prompt = f"""Analyse crypto news for {symbol}. Signal direction: {signal_direction}
Headlines: {chr(10).join(formatted)}
Return ONLY valid JSON:
{{"sentiment_score":<0-100>,"confidence":<0-100>,"direction_bias":"<BULLISH|BEARISH|NEUTRAL>","urgency":"<HIGH|MEDIUM|LOW>","key_reason":"<100 chars>","price_impact_window":"<minutes|hours|days>","no_trade":<bool>}}"""
        data = self._parse_json(await self._call(prompt, call_label=f"news_{symbol}"))
        if not data:
            return None
        return SentimentResult(
            sentiment_score=float(data.get("sentiment_score", 50)),
            confidence=float(data.get("confidence", 50)),
            direction_bias=data.get("direction_bias", "NEUTRAL"),
            urgency=data.get("urgency", "LOW"),
            key_reason=data.get("key_reason", ""),
            price_impact_window=data.get("price_impact_window", "hours"),
            no_trade=bool(data.get("no_trade", False)),
        )

    # ── 3: Dead Signal Analysis ───────────────────────────────

    def buffer_dead_signal(self, symbol: str, direction: str, strategy: str,
                            kill_reason: str, rr: float, confidence: float, regime: str):
        self._dead_signal_buffer.append({
            "symbol": symbol, "direction": direction, "strategy": strategy,
            "kill_reason": kill_reason, "rr": rr, "confidence": confidence,
            "regime": regime, "ts": time.time(),
        })
        if len(self._dead_signal_buffer) > 500:
            self._dead_signal_buffer = self._dead_signal_buffer[-500:]

    async def analyse_dead_signals(self, regime: str) -> Optional[DeadSignalAnalysis]:
        if self._mode == "off" or not self._dead_signal_buffer:
            return None
        if time.time() - self._last_dead_analysis < self._dead_analysis_interval:
            return None

        kill_counts = Counter(s["kill_reason"] for s in self._dead_signal_buffer)
        strat_counts = Counter(s["strategy"] for s in self._dead_signal_buffer)
        avg_rr = sum(s["rr"] for s in self._dead_signal_buffer) / len(self._dead_signal_buffer)
        avg_conf = sum(s["confidence"] for s in self._dead_signal_buffer) / len(self._dead_signal_buffer)
        total = len(self._dead_signal_buffer)

        # V2.09: Per-strategy direction breakdown so the AI can spot structural bias
        strat_dir: Dict[str, Dict[str, int]] = {}
        for s in self._dead_signal_buffer:
            if s["strategy"] not in strat_dir:
                strat_dir[s["strategy"]] = {"LONG": 0, "SHORT": 0}
            strat_dir[s["strategy"]][s["direction"]] = strat_dir[s["strategy"]].get(s["direction"], 0) + 1

        biased = {
            k: v for k, v in strat_dir.items()
            if v["SHORT"] + v["LONG"] >= 5 and (
                v["SHORT"] / max(1, v["SHORT"] + v["LONG"]) > 0.85 or
                v["LONG"] / max(1, v["SHORT"] + v["LONG"]) > 0.85
            )
        }

        ss = self._session_stats

        # Count ensemble suppressions
        ensemble_kills = kill_counts.get("ENSEMBLE_SUPPRESS", 0)
        ensemble_pct   = ensemble_kills / max(total, 1) * 100

        prompt = f"""Analysing why a crypto bot's signals fail quality filters.
Period: 2h | Regime: {regime} | Total dead: {total}
Kill reasons: {json.dumps(dict(kill_counts.most_common(10)))}
Strategy fires: {json.dumps(dict(strat_counts.most_common(10)))}
Per-strategy direction (LONG/SHORT counts): {json.dumps(strat_dir)}
Strategies with >85% single-direction bias: {json.dumps(biased)}
Avg R:R: {avg_rr:.2f} | Avg confidence: {avg_conf:.1f}
HTF blocks: LONG={ss['htf_blocked_long']} SHORT={ss['htf_blocked_short']}
Whale flow: BUY={ss['whale_buy_events']}(${ss['whale_buy_usd']/1000:.0f}k) SELL={ss['whale_sell_events']}(${ss['whale_sell_usd']/1000:.0f}k)
Signals vs whale flow conflict: {ss['signals_vs_whale_conflict']}
Ensemble voter suppressions: {ensemble_kills}/{total} ({ensemble_pct:.0f}%) — multi-source voting killed these

Analysis rules:
- If whale_buy_usd > whale_sell_usd × 3 and LONG signals are being killed, note whale tailwind is wasted
- If whale_sell_usd > whale_buy_usd × 3 and SHORT signals are killed, note whale tailwind is wasted
- If HTF blocks one direction 3x more than other, that is the real filter not a bug
- If rr < 1.3 across most signals, set rr_issue=true
- If dominant_kill_reason is AGG_THRESHOLD and avg confidence is near floor, threshold may be misconfigured

Return ONLY valid JSON:
{{"dominant_kill_reason":"<str>","regime_mismatch":<bool>,"strategy_to_suppress":"<str|null>","rr_issue":<bool>,"direction_bias_detected":<bool>,"ensemble_over_suppressing":<bool>,"whale_alignment_note":"<brief note on whale flow vs signal direction or empty string>","recommendation":"<150 chars>"}}"""

        data = self._parse_json(await self._call(prompt, model=_MODEL_BATCH, call_label="dead_signal_batch"))
        self._dead_signal_buffer.clear()
        self._last_dead_analysis = time.time()
        if not data:
            return None

        result = DeadSignalAnalysis(
            dominant_kill_reason=data.get("dominant_kill_reason", ""),
            regime_mismatch=bool(data.get("regime_mismatch", False)),
            strategy_to_suppress=data.get("strategy_to_suppress"),
            rr_issue=bool(data.get("rr_issue", False)),
            recommendation=data.get("recommendation", ""),
        )
        bias = " ⚠️ DIRECTION BIAS" if data.get("direction_bias_detected") else ""
        logger.info(f"🤖 Dead signals: {result.dominant_kill_reason} | {result.recommendation}{bias}")
        _ai_log.info(f"[DEAD_SIGNAL_AUDIT] {json.dumps(data)}")
        return result

    # ── 4: Structural Audit (NEW — deep model, 1x/day) ────────

    async def structural_audit(self, force: bool = False, deep: bool = False) -> Optional[StructuralAuditResult]:
        """
        Structural bias audit — finds direction-biased strategies, HTF asymmetry,
        whale conflicts, and regime mismatches the 2h dead-signal batch can't catch.

        Args:
            force: bypass the time throttle entirely (use for on-demand /audit command)
            deep:  use Nemotron 120B (deep tier) instead of Llama fast tier.
                   Deep is more thorough but costs 1 of the 50/day Nemotron free quota.
                   Auto-schedule runs on Llama. On-demand /audit deep uses Nemotron.

        Auto-schedule: Llama fast tier, every 1h, ~1000 tokens/call.
        On-demand:     Nemotron deep tier on request, max 1x/day unless force=True.
        """
        if self._mode == "off":
            return None
        now = time.time()
        if not force:
            if now - self._last_structural_audit < self._structural_audit_interval:
                return None
        if deep and not force:
            # Deep audit (Nemotron) self-throttles to 1x/day regardless of force
            if now - self._last_deep_audit < self._deep_audit_interval:
                logger.info("Deep audit already ran in the last 24h — use force=True to override")
                deep = False  # Fall back to fast tier

        ss = self._session_stats
        direction_bias = {}
        for strat, counts in ss["strategy_direction"].items():
            total = counts.get("LONG", 0) + counts.get("SHORT", 0)
            if total >= 10:
                sp = counts.get("SHORT", 0) / total * 100
                lp = counts.get("LONG", 0) / total * 100
                direction_bias[strat] = {
                    "total": total,
                    "SHORT": counts.get("SHORT", 0), "LONG": counts.get("LONG", 0),
                    "SHORT%": round(sp, 1), "LONG%": round(lp, 1),
                    "biased": sp > 85 or lp > 85,
                }

        whale_dom = "BUY" if ss["whale_buy_usd"] > ss["whale_sell_usd"] else (
            "SELL" if ss["whale_sell_usd"] > ss["whale_buy_usd"] else "NEUTRAL"
        )
        htf_asym = (
            ss["htf_blocked_long"] > ss["htf_blocked_short"] * 3 or
            ss["htf_blocked_short"] > ss["htf_blocked_long"] * 3
        )
        total_raw = ss.get("total_raw_signals", 0)

        prompt = f"""You are a senior quant engineer auditing a crypto trading bot for structural bugs.
Find systematic direction biases, broken strategy logic, and filter asymmetries.

TOTAL RAW SIGNALS (pre-filter): {total_raw}

STRATEGY DIRECTION BREAKDOWN (raw signals, before HTF or aggregator filtering):
{json.dumps(direction_bias, indent=2)}

HTF GUARDRAIL BLOCKS (signals killed by higher-timeframe filter, after raw generation):
LONG blocked={ss['htf_blocked_long']} | SHORT blocked={ss['htf_blocked_short']} | Asymmetric={htf_asym}
(If one direction is blocked 3x+ more, the guardrail is a one-sided factory)

WHALE FLOW vs SIGNALS:
Whale BUY: {ss['whale_buy_events']} events (${ss['whale_buy_usd']/1000:.0f}k)
Whale SELL: {ss['whale_sell_events']} events (${ss['whale_sell_usd']/1000:.0f}k)
Dominant whale: {whale_dom} | Signals sent AGAINST whale flow: {ss['signals_vs_whale_conflict']}

REGIME vs SIGNAL DIRECTION (signals that survived HTF, before aggregator):
{json.dumps(ss['regime_at_signal'], indent=2)}

Find: (1) strategies with >80% single-direction bias, (2) HTF one-sided factory ONLY when it is blocking signals that SHOULD go through, (3) systematic trading against whale flow, (4) signals opposite to regime.

STRICT RULES — violation means invalid output:
- biased_strategies: ONLY include strategies with >80% bias AND at least 10 signals. Below 10, directional skew is statistically expected noise. 75% bias is NOT a bug.
- htf_asymmetry: ONLY flag true if HTF is blocking the direction that ALIGNS with the weekly trend (e.g. blocking LONGs when weekly is BULLISH, or blocking SHORTs when weekly is BEARISH). Blocking LONGs in a BEARISH weekly trend is CORRECT behavior — do NOT flag it. Note: weekly trend context is not available in this data — if you cannot determine weekly trend from the regime data, set htf_asymmetry=false.
- If fewer than 20 total raw signals exist, set severity=LOW and root_causes=["Insufficient data — need 20+ signals for meaningful analysis"]
- Every recommendation must cite a SPECIFIC number from the data above

Return ONLY valid JSON:
{{"biased_strategies":["<name>",...],"htf_asymmetry":<bool>,"whale_signal_conflict":<bool>,"regime_signal_conflict":<bool>,"root_causes":["<cause1>","<cause2>"],"recommendations":["<fix1>","<fix2>"],"severity":"<LOW|MEDIUM|HIGH|CRITICAL>"}}"""

        use_model = _MODEL_DEEP if deep else _MODEL_FAST
        call_label = "structural_audit_deep" if deep else "structural_audit_fast"
        data = self._parse_json(await self._call(prompt, model=use_model, call_label=call_label))
        self._last_structural_audit = time.time()
        if deep:
            self._last_deep_audit = time.time()

        if not data:
            # Don't reset stats on failure — preserve for next attempt
            return None

        # Reset session stats only after a successful audit parse
        self._session_stats = {
            "strategy_direction": {}, "htf_blocked_long": 0, "htf_blocked_short": 0,
            "whale_buy_events": 0, "whale_sell_events": 0, "whale_buy_usd": 0.0,
            "whale_sell_usd": 0.0, "signals_vs_whale_conflict": 0, "regime_at_signal": {},
            "total_raw_signals": 0,
        }

        result = StructuralAuditResult(
            biased_strategies=data.get("biased_strategies", []),
            htf_asymmetry=bool(data.get("htf_asymmetry", False)),
            whale_signal_conflict=bool(data.get("whale_signal_conflict", False)),
            regime_signal_conflict=bool(data.get("regime_signal_conflict", False)),
            root_causes=data.get("root_causes", []),
            recommendations=data.get("recommendations", []),
            severity=data.get("severity", "LOW"),
        )
        emoji = {"LOW": "ℹ️", "MEDIUM": "⚠️", "HIGH": "🚨", "CRITICAL": "🔥"}.get(result.severity, "ℹ️")
        tier_label = "DEEP (Nemotron)" if deep else "FAST (Llama)"
        logger.info(f"{emoji} STRUCTURAL AUDIT [{result.severity}] [{tier_label}]: biased={result.biased_strategies} htf_asym={result.htf_asymmetry} whale_conflict={result.whale_signal_conflict}")
        for rec in result.recommendations:
            logger.info(f"  → {rec}")
        _ai_log.info(f"[STRUCTURAL_AUDIT] severity={result.severity} {json.dumps(data)}")
        return result

    # ── 5: Explain signal death (/ai why SYMBOL) ──────────────

    # ── Scan Notebook: deep symbol analysis (Arcee Trinity) ──────

    async def analyse_symbol(
        self,
        symbol: str,
        ohlcv_data: list,
        signals: list,
        whale_data: dict,
        news: list,
        regime: str,
        question: str = "",
    ) -> str:
        """
        Full symbol analysis for the scan notebook.
        Uses Arcee Trinity Large (400B, Reasoning) for deep multi-factor analysis.
        Returns plain-text analysis with reasoning visible.
        """
        if self._api_key is None:
            return "AI Analyst not initialised."

        # Summarise OHLCV into key metrics
        closes = [float(c[4]) for c in ohlcv_data[-20:]] if ohlcv_data else []
        highs  = [float(c[2]) for c in ohlcv_data[-20:]] if ohlcv_data else []
        lows   = [float(c[3]) for c in ohlcv_data[-20:]] if ohlcv_data else []
        vols   = [float(c[5]) for c in ohlcv_data[-20:]] if ohlcv_data else []

        price_now  = closes[-1] if closes else 0
        price_open = closes[0]  if closes else 0
        pct_move   = (price_now - price_open) / price_open * 100 if price_open else 0
        avg_vol    = sum(vols) / len(vols) if vols else 0
        last_vol   = vols[-1] if vols else 0
        vol_ratio  = last_vol / avg_vol if avg_vol else 1

        range_high = max(highs) if highs else price_now
        range_low  = min(lows)  if lows  else price_now
        range_pct  = (range_high - range_low) / range_low * 100 if range_low else 0

        # Summarise signals
        sig_summary = []
        for s in signals[:6]:
            sig_summary.append(
                f"  {s.get('strategy','?')}: {s.get('direction','?')} "
                f"conf={s.get('confidence',0):.0f} rr={s.get('rr_ratio',0):.1f}R "
                f"grade={s.get('alpha_grade','?')}"
            )

        # Whale context
        whale_buy  = whale_data.get('buy_usd', 0)
        whale_sell = whale_data.get('sell_usd', 0)
        whale_dom  = "BUY" if whale_buy > whale_sell else "SELL" if whale_sell > whale_buy else "NEUTRAL"

        # News
        news_lines = [f"  [{n.get('source','?')}] {n.get('title','')}" for n in news[:4]]

        user_q = f"\n\nSpecific question: {question}" if question else ""

        prompt = f"""You are a professional crypto futures analyst. Analyse {symbol} comprehensively.

PRICE ACTION (last 20 bars, 1h):
  Current: ${price_now:.6f}
  Period move: {pct_move:+.2f}%
  Range: ${range_low:.6f} – ${range_high:.6f} ({range_pct:.1f}% range)
  Volume vs avg: {vol_ratio:.2f}x (last bar vs 20-bar average)

MARKET REGIME: {regime}

STRATEGY SIGNALS GENERATED:
{chr(10).join(sig_summary) if sig_summary else "  No signals generated"}

WHALE FLOW (last 10 min):
  Buy volume: ${whale_buy/1000:.0f}k | Sell volume: ${whale_sell/1000:.0f}k
  Dominant: {whale_dom}

NEWS (last 2h):
{chr(10).join(news_lines) if news_lines else "  No relevant news"}
{user_q}

Provide:
1. Overall assessment (2-3 sentences on what the chart is telling you)
2. Key risks for any trade right now
3. What would need to happen to confirm a strong setup
4. Your confidence in this setup (0-100) and why

Be specific about price levels. Be honest if the setup is weak. No generic advice."""

        return await self._call(prompt, model=_MODEL_ANALYST, call_label=f"scan_notebook_{symbol}") or "Analysis unavailable — check API key."

    async def chat_about_symbol(
        self,
        symbol: str,
        question: str,
        context: dict,
        history: list = None,
    ) -> str:
        """
        Conversational AI chat about a symbol in the scan notebook.
        Uses fast Llama for quick questions, Arcee for deep questions.
        Auto-routes based on question complexity.
        """
        if self._api_key is None:
            return "AI Analyst not initialised."

        # Route: deep questions → Arcee, quick → Llama
        deep_triggers = ["why", "explain", "risk", "what if", "predict", "compare", "analyse", "how does"]
        use_deep = any(t in question.lower() for t in deep_triggers)
        model = _MODEL_ANALYST if use_deep else _MODEL_FAST
        call_label = f"chat_deep_{symbol}" if use_deep else f"chat_fast_{symbol}"

        # Build conversation history
        messages = []
        system = (
            f"You are TitanBot's AI analyst. You're helping a trader analyse {symbol}. "
            f"Current regime: {context.get('regime','?')}. "
            f"Recent price: ${context.get('price',0):.6f}. "
            f"Keep answers concise and specific. Reference actual numbers."
        )

        if history:
            for h in history[-6:]:  # last 3 exchanges
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question})

        # Build a single prompt from system + history + question and route through _call
        hist_text = ""
        if history:
            for h in history[-6:]:
                role = h.get("role", "user")
                hist_text += f"\n[{role.upper()}]: {h.get('content', '')}"

        prompt = f"{system}{hist_text}\n\n{question}"
        raw = await self._call(prompt, model=model, call_label=call_label)
        if raw:
            _ai_log.debug(f"[CHAT:{call_label}] q={question[:60]} → {raw[:80]}")
        return raw or "No response."

    # ── AI Diagnostics: full pipeline review ─────────────────────

    async def pipeline_diagnostics(self, session_data: dict) -> dict:
        """
        Holistic pipeline review — looks at the full signal pipeline and identifies
        where value is being lost. Uses Step Flash (fast MoE) since this is called
        frequently and needs to be cheap.

        Returns structured findings with severity and plain-English explanations.
        """
        if self._mode == "off" or self._api_key is None:
            return {}

        sd = session_data
        prompt = f"""You are auditing a crypto trading bot's signal pipeline end-to-end.
Identify exactly where signals are being lost, filtered, or going wrong.

CRITICAL REGIME CONTEXT (read this before making any judgement):
  Current regime: {sd.get('regime', 'UNKNOWN')}
  Fear & Greed: {sd.get('fear_greed', '?')}
  Weekly bias: {sd.get('weekly_bias', '?')}
  IMPORTANT: In VOLATILE or BEAR_TREND regime with weekly BEARISH bias:
    - Blocking LONGs is CORRECT and expected — NOT a bug
    - Zero LONG signals is expected behavior, NOT directional bias
    - Low win rate is expected when bot has been running <2 hours (no completed trades)
    - HTF guardrail blocking LONGs is working as designed
    - Only flag issues that are problems GIVEN the current regime context

PIPELINE STATS (last session):
  Symbols scanned: {sd.get('symbols_scanned', 0)}
  Raw signals generated: {sd.get('raw_signals', 0)}
  Killed by RR floor: {sd.get('killed_rr', 0)}
  Killed by HTF guardrail: {sd.get('killed_htf', 0)}
  Killed by confidence threshold: {sd.get('killed_conf', 0)}
  Killed by no-trade zones: {sd.get('killed_ntze', 0)}
  Killed by aggregator: {sd.get('killed_agg', 0)}
  Published: {sd.get('published', 0)}
  Win rate on published: {sd.get('win_rate', 0):.1f}% (ignore if <10 completed trades)

STRATEGY BREAKDOWN:
{json.dumps(sd.get('strategy_breakdown', {}), indent=2)}

DIRECTION IMBALANCE:
  LONG signals: {sd.get('long_count', 0)} | SHORT signals: {sd.get('short_count', 0)}
  HTF LONG blocks: {sd.get('htf_long_blocks', 0)} | HTF SHORT blocks: {sd.get('htf_short_blocks', 0)}

TOP KILL REASONS:
{json.dumps(sd.get('kill_reasons', {}), indent=2)}

Identify:
1. The biggest bottleneck (where most value is lost)
2. Any structural bias problems
3. Whether the filters are too aggressive or too permissive
4. Top 3 specific improvements with expected impact

Return ONLY valid JSON:
{{
  "bottleneck": "<where most signals are lost and why>",
  "structural_issues": ["<issue 1>", "<issue 2>"],
  "filter_verdict": "<too aggressive / too permissive / balanced>",
  "improvements": [
    {{"change": "<what to change>", "expected": "<what improves>", "priority": "<HIGH|MEDIUM|LOW>"}}
  ],
  "overall_health": "<CRITICAL|POOR|FAIR|GOOD|EXCELLENT>",
  "health_score": <0-100>
}}"""

        raw = await self._call(prompt, model=_MODEL_BATCH, call_label="pipeline_diag")
        data = self._parse_json(raw)
        if data:
            _ai_log.info(f"[PIPELINE_DIAG] health={data.get('overall_health')} score={data.get('health_score')}")
        return data or {}

    async def explain_signal_death(self, symbol: str, recent_deaths: List[dict], recent_log_lines: List[str]) -> str:
        if self._api_key is None:
            return "AI Analyst not initialised — run initialize() first."
        if not recent_deaths and not recent_log_lines:
            return f"No recent signal activity found for {symbol}."
        prompt = f"""Trader asking why bot signals for {symbol} keep failing.
Recent deaths: {json.dumps(recent_deaths[:10], indent=2)}
Log lines: {chr(10).join(recent_log_lines[:20])}
Explain in 3-5 plain English sentences. Mention exact kill reason and what could fix it. No bullet points."""
        return await self._call(prompt, call_label=f"explain_{symbol}") or f"Could not analyse {symbol} — try again."

    # ── 6: Narrative review (compat) ─────────────────────────

    async def review_narrative(self, narrative: str, symbol: str, direction: str, grade: str) -> Optional[dict]:
        if self._mode != "full" or not narrative:
            return None
        prompt = f"""Rate this signal narrative for {symbol} {direction} {grade}-grade:
\"\"\"{narrative}\"\"\"
Return ONLY valid JSON:
{{"clarity":<1-10>,"actionability":<1-10>,"too_technical":<bool>,"too_long":<bool>,"too_vague":<bool>,"improvement":"<str|null>"}}"""
        return self._parse_json(await self._call(prompt, call_label="review_narrative"))

    # ── GROUP 4A: Execution Funnel Audit ──────────────────────────────────────

    async def execution_funnel_audit(self, signals: List[dict], regime: str) -> dict:
        """
        GROUP 4A: Post-publication execution analysis.

        This is the audit the AI was MISSING. The pipeline_diag prompt only saw
        pre-publication kill counts. This prompt sees what happened AFTER publication:
        - Did price reach the entry zone?
        - How far was the zone from publish price?
        - What was the fill rate by strategy / entry zone distance?
        - Were expired signals directionally correct anyway?

        Called by diagnostic_engine every 2h when there are ≥3 expired signals.
        """
        if self._mode == "off" or self._api_key is None or not signals:
            return {}

        filled   = [s for s in signals if s.get("zone_reached") and s.get("outcome") not in ("EXPIRED", None)]
        expired  = [s for s in signals if s.get("outcome") == "EXPIRED"]
        active   = [s for s in signals if s.get("outcome") is None and not s.get("outcome")]

        def _zone_dist(s):
            """Distance % from publish_price to entry zone mid."""
            pp = s.get("publish_price")
            el = s.get("entry_low")
            eh = s.get("entry_high")
            if not pp or not el or not eh or pp <= 0:
                return None
            entry_mid = (el + eh) / 2
            return round(abs(pp - entry_mid) / pp * 100, 2)

        filled_lines = []
        for s in filled:
            d = _zone_dist(s)
            filled_lines.append(
                f"  {s['symbol']} {s['direction']}: publish ${s.get('publish_price','?')} "
                f"zone {s.get('entry_low','?')}-{s.get('entry_high','?')} "
                f"({d}% away) → FILLED"
            )

        expired_lines = []
        for s in expired:
            d = _zone_dist(s)
            pep = s.get("post_expiry_price")
            direction = s.get("direction", "")
            entry_mid = ((s.get("entry_low", 0) + s.get("entry_high", 0)) / 2) if s.get("entry_low") else 0
            pp = s.get("publish_price", 0) or 0

            # Was price direction correct after expiry?
            if pep and pp > 0:
                if direction == "LONG":
                    correct = "YES ✅" if pep > pp else "NO ❌"
                    move = f"+{(pep-pp)/pp*100:.1f}%" if pep > pp else f"{(pep-pp)/pp*100:.1f}%"
                else:
                    correct = "YES ✅" if pep < pp else "NO ❌"
                    move = f"{(pep-pp)/pp*100:.1f}%" if pep < pp else f"+{(pep-pp)/pp*100:.1f}%"
                direction_str = f"post-expiry move: {move} direction_correct={correct}"
            else:
                direction_str = "post-expiry price not available"

            zone_reached = "zone_reached=YES" if s.get("zone_reached") else "zone_reached=NO"
            expired_lines.append(
                f"  {s['symbol']} {direction}: publish ${pp:.4f} "
                f"zone {s.get('entry_low','?'):.4f}-{s.get('entry_high','?'):.4f} "
                f"({d}% away) {zone_reached} | {direction_str}"
            )

        fill_rate = f"{len(filled)}/{len(signals)}" if signals else "0/0"
        zone_reached_count = sum(1 for s in signals if s.get("zone_reached"))

        prompt = f"""You are auditing a crypto trading bot's signal EXECUTION quality — not signal generation.
The pipeline_diag already covers pre-publication kills. This audit covers what happens AFTER publication.

REGIME: {regime}
SIGNALS THIS PERIOD: {len(signals)} total | fill_rate={fill_rate} | zone_reached={zone_reached_count}

FILLED SIGNALS ({len(filled)}):
{chr(10).join(filled_lines) if filled_lines else "  None"}

EXPIRED UNFILLED ({len(expired)}):
{chr(10).join(expired_lines) if expired_lines else "  None"}

ACTIVE/PENDING ({len(active)}): {[s.get('symbol') for s in active]}

IMPORTANT CONTEXT:
- "ACTIVE/PENDING" signals have not yet had time to fill — DO NOT count them as failed fills
- Only judge fill rate based on EXPIRED UNFILLED signals
- If EXPIRED UNFILLED is 0, fill_rate verdict is INSUFFICIENT_DATA, severity=LOW
- If ACTIVE/PENDING > EXPIRED UNFILLED × 2, it is too early to draw conclusions
- zone_placement_verdict should be "balanced" unless you have clear evidence from filled vs expired patterns

KEY QUESTIONS (only for expired signals, not active):
1. What is the fill rate pattern among EXPIRED signals?
2. Were expired signals directionally correct vs price action?
3. What is causing low fills — entry zone depth, market conditions, or regime?
4. What specific fixes would improve fill rate?

Return ONLY valid JSON:
{{
  "fill_rate_pct": <0-100>,
  "zone_placement_verdict": "<too deep|immediate|balanced>",
  "direction_accuracy_on_expired": "<X/Y (Z%)>",
  "root_cause": "<primary reason signals are not filling>",
  "execution_issues": ["<issue 1>", "<issue 2>"],
  "recommendations": [
    {{"change": "<what>", "expected": "<impact>", "priority": "<HIGH|MEDIUM|LOW>"}}
  ],
  "severity": "<CRITICAL|HIGH|MEDIUM|LOW>"
}}"""

        raw = await self._call(prompt, model=_MODEL_BATCH, call_label="execution_funnel_audit")
        data = self._parse_json(raw)
        if data:
            _ai_log.info(
                f"[EXECUTION_FUNNEL] fill={data.get('fill_rate_pct')}% "
                f"zone={data.get('zone_placement_verdict')} "
                f"root_cause={data.get('root_cause','?')[:60]}"
            )
        return data or {}

    # ── GROUP 4B: Post-Expiry Direction Accuracy Audit ────────────────────────

    async def post_expiry_direction_audit(self, expired_signals: List[dict]) -> dict:
        """
        GROUP 4B: Are expired signals directionally correct?

        Fetches this from the outcomes table post_expiry_price field.
        If 100% of expired signals moved in the predicted direction,
        the bot has signal quality but execution failure.
        If <50% were correct, signal quality itself is the problem.

        Called by outcome_monitor after every batch of 3+ signals expires.
        """
        if self._mode == "off" or self._api_key is None or not expired_signals:
            return {}

        # Only analyse signals where we have post_expiry_price
        analysable = [
            s for s in expired_signals
            if s.get("post_expiry_price") and s.get("publish_price")
        ]

        if not analysable:
            return {"note": "No post_expiry_price data yet — will improve after first full session"}

        lines = []
        correct_count = 0
        for s in analysable:
            pp    = s["publish_price"]
            pep   = s["post_expiry_price"]
            dirn  = s.get("direction", "LONG")
            sym   = s.get("symbol", "?")
            move  = (pep - pp) / pp * 100

            if dirn == "LONG":
                correct = pep > pp
                move_str = f"+{move:.1f}%" if move >= 0 else f"{move:.1f}%"
            else:
                correct = pep < pp
                move_str = f"{move:.1f}%" if move <= 0 else f"+{move:.1f}%"

            if correct:
                correct_count += 1
            lines.append(
                f"  {sym} {dirn}: publish ${pp:.4f} → expiry ${pep:.4f} "
                f"({move_str}) direction={'CORRECT ✅' if correct else 'WRONG ❌'}"
            )

        accuracy = correct_count / len(analysable) * 100

        prompt = f"""Audit of {len(analysable)} expired crypto signals — did price move in the predicted direction after expiry?

{chr(10).join(lines)}

DIRECTION ACCURACY: {correct_count}/{len(analysable)} = {accuracy:.0f}%

If accuracy is HIGH (>70%): signals are CORRECT but EXECUTION is failing (entry zones, speed, or triggers).
If accuracy is LOW (<50%): signal QUALITY itself is the problem (wrong direction predicted).
If accuracy is MEDIUM (50-70%): mixed — some regime mismatch or noise.

Return ONLY valid JSON:
{{
  "direction_accuracy_pct": {accuracy:.0f},
  "verdict": "<EXECUTION_FAILURE|SIGNAL_QUALITY_FAILURE|MIXED>",
  "analysis": "<2-3 sentences explaining what this means for the bot>",
  "primary_fix": "<most important thing to fix based on this data>",
  "worst_misses": ["<symbol that missed biggest opportunity>"]
}}"""

        raw = await self._call(prompt, model=_MODEL_BATCH, call_label="post_expiry_direction_audit")
        data = self._parse_json(raw)
        if data:
            _ai_log.info(
                f"[POST_EXPIRY_AUDIT] accuracy={data.get('direction_accuracy_pct')}% "
                f"verdict={data.get('verdict')} "
                f"fix={data.get('primary_fix','?')[:60]}"
            )
        return data or {}

    # ── GROUP 4C: Strategy × Regime Performance Audit ────────────────────────

    async def strategy_regime_audit(self, perf_matrix: dict, regime: str) -> dict:
        """
        GROUP 4C: Strategy performance broken down by regime.

        perf_matrix format:
        {
          "IchimokuCloud|BULL_TREND|LONG": {"signals": 8, "filled": 0, "wins": 0, "expired": 8},
          "SMC|BULL_TREND|LONG": {"signals": 12, "filled": 2, "wins": 1, "expired": 10},
          ...
        }

        This would have caught: "IchimokuCloud LONG in BULL_TREND: 0/8 filled (0%)"
        Called by diagnostic_engine every 4h.
        """
        if self._mode == "off" or self._api_key is None or not perf_matrix:
            return {}

        # Format the matrix for the prompt
        lines = []
        for key, stats in sorted(perf_matrix.items(), key=lambda x: x[1].get("signals", 0), reverse=True):
            s = stats.get("signals", 0)
            if s < 2:
                continue
            f  = stats.get("filled", 0)
            w  = stats.get("wins", 0)
            e  = stats.get("expired", 0)
            fill_pct = round(f / s * 100) if s > 0 else 0
            win_pct  = round(w / f * 100) if f > 0 else 0
            lines.append(
                f"  {key}: {s} signals | fill={fill_pct}% ({f}/{s}) | "
                f"win={win_pct}% ({w}/{f if f>0 else '?'}) | expired={e}"
            )

        if not lines:
            return {}

        prompt = f"""Audit of a crypto trading bot's strategy performance broken down by strategy × regime × direction.
Current regime: {regime}

STRATEGY × REGIME × DIRECTION MATRIX:
{chr(10).join(lines)}

Find:
1. Combinations with 0% fill rate (entry zones always missed)
2. Combinations with high fill but low win rate (bad signal quality in that context)
3. Combinations that should be suppressed in the current regime
4. Combinations that are outperforming and should get higher allocation

Return ONLY valid JSON:
{{
  "suppress_immediately": ["<strategy|regime|direction combos with 0% fill for 5+ signals>"],
  "investigate": ["<combos with fill but poor win rate>"],
  "promote": ["<best performing combos>"],
  "regime_recommendation": "<what should bot do differently in {regime}>",
  "key_finding": "<single most important finding in one sentence>"
}}"""

        raw = await self._call(prompt, model=_MODEL_BATCH, call_label="strategy_regime_audit")
        data = self._parse_json(raw)
        if data:
            _ai_log.info(
                f"[STRATEGY_REGIME_AUDIT] regime={regime} "
                f"suppress={data.get('suppress_immediately',[])} "
                f"finding={data.get('key_finding','?')[:60]}"
            )
        return data or {}

    # ── REMOVED: generate_signal_narrative ───────────────────
    # V2.09: Removed. Burned ~20 calls/day for cosmetic Telegram text.
    # The formatter produces quality signal cards from confluence notes.


# ── Singleton ─────────────────────────────────────────────────
ai_analyst = AIAnalyst()
