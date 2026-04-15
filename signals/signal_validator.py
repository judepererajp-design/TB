"""
TitanBot Pro — Signal Data Validator ("Skeptical LLM Mode")
=============================================================
Dual-layer defense against bad calculations and wired-incorrectly logic:

  Layer A: Programmatic Validation (STRICT)
    Hard constraints that catch impossible values — RSI out of range,
    negative prices, inverted entry zones, direction/indicator conflicts.
    These run every cycle, are zero-cost, and never miss.

  Layer B: LLM Anomaly Detection (FLEXIBLE)
    Pattern-aware checks for *suspicious* combinations that pass Layer A
    but still smell wrong — e.g. BUY in overbought, trend/signal conflict,
    conflicting indicators.  Uses the FAST model (1 call, ~2-3 s).

Design:
  1. Layer A runs first (instant, free).
  2. If Layer A flags ERROR → signal is killed immediately, no LLM call.
  3. If Layer A passes → Layer B runs (LLM anomaly check).
  4. Layer B returns a data_quality score and optional warnings.
  5. Caller decides: ERROR → reject; WARNING with low quality → penalise.

Integration point: called inside aggregator.process() BEFORE scoring.
The validator NEVER blocks on its own — it returns a ValidationResult
that the aggregator interprets (reject, penalise, or pass through).

Feature flag: SIGNAL_VALIDATOR (off | shadow | live)
  - off:    skip entirely
  - shadow: run both layers, log results, but don't affect signals
  - live:   full enforcement — errors reject, warnings penalise confidence
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Validation result ─────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of the dual-layer validation check."""
    status: str = "OK"             # OK | WARNING | ERROR
    data_quality: str = "HIGH"     # HIGH | MEDIUM | LOW
    issues: List[str] = field(default_factory=list)
    # Layer A details
    hard_errors: List[str] = field(default_factory=list)
    # Layer B details (LLM)
    llm_warnings: List[str] = field(default_factory=list)
    llm_confidence_in_data: int = 100  # 0-100: how much the LLM trusts the input
    # Timing
    layer_a_ms: float = 0.0
    layer_b_ms: float = 0.0


# ── Programmatic bounds (Layer A) ─────────────────────────────────────

# These are physical/mathematical impossibilities — if violated, the
# calculation engine has a bug.
HARD_RULES: Dict[str, dict] = {
    "rsi":       {"min": 0, "max": 100, "label": "RSI"},
    "adx":       {"min": 0, "max": 100, "label": "ADX"},
    "confidence": {"min": 0, "max": 100, "label": "Confidence"},
}


class SignalValidator:
    """
    Dual-layer signal data validator.

    Layer A: instant programmatic checks (always runs).
    Layer B: LLM anomaly detection (runs only when AI mode is active
             and Layer A passes).
    """

    # LLM validation throttle — avoid burning calls on every signal.
    # Only call LLM if ≥ this many seconds since last LLM validation.
    _LLM_COOLDOWN_SECS = 30

    def __init__(self):
        self._last_llm_validation: float = 0.0
        self._stats = {
            "total_checked": 0,
            "hard_errors": 0,
            "llm_warnings": 0,
            "passed_clean": 0,
        }

    # ── Public API ────────────────────────────────────────────────────

    async def validate(
        self,
        signal_data: dict,
        run_llm: bool = True,
    ) -> ValidationResult:
        """
        Run dual-layer validation on signal data.

        Args:
            signal_data: dict with keys like rsi, adx, price, volume,
                         direction, strategy_signal, regime, confidence, etc.
            run_llm: whether to attempt Layer B (LLM check). Set False
                     to skip LLM (e.g. when AI mode is off or budget exhausted).

        Returns:
            ValidationResult with status, issues, and data_quality score.
        """
        self._stats["total_checked"] += 1
        result = ValidationResult()

        # ── Layer A: hard programmatic checks ─────────────────────
        t0 = time.time()
        self._run_hard_checks(signal_data, result)
        result.layer_a_ms = (time.time() - t0) * 1000

        if result.hard_errors:
            result.status = "ERROR"
            result.data_quality = "LOW"
            result.issues = list(result.hard_errors)
            self._stats["hard_errors"] += 1
            logger.warning(
                "🛑 VALIDATOR Layer A ERROR: %s | data=%s",
                "; ".join(result.hard_errors),
                _summarise_data(signal_data),
            )
            return result

        # ── Layer B: LLM anomaly detection ────────────────────────
        if run_llm and self._should_run_llm():
            t1 = time.time()
            await self._run_llm_check(signal_data, result)
            result.layer_b_ms = (time.time() - t1) * 1000
            self._last_llm_validation = time.time()

        # ── Aggregate ─────────────────────────────────────────────
        if result.llm_warnings:
            result.status = "WARNING"
            result.issues = list(result.llm_warnings)
            self._stats["llm_warnings"] += 1
            # Data quality based on LLM confidence
            if result.llm_confidence_in_data < 40:
                result.data_quality = "LOW"
            elif result.llm_confidence_in_data < 70:
                result.data_quality = "MEDIUM"
        else:
            self._stats["passed_clean"] += 1

        return result

    def get_stats(self) -> dict:
        """Return validation statistics for monitoring."""
        return dict(self._stats)

    # ── Layer A: Programmatic hard checks ─────────────────────────

    def _run_hard_checks(self, data: dict, result: ValidationResult) -> None:
        """
        Check for mathematically impossible values.
        These are bugs in the calculation engine — not market conditions.
        """
        # 1. Range checks for known indicators
        for key, rules in HARD_RULES.items():
            val = data.get(key)
            if val is not None:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    result.hard_errors.append(
                        f"{rules['label']} is not a number: {val!r}"
                    )
                    continue
                if val < rules["min"] or val > rules["max"]:
                    result.hard_errors.append(
                        f"{rules['label']} = {val} is outside valid range "
                        f"[{rules['min']}, {rules['max']}]. Possible calculation bug."
                    )

        # 2. Price must be positive
        price = data.get("price")
        if price is None:
            price = data.get("entry_mid")
        if price is not None:
            try:
                price = float(price)
                if price <= 0:
                    result.hard_errors.append(
                        f"Price = {price} must be > 0."
                    )
            except (TypeError, ValueError):
                result.hard_errors.append(
                    f"Price is not a number: {price!r}"
                )

        # 3. Volume cannot be negative
        volume = data.get("volume")
        if volume is not None:
            try:
                volume = float(volume)
                if volume < 0:
                    result.hard_errors.append(
                        f"Volume = {volume} cannot be negative."
                    )
            except (TypeError, ValueError):
                pass  # Volume not always present

        # 4. Stop-loss geometry (direction-aware)
        direction = data.get("direction", "").upper()
        entry_low = data.get("entry_low")
        entry_high = data.get("entry_high")
        stop_loss = data.get("stop_loss")

        if entry_low is not None and entry_high is not None:
            try:
                el, eh = float(entry_low), float(entry_high)
                if el > eh:
                    result.hard_errors.append(
                        f"Entry zone inverted: entry_low ({el}) > entry_high ({eh})."
                    )
            except (TypeError, ValueError):
                pass

        if stop_loss is not None and entry_low is not None and entry_high is not None:
            try:
                sl = float(stop_loss)
                el, eh = float(entry_low), float(entry_high)
                if direction == "LONG" and sl >= el:
                    result.hard_errors.append(
                        f"LONG signal has SL ({sl}) >= entry_low ({el}). "
                        f"SL must be below entry for longs."
                    )
                elif direction == "SHORT" and sl <= eh:
                    result.hard_errors.append(
                        f"SHORT signal has SL ({sl}) <= entry_high ({eh}). "
                        f"SL must be above entry for shorts."
                    )
            except (TypeError, ValueError):
                pass

        # 5. ATR must be positive if present
        atr = data.get("atr")
        if atr is not None:
            try:
                atr = float(atr)
                if atr < 0:
                    result.hard_errors.append(
                        f"ATR = {atr} cannot be negative."
                    )
            except (TypeError, ValueError):
                pass

        # 6. R:R ratio must be positive
        rr = data.get("rr_ratio")
        if rr is not None:
            try:
                rr = float(rr)
                if rr < 0:
                    result.hard_errors.append(
                        f"R:R ratio = {rr} cannot be negative."
                    )
            except (TypeError, ValueError):
                pass

    # ── Layer B: LLM anomaly detection ────────────────────────────

    def _should_run_llm(self) -> bool:
        """Throttle LLM calls to avoid burning the daily budget."""
        return time.time() - self._last_llm_validation >= self._LLM_COOLDOWN_SECS

    async def _run_llm_check(self, data: dict, result: ValidationResult) -> None:
        """
        Ask the LLM to evaluate the signal data for logical inconsistencies.
        The LLM acts as a pattern anomaly detector — it catches weird
        combinations that hard rules can't anticipate.
        """
        try:
            from utils.free_llm import call_llm, parse_json_response, MODEL_FAST

            prompt = self._build_llm_prompt(data)
            raw = await call_llm(
                prompt, model=MODEL_FAST, temperature=0.1, max_tokens=400
            )
            if not raw:
                logger.debug("Signal validator LLM returned empty — skipping Layer B")
                return

            parsed = parse_json_response(raw)
            if not parsed:
                logger.debug("Signal validator LLM response not parseable — skipping")
                return

            # Extract LLM verdict
            status = parsed.get("status", "OK").upper()
            confidence = int(parsed.get("data_confidence", 100))
            issues = parsed.get("issues", [])

            result.llm_confidence_in_data = max(0, min(100, confidence))

            if status in ("ERROR", "WARNING") and issues:
                for issue in issues[:3]:  # cap at 3 issues
                    if isinstance(issue, str):
                        result.llm_warnings.append(issue)
            elif confidence < 50:
                # Low confidence even without explicit issues
                result.llm_warnings.append(
                    f"LLM data confidence is low ({confidence}/100) — "
                    f"indicators may be inconsistent."
                )

        except ImportError:
            logger.debug("free_llm not available — skipping Layer B")
        except Exception as e:
            logger.debug("Signal validator LLM check failed: %s", e)

    @staticmethod
    def _build_llm_prompt(data: dict) -> str:
        """
        Build the validation prompt. Teaches the LLM the sanity rules
        and asks it to evaluate the specific signal data.
        """
        # Extract key fields for the prompt
        direction = data.get("direction", "UNKNOWN")
        strategy = data.get("strategy", "UNKNOWN")
        regime = data.get("regime", "UNKNOWN")

        # Build indicator summary
        indicators = []
        for key in ("rsi", "adx", "macd_histogram", "volume_ratio",
                     "bollinger_position", "funding_rate"):
            val = data.get(key)
            if val is not None:
                indicators.append(f"  {key}: {val}")
        indicator_block = "\n".join(indicators) if indicators else "  (no indicators provided)"

        confidence = data.get("confidence", "?")

        return f"""You are a trading signal DATA VALIDATOR. Your ONLY job is to check
if the signal data is logically consistent. You are NOT making a trading decision.

=== SIGNAL DATA ===
Direction: {direction}
Strategy:  {strategy}
Regime:    {regime}
Confidence: {confidence}
Indicators:
{indicator_block}

=== VALIDATION RULES ===
1. RSI > 70 with BUY/LONG signal = suspicious (overbought → usually sell)
2. RSI < 30 with SELL/SHORT signal = suspicious (oversold → usually buy)
3. Strong uptrend regime + SHORT signal = suspicious (unless reversal strategy)
4. Strong downtrend regime + LONG signal = suspicious (unless reversal strategy)
5. MACD histogram positive + SHORT = suspicious (momentum is bullish)
6. MACD histogram negative + LONG = suspicious (momentum is bearish)
7. Funding rate extremely positive + LONG = crowded trade risk
8. Funding rate extremely negative + SHORT = crowded trade risk
9. ADX < 20 with trend-following strategy = suspicious (no trend to follow)
10. Multiple indicators conflicting with signal direction = suspicious

IMPORTANT: Reversal strategies (ExtremeReversal, MeanReversion) are EXPECTED to
trade against indicators — do NOT flag them for rules 1-2.

Return ONLY valid JSON:
{{"status": "OK|WARNING|ERROR", "data_confidence": 0-100, "issues": ["issue1", "issue2"]}}

- OK: data looks consistent
- WARNING: unusual combination detected (might be intentional)
- ERROR: data is clearly wrong (impossible values or blatant logic error)
- data_confidence: how much you trust this data (100=perfect, 0=garbage)
- issues: list of specific problems found (empty if OK)"""


# ── Helper ────────────────────────────────────────────────────────────

def _summarise_data(data: dict) -> str:
    """Short summary of signal data for log messages."""
    parts = []
    for key in ("symbol", "direction", "strategy", "rsi", "adx",
                "confidence", "regime", "price"):
        val = data.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    return " | ".join(parts) if parts else "(empty)"


# ── Singleton ─────────────────────────────────────────────────────────
signal_validator = SignalValidator()
