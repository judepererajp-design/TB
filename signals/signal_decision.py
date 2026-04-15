"""
TitanBot Pro — On-demand AI signal decision review
==================================================
Builds a richer, per-signal context payload for the dashboard and asks the LLM
for a human-in-the-loop verdict: TAKE, WAIT, or AVOID.
"""

import logging
from typing import Any, Dict

from utils.free_llm import (
    MODEL_FAST as _MODEL_FAST,
    call_llm as _free_llm_call,
    parse_json_response as _parse_json_response,
)

logger = logging.getLogger(__name__)


async def review_signal(signal: Dict[str, Any], detail: Dict[str, Any], openrouter_api_key: str = "") -> Dict[str, Any]:
    ctx = _gather_context(signal, detail)
    prompt = _build_prompt(detail, ctx)
    verdict = None
    try:
        verdict = await _call_ai(prompt, openrouter_api_key)
    except Exception as exc:
        logger.warning("signal_decision LLM failed for %s: %s", detail.get("symbol"), exc)
    if not isinstance(verdict, dict):
        verdict = _rule_based_fallback(detail, ctx)
    return _normalise_review(verdict, detail, ctx)


def _gather_context(signal: Dict[str, Any], detail: Dict[str, Any]) -> Dict[str, Any]:
    entry_low = float(detail.get("entry_low") or 0)
    entry_high = float(detail.get("entry_high") or 0)
    stop_loss = float(detail.get("stop_loss") or 0)
    tp1 = float(detail.get("tp1") or 0)
    zone_mid = (entry_low + entry_high) / 2 if entry_low and entry_high else 0.0
    current_price = 0.0
    try:
        from core.price_cache import price_cache
        current_price = float(price_cache.get(detail.get("symbol", "")) or 0.0)
    except Exception:
        current_price = 0.0
    if current_price <= 0:
        current_price = float(detail.get("expected_fill_mid") or zone_mid or entry_low or entry_high or 0.0)

    distance_pct = 0.0
    in_entry_zone = False
    if current_price > 0 and entry_low > 0 and entry_high > 0:
        in_entry_zone = entry_low <= current_price <= entry_high
        if current_price < entry_low:
            distance_pct = ((entry_low - current_price) / entry_low) * 100
        elif current_price > entry_high:
            distance_pct = ((current_price - entry_high) / entry_high) * 100

    regime_now = detail.get("regime") or signal.get("regime") or "UNKNOWN"
    try:
        from analyzers.regime import regime_analyzer
        regime_now = getattr(regime_analyzer.regime, "value", regime_now) or regime_now
    except Exception:
        pass

    funding_now = None
    try:
        from analyzers.derivatives import derivatives_analyzer
        coin = (detail.get("symbol") or "").replace("/USDT", "").replace("/BUSD", "")
        deriv = derivatives_analyzer.get_coin_intel(coin)
        if deriv:
            funding_now = deriv.funding_rate
    except Exception:
        funding_now = None

    btc_weekly_bearish = None
    btc_4h_bouncing = None
    try:
        from analyzers.htf_guardrail import htf_guardrail
        btc_weekly_bearish = getattr(htf_guardrail, "_weekly_bearish", None)
        btc_4h_bouncing = getattr(htf_guardrail, "_btc_4h_bouncing", None)
    except Exception:
        pass

    return {
        "current_price": current_price,
        "in_entry_zone": in_entry_zone,
        "distance_to_zone_pct": round(distance_pct, 2),
        "regime_now": regime_now,
        "regime_changed": regime_now != (signal.get("regime") or detail.get("regime") or regime_now),
        "funding_now": funding_now,
        "btc_weekly_bearish": btc_weekly_bearish,
        "btc_4h_bouncing": btc_4h_bouncing,
        "state": detail.get("exec_state") or signal.get("exec_state") or "WATCHING",
        "reward_to_tp1_pct": round(((tp1 - current_price) / current_price) * 100, 2) if current_price > 0 and tp1 > 0 else None,
        "risk_to_sl_pct": round((abs(current_price - stop_loss) / current_price) * 100, 2) if current_price > 0 and stop_loss > 0 else None,
    }


def _build_inputs_used(detail: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    confluence = [str(c) for c in (detail.get("confluence") or [])[:6]]
    signal_inputs = [
        f"{detail.get('symbol', '—')} {detail.get('direction', '—')} {detail.get('strategy', '—')}",
        f"confidence {round(detail.get('confidence') or 0)}/100 · grade {detail.get('alpha_grade') or '—'}",
        f"entry {detail.get('entry_low') or '—'} → {detail.get('entry_high') or '—'} · SL {detail.get('stop_loss') or '—'} · TP1 {detail.get('tp1') or '—'}",
        f"R/R {detail.get('rr_ratio') or '—'} · adj R/R {detail.get('fee_adjusted_rr') or '—'}",
    ]
    market_inputs = [
        f"regime at decision {ctx.get('regime_now', 'UNKNOWN')}",
        f"live price {ctx.get('current_price') or '—'} · state {ctx.get('state', 'WATCHING')}",
        f"price is {'inside' if ctx.get('in_entry_zone') else 'outside'} entry zone",
        f"distance to zone {ctx.get('distance_to_zone_pct', 0)}%",
    ]
    if detail.get("rsi"):
        market_inputs.append(f"RSI {round(detail.get('rsi') or 0, 1)}")
    if detail.get("adx"):
        market_inputs.append(f"ADX {round(detail.get('adx') or 0, 1)}")
    if detail.get("funding_rate") is not None:
        market_inputs.append(f"funding at signal {detail.get('funding_rate')}")
    if ctx.get("funding_now") is not None:
        market_inputs.append(f"funding now {ctx.get('funding_now')}")
    execution_inputs = [
        f"execution score {detail.get('execution_score') if detail.get('execution_score') is not None else '—'}",
        f"session warning {detail.get('session_warning') or 'none'}",
        f"skip rule {detail.get('skip_rule') or 'none'}",
        f"reward to TP1 {ctx.get('reward_to_tp1_pct') if ctx.get('reward_to_tp1_pct') is not None else '—'}% · risk to SL {ctx.get('risk_to_sl_pct') if ctx.get('risk_to_sl_pct') is not None else '—'}%",
    ]
    if confluence:
        execution_inputs.append("confluence: " + " | ".join(confluence))
    return {
        "signal": signal_inputs,
        "market": market_inputs,
        "execution": execution_inputs,
    }


def _build_prompt(detail: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    inputs = _build_inputs_used(detail, ctx)
    confluence = "\n".join(f"- {item}" for item in (detail.get("confluence") or [])[:8]) or "- none"
    exec_factors = detail.get("execution_factors") or {}
    exec_factor_text = "\n".join(f"- {k}: {round(v, 1)}" for k, v in exec_factors.items()) or "- none"
    return f"""You are a discretionary crypto trade reviewer.
Decide whether this signal should be acted on RIGHT NOW.
You are not validating data quality; you are deciding TAKE, WAIT, or AVOID.

=== SIGNAL ===
Symbol: {detail.get('symbol')}
Direction: {detail.get('direction')}
Strategy: {detail.get('strategy')}
Confidence: {detail.get('confidence')}
Alpha grade: {detail.get('alpha_grade')}
Regime at signal: {detail.get('regime')}
Setup class: {detail.get('setup_class')}
Sector: {detail.get('sector')}

=== TRADE STRUCTURE ===
Entry zone: {detail.get('entry_low')} -> {detail.get('entry_high')}
Stop loss: {detail.get('stop_loss')}
TP1: {detail.get('tp1')}
TP2: {detail.get('tp2')}
TP3: {detail.get('tp3')}
Raw R/R: {detail.get('rr_ratio')}
Adjusted R/R: {detail.get('fee_adjusted_rr')}
Expected fill: {detail.get('expected_fill_low')} -> {detail.get('expected_fill_high')}
Skip rule: {detail.get('skip_rule')}
Session warning: {detail.get('session_warning')}

=== LIVE DECISION CONTEXT ===
Exec state: {ctx.get('state')}
Current price: {ctx.get('current_price')}
Current regime: {ctx.get('regime_now')}
Regime changed: {ctx.get('regime_changed')}
In entry zone: {ctx.get('in_entry_zone')}
Distance to zone: {ctx.get('distance_to_zone_pct')}%
Reward to TP1: {ctx.get('reward_to_tp1_pct')}%
Risk to SL: {ctx.get('risk_to_sl_pct')}%
BTC weekly bearish: {ctx.get('btc_weekly_bearish')}
BTC 4h bouncing: {ctx.get('btc_4h_bouncing')}

=== INDICATORS ===
RSI: {detail.get('rsi')}
ADX: {detail.get('adx')}
Funding at signal: {detail.get('funding_rate')}
Funding now: {ctx.get('funding_now')}
Volume ratio: {detail.get('vol_ratio')}

=== CONFLUENCE ===
{confluence}

=== EXECUTION GATE ===
Execution score: {detail.get('execution_score')}
Execution factors:
{exec_factor_text}
Bad factors: {detail.get('execution_bad_factors')}
Kill combo: {detail.get('execution_kill_combo')}

Rules:
- TAKE only if reward/risk is still attractive AND timing is acceptable now.
- WAIT if the thesis may be fine but price location/timing is not ideal yet.
- AVOID if the setup is too degraded, crowded, late, or structurally weak.
- Mention the single most decision-relevant reason first.
- Be strict and concise.

Return ONLY valid JSON:
{{
  "verdict": "TAKE|WAIT|AVOID",
  "confidence": 0,
  "summary": "one sentence",
  "strengths": ["point 1", "point 2"],
  "risks": ["point 1", "point 2"],
  "entry_guidance": "specific execution advice for now"
}}

Context used for the decision:
Signal:
{chr(10).join("- " + s for s in inputs["signal"])}
Market:
{chr(10).join("- " + s for s in inputs["market"])}
Execution:
{chr(10).join("- " + s for s in inputs["execution"])}"""


async def _call_ai(prompt: str, api_key: str = "") -> Dict[str, Any]:
    raw = await _free_llm_call(prompt, model=_MODEL_FAST, temperature=0.1, max_tokens=500)
    if not raw:
        raise ValueError("No response from AI provider")
    result = _parse_json_response(raw)
    if result is None:
        raise ValueError(f"Failed to parse AI response: {raw[:200]}")
    return result


def _rule_based_fallback(detail: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    strengths = []
    risks = []
    score = 0

    confidence = float(detail.get("confidence") or 0)
    fee_rr = float(detail.get("fee_adjusted_rr") or detail.get("rr_ratio") or 0)
    exec_score = float(detail.get("execution_score") or 0)

    if confidence >= 70:
        strengths.append(f"High model confidence ({round(confidence)})")
        score += 1
    elif confidence < 55:
        risks.append(f"Confidence is only {round(confidence)}")

    if fee_rr >= 2:
        strengths.append(f"Reward/risk remains acceptable ({round(fee_rr, 1)}R)")
        score += 1
    else:
        risks.append(f"Reward/risk is thin ({round(fee_rr, 1)}R)")

    if ctx.get("in_entry_zone"):
        strengths.append("Price is already inside the entry zone")
        score += 2
    elif ctx.get("distance_to_zone_pct", 0) <= 1:
        risks.append(f"Price is still {ctx.get('distance_to_zone_pct')}% away from entry")
    else:
        risks.append(f"Price is extended {ctx.get('distance_to_zone_pct')}% away from the zone")

    if exec_score >= 50:
        strengths.append(f"Execution gate is supportive ({round(exec_score)}/100)")
        score += 1
    elif exec_score:
        risks.append(f"Execution gate is weak ({round(exec_score)}/100)")

    if detail.get("session_warning"):
        risks.append(str(detail.get("session_warning")))
    if detail.get("execution_kill_combo"):
        risks.append(f"Execution kill combo present: {detail.get('execution_kill_combo')}")
    if detail.get("skip_rule"):
        risks.append(str(detail.get("skip_rule")))

    verdict = "WAIT"
    if score >= 4 and ctx.get("in_entry_zone") and not detail.get("session_warning"):
        verdict = "TAKE"
    elif len(risks) >= 3 and not ctx.get("in_entry_zone"):
        verdict = "AVOID"

    return {
        "verdict": verdict,
        "confidence": min(85, max(35, int(confidence))),
        "summary": risks[0] if verdict != "TAKE" and risks else strengths[0] if strengths else "Mixed setup quality.",
        "strengths": strengths[:3] or ["Signal structure remains acceptable."],
        "risks": risks[:3] or ["No major tactical risk detected."],
        "entry_guidance": (
            "Enter only if price holds inside the entry zone and execution quality stays supportive."
            if verdict == "TAKE"
            else "Wait for price to return to the zone and re-check execution conditions."
            if verdict == "WAIT"
            else "Skip this setup unless market structure materially improves."
        ),
    }


def _normalise_review(verdict: Dict[str, Any], detail: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    clean_verdict = str(verdict.get("verdict") or "WAIT").upper()
    if clean_verdict not in {"TAKE", "WAIT", "AVOID"}:
        clean_verdict = "WAIT"
    confidence = int(max(0, min(100, verdict.get("confidence") or 0)))
    strengths = verdict.get("strengths") if isinstance(verdict.get("strengths"), list) else []
    risks = verdict.get("risks") if isinstance(verdict.get("risks"), list) else []
    return {
        "verdict": clean_verdict,
        "confidence": confidence,
        "summary": str(verdict.get("summary") or "No summary returned."),
        "strengths": [str(x) for x in strengths[:4]],
        "risks": [str(x) for x in risks[:4]],
        "entry_guidance": str(verdict.get("entry_guidance") or ""),
        "inputs_used": _build_inputs_used(detail, ctx),
    }
