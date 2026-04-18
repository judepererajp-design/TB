"""
TitanBot Pro — Mid-Trade Thesis Checker
========================================
When user taps "🔍 Check Signal" during an active trade, this module:

1. Fetches current OHLCV for the coin (last 8 candles on signal timeframe)
2. Compares against the original signal's thesis conditions
3. Checks current regime, funding, BTC pressure, liquidation clusters
4. Calls AI to give a cold-blooded verdict: INTACT / WEAKENED / INVALIDATED
5. Returns a formatted Telegram message with specific evidence

Design principle: the AI is a co-pilot that stays unemotional when you
can't. It reads the same data you see but doesn't have skin in the game.

The check runs in ~5-8 seconds (one AI call, fast model).
"""

import asyncio
import logging
import time
from typing import Optional

from utils.free_llm import call_llm as _free_llm_call, parse_json_response as _parse_json_response, MODEL_FAST as _MODEL_FAST

logger = logging.getLogger(__name__)


async def check_thesis(
    signal_id: int,
    symbol: str,
    direction: str,
    strategy: str,
    entry_price: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    created_at: float,          # signal publish time
    activated_at: float,        # entry fill time
    raw_data: dict,             # original signal raw_data
    current_price: float,
    current_r: float,           # current P&L in R
    max_r: float,               # peak R seen
    openrouter_api_key: str = "",  # kept for backwards compatibility, ignored
) -> str:
    """
    Run a full thesis check and return a formatted Telegram message.
    Called when user presses "🔍 Check Signal" mid-trade.
    """
    try:
        # ── Gather current market context ─────────────────────────
        context = await _gather_context(symbol, direction, strategy, raw_data)

        # ── Build the AI prompt ───────────────────────────────────
        prompt = _build_prompt(
            symbol, direction, strategy,
            entry_price, stop_loss, tp1, tp2,
            created_at, activated_at,
            raw_data, current_price, current_r, max_r,
            context,
        )

        # ── Call AI ───────────────────────────────────────────────
        verdict_json = await _call_ai(prompt, openrouter_api_key)

        # ── Format output ─────────────────────────────────────────
        return _format_verdict(
            symbol, direction, current_price, current_r,
            stop_loss, tp1, verdict_json, context
        )

    except Exception as e:
        logger.error(f"thesis_checker error for {symbol}: {e}")
        return (
            f"🔍 <b>Signal Check — {symbol}</b>\n\n"
            f"⚠️ Could not complete check: {str(e)[:100]}\n\n"
            f"<i>SL at <code>{stop_loss:.6g}</code> is still your arbiter.</i>"
        )


async def _gather_context(symbol: str, direction: str,
                           strategy: str, raw_data: dict) -> dict:
    """Fetch current market data for comparison against original signal."""
    ctx = {}

    # Current regime
    try:
        from analyzers.regime import regime_analyzer
        ctx["regime"] = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        ctx["regime_changed"] = (
            ctx["regime"] != raw_data.get("regime_at_signal", ctx["regime"])
        )
    except Exception:
        ctx["regime"] = "UNKNOWN"
        ctx["regime_changed"] = False

    # Current funding rate (for FundingRateArb signals)
    try:
        from analyzers.derivatives import derivatives_analyzer
        coin = symbol.replace("/USDT", "").replace("/BUSD", "")
        deriv = derivatives_analyzer.get_coin_intel(coin)
        if deriv:
            ctx["funding_now"]      = deriv.funding_rate
            ctx["funding_original"] = raw_data.get("funding_rate", 0)
            ctx["funding_flipped"]  = (
                (ctx["funding_original"] < 0 and ctx["funding_now"] > 0.0001)
                or (ctx["funding_original"] > 0 and ctx["funding_now"] < -0.0001)
            )
    except Exception:
        pass

    # BTC pressure
    try:
        from core.price_cache import price_cache
        btc = price_cache.get("BTC/USDT")
        if btc:
            ctx["btc_price"] = btc
        from analyzers.htf_guardrail import htf_guardrail
        ctx["btc_weekly_bearish"] = htf_guardrail._weekly_bearish
        ctx["btc_4h_bouncing"]    = htf_guardrail._btc_4h_bouncing
    except Exception:
        pass

    # Liquidation cluster around SL — already in cache
    try:
        from signals.liquidation_heatmap import liquidation_heatmap as _lh
        clusters = _lh._cluster_cache.get(symbol, [])
        ctx["liq_clusters"] = len(clusters)
        # FIX: previous divisor `max(stop_loss, 1)` broke for low-priced symbols
        # (e.g. SHIB at $0.000025 → divisor floors to 1.0 → abs-price-diff/1 is
        # always <0.005, so cluster_near_sl was always True). Use the stop_loss
        # value itself with a tiny epsilon, so the threshold is a true percentage.
        _sl = float(raw_data.get("stop_loss") or 0.0)
        if _sl > 0:
            ctx["liq_cluster_near_sl"] = any(
                abs(c.price_low - _sl) / _sl < 0.005
                for c in clusters
            )
        else:
            ctx["liq_cluster_near_sl"] = False
    except Exception:
        ctx["liq_clusters"] = 0
        ctx["liq_cluster_near_sl"] = False

    # Recent OHLCV summary (last 4 candles on 1h)
    try:
        from data.api_client import api_client
        ohlcv = await api_client.fetch_ohlcv(symbol, "1h", limit=6)
        if ohlcv and len(ohlcv) >= 4:
            recent = ohlcv[-4:]
            closes = [c[4] for c in recent]
            opens  = [c[1] for c in recent]
            highs  = [c[2] for c in recent]
            lows   = [c[3] for c in recent]
            ctx["candles_summary"] = {
                "closes":    [round(c, 6) for c in closes],
                "direction": "up" if closes[-1] > closes[0] else "down",
                "wick_hunt": (
                    direction == "LONG" and min(lows) < raw_data.get("stop_loss", 0) * 1.002
                ) or (
                    direction == "SHORT" and max(highs) > raw_data.get("stop_loss", 0) * 0.998
                ),
            }
    except Exception:
        pass

    return ctx


def _build_prompt(
    symbol, direction, strategy,
    entry, sl, tp1, tp2,
    created_at, activated_at,
    raw_data, current_price, current_r, max_r,
    ctx
) -> str:
    import time as _t

    mins_in_trade = int((_t.time() - (activated_at or created_at)) / 60)
    entry_mid = entry or (raw_data.get("entry_low", 0) + raw_data.get("entry_high", 0)) / 2

    regime_at_signal = raw_data.get("regime_at_signal", "UNKNOWN")
    regime_now       = ctx.get("regime", "UNKNOWN")
    regime_change    = f"CHANGED ({regime_at_signal} → {regime_now})" if ctx.get("regime_changed") else f"Unchanged ({regime_now})"

    funding_block = ""
    if "funding_now" in ctx:
        fn = ctx["funding_now"] * 100
        fo = ctx.get("funding_original", 0) * 100
        flipped = ctx.get("funding_flipped", False)
        funding_block = (
            f"Funding at signal: {fo:+.4f}%/8h\n"
            f"Funding now:       {fn:+.4f}%/8h\n"
            f"Funding flipped:   {'YES ⚠️' if flipped else 'No'}\n"
        )

    candles_block = ""
    cs = ctx.get("candles_summary", {})
    if cs:
        closes_str = " → ".join(f"{c:.4g}" for c in cs.get("closes", []))
        candles_block = (
            f"Last 4 candle closes: {closes_str}\n"
            f"Price direction: {cs.get('direction', '?')}\n"
            f"Wick hunted SL area: {'YES' if cs.get('wick_hunt') else 'No'}\n"
        )

    return f"""You are a trading analyst doing a mid-trade thesis check.
A user is {mins_in_trade} minutes into a live trade and wants to know if the original thesis still holds.
Be cold, specific, and grounded in the numbers. No hedging. No "it depends."

=== ORIGINAL SIGNAL ===
Symbol:    {symbol}
Direction: {direction}
Strategy:  {strategy}
Entry:     {entry_mid:.6g}
SL:        {sl:.6g}
TP1:       {tp1:.6g}
TP2:       {tp2:.6g}
Confidence at signal: {raw_data.get('confidence', '?')}

=== CURRENT STATE ===
Current price: {current_price:.6g}
P&L now:       {current_r:+.2f}R
Peak R seen:   {max_r:+.2f}R
Time in trade: {mins_in_trade} minutes

Regime: {regime_change}
{funding_block}{candles_block}
BTC weekly bearish: {ctx.get('btc_weekly_bearish', '?')}
BTC 4h bouncing: {ctx.get('btc_4h_bouncing', '?')}
Liq clusters near SL: {'Yes' if ctx.get('liq_cluster_near_sl') else 'No'}

=== THESIS CHECK RULES ===
For FundingRateArb: thesis INVALIDATED if funding flipped direction.
For IchimokuCloud: thesis WEAKENED if price closed below cloud (long) or above (short).
For ElliottWave: thesis INVALIDATED if Wave 2 low (long) or high (short) is taken.
For SMC/Breakout: thesis WEAKENED if price returned to breakout zone.
For all: regime change to VOLATILE during a LONG = WEAKENED.
For all: SL is always the final arbiter — if thesis is WEAKENED but SL holds, stay.

Return ONLY valid JSON:
{{
  "verdict": "INTACT|WEAKENED|INVALIDATED",
  "confidence": 0-100,
  "primary_reason": "<one sentence — most important factor>",
  "supporting_evidence": ["<point 1>", "<point 2>"],
  "sl_still_valid": true|false,
  "action": "HOLD|TIGHTEN_TO_TP1|EXIT_NOW",
  "action_reason": "<why — specific numbers>"
}}"""


async def _call_ai(prompt: str, api_key: str = "") -> dict:
    """Call the fast AI model for thesis verdict via Pollinations.ai (no key needed)."""
    raw = await _free_llm_call(prompt, model=_MODEL_FAST, temperature=0.1, max_tokens=400)
    if not raw:
        raise ValueError("No response from AI provider")
    result = _parse_json_response(raw)
    if result is None:
        raise ValueError(f"Failed to parse AI response: {raw[:200]}")
    return result


def _format_verdict(
    symbol, direction, current_price, current_r,
    stop_loss, tp1, verdict: dict, ctx: dict
) -> str:
    """Format the AI verdict as a Telegram message."""
    v   = verdict.get("verdict", "UNKNOWN")
    conf = verdict.get("confidence", 0)
    reason = verdict.get("primary_reason", "")
    evidence = verdict.get("supporting_evidence", [])
    action = verdict.get("action", "HOLD")
    action_reason = verdict.get("action_reason", "")
    sl_valid = verdict.get("sl_still_valid", True)

    verdict_emoji = {
        "INTACT":      "✅",
        "WEAKENED":    "🟡",
        "INVALIDATED": "❌",
    }.get(v, "⚪")

    action_emoji = {
        "HOLD":         "🛡 Hold position",
        "TIGHTEN_TO_TP1": "🎯 Tighten target to TP1",
        "EXIT_NOW":     "🚪 Consider early exit",
    }.get(action, action)

    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    pnl_emoji = "📈" if current_r >= 0 else "📉"

    msg  = f"🔍 <b>Signal Check — {symbol} {dir_emoji}</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"{verdict_emoji} <b>Thesis: {v}</b>  ({conf}% confident)\n\n"
    msg += f"{pnl_emoji} P&L: <b>{current_r:+.2f}R</b>  ·  Price: <code>{current_price:.6g}</code>\n\n"
    msg += f"<b>Verdict:</b> {reason}\n"

    if evidence:
        msg += "\n<b>Evidence:</b>\n"
        for e in evidence[:3]:
            msg += f"  • {e}\n"

    msg += f"\n<b>Suggestion:</b> {action_emoji}\n"
    if action_reason:
        msg += f"<i>{action_reason}</i>\n"

    if not sl_valid:
        msg += f"\n⚠️ <b>SL may be compromised</b> — review <code>{stop_loss:.6g}</code>\n"

    # Regime note
    if ctx.get("regime_changed"):
        msg += f"\n🔄 Regime changed since signal — factor this in.\n"

    msg += f"\n<i>SL at <code>{stop_loss:.6g}</code> is still your final exit rule.</i>"

    return msg
