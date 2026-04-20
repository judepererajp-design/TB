"""
TitanBot Pro — Telegram Formatter (v4 — Interactive Trading Terminal)
======================================================================
Overhaul based on UX spec:
  - Compact 10-line default signal card
  - Analysis hidden behind buttons (Market, Trade Plan, Logic, Metrics, Profile)
  - Lifecycle messages (setup → entry → active → TP1 → closed)
  - Market digest replaces regime spam
  - Zero redundancy
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# Entry-status thresholds — must match frontend (index.html) and backend (execution_engine.py)
_ENTRY_LATE_PCT: float = 2.0     # overshoot ≤ this → LATE ENTRY; > this → EXTENDED

from signals.aggregator import ScoredSignal
from analyzers.regime import regime_analyzer
from analyzers.altcoin_rotation import rotation_tracker
from config.loader import cfg
from utils.formatting import fmt_price
from utils.signal_guidance import guidance_payload
from strategies.base import direction_str


TF_MINUTES = {
    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440,
}

# Maps setup_class → recommended entry timeframe and confirmation timeframe.
# Kept in sync with strategies/base.py SETUP_CLASS_ENTRY_TF / SETUP_CLASS_CONFIRM_TF.
_SC_ENTRY_TF: Dict[str, str] = {
    'scalp':      '15m',
    'intraday':   '1h',
    'swing':      '4h',
    'positional': '1d',
}
_SC_CONFIRM_TF: Dict[str, str] = {
    'scalp':      '5m',
    'intraday':   '15m',
    'swing':      '1h',
    'positional': '4h',
}

STRATEGY_EXPIRY_CANDLES = {
    'SmartMoneyConcepts':    6,
    'WyckoffAccDist':        8,
    'Wyckoff':               8,   # Legacy alias
    'Ichimoku':              8,   # BUG-12: was 5 — Ichimoku is swing, not scalp
    'IchimokuCloud':         8,   # Legacy alias
    'ElliottWave':           6,
    'InstitutionalBreakout': 4,
    'Momentum':              3,
    'MomentumContinuation':  3,   # Legacy alias
    'FundingRateArb':        4,
    'PriceAction':           4,
    'ExtremeReversal':       6,
    'MeanReversion':         4,
    'RangeScalper':          8,
    'HarmonicPattern':       6,
    'GeometricPattern':      6,
}


def _trade_management_sentence(
    setup_class: str,
    strategy: str,
    regime: str,
    direction: str,
    rr_ratio: float,
    tp1: float,
    tp2: float,
    tp3,
    entry_mid: float,
) -> str:
    """
    Generate a single, plain-English trade management instruction
    from first principles — replacing the raw timeframe label.

    Examples:
      "Swing trade. Move SL to BE after TP1, target TP2 over 2–4 days."
      "Intraday. Close by end of session if TP1 not reached. Tight trail."
      "Scalp. Take full size at TP1. Do not hold overnight."
    """
    # ── Horizon phrase ───────────────────────────────────────
    horizon = {
        'swing':    'Swing trade',
        'intraday': 'Intraday setup',
        'scalp':    'Scalp',
    }.get(setup_class, 'Setup')

    # ── TP management ────────────────────────────────────────
    # How far is TP2 as a % move?
    tp2_pct = abs(tp2 - entry_mid) / entry_mid * 100 if entry_mid else 0

    if setup_class == 'scalp':
        tp_action = "Take full size at TP1. Do not hold overnight."
    elif rr_ratio >= 3.0:
        # High RR — worth running for TP2/TP3
        if tp3:
            tp_action = "Scale out: ⅓ at TP1, ⅓ at TP2, trail remainder to TP3."
        else:
            tp_action = "Take ½ at TP1, move SL to BE, target TP2 in full."
    else:
        # Moderate RR — protect at TP1
        tp_action = "Move SL to BE after TP1 hits. Target TP2."

    # ── Hold time phrase ─────────────────────────────────────
    hold = {
        'swing':    '2–5 days',
        'intraday': 'end of session',
        'scalp':    '2–4 hours',
    }.get(setup_class, '')

    # ── Regime modifier ──────────────────────────────────────
    regime_note = ""
    if regime == "CHOPPY":
        if setup_class == 'swing':
            regime_note = " Range market — TP1 is the priority, TP2 is a bonus."
        else:
            regime_note = " Choppy conditions — don't overstay."
    elif regime == "BEAR_TREND" and direction == "LONG":
        regime_note = " Counter-trend long — TP1 only, no trailing."
    elif regime == "BULL_TREND" and direction == "SHORT":
        regime_note = " Counter-trend short — TP1 only, no trailing."
    elif regime == "VOLATILE":
        regime_note = " High volatility — reduce size, TP1 target only."
    elif regime == "VOLATILE_PANIC":
        regime_note = " 🚨 Panic regime — emergency sizing, TP1 only, no trailing."
    elif regime in ("BULL_TREND", "BEAR_TREND") and setup_class == 'swing':
        regime_note = f" Trending market — give it room to run."

    # ── Strategy-specific nuance ─────────────────────────────
    strat_note = ""
    if 'Elliott' in strategy:
        strat_note = " Wave count invalidates on close past SL."
    elif 'Wyckoff' in strategy:
        strat_note = " Wait for spring/upthrust retest before adding."
    elif 'Funding' in strategy:
        strat_note = " Close if funding flips neutral."
    elif 'Ichimoku' in strategy and setup_class == 'swing':
        strat_note = " Cloud support/resistance acts as dynamic TP guide."
    elif 'MeanReversion' in strategy or 'Range' in strategy:
        strat_note = " Mean reversion — take profits early, don't trail."

    # ── Assemble ─────────────────────────────────────────────
    hold_clause = f" Hold up to {hold}." if hold else ""
    sentence = f"{horizon}.{hold_clause} {tp_action}{regime_note}{strat_note}"
    return sentence.strip()

STRATEGY_PERSONALITY = {
    'SmartMoneyConcepts': {
        'type': 'Smart Money Reversal / Continuation',
        'duration': '8-24 hours',
        'pullbacks': 'Expected - price may retest OB before moving',
        'win_style': 'Moderate frequency / high R:R',
        'exit_note': 'TP2 is primary target. Trail stop after TP1.',
    },
    'Wyckoff': {
        'type': 'Accumulation / Distribution Phase Exit',
        'duration': '24-72 hours',
        'pullbacks': 'Minimal after Spring/UTAD - move should be impulsive',
        'win_style': 'Low frequency / very high R:R',
        'exit_note': 'Let it run to TP3. This is a big structural move.',
    },
    'Ichimoku': {
        'type': 'Trend Continuation (cloud support)',
        'duration': '12-48 hours',
        'pullbacks': 'Shallow pullbacks to Kijun expected - hold through',
        'win_style': 'Moderate frequency / moderate R:R',
        'exit_note': 'Exit if Chikou crosses back into price.',
    },
    'IchimokuCloud': {  # Legacy alias
        'type': 'Trend Continuation (cloud support)',
        'duration': '12-48 hours',
        'pullbacks': 'Shallow pullbacks to Kijun expected - hold through',
        'win_style': 'Moderate frequency / moderate R:R',
        'exit_note': 'Exit if Chikou crosses back into price.',
    },
    'ElliottWave': {
        'type': 'Impulse Wave Entry (W3 or W5)',
        'duration': '6-18 hours',
        'pullbacks': 'Low in W3. W5 may show early reversal - watch divergence.',
        'win_style': 'Low frequency / excellent R:R on W3',
        'exit_note': 'W3: target 161.8% extension. W5: exit before RSI diverges.',
    },
    'InstitutionalBreakout': {
        'type': 'Volume Breakout Continuation',
        'duration': '2-8 hours',
        'pullbacks': 'Retest of breakout level is common - valid if holds',
        'win_style': 'Moderate / moderate R:R',
        'exit_note': 'Exit if price closes back inside channel.',
    },
    'Momentum': {
        'type': 'Trend Momentum Surge',
        'duration': '1-4 hours',
        'pullbacks': 'Shallow - do not exit early',
        'win_style': 'High frequency / lower R:R',
        'exit_note': 'Exit if MACD histogram turns negative.',
    },
    'MomentumContinuation': {  # Legacy alias
        'type': 'Trend Momentum Surge',
        'duration': '1-4 hours',
        'pullbacks': 'Shallow - do not exit early',
        'win_style': 'High frequency / lower R:R',
        'exit_note': 'Exit if MACD histogram turns negative.',
    },
    'FundingRateArb': {
        'type': 'Crowd Fade / Funding Reset',
        'duration': '4-16 hours',
        'pullbacks': 'Initial move can be sharp - do not chase if missed zone',
        'win_style': 'Low frequency / high conviction',
        'exit_note': 'Watch for funding normalization - that is your exit signal.',
    },
    'PriceAction': {
        'type': 'Candlestick Reversal at Key Level',
        'duration': '2-8 hours',
        'pullbacks': 'Unlikely - pattern is already the rejection',
        'win_style': 'High frequency / moderate R:R',
        'exit_note': 'Quick moves expected. Take TP1 early.',
    },
    'ExtremeReversal': {
        'type': 'Extreme Oversold/Overbought Fade',
        'duration': '1-3 hours',
        'pullbacks': 'Initial reversal may be sharp',
        'win_style': 'High frequency / low R:R scalp',
        'exit_note': 'Target Bollinger midband (TP2). Do not overstay.',
    },
    'MeanReversion': {
        'type': 'Z-Score Extreme Fade',
        'duration': '2-6 hours',
        'pullbacks': 'Trade may retrace toward mean gradually',
        'win_style': 'High frequency / moderate R:R',
        'exit_note': 'TP2 is the mean - that is the full target.',
    },
    'RangeScalper': {
        'type': 'Range Boundary Bounce',
        'duration': '0.5-2 hours',
        'pullbacks': 'Move to midpoint is the target - do not hold past it',
        'win_style': 'High frequency / low R:R',
        'exit_note': 'Exit at equilibrium. This is a scalp only.',
    },
}

GRADE_CONFIG = {
    "A+": ("⚡", "A+ SETUP — ENTER NOW",
           "👉 Execute immediately. Multi-cluster alignment. Rare — do not skip."),
    "A":  ("🔥", "A SETUP — HIGH CONVICTION",
           "👉 Prepare your order. Entry on first confirmation trigger."),
    "B+": ("📊", "B+ SETUP — SOLID SETUP",
           "👉 Monitoring for entry. Levels revealed when price approaches."),
    "B":  ("📊", "B SETUP — DEVELOPING",
           "👉 Setup identified. Monitoring for entry triggers."),
    "C":  ("📋", "C SETUP — CONTEXT SIGNAL",
           "👉 Informational only. Wait for A/B grade before risking capital."),
}


def _confidence_label(score: float) -> str:
    if score >= 80:
        return "🟢 STRONG"
    elif score >= 70:
        return "🟡 GOOD"
    elif score >= 65:
        return "🔵 OK"
    return "🔴 WEAK"


def _entry_status(entry_low: float, entry_high: float, direction: str, current_price: float) -> str:
    """Return entry status emoji+label based on current price vs entry zone.

    Returns empty string when price has not yet reached the zone (normal waiting state).
    """
    if current_price <= 0:
        return ""
    if entry_low <= current_price <= entry_high:
        return "🟢 IN ZONE"
    if direction == "LONG":
        if current_price < entry_low:
            return ""  # approaching — normal waiting state
        overshoot = (current_price - entry_high) / entry_high * 100
        return "🟡 LATE ENTRY" if overshoot <= _ENTRY_LATE_PCT else "🔴 EXTENDED"
    else:  # SHORT
        if current_price > entry_high:
            return ""  # approaching — normal waiting state
        overshoot = (entry_low - current_price) / entry_low * 100
        return "🟡 LATE ENTRY" if overshoot <= _ENTRY_LATE_PCT else "🔴 EXTENDED"


def _live_rr_ratio(
    entry_low: float, entry_high: float,
    stop_loss: float, tp1: float, tp2: float,
    direction: str, current_price: float,
) -> Optional[float]:
    """Compute live R/R using current price as the effective entry, targeting TP2 (or TP1).

    Only meaningful when price is at or past the entry zone.  Returns None when
    price has not yet reached the zone or when risk would be zero/negative.
    """
    if current_price <= 0 or not stop_loss:
        return None
    target = tp2 or tp1
    if not target:
        return None
    if direction == "LONG":
        risk = current_price - stop_loss
        reward = target - current_price
    else:
        risk = stop_loss - current_price
        reward = current_price - target
    if risk <= 0:
        return None
    return reward / risk


class TelegramFormatter:
    """Generates all formatted Telegram messages — Interactive Terminal Edition."""

    def format_signal(
        self,
        scored: ScoredSignal,
        signal_id: int = 0,
        alpha_score=None,
        prob_estimate=None,
        sizing=None,
        confluence=None,
        current_price: float = 0.0,
    ) -> str:
        sig = scored.base_signal
        direction = direction_str(sig)
        # PHASE 3 AUDIT FIX (P3-4): Use alpha model grade (the grade that drives
        # execution gating) instead of the aggregator confidence grade.  This
        # ensures the Telegram card shows the same grade that execution_engine
        # and all downstream systems use.  Falls back to scored.grade when
        # alpha_score is unavailable (e.g. backtester, test paths).
        grade = alpha_score.grade if alpha_score and hasattr(alpha_score, 'grade') else scored.grade
        conf = scored.final_confidence or sig.confidence
        raw = sig.raw_data or {}
        strategy = sig.strategy or "Unknown"
        if sizing and getattr(sizing, "position_size_usdt", 0) > 0:
            raw["intended_size_usd"] = sizing.position_size_usdt
        guidance = guidance_payload(
            sig,
            confluence=sig.confluence,
            current_session=getattr(regime_analyzer.session, 'value', ''),
        )

        dir_emoji = "🟢" if direction == "LONG" else "🔴"
        dir_label = "LONG ▲" if direction == "LONG" else "SHORT ▼"
        entry_mid = (sig.entry_low + sig.entry_high) / 2
        tp2_pct = abs(sig.tp2 - entry_mid) / entry_mid * 100 if entry_mid else 0

        grade_emoji, grade_label, grade_action = GRADE_CONFIG.get(grade, GRADE_CONFIG["B"])
        conf_label = _confidence_label(conf)
        regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        regime_readable = {
            "BULL_TREND": "Trending", "BEAR_TREND": "Bear Trend",
            "CHOPPY": "Range / Chop", "VOLATILE": "Volatile", "VOLATILE_PANIC": "Risk-Off",
        }.get(regime, regime)

        kz_line = ""
        if scored.is_killzone:
            kz_line = f" ·  ⏰ {getattr(regime_analyzer.session, 'value', 'UNKNOWN')} Killzone +{scored.killzone_bonus:.0f}"

        _sc_label = {"swing": "Swing", "intraday": "Intraday", "scalp": "Scalp"}.get(
            getattr(sig, 'setup_class', 'intraday'),
            getattr(sig, 'setup_class', 'Intraday').capitalize()
        )

        candles = STRATEGY_EXPIRY_CANDLES.get(strategy, 6)
        tf_mins = TF_MINUTES.get(getattr(sig, 'entry_timeframe', sig.timeframe), 60)
        total_mins = candles * tf_mins
        h = total_mins // 60
        m = total_mins % 60
        duration_str = f"{h}h" if not m else f"{h}h{m}m"

        if any(k in strategy for k in ('Breakout',)):
            entry_type = "Breakout"
        elif raw.get('has_ob') or raw.get('wave_type') == 'WAVE3':
            entry_type = "Pullback"
        else:
            entry_type = "Immediate"

        warning_lines: List[str] = []
        if scored.volume_score < 45:
            warning_lines.append("⚠️ Weak volume confirmation — wait.")
        elif regime == "VOLATILE":
            warning_lines.append("⚠️ High volatility — reduce size, TP1 target only.")
        elif regime == "BEAR_TREND" and direction == "LONG":
            warning_lines.append("⚠️ LONG in bear trend — use TP1 as target.")
        if guidance.get("session_warning"):
            warning_lines.append(guidance["session_warning"])

        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")

        # ── C grade: narrative only, zero trade numbers ──────────────────────
        if grade == "C":
            warning_lines = []  # C = informational only; risk warnings are contradictory
            pressure = "upward" if direction == "LONG" else "downward"
            strategy_narrative = {
                'SmartMoneyConcepts':    "Smart money positioning",
                'Ichimoku':              "Trend alignment forming",
                'IchimokuCloud':         "Trend alignment forming",
                'ElliottWave':           "Wave structure developing",
                'WyckoffAccDist':        "Accumulation / distribution phase",
                'Wyckoff':               "Accumulation / distribution phase",
                'InstitutionalBreakout': "Breakout pressure building",
                'Momentum':              "Momentum building",
                'MomentumContinuation':  "Momentum building",
                'FundingRateArb':        "Funding extreme forming",
                'MeanReversion':         "Statistical deviation detected",
                'PriceAction':           "Price structure forming",
                'ExtremeReversal':       "Extreme condition detected",
                'RangeScalper':          "Range boundary approached",
                'HarmonicPattern':       "Harmonic pattern completing",
                'GeometricPattern':      "Chart pattern forming",
            }.get(strategy, "Setup forming")
            msg  = f"{grade_emoji} <b>{grade_label}</b>\n"
            msg += f"<i>{grade_action}</i>\n\n"
            msg += f"{dir_emoji} <b>{dir_label}  {sig.symbol}  ·  {_sc_label}</b>{kz_line}\n\n"
            msg += f"{strategy_narrative}.\n"
            msg += f"Potential {pressure} pressure developing.\n\n"
            msg += f"Confidence: {conf_label} ({conf:.0f})\n"
            msg += f"Strategy: <i>{strategy}</i>\n"
            _sc_c = getattr(sig, 'setup_class', 'intraday')
            _etf_c = getattr(sig, 'entry_timeframe', None) or _SC_ENTRY_TF.get(_sc_c, '1h')
            msg += f"Chart: {_etf_c} entry  ·  {_SC_CONFIRM_TF.get(_sc_c, '15m')} confirm\n"
            msg += f"Regime: {regime_readable}\n\n"
            msg += f"⏳ Monitoring for {candles} candles ({duration_str})"
            if warning_lines:
                msg += "\n" + "\n".join(warning_lines)
            msg += f"\n⏱ {now_utc}  |  #{signal_id}"
            return msg

        # ── Live entry context (when publish_price is available) ─────────────
        _zone_status = _entry_status(sig.entry_low, sig.entry_high, direction, current_price)
        _live_rr     = _live_rr_ratio(
            sig.entry_low, sig.entry_high, sig.stop_loss, sig.tp1, sig.tp2,
            direction, current_price,
        ) if _zone_status else None  # only compute when price is at/past zone
        _zone_suffix = f"  {_zone_status}" if _zone_status else ""
        _adj_rr = guidance.get("fee_adjusted_rr")
        _rr_line: str
        if _live_rr is not None and abs(_live_rr - sig.rr_ratio) > 0.1:
            _rr_line = f"R/R:   {sig.rr_ratio:.1f}R (planned)  ·  {_live_rr:.1f}R (live)\n"
        else:
            _rr_line = f"R/R:   {sig.rr_ratio:.1f}R\n"
        _adj_rr_line = f"Adj R/R: {_adj_rr:.1f}R after fees/slippage\n" if _adj_rr is not None else ""
        _skip_line = f"Skip: {guidance['skip_rule']}\n" if guidance.get("skip_rule") else ""

        # ── V11: Grade-aware level visibility ────────────────────────
        # A+ = everything shown (execute immediately)
        # A  = zone + SL shown (prepare order, TPs behind button)
        # B+/B = R:R + sizing only (await triggers, full levels at APPROACHING)
        _pa_badge = "  ⚡ <b>Power Aligned</b>" if getattr(scored, 'power_aligned', False) else ""
        msg  = f"{grade_emoji} <b>{grade_label}</b>{_pa_badge}\n"
        msg += f"<i>{grade_action}</i>\n\n"
        msg += f"{dir_emoji} <b>{dir_label}  {sig.symbol}  ·  {_sc_label}</b>{kz_line}\n\n"
        msg += f"Strategy: <i>{strategy}</i>\n"
        _sc = getattr(sig, 'setup_class', 'intraday')
        _entry_tf   = getattr(sig, 'entry_timeframe', None) or _SC_ENTRY_TF.get(_sc, '1h')
        _confirm_tf = _SC_CONFIRM_TF.get(_sc, '15m')
        msg += f"Chart:    {_entry_tf} entry  ·  {_confirm_tf} confirm\n"

        if grade == "A+":
            # A+ — FULL LEVELS (execute immediately)
            msg += f"Entry: {entry_type}\n"
            msg += f"Zone:  <code>{fmt_price(sig.entry_low)}</code> — <code>{fmt_price(sig.entry_high)}</code>{_zone_suffix}  <i>(expected fill)</i>\n"
            msg += f"SL:    <code>{fmt_price(sig.stop_loss)}</code>\n"
            msg += f"TP1:   <code>{fmt_price(sig.tp1)}</code>\n"
            msg += f"TP2:   <code>{fmt_price(sig.tp2)}</code>\n"
            if sig.tp3:
                msg += f"TP3:   <code>{fmt_price(sig.tp3)}</code>\n"
            msg += _rr_line
            msg += _adj_rr_line
            msg += _skip_line
        elif grade == "A":
            # V17: A — show all levels (users need them to set orders)
            msg += f"Entry: {entry_type}\n"
            msg += f"Zone:  <code>{fmt_price(sig.entry_low)}</code> — <code>{fmt_price(sig.entry_high)}</code>{_zone_suffix}  <i>(expected fill)</i>\n"
            msg += f"SL:    <code>{fmt_price(sig.stop_loss)}</code>\n"
            msg += f"TP1:   <code>{fmt_price(sig.tp1)}</code> → move SL to BE\n"
            msg += f"TP2:   <code>{fmt_price(sig.tp2)}</code>\n"
            if sig.tp3:
                msg += f"TP3:   <code>{fmt_price(sig.tp3)}</code>\n"
            msg += _rr_line
            msg += _adj_rr_line
            msg += _skip_line
        else:
            # V17: B/B+ — show zone + SL (users need these to set alerts)
            msg += f"Entry: {entry_type}\n"
            msg += f"Zone:  <code>{fmt_price(sig.entry_low)}</code> — <code>{fmt_price(sig.entry_high)}</code>  <i>(expected fill)</i>\n"
            msg += f"SL:    <code>{fmt_price(sig.stop_loss)}</code>\n"
            msg += f"R/R:   {sig.rr_ratio:.1f}R\n"
            msg += _adj_rr_line
            msg += _skip_line
            msg += f"<i>Awaiting triggers. TP levels at confirmation.</i>\n"

        msg += f"\nConfidence: {conf_label} ({conf:.0f})\n"
        msg += f"Regime: {regime_readable}\n"

        # Fix U1: show quant metrics on card for B/A/A+ (not for C — those are narrative only)
        if prob_estimate and alpha_score:
            p_pct = int(prob_estimate.p_win * 100)
            ev_r  = alpha_score.expected_value_r
            kelly = getattr(alpha_score, 'kelly_fraction', 0)
            msg += f"P(win): {p_pct}%  ·  EV: {ev_r:+.2f}R  ·  Kelly: {kelly:.0%}\n"

        # V14: Show position sizing with leverage context
        if sizing and hasattr(sizing, 'position_size_usdt') and sizing.position_size_usdt > 0:
            _lev = getattr(sizing, 'leverage_suggested', 1)
            _risk_pct_str = f" ({sizing.risk_amount_usdt/sizing.position_size_usdt*100:.1f}%)" if sizing.position_size_usdt > 0 else ""
            msg += f"💰 Size: ${sizing.position_size_usdt:,.0f} (1x)  ·  Risk: ${sizing.risk_amount_usdt:,.0f}{_risk_pct_str}\n"
            if _lev and _lev > 1:
                msg += f"⚠️ At {_lev}x: size=${sizing.position_size_usdt*_lev:,.0f}, risk=${sizing.risk_amount_usdt*_lev:,.0f}\n"

        # Trade management sentence
        mgmt = _trade_management_sentence(
            setup_class=getattr(sig, 'setup_class', 'intraday'),
            strategy=strategy,
            regime=regime,
            direction=direction,
            rr_ratio=sig.rr_ratio,
            tp1=sig.tp1,
            tp2=sig.tp2,
            tp3=sig.tp3,
            entry_mid=entry_mid,
        )
        msg += f"\n💡 {mgmt}"

        # ── Ensemble verdict line ─────────────────────────────────────────────
        # Show a compact summary of what the ensemble voter decided.
        # Only shown when the verdict is BOOST or when CVD/SM both confirmed.
        try:
            _conf_list = confluence or []
            _ensemble_line = next(
                (c for c in _conf_list if "Ensemble" in str(c)), None
            )
            _cvd_line = next(
                (c for c in _conf_list if "CVD" in str(c)), None
            )
            _sm_line = next(
                (c for c in _conf_list if "Smart Money" in str(c)), None
            )
            _liq_line = next(
                (c for c in _conf_list if "Liq cluster" in str(c) or "liq cluster" in str(c)), None
            )
            _oi_line = next(
                (c for c in _conf_list if "OI" in str(c) and ("rising" in str(c).lower() or "conviction" in str(c).lower())), None
            )

            # ── CTX-4: Grouped confluence hierarchy ──────────────────────
            # Tier 1: Structural (highest signal quality)
            # Tier 2: Microstructure (market intel)
            # Tier 3: Supporting (lower weight)
            _conf_list_str = [str(c).strip() for c in (_conf_list or [])]

            _tier1, _tier2, _tier3 = [], [], []
            for c in _conf_list_str:
                cl = c.lower()
                # Tier 1 — structural / HTF / SMC
                if any(x in cl for x in ('htf', 'weekly', '4h', 'bos', 'choch', 'order block',
                                         'ob+fvg', 'sweep', 'premium', 'discount', 'wick',
                                         'wave', 'wyckoff', 'structure', 'ensemble boost')):
                    _tier1.append(c)
                # Tier 2 — microstructure / market intel
                elif any(x in cl for x in ('cvd', 'smart money', 'whale', 'oi', 'funding',
                                           'liq cluster', 'open interest', 'netflow', 'basis')):
                    _tier2.append(c)
                # Tier 3 — supporting
                else:
                    _tier3.append(c)

            if _tier1 or _tier2 or _tier3:
                msg += "\n"
                if _tier1:
                    msg += "\n<b>Structure:</b>"
                    for c in _tier1[:4]:
                        msg += f"\n  {c}"
                if _tier2:
                    msg += "\n<b>Market Intel:</b>"
                    for c in _tier2[:3]:
                        msg += f"\n  {c}"
                if _tier3:
                    msg += "\n<b>Supporting:</b>"
                    for c in _tier3[:3]:
                        msg += f"\n  {c}"
            else:
                # Fallback: original micro_parts path
                _micro_parts = []
                if _cvd_line:
                    _micro_parts.append(str(_cvd_line).strip())
                if _sm_line:
                    _micro_parts.append(str(_sm_line).strip())
                if _oi_line:
                    _micro_parts.append(str(_oi_line).strip())
                _has_micro = bool(_micro_parts or _cvd_line or _sm_line or _oi_line or _liq_line)
                if _micro_parts:
                    msg += f"\n\n<b>Market Intel:</b>"
                    for part in _micro_parts[:3]:
                        msg += f"\n  {part}"
                if _liq_line:
                    msg += f"\n  {str(_liq_line).strip()}"

            if _ensemble_line and "BOOST" in str(_ensemble_line):
                msg += f"\n  {str(_ensemble_line).strip()}"

        except Exception:
            pass

        # ── AI Narrative (Nemotron-3-Super) ──────────────────────────────────
        # If AI analyst generated a narrative for this signal, add it.
        # Falls back gracefully (no narrative shown) when AI is off or unavailable.
        _ai_narrative = sig.raw_data.get('ai_narrative')
        if _ai_narrative and isinstance(_ai_narrative, str) and len(_ai_narrative) > 20:
            msg += f"\n\n🤖 <i>{_ai_narrative.strip()}</i>"

        # ── CTX-2: Funding rate context (perpetuals only) ─────────────────────
        # Shows funding rate when it's meaningful (>0.03%/8h or negative).
        # Helps traders know if they're fighting or riding the funding flow.
        try:
            _fr = float(raw.get('funding_rate', 0))
            _fa = float(raw.get('funding_annual', 0))
            if abs(_fr) >= 0.0003:  # Only show when rate is significant
                _fr_pct = _fr * 100
                _fr_sign = "+" if _fr > 0 else ""
                _fr_color = "🔴" if _fr > 0.001 else ("🟢" if _fr < -0.0003 else "⚪")
                _fr_note = ""
                if _fr > 0.001 and direction == "LONG":
                    _fr_note = " — longs paying premium"
                elif _fr < -0.0003 and direction == "SHORT":
                    _fr_note = " — shorts paying premium"
                elif _fr > 0.001 and direction == "SHORT":
                    _fr_note = " — funding tailwind ✓"
                elif _fr < -0.0003 and direction == "LONG":
                    _fr_note = " — funding tailwind ✓"
                msg += f"\n{_fr_color} Funding: {_fr_sign}{_fr_pct:.4f}%/8h  ({_fa:+.1f}% ann.){_fr_note}"
        except Exception:
            pass

        # ── CTX-7: Volatility context (ATR percentile + expansion state) ──────
        # Tells traders whether they're entering in low or high vol environment.
        try:
            _atr_pct = float(raw.get('atr_pct', 0))
            _vol_rank = float(raw.get('vol_percentile', 0.5))
            _vol_exp = raw.get('vol_expanding', False)
            if _atr_pct > 0:
                _vr_pct = int(_vol_rank * 100)
                _vol_label = (
                    "🔥 High vol" if _vol_rank >= 0.75 else
                    "🔷 Moderate vol" if _vol_rank >= 0.4 else
                    "😴 Low vol"
                )
                _exp_tag = " · expanding" if _vol_exp else (" · contracting" if _vol_rank < 0.35 else "")
                msg += f"\n📊 ATR: {_atr_pct*100:.2f}%  ·  Vol rank: {_vr_pct}th pct  ({_vol_label}{_exp_tag})"
        except Exception:
            pass

        # ── Institutional Flow context ─────────────────────────────────────
        try:
            from analyzers.institutional_flow import institutional_flow as _if_eng
            _if_ctx = _if_eng.get_telegram_context(sig.symbol)
            if _if_ctx:
                msg += f"\n{_if_ctx}"
        except Exception:
            pass

        # ── Liquidation cluster SL check ──────────────────────────────────
        # If SL price is near a liquidation cluster, warn the trader.
        # Pre-fetched sync from LiquidationHeatmap cache (no await needed).
        try:
            from signals.liquidation_heatmap import liquidation_heatmap as _lh
            _cached_clusters = _lh.get_cached_clusters(sig.symbol)
            if _cached_clusters:
                _sl_danger, _sl_widen, _sl_note = _lh.check_sl_near_cluster(
                    sig.stop_loss, _cached_clusters
                )
                if _sl_danger:
                    msg += f"\n🎯 {_sl_note}"
        except Exception:
            pass

        msg += f"\n\n⏳ Valid: {candles} candles ({duration_str})"
        if warning_lines:
            msg += "\n" + "\n".join(warning_lines)
        msg += f"\nInvalidation: Close past <code>{fmt_price(sig.stop_loss)}</code>"

        # I9: BTC context line — gives immediate macro context on every signal
        try:
            from core.price_cache import price_cache as _pc
            btc_price = _pc.get("BTC/USDT")
            if btc_price:
                from analyzers.htf_guardrail import htf_guardrail as _htf
                _btc_bias = "↗ bouncing" if _htf._btc_4h_bouncing else ("↘ trending down" if _htf._btc_4h_resuming_down else "→ neutral")
                msg += f"\n₿ BTC: <code>{fmt_price(btc_price)}</code>  ({_btc_bias})"
        except Exception:
            pass

        # I8: TradingView link for quick chart verification
        _tv_symbol = sig.symbol.replace("/USDT", "USDT").replace("/", "")
        msg += f"\n📊 <a href=\"https://www.tradingview.com/chart/?symbol=BINANCE:{_tv_symbol}PERP\">View on TradingView</a>"

        msg += f"\n⏱ {now_utc}  |  #{signal_id}"

        return msg

    def format_market_panel(self, scored: ScoredSignal) -> str:
        regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        fg = regime_analyzer.fear_greed
        alt = regime_analyzer.alt_season_score

        trend = {
            "BULL_TREND": "Bullish", "BEAR_TREND": "Bearish",
            "CHOPPY": "Ranging", "VOLATILE": "Volatile", "VOLATILE_PANIC": "Risk-Off",
        }.get(regime, regime)

        if regime in ("VOLATILE", "VOLATILE_PANIC"):
            vol, risk = "High", "HIGH"
        elif regime == "CHOPPY":
            vol, risk = "Low", "LOW"
        else:
            vol, risk = "Normal", "NORMAL"

        if fg < 25:
            liq = "Extreme Fear"
        elif fg > 75:
            liq = "Extreme Greed"
        else:
            liq = "Normal"

        sig = scored.base_signal
        strategy = sig.strategy or "Unknown"
        raw = sig.raw_data or {}
        direction = direction_str(sig)
        edge_types = []
        if raw.get('has_ob') and raw.get('has_fvg'):
            edge_types.append("Smart Money")
        elif raw.get('has_sweep'):
            edge_types.append("Liquidity")
        if 'Ichimoku' in strategy:
            edge_types.append("IchimokuCloud")
        if 'Elliott' in strategy:
            edge_types.append("ElliottWave")
        edge_label = " + ".join(edge_types[:2]) if edge_types else strategy
        bias = "Bullish" if direction == "LONG" else "Bearish"

        msg = (
            f"🌍 <b>MARKET CONTEXT</b>\n\n"
            f"Trend:        {trend}\n"
            f"Volatility:   {vol}\n"
            f"Liquidity:    {liq}\n"
            f"Fear &amp; Greed: {fg}\n"
            f"Alt Index:    {alt}/100\n"
            f"Risk:         {risk}\n\n"
            f"Edge:  {edge_label}\n"
            f"Bias:  {bias}"
        )

        # ── News headlines (RSS via news_scraper) ─────────────
        news = raw.get('news_headlines', [])
        if news:
            msg += "\n\n📰 <b>Recent News</b>"
            for item in news[:3]:
                title = item.get('title', '')[:80]
                source = item.get('source', '')
                # news_scraper provides sentiment_score (keyword-based, 0-100)
                score = item.get('sentiment_score')
                if score is not None:
                    if score >= 65:
                        s_emoji = "🟢"
                    elif score <= 35:
                        s_emoji = "🔴"
                    else:
                        s_emoji = "⚪"
                    msg += f"\n{s_emoji} <i>{title}</i>  <code>{source}</code>"
                else:
                    msg += f"\n• <i>{title}</i>  <code>{source}</code>"

        # ── HyperTracker cohort bias (A/A+ signals only) ──────
        ht_mp_bias = raw.get('ht_mp_bias')
        ht_sm_bias = raw.get('ht_sm_bias')
        if ht_mp_bias or ht_sm_bias:
            msg += "\n\n🐋 <b>Smart Money (Hyperliquid)</b>"
            if ht_mp_bias:
                b_score = raw.get('ht_mp_bias_score', 50)
                in_pos  = raw.get('ht_mp_in_pos', 0)
                b_emoji = "🟢" if b_score >= 60 else ("🔴" if b_score <= 40 else "⚪")
                msg += f"\n{b_emoji} Money Printer: {ht_mp_bias}  ·  {in_pos}% active"
            if ht_sm_bias:
                b_score = raw.get('ht_sm_bias_score', 50)
                in_pos  = raw.get('ht_sm_in_pos', 0)
                b_emoji = "🟢" if b_score >= 60 else ("🔴" if b_score <= 40 else "⚪")
                msg += f"\n{b_emoji} Smart Money: {ht_sm_bias}  ·  {in_pos}% active"

        # OI crowd warning
        if raw.get('ht_is_crowded'):
            long_pct  = raw.get('ht_long_pct', 50)
            short_pct = raw.get('ht_short_pct', 50)
            dom_side  = "LONG" if long_pct >= short_pct else "SHORT"
            dom_pct   = max(long_pct, short_pct)
            is_same   = dom_side == direction
            w_emoji   = "⚠️" if is_same else "✅"
            crowd_label = "crowded" if is_same else "uncrowded — you're contrarian"
            msg += f"\n{w_emoji} HL OI: {dom_pct:.0f}% {dom_side} ({crowd_label})"

        liq_risk = raw.get('ht_liq_risk_pct', 0)
        if liq_risk >= 10:
            msg += f"\n💥 {liq_risk:.0f}% of Smart Money OI near liquidation"

        return msg

    def format_trade_plan_panel(self, scored: ScoredSignal,
                                 alpha_score=None, prob_estimate=None, sizing=None) -> str:
        sig = scored.base_signal
        direction = direction_str(sig)
        entry_mid = (sig.entry_low + sig.entry_high) / 2
        sl_pct = abs(entry_mid - sig.stop_loss) / entry_mid * 100 if entry_mid else 0
        tp1_pct = abs(sig.tp1 - entry_mid) / entry_mid * 100 if entry_mid else 0
        tp2_pct = abs(sig.tp2 - entry_mid) / entry_mid * 100 if entry_mid else 0
        tp3_pct = abs(sig.tp3 - entry_mid) / entry_mid * 100 if (sig.tp3 and entry_mid) else 0
        sign = "-" if direction == "LONG" else "+"
        tp_sign = "+" if direction == "LONG" else "-"

        p_win_str = f"\nP(win): {prob_estimate.p_win*100:.0f}%" if prob_estimate else ""
        ev_str = f"\nEV: {alpha_score.expected_value_r:+.2f}R" if alpha_score else ""
        tp3_line = f"\nTP3: <code>{fmt_price(sig.tp3)}</code>" if sig.tp3 else ""
        raw = sig.raw_data or {}
        if sizing and getattr(sizing, "position_size_usdt", 0) > 0:
            raw["intended_size_usd"] = sizing.position_size_usdt
        guidance = guidance_payload(
            sig,
            confluence=sig.confluence,
            current_session=getattr(regime_analyzer.session, 'value', ''),
        )
        adj_rr_line = f"\nAdj R/R: {guidance['fee_adjusted_rr']:.1f}R" if guidance.get("fee_adjusted_rr") is not None else ""
        skip_line = f"\n{guidance['skip_rule']}" if guidance.get("skip_rule") else ""
        warn_line = f"\n{guidance['session_warning']}" if guidance.get("session_warning") else ""
        size_str = ""
        if sizing and getattr(sizing, 'position_size_usdt', 0) > 0:
            size_str = (
                f"\n\nSuggested Size: ${sizing.position_size_usdt:,.0f}\n"
                f"Risk: ${sizing.risk_amount_usdt:,.0f}"
            )

        return (
            f"💰 <b>TRADE PLAN</b>\n\n"
            f"<b>ENTRY ZONE</b>\n"
            f"<code>{fmt_price(sig.entry_low)}</code> — <code>{fmt_price(sig.entry_high)}</code>\n\n"
            f"SL:  <code>{fmt_price(sig.stop_loss)}</code>  ({sign}{sl_pct:.1f}%)\n\n"
            f"TP1: <code>{fmt_price(sig.tp1)}</code> → BE\n"
            f"TP2: <code>{fmt_price(sig.tp2)}</code>"
            f"{tp3_line}\n"
            f"{p_win_str}{ev_str}{adj_rr_line}{skip_line}{warn_line}"
            f"{size_str}"
        )

    def format_logic_panel(self, scored: ScoredSignal) -> str:
        sig = scored.base_signal
        strategy = sig.strategy or "Unknown"
        raw = sig.raw_data or {}
        direction = direction_str(sig)

        lines = [f"🧠 <b>CONFLUENCE LOGIC</b>\n", f"<b>{strategy}</b>\n"]

        if raw.get('has_ob'):
            lines.append("✅ Order Block (institutional level)")
        if raw.get('has_fvg'):
            lines.append("✅ Fair Value Gap (imbalance)")
        if raw.get('has_sweep'):
            sweep_dir = "below" if direction == "LONG" else "above"
            lines.append(f"✅ Liquidity sweep {sweep_dir} key level")
        if 'Ichimoku' in strategy:
            lines.append("✅ Price above cloud" if direction == "LONG" else "✅ Price below cloud")
            lines.append("✅ TK aligned")
            lines.append("✅ Chikou clear of price")
        wave = raw.get('wave_type', '')
        if wave == 'WAVE3':
            lines.append("✅ Elliott Wave 3 — strongest impulse")
        elif wave == 'WAVE5':
            lines.append("✅ Elliott Wave 5 — final impulse")
        wyckoff = raw.get('wyckoff_event', '')
        if wyckoff == 'SPRING':
            lines.append("✅ Wyckoff Spring — stop hunt confirmed")
        elif wyckoff == 'UTAD':
            lines.append("✅ Wyckoff UTAD — false breakout")
        htf = raw.get('htf_structure', '')
        if htf:
            lines.append(f"✅ HTF Structure: {htf}")
        for item in scored.all_confluence[:4]:
            if not any(item in l for l in lines):
                lines.append(f"• {item}")

        return "\n".join(lines)

    def format_metrics_panel(self, scored: ScoredSignal) -> str:
        conf = scored.final_confidence or 0
        t = scored.technical_score
        v = scored.volume_score
        o = scored.orderflow_score
        d = scored.derivatives_score
        s = scored.sentiment_score

        def bar(score):
            filled = int(round(score / 10))
            return "█" * filled + "░" * (10 - filled)

        der_block = ""
        if scored.derivatives_data:
            dd = scored.derivatives_data
            squeeze_emoji = {"HIGH": "🚀", "MEDIUM": "⚡", "LOW": "—"}.get(dd.squeeze_risk, "—")
            der_block = (
                f"\n\nFunding:  <code>{dd.funding_rate:+.4f}%</code>\n"
                f"Squeeze:  {squeeze_emoji} {dd.squeeze_risk}\n"
                f"L/S Ratio: {dd.long_short_ratio:.2f}"
            )

        # Power alignment line
        _pa_line = ""
        if getattr(scored, 'power_aligned', False):
            _pa_reason = getattr(scored, 'power_alignment_reason', '')
            _pa_line = f"\n\n⚡ <b>Power Aligned</b>  —  {_pa_reason}"

        return (
            f"📊 <b>CONFIDENCE METRICS</b>\n\n"
            f"Overall: <b>{conf:.0f}/100</b>  {_confidence_label(conf)}\n\n"
            f"<code>"
            f"Structure  {bar(t)} {t:.0f}\n"
            f"Volume     {bar(v)} {v:.0f}\n"
            f"Orderflow  {bar(o)} {o:.0f}\n"
            f"Derivativ  {bar(d)} {d:.0f}\n"
            f"Sentiment  {bar(s)} {s:.0f}"
            f"</code>"
            f"{der_block}"
            f"{_pa_line}"
        )

    def format_profile_panel(self, scored: ScoredSignal) -> str:
        sig = scored.base_signal
        strategy = sig.strategy or "Unknown"
        p = STRATEGY_PERSONALITY.get(strategy, {
            'type': strategy, 'duration': '2-8 hours',
            'pullbacks': 'Variable', 'win_style': 'Standard',
            'exit_note': 'Follow TP levels as labeled.',
        })
        return (
            f"🧬 <b>TRADE PROFILE</b>\n\n"
            f"Type:      {p['type']}\n"
            f"Duration:  {p['duration']}\n\n"
            f"Pullbacks: {p['pullbacks']}\n"
            f"Edge:      {p['win_style']}\n\n"
            f"<i>{p['exit_note']}</i>"
        )

    # ── Lifecycle messages ────────────────────────────────────

    def format_entry_active(self, symbol: str, direction: str,
                             fill_price: float, signal_id: int,
                             scored=None) -> str:
        """
        IN TRADE card — levels always visible so you never have to hunt for them.
        scored: optional ScoredSignal — used to show entry zone / SL / TP levels.
        """
        dir_emoji = "🟢" if direction == "LONG" else "🔴"
        dir_label = "LONG ▲" if direction == "LONG" else "SHORT ▼"
        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")
        msg = (
            f"✅ <b>IN TRADE</b>  ·  {dir_emoji} <b>{dir_label}  {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Filled:  <code>{fmt_price(fill_price)}</code>\n"
        )
        if scored:
            sig = scored.base_signal
            msg += f"Zone:    <code>{fmt_price(sig.entry_low)}</code> — <code>{fmt_price(sig.entry_high)}</code>  ✓\n"
            msg += f"SL:      <code>{fmt_price(sig.stop_loss)}</code>\n"
            msg += f"TP1:     <code>{fmt_price(sig.tp1)}</code>  → move SL to BE\n"
            msg += f"TP2:     <code>{fmt_price(sig.tp2)}</code>\n"
            if sig.tp3:
                msg += f"TP3:     <code>{fmt_price(sig.tp3)}</code>\n"
            msg += f"R/R:     {sig.rr_ratio:.1f}R\n"
            strategy = sig.strategy or ""
            if strategy:
                msg += f"\nStrategy: <i>{strategy}</i>\n"
        # SL cluster warning on active card too
        try:
            from signals.liquidation_heatmap import liquidation_heatmap as _lh
            if scored:
                _sl = scored.base_signal.stop_loss
                _sym = scored.base_signal.symbol
                _clusters = _lh.get_cached_clusters(_sym)
                if _clusters:
                    _danger, _widen, _sl_note = _lh.check_sl_near_cluster(_sl, _clusters)
                    if _danger:
                        msg += f"\n{_sl_note}"
        except Exception:
            pass

        msg += f"\n⏱ {now_utc}  |  #{signal_id}"
        return msg

    def format_trade_active(self, symbol: str, direction: str,
                             pnl_pct: float, signal_id: int,
                             scored=None) -> str:
        """
        TRADE ACTIVE card — PnL shown at top, levels always visible below.
        scored: optional ScoredSignal — used to show SL/TP levels.
        """
        dir_emoji = "🟢" if direction == "LONG" else "🔴"
        dir_label = "LONG ▲" if direction == "LONG" else "SHORT ▼"
        pnl_emoji = "📈" if pnl_pct >= 0 else "📉"
        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")
        # pnl_pct field is actually passed as R value from the outcome monitor
        # Display as R so it's consistent with the TRADE ACTIVE heartbeat log
        _pnl_display = f"{pnl_pct:+.2f}R" if abs(pnl_pct) < 20 else f"{pnl_pct:+.2f}%"
        msg = (
            f"🟡 <b>TRADE ACTIVE</b>  ·  {dir_emoji} <b>{dir_label}  {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{pnl_emoji} PnL:  <b>{_pnl_display}</b>\n\n"
        )
        if scored:
            sig = scored.base_signal
            msg += f"SL:   <code>{fmt_price(sig.stop_loss)}</code>\n"
            msg += f"TP1:  <code>{fmt_price(sig.tp1)}</code>\n"
            msg += f"TP2:  <code>{fmt_price(sig.tp2)}</code>\n"
            if sig.tp3:
                msg += f"TP3:  <code>{fmt_price(sig.tp3)}</code>\n"
        msg += f"\n⏱ {now_utc}  |  #{signal_id}"
        return msg

    def format_tp1_hit(self, symbol: str, direction: str,
                        price: float, be_stop: float, signal_id: int,
                        scored=None) -> str:
        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")
        dir_emoji = "🟢" if direction == "LONG" else "🔴"
        dir_label = "LONG ▲" if direction == "LONG" else "SHORT ▼"
        msg = (
            f"🎯 <b>TP1 HIT</b>  ·  {dir_emoji} <b>{dir_label}  {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"✅ TP1:  <code>{fmt_price(price)}</code>  — taken\n"
            f"🛡 SL → BE:  <code>{fmt_price(be_stop)}</code>  — risk-free\n"
        )
        if scored:
            sig = scored.base_signal
            msg += f"\nTP2:  <code>{fmt_price(sig.tp2)}</code>  ← next target\n"
            if sig.tp3:
                msg += f"TP3:  <code>{fmt_price(sig.tp3)}</code>\n"
        msg += f"\n⏱ {now_utc}  |  #{signal_id}"
        return msg

    def format_trade_closed(self, symbol: str, direction: str, result_pct: float,
                             duration_h: float, grade: str, signal_id: int) -> str:
        result_emoji = "✅" if result_pct > 0 else ("➡️" if result_pct == 0 else "🔴")
        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")
        # FIX #10: Display as R-multiple (result_pct is now pnl_r from outcome monitor)
        r_display = f"{result_pct:+.2f}R" if result_pct != 0 else "BE"
        dur_display = f"{duration_h:.1f}h" if duration_h > 0 else "<1m"
        return (
            f"✅ <b>TRADE CLOSED</b>\n"
            f"<b>{symbol} {direction}</b>\n\n"
            f"Result:   {result_emoji} <b>{r_display}</b>\n"
            f"Duration: {dur_display}\n"
            f"Grade:    {grade}\n"
            f"⏱ {now_utc}  |  #{signal_id}"
        )

    def format_sl_hit(self, symbol: str, direction: str, sl_price: float, signal_id: int) -> str:
        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")
        return (
            f"🔴 <b>STOP LOSS HIT</b>\n"
            f"<b>{symbol} {direction}</b>\n\n"
            f"SL: <code>{fmt_price(sl_price)}</code>\n"
            f"Position closed.\n"
            f"⏱ {now_utc}  |  #{signal_id}"
        )

    def format_signal_invalidated(self, symbol: str, direction: str,
                                   reason: str, signal_id: int) -> str:
        return (
            f"❌ <b>SIGNAL INVALIDATED</b>\n"
            f"<b>{symbol} {direction}</b>\n\n"
            f"Reason: {reason}\n\n"
            f"<i>Structure changed. Do NOT enter this trade.</i>\n"
            f"#{signal_id}"
        )

    def format_signal_expired(self, symbol: str, direction: str,
                               candles: int, signal_id: int) -> str:
        return (
            f"⏰ <b>SIGNAL EXPIRED</b>\n"
            f"<b>{symbol} {direction}</b>\n\n"
            f"Entry zone not reached in {candles} candles.\n"
            f"<i>Edge has diminished. Skip this trade.</i>\n"
            f"#{signal_id}"
        )

    # ── Market digest ─────────────────────────────────────────

    def format_exit_rules_panel(self, scored: "ScoredSignal") -> str:
        """
        Exit Rules panel — shown when user taps 'Exit Rules' on an active trade card.
        Covers: TP management, trailing, invalidation, regime-based early exit.
        """
        sig = scored.base_signal
        direction = direction_str(sig)
        strategy = sig.strategy or "Unknown"
        regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        is_long = direction == "LONG"
        sign_away = "above" if is_long else "below"

        # TP management lines
        tp_mgmt = (
            f"✅ <b>TP1</b> hit → Move SL to BE immediately\n"
            f"📈 <b>TP2</b> → Full close (or leave ½ runner)\n"
        )
        if sig.tp3:
            tp_mgmt += f"🚀 <b>TP3</b> → Trail remainder with tight stop\n"

        # Regime-specific exit note
        regime_exit = {
            "VOLATILE":       "⚡ VOLATILE: Take TP1, close immediately — do not trail",
            "VOLATILE_PANIC": "🚨 PANIC: Exit at TP1 regardless. No trailing. Protect capital.",
            "BEAR_TREND":     "📉 BEAR TREND: LONGs — TP1 only. SHORTs — let it run to TP2.",
            "BULL_TREND":     "📈 BULL TREND: SHORTs — TP1 only. LONGs — trail to TP3.",
            "CHOPPY":         "〰 CHOPPY: TP1 is the full target. Do not trail.",
        }.get(regime, "")

        # Strategy-specific exit
        strat_exit = {
            "WyckoffAccDist":    "Wait for phase completion. Spring target = markup start.",
            "Wyckoff":           "Wait for phase completion. Spring target = markup start.",
            "Ichimoku":          "Exit if Chikou crosses back into price structure.",
            "IchimokuCloud":     "Exit if Chikou crosses back into price structure.",
            "ElliottWave":       "W3: exit at 161.8% ext. W5: watch for RSI divergence early.",
            "FundingRateArb":    "Close when funding normalises (rate returns near 0).",
            "MeanReversion":     "TP2 = the mean. That is the full target — do not overstay.",
            "SmartMoneyConcepts":"Exit if price closes back through the order block.",
            "Momentum":          "Exit if MACD histogram turns against you.",
            "MomentumContinuation": "Exit if MACD histogram turns against you.",
            "ExtremeReversal":   "TP2 = BB midband. Take it. Do not hold beyond.",
            "HarmonicPattern":   "Exit at D-point target. Pattern invalidation = SL.",
            "GeometricPattern":  "Exit at measured move target.",
        }.get(strategy, "Follow TP levels as labelled.")

        msg = (
            f"🚪 <b>EXIT RULES</b>\n\n"
            f"{tp_mgmt}\n"
            f"<b>Invalidation:</b> Close {sign_away} <code>{fmt_price(sig.stop_loss)}</code>\n\n"
        )
        if regime_exit:
            msg += f"{regime_exit}\n\n"
        msg += f"<b>Strategy note:</b>\n<i>{strat_exit}</i>"
        return msg

    def format_market_digest(self) -> str:
        regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        session = getattr(regime_analyzer, 'session', None)
        session_name = session.value if session else "Unknown"
        fg = regime_analyzer.fear_greed
        alt = regime_analyzer.alt_season_score
        # Show staleness warning when F&G hasn't been successfully fetched yet
        fg_display = f"{fg}" if regime_analyzer.fear_greed_is_fresh else f"{fg} ⚠️ stale"

        rotation = rotation_tracker.get_rotation_summary()
        hot = rotation.get('hot', [])[:3]

        regime_label = {
            "BULL_TREND": "TRENDING BULLISH", "BEAR_TREND": "TRENDING BEARISH",
            "CHOPPY": "CHOPPY", "VOLATILE": "VOLATILE", "VOLATILE_PANIC": "RISK-OFF",
        }.get(regime, regime)

        if regime == "BULL_TREND":
            edges = ["Trend Continuation", "Breakout Long"]
            avoid_types = ["Short reversals"]
        elif regime == "BEAR_TREND":
            edges = ["Trend Short", "Breakdown"]
            avoid_types = ["Long reversals"]
        elif regime == "CHOPPY":
            edges = ["Mean Reversion", "Range Trades"]
            avoid_types = ["Breakouts"]
        elif regime == "VOLATILE":
            edges = ["Funding Rate Arb", "Extreme Reversal"]
            avoid_types = ["Trend following"]
        else:
            edges = ["Selective only"]
            avoid_types = ["High risk setups"]

        edges_text = "\n".join(f"✓ {e}" for e in edges)
        avoid_text = "\n".join(f"✗ {a}" for a in avoid_types)
        hot_text = "  ".join(f"🔥 {s}" for s, _ in hot) if hot else "None"
        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")

        return (
            f"🌍 <b>Market Digest — {session_name}</b>\n\n"
            f"Regime: <b>{regime_label}</b>\n"
            f"Fear &amp; Greed: {fg_display}  ·  Alt Index: {alt}/100\n\n"
            f"<b>Best Edges:</b>\n{edges_text}\n\n"
            f"<b>Avoid:</b>\n{avoid_text}\n\n"
            f"<b>Hot Sectors:</b> {hot_text}\n\n"
            f"⏱ {now_utc}"
        )

    def format_watchlist_snapshot(self, symbol: str, data: dict) -> str:
        score = data.get('score', 0)
        reasons = data.get('reasons', [])
        reason_text = "\n".join(f"• {r}" for r in reasons[:4]) or "• Coiling detected"
        # V10: Infer likely direction from reasons
        _bull_hints = sum(1 for r in reasons if any(k in r.lower() for k in ('bullish', 'accumulation', 'support', 'oversold')))
        _bear_hints = sum(1 for r in reasons if any(k in r.lower() for k in ('bearish', 'distribution', 'resistance', 'overbought')))
        _dir_hint = ""
        if _bull_hints > _bear_hints:
            _dir_hint = "\n🔍 <b>Watch for:</b> Breakout LONG"
        elif _bear_hints > _bull_hints:
            _dir_hint = "\n🔍 <b>Watch for:</b> Breakdown SHORT"
        else:
            _dir_hint = "\n🔍 <b>Watch for:</b> Volatility expansion (direction TBD)"
        return (
            f"👁 <b>{symbol} — Watchlist Snapshot</b>\n\n"
            f"Score: <b>{score:.0f}/100</b>\n\n"
            f"{reason_text}\n"
            f"{_dir_hint}\n\n"
            f"<i>No executable signal yet. Watching for confirmation.</i>"
        )

    # ── Status / admin formatters ─────────────────────────────

    def format_status(self, stats: Dict) -> str:
        uptime = stats.get('uptime_str', '—')
        regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        session = getattr(regime_analyzer.session, 'value', 'UNKNOWN')
        alt_score = regime_analyzer.alt_season_score
        fg = regime_analyzer.fear_greed
        signals_today = stats.get('signals_today', 0)
        cpu = stats.get('cpu_pct', 0)
        ram_gb = stats.get('ram_gb', 0)
        t1 = stats.get('tier1_count', 0)
        t2 = stats.get('tier2_count', 0)
        t3 = stats.get('tier3_count', 0)
        watchlist = stats.get('watchlist_count', 0)

        regime_emoji = {
            "BULL_TREND": "📈", "BEAR_TREND": "📉", "CHOPPY": "〰",
            "VOLATILE": "⚡", "VOLATILE_PANIC": "🚨",
        }.get(regime, "?")

        return (
            f"<b>TITANBOT PRO — STATUS</b>\n\n"
            f"Uptime: <b>{uptime}</b>  ·  Signals today: <b>{signals_today}</b>\n\n"
            f"<b>SCANNING</b>\n"
            f"Tier 1: {t1}  Tier 2: {t2}  Tier 3: {t3}\n"
            f"Watchlist: {watchlist} setups\n\n"
            f"<b>MARKET: {regime_emoji} {regime}</b>\n"
            f"Session: {session}  ·  F&amp;G: {fg}  ·  Alt: {alt_score}/100\n\n"
            f"<b>SYSTEM</b>\n"
            f"CPU: {cpu}%  ·  RAM: {ram_gb:.1f}GB"
        )

    def format_performance(self, stats: Dict, period: str = "30 days") -> str:
        total = stats.get('total_signals') or 0
        taken = stats.get('trades_taken') or 0
        wins = stats.get('wins') or 0
        losses = stats.get('losses') or 0
        avg_r = stats.get('avg_r') or 0
        win_rate = (wins / taken * 100) if taken > 0 else 0
        by_strategy = stats.get('by_strategy', [])

        strategy_text = ""
        for row in by_strategy[:5]:
            strat = row.get('strategy', '?')[:16]
            w = row.get('wins', 0)
            l = row.get('losses', 0)
            tot = w + l
            wr = f"{w/tot*100:.0f}%" if tot > 0 else "—"
            strategy_text += f"\n  {strat:<16} {w}W/{l}L  {wr}"

        return (
            f"📊 <b>PERFORMANCE — Last {period}</b>\n\n"
            f"Signals:      <b>{total}</b>\n"
            f"Trades taken: <b>{taken}</b>\n"
            f"Win rate:     <b>{win_rate:.1f}%</b>  ({wins}W / {losses}L)\n"
            f"Avg R:        <b>{avg_r:.2f}R</b>\n\n"
            f"<b>BY STRATEGY</b>\n"
            f"<code>{strategy_text}</code>"
        )

    def format_market(self) -> str:
        return self.format_market_digest()

    def format_watchlist_alert(self, symbol: str, score: float, reasons: List[str]) -> str:
        reasons_text = "\n".join(f"  • {r}" for r in reasons[:4])
        return (
            f"👁 <b>WATCHLIST — {symbol}</b>\n"
            f"Score: {score:.0f}/100\n\n"
            f"{reasons_text}"
        )

    def format_whale_alert(self, symbol: str, order_usd: float, volume_mult: float) -> str:
        return (
            f"🐋 <b>WHALE — {symbol}</b>\n"
            f"${order_usd:,.0f}  ·  {volume_mult:.1f}x avg volume"
        )

    def format_tier_promotion(self, symbol: str, from_tier: int, to_tier: int,
                               volume_mult: float, volume_24h: float) -> str:
        return (
            f"🚀 <b>TIER PROMOTION — {symbol}</b>\n"
            f"Tier {from_tier} → Tier {to_tier}  ·  {volume_mult:.1f}x volume spike"
        )

    def format_daily_summary(self, summary: Dict) -> str:
        date = datetime.now(timezone.utc).strftime("%b %d, %Y")
        signals_yday = summary.get('signals_yesterday', 0)
        aplus = summary.get('aplus_count', 0)
        watchlist = summary.get('watchlist_today', [])
        watch_text = ""
        for w in watchlist[:3]:
            watch_text += f"\n  • {w.get('symbol', '?')}"

        return (
            f"📋 <b>DAILY SUMMARY — {date}</b>\n\n"
            f"Signals yesterday: {signals_yday}  ·  A+: {aplus}\n\n"
            f"<b>WATCHLIST</b>{watch_text or chr(10) + '  Nothing coiling yet'}\n\n"
            f"Regime: {getattr(regime_analyzer.regime, 'value', 'UNKNOWN')}  ·  Alt: {regime_analyzer.alt_season_score}/100"
        )

    def format_circuit_breaker_active(self, reason: str, resume_time: str) -> str:
        return (
            f"🚨 <b>CIRCUIT BREAKER</b>\n\n"
            f"Reason: {reason}\n"
            f"Resume: {resume_time} UTC"
        )

    def format_circuit_breaker_cleared(self) -> str:
        return "✅ <b>Circuit breaker cleared.</b> Resuming normal scanning."

    def format_error(self, error: str, recovering: bool = True) -> str:
        recovery = "🔄 Auto-recovering..." if recovering else "⚠️ Manual intervention needed."
        return f"⚠️ <b>ERROR:</b> {error}\n{recovery}"

    # Backward compat shims
    def format_entry_reached(self, symbol: str, direction: str,
                              price: float, strategy: str) -> str:
        return self.format_entry_active(symbol, direction, price, 0)

    def format_invalidated(self, symbol: str, direction: str,
                            reason: str, strategy: str) -> str:
        return self.format_signal_invalidated(symbol, direction, reason, 0)

    def format_expired(self, symbol: str, direction: str, strategy: str, signal_id: int = 0) -> str:
        candles = STRATEGY_EXPIRY_CANDLES.get(strategy, 6)
        return self.format_signal_expired(symbol, direction, candles, signal_id)

    def format_conflict(self, symbol: str, original_dir: str,
                         new_dir: str, strategy: str) -> str:
        return (
            f"⚠️ <b>CONFLICTING SIGNAL — {symbol}</b>\n"
            f"Had: {original_dir}  ·  New: {new_dir}\n\n"
            f"<i>Stand aside or trade the higher grade.</i>"
        )

    def format_outcome(self, symbol: str, direction: str, outcome: str,
                        pnl_r: float, note: str) -> str:
        outcome_map = {
            "WIN": ("✅", "WIN"), "LOSS": ("🔴", "LOSS"),
            "BREAKEVEN": ("➡️", "BREAKEVEN"), "EXPIRED": ("⏰", "EXPIRED"),
        }
        emoji, label = outcome_map.get(outcome, ("📊", outcome))
        r_str = f"{pnl_r:+.1f}R" if pnl_r != 0 else "—"
        return f"{emoji} <b>{label}</b>  {r_str}\n{note}"

    def format_signal_detail(self, scored: ScoredSignal) -> str:
        return self.format_logic_panel(scored) + "\n\n" + self.format_metrics_panel(scored)

    @staticmethod
    def _confidence_bar(confidence: float) -> str:
        filled = int(confidence / 10)
        return "█" * filled + "░" * (10 - filled)

    @staticmethod
    def _mini_bar(wins: int, total: int) -> str:
        if total == 0:
            return "░░░░░"
        filled = int(wins / total * 5)
        return "█" * filled + "░" * (5 - filled)

    def format_context_signal(self, scored, signal_id: int, up_score: float) -> str:
        """
        C signal with meaningful upgrade probability.
        Deliberately strips all tradeable numbers.
        Psychological design: narrative = observation, not action.
        """
        sig = scored.base_signal
        direction = direction_str(sig)
        strategy = sig.strategy or "Unknown"

        regime = scored.regime or getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        regime_readable = {
            "BULL_TREND": "Trending Bullish", "BEAR_TREND": "Trending Bearish",
            "CHOPPY": "Range / Chop", "VOLATILE": "High Volatility",
            "VOLATILE_PANIC": "Risk-Off",
        }.get(regime, regime)

        # Direction in narrative language, not trading language
        pressure = "upward" if direction == "LONG" else "downward"

        # What's forming (strategy-readable)
        strategy_narrative = {
            'SmartMoneyConcepts':    "Smart money positioning",
            'Ichimoku':              "Trend alignment forming",
            'IchimokuCloud':         "Trend alignment forming",
            'ElliottWave':           "Wave structure developing",
            'WyckoffAccDist':        "Accumulation / distribution phase",
            'Wyckoff':               "Accumulation / distribution phase",
            'InstitutionalBreakout': "Breakout pressure building",
            'Momentum':              "Momentum building",
            'MomentumContinuation':  "Momentum building",
            'FundingRateArb':        "Funding extreme forming",
            'MeanReversion':         "Statistical deviation detected",
            'PriceAction':           "Price structure forming",
            'ExtremeReversal':       "Extreme condition detected",
            'RangeScalper':          "Range boundary approached",
            'HarmonicPattern':       "Harmonic pattern completing",
            'GeometricPattern':      "Chart pattern forming",
        }.get(strategy, "Setup forming")

        # Candle validity
        # FIX: removed self-import (tg/formatter.py importing from itself). STRATEGY_EXPIRY_CANDLES and TF_MINUTES are already in module scope above.
        candles = STRATEGY_EXPIRY_CANDLES.get(strategy, 6)

        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")

        _sc_label2 = {"swing": "Swing", "intraday": "Intraday", "scalp": "Scalp"}.get(getattr(sig, 'setup_class', 'intraday'), 'Intraday')
        msg  = f"👀 <b>MARKET CONTEXT — {sig.symbol}  ·  {_sc_label2}</b>\n\n"
        msg += f"{strategy_narrative}.\n"
        msg += f"Potential {pressure} pressure on this timeframe.\n\n"
        msg += f"Regime: {regime_readable}\n"
        msg += f"Confidence: {scored.final_confidence:.0f}  (threshold: 72)\n\n"
        msg += f"<b>This is NOT a trade signal.</b>\n"
        msg += f"Monitoring for confirmation. Valid for {candles} candles.\n\n"
        msg += f"👉 Wait for upgrade to A/B grade.\n"
        msg += f"⏱ {now_utc}  |  #{signal_id}"

        return msg

    def format_upgrade_message(self, symbol: str, direction: str,
                                old_grade: str, new_grade: str,
                                signal_id: int) -> str:
        """
        Upgrade notification — the reward event in the attention loop.
        Signals: context → confirmation → execution flow.
        """
        dir_emoji = "🟢" if direction == "LONG" else "🔴"
        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")
        return (
            f"⬆️ <b>SIGNAL UPGRADE</b>\n\n"
            f"{dir_emoji} <b>{symbol} {direction}</b>\n"
            f"Grade: <b>{old_grade} → {new_grade}</b>\n\n"
            f"Confirmation received. Trade details follow.\n"
            f"⏱ {now_utc}  |  #{signal_id}"
        )

    def format_admin_c_signal(self, scored, signal_id: int, up_score: float) -> str:
        """
        Compact admin-channel message for suppressed C signals.
        Operators can see what was filtered and why.
        """
        sig = scored.base_signal
        direction = direction_str(sig)
        entry_mid = (sig.entry_low + sig.entry_high) / 2
        tp2_pct = abs(sig.tp2 - entry_mid) / entry_mid * 100 if entry_mid else 0
        return (
            f"🔇 <b>C suppressed</b>  {sig.symbol} {direction}\n"
            f"conf={scored.final_confidence:.0f}  UP={up_score:.2f}  "
            f"TP2={tp2_pct:.1f}%  regime={scored.regime}\n"
            f"#{signal_id}"
        )

formatter = TelegramFormatter()


def format_market_brain_panel(profile: dict) -> str:
    if not profile:
        return ""
    regime = profile.get("regime", "UNKNOWN")
    energy = profile.get("energy_score", None)
    confirmations = profile.get("confirmation_required", "?")
    energy_txt = f"{energy:.2f}" if isinstance(energy, (int, float)) else "N/A"
    return (
        "\n🧠 Market Mode: {regime}"
        "\n⚡ Energy: {energy_txt}"
        "\n🎯 Confirmations: {confirmations}"
    ).format(regime=regime, energy_txt=energy_txt, confirmations=confirmations)
