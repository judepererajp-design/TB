"""
TitanBot Pro — Sentinel Intelligence Features
===============================================
Eight AI-powered analysis capabilities that go beyond signal generation.
All use the fast tier (Llama 3.3 70B) unless noted. All are optional and
fail gracefully — any exception returns None or a safe fallback.

Features:
  1. postmortem()            — Analyses last N closed trades, finds losing patterns
  2. regime_oracle()         — Predicts next regime shift 30min ahead (leading signals)
  3. strategy_doctor()       — Deep dive on one strategy's performance problems
  4. whale_intent()          — Classifies whale flow as accumulation/distribution/squeeze
  5. symbol_blacklist_scan() — Finds symbols with chronic false positives, temp-bans them
  6. threshold_calibration() — Suggests specific parameter numbers from outcome data
  7. correlation_radar()     — Detects BTC correlation breakdowns (independent move alert)
  8. news_spike_scan()       — Batch sentiment scan across all symbols, flags catalysts

Usage:
  from analyzers.sentinel_features import sentinel
  result = await sentinel.postmortem()
  result = await sentinel.regime_oracle(context)
  result = await sentinel.strategy_doctor("GeometricPattern:RisingWedge")
  ...

All methods also callable via Telegram: /sentinel <feature> [args]
"""

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────

@dataclass
class PostmortemResult:
    total_trades: int = 0
    win_rate: float = 0.0
    dominant_loss_pattern: str = ""
    direction_breakdown: dict = field(default_factory=dict)   # {"LONG": {"wins": 3, "losses": 7}}
    regime_breakdown: dict = field(default_factory=dict)      # {"BULL_TREND": {"wins": 5, "losses": 2}}
    worst_strategy: str = ""
    worst_strategy_loss_rate: float = 0.0
    insight: str = ""        # Plain English summary paragraph
    actions: List[str] = field(default_factory=list)  # Specific actionable fixes


@dataclass
class RegimeOracleResult:
    current_regime: str = ""
    predicted_next: str = ""
    confidence: float = 0.0          # 0-100
    estimated_time_to_shift: str = ""  # "10-20 min", "1-2 hours", "stable"
    leading_signals: List[str] = field(default_factory=list)
    trade_implication: str = ""       # What to do NOW before the shift


@dataclass
class StrategyDoctorResult:
    strategy: str = ""
    total_signals: int = 0
    win_rate: float = 0.0
    avg_rr_achieved: float = 0.0
    vs_rr_expected: float = 0.0
    top_kill_reasons: List[Tuple[str, int]] = field(default_factory=list)
    best_regime: str = ""
    worst_regime: str = ""
    diagnosis: str = ""
    prescription: List[str] = field(default_factory=list)


@dataclass
class WhaleIntentResult:
    symbol: str = ""
    classification: str = ""   # ACCUMULATION | DISTRIBUTION | SHORT_SQUEEZE | PANIC_SELL | NOISE
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    signal_implication: str = ""    # LONG_BIAS | SHORT_BIAS | AVOID | NEUTRAL
    urgency: str = "LOW"            # HIGH | MEDIUM | LOW


@dataclass
class BlacklistEntry:
    symbol: str
    reason: str
    false_positive_rate: float
    sample_size: int
    banned_until: float      # unix timestamp


@dataclass
class ThresholdSuggestion:
    parameter: str           # e.g. "rr_floor", "htf_block_threshold"
    strategy: str            # which strategy it applies to (or "global")
    current_value: float
    suggested_value: float
    evidence: str            # why — one sentence
    expected_improvement: str
    risk: str                # LOW | MEDIUM | HIGH


@dataclass
class CorrelationBreakResult:
    symbol: str = ""
    normal_correlation: float = 0.0
    current_correlation: float = 0.0
    break_magnitude: float = 0.0    # how far from normal
    direction: str = ""             # DIVERGING_UP | DIVERGING_DOWN
    confidence: float = 0.0
    alert: str = ""


@dataclass
class NewsSpike:
    symbol: str
    sentiment: str           # BULLISH | BEARISH
    urgency: str             # HIGH | MEDIUM
    headline_count: int
    key_headline: str
    estimated_move_window: str   # "minutes" | "hours"


class SentinelFeatures:
    """
    Eight AI-powered analysis capabilities beyond signal generation.
    Singleton — use `sentinel` at module bottom.
    """

    def __init__(self):
        # Simple in-memory blacklist: symbol -> BlacklistEntry
        self._blacklist: Dict[str, BlacklistEntry] = {}
        # Cache for correlation baseline: symbol -> (baseline_corr, timestamp)
        self._corr_baseline: Dict[str, Tuple[float, float]] = {}

    def _get_ai(self):
        """Lazy import to avoid circular deps."""
        from analyzers.ai_analyst import ai_analyst
        return ai_analyst

    # ── 1. Postmortem ─────────────────────────────────────────

    async def postmortem(self, lookback_trades: int = 30) -> Optional[PostmortemResult]:
        """
        Analyses last N closed trades to find systematic losing patterns.
        Reads from diagnostic_engine._outcome_log (live outcomes) and
        learning_loop._trade_history (paper/live trade records).

        Returns a PostmortemResult with plain-English insight and specific actions.
        """
        try:
            from core.diagnostic_engine import diagnostic_engine
            from core.learning_loop import learning_loop

            # Gather outcome data from both sources
            outcomes = list(diagnostic_engine._outcome_log[-lookback_trades:])
            if len(outcomes) < 5:
                logger.info("Postmortem: not enough outcome data yet (need 5+)")
                return None

            # Build compact summary for AI
            direction_stats: dict = defaultdict(lambda: {"wins": 0, "losses": 0})
            regime_stats: dict = defaultdict(lambda: {"wins": 0, "losses": 0})
            strat_stats: dict = defaultdict(lambda: {"wins": 0, "losses": 0})

            for o in outcomes:
                d = o.get("direction", "?")
                r = o.get("regime", "?")
                s = o.get("strategy", "?")
                won = o.get("outcome") == "WIN"
                direction_stats[d]["wins" if won else "losses"] += 1
                regime_stats[r]["wins" if won else "losses"] += 1
                strat_stats[s]["wins" if won else "losses"] += 1

            wins = sum(1 for o in outcomes if o.get("outcome") == "WIN")
            win_rate = wins / len(outcomes)
            avg_pnl_r = sum(o.get("pnl_r", 0) for o in outcomes) / len(outcomes)

            prompt = f"""You are a trading performance analyst reviewing a crypto futures bot's recent trades.

TRADE SUMMARY (last {len(outcomes)} closed trades):
Win rate: {win_rate*100:.1f}% ({wins} wins / {len(outcomes)-wins} losses)
Average P&L in R: {avg_pnl_r:+.2f}R

Direction breakdown:
{json.dumps(dict(direction_stats), indent=2)}

Regime breakdown:
{json.dumps(dict(regime_stats), indent=2)}

Strategy breakdown:
{json.dumps(dict(strat_stats), indent=2)}

Sample of recent losses (last 5):
{json.dumps([o for o in outcomes if o.get('outcome') != 'WIN'][-5:], indent=2)}

Identify:
1. The single most damaging pattern (e.g. "shorting in BULL_TREND", "FundingArb losses in trending markets")
2. The worst-performing strategy by loss rate
3. The regime where the bot performs best and worst
4. Two or three specific, actionable fixes

Return ONLY valid JSON:
{{
  "dominant_loss_pattern": "<one sentence describing the biggest losing pattern>",
  "worst_strategy": "<strategy name>",
  "worst_strategy_loss_rate": <float 0-1>,
  "best_regime": "<regime name>",
  "worst_regime": "<regime name>",
  "insight": "<2-3 sentence plain English summary of what's wrong and why>",
  "actions": ["<specific fix 1>", "<specific fix 2>", "<specific fix 3>"]
}}"""

            ai = self._get_ai()
            data = ai._parse_json(await ai._call(prompt, call_label="postmortem"))
            if not data:
                return None

            return PostmortemResult(
                total_trades=len(outcomes),
                win_rate=win_rate,
                dominant_loss_pattern=data.get("dominant_loss_pattern", ""),
                direction_breakdown=dict(direction_stats),
                regime_breakdown=dict(regime_stats),
                worst_strategy=data.get("worst_strategy", ""),
                worst_strategy_loss_rate=float(data.get("worst_strategy_loss_rate", 0)),
                insight=data.get("insight", ""),
                actions=data.get("actions", []),
            )

        except Exception as e:
            logger.error(f"Postmortem error: {e}")
            return None

    # ── 2. Regime Oracle ──────────────────────────────────────

    async def regime_oracle(self, market_context: dict) -> Optional[RegimeOracleResult]:
        """
        Predicts the next regime shift using leading indicators:
        - Funding rate momentum (rate of change, not level)
        - OI change rate (acceleration, not total)
        - Whale accumulation velocity (buys per minute, increasing or decreasing)
        - BTC 15m momentum vs 1h trend (divergence = early warning)
        - Stablecoin inflow rate

        This fires ahead of the HTF guardrail recalculation, giving the signal
        engine a 15-30 minute head start on the next regime.
        """
        try:
            from signals.whale_aggregator import whale_aggregator
            whale_stats = whale_aggregator.get_session_stats()

            prompt = f"""You are a regime prediction model for a crypto futures trading bot.
Your job is to predict the NEXT regime shift, not describe the current one.

Current state:
- Current regime: {market_context.get('regime', 'UNKNOWN')}
- BTC 15m change: {market_context.get('btc_15m_change', 0):+.2f}%
- BTC 1h change: {market_context.get('btc_1h_change', 0):+.2f}%
- BTC 4h change: {market_context.get('btc_4h_change', 0):+.2f}%
- Funding rate now: {market_context.get('funding_rate', 0):+.4f}%
- Funding rate 1h ago: {market_context.get('funding_rate_1h_ago', 0):+.4f}%
- OI change last 30min: {market_context.get('oi_change_30m', 0):+.2f}%
- Whale buy velocity (last 30min): {whale_stats.get('total_buy_volume', 0)/1000:.0f}k/30min
- Fear & Greed: {market_context.get('fear_greed', 50)}
- ADX: {market_context.get('adx', 0):.1f}
- Chop index: {market_context.get('chop', 0):.2f}

Leading signal focus:
- If 15m diverges from 4h, regime is about to flip
- If funding momentum is accelerating negative, short squeeze imminent
- If whale buy velocity is increasing while price falls, accumulation → pump
- If OI rising while price falling, trapped longs → flush incoming

Return ONLY valid JSON:
{{
  "predicted_next_regime": "<BULL_TREND|BEAR_TREND|CHOPPY|REVERSAL_IMMINENT|STABLE>",
  "confidence": <integer 0-100>,
  "estimated_time_to_shift": "<5-15 min|15-30 min|30-60 min|1-2 hours|stable — no shift expected>",
  "leading_signals": ["<signal 1>", "<signal 2>", "<signal 3>"],
  "trade_implication": "<what to do right now given the predicted shift — 1 sentence>"
}}"""

            ai = self._get_ai()
            data = ai._parse_json(await ai._call(prompt, call_label="regime_oracle"))
            if not data:
                return None

            return RegimeOracleResult(
                current_regime=market_context.get("regime", "UNKNOWN"),
                predicted_next=data.get("predicted_next_regime", "STABLE"),
                confidence=float(data.get("confidence", 0)),
                estimated_time_to_shift=data.get("estimated_time_to_shift", "stable"),
                leading_signals=data.get("leading_signals", []),
                trade_implication=data.get("trade_implication", ""),
            )

        except Exception as e:
            logger.error(f"Regime oracle error: {e}")
            return None

    # ── 3. Strategy Doctor ────────────────────────────────────

    async def strategy_doctor(self, strategy_name: str) -> Optional[StrategyDoctorResult]:
        """
        Deep dive on a single strategy's recent performance.
        Reads both the death log (why signals died before firing) and
        the outcome log (how signals that DID fire actually performed).
        """
        try:
            from core.diagnostic_engine import diagnostic_engine

            # Deaths for this strategy
            deaths = [
                d for d in diagnostic_engine._death_log
                if strategy_name.lower() in d.get("strategy", "").lower()
            ]

            # Outcomes for this strategy
            outcomes = [
                o for o in diagnostic_engine._outcome_log
                if strategy_name.lower() in o.get("strategy", "").lower()
            ]

            if len(deaths) + len(outcomes) < 5:
                logger.info(f"Strategy doctor: insufficient data for {strategy_name}")
                return None

            wins = sum(1 for o in outcomes if o.get("outcome") == "WIN")
            win_rate = wins / len(outcomes) if outcomes else 0
            avg_rr = sum(o.get("pnl_r", 0) for o in outcomes) / len(outcomes) if outcomes else 0
            kill_counts = Counter(d.get("kill_reason", "?") for d in deaths)
            regime_wins = defaultdict(lambda: {"wins": 0, "losses": 0})
            for o in outcomes:
                r = o.get("regime", "?")
                regime_wins[r]["wins" if o.get("outcome") == "WIN" else "losses"] += 1

            prompt = f"""You are diagnosing performance problems for one strategy in a crypto futures bot.

Strategy: {strategy_name}
Total signals generated (reached outcome): {len(outcomes)}
Total signals killed (never fired): {len(deaths)}
Win rate on fired signals: {win_rate*100:.1f}%
Average R achieved: {avg_rr:+.2f}R

Why signals were killed before firing (top reasons):
{json.dumps(dict(kill_counts.most_common(8)))}

Performance by regime:
{json.dumps(dict(regime_wins))}

Last 3 losses (for pattern detection):
{json.dumps([o for o in outcomes if o.get('outcome') != 'WIN'][-3:], indent=2)}

Diagnose:
1. What is the core problem with this strategy right now?
2. In which regime does it perform best, and worst?
3. Are the kill reasons filtering out good signals or bad ones?
4. Give 2-3 specific prescription changes (parameter values, not vague advice)

Return ONLY valid JSON:
{{
  "best_regime": "<regime name>",
  "worst_regime": "<regime name>",
  "diagnosis": "<2-3 sentence plain English diagnosis>",
  "kill_reason_verdict": "<are the filter kills helping or hurting — 1 sentence>",
  "prescription": ["<specific fix with numbers if possible>", "<specific fix 2>", "<specific fix 3>"]
}}"""

            ai = self._get_ai()
            data = ai._parse_json(await ai._call(prompt, call_label=f"doctor_{strategy_name}"))
            if not data:
                return None

            return StrategyDoctorResult(
                strategy=strategy_name,
                total_signals=len(outcomes),
                win_rate=win_rate,
                avg_rr_achieved=avg_rr,
                top_kill_reasons=kill_counts.most_common(5),
                best_regime=data.get("best_regime", ""),
                worst_regime=data.get("worst_regime", ""),
                diagnosis=data.get("diagnosis", ""),
                prescription=data.get("prescription", []),
            )

        except Exception as e:
            logger.error(f"Strategy doctor error: {e}")
            return None

    # ── 4. Whale Intent Classifier ────────────────────────────

    async def whale_intent(self, symbol: str, lookback_secs: int = 600) -> Optional[WhaleIntentResult]:
        """
        Classifies recent whale activity on a symbol as one of:
          ACCUMULATION  — large buys, increasing velocity, price not moving up yet
          DISTRIBUTION  — large sells, decreasing buy pressure, smart exit
          SHORT_SQUEEZE — extreme negative funding + whale buys = squeeze setup
          PANIC_SELL    — sudden large sells, volume spike, likely cascade
          NOISE         — random activity, no discernible pattern

        More actionable than raw buy/sell counts because it captures timing and pattern.
        """
        try:
            from signals.whale_aggregator import whale_aggregator
            events = whale_aggregator.get_recent_events(symbol=symbol, max_age_secs=lookback_secs)

            if not events or len(events) < 2:
                return None

            # Build event timeline for AI
            timeline = []
            for e in sorted(events, key=lambda x: x.timestamp):
                age_s = int(time.time() - e.timestamp)
                timeline.append({
                    "side": e.side,
                    "usd": round(e.order_usd),
                    "age_seconds_ago": age_s,
                })

            total_buy = sum(e.order_usd for e in events if e.side == "buy")
            total_sell = sum(e.order_usd for e in events if e.side == "sell")
            buy_events = [e for e in events if e.side == "buy"]
            # Velocity: are buys accelerating (most buys in last 2min vs first 2min)?
            mid_ts = time.time() - lookback_secs / 2
            recent_buys = sum(e.order_usd for e in buy_events if e.timestamp > mid_ts)
            early_buys = total_buy - recent_buys
            velocity_trend = "ACCELERATING" if recent_buys > early_buys * 1.5 else (
                "DECELERATING" if early_buys > recent_buys * 1.5 else "STEADY"
            )

            prompt = f"""Classify the whale activity pattern for {symbol} in the last {lookback_secs//60} minutes.

Event timeline (chronological):
{json.dumps(timeline)}

Summary:
- Total BUY volume: ${total_buy:,.0f}
- Total SELL volume: ${total_sell:,.0f}
- Buy velocity trend: {velocity_trend} (comparing first vs second half of window)
- Buy/sell ratio: {total_buy/(total_sell+1):.2f}x

Classifications to choose from:
- ACCUMULATION: smart money buying quietly, velocity increasing, price may lag
- DISTRIBUTION: institutional exits, sell volume dominating, price support weakening
- SHORT_SQUEEZE: heavy buying into negative funding = shorts about to get squeezed
- PANIC_SELL: sudden large sells, likely triggered by news or cascade liquidation
- NOISE: random mixed flow, no actionable pattern

Return ONLY valid JSON:
{{
  "classification": "<one of the 5 options above>",
  "confidence": <integer 0-100>,
  "evidence": ["<evidence point 1>", "<evidence point 2>"],
  "signal_implication": "<LONG_BIAS|SHORT_BIAS|AVOID|NEUTRAL>",
  "urgency": "<HIGH if action needed in <15min, MEDIUM if <1h, LOW otherwise>"
}}"""

            ai = self._get_ai()
            data = ai._parse_json(await ai._call(prompt, call_label=f"whale_intent_{symbol}"))
            if not data:
                return None

            return WhaleIntentResult(
                symbol=symbol,
                classification=data.get("classification", "NOISE"),
                confidence=float(data.get("confidence", 0)),
                evidence=data.get("evidence", []),
                signal_implication=data.get("signal_implication", "NEUTRAL"),
                urgency=data.get("urgency", "LOW"),
            )

        except Exception as e:
            logger.error(f"Whale intent error for {symbol}: {e}")
            return None

    # ── 5. Symbol Blacklist Scanner ───────────────────────────

    async def symbol_blacklist_scan(
        self,
        ban_duration_hours: int = 24,
        min_sample: int = 20,
        max_fpr: float = 0.85,
    ) -> List[BlacklistEntry]:
        """
        Scans the death log for symbols with chronic false positive rates.
        A symbol is a candidate for temp-banning if:
          - It has generated >= min_sample signals in the last 48h
          - >= max_fpr (85%) of them were killed by the same filter
          - None of the signals that passed have WON

        Typical culprits: wash-traded low-caps, manipulated pumps,
        micro-cap tokens with spread issues.

        Writes results to self._blacklist. The engine checks this via
        is_blacklisted() before running strategies.
        """
        try:
            from core.diagnostic_engine import diagnostic_engine
            import time as _t
            now = _t.time()

            recent_deaths = [
                d for d in diagnostic_engine._death_log
                if now - d["ts"] < 172800  # 48h
            ]

            # Count deaths per symbol
            symbol_deaths: dict = defaultdict(list)
            for d in recent_deaths:
                symbol_deaths[d["symbol"]].append(d)

            candidates = []
            for symbol, deaths in symbol_deaths.items():
                if len(deaths) < min_sample:
                    continue
                kill_counts = Counter(d.get("kill_reason", "?") for d in deaths)
                top_kill, top_count = kill_counts.most_common(1)[0]
                fpr = top_count / len(deaths)
                if fpr >= max_fpr:
                    candidates.append({
                        "symbol": symbol,
                        "total_deaths": len(deaths),
                        "top_kill_reason": top_kill,
                        "false_positive_rate": round(fpr, 3),
                    })

            if not candidates:
                logger.info("Blacklist scan: no candidates found")
                return []

            # Ask AI to confirm — mechanical scan can have false alarms
            prompt = f"""You are reviewing symbols for temporary trading suspension in a crypto bot.
A symbol is flagged if it consistently generates signals that die on the same filter,
suggesting the market for that symbol is either illiquid, manipulated, or incompatible
with the bot's strategy logic.

Flagged candidates (sorted by false positive rate):
{json.dumps(sorted(candidates, key=lambda x: -x['false_positive_rate'])[:15], indent=2)}

For each symbol, decide: BAN (temp suspend) or KEEP (likely legitimate).
BAN if: consistently killed by spread/liquidity filters, obvious micro-cap, or suspicious name.
KEEP if: major coin (BTC, ETH, SOL, etc.) or the kill reason suggests a fixable config issue.

Return ONLY valid JSON (array):
[
  {{"symbol": "<SYM/USDT>", "verdict": "<BAN|KEEP>", "reason": "<1 sentence>"}}
]"""

            ai = self._get_ai()
            raw = await ai._call(prompt, call_label="blacklist_scan")
            verdicts = ai._parse_json(raw)
            if not verdicts or not isinstance(verdicts, list):
                # Fall back: mechanically ban candidates, AI unavailable
                verdicts = [{"symbol": c["symbol"], "verdict": "BAN", "reason": "auto-ban (AI unavailable)"} for c in candidates]

            banned = []
            ban_until = time.time() + ban_duration_hours * 3600

            for v in verdicts:
                if v.get("verdict") == "BAN":
                    sym = v["symbol"]
                    cand = next((c for c in candidates if c["symbol"] == sym), None)
                    if not cand:
                        continue
                    entry = BlacklistEntry(
                        symbol=sym,
                        reason=v.get("reason", "chronic false positive"),
                        false_positive_rate=cand["false_positive_rate"],
                        sample_size=cand["total_deaths"],
                        banned_until=ban_until,
                    )
                    self._blacklist[sym] = entry
                    banned.append(entry)
                    logger.warning(
                        f"🚫 Blacklisted {sym} for {ban_duration_hours}h — "
                        f"FPR={cand['false_positive_rate']:.0%} ({cand['total_deaths']} deaths): "
                        f"{v.get('reason')}"
                    )

            return banned

        except Exception as e:
            logger.error(f"Blacklist scan error: {e}")
            return []

    def is_blacklisted(self, symbol: str) -> Tuple[bool, str]:
        """Check if symbol is currently blacklisted. Called from engine.py."""
        if symbol not in self._blacklist:
            return False, ""
        entry = self._blacklist[symbol]
        if time.time() > entry.banned_until:
            del self._blacklist[symbol]
            return False, ""
        hours_left = (entry.banned_until - time.time()) / 3600
        return True, f"Blacklisted {hours_left:.1f}h remaining: {entry.reason}"

    def get_blacklist(self) -> List[BlacklistEntry]:
        """Returns all active blacklist entries (expired ones removed)."""
        now = time.time()
        self._blacklist = {k: v for k, v in self._blacklist.items() if v.banned_until > now}
        return list(self._blacklist.values())

    def remove_from_blacklist(self, symbol: str) -> bool:
        """Manually remove a symbol from blacklist (for /sentinel unban SYM)."""
        if symbol in self._blacklist:
            del self._blacklist[symbol]
            logger.info(f"✅ {symbol} manually removed from blacklist")
            return True
        return False

    # ── 6. Threshold Calibration ──────────────────────────────

    async def threshold_calibration(self) -> List[ThresholdSuggestion]:
        """
        Reads actual outcome data and suggests specific parameter value changes.
        Not vague ("improve RR floor") — actual numbers ("raise rr_floor from 1.8 to 2.1
        for GeometricPattern based on 45 trades where 2.0+ R had 71% win rate vs 38% below").

        Returns a list of ThresholdSuggestion objects that can be converted to
        Sentinel approval requests.
        """
        try:
            from core.diagnostic_engine import diagnostic_engine
            from config.loader import cfg

            outcomes = list(diagnostic_engine._outcome_log)
            deaths = list(diagnostic_engine._death_log)

            if len(outcomes) < 20:
                logger.info("Threshold calibration: need 20+ outcomes")
                return []

            # RR distribution analysis: where does win rate break?
            rr_buckets = defaultdict(lambda: {"wins": 0, "losses": 0})
            for o in outcomes:
                rr_bucket = round(o.get("pnl_r", 0) * 2) / 2  # bucket to 0.5R
                rr_buckets[rr_bucket]["wins" if o.get("outcome") == "WIN" else "losses"] += 1

            # Confidence distribution: where does win rate break?
            conf_buckets = defaultdict(lambda: {"wins": 0, "losses": 0})
            for o in outcomes:
                bucket = (o.get("confidence", 50) // 5) * 5  # bucket to 5-point bands
                conf_buckets[bucket]["wins" if o.get("outcome") == "WIN" else "losses"] += 1

            # Strategy-regime win rates
            strat_regime = defaultdict(lambda: {"wins": 0, "losses": 0})
            for o in outcomes:
                key = f"{o.get('strategy','?')}@{o.get('regime','?')}"
                strat_regime[key]["wins" if o.get("outcome") == "WIN" else "losses"] += 1

            # Current config values (best-effort)
            try:
                current_rr = cfg.risk.min_rr if hasattr(cfg, 'risk') else 1.8
            except Exception:
                current_rr = 1.8

            prompt = f"""You are a quant analyst calibrating trading bot parameters based on actual outcome data.
Suggest SPECIFIC parameter changes with exact numbers, not vague advice.

Win rate by R achieved (in R multiples):
{json.dumps({str(k): v for k, v in sorted(rr_buckets.items())}, indent=2)}

Win rate by signal confidence (in 5-point bands):
{json.dumps({str(k): v for k, v in sorted(conf_buckets.items())}, indent=2)}

Win rate by strategy+regime combination (top 10 by volume):
{json.dumps({k: v for k, v in list(sorted(strat_regime.items(), key=lambda x: -(x[1]['wins']+x[1]['losses'])))[:10]}, indent=2)}

Current known parameters:
- RR floor: {current_rr}
- HTF block confidence threshold: ~85 (weekly bearish)
- Signal confidence minimum: ~65

Total outcomes in dataset: {len(outcomes)}
Total deaths analysed: {len(deaths)}

Identify 2-4 specific parameter changes that would improve win rate or remove clearly
losing setups. Each must include exact numbers, not ranges.

Return ONLY valid JSON (array):
[
  {{
    "parameter": "<exact parameter name>",
    "strategy": "<strategy name or 'global'>",
    "current_value": <number>,
    "suggested_value": <number>,
    "evidence": "<one sentence citing the data that supports this>",
    "expected_improvement": "<e.g. 'raises win rate from 42% to 61% based on 45 outcomes'>",
    "risk": "<LOW|MEDIUM|HIGH>"
  }}
]"""

            ai = self._get_ai()
            raw = await ai._call(prompt, call_label="threshold_calibration")
            data = ai._parse_json(raw)
            if not data or not isinstance(data, list):
                return []

            suggestions = []
            for item in data:
                try:
                    suggestions.append(ThresholdSuggestion(
                        parameter=item.get("parameter", ""),
                        strategy=item.get("strategy", "global"),
                        current_value=float(item.get("current_value", 0)),
                        suggested_value=float(item.get("suggested_value", 0)),
                        evidence=item.get("evidence", ""),
                        expected_improvement=item.get("expected_improvement", ""),
                        risk=item.get("risk", "MEDIUM"),
                    ))
                except Exception:
                    continue

            logger.info(f"🔧 Threshold calibration: {len(suggestions)} suggestions generated")
            return suggestions

        except Exception as e:
            logger.error(f"Threshold calibration error: {e}")
            return []

    # ── 7. Correlation Radar ──────────────────────────────────

    async def correlation_radar(
        self,
        symbols: List[str],
        ohlcv_dict_map: dict,
        break_threshold: float = 0.25,
    ) -> List[CorrelationBreakResult]:
        """
        Detects BTC correlation breakdowns — when a coin stops moving with BTC.
        A divergence of >= break_threshold from the coin's normal correlation
        is flagged as a potential independent breakout/breakdown.

        Does NOT call the AI — pure math. AI is only called if the divergence is
        large enough to warrant a classification. This keeps it zero-cost normally.
        """
        try:
            from data.api_client import api
            import numpy as np

            # Get BTC returns
            btc_data = ohlcv_dict_map.get("BTC/USDT", {}).get("1h", [])
            if not btc_data or len(btc_data) < 24:
                return []

            btc_closes = [float(c[4]) for c in btc_data[-48:]]
            btc_returns = [
                (btc_closes[i] - btc_closes[i-1]) / btc_closes[i-1]
                for i in range(1, len(btc_closes))
            ]

            breaks = []
            now = time.time()

            for symbol in symbols:
                if symbol == "BTC/USDT":
                    continue
                sym_data = ohlcv_dict_map.get(symbol, {}).get("1h", [])
                if not sym_data or len(sym_data) < 24:
                    continue

                sym_closes = [float(c[4]) for c in sym_data[-48:]]
                sym_returns = [
                    (sym_closes[i] - sym_closes[i-1]) / sym_closes[i-1]
                    for i in range(1, len(sym_closes))
                ]

                min_len = min(len(btc_returns), len(sym_returns))
                if min_len < 12:
                    continue

                btc_r = btc_returns[-min_len:]
                sym_r = sym_returns[-min_len:]

                # Full window correlation (baseline)
                baseline_corr = float(np.corrcoef(btc_r, sym_r)[0, 1])
                # Recent window correlation (last 6h)
                recent_corr = float(np.corrcoef(btc_r[-6:], sym_r[-6:])[0, 1]) if min_len >= 6 else baseline_corr

                # AUDIT FIX: ``np.corrcoef`` returns NaN when either series
                # has zero variance (e.g. stuck OHLCV, illiquid micro-caps,
                # or a data-feed outage).  ``NaN >= threshold`` silently
                # evaluates to False, so the correlation-break alert — a
                # primary independent-move detector — disappears exactly
                # when the data is garbage.  Skip the symbol instead.
                if np.isnan(baseline_corr) or np.isnan(recent_corr):
                    continue

                # Use cached baseline if available (more stable reference)
                cached = self._corr_baseline.get(symbol)
                if cached and (now - cached[1]) < 14400:   # 4h cache
                    reference_corr = cached[0]
                else:
                    reference_corr = baseline_corr
                    self._corr_baseline[symbol] = (baseline_corr, now)

                divergence = abs(recent_corr - reference_corr)

                if divergence >= break_threshold:
                    # Determine direction from recent price moves
                    sym_recent_return = (sym_closes[-1] - sym_closes[-7]) / sym_closes[-7]
                    btc_recent_return = (btc_closes[-1] - btc_closes[-7]) / btc_closes[-7]
                    direction = (
                        "DIVERGING_UP" if sym_recent_return > btc_recent_return
                        else "DIVERGING_DOWN"
                    )
                    confidence = min(95, divergence * 200)  # rough confidence from divergence magnitude

                    breaks.append(CorrelationBreakResult(
                        symbol=symbol,
                        normal_correlation=round(reference_corr, 3),
                        current_correlation=round(recent_corr, 3),
                        break_magnitude=round(divergence, 3),
                        direction=direction,
                        confidence=round(confidence, 1),
                        alert=(
                            f"{symbol} correlation broke: {reference_corr:.2f} → {recent_corr:.2f} "
                            f"({'moving independently upward' if direction == 'DIVERGING_UP' else 'moving independently downward'})"
                        ),
                    ))

            # Sort by break magnitude
            breaks.sort(key=lambda x: -x.break_magnitude)

            if breaks:
                logger.info(f"📡 Correlation radar: {len(breaks)} breaks detected: {[b.symbol for b in breaks[:5]]}")

            return breaks[:10]  # Cap at 10 most significant

        except Exception as e:
            logger.error(f"Correlation radar error: {e}")
            return []

    # ── 8. News Spike Scanner ─────────────────────────────────

    async def news_spike_scan(
        self,
        symbols: List[str],
        batch_size: int = 40,
    ) -> List[NewsSpike]:
        """
        Batch sentiment scan across all symbols in a single API call.
        Groups symbols into batches of `batch_size`, sends one call per batch.
        Flags any coin with HIGH-urgency bullish or bearish news catalyst.

        Much cheaper than per-symbol news calls because it groups symbols:
        40 symbols = 1 call instead of 40.
        """
        try:
            from analyzers.news_scraper import news_scraper

            # Collect recent headlines per symbol
            symbol_news = {}
            for sym in symbols:
                base = sym.split("/")[0]
                headlines = news_scraper.get_news_for_symbol(sym)
                if headlines:
                    symbol_news[base] = [h.get("title", "") for h in headlines[:3]]

            if not symbol_news:
                return []

            spikes = []
            symbol_list = list(symbol_news.items())

            # Process in batches to keep tokens manageable
            for i in range(0, len(symbol_list), batch_size):
                batch = symbol_list[i:i + batch_size]
                batch_dict = {sym: headlines for sym, headlines in batch}

                prompt = f"""You are a crypto news analyst. For each coin, quickly assess if there's a HIGH-impact news catalyst in the last hour that would move price significantly.

News headlines by coin:
{json.dumps(batch_dict, indent=2)}

Return ONLY coins with HIGH or MEDIUM urgency news. Skip coins with no significant news.
Return ONLY valid JSON (array, can be empty []):
[
  {{
    "symbol": "<COIN (no /USDT)>",
    "sentiment": "<BULLISH|BEARISH>",
    "urgency": "<HIGH|MEDIUM>",
    "key_headline": "<the most impactful headline, max 80 chars>",
    "estimated_move_window": "<minutes|hours>"
  }}
]"""

                ai = self._get_ai()
                raw = await ai._call(prompt, call_label=f"news_spike_batch_{i//batch_size}")
                batch_results = ai._parse_json(raw)

                if batch_results and isinstance(batch_results, list):
                    for item in batch_results:
                        base = item.get("symbol", "")
                        if not base:
                            continue
                        spikes.append(NewsSpike(
                            symbol=f"{base}/USDT",
                            sentiment=item.get("sentiment", "NEUTRAL"),
                            urgency=item.get("urgency", "MEDIUM"),
                            headline_count=len(symbol_news.get(base, [])),
                            key_headline=item.get("key_headline", ""),
                            estimated_move_window=item.get("estimated_move_window", "hours"),
                        ))

            # Sort HIGH urgency first
            spikes.sort(key=lambda x: (0 if x.urgency == "HIGH" else 1, x.symbol))

            if spikes:
                high = [s for s in spikes if s.urgency == "HIGH"]
                logger.info(
                    f"📰 News spike scan: {len(spikes)} catalysts found "
                    f"({len(high)} HIGH urgency)"
                )

            return spikes

        except Exception as e:
            logger.error(f"News spike scan error: {e}")
            return []


# ── Singleton ─────────────────────────────────────────────────
sentinel = SentinelFeatures()
