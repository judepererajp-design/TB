"""
TitanBot Pro — Three-AI Veto System
=====================================
v2.10: Autonomous improvement system using three AI models in a debate pattern.

ARCHITECTURE:
  Proposer  (Llama 3.3 70B)   — generates a specific change with evidence
  Challenger (Qwen3 80B)      — attacks the proposal, finds flaws
  Arbitrator (GPT-OSS 120B)   — weighs both sides, decides confidence score

THREE TIERS (what can auto-execute vs requires approval):

  Tier 1 — AUTO-EXECUTE (no approval, fully reversible, small blast radius):
    - Symbol blacklist (24h temp ban for chronic false positives)
    - Strategy cool-down (pause strategy for 4h if loss streak detected)
    - No-trade zone additions (flag symbol as dangerous for 2h)

  Tier 2 — VETO REQUIRED (auto-execute only if Arbitrator confidence ≥ 80%):
    - Dead signal suppression (reduce weight of a failing strategy)
    - Cohort bias overrides (lower signal confidence in HTF conflict)
    - Regime-specific strategy disabling (e.g. disable RisingWedge in BULL_TREND)

  Tier 3 — ALWAYS HUMAN APPROVAL (AI presents reasoning, you click Apply):
    - RR floor changes
    - HTF guardrail threshold changes
    - Confidence minimum changes
    - Position sizing changes

SAFETY GATES:
  - Every auto-applied change is logged with full reasoning chain
  - Performance gate: if win rate drops >5% in 4h after any Tier 1/2 change,
    auto-reverts without human intervention
  - Hard parameter bounds: AI cannot set values outside safe ranges
    (e.g. RR floor cannot go below 1.3 or above 3.5)
  - Max 3 auto-changes per 24h window to prevent runaway adaptation
  - All Tier 3 proposals appear in the dashboard Approvals panel

PLAIN ENGLISH AUDIT TRAIL:
  Every change generates a human-readable summary:
  "FundingRateArb was generating 0% SHORT signals for 6h. Paused in BULL_TREND
   regime for 4h while monitoring. Win rate before pause: 38%. Arbitrator
   confidence: 87% (Challenger found no flaws in the evidence)."
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Model assignments ─────────────────────────────────────────
from utils.free_llm import call_llm, parse_json_response, MODEL_PROPOSER, MODEL_CHALLENGER, MODEL_ARBITRATOR

_MODEL_PROPOSER   = MODEL_PROPOSER
_MODEL_CHALLENGER = MODEL_CHALLENGER
_MODEL_ARBITRATOR = MODEL_ARBITRATOR

# Safe parameter bounds (AI cannot exceed these regardless of confidence)
PARAM_BOUNDS = {
    "rr_floor":              (1.3, 3.5),
    "confidence_floor":      (55.0, 85.0),
    "htf_block_threshold":   (70.0, 92.0),
    "position_size_pct":     (0.5, 5.0),
}

# Plain-English explanations for each Tier 3 change type
# Shown in the Approvals dashboard so non-traders can make informed decisions
PLAIN_ENGLISH_GUIDE = {
    'adjust_rr_floor': {
        'what':         'The minimum reward-to-risk ratio a signal must have before publishing.',
        'higher_means': 'Fewer signals but each needs more profit potential relative to risk.',
        'lower_means':  'More signals publish, including ones with tighter profit targets.',
        'safe_range':   '1.5–2.5 is typical. Below 1.3 is risky. Above 3.0 = almost nothing publishes.',
        'approve_when': 'Win rate is low and signals keep hitting SL before TP.',
        'reject_when':  'You are already getting too few signals.',
    },
    'adjust_confidence_floor': {
        'what':         'The minimum confidence score (0–100) a signal needs before publishing.',
        'higher_means': 'Only high-conviction signals publish — fewer but higher quality.',
        'lower_means':  'More signals publish, including ones the bot is less certain about.',
        'safe_range':   '60–75 is typical. Below 55 = noise. Above 85 = almost nothing publishes.',
        'approve_when': 'Too many weak signals are publishing and losing.',
        'reject_when':  'You are missing good moves because signals are being blocked.',
    },
    'adjust_htf_threshold': {
        'what':         'How strong a signal must be to trade against the weekly trend direction.',
        'higher_means': 'Harder to trade counter-trend. More conservative.',
        'lower_means':  'More counter-trend signals get through. More signals, more risk.',
        'safe_range':   '72–85 is typical.',
        'approve_when': 'Clear reversals are being blocked by an outdated weekly trend bias.',
        'reject_when':  'The weekly trend is strong and clear — trust it.',
    },
    'adjust_position_size': {
        'what':         'What percentage of your account is risked on each trade.',
        'higher_means': 'Larger positions. More profit if right, more loss if wrong.',
        'lower_means':  'Smaller positions. Lower risk per trade.',
        'safe_range':   '0.5–2% per trade. Never approve above 3%.',
        'approve_when': 'Win rate is consistently above 60% over 30+ trades.',
        'reject_when':  'Win rate is below 55% OR you have fewer than 20 closed trades.',
    },
    'execution_config': {
        'what':         'How the bot places and manages orders (entry zones, timing, triggers).',
        'higher_means': 'Varies by specific parameter.',
        'lower_means':  'Varies by specific parameter.',
        'safe_range':   'Check the specific change description.',
        'approve_when': 'Fill rate has been consistently low over many signals (not just 1 day).',
        'reject_when':  'The bot has fewer than 20 expired signals — too early to judge.',
    },
}

MAX_AUTO_CHANGES_PER_DAY   = 3
PERF_GATE_HOURS            = 4      # hours to monitor after a change
PERF_GATE_DROP_THRESHOLD   = 0.05   # 5% win rate drop triggers auto-revert
MIN_ARBITRATOR_CONFIDENCE  = 80.0   # minimum for Tier 2 auto-execute
MIN_TRADE_SAMPLE           = 20     # floor: require at least this many closed trades (7d)
AUTO_CHANGE_COOLDOWN_H     = 6.0    # minimum hours between auto-changes of the same type
ADAPTIVE_TRADE_SAMPLE_PCT  = 0.05   # adaptive floor = max(MIN_TRADE_SAMPLE, this × 7d volume)
BLEND_WEIGHT_7D            = 0.70   # 7-day weight in blended win rate (normal regime)
BLEND_WEIGHT_1D            = 0.30   # 24-hour weight in blended win rate (normal regime)
REGIME_SHIFT_THRESHOLD     = 0.15   # If |wr_24h − wr_7d| exceeds this, use 50/50 blend


@dataclass
class VetoProposal:
    proposal_id: str
    tier: int                    # 1, 2, or 3
    change_type: str             # 'blacklist_symbol', 'pause_strategy', 'adjust_param', etc.
    description: str             # plain English
    change_data: dict            # the actual change {param: new_value, ...}
    evidence: str                # data that supports this change
    proposer_reasoning: str
    challenger_objections: str
    arbitrator_verdict: str
    arbitrator_confidence: float # 0-100
    created_at: float = field(default_factory=time.time)
    auto_applied: bool = False
    applied_at: float = 0.0
    reverted: bool = False
    reverted_at: float = 0.0
    revert_reason: str = ""
    win_rate_before: float = 0.0
    win_rate_after:  float = 0.0
    # AI attempt log: which model answered each role (proposer/challenger/arbitrator).
    # Each entry: {"role": str, "model": str, "attempts": [{"attempt": int, "model": str, "succeeded": bool}]}
    # Populated during run_debate_cycle(). Enables post-hoc debugging of inconsistent veto decisions.
    ai_attempts: List[dict] = field(default_factory=list)


class VetoSystem:
    """
    Three-AI debate system for autonomous bot improvement.
    Uses Proposer/Challenger/Arbitrator pattern to validate changes
    before auto-applying or escalating to human approval.
    """

    def __init__(self):
        self._enabled = True
        self._proposals: List[VetoProposal] = []
        self._max_proposals = 500  # Prevent unbounded growth
        self._auto_changes_today = 0
        self._day_reset = time.time()
        self._last_run = 0.0
        self._run_interval = 7200.0   # Run debate every 2h (same as audit batch)
        self._perf_monitoring: Dict[str, dict] = {}  # proposal_id -> monitoring state
        self._lock = asyncio.Lock()
        # Per-change-type cooldown: tracks last applied timestamp to prevent
        # rapid re-application of the same type of change (feedback oscillation).
        self._last_change_by_type: Dict[str, float] = {}

    def initialize(self, api_key: str = ""):
        """Initialize veto system. No API key needed (uses free LLMs)."""
        self._enabled = True
        logger.info("⚖️  Veto system initialised — 3-model debate active (Pollinations.ai, no key)")

    async def _call(
        self,
        prompt: str,
        model: str,
        label: str,
        attempts_out: Optional[List[dict]] = None,
    ) -> Optional[str]:
        if not self._enabled:
            return None
        raw = await call_llm(
            prompt, model=model, temperature=0.15, max_tokens=800,
            attempts_out=attempts_out,
        )
        if raw:
            logger.debug(f"[VETO:{label}] {raw[:120]}")
        return raw

    def _parse_json(self, raw: str) -> Optional[dict]:
        return parse_json_response(raw)

    # ── Main debate loop ──────────────────────────────────────

    async def run_debate_cycle(self, session_data: dict) -> List[VetoProposal]:
        """
        Full Proposer → Challenger → Arbitrator debate cycle.
        Called every 2h from the diagnostic loop.
        Returns list of new proposals (some auto-applied, some awaiting approval).
        """
        if not self._enabled:
            return []
        if time.time() - self._last_run < self._run_interval:
            return []

        # Reset daily counter
        if time.time() - self._day_reset > 86400:
            self._auto_changes_today = 0
            self._day_reset = time.time()

        async with self._lock:
            self._last_run = time.time()
            proposals = []

            try:
                # Step 1: Proposer generates candidate changes
                prop_raw = await self._proposer_call(session_data)
                if not prop_raw:
                    return []

                prop_data = self._parse_json(prop_raw)
                if not prop_data or not isinstance(prop_data.get("proposals"), list):
                    return []

                for p in prop_data["proposals"][:3]:  # max 3 proposals per cycle
                    try:
                        proposal = await self._debate_one(p, session_data)
                        if proposal:
                            proposals.append(proposal)
                            await self._handle_proposal(proposal)
                    except Exception as e:
                        logger.debug(f"Debate error for proposal: {e}")

            except Exception as e:
                logger.error(f"Veto debate cycle error: {e}")

            return proposals

    async def _proposer_call(self, sd: dict) -> Optional[str]:
        """Proposer: analyse session data and suggest specific changes."""
        prompt = f"""You are TitanBot's improvement system. Analyse the session data and propose specific changes.

SESSION DATA:
  Win rate: {sd.get('win_rate', 0):.1f}%
  Signals published: {sd.get('published', 0)}
  Top kill reason: {sd.get('top_kill', 'unknown')}
  Direction bias (LONG/SHORT): {sd.get('long_count', 0)}/{sd.get('short_count', 0)}
  HTF blocks (LONG/SHORT): {sd.get('htf_long_blocks', 0)}/{sd.get('htf_short_blocks', 0)}
  Strategy win rates: {json.dumps(sd.get('strategy_win_rates', {}), indent=2)}
  Current regime: {sd.get('regime', 'UNKNOWN')}
  Chronic false positive symbols: {json.dumps(sd.get('high_fpr_symbols', []))}
  Loss streak strategies: {json.dumps(sd.get('loss_streak_strategies', []))}

CHANGE TIERS:
  Tier 1 (auto, no approval): blacklist_symbol, pause_strategy, add_no_trade_zone
  Tier 2 (auto if 80%+ confidence): suppress_strategy_weight, disable_strategy_in_regime
  Tier 3 (always human approval): adjust_rr_floor, adjust_confidence_floor, adjust_htf_threshold

Propose 1-3 specific, evidence-based changes. Only propose what the data clearly supports.
If everything looks fine, return an empty proposals list.

Return ONLY valid JSON:
{{
  "proposals": [
    {{
      "tier": <1, 2, or 3>,
      "change_type": "<type from above>",
      "description": "<plain English, what changes and why, 1-2 sentences>",
      "change_data": {{}},
      "evidence": "<specific data points that support this change>"
    }}
  ]
}}"""
        return await self._call(prompt, _MODEL_PROPOSER, "proposer")

    async def _debate_one(self, proposal_data: dict, sd: dict) -> Optional[VetoProposal]:
        """Run Challenger + Arbitrator for a single proposal."""
        import uuid

        desc = proposal_data.get("description", "")
        evidence = proposal_data.get("evidence", "")
        tier = int(proposal_data.get("tier", 3))

        # Step 2: Challenger attacks the proposal
        challenge_prompt = f"""You are a sceptical risk manager reviewing a proposed bot change.
Your job is to find flaws, edge cases, and risks.

PROPOSED CHANGE:
  Description: {desc}
  Evidence: {evidence}
  Change data: {json.dumps(proposal_data.get('change_data', {}))}
  Tier: {tier}

Find problems with this proposal:
- Is the evidence statistically sufficient? (small sample = bad)
- Could there be a confounding factor?
- What's the downside if this is wrong?
- Is the change reversible?

Return ONLY valid JSON:
{{
  "objections": ["<objection 1>", "<objection 2>"],
  "fatal_flaw": <true if this change should definitely be rejected>,
  "risk_level": "<LOW|MEDIUM|HIGH>",
  "challenger_verdict": "<approve with caution|reject|approve>"
}}"""

        _challenger_attempts: List[dict] = []
        challenge_raw = await self._call(
            challenge_prompt, _MODEL_CHALLENGER, "challenger",
            attempts_out=_challenger_attempts,
        )
        challenge_data = self._parse_json(challenge_raw) or {}

        # If Challenger found a fatal flaw, don't proceed
        if challenge_data.get("fatal_flaw", False):
            logger.info(f"⚖️  Challenger rejected: {desc[:60]}")
            return None

        objections = challenge_data.get("objections", [])

        # Step 3: Arbitrator weighs both sides
        arb_prompt = f"""You are the final decision maker for a bot improvement proposal.
Weigh the proposal against the challenger's objections and assign a confidence score.

PROPOSAL:
  {desc}
  Evidence: {evidence}

CHALLENGER OBJECTIONS:
  {json.dumps(objections)}
  Challenger verdict: {challenge_data.get('challenger_verdict', 'unclear')}

SAFE PARAMETER BOUNDS (changes must stay within):
  {json.dumps(PARAM_BOUNDS)}

Consider:
- Is the evidence strong enough to act on?
- Are the objections valid or nitpicking?
- What's the worst case if this is wrong?

Return ONLY valid JSON:
{{
  "confidence": <0-100, how confident you are this change is beneficial>,
  "verdict": "<approve|reject|escalate_to_human>",
  "reasoning": "<1-2 sentences on why>",
  "adjusted_change_data": {{}}
}}"""

        _arbitrator_attempts: List[dict] = []
        arb_raw = await self._call(
            arb_prompt, _MODEL_ARBITRATOR, "arbitrator",
            attempts_out=_arbitrator_attempts,
        )
        arb_data = self._parse_json(arb_raw) or {}

        confidence = float(arb_data.get("confidence", 0))
        verdict = arb_data.get("verdict", "escalate_to_human")

        # If Arbitrator says reject, skip
        if verdict == "reject":
            logger.info(f"⚖️  Arbitrator rejected (conf={confidence:.0f}%): {desc[:60]}")
            return None

        # Use adjusted change data if provided
        change_data = arb_data.get("adjusted_change_data") or proposal_data.get("change_data", {})

        # Validate bounds for parameter changes
        for param, value in change_data.items():
            if param in PARAM_BOUNDS:
                lo, hi = PARAM_BOUNDS[param]
                if not (lo <= float(value) <= hi):
                    logger.warning(f"⚖️  Veto: {param}={value} out of bounds [{lo},{hi}] — clamped")
                    change_data[param] = max(lo, min(hi, float(value)))

        proposal = VetoProposal(
            proposal_id=str(uuid.uuid4())[:8],
            tier=tier,
            change_type=proposal_data.get("change_type", "unknown"),
            description=desc,
            change_data=change_data,
            evidence=evidence,
            proposer_reasoning=evidence,
            challenger_objections="\n".join(objections),
            arbitrator_verdict=arb_data.get("reasoning", ""),
            arbitrator_confidence=confidence,
            ai_attempts=[
                {"role": "challenger", "model": _MODEL_CHALLENGER, "attempts": _challenger_attempts},
                {"role": "arbitrator", "model": _MODEL_ARBITRATOR, "attempts": _arbitrator_attempts},
            ],
        )
        return proposal

    async def _handle_proposal(self, proposal: VetoProposal):
        """Decide what to do with a vetted proposal based on tier and confidence."""
        tier = proposal.tier
        conf = proposal.arbitrator_confidence

        if tier == 1:
            # Always auto-execute Tier 1 (reversible, low risk)
            if self._auto_changes_today < MAX_AUTO_CHANGES_PER_DAY:
                if await self._auto_change_allowed(proposal):
                    await self._apply_change(proposal, reason="Tier 1 auto-execute")
                else:
                    self._proposals.append(proposal)
                    await self._notify_human(proposal)
            else:
                logger.info(f"⚖️  Tier 1 skipped — daily auto-change limit reached")
                self._proposals.append(proposal)

        elif tier == 2:
            if conf >= MIN_ARBITRATOR_CONFIDENCE and self._auto_changes_today < MAX_AUTO_CHANGES_PER_DAY:
                if await self._auto_change_allowed(proposal):
                    await self._apply_change(proposal, reason=f"Tier 2 auto (confidence={conf:.0f}%)")
                else:
                    self._proposals.append(proposal)
                    await self._notify_human(proposal)
            else:
                # Escalate to human with confidence info
                logger.info(f"⚖️  Tier 2 escalated to human (conf={conf:.0f}% < {MIN_ARBITRATOR_CONFIDENCE}%)")
                self._proposals.append(proposal)
                await self._notify_human(proposal)

        elif tier == 3:
            # Always escalate Tier 3
            logger.info(f"⚖️  Tier 3 proposal awaiting human approval: {proposal.description[:60]}")
            self._proposals.append(proposal)
            await self._notify_human(proposal)

        # Prevent unbounded growth: keep only recent proposals
        if len(self._proposals) > self._max_proposals:
            self._proposals = self._proposals[-self._max_proposals:]

    async def _get_current_win_rate(self) -> float:
        """Return blended win rate in 0–100 scale.

        Normal regime:   70% × 7d  +  30% × 24h
        Regime shift:    50% × 7d  +  50% × 24h  (when |wr_24h − wr_7d| > REGIME_SHIFT_THRESHOLD)

        The deviation check acts as a fast-path trigger: if recent performance
        has diverged sharply from the weekly baseline (e.g., a bull run reversed
        in the last 24 h), the system should react faster than a pure 70/30 blend
        would allow.  Falls back to whichever window has data; returns 0.0 on failure.
        """
        try:
            from data.database import db as _db
            _stats_7d, _stats_1d = (
                await _db.get_performance_stats(days=7),
                await _db.get_performance_stats(days=1),
            )

            def _wr(stats: dict):
                _w = (stats or {}).get("wins",   0) or 0
                _l = (stats or {}).get("losses", 0) or 0
                _t = _w + _l
                return (_w / _t) if _t else None

            _wr_7d = _wr(_stats_7d)
            _wr_1d = _wr(_stats_1d)

            if _wr_7d is None and _wr_1d is None:
                return 0.0
            if _wr_7d is None:
                return round(_wr_1d * 100, 1)
            if _wr_1d is None:
                return round(_wr_7d * 100, 1)

            # Regime shift check: large 7d↔24h divergence means the market has
            # changed recently — react with equal weighting on both windows.
            _deviation = abs(_wr_1d - _wr_7d)
            if _deviation > REGIME_SHIFT_THRESHOLD:
                logger.info(
                    f"⚖️  Regime shift detected — |wr_24h − wr_7d| = "
                    f"{_deviation:.2f} > {REGIME_SHIFT_THRESHOLD}. "
                    f"Switching to 50/50 blend (7d={_wr_7d:.2%}, 24h={_wr_1d:.2%})"
                )
                return round((0.5 * _wr_7d + 0.5 * _wr_1d) * 100, 1)

            return round((BLEND_WEIGHT_7D * _wr_7d + BLEND_WEIGHT_1D * _wr_1d) * 100, 1)
        except Exception:
            return 0.0

    async def _auto_change_allowed(self, proposal: VetoProposal) -> bool:
        """
        Return True only if:
          1. Enough closed trades exist (MIN_TRADE_SAMPLE) so win-rate data is stable.
          2. The same change_type has not been auto-applied within AUTO_CHANGE_COOLDOWN_H.
        Prevents feedback oscillation when the bot has few trades or reacts too quickly.
        """
        # 1. Minimum trade sample guard (adaptive floor)
        # In high-frequency systems, 20 trades represents only a tiny slice of
        # weekly volume — decisions made on that slice are statistically fragile.
        # Require max(MIN_TRADE_SAMPLE, 5% of 7d volume) so the floor scales
        # proportionally with system activity.
        try:
            from data.database import db as _db
            _stats = await _db.get_performance_stats(days=7)
            _wins   = (_stats or {}).get("wins",   0) or 0
            _losses = (_stats or {}).get("losses", 0) or 0
            _closed = _wins + _losses
            _adaptive_min = max(MIN_TRADE_SAMPLE, int(_closed * ADAPTIVE_TRADE_SAMPLE_PCT))
            if _closed < _adaptive_min:
                logger.info(
                    f"⚖️  Auto-change deferred — only {_closed}/{_adaptive_min} closed "
                    f"trades (7d, adaptive min). Escalating to human."
                )
                return False
        except Exception:
            pass  # If DB unavailable allow the change; don't hard-block

        # 2. Per-type cooldown guard
        _last = self._last_change_by_type.get(proposal.change_type, 0.0)
        _elapsed_h = (time.time() - _last) / 3600
        if _elapsed_h < AUTO_CHANGE_COOLDOWN_H:
            logger.info(
                f"⚖️  Auto-change deferred — '{proposal.change_type}' was last applied "
                f"{_elapsed_h:.1f}h ago (cooldown={AUTO_CHANGE_COOLDOWN_H}h). Escalating to human."
            )
            return False

        return True

    async def _apply_change(self, proposal: VetoProposal, reason: str):
        """Apply an auto-approved change and start performance monitoring."""
        try:
            # Record win rate before change (for performance gate)
            try:
                proposal.win_rate_before = await self._get_current_win_rate()
            except Exception:
                proposal.win_rate_before = 0.0

            # Apply based on change type
            applied = await self._execute_change(proposal)
            if not applied:
                return

            proposal.auto_applied = True
            proposal.applied_at = time.time()
            self._auto_changes_today += 1
            self._proposals.append(proposal)
            # Record per-type timestamp for cooldown guard
            self._last_change_by_type[proposal.change_type] = proposal.applied_at

            # Start performance monitoring
            self._perf_monitoring[proposal.proposal_id] = {
                "proposal": proposal,
                "start_time": time.time(),
                "baseline_wr": proposal.win_rate_before,
            }

            logger.info(
                f"✅ VETO AUTO-APPLY [{reason}]: {proposal.description[:80]}\n"
                f"   Arbitrator confidence: {proposal.arbitrator_confidence:.0f}%\n"
                f"   Challenger said: {proposal.challenger_objections[:80]}"
            )

        except Exception as e:
            logger.error(f"Veto apply error: {e}")

    async def _execute_change(self, proposal: VetoProposal) -> bool:
        """Execute the actual change based on change_type."""
        ct = proposal.change_type
        cd = proposal.change_data

        try:
            if ct == "blacklist_symbol":
                from analyzers.sentinel_features import sentinel
                symbol = cd.get("symbol", "")
                if symbol:
                    from analyzers.sentinel_features import BlacklistEntry
                    entry = BlacklistEntry(
                        symbol=symbol,
                        reason=f"Veto auto-ban: {proposal.description[:80]}",
                        false_positive_rate=float(cd.get("fpr", 0.9)),
                        sample_size=int(cd.get("sample_size", 20)),
                        banned_until=time.time() + float(cd.get("hours", 24)) * 3600,
                    )
                    sentinel._blacklist[symbol] = entry
                    return True

            elif ct == "pause_strategy":
                strategy = cd.get("strategy", "")
                hours = float(cd.get("hours", 4))
                if strategy:
                    # Store pause in a simple dict on the engine if available
                    try:
                        from core.engine import engine
                        if not hasattr(engine, "_strategy_pauses"):
                            engine._strategy_pauses = {}
                        engine._strategy_pauses[strategy] = time.time() + hours * 3600
                        logger.info(f"⏸️  Strategy paused: {strategy} for {hours}h")
                        return True
                    except Exception:
                        return False

            elif ct == "disable_strategy_in_regime":
                strategy = cd.get("strategy", "")
                regime   = cd.get("regime", "")
                hours    = float(cd.get("hours", 6))
                if strategy and regime:
                    try:
                        from core.engine import engine
                        if not hasattr(engine, "_regime_strategy_bans"):
                            engine._regime_strategy_bans = {}
                        engine._regime_strategy_bans[f"{strategy}:{regime}"] = time.time() + hours * 3600
                        return True
                    except Exception:
                        return False

            # Tier 3 changes (parameter adjustments) — stored as pending for human
            elif ct in ("adjust_rr_floor", "adjust_confidence_floor", "adjust_htf_threshold"):
                # These never auto-apply, they go to human approval queue
                return False

        except Exception as e:
            logger.error(f"Execute change error ({ct}): {e}")
            return False

        return False

    async def _notify_human(self, proposal: VetoProposal):
        """Send proposal to Telegram approval queue and dashboard."""
        try:
            from core.diagnostic_engine import diagnostic_engine
            from dataclasses import dataclass as _dc

            # Convert to PendingApproval format
            risk = "LOW" if proposal.tier == 1 else "MEDIUM" if proposal.tier == 2 else "HIGH"
            from core.diagnostic_engine import PendingApproval
            approval = PendingApproval(
                approval_id=proposal.proposal_id,
                change_type=proposal.change_type,
                description=proposal.description,
                old_value=str(proposal.change_data.get("current_value", "—")),
                new_value=str(proposal.change_data.get("new_value", str(proposal.change_data))),
                reason=(
                    f"Proposer: {proposal.evidence[:200]}\n"
                    f"Challenger: {proposal.challenger_objections[:200]}\n"
                    f"Arbitrator ({proposal.arbitrator_confidence:.0f}% conf): {proposal.arbitrator_verdict[:200]}"
                ),
                risk_level=risk,
                estimated_impact=f"Arbitrator confidence: {proposal.arbitrator_confidence:.0f}%",
            )
            diagnostic_engine._pending_approvals[proposal.proposal_id] = approval
            if diagnostic_engine.on_send_approval:
                await diagnostic_engine.on_send_approval(approval)
        except Exception as e:
            logger.debug(f"Notify human error: {e}")

    # ── Performance gate ──────────────────────────────────────

    async def check_performance_gate(self):
        """
        Called every 30min by diagnostic loop.
        If win rate dropped >5% since an auto-change, auto-revert.
        """
        now = time.time()
        to_revert = []

        for pid, monitor in list(self._perf_monitoring.items()):
            elapsed = now - monitor["start_time"]
            if elapsed < 1800:  # Wait at least 30min for data
                continue
            if elapsed > PERF_GATE_HOURS * 3600:
                del self._perf_monitoring[pid]
                continue

            try:
                current_wr = await self._get_current_win_rate()
                drop = monitor["baseline_wr"] - current_wr

                if drop > PERF_GATE_DROP_THRESHOLD * 100:
                    to_revert.append((pid, monitor["proposal"], drop))
            except Exception:
                pass

        for pid, proposal, drop in to_revert:
            await self._revert_change(
                proposal,
                reason=f"Performance gate: win rate dropped {drop:.1f}% after change"
            )
            del self._perf_monitoring[pid]

    async def _revert_change(self, proposal: VetoProposal, reason: str):
        """Undo an auto-applied change."""
        ct = proposal.change_type
        cd = proposal.change_data

        try:
            if ct == "blacklist_symbol":
                from analyzers.sentinel_features import sentinel
                sentinel.remove_from_blacklist(cd.get("symbol", ""))

            elif ct == "pause_strategy":
                from core.engine import engine
                if hasattr(engine, "_strategy_pauses"):
                    engine._strategy_pauses.pop(cd.get("strategy", ""), None)

            elif ct == "disable_strategy_in_regime":
                from core.engine import engine
                if hasattr(engine, "_regime_strategy_bans"):
                    key = f"{cd.get('strategy','')}:{cd.get('regime','')}"
                    engine._regime_strategy_bans.pop(key, None)

            proposal.reverted = True
            proposal.reverted_at = time.time()
            proposal.revert_reason = reason

            logger.warning(
                f"↩️  VETO AUTO-REVERT: {proposal.description[:80]}\n"
                f"   Reason: {reason}"
            )
        except Exception as e:
            logger.error(f"Revert error: {e}")

    # ── Public API ────────────────────────────────────────────

    def get_proposals(self, limit: int = 20) -> List[dict]:
        """Get recent proposals for dashboard display."""
        proposals = sorted(self._proposals, key=lambda p: p.created_at, reverse=True)[:limit]
        return [
            {
                "id":           p.proposal_id,
                "tier":         p.tier,
                "type":         p.change_type,
                "description":  p.description,
                "evidence":     p.evidence,
                "confidence":   p.arbitrator_confidence,
                "challenger":   p.challenger_objections[:200],
                "arbitrator":   p.arbitrator_verdict,
                "auto_applied": p.auto_applied,
                "applied_at":   p.applied_at,
                "reverted":     p.reverted,
                "revert_reason":p.revert_reason,
                "created_at":   p.created_at,
                "win_rate_before": p.win_rate_before,
                "win_rate_after":  p.win_rate_after,
                "plain_english": PLAIN_ENGLISH_GUIDE.get(p.change_type, {}),
            }
            for p in proposals
        ]

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "auto_changes_today": self._auto_changes_today,
            "max_auto_per_day": MAX_AUTO_CHANGES_PER_DAY,
            "proposals_total": len(self._proposals),
            "proposals_auto_applied": sum(1 for p in self._proposals if p.auto_applied),
            "proposals_reverted": sum(1 for p in self._proposals if p.reverted),
            "monitoring_count": len(self._perf_monitoring),
            "models": {
                "proposer":   _MODEL_PROPOSER,
                "challenger": _MODEL_CHALLENGER,
                "arbitrator": _MODEL_ARBITRATOR,
            }
        }

    async def manual_revert_last(self) -> bool:
        """Revert the most recent auto-applied change. Called from dashboard."""
        for p in sorted(self._proposals, key=lambda x: x.applied_at, reverse=True):
            if p.auto_applied and not p.reverted:
                await self._revert_change(p, reason="Manual revert via dashboard")
                return True
        return False


# ── Singleton ─────────────────────────────────────────────────
veto_system = VetoSystem()
