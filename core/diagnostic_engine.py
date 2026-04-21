from collections import Counter, deque as _deque
"""
TitanBot Pro — Self-Healing Diagnostic Engine
===============================================
Observes the bot's own behaviour and generates actionable reports.

What it does:
  1. Monitors error patterns in real-time from structured log records
  2. Analyses performance gaps (strategy WR, confidence calibration)
  3. Detects edge cases (impossible values, feedback loops, stale data)
  4. Sends structured diagnostic reports to a separate Telegram channel
  5. Proposes runtime config changes — applied only after user YES tap
  6. Makes safe auto-actions within pre-approved bounds (circuit breaker,
     temporary symbol exclusion, strategy suppression)

What it NEVER does:
  - Modify Python source files
  - Modify settings.yaml
  - Touch open positions or live order management
  - Apply any change that isn't immediately reversible

All pending approvals are stored in DB and presented via Telegram buttons.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

# ── Data classes ──────────────────────────────────────────────

@dataclass
class ErrorRecord:
    error_type: str
    module: str
    message: str
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    symbols_affected: List[str] = field(default_factory=list)


@dataclass
class PendingApproval:
    approval_id: str
    change_type: str          # 'rr_floor' | 'confidence_floor' | 'suppress_strategy' | 'exclude_symbol'
    description: str
    old_value: object
    new_value: object
    reason: str
    risk_level: str           # 'LOW' | 'MEDIUM' | 'HIGH'
    estimated_impact: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0   # 0 = no expiry
    # Personalised AI verdict generated at proposal time using live bot metrics
    user_verdict: str = ""    # "Your win rate is X%. AI says: approve/wait because..."
    verdict_approve: bool = True  # True=lean approve, False=lean reject/wait


@dataclass
class DiagnosticReport:
    period_hours: float
    scan_count: int
    signals_generated: int
    signals_published: int
    criticals: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    performance_gaps: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    healthy_items: List[str] = field(default_factory=list)
    auto_actions_taken: List[str] = field(default_factory=list)
    ai_recommendations: List[str] = field(default_factory=list)


class DiagnosticEngine:
    """
    Always-on observer. Runs even when signal AI is disabled.
    """

    def __init__(self):
        # Error tracking
        self._error_counts: Dict[str, ErrorRecord] = {}  # key: "module:error_type"
        self._error_rate_window: _deque = _deque(maxlen=10_000)  # timestamps of recent errors

        # Signal death tracking
        self._death_log: _deque = _deque(maxlen=1000)  # last 1000 signal deaths

        # Trade outcome tracking
        self._outcome_log: List[dict] = []                # last 500 outcomes

        # ── NEW: Broader observation storage ─────────────────
        self._telegram_log: List[dict] = []               # last 200 telegram sends
        self._signal_card_log: List[dict] = []            # last 100 signal cards
        self._api_latency: Dict[str, List[dict]] = {}     # per-endpoint latency
        self._narrative_quality_log: List[dict] = []      # AI narrative quality scores
        self._scan_timings: List[dict] = []               # per-symbol scan durations
        self._db_timings: List[dict] = []                 # DB operation timings

        # Scan stats (per-report cycle — reset after each 6h diagnostic report)
        self._scan_count: int = 0
        self._signals_generated: int = 0
        # Cumulative totals (never reset — true uptime stats for dashboard)
        self._total_scan_count: int = 0
        self._total_signals_generated: int = 0
        self._total_signals_published: int = 0
        self._signals_published: int = 0
        self._last_report_time: float = 0.0
        self._report_interval: float = 21600.0            # 6 hours

        # GROUP 4: New audit timers — track when each new AI audit last ran
        self._last_execution_funnel_audit: float = 0.0   # 4A: runs every 2h
        self._last_strategy_regime_audit: float  = 0.0   # 4C: runs every 4h
        self._execution_audit_interval:  float   = 7200  # 2h
        self._strategy_audit_interval:   float   = 14400 # 4h

        # GROUP 5: Adaptive parameter tuner — once per 24h
        # Adjusts penalty/multiplier magnitudes based on per-market-state win rates.
        # The ParamTuner has its own internal _last_run guard so no separate timer
        # is needed here; we just call it on every watch loop iteration and let it
        # self-throttle.
        self._param_tuner_enabled: bool = True

        # Pending approvals
        self._pending_approvals: Dict[str, PendingApproval] = {}
        self._applied_overrides: Dict[str, dict] = {}    # key -> {value, previous, applied_at}

        # Proposal deduplication — track last time each proposal type was sent
        # Prevents spamming the same proposal every 60s
        self._last_proposals: Dict[str, float] = {}

        # Global rate cap — max 4 proposals per hour across ALL types.
        # Defense-in-depth: even if individual dedup fails, user won't get 25+ messages.
        self._proposal_timestamps: _deque = _deque(maxlen=20)  # timestamps of recent proposals
        self._max_proposals_per_hour: int = 4

        # Callbacks (set by engine.py)
        self.on_send_report: Optional[Callable] = None       # async (text: str) -> None
        self.on_send_approval: Optional[Callable] = None     # async (approval: PendingApproval) -> None
        self.on_apply_override: Optional[Callable] = None    # async (key, value) -> None

        # Background task
        self._running = False
        self._task = None

    def start(self):
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("🔬 Diagnostic engine started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    # ── Observation feeds (called by engine.py) ───────────────

    def record_error(self, module: str, error_type: str, message: str, symbol: str = ""):
        """Called by engine whenever an error occurs."""
        key = f"{module}:{error_type}"
        now = time.time()
        if key in self._error_counts:
            rec = self._error_counts[key]
            rec.count += 1
            rec.last_seen = now
            if symbol and symbol not in rec.symbols_affected:
                rec.symbols_affected.append(symbol)
        else:
            self._error_counts[key] = ErrorRecord(
                error_type=error_type,
                module=module,
                message=message,
                symbols_affected=[symbol] if symbol else [],
            )
        # Track rate for circuit breaker check
        self._error_rate_window.append(now)
        # Deque maxlen=10,000 caps the upper bound automatically;
        # just purge entries older than 60s for the rate calculation.
        while self._error_rate_window and now - self._error_rate_window[0] > 60:
            self._error_rate_window.popleft()

    def record_signal_death(self, symbol: str, direction: str, strategy: str,
                             kill_reason: str, rr: float, confidence: float,
                             regime: str, setup_class: str = "intraday"):
        """Called by engine for every signal that dies in the pipeline."""
        self._death_log.append({
            "symbol": symbol,
            "direction": direction,
            "strategy": strategy,
            "kill_reason": kill_reason,
            "rr": rr,
            "confidence": confidence,
            "regime": regime,
            "setup_class": setup_class,
            "ts": time.time(),
        })
        # deque(maxlen=1000) auto-evicts oldest entries

        # Also feed to AI analyst buffer
        try:
            from analyzers.ai_analyst import ai_analyst
            ai_analyst.buffer_dead_signal(
                symbol, direction, strategy, kill_reason, rr, confidence, regime
            )
        except Exception:
            pass

    def record_scan(self):
        self._scan_count += 1
        self._total_scan_count += 1

    def record_signal_generated(self):
        self._signals_generated += 1
        self._total_signals_generated += 1

    def record_signal_published(self):
        self._signals_published += 1
        self._total_signals_published += 1

    def record_outcome(self, symbol: str, strategy: str, direction: str,
                        outcome: str, pnl_r: float, confidence: float, regime: str):
        """Called by learning_loop when a trade closes."""
        self._outcome_log.append({
            "symbol": symbol,
            "strategy": strategy,
            "direction": direction,
            "outcome": outcome,  # WIN | LOSS | BREAKEVEN | EXPIRED
            "pnl_r": pnl_r,
            "confidence": confidence,
            "regime": regime,
            "ts": time.time(),
        })
        if len(self._outcome_log) > 500:
            self._outcome_log = self._outcome_log[-500:]

    # ── NEW: Telegram observation ─────────────────────────────

    def record_telegram_message(self, message_type: str, char_count: int,
                                  parse_mode: str = "HTML", has_keyboard: bool = False,
                                  send_ok: bool = True, error: str = ""):
        """
        Called by TelegramObserver every time a message is sent.
        Tracks formatting issues, length violations, send errors.
        """
        now = time.time()
        record = {
            "type": message_type,
            "chars": char_count,
            "parse_mode": parse_mode,
            "has_keyboard": has_keyboard,
            "send_ok": send_ok,
            "error": error,
            "ts": now,
        }
        self._telegram_log.append(record)
        if len(self._telegram_log) > 200:
            self._telegram_log = self._telegram_log[-200:]

        # Flag issues immediately
        if char_count > 4000:
            self.record_error("telegram", "MESSAGE_TOO_LONG",
                              f"Message type={message_type} is {char_count} chars (limit 4096)")
        if not send_ok and error:
            self.record_error("telegram", "SEND_ERROR",
                              f"type={message_type}: {error[:120]}")

    def record_telegram_issue(self, issue_type: str, detail: str):
        """Record a specific Telegram formatting or delivery issue."""
        self.record_error("telegram", issue_type, detail)

    def record_signal_card(self, symbol: str, grade: str, char_count: int,
                            had_narrative: bool, confluence_count: int,
                            had_tp3: bool, had_btc_context: bool,
                            parse_ok: bool = True):
        """
        Called after each signal card is published.
        Tracks card completeness and formatting health.
        """
        self._signal_card_log.append({
            "symbol": symbol,
            "grade": grade,
            "chars": char_count,
            "had_narrative": had_narrative,
            "confluence_count": confluence_count,
            "had_tp3": had_tp3,
            "had_btc_context": had_btc_context,
            "parse_ok": parse_ok,
            "ts": time.time(),
        })
        if len(self._signal_card_log) > 100:
            self._signal_card_log = self._signal_card_log[-100:]

    # ── NEW: API performance observation ─────────────────────

    def record_api_call(self, endpoint: str, latency_ms: float, success: bool,
                         exchange: str = "binance"):
        """
        Called by api_client for every exchange API call.
        Tracks latency trends and failure rates.
        """
        key = f"{exchange}:{endpoint}"
        if key not in self._api_latency:
            self._api_latency[key] = []
        self._api_latency[key].append({
            "ms": latency_ms,
            "ok": success,
            "ts": time.time(),
        })
        # Keep only last 100 calls per endpoint
        if len(self._api_latency[key]) > 100:
            self._api_latency[key] = self._api_latency[key][-100:]

    # ── NEW: AI output quality observation ───────────────────

    def record_narrative_quality(self, symbol: str, grade: str, quality: dict):
        """
        Called after AI self-reviews its own narrative.
        Tracks narrative quality trends so we can improve prompts.
        """
        self._narrative_quality_log.append({
            "symbol": symbol,
            "grade": grade,
            "clarity": quality.get("clarity", 5),
            "actionability": quality.get("actionability", 5),
            "too_technical": quality.get("too_technical", False),
            "too_long": quality.get("too_long", False),
            "too_vague": quality.get("too_vague", False),
            "improvement": quality.get("improvement"),
            "ts": time.time(),
        })
        if len(self._narrative_quality_log) > 50:
            self._narrative_quality_log = self._narrative_quality_log[-50:]

    # ── NEW: System performance observation ──────────────────

    def record_scan_timing(self, symbol: str, duration_ms: float):
        """Track how long individual symbol scans take."""
        self._scan_timings.append({"symbol": symbol, "ms": duration_ms, "ts": time.time()})
        if len(self._scan_timings) > 500:
            self._scan_timings = self._scan_timings[-500:]

    def record_db_operation(self, operation: str, duration_ms: float, success: bool):
        """Track DB write/read performance."""
        self._db_timings.append({
            "op": operation, "ms": duration_ms,
            "ok": success, "ts": time.time()
        })
        if len(self._db_timings) > 200:
            self._db_timings = self._db_timings[-200:]

    # ── Approval system ───────────────────────────────────────

    @staticmethod
    def _proposal_target_key(change_type: str, new_value) -> str:
        """
        Build a stable key identifying the TARGET of a proposal (not its value),
        so two proposals that mutate the same knob (e.g., swing RR floor 1.75 vs
        1.77) collapse to one pending approval instead of stacking.
        """
        try:
            if isinstance(new_value, dict):
                if change_type == "rr_floor":
                    return f"rr_floor:{new_value.get('setup_class', '_')}"
                if change_type == "suppress_strategy":
                    return f"suppress_strategy:{new_value.get('strategy', '_')}"
                if change_type == "exclude_symbol":
                    return f"exclude_symbol:{new_value.get('symbol', '_')}"
        except Exception:
            pass
        return change_type

    async def propose_change(self, change_type: str, description: str,
                              old_value, new_value, reason: str,
                              risk_level: str = "LOW",
                              estimated_impact: str = "Unknown"):
        """
        Propose a runtime config change. Sends Telegram approval request.
        Does NOT apply anything until user taps YES.
        """
        # ── Dedup against existing non-expired pending approvals ─────────────
        # Without this, if the user never taps YES/NO, the diagnostic loop keeps
        # firing a new approval every 30 min (value-specific time dedup at the
        # call site only prevents *call* frequency, not queue buildup).
        # Result seen in the field: 6 identical "Lower swing RR floor: 2.0 → 1.x"
        # proposals piled up with 5 still pending, all showing From: 2.0.
        # Instead of creating a new approval, update the existing one in place
        # with the fresher proposed value/reason and keep the same approval_id
        # so the Telegram card can be edited rather than duplicated.
        now = time.time()
        try:
            _target_key = self._proposal_target_key(change_type, new_value)
        except Exception:
            _target_key = None
        if _target_key:
            for _existing in list(self._pending_approvals.values()):
                if _existing.change_type != change_type:
                    continue
                if _existing.expires_at and _existing.expires_at <= now:
                    continue
                try:
                    _existing_key = self._proposal_target_key(
                        _existing.change_type, _existing.new_value
                    )
                except Exception:
                    _existing_key = None
                if _existing_key != _target_key:
                    continue
                # Found a matching pending proposal — refresh it in place.
                _existing.description = description
                _existing.new_value = new_value
                _existing.reason = reason
                _existing.risk_level = risk_level
                _existing.estimated_impact = estimated_impact
                logger.info(
                    f"🔬 Approval refreshed [{_existing.approval_id}] "
                    f"(duplicate suppressed): {description}"
                )
                if self.on_send_approval:
                    try:
                        await self.on_send_approval(_existing)
                    except Exception as _e:
                        logger.debug(f"on_send_approval on refresh failed: {_e}")
                return

        # ── Global rate cap — prevent more than N proposals per hour ──────────
        cutoff = now - 3600
        recent_proposals = [t for t in self._proposal_timestamps if t > cutoff]
        if len(recent_proposals) >= self._max_proposals_per_hour:
            logger.info(
                f"🔬 Proposal rate-capped ({len(recent_proposals)}/{self._max_proposals_per_hour}/hr): "
                f"{description[:60]}"
            )
            return
        self._proposal_timestamps.append(now)

        import uuid
        approval_id = str(uuid.uuid4())[:8]
        approval = PendingApproval(
            approval_id=approval_id,
            change_type=change_type,
            description=description,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            risk_level=risk_level,
            estimated_impact=estimated_impact,
            expires_at=time.time() + 86400,  # Expires in 24h
        )
        # ── Generate personalised "should you approve?" verdict ──────────────
        # Pull live bot metrics and ask AI to give a specific recommendation
        # using YOUR actual numbers, not generic rules.
        try:
            from analyzers.ai_analyst import ai_analyst
            from data.database import db as _db
            from analyzers.regime import regime_analyzer
            _stats = await _db.get_performance_stats(days=7)
            _wins = (_stats or {}).get('wins', 0) or 0
            _losses = (_stats or {}).get('losses', 0) or 0
            _total_closed = _wins + _losses
            _wr = round(_wins / _total_closed * 100, 1) if _total_closed else None
            _total_signals = (_stats or {}).get('total_signals', 0) or 0
            _regime = getattr(getattr(regime_analyzer, 'regime', None), 'value', 'UNKNOWN')

            _verdict_prompt = f"""You are advising a non-expert trader whether to approve a bot configuration change.
Give a single paragraph of plain English (max 60 words). Be direct: start with "Approve" or "Wait".
Use their actual numbers — do not give generic advice.

PROPOSED CHANGE: {description}
Current value: {old_value} → Proposed: {new_value}
Reason from AI: {reason}

THEIR BOT RIGHT NOW:
- Win rate (7d): {f"{_wr}% over {_total_closed} closed trades" if _wr else f"No closed trades yet ({_total_signals} signals pending)"}
- Current regime: {_regime}
- Risk level of this change: {risk_level}

In 1-2 sentences: should they approve or wait? Start with "Approve" or "Wait —" and explain why using their numbers."""

            if ai_analyst._client:
                _verdict_raw = await ai_analyst._call(_verdict_prompt, call_label="approval_verdict")
                if _verdict_raw and len(_verdict_raw) > 20:
                    approval.user_verdict = _verdict_raw.strip()[:300]
                    approval.verdict_approve = _verdict_raw.strip().lower().startswith('approve')
        except Exception as _ve:
            logger.debug(f"Approval verdict generation failed (non-fatal): {_ve}")

        self._pending_approvals[approval_id] = approval
        logger.info(f"🔬 Approval proposed [{approval_id}]: {description}")

        if self.on_send_approval:
            await self.on_send_approval(approval)

    async def apply_approval(self, approval_id: str) -> bool:
        """
        Called when user taps YES. Applies the runtime change immediately.
        """
        approval = self._pending_approvals.get(approval_id)
        if not approval:
            return False

        try:
            # Apply via runtime config system
            if self.on_apply_override:
                await self.on_apply_override(approval.change_type, approval.new_value)

            # Also apply known in-memory changes directly
            await self._apply_known_change(approval)

            # Store for UNDO
            self._applied_overrides[approval_id] = {
                "change_type": approval.change_type,
                "new_value": approval.new_value,
                "previous": approval.old_value,
                "applied_at": time.time(),
                "description": approval.description,
            }

            del self._pending_approvals[approval_id]
            logger.info(f"✅ Approval {approval_id} applied: {approval.description}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply approval {approval_id}: {e}")
            return False

    async def undo_change(self, approval_id: str) -> bool:
        """Revert an applied change."""
        override = self._applied_overrides.get(approval_id)
        if not override:
            return False
        try:
            if self.on_apply_override:
                await self.on_apply_override(override["change_type"], override["previous"])
            await self._apply_known_change_direct(override["change_type"], override["previous"])
            del self._applied_overrides[approval_id]
            logger.info(f"↩️  Reverted change: {override['description']}")
            return True
        except Exception as e:
            logger.error(f"Failed to undo {approval_id}: {e}")
            return False

    async def _apply_known_change(self, approval: PendingApproval):
        await self._apply_known_change_direct(approval.change_type, approval.new_value)

    async def _apply_known_change_direct(self, change_type: str, value):
        """Apply known safe changes in-memory immediately."""
        try:
            if change_type == "rr_floor":
                # value = {'setup_class': 'swing', 'new_floor': 1.35}
                setup_class = value.get("setup_class", "swing")
                new_floor = float(value.get("new_floor", 1.35))
                from strategies.base import set_rr_floor_override
                set_rr_floor_override(setup_class, new_floor)
                logger.info(f"  → RR floor override applied: {setup_class} = {new_floor}")

            elif change_type == "min_confidence":
                from signals.aggregator import signal_aggregator
                signal_aggregator._min_confidence = float(value)
                logger.info(f"  → min_confidence updated in-memory: {value}")

            elif change_type == "suppress_strategy":
                from governance.performance_tracker import performance_tracker
                strat, duration_mins = value["strategy"], value["duration_mins"]
                performance_tracker.suppress_strategy(strat, duration_mins=duration_mins)
                logger.info(f"  → Strategy suppressed: {strat} for {duration_mins}min")

            elif change_type == "exclude_symbol":
                from scanner.scanner import scanner
                scanner.exclude_symbol(value["symbol"])
                logger.info(f"  → Symbol excluded: {value['symbol']}")

            elif change_type == "execution_config":
                # Execution funnel proposals are advisory — log that user acknowledged.
                # The actual config change description is stored in the approval for audit.
                logger.info(f"  → Execution config acknowledged: {str(value)[:120]}")

        except Exception as e:
            logger.warning(f"In-memory apply failed ({change_type}): {e}")

    def reject_approval(self, approval_id: str):
        """Called when user taps NO."""
        if approval_id in self._pending_approvals:
            del self._pending_approvals[approval_id]

    def snooze_approval(self, approval_id: str, minutes: int = 60):
        """Called when user taps 'Ask me later'."""
        if approval_id in self._pending_approvals:
            self._pending_approvals[approval_id].expires_at = time.time() + (minutes * 60)

    def get_pending_approvals(self) -> List[PendingApproval]:
        now = time.time()
        return [a for a in self._pending_approvals.values()
                if a.expires_at == 0 or a.expires_at > now]

    def get_applied_overrides(self) -> List[dict]:
        return list(self._applied_overrides.values())

    # ── Background watch loop ─────────────────────────────────

    async def _watch_loop(self):
        """Continuously monitor for issues."""
        await asyncio.sleep(30)  # Let bot warm up first
        while self._running:
            try:
                await self._check_error_rate()
                await self._check_signal_drought()
                await self._check_edge_cases()
                await self._check_performance_gaps()

                # Run AI dead signal analysis (V2.09: every 2h, was 30min — saves ~36 calls/day)
                try:
                    from analyzers.ai_analyst import ai_analyst
                    from analyzers.regime import regime_analyzer
                    regime = regime_analyzer.regime.value if hasattr(regime_analyzer, 'regime') else "UNKNOWN"
                    analysis = await ai_analyst.analyse_dead_signals(regime)
                    if analysis:
                        await self._act_on_dead_signal_analysis(analysis)
                except Exception as e:
                    logger.debug(f"Dead signal AI analysis error: {e}")

                # V2.10: Run veto debate cycle every 2h
                try:
                    from analyzers.veto_system import veto_system
                    from analyzers.ai_analyst import ai_analyst
                    if veto_system._enabled:
                        ss = ai_analyst._session_stats
                        # Build session data for veto proposer
                        sd = {
                            "win_rate": self.get_stats_summary().get("win_rate_1h", 0),
                            "published": self._signals_published,
                            "top_kill": (Counter(d.get("kill_reason", "?") for d in self._death_log[-100:])
                                         .most_common(1)[0][0] if self._death_log else "unknown"),
                            "long_count": ss.get("regime_at_signal", {}).get("BULL_TREND", {}).get("LONG", 0),
                            "short_count": ss.get("regime_at_signal", {}).get("BULL_TREND", {}).get("SHORT", 0),
                            "htf_long_blocks": ss.get("htf_blocked_long", 0),
                            "htf_short_blocks": ss.get("htf_blocked_short", 0),
                            "regime": "UNKNOWN",
                            "high_fpr_symbols": [],
                            "loss_streak_strategies": [],
                            "strategy_win_rates": {},
                        }
                        await veto_system.run_debate_cycle(sd)
                        await veto_system.check_performance_gate()
                except Exception as _ve:
                    logger.debug(f"Veto cycle error: {_ve}")

                # V2.09: Once-per-day structural audit (deep model — checks direction bias,
                # HTF asymmetry, whale conflicts). Runs automatically, no rate limit concern.
                try:
                    from analyzers.ai_analyst import ai_analyst
                    audit = await ai_analyst.structural_audit()
                    if audit and audit.severity in ("HIGH", "CRITICAL"):
                        # Surface critical structural bugs to the diagnostic report immediately
                        _audit_lines = [f"🔥 STRUCTURAL AUDIT [{audit.severity}]"]
                        if audit.biased_strategies:
                            _audit_lines.append(f"  Biased strategies: {', '.join(audit.biased_strategies)}")
                        if audit.htf_asymmetry:
                            _audit_lines.append("  ⚠️ HTF guardrail blocking one direction 3x+ more")
                        if audit.whale_signal_conflict:
                            _audit_lines.append("  ⚠️ Signals frequently trade against whale flow")
                        for rec in audit.recommendations[:3]:
                            _audit_lines.append(f"  → {rec}")
                        logger.warning(chr(10).join(_audit_lines))
                except Exception as _audit_err:
                    logger.debug(f"Structural audit error: {_audit_err}")

                # GROUP 4A: Execution funnel audit — every 2h.
                # Feeds AI actual fill rates, entry zone distances, and post-expiry direction
                # accuracy. This is the audit the old pipeline_diag was completely missing.
                _now = time.time()
                if _now - self._last_execution_funnel_audit > self._execution_audit_interval:
                    try:
                        from analyzers.ai_analyst import ai_analyst
                        from data.database import db
                        from analyzers.regime import regime_analyzer
                        _recent_sigs = await db.get_recent_signals(hours=4)
                        # Only audit signals ≥45min old — fill_rate=0 at T+3min is normal
                        import time as _tck
                        _cutoff_ts = _tck.time() - 2700
                        _aged = []
                        for _s in _recent_sigs:
                            try:
                                import datetime
                                _ca = _s.get('created_at')
                                _dt_ts = None
                                if isinstance(_ca, (int, float)):
                                    _dt_ts = float(_ca)
                                elif _ca is not None:
                                    _st = str(_ca)[:19]
                                    try:
                                        _dt_ts = datetime.datetime.strptime(
                                            _st, '%Y-%m-%d %H:%M:%S'
                                        ).timestamp()
                                    except ValueError:
                                        # Try ISO 8601 with 'T' separator
                                        try:
                                            _dt_ts = datetime.datetime.fromisoformat(
                                                str(_ca).replace('Z', '+00:00')
                                            ).timestamp()
                                        except ValueError:
                                            _dt_ts = None
                                if _dt_ts is not None and _dt_ts < _cutoff_ts:
                                    _aged.append(_s)
                            except Exception:
                                pass
                        if len(_aged) >= 2:
                            _recent_sigs = _aged
                        elif len(_recent_sigs) < 5:
                            # Too early to audit — no aged signals yet
                            self._last_execution_funnel_audit = _tck.time()
                            raise Exception('Insufficient aged signal data for funnel audit')
                        if len(_recent_sigs) >= 2:
                            _regime = regime_analyzer.regime.value if hasattr(regime_analyzer.regime, 'value') else str(regime_analyzer.regime)
                            _funnel = await ai_analyst.execution_funnel_audit(_recent_sigs, _regime)
                            if _funnel and _funnel.get("severity") in ("CRITICAL", "HIGH"):
                                logger.warning(
                                    f"⚡ EXECUTION FUNNEL [{_funnel.get('severity')}] "
                                    f"fill={_funnel.get('fill_rate_pct')}% "
                                    f"zone={_funnel.get('zone_placement_verdict')} "
                                    f"cause={_funnel.get('root_cause','?')[:80]}"
                                )
                                # Surface top recommendation as an approval proposal
                                recs = _funnel.get("recommendations", [])
                                if recs:
                                    top = recs[0]
                                    _dedup = f"exec_funnel_{_funnel.get('zone_placement_verdict','?')}"
                                    if _now - self._last_proposals.get(_dedup, 0) > 7200:
                                        self._last_proposals[_dedup] = _now
                                        await self.propose_change(
                                            change_type="execution_config",
                                            description=f"Execution funnel: {top.get('change','?')[:80]}",
                                            old_value="current_config",
                                            new_value=top.get("change", ""),
                                            reason=f"AI execution audit: fill_rate={_funnel.get('fill_rate_pct')}% | {_funnel.get('root_cause','?')[:120]}",
                                            risk_level="LOW",
                                            estimated_impact=top.get("expected", "Improve fill rate"),
                                        )
                            self._last_execution_funnel_audit = _now
                    except Exception as _efa_err:
                        logger.debug(f"Execution funnel audit error: {_efa_err}")

                # GROUP 4C: Strategy × regime performance audit — every 4h.
                # Catches "IchimokuCloud LONG in BULL_TREND: 0/8 filled (0%)" type findings.
                if _now - self._last_strategy_regime_audit > self._strategy_audit_interval:
                    try:
                        from analyzers.ai_analyst import ai_analyst
                        from data.database import db
                        from analyzers.regime import regime_analyzer
                        _all_sigs = await db.get_recent_signals(hours=24)
                        if len(_all_sigs) >= 5:
                            # Build strategy × regime × direction performance matrix
                            _perf_matrix: dict = {}
                            for _sig in _all_sigs:
                                _key = f"{(_sig.get('strategy','?') or '?')[:20]}|{_sig.get('regime','?')}|{_sig.get('direction','?')}"
                                if _key not in _perf_matrix:
                                    _perf_matrix[_key] = {"signals": 0, "filled": 0, "wins": 0, "expired": 0}
                                _perf_matrix[_key]["signals"] += 1
                                if _sig.get("zone_reached"):
                                    _perf_matrix[_key]["filled"] += 1
                                if _sig.get("outcome") == "WIN":
                                    _perf_matrix[_key]["wins"] += 1
                                if _sig.get("outcome") == "EXPIRED":
                                    _perf_matrix[_key]["expired"] += 1
                            _regime = regime_analyzer.regime.value if hasattr(regime_analyzer.regime, 'value') else str(regime_analyzer.regime)
                            _sra = await ai_analyst.strategy_regime_audit(_perf_matrix, _regime)
                            if _sra and _sra.get("suppress_immediately"):
                                logger.warning(
                                    f"🎯 STRATEGY REGIME AUDIT: suppress={_sra.get('suppress_immediately')} "
                                    f"finding={_sra.get('key_finding','?')[:80]}"
                                )
                        self._last_strategy_regime_audit = _now
                    except Exception as _sra_err:
                        logger.debug(f"Strategy regime audit error: {_sra_err}")

                # GROUP 5: Adaptive parameter tuner — runs once per 24h.
                # Adjusts penalty magnitudes and market-state multipliers based on
                # per-market-state win rates (pure statistical feedback, no LLM).
                if self._param_tuner_enabled:
                    try:
                        from governance.param_tuner import param_tuner as _pt
                        _pt_changes = await _pt.run_tuning_cycle()
                        if _pt_changes and self.on_send_report:
                            _report = (
                                "⚙️ <b>Adaptive param tuner</b> — "
                                f"{len(_pt_changes)} adjustment(s):\n"
                                + "\n".join(f"  • {c}" for c in _pt_changes)
                            )
                            await self.on_send_report(_report)
                    except Exception as _pt_err:
                        logger.debug(f"Param tuner cycle error: {_pt_err}")

                # 6-hour diagnostic report
                if time.time() - self._last_report_time > self._report_interval:
                    await self._send_diagnostic_report()
                    self._last_report_time = time.time()

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Diagnostic watch error: {e}")
                await asyncio.sleep(60)

    async def _check_error_rate(self):
        """If errors > 100/min, trigger circuit breaker."""
        rate = len(self._error_rate_window)
        if rate > 100:
            try:
                from risk.manager import risk_manager
                if not risk_manager.circuit_breaker.is_active:
                    risk_manager.circuit_breaker.trigger(
                        reason=f"Diagnostic: critical error rate {rate}/min"
                    )
                    msg = (f"🚨 <b>Circuit breaker auto-triggered</b>\n"
                           f"Error rate: {rate}/min\nBot paused 15 minutes.\n"
                           f"Tap [RESUME] to override immediately.")
                    if self.on_send_report:
                        await self.on_send_report(msg)
            except Exception as e:
                logger.debug(f"Circuit breaker trigger error: {e}")

    async def _check_signal_drought(self):
        """If zero signals published in 4+ hours, propose changes."""
        now = time.time()
        recent_deaths = [d for d in self._death_log if now - d["ts"] < 14400]  # 4h
        if len(recent_deaths) < 10:
            return  # Not enough data yet

        # ── Evidence gates — prevent premature approvals ──────────────────────
        # Gate 1: Need minimum 500 scans before the pipeline has seen enough symbols
        # to distinguish "filter is wrong" from "nothing good to trade right now"
        if self._scan_count < 500:
            return

        # Gate 2: Dead zone filter — 03:00–07:00 UTC is low-liquidity.
        # Every AGG_THRESHOLD / RR_FLOOR kill during this window is expected behaviour,
        # not a calibration problem. Don't generate approvals from dead-zone data.
        from datetime import datetime, timezone as _tz
        _hour = datetime.now(_tz.utc).hour
        if 3 <= _hour < 7:
            return

        # Gate 3: Need at least 30 recent deaths to have statistical significance.
        # 10 deaths from a 2-minute window are noise; 30 over 4h is a pattern.
        if len(recent_deaths) < 30:
            return

        # Count kill reasons in last 4 hours
        kill_counts = Counter(d["kill_reason"] for d in recent_deaths)
        if not kill_counts:
            return
        top_killer, top_count = kill_counts.most_common(1)[0]
        pct = top_count / len(recent_deaths) * 100

        # Propose fix when one kill reason dominates AND signal throughput is very low.
        # Old gate: required exactly zero published, meaning a single published signal
        # would block all calibration proposals even if 99% of signals were dying.
        # New gate: published rate < 5% of generated signals, or zero published.
        publish_rate = self._signals_published / max(self._signals_generated, 1)
        if publish_rate < 0.05 and pct > 70:
            # One kill reason is dominating — propose fix
            if top_killer == "RR_FLOOR":
                avg_rr = sum(d["rr"] for d in recent_deaths) / len(recent_deaths)
                proposed_floor = round(avg_rr * 1.05, 2)  # 5% above actual avg RR

                # Only propose if meaningfully different from current
                try:
                    from strategies.base import cfg_min_rr, get_rr_floor_overrides
                    current_floor = get_rr_floor_overrides().get("swing") or cfg_min_rr("swing")
                    if proposed_floor < current_floor - 0.03:
                        # Deduplicate — don't re-propose RR floor within 30 minutes
                        # (type-level, not value-specific, so 1.24→1.25 shifts don't spam)
                        _dedup_key = "rr_floor_swing"
                        _now = time.time()
                        if _now - self._last_proposals.get(_dedup_key, 0) > 1800:
                            self._last_proposals[_dedup_key] = _now
                            await self.propose_change(
                                change_type="rr_floor",
                                description=f"Lower swing RR floor: {current_floor} → {proposed_floor}",
                                old_value=current_floor,
                                new_value={"setup_class": "swing", "new_floor": proposed_floor},
                                reason=(f"RR_FLOOR killing {pct:.0f}% of signals. "
                                        f"Avg actual R:R is {avg_rr:.2f} — floor {current_floor} "
                                        f"may be too strict for current regime."),
                                risk_level="LOW",
                                estimated_impact=f"~{top_count//10} more signals/hour estimated",
                            )
                except Exception:
                    pass

            elif top_killer == "AGG_THRESHOLD":
                avg_conf = sum(d["confidence"] for d in recent_deaths) / len(recent_deaths)
                # If avg dying confidence is within 5 points of threshold, suggest lowering
                try:
                    from signals.aggregator import signal_aggregator
                    import numpy as _np_diag
                    raw_min = signal_aggregator._min_confidence

                    # FIX: Read the EFFECTIVE min_confidence (includes adaptive floor)
                    # so the approval reflects what's actually killing signals,
                    # not just the raw config floor. Raw floor=60 but adaptive can be 63-68.
                    recent_buf = list(getattr(signal_aggregator, '_recent_score_buffer', []))
                    if len(recent_buf) >= 20:
                        adaptive = float(_np_diag.percentile(recent_buf, 40))
                        current_min = max(raw_min, min(68, adaptive))
                    else:
                        current_min = raw_min

                    if avg_conf > current_min - 6:
                        proposed = max(55, round(current_min - 3))
                        # Guard: never propose a no-op or upward change
                        if proposed >= current_min:
                            pass  # Already at minimum or can't go lower
                        elif proposed < 55:
                            pass  # Safety floor — never go below 55
                        else:
                            # Deduplicate — don't re-propose confidence floor within 30 minutes
                            # (type-level key so small value shifts don't bypass)
                            _dedup_key = "min_confidence_lower"
                            _now = time.time()
                            if _now - self._last_proposals.get(_dedup_key, 0) > 1800:
                                self._last_proposals[_dedup_key] = _now
                                await self.propose_change(
                                    change_type="min_confidence",
                                    description=f"Lower confidence floor: {current_min:.0f} → {proposed}",
                                    old_value=current_min,
                                    new_value=proposed,
                                    reason=(f"AGG_THRESHOLD killing {pct:.0f}% of signals. "
                                            f"Avg dying confidence: {avg_conf:.1f} (effective floor: {current_min:.0f})."),
                                    risk_level="LOW",
                                    estimated_impact=f"~{top_count//8} more signals/hour estimated",
                                )
                except Exception:
                    pass

    async def _check_edge_cases(self):
        """Detect anomalous patterns in recent activity."""
        now = time.time()
        recent = [d for d in self._death_log if now - d["ts"] < 3600]  # Last 1h

        if not recent:
            return

        # Check: same symbol appearing > 10 times with same setup
        symbol_counts = Counter(f"{d['symbol']}:{d['strategy']}" for d in recent)
        for key, count in symbol_counts.items():
            if count > 10:
                symbol, strategy = key.split(":", 1)
                # Auto-exclude for 1 hour to stop the spam
                try:
                    from scanner.scanner import scanner
                    from config.constants import Diagnostics
                    scanner.temporarily_exclude_symbol(
                        symbol,
                        duration_secs=Diagnostics.EDGE_CASE_CHECK_WINDOW_SECS,
                        reason=f"edge-case repeat with {strategy}",
                    )
                    logger.info(f"🔬 Edge case: {symbol} appearing {count}x/hour "
                                f"with {strategy} — temporarily excluded")
                except Exception:
                    pass

        # Check: strategy crashing repeatedly
        error_strats = Counter()
        for key, rec in self._error_counts.items():
            if time.time() - rec.last_seen < 3600 and rec.count > 20:
                strat_hint = rec.message
                error_strats[rec.module] += rec.count

        for module, count in error_strats.most_common(3):
            if count > 50:
                logger.warning(f"🔬 High error rate: {module} — {count} errors in last hour")

    async def _check_performance_gaps(self):
        """Analyse recent outcomes for underperforming strategies."""
        if len(self._outcome_log) < 10:
            return

        now = time.time()
        recent = [o for o in self._outcome_log if now - o["ts"] < 86400]  # Last 24h

        if not recent:
            return

        # Group by strategy
        strat_outcomes = defaultdict(list)
        for o in recent:
            strat_outcomes[o["strategy"]].append(o["outcome"] == "WIN")

        for strat, results in strat_outcomes.items():
            if len(results) < 5:
                continue  # Not enough data
            wr = sum(results) / len(results)
            if wr < 0.30:  # Below 30% win rate
                try:
                    from governance.performance_tracker import performance_tracker
                    if not performance_tracker.is_strategy_disabled(strat):
                        # Deduplicate — don't re-propose same strategy within 4 hours
                        _dedup_key = f"perf_suppress:{strat}"
                        if now - self._last_proposals.get(_dedup_key, 0) > 14400:
                            self._last_proposals[_dedup_key] = now
                            await self.propose_change(
                                change_type="suppress_strategy",
                                description=f"Suppress {strat} for 4h (win rate: {wr:.0%})",
                                old_value={"strategy": strat, "suppressed": False},
                                new_value={"strategy": strat, "duration_mins": 240},
                                reason=(f"{strat} win rate: {wr:.0%} over {len(results)} trades "
                                        f"in last 24h. Below 30% threshold."),
                                risk_level="LOW",
                                estimated_impact="Reduces losing trades, fewer signals",
                            )
                except Exception:
                    pass

        # ── Confidence floor raise-back: detect if a previously-lowered
        # confidence floor is now producing too many low-quality signals ─────
        await self._check_confidence_raise_back(recent)

    async def _check_confidence_raise_back(self, recent_outcomes):
        """If confidence floor was lowered and win rate dropped, propose raising it back."""
        if not recent_outcomes or len(recent_outcomes) < 10:
            return

        # Only check if we have an active min_confidence override
        active_conf_override = None
        for aid, ov in self._applied_overrides.items():
            if ov.get("change_type") == "min_confidence":
                active_conf_override = ov
                break
        if not active_conf_override:
            return

        # Need at least 4h of data since the override was applied
        override_age = time.time() - active_conf_override.get("applied_at", 0)
        if override_age < 14400:  # 4 hours
            return

        # Calculate win rate of signals since the override was applied
        applied_at = active_conf_override.get("applied_at", 0)
        post_change = [o for o in recent_outcomes if o["ts"] > applied_at]
        if len(post_change) < 8:
            return

        post_wr = sum(1 for o in post_change if o["outcome"] == "WIN") / len(post_change)
        if post_wr < 0.35:
            previous_val = active_conf_override.get("previous")
            current_val = active_conf_override.get("new_value")
            if previous_val is not None and current_val is not None:
                _dedup_key = "raise_confidence_back"
                _now = time.time()
                if _now - self._last_proposals.get(_dedup_key, 0) > 7200:
                    self._last_proposals[_dedup_key] = _now
                    await self.propose_change(
                        change_type="min_confidence",
                        description=f"Raise confidence floor back: {current_val} → {previous_val}",
                        old_value=current_val,
                        new_value=previous_val,
                        reason=(f"Win rate dropped to {post_wr:.0%} over {len(post_change)} trades "
                                f"since confidence was lowered. Reverting to previous floor."),
                        risk_level="LOW",
                        estimated_impact="Fewer but higher-quality signals",
                    )

    async def _act_on_dead_signal_analysis(self, analysis):
        """Act on AI dead signal analysis results."""
        if analysis.strategy_to_suppress:
            try:
                from governance.performance_tracker import performance_tracker
                if not performance_tracker.is_strategy_disabled(analysis.strategy_to_suppress):
                    # Dedup: don't re-propose the same strategy suppression within 6h
                    _sup_dedup_key = f"suppress_strategy:{analysis.strategy_to_suppress}"
                    _sup_cooldown = 6 * 3600  # 6 hours
                    if time.time() - self._last_proposals.get(_sup_dedup_key, 0) > _sup_cooldown:
                        self._last_proposals[_sup_dedup_key] = time.time()

                        # In VOLATILE regime, IchimokuCloud 100% SHORT is expected behaviour
                        # The AI should not flag this as a problem
                        try:
                            from analyzers.regime import regime_analyzer as _ra
                            _cur_regime = getattr(getattr(_ra, 'regime', None), 'value', '')
                        except Exception:
                            _cur_regime = ''
                        _ichimoku_expected_in_volatile = (
                            'ichimoku' in analysis.strategy_to_suppress.lower() and
                            _cur_regime in ('VOLATILE', 'VOLATILE_PANIC', 'BEAR_TREND')
                        )
                        if _ichimoku_expected_in_volatile:
                            logger.info(
                                f"Suppression proposal skipped: {analysis.strategy_to_suppress} "
                                f"100% SHORT is expected in {_cur_regime} regime"
                            )
                        else:
                            await self.propose_change(
                                change_type="suppress_strategy",
                                description=f"AI suggests suppressing {analysis.strategy_to_suppress}",
                                old_value={"strategy": analysis.strategy_to_suppress, "suppressed": False},
                                new_value={"strategy": analysis.strategy_to_suppress, "duration_mins": 120},
                                reason=f"AI analysis: {analysis.recommendation}",
                                risk_level="LOW",
                                estimated_impact="Reduces signal noise from underperforming strategy",
                            )
            except Exception:
                pass

    # ── Diagnostic report ─────────────────────────────────────

    async def _send_diagnostic_report(self):
        """Build and send the 6-hour diagnostic report."""
        if not self.on_send_report:
            return

        now = time.time()
        period_hours = self._report_interval / 3600
        recent_deaths = [d for d in self._death_log if now - d["ts"] < self._report_interval]
        recent_outcomes = [o for o in self._outcome_log if now - o["ts"] < self._report_interval]

        # Build report sections
        criticals = []
        errors = []
        perf_gaps = []
        edge_cases = []
        healthy = []
        auto_actions = []

        # Check signal drought
        if self._signals_published == 0 and self._signals_generated > 10:
            kill_counts = Counter(d["kill_reason"] for d in recent_deaths)
            top_k, top_v = kill_counts.most_common(1)[0] if kill_counts else ("?", 0)
            pct = top_v / len(recent_deaths) * 100 if recent_deaths else 0
            criticals.append(
                f"Zero signals published in {period_hours:.0f}h\n"
                f"   Dominant kill: {top_k} ({pct:.0f}% of {len(recent_deaths)} deaths)"
            )

        # High-frequency errors
        for key, rec in self._error_counts.items():
            if now - rec.last_seen < self._report_interval and rec.count > 10:
                errors.append(
                    f"{rec.module} — {rec.error_type} ({rec.count}x)\n"
                    f"   Msg: {rec.message[:80]}"
                    + (f"\n   Affects: {', '.join(rec.symbols_affected[:5])}" if rec.symbols_affected else "")
                )

        # Strategy performance
        strat_outcomes = defaultdict(list)
        for o in self._outcome_log:
            strat_outcomes[o["strategy"]].append(o["outcome"] == "WIN")
        for strat, results in sorted(strat_outcomes.items(), key=lambda x: len(x[1]), reverse=True):
            if len(results) >= 3:
                wr = sum(results) / len(results)
                if wr < 0.35:
                    perf_gaps.append(f"{strat}: {wr:.0%} WR ({len(results)} trades) ← BELOW THRESHOLD")
                elif wr > 0.60:
                    healthy.append(f"{strat}: {wr:.0%} WR ({len(results)} trades) ← STRONG")

        # Confidence calibration
        conf_buckets = defaultdict(list)
        for o in self._outcome_log:
            bucket = int(o["confidence"] // 5) * 5
            conf_buckets[bucket].append(o["outcome"] == "WIN")
        for bucket, results in sorted(conf_buckets.items()):
            if len(results) >= 3:
                wr = sum(results) / len(results)
                if wr < 0.40 and bucket < 80:
                    perf_gaps.append(
                        f"Confidence {bucket}-{bucket+4}: {wr:.0%} WR over {len(results)} trades "
                        f"← signals at this level are near coin-flip"
                    )

        # Applied overrides summary
        for ov in self._applied_overrides.values():
            auto_actions.append(
                f"{ov['description']} (applied {(now-ov['applied_at'])/3600:.1f}h ago)"
            )

        # Healthy checks
        if not errors:
            healthy.append("No recurring errors in last 6h")
        if self._scan_count > 0:
            healthy.append(f"Scanner healthy — {self._scan_count} scans completed")

        # Get AI recommendations if available
        ai_recs = []
        try:
            from analyzers.ai_analyst import ai_analyst
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer, 'regime', None)
            regime_val = regime.value if regime else "UNKNOWN"
            analysis = await ai_analyst.analyse_dead_signals(regime_val)
            if analysis and analysis.recommendation:
                ai_recs.append(analysis.recommendation)
        except Exception:
            pass

        # Format report
        lines = [
            "🔬 <b>TitanBot Self-Diagnostic Report</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Period: {period_hours:.0f}h | Scans: {self._scan_count} | "
            f"Generated: {self._signals_generated} | Published: {self._signals_published}",
            "",
        ]

        if criticals:
            lines.append(f"🚨 <b>CRITICAL ({len(criticals)})</b>")
            for c in criticals:
                lines.append(f"→ {c}")
            lines.append("")

        if errors:
            lines.append(f"⚡ <b>ERRORS ({len(errors)})</b>")
            for e in errors[:5]:  # Max 5 to keep message manageable
                lines.append(f"→ {e}")
            lines.append("")

        if perf_gaps:
            lines.append(f"📊 <b>PERFORMANCE GAPS ({len(perf_gaps)})</b>")
            for p in perf_gaps[:5]:
                lines.append(f"→ {p}")
            lines.append("")

        if edge_cases:
            lines.append(f"🔍 <b>EDGE CASES ({len(edge_cases)})</b>")
            for ec in edge_cases:
                lines.append(f"→ {ec}")
            lines.append("")

        if ai_recs:
            lines.append("🤖 <b>AI RECOMMENDATIONS</b>")
            for rec in ai_recs:
                lines.append(f"→ {rec}")
            lines.append("")

        if auto_actions:
            lines.append(f"⚙️ <b>AUTO-ACTIONS TAKEN ({len(auto_actions)})</b>")
            for a in auto_actions:
                lines.append(f"→ {a}")
            lines.append("")

        if healthy:
            lines.append(f"✅ <b>HEALTHY ({len(healthy)})</b>")
            for h in healthy[:5]:
                lines.append(f"→ {h}")
            lines.append("")

        pending = self.get_pending_approvals()
        if pending:
            lines.append(f"⏳ {len(pending)} pending approval(s) — use /ai approve to review")
            lines.append("")

        lines.append(f"Next report in: {period_hours:.0f} hours | /ai report for on-demand")

        await self.on_send_report("\n".join(lines))

        # ── NEW: Send supplementary reports as separate messages ──
        await self._send_telegram_health_report()
        await self._send_system_performance_report()
        await self._send_narrative_quality_report()

        # Reset counters after report
        self._scan_count = 0
        self._signals_generated = 0
        self._signals_published = 0
        self._error_counts.clear()

    async def _send_telegram_health_report(self):
        """Report on Telegram message formatting and delivery health."""
        if not self.on_send_report or not self._telegram_log:
            return

        now = time.time()
        recent = [m for m in self._telegram_log if now - m["ts"] < self._report_interval]
        if not recent:
            return

        issues = []
        # Check for oversized messages
        long_msgs = [m for m in recent if m["chars"] > 3800]
        if long_msgs:
            avg_len = sum(m["chars"] for m in long_msgs) / len(long_msgs)
            issues.append(
                f"⚠️ {len(long_msgs)} messages near/over Telegram 4096 char limit "
                f"(avg {avg_len:.0f} chars)\n"
                f"   Largest type: {max(long_msgs, key=lambda x: x['chars'])['type']}\n"
                f"   Fix: trim confluence notes or split long messages"
            )

        # Check for send errors
        failed = [m for m in recent if not m["send_ok"]]
        if failed:
            errors = Counter(m["error"][:60] for m in failed)
            top_err, top_count = errors.most_common(1)[0]
            issues.append(
                f"❌ {len(failed)} Telegram send failures\n"
                f"   Most common: {top_err} ({top_count}x)"
            )

        # Check signal card completeness
        if self._signal_card_log:
            cards = [c for c in self._signal_card_log if now - c["ts"] < self._report_interval]
            if cards:
                no_narrative = sum(1 for c in cards if not c["had_narrative"])
                no_btc = sum(1 for c in cards if not c["had_btc_context"])
                parse_fails = sum(1 for c in cards if not c["parse_ok"])
                avg_confluence = sum(c["confluence_count"] for c in cards) / len(cards)

                if no_narrative > 0 and len(cards) > 3:
                    issues.append(
                        f"📋 {no_narrative}/{len(cards)} signal cards missing AI narrative\n"
                        f"   Check AI mode is 'full' and OpenRouter key is valid"
                    )
                if parse_fails > 0:
                    issues.append(
                        f"🔴 {parse_fails} signal cards had HTML parse errors\n"
                        f"   Check fmt_price() output for special characters"
                    )
                if avg_confluence < 2:
                    issues.append(
                        f"📊 Avg confluence notes per signal: {avg_confluence:.1f} (low)\n"
                        f"   Signals may lack supporting context — check analyzer outputs"
                    )

        if not issues:
            return  # All good — no need to send empty report

        lines = [
            "📱 <b>Telegram Health Report</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
        ] + issues
        await self.on_send_report("\n".join(lines))

    async def _send_system_performance_report(self):
        """Report on API latency, DB performance, and scan speed."""
        if not self.on_send_report:
            return

        issues = []
        suggestions = []
        now = time.time()
        cutoff = now - self._report_interval

        # API latency analysis
        for endpoint_key, calls in self._api_latency.items():
            recent_calls = [c for c in calls if c["ts"] > cutoff]
            if len(recent_calls) < 5:
                continue
            avg_ms = sum(c["ms"] for c in recent_calls) / len(recent_calls)
            fail_rate = sum(1 for c in recent_calls if not c["ok"]) / len(recent_calls)

            if avg_ms > 2000:
                issues.append(
                    f"🐢 {endpoint_key}: avg latency {avg_ms:.0f}ms (slow)\n"
                    f"   Consider adding caching or reducing call frequency"
                )
            if fail_rate > 0.1:
                issues.append(
                    f"⚠️ {endpoint_key}: {fail_rate:.0%} failure rate\n"
                    f"   Check network stability and retry logic"
                )

        # DB performance
        if self._db_timings:
            recent_db = [d for d in self._db_timings if d["ts"] > cutoff]
            if recent_db:
                slow_writes = [d for d in recent_db if d["ms"] > 500]
                failed_writes = [d for d in recent_db if not d["ok"]]
                if slow_writes:
                    issues.append(
                        f"🗄️ {len(slow_writes)} slow DB operations (>500ms)\n"
                        f"   Consider increasing database.timeout in settings.yaml"
                    )
                if failed_writes:
                    issues.append(
                        f"❌ {len(failed_writes)} DB write failures\n"
                        f"   Check disk space and DB file permissions"
                    )

        # Scan speed analysis
        if self._scan_timings:
            recent_scans = [s for s in self._scan_timings if s["ts"] > cutoff]
            if recent_scans:
                avg_scan_ms = sum(s["ms"] for s in recent_scans) / len(recent_scans)
                slow_symbols = [s for s in recent_scans if s["ms"] > 5000]
                if avg_scan_ms > 3000:
                    # Check if specific slow symbols are dragging down the average
                    _slow_names_local = Counter(s["symbol"] for s in slow_symbols)
                    _top_slow = [sym for sym, _ in _slow_names_local.most_common(3)] if slow_symbols else []
                    if _top_slow:
                        # Specific symbols are causing it — auto-exclude recommendation
                        _sym_list = ", ".join(_top_slow)
                        issues.append(
                            f"⏱️ Avg scan time: {avg_scan_ms/1000:.1f}s per symbol\n"
                            f"   Root cause: API timeouts on {_sym_list}\n"
                            f"   ✅ Fix: These symbols are blocking the queue. "
                            f"Auto-excluding them for 2h would restore normal performance."
                        )
                        # Actually auto-exclude them if they've been slow for 3+ scans
                        for _slow_sym in _top_slow:
                            _sym_count = _slow_names_local.get(_slow_sym, 0)
                            if _sym_count >= 3:
                                try:
                                    from analyzers.sentinel_features import sentinel
                                    sentinel.add_to_blacklist(
                                        _slow_sym,
                                        reason=f"API timeout ({avg_scan_ms/1000:.0f}s avg scan)",
                                        hours=2,
                                    )
                                except Exception:
                                    pass
                    else:
                        # General slowness — could be workers
                        issues.append(
                            f"⏱️ Avg scan time: {avg_scan_ms/1000:.1f}s per symbol (slow)\n"
                            f"   Check exchange API latency or reduce batch_size in settings"
                        )
                if slow_symbols:
                    slow_names = Counter(s["symbol"] for s in slow_symbols)
                    top_slow = ", ".join(f"{sym}({ct}x)" for sym, ct in slow_names.most_common(3))
                    # Only show if we haven't already merged it into the avg scan issue above
                    if avg_scan_ms <= 3000:
                        issues.append(
                            f"🐌 Slow symbols: {top_slow}\n"
                            f"   API issues detected — auto-excluded for 2h if 3+ incidents"
                        )

        # Config suggestions based on observed behaviour
        if self._scan_count > 0 and self._signals_published == 0:
            suggestions.append(
                "💡 Zero signals published this period — consider sharing "
                "the diagnostic log with your developer to analyse pipeline bottlenecks"
            )

        if not issues and not suggestions:
            return

        lines = ["⚙️ <b>System Performance Report</b>", "━━━━━━━━━━━━━━━━━━━━━━━━"]
        if issues:
            lines += issues
        if suggestions:
            lines += ["", "💡 <b>Suggestions</b>"] + suggestions

        await self.on_send_report("\n".join(lines))

    async def _send_narrative_quality_report(self):
        """Report on AI narrative quality and suggest prompt improvements."""
        if not self.on_send_report or not self._narrative_quality_log:
            return

        now = time.time()
        recent = [n for n in self._narrative_quality_log
                  if now - n["ts"] < self._report_interval]
        if len(recent) < 3:
            return

        avg_clarity = sum(n["clarity"] for n in recent) / len(recent)
        avg_action = sum(n["actionability"] for n in recent) / len(recent)
        too_technical = sum(1 for n in recent if n["too_technical"])
        too_long = sum(1 for n in recent if n["too_long"])
        too_vague = sum(1 for n in recent if n["too_vague"])

        # Only send if there are quality issues worth reporting
        has_issues = (avg_clarity < 7 or avg_action < 7 or
                      too_technical > 1 or too_long > 1 or too_vague > 1)
        if not has_issues:
            return

        # Collect unique improvement suggestions
        improvements = list({n["improvement"] for n in recent
                             if n.get("improvement")})[:3]

        lines = [
            "🤖 <b>AI Narrative Quality Report</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Narratives reviewed: {len(recent)}",
            f"Avg clarity: {avg_clarity:.1f}/10",
            f"Avg actionability: {avg_action:.1f}/10",
            "",
        ]

        if too_technical > 0:
            lines.append(f"⚠️ {too_technical} narratives flagged as too technical")
        if too_long > 0:
            lines.append(f"⚠️ {too_long} narratives too long (>200 words)")
        if too_vague > 0:
            lines.append(f"⚠️ {too_vague} narratives too vague (missing specific levels)")

        if improvements:
            lines += ["", "💡 <b>Improvement suggestions from AI self-review:</b>"]
            for imp in improvements:
                lines.append(f"→ {imp}")

        lines += [
            "",
            "<i>These suggestions will be used to improve AI prompts automatically.</i>",
            "<i>Share this log with your developer to apply prompt improvements.</i>",
        ]

        await self.on_send_report("\n".join(lines))

    # ── Telegram-facing API ───────────────────────────────────

    async def get_report_on_demand(self) -> str:
        """Called by /ai report Telegram command."""
        old_interval = self._report_interval
        self._report_interval = 0  # Force immediate report
        await self._send_diagnostic_report()
        self._report_interval = old_interval
        return "Report sent ✅"

    def get_stats_summary(self) -> dict:
        """For Telegram /ai status panel."""
        now = time.time()
        recent_deaths = [d for d in self._death_log if now - d["ts"] < 3600]
        kill_counts = Counter(d["kill_reason"] for d in recent_deaths)
        return {
            "scan_count": self._scan_count,
            "total_scan_count": self._total_scan_count,
            "total_signals_generated": self._total_signals_generated,
            "total_signals_published": self._total_signals_published,
            "signals_generated": self._signals_generated,
            "signals_published": self._signals_published,
            "death_count_1h": len(recent_deaths),
            "top_kill_reason": kill_counts.most_common(1)[0][0] if kill_counts else "none",
            "pending_approvals": len(self._pending_approvals),
            "active_overrides": len(self._applied_overrides),
            "error_rate_per_min": len(self._error_rate_window),
        }

    def get_death_breakdown(self, hours: int = 24) -> dict:
        """
        FIX #35: Aggregated signal death analysis sorted by frequency.
        Called by /api/diagnostics endpoint so dashboard Kill Reasons chart
        and /ai explain both show which filter kills the most signals.
        """
        now = time.time()
        cutoff = now - (hours * 3600)
        recent = [d for d in self._death_log if d.get("ts", 0) >= cutoff]

        if not recent:
            return {
                "total_deaths": 0, "by_reason": {}, "by_strategy": {},
                "top_killer": "none", "worst_strategy": "none", "period_hours": hours,
            }

        reason_counts = Counter(d.get("kill_reason", "UNKNOWN") for d in recent)

        strat_data: dict = {}
        for d in recent:
            s = d.get("strategy", "UNKNOWN")
            if s not in strat_data:
                strat_data[s] = Counter()
            strat_data[s][d.get("kill_reason", "UNKNOWN")] += 1

        by_strategy = {
            s: {
                "deaths": sum(c.values()),
                "top_reason": c.most_common(1)[0][0] if c else "UNKNOWN",
            }
            for s, c in strat_data.items()
        }

        top_killer   = reason_counts.most_common(1)[0][0] if reason_counts else "none"
        worst        = max(by_strategy, key=lambda s: by_strategy[s]["deaths"]) if by_strategy else "none"

        return {
            "total_deaths":  len(recent),
            "by_reason":     dict(sorted(reason_counts.items(), key=lambda x: -x[1])),
            "by_strategy":   dict(sorted(by_strategy.items(), key=lambda x: -x[1]["deaths"])),
            "top_killer":    top_killer,
            "worst_strategy": worst,
            "period_hours":  hours,
        }



# ── Singleton ─────────────────────────────────────────────────
diagnostic_engine = DiagnosticEngine()
