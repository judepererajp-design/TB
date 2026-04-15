from __future__ import annotations

from typing import Callable, Dict


class BotHealthService:
    """Compact operator-focused health dashboard for Telegram."""

    def __init__(self, runtime_provider: Callable[[], Dict]) -> None:
        self._runtime_provider = runtime_provider

    async def render_dashboard(self) -> str:
        runtime = self._runtime_provider()
        ext_summary, diag_stats, ai_mode, circuit = self._collect_runtime_health()
        from data.database import db
        events = await db.get_recent_events(limit=5)

        event_lines = [
            f"• <b>{e.get('event_type', '?')}</b> — {str(e.get('message', '') or '')[:52]}"
            for e in events[:3]
        ] or ["• No recent incidents logged."]

        lines = [
            "🩺 <b>Bot Health</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Bot: <b>{'Paused' if runtime.get('paused') else 'Active'}</b>",
            f"Scans: <b>{runtime.get('scan_count', 0):,}</b> · Signals: <b>{runtime.get('signal_count', 0)}</b>",
            f"Telegram: <b>{ext_summary.get('telegram', 'NO_DATA')}</b>",
            f"Database: <b>{ext_summary.get('database', 'NO_DATA')}</b> · API: <b>{ext_summary.get('api', 'NO_DATA')}</b>",
            f"AI: <b>{ai_mode}</b> · Circuit: <b>{circuit}</b>",
            f"Errors/min: <b>{diag_stats.get('error_rate_per_min', 0)}</b>",
            f"Deaths 1h: <b>{diag_stats.get('death_count_1h', 0)}</b> · Top kill: <b>{diag_stats.get('top_kill_reason', 'none')}</b>",
            "",
            "<b>Recent incidents</b>",
            *event_lines,
        ]
        return "\n".join(lines)

    async def render_section(self, section: str) -> str:
        section = (section or "overview").lower()
        if section in ("overview", "refresh"):
            return await self.render_dashboard()
        if section == "scans":
            return self._render_scans()
        if section == "telegram":
            return self._render_telegram()
        if section == "ai":
            return self._render_ai()
        if section == "circuit":
            return self._render_circuit()
        if section == "events":
            return await self._render_events()
        return await self.render_dashboard()

    def _render_scans(self) -> str:
        runtime = self._runtime_provider()
        from core.diagnostic_engine import diagnostic_engine

        diag_stats = diagnostic_engine.get_stats_summary()
        return (
            "📡 <b>Scan Health</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Scans this hour: <b>{diag_stats.get('scan_count', 0)}</b>\n"
            f"Signals generated: <b>{diag_stats.get('signals_generated', 0)}</b>\n"
            f"Signals published: <b>{diag_stats.get('signals_published', 0)}</b>\n"
            f"Runtime scans: <b>{runtime.get('scan_count', 0):,}</b>\n"
            f"Runtime signals: <b>{runtime.get('signal_count', 0)}</b>\n"
            f"Queue pressure: <b>{'HIGH' if diag_stats.get('error_rate_per_min', 0) > 10 else 'NORMAL'}</b>"
        )

    def _render_telegram(self) -> str:
        from core.extended_health import ext_health

        report = ext_health.get_report().get("telegram")
        return (
            "📱 <b>Telegram Health</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Status: <b>{report.status}</b>\n"
            f"Calls: <b>{report.total_calls}</b>\n"
            f"Avg latency: <b>{report.avg_latency_ms:.1f}ms</b>\n"
            f"P95 latency: <b>{report.p95_latency_ms:.1f}ms</b>\n"
            f"Errors: <b>{report.error_count}</b> ({report.error_rate:.1%})\n"
            f"Slow calls: <b>{report.slow_count}</b>"
        )

    def _render_ai(self) -> str:
        from analyzers.ai_analyst import ai_analyst
        from core.diagnostic_engine import diagnostic_engine

        status = ai_analyst.get_status()
        diag_stats = diagnostic_engine.get_stats_summary()
        return (
            "🤖 <b>AI Health</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Mode: <b>{status.get('mode', 'unknown')}</b>\n"
            f"Initialized: <b>{'YES' if status.get('initialized') else 'NO'}</b>\n"
            f"Calls today: <b>{status.get('calls_total_today', 0)}</b>\n"
            f"Avg latency: <b>{status.get('avg_latency_ms', 0)}ms</b>\n"
            f"Pending approvals: <b>{diag_stats.get('pending_approvals', 0)}</b>\n"
            f"Active overrides: <b>{diag_stats.get('active_overrides', 0)}</b>"
        )

    def _render_circuit(self) -> str:
        from risk.manager import risk_manager

        breaker = getattr(risk_manager, "circuit_breaker", None)
        active = bool(getattr(breaker, "_paused", False) or getattr(breaker, "_hard_kill_active", False))
        reason = getattr(breaker, "_reason", "") if breaker else ""
        resume_at = getattr(breaker, "_resume_at", None) if breaker else None
        return (
            "🛡️ <b>Circuit Breaker</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"State: <b>{'ACTIVE' if active else 'CLEAR'}</b>\n"
            f"Reason: <b>{reason or 'None'}</b>\n"
            f"Resume at: <b>{resume_at or 'n/a'}</b>\n"
            "Snapshot: <b>live</b>"
        )

    async def _render_events(self) -> str:
        from data.database import db
        events = await db.get_recent_events(limit=12)
        lines = ["🧾 <b>Recent Incidents</b>", "━━━━━━━━━━━━━━━━━━━━━━━━"]
        if not events:
            lines.append("No recent incidents logged.")
        else:
            for event in events[:8]:
                lines.append(
                    f"• <b>{event.get('event_type', '?')}</b> — {str(event.get('message', '') or '')[:90]}"
                )
        return "\n".join(lines)

    def _collect_runtime_health(self):
        from analyzers.ai_analyst import ai_analyst
        from core.diagnostic_engine import diagnostic_engine
        from core.extended_health import ext_health
        from risk.manager import risk_manager

        ext_summary = ext_health.get_summary()
        diag_stats = diagnostic_engine.get_stats_summary()
        ai_mode = ai_analyst.get_status().get("mode", "unknown")
        breaker = getattr(risk_manager, "circuit_breaker", None)
        circuit = "ACTIVE" if (
            getattr(breaker, "_paused", False) or getattr(breaker, "_hard_kill_active", False)
        ) else "CLEAR"
        return ext_summary, diag_stats, ai_mode, circuit
