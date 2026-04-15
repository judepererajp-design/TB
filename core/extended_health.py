"""
TitanBot Pro — Extended Health Monitoring
==========================================
Monitors subsystem latencies (DB, API, Telegram) and exposes
metrics for the diagnostic engine and dashboard.

Supplements core/health_monitor.py (signal/OHLCV validation) with
infrastructure-level health checks.

Usage:
    from core.extended_health import ext_health
    ext_health.record_db_latency(op_name, latency_ms)
    ext_health.record_api_latency(endpoint, latency_ms, success)
    ext_health.record_telegram_latency(latency_ms, success)
    report = ext_health.get_report()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.constants import Diagnostics as DIAG

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────
_DB_SLOW_MS = DIAG.DB_SLOW_OPERATION_MS            # 500 ms
_API_WARN_MS = DIAG.API_LATENCY_WARNING_MS          # 2000 ms
_BUFFER_SIZE = DIAG.API_LATENCY_BUFFER_SIZE          # 100 entries


@dataclass
class LatencySample:
    """Single latency measurement."""
    name: str
    latency_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class SubsystemReport:
    """Health summary for one subsystem."""
    name: str
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    total_calls: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    slow_count: int = 0
    status: str = "HEALTHY"  # HEALTHY | DEGRADED | UNHEALTHY


class ExtendedHealthMonitor:
    """Infrastructure-level health monitoring."""

    def __init__(self) -> None:
        self._db_samples: deque[LatencySample] = deque(maxlen=_BUFFER_SIZE)
        self._api_samples: deque[LatencySample] = deque(maxlen=_BUFFER_SIZE)
        self._telegram_samples: deque[LatencySample] = deque(maxlen=_BUFFER_SIZE)

    # ── Recording ───────────────────────────────────────────

    def record_db_latency(
        self, op_name: str, latency_ms: float, success: bool = True
    ) -> None:
        """Record a database operation latency."""
        self._db_samples.append(
            LatencySample(name=op_name, latency_ms=latency_ms, success=success)
        )
        if latency_ms > _DB_SLOW_MS:
            logger.warning(
                f"🐢 Slow DB op: {op_name} took {latency_ms:.0f}ms "
                f"(threshold={_DB_SLOW_MS}ms)"
            )

    def record_api_latency(
        self, endpoint: str, latency_ms: float, success: bool = True
    ) -> None:
        """Record an API call latency."""
        self._api_samples.append(
            LatencySample(name=endpoint, latency_ms=latency_ms, success=success)
        )
        if latency_ms > _API_WARN_MS:
            logger.warning(
                f"🐢 Slow API call: {endpoint} took {latency_ms:.0f}ms "
                f"(threshold={_API_WARN_MS}ms)"
            )

    def record_telegram_latency(
        self, latency_ms: float, success: bool = True
    ) -> None:
        """Record a Telegram API call latency."""
        self._telegram_samples.append(
            LatencySample(name="telegram", latency_ms=latency_ms, success=success)
        )

    # ── Reporting ───────────────────────────────────────────

    def _compute_report(
        self, name: str, samples: deque, slow_threshold_ms: float
    ) -> SubsystemReport:
        """Compute a health report from a sample buffer."""
        if not samples:
            return SubsystemReport(name=name, status="NO_DATA")

        latencies = [s.latency_ms for s in samples]
        errors = [s for s in samples if not s.success]
        slow = [s for s in samples if s.latency_ms > slow_threshold_ms]

        total = len(samples)
        err_count = len(errors)
        err_rate = err_count / total if total > 0 else 0.0

        sorted_lat = sorted(latencies)
        p95_idx = max(0, int((len(sorted_lat) - 1) * 0.95))

        status = "HEALTHY"
        if err_rate > DIAG.API_FAILURE_RATE_WARNING:
            status = "UNHEALTHY"
        elif len(slow) > total * 0.2:  # >20% slow
            status = "DEGRADED"

        return SubsystemReport(
            name=name,
            avg_latency_ms=round(sum(latencies) / len(latencies), 1),
            p95_latency_ms=round(sorted_lat[p95_idx], 1),
            max_latency_ms=round(max(latencies), 1),
            total_calls=total,
            error_count=err_count,
            error_rate=round(err_rate, 4),
            slow_count=len(slow),
            status=status,
        )

    def get_report(self) -> Dict[str, SubsystemReport]:
        """Get health reports for all subsystems."""
        return {
            "database": self._compute_report("database", self._db_samples, _DB_SLOW_MS),
            "api": self._compute_report("api", self._api_samples, _API_WARN_MS),
            "telegram": self._compute_report(
                "telegram", self._telegram_samples, _API_WARN_MS
            ),
        }

    def get_summary(self) -> Dict[str, str]:
        """Quick status for each subsystem."""
        report = self.get_report()
        return {name: r.status for name, r in report.items()}

    def is_healthy(self) -> bool:
        """True if all subsystems are HEALTHY or have NO_DATA."""
        return all(
            r.status in ("HEALTHY", "NO_DATA")
            for r in self.get_report().values()
        )


# ── Singleton ───────────────────────────────────────────────────
ext_health = ExtendedHealthMonitor()
