"""
Exchange Health Monitor
========================
Lightweight health check that derives an aggregate status from
:mod:`data.api_client`'s telemetry. Used by the aggregator as an
early gate so signals are not published while the exchange API is
down or stalling.

Why: during a real exchange outage, REST tickers freeze for several
minutes. Strategies still receive (stale) prices, fire breakout
triggers based on the last known close, then immediately invalidate
when the feed catches up — usually after the bot has already taken
a fill at a worse price. Suppressing publication during these
windows is one of the highest-leverage edge cases in the signal
pipeline.

Status model:
  * HEALTHY   — publish normally
  * DEGRADED  — publish with caution (logged but not blocked here;
                downstream ``ExchangeHealth`` thresholds may apply)
  * UNHEALTHY — block all new signals

Usage:
    from signals.exchange_health import exchange_health
    if exchange_health.should_block_signals():
        return None
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config.constants import ExchangeHealth as EHC

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


@dataclass
class HealthSnapshot:
    status: HealthStatus
    avg_latency_ms: float
    error_rate: float
    consecutive_failures: int
    is_healthy_flag: bool
    reason: str = ""

    @property
    def should_block(self) -> bool:
        return self.status == HealthStatus.UNHEALTHY


class ExchangeHealthMonitor:
    """Singleton health monitor reading from ``data.api_client.api``."""

    def __init__(self) -> None:
        self._consecutive_failures: int = 0
        self._last_recovery_ts: float = 0.0
        self._last_status: HealthStatus = HealthStatus.HEALTHY

    # ── Public API ─────────────────────────────────────────────

    def record_failure(self) -> None:
        """Increment the consecutive-failure counter (publish-time)."""
        self._consecutive_failures += 1

    def record_success(self) -> None:
        """Reset the consecutive-failure counter on a successful publish."""
        if self._consecutive_failures > 0:
            self._last_recovery_ts = time.time()
        self._consecutive_failures = 0

    def check_health(self) -> HealthSnapshot:
        """Compute current health from api_client telemetry + counters."""
        avg_lat = 0.0
        err_rate = 0.0
        is_healthy_flag = True
        total_reqs = 0
        try:
            from data.api_client import api
            stats = api.get_request_stats() or {}
            avg_lat = float(stats.get("avg_latency_ms", 0.0) or 0.0)
            total_reqs = int(stats.get("total_requests", 0) or 0)
            total_errs = int(stats.get("total_errors", 0) or 0)
            err_rate = total_errs / max(1, total_reqs)
            is_healthy_flag = bool(getattr(api, "is_healthy", True))
        except Exception as exc:
            # Defensive: if telemetry is unavailable we treat the
            # exchange as healthy. Failing-closed here would block
            # every signal during config bootstrap.
            logger.debug(f"exchange_health: telemetry unavailable ({exc})")

        # Hard fails first
        if self._consecutive_failures >= EHC.UNHEALTHY_CONSEC_FAILS:
            snap = HealthSnapshot(
                status=HealthStatus.UNHEALTHY,
                avg_latency_ms=avg_lat,
                error_rate=err_rate,
                consecutive_failures=self._consecutive_failures,
                is_healthy_flag=is_healthy_flag,
                reason=f"{self._consecutive_failures} consecutive publish failures",
            )
        elif total_reqs > 0 and not is_healthy_flag:
            # Only trust the api_client's is_healthy flag once it has
            # actually made calls — at process startup the flag is
            # False until the first successful fetch, which would
            # otherwise block every test/early-startup signal.
            snap = HealthSnapshot(
                status=HealthStatus.UNHEALTHY,
                avg_latency_ms=avg_lat,
                error_rate=err_rate,
                consecutive_failures=self._consecutive_failures,
                is_healthy_flag=is_healthy_flag,
                reason="api_client reports unhealthy",
            )
        elif (avg_lat >= EHC.UNHEALTHY_AVG_LATENCY_MS
              or err_rate >= EHC.UNHEALTHY_ERROR_RATE):
            snap = HealthSnapshot(
                status=HealthStatus.UNHEALTHY,
                avg_latency_ms=avg_lat,
                error_rate=err_rate,
                consecutive_failures=self._consecutive_failures,
                is_healthy_flag=is_healthy_flag,
                reason=(
                    f"latency={avg_lat:.0f}ms err_rate={err_rate:.1%} "
                    f"exceeds UNHEALTHY thresholds"
                ),
            )
        elif (avg_lat >= EHC.DEGRADED_AVG_LATENCY_MS
              or err_rate >= EHC.DEGRADED_ERROR_RATE):
            snap = HealthSnapshot(
                status=HealthStatus.DEGRADED,
                avg_latency_ms=avg_lat,
                error_rate=err_rate,
                consecutive_failures=self._consecutive_failures,
                is_healthy_flag=is_healthy_flag,
                reason=(
                    f"latency={avg_lat:.0f}ms err_rate={err_rate:.1%} "
                    f"exceeds DEGRADED thresholds"
                ),
            )
        else:
            # Recovery grace: stay DEGRADED for RECOVERY_GRACE_SECS
            # after the last consecutive-failure reset to avoid a
            # single recovered ticker re-opening the floodgates.
            now = time.time()
            in_grace = (
                self._last_recovery_ts > 0
                and now - self._last_recovery_ts < EHC.RECOVERY_GRACE_SECS
            )
            snap = HealthSnapshot(
                status=HealthStatus.DEGRADED if in_grace else HealthStatus.HEALTHY,
                avg_latency_ms=avg_lat,
                error_rate=err_rate,
                consecutive_failures=self._consecutive_failures,
                is_healthy_flag=is_healthy_flag,
                reason="recovery grace window" if in_grace else "",
            )

        # Log transitions only (avoid log spam every cycle)
        if snap.status != self._last_status:
            logger.warning(
                f"🛰️  Exchange health: {self._last_status.value} → "
                f"{snap.status.value} | {snap.reason}"
            )
            self._last_status = snap.status

        return snap

    def should_block_signals(self) -> bool:
        """Convenience: True iff the current status is UNHEALTHY."""
        return self.check_health().should_block


# ── Singleton ─────────────────────────────────────────────────
exchange_health = ExchangeHealthMonitor()
