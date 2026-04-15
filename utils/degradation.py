"""
TitanBot Pro — Degradation Engine
====================================
Project Inheritance Feature 10.

4-layer fallback chain per data source:
  1. Primary source
  2. Fallback source
  3. Stale cache (with age tag)
  4. Disable + alert

Per-source circuit breakers: 5 failures → OPEN (30s cooldown) → HALF_OPEN probe.

System mode tracking:
  news_mode:      PRIMARY | DEGRADED | BLIND
  execution_mode: FULL | SAFE | DISABLED

News blindness rules:
  - <30% RSS success → DEGRADED (news weight 0.3×)
  - 0% success       → BLIND (disable news gating, alert Telegram)

Database failure fallback (Feature 11):
  - On DB write failure: write to local file, queue for retry, alert.
  - Retry queue flushes on configurable interval.
  - NEVER silently drop a signal.

Fallback: if degradation engine itself fails, all sources treated as PRIMARY.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config.feature_flags import ff
from config.shadow_mode import shadow_log

logger = logging.getLogger(__name__)


# ── System modes ─────────────────────────────────────────────

class NewsMode(str, Enum):
    PRIMARY  = "PRIMARY"
    DEGRADED = "DEGRADED"
    BLIND    = "BLIND"


class ExecutionMode(str, Enum):
    FULL     = "FULL"
    SAFE     = "SAFE"
    DISABLED = "DISABLED"


# ── Circuit Breaker ──────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED    = "CLOSED"     # Normal operation
    OPEN      = "OPEN"       # Failures exceeded threshold, skip calls
    HALF_OPEN = "HALF_OPEN"  # Probing after cooldown


@dataclass
class CircuitBreaker:
    """
    Per-source circuit breaker.
    5 consecutive failures → OPEN for 30 seconds → HALF_OPEN probe.
    """
    name: str
    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

    def record_success(self):
        """Record a successful call — reset to CLOSED."""
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED
        self.last_success_time = time.time()

    def record_failure(self):
        """Record a failed call — may trip to OPEN."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(
                    f"⚡ Circuit OPEN: {self.name} — "
                    f"{self.consecutive_failures} consecutive failures"
                )
            self.state = CircuitState.OPEN

    def should_allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            # Check if cooldown has passed
            if time.time() - self.last_failure_time >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"⚡ Circuit HALF_OPEN: {self.name} — probing")
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True  # Allow probe
        return True


# ── Stale Cache ──────────────────────────────────────────────

@dataclass
class CachedData:
    """Cached data with age tracking."""
    data: Any
    cached_at: float
    source: str

    @property
    def age_seconds(self) -> float:
        return time.time() - self.cached_at

    @property
    def age_label(self) -> str:
        age = self.age_seconds
        if age < 60:
            return f"{age:.0f}s"
        if age < 3600:
            return f"{age / 60:.0f}m"
        return f"{age / 3600:.1f}h"

    @property
    def is_stale(self) -> bool:
        """Data older than 6 hours is considered too stale."""
        return self.age_seconds > 21600  # 6h


# ── Degradation Engine ───────────────────────────────────────

class DegradationEngine:
    """
    Manages system degradation across all external data sources.
    Tracks circuit breakers, cached data, and system modes.
    """

    def __init__(self):
        self._news_mode = NewsMode.PRIMARY
        self._execution_mode = ExecutionMode.FULL

        # Circuit breakers per source
        self._breakers: Dict[str, CircuitBreaker] = {}

        # Stale cache per source
        self._cache: Dict[str, CachedData] = {}

        # Telegram alert callback (wired at startup)
        self._on_degradation_alert: Optional[Callable] = None

        # DB failure fallback
        self._db_retry_queue: List[Dict] = []
        self._db_fallback_path = Path("logs/db_fallback.jsonl")

        # RSS success tracking
        self._last_rss_success_rate: float = 1.0

    def get_or_create_breaker(self, source: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a source."""
        if source not in self._breakers:
            self._breakers[source] = CircuitBreaker(name=source)
        return self._breakers[source]

    def cache_data(self, source: str, data: Any):
        """Cache data from a source for fallback use."""
        self._cache[source] = CachedData(
            data=data,
            cached_at=time.time(),
            source=source,
        )

    def get_cached(self, source: str) -> Optional[CachedData]:
        """Get cached data for a source, if available and not too stale."""
        cached = self._cache.get(source)
        if cached and not cached.is_stale:
            return cached
        return None

    # ── News mode management ─────────────────────────────────

    @property
    def news_mode(self) -> NewsMode:
        return self._news_mode

    @property
    def execution_mode(self) -> ExecutionMode:
        return self._execution_mode

    @property
    def news_weight_multiplier(self) -> float:
        """Returns the weight multiplier for news signals based on mode."""
        if self._news_mode == NewsMode.DEGRADED:
            return 0.3
        if self._news_mode == NewsMode.BLIND:
            return 0.0
        return 1.0

    def update_rss_success_rate(self, successes: int, total: int):
        """
        Update news mode based on RSS feed success rate.
        Called after each polling cycle.

        Rules:
          - <30% success → DEGRADED (reduce news weight to 0.3×)
          - 0% success   → BLIND (disable news gating)
        """
        if not ff.is_active("DEGRADATION_ENGINE"):
            return

        rate = successes / max(total, 1)
        self._last_rss_success_rate = rate
        old_mode = self._news_mode

        if rate == 0.0:
            self._news_mode = NewsMode.BLIND
        elif rate < 0.30:
            self._news_mode = NewsMode.DEGRADED
        else:
            self._news_mode = NewsMode.PRIMARY

        from config.fcn_logger import fcn_log
        import logging as _logging
        _de_mode = "shadow" if ff.is_shadow("DEGRADATION_ENGINE") else "live"

        if self._news_mode != old_mode:
            logger.warning(
                f"📡 News mode: {old_mode.value} → {self._news_mode.value} "
                f"(RSS success rate: {rate:.0%}, {successes}/{total})"
            )

            fcn_log("DEGRADATION_ENGINE",
                    f"{_de_mode} | news mode: {old_mode.value} → {self._news_mode.value} RSS={rate:.0%} ({successes}/{total})",
                    level=_logging.WARNING)
        else:
            fcn_log("DEGRADATION_ENGINE",
                    f"{_de_mode} | checked — mode={self._news_mode.value} RSS={rate:.0%} ({successes}/{total})")

            if ff.is_shadow("DEGRADATION_ENGINE"):
                shadow_log("DEGRADATION_ENGINE", {
                    "type": "news_mode_change",
                    "old_mode": old_mode.value,
                    "new_mode": self._news_mode.value,
                    "rss_success_rate": rate,
                })
            elif self._on_degradation_alert:
                asyncio.ensure_future(
                    self._send_alert(
                        f"📡 News mode: {old_mode.value} → {self._news_mode.value}\n"
                        f"RSS success rate: {rate:.0%} ({successes}/{total})"
                    )
                )

    async def _send_alert(self, message: str):
        """Send a degradation alert via Telegram callback."""
        if self._on_degradation_alert:
            try:
                await self._on_degradation_alert(message)
            except Exception as e:
                logger.error(f"Degradation alert send failed: {e}")

    # ── Database failure fallback (Feature 11) ───────────────

    def handle_db_write_failure(self, record: Dict, error: Exception):
        """
        Called when a database write fails.
        Writes the signal to a local file, queues for retry, and alerts.
        Under NO circumstances should a signal be silently dropped.

        Safety limits:
          - Max queue size: DB_RETRY_MAX_QUEUE_SIZE (oldest records dropped)
          - Max retries per record: DB_RETRY_MAX_RETRIES_PER_RECORD
          - Alert when queue exceeds DB_RETRY_QUEUE_ALERT_THRESHOLD
        """
        from config.constants import NewsIntelligence as NI

        logger.error(
            f"🗄️ DB write FAILED: {error} — "
            f"writing to fallback file and queueing retry"
        )

        # Write to local fallback file immediately
        try:
            self._db_fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._db_fallback_path, "a") as f:
                entry = {
                    "ts": time.time(),
                    "error": str(error),
                    "record": record,
                }
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as write_err:
            logger.critical(
                f"🗄️ CRITICAL: Cannot write to fallback file: {write_err}"
            )

        # Enforce max queue size — drop oldest if at capacity
        max_size = NI.DB_RETRY_MAX_QUEUE_SIZE
        if len(self._db_retry_queue) >= max_size:
            dropped = self._db_retry_queue.pop(0)
            logger.warning(
                f"🗄️ DB retry queue at max ({max_size}), dropped oldest record "
                f"(first_failed_at={dropped.get('first_failed_at', '?')})"
            )

        # Queue for retry
        self._db_retry_queue.append({
            "record": record,
            "error": str(error),
            "first_failed_at": time.time(),
            "retry_count": 0,
        })

        # Alert if queue is growing large
        alert_threshold = NI.DB_RETRY_QUEUE_ALERT_THRESHOLD
        qsize = len(self._db_retry_queue)
        if qsize >= alert_threshold and qsize % alert_threshold == 0:
            logger.warning(
                f"⚠️ DB retry queue size: {qsize} records pending. "
                f"Check database connectivity."
            )
            if self._on_degradation_alert:
                try:
                    asyncio.ensure_future(self._send_alert(
                        f"⚠️ DB retry queue size: {qsize} records pending. "
                        f"Check database connectivity."
                    ))
                except RuntimeError:
                    pass  # No event loop available (e.g. during tests)

    def get_retry_queue_size(self) -> int:
        """Return the number of records waiting for DB retry."""
        return len(self._db_retry_queue)

    def pop_retry_batch(self, batch_size: int = 10) -> List[Dict]:
        """
        Pop a batch of records from the retry queue for retry.
        Increments retry_count. Records exceeding max retries are
        dead-lettered (logged and discarded) rather than retried forever.
        """
        from config.constants import NewsIntelligence as NI
        max_retries = NI.DB_RETRY_MAX_RETRIES_PER_RECORD

        batch = self._db_retry_queue[:batch_size]
        self._db_retry_queue = self._db_retry_queue[batch_size:]

        retryable = []
        for item in batch:
            item["retry_count"] = item.get("retry_count", 0) + 1
            if item["retry_count"] > max_retries:
                logger.error(
                    f"🗄️ DB record dead-lettered after {max_retries} retries: "
                    f"first_failed_at={item.get('first_failed_at', '?')}"
                )
                continue  # Drop — already written to fallback file
            retryable.append(item)

        return retryable

    # ── Status / observability ───────────────────────────────

    def get_status(self) -> Dict:
        """Return full degradation status for dashboards."""
        return {
            "news_mode": self._news_mode.value,
            "execution_mode": self._execution_mode.value,
            "rss_success_rate": self._last_rss_success_rate,
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failures": cb.consecutive_failures,
                }
                for name, cb in self._breakers.items()
            },
            "cached_sources": {
                name: {
                    "age": c.age_label,
                    "is_stale": c.is_stale,
                }
                for name, c in self._cache.items()
            },
            "db_retry_queue_size": len(self._db_retry_queue),
        }


# ── Singleton ─────────────────────────────────────────────────
degradation_engine = DegradationEngine()
