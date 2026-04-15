"""
TitanBot Pro — Network Monitor
=================================
Detects connectivity loss and manages trading state during outages.

States:
  ONLINE      → Normal operation
  OFFLINE     → Connectivity lost, execution paused
  RECOVERING  → Connection restored, revalidating signals

Architecture:
  - Pings exchange every 10 seconds
  - Auto-pauses execution on offline (does NOT clear signals)
  - Auto-revalidates pending signals on reconnect
  - Logs network vs compute time separately
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class NetworkState(Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    RECOVERING = "RECOVERING"


class NetworkMonitor:
    """Monitors exchange connectivity and manages trading state during outages."""

    def __init__(self):
        self._state = NetworkState.ONLINE
        self._last_ping: float = 0
        self._ping_interval: float = 10.0    # Check every 10 seconds
        self._offline_since: Optional[float] = None
        self._total_offline_time: float = 0
        self._offline_count: int = 0
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self.on_offline = None      # Called when connectivity lost
        self.on_reconnect = None    # Called when connectivity restored

    @property
    def state(self) -> NetworkState:
        return self._state

    @property
    def is_online(self) -> bool:
        return self._state == NetworkState.ONLINE

    @property
    def is_offline(self) -> bool:
        return self._state in (NetworkState.OFFLINE, NetworkState.RECOVERING)

    def start(self):
        """Start background connectivity monitoring"""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("🌐 Network monitor started (ping every 10s)")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Background loop that pings exchange periodically"""
        from data.api_client import api

        consecutive_failures = 0

        while self._running:
            try:
                await asyncio.sleep(self._ping_interval)

                try:
                    # Quick connectivity check
                    result = await asyncio.wait_for(
                        api.fetch_ticker("BTC/USDT"), timeout=8.0
                    )
                    if result:
                        consecutive_failures = 0

                        if self._state != NetworkState.ONLINE:
                            # We're back online!
                            downtime = time.time() - self._offline_since if self._offline_since else 0
                            self._total_offline_time += downtime
                            logger.warning(
                                f"🌐 Connection RESTORED after {downtime:.0f}s offline"
                            )

                            # Trigger reconnect handler before going ONLINE
                            if self.on_reconnect:
                                try:
                                    await self.on_reconnect()
                                except Exception as e:
                                    logger.error(f"Reconnect handler error: {e}")

                            # Set ONLINE atomically — skip RECOVERING to avoid
                            # race window where other coroutines see non-ONLINE state
                            self._state = NetworkState.ONLINE
                            logger.info("🌐 Network state: ONLINE")

                except (asyncio.TimeoutError, Exception) as e:
                    consecutive_failures += 1

                    if consecutive_failures >= 3 and self._state == NetworkState.ONLINE:
                        # Go offline after 3 consecutive failures
                        self._state = NetworkState.OFFLINE
                        self._offline_since = time.time()
                        self._offline_count += 1
                        logger.error(
                            f"🌐 Connection LOST — pausing execution "
                            f"({consecutive_failures} failed pings)"
                        )

                        if self.on_offline:
                            try:
                                await self.on_offline()
                            except Exception:
                                pass

                    elif self._state == NetworkState.OFFLINE:
                        logger.debug(
                            f"🌐 Still offline... "
                            f"({time.time() - self._offline_since:.0f}s)"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Network monitor error: {e}")

    def get_stats(self) -> dict:
        """Get network stats for heartbeat display"""
        current_downtime = 0
        if self._offline_since and self._state != NetworkState.ONLINE:
            current_downtime = time.time() - self._offline_since

        return {
            'state': self._state.value,
            'total_offline_time': self._total_offline_time + current_downtime,
            'offline_count': self._offline_count,
        }


# ── Singleton ──────────────────────────────────────────────
network_monitor = NetworkMonitor()
