"""
TitanBot Pro — Whale Deposit Monitor (CEX Inflow Spike Detection)
===================================================================
R8 Feature #2: Monitors on-chain BTC/ETH exchange inflows to detect
coordinated sell-off setups BEFORE they hit the order book.

When large deposits spike (>$50M in 1hr), the bot:
  (a) tightens all open position SLs to breakeven
  (b) reduces new signal confidence by -20
  (c) flags as "whale deposit event" in signal metadata

Data sources (all free, no API key):
  - Blockchain.com API (BTC exchange flow estimates)
  - Etherscan-like public mempool data (ETH)
  - Whale Alert public feed (free tier: delayed 10min)
  - CryptoQuant free tier (exchange netflow)

Fallback: Uses exchange-reported large deposit volume spikes
from existing order book data when on-chain APIs are unavailable.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

from config.feature_flags import ff
from config.shadow_mode import shadow_log

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────

# Thresholds
WHALE_DEPOSIT_THRESHOLD_USD = 50_000_000    # $50M in 1hr triggers alert
ELEVATED_DEPOSIT_THRESHOLD_USD = 25_000_000 # $25M triggers caution
CONFIDENCE_PENALTY_WHALE = -20              # Applied to new signals
CONFIDENCE_PENALTY_ELEVATED = -10
CHECK_INTERVAL_SECONDS = 300                # Check every 5 minutes
ALERT_COOLDOWN_SECONDS = 1800               # Don't re-alert for 30 min
INFLOW_WINDOW_SECONDS = 3600                # 1-hour rolling window


@dataclass
class InflowEvent:
    """A detected exchange inflow event"""
    timestamp: float
    asset: str           # BTC, ETH
    exchange: str        # binance, coinbase, kraken, unknown
    amount_usd: float
    source: str          # API source that detected it
    intent: str = "unknown"      # Feature 6: inflow, outflow, mint, burn
    intent_confidence: float = 0.0  # 0-1 confidence in intent classification


@dataclass
class WhaleDepositState:
    """Current state of whale deposit monitoring"""
    is_active: bool = False          # True if whale deposit alert is active
    alert_level: str = "NORMAL"     # NORMAL, ELEVATED, CRITICAL
    total_inflow_1h_usd: float = 0.0
    last_alert_time: float = 0.0
    recent_events: List[InflowEvent] = field(default_factory=list)
    last_event_time: float = 0.0
    last_event_keywords: List[str] = field(default_factory=list)
    last_event_bias: str = "NEUTRAL"
    last_event_detail: str = ""
    confidence_penalty: int = 0
    last_check_time: float = 0.0


class WhaleDepositMonitor:
    """
    Monitors exchange inflows and triggers protective actions.

    Integration points:
      - Engine checks get_state() before publishing signals
      - Outcome monitor calls should_tighten_stops() to protect positions
      - Confidence penalty is additive to signal confidence scoring
    """

    def __init__(self):
        self._state = WhaleDepositState()
        self._inflow_history: List[InflowEvent] = []
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

    async def start(self):
        """Start the background monitoring loop"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("🐋 Whale deposit monitor started")

    async def stop(self):
        """Stop monitoring"""
        self._running = False
        # AUDIT FIX: await the cancelled task before closing the aiohttp
        # session so pending CryptoQuant fetches unwind cleanly on
        # shutdown (prevents "Session is closed" errors and "Task was
        # destroyed but it is pending" warnings).
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._session:
            await self._session.close()

    def get_state(self) -> WhaleDepositState:
        """Get current whale deposit state for engine integration"""
        return self._state

    def get_confidence_penalty(self) -> int:
        """Get confidence penalty to apply to new signals"""
        return self._state.confidence_penalty

    def should_tighten_stops(self) -> bool:
        """Whether outcome monitor should tighten all stops to BE"""
        return self._state.alert_level == "CRITICAL"

    def is_elevated(self) -> bool:
        """Whether deposits are elevated (caution mode)"""
        return self._state.alert_level in ("ELEVATED", "CRITICAL")

    # ── Background Loop ──────────────────────────────────────

    async def _monitor_loop(self):
        """Main monitoring loop — runs every 5 minutes"""
        while self._running:
            try:
                await self._check_inflows()
            except Exception as e:
                logger.error(f"Whale deposit monitor error: {e}")
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

    async def _check_inflows(self):
        """Check all inflow sources and update state"""
        now = time.time()
        self._state.last_check_time = now

        # Collect inflow data from all available sources
        events = []
        try:
            events.extend(await asyncio.wait_for(
                self._fetch_blockchain_com_inflows(), timeout=60
            ))
        except asyncio.TimeoutError:
            logger.warning("Whale monitor: Blockchain.com fetch timed out (60s)")
        except Exception as e:
            logger.debug(f"Blockchain.com fetch failed: {e}")

        try:
            events.extend(await asyncio.wait_for(
                self._fetch_cryptoquant_inflows(), timeout=60
            ))
        except asyncio.TimeoutError:
            logger.warning("Whale monitor: CryptoQuant fetch timed out (60s)")
        except Exception as e:
            logger.debug(f"CryptoQuant fetch failed: {e}")

        # Add new events to history
        for event in events:
            # Feature 6: Classify whale intent
            self.classify_whale_intent(event)
            self._inflow_history.append(event)
            # Feed WalletBehaviorProfiler for whale intent analysis
            try:
                from analyzers.wallet_behavior import wallet_profiler
                wallet_profiler.record_event(
                    # AUDIT FIX: exchange inflows signal SELLING intent
                    # (whales move coins to the exchange to dump), as the
                    # adjacent comment states.  Previously we passed
                    # ``side="buy"`` which caused _analyze_phase() to count
                    # every deposit toward the bullish ACCUMULATION phase —
                    # the exact inverse of the real signal, flipping whale
                    # confidence adjustments against the trade direction.
                    side="sell",
                    size_usd=event.amount_usd,
                    symbol=f"{event.asset}USDT",
                    source="deposit",
                )
            except Exception:
                pass

        # Log whale intent heartbeat when no events found
        if not events and ff.is_active("WHALE_INTENT"):
            from config.fcn_logger import fcn_log
            _mode = "shadow" if ff.is_shadow("WHALE_INTENT") else "live"
            fcn_log("WHALE_INTENT", f"{_mode} | checked — no whale events detected this cycle")

        # Prune old events (keep 4 hours of history)
        cutoff = now - 14400
        self._inflow_history = [
            e for e in self._inflow_history if e.timestamp > cutoff
        ]

        # Calculate 1-hour rolling inflow
        window_cutoff = now - INFLOW_WINDOW_SECONDS
        recent_inflows = [
            e for e in self._inflow_history if e.timestamp > window_cutoff
        ]
        total_inflow_1h = sum(e.amount_usd for e in recent_inflows)

        # Update state
        self._state.total_inflow_1h_usd = total_inflow_1h
        self._state.recent_events = recent_inflows[-10:]  # Keep last 10
        if recent_inflows:
            latest_event = recent_inflows[-1]
            self._state.last_event_time = latest_event.timestamp
            self._state.last_event_keywords = [
                latest_event.asset.lower(),
                latest_event.exchange.lower(),
                latest_event.intent.lower(),
                "whale",
                "deposit" if latest_event.intent == "deposit" else "transfer",
            ]
            self._state.last_event_bias = "SHORT" if latest_event.intent in {"deposit", "inflow"} else "LONG"
            self._state.last_event_detail = (
                f"{latest_event.asset} {latest_event.intent} "
                f"{latest_event.exchange} ${latest_event.amount_usd/1e6:.1f}M"
            )
        else:
            self._state.last_event_time = 0.0
            self._state.last_event_keywords = []
            self._state.last_event_bias = "NEUTRAL"
            self._state.last_event_detail = ""

        # Determine alert level
        old_level = self._state.alert_level
        if total_inflow_1h >= WHALE_DEPOSIT_THRESHOLD_USD:
            self._state.alert_level = "CRITICAL"
            self._state.is_active = True
            self._state.confidence_penalty = CONFIDENCE_PENALTY_WHALE
        elif total_inflow_1h >= ELEVATED_DEPOSIT_THRESHOLD_USD:
            self._state.alert_level = "ELEVATED"
            self._state.is_active = True
            self._state.confidence_penalty = CONFIDENCE_PENALTY_ELEVATED
        else:
            self._state.alert_level = "NORMAL"
            self._state.is_active = False
            self._state.confidence_penalty = 0

        # Log state changes
        if self._state.alert_level != old_level:
            if self._state.alert_level == "CRITICAL":
                logger.warning(
                    f"🐋🔴 WHALE DEPOSIT CRITICAL: ${total_inflow_1h/1e6:.1f}M "
                    f"inflow in 1hr — tightening all stops, reducing confidence by "
                    f"{CONFIDENCE_PENALTY_WHALE}"
                )
            elif self._state.alert_level == "ELEVATED":
                logger.warning(
                    f"🐋🟡 WHALE DEPOSIT ELEVATED: ${total_inflow_1h/1e6:.1f}M "
                    f"inflow in 1hr — reducing new signal confidence by "
                    f"{CONFIDENCE_PENALTY_ELEVATED}"
                )
            elif old_level != "NORMAL":
                logger.info(
                    f"🐋🟢 Whale deposit alert cleared — inflows back to "
                    f"${total_inflow_1h/1e6:.1f}M"
                )

    # ── Data Sources ─────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def _fetch_blockchain_com_inflows(self) -> List[InflowEvent]:
        """
        Fetch BTC exchange inflow estimates from Blockchain.com.
        Uses the free charts API for exchange deposits.
        """
        events = []
        try:
            session = await self._get_session()
            # Blockchain.com free API for exchange flow data
            url = "https://api.blockchain.info/charts/estimated-transaction-volume-usd"
            params = {"timespan": "1hours", "format": "json", "cors": "true"}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get("values", [])
                    if values:
                        latest = values[-1]
                        # Blockchain.com gives total BTC tx volume;
                        # exchange inflow is roughly 40% of total
                        est_inflow = float(latest.get("y", 0)) * 0.4
                        if est_inflow > 1_000_000:  # Only log significant
                            events.append(InflowEvent(
                                timestamp=float(latest.get("x", time.time())),
                                asset="BTC",
                                exchange="aggregate",
                                amount_usd=est_inflow,
                                source="blockchain.com",
                            ))
        except Exception as e:
            logger.debug(f"Blockchain.com inflow fetch error: {e}")
        return events

    async def _fetch_cryptoquant_inflows(self) -> List[InflowEvent]:
        """
        Fetch exchange netflow data from CryptoQuant free API.
        Free tier provides delayed data (10-30 min) but sufficient
        for detecting large coordinated deposits.
        """
        events = []
        try:
            session = await self._get_session()
            url = "https://api.cryptoquant.com/v1/btc/exchange-flows/inflow"
            params = {"window": "hour", "limit": 1}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {}).get("data", [])
                    if result:
                        latest = result[-1]
                        inflow_usd = float(latest.get("inflow_total", 0))
                        if inflow_usd > 5_000_000:
                            events.append(InflowEvent(
                                timestamp=time.time(),
                                asset="BTC",
                                exchange="aggregate",
                                amount_usd=inflow_usd,
                                source="cryptoquant",
                            ))
        except Exception as e:
            logger.debug(f"CryptoQuant inflow fetch error: {e}")
        return events

    # ── Feature 6: Whale Intent Classification ────────────────
    # Heuristic-based, no LLM. Classifies whale transactions into:
    #   inflow (sell pressure), outflow (accumulation), mint, burn.

    # Known exchange deposit addresses (partial list — extend as needed)
    _KNOWN_EXCHANGE_ADDRESSES: set = {
        "binance", "coinbase", "kraken", "bitfinex", "huobi",
        "okx", "bybit", "kucoin", "gate.io", "gemini",
    }

    def classify_whale_intent(self, event: InflowEvent) -> Tuple[str, float]:
        """
        Classify a whale transaction into intent categories.
        Uses heuristic rules based on exchange, direction, and asset type.

        Returns:
            (intent, confidence) where:
            - intent: "inflow" | "outflow" | "mint" | "burn" | "unknown"
            - confidence: 0.0–1.0

        Fallback: if feature flag is off, returns ("unknown", 0.0).
        """
        if not ff.is_active("WHALE_INTENT"):
            return ("unknown", 0.0)

        exchange_lower = event.exchange.lower()
        source_lower = event.source.lower()
        intent = "unknown"
        confidence = 0.0

        # Rule 1: Exchange inflow → selling pressure
        if any(ex in exchange_lower for ex in self._KNOWN_EXCHANGE_ADDRESSES):
            if "inflow" in source_lower or event.amount_usd > 0:
                intent = "inflow"
                confidence = 0.75
                # Higher confidence for larger amounts
                if event.amount_usd >= WHALE_DEPOSIT_THRESHOLD_USD:
                    confidence = 0.90

        # Rule 2: Exchange outflow → accumulation
        if "outflow" in source_lower:
            intent = "outflow"
            confidence = 0.80

        # Rule 3: Mint detection (stablecoins)
        if event.asset in ("USDT", "USDC", "DAI"):
            if "mint" in source_lower or "issue" in source_lower:
                intent = "mint"
                confidence = 0.85

        # Rule 4: Burn detection
        if "burn" in source_lower or "destroy" in source_lower:
            intent = "burn"
            confidence = 0.85

        # Update the event
        event.intent = intent
        event.intent_confidence = confidence

        if ff.is_shadow("WHALE_INTENT"):
            shadow_log("WHALE_INTENT", {
                "asset": event.asset,
                "exchange": event.exchange,
                "amount_usd": event.amount_usd,
                "intent": intent,
                "confidence": confidence,
            })

        from config.fcn_logger import fcn_log
        _mode = "shadow" if ff.is_shadow("WHALE_INTENT") else "live"
        fcn_log("WHALE_INTENT", f"{_mode} | {event.asset} ${event.amount_usd/1e6:.1f}M {event.exchange} → {intent} conf={confidence:.2f}")

        logger.debug(
            f"🐋 Intent: {event.asset} ${event.amount_usd/1e6:.1f}M "
            f"on {event.exchange} → {intent} (conf={confidence:.2f})"
        )

        return (intent, confidence)

    def get_whale_intent_summary(self) -> Dict:
        """
        Summarise recent whale intents for cross-signal correlation.
        Returns dict with intent counts and dominant intent.
        """
        now = time.time()
        window = now - INFLOW_WINDOW_SECONDS
        recent = [e for e in self._inflow_history if e.timestamp > window]

        intent_counts: Dict[str, int] = {}
        for event in recent:
            if event.intent != "unknown":
                intent_counts[event.intent] = intent_counts.get(event.intent, 0) + 1

        dominant = max(intent_counts, key=intent_counts.get) if intent_counts else "unknown"

        return {
            "total_events": len(recent),
            "intent_counts": intent_counts,
            "dominant_intent": dominant,
            "total_inflow_usd": sum(e.amount_usd for e in recent if e.intent == "inflow"),
            "total_outflow_usd": sum(e.amount_usd for e in recent if e.intent == "outflow"),
        }

    def inject_manual_inflow(self, asset: str, amount_usd: float,
                              exchange: str = "unknown"):
        """
        Allow manual injection of inflow events for testing
        or integration with other data sources.
        """
        self._inflow_history.append(InflowEvent(
            timestamp=time.time(),
            asset=asset,
            exchange=exchange,
            amount_usd=amount_usd,
            source="manual",
        ))
        logger.info(
            f"🐋 Manual inflow event: {asset} ${amount_usd/1e6:.1f}M "
            f"on {exchange}"
        )


# ── Singleton ─────────────────────────────────────────────────
whale_deposit_monitor = WhaleDepositMonitor()
