"""
TitanBot Pro — WebSocket Price Feed
====================================
Real-time price feed via Binance Futures WebSocket combined streams.
Replaces 10–12s REST polling latency with sub-100ms push updates.

Feeds into the shared PriceCache so all monitors (Execution, Outcome,
Invalidation) benefit automatically.

Usage:
    from data.websocket_feed import ws_feed

    await ws_feed.start(["BTC/USDT", "ETH/USDT"])
    price = ws_feed.get_price("BTC/USDT")
    await ws_feed.stop()

Architecture:
    - Connects to Binance combined stream endpoint
    - Subscribes to @ticker (real-time price) and @kline_1m (candle updates)
    - Auto-reconnects with exponential backoff on disconnect
    - Heartbeat ping/pong monitoring
    - Respects Binance 200-stream-per-connection limit (shards automatically)
    - Thread-safe price lookups via dict (asyncio single-threaded)
"""

import asyncio
import json
import logging
import math
import time
from typing import Dict, List, Optional, Any, Callable

import aiohttp

from config.constants import WebSocket as WSConst

logger = logging.getLogger(__name__)


def _to_ws_symbol(symbol: str) -> str:
    """Convert exchange symbol format to Binance WS format.

    'BTC/USDT' → 'btcusdt'
    'BTCUSDT'  → 'btcusdt'
    """
    return symbol.replace("/", "").lower()


def _to_exchange_symbol(ws_symbol: str) -> str:
    """Convert Binance WS symbol back to exchange format.

    'btcusdt' → 'BTC/USDT' (best-effort: insert slash before USDT)
    """
    upper = ws_symbol.upper()
    for quote in ("USDT", "BUSD", "USDC"):
        if upper.endswith(quote):
            base = upper[: -len(quote)]
            return f"{base}/{quote}"
    return upper


class _ConnectionShard:
    """A single WebSocket connection handling up to MAX_STREAMS streams."""

    def __init__(self, shard_id: int, streams: List[str],
                 on_message: Callable[[dict], None]):
        self.shard_id = shard_id
        self.streams = list(streams)
        self._on_message = on_message
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # FIX AUDIT-8: Track actual WS handshake confirmation per shard.
        self.is_connected = False

    @property
    def url(self) -> str:
        joined = "/".join(self.streams)
        return f"{WSConst.BASE_URL}/stream?streams={joined}"

    async def connect(self, session: aiohttp.ClientSession):
        self._session = session
        self._running = True
        self._task = asyncio.create_task(self._listen(), name=f"ws-shard-{self.shard_id}")

    async def disconnect(self):
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _listen(self):
        while self._running:
            try:
                self._ws = await self._session.ws_connect(
                    self.url,
                    heartbeat=WSConst.PING_INTERVAL_SECS,
                    timeout=WSConst.PONG_TIMEOUT_SECS,
                )
                # FIX AUDIT-8: Mark connected only after actual WS handshake succeeds
                self.is_connected = True
                logger.info(
                    f"WS shard {self.shard_id} connected "
                    f"({len(self.streams)} streams)"
                )
                async for msg in self._ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            self._on_message(data)
                        except json.JSONDecodeError:
                            logger.debug("WS: malformed JSON ignored")
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
                # Connection dropped — mark disconnected
                self.is_connected = False

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self.is_connected = False
                if self._running:
                    logger.warning(
                        f"WS shard {self.shard_id} error: {exc}"
                    )
            except Exception as exc:
                self.is_connected = False
                if self._running:
                    logger.error(
                        f"WS shard {self.shard_id} unexpected error: {exc}"
                    )

            if self._running:
                logger.info(f"WS shard {self.shard_id} will reconnect…")
                # Caller (WebSocketFeed) handles reconnect backoff at top level,
                # but shards also do a small pause to avoid tight loops.
                await asyncio.sleep(WSConst.RECONNECT_BASE_DELAY_SECS)


class WebSocketFeed:
    """
    Real-time Binance Futures WebSocket feed.

    Manages one or more connection shards (each limited to
    MAX_STREAMS_PER_CONNECTION streams) and publishes price updates into
    the shared PriceCache.
    """

    def __init__(self):
        # Latest prices: exchange-format symbol → price
        self._prices: Dict[str, float] = {}
        self._timestamps: Dict[str, float] = {}

        # Subscribed symbols in exchange format (e.g. "BTC/USDT")
        self._symbols: List[str] = []

        # Connection management
        self._shards: List[_ConnectionShard] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._running = False
        self._reconnect_task: Optional[asyncio.Task] = None

        # Callbacks for price updates
        self._callbacks: List[Callable[[str, float], Any]] = []

        # Stats
        self._started_at: Optional[float] = None
        self._messages_received: int = 0
        self._reconnections: int = 0
        self._errors: int = 0
        self._last_message_at: Optional[float] = None

    # ── Public API ──────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """FIX AUDIT-8: True only when at least one shard has confirmed WS handshake."""
        if not self._shards:
            return self._connected  # fallback for pre-shard state
        return any(s.is_connected for s in self._shards)

    async def start(self, symbols: List[str]) -> None:
        """Connect to Binance WS and start receiving price updates."""
        if self._running:
            logger.warning("WebSocketFeed already running")
            return

        self._symbols = list(symbols)
        self._running = True
        self._started_at = time.time()

        self._session = aiohttp.ClientSession()
        try:
            await self._build_and_connect_shards()
        except Exception:
            await self._session.close()
            self._session = None
            self._running = False
            raise

        # Start health monitor
        self._reconnect_task = asyncio.create_task(
            self._health_monitor(), name="ws-health-monitor"
        )
        logger.info(
            f"🔌 WebSocketFeed started with {len(self._symbols)} symbols "
            f"across {len(self._shards)} shard(s)"
        )

    async def stop(self) -> None:
        """Clean shutdown of all connections."""
        self._running = False
        self._connected = False

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        for shard in self._shards:
            await shard.disconnect()
        self._shards.clear()

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        logger.info("🔌 WebSocketFeed stopped")

    async def subscribe(self, symbols: List[str]) -> None:
        """Add symbols dynamically. Rebuilds shards if needed."""
        new_symbols = [s for s in symbols if s not in self._symbols]
        if not new_symbols:
            return

        self._symbols.extend(new_symbols)
        logger.info(f"WS: subscribing to {len(new_symbols)} new symbol(s)")

        if self._running:
            await self._rebuild_shards()

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Remove symbols dynamically. Rebuilds shards if needed."""
        removed = [s for s in symbols if s in self._symbols]
        if not removed:
            return

        for sym in removed:
            self._symbols.remove(sym)
            self._prices.pop(sym, None)
            self._timestamps.pop(sym, None)

        logger.info(f"WS: unsubscribed from {len(removed)} symbol(s)")

        if self._running:
            await self._rebuild_shards()

    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol. Thread-safe dict lookup."""
        return self._prices.get(symbol)

    def get_all_prices(self) -> Dict[str, float]:
        """Return a snapshot of all current prices."""
        return dict(self._prices)

    # D-1 FIX: removed second (stale) `is_connected` property that shadowed the
    # correct per-shard implementation defined earlier in the class.  The duplicate
    # `return self._connected and self._running` used a legacy flag that was never
    # updated after shards were introduced, causing api_client fallback logic to
    # treat the feed as disconnected even when all shards were healthy.

    def connection_stats(self) -> Dict:
        """Return connection diagnostics."""
        uptime = time.time() - self._started_at if self._started_at else 0
        return {
            "connected": self._connected,
            "running": self._running,
            "uptime_secs": round(uptime, 1),
            "symbols": len(self._symbols),
            "shards": len(self._shards),
            "messages_received": self._messages_received,
            "reconnections": self._reconnections,
            "errors": self._errors,
            "cached_prices": len(self._prices),
            "last_message_age_secs": (
                round(time.time() - self._last_message_at, 1)
                if self._last_message_at else None
            ),
        }

    def on_price_update(self, callback: Callable[[str, float], Any]) -> None:
        """Register a callback for price updates: callback(symbol, price)."""
        self._callbacks.append(callback)

    # ── Internal ────────────────────────────────────────────────

    def _build_streams(self, symbols: List[str]) -> List[str]:
        """Build Binance WS stream names for a list of symbols."""
        streams = []
        for sym in symbols:
            ws_sym = _to_ws_symbol(sym)
            streams.append(f"{ws_sym}@ticker")
            streams.append(f"{ws_sym}@kline_1m")
        return streams

    async def _build_and_connect_shards(self):
        """Create shards respecting the per-connection stream limit."""
        all_streams = self._build_streams(self._symbols)
        max_per = WSConst.MAX_STREAMS_PER_CONNECTION

        # Chunk streams into groups of max_per
        chunks = [
            all_streams[i: i + max_per]
            for i in range(0, len(all_streams), max_per)
        ]

        for idx, chunk in enumerate(chunks):
            shard = _ConnectionShard(idx, chunk, self._handle_message)
            await shard.connect(self._session)
            self._shards.append(shard)

        # FIX AUDIT-8: Don't set _connected here — shard.connect() only spawns
        # a background task, the actual WS handshake happens in _listen().
        # Use the is_connected property which checks per-shard status.

    async def _rebuild_shards(self):
        """Disconnect all shards and rebuild with current symbol list."""
        for shard in self._shards:
            await shard.disconnect()
        self._shards.clear()
        self._connected = False

        if self._symbols and self._session and not self._session.closed:
            await self._build_and_connect_shards()

    def _handle_message(self, raw: dict):
        """Process a combined-stream message from Binance."""
        self._messages_received = (self._messages_received + 1) % WSConst.COUNTER_WRAP
        self._last_message_at = time.time()

        # Combined stream format: {"stream": "btcusdt@ticker", "data": {...}}
        data = raw.get("data", raw)
        event_type = data.get("e")

        if event_type == "24hrTicker":
            self._handle_ticker(data)
        elif event_type == "kline":
            self._handle_kline(data)

    # D-3: Maximum believable price for any crypto pair quoted in USDT.
    # A price above this (e.g. BTC/USDT at $10 million) is plainly malformed.
    _MAX_SANE_PRICE: float = 1_000_000_000.0  # $1 billion per unit

    def _validate_price(self, price: float, symbol: str) -> bool:
        """Return True only if price is a finite, positive, sane value.

        D-3 FIX: Binance occasionally emits malformed ticks (nan, inf,
        negative, or astronomically large values) during exchange issues.
        Accepting them without validation poisons all downstream math:
        ``if price > entry_high`` becomes permanently True for +inf, and
        NaN propagates silently through slippage and R-multiple calculations.
        """
        if not math.isfinite(price):
            logger.warning("WS: non-finite price %.6g for %s — discarded", price, symbol)
            return False
        if price <= 0:
            logger.warning("WS: non-positive price %.6g for %s — discarded", price, symbol)
            return False
        if price > self._MAX_SANE_PRICE:
            logger.warning("WS: absurd price %.6g for %s — discarded", price, symbol)
            return False
        return True

    def _handle_ticker(self, data: dict):
        """Extract price from 24hr ticker event."""
        ws_symbol = data.get("s", "")  # e.g. "BTCUSDT"
        price_str = data.get("c")      # last price (string)
        if not ws_symbol or price_str is None:
            return

        try:
            price = float(price_str)
        except (ValueError, TypeError):
            return

        # D-3 FIX: reject nan/inf/negative/absurd values before storing
        symbol = _to_exchange_symbol(ws_symbol.lower())
        if not self._validate_price(price, symbol):
            return

        self._prices[symbol] = price
        self._timestamps[symbol] = time.time()

        self._publish_to_price_cache(symbol, price)
        self._notify_callbacks(symbol, price)

    def _handle_kline(self, data: dict):
        """Extract close price from kline event."""
        kline = data.get("k", {})
        ws_symbol = kline.get("s", "")
        price_str = kline.get("c")     # kline close price
        if not ws_symbol or price_str is None:
            return

        try:
            price = float(price_str)
        except (ValueError, TypeError):
            return

        # D-3 FIX: reject nan/inf/negative/absurd values before storing
        symbol = _to_exchange_symbol(ws_symbol.lower())
        if not self._validate_price(price, symbol):
            return

        self._prices[symbol] = price
        self._timestamps[symbol] = time.time()

        self._publish_to_price_cache(symbol, price)
        self._notify_callbacks(symbol, price)

    def _publish_to_price_cache(self, symbol: str, price: float):
        """Push price into the shared PriceCache if available."""
        try:
            from core.price_cache import price_cache
            if symbol in price_cache.subscribed_symbols:
                price_cache._prices[symbol] = price
                price_cache._timestamps[symbol] = time.time()
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"WS: price_cache update failed: {exc}")

    def _notify_callbacks(self, symbol: str, price: float):
        """Fire all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(symbol, price)
            except Exception as exc:
                logger.debug(f"WS callback error: {exc}")

    async def _health_monitor(self):
        """Monitor connection health; trigger reconnect if stale."""
        while self._running:
            await asyncio.sleep(WSConst.PING_INTERVAL_SECS)

            if not self._running:
                break

            if self._last_message_at is None:
                continue

            silence = time.time() - self._last_message_at
            if silence > WSConst.PONG_TIMEOUT_SECS:
                logger.warning(
                    f"WS: no messages for {silence:.0f}s — reconnecting"
                )
                self._reconnections = (self._reconnections + 1) % WSConst.COUNTER_WRAP
                self._connected = False
                # D-2 FIX: clear stale price cache before reconnecting so that
                # downstream consumers never read prices from the dead connection
                # window.  Without this, entry-zone math and slippage estimates
                # kept using the last known price for the entire reconnect period.
                self._prices.clear()
                self._timestamps.clear()
                await self._reconnect_with_backoff()

    async def _reconnect_with_backoff(self):
        """Reconnect all shards with exponential backoff."""
        delay = WSConst.RECONNECT_BASE_DELAY_SECS
        max_delay = WSConst.RECONNECT_MAX_DELAY_SECS

        while self._running:
            try:
                await self._rebuild_shards()
                if self._connected:
                    logger.info("WS: reconnected successfully")
                    return
            except Exception as exc:
                self._errors += 1
                logger.warning(f"WS reconnect failed: {exc}")

            logger.info(f"WS: retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)


# ── Singleton ──────────────────────────────────────────────
ws_feed = WebSocketFeed()
