"""
TitanBot Pro — Whale Intelligence Aggregator
================================================
Buffers whale detections and sends aggregated intelligence summaries
to Telegram instead of per-event spam.

Architecture:
  Scanner detects whale → engine adds to buffer → aggregator flushes periodically

Output: One clean Telegram summary every 60-90 seconds with:
  - Directional bias (buy vs sell dominance)
  - Top whale events with price context
  - Market interpretation
  - Range position context

Console still gets every individual detection (developer telemetry).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from utils.formatting import fmt_price, fmt_price_raw

logger = logging.getLogger(__name__)


@dataclass
class WhaleEvent:
    """Single whale detection"""
    symbol: str
    side: str           # 'buy' or 'sell'
    order_usd: float
    price: float
    qty: float
    fill_prob: float = 0.5    # 0.0–1.0 heuristic fill probability
    spread_pct: float = 0.0   # bid/ask spread at detection time
    timestamp: float = field(default_factory=time.time)


class WhaleAggregator:
    """
    Buffers whale detections and publishes aggregated summaries.
    
    Smart notification policy:
    - Flush every 5 minutes (not 90s)
    - Need 3+ significant events (not 2)
    - Only $250k+ trades count for Telegram (not $150k)
    - Max 8 whale summaries per day (prevents notification fatigue)
    - During CHOPPY regime: only send if directional bias is strong
    - Console still gets every detection (developer telemetry)
    """

    def __init__(self):
        self._buffer: List[WhaleEvent] = []
        self._last_flush: float = 0
        self._flush_interval: float = 300.0     # Flush every 5 min
        self._min_events_to_send: int = 3
        self._min_usd_for_telegram: float = 250_000
        self._lock = asyncio.Lock()

        # Daily message cap
        self._daily_sends: int = 0
        self._max_daily_sends: int = 8
        self._daily_reset_time: float = 0

        # Stats tracking
        self._total_buy_volume: float = 0
        self._total_sell_volume: float = 0
        self._total_events: int = 0

        # 30-day rolling history for percentile ranking
        # Stores (timestamp, order_usd) for each whale event
        # Used to answer: "Is this whale large by historical standards?"
        from collections import deque
        self._size_history: deque = deque(maxlen=2000)  # ~30d at current flow rate
        self._daily_volume_history: deque = deque(maxlen=30)  # daily totals
        self._current_day_volume: float = 0.0
        # FIX B12: seed with current time, not 0.  A zero seed makes the
        # first event trip `now - 0 > 86400` and rolls what is effectively
        # an empty day into daily history.  Harmless today but trap-prone
        # once downstream consumers compute daily-volume moving averages.
        self._current_day_start: float = time.time()

        # FIX Q15: per-symbol telegram thresholds scaled by 24h volume.
        # A flat 250k threshold is noise on BTC ($30B+ daily) and noise-free
        # on a $20M-daily alt.  Cache computed thresholds for ~1h to avoid
        # hammering the symbol-volume cache each whale event.
        self._symbol_threshold_cache: Dict[str, Tuple[float, float]] = {}  # symbol -> (threshold_usd, expires_at)

    def add_event(self, whale: Dict):
        """Add a whale detection, recording to 30d history for percentile ranking."""
        import time as _t
        event = WhaleEvent(
            symbol=whale['symbol'],
            side=whale['side'],
            order_usd=whale['order_usd'],
            price=whale.get('price', 0),
            qty=whale.get('qty', 0),
            fill_prob=whale.get('fill_prob', 0.5),
            spread_pct=whale.get('spread_pct', 0.0),
        )
        self._buffer.append(event)
        self._total_events += 1

        if event.side == 'buy':
            self._total_buy_volume += event.order_usd
        else:
            self._total_sell_volume += event.order_usd

        # Feed WalletBehaviorProfiler for whale intent analysis
        try:
            from analyzers.wallet_behavior import wallet_profiler
            wallet_profiler.record_event(
                side=event.side,
                size_usd=event.order_usd,
                symbol=event.symbol,
                source="orderbook",
            )
        except Exception:
            pass

        # Record to 30d size history for percentile ranking
        now = _t.time()
        self._size_history.append((now, event.order_usd))

        # Track daily volume
        if now - self._current_day_start > 86400:
            if self._current_day_volume > 0:
                self._daily_volume_history.append(self._current_day_volume)
            self._current_day_volume = 0.0
            self._current_day_start = now
        self._current_day_volume += event.order_usd

        # Compute percentile rank for this event
        pct = self.size_percentile(event.order_usd)
        pct_label = (
            "🔥 MEGA" if pct >= 0.95 else
            "⚡ LARGE" if pct >= 0.80 else
            "📊 Notable" if pct >= 0.60 else ""
        )

        logger.info(
            f"🐋 Whale: {event.symbol} {event.side.upper()} "
            f"${event.order_usd:,.0f} @ {fmt_price_raw(event.price)}"
            + (f" {pct_label} ({pct:.0%} of 30d)" if pct_label else "")
        )

    def size_percentile(self, order_usd: float) -> float:
        """Return percentile rank of this order size vs 30d history (0-1)."""
        import numpy as _np
        if len(self._size_history) < 10:
            return 0.5  # Not enough history
        sizes = [v for _, v in self._size_history]
        return float(_np.searchsorted(sorted(sizes), order_usd)) / len(sizes)

    def _effective_telegram_threshold(self, symbol: str) -> float:
        """
        FIX Q15: return the minimum whale order-size (USD) that qualifies
        for a Telegram summary on this specific symbol.

        Baseline floor: ``self._min_usd_for_telegram`` (250k default).
        Scaled floor:   0.5% of rolling 24h quote volume.

        The larger of the two wins so that majors require genuinely large
        prints before being summarised, while small-caps still surface
        their (smaller but still-material) whale prints.  Cached for 1h
        to avoid hitting the volume cache on every event.
        """
        now = time.time()
        cached = self._symbol_threshold_cache.get(symbol)
        if cached and cached[1] > now:
            return cached[0]

        threshold = self._min_usd_for_telegram
        try:
            # Read 24h quote volume from the scanner/price cache when available.
            # Be defensive — this helper must never raise during flush.
            from core.price_cache import price_cache as _pc
            vol_24h = 0.0
            getter = getattr(_pc, 'get_24h_volume', None)
            if callable(getter):
                vol_24h = float(getter(symbol) or 0.0)
            if vol_24h > 0:
                # 0.5% of 24h volume (small enough that alts qualify, large
                # enough that BTC/ETH need genuine six-figure prints).
                scaled = 0.005 * vol_24h
                threshold = max(self._min_usd_for_telegram, scaled)
        except Exception:
            pass

        self._symbol_threshold_cache[symbol] = (threshold, now + 3600.0)
        return threshold

    def should_flush(self) -> bool:
        """Check if it's time to flush the buffer"""
        if not self._buffer:
            return False
        return time.time() - self._last_flush >= self._flush_interval

    async def flush(self) -> Optional[str]:
        """
        Flush buffer and return formatted Telegram summary.
        Returns None if nothing worth sending.
        
        Smart filtering:
          - Respects daily message cap
          - During CHOPPY: only sends if strong directional bias
          - Requires minimum significant events
        """
        async with self._lock:
            if not self._buffer:
                return None

            events = self._buffer.copy()
            self._buffer.clear()
            self._last_flush = time.time()

        # Daily cap reset check
        now = time.time()
        if now - self._daily_reset_time > 86400:
            self._daily_sends = 0
            self._daily_reset_time = now

        # Respect daily cap
        if self._daily_sends >= self._max_daily_sends:
            logger.debug(
                f"Whale summary suppressed — daily cap ({self._max_daily_sends}) reached"
            )
            return None

        # Filter: only include events above Telegram threshold.
        # FIX Q15: scale the threshold per-symbol by 24h volume.  A fixed
        # 250k is noise on BTC and gated-too-high on a thin alt.  Use
        # max(250k, 0.5% × 24h_volume) so majors require bigger prints
        # while small-caps still surface meaningful flow.
        significant = [
            e for e in events
            if e.order_usd >= self._effective_telegram_threshold(e.symbol)
        ]

        # If not enough significant events, skip Telegram
        if len(significant) < self._min_events_to_send:
            return None

        # During CHOPPY regime: only send if strong directional bias
        try:
            from analyzers.regime import regime_analyzer, Regime
            if regime_analyzer.regime == Regime.CHOPPY:
                buy_vol = sum(e.order_usd for e in significant if e.side == 'buy')
                sell_vol = sum(e.order_usd for e in significant if e.side == 'sell')
                total = buy_vol + sell_vol
                if total > 0:
                    dominance = max(buy_vol, sell_vol) / total
                    if dominance < 0.70:
                        # Balanced flow during chop — not interesting enough
                        logger.debug(
                            f"Whale summary suppressed in CHOPPY — "
                            f"no strong bias ({dominance:.0%})"
                        )
                        return None
        except Exception:
            pass

        # Use significant events for display
        display_events = sorted(significant, key=lambda e: e.order_usd, reverse=True)[:6]

        self._daily_sends += 1
        return self._format_summary(events, display_events)

    def _format_summary(self, all_events: List[WhaleEvent], display_events: List[WhaleEvent]) -> str:
        """Format aggregated whale intelligence summary for Telegram"""
        buy_vol = sum(e.order_usd for e in all_events if e.side == 'buy')
        sell_vol = sum(e.order_usd for e in all_events if e.side == 'sell')
        total_vol = buy_vol + sell_vol
        buy_count = sum(1 for e in all_events if e.side == 'buy')
        sell_count = sum(1 for e in all_events if e.side == 'sell')

        # Determine directional bias
        if total_vol == 0:
            bias = "NEUTRAL"
            bias_emoji = "⚖️"
        elif buy_vol > sell_vol * 1.5:
            bias = "BUY DOMINANT"
            bias_emoji = "🟢"
        elif sell_vol > buy_vol * 1.5:
            bias = "SELL DOMINANT"
            bias_emoji = "🔴"
        elif buy_vol > sell_vol:
            bias = "MILD BUY"
            bias_emoji = "🟡"
        elif sell_vol > buy_vol:
            bias = "MILD SELL"
            bias_emoji = "🟡"
        else:
            bias = "BALANCED"
            bias_emoji = "⚖️"

        # Build event list
        lines = []
        # Sort by size descending
        display_events.sort(key=lambda e: e.order_usd, reverse=True)
        for e in display_events[:6]:
            side_emoji = "🟢" if e.side == 'buy' else "🔴"
            lines.append(
                f"  {side_emoji} {e.symbol.replace('/USDT', '')} — "
                f"${e.order_usd / 1000:,.0f}k {e.side} @ {fmt_price(e.price)}"
            )

        event_list = "\n".join(lines)

        # Interpretation
        if bias in ("BUY DOMINANT",):
            interpretation = "Accumulation detected — watch for upside expansion"
        elif bias in ("SELL DOMINANT",):
            interpretation = "Distribution detected — watch for downside pressure"
        elif len(all_events) >= 5:
            interpretation = "High whale activity — volatility expansion likely"
        else:
            interpretation = "Normal institutional flow"

        # Add percentile context if history available
        vol_pct = self.size_percentile(total_vol) if total_vol > 0 else 0.5
        pct_note = ""
        if vol_pct >= 0.90:
            pct_note = f" — {vol_pct:.0%} of 30d history (exceptional)"
        elif vol_pct >= 0.75:
            pct_note = f" — {vol_pct:.0%} of 30d history (elevated)"

        text = (
            f"🐋 <b>WHALE FLOW ({len(all_events)} events)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{bias_emoji} <b>{bias}</b>\n"
            f"💰 Buy: ${buy_vol / 1000:,.0f}k ({buy_count}) | "
            f"Sell: ${sell_vol / 1000:,.0f}k ({sell_count}){pct_note}\n\n"
            f"{event_list}\n\n"
            f"<i>{interpretation}</i>"
        )

        return text

    def get_session_stats(self) -> Dict:
        """Get running session stats for heartbeat"""
        return {
            'total_events': self._total_events,
            'total_buy_volume': self._total_buy_volume,
            'total_sell_volume': self._total_sell_volume,
            'buffer_size': len(self._buffer),
        }

    def get_recent_events(self, symbol: str = None, max_age_secs: float = 300) -> list:
        """BUG-16 FIX: Public API for recent whale events — replaces direct _buffer access
        in engine.py. Accepts optional symbol filter and recency window.
        Returns a snapshot copy so callers can't accidentally mutate the buffer.
        """
        import time as _t
        now = _t.time()
        return [
            e for e in list(self._buffer)
            if (now - e.timestamp) < max_age_secs
            and (symbol is None or e.symbol == symbol)
        ]

    def get_recent_dominant_side(
        self,
        symbol: str = None,
        max_age_secs: float = 300,
        min_events: int = 2,
        min_dominance_ratio: float = 0.60,
    ) -> str:
        """Return LONG/SHORT when recent whale flow has a clear dominant side.

        Volume is weighted by fill_prob so high-probability walls contribute
        more than far-from-mid walls that are unlikely to be filled.
        """
        recent = self.get_recent_events(symbol=symbol, max_age_secs=max_age_secs)
        if len(recent) < min_events:
            return ""

        buy_vol = sum(e.order_usd * e.fill_prob for e in recent if e.side == "buy")
        sell_vol = sum(e.order_usd * e.fill_prob for e in recent if e.side == "sell")
        total_vol = buy_vol + sell_vol
        if total_vol <= 0:
            return ""

        dominance = max(buy_vol, sell_vol) / total_vol
        if dominance < min_dominance_ratio:
            return ""

        if buy_vol > sell_vol:
            return "LONG"
        if sell_vol > buy_vol:
            return "SHORT"
        return ""


# ── Singleton ──────────────────────────────────────────────
whale_aggregator = WhaleAggregator()
