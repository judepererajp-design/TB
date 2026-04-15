"""
TitanBot Pro — HyperTracker (CMM) Client
==========================================
Provides smart-money intelligence from Hyperliquid's 1.7M tracked wallets.

API base: https://ht-api.coinmarketman.com
Auth:     Authorization: Bearer <token>
Key:      Set in config/settings.yaml → global.hypertracker_api_key

Free tier:  100 req/day
Pulse tier: 50,000 req/month (~1,666/day)

Usage strategy (free tier safe):
  - Cohort summary (Money Printer + Smart Money): called only when a signal
    reaches publication stage, cached 15 minutes per cohort.
  - Watched wallet positions: polled every 30 minutes, cached aggressively.
  - Per-coin OI breakdown: only called for A/A+ signals, cached 10 min per coin.
  - Daily budget at free tier with normal signal volume (5-10 signals/day):
    ~5×2 cohort calls + 1×48 wallet polls + ~5 coin calls ≈ 63 calls/day ✓

Cohort IDs (from HyperTracker legend):
  money_printer = "money_printer"   — consistent winners, highest all-time PnL
  smart_money   = "smart_money"     — strong PnL performers
  grinder       = "grinder"         — high-volume, moderate PnL
  whale         = "whale"           — large position size
  leviathan     = "leviathan"       — largest wallets
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

from config.loader import cfg

logger = logging.getLogger(__name__)

BASE_URL = "https://ht-api.coinmarketman.com"

# Cohort segment IDs (verified from HyperTracker legend page)
COHORT_IDS = {
    # Numeric IDs as per HyperTracker API docs /api/external/segments
    "money_printer": "8",    # PnL > $1M  💰
    "smart_money":   "9",    # PnL $100k–$1M  📈
    "grinder":       "10",   # PnL $10k–$100k  📊
    "small_whale":   "4",    # Size $100k–$500k  🐋
    "whale":         "5",    # Size $500k–$1M  🐳
    "leviathan":     "7",    # Size > $5M  🐉
    "apex_predator": "3",    # Size $50k–$100k  🦈
}

# The two cohorts we weight most heavily in signal decisions
PRIMARY_COHORTS = ["money_printer", "smart_money"]

# Bias string → numeric score mapping (for combining multiple cohorts)
_BIAS_SCORES = {
    "Very Bullish":     90,
    "Bullish":          75,
    "Slightly Bullish": 62,
    "Indecisive":       50,
    "Slightly Bearish": 38,
    "Bearish":          25,
    "Very Bearish":     10,
}


@dataclass
class CohortSnapshot:
    """Snapshot of one cohort's current position state."""
    cohort_id: str
    name: str
    bias: str               # e.g. "Slightly Bearish"
    bias_score: int         # 0-100
    leverage: float         # average leverage
    in_positions_pct: int   # % of cohort wallets currently in a position
    trader_count: int
    top_coins: List[str] = field(default_factory=list)
    fetched_at: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > 900  # 15 min TTL


@dataclass
class WalletPosition:
    """A single open position on a tracked wallet."""
    address: str
    symbol: str
    direction: str   # LONG or SHORT
    size_usd: float
    entry_price: float
    pnl_usd: float
    leverage: float
    dist_to_liq_pct: float
    opened_at: float


@dataclass
class CoinOISnapshot:
    """Open interest breakdown for one coin."""
    symbol: str
    total_oi_usd: float
    long_pct: float
    short_pct: float
    cohort_breakdown: Dict[str, dict] = field(default_factory=dict)
    liq_risk_pct: float = 0.0    # % of OI within 75% of liquidation
    fetched_at: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > 600   # 10 min

    @property
    def is_crowded(self) -> bool:
        """True if dominant side exceeds 70% of OI — crowded trade warning."""
        return max(self.long_pct, self.short_pct) > 70

    @property
    def dominant_side(self) -> str:
        return "LONG" if self.long_pct >= self.short_pct else "SHORT"


class HyperTrackerClient:
    """
    Async HyperTracker API client with aggressive caching to stay within free tier.
    All public methods return cached data immediately; background refresh happens
    lazily on cache miss or staleness.
    """

    def __init__(self):
        self._api_key: str = ""
        self._enabled: bool = False

        # Cohort cache: cohort_id → CohortSnapshot
        self._cohort_cache: Dict[str, CohortSnapshot] = {}
        self._cohort_fetch_lock: Dict[str, asyncio.Lock] = {}

        # Wallet positions cache: address → list of WalletPosition
        self._wallet_cache: Dict[str, List[WalletPosition]] = {}
        self._wallet_last_fetch: float = 0.0
        self._wallet_ttl: float = 1800.0   # 30 min

        # Coin OI cache: symbol → CoinOISnapshot
        self._oi_cache: Dict[str, CoinOISnapshot] = {}

        # Watched wallet addresses (from settings or manually added)
        self._watched_wallets: List[str] = []

        # Breaking position change callback
        self.on_whale_position_change = None   # callback(address, symbol, direction, size_usd)

        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def initialize(self) -> bool:
        """Load API key from config. Returns True if enabled."""
        try:
            key = cfg.global_cfg.get("hypertracker_api_key", "")
            watched = cfg.global_cfg.get("hypertracker_watched_wallets", [])
            if key and key not in ("", "YOUR_KEY_HERE"):
                self._api_key = key
                self._enabled = True
                self._watched_wallets = watched or []
                logger.info(f"🐋 HyperTracker enabled. Watched wallets: {len(self._watched_wallets)}")
                return True
            else:
                logger.info("🐋 HyperTracker disabled — no API key in settings.yaml")
                return False
        except Exception as e:
            logger.warning(f"HyperTracker init error: {e}")
            return False

    def start(self):
        if self._enabled and not self._running:
            self._running = True
            self._task = asyncio.create_task(self._poll_loop())
            logger.info("🐋 HyperTracker background polling started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def _poll_loop(self):
        """Background loop: fetch cohorts on startup, then refresh wallets every 30 min."""
        await asyncio.sleep(5)  # short delay after startup

        # Proactively fetch cohorts on startup so they're cached before dashboard asks
        logger.info("🐋 HyperTracker: pre-fetching cohorts on startup...")
        try:
            ok = await self._fetch_all_cohorts_bias()
            if ok:
                logger.info("🐋 HyperTracker: cohort cache warmed")
            else:
                logger.warning("🐋 HyperTracker: startup cohort fetch returned no data")
        except Exception as e:
            logger.warning(f"🐋 HyperTracker startup cohort fetch failed: {e}")

        while self._running:
            try:
                if self._watched_wallets:
                    await self._refresh_wallet_positions()
                # Re-fetch cohorts every 30 min to keep cache fresh
                await self._fetch_all_cohorts_bias()
                await asyncio.sleep(1800)   # 30 min
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"HyperTracker poll error: {e}")
                await asyncio.sleep(300)

    # ── Internal fetch methods ────────────────────────────────

    async def _get(self, path: str, params: dict = None) -> Optional[dict]:
        if not self._enabled:
            return None
        session = await self._get_session()
        url = f"{BASE_URL}{path}"
        key_hint = self._api_key[:8] + "..." if self._api_key else "EMPTY"
        logger.info(f"🐋 HyperTracker GET {url} (key={key_hint})")
        try:
            async with session.get(url, params=params) as resp:
                logger.info(f"🐋 HyperTracker response: HTTP {resp.status} for {path}")
                if resp.status == 401:
                    logger.warning(f"🐋 HyperTracker: 401 Unauthorized — API key invalid or expired (key={key_hint})")
                    self._enabled = False
                    return None
                if resp.status == 403:
                    logger.warning(f"🐋 HyperTracker: 403 Forbidden — key may lack permissions (key={key_hint})")
                    return None
                if resp.status == 429:
                    logger.warning("🐋 HyperTracker: 429 Rate limit hit")
                    return None
                if resp.status == 404 and "/api/external/" in path:
                    # Free tier (100 req/day) does NOT include /api/external/ endpoints.
                    # Pulse plan ($179/mo) required. Disable to stop log spam.
                    if not getattr(self, '_free_tier_warned', False):
                        logger.warning(
                            "🐋 HyperTracker: Free tier does not include /api/external/ endpoints "
                            "(Smart Money, Cohorts, OI data). "
                            "Upgrade to Pulse plan at app.coinmarketman.com/hypertracker/api — "
                            "disabling HyperTracker to stop log spam."
                        )
                        self._free_tier_warned = True
                    self._enabled = False
                    return None
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(f"🐋 HyperTracker {path}: HTTP {resp.status} — {body[:200]}")
                    return None
                return await resp.json()
        except asyncio.TimeoutError:
            logger.warning(f"🐋 HyperTracker timeout (15s): {path}")
            return None
        except Exception as e:
            logger.warning(f"🐋 HyperTracker request error {path}: {e}")
            return None

    async def _fetch_all_cohorts_bias(self) -> bool:
        """
        Fetch all cohort biases in ONE call using /api/external/segments/bias.
        Returns True if successful. Populates _cohort_cache for all known cohorts.
        NOTE: /api/external/ endpoints require Pulse tier ($179/mo) or higher.
              Free tier (100 req/day) returns 404 on all /api/external/ routes.
        """
        data = await self._get("/api/external/segments/bias")
        if not data:
            return False

        # data is a list of segment bias objects
        if not isinstance(data, list):
            logger.warning(f"🐋 HyperTracker: unexpected segments/bias response type: {type(data)}")
            return False

        # Build reverse map: numeric_id → our cohort_id name
        id_to_name = {str(v): k for k, v in COHORT_IDS.items()}

        parsed = 0
        now = time.time()
        for item in data:
            try:
                seg = item.get("segment") or {}
                seg_id = str(seg.get("id", ""))
                seg_name = seg.get("name", "")
                cohort_id = id_to_name.get(seg_id)
                if not cohort_id:
                    continue  # not one of our tracked cohorts

                # bias_history is list of snapshots, newest first
                bias_list = item.get("bias") or []
                # Use the most recent snapshot
                latest = bias_list[0] if bias_list else {}
                perp_bias = float(latest.get("bias") or 0.0)

                # Map -1..1 to 0..100 score and label
                bias_score = int((perp_bias + 1) / 2 * 100)
                if perp_bias >= 0.5:
                    bias_raw = "Very Bullish"
                elif perp_bias >= 0.2:
                    bias_raw = "Bullish"
                elif perp_bias >= -0.2:
                    bias_raw = "Neutral"
                elif perp_bias >= -0.5:
                    bias_raw = "Bearish"
                else:
                    bias_raw = "Very Bearish"

                leverage = round(float(latest.get("exposureRatio") or 1.0), 2)

                snapshot = CohortSnapshot(
                    cohort_id=cohort_id,
                    name=seg_name or cohort_id.replace("_", " ").title(),
                    bias=bias_raw,
                    bias_score=bias_score,
                    leverage=leverage,
                    in_positions_pct=0,   # not in this endpoint
                    trader_count=0,
                    top_coins=[],
                    fetched_at=now,
                )
                self._cohort_cache[cohort_id] = snapshot
                parsed += 1
                logger.info(f"🐋 Cohort {seg_name}: {bias_raw} (bias={perp_bias:.2f})")

            except Exception as e:
                logger.warning(f"🐋 HyperTracker cohort parse error: {e}")

        logger.info(f"🐋 HyperTracker: parsed {parsed} cohorts from segments/bias")
        return parsed > 0

    async def _fetch_cohort(self, cohort_id: str) -> Optional[CohortSnapshot]:
        """Return cohort from cache (populated by _fetch_all_cohorts_bias)."""
        cached = self._cohort_cache.get(cohort_id)
        if cached and not cached.is_stale:
            return cached
        # If cache is empty/stale, try a full refresh
        await self._fetch_all_cohorts_bias()
        return self._cohort_cache.get(cohort_id)

    async def _fetch_coin_oi(self, coin: str) -> Optional[CoinOISnapshot]:
        """Fetch open interest breakdown for one coin."""
        cached = self._oi_cache.get(coin)
        if cached and not cached.is_stale:
            return cached

        # Use the positions endpoint filtered by coin — get summary stats
        data = await self._get("/api/external/positions", params={
            "coin": coin, "open": "true", "limit": 1
        })
        if not data:
            return None

        # Also fetch liquidation risk for money_printer cohort on this coin
        liq_data = await self._get(
            f"/api/external/money_printer/assets/liquidation-risk",
            params={"limit": 50}
        )

        try:
            summary = (data.get("data") or {})
            total_long_usd  = float(summary.get("total_long_usd") or 0)
            total_short_usd = float(summary.get("total_short_usd") or 0)
            total_oi        = total_long_usd + total_short_usd
            long_pct  = round(total_long_usd  / total_oi * 100, 1) if total_oi else 50.0
            short_pct = round(total_short_usd / total_oi * 100, 1) if total_oi else 50.0

            # Find liquidation risk for this coin
            liq_risk = 0.0
            if liq_data:
                for asset in (liq_data.get("data") or []):
                    if (asset.get("coin") or "").upper() == coin.upper():
                        liq_risk = float(asset.get("liq_risk_pct") or 0)
                        break

            snapshot = CoinOISnapshot(
                symbol=coin,
                total_oi_usd=total_oi,
                long_pct=long_pct,
                short_pct=short_pct,
                liq_risk_pct=liq_risk,
                fetched_at=time.time(),
            )
            self._oi_cache[coin] = snapshot
            return snapshot
        except Exception as e:
            logger.debug(f"HyperTracker coin OI parse ({coin}): {e}")
            return None

    async def _refresh_wallet_positions(self):
        """Refresh positions for all watched wallets. Called every 30 min."""
        if not self._watched_wallets:
            return

        previous: Dict[str, List[WalletPosition]] = dict(self._wallet_cache)
        new_cache: Dict[str, List[WalletPosition]] = {}

        for address in self._watched_wallets:
            data = await self._get("/api/external/positions", params={
                "address": address,
                "open": "true",
                "limit": 20,
            })
            if not data:
                continue
            positions = []
            for pos in (data.get("data") or data.get("results") or []):
                try:
                    direction = "LONG" if float(pos.get("size") or pos.get("amount") or 0) > 0 else "SHORT"
                    wp = WalletPosition(
                        address=address,
                        symbol=(pos.get("coin") or pos.get("symbol") or "").upper(),
                        direction=direction,
                        size_usd=abs(float(pos.get("value") or pos.get("size_usd") or 0)),
                        entry_price=float(pos.get("entryPx") or pos.get("entry_price") or 0),
                        pnl_usd=float(pos.get("unrealizedPnl") or pos.get("pnl") or 0),
                        leverage=float(pos.get("leverage") or 1.0),
                        dist_to_liq_pct=float(pos.get("distToLiq") or pos.get("dist_to_liq_pct") or 0),
                        opened_at=float(pos.get("openedAt") or pos.get("opened_at") or time.time()),
                    )
                    positions.append(wp)
                except Exception:
                    pass
            new_cache[address] = positions

            # Detect new/closed positions and fire callback
            if self.on_whale_position_change and address in previous:
                old_syms = {(p.symbol, p.direction) for p in previous[address]}
                new_syms = {(p.symbol, p.direction) for p in positions}
                for pos in positions:
                    if (pos.symbol, pos.direction) not in old_syms:
                        try:
                            await self.on_whale_position_change(
                                address, pos.symbol, pos.direction, pos.size_usd, "opened"
                            )
                        except Exception:
                            pass
                for old_pos in previous.get(address, []):
                    if (old_pos.symbol, old_pos.direction) not in new_syms:
                        try:
                            await self.on_whale_position_change(
                                address, old_pos.symbol, old_pos.direction, old_pos.size_usd, "closed"
                            )
                        except Exception:
                            pass

        self._wallet_cache = new_cache
        self._wallet_last_fetch = time.time()

    # ── Public API ─────────────────────────────────────────────

    async def get_primary_cohort_bias(self) -> Tuple[Optional[CohortSnapshot], Optional[CohortSnapshot]]:
        """
        Get Money Printer and Smart Money cohort snapshots.
        Called when TitanBot is about to publish a signal.
        Returns (money_printer, smart_money) — either can be None on API failure.
        """
        if not self._enabled:
            return None, None
        mp = await self._fetch_cohort("money_printer")
        sm = await self._fetch_cohort("smart_money")
        return mp, sm

    async def get_all_cohorts(self) -> List[CohortSnapshot]:
        """Fetch all cohorts using /api/external/segments/bias (one call for all)."""
        if not self._enabled:
            return []
        # Try to refresh if cache is empty or stale
        if not self._cohort_cache:
            await self._fetch_all_cohorts_bias()
        return [s for s in self._cohort_cache.values() if isinstance(s, CohortSnapshot)]

    async def get_coin_intelligence(self, symbol: str) -> Optional[CoinOISnapshot]:
        """
        Get OI breakdown + liquidation risk for a coin.
        Only called for A/A+ signals to stay within API budget.
        """
        if not self._enabled:
            return None
        coin = symbol.replace("/USDT", "").replace("/BUSD", "").upper()
        return await self._fetch_coin_oi(coin)

    def get_watched_wallet_positions(self) -> Dict[str, List[WalletPosition]]:
        """Return cached wallet positions (no API call)."""
        return self._wallet_cache

    def format_signal_intel(
        self,
        mp: Optional[CohortSnapshot],
        sm: Optional[CohortSnapshot],
        oi: Optional[CoinOISnapshot],
        signal_direction: str,
    ) -> str:
        """
        Format HyperTracker intel for the Telegram Market panel.
        Returns empty string if no data available.
        """
        lines = []

        if mp or sm:
            lines.append("\n🐋 <b>Smart Money Positioning</b>")
            for snap in [mp, sm]:
                if not snap:
                    continue
                bias_emoji = "🟢" if snap.bias_score >= 60 else ("🔴" if snap.bias_score <= 40 else "⚪")
                lines.append(
                    f"{bias_emoji} <b>{snap.name}</b>: {snap.bias} "
                    f"· {snap.in_positions_pct}% in pos "
                    f"· {snap.leverage}x avg lev"
                )

        if oi:
            # Crowded trade warning
            if oi.is_crowded:
                dom = oi.dominant_side
                dom_pct = oi.long_pct if dom == "LONG" else oi.short_pct
                is_with  = dom == signal_direction
                warn = "⚠️ Crowded trade" if is_with else "✅ Uncrowded"
                lines.append(
                    f"\n{warn}: {dom_pct:.0f}% of HL OI is {dom} "
                    f"({'same side as you' if is_with else 'opposite side'})"
                )

            # Liquidation cluster
            if oi.liq_risk_pct >= 10:
                lines.append(
                    f"💥 {oi.liq_risk_pct:.0f}% of Smart Money OI near liquidation "
                    f"— potential cascade zone"
                )

        # Conflict with signal direction
        if mp and sm:
            mp_against = (mp.bias_score < 45 and signal_direction == "LONG") or \
                         (mp.bias_score > 55 and signal_direction == "SHORT")
            sm_against = (sm.bias_score < 45 and signal_direction == "LONG") or \
                         (sm.bias_score > 55 and signal_direction == "SHORT")
            if mp_against and sm_against:
                lines.append("\n⚠️ Both Money Printer & Smart Money positioned against this trade")
            elif mp_against:
                lines.append("\n⚠️ Money Printer cohort positioned against this trade")

        return "\n".join(lines)

    def format_whales_summary(self) -> str:
        """Format all cohort data for the /whales Telegram command."""
        if not self._enabled:
            return "🐋 <b>HyperTracker</b>\n\nNot configured — add hypertracker_api_key to settings.yaml."
        if not self._cohort_cache:
            return "🐋 <b>Cohort data loading...</b>"

        lines = ["🐋 <b>Hyperliquid Smart Money</b>\n"]
        for cid in ["money_printer", "smart_money", "whale", "leviathan", "grinder"]:
            snap = self._cohort_cache.get(cid)
            if not snap:
                continue
            b_emoji = "🟢" if snap.bias_score >= 60 else ("🔴" if snap.bias_score <= 40 else "⚪")
            lines.append(
                f"{b_emoji} <b>{snap.name}</b>  "
                f"{snap.bias}  ·  {snap.in_positions_pct}% active  "
                f"·  {snap.leverage}x lev"
            )
        lines.append(f"\n<i>Updated: {int((time.time() - min(s.fetched_at for s in self._cohort_cache.values())) / 60)}m ago</i>")
        return "\n".join(lines)

    def format_wallet_positions(self) -> str:
        """Format tracked wallet positions for /whales detail view."""
        if not self._wallet_cache:
            return "No tracked wallets or no open positions."
        lines = ["📍 <b>Tracked Wallets — Open Positions</b>\n"]
        for address, positions in self._wallet_cache.items():
            short_addr = f"{address[:6]}...{address[-4:]}"
            if not positions:
                lines.append(f"<b>{short_addr}</b>: No open positions")
                continue
            lines.append(f"<b>{short_addr}</b>:")
            for p in positions[:5]:
                d_emoji = "🟢" if p.direction == "LONG" else "🔴"
                pnl_str = f"+${p.pnl_usd/1000:.1f}K" if p.pnl_usd >= 0 else f"-${abs(p.pnl_usd)/1000:.1f}K"
                lines.append(
                    f"  {d_emoji} {p.symbol} {p.direction}  "
                    f"${p.size_usd/1e6:.2f}M  "
                    f"PnL: {pnl_str}  "
                    f"Liq: {p.dist_to_liq_pct:.0f}% away"
                )
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────────
hypertracker = HyperTrackerClient()
