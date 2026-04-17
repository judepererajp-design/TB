"""
TitanBot Pro — Hyperliquid Smart Money Tracker
===============================================
Tracks the top traders on Hyperliquid and aggregates their
open positions to generate a free "smart money" directional
signal for each coin.

No API keys required. Uses Hyperliquid's public API:
  https://api.hyperliquid.xyz/info

How it works:
  1. Fetch top 30 traders by all-time PnL from Hyperliquid stats API
  2. Query each wallet's current open positions (clearinghouseState)
  3. Aggregate per coin: how many top traders are long vs short
  4. Generate directional bias per coin

Bot integration:
  - Signal gets +6 confidence if top traders align with direction
  - Signal gets -6 confidence if top traders are strongly opposed
  - Dashboard Smart Money section populated (replaces HyperTracker)
  - Telegram signal cards show "🧠 Smart Money: 70% LONG (14/20 wallets)"

Refresh: every 30 minutes (respects public API rate limits)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

_HL_INFO   = "https://api.hyperliquid.xyz/info"
_HL_STATS  = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

_REFRESH_INTERVAL = 1800  # 30 min
_TOP_N_TRADERS    = 25    # how many top wallets to track
_MIN_ALIGN_PCT    = 0.60  # 60%+ alignment = significant signal


@dataclass
class WalletPosition:
    """One position held by a top trader."""
    address: str
    coin:    str
    side:    str    # "LONG" or "SHORT"
    size_usd: float
    pnl:     float  # all-time PnL for ranking


@dataclass
class CoinBias:
    """Aggregated smart money direction for one coin."""
    coin:       str
    long_count: int   = 0  # number of top wallets long
    short_count: int  = 0  # number of top wallets short
    long_usd:   float = 0.0
    short_usd:  float = 0.0
    total_wallets: int = 0
    fetched_at: float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > 3600  # 1 hour

    @property
    def total_count(self) -> int:
        return self.long_count + self.short_count

    @property
    def long_pct(self) -> float:
        t = self.total_count
        return self.long_count / t if t > 0 else 0.5

    @property
    def direction(self) -> str:
        if self.total_count < 3:
            return "NEUTRAL"  # not enough data
        lp = self.long_pct
        if lp >= 0.70: return "STRONG_LONG"
        if lp >= 0.55: return "LEAN_LONG"
        if lp <= 0.30: return "STRONG_SHORT"
        if lp <= 0.45: return "LEAN_SHORT"
        return "NEUTRAL"

    @property
    def bias_label(self) -> str:
        d = self.direction
        labels = {
            "STRONG_LONG":  "📈 Strong Long",
            "LEAN_LONG":    "🔼 Lean Long",
            "STRONG_SHORT": "📉 Strong Short",
            "LEAN_SHORT":   "🔽 Lean Short",
            "NEUTRAL":      "⚖️ Neutral",
        }
        return labels.get(d, "—")

    def confidence_delta(self, signal_direction: str) -> Tuple[int, str]:
        """
        Returns (delta, note) for engine confidence adjustment.
        """
        if self.total_count < 3:
            return 0, ""  # not enough wallets tracking this coin

        lp = self.long_pct
        aligned   = self.long_count  if signal_direction == "LONG" else self.short_count
        opposed   = self.short_count if signal_direction == "LONG" else self.long_count
        total     = self.total_count

        if lp >= _MIN_ALIGN_PCT and signal_direction == "LONG":
            delta = 6
            note  = (f"🧠 Smart Money: {lp:.0%} long "
                     f"({aligned}/{total} top wallets)")
        elif lp <= (1 - _MIN_ALIGN_PCT) and signal_direction == "SHORT":
            delta = 6
            note  = (f"🧠 Smart Money: {1-lp:.0%} short "
                     f"({opposed}/{total} top wallets)")
        elif lp >= _MIN_ALIGN_PCT and signal_direction == "SHORT":
            delta = -6
            note  = (f"⚠️ Smart Money OPPOSES short: "
                     f"{lp:.0%} long ({aligned}/{total})")
        elif lp <= (1 - _MIN_ALIGN_PCT) and signal_direction == "LONG":
            delta = -6
            note  = (f"⚠️ Smart Money OPPOSES long: "
                     f"{1-lp:.0%} short ({opposed}/{total})")
        else:
            delta = 0
            note  = (f"🧠 Smart Money mixed: {lp:.0%} long "
                     f"({total} wallets)")

        return delta, note


class SmartMoneyClient:
    """
    Tracks top Hyperliquid traders' positions.
    Provides free smart money intelligence for signal decisions.
    """

    def __init__(self):
        self._coin_bias:   Dict[str, CoinBias] = {}
        self._top_wallets: List[dict] = []
        self._session:     Optional[aiohttp.ClientSession] = None
        self._running      = False
        self._task:        Optional[asyncio.Task] = None
        self._last_refresh = 0.0
        self._wallet_count = 0

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._main_loop())
            logger.info(
                "🧠 SmartMoneyClient started "
                "(Hyperliquid top traders — free public API)"
            )

    async def stop(self):
        self._running = False
        # AUDIT FIX: await the cancelled task before closing the aiohttp
        # session so pending Hyperliquid API calls unwind cleanly.
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._session and not self._session.closed:
            await self._session.close()

    def get_bias(self, coin: str) -> Optional[CoinBias]:
        """Get smart money bias for a coin. None if no data."""
        b = self._coin_bias.get(coin.upper())
        return b if b and not b.is_stale else None

    def get_all_biases(self) -> Dict[str, CoinBias]:
        return {k: v for k, v in self._coin_bias.items() if not v.is_stale}

    def get_signal_intel(self, symbol: str, direction: str) -> Tuple[int, str]:
        """Main interface for engine. Returns (confidence_delta, note)."""
        coin = symbol.replace("/USDT", "").replace("/BUSD", "").upper()
        bias = self.get_bias(coin)
        if not bias:
            return 0, ""
        return bias.confidence_delta(direction)

    @property
    def is_ready(self) -> bool:
        return bool(self._coin_bias) and self._wallet_count > 0

    @property
    def wallet_count(self) -> int:
        return self._wallet_count

    # ── Session helpers ───────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                headers={"User-Agent": "TitanBot/3.0"},
            )
        return self._session

    async def _post(self, url: str, payload: dict) -> Optional[dict]:
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            pass
        return None

    async def _get(self, url: str) -> Optional[dict]:
        try:
            session = await self._get_session()
            async with session.get(url) as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            pass
        return None

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _main_loop(self):
        await asyncio.sleep(30)  # let bot warm up
        while self._running:
            try:
                await self._refresh()
                await asyncio.sleep(_REFRESH_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"SmartMoneyClient loop error: {e}")
                await asyncio.sleep(300)

    async def _refresh(self):
        """Full refresh: get leaderboard → query positions → aggregate."""
        logger.info("🧠 SmartMoney: fetching top trader positions...")

        # Step 1: get top wallet addresses
        wallets = await self._get_top_wallets()
        if not wallets:
            logger.warning("🧠 SmartMoney: no leaderboard data available")
            return

        # Step 2: query each wallet's open positions (staggered)
        all_positions: List[WalletPosition] = []
        for i, wallet in enumerate(wallets[:_TOP_N_TRADERS]):
            try:
                positions = await self._get_wallet_positions(wallet)
                all_positions.extend(positions)
            except Exception as e:
                logger.warning(f"SmartMoney wallet query error: {e}")
            if i % 5 == 4:
                await asyncio.sleep(1)  # brief pause every 5 wallets

        # Step 3: aggregate per coin
        if all_positions:
            self._aggregate(all_positions)
            self._wallet_count = len(wallets[:_TOP_N_TRADERS])
            self._last_refresh = time.time()
            logger.info(
                f"🧠 SmartMoney: {self._wallet_count} wallets tracked, "
                f"{len(self._coin_bias)} coins with positions"
            )
        else:
            logger.warning("🧠 SmartMoney: no positions found")

    async def _get_top_wallets(self) -> List[dict]:
        """
        Fetch top traders from Hyperliquid stats leaderboard.
        Falls back to direct info API if stats API unavailable.
        """
        wallets = []

        # Try stats API first
        try:
            data = await self._get(_HL_STATS)
            if data and isinstance(data, dict):
                leaderboard = data.get("leaderboardRows", [])
                # ── Recency-weighted composite score ─────────────────────────────
                # Problem: ranking by all-time PnL gives top weight to traders
                # who made a killing in 2021 but may not be relevant today.
                # Solution: composite score = 0.4×(30d PnL) + 0.3×(7d PnL) + 0.3×(30d ROI)
                # This balances recent performance with sustained edge.
                scored_wallets = []
                for row in leaderboard:
                    addr = row.get("ethAddress", "")
                    if not addr:
                        continue
                    perfs = {p[0]: p[1] for p in row.get("windowPerformances", [])
                             if isinstance(p, list) and len(p) == 2}
                    pnl_30d  = float((perfs.get("month", {}) or {}).get("pnl", 0) or 0)
                    pnl_7d   = float((perfs.get("week", {}) or {}).get("pnl", 0) or 0)
                    roi_30d  = float((perfs.get("month", {}) or {}).get("roi", 0) or 0)
                    pnl_all  = float((perfs.get("allTime", {}) or {}).get("pnl", 0) or 0)
                    # Composite: recent PnL matters more than all-time
                    score = 0.4 * pnl_30d + 0.3 * pnl_7d + 0.3 * roi_30d * 10000
                    # Require positive 30d performance (filter out dormant whales)
                    if pnl_30d > 0 or pnl_all > 500_000:
                        scored_wallets.append({"address": addr, "pnl": pnl_all, "score": score})
                scored_wallets.sort(key=lambda x: x["score"], reverse=True)
                wallets = scored_wallets[:_TOP_N_TRADERS]
                if wallets:
                    logger.debug(f"🧠 Leaderboard: {len(wallets)} wallets (recency-weighted)")
                    return wallets

                # Legacy fallback: raw all-time sort
                for row in leaderboard[:_TOP_N_TRADERS]:
                    addr = row.get("ethAddress", "")
                    # windowPerformances is a list of [windowName, {pnl, roi, vlm}] pairs
                    pnl = 0.0
                    for perf in row.get("windowPerformances", []):
                        if isinstance(perf, list) and len(perf) >= 2 and perf[0] == "allTime":
                            try:
                                pnl = float(perf[1].get("pnl", 0) or 0)
                            except (ValueError, TypeError):
                                pass
                        elif isinstance(perf, dict) and perf.get("window") == "allTime":
                            try:
                                pnl = float(perf.get("pnl", 0) or 0)
                            except (ValueError, TypeError):
                                pass
                    if addr:
                        wallets.append({"address": addr, "pnl": pnl})
                if wallets:
                    logger.debug(f"🧠 Leaderboard: {len(wallets)} wallets from stats API")
                    return wallets
        except Exception as e:
            logger.warning(f"SmartMoney stats API error: {e}")

        # Fallback: try direct Hyperliquid info leaderboard
        try:
            data = await self._post(_HL_INFO, {"type": "leaderboard"})
            if data and isinstance(data, list):
                for row in data[:_TOP_N_TRADERS]:
                    addr = row.get("user", row.get("address", ""))
                    pnl  = float(row.get("pnl", row.get("allTimePnl", 0)) or 0)
                    if addr:
                        wallets.append({"address": addr, "pnl": pnl})
                if wallets:
                    logger.debug(f"🧠 Leaderboard: {len(wallets)} wallets from info API")
                    return wallets
        except Exception as e:
            logger.warning(f"SmartMoney leaderboard fallback error: {e}")

        # Last fallback: use known high-PnL Hyperliquid addresses
        # These are publicly visible on the Hyperliquid leaderboard
        known_whales = [
            "0x563c175e6f11582f65d6d9e360a6614d1ed8b71a",
            "0xd20bf4e5cd7df1c88c7ce8e44e949a5c5b00db87",
            "0x44a99e138a3e85b1fcae3d8e34d4820f793dcfdf",
            "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7",
            "0x851b764deb6eb59d00e7abe5ea61d1c4fb76b59d",
            "0x0ab7a75843e3ab0168b3e5bc4b2024b10c79de8a",
            "0x8cb17b3a3f24c2d9acf30d9e40caf26fe09f6a29",
            "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",
            "0xa5ca62d95d24a4a350983d5b8ac4eb8638887396",
            "0x4cc5e2cd6a0e2400caac23a12a5f6aef6efe6337",
        ]
        for addr in known_whales:
            wallets.append({"address": addr, "pnl": 0})
        logger.debug(f"🧠 Using {len(wallets)} known whale addresses as fallback")
        return wallets

    async def _get_wallet_positions(self, wallet: dict) -> List[WalletPosition]:
        """Query a single wallet's open perpetual positions."""
        positions = []
        address = wallet.get("address", "")
        pnl     = wallet.get("pnl", 0)

        if not address:
            return positions

        data = await self._post(_HL_INFO, {
            "type": "clearinghouseState",
            "user": address
        })
        if not data:
            return positions

        try:
            asset_positions = data.get("assetPositions", [])
            for ap in asset_positions:
                pos = ap.get("position", {})
                coin = pos.get("coin", "")
                if not coin:
                    continue
                szi   = float(pos.get("szi", 0) or 0)  # positive=long, negative=short
                px    = float(pos.get("entryPx", 0) or 0)
                if szi == 0 or px == 0:
                    continue
                side    = "LONG" if szi > 0 else "SHORT"
                size_usd = abs(szi) * px
                positions.append(WalletPosition(
                    address  = address,
                    coin     = coin.upper(),
                    side     = side,
                    size_usd = size_usd,
                    pnl      = pnl,
                ))
        except Exception as e:
            logger.debug(f"SmartMoney position parse error {address[:8]}: {e}")

        return positions

    def _aggregate(self, positions: List[WalletPosition]):
        """Build per-coin bias from all collected positions."""
        from collections import defaultdict
        coin_data: Dict[str, dict] = defaultdict(lambda: {
            "long_count": 0, "short_count": 0,
            "long_usd": 0.0,  "short_usd": 0.0,
            "wallets": set()
        })

        for pos in positions:
            coin = pos.coin.upper()
            cd   = coin_data[coin]
            cd["wallets"].add(pos.address)
            if pos.side == "LONG":
                cd["long_count"] += 1
                cd["long_usd"]   += pos.size_usd
            else:
                cd["short_count"] += 1
                cd["short_usd"]   += pos.size_usd

        now = time.time()
        new_biases: Dict[str, CoinBias] = {}
        for coin, cd in coin_data.items():
            bias = CoinBias(
                coin         = coin,
                long_count   = cd["long_count"],
                short_count  = cd["short_count"],
                long_usd     = cd["long_usd"],
                short_usd    = cd["short_usd"],
                total_wallets= len(cd["wallets"]),
                fetched_at   = now,
            )
            new_biases[coin] = bias

        self._coin_bias = new_biases

    def get_summary_for_dashboard(self) -> List[dict]:
        """Returns smart money data sorted by total exposure (for dashboard)."""
        biases = self.get_all_biases()
        sorted_coins = sorted(
            biases.values(),
            key=lambda b: b.long_usd + b.short_usd,
            reverse=True
        )
        result = []
        for bias in sorted_coins[:20]:
            result.append({
                "coin":          bias.coin,
                "direction":     bias.direction,
                "bias_label":    bias.bias_label,
                "long_pct":      round(bias.long_pct * 100, 1),
                "long_count":    bias.long_count,
                "short_count":   bias.short_count,
                "total_count":   bias.total_count,
                "total_wallets": bias.total_wallets,
                "long_usd":      bias.long_usd,
                "short_usd":     bias.short_usd,
            })
        return result


# ── Singleton ──────────────────────────────────────────────────────────────────
smart_money = SmartMoneyClient()
