"""
TitanBot Pro — Altcoin Rotation Tracker
=========================================
Tracks capital flow between crypto sectors to identify which
sectors are HOT (receiving money) vs COLD (bleeding out).

This is one of the highest-edge tools in crypto trading.
Money rotates in a predictable pattern:
  BTC → ETH → Large caps → Mid caps → Small caps → Meme

Understanding WHERE in this cycle we are and WHICH SECTOR
is currently receiving capital gives massive edge.

Output:
  - Sector scores (HOT/WARM/NEUTRAL/COLD)
  - Rotation phase detection
  - Per-signal sector bonus/penalty
  - Alt season cycle position
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from config.loader import cfg
from data.api_client import api

logger = logging.getLogger(__name__)


class SectorStatus(str, Enum):
    HOT      = "HOT"       # Strong capital inflow, >5% sector avg
    WARM     = "WARM"      # Mild inflow, 1-5%
    NEUTRAL  = "NEUTRAL"   # Flat, -1 to +1%
    COOL     = "COOL"      # Mild outflow, -1 to -3%
    COLD     = "COLD"      # Strong outflow, <-3%


@dataclass
class SectorData:
    name: str
    symbols: List[str] = field(default_factory=list)
    avg_change_24h: float = 0.0
    avg_change_7d: float = 0.0
    # AUDIT FIX: the previous docstring promised "Volume vs 7-day average"
    # but _fetch_sector_data was storing the absolute mean quoteVolume here
    # (a USD figure, not a %), silently mis-typed downstream.  Renaming to
    # an explicit absolute field prevents any future consumer from treating
    # this as a ratio.  Kept as a float so existing consumers see a 0.0
    # default and not a KeyError.
    avg_quote_volume_24h: float = 0.0   # Average 24h quote volume (USD)
    status: SectorStatus = SectorStatus.NEUTRAL
    score: float = 50.0               # 0-100, used for signal adjustment
    momentum: float = 0.0             # Rate of change of score
    leader: Optional[str] = None      # Best performing symbol in sector
    laggard: Optional[str] = None     # Worst performing
    last_updated: float = 0.0


@dataclass
class RotationState:
    """Overall market rotation state"""
    phase: str = "NEUTRAL"              # BTC_SEASON | ALT_ACCUMULATION | ALT_SEASON | MEME_PHASE | RESET
    hot_sectors: List[str] = field(default_factory=list)
    cold_sectors: List[str] = field(default_factory=list)
    rotation_velocity: float = 0.0      # How fast rotation is happening
    btc_dominance_trend: str = "NEUTRAL"  # RISING | FALLING | NEUTRAL
    recommended_focus: List[str] = field(default_factory=list)
    avoid_sectors: List[str] = field(default_factory=list)


class AltcoinRotationTracker:
    """
    Tracks sector rotation across the crypto market.
    Updates every rotation_interval seconds (default: 10 min).
    """

    def __init__(self):
        self._rot_cfg = cfg.analyzers.rotation
        self._sectors: Dict[str, SectorData] = {}
        self._rotation_state = RotationState()
        self._symbol_sector_map: Dict[str, str] = {}
        self._last_update: float = 0
        self._cache_ttl = cfg.cache.get('rotation_ttl', 600)
        self._lock = asyncio.Lock()

        self._build_sector_map()
        # BUG-7 FIX: was 0.0 — caused immediate refresh on first boot which delisted
        # 20 symbols from the freshly-built hardcoded map (73→53 in the first 9 seconds).
        # Initialize to (now - 6 days) so the first refresh happens after 24h of uptime,
        # by which point we have real trade data to decide what's actually delisted.
        self._last_sector_refresh: float = time.time() - (6 * 86400)
        self._sector_refresh_interval: float = 7 * 86400  # weekly refresh

    def _build_sector_map(self):
        """Build reverse mapping: symbol → sector"""
        # Try config first
        sectors_cfg = self._rot_cfg.sectors
        sector_dict = {}

        if hasattr(sectors_cfg, 'to_dict'):
            sector_dict = sectors_cfg.to_dict()
        elif isinstance(sectors_cfg, dict):
            sector_dict = sectors_cfg
        elif hasattr(sectors_cfg, '__dict__'):
            # OmegaConf / DictConfig — iterate attributes
            for key in dir(sectors_cfg):
                if not key.startswith('_'):
                    val = getattr(sectors_cfg, key, None)
                    if isinstance(val, (list, tuple)):
                        sector_dict[key] = list(val)

        # Hardcoded fallback if config parsing fails
        if not sector_dict:
            logger.warning("Sector config empty — using hardcoded fallback")
            sector_dict = {
                "LAYER1":   ["BTC", "ETH", "SOL", "ADA", "AVAX", "DOT", "ATOM", "NEAR", "APT", "SUI"],
                "LAYER2":   ["MATIC", "ARB", "OP", "IMX", "STRK", "ZK", "MANTA"],
                "DEFI":     ["UNI", "AAVE", "CRV", "SUSHI", "COMP", "MKR", "SNX"],
                "AI":       ["FET", "AGIX", "OCEAN", "RENDER", "TAO", "WLD", "ARKM"],
                "GAMING":   ["AXS", "SAND", "MANA", "ENJ", "GALA", "BEAM", "PIXEL"],
                "MEME":     ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF"],
                "RWA":      ["ONDO", "POLYX", "CFG", "TRU"],
                "DEPIN":    ["HNT", "FIL", "AR", "STORJ", "GRT"],
                "EXCHANGE":  ["BNB", "OKB", "CRO"],
                "LSD":      ["LDO", "RPL", "FXS", "ANKR"],
            }

        for sector_name, symbols in sector_dict.items():
            if not isinstance(symbols, (list, tuple)):
                continue
            sym_list = [f"{s}/USDT" for s in symbols]
            self._sectors[sector_name] = SectorData(
                name=sector_name,
                symbols=sym_list,
            )
            for symbol in symbols:
                self._symbol_sector_map[f"{symbol}/USDT"] = sector_name

        logger.info(f"Sector map: {len(self._sectors)} sectors, {len(self._symbol_sector_map)} symbols mapped")

    # ── Public interface ─────────────────────────────────────

    def get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        """Get the sector a symbol belongs to"""
        return self._symbol_sector_map.get(symbol)

    def get_sector_status(self, sector: str) -> Optional[SectorStatus]:
        """Get current status of a sector"""
        if sector in self._sectors:
            return self._sectors[sector].status
        return None

    def get_signal_adjustment(self, symbol: str, direction: str) -> Tuple[float, str]:
        """
        Get confidence adjustment based on sector rotation.
        Returns (adjustment, reason).
        High adjustment for symbols in hot sectors.
        Negative adjustment for cold sectors.
        """
        sector = self.get_sector_for_symbol(symbol)
        if not sector or sector not in self._sectors:
            return 0.0, ""

        sector_data = self._sectors[sector]

        if sector_data.status == SectorStatus.HOT:
            if direction == "LONG":
                return 8.0, f"🔥 {sector} sector in rotation — HOT"
            else:
                return -5.0, f"⚠️ {sector} sector HOT — shorting against rotation"

        elif sector_data.status == SectorStatus.WARM:
            if direction == "LONG":
                return 4.0, f"📈 {sector} sector warming up"
            else:
                return -2.0, ""

        elif sector_data.status == SectorStatus.COOL:
            if direction == "LONG":
                return -4.0, f"📉 {sector} sector cooling — caution on longs"
            else:
                return 3.0, f"✅ {sector} sector cooling — shorts aligned"

        elif sector_data.status == SectorStatus.COLD:
            if direction == "LONG":
                return -10.0, f"❄️ {sector} sector COLD — avoid longs"
            else:
                return 6.0, f"🔴 {sector} sector COLD — short rotation aligned"

        return 0.0, ""

    def get_sector_weight(self, symbol: str, direction: str) -> Tuple[float, str]:
        """
        Get sector rotation MULTIPLIER (0.6 to 1.35) for position confidence.
        Uses momentum (acceleration) not just 24h change.

        Returns: (weight_multiplier, reason)
        """
        sector = self.get_sector_for_symbol(symbol)
        if not sector or sector not in self._sectors:
            return 1.0, ""

        sd = self._sectors[sector]

        # AUDIT FIX: center the mapping on score=50 → weight=1.00 so a
        # NEUTRAL sector imparts no systematic drag.  Previous formula
        # 0.6 + score/100*0.75 made score=50 map to 0.975 (-2.5% silent
        # penalty on every neutral-sector signal).
        # New: 50 → 1.00, 0 → 0.65, 100 → 1.35 (linear on each side).
        if sd.score >= 50:
            weight = 1.0 + (sd.score - 50) / 50.0 * 0.35    # 50→1.00, 100→1.35
        else:
            weight = 1.0 - (50 - sd.score) / 50.0 * 0.35    # 50→1.00, 0→0.65
        weight = round(max(0.65, min(1.35, weight)), 3)

        # Direction alignment penalty/bonus
        if direction == "LONG" and sd.status in (SectorStatus.COLD, SectorStatus.COOL):
            weight *= 0.85  # Penalize longs in cold sectors
        elif direction == "SHORT" and sd.status in (SectorStatus.HOT, SectorStatus.WARM):
            weight *= 0.85  # Penalize shorts in hot sectors
        elif direction == "LONG" and sd.status == SectorStatus.HOT:
            weight *= 1.10  # Boost longs in hot sectors
        elif direction == "SHORT" and sd.status == SectorStatus.COLD:
            weight *= 1.10  # Boost shorts in cold sectors

        weight = round(max(0.5, min(1.45, weight)), 3)

        reason = ""
        if weight >= 1.15:
            reason = f"🔥 {sector} sector rotation tailwind ({weight:.2f}x)"
        elif weight <= 0.75:
            reason = f"❄️ {sector} sector headwind ({weight:.2f}x)"

        return weight, reason

    def get_rotation_summary(self) -> Dict:
        """Get full rotation state for Telegram display"""
        hot = [(s, d) for s, d in self._sectors.items()
               if d.status == SectorStatus.HOT]
        warm = [(s, d) for s, d in self._sectors.items()
                if d.status == SectorStatus.WARM]
        cold = [(s, d) for s, d in self._sectors.items()
                if d.status == SectorStatus.COLD]
        cool = [(s, d) for s, d in self._sectors.items()
                if d.status == SectorStatus.COOL]

        return {
            'phase': self._rotation_state.phase,
            'hot': [(s, d.avg_change_24h) for s, d in hot],
            'warm': [(s, d.avg_change_24h) for s, d in warm],
            'cool': [(s, d.avg_change_24h) for s, d in cool],
            'cold': [(s, d.avg_change_24h) for s, d in cold],
            'recommended_focus': self._rotation_state.recommended_focus,
            'avoid_sectors': self._rotation_state.avoid_sectors,
            'btc_dominance_trend': self._rotation_state.btc_dominance_trend,
        }

    # ── Update cycle ─────────────────────────────────────────

    async def update(self):
        """Refresh sector rotation data"""
        async with self._lock:
            if time.time() - self._last_update < self._cache_ttl:
                return

            try:
                # PHASE 3 FIX (SECTOR-STALE): refresh sector map weekly from exchange
                await self._maybe_refresh_sector_map()

                await self._fetch_sector_data()
                self._classify_sectors()
                self._determine_rotation_phase()
                self._last_update = time.time()

                hot_names = [s for s, d in self._sectors.items()
                             if d.status == SectorStatus.HOT]
                cold_names = [s for s, d in self._sectors.items()
                              if d.status == SectorStatus.COLD]

                logger.info(
                    f"Rotation update — Phase: {self._rotation_state.phase} | "
                    f"HOT: {hot_names} | COLD: {cold_names}"
                )

            except Exception as e:
                logger.error(f"Rotation update error: {e}")

    async def force_update(self):
        """Force refresh bypassing cache"""
        self._last_update = 0
        await self.update()

    async def _fetch_sector_data(self):
        """Fetch ticker data for all sector symbols"""
        # Get all tickers in one call (efficient)
        all_tickers = await api.fetch_tickers()
        if not all_tickers:
            return

        for sector_name, sector_data in self._sectors.items():
            changes_24h = []
            volumes = []
            best_symbol = None
            best_change = -999
            worst_symbol = None
            worst_change = 999

            for symbol in sector_data.symbols:
                ticker = all_tickers.get(symbol)
                if not ticker:
                    continue

                change = float(ticker.get('percentage', 0) or 0)
                volume = float(ticker.get('quoteVolume', 0) or 0)

                changes_24h.append(change)
                volumes.append(volume)

                if change > best_change:
                    best_change = change
                    best_symbol = symbol

                if change < worst_change:
                    worst_change = change
                    worst_symbol = symbol

            if changes_24h:
                sector_data.avg_change_24h = float(np.mean(changes_24h))
                sector_data.avg_quote_volume_24h = float(np.mean(volumes)) if volumes else 0
                sector_data.leader = best_symbol
                sector_data.laggard = worst_symbol
                sector_data.last_updated = time.time()

    def _classify_sectors(self):
        """Classify each sector's status based on performance + momentum"""
        hot_threshold = cfg.analyzers.rotation.get('hot_sector_gain', 5.0)
        cold_threshold = cfg.analyzers.rotation.get('cold_sector_loss', -3.0)
        # AUDIT FIX (HOT/WARM hysteresis): without an exit band, a sector
        # oscillating around the 5 % boundary flip-flops between HOT and WARM
        # each refresh, causing rotation-weight thrashing in sideways markets.
        # Add an asymmetric exit band so HOT only drops back to WARM once the
        # sector cools ~1 pp below the hot threshold.
        hot_exit_threshold = max(1.0, hot_threshold - 1.0)

        for sector_name, sector_data in self._sectors.items():
            change = sector_data.avg_change_24h
            was_hot = sector_data.status == SectorStatus.HOT

            # Calculate momentum (acceleration): how fast is the score changing?
            old_score = sector_data.score
            if change >= hot_threshold:
                sector_data.status = SectorStatus.HOT
                sector_data.score = min(100, 70 + (change - hot_threshold) * 3)
            elif was_hot and change >= hot_exit_threshold:
                # Hysteresis: stay HOT until we fall clearly below the entry
                # band.  Score from a reduced anchor (hot_exit_threshold) so a
                # sector sitting in the hysteresis band doesn't end up scoring
                # LOWER than at its HOT entry point (which would happen if we
                # still subtracted hot_threshold here — change-hot_threshold
                # goes negative in the band).
                sector_data.status = SectorStatus.HOT
                sector_data.score = min(100, 70 + (change - hot_exit_threshold) * 3)
            elif change >= 1.0:
                sector_data.status = SectorStatus.WARM
                # BUG-2 FIX: cap at 69.9 — without cap, WARM near the HOT threshold
                # (e.g. change=4.9%) scores 75.6, HIGHER than HOT (change=5.0%) at 70.0.
                # That inverts the weight order: WARM sectors get more rotation weight
                # than HOT sectors near the boundary.
                sector_data.score = min(69.9, 60 + (change - 1.0) * 4)
            elif change >= cold_threshold:
                sector_data.status = SectorStatus.NEUTRAL if change >= 0 else SectorStatus.COOL
                sector_data.score = 50 + change * 3
            else:
                sector_data.status = SectorStatus.COLD
                sector_data.score = max(0, 30 + (change - cold_threshold) * 2)

            # Momentum = change in score (positive = accelerating into sector)
            sector_data.momentum = sector_data.score - old_score

    def _determine_rotation_phase(self):
        """
        Determine the current market rotation phase.

        Typical crypto rotation cycle:
        1. BTC_SEASON — BTC outperforms, dominance rising
        2. ETH_SEASON — ETH follows BTC breakout
        3. ALT_ACCUMULATION — Smart money positioning in alts
        4. ALT_SEASON — Alts exploding across the board
        5. MEME_PHASE — Late-cycle mania (memes pumping)
        6. RESET — Correction, back to BTC safety
        """
        hot_sectors = [s for s, d in self._sectors.items()
                       if d.status in (SectorStatus.HOT, SectorStatus.WARM)]
        cold_sectors = [s for s, d in self._sectors.items()
                        if d.status in (SectorStatus.COLD, SectorStatus.COOL)]

        self._rotation_state.hot_sectors = hot_sectors
        self._rotation_state.cold_sectors = cold_sectors

        # Determine phase
        meme_hot = 'MEME' in hot_sectors
        layer1_hot = 'LAYER1' in hot_sectors
        layer2_hot = 'LAYER2' in hot_sectors
        defi_hot = 'DEFI' in hot_sectors
        ai_hot = 'AI' in hot_sectors

        n_hot = len(hot_sectors)

        if meme_hot and n_hot >= 4:
            phase = "MEME_PHASE"      # Late-cycle mania
        elif layer2_hot and defi_hot and n_hot >= 3:
            phase = "ALT_SEASON"      # Peak alt season
        elif (layer1_hot or layer2_hot) and not meme_hot:
            phase = "ALT_ACCUMULATION"
        elif len(cold_sectors) >= 6 and not layer1_hot:
            phase = "BTC_SEASON"      # Everything cold except BTC
        elif n_hot == 0:
            phase = "RESET"
        else:
            phase = "NEUTRAL"

        self._rotation_state.phase = phase

        # Recommendations
        focus = []
        avoid = []

        if phase == "ALT_SEASON":
            focus = [s for s in hot_sectors if s != 'MEME']
            avoid = cold_sectors
        elif phase == "MEME_PHASE":
            focus = ['MEME']
            avoid = [s for s in cold_sectors]
        elif phase == "BTC_SEASON":
            focus = ['LAYER1']
            avoid = list(self._sectors.keys())  # Avoid most alts
        elif phase == "ALT_ACCUMULATION":
            focus = ['LAYER1', 'LAYER2', 'AI']
            avoid = ['MEME']
        else:
            focus = hot_sectors[:3]
            avoid = cold_sectors

        self._rotation_state.recommended_focus = focus
        self._rotation_state.avoid_sectors = avoid


    async def _maybe_refresh_sector_map(self):
        """
        PHASE 3 FIX (SECTOR-STALE): Refresh the sector symbol map weekly.
        Fetches active perpetual futures from Binance and maps any new symbols
        to their sector using their base currency. New listings that match known
        sectors get added; symbols no longer trading get cleaned up.
        """
        if time.time() - self._last_sector_refresh < self._sector_refresh_interval:
            return

        try:
            logger.info("Refreshing sector map from exchange listings...")
            markets = await api.fetch_markets()
            if not markets:
                return

            # Build set of active USDT perpetual symbols
            # FIX ROTATION-1: Binance futures symbols come back as "SOL/USDT:USDT"
            # (linear perpetuals use the ":USDT" settlement suffix in ccxt). Our sector
            # map stores them as "SOL/USDT". Without normalization, every symbol looks
            # "delisted" and the map shrinks to ~1 entry after the first weekly refresh.
            #
            # V17 FIX: fetch_markets() returns a dict keyed by symbol string,
            # not a list. Iterating it directly yields keys (strings), not market dicts.
            # Use .values() to get the actual market info dicts.
            active_symbols = set()
            _markets_iter = markets.values() if isinstance(markets, dict) else markets
            for market in _markets_iter:
                if (isinstance(market, dict) and
                        market.get('quote') == 'USDT' and
                        market.get('active', True) and
                        market.get('type') in ('swap', 'future', 'spot')):
                    symbol = market.get('symbol', '')
                    if symbol:
                        # Normalize: strip settlement suffix so "SOL/USDT:USDT" → "SOL/USDT"
                        normalized = symbol.split(':')[0] if ':' in symbol else symbol
                        active_symbols.add(normalized)
                        active_symbols.add(symbol)  # also keep original for spot markets

            # Map new symbols that match known sector bases
            # Build reverse lookup: base → sector
            _base_to_sector = {}
            for sector_name, sector_data in self._sectors.items():
                for sym in sector_data.symbols:
                    base = sym.replace('/USDT', '')
                    _base_to_sector[base] = sector_name

            added = 0
            for active_sym in active_symbols:
                if active_sym in self._symbol_sector_map:
                    continue  # Already mapped
                base = active_sym.replace('/USDT', '')
                sector = _base_to_sector.get(base)
                if sector and sector in self._sectors:
                    self._symbol_sector_map[active_sym] = sector
                    if active_sym not in self._sectors[sector].symbols:
                        self._sectors[sector].symbols.append(active_sym)
                    added += 1

            # Remove delisted symbols (those in our map but no longer active)
            removed = 0
            for sym in list(self._symbol_sector_map.keys()):
                if sym not in active_symbols and sym != 'BTC/USDT':  # never remove BTC
                    del self._symbol_sector_map[sym]
                    removed += 1

            self._last_sector_refresh = time.time()
            logger.info(
                f"Sector map refreshed: +{added} new / -{removed} delisted | "
                f"{len(self._symbol_sector_map)} total symbols mapped"
            )

        except Exception as e:
            logger.warning(f"Sector map refresh failed (non-fatal): {e}")
            self._last_sector_refresh = time.time()  # Back off for a week even on error

# ── Singleton ──────────────────────────────────────────────
rotation_tracker = AltcoinRotationTracker()
