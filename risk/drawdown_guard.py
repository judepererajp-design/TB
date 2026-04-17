"""
TitanBot Pro — Drawdown Reduction Module
==========================================
Implements four key drawdown reduction mechanisms:

1. Correlation filter — blocks signals when 3+ open positions in same sector
2. Regime transition rapid exit — tightens stops on regime shift
3. Compression warning — alerts range strategies when compression imminent
4. Position heat management — reduces sizing on equity drawdown

Usage:
    from risk.drawdown_guard import drawdown_guard
    # Check before opening a new position:
    if drawdown_guard.should_block_correlated(symbol, sector, open_positions):
        skip signal

    # On regime change:
    new_stops = drawdown_guard.tighten_stops_on_transition(
        old_regime, new_regime, positions
    )

    # Check equity heat:
    size_mult = drawdown_guard.get_heat_multiplier(current_equity, daily_peak)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Sector mappings for crypto assets ────────────────────────────────────
# Common sector groupings for correlation filtering
SECTOR_MAP: Dict[str, str] = {
    # Layer 1
    "BTC/USDT": "btc", "ETH/USDT": "eth",
    "SOL/USDT": "l1", "AVAX/USDT": "l1", "ADA/USDT": "l1",
    "DOT/USDT": "l1", "ATOM/USDT": "l1", "NEAR/USDT": "l1",
    "APT/USDT": "l1", "SUI/USDT": "l1", "SEI/USDT": "l1",
    "ICP/USDT": "l1", "FTM/USDT": "l1", "TIA/USDT": "l1",
    # Layer 2
    "ARB/USDT": "l2", "OP/USDT": "l2", "MATIC/USDT": "l2",
    "STRK/USDT": "l2", "MANTA/USDT": "l2", "METIS/USDT": "l2",
    # DeFi
    "UNI/USDT": "defi", "AAVE/USDT": "defi", "MKR/USDT": "defi",
    "COMP/USDT": "defi", "CRV/USDT": "defi", "SUSHI/USDT": "defi",
    "DYDX/USDT": "defi", "SNX/USDT": "defi", "LINK/USDT": "oracle",
    # AI / Compute
    "FET/USDT": "ai", "AGIX/USDT": "ai", "RNDR/USDT": "ai",
    "TAO/USDT": "ai", "WLD/USDT": "ai", "OCEAN/USDT": "ai",
    # Meme
    "DOGE/USDT": "meme", "SHIB/USDT": "meme", "PEPE/USDT": "meme",
    "FLOKI/USDT": "meme", "BONK/USDT": "meme", "WIF/USDT": "meme",
    # Gaming
    "AXS/USDT": "gaming", "SAND/USDT": "gaming", "MANA/USDT": "gaming",
    "GALA/USDT": "gaming", "IMX/USDT": "gaming",
}


def get_sector(symbol: str) -> str:
    """Get sector for a symbol. Falls back to 'other' if unknown."""
    return SECTOR_MAP.get(symbol, "other")


@dataclass
class DrawdownGuardState:
    """Internal state for tracking drawdown metrics."""
    daily_peak_equity: float = 0.0
    last_regime: str = "UNKNOWN"
    regime_change_ts: float = 0.0
    open_sectors: Dict[str, int] = field(default_factory=dict)  # sector -> count


class DrawdownGuard:
    """
    Multi-layered drawdown reduction system.
    """

    # Maximum same-sector positions before blocking
    MAX_SECTOR_POSITIONS = 3

    # Equity drawdown thresholds for position heat management
    HEAT_LEVELS = [
        (0.03, 0.75),   # 3% drawdown → 75% sizing
        (0.05, 0.50),   # 5% drawdown → 50% sizing
        (0.08, 0.25),   # 8% drawdown → 25% sizing
        (0.10, 0.0),    # 10% drawdown → stop trading
    ]

    # Stop tightening factors by regime transition
    STOP_TIGHTEN_MAP = {
        # (old_regime, new_regime) → tightening factor (multiply current stop distance)
        ("BULL_TREND", "VOLATILE"):      0.60,  # Trend → volatile: tighten hard
        ("BULL_TREND", "CHOPPY"):        0.70,  # Trend → chop: tighten medium
        ("BEAR_TREND", "VOLATILE"):      0.60,
        ("BEAR_TREND", "CHOPPY"):        0.70,
        ("VOLATILE", "VOLATILE_PANIC"):  0.40,  # Volatile → panic: emergency tighten
        ("BULL_TREND", "BEAR_TREND"):    0.50,  # Trend reversal: tighten maximum
        ("BEAR_TREND", "BULL_TREND"):    0.50,
    }

    def __init__(self):
        self._state = DrawdownGuardState()
        self._lock_time = 0.0

    # ── 1. Correlation Filter ──────────────────────────────────────────

    def should_block_correlated(
        self,
        symbol: str,
        open_positions: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        Check if opening a new position would create excessive sector concentration.

        Args:
            symbol: Symbol to check (e.g., "SOL/USDT")
            open_positions: List of open position dicts with 'symbol' key

        Returns:
            (should_block: bool, reason: str)
        """
        target_sector = get_sector(symbol)

        if target_sector in ("btc", "eth", "other"):
            # BTC/ETH are standalone sectors — don't block them
            return False, ""

        same_sector_count = 0
        same_sector_symbols = []
        for pos in open_positions:
            pos_symbol = pos.get("symbol", "")
            if get_sector(pos_symbol) == target_sector:
                same_sector_count += 1
                same_sector_symbols.append(pos_symbol)

        if same_sector_count >= self.MAX_SECTOR_POSITIONS:
            reason = (
                f"🚫 Sector concentration: {same_sector_count} open {target_sector} "
                f"positions ({', '.join(same_sector_symbols[:3])}). "
                f"Max {self.MAX_SECTOR_POSITIONS} per sector."
            )
            return True, reason

        return False, ""

    def get_sector_sizing_penalty(
        self,
        symbol: str,
        open_positions: List[Dict[str, Any]],
    ) -> float:
        """
        Returns a size multiplier (0.5-1.0) based on sector concentration.
        Even if not blocked, having 2 same-sector positions reduces size.
        """
        target_sector = get_sector(symbol)
        if target_sector in ("btc", "eth", "other"):
            return 1.0

        count = sum(
            1 for pos in open_positions
            if get_sector(pos.get("symbol", "")) == target_sector
        )

        if count >= 2:
            return 0.60  # Already 2 in sector → reduce to 60%
        elif count >= 1:
            return 0.80  # Already 1 → reduce to 80%
        return 1.0

    # ── 2. Regime Transition Rapid Exit ────────────────────────────────

    def get_stop_tightening_factor(
        self,
        old_regime: str,
        new_regime: str,
    ) -> float:
        """
        When regime shifts mid-position, returns a factor to tighten stops.

        Returns:
            float: Multiply current stop distance by this factor (1.0 = no change).
                   Values < 1.0 mean tighten the stop.
        """
        key = (old_regime, new_regime)
        factor = self.STOP_TIGHTEN_MAP.get(key, 1.0)

        if factor < 1.0:
            self._state.regime_change_ts = time.time()
            self._state.last_regime = new_regime
            logger.info(
                f"🛡️ Regime transition {old_regime}→{new_regime}: "
                f"tighten stops by {(1-factor)*100:.0f}%"
            )

        return factor

    def compute_tightened_stops(
        self,
        positions: List[Dict[str, Any]],
        old_regime: str,
        new_regime: str,
    ) -> List[Dict[str, Any]]:
        """
        Calculate new stop-loss levels for all open positions after regime shift.

        Each position dict should have:
          - symbol, direction, entry_price, stop_loss, current_price

        Returns list of dicts with 'symbol' and 'new_stop_loss'.
        """
        factor = self.get_stop_tightening_factor(old_regime, new_regime)
        if factor >= 1.0:
            return []  # No tightening needed

        results = []
        for pos in positions:
            entry = pos.get("entry_price", 0)
            current_sl = pos.get("stop_loss", 0)
            current_price = pos.get("current_price", entry)
            direction = pos.get("direction", "LONG")

            if direction == "LONG":
                # Current stop distance = entry - SL
                stop_dist = entry - current_sl
                new_stop_dist = stop_dist * factor
                new_sl = entry - new_stop_dist
                # Never move stop below current price (break-even minimum)
                new_sl = min(new_sl, current_price * 0.995)
                # Never move stop backwards (less protective)
                new_sl = max(new_sl, current_sl)
            else:
                stop_dist = current_sl - entry
                new_stop_dist = stop_dist * factor
                new_sl = entry + new_stop_dist
                new_sl = max(new_sl, current_price * 1.005)
                new_sl = min(new_sl, current_sl)

            if new_sl != current_sl:
                results.append({
                    "symbol": pos.get("symbol", ""),
                    "new_stop_loss": round(new_sl, 8),
                    "old_stop_loss": current_sl,
                    "tighten_factor": factor,
                    "reason": f"Regime {old_regime}→{new_regime}",
                })

        return results

    # ── 3. Compression Warning ─────────────────────────────────────────

    @staticmethod
    def check_compression_warning(
        bb_squeeze_data: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Check if BB squeeze data indicates imminent compression breakout.
        Range strategies should exit or reduce when this fires.

        Args:
            bb_squeeze_data: Result from BaseStrategy.detect_bb_squeeze()

        Returns:
            (is_warning: bool, warning_message: str)
        """
        if not bb_squeeze_data:
            return False, ""

        is_squeeze = bb_squeeze_data.get("is_squeeze", False)
        comp_bars = bb_squeeze_data.get("compression_bars", 0)
        bw_pctile = bb_squeeze_data.get("bandwidth_pctile", 0.5)

        if is_squeeze and comp_bars >= 6:
            return True, (
                f"⚠️ COMPRESSION WARNING: BB squeeze active for {comp_bars} bars "
                f"(bandwidth at {bw_pctile:.0%} percentile). "
                f"Range breakout imminent — reduce range positions."
            )

        if bw_pctile < 0.10 and comp_bars >= 3:
            return True, (
                f"⚠️ COMPRESSION WARNING: Bandwidth at {bw_pctile:.0%} percentile, "
                f"{comp_bars} bars compressed. Breakout likely within 2-4 bars."
            )

        return False, ""

    # ── 4. Position Heat Management ────────────────────────────────────

    def update_daily_peak(self, current_equity: float) -> None:
        """Update the daily peak equity for drawdown tracking."""
        if current_equity > self._state.daily_peak_equity:
            self._state.daily_peak_equity = current_equity

    def get_heat_multiplier(self, current_equity: float) -> float:
        """
        Returns position size multiplier based on intraday equity drawdown.

        Reduces sizing progressively as equity drops from daily high:
          - 3% drawdown → 75% sizing
          - 5% drawdown → 50% sizing
          - 8% drawdown → 25% sizing
          - 10% drawdown → 0% (stop trading)

        Returns:
            float: Size multiplier (0.0 to 1.0)
        """
        peak = self._state.daily_peak_equity
        if peak <= 0:
            return 1.0

        drawdown = (peak - current_equity) / peak

        for threshold, mult in self.HEAT_LEVELS:
            if drawdown >= threshold:
                last_mult = mult
            else:
                break
        else:
            last_mult = self.HEAT_LEVELS[-1][1]

        # Find the appropriate level
        multiplier = 1.0
        for threshold, mult in self.HEAT_LEVELS:
            if drawdown >= threshold:
                multiplier = mult

        if multiplier < 1.0:
            logger.info(
                f"🌡️ Position heat: equity down {drawdown:.1%} from daily peak "
                f"({peak:.2f} → {current_equity:.2f}). "
                f"Size multiplier: {multiplier:.0%}"
            )

        return multiplier

    def reset_daily(self, current_equity: float = 0.0) -> None:
        """Reset daily tracking at midnight UTC."""
        self._state.daily_peak_equity = current_equity
        self._state.open_sectors.clear()
        logger.info("🔄 DrawdownGuard daily reset")


# Module-level singleton
drawdown_guard = DrawdownGuard()
