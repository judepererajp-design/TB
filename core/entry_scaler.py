"""
Tiered entry / DCA engine.

Computes scaled entry tiers for a signal's entry zone, supporting
SINGLE (midpoint), SCALED (3-tier), and DCA (4-tier) modes.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from config.constants import EntryScaling
from config.loader import cfg

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────

class EntryMode(str, Enum):
    SINGLE = "SINGLE"
    SCALED = "SCALED"
    DCA = "DCA"


@dataclass
class EntryTier:
    """One layer of a tiered entry plan."""
    price: float
    size_pct: float
    label: str


# ── Core class ───────────────────────────────────────────────────

class EntryScaler:
    """Computes entry tiers and weighted average fill prices."""

    def compute_tiers(
        self,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        mode: str = "SINGLE",
    ) -> List[EntryTier]:
        """Return a list of ``EntryTier`` objects for the requested mode.

        Parameters
        ----------
        entry_low:  Lower bound of the entry zone.
        entry_high: Upper bound of the entry zone.
        stop_loss:  Stop-loss price (used only in DCA mode).
        mode:       One of ``SINGLE``, ``SCALED``, ``DCA``.
        """
        try:
            resolved_mode = EntryMode(mode.upper())
        except ValueError:
            logger.warning(
                f"⚠️  Unknown entry mode '{mode}', falling back to SINGLE"
            )
            resolved_mode = EntryMode.SINGLE

        midpoint = (entry_low + entry_high) / 2

        # Guard: entry_low must be below entry_high; swap if inverted
        if entry_low > entry_high:
            entry_low, entry_high = entry_high, entry_low
            midpoint = (entry_low + entry_high) / 2

        if resolved_mode == EntryMode.SINGLE:
            return [EntryTier(price=midpoint, size_pct=1.0, label="mid")]

        if resolved_mode == EntryMode.SCALED:
            pcts = EntryScaling.SCALED_TIER_PCTS
            return [
                EntryTier(price=entry_low, size_pct=pcts[0], label="low"),
                EntryTier(price=midpoint,  size_pct=pcts[1], label="mid"),
                EntryTier(price=entry_high, size_pct=pcts[2], label="high"),
            ]

        # DCA mode
        pcts = EntryScaling.DCA_TIER_PCTS
        below_entry = entry_low - (
            abs(entry_low - stop_loss) * EntryScaling.DCA_BELOW_ENTRY_FACTOR
        )
        return [
            EntryTier(price=entry_high,  size_pct=pcts[0], label="high"),
            EntryTier(price=midpoint,    size_pct=pcts[1], label="mid"),
            EntryTier(price=entry_low,   size_pct=pcts[2], label="low"),
            EntryTier(price=below_entry, size_pct=pcts[3], label="below_low"),
        ]

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def compute_avg_entry(
        tiers: List[EntryTier],
        fills: Dict[str, float],
    ) -> float:
        """Compute the weighted-average fill price.

        Parameters
        ----------
        tiers: The planned entry tiers.
        fills: Mapping of tier label → actual fill price.
               Only tiers present in *fills* contribute to the average.

        Returns
        -------
        Weighted average price, or 0.0 if nothing was filled.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for tier in tiers:
            fill_price = fills.get(tier.label)
            if fill_price is not None:
                weighted_sum += fill_price * tier.size_pct
                total_weight += tier.size_pct
        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def configured_mode() -> str:
        """Read the entry mode from config, defaulting to SINGLE."""
        try:
            return cfg.execution.get("entry_mode", "SINGLE")
        except Exception:
            return "SINGLE"
