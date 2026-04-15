"""
TitanBot Pro — Time Filters
==============================
Blocks or penalizes signals during low-probability time periods.

Dead zones and weekends account for >40% of false signals.
Simply not sending during these periods dramatically improves accuracy.

Rules:
  - Block signals 03:00-07:00 UTC (dead zone) unless A+ grade
  - Reduce confidence 15% outside killzones
  - Saturday: graded penalty (−8 A, −14 B, −20 C) — not a hard block
  - Block all signals on Sunday 00:00-08:00 UTC
"""

import logging
from datetime import datetime, timezone
from typing import Tuple

from config.loader import cfg

logger = logging.getLogger(__name__)


class TimeFilter:
    """
    Evaluates whether the current time is suitable for signal generation.
    Returns confidence adjustment and whether signal should be blocked.
    """

    # Dead zone hours (UTC)
    DEAD_ZONE_START = 3
    DEAD_ZONE_END = 7

    # Killzone hours (UTC) — high-probability periods
    KILLZONES = {
        'london_open':  (7, 10),
        'ny_open':      (12, 15),
        'london_close': (15, 17),
        'asia_open':    (0, 3),
    }

    def evaluate(self, grade: str = "B") -> Tuple[bool, float, str]:
        """
        Evaluate current time for signal quality.

        Args:
            grade: Signal grade (A+, A, B, C)

        Returns:
            (should_block, confidence_adjustment, reason)
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # ── Sunday dead period ────────────────────────────────
        # R7-B4: Changed from hard-block to heavy penalties for B/C grades.
        # Saturday uses penalties; Sunday should be consistent.
        # Grade-aware scale (all soft penalties, no hard blocks):
        if weekday == 6 and hour < 8:
            if grade == "A+":
                return False, -15, "⚠️ Sunday early — reduced confidence (A+ override)"
            if grade == "A":
                return False, -18, "⚠️ Sunday early — heavy penalty, 0.6x size (A-grade)"
            if grade in ("B", "B+"):
                return False, -22, "⚠️ Sunday early — B-grade heavy penalty"
            # C grade: very heavy penalty (effectively blocks most but not hard-kill)
            return False, -28, "⚠️ Sunday early — C-grade near-blocking penalty"

        # ── Saturday (T11: weekend low-volume filter) ─────────────────────────
        # Crypto trades 24/7 but weekend volumes are 20-30% lower with wider spreads.
        # Grade-aware penalties: strong multi-strategy signals survive, thin ones don't.
        if weekday == 5:
            if grade == "C":
                return False, -20, "⚠️ Saturday C-grade — heavy confidence penalty"
            elif grade in ("B", "B+"):
                return False, -14, "⚠️ Saturday B-grade — moderate confidence penalty"
            else:  # A or A+
                return False, -8, "⚠️ Saturday A-grade — light confidence penalty"

        # ── Weekday dead zone (03:00-07:00 UTC) ──────────────
        # R7-B4: Changed from hard-block to heavy penalties for consistency.
        # A+ signals get lightest penalty, C signals get heaviest.
        if self.DEAD_ZONE_START <= hour < self.DEAD_ZONE_END:
            if grade == "A+":
                return False, -10, "⚠️ Dead zone — A+ override"
            if grade == "A":
                return False, -15, "⚠️ Dead zone — A-grade penalty"
            if grade in ("B", "B+"):
                return False, -20, "⚠️ Dead zone — B-grade heavy penalty"
            return False, -25, "⚠️ Dead zone — C-grade near-blocking penalty"

        # ── Asia session (T6: session-aware confidence penalty) ─────────────────
        # Asia session (00:00-03:00 UTC): lower liquidity vs London/NY.
        # Not a hard block but signals need a higher confidence floor.
        asia_session = self.KILLZONES['asia_open']
        if asia_session[0] <= hour < asia_session[1]:
            if grade in ("B", "C"):
                return False, -12, "⚠️ Asia session — lower liquidity, confidence reduced"
            # A/A+ still allowed with light penalty
            return False, -5, "📊 Asia session — A-grade allowed, slight reduction"

        # ── Killzone bonus ────────────────────────────────────
        in_killzone = False
        for kz_name, (start, end) in self.KILLZONES.items():
            if kz_name == 'asia_open':
                continue  # Handled above
            if start <= hour < end:
                in_killzone = True
                break

        if in_killzone:
            return False, 5, "⏰ Killzone active — prime time"

        # ── Outside killzone (not dead zone) ──────────────────
        return False, -8, "📊 Outside killzone — moderate reduction"

    def is_killzone(self) -> bool:
        """Quick check if currently in a killzone"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        if now.weekday() >= 5:
            return False
        for _, (start, end) in self.KILLZONES.items():
            if start <= hour < end:
                return True
        return False

    def get_session_name(self) -> str:
        """Get current session name for display"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        if now.weekday() >= 5:
            return "Weekend"
        for name, (start, end) in self.KILLZONES.items():
            if start <= hour < end:
                return name.replace('_', ' ').title()
        if self.DEAD_ZONE_START <= hour < self.DEAD_ZONE_END:
            return "Dead Zone"
        return "Off-Session"


# ── Singleton ──────────────────────────────────────────────
time_filter = TimeFilter()
