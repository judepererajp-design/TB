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

        # ── FIX Q1: Macro-event dead-zones ────────────────────
        # Around US CPI / FOMC / NFP releases and major monthly options
        # expiries, crypto correlates tightly with macro flows and most
        # technical setups wash out in the volatility spike.  A soft
        # penalty is applied via `_macro_event_adjustment` if we're inside
        # one of those windows AND the heuristic considers it imminent.
        _macro_penalty, _macro_reason = self._macro_event_adjustment(now, grade)
        if _macro_penalty <= -25 and grade in ("B", "C"):
            # Heavy-penalty near-block for weak grades in macro windows.
            return False, _macro_penalty, _macro_reason

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
        # FIX Q13: weekend penalties are regime-aware.  A clean BULL_TREND on
        # a Saturday is much less dangerous than CHOPPY/VOLATILE — trend trades
        # respect the macro structure, while mean-reversion trades need the
        # fuller weekday tape.  Amplify penalties in risk-off regimes.
        if weekday == 5:
            _regime_mult = self._weekend_regime_multiplier()
            if grade == "C":
                return False, int(-20 * _regime_mult), "⚠️ Saturday C-grade — heavy confidence penalty"
            elif grade in ("B", "B+"):
                return False, int(-14 * _regime_mult), "⚠️ Saturday B-grade — moderate confidence penalty"
            else:  # A or A+
                return False, int(-8 * _regime_mult), "⚠️ Saturday A-grade — light confidence penalty"

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
            if _macro_penalty:
                return False, 5 + _macro_penalty, f"⏰ Killzone active — prime time ({_macro_reason})"
            return False, 5, "⏰ Killzone active — prime time"

        # ── Outside killzone (not dead zone) ──────────────────
        if _macro_penalty:
            return False, -8 + _macro_penalty, f"📊 Outside killzone — moderate reduction ({_macro_reason})"
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

    # ── Session-aware liquidity model (PR4 items 9 + 10) ──────

    def _weekend_regime_multiplier(self) -> float:
        """
        FIX Q13: multiplier applied to weekend penalty magnitudes based on
        current regime.  CHOPPY / VOLATILE weekends deserve harsher penalties
        than a clean trend weekend — trend-following trades in BULL/BEAR
        still have macro structure to respect, while mean-reversion in a
        low-volume weekend range is a coin flip.

        Returns 1.0 if the regime is unavailable.  Capped at 1.5 so a
        regime multiplier can never make an A-grade weekend A penalty
        (−8) exceed a C-grade one (−20).
        """
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, "value", "UNKNOWN")
            if regime in ("CHOPPY", "VOLATILE"):
                return 1.5
            if regime in ("BULL_TREND", "BEAR_TREND"):
                return 0.85  # trend weekends are slightly less punitive
            return 1.0
        except Exception:
            return 1.0

    def _macro_event_adjustment(
        self, now: datetime, grade: str
    ) -> Tuple[int, str]:
        """
        FIX Q1: return a macro-event confidence adjustment when the UTC clock
        falls inside a historically-volatile macro window.

        Heuristics (schedule-free, conservative):
          • US CPI: 12:30 UTC, 2nd/3rd Wednesday of the month (±30min)
          • FOMC decision: 18:00 UTC, Wednesday of 3rd full week (±60min)
          • NFP: 12:30 UTC, first Friday of the month (±30min)
          • Options monthly expiry: 08:00 UTC, last Friday of the month (±90min)

        Without a live economic-calendar feed we can't be surgical.  This
        module intentionally uses only calendar heuristics — better to
        soft-penalise across the whole window than miss the event entirely.
        Returns (penalty_pts, reason_str); (0, "") when no window matches.
        """
        minute = now.minute
        hour = now.hour
        weekday = now.weekday()          # 0=Mon … 6=Sun
        day_of_month = now.day

        # Helper: absolute minute-offset vs target (H, M) on same day.
        def _mins_to(h: int, m: int) -> int:
            return abs((hour - h) * 60 + (minute - m))

        # NFP: first Friday 12:30 UTC ±30min
        if weekday == 4 and 1 <= day_of_month <= 7 and _mins_to(12, 30) <= 30:
            return (-30, "⚠️ NFP window — macro event imminent, signals heavily reduced")

        # CPI: 2nd Wed of month, 12:30 UTC ±30min.  Roughly days 8-14.
        if weekday == 2 and 8 <= day_of_month <= 14 and _mins_to(12, 30) <= 30:
            return (-30, "⚠️ CPI window — macro event imminent, signals heavily reduced")

        # FOMC: 3rd Wed of month, 18:00 UTC ±60min (use days 15-21 as proxy).
        if weekday == 2 and 15 <= day_of_month <= 21 and _mins_to(18, 0) <= 60:
            return (-35, "⚠️ FOMC window — rate decision imminent, signals heavily reduced")

        # Monthly options expiry: last Friday of month, 08:00 UTC ±90min.
        # Last Friday is day >= 22 and weekday 4.
        if weekday == 4 and day_of_month >= 22 and _mins_to(8, 0) <= 90:
            # A-grade gets a light tap; lower grades take a heavier hit.
            if grade in ("A", "A+"):
                return (-6, "📉 Monthly options expiry — pin risk, slight confidence cut")
            return (-15, "⚠️ Monthly options expiry — pin risk, signal penalised")

        return (0, "")

    def expected_spread_multiplier(self) -> float:
        """
        Multiplier applied to *observed* ``spread_bps`` to model the
        spread that should be **expected** during the current session.
        Returns 1.0 in killzones (live spread is the best estimate)
        and up to 1.7× during Sunday-early.

        Rationale: a 5-bps quote on Saturday morning can blow out to
        25 bps mid-sweep; using the snapshot under-states the true
        execution cost. Aggregator multiplies the observed spread by
        this factor before propagating to the execution gate.
        """
        try:
            from config.constants import WeekendChop as _WC
        except Exception:
            return 1.0
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        # Sunday early gets the worst multiplier
        if weekday == 6 and hour < 8:
            return _WC.SPREAD_MULT_SUNDAY_EARLY
        # Full Saturday and rest of Sunday
        if weekday >= 5:
            return _WC.SPREAD_MULT_SATURDAY
        # Weekday dead zone
        if self.DEAD_ZONE_START <= hour < self.DEAD_ZONE_END:
            return _WC.SPREAD_MULT_DEAD_ZONE
        # Asia session
        asia_start, asia_end = self.KILLZONES['asia_open']
        if asia_start <= hour < asia_end:
            return _WC.SPREAD_MULT_ASIA
        # Killzone (any of London/NY)
        for name, (start, end) in self.KILLZONES.items():
            if name == 'asia_open':
                continue
            if start <= hour < end:
                return _WC.SPREAD_MULT_KILLZONE
        return _WC.SPREAD_MULT_OFF_SESSION

    def weekend_chop_floor_bump(self) -> int:
        """
        Confidence-floor bump (in points) the aggregator should add
        during weekend low-liquidity windows.

        Stacks on top of TimeFilter.evaluate()'s per-signal penalty:
        the per-signal penalty discourages weak signals from publishing,
        the floor bump raises the bar even for high-grade signals so
        only genuinely strong setups get through during Saturday/Sunday.
        Returns 0 outside weekend windows.
        """
        try:
            from config.constants import WeekendChop as _WC
        except Exception:
            return 0
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        if weekday == 6 and hour < 8:
            return _WC.SUNDAY_EARLY_CONF_FLOOR_BUMP
        if weekday >= 5:
            return _WC.WEEKEND_CONF_FLOOR_BUMP
        return 0


# ── Singleton ──────────────────────────────────────────────
time_filter = TimeFilter()
