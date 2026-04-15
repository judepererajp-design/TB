"""
TitanBot Pro — Signal Aggregator
===================================
The brain of the system. Takes raw strategy signals and runs them through
a multi-factor scoring pipeline before deciding whether to publish.

Pipeline:
  1. Strategy signal arrives (confidence 0-100)
  2. Each analyzer scores the setup independently
  3. Regime-adjusted weights applied to each score
  4. Weighted blend produces final confidence
  5. Deduplication check (no repeat signals)
  6. Minimum confidence gate
  7. Risk/R/R validation
  8. Published to Telegram

Key philosophy: No single factor dominates.
Price action + volume + derivatives + regime must broadly agree.
"""

import asyncio
import logging
import math
import sqlite3
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from config.loader import cfg
from signals.context_contracts import build_setup_context
from strategies.base import SignalResult, SignalDirection

# Named tuple for a single dedup entry — consolidates the three parallel dicts
# (_recent, _recent_conf, _recent_grade) that previously had to be kept in sync.
_DedupEntry = namedtuple('_DedupEntry', ['ts', 'conf', 'grade'])
from analyzers.regime import regime_analyzer, Regime
from analyzers.volume import volume_analyzer, VolumeData
from analyzers.derivatives import derivatives_analyzer, DerivativesData
from analyzers.altcoin_rotation import rotation_tracker
from data.api_client import api
from config.constants import (
    Grading, RateLimiting, Penalties, AggregatorScoring, OutcomeTracking,
)

logger = logging.getLogger(__name__)
try:
    from utils.trade_logger import trade_logger as _tl
except Exception:
    _tl = None


def _validator_issue_text(result: Any, fallback: str = "validator flagged signal") -> str:
    """Return a safe first validator issue for logging/UI text."""
    issues = getattr(result, "issues", None) or []
    if not issues:
        return fallback
    issue = str(issues[0] or "").strip()
    return issue or fallback


@dataclass
class ScoredSignal:
    """A strategy signal that has been fully scored and is ready to publish"""
    base_signal: SignalResult

    # Individual analyzer scores (0-100)
    technical_score: float = Grading.NEUTRAL_SCORE
    volume_score: float = Grading.NEUTRAL_SCORE
    orderflow_score: float = Grading.NEUTRAL_SCORE
    derivatives_score: float = Grading.NEUTRAL_SCORE
    sentiment_score: float = Grading.NEUTRAL_SCORE
    correlation_score: float = Grading.NEUTRAL_SCORE

    # Final
    final_confidence: float = Grading.NEUTRAL_SCORE
    regime: str = "UNKNOWN"
    is_killzone: bool = False
    killzone_bonus: float = 0.0
    sector_adjustment: float = 0.0
    sector_note: str = ""
    derivatives_data: Optional[DerivativesData] = None
    volume_data: Optional[VolumeData] = None
    grade: str = "B"   # A+ | A | B | C
    all_confluence: List[str] = field(default_factory=list)
    confluence_multiplier: float = 1.0  # From ConfluenceScorer (1.30 = 3+ independent clusters)
    power_aligned: bool = False         # True when cross-category strength detected (⚡ badge)
    power_alignment_reason: str = ""    # Human-readable reason (e.g. "4/5 categories confirmed")
    # v2 Power Alignment fields
    power_alignment_tier: str = ""      # "STRONG" | "MODERATE" | "" (empty = not aligned)
    power_alignment_score: float = 0.0  # 0-100 composite power score
    power_alignment_boost: int = 0      # Confidence points added by power alignment


class SignalDeduplicator:
    """
    Prevents the same signal from being sent twice within the dedup window.

    DEDUP-FIX (3 bugs fixed):
    1. PERSISTENCE: State is saved to DB on every mark_sent and loaded on startup.
       Previously in-memory only — a bot restart would wipe the dedup dict and
       immediately resend all signals from the last 3 hours.

    2. CONFIDENCE CONSISTENCY: Both is_duplicate() and mark_sent() now use the same
       confidence value (final scored confidence). Previously is_duplicate() received
       raw strategy confidence (e.g. 72) while mark_sent() stored scored final
       confidence (e.g. 80). The +15 breakthrough check was comparing apples/oranges
       and could fire when it shouldn't.

    3. C-GRADE WINDOW: C-grade (informational) signals use a shorter dedup window
       (cfg.aggregator.c_grade_dedup_minutes, default 30 min) so they don't permanently
       block A/B signals on the same symbol. A 3h C-grade block was a real problem in
       choppy markets where every scan produced a 72-confidence C signal.
    """

    def __init__(self):
        # Single consolidated dict replaces the previous three parallel dicts
        # (_recent, _recent_conf, _recent_grade).  Eliminates three separate O(1)
        # lookups and reduces the chance of the dicts drifting out of sync.
        self._recent: Dict[str, _DedupEntry] = {}   # key -> (ts, conf, grade)
        self._window = cfg.aggregator.get('dedup_window_minutes', 180) * 60
        self._c_window = cfg.aggregator.get('c_grade_dedup_minutes', 60) * 60
        self._conf_breakthrough = RateLimiting.CONF_BREAKTHROUGH_PTS  # Allow resend if final_confidence ↑ this much
        self._db_loaded = False
        # Post-expiry extended cooldown: symbol:direction -> timestamp of expiry
        self._post_expiry: Dict[str, float] = {}
        self._post_expiry_window = 4 * 3600    # 4h cooldown after expiry
        self._post_expiry_breakthrough = RateLimiting.POST_EXPIRY_BREAKTHROUGH_PTS  # Require +20 pts to break through post-expiry
        # asyncio.Lock for atomic check-and-mark to prevent concurrent scan tasks from
        # both passing dedup (the race window is between process() returning and the
        # Telegram send completing, during which another asyncio.gather task can
        # process the same symbol and also see is_duplicate=False).
        # Initialized eagerly here (not lazily) so all callers share the same Lock
        # instance; lazy creation risks two coroutines each creating their own Lock.
        self._lock: asyncio.Lock = asyncio.Lock()

    async def load_from_db(self):
        """
        DEDUP-FIX (PERSISTENCE): Load unexpired dedup entries from DB on startup.
        Called once during engine startup. Non-fatal if DB read fails.

        T4-FIX: Uses asyncio.to_thread to run synchronous sqlite3 calls off the
        event loop, preventing blocking during startup DB reads.
        """
        if self._db_loaded:
            return
        self._db_loaded = True
        try:
            import os
            db_path = cfg.database.get('path', 'data/titanbot.db') if hasattr(cfg.database, 'get') else getattr(cfg.database, 'path', 'data/titanbot.db')
            if not os.path.exists(db_path):
                return

            def _sync_load():
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                now = time.time()
                rows = conn.execute(
                    "SELECT symbol, direction, sent_at, final_confidence, grade "
                    "FROM signal_dedup WHERE sent_at > ? ORDER BY sent_at DESC",
                    (now - self._window,)
                ).fetchall()
                conn.close()
                return rows

            rows = await asyncio.to_thread(_sync_load)
            loaded = 0
            for row in rows:
                key = f"{row['symbol']}:{row['direction']}"
                if key not in self._recent:  # Most recent wins
                    self._recent[key] = _DedupEntry(
                        ts=float(row['sent_at']),
                        conf=float(row['final_confidence'] or 0),
                        grade=row['grade'] or 'B',
                    )
                    loaded += 1
            if loaded:
                logger.info(f"Dedup: loaded {loaded} unexpired entries from DB (restart-safe)")
        except Exception as e:
            logger.debug(f"Dedup DB load failed (non-fatal): {e}")

    def _effective_window(self, key: str) -> float:
        """
        Grade-aware dedup windows:
          A+ / A : 90 min  — high conviction setups can re-fire if market keeps offering them
          B      : 180 min — standard dedup window
          C      : 60 min  — context signal, moderate window to prevent hovering-price spam
          Post-expiry: 4h cooldown regardless of grade
        """
        _expiry_ts = self._post_expiry.get(key, 0)
        if _expiry_ts and time.time() - _expiry_ts < self._post_expiry_window:
            return self._post_expiry_window
        entry = self._recent.get(key)
        grade = entry.grade if entry else 'B'
        if grade == 'C':
            return self._c_window                  # 60 min
        elif grade in ('A+', 'A'):
            return self._window * 0.5              # 90 min (half of standard)
        return self._window                        # 180 min for B

    def is_duplicate(self, symbol: str, direction: str, new_final_confidence: float = 0.0) -> bool:
        """
        DEDUP-FIX (CONFIDENCE CONSISTENCY): now receives final_confidence, not raw confidence.
        Call this AFTER scoring, not before. See aggregator.process() for updated call site.
        """
        key = f"{symbol}:{direction}"
        entry = self._recent.get(key)
        if entry is not None:
            window = self._effective_window(key)
            if time.time() - entry.ts < window:
                # Breakthrough exception: allow if significantly better final confidence
                # Use higher breakthrough threshold if signal recently expired
                _bt = (self._post_expiry_breakthrough
                       if self._post_expiry.get(key, 0) and
                          time.time() - self._post_expiry.get(key, 0) < self._post_expiry_window
                       else self._conf_breakthrough)
                if new_final_confidence > 0 and new_final_confidence >= entry.conf + _bt:
                    return False  # Meaningful upgrade — allow resend
                return True
        return False

    def mark_sent(self, symbol: str, direction: str, confidence: float = 0.0, grade: str = 'B'):
        """Persist to DB and update in-memory dict.

        T4-FIX: DB write is offloaded to a background thread via asyncio.to_thread
        so the synchronous sqlite3 call doesn't block the event loop. The in-memory
        update happens immediately (synchronous) for correctness; only the DB
        persistence is deferred.
        """
        key = f"{symbol}:{direction}"
        now = time.time()  # BUG-6 FIX: capture once — prevents in-memory vs DB timestamp skew
        self._recent[key] = _DedupEntry(ts=now, conf=confidence, grade=grade)
        # Persist to DB so restarts don't wipe the window — non-blocking
        try:
            import os
            db_path = cfg.database.get('path', 'data/titanbot.db') if hasattr(cfg.database, 'get') else getattr(cfg.database, 'path', 'data/titanbot.db')
            if os.path.exists(db_path):
                def _sync_write():
                    conn = sqlite3.connect(db_path)
                    conn.execute(
                        "INSERT OR REPLACE INTO signal_dedup "
                        "(symbol, direction, sent_at, final_confidence, grade) VALUES (?,?,?,?,?)",
                        (symbol, direction, now, confidence, grade)
                    )
                    conn.commit()
                    conn.close()

                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(asyncio.to_thread(_sync_write))
                    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                except RuntimeError:
                    # No running event loop (e.g. called from sync context) — fallback to sync
                    _sync_write()
        except Exception:
            pass  # Non-fatal — in-memory still works

    def unmark(self, symbol: str, direction: str) -> None:
        """PHASE 3 AUDIT FIX (P3-2): Remove a dedup mark for a signal that was
        marked but later killed by a post-dedup gate (health monitor, throttle,
        circuit breaker).  Without this, killed signals poison the dedup window
        and suppress valid signals for the remainder of the window.
        """
        key = f"{symbol}:{direction}"
        self._recent.pop(key, None)
        # Also remove from DB so restarts don't resurrect stale marks
        try:
            import os
            db_path = cfg.database.get('path', 'data/titanbot.db') if hasattr(cfg.database, 'get') else getattr(cfg.database, 'path', 'data/titanbot.db')
            if os.path.exists(db_path):
                def _sync_delete():
                    conn = sqlite3.connect(db_path)
                    conn.execute(
                        "DELETE FROM signal_dedup WHERE symbol = ? AND direction = ?",
                        (symbol, direction)
                    )
                    conn.commit()
                    conn.close()
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(asyncio.to_thread(_sync_delete))
                    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                except RuntimeError:
                    _sync_delete()
        except Exception:
            pass  # Non-fatal — in-memory already cleared

    def clear_expired(self):
        now = time.time()
        expired = [k for k, entry in self._recent.items()
                   if now - entry.ts > self._effective_window(k)]
        for k in expired:
            del self._recent[k]

    async def check_and_mark(
        self,
        symbol: str,
        direction: str,
        confidence: float = 0.0,
        grade: str = 'B',
    ) -> bool:
        """
        Atomically check for duplicate AND mark as sent under a lock.

        Returns True  → duplicate (caller should discard the signal).
        Returns False → not a duplicate (signal is now pre-marked, caller may publish).

        Holding the lock during both check and mark eliminates the TOCTOU race
        where two concurrent asyncio tasks for the same symbol both pass
        is_duplicate() before either calls mark_sent(), resulting in duplicate
        Telegram messages.
        """
        async with self._lock:
            if self.is_duplicate(symbol, direction, new_final_confidence=confidence):
                return True
            self.mark_sent(symbol, direction, confidence=confidence, grade=grade)
            return False


class SignalAggregator:
    """
    Multi-factor signal scoring engine.
    All strategy signals pass through here before going to Telegram.
    """

    # Grade thresholds — referenced by both the hourly-limit bypass and _grade_signal.
    # Keeping them here avoids silent drift if the grading criteria changes.
    _GRADE_APLUS_MIN_CONF: int = Grading.APLUS_MIN_CONF  # minimum confidence for an A+ grade signal
    _GRADE_A_MIN_CONF: int = Grading.A_MIN_CONF   # minimum confidence for an A-grade signal
    _GRADE_B_MIN_CONF: int = Grading.B_MIN_CONF   # minimum confidence for a B-grade signal
    _RATE_LIMIT_BYPASS_MIN_RR: float = RateLimiting.RATE_LIMIT_BYPASS_MIN_RR  # minimum RR for rate limit bypass
    _SLOT_REUSE_COOLDOWN_SECS: int = RateLimiting.SLOT_REUSE_COOLDOWN_SECS  # seconds before a released slot can be reused

    def __init__(self):
        self._deduplicator = SignalDeduplicator()
        self._agg_cfg = cfg.aggregator
        self._min_confidence = getattr(self._agg_cfg, 'min_confidence', 72)
        self._daily_count: Dict[str, int] = {}  # symbol -> signals today
        self._daily_count_times: Dict[str, list] = {}  # symbol -> list of publish timestamps
        self._hourly_total = 0
        self._hour_reset_time = time.time()
        self._last_slot_release_time: float = 0.0  # cooldown: prevents burst after slot release
        self._max_per_hour: int = cfg.aggregator.get(
            'max_signals_per_hour',
            cfg.telegram.get('max_signals_per_hour', 12)  # FIX #13: sync with telegram cap
        )
        # Rolling buffer for adaptive confidence threshold (initialized eagerly to avoid
        # defensive hasattr/AttributeError checks inside the hot process() path)
        self._recent_score_buffer: deque = deque(maxlen=100)
        # FIX M5: track UTC date to detect day rollover without external call
        import datetime as _dt
        self._current_date_utc: str = _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d')
        # CLARITY-SPAM FIX: track per-(symbol, direction) AGG_THRESHOLD failures within
        # the current hour.  Once a symbol fails AGG_THRESHOLD N times while using the
        # clarity bypass, the bypass is suspended for that symbol for the hour.
        # Prevents AIOT-style re-generation storms (30+ bypass attempts, all failing
        # at 67.x/68, burning API calls and log space with zero signal output).
        # Key: (symbol, direction_str)  Value: (failure_count, first_failure_ts)
        self._clarity_failure_counts: Dict[tuple, tuple] = {}
        self._clarity_max_failures: int = 3    # suspend after 3 failures
        self._clarity_suspend_secs: int = 3600  # suspend for 1 hour

    async def load_hourly_counter(self):
        """L3-FIX: Restore hourly counter from DB on startup.
        Prevents 2× hourly limit after mid-hour restart."""
        try:
            from data.database import db
            state = await db.load_learning_state('hourly_signal_counter')
            if state:
                saved_reset = float(state.get('reset_time', 0))
                saved_count = int(state.get('count', 0))
                # Only restore if the saved hour hasn't expired
                if time.time() - saved_reset < 3600:
                    self._hourly_total = saved_count
                    self._hour_reset_time = saved_reset
                    logger.info(
                        f"Hourly counter restored from DB: {saved_count} signals "
                        f"(resets in {3600 - (time.time() - saved_reset):.0f}s)"
                    )
        except Exception as e:
            logger.debug(f"Hourly counter load failed (non-fatal): {e}")

    def _persist_hourly_counter(self):
        """L3-FIX: Fire-and-forget persistence of the hourly counter to DB."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._save_hourly_counter())
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        except RuntimeError:
            pass  # No running loop — skip persist

    async def _save_hourly_counter(self):
        """L3-FIX: Async save of hourly counter state."""
        try:
            from data.database import db
            await db.save_learning_state('hourly_signal_counter', {
                'count': self._hourly_total,
                'reset_time': self._hour_reset_time,
            })
        except Exception:
            pass  # Non-fatal

    def unmark_signal(self, symbol: str, direction: str) -> None:
        """PHASE 3 FIX (P3-2): Public API to rollback a dedup mark.

        Called by engine.py when a post-dedup gate (circuit breaker, health
        monitor, burst throttle) blocks a signal that was already marked as
        sent by the deduplicator.  Without this, the killed signal would
        suppress valid signals for the remainder of the dedup window.
        """
        self._deduplicator.unmark(symbol, direction)

    def _check_day_rollover(self):
        """FIX M5: Reset daily counts when UTC date changes (survives restarts via date check).
        FIX MEMORY-LEAK: Also purge stale entries from _daily_count_times to prevent
        unbounded dict growth.  Previously only _daily_count was reset on rollover while
        _daily_count_times grew forever — new symbols added daily but entries for symbols
        that stopped appearing were never removed."""
        import datetime as _dt
        today = _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d')
        if today != self._current_date_utc:
            self._daily_count = {}
            # Purge _daily_count_times: remove entries older than 24h and delete
            # empty symbol keys to prevent unbounded dict growth.
            _now = time.time()
            _stale_keys = []
            for _sym, _times in self._daily_count_times.items():
                fresh = [t for t in _times if _now - t < 86400]
                if fresh:
                    self._daily_count_times[_sym] = fresh
                else:
                    _stale_keys.append(_sym)
            for _sym in _stale_keys:
                del self._daily_count_times[_sym]
            self._current_date_utc = today
            logger.info(
                f"Daily signal counts reset for {today} "
                f"(purged {len(_stale_keys)} stale symbol entries from _daily_count_times)"
            )

    async def process(self, signal: SignalResult, raw_signals: list = None) -> Optional[ScoredSignal]:
        """
        Main entry point. Takes a raw strategy signal and returns a
        fully-scored ScoredSignal ready for publishing, or None if rejected.

        raw_signals: Optional list of all strategy signals from the same scan cycle.
        Used by the SMC veto to detect structural conflicts. (BUG-E fix)
        """
        self._check_day_rollover()  # FIX M5: reset daily counts at midnight

        # ── 0. Validate signal geometry + recompute R:R ──────
        # Never trust strategy math — recompute centrally using TP2 (primary target)
        entry_mid = (signal.entry_low + signal.entry_high) / 2
        # FIX (P3-A): Clamp TP2/TP3 to a minimum positive value for SHORT signals.
        # Strategies like SMC compute TP2 = entry - X*ATR which can go negative on
        # low-priced tokens (e.g. AIOT at $0.03 → TP2=-$0.001). Instead of rejecting
        # (which wastes the analysis), clamp to 0.1% of entry (tiny but positive).
        _min_tp_price = entry_mid * Penalties.TP_NEGATIVE_FLOOR_PCT  # 0.1% of entry price
        if signal.tp2 <= 0 and _min_tp_price > 0:
            logger.debug(
                f"TP2 clamped {signal.symbol}: {signal.tp2:.6f} → {_min_tp_price:.6f} "
                f"(negative TP2 floor)"
            )
            signal.tp2 = _min_tp_price
        if signal.tp1 <= 0 and _min_tp_price > 0:
            signal.tp1 = _min_tp_price
        if hasattr(signal, 'tp3') and signal.tp3 is not None and signal.tp3 <= 0 and _min_tp_price > 0:
            logger.debug(
                f"TP3 clamped {signal.symbol}: {signal.tp3:.6f} → {_min_tp_price:.6f} "
                f"(negative TP3 floor)"
            )
            signal.tp3 = _min_tp_price

        # ── TP3 ATR clamp: bound structural TPs to realistic market movement ──
        # Strategies like Elliott Wave and Breakout use Fibonacci/measured-move
        # targets that can produce unbounded TP3 distances.  This normalization
        # layer caps |TP3 - entry| at TP3_MAX_ATR_MULT × ATR while leaving the
        # strategy's structural logic intact.
        _sig_atr = getattr(signal, 'atr', None)
        if _sig_atr and _sig_atr > 0 and signal.tp3 is not None:
            _max_tp3_dist = _sig_atr * Penalties.TP3_MAX_ATR_MULT
            _tp3_dist = abs(signal.tp3 - entry_mid)
            if _tp3_dist > _max_tp3_dist:
                _old_tp3 = signal.tp3
                signal.tp3 = entry_mid + math.copysign(_max_tp3_dist, signal.tp3 - entry_mid)
                logger.debug(
                    f"TP3 ATR-clamped {signal.symbol} {signal.strategy}: "
                    f"{_old_tp3:.6f} → {signal.tp3:.6f} "
                    f"(dist {_tp3_dist / _sig_atr:.1f}×ATR → {Penalties.TP3_MAX_ATR_MULT}×ATR)"
                )

        # TP levels must be positive — a price of ≤0 is physically impossible and
        # causes inflated R:R (e.g. entry_mid - (-0.003) = large reward on a $0.03 coin).
        for _tp_name, _tp_val in (("tp1", signal.tp1), ("tp2", signal.tp2)):
            if _tp_val <= 0:
                logger.info(
                    f"Geometry rejected {signal.symbol}: {_tp_name}={_tp_val:.6f} <= 0"
                )
                if _tl:
                    _dir = "LONG" if signal.direction == SignalDirection.LONG else "SHORT"
                    _tl.signal(symbol=signal.symbol, direction=_dir, grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result=f"REJECTED(GEOMETRY_INVALID {_tp_name}<=0)")
                return None

        if signal.direction == SignalDirection.LONG:
            if signal.stop_loss >= signal.entry_low:
                logger.info(f"Geometry rejected {signal.symbol}: SL >= entry_low (LONG)")
                if _tl:
                    _tl.signal(symbol=signal.symbol, direction="LONG", grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result="REJECTED(GEOMETRY_INVALID sl>=entry_low LONG)")
                return None
            # FIX: TP1 must be above entry_mid (not entry_high — tight zones would fail)
            if signal.tp1 <= entry_mid:
                logger.info(f"Geometry rejected {signal.symbol}: TP1 {signal.tp1:.6f} <= entry_mid {entry_mid:.6f} (LONG)")
                if _tl:
                    _tl.signal(symbol=signal.symbol, direction="LONG", grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result="REJECTED(GEOMETRY_INVALID tp1<=entry_mid LONG)")
                return None
            # BUG-3 FIX: TP2 must be above TP1 for LONG — reversed ordering causes early
            # exit at the wrong (lower) target and inflated reported R:R.
            if signal.tp2 <= signal.tp1:
                logger.info(f"Geometry rejected {signal.symbol}: TP2 {signal.tp2:.6f} <= TP1 {signal.tp1:.6f} (LONG)")
                if _tl:
                    _tl.signal(symbol=signal.symbol, direction="LONG", grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result="REJECTED(GEOMETRY_INVALID tp2<=tp1 LONG)")
                return None
            risk = entry_mid - signal.stop_loss
            reward = signal.tp2 - entry_mid   # Use TP2 — the primary target
        else:
            if signal.stop_loss <= signal.entry_high:
                logger.info(f"Geometry rejected {signal.symbol}: SL <= entry_high (SHORT)")
                if _tl:
                    _tl.signal(symbol=signal.symbol, direction="SHORT", grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result="REJECTED(GEOMETRY_INVALID sl<=entry_high SHORT)")
                return None
            # FIX: TP1 must be below entry_mid (was entry_low — killed ALL short signals with narrow zones)
            if signal.tp1 >= entry_mid:
                logger.info(f"Geometry rejected {signal.symbol}: TP1 {signal.tp1:.6f} >= entry_mid {entry_mid:.6f} (SHORT)")
                if _tl:
                    _tl.signal(symbol=signal.symbol, direction="SHORT", grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result="REJECTED(GEOMETRY_INVALID tp1>=entry_mid SHORT)")
                return None
            # BUG-3 FIX: TP2 must be below TP1 for SHORT — reversed ordering causes early
            # exit at the wrong (higher) target and inflated reported R:R.
            if signal.tp2 >= signal.tp1:
                logger.info(f"Geometry rejected {signal.symbol}: TP2 {signal.tp2:.6f} >= TP1 {signal.tp1:.6f} (SHORT)")
                if _tl:
                    _tl.signal(symbol=signal.symbol, direction="SHORT", grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result="REJECTED(GEOMETRY_INVALID tp2>=tp1 SHORT)")
                return None
            risk = signal.stop_loss - entry_mid
            reward = entry_mid - signal.tp2   # Use TP2 — the primary target

        if risk <= 0:
            return None
        verified_rr = reward / risk
        if abs(verified_rr - signal.rr_ratio) > Penalties.RR_CORRECTION_TOLERANCE:
            logger.debug(
                f"RR correction {signal.symbol}: strategy said {signal.rr_ratio:.2f}, "
                f"actual={verified_rr:.2f}"
            )
        signal.rr_ratio = round(verified_rr, 2)  # Override with verified value

        # ── RR sanity cap: flag absurd R/R that indicates strategy miscalculation ──
        # In perpetual futures, a genuine 10R+ setup is extremely rare.
        # If we see 13.6R it almost always means TP3 was calculated incorrectly.
        # Cap display at 8R and add a flag so the user knows the levels may be off.
        _RR_SANITY_CAP = Penalties.RR_SANITY_CAP
        if signal.rr_ratio > _RR_SANITY_CAP:
            logger.warning(
                f"⚠️ RR sanity: {signal.symbol} {signal.strategy} "
                f"R/R={signal.rr_ratio:.1f}R exceeds cap — TP levels may be miscalculated. "
                f"Capping display at {_RR_SANITY_CAP}R."
            )
            signal.rr_ratio = _RR_SANITY_CAP
            signal.confluence.append(
                f"⚠️ R/R capped at {_RR_SANITY_CAP}R — original {verified_rr:.1f}R suggests "
                f"TP calculation may be off. Verify TP3 before sizing."
            )

        # ── Enforce max_rr from risk config ──────────────────────────
        # The user-configurable max_rr (default 6.0) was defined in settings.yaml
        # and schema.py but never actually enforced in the pipeline. This closes
        # the gap: signals above max_rr are capped to the configured limit.
        # FIX: Previous version used try/except that silently swallowed config
        # load errors, allowing R:R to exceed max_rr (seen: PIPPIN 8.0R, ZEC 7.2R).
        # Now uses RR_SANITY_CAP as fallback so max_rr ALWAYS applies.
        _max_rr_fallback = Penalties.RR_SANITY_CAP  # Already imported; single source of truth
        try:
            from config.loader import cfg as _rr_cfg
            _user_max_rr = getattr(_rr_cfg.risk, 'max_rr', _max_rr_fallback)
        except (ImportError, AttributeError):
            _user_max_rr = _max_rr_fallback
        if _user_max_rr and signal.rr_ratio > _user_max_rr:
            logger.info(
                f"⚡ R/R {signal.symbol}: {signal.rr_ratio:.1f}R capped to "
                f"{_user_max_rr:.1f}R (risk.max_rr)"
            )
            signal.rr_ratio = round(_user_max_rr, 2)

        # ── RR floor gate (single authoritative check) ───────
        # Strategies may do their own check but this is the canonical gate.
        # Uses setup_class-aware floor — EV gate in alpha_model owns profitability.
        from strategies.base import cfg_min_rr as _min_rr
        _rr_floor = _min_rr(getattr(signal, 'setup_class', 'intraday'))

        # ── Dynamic RR discount for very-high-confidence signals ───────────
        # EV(95% × 1.5R) = 1.43R  >> EV(68% × 2.0R) = 1.36R
        # When the ensemble is ≥ 95% confident, the rigid floor actively
        # kills positive-EV setups.  Scale it down by 0.75×.
        from config.constants import DynamicRR
        if signal.confidence >= DynamicRR.HIGH_CONF_THRESHOLD:
            _rr_floor_orig = _rr_floor
            _rr_floor *= DynamicRR.HIGH_CONF_RR_DISCOUNT
            _rr_floor = round(_rr_floor, 2)
            logger.info(
                f"📐 Dynamic RR: {signal.symbol} conf={signal.confidence:.0f} ≥ "
                f"{DynamicRR.HIGH_CONF_THRESHOLD} → floor {_rr_floor_orig:.2f} → {_rr_floor:.2f}"
            )

        # ── PHASE 2 FIX (P2-C): SMC structural veto ───────────────────────────
        # If SmartMoneyConcepts generates a SHORT signal on this symbol AND we are
        # about to publish a LONG (or vice versa), apply a strong confidence penalty.
        # SMC reads market structure directly — it should not be outvoted by 4
        # momentum/lagging strategies. This doesn't block the signal entirely but
        # makes it very hard to pass the confidence floor.
        #
        # BUG-E FIX: The original code used `hasattr(self, '_last_raw_signals')` which
        # was never set, so this veto NEVER fired. Fixed by passing raw_signals as an
        # explicit parameter to process() and using it directly here.
        try:
            _smc_direction = None
            _smc_search_list = raw_signals if raw_signals else []
            for _s in _smc_search_list:
                if _s.strategy == "SmartMoneyConcepts":
                    _smc_direction = _s.direction.value if hasattr(_s.direction, 'value') else _s.direction
                    break
            if _smc_direction and _smc_direction != getattr(signal.direction, 'value', str(signal.direction)):
                # FIX SMC-VETO: was -20, changed to -12.
                # -20 on a 65-confidence signal dropped it to 45, killing it before
                # the aggregator's weighted scoring had a chance to run. The veto
                # should flag disagreement (soft penalty) not act as a hard kill.
                # -12 still makes it very hard to clear the confidence floor when
                # SMC disagrees, but a legitimately high-scoring signal can still pass.
                signal.confidence = max(Penalties.SMC_VETO_CONF_FLOOR, signal.confidence - Penalties.SMC_VETO_PENALTY_PTS)
                signal.confluence.append(
                    f"⚠️ SMC structural conflict: SMC={_smc_direction}, "
                    f"signal={getattr(signal.direction, 'value', str(signal.direction))} — confidence -12"
                )
                logger.info(
                    f"P2-C SMC veto: SMC {_smc_direction} vs {getattr(signal.direction, 'value', str(signal.direction))} → -12"
                )
        except Exception:
            pass

        if signal.rr_ratio < _rr_floor:
            logger.info(
                f"❌ Signal died (agg) | {signal.symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                f"| reason=RR_FLOOR | rr={signal.rr_ratio:.2f} < {_rr_floor:.2f} "
                f"({getattr(signal, 'setup_class', 'intraday')})"
            )
            if _tl:
                _dir = getattr(signal.direction, 'value', str(signal.direction))
                _rr_regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                _tl.signal(symbol=signal.symbol, direction=_dir, grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime=_rr_regime, result=f"REJECTED(RR_FLOOR rr={signal.rr_ratio:.2f}<{_rr_floor:.2f})")
            # Feed diagnostic engine
            try:
                from core.diagnostic_engine import diagnostic_engine
                diagnostic_engine.record_signal_death(
                    symbol=signal.symbol, direction=getattr(signal.direction, 'value', str(signal.direction)),
                    strategy=signal.strategy, kill_reason="RR_FLOOR",
                    rr=signal.rr_ratio, confidence=signal.confidence,
                    regime=self._agg_cfg.get('regime', 'UNKNOWN') if hasattr(self._agg_cfg, 'get') else 'UNKNOWN',
                    setup_class=getattr(signal, 'setup_class', 'intraday'),
                )
            except Exception:
                pass
            return None

        # ── 0b. Signal data validation ("Skeptical LLM Mode") ────────
        # Dual-layer defense: Layer A (programmatic hard checks) catches
        # impossible values; Layer B (LLM anomaly detection) catches
        # suspicious indicator/direction combinations.
        # Feature flag: SIGNAL_VALIDATOR (off | shadow | live)
        try:
            from config.feature_flags import ff
            from signals.signal_validator import signal_validator

            _ff_state = ff.get_state("SIGNAL_VALIDATOR")
            if _ff_state in ("live", "shadow"):
                _dir_str_v = getattr(signal.direction, 'value', str(signal.direction))
                _regime_str_v = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                _val_data = {
                    "symbol": signal.symbol,
                    "direction": _dir_str_v,
                    "strategy": signal.strategy,
                    "confidence": signal.confidence,
                    "regime": _regime_str_v,
                    "entry_low": signal.entry_low,
                    "entry_high": signal.entry_high,
                    "stop_loss": signal.stop_loss,
                    "price": entry_mid,
                    "rr_ratio": signal.rr_ratio,
                }
                # Include raw_data indicators if available
                _rd = signal.raw_data or {}
                for _ik in ("rsi", "adx", "macd_histogram", "volume_ratio",
                            "funding_rate", "bollinger_position", "atr"):
                    if _ik in _rd:
                        _val_data[_ik] = _rd[_ik]
                if signal.atr is not None:
                    _val_data["atr"] = signal.atr

                # Run LLM layer only when AI mode is active
                _run_llm = False
                try:
                    from analyzers.ai_analyst import ai_analyst
                    _run_llm = getattr(ai_analyst, '_mode', 'off') != 'off'
                except Exception:
                    pass

                _vr = await signal_validator.validate(_val_data, run_llm=_run_llm)

                if _ff_state == "live":
                    _primary_issue = _validator_issue_text(_vr)
                    if _vr.status == "ERROR" or _vr.kill_switch:
                        logger.warning(
                            "🛑 Signal rejected by validator | %s %s %s | %s%s",
                            signal.symbol, _dir_str_v, signal.strategy,
                            "; ".join(_vr.issues) if _vr.issues else _primary_issue,
                            " [KILL SWITCH]" if _vr.kill_switch else "",
                        )
                        if _tl:
                            _tl.signal(
                                symbol=signal.symbol, direction=_dir_str_v,
                                grade="?", confidence=signal.confidence,
                                entry_low=signal.entry_low, entry_high=signal.entry_high,
                                stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2,
                                rr=signal.rr_ratio, strategy=signal.strategy,
                                regime=_regime_str_v,
                                result=f"REJECTED(VALIDATOR_ERROR {_primary_issue[:60]})",
                            )
                        # Record in diagnostic engine so it appears in Kill Reasons chart
                        try:
                            from core.diagnostic_engine import diagnostic_engine
                            _kill_reason = "VALIDATOR_KILL_SWITCH" if _vr.kill_switch else "VALIDATOR_ERROR"
                            diagnostic_engine.record_signal_death(
                                symbol=signal.symbol, direction=_dir_str_v,
                                strategy=signal.strategy, kill_reason=_kill_reason,
                                rr=signal.rr_ratio, confidence=signal.confidence,
                                regime=_regime_str_v,
                                setup_class=getattr(signal, 'setup_class', 'intraday'),
                            )
                        except Exception:
                            pass
                        return None
                    elif _vr.status == "WARNING" and _vr.data_quality in ("LOW", "MEDIUM"):
                        # Dynamic penalty: scales with LLM distrust level
                        from signals.signal_validator import DEFAULT_WARNING_PENALTY
                        _penalty = _vr.dynamic_penalty if _vr.dynamic_penalty > 0 else DEFAULT_WARNING_PENALTY
                        signal.confidence = max(40, signal.confidence - _penalty)
                        signal.confluence.append(
                            f"⚠️ Data validator: {_primary_issue[:80]} (confidence -{_penalty})"
                        )
                        logger.info(
                            "⚠️ Validator warning | %s %s | quality=%s | %s | conf -%d",
                            signal.symbol, signal.strategy, _vr.data_quality,
                            _primary_issue[:60],
                            _penalty,
                        )
                else:
                    # Shadow mode — log only, don't affect signal
                    if _vr.status != "OK":
                        logger.info(
                            "👻 VALIDATOR shadow | %s %s %s | status=%s quality=%s | %s",
                            signal.symbol, _dir_str_v, signal.strategy,
                            _vr.status, _vr.data_quality,
                            "; ".join(_vr.issues)[:120],
                        )
        except ImportError:
            pass  # validator not available — skip gracefully
        except Exception as _val_err:
            logger.debug("Signal validator error (non-fatal): %s", _val_err)

        # ── 1. Direction string (dedup check moved to post-scoring, step 13a) ──
        # DEDUP-FIX: Dedup now runs AFTER scoring so it compares final vs final confidence.
        # Previously it compared raw strategy confidence (pre-scoring) which caused the
        # breakthrough exception to fire incorrectly.
        direction_str = getattr(signal.direction, 'value', str(signal.direction)) if hasattr(signal.direction, 'value') else str(signal.direction)

        # ── 2. Daily signal limit per symbol ────────────────
        # A/A+ signals get a higher daily symbol limit — they represent persistent
        # institutional setups that deserve multiple looks in a day
        # Estimate grade from raw confidence for the purpose of the symbol limit.
        # Full scoring hasn't run yet, but raw confidence is a reasonable proxy.
        # A/A+ typically come from conf≥80; use 78 as threshold to be generous.
        _raw_conf = getattr(signal, 'confidence', 0)
        _sig_grade_est = 'A' if _raw_conf >= Grading.A_GRADE_ESTIMATE_RAW_CONF else 'B'
        _grade_limit_mult = AggregatorScoring.GRADE_LIMIT_MULT_HIGH if _sig_grade_est in ('A+', 'A') else AggregatorScoring.GRADE_LIMIT_MULT_DEFAULT
        max_per_symbol = getattr(self._agg_cfg, 'max_signals_per_symbol', 3) * _grade_limit_mult
        # Rolling 24h window for symbol count — not midnight reset
        _now_ts = time.time()
        _sym_times = self._daily_count_times.get(signal.symbol, [])
        _sym_times = [t for t in _sym_times if _now_ts - t < 86400]  # last 24h only
        self._daily_count_times[signal.symbol] = _sym_times
        symbol_count = len(_sym_times)  # rolling count
        if symbol_count >= max_per_symbol:
            logger.info(
                f"❌ Signal died (agg) | {signal.symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                f"| reason=DAILY_SYMBOL_LIMIT | count={symbol_count}/{max_per_symbol}"
            )
            if _tl:
                _dir = getattr(signal.direction, 'value', str(signal.direction))
                _tl.signal(symbol=signal.symbol, direction=_dir, grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result=f"REJECTED(DAILY_SYMBOL_LIMIT count={symbol_count}/{max_per_symbol})")
            return None

        # ── 3. Hourly total limit ────────────────────────────
        self._refresh_hourly_counter()
        max_hourly = getattr(self, '_max_per_hour', cfg.aggregator.get('max_signals_per_hour', 6))

        # FIX: Quality-tiered slot reservation. Split hourly capacity:
        #   Slots 1-8  → open to all grades (A+, A, B)
        #   Slots 9-12 → reserved for A/A+ only (B rejected early)
        # This prevents 12 fast B-grade signals from crowding out
        # later high-quality A-grade setups (seen: 33 signals/hour,
        # best signals rejected because B-grades filled all slots).
        _quality_reserve_threshold = min(max_hourly - 4, max_hourly * 2 // 3)

        # FIX: Trend-aligned signals (SHORT in BEAR_TREND, LONG in BULL_TREND)
        # are exempt from the hourly rate limit — they should never be killed
        # by counter-trend longs filling the quota first.
        # Evidence: BANANAS31 SHORT (3.4R, best signal) was killed by HOURLY_TOTAL_LIMIT
        # because 4 counter-trend longs published first in same hour.
        _is_htf_aligned_exempt = False
        try:
            from analyzers.htf_guardrail import htf_guardrail as _agg_htf
            _is_htf_aligned_exempt = (
                (getattr(signal.direction, 'value', str(signal.direction)) == "SHORT" and _agg_htf._weekly_bias == "BEARISH") or
                (getattr(signal.direction, 'value', str(signal.direction)) == "LONG" and _agg_htf._weekly_bias == "BULLISH")
            )
        except Exception:
            pass

        # Only A+ grade signals bypass the hourly cap — truly exceptional setups
        # should never be killed by rate limits.  Regular A-grade (conf ≥ 80) is too
        # common to bypass; only A+ (conf ≥ 88) with strong RR qualifies.
        # Provisional threshold mirrors _grade_signal: A+≥88.
        # Secondary gate: rr_ratio must be ≥2.5 to guard against confidence inflation
        # silently making every signal exempt and rendering the hourly cap meaningless.
        _raw_conf = signal.confidence or 0
        _raw_rr = signal.rr_ratio or 0.0
        _is_high_grade_exempt = _raw_conf >= self._GRADE_APLUS_MIN_CONF and _raw_rr >= self._RATE_LIMIT_BYPASS_MIN_RR

        # FIX: clarity=100 signals killed by hourly limit.  A clarity ≥ 95
        # signal is exceptionally clean — bypasses hourly cap even if
        # confidence is slightly below A+ threshold (80 vs 88).
        _clarity = getattr(signal, 'clarity_score', 0) or 0
        _dir_str_cl = getattr(signal.direction, 'value', str(signal.direction))
        _clarity_key = (signal.symbol, _dir_str_cl)

        # CLARITY-SPAM FIX: check if this symbol×direction has already exhausted
        # its clarity bypass budget for this hour.  AIOT had clarity=100 but kept
        # failing AGG_THRESHOLD at 67.x/68 — bypassing the hourly cap 30+ times
        # and generating nothing but log noise.  After _clarity_max_failures
        # consecutive AGG_THRESHOLD failures, suspend the bypass for 1 hour.
        _clarity_suspended = False
        if _clarity >= RateLimiting.CLARITY_BYPASS_MIN_SCORE:
            _cf = self._clarity_failure_counts.get(_clarity_key)
            if _cf is not None:
                _cf_count, _cf_ts = _cf
                _cf_age = time.time() - _cf_ts
                if _cf_count >= self._clarity_max_failures and _cf_age < self._clarity_suspend_secs:
                    _clarity_suspended = True
                    logger.debug(
                        f"Clarity bypass suspended | {signal.symbol} {_dir_str_cl} "
                        f"| {_cf_count} AGG_THRESHOLD failures in last {_cf_age/60:.0f}min"
                    )
                elif _cf_age >= self._clarity_suspend_secs:
                    # Suspension window expired — reset counter
                    del self._clarity_failure_counts[_clarity_key]

        _is_clarity_exempt = (
            not _clarity_suspended
            and _clarity >= RateLimiting.CLARITY_BYPASS_MIN_SCORE
            and _raw_conf >= RateLimiting.CLARITY_BYPASS_MIN_CONF
            and _raw_rr >= RateLimiting.CLARITY_BYPASS_MIN_RR
        )

        _is_exempt = _is_htf_aligned_exempt or _is_high_grade_exempt or _is_clarity_exempt

        # Quality reserve gate: once open slots are exhausted, only A/A+ pass
        _is_a_or_above = _raw_conf >= self._GRADE_A_MIN_CONF
        if (self._hourly_total >= _quality_reserve_threshold
                and not _is_a_or_above
                and not _is_exempt
                and self._hourly_total < max_hourly):
            logger.info(
                f"❌ Signal died (agg) | {signal.symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                f"| reason=HOURLY_QUALITY_RESERVE | total={self._hourly_total}/{max_hourly} "
                f"(reserve threshold={_quality_reserve_threshold}, conf={_raw_conf:.0f})"
            )
            if _tl:
                _dir = getattr(signal.direction, 'value', str(signal.direction))
                _tl.signal(symbol=signal.symbol, direction=_dir, grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result=f"REJECTED(HOURLY_QUALITY_RESERVE total={self._hourly_total}/{max_hourly})")
            return None

        if self._hourly_total >= max_hourly and not _is_exempt:
            logger.info(
                f"❌ Signal died (agg) | {signal.symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                f"| reason=HOURLY_TOTAL_LIMIT | total={self._hourly_total}/{max_hourly}"
            )
            if _tl:
                _dir = getattr(signal.direction, 'value', str(signal.direction))
                _tl.signal(symbol=signal.symbol, direction=_dir, grade="?", confidence=signal.confidence, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime="UNKNOWN", result=f"REJECTED(HOURLY_TOTAL_LIMIT total={self._hourly_total}/{max_hourly})")
            return None
        elif _is_exempt and self._hourly_total >= max_hourly:
            _exempt_reason = (
                "HTF-aligned" if _is_htf_aligned_exempt
                else f"clarity({_clarity})" if _is_clarity_exempt
                else f"high-confidence({_raw_conf:.0f})"
            )
            logger.info(
                f"✅ {_exempt_reason} signal bypasses hourly limit | {signal.symbol} "
                f"{getattr(signal.direction, 'value', str(signal.direction))} | limit={self._hourly_total}/{max_hourly}"
            )

        # ── 3b. Slot-reuse cooldown ───────────────────────────
        # If a slot was just released by an expiry/invalidation, enforce a 10-second
        # window before it can be filled by another signal — prevents a rapid churn of
        # invalidations from creating a short burst that defeats the hourly cap.
        _slot_cooldown_secs = self._SLOT_REUSE_COOLDOWN_SECS
        _since_release = time.time() - self._last_slot_release_time
        if _since_release < _slot_cooldown_secs and not _is_exempt:
            logger.info(
                f"⏸️ Signal slot cooldown | {signal.symbol} "
                f"({_slot_cooldown_secs - _since_release:.0f}s remaining after last release)"
            )
            return None

        # ── 4. Fetch supporting data concurrently ────────────
        # FIX: reuse 1h OHLCV already injected into signal.raw_data by engine._scan_symbol.
        # The engine writes signal.raw_data['ohlcv_1h'] before calling aggregator.process(),
        # so a second api.fetch_ohlcv() call here is a redundant exchange round-trip.
        ohlcv_1h_cached = signal.raw_data.get('ohlcv_1h') if signal.raw_data else None
        if ohlcv_1h_cached:
            ohlcv_1h = ohlcv_1h_cached
            derivatives, order_book = await asyncio.gather(
                derivatives_analyzer.analyze(signal.symbol),
                api.fetch_order_book(signal.symbol, limit=50),
                return_exceptions=True
            )
        else:
            ohlcv_1h, derivatives, order_book = await asyncio.gather(
                api.fetch_ohlcv(signal.symbol, "1h", limit=100),
                derivatives_analyzer.analyze(signal.symbol),
                api.fetch_order_book(signal.symbol, limit=50),
                return_exceptions=True
            )

        # ── 5. Score each component ──────────────────────────
        scored = ScoredSignal(base_signal=signal)
        scored.technical_score = signal.confidence
        scored.regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        scored.derivatives_data = derivatives if not isinstance(derivatives, Exception) else None
        if signal.raw_data is None:
            signal.raw_data = {}
        _volumes_np, _closes_np, _atr_proxy = self._extract_market_arrays(ohlcv_1h)

        # Volume score
        if isinstance(ohlcv_1h, list) and ohlcv_1h:
            vol_data = volume_analyzer.analyze(ohlcv_1h, signal.entry_high)
            scored.volume_score = vol_data.score
            scored.volume_data = vol_data
            self._extend_unique_confluence(signal, vol_data.notes[:2])
            try:
                _trade_type = self._infer_volume_trade_type(signal)
                _entry_mid = (signal.entry_low + signal.entry_high) / 2
                volume_quality = volume_analyzer.assess_volume_quality(
                    ohlcv_1h,
                    _entry_mid,
                    direction=direction_str,
                    trade_type=_trade_type,
                )
                scored.volume_score = self._clamp_score(
                    scored.volume_score * 0.65
                    + volume_quality.quality_score * 0.35
                    + volume_quality.confidence_delta
                )
                self._extend_unique_confluence(signal, volume_quality.notes[:2])
                signal.raw_data.update({
                    "volume_trade_type": _trade_type,
                    "volume_quality_score": volume_quality.quality_score,
                    "volume_quality_label": volume_quality.quality_label,
                    "volume_quality_context": volume_quality.context_label,
                    "spread_bps": volume_quality.spread_bps,
                    "volume_quality_dry_volume": volume_quality.dry_volume,
                    "volume_quality_delta": volume_quality.confidence_delta,
                })
            except Exception:
                pass
        else:
            scored.volume_score = 50.0

        try:
            from analyzers.trigger_quality import trigger_quality_analyzer
            trigger_inputs = self._build_trigger_inputs(signal)
            if trigger_inputs:
                trigger_quality = trigger_quality_analyzer.analyze(
                    trigger_inputs,
                    volumes=_volumes_np,
                    closes=_closes_np,
                    atr=_atr_proxy,
                )
                scored.technical_score = self._clamp_score(
                    scored.technical_score * 0.90
                    + trigger_quality.quality_score * 100.0 * 0.10
                    + trigger_quality.confidence_delta
                )
                self._extend_unique_confluence(signal, trigger_quality.notes[:2])
                signal.raw_data.update({
                    "trigger_quality_score": trigger_quality.quality_score,
                    "trigger_quality_label": trigger_quality.quality_label,
                    "trigger_quality_effective_count": trigger_quality.effective_trigger_count,
                    "trigger_quality_volume_context": trigger_quality.volume_context,
                    "trigger_quality_fast_move": trigger_quality.fast_move,
                    "trigger_quality_delta": trigger_quality.confidence_delta,
                })
            else:
                signal.raw_data.setdefault("trigger_quality_score", 0.0)
                signal.raw_data.setdefault("trigger_quality_label", "LOW")
        except Exception:
            pass

        # Derivatives score
        if scored.derivatives_data:
            scored.derivatives_score = scored.derivatives_data.score
            is_valid, conf_adj, der_notes = derivatives_analyzer.assess_entry_validity(
                scored.derivatives_data, direction_str
            )
            if not is_valid:
                logger.info(
                    f"❌ Signal died (agg) | {signal.symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                    f"| reason=DERIVATIVES_INVALID | strategy={signal.strategy}"
                )
                return None
            signal.confluence.extend(der_notes[:2])
        else:
            scored.derivatives_score = 50.0

        # Order flow score
        # FIX: was passing signal.entry_high as the price anchor. entry_high is the TOP
        # of the entry zone — for a LONG at $100–102, the orderflow was measured from $102,
        # missing bid support stacked at $99–101. Use entry_mid for a symmetric measurement.
        if isinstance(order_book, dict) and order_book:
            _of_entry_mid = (signal.entry_low + signal.entry_high) / 2
            scored.orderflow_score = self._score_orderflow(
                order_book, _of_entry_mid, direction_str
            )
            # Wire absorption detection from spoof detector
            try:
                from analyzers.orderflow import orderflow_analyzer as _of_ref
                _spoof = _of_ref.get_spoof_detector()
                _bids_raw = order_book.get('bids', [])
                _asks_raw = order_book.get('asks', [])
                _bid_sizes = [q for _, q in _bids_raw[:20]]
                _ask_sizes = [q for _, q in _asks_raw[:20]]
                _avg_sz = (sum(_bid_sizes + _ask_sizes) / max(len(_bid_sizes + _ask_sizes), 1)
                           if _bid_sizes or _ask_sizes else 1.0)
                _avg_sz = max(_avg_sz, 1e-10)
                _wall_mult = cfg.analyzers.orderflow.get('wall_threshold_mult', 4.0)
                _bid_walls_pairs = [(p, q) for p, q in _bids_raw[:20] if q > _avg_sz * _wall_mult]
                _ask_walls_pairs = [(p, q) for p, q in _asks_raw[:20] if q > _avg_sz * _wall_mult]
                _wall_analysis = _spoof.analyze_walls(
                    _bid_walls_pairs, _ask_walls_pairs, _of_entry_mid, _avg_sz,
                )
                if _wall_analysis.absorption_detected:
                    from config.constants import SpoofingDetection as _SD
                    scored.orderflow_score = self._clamp_score(
                        scored.orderflow_score + _SD.ABSORPTION_CONF_BONUS
                    )
                    signal.raw_data.update({
                        "absorption_detected": True,
                        "absorption_side": _wall_analysis.absorption_side,
                        "absorption_touches": _wall_analysis.absorption_touches,
                    })
                    self._extend_unique_confluence(signal, _wall_analysis.notes[:1])
            except Exception:
                pass
        else:
            scored.orderflow_score = 50.0

        # ── 5b. Zero-data signal gate ────────────────────────
        # If ALL three primary scores (tech, vol, orderflow) are at neutral
        # default (50 ± 2), it means no real analyzer data fed into the
        # signal.  Publishing these "phantom confidence" signals was a major
        # source of losses in production (0% win rate).
        _neutral = Grading.NEUTRAL_SCORE
        _margin  = Grading.ZERO_DATA_MARGIN
        if (abs(scored.technical_score - _neutral) <= _margin
                and abs(scored.volume_score - _neutral) <= _margin
                and abs(scored.orderflow_score - _neutral) <= _margin):
            logger.info(
                f"❌ Signal died (agg) | {signal.symbol} "
                f"{getattr(signal.direction, 'value', str(signal.direction))} "
                f"| reason=ZERO_DATA_SIGNAL | tech={scored.technical_score:.1f} "
                f"vol={scored.volume_score:.1f} of={scored.orderflow_score:.1f}"
            )
            if _tl:
                _dir = getattr(signal.direction, 'value', str(signal.direction))
                _tl.signal(
                    symbol=signal.symbol, direction=_dir, grade="?",
                    confidence=signal.confidence,
                    entry_low=signal.entry_low, entry_high=signal.entry_high,
                    stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2,
                    rr=signal.rr_ratio, strategy=signal.strategy,
                    regime="UNKNOWN",
                    result="REJECTED(ZERO_DATA_SIGNAL)",
                )
            return None

        # ── 5c. Per-component contradiction floor ─────────────
        # The ZERO_DATA gate above only catches all-neutral scores.
        # A signal with tech=95, flow=18, deriv=18 passes it — the high technical
        # score masks the fact that derivatives and orderflow both said "no".
        # This gate requires at least ONE of (derivatives, orderflow) to show
        # meaningful confirmation.  If both are in the low zone, the composite
        # confidence is misleading regardless of how strong the technical score is.
        #
        # Floor is intentionally lenient (38) to allow slightly noisy market data
        # without over-filtering.  It only blocks genuinely contradictory setups
        # where the market's own structure (orderflow + derivatives) disagrees.
        #
        # Exempt: A+ grade signals (already require derivatives ≥ 65 via grading paths).
        _COMPONENT_FLOOR = 38.0
        _is_aplus_est = (signal.confidence or 0) >= self._GRADE_APLUS_MIN_CONF
        if (not _is_aplus_est
                and scored.derivatives_score < _COMPONENT_FLOOR
                and scored.orderflow_score < _COMPONENT_FLOOR):
            _dir_cf = getattr(signal.direction, 'value', str(signal.direction))
            logger.info(
                f"❌ Signal died (agg) | {signal.symbol} {_dir_cf} "
                f"| reason=COMPONENT_FLOOR | deriv={scored.derivatives_score:.1f} "
                f"flow={scored.orderflow_score:.1f} (both < {_COMPONENT_FLOOR}) "
                f"tech={scored.technical_score:.1f}"
            )
            if _tl:
                _tl.signal(
                    symbol=signal.symbol, direction=_dir_cf, grade="?",
                    confidence=signal.confidence,
                    entry_low=signal.entry_low, entry_high=signal.entry_high,
                    stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2,
                    rr=signal.rr_ratio, strategy=signal.strategy,
                    regime=scored.regime,
                    result=f"REJECTED(COMPONENT_FLOOR deriv={scored.derivatives_score:.0f} flow={scored.orderflow_score:.0f})",
                )
            return None

        # Sentiment score (from Fear & Greed + derivatives data)
        scored.sentiment_score = self._score_sentiment(direction_str, scored.derivatives_data)
        try:
            from analyzers.wallet_behavior import wallet_profiler
            whale_intent = wallet_profiler.get_advanced_intent(direction_str)
            whale_component = self._clamp_score(50.0 + whale_intent.confidence_delta * 4.0)
            scored.sentiment_score = self._clamp_score(
                scored.sentiment_score * 0.75 + whale_component * 0.25
            )
            whale_summary = (
                f"🐋 Whale intent: {whale_intent.intent} "
                f"({whale_intent.directional_bias.lower()}, conf {whale_intent.confidence:.0%}, Δ{whale_intent.confidence_delta:+d})"
            )
            self._extend_unique_confluence(
                signal,
                [whale_summary] + list(whale_intent.notes[:1]),
            )
            signal.raw_data.update({
                "whale_intent": whale_intent.intent,
                "whale_intent_bias": whale_intent.directional_bias,
                "whale_intent_confidence": whale_intent.confidence,
                "whale_buy_ratio": (
                    min(1.0, 0.5 + whale_intent.confidence * 0.5)
                    if whale_intent.directional_bias == "BULLISH"
                    else max(0.0, 0.5 - whale_intent.confidence * 0.5)
                    if whale_intent.directional_bias == "BEARISH"
                    else 0.5
                ),
                "whale_intent_delta": whale_intent.confidence_delta,
                "whale_intent_is_market_maker": whale_intent.is_market_maker,
                "whale_intent_is_coordinated": whale_intent.is_coordinated,
                "whale_intent_shock_active": whale_intent.shock_active,
            })
        except Exception:
            pass

        # Correlation score
        scored.correlation_score = self._score_correlation(signal.symbol)

        # ── 6. Killzone bonus ─────────────────────────────────
        in_killzone, kz_bonus = regime_analyzer.is_killzone()
        scored.is_killzone = in_killzone
        scored.killzone_bonus = kz_bonus
        if in_killzone:
            signal.confluence.append(
                f"⏰ {getattr(regime_analyzer.session, 'value', 'UNKNOWN')} killzone active (+{kz_bonus} confidence)"
            )

        # ── 7. Sector rotation adjustment ─────────────────────
        sector_adj, sector_note = rotation_tracker.get_signal_adjustment(
            signal.symbol, direction_str
        )
        scored.sector_adjustment = sector_adj
        scored.sector_note = sector_note
        if sector_note:
            signal.confluence.append(sector_note)

        # ── 7b. Candlestick pattern confluence ───────────────
        # Wire patterns/candlestick.py — previously orphaned, never called.
        # Adds confluence notes + small confidence bonus for confirming patterns.
        _candle_bonus = 0.0
        try:
            from patterns.candlestick import detect_all as _detect_candles
            _raw = signal.raw_data
            _ohlcv_raw = _raw.get('ohlcv_1h') or _raw.get('ohlcv_15m') or _raw.get('ohlcv')
            if _ohlcv_raw and len(_ohlcv_raw) >= 5:
                import numpy as _np
                _o = _np.array([c[1] for c in _ohlcv_raw], dtype=float)
                _h = _np.array([c[2] for c in _ohlcv_raw], dtype=float)
                _l = _np.array([c[3] for c in _ohlcv_raw], dtype=float)
                _c = _np.array([c[4] for c in _ohlcv_raw], dtype=float)
                _dir = direction_str
                for p in _detect_candles(_o, _h, _l, _c):
                    if p['direction'] in (_dir, 'NEUTRAL'):
                        _candle_bonus += p['strength'] * 5.0  # max ~5 pts per pattern
                        signal.confluence.append(
                            f"🕯 {p['pattern']} ({p['direction'].lower()}, "
                            f"str={p['strength']:.1f})"
                        )
        except Exception:
            _candle_bonus = 0.0

        # ── 8. Get regime-adjusted weights ────────────────────
        weights = self._get_adjusted_weights()

        # ── 9. Calculate final confidence ─────────────────────
        base_score = (
            weights['technical']    * scored.technical_score +
            weights['volume']       * scored.volume_score +
            weights['orderflow']    * scored.orderflow_score +
            weights['derivatives']  * scored.derivatives_score +
            weights['sentiment']    * scored.sentiment_score +
            weights['correlation']  * scored.correlation_score
        )

        # Normalize from weighted average to 0-100
        # The weighted average will naturally be 0-100 since inputs are 0-100

        # Apply bonuses
        final = base_score + scored.killzone_bonus + scored.sector_adjustment + _candle_bonus
        final = max(0.0, min(99.0, final))

        # BUG-7 FIX: Apply slippage penalty to confidence.
        # get_adjustment_factor() was calculated but never wired in, meaning thin
        # alts with chronic adverse slippage executed at the same confidence as BTC.
        try:
            from core.slippage_tracker import slippage_tracker as _st
            _slip_factor = _st.get_adjustment_factor(signal.symbol, direction_str)
            if _slip_factor < 1.0:
                final = final * _slip_factor
        except Exception:
            pass

        scored.final_confidence = final

        # ── 10. Apply regime minimum confidence override ──────
        # NOTE: This is a FIRST-PASS geometry filter only.
        # The engine's get_adaptive_min_confidence() gate (direction-aware,
        # F&G-corrected for HTF-aligned shorts) is the real quality gate.
        # Keep this threshold low enough that valid signals reach the engine.
        # ── Adaptive percentile-ranked threshold ─────────────────────────────
        # Instead of a flat floor, the threshold adapts to the current
        # distribution of confidence scores seen in the last 4 hours.
        # This prevents the bot from being too strict in low-signal periods
        # and too loose when many high-quality setups are available.
        #
        # Method: maintain a rolling buffer of recent signal scores.
        # Threshold = max(config_floor, 40th percentile of recent scores).
        # If few signals → floor stays at config value.
        # If many high-quality signals → floor rises to keep only top 60%.
        recent_scores = list(self._recent_score_buffer)
        if len(recent_scores) >= Grading.ADAPTIVE_FLOOR_MIN_HISTORY:
            # Enough history: use 40th percentile as adaptive floor
            adaptive_floor = float(np.percentile(recent_scores, Grading.ADAPTIVE_FLOOR_PERCENTILE))
            # Cap: never go above 68 (would kill too many signals) or below config
            min_conf = max(self._min_confidence, min(Grading.ADAPTIVE_FLOOR_MAX_CAP, adaptive_floor))
        else:
            # Not enough history: use config floor
            min_conf = self._min_confidence

        regime_override = regime_analyzer.get_min_confidence_override()
        if regime_override:
            min_conf = min(max(min_conf, regime_override), 65)

        # BUG-2 FIX: Only record scores that passed the AGG_THRESHOLD gate.
        # Previously appended BEFORE the gate, poisoning the adaptive floor percentile
        # with rejected signals and keeping the threshold artificially low.

        if final < min_conf:
            logger.info(
                f"❌ Signal died (agg) | {signal.symbol} {direction_str} "
                f"| reason=AGG_THRESHOLD | score={final:.1f}/{min_conf} "
                f"| strategy={signal.strategy}"
            )
            if _tl:
                _tl.signal(symbol=signal.symbol, direction=direction_str, grade="?", confidence=final, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime=scored.regime, tech=scored.technical_score, vol=scored.volume_score, flow=scored.orderflow_score, deriv=scored.derivatives_score, sent=scored.sentiment_score, corr=scored.correlation_score, result=f"REJECTED(AGG_THRESHOLD score={final:.1f}<{min_conf})")
            # CLARITY-SPAM FIX: record this failure so repeated clarity bypasses
            # that never clear AGG_THRESHOLD eventually get suspended.
            try:
                _cl_fail_key = (signal.symbol, direction_str)
                _existing_cf = self._clarity_failure_counts.get(_cl_fail_key)
                if _existing_cf is None:
                    self._clarity_failure_counts[_cl_fail_key] = (1, time.time())
                else:
                    _existing_count, _existing_ts = _existing_cf
                    self._clarity_failure_counts[_cl_fail_key] = (_existing_count + 1, _existing_ts)
            except Exception:
                pass
            # Feed diagnostic engine
            try:
                from core.diagnostic_engine import diagnostic_engine
                from analyzers.regime import regime_analyzer as _ra_diag
                diagnostic_engine.record_signal_death(
                    symbol=signal.symbol, direction=direction_str,
                    strategy=signal.strategy, kill_reason="AGG_THRESHOLD",
                    rr=signal.rr_ratio, confidence=final,
                    regime=getattr(_ra_diag, 'regime', type('', (), {'value': 'UNKNOWN'})()).value,
                    setup_class=getattr(signal, 'setup_class', 'intraday'),
                )
            except Exception:
                pass
            return None

        # ── 11. Grade the signal ──────────────────────────────
        # BUG-2 FIX (cont): record score AFTER the gate — only approved signals
        # should influence the adaptive floor percentile calculation.
        self._recent_score_buffer.append(final)

        scored.confluence_multiplier = getattr(signal, 'confluence_multiplier', 1.0)
        scored.grade = self._grade_signal(final, scored)
        scored.all_confluence = signal.confluence

        # ── 11b. Power alignment badge ────────────────────────
        # Cross-category strength detection — independent of grade.
        # Gated by POWER_ALIGNMENT feature flag.
        try:
            from config.feature_flags import ff
            if ff.is_active("POWER_ALIGNMENT"):
                _pa, _pa_reason = self._compute_power_alignment(scored)
                _pa_mode = "shadow" if ff.is_shadow("POWER_ALIGNMENT") else "live"
                from config.fcn_logger import fcn_log

                # v2 enhanced alignment with HTF concordance and tiers
                _htf_bias = "NEUTRAL"
                try:
                    from analyzers.htf_guardrail import htf_guardrail
                    _htf_bias = getattr(htf_guardrail, '_weekly_bias', 'NEUTRAL')
                except Exception:
                    pass

                # Derive 1H (MTF) structure bias from volume data trend
                _mtf_bias = "NEUTRAL"
                try:
                    _vol_data = scored.volume_data
                    if _vol_data and hasattr(_vol_data, 'obv_divergence'):
                        if _vol_data.obv_divergence == "BULLISH":
                            _mtf_bias = "BULLISH"
                        elif _vol_data.obv_divergence == "BEARISH":
                            _mtf_bias = "BEARISH"
                    # Strengthen with accumulation/distribution if available
                    if _vol_data and hasattr(_vol_data, 'accumulation'):
                        if _vol_data.accumulation == "ACCUMULATION" and _mtf_bias != "BEARISH":
                            _mtf_bias = "BULLISH"
                        elif _vol_data.accumulation == "DISTRIBUTION" and _mtf_bias != "BULLISH":
                            _mtf_bias = "BEARISH"
                except Exception:
                    pass

                _pa2, _pa2_reason, _pa2_tier, _pa2_score, _pa2_boost = \
                    self._compute_power_alignment_v2(
                        scored, htf_bias=_htf_bias, regime=scored.regime,
                        mtf_structure_bias=_mtf_bias,
                    )

                if ff.is_enabled("POWER_ALIGNMENT"):
                    scored.power_aligned = _pa
                    scored.power_alignment_reason = _pa_reason
                    # v2 fields
                    scored.power_alignment_tier = _pa2_tier
                    scored.power_alignment_score = _pa2_score
                    scored.power_alignment_boost = _pa2_boost
                    # Apply confidence boost from power alignment v2
                    if _pa2_boost > 0:
                        final = min(99, final + _pa2_boost)
                        scored.final_confidence = final
                        scored.grade = self._grade_signal(final, scored)
                    fcn_log("POWER_ALIGNMENT",
                            f"{_pa_mode} | {signal.symbol} aligned={_pa} "
                            f"tier={_pa2_tier} score={_pa2_score:.0f} "
                            f"boost={_pa2_boost} reason={_pa2_reason}")
                else:
                    # Shadow mode: log only, don't set on scored
                    fcn_log("POWER_ALIGNMENT",
                            f"shadow | {signal.symbol} aligned={_pa} "
                            f"tier={_pa2_tier} score={_pa2_score:.0f} "
                            f"reason={_pa2_reason}")
                    if _pa:
                        from config.shadow_mode import shadow_log
                        shadow_log("POWER_ALIGNMENT", {
                            "symbol": signal.symbol,
                            "aligned": _pa,
                            "tier": _pa2_tier,
                            "score": _pa2_score,
                            "boost": _pa2_boost,
                            "reason": _pa2_reason,
                        })
        except Exception:
            pass

        # ── 12. Update signal with final data ────────────────
        signal.final_confidence = final
        signal.regime = scored.regime
        signal.regime_time = time.time()  # Snapshot timestamp for staleness check
        signal.sector = rotation_tracker.get_sector_for_symbol(signal.symbol)
        signal.setup_context = build_setup_context(signal)
        if signal.raw_data is None:
            signal.raw_data = {}
        signal.raw_data["setup_context"] = signal.setup_context

        # ── 13a. Post-scoring dedup check (atomic check-and-mark) ────────────
        # DEDUP-FIX (CONFIDENCE CONSISTENCY): Now that we have final_confidence,
        # do the real dedup check. This compares final vs stored-final so the
        # +15 breakthrough is meaningful. First-time signals always pass (key absent).
        #
        # RACE-FIX: check_and_mark() holds an asyncio.Lock across the check AND mark
        # so two concurrent asyncio.gather tasks for the same symbol cannot both
        # pass the dedup gate before either records the mark. Previously, with
        # mark_sent deferred to after Telegram send (V17), both concurrent scan tasks
        # would see is_duplicate=False and both publish the same signal.
        if await self._deduplicator.check_and_mark(
            signal.symbol, direction_str,
            confidence=final, grade=scored.grade,
        ):
            logger.debug(
                f"Duplicate signal filtered (post-scoring): "
                f"{signal.symbol} {direction_str} final={final:.1f}"
            )
            return None

        # BEH-6/10: C grade signals are informational — don't count against
        # daily/hourly limits so they can't crowd out real A/B signals
        if scored.grade != "C":
            self._daily_count[signal.symbol] = symbol_count + 1
            # Record timestamp for rolling 24h window
            _ts_list = self._daily_count_times.get(signal.symbol, [])
            _ts_list.append(time.time())
            self._daily_count_times[signal.symbol] = _ts_list
            # FIX: Exempt signals (HTF-aligned, A+, clarity) bypass the hourly
            # cap — they must NOT inflate _hourly_total above max.  Previously
            # every approved signal incremented unconditionally, pushing the
            # counter to e.g. 15/12 and permanently blocking all non-exempt
            # signals for the rest of the hour.  Exempt signals are "off-quota".
            if self._hourly_total < self._max_per_hour:
                self._hourly_total += 1
            self._persist_hourly_counter()  # L3-FIX: persist to survive restarts

        logger.info(
            f"✅ Signal approved: {signal.symbol} {direction_str} "
            f"grade={scored.grade} confidence={final:.1f} "
            f"strategy={signal.strategy}"
        )
        if _tl:
            _tl.signal(symbol=signal.symbol, direction=direction_str, grade=scored.grade, confidence=final, entry_low=signal.entry_low, entry_high=signal.entry_high, stop_loss=signal.stop_loss, tp1=signal.tp1, tp2=signal.tp2, rr=signal.rr_ratio, strategy=signal.strategy, regime=scored.regime, tech=scored.technical_score, vol=scored.volume_score, flow=scored.orderflow_score, deriv=scored.derivatives_score, sent=scored.sentiment_score, corr=scored.correlation_score, result="APPROVED")

        return scored

    def _get_adjusted_weights(self) -> Dict[str, float]:
        """Get weights adjusted for current market regime"""
        weights_cfg = self._agg_cfg.weights
        weights = {
            'technical':   getattr(weights_cfg, 'technical', 0.30),
            'volume':      getattr(weights_cfg, 'volume', 0.15),
            'orderflow':   getattr(weights_cfg, 'orderflow', 0.18),
            'derivatives': getattr(weights_cfg, 'derivatives', 0.20),
            'sentiment':   getattr(weights_cfg, 'sentiment', 0.07),
            'correlation': getattr(weights_cfg, 'correlation', 0.10),
        }

        # Apply regime adjustments
        regime_adjustments = regime_analyzer.get_weight_adjustments()
        for key, adj in regime_adjustments.items():
            if key in weights:
                weights[key] = max(0.0, weights[key] + adj)

        # Re-normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _score_orderflow(
        self, order_book: Dict, price: float, direction: str
    ) -> float:
        """
        Score the order book for signal direction.
        Look for support walls (longs) or resistance walls (shorts).
        """
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                return Grading.NEUTRAL_SCORE

            # BUG-7 FIX: Use symmetric band around current price for both sides.
            # FIX #14: Band width is price-adaptive. For sub-penny coins (price < $0.01),
            # ±1% = only $0.0001 — a single tick — giving an empty band and forcing a
            # 50.0 neutral score for every Tier 3 signal. Widen band for low-price coins.
            if price < AggregatorScoring.MICRO_PRICE_THRESHOLD:
                band = AggregatorScoring.BAND_MICRO   # ±3% for micro-price coins
            elif price < AggregatorScoring.SUBPENNY_THRESHOLD:
                band = AggregatorScoring.BAND_SUBPENNY   # ±2% for sub-penny coins
            elif price < AggregatorScoring.NORMAL_PRICE_THRESHOLD:
                band = AggregatorScoring.BAND_CENTS  # ±1.5% for cents-range coins
            else:
                band = AggregatorScoring.BAND_NORMAL   # ±1% standard (BTC, ETH, etc.)

            nearby_bid_vol = sum(
                qty for price_level, qty in bids
                if price * (1 - band) <= price_level <= price * (1 + band)
            )
            nearby_ask_vol = sum(
                qty for price_level, qty in asks
                if price * (1 - band) <= price_level <= price * (1 + band)
            )

            total = nearby_bid_vol + nearby_ask_vol
            if total == 0:
                return 50.0

            bid_ratio = nearby_bid_vol / total

            # Wall detection — large orders relative to average
            all_bid_sizes = [qty for _, qty in bids[:20]]
            all_ask_sizes = [qty for _, qty in asks[:20]]
            avg_size = (np.mean(all_bid_sizes + all_ask_sizes)
                        if all_bid_sizes + all_ask_sizes else 1)
            wall_threshold = avg_size * cfg.analyzers.orderflow.get('wall_threshold_mult', 4.0)

            # Pass 10 FIX: was [:10] for both bid_walls and ask_walls while the average
            # was computed over [:20]. A wall at position 10-19 inflated the threshold
            # (making it harder to detect any wall) but was invisible to detection itself.
            # Use the same 20-level depth for both average and detection.
            bid_walls = [p for p, q in bids[:20] if q > wall_threshold]
            ask_walls = [p for p, q in asks[:20] if q > wall_threshold]

            score = 50.0

            if direction == "LONG":
                score += (bid_ratio - 0.5) * 40  # More bids = bullish
                if bid_walls:
                    score += 8  # Bid wall = support
                if ask_walls:
                    score -= 5  # Ask wall = resistance
            else:
                score += (0.5 - bid_ratio) * 40  # More asks = bearish
                if ask_walls:
                    score += 8
                if bid_walls:
                    score -= 5

            return max(0.0, min(100.0, score))

        except Exception:
            return Grading.NEUTRAL_SCORE

    def _score_sentiment(self, direction: str, derivatives_data=None) -> float:
        """Score based on Fear & Greed and derivatives sentiment"""
        fg = regime_analyzer.fear_greed
        score = Grading.NEUTRAL_SCORE

        extreme_fear = cfg.analyzers.sentiment.get('extreme_fear_threshold', 25)
        extreme_greed = cfg.analyzers.sentiment.get('extreme_greed_threshold', 75)

        if direction == "LONG":
            if fg < extreme_fear:
                score += Penalties.SENTIMENT_EXTREME_FEAR_LONG_BOOST  # Extreme fear = buy signal (contrarian)
            elif fg < 40:
                score += Penalties.SENTIMENT_MODERATE_FEAR_BOOST
            elif fg > extreme_greed:
                score -= Penalties.SENTIMENT_EXTREME_GREED_LONG_PENALTY  # Extreme greed = caution for longs
        else:  # SHORT
            if fg > extreme_greed:
                score += Penalties.SENTIMENT_EXTREME_GREED_SHORT_BOOST  # Extreme greed = sell signal
            elif fg > 60:
                score += Penalties.SENTIMENT_MODERATE_GREED_BOOST
            elif fg < extreme_fear:
                score -= Penalties.SENTIMENT_EXTREME_FEAR_SHORT_PENALTY  # Extreme fear = don't short into panic

        # Incorporate derivatives data — most powerful sentiment signal in crypto
        # FIX 10: was `if derivatives_data:` — a dataclass instance with all-zero/empty
        # fields evaluates as True in Python, but a zeroed DerivativesData (e.g. API
        # returned no data, all fields defaulted) is indistinguishable from valid data.
        # Use `is not None` so we only skip when the object was never created.
        if derivatives_data is not None:
            bias = derivatives_data.signal_bias
            if direction == "LONG":
                if bias in ("BULLISH", "EXTREME_SHORT"):
                    # BULLISH = funding favours longs / EXTREME_SHORT = squeeze setup
                    score += 12
                elif bias == "EXTREME_LONG":
                    # Crowd max long = contra signal for new longs
                    score -= 8
                if derivatives_data.squeeze_risk == "HIGH":
                    score += 15  # Imminent squeeze = strong long sentiment edge
            else:  # SHORT
                if bias in ("BEARISH", "EXTREME_LONG"):
                    score += 12
                elif bias == "EXTREME_SHORT":
                    score -= 8
                if derivatives_data.liquidation_risk == "HIGH":
                    score += 10  # Mass long liquidation risk = short sentiment edge

        return max(0.0, min(100.0, score))

    def _score_correlation(self, symbol: str) -> float:
        """
        Penalize if we already have a highly correlated signal active.
        Prevents double exposure to the same sector.
        """
        sector = rotation_tracker.get_sector_for_symbol(symbol)
        if not sector:
            return 60.0  # Unknown sector = neutral

        # Get rotation score for this sector
        sector_summary = rotation_tracker.get_rotation_summary()
        hot_sectors = [s for s, _ in sector_summary.get('hot', [])]
        cold_sectors = [s for s, _ in sector_summary.get('cold', [])]

        if sector in hot_sectors:
            return 70.0  # Hot sector = good correlation score
        elif sector in cold_sectors:
            return 30.0  # Cold sector = poor correlation
        return 50.0

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    @staticmethod
    def _extend_unique_confluence(signal: SignalResult, notes: List[str], limit: int = 2) -> None:
        if not notes:
            return
        existing = set(signal.confluence or [])
        added = 0
        for note in notes:
            if not note or note in existing:
                continue
            signal.confluence.append(note)
            existing.add(note)
            added += 1
            if added >= limit:
                break

    @staticmethod
    def _extract_market_arrays(ohlcv: List) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        if not isinstance(ohlcv, list) or len(ohlcv) < 2:
            return None, None, 0.0
        try:
            closes = np.array([float(c[4]) for c in ohlcv], dtype=float)
            volumes = np.array([float(c[5]) for c in ohlcv], dtype=float)
            highs = np.array([float(c[2]) for c in ohlcv[-15:]], dtype=float)
            lows = np.array([float(c[3]) for c in ohlcv[-15:]], dtype=float)
            prev_closes = closes[-15:-1] if len(closes) >= 15 else closes[:-1]
            atr_proxy = 0.0
            if len(highs) >= 2 and len(prev_closes) >= 1:
                tr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(
                        np.abs(highs[1:] - prev_closes),
                        np.abs(lows[1:] - prev_closes),
                    ),
                )
                if len(tr):
                    atr_proxy = float(np.mean(tr[-14:]))
            return volumes, closes, atr_proxy
        except Exception:
            return None, None, 0.0

    @staticmethod
    def _infer_volume_trade_type(signal: SignalResult) -> str:
        raw_data = signal.raw_data or {}
        regime_trade_type = str(raw_data.get("regime_trade_type", ""))
        strategy = str(getattr(signal, "strategy", ""))

        if strategy == "InstitutionalBreakout":
            return "breakout"
        if regime_trade_type.startswith(("PULLBACK_", "LOCAL_CONTINUATION_")):
            return "pullback"
        if strategy in {"ExtremeReversal", "MeanReversion"}:
            return "reversal"
        if strategy == "RangeScalper":
            return "range"
        return ""

    @staticmethod
    def _build_trigger_inputs(signal: SignalResult):
        """Build TriggerQuality inputs from a live signal's raw_data and confluence.

        Raw-data keys provide the strongest hints for structure/momentum/derivatives
        categories, while human-readable confluence notes are pattern-matched into
        structure, momentum, volume, derivatives, or orderflow buckets. The result is
        a deduplicated trigger list suitable for TriggerQualityAnalyzer.analyze().
        """
        try:
            from analyzers.trigger_quality import (
                CATEGORY_DERIVATIVES,
                CATEGORY_MOMENTUM,
                CATEGORY_ORDERFLOW,
                CATEGORY_STRUCTURE,
                CATEGORY_VOLUME,
                Trigger,
            )
        except Exception:
            return []

        raw_data = signal.raw_data or {}
        confluence = list(getattr(signal, "confluence", []) or [])
        lower_notes = [str(note).lower() for note in confluence]
        vol_ratio = float(raw_data.get("vol_ratio", 0.0) or 0.0)
        volume_confirmed = (
            vol_ratio >= 1.5
            or any("volume" in note for note in lower_notes)
        )

        def _add_trigger(name: str, category: str, strength: float, confirmed: bool = False):
            triggers.append(
                Trigger(
                    name=name,
                    category=category,
                    raw_strength=max(0.2, min(1.0, strength)),
                    volume_confirmed=confirmed or volume_confirmed,
                    volume_mult=max(1.0, vol_ratio) if vol_ratio > 0 else 1.0,
                )
            )

        triggers = []

        structure_keys = (
            "primary_pattern", "bos_level", "channel_high", "channel_low", "key_level",
            "wyckoff_phase", "range_high", "range_low", "wave1_start", "ob_low",
            "ob_high", "fvg_low", "fvg_high", "sweep_low", "sweep_high",
        )
        momentum_keys = ("macd", "rsi", "adx", "z_score", "tenkan", "kijun")
        derivatives_keys = ("funding_rate", "oi_change")

        for key in structure_keys:
            if key in raw_data:
                _add_trigger(f"raw:{key}", CATEGORY_STRUCTURE, 0.85)
        for key in momentum_keys:
            if key in raw_data:
                _add_trigger(f"raw:{key}", CATEGORY_MOMENTUM, 0.75)
        for key in derivatives_keys:
            if key in raw_data:
                _add_trigger(f"raw:{key}", CATEGORY_DERIVATIVES, 0.8)

        for note, lowered in zip(confluence, lower_notes):
            strength = 0.85 if "✅" in note else 0.65 if "📊" in note else 0.5
            if any(token in lowered for token in ("volume", "vwap", "obv", "accumulation", "distribution")):
                _add_trigger(f"note:{lowered[:32]}", CATEGORY_VOLUME, strength, confirmed=True)
            elif any(token in lowered for token in ("funding", "oi ", "oi:", "liquidation", "squeeze")):
                _add_trigger(f"note:{lowered[:32]}", CATEGORY_DERIVATIVES, strength)
            elif any(token in lowered for token in ("orderflow", "wall", "bid", "ask", "delta")):
                _add_trigger(f"note:{lowered[:32]}", CATEGORY_ORDERFLOW, strength)
            elif any(token in lowered for token in ("rsi", "macd", "adx", "tenkan", "kijun", "z-score", "stoch")):
                _add_trigger(f"note:{lowered[:32]}", CATEGORY_MOMENTUM, strength)
            elif any(token in lowered for token in (
                "pattern", "bos", "breakout", "breakdown", "wave", "wyckoff",
                "range", "rejection", "liquidity sweep", "fvg", "ob ", "key level",
            )):
                _add_trigger(f"note:{lowered[:32]}", CATEGORY_STRUCTURE, strength)

        deduped = {}
        for trigger in triggers:
            key = (trigger.name, trigger.category)
            if key not in deduped or deduped[key].raw_strength < trigger.raw_strength:
                deduped[key] = trigger
        return list(deduped.values())

    def _grade_signal(self, confidence: float, scored: ScoredSignal) -> str:
        """
        Grade signal quality — Trading Assistant Edition.

        A+ paths (institutional-grade, act immediately):
          Path 1: ≥88 conf + killzone + good derivatives + volume
          Path 2: ≥90 conf + strong derivatives + strong volume (killzone not required)
          Path 3: ≥85 conf + multi-cluster confluence (confluence_mult ≥ 1.15)
          Path 4: ≥88 conf + best regime (BULL_TREND for LONG, BEAR_TREND for SHORT)

        A: ≥80 conf — high conviction, enter on confirmation candle
        B: ≥72 conf — solid setup, monitor 1-2 candles
        C:  <72    — context signal only
        """
        # Confluence multiplier stored on scored if available
        conf_mult = getattr(scored, 'confluence_multiplier', 1.0)

        # Path 1: Classic A+ — high conf + killzone + derivatives + volume
        if (confidence >= self._GRADE_APLUS_MIN_CONF
                and scored.is_killzone
                and scored.derivatives_score >= Grading.APLUS_PATH1_DERIVATIVES_MIN
                and scored.volume_score >= Grading.APLUS_PATH1_VOLUME_MIN):
            return "A+"

        # Path 2: Very high conf + strong signals (no killzone required)
        if (confidence >= Grading.APLUS_PATH2_CONF_MIN
                and scored.derivatives_score >= Grading.APLUS_PATH2_DERIVATIVES_MIN
                and scored.volume_score >= Grading.APLUS_PATH2_VOLUME_MIN):
            return "A+"

        # Path 3: Multi-cluster confluence boost (≥2 independent clusters agreed)
        if confidence >= Grading.APLUS_PATH3_CONF_MIN and conf_mult >= Grading.APLUS_PATH3_CONFLUENCE_MULT_MIN:
            return "A+"

        # Path 4: High conf in aligned regime
        regime = getattr(scored, 'regime', '')
        # FIX: must verify signal direction aligns with regime — otherwise a SHORT in
        # BULL_TREND at conf ≥ 88 would get A+ and bypass the execution engine entirely.
        _direction = getattr(scored.base_signal.direction, 'value', str(scored.base_signal.direction))
        _regime_aligned = (
            (regime == 'BULL_TREND' and _direction == 'LONG') or
            (regime == 'BEAR_TREND' and _direction == 'SHORT')
        )
        if (confidence >= self._GRADE_APLUS_MIN_CONF
                and _regime_aligned
                and scored.volume_score >= 62):
            return "A+"

        if confidence >= self._GRADE_A_MIN_CONF:
            return "A"
        if confidence >= self._GRADE_B_MIN_CONF:
            return "B"
        return "C"

    @staticmethod
    def _compute_power_alignment(scored: ScoredSignal) -> tuple:
        """
        Detect cross-category power alignment — regime-agnostic.

        Returns (is_aligned: bool, reason: str).

        Power alignment means the signal has meaningful confirmation from
        multiple independent scoring categories.  This is orthogonal to
        the overall grade: a B-grade signal can be power-aligned when its
        individual components are all above-average, even if the weighted
        composite didn't reach A+ thresholds.  Conversely, an A-grade
        signal driven by a single dominant category won't earn the badge.
        """
        hits = []
        if scored.technical_score >= Grading.POWER_ALIGN_STRUCTURE_MIN:
            hits.append("structure")
        if scored.volume_score >= Grading.POWER_ALIGN_VOLUME_MIN:
            hits.append("volume")
        if scored.derivatives_score >= Grading.POWER_ALIGN_DERIVATIVES_MIN:
            hits.append("derivatives")
        if scored.orderflow_score >= Grading.POWER_ALIGN_ORDERFLOW_MIN:
            hits.append("orderflow")
        # Context: sentiment OR correlation above neutral confirms broader tailwind
        if scored.sentiment_score >= 55 or scored.correlation_score >= 60:
            hits.append("context")

        aligned = len(hits) >= Grading.POWER_ALIGN_MIN_CATEGORIES
        if aligned:
            reason = f"{len(hits)}/5 categories confirmed ({', '.join(hits)})"
        else:
            reason = ""
        return aligned, reason

    @staticmethod
    def _compute_power_alignment_v2(
        scored: ScoredSignal,
        htf_bias: str = "NEUTRAL",
        regime: str = "UNKNOWN",
        mtf_structure_bias: str = "NEUTRAL",
    ) -> tuple:
        """
        Enhanced power alignment v2 — weighted scoring, HTF concordance,
        MTF structure alignment, regime-aware thresholds, and strength tier.

        Parameters
        ----------
        scored : ScoredSignal
        htf_bias : str — 4H directional bias (BULLISH/BEARISH/NEUTRAL)
        regime : str — current regime label
        mtf_structure_bias : str — 1H structure bias (BULLISH/BEARISH/NEUTRAL)

        Returns (is_aligned: bool, reason: str, tier: str, score: float, boost: int).
        """
        # ── 1. Category hit detection (same as v1 for backward compat)
        hits = []
        if scored.technical_score >= Grading.POWER_ALIGN_STRUCTURE_MIN:
            hits.append("structure")
        if scored.volume_score >= Grading.POWER_ALIGN_VOLUME_MIN:
            hits.append("volume")
        if scored.derivatives_score >= Grading.POWER_ALIGN_DERIVATIVES_MIN:
            hits.append("derivatives")
        if scored.orderflow_score >= Grading.POWER_ALIGN_ORDERFLOW_MIN:
            hits.append("orderflow")
        if scored.sentiment_score >= 55 or scored.correlation_score >= 60:
            hits.append("context")

        aligned = len(hits) >= Grading.POWER_ALIGN_MIN_CATEGORIES

        if not aligned:
            return False, "", "", 0.0, 0

        # ── 2. Weighted composite power score ──────────────────
        # Normalize each score from threshold to 100 (threshold=0, 100=1)
        # This ensures a signal that just meets thresholds gets ~50 composite
        # and one with all scores near 100 gets ~100
        struct_norm = max(0, scored.technical_score - Grading.POWER_ALIGN_STRUCTURE_MIN) / (100 - Grading.POWER_ALIGN_STRUCTURE_MIN)
        vol_norm = max(0, scored.volume_score - Grading.POWER_ALIGN_VOLUME_MIN) / (100 - Grading.POWER_ALIGN_VOLUME_MIN)
        deriv_norm = max(0, scored.derivatives_score - Grading.POWER_ALIGN_DERIVATIVES_MIN) / (100 - Grading.POWER_ALIGN_DERIVATIVES_MIN)
        of_norm = max(0, scored.orderflow_score - Grading.POWER_ALIGN_ORDERFLOW_MIN) / (100 - Grading.POWER_ALIGN_ORDERFLOW_MIN)
        ctx_score = max(scored.sentiment_score, scored.correlation_score)
        ctx_norm = max(0, ctx_score - 50) / 50.0

        # Base score of 50 since we already passed the category gate
        raw_power = (
            struct_norm * Grading.POWER_ALIGN_WEIGHT_STRUCTURE +
            vol_norm * Grading.POWER_ALIGN_WEIGHT_VOLUME +
            deriv_norm * Grading.POWER_ALIGN_WEIGHT_DERIVATIVES +
            of_norm * Grading.POWER_ALIGN_WEIGHT_ORDERFLOW +
            ctx_norm * Grading.POWER_ALIGN_WEIGHT_CONTEXT
        )
        power_score = 50.0 + raw_power * 50.0  # Range: 50-100

        # ── 3. HTF concordance adjustment ──────────────────────
        direction = getattr(scored.base_signal, 'direction', None)
        dir_str = direction.value if hasattr(direction, 'value') else str(direction or "")
        htf_aligned = False
        if htf_bias != "NEUTRAL" and dir_str:
            htf_aligned = (
                (dir_str == "LONG" and htf_bias == "BULLISH") or
                (dir_str == "SHORT" and htf_bias == "BEARISH")
            )
            if htf_aligned:
                power_score += Grading.POWER_ALIGN_HTF_CONCORDANCE_BONUS
            else:
                power_score += Grading.POWER_ALIGN_HTF_CONFLICT_PENALTY

        # ── 3b. MTF Structure Alignment ───────────────────────
        # 4H (HTF) sets direction, 1H (MTF) confirms structure, entry confirms timing
        # Full alignment = all three agree → bonus
        # MTF conflicts with HTF/entry → penalty
        mtf_aligned = False
        if mtf_structure_bias != "NEUTRAL" and dir_str:
            mtf_aligned = (
                (dir_str == "LONG" and mtf_structure_bias == "BULLISH") or
                (dir_str == "SHORT" and mtf_structure_bias == "BEARISH")
            )
            if mtf_aligned and htf_aligned:
                # Full 4H→1H→entry alignment
                power_score += Grading.MTF_ALIGNMENT_BONUS
            elif not mtf_aligned:
                # 1H structure opposes direction
                power_score += Grading.MTF_CONFLICT_PENALTY

        # ── 4. Regime-aware threshold adjustment ──────────────
        effective_strong = Grading.POWER_ALIGN_TIER_STRONG
        effective_moderate = Grading.POWER_ALIGN_TIER_MODERATE

        if regime == "CHOPPY":
            effective_strong += Grading.POWER_ALIGN_CHOPPY_PENALTY
            effective_moderate += Grading.POWER_ALIGN_CHOPPY_PENALTY
        elif regime in ("BULL_TREND", "BEAR_TREND"):
            # With-trend gets lower bar
            if htf_aligned:
                effective_strong += Grading.POWER_ALIGN_TREND_BONUS
                effective_moderate += Grading.POWER_ALIGN_TREND_BONUS

        power_score = max(0.0, min(100.0, power_score))

        # ── 5. Strength tier classification ───────────────────
        if power_score >= effective_strong:
            tier = "STRONG"
            boost = Grading.POWER_ALIGN_CONF_BOOST_STRONG
        elif power_score >= effective_moderate:
            tier = "MODERATE"
            boost = Grading.POWER_ALIGN_CONF_BOOST_MODERATE
        else:
            tier = ""
            boost = 0

        reason = (
            f"{len(hits)}/5 categories confirmed ({', '.join(hits)}) — "
            f"{tier or 'ALIGNED'} (score={power_score:.0f})"
        )
        if htf_bias != "NEUTRAL":
            reason += f" HTF={'✅' if htf_aligned else htf_bias}"
        if mtf_structure_bias != "NEUTRAL":
            reason += f" MTF={'✅' if mtf_aligned else '⚠️'}"

        return True, reason, tier, power_score, boost

    def release_hourly_slot(self):
        """
        Decrement the hourly counter when a watched signal is invalidated or
        expires before entering — freeing the slot for a replacement signal.
        Stamps _last_slot_release_time so process() can enforce a reuse cooldown.
        """
        if self._hourly_total > 0:
            self._hourly_total -= 1
            self._last_slot_release_time = time.time()
            self._persist_hourly_counter()
            logger.debug(f"Hourly slot released — counter now {self._hourly_total}/{self._max_per_hour}")

    def _refresh_hourly_counter(self):
        """Reset hourly counter if an hour has passed"""
        if time.time() - self._hour_reset_time >= 3600:
            self._hourly_total = 0
            self._hour_reset_time = time.time()
            self._persist_hourly_counter()  # L3-FIX: persist reset to DB

    def reset_daily_counts(self):
        """Call at midnight to reset daily signal counts"""
        self._daily_count = {}
        self._deduplicator.clear_expired()


# ── Singleton ──────────────────────────────────────────────

# ============================================================
# Global Aggregator Instance (singleton)
# ============================================================
signal_aggregator = SignalAggregator()
