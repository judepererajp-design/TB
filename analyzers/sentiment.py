"""
TitanBot Pro — Sentiment Analyzer
====================================
Fetches and interprets market sentiment data.
Primary source: Alternative.me Fear & Greed Index (free, no API key).

Fear & Greed scale:
  0-24  : Extreme Fear   → contrarian BUY signal
  25-44 : Fear           → cautious longs
  45-55 : Neutral
  56-74 : Greed          → cautious shorts
  75-100: Extreme Greed  → contrarian SELL signal
"""

import asyncio
import logging
import math
import time
from typing import Dict, Optional

import aiohttp

from analyzers._common.stats import blom_percentile as _blom_percentile
from config.constants import NewsIntelligence, Sentiment as SentimentConst
from config.feature_flags import ff
from config.shadow_mode import shadow_log
from config.loader import cfg

logger = logging.getLogger(__name__)


def compute_fear_greed_raw_adjustment(fear_greed: int, direction: str) -> float:
    """
    Return the raw Fear & Greed adjustment for contextual-cap composition.
    """
    max_adj = NewsIntelligence.FEAR_GREED_MAX_ADJUSTMENT_PCT
    extreme_fear = NewsIntelligence.FEAR_GREED_EXTREME_FEAR_THRESHOLD
    extreme_greed = NewsIntelligence.FEAR_GREED_EXTREME_GREED_THRESHOLD
    adjustment = 0.0

    if direction == "LONG" and fear_greed < extreme_fear:
        severity = (extreme_fear - fear_greed) / extreme_fear
        adjustment = -(0.06 + severity * 0.09)  # Steeper: -6% base up to -15%
    elif direction == "SHORT" and fear_greed > extreme_greed:
        severity = (fear_greed - extreme_greed) / (100 - extreme_greed)
        adjustment = -(0.06 + severity * 0.09)  # Steeper: -6% base up to -15%

    return max(-max_adj, min(max_adj, adjustment))


class SentimentAnalyzer:
    """
    Fear & Greed index with decay-weighted history and percentile normalisation.

    Improvements over raw F&G:
    1. Exponential decay: recent readings weighted more than old ones
       (EWMA with half-life = 6 hours → yesterday's panic fades by morning)
    2. Percentile rank vs 30-day history: F&G=25 in a bear market is normal;
       F&G=25 in a bull market is an extreme event. Context matters.
    3. Velocity: rate of change (d(F&G)/dt) predicts inflection points
       better than the raw level — a falling F&G is more bearish than a
       stable low reading.
    """

    def __init__(self):
        self._fear_greed      = 50          # Current reading
        self._label           = "Neutral"
        self._last_update     = 0.0
        self._cache_ttl       = cfg.cache.get('sentiment_ttl', 1800)
        self._api_url         = "https://api.alternative.me/fng/?limit=30"  # 30 days
        self._lock            = asyncio.Lock()
        # History for percentile and velocity calculations
        self._history: list   = []          # list of (timestamp, value) tuples
        self._ewma: float     = 50.0        # Exponentially weighted moving avg
        self._ewma_alpha: float = 0.15      # EWMA decay: ~6h half-life on 30min updates
        self._velocity: float  = 0.0        # Rate of change (F&G units/hour)
        self._pct_rank: float  = 0.5        # Percentile rank vs 30d history (0-1)

    @property
    def fear_greed(self) -> int:
        return self._fear_greed

    @property
    def label(self) -> str:
        return self._label

    @property
    def is_extreme_fear(self) -> bool:
        return self._fear_greed <= cfg.analyzers.sentiment.get('extreme_fear_threshold', 25)

    @property
    def is_extreme_greed(self) -> bool:
        return self._fear_greed >= cfg.analyzers.sentiment.get('extreme_greed_threshold', 75)

    async def update(self):
        """Fetch Fear & Greed with 30-day history for percentile + velocity."""
        if not cfg.analyzers.sentiment.get('fear_greed_enabled', True):
            return

        async with self._lock:
            if time.time() - self._last_update < self._cache_ttl:
                return

            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(self._api_url) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            entries = data.get('data', [{}])
                            if not entries:
                                return

                            # Latest reading — CLAMP to the [0, 100] API contract
                            # before it contaminates the EWMA / history. Upstream
                            # error responses have historically returned e.g. 255
                            # or -1 (int cast of a string).
                            latest = entries[0]
                            try:
                                raw_val = int(latest['value'])
                            except (TypeError, ValueError, KeyError):
                                return
                            new_val = max(
                                SentimentConst.API_MIN,
                                min(SentimentConst.API_MAX, raw_val),
                            )
                            self._fear_greed = new_val
                            self._label = latest.get('value_classification', 'Neutral')

                            # Build history (newest first from API, reverse to oldest first)
                            now = time.time()
                            _history = []
                            for i, e in enumerate(entries):
                                _v = e.get('value')
                                if _v is None:
                                    continue
                                try:
                                    _iv = int(_v)
                                except (TypeError, ValueError):
                                    continue
                                _iv = max(
                                    SentimentConst.API_MIN,
                                    min(SentimentConst.API_MAX, _iv),
                                )
                                _history.append((now - i * 86400, _iv))
                            self._history = _history

                            # Update EWMA — guard against NaN/Inf from a
                            # malformed upstream response (already clamped
                            # above, but defend against prior _ewma state
                            # that may have drifted).
                            if not math.isfinite(self._ewma):
                                self._ewma = float(new_val)
                            _next_ewma = (
                                self._ewma_alpha * new_val
                                + (1 - self._ewma_alpha) * self._ewma
                            )
                            if math.isfinite(_next_ewma):
                                self._ewma = _next_ewma
                            else:
                                self._ewma = float(new_val)

                            # Velocity: change vs 24h ago (index 1 = yesterday)
                            if len(self._history) >= 2:
                                prev_24h = self._history[1][1]
                                self._velocity = (new_val - prev_24h) / 24  # per hour

                            # Percentile rank vs 30d history using the
                            # Blom plotting-position formula — handles
                            # ties, small-sample bias and NaN input
                            # correctly (the previous searchsorted hack
                            # mis-ranked ties and returned 1.0 on an
                            # all-equal history).
                            if len(self._history) >= 5:
                                vals = [v for _, v in self._history]
                                _pct = _blom_percentile(new_val, vals, default=0.5)
                                self._pct_rank = float(_pct) if _pct is not None else 0.5

                            self._last_update = now
                            logger.debug(
                                f"F&G: {self._fear_greed} ({self._label}) "
                                f"EWMA={self._ewma:.1f} vel={self._velocity:+.2f}/h "
                                f"pct={self._pct_rank:.0%}"
                            )
                        else:
                            logger.warning(f"F&G API HTTP {resp.status} — keeping cached {self._fear_greed}")
            except Exception as e:
                logger.warning(f"F&G fetch failed: {e} — cached={self._fear_greed}")

    def score_for_direction(self, direction: str) -> float:
        """
        Sentiment score for a trade direction using EWMA + velocity + percentile rank.

        Three components:
        1. Level score: where F&G sits (contrarian: fear=buy, greed=sell)
        2. Velocity adjustment: fast-falling F&G = bearish, fast-rising = bullish
        3. Percentile adjustment: extreme percentile rank amplifies the level signal

        Returns 0-100 where 50=neutral, 80+=strong alignment.
        """
        fg   = self._ewma       # EWMA (decay-weighted) is more stable than raw
        vel  = self._velocity   # per-hour rate of change
        pct  = self._pct_rank   # 0-1 percentile vs 30d history

        extreme_fear  = cfg.analyzers.sentiment.get('extreme_fear_threshold', 25)
        extreme_greed = cfg.analyzers.sentiment.get('extreme_greed_threshold', 75)

        # ── Base score (contrarian) ─────────────────────────────
        if direction == "LONG":
            if fg <= extreme_fear:   base = 80.0
            elif fg <= 40:           base = 65.0
            elif fg >= extreme_greed: base = 32.0
            else:                    base = 50.0
        else:
            if fg >= extreme_greed:  base = 80.0
            elif fg >= 60:           base = 65.0
            elif fg <= extreme_fear: base = 30.0
            else:                    base = 50.0

        # ── Velocity adjustment (momentum of sentiment) ─────────
        # If F&G is falling fast, that's bearish (even if currently high)
        # If F&G is rising fast, that's bullish (even if currently low)
        #
        # AUDIT FIX: the old code was `min(8, vel * 4)` which is *not*
        # a symmetric cap — a velocity of -10 for LONG produced -40
        # (passes through min because -40 < 8), blowing the bound.
        # Use `max(-cap, min(cap, signed))` so the magnitude is clamped
        # on both sides while the sign is preserved.
        vel_adj = 0.0
        _vel_cap = SentimentConst.VELOCITY_CAP_PTS
        _vel_slope = SentimentConst.VELOCITY_PTS_PER_UNIT
        if abs(vel) > 0.5 and math.isfinite(vel):  # Meaningful velocity
            if direction == "LONG":
                vel_adj = max(-_vel_cap, min(_vel_cap, vel * _vel_slope))
            else:
                vel_adj = max(-_vel_cap, min(_vel_cap, -vel * _vel_slope))

        # ── Percentile amplification ────────────────────────────
        # Being at an extreme percentile amplifies the contrarian signal
        pct_extreme = max(pct, 1.0 - pct)  # 0.5=middle, 1.0=all-time extreme
        pct_bonus = (pct_extreme - 0.5) * 20  # up to +10 at 100th percentile

        return min(95.0, max(25.0, base + vel_adj + pct_bonus))

    @property
    def ewma(self) -> float:
        """EWMA-smoothed Fear & Greed (more stable than raw value)."""
        return self._ewma

    @property
    def velocity(self) -> float:
        """Rate of change in F&G units per hour."""
        return self._velocity

    @property
    def percentile_rank(self) -> float:
        """Percentile rank vs 30-day history (0=lowest, 1=highest)."""
        return self._pct_rank

    def get_summary(self) -> Dict:
        return {
            'value':           self._fear_greed,
            'label':           self._label,
            'is_extreme_fear': self.is_extreme_fear,
            'is_extreme_greed':self.is_extreme_greed,
        }

    # ── Feature 7: Fear & Greed Regime Overlay ─────────────────
    def get_regime_threshold_adjustment(self, direction: str) -> float:
        """
        Returns a threshold adjustment multiplier based on F&G regime.

        Rules:
          - Extreme Fear (F&G < 20): tighten LONG entry thresholds by 5-10%
          - Extreme Greed (F&G > 80): tighten SHORT entry thresholds by 5-10%

        Adjustment is ±5-10%, never larger. This is a context signal —
        it must NEVER independently trigger a trade.

        Returns:
            Float multiplier (e.g. 0.93 = 7% tighter, 1.0 = no adjustment).
            Always returns 1.0 if feature flag is off.
        """
        if not ff.is_active("FEAR_GREED"):
            return 1.0

        fg = self._fear_greed
        adjustment = compute_fear_greed_raw_adjustment(fg, direction)
        multiplier = 1.0 + adjustment

        if ff.is_shadow("FEAR_GREED"):
            shadow_log("FEAR_GREED", {
                "fear_greed": fg,
                "direction": direction,
                "adjustment": adjustment,
                "multiplier": multiplier,
            })
            from config.fcn_logger import fcn_log
            fcn_log("FEAR_GREED", f"shadow | F&G={fg} dir={direction} adj={adjustment:+.3f} mult={multiplier:.3f}")
            return 1.0  # Shadow: no live effect

        from config.fcn_logger import fcn_log
        fcn_log("FEAR_GREED", f"live | F&G={fg} dir={direction} adj={adjustment:+.3f} mult={multiplier:.3f}")
        return multiplier

    def get_raw_adjustment(self, direction: str) -> float:
        """
        Returns the raw adjustment amount (not multiplier) for use in
        global contextual confidence cap enforcement.
        E.g. -0.07 means 7% tighter.  Returns 0.0 if feature is off.
        """
        if not ff.is_active("FEAR_GREED") or not ff.is_enabled("FEAR_GREED"):
            return 0.0

        return compute_fear_greed_raw_adjustment(self._fear_greed, direction)

    def get_7d_trend(self) -> str:
        """
        Returns 7-day trend direction: 'rising', 'falling', or 'flat'.
        Based on velocity and history comparison.
        """
        if len(self._history) < 2:
            return "flat"
        latest = self._history[0][1] if self._history else self._fear_greed
        oldest_7d = self._history[-1][1] if self._history else self._fear_greed
        diff = latest - oldest_7d
        if diff > 5:
            return "rising"
        if diff < -5:
            return "falling"
        return "flat"


# ── Singleton ──────────────────────────────────────────────
sentiment_analyzer = SentimentAnalyzer()
