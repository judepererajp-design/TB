"""
TitanBot Pro — HTF Weekly Guardrail
=====================================
"Don't fight the higher timeframe."

PHASE 1 FIXES (P1-A, P1-B):
  - P1-A: warm_up() forces a blocking _refresh() at startup before the first scan.
           Eliminates the cold-start race where ADX=0.0 triggered the soft penalty
           path instead of the hard block, allowing the SEI/USDT loss to slip through.
  - P1-B: is_hard_blocked() is a new method called by engine.py BEFORE the aggregator
           runs. This makes the HTF check a true pre-filter gate, not a post-approval
           penalty. The old check() method is kept for backward compat (soft penalty path
           for non-FundingRateArb strategies that legitimately counter-trade).
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.api_client import api

logger = logging.getLogger(__name__)


@dataclass
class HTFGuardrailResult:
    blocked: bool
    penalty: int
    weekly_bias: str
    weekly_adx: float
    reason: str


class HTFWeeklyGuardrail:
    """
    Checks weekly trend before allowing counter-trend signals.
    Runs on weekly OHLCV data with 4-hour cache.

    P1-A fix: warm_up() must be called during engine startup.
    P1-B fix: is_hard_blocked() is the pre-aggregation hard gate.
    """

    def __init__(self):
        self._weekly_bias: str = "NEUTRAL"
        self._weekly_adx: float = 0.0
        self._weekly_ma_slope: float = 0.0
        self._last_update: float = 0
        self._cache_ttl: int = 1800  # V14: 30 min (was 4h — too slow for bounce detection)
        self._lock = asyncio.Lock()
        self._warmed: bool = False  # P1-A: track whether warm_up() has been called
        self._btc_4h_bouncing: bool = False    # V14: BTC 4h making higher lows
        self._btc_4h_resuming_down: bool = False  # V14: BTC 4h resuming bearish

        # Stablecoin-linked dynamic ADX threshold adjustment
        self._stablecoin_adx_offset: int = 0   # Applied to base 25 ADX threshold

        # Session block counters — reset each bot restart, used by Sentinel dashboard
        self._blocks_long_session:  int = 0
        self._blocks_short_session: int = 0

    # ── P1-A: Startup warm-up ─────────────────────────────────────────────────
    async def warm_up(self):
        """
        PHASE 1 FIX (P1-A): Force a blocking refresh before the first scan loop.
        Must be called in engine.py startup AFTER exchange connection is confirmed.
        Prevents cold-start race where ADX=0.0 bypasses the hard block path.
        """
        logger.info("HTF guardrail warming up (blocking refresh)...")
        try:
            await self._refresh()
            self._warmed = True
            logger.info(
                f"✅ HTF guardrail warmed: bias={self._weekly_bias}, "
                f"adx={self._weekly_adx:.1f}"
            )
        except Exception as e:
            # Even on failure, mark warmed so the bot doesn't hang.
            # Use NEUTRAL as safe fallback — no signals will be hard-blocked.
            self._warmed = True
            logger.warning(f"HTF guardrail warm-up failed (using NEUTRAL fallback): {e}")

    # ── P1-B: Pre-aggregation hard gate ──────────────────────────────────────
    def is_hard_blocked(
        self,
        signal_direction: str,
        raw_confidence: float,
        strategy_name: str,
    ) -> tuple:
        """
        PHASE 1 FIX (P1-B): Hard gate called BEFORE the aggregator.
        Returns (blocked: bool, reason: str).

        Direction handling by state (fixes the deadlock):

        ┌────────────────────────────────┬──────────────┬─────────────────────────────────────────┐
        │ State                          │ LONG         │ SHORT                                   │
        ├────────────────────────────────┼──────────────┼─────────────────────────────────────────┤
        │ Weekly BEARISH, no 4h bounce   │ Hard block   │ FREE (aligned with weekly)              │
        │                                │ (need 82+)   │                                         │
        ├────────────────────────────────┼──────────────┼─────────────────────────────────────────┤
        │ Weekly BEARISH + 4h bouncing   │ Softer block │ FREE (BEST setup: weekly trend + fade   │
        │                                │ (need 78+)   │ rally into resistance) — soft penalty   │
        │                                │              │ applied in aggregator, not hard block.  │
        ├────────────────────────────────┼──────────────┼─────────────────────────────────────────┤
        │ Weekly BULLISH + 4h declining  │ FREE         │ Soft penalty only, not hard block       │
        ├────────────────────────────────┼──────────────┼─────────────────────────────────────────┤
        │ Weekly NEUTRAL                 │ FREE         │ FREE                                    │
        └────────────────────────────────┴──────────────┴─────────────────────────────────────────┘

        Deadlock prevention: if BOTH directions would be hard-blocked simultaneously,
        the system is misconfigured. We log it and release the trend-aligned direction.
        """
        if not self._warmed:
            return False, "HTF guardrail: not warmed — allowing"

        if self._weekly_bias == "NEUTRAL":
            # When weekly is NEUTRAL the original guardrail allowed everything.
            # P0-B FIX: when the intraday regime is strongly directional we apply
            # a lightweight confidence threshold so counter-trend signals face at
            # least some structural gate instead of only strategy-level penalties.
            #
            # Logic: if intraday = BULL_TREND and signal = SHORT (counter-trend),
            #        or intraday = BEAR_TREND and signal = LONG, require a minimum
            #        confidence of 78 (the lowest bar used in any weekly-biased path).
            #        Signals at or above that pass freely; below that are blocked.
            # This closes the "P0-B weakest gate" without turning NEUTRAL into a
            # fully-biased weekly state — the threshold is lower than the BULLISH/
            # BEARISH paths (78 vs 81-88) to reflect genuine weekly ambiguity.
            try:
                from analyzers.regime import regime_analyzer as _htf_ra
                _intraday = getattr(_htf_ra.regime, 'value', 'UNKNOWN')
            except (ImportError, AttributeError):
                _intraday = 'UNKNOWN'

            _is_counter_intraday = (
                (signal_direction == "SHORT" and _intraday == "BULL_TREND") or
                (signal_direction == "LONG"  and _intraday == "BEAR_TREND")
            )
            if _is_counter_intraday:
                _neutral_threshold = 78.0  # Lowest bar — weekly is ambiguous
                if raw_confidence < _neutral_threshold:
                    reason = (
                        f"🚫 HTF weekly NEUTRAL + intraday {_intraday}: "
                        f"{signal_direction} blocked — "
                        f"need {_neutral_threshold:.0f}+ conf (got {raw_confidence:.0f}). "
                        f"Use strategy-level overrides or wait for weekly bias to firm up."
                    )
                    if signal_direction == "LONG":
                        self._blocks_long_session += 1
                    else:
                        self._blocks_short_session += 1
                    return True, reason
                return False, (
                    f"HTF NEUTRAL+{_intraday} counter-trend override: "
                    f"conf={raw_confidence:.0f} >= {_neutral_threshold:.0f}"
                )
            return False, "HTF guardrail: NEUTRAL — regime-level controls apply"

        is_counter = (
            (signal_direction == "LONG"  and self._weekly_bias == "BEARISH") or
            (signal_direction == "SHORT" and self._weekly_bias == "BULLISH")
        )

        # ── Trend-aligned signals ─────────────────────────────────────────────
        if not is_counter:
            # V14 bounce guard — the original implementation hard-blocked ALL shorts
            # during a 4h bounce, which combined with the LONG block created a complete
            # trading deadlock. The fix: never hard-block trend-aligned signals.
            # The bounce context is handled as a soft penalty in the aggregator
            # (shorts during bounce get -5 confidence to prefer resistance setups).
            return False, f"HTF aligned ({self._weekly_bias})"

        # ── Counter-trend signals ─────────────────────────────────────────────
        _dyn_adx = self.get_dynamic_adx_threshold()
        if self._weekly_adx < _dyn_adx:
            # Weak weekly trend — don't hard block, soft penalty handles it
            return False, f"HTF weak counter-trend (ADX={self._weekly_adx:.0f}<{_dyn_adx})"

        # Context-aware threshold: 4h structural pattern provides backing for counter-trend signals
        # - LONG vs BEARISH weekly when BTC 4h is bouncing (higher-lows/BOS up)
        # - SHORT vs BULLISH weekly when BTC 4h is resuming down (lower-highs, no BOS up)
        # AUDIT FIX: previously `_btc_4h_resuming_down` was computed but never read,
        # so counter-trend shorts into a weekly-bull regime had no structural tailwind.
        if signal_direction == "LONG" and self._weekly_bias == "BEARISH" and self._btc_4h_bouncing:
            # 4h bounce validates counter-trend long — require 78 (down from ~86)
            base_threshold = 78.0
        elif signal_direction == "SHORT" and self._weekly_bias == "BULLISH" and self._btc_4h_resuming_down:
            # 4h resuming-down validates counter-trend short
            base_threshold = 78.0
        else:
            # No bounce support: standard formula, capped at 88 (not 93 — too restrictive)
            base_threshold = min(88.0, 78.0 + 0.25 * (self._weekly_adx - 25))  # At ADX=69: 88.0

        # FIX-9: Extreme Fear/Greed adjustment
        # Uses centralized thresholds from config/constants.py
        # Both cases warrant a threshold reduction for the contrarian trade.
        try:
            from analyzers.regime import regime_analyzer as _ra
            from config.constants import FearGreedThresholds as _FGT
            _fg = _ra.fear_greed
            if signal_direction == "LONG" and _fg < _FGT.EXTREME_FEAR:
                # Extreme fear = sentiment capitulation → counter-trend longs have tailwind
                base_threshold = max(72.0, base_threshold - 5.0)
            elif signal_direction == "SHORT" and _fg > _FGT.EXTREME_GREED:
                # Extreme greed + weekly bearish = classic distribution → reduce threshold
                base_threshold = max(72.0, base_threshold - 3.0)
        except Exception:
            pass

        # Counter-trend-specialised strategies get a further 5pt reduction
        # Counter-trend-specialised: -5pt reduction. All others stay at full threshold.
        _COUNTER_TREND_STRATEGIES = {"MeanReversion", "RangeScalper", "ExtremeReversal"}
        # Apply pre-threshold confidence penalty for counter-regime counter-trend signals
        # This ensures strategies like Wyckoff/PriceAction/FundingRateArb face true bar
        # even if their raw confidence is just above the 0.2pt margin
        _TREND_FOLLOW_STRATEGIES = {"WyckoffAccDist", "Wyckoff", "ElliottWave", "Momentum", "MomentumContinuation", "Ichimoku", "IchimokuCloud", "SmartMoneyConcepts"}
        if strategy_name in _TREND_FOLLOW_STRATEGIES:
            # Trend-following counter-direction: +3pt (not +5) — leaves small window for extremes
            effective_threshold = min(91.0, base_threshold + 3.0)
        else:
            # Counter-trend strategies (MeanReversion, RangeScalper etc.) get a lower bar (-5pt).
            # Strategies in neither list (e.g. SMC, Breakout) use the base threshold unchanged.
            effective_threshold = (
                max(72.0, base_threshold - 5.0)
                if strategy_name in _COUNTER_TREND_STRATEGIES
                else base_threshold
            )

        if raw_confidence >= effective_threshold:
            return False, (
                f"HTF counter-trend override: conf={raw_confidence:.0f} "
                f">= {effective_threshold:.0f}"
            )

        reason = (
            f"🚫 HTF weekly guardrail: {signal_direction} blocked — "
            f"weekly {self._weekly_bias} (ADX={self._weekly_adx:.0f}). "
            f"Need {effective_threshold:.0f}+ confidence "
            f"({'4h bounce active — lower bar' if self._btc_4h_bouncing else ('4h resuming down — lower bar' if self._btc_4h_resuming_down and signal_direction == 'SHORT' else 'no structural support')})."
        )
        # Increment session block counters for Sentinel dashboard
        if signal_direction == "LONG":
            self._blocks_long_session += 1
        else:
            self._blocks_short_session += 1
        return True, reason

    # ── Original async check (soft penalty, kept for backward compat) ─────────
    async def check(
        self,
        signal_direction: str,
        signal_confidence: float,
    ) -> HTFGuardrailResult:
        """
        Soft-penalty path used in the existing pipeline for post-aggregation
        penalty application. Still used for non-blocked signals to apply
        the soft confidence penalty (e.g., -5 for aligned trend signals).
        """
        await self._maybe_refresh()

        is_counter_trend = (
            (signal_direction == "LONG" and self._weekly_bias == "BEARISH")
            or (signal_direction == "SHORT" and self._weekly_bias == "BULLISH")
        )

        if not is_counter_trend or self._weekly_bias == "NEUTRAL":
            return HTFGuardrailResult(
                blocked=False, penalty=0,
                weekly_bias=self._weekly_bias, weekly_adx=self._weekly_adx,
                reason=(
                    f"✅ Weekly trend aligned ({self._weekly_bias}) "
                    f"or neutral (ADX={self._weekly_adx:.0f})"
                ),
            )

        if self._weekly_adx < self.get_dynamic_adx_threshold():
            return HTFGuardrailResult(
                blocked=False, penalty=8,
                weekly_bias=self._weekly_bias, weekly_adx=self._weekly_adx,
                reason=(
                    f"⚠️ Weak counter-trend weekly ({self._weekly_bias}, "
                    f"ADX={self._weekly_adx:.0f}) — small confidence penalty only"
                ),
            )

        override_threshold = round(min(93.0, 80.0 + 0.2 * (self._weekly_adx - 25)), 1)

        if signal_confidence >= override_threshold:
            return HTFGuardrailResult(
                blocked=False, penalty=5,
                weekly_bias=self._weekly_bias, weekly_adx=self._weekly_adx,
                reason=(
                    f"⚡ Counter-trend override — weekly={self._weekly_bias} "
                    f"(ADX={self._weekly_adx:.0f}) but confidence={signal_confidence:.0f} "
                    f">= {override_threshold} threshold"
                ),
            )

        return HTFGuardrailResult(
            blocked=True, penalty=0,
            weekly_bias=self._weekly_bias, weekly_adx=self._weekly_adx,
            reason=(
                f"🚫 HTF weekly guardrail: {signal_direction} blocked — "
                f"weekly {self._weekly_bias} (ADX={self._weekly_adx:.0f}). "
                f"Need {override_threshold}+ confidence for counter-trend scalps."
            ),
        )

    async def _maybe_refresh(self):
        if time.time() - self._last_update < self._cache_ttl:
            return
        async with self._lock:
            if time.time() - self._last_update < self._cache_ttl:
                return
            try:
                await self._refresh()
            except Exception as e:
                logger.warning(f"HTF guardrail refresh failed: {e}")

    async def _refresh(self):
        """Fetch BTC weekly OHLCV and classify macro trend.
        V14: Also fetch BTC 4h to detect counter-trend bounces."""
        # V14: BTC 4h bounce detection — requires meaningful structure, not just noise
        try:
            ohlcv_4h = await api.fetch_ohlcv("BTC/USDT", "4h", limit=30)
            if ohlcv_4h and len(ohlcv_4h) >= 10:
                closes_4h = [float(c[4]) for c in ohlcv_4h]
                lows_4h   = [float(c[3]) for c in ohlcv_4h]
                highs_4h  = [float(c[2]) for c in ohlcv_4h]

                # Higher lows: require >= 0.5% separation to count as meaningful
                # (the old 0.2% tolerance caused nearly every sideways market to
                # register as "bouncing", permanently activating the bounce guard)
                recent_lows = lows_4h[-6:]
                _hl_count = sum(
                    1 for i in range(1, len(recent_lows))
                    if recent_lows[i] > recent_lows[i-1] * 1.005  # +0.5% minimum
                )
                _higher_lows = _hl_count >= 3  # majority of recent swings are HLs

                # BOS upward: close must be meaningfully above the prior swing high
                _recent_high = max(highs_4h[-8:-3]) if len(highs_4h) >= 8 else 0
                _bos_up = closes_4h[-1] > _recent_high * 1.005 if _recent_high > 0 else False

                self._btc_4h_bouncing = _higher_lows or _bos_up

                # Resumption: lower highs with meaningful separation
                recent_highs = highs_4h[-6:]
                _lh_count = sum(
                    1 for i in range(1, len(recent_highs))
                    if recent_highs[i] < recent_highs[i-1] * 0.995  # -0.5% minimum
                )
                self._btc_4h_resuming_down = _lh_count >= 3 and not _bos_up
            else:
                self._btc_4h_bouncing = False
                self._btc_4h_resuming_down = False
        except Exception as _4h_err:
            logger.debug(f"BTC 4h fetch failed: {_4h_err}")
            self._btc_4h_bouncing = False
            self._btc_4h_resuming_down = False

        ohlcv = await api.fetch_ohlcv("BTC/USDT", "1w", limit=30)
        if not ohlcv or len(ohlcv) < 12:
            return

        closes = np.array([float(c[4]) for c in ohlcv])
        highs  = np.array([float(c[2]) for c in ohlcv])
        lows   = np.array([float(c[3]) for c in ohlcv])

        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
        ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)

        self._weekly_adx = self._calc_adx(highs, lows, closes)

        weekly_hh = closes[-1] > np.max(closes[-8:-1]) if len(closes) >= 8 else True
        weekly_ll = closes[-1] < np.min(lows[-8:-1])   if len(closes) >= 8 else False

        ma10_recent = np.mean(closes[-5:])
        ma10_prev   = np.mean(closes[-10:-5])
        self._weekly_ma_slope = (ma10_recent - ma10_prev) / ma10_prev if ma10_prev > 0 else 0.0

        bullish_signals = sum([
            closes[-1] > ma20,
            closes[-1] > ma50,
            ma20 > ma50,
            weekly_hh,
            self._weekly_ma_slope > 0.02,
        ])
        bearish_signals = sum([
            closes[-1] < ma20,
            closes[-1] < ma50,
            ma20 < ma50,
            weekly_ll,
            self._weekly_ma_slope < -0.02,
        ])

        if bullish_signals >= 4 and self._weekly_adx > 22:
            self._weekly_bias = "BULLISH"
        elif bearish_signals >= 4 and self._weekly_adx > 22:
            self._weekly_bias = "BEARISH"
        else:
            self._weekly_bias = "NEUTRAL"

        self._last_update = time.time()
        logger.info(
            f"📅 HTF weekly: bias={self._weekly_bias} | "
            f"ADX={self._weekly_adx:.0f} | slope={self._weekly_ma_slope:.3f} | "
            f"4h_bounce={self._btc_4h_bouncing}"
        )

    @staticmethod
    def _calc_adx(highs, lows, closes, period=10) -> float:
        if len(closes) < period + 2:
            return 0.0
        try:
            high_diff = np.diff(highs)
            low_diff  = -np.diff(lows)
            plus_dm  = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])
                )
            )
            atr_s   = np.mean(tr[-period:])
            plus_s  = np.mean(plus_dm[-period:])
            minus_s = np.mean(minus_dm[-period:])
            if atr_s == 0:
                return 0.0
            plus_di  = 100 * plus_s / atr_s
            minus_di = 100 * minus_s / atr_s
            return float(min(100.0, 100 * abs(plus_di - minus_di) / max(1e-6, plus_di + minus_di)))
        except Exception:
            return 0.0

    # ── Stablecoin-linked dynamic ADX threshold ────────────────────────────────
    def update_stablecoin_adx_offset(self, stablecoin_signal: str):
        """
        Adjust HTF ADX threshold based on stablecoin flow signal.

        When capital is DRAINING → raise the bar (more restrictive).
        When capital is AMPLE → lower the bar (more permissive).
        Called by engine.py after stablecoin analyzer update.
        """
        from config.constants import StablecoinHTF as _SH
        if stablecoin_signal == "DRAINING":
            self._stablecoin_adx_offset = _SH.OUTFLOW_ADX_BOOST  # +3 → threshold 28
        elif stablecoin_signal == "AMPLE":
            self._stablecoin_adx_offset = -_SH.INFLOW_ADX_REDUCTION  # -2 → threshold 23
        else:
            self._stablecoin_adx_offset = 0

    def get_dynamic_adx_threshold(self) -> int:
        """Return the effective ADX threshold after stablecoin adjustment."""
        from config.constants import StablecoinHTF as _SH
        return _SH.BASE_ADX_THRESHOLD + self._stablecoin_adx_offset

    def get_weekly_adx(self) -> float:
        """Return the most recent cached weekly ADX value (0-100).

        Public accessor for strategies that need to gate on weekly trend strength
        without directly reaching into the private ``_weekly_adx`` attribute.
        Returns 0.0 before the first successful refresh.
        """
        return self._weekly_adx


# ── Singleton ──────────────────────────────────────────────────────────────
htf_guardrail = HTFWeeklyGuardrail()
