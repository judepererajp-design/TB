"""
TitanBot Pro — Market Regime Analyzer
======================================
Determines the overall market regime and acts as the TOP-LEVEL GATE
for all signals. No signal fires without passing regime analysis.

Regimes:
  BULL_TREND      → All strategies active, aggressive mode
  BEAR_TREND      → Only short strategies and high-conviction longs
  CHOPPY          → Raise confidence bar, reduce signal frequency
  VOLATILE        → High-conviction setups, all strategies allowed, tighter sizing
  VOLATILE_PANIC  → Near-shutdown: only Wyckoff/SMC/FundingRateArb, max 2 signals/hr
  NEUTRAL         → HTF guardrail only (weekly bias undecided)

Also tracks:
  - BTC dominance trend (rising = BTC season, falling = alt season)
  - Alt season score (0-100)
  - Fear & Greed index
  - Session awareness (Asia/London/NY/Weekend)
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import aiohttp

from config.loader import cfg
from data.api_client import api

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    BULL_TREND      = "BULL_TREND"
    BEAR_TREND      = "BEAR_TREND"
    CHOPPY          = "CHOPPY"
    VOLATILE        = "VOLATILE"
    VOLATILE_PANIC  = "VOLATILE_PANIC"
    NEUTRAL         = "NEUTRAL"


class Session(str, Enum):
    ASIA           = "ASIA"
    LONDON_OPEN    = "LONDON_OPEN"
    NY_OPEN        = "NY_OPEN"
    LONDON_CLOSE   = "LONDON_CLOSE"
    OVERLAP        = "OVERLAP"
    DEAD_ZONE      = "DEAD_ZONE"
    WEEKEND        = "WEEKEND"


class RegimeAnalyzer:
    """
    Analyzes macro market conditions to determine regime.
    Results are cached and refreshed every N minutes.
    """

    def __init__(self):
        self._regime: Regime = Regime.CHOPPY
        self._session: Session = Session.DEAD_ZONE
        self._provisional_regime: "Regime" = Regime.CHOPPY  # 1-cycle-ahead regime estimate
        self._btc_dominance: float = 52.0
        self._btc_dominance_last_update: float = 0  # 0 = never fetched (default/stale)
        self._alt_season_score: int = 50
        self._fear_greed: int = 50
        self._fear_greed_last_update: float = 0  # 0 = never fetched (default/stale)
        self._volatility_regime: str = "NORMAL"
        self._last_update: float = 0
        self._cache_ttl = cfg.cache.get('regime_ttl', 300)
        self._lock = asyncio.Lock()

        # Regime transition cooldown — require 2 consecutive matching cycles
        # before committing to a new regime (prevents rapid flipping near thresholds)
        self._regime_candidate: Regime = Regime.CHOPPY
        self._regime_candidate_count: int = 0
        # First-update flag — bypasses confirmation cooldown on startup so the
        # bot uses the real regime immediately instead of defaulting to CHOPPY
        # for the first ~5 minutes (one full update cycle).
        self._is_initialized: bool = False

        # Post-commit hysteresis: timestamp (epoch secs) of the last
        # regime flip. Aggregator consults this to bump min-confidence
        # for the noisy window right after a transition is committed.
        # 0.0 sentinel = no transition seen since startup.
        self._last_regime_change_at: float = 0.0

        # Regime state
        self._btc_trend_bars: Optional[np.ndarray] = None
        self._btc_adx: float = 0
        self._btc_atr_ratio: float = 1.0
        # Raw ATR values stored for the volatility spike fast-path in _classify_regime().
        # M4-FIX: the fast-path previously checked hasattr(self, '_btc_atr') which was
        # always False — these attributes were never assigned, so the fast-path was dead.
        self._btc_atr: float = 0.0
        self._btc_atr_avg: float = 0.0

        # Cross-asset macro state (early risk-off detection)
        # Fetched from Yahoo Finance / public APIs — no key needed
        self._dxy_rising: bool = False        # Dollar strengthening = risk-off signal
        self._spx_below_ma20: bool = False    # S&P500 below 20-day = macro risk-off
        self._vix_elevated: bool = False      # VIX > 25 = equity fear = crypto risk-off
        self._macro_risk_off: bool = False    # Combined cross-asset risk-off flag
        self._macro_risk_on: bool = False     # Combined risk-on flag (for signal modifier)
        self._macro_last_update: float = 0
        self._macro_ttl: int = 1800           # 30 min — macro data moves slowly

        # Chop mode continuous scoring (0.0 = strong trend, 1.0 = pure chop)
        self._chop_strength: float = 0.5
        self._range_high: float = 0.0    # Upper boundary of detected range
        self._range_low: float = 0.0     # Lower boundary of detected range
        self._range_eq: float = 0.0      # Equilibrium (midpoint)

        # State confidence: how decisively was the current regime classified (0–1)
        self._state_confidence: float = 0.5

        # Volume chop factor: erratic volume = chop signal (used in composite)
        self._volume_chop_factor: float = 0.5

        # Fast crash detection flag (5m price drop + volume spike)
        self._crash_detected: bool = False

        # Regime transition early warning (3-factor detection)
        self._adx_history: list = []          # Rolling ADX values for decline detection
        self._transition_warning: bool = False
        self._transition_factors: list = []   # Which factors triggered
        self._transition_factor_count: int = 0

    # ── Public interface ────────────────────────────────────

    @property
    def regime(self) -> Regime:
        return self._regime

    @property
    def session(self) -> Session:
        return self._get_current_session()

    @property
    def alt_season_score(self) -> int:
        return self._alt_season_score

    @property
    def fear_greed(self) -> int:
        return self._fear_greed

    @property
    def fear_greed_is_fresh(self) -> bool:
        """True if Fear & Greed has been successfully fetched at least once this session."""
        return self._fear_greed_last_update > 0

    @property
    def btc_dominance_is_fresh(self) -> bool:
        """True if BTC dominance has been successfully fetched at least once this session."""
        return self._btc_dominance_last_update > 0

    @property
    def btc_dominance(self) -> float:
        return self._btc_dominance

    @property
    def state_confidence(self) -> float:
        """
        Confidence that the current regime classification is correct [0, 1].
        High when indicators strongly support the classified regime; low near thresholds.
        Used by callers as: final_penalty = base_penalty * state_confidence
        """
        return self._state_confidence

    # ── Post-commit transition hysteresis ───────────────────
    # The 2-cycle confirmation in _update_regime is a *pre-commit*
    # hysteresis. These helpers expose a *post-commit* window so
    # downstream consumers (e.g. aggregator min-confidence floor)
    # can apply extra caution for the noisy minutes after a flip.

    def time_since_last_transition(self) -> Optional[float]:
        """Seconds since the last committed regime change, or None if never."""
        if self._last_regime_change_at <= 0:
            return None
        return max(0.0, time.time() - self._last_regime_change_at)

    def is_recently_transitioned(self, within_secs: int) -> bool:
        """True iff a regime change committed within the last ``within_secs``."""
        elapsed = self.time_since_last_transition()
        return elapsed is not None and elapsed < within_secs

    @property
    def is_alt_season(self) -> bool:
        return self._alt_season_score >= 60

    @property
    def is_bull(self) -> bool:
        return self._regime == Regime.BULL_TREND

    @property
    def is_choppy(self) -> bool:
        return self._regime in (Regime.CHOPPY, Regime.VOLATILE)

    @property
    def chop_strength(self) -> float:
        """Continuous chop score: 0.0 = trending, 1.0 = pure chop"""
        return self._chop_strength

    @property
    def range_high(self) -> float:
        """Upper boundary of detected trading range"""
        return self._range_high

    @property
    def range_low(self) -> float:
        """Lower boundary of detected trading range"""
        return self._range_low

    @property
    def range_eq(self) -> float:
        """Equilibrium / midpoint of range"""
        return self._range_eq

    def get_min_confidence_override(self) -> Optional[int]:
        """
        Regime can raise the minimum confidence threshold.
        Returns None if no override (use settings.yaml default).
        """
        reg_cfg = cfg.aggregator.regime_adjustments
        if self._regime == Regime.CHOPPY:
            return getattr(reg_cfg.get('CHOPPY', {}), 'min_confidence_override', None)

    def get_adaptive_min_confidence(
        self,
        base_min: int,
        recent_win_rate: float = -1,
        direction: str = "",
    ) -> int:
        """
        Adaptive minimum confidence that adjusts based on:
          1. Current regime (CHOPPY stricter, TREND looser)
          2. Recent signal win rate feedback
          3. Chop strength (continuous scaling)
          4. Fear & Greed direction-awareness (FIX #15)

        FIX #15: Added direction param. Extreme fear is NOT a reason to raise
        the floor for SHORT signals — they are WITH the panic, highest conviction.
        Previously a flat +3 was applied to both directions; now it's directional.
        """
        adjusted = base_min

        # Chop scaling: chop=0 → -5 (trend), chop=1 → +8 (chop)
        _effective_chop = self._chop_strength
        if hasattr(self, '_provisional_regime') and self._provisional_regime != self._regime:
            if self._provisional_regime.value in ('CHOPPY', 'BEAR_TREND', 'VOLATILE'):
                _effective_chop = min(1.0, self._chop_strength + 0.2)
        chop_adj = int(-5 + 13 * _effective_chop)
        adjusted += chop_adj

        # Win rate feedback
        if recent_win_rate >= 0:
            if recent_win_rate >= 0.65:
                adjusted -= 3
            elif recent_win_rate < 0.40:
                adjusted += 5
            elif recent_win_rate < 0.50:
                adjusted += 2

        # FIX #15: Direction-aware F&G
        from config.constants import FearGreedThresholds as _FGT
        if self._fear_greed < _FGT.EXTREME_FEAR:
            if direction == "LONG":
                adjusted += 4   # Contrarian long in extreme fear — needs conviction
            elif direction == "SHORT":
                pass            # WITH the panic — no extra penalty
            else:
                adjusted += 2   # Unknown direction — mild caution
        elif self._fear_greed > _FGT.EXTREME_GREED:
            if direction == "SHORT":
                adjusted += 4   # Contrarian short in extreme greed — needs conviction
            elif direction == "LONG":
                pass            # WITH the mania — no extra penalty
            else:
                adjusted += 2   # Unknown direction — mild caution

        return max(60, min(95, adjusted))


    def get_weight_adjustments(self) -> Dict[str, float]:
        """Get weight adjustment deltas for current regime"""
        adjustments = {}
        reg_cfg = cfg.aggregator.regime_adjustments
        regime_name = self._regime.value
        regime_adj = reg_cfg.get(regime_name, {})
        if hasattr(regime_adj, 'to_dict'):
            for key, val in regime_adj.to_dict().items():
                if key != 'min_confidence_override' and val != 0:
                    adjustments[key] = val
        return adjustments

    def is_killzone(self) -> Tuple[bool, int]:
        """
        Returns (is_in_killzone, confidence_bonus).
        Killzones are high-probability session times for SMC setups.
        """
        session = self._get_current_session()
        if session in (Session.LONDON_OPEN, Session.NY_OPEN):
            bonus = cfg.strategies.smc.killzones.get('killzone_bonus', 10)
            return True, bonus
        elif session == Session.LONDON_CLOSE:
            return True, 5
        return False, 0

    # ── Update cycle ────────────────────────────────────────

    async def update(self):
        """
        Refresh all regime data. Called every regime_interval seconds.
        """
        async with self._lock:
            if time.time() - self._last_update < self._cache_ttl:
                return

            try:
                await self._update_btc_trend()
                await self._update_fear_greed()
                await self._fetch_macro_signals()   # Cross-asset early warning
                await self._classify_regime()
                self._last_update = time.time()

                logger.info(
                    f"Regime: {self._regime.value} | "
                    f"Chop: {self._chop_strength:.2f} | "
                    f"ADX: {self._btc_adx:.1f} | "
                    f"Session: {self._get_current_session().value} | "
                    f"Alt Season: {self._alt_season_score}/100 | "
                    f"F&G: {self._fear_greed}"
                )

            except Exception as e:
                logger.error(f"Regime update error: {e}")

    async def force_update(self):
        """Force an immediate regime refresh (bypasses cache)"""
        self._last_update = 0
        await self.update()

    # ── Private analysis methods ─────────────────────────────

    async def _update_btc_trend(self):
        """Analyze BTC trend and volatility"""
        ohlcv = await api.fetch_ohlcv("BTC/USDT", "4h", limit=100)
        if not ohlcv or len(ohlcv) < 60:
            return

        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # Store for chop_strength autocorrelation calculation
        self._btc_trend_bars = closes.copy()

        # SMA trend
        sma_fast = cfg.analyzers.regime.get('btc_sma_fast', 20)
        sma_slow = cfg.analyzers.regime.get('btc_sma_slow', 50)
        ma_fast = np.mean(closes[-sma_fast:])
        ma_slow = np.mean(closes[-sma_slow:])
        self._btc_above_fast_ma = closes[-1] > ma_fast
        self._btc_above_slow_ma = closes[-1] > ma_slow
        self._btc_ma_bullish = ma_fast > ma_slow

        # ADX calculation (trend strength)
        self._btc_adx = self._calculate_adx(highs, lows, closes, period=14)

        # ATR for volatility
        # C3-FIX: both calls previously received different-length slices but
        # _calculate_atr() always reads the LAST `period` bars of whatever
        # array is passed — so both calls hit the same 14 bars and the ratio
        # was always ≈ 1.0, making volatile_atr_mult = 2.0 unreachable.
        # Fix: atr_now uses the most-recent 14 bars; atr_baseline uses the
        # 14-bar window that ended 14 bars ago (bars [-28:-14]) to give a
        # true historical average that the current period can be compared to.
        atr_now = self._calculate_atr(highs[-14:], lows[-14:], closes[-14:], period=14)
        atr_baseline = self._calculate_atr(highs[-50:-14], lows[-50:-14], closes[-50:-14], period=14)
        self._btc_atr_ratio = atr_now / atr_baseline if atr_baseline > 0 else 1.0
        # M4-FIX: store raw values so the volatility-spike fast-path in
        # _classify_regime() can reference them (previously checked via
        # hasattr which was always False, keeping the fast-path permanently dead).
        self._btc_atr = atr_now
        self._btc_atr_avg = atr_baseline

        # Higher highs / higher lows (simple structure check)
        self._btc_higher_highs = self._check_higher_highs(highs[-20:])
        self._btc_lower_lows = self._check_lower_lows(lows[-20:])

        # Volume chop factor: high coefficient of variation = erratic = chop signal.
        # Low CV = consistent = trend-supporting.  Computed over the same 100-bar window.
        # Conditioning: when ADX indicates a strong trend (> 25), high volume variance is
        # EXPECTED (breakout/expansion bars drive large candles) and should NOT be counted
        # as chop evidence.  Scale the factor down proportionally to ADX trend strength.
        vol = df['volume'].values
        if len(vol) >= 20:
            _vol_mean = np.mean(vol[-20:])
            _vol_cv = np.std(vol[-20:]) / (_vol_mean + 1e-10)
            # CV=0 (steady) → 0.0; CV=2+ (erratic) → 1.0 — baseline chop factor
            _raw_vol_chop = min(1.0, _vol_cv * 0.5)
            # Trend-conditioning: reduce vol chop contribution when ADX > 25.
            # At ADX=25 → trend_damp=0 (no reduction); at ADX=45 → trend_damp=1.0 (zero out).
            # Prevents trending breakout volume from being misread as chop.
            _trend_damp = min(1.0, max(0.0, (self._btc_adx - 25) / 20))
            self._volume_chop_factor = float(_raw_vol_chop * (1.0 - _trend_damp))
        else:
            self._volume_chop_factor = 0.5

        # BTC dominance — fetch real data from CoinGecko/Binance
        await self._fetch_btc_dominance()
        # Fast crash detection on 5m timeframe
        await self._fetch_crash_detection()

    async def _update_fear_greed(self):
        """Fetch Fear & Greed index from Alternative.me"""
        if not cfg.analyzers.sentiment.get('fear_greed_enabled', True):
            return

        try:
            url = cfg.analyzers.regime.get('fear_greed_api',
                                           'https://api.alternative.me/fng/')
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._fear_greed = int(data['data'][0]['value'])
                        self._fear_greed_last_update = time.time()
                        logger.debug(f"F&G updated: {self._fear_greed}")
                    else:
                        logger.warning(
                            f"Fear & Greed API returned HTTP {resp.status} — "
                            f"using {'default 50' if not self.fear_greed_is_fresh else 'cached'} value {self._fear_greed}"
                        )
        except Exception as e:
            logger.warning(
                f"Fear & Greed fetch failed: {e} — "
                f"using {'default 50 (never fetched this session)' if not self.fear_greed_is_fresh else f'cached {self._fear_greed}'}"
            )

    async def _classify_regime(self):
        """
        Classify the current market regime based on all available data.

        Decision tree:
        1. Extreme fear + BTC in downtrend → VOLATILE
        2. High volatility (ATR spike) → VOLATILE
        3. Low ADX → CHOPPY
        4. BTC uptrend → BULL_TREND
        5. BTC downtrend → BEAR_TREND

        Also computes continuous chop_strength (0.0 to 1.0).
        """
        chop_adx_max = cfg.analyzers.regime.get('chop_adx_max', 20)
        volatile_atr_mult = cfg.analyzers.regime.get('volatile_atr_mult', 2.0)

        # ── Fast crash detection override ─────────────────────
        # 5-minute price drop >2% + volume spike >1.5× detected: this is a flash
        # crash / capitulation event.  Skip all other checks — commit immediately.
        if self._crash_detected:
            self._regime = Regime.VOLATILE_PANIC
            self._regime_candidate = Regime.VOLATILE_PANIC
            self._regime_candidate_count = 0
            self._provisional_regime = Regime.VOLATILE_PANIC
            self._state_confidence = 0.90  # High but not absolute — crash signals can be false positives
            self._calculate_alt_season_score()
            return

        # ── Compute continuous chop_strength ──────────────────
        # Based on: low ADX + low autocorrelation + overlapping structure
        # ADX component: ADX < 15 = full chop, ADX > 30 = full trend
        adx_chop = max(0.0, min(1.0, 1.0 - (self._btc_adx - 15) / 15))

        # Autocorrelation component (negative autocorr = chop)
        autocorr_chop = 0.5  # Default neutral
        if self._btc_trend_bars is not None and len(self._btc_trend_bars) >= 30:
            try:
                closes = self._btc_trend_bars
                returns = np.diff(closes) / closes[:-1]
                if len(returns) >= 20:
                    r1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                    if not np.isnan(r1):
                        # Negative autocorr = chop, positive = trending
                        autocorr_chop = max(0.0, min(1.0, 0.5 - r1 * 2.0))
            except Exception:
                pass

        # Structure overlap component: higher highs AND lower lows = chop
        structure_chop = 0.5
        if self._btc_higher_highs and self._btc_lower_lows:
            structure_chop = 0.9  # Overlapping structure = choppy
        elif not self._btc_higher_highs and not self._btc_lower_lows:
            structure_chop = 0.8  # No clear structure
        elif self._btc_higher_highs and not self._btc_lower_lows:
            structure_chop = 0.2  # Bullish trend
        elif self._btc_lower_lows and not self._btc_higher_highs:
            structure_chop = 0.2  # Bearish trend

        # Weighted blend (4 components: ADX, autocorrelation, structure, volume)
        # Volume CV is erratic in chop; consistent in trends — 10% weight
        raw_chop = (0.40 * adx_chop
                    + 0.25 * autocorr_chop
                    + 0.25 * structure_chop
                    + 0.10 * self._volume_chop_factor)

        # FIX-8b: ADX > 40 contradicts CHOPPY classification.
        # When ADX is strong, reduce chop_strength — strong directional force
        # present even if price structure appears range-bound.
        if self._btc_adx > 40:
            _adx_damp = min(1.0, (self._btc_adx - 40) / 20)  # 0 at ADX=40, 1 at ADX=60
            raw_chop *= (1.0 - 0.35 * _adx_damp)  # Up to 35% reduction at ADX=60

        self._chop_strength = round(raw_chop, 3)

        # ── Detect trading range boundaries ───────────────────
        if self._btc_trend_bars is not None and len(self._btc_trend_bars) >= 20:
            try:
                recent = self._btc_trend_bars[-20:]
                self._range_high = float(np.max(recent))
                self._range_low = float(np.min(recent))
                self._range_eq = (self._range_high + self._range_low) / 2.0
            except Exception:
                pass

        # ── Classify discrete regime ───────────────────────────
        # Determine the proposed new regime this cycle.
        #
        # Decision priority (revised):
        #   1. VOLATILE  — emergency: extreme fear + confirmed downtrend
        #   2. BULL_TREND — price above both MAs + higher highs (fast signal, no MA cross lag)
        #   3. BEAR_TREND — price below both MAs + lower lows
        #   4. VOLATILE  — high ATR but NO directional structure (undirected whipsaw)
        #   5. CHOPPY    — low ADX, no trend
        #   6. Indeterminate fallback
        #
        # Key fix: ATR spike is moved AFTER directional checks.
        # A bull/bear expansion WILL produce a high ATR — that is what expansion looks like.
        # ATR alone cannot mean VOLATILE; it only means VOLATILE when direction is ambiguous.
        # Previously, ATR check fired before direction check, causing every sharp rally
        # to be misclassified as VOLATILE (blocking long_bias + lowered conf threshold).
        #
        # BULL_TREND condition changed from (ma_fast>ma_slow AND above_fast AND HH)
        # to (above_fast AND above_slow AND HH). MA cross lags by weeks after a reversal.
        # Price being above both MAs is a sufficient near-term directional signal.

        proposed_regime: Regime

        # 0. VOLATILE_PANIC: extreme fear + crash + volatility spike
        # All three must be true: panic sentiment + structural breakdown + abnormal vol
        # This is the "flash crash / capitulation event" state — near-shutdown mode
        if (self._fear_greed < 10
                and self._btc_lower_lows
                and not self._btc_ma_bullish
                and self._btc_atr_ratio > 2.5):
            proposed_regime = Regime.VOLATILE_PANIC

        # 1. Risk-off: extreme fear + confirmed downtrend (less severe than VOLATILE_PANIC)
        # Also fire if macro cross-asset risk-off (DXY+SPX+VIX) aligns with BTC downtrend
        elif (self._fear_greed < 15
                and not self._btc_ma_bullish
                and self._btc_lower_lows):
            proposed_regime = Regime.VOLATILE

        # 1b. Macro accelerated risk-off: cross-asset signals fire even before F&G drops
        # DXY rising + SPX weakening + VIX elevated = macro risk-off precedes BTC by 1-2h
        elif (self._macro_risk_off
                and not self._btc_ma_bullish
                and self._btc_lower_lows):
            proposed_regime = Regime.BEAR_TREND  # Not VOLATILE yet — macro warns, BTC confirms

        # 2. Bull trend: price above both MAs + higher highs structure
        #    Uses price position vs MAs (fast-reacting) rather than MA cross (lagging)
        elif (self._btc_above_fast_ma
              and self._btc_above_slow_ma
              and self._btc_higher_highs):
            # Still tag as VOLATILE if ATR is extreme AND ADX is weak (spiky but directionless)
            if self._btc_atr_ratio > volatile_atr_mult and self._btc_adx < chop_adx_max:
                proposed_regime = Regime.VOLATILE
            else:
                proposed_regime = Regime.BULL_TREND

        # 3. Bear trend: price below both MAs + lower lows structure
        elif (not self._btc_above_slow_ma
              and not self._btc_above_fast_ma
              and self._btc_lower_lows):
            if self._btc_atr_ratio > volatile_atr_mult and self._btc_adx < chop_adx_max:
                proposed_regime = Regime.VOLATILE
            else:
                proposed_regime = Regime.BEAR_TREND

        # 4. Choppy vs Volatile (no directional structure confirmed):
        #    - Low ADX + normal ATR → CHOPPY (pure sideways)
        #    - Low ADX + high ATR → VOLATILE (big candles but whipsawing, no direction)
        # FIX-8: Hysteresis — ADX must exceed chop_adx_max+5 to EXIT CHOPPY.
        # Prevents rapid CHOPPY↔BEAR_TREND flipping when ADX hovers at 21-22.
        elif self._btc_adx < chop_adx_max or (
            self._regime == Regime.CHOPPY and self._btc_adx < chop_adx_max + 5
        ):
            if self._btc_atr_ratio > volatile_atr_mult:
                proposed_regime = Regime.VOLATILE  # Big candles but directionless = true whipsaw
            else:
                proposed_regime = Regime.CHOPPY

        # 5. VOLATILE: high ATR with NO clear directional structure
        #    (strong move but ambiguous direction — not a clean trend expansion)
        elif self._btc_atr_ratio > volatile_atr_mult:
            proposed_regime = Regime.VOLATILE

        # 6. Indeterminate structure: all primary checks failed, use ADX strength as tiebreaker
        else:
            # Use MA cross as secondary direction signal when price position is ambiguous
            if self._btc_ma_bullish:
                proposed_regime = Regime.BULL_TREND
            elif not self._btc_ma_bullish and self._btc_lower_lows:
                proposed_regime = Regime.BEAR_TREND
            elif self._btc_adx > 30:
                # ADX > 30 = strong trending force but no clean directional structure.
                # This is the "recovering from downtrend" state: price above fast MA but
                # not yet above slow MA, SMA cross lagging, structure ambiguous.
                # VOLATILE is correct here — strong move, wait for clarity.
                # Calling it CHOPPY (as before) is wrong and crushes trend strategy weights.
                proposed_regime = Regime.VOLATILE
            else:
                proposed_regime = Regime.CHOPPY

        # ── First-update fast-path ────────────────────────────────────────────────────────
        # On startup _regime defaults to CHOPPY as a placeholder, but there is no real
        # prior state to protect against rapid flipping.  Requiring 2 confirmation cycles
        # (startup + one 300s tick) would leave the bot using wrong strategy weights for
        # the first ~5 minutes.  Commit the data-driven regime immediately on first call.
        if not self._is_initialized:
            self._is_initialized = True
            self._regime = proposed_regime
            self._regime_candidate = proposed_regime
            self._regime_candidate_count = 0
            self._provisional_regime = proposed_regime
            self._state_confidence = self._compute_state_confidence(volatile_atr_mult, chop_adx_max)
            self._calculate_alt_season_score()
            return

        # ── Cooldown: require 2 consecutive matching cycles to commit (FIX M4 enhanced) ────
        # FIX M4: Don't fully reset candidate count when a different regime appears once.
        # A single "blip" detection (e.g. VOLATILE during a rally) would previously
        # reset count to 0, causing the real trend transition to never be detected.
        # Now we decay the counter by 1 on a mismatch instead of resetting to 0.
        # FIX #30: Volatility spike fast-path.
        # If ATR has expanded > 2.5× its 20-bar average, this is a real volatility event
        # (flash crash, major news). The 5-cycle delay would leave the bot in NEUTRAL/CHOPPY
        # during the first wave of a crash. Commit VOLATILE immediately on a spike.
        _is_vol_spike = False
        try:
            if proposed_regime == Regime.VOLATILE and hasattr(self, '_btc_atr') and hasattr(self, '_btc_atr_avg'):
                _is_vol_spike = (self._btc_atr > 0 and self._btc_atr_avg > 0
                                 and self._btc_atr / self._btc_atr_avg > 2.5)
        except Exception:
            pass

        if _is_vol_spike:
            # Immediate commit for volatility spikes — don't wait for confirmation
            if self._regime != Regime.VOLATILE:
                self._last_regime_change_at = time.time()
            self._regime = Regime.VOLATILE
            self._regime_candidate = Regime.VOLATILE
            self._regime_candidate_count = 0
        elif proposed_regime == self._regime:
            # Already in this regime — reset candidate and stay
            self._regime_candidate = proposed_regime
            self._regime_candidate_count = 0
        elif proposed_regime == self._regime_candidate:
            # Same candidate as last cycle — increment counter.
            self._regime_candidate_count += 1
            # Historical (pre-Apr-2026) behaviour: `_large_distance` flips such as
            # CHOPPY→BULL committed after ONE matching cycle. In practice this
            # caused the regime to flap between CHOPPY and BULL_TREND when the
            # discriminating feature (e.g. BTC ADX) hovered right at the
            # classification threshold — chop=0.64 with ADX ≈ chop_adx_max+5
            # would re-commit on every 300s tick. Require 2 matching cycles for
            # *all* discretionary transitions; keep the fast-path only for true
            # VOLATILE events (ATR spikes are a real discontinuity and need
            # immediate action).
            _threshold = 1 if proposed_regime == Regime.VOLATILE else 2
            if self._regime_candidate_count >= _threshold:
                if self._regime != proposed_regime:
                    self._last_regime_change_at = time.time()
                self._regime = proposed_regime
                self._regime_candidate_count = 0
        else:
            # Different candidate — decay count by 1 instead of full reset
            self._regime_candidate_count = max(0, self._regime_candidate_count - 1)
            if self._regime_candidate_count == 0:
                self._regime_candidate = proposed_regime
                self._regime_candidate_count = 1

        # PHASE 3 FIX (REGIME-LAG): Track provisional regime (1-confirmation) separately
        # from committed regime (2-confirmation). Confidence floors use provisional
        # so filters adapt 1 cycle faster. Strategy selection still uses committed regime.
        self._provisional_regime = proposed_regime

        # State confidence: how decisively was the regime identified?
        self._state_confidence = self._compute_state_confidence(volatile_atr_mult, chop_adx_max)

        # Alt season score
        self._calculate_alt_season_score()

        # Regime transition early warning (3-factor check)
        self._check_transition_warning()

    def _check_transition_warning(self):
        """
        4-factor regime transition early warning.
        Factors: (1) ADX declining, (2) vol compressing, (3) flow decelerating,
        (4) OI/price exhaustion.
        Requires multi-signal agreement (MIN_FACTORS_FOR_WARNING).
        """
        from config.constants import RegimeTransition as RT

        # Track ADX history
        self._adx_history.append(self._btc_adx)
        if len(self._adx_history) > RT.ADX_PEAK_LOOKBACK + 5:
            self._adx_history = self._adx_history[-(RT.ADX_PEAK_LOOKBACK + 5):]

        self._refresh_transition_warning()

    def _refresh_transition_warning(self):
        """Recompute transition warning from current tracked inputs."""
        from config.constants import RegimeTransition as RT
        from analyzers.regime_transition import detect_transition_pattern

        factors = []
        external_inputs_available = 0

        # Factor 1: ADX declining from recent peak
        if len(self._adx_history) >= RT.ADX_PEAK_LOOKBACK:
            lookback = self._adx_history[-RT.ADX_PEAK_LOOKBACK:]
            recent_peak = max(lookback)
            current_adx = self._adx_history[-1]
            if recent_peak - current_adx >= RT.ADX_DECLINE_THRESHOLD:
                factors.append(f"ADX↓{recent_peak - current_adx:.1f}")

        # Factor 2: Vol compressing (check external vol regime)
        # This is set by the engine via set_vol_regime_for_transition()
        if hasattr(self, '_external_vol_regime') and self._external_vol_regime in RT.VOL_COMPRESS_REGIMES:
            factors.append(f"Vol:{self._external_vol_regime}")
            external_inputs_available += 1

        # Factor 3: Flow divergence (stablecoin deceleration)
        # This is set by the engine via set_flow_accel_for_transition()
        if hasattr(self, '_external_flow_accel') and self._external_flow_accel < RT.FLOW_DECEL_THRESHOLD:
            factors.append(f"FlowDecel:{self._external_flow_accel:.1f}%")
            external_inputs_available += 1

        # Factor 4: OI/price exhaustion — OI rising but price stagnant
        # (divergence suggests trend is running on leverage, not conviction)
        if hasattr(self, '_external_oi_change') and hasattr(self, '_external_price_change'):
            if (self._external_oi_change >= RT.EXHAUSTION_OI_THRESHOLD
                    and abs(self._external_price_change) <= RT.EXHAUSTION_PRICE_THRESHOLD):
                factors.append(
                    f"Exhaust:OI+{self._external_oi_change:.1f}%/price{self._external_price_change:+.1f}%"
                )
                external_inputs_available += 1

        if external_inputs_available < RT.MIN_EXTERNAL_INPUTS_FOR_WARNING_BACKSTOP:
            backstop = detect_transition_pattern(
                current_regime=getattr(self._regime, 'value', str(self._regime)),
                chop_strength=self._chop_strength,
                vol_ratio=self._btc_atr_ratio,
                adx=self._btc_adx,
                adx_history=self._adx_history,
            )
            if backstop:
                factors.append(f"Backstop:{backstop['transition_type']}")

        self._transition_factor_count = len(factors)
        self._transition_factors = factors
        self._transition_warning = len(factors) >= RT.MIN_FACTORS_FOR_WARNING

        if self._transition_warning:
            logger.info(
                f"⚠️ REGIME TRANSITION WARNING: {len(factors)}/{RT.MIN_FACTORS_FOR_WARNING} factors "
                f"active — {', '.join(factors)}"
            )

    def set_vol_regime_for_transition(self, vol_regime: str):
        """Set external vol regime input for transition detection."""
        self._external_vol_regime = vol_regime
        self._refresh_transition_warning()

    def set_flow_accel_for_transition(self, flow_acceleration: float):
        """Set external flow acceleration input for transition detection."""
        self._external_flow_accel = flow_acceleration
        self._refresh_transition_warning()

    def set_exhaustion_data(self, oi_change_pct: float, price_change_pct: float):
        """Set OI change and price change for exhaustion detection (factor 4)."""
        self._external_oi_change = oi_change_pct
        self._external_price_change = price_change_pct
        self._refresh_transition_warning()

    def update_transition_inputs(
        self,
        vol_regime: str | None = None,
        flow_acceleration: float | None = None,
        oi_change_pct: float | None = None,
        price_change_pct: float | None = None,
    ):
        """Batch-set multiple transition inputs and refresh once.

        Prefer this over calling individual setters in sequence; it avoids
        redundant _refresh_transition_warning() calls (harmless today but
        wasteful if the warning logic grows more expensive).
        """
        if vol_regime is not None:
            self._external_vol_regime = vol_regime
        if flow_acceleration is not None:
            self._external_flow_accel = flow_acceleration
        if oi_change_pct is not None and price_change_pct is not None:
            self._external_oi_change = oi_change_pct
            self._external_price_change = price_change_pct
        self._refresh_transition_warning()

    def get_transition_warning(self) -> dict:
        """
        Returns transition warning state for engine consumption.

        Returns:
            dict with:
                warning: bool — True if transition detected
                factor_count: int — number of active factors
                factors: list[str] — active factor descriptions
        """
        return {
            "warning": self._transition_warning,
            "factor_count": self._transition_factor_count,
            "factors": self._transition_factors,
        }

    def _compute_state_confidence(self, volatile_atr_mult: float, chop_adx_max: float) -> float:
        """
        Compute how decisively the committed regime was classified [0, 1].

        0.5 = ambiguous (near threshold)
        0.95 = maximum output — we never return 1.0 here since absolute certainty
               in any market signal is unwarranted and risks extreme downstream compounding.

        Callers should still clamp results:
            final_penalty = min(max_penalty, base_penalty * state_confidence)
        """
        regime = self._regime
        _MAX = 0.95  # Hard ceiling — preserves a small uncertainty margin in all cases

        if regime == Regime.VOLATILE_PANIC:
            return _MAX  # Highest possible, but still not absolute

        if regime in (Regime.BULL_TREND, Regime.BEAR_TREND):
            # Confidence from ADX strength + MA alignment + structure.
            # ADX is used here only as a *margin from threshold* (25→45), not the raw
            # value, so it contributes different information from detection (which checks
            # whether ADX exceeds thresholds at all).
            adx_conf = min(1.0, max(0.0, (self._btc_adx - 25) / 20))  # 0 at ADX=25, 1 at ADX=45
            ma_aligned = (
                (regime == Regime.BULL_TREND
                 and self._btc_above_fast_ma and self._btc_above_slow_ma and self._btc_ma_bullish)
                or
                (regime == Regime.BEAR_TREND
                 and not self._btc_above_fast_ma and not self._btc_above_slow_ma
                 and not self._btc_ma_bullish)
            )
            ma_conf = 0.8 if ma_aligned else 0.4
            struct_conf = 0.8 if (
                (regime == Regime.BULL_TREND and self._btc_higher_highs) or
                (regime == Regime.BEAR_TREND and self._btc_lower_lows)
            ) else 0.4
            raw = 0.40 * adx_conf + 0.35 * ma_conf + 0.25 * struct_conf
            return round(min(_MAX, max(0.25, raw + 0.1)), 3)

        if regime == Regime.VOLATILE:
            excess = max(0.0, self._btc_atr_ratio - volatile_atr_mult) / max(1e-10, volatile_atr_mult)
            fear_boost = 0.2 if self._fear_greed < 20 else 0.0
            return round(min(_MAX, max(0.25, 0.5 + excess * 0.4 + fear_boost)), 3)

        if regime == Regime.CHOPPY:
            if self._btc_adx < chop_adx_max:
                depth = 1.0 - (self._btc_adx / chop_adx_max)  # 1.0 at ADX=0, 0.0 at threshold
                return round(min(_MAX, max(0.25, 0.5 + depth * 0.5)), 3)
            return 0.4  # Just above threshold — low confidence

        return 0.5  # NEUTRAL or unrecognised regime

    async def _fetch_crash_detection(self):
        """
        Fast crash detection on the 5m timeframe.
        Sets ``self._crash_detected = True`` when BOTH conditions hold:
          - Latest 5m bar closed >2% below the prior bar's close (sharp drop)
          - Latest 5m bar volume is >1.5× the 20-bar rolling average (panic volume)

        When True, ``_classify_regime()`` immediately commits VOLATILE_PANIC and
        returns without running any further classification logic — override everything.
        """
        try:
            ohlcv_5m = await api.fetch_ohlcv("BTC/USDT", "5m", limit=22)
            if not ohlcv_5m or len(ohlcv_5m) < 10:
                self._crash_detected = False
                return

            df5 = pd.DataFrame(ohlcv_5m, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            latest_close = float(df5['close'].iloc[-1])
            prev_close = float(df5['close'].iloc[-2])
            # Positive value = price dropped
            price_drop_pct = (prev_close - latest_close) / prev_close

            avg_volume = float(df5['volume'].iloc[:-1].mean())
            latest_volume = float(df5['volume'].iloc[-1])
            volume_spike = latest_volume / avg_volume if avg_volume > 0 else 1.0

            self._crash_detected = (price_drop_pct > 0.02 and volume_spike > 1.5)
            if self._crash_detected:
                logger.warning(
                    f"⚡ CRASH DETECTED: 5m drop={price_drop_pct:.2%} "
                    f"vol_spike={volume_spike:.2f}x → forcing VOLATILE_PANIC"
                )
        except Exception as e:
            logger.debug(f"Crash detection fetch failed (non-fatal): {e}")
            self._crash_detected = False

    def _calculate_alt_season_score(self):
        """
        Estimate alt season likelihood (0-100).
        High score = alts outperforming BTC = rotate to alts.
        """
        score = 50  # Neutral start

        # BTC dominance falling = alts gaining
        dom_bull = cfg.analyzers.regime.get('btc_dominance_bull', 50.0)
        dom_bear = cfg.analyzers.regime.get('btc_dominance_bear', 58.0)

        if self._btc_dominance < dom_bull:
            score += 25  # Strong alt season signal
        elif self._btc_dominance < (dom_bull + dom_bear) / 2:
            score += 10

        # Fear & Greed
        if self._fear_greed > 65:
            score += 15  # Greed = risk-on = alt season
        elif self._fear_greed < 35:
            score -= 15  # Fear = BTC dominance

        # BTC trend strong = dominance strengthens
        if self._btc_adx > 40 and self._btc_ma_bullish:
            score -= 10  # BTC pumping hard = alts lag

        self._alt_season_score = max(0, min(100, score))

    async def _fetch_btc_dominance(self):
        """
        Fetch real BTC dominance from CoinGecko free /global endpoint.
        Falls back to Binance BTCDOMUSDT if CoinGecko is rate-limited.
        Updates self._btc_dominance with actual value.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Try CoinGecko first (no API key needed)
                try:
                    async with session.get(
                        "https://api.coingecko.com/api/v3/global",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            dom = data.get('data', {}).get('market_cap_percentage', {}).get('btc')
                            if dom is not None:
                                self._btc_dominance = float(dom)
                                self._btc_dominance_last_update = time.time()
                                logger.debug(f"BTC.D from CoinGecko: {self._btc_dominance:.1f}%")
                                return
                except Exception:
                    pass

                # Fallback: try Binance BTCDOMUSDT perpetual
                try:
                    ticker = await api.fetch_ticker("BTCDOMUSDT")
                    if ticker and 'last' in ticker:
                        self._btc_dominance = float(ticker['last'])
                        self._btc_dominance_last_update = time.time()
                        logger.debug(f"BTC.D from Binance: {self._btc_dominance:.1f}%")
                        return
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Could not fetch BTC dominance: {e}")
        # If all fails, keep whatever we had before

    async def _fetch_macro_signals(self):
        """
        Fetch cross-asset macro signals for early risk-off/risk-on detection.
        Uses Yahoo Finance query endpoint (no API key required).

        Risk-OFF signals (precede BTC drops by 1-2h):
          - DXY rising (dollar strengthening = capital fleeing risk assets)
          - SPX below 20-day MA (equity market weakening)
          - VIX > 25 (equity fear spiking)

        Risk-ON signals (confirm alt season potential):
          - DXY falling
          - SPX above 20-day MA
          - VIX < 16

        Updates: every 30 minutes (macro data moves slowly)
        Falls back silently on any network error — bot continues on internal signals.
        """
        if time.time() - self._macro_last_update < self._macro_ttl:
            return

        try:
            dxy_close = None; spx_close = None; spx_ma20 = None; vix_close = None

            _yf_base = "https://query1.finance.yahoo.com/v8/finance/chart"

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8)
            ) as session:

                async def _fetch_dxy():
                    try:
                        async with session.get(f"{_yf_base}/DX-Y.NYB?interval=1d&range=5d") as r:
                            if r.status == 200:
                                d = await r.json()
                                raw = (d.get("chart") or {})
                                result = (raw.get("result") or [{}])[0]
                                closes = [c for c in (result.get("indicators", {}).get("quote", [{}])[0].get("close") or []) if c is not None]
                                if len(closes) >= 2:
                                    return closes[-1], closes[-2]
                    except Exception:
                        pass
                    return None, None

                async def _fetch_spx():
                    try:
                        async with session.get(f"{_yf_base}/^GSPC?interval=1d&range=30d") as r:
                            if r.status == 200:
                                d = await r.json()
                                raw = (d.get("chart") or {})
                                result = (raw.get("result") or [{}])[0]
                                closes = [c for c in (result.get("indicators", {}).get("quote", [{}])[0].get("close") or []) if c is not None]
                                if len(closes) >= 20:
                                    return closes[-1], sum(closes[-20:]) / 20
                    except Exception:
                        pass
                    return None, None

                async def _fetch_vix():
                    try:
                        async with session.get(f"{_yf_base}/^VIX?interval=1d&range=5d") as r:
                            if r.status == 200:
                                d = await r.json()
                                raw = (d.get("chart") or {})
                                result = (raw.get("result") or [{}])[0]
                                closes = [c for c in (result.get("indicators", {}).get("quote", [{}])[0].get("close") or []) if c is not None]
                                if closes:
                                    return closes[-1]
                    except Exception:
                        pass
                    return None

                # Yahoo Finance query1 endpoint — free, no auth required.
                # All three requests are made concurrently to cut macro fetch time by ~2/3.
                (dxy_close, dxy_prev), (spx_close, spx_ma20), vix_close = await asyncio.gather(
                    _fetch_dxy(), _fetch_spx(), _fetch_vix()
                )

            # Combine into composite flags
            if dxy_close is not None and dxy_prev is not None:
                self._dxy_rising = dxy_close > dxy_prev * 1.002  # +0.2% = meaningful
            if spx_close is not None and spx_ma20 is not None:
                self._spx_below_ma20 = spx_close < spx_ma20 * 0.99  # 1% below MA
            if vix_close is not None:
                self._vix_elevated = vix_close > 25

            # Risk-OFF: at least 2 of 3 cross-asset signals fire
            risk_off_signals = sum([
                self._dxy_rising,
                self._spx_below_ma20,
                self._vix_elevated,
            ])
            self._macro_risk_off = risk_off_signals >= 2

            # Risk-ON: DXY falling + SPX above MA + VIX low
            self._macro_risk_on = (
                not self._dxy_rising
                and not self._spx_below_ma20
                and not self._vix_elevated
                and vix_close is not None and vix_close < 16
            )

            self._macro_last_update = time.time()
            logger.info(
                f"📊 Macro: DXY={'↑' if self._dxy_rising else '↓'} "
                f"SPX={'<MA20' if self._spx_below_ma20 else '>MA20'} "
                f"VIX={f'{vix_close:.0f}' if vix_close else '?'} "
                f"→ risk_off={self._macro_risk_off} risk_on={self._macro_risk_on}"
            )

        except Exception as e:
            logger.debug(f"Macro signal fetch failed (non-fatal): {e}")

    def _get_current_session(self) -> Session:
        """Determine current trading session by UTC hour"""
        now = datetime.now(timezone.utc)

        # Weekend check (Saturday/Sunday)
        if now.weekday() >= 5:
            return Session.WEEKEND

        hour = now.hour
        killzones = cfg.strategies.smc.killzones

        london_open = getattr(killzones, 'london_open', [7, 10])
        ny_open = getattr(killzones, 'ny_open', [12, 15])
        london_close = getattr(killzones, 'london_close', [15, 17])
        asia_open = getattr(killzones, 'asia_open', [0, 3])

        if isinstance(london_open, (list, tuple)) and london_open[0] <= hour < london_open[1]:
            return Session.LONDON_OPEN
        elif isinstance(ny_open, (list, tuple)) and ny_open[0] <= hour < ny_open[1]:
            return Session.NY_OPEN
        elif isinstance(london_close, (list, tuple)) and london_close[0] <= hour < london_close[1]:
            return Session.LONDON_CLOSE
        elif isinstance(asia_open, (list, tuple)) and len(asia_open) >= 2:
            # Handle midnight-spanning sessions (e.g. [23, 3]) and normal ranges (e.g. [0, 3]).
            # For [0, 3]: the naïve `(hour >= 0 or hour < 3)` is ALWAYS True because
            # `hour >= 0` holds for every valid hour value — dead zone (3–7) would be
            # unreachable.  Use the non-spanning form when start ≤ end.
            if asia_open[0] <= asia_open[1]:
                asia_match = asia_open[0] <= hour < asia_open[1]
            else:  # spans midnight (e.g. [23, 3])
                asia_match = hour >= asia_open[0] or hour < asia_open[1]
            if asia_match:
                return Session.ASIA
        if 3 <= hour < 7:
            return Session.DEAD_ZONE
        elif 17 <= hour < 24:
            return Session.DEAD_ZONE
        else:
            return Session.OVERLAP

    # ── Technical calculations ───────────────────────────────

    @staticmethod
    def _wilder_smooth(data: np.ndarray, period: int) -> np.ndarray:
        """
        Wilder's smoothed moving average (EMA with alpha = 1/period).

        M5-FIX: The previous _calculate_adx() used np.convolve with a uniform
        kernel (simple moving average).  Standard ADX uses Wilder's smoothing
        which reacts faster to recent changes and produces higher ADX values in
        trending conditions.  SMA produces values ~10–30% lower, causing
        trending markets to be mis-classified as CHOPPY more often and
        suppressing trend-following signals at the wrong times.

        First value = sum of the first `period` elements (Wilder convention).
        Subsequent values = prev × (period−1)/period + current.
        Returns an array of length max(0, len(data)−period+1).
        """
        if len(data) < period:
            return np.array([])
        result = np.empty(len(data) - period + 1)
        result[0] = float(np.sum(data[:period]))
        k = (period - 1) / period
        for i in range(1, len(result)):
            result[i] = result[i - 1] * k + data[period - 1 + i]
        return result

    @staticmethod
    def _calculate_adx(highs, lows, closes, period=14) -> float:
        """Average Directional Index — measures trend strength (Wilder smoothing)"""
        try:
            if len(closes) < period + 1:
                return 0.0

            # True Range
            tr_list = []
            for i in range(1, len(closes)):
                hl = highs[i] - lows[i]
                hpc = abs(highs[i] - closes[i-1])
                lpc = abs(lows[i] - closes[i-1])
                tr_list.append(max(hl, hpc, lpc))

            tr = np.array(tr_list)

            # +DM / -DM
            pdm = np.maximum(np.diff(highs), 0)
            ndm = np.maximum(-np.diff(lows), 0)
            # Where +DM > -DM and +DM > 0
            pdm = np.where(pdm > ndm, pdm, 0)
            ndm = np.where(ndm > pdm, ndm, 0)

            # M5-FIX: Wilder smoothing instead of SMA
            atr_s = RegimeAnalyzer._wilder_smooth(tr, period)
            pdm_s = RegimeAnalyzer._wilder_smooth(pdm[:len(tr)], period)
            ndm_s = RegimeAnalyzer._wilder_smooth(ndm[:len(tr)], period)

            min_len = min(len(atr_s), len(pdm_s), len(ndm_s))
            if min_len == 0:
                return 0.0

            atr_s = atr_s[-min_len:]
            pdm_s = pdm_s[-min_len:]
            ndm_s = ndm_s[-min_len:]

            pdi = 100 * pdm_s / (atr_s + 1e-10)
            ndi = 100 * ndm_s / (atr_s + 1e-10)
            dx = 100 * np.abs(pdi - ndi) / (pdi + ndi + 1e-10)

            # Wilder-smooth DX to produce ADX, return the most recent value.
            # FIX (P0-A): _wilder_smooth initializes with sum(first N values)
            # which is correct for intermediate TR/DM smoothing (the scale
            # cancels in the DI ratio). But DX is already 0-100, so the sum
            # initialization inflates ADX by ~period (e.g. ADX=345 when true
            # ADX=24.6). Divide by period to normalize back to 0-100 range.
            adx_s = RegimeAnalyzer._wilder_smooth(dx, period)
            if len(adx_s) == 0:
                return 0.0
            adx_final = float(adx_s[-1]) / period
            return min(100.0, max(0.0, adx_final))

        except Exception:
            return 0.0

    @staticmethod
    def _calculate_atr(highs, lows, closes, period=14) -> float:
        """Average True Range — measures volatility"""
        try:
            if len(closes) < 2:
                return 0.0
            trs = []
            for i in range(1, min(len(closes), period + 1)):
                idx = len(closes) - i
                hl = highs[idx] - lows[idx]
                hpc = abs(highs[idx] - closes[idx-1])
                lpc = abs(lows[idx] - closes[idx-1])
                trs.append(max(hl, hpc, lpc))
            return float(np.mean(trs)) if trs else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _check_higher_highs(highs: np.ndarray) -> bool:
        """
        Check for ascending swing-high structure.

        L1-FIX: Previously triggered on just 2 swing highs where the last
        exceeded the previous — a single bounce on any 20-bar window qualified
        an entire bear trend as "higher highs".  Now requires at least 3 swing
        highs detected; the check still only requires the last to exceed the
        second-to-last, but the 3-point minimum confirms there is an actual
        swing structure rather than a lone local peak.
        """
        if len(highs) < 6:
            return False
        swing_highs = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                swing_highs.append(highs[i])
        if len(swing_highs) >= 3:
            return swing_highs[-1] > swing_highs[-2]
        return False

    @staticmethod
    def _check_lower_lows(lows: np.ndarray) -> bool:
        """
        Check for descending swing-low structure.

        L1-FIX (symmetric): same 3-swing-point minimum as _check_higher_highs.
        """
        if len(lows) < 6:
            return False
        swing_lows = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                swing_lows.append(lows[i])
        if len(swing_lows) >= 3:
            return swing_lows[-1] < swing_lows[-2]
        return False

    def get_summary(self) -> Dict:
        """Get a summary dict for Telegram display"""
        return {
            'regime': self._regime.value,
            'session': self._get_current_session().value,
            'alt_season_score': self._alt_season_score,
            'fear_greed': self._fear_greed,
            'btc_dominance': round(self._btc_dominance, 1),
            'btc_dominance_is_fresh': self.btc_dominance_is_fresh,
            'btc_adx': round(self._btc_adx, 1),
            'is_alt_season': self.is_alt_season,
            'is_killzone': self.is_killzone()[0],
            'macro_risk_off': self._macro_risk_off,
            'macro_risk_on': self._macro_risk_on,
            'dxy_rising': self._dxy_rising,
            'vix_elevated': self._vix_elevated,
            'state_confidence': round(self._state_confidence, 3),
            'crash_detected': self._crash_detected,
        }


# ── Singleton ──────────────────────────────────────────────
regime_analyzer = RegimeAnalyzer()
