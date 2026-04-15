"""
TitanBot Pro — Market State Engine (Institutional Upgrade)
===========================================================
Evaluates the MARKET ENVIRONMENT before evaluating any trade.

Institutions ask: "What environment are we trading in?"
Before they ask: "Is this a setup?"

Market States:
  TRENDING       → Clear directional move, trend-following strategies active
  EXPANSION      → Volatility expanding from compression, breakouts likely
  COMPRESSION    → Range tightening, volatility declining, breakout imminent
  LIQUIDITY_HUNT → Price sweeping stops before reversal (institutional behavior)
  ROTATION       → Sector/asset rotation in progress
  VOLATILE_PANIC → Fear/de-risking event, altcoins especially vulnerable
  NEUTRAL        → No dominant state, normal scanning

Why this matters:
  Same signal + different environment = very different outcome.
  Example: Breakout signal in COMPRESSION → likely winner.
           Breakout signal in LIQUIDITY_HUNT → likely stop hunt/fade.

Usage:
  from analyzers.market_state_engine import market_state_engine
  state = await market_state_engine.get_state()
  if state.blocks_strategy("InstitutionalBreakout"):
      skip signal
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.loader import cfg
from data.api_client import api

logger = logging.getLogger(__name__)

BREAKOUT_EXEMPTION_MAX_AGE_SECS = 600


class MarketState(str, Enum):
    TRENDING        = "TRENDING"
    EXPANSION       = "EXPANSION"
    COMPRESSION     = "COMPRESSION"
    LIQUIDITY_HUNT  = "LIQUIDITY_HUNT"
    ROTATION        = "ROTATION"
    VOLATILE_PANIC  = "VOLATILE_PANIC"
    NEUTRAL         = "NEUTRAL"


# ── Strategy behavior per market state ──────────────────────────────────
# Maps state → strategies that should be suppressed or penalized
STATE_STRATEGY_MATRIX: Dict[str, Dict] = {
    MarketState.TRENDING: {
        "boost":    ["Momentum", "MomentumContinuation", "Ichimoku", "IchimokuCloud", "ElliottWave"],
        "penalize": ["MeanReversion", "RangeScalper"],
        "block":    [],
        "confidence_mult": 1.05,
        "description": "Clear trend — ride momentum, avoid fading",
    },
    MarketState.EXPANSION: {
        "boost":    ["InstitutionalBreakout", "Momentum", "MomentumContinuation", "SmartMoneyConcepts"],
        "penalize": ["MeanReversion", "RangeScalper"],
        "block":    [],
        "confidence_mult": 1.08,
        "description": "Vol expanding — breakouts activating",
    },
    MarketState.COMPRESSION: {
        "boost":    ["RangeScalper", "MeanReversion", "FundingRateArb"],
        "penalize": ["InstitutionalBreakout", "Momentum", "MomentumContinuation"],
        "block":    [],
        "confidence_mult": 0.92,
        "description": "Vol compressed — range plays only, breakout risk low",
    },
    MarketState.LIQUIDITY_HUNT: {
        "boost":    ["SmartMoneyConcepts", "WyckoffAccDist", "Wyckoff", "ExtremeReversal"],
        "penalize": ["InstitutionalBreakout", "Momentum", "MomentumContinuation", "ElliottWave"],
        # InstitutionalBreakout moved from hard-block to heavy penalize so that
        # very high-conviction breakouts (conf 90+) can still survive the penalty
        # rather than being lost entirely. Markets are not binary — even stop-hunt
        # environments occasionally produce genuine breakouts.
        "block":    [],
        "confidence_mult": 0.85,
        "description": "Stop hunt in progress — fade the move, SMC only",
    },
    MarketState.ROTATION: {
        "boost":    ["SmartMoneyConcepts", "PriceAction"],
        "penalize": ["Momentum", "MomentumContinuation", "ElliottWave"],
        "block":    [],
        "confidence_mult": 0.90,
        "description": "Rotation — correlation breakdown, reduced size",
    },
    MarketState.VOLATILE_PANIC: {
        "boost":    ["FundingRateArb"],
        "penalize": ["Momentum", "MomentumContinuation", "InstitutionalBreakout", "ElliottWave", "Ichimoku", "IchimokuCloud"],
        "block":    ["Momentum", "MomentumContinuation", "InstitutionalBreakout"],
        "confidence_mult": 0.70,
        "description": "Panic mode — avoid altcoin longs, SMC/funding only",
    },
    MarketState.NEUTRAL: {
        "boost":    [],
        "penalize": [],
        "block":    [],
        "confidence_mult": 1.0,
        "description": "No dominant state — normal scan rules apply",
    },
}


@dataclass
class MarketStateResult:
    state: MarketState
    confidence: float           # 0.0 to 1.0 how confident in this classification
    direction_bias: str         # "BULLISH", "BEARISH", "NEUTRAL"
    btc_momentum_fast: float    # BTC short-term momentum (-1 to +1) — 6-bar
    btc_momentum_slow: float    # BTC longer momentum (-1 to +1) — 24-bar (Fix S1)
    vol_ratio: float            # Current ATR / 30-period ATR
    compression_bars: int       # How many bars in compression
    description: str
    signals: List[str] = field(default_factory=list)  # Evidence signals

    def blocks_strategy(self, strategy: str) -> bool:
        matrix = STATE_STRATEGY_MATRIX.get(self.state, {})
        return strategy in matrix.get("block", [])

    def get_confidence_mult(self, strategy: str) -> float:
        from governance.adaptive_params import adaptive_params as _ap
        matrix = STATE_STRATEGY_MATRIX.get(self.state, {})
        # Use live adaptive value if available; fall back to matrix default.
        # The ParamTuner adjusts these over time based on per-state win rates.
        _param_key = f"mse_mult_{self.state.value.lower()}"
        base = _ap.get(_param_key, matrix.get("confidence_mult", 1.0))

        # Scale the deviation from neutral (1.0) by classification confidence.
        # A VOLATILE_PANIC at 55% confidence should only half-apply its ×0.70 penalty:
        #   effective = 1.0 + (0.70 - 1.0) × 0.55  →  ×0.835 instead of ×0.70
        # This prevents low-confidence state classifications from imposing full penalties.
        if strategy in matrix.get("boost", []):
            raw = min(1.20, base + 0.08)
        elif strategy in matrix.get("penalize", []):
            raw = max(0.60, base - 0.12)
        else:
            raw = base

        deviation = raw - 1.0
        # Clamp state confidence to [0.5, 1.0] so a 50%-confident state still applies
        # half the penalty (avoids silently zeroing out all regime effects).
        scaled_conf = max(0.5, min(1.0, self.confidence))
        return round(1.0 + deviation * scaled_conf, 4)

    def get_description(self) -> str:
        matrix = STATE_STRATEGY_MATRIX.get(self.state, {})
        return matrix.get("description", self.description)


class MarketStateEngine:
    """
    Classifies the current macro market environment every cycle.
    Single source of truth for "what is the market doing right now?"
    """

    def __init__(self):
        self._state: Optional[MarketStateResult] = None
        self._last_update: float = 0
        self._cache_ttl: int = 180  # 3 minutes
        self._lock = asyncio.Lock()

        # State history for transition detection
        self._state_history: List[Tuple[float, MarketState]] = []

    # ─── Public API ────────────────────────────────────────────────────

    async def get_state(self) -> MarketStateResult:
        """Get current market state, refreshing if stale."""
        if time.time() - self._last_update < self._cache_ttl and self._state:
            return self._state

        async with self._lock:
            if time.time() - self._last_update < self._cache_ttl and self._state:
                return self._state
            try:
                await self._compute_state()
            except Exception as e:
                logger.error(f"Market state engine error: {e}")
                if not self._state:
                    self._state = MarketStateResult(
                        state=MarketState.NEUTRAL,
                        confidence=0.3,
                        direction_bias="NEUTRAL",
                        btc_momentum_fast=0.0,
                        btc_momentum_slow=0.0,
                        vol_ratio=1.0,
                        compression_bars=0,
                        description="Fallback — data unavailable",
                        signals=["data_error"],
                    )

        return self._state

    def get_transition_risk(self) -> float:
        """
        Returns 0.0–1.0 indicating how likely a regime transition is.
        High = reduce signal volume. Based on recent state changes.

        NOTE: callers should also check get_transition_type() — breakout
        transitions (COMPRESSION → EXPANSION/TRENDING) should NOT suppress
        signals even if the risk score is high.
        """
        if len(self._state_history) < 3:
            return 0.0

        recent = self._state_history[-5:]
        unique_states = len(set(s for _, s in recent))
        # Many different states recently = transition period
        return min(1.0, (unique_states - 1) / 4.0)

    def get_transition_type(self) -> str:
        """
        Returns the qualitative nature of the current transition:
          'breakout' — COMPRESSION → EXPANSION or TRENDING (best entry, do NOT suppress)
          'noise'    — rapid oscillation between unrelated states (suppress)
          'stable'   — no meaningful transition in progress

        Use this alongside get_transition_risk() so that breakout transitions
        don't get penalised the same way as random noise flipping.
        """
        if len(self._state_history) < 2:
            return "stable"

        recent_history = self._state_history[-4:]
        recent_states = [s for _, s in recent_history]

        # Directional breakout: compression resolving into expansion or trend
        _breakout_targets = {MarketState.EXPANSION, MarketState.TRENDING}
        if len(recent_history) >= 2:
            prev_ts, prev = recent_history[-2]
            curr_ts, curr = recent_history[-1]
            if (
                prev == MarketState.COMPRESSION
                and curr in _breakout_targets
                and (curr_ts - prev_ts) <= BREAKOUT_EXEMPTION_MAX_AGE_SECS
            ):
                return "breakout"
            # Multi-step: compression two reads ago, now expanding
            if len(recent_history) >= 3:
                prior_ts, prior = recent_history[-3]
                if (
                    prior == MarketState.COMPRESSION
                    and curr in _breakout_targets
                    and (curr_ts - prior_ts) <= BREAKOUT_EXEMPTION_MAX_AGE_SECS
                ):
                    return "breakout"

        # Noise: many distinct states in a short window
        unique = len(set(recent_states))
        if unique >= 3:
            return "noise"

        return "stable"

    # ─── Internal computation ───────────────────────────────────────────

    async def _compute_state(self):
        """Core classification logic — runs on BTC 1h + 15m data."""
        ohlcv_1h = await api.fetch_ohlcv("BTC/USDT", "1h", limit=100)
        ohlcv_15m = await api.fetch_ohlcv("BTC/USDT", "15m", limit=60)

        if not ohlcv_1h or len(ohlcv_1h) < 50:
            return

        closes_1h = np.array([float(c[4]) for c in ohlcv_1h])
        highs_1h  = np.array([float(c[2]) for c in ohlcv_1h])
        lows_1h   = np.array([float(c[3]) for c in ohlcv_1h])
        vols_1h   = np.array([float(c[5]) for c in ohlcv_1h])

        signals: List[str] = []

        # ── ATR ratio (volatility regime) ──────────────────────────────
        atr_now   = self._calc_atr(highs_1h[-14:], lows_1h[-14:], closes_1h[-14:])
        atr_slow  = self._calc_atr(highs_1h[-50:], lows_1h[-50:], closes_1h[-50:], period=50)
        vol_ratio = atr_now / atr_slow if atr_slow > 0 else 1.0

        # ── BTC momentum (fast & slow) ──────────────────────────────────
        btc_mom_fast = self._calc_momentum(closes_1h, period=6)   # 6h momentum
        btc_mom_slow = self._calc_momentum(closes_1h, period=24)  # 24h momentum

        # ── Compression detection ───────────────────────────────────────
        compression_bars, bb_squeeze = self._detect_compression(closes_1h, highs_1h, lows_1h)

        # ── Volume spike (expansion signal) ────────────────────────────
        vol_ma = np.mean(vols_1h[-20:])
        recent_vol = vols_1h[-3:].mean()
        vol_spike = recent_vol / vol_ma if vol_ma > 0 else 1.0

        # ── Liquidity hunt detection (stop sweep pattern) ──────────────
        liq_hunt = self._detect_liquidity_hunt(closes_1h, highs_1h, lows_1h, ohlcv_15m, volumes=vols_1h)

        # ── ADX trend strength ──────────────────────────────────────────
        adx = self._calc_adx(highs_1h[-30:], lows_1h[-30:], closes_1h[-30:])

        # ── BTC dominance / rotation proxy ─────────────────────────────
        # Rapid BTC mom divergence from price = rotation
        price_pct_1h = (closes_1h[-1] - closes_1h[-2]) / closes_1h[-2]

        # ── Classify ────────────────────────────────────────────────────
        state, confidence = self._classify(
            vol_ratio=vol_ratio,
            btc_mom_fast=btc_mom_fast,
            btc_mom_slow=btc_mom_slow,
            compression_bars=compression_bars,
            bb_squeeze=bb_squeeze,
            vol_spike=vol_spike,
            liq_hunt=liq_hunt,
            adx=adx,
            signals=signals,
        )

        # ── Direction bias ──────────────────────────────────────────────
        if btc_mom_fast > 0.015:
            direction_bias = "BULLISH"
        elif btc_mom_fast < -0.015:
            direction_bias = "BEARISH"
        else:
            direction_bias = "NEUTRAL"

        result = MarketStateResult(
            state=state,
            confidence=round(confidence, 3),
            direction_bias=direction_bias,
            btc_momentum_fast=round(btc_mom_fast, 5),
            btc_momentum_slow=round(btc_mom_slow, 5),   # Fix S1: expose 24-bar momentum
            vol_ratio=round(vol_ratio, 3),
            compression_bars=compression_bars,
            description=STATE_STRATEGY_MATRIX.get(state, {}).get("description", ""),
            signals=signals,
        )

        # Track state history for transition detection
        self._state_history.append((time.time(), state))
        if len(self._state_history) > 20:
            self._state_history.pop(0)

        self._state = result
        self._last_update = time.time()

        logger.info(
            f"🧠 Market State: {state.value} | bias={direction_bias} | "
            f"vol_ratio={vol_ratio:.2f} | compress={compression_bars}bars | "
            f"liq_hunt={liq_hunt} | conf={confidence:.2f}"
        )

    def _classify(
        self,
        vol_ratio: float,
        btc_mom_fast: float,
        btc_mom_slow: float,
        compression_bars: int,
        bb_squeeze: bool,
        vol_spike: bool,
        liq_hunt: bool,
        adx: float,
        signals: List[str],
    ) -> Tuple[MarketState, float]:
        """Multi-factor state classification."""

        # ── LIQUIDITY_HUNT (highest priority — overrides everything) ──
        if liq_hunt:
            signals.append("liquidity_hunt_pattern")
            return MarketState.LIQUIDITY_HUNT, 0.75

        # ── VOLATILE_PANIC ─────────────────────────────────────────────
        # BTC down hard + vol spike + negative momentum both timeframes
        if (btc_mom_fast < -0.04
                and vol_ratio > 1.8
                and btc_mom_slow < -0.05):
            signals.append("panic_cascade")
            return MarketState.VOLATILE_PANIC, 0.80

        # ── EXPANSION ─────────────────────────────────────────────────
        # ATR rising + volume spike = expansion from compression
        if vol_ratio > 1.5 and vol_spike and compression_bars >= 4:
            signals.append("compression_breakout")
            return MarketState.EXPANSION, 0.78

        if vol_ratio > 1.8 and adx > 25:
            signals.append("vol_expansion")
            return MarketState.EXPANSION, 0.70

        # ── COMPRESSION ───────────────────────────────────────────────
        if bb_squeeze and compression_bars >= 6:
            signals.append("bb_squeeze")
            return MarketState.COMPRESSION, 0.72

        if vol_ratio < 0.65 and compression_bars >= 4:
            signals.append("atr_compression")
            return MarketState.COMPRESSION, 0.65

        # ── TRENDING ──────────────────────────────────────────────────
        if adx > 28 and abs(btc_mom_fast) > 0.02 and abs(btc_mom_slow) > 0.02:
            # Both timeframes agree on direction
            if btc_mom_fast * btc_mom_slow > 0:
                signals.append("strong_trend")
                return MarketState.TRENDING, 0.75

        # ── ROTATION ──────────────────────────────────────────────────
        # Moderate BTC move but high volume divergence
        if 0.005 < abs(btc_mom_fast) < 0.03 and vol_ratio > 1.3 and adx < 22:
            signals.append("rotation_signal")
            return MarketState.ROTATION, 0.55

        # ── NEUTRAL ───────────────────────────────────────────────────
        signals.append("no_dominant_state")
        return MarketState.NEUTRAL, 0.50

    # ─── Technical helpers ──────────────────────────────────────────────

    @staticmethod
    def _calc_atr(highs, lows, closes, period=14) -> float:
        if len(closes) < 2:
            return 0.0
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))

    @staticmethod
    def _calc_momentum(closes: np.ndarray, period: int = 6) -> float:
        """Normalized momentum: (close[-1] - close[-period]) / close[-period]"""
        if len(closes) < period + 1:
            return 0.0
        return float((closes[-1] - closes[-period]) / closes[-period])

    @staticmethod
    def _calc_adx(highs, lows, closes, period=14) -> float:
        if len(closes) < period + 2:
            return 0.0
        try:
            high_diff = np.diff(highs)
            low_diff  = -np.diff(lows)
            plus_dm   = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
            minus_dm  = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
            )
            atr_s   = np.mean(tr[-period:])
            plus_s  = np.mean(plus_dm[-period:])
            minus_s = np.mean(minus_dm[-period:])
            if atr_s == 0:
                return 0.0
            plus_di  = 100 * plus_s  / atr_s
            minus_di = 100 * minus_s / atr_s
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            return float(dx)
        except Exception:
            return 0.0

    @staticmethod
    def _detect_compression(closes, highs, lows, window=20) -> Tuple[int, bool]:
        """
        Detect Bollinger Band squeeze + count consecutive compression bars.
        Returns (compression_bars, is_squeezed)
        """
        if len(closes) < window + 5:
            return 0, False
        try:
            recent = closes[-window:]
            std_now  = np.std(recent[-5:])
            std_slow = np.std(recent)
            bb_squeeze = std_now < std_slow * 0.70  # Bands tightening

            # Count bars where each candle's range < 30-period avg range
            ranges = highs - lows
            avg_range = np.mean(ranges[-30:]) if len(ranges) >= 30 else np.mean(ranges)
            comp_bars = 0
            for i in range(len(ranges) - 1, max(0, len(ranges) - 20), -1):
                if ranges[i] < avg_range * 0.65:
                    comp_bars += 1
                else:
                    break  # Must be consecutive

            return comp_bars, bb_squeeze
        except Exception:
            return 0, False

    def _detect_liquidity_hunt(
        self, closes, highs, lows, ohlcv_15m, volumes=None
    ) -> bool:
        """
        Detect institutional stop-hunt pattern:
        - Price sweeps a recent high/low (wick > 2x body)
        - Then closes back inside previous range (rejection)
        - Volume spike on the sweep candle (Phase 9-12 audit fix)

        Previously the docstring documented volume confirmation but the
        implementation never checked it, leading to false positives on
        thin-wick candles without unusual volume.
        """
        if len(closes) < 10:
            return False
        if volumes is None:
            logger.debug(
                "_detect_liquidity_hunt: volumes=None — volume confirmation "
                "bypassed; sweep candles accepted without volume check"
            )
        try:
            # Pre-compute median volume for spike confirmation
            _has_vol = volumes is not None and len(volumes) >= 10
            _vol_median = float(np.median(volumes[-20:])) if _has_vol else 0.0

            # Check last 3 candles on 1h
            for i in range(-3, 0):
                candle_high = highs[i]
                candle_low  = lows[i]
                prev_high   = np.max(highs[i-10:i])
                prev_low    = np.min(lows[i-10:i])

                # Volume confirmation: candle must have ≥1.3× median volume
                # (institutional sweeps carry above-average volume)
                if _has_vol and _vol_median > 0:
                    candle_vol = float(volumes[i])
                    if candle_vol < _vol_median * 1.3:
                        continue  # skip — no volume spike on this candle

                # Upper wick sweep: exceeded recent high but closed back below it
                upper_wick = candle_high - max(closes[i], closes[i-1])
                body       = abs(closes[i] - closes[i-1])

                if candle_high > prev_high * 1.002 and upper_wick > body * 1.8:
                    return True

                # Lower wick sweep: exceeded recent low but closed back above it
                lower_wick = min(closes[i], closes[i-1]) - candle_low
                if candle_low < prev_low * 0.998 and lower_wick > body * 1.8:
                    return True

        except Exception:
            pass
        return False


# ── Singleton ───────────────────────────────────────────────────────────
market_state_engine = MarketStateEngine()
