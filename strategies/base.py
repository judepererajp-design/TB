"""
TitanBot Pro — Base Strategy
==============================
All strategies extend BaseStrategy and return SignalResult objects.

This module also provides:
  - SignalResult: data class for a complete trading signal
  - SignalDirection: LONG / SHORT enum
  - cfg_min_rr(setup_class): returns minimum R:R for a given setup class
  - SETUP_CLASS_ENTRY_TF: maps setup class → primary entry timeframe
  - set_rr_floor_override / get_rr_floor_overrides: runtime R:R overrides
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.loader import cfg

logger = logging.getLogger(__name__)

# ── R:R floor overrides (set by DiagnosticEngine at runtime) ──────────────
_rr_floor_overrides: Dict[str, float] = {}


def set_rr_floor_override(strategy: str, rr: float) -> None:
    """Override the minimum R:R floor for a specific strategy at runtime."""
    _rr_floor_overrides[strategy] = rr


def get_rr_floor_overrides() -> Dict[str, float]:
    """Return a copy of all active R:R floor overrides."""
    return dict(_rr_floor_overrides)


def cfg_min_rr(setup_class: str = "intraday") -> float:
    """
    Return minimum R:R for the given setup class.
    
    Reads from cfg.risk.min_rr_by_class first, then falls back to
    hard-coded defaults. Diagnostic engine overrides are checked last.
    """
    defaults = {
        "scalp":     1.3,
        "intraday":  1.5,
        "swing":     2.0,
        "positional": 2.5,
    }
    try:
        by_class = cfg.risk.get('min_rr_by_class', {})
        if isinstance(by_class, dict):
            base = float(by_class.get(setup_class, defaults.get(setup_class, 1.5)))
        else:
            base = defaults.get(setup_class, 1.5)
    except Exception:
        base = defaults.get(setup_class, 1.5)
    return base


# ── Setup class → primary entry timeframe ────────────────────────────────
SETUP_CLASS_ENTRY_TF: Dict[str, str] = {
    "scalp":      "15m",
    "intraday":   "1h",
    "swing":      "4h",
    "positional": "1d",
}

# ── Setup class → confirmation / trigger timeframe ───────────────────────
# Execution timing should stay setup-aware without dropping all the way to
# noisy 5m candles for every signal class. These are one tier faster than the
# primary entry timeframe so trigger checks remain responsive but context-aware.
SETUP_CLASS_CONFIRM_TF: Dict[str, str] = {
    "scalp":      "5m",
    "intraday":   "15m",
    "swing":      "1h",
    "positional": "4h",
}


# ── Signal Direction ──────────────────────────────────────────────────────
class SignalDirection(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


# ── Signal Result ─────────────────────────────────────────────────────────
@dataclass
class SignalResult:
    """
    A complete, validated trading signal emitted by a strategy.
    
    All prices are in the quote currency (USDT for most pairs).
    """
    symbol: str
    direction: SignalDirection
    strategy: str
    confidence: float           # 0-100

    # Entry zone (inclusive)
    entry_low:  float
    entry_high: float

    # Risk levels
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float] = None

    rr_ratio: float = 0.0
    atr: Optional[float] = None          # ATR at signal time — used for TP3 clamp

    # Metadata
    setup_class: str = "intraday"         # scalp | intraday | swing | positional
    timeframe: str = "1h"                 # Primary analysis timeframe
    entry_timeframe: str = ""             # Entry trigger timeframe (set by engine)
    analysis_timeframes: List[str] = field(default_factory=list)

    # Confluence and context
    confluence: List[str] = field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None
    setup_context: Optional[Dict[str, Any]] = None
    execution_context: Optional[Dict[str, Any]] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    regime: Optional[str] = None
    sector: Optional[str] = None
    tier: Optional[int] = None


# ── Base Strategy ─────────────────────────────────────────────────────────
class BaseStrategy(ABC):
    """
    Abstract base class for all TitanBot strategies.
    
    Provides:
      - Indicator calculation helpers (RSI, ATR, ADX, Bollinger Bands)
      - Signal validation (geometry, R:R floor, direction consistency)
      - Config accessor shortcut
    """

    name: str = "BaseStrategy"
    description: str = ""

    # Class-level cache shared across all strategy instances for a given scan cycle.
    # Cleared by clear_indicator_cache() before each symbol is processed.
    _class_indicator_cache: Dict[str, Any] = {}

    def __init__(self):
        self._indicator_cache: Dict[str, Any] = {}

    @classmethod
    def clear_indicator_cache(cls) -> None:
        """Clear the class-level indicator cache before processing a new symbol."""
        cls._class_indicator_cache.clear()

    # ── Abstract interface ─────────────────────────────────────────────
    @abstractmethod
    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        """
        Analyse the symbol and return a SignalResult if a setup is found.
        Returns None if no setup qualifies.
        """

    # ── Indicator helpers ──────────────────────────────────────────────
    @staticmethod
    def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
        """Wilder RSI. Returns the most recent RSI value (0-100)."""
        closes = np.asarray(closes, dtype=float)
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        # Wilder's smoothing
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 2)

    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                       period: int = 14) -> float:
        """Average True Range using Wilder's smoothing."""
        highs  = np.asarray(highs,  dtype=float)
        lows   = np.asarray(lows,   dtype=float)
        closes = np.asarray(closes, dtype=float)
        if len(highs) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            )
            trs.append(tr)
        trs = np.array(trs)
        atr = np.mean(trs[:period])
        for i in range(period, len(trs)):
            atr = (atr * (period - 1) + trs[i]) / period
        return float(atr)

    @staticmethod
    def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                       period: int = 14) -> float:
        """Wilder ADX. Returns the most recent ADX value (0-100)."""
        highs  = np.asarray(highs,  dtype=float)
        lows   = np.asarray(lows,   dtype=float)
        closes = np.asarray(closes, dtype=float)
        if len(highs) < period * 2 + 1:
            return 0.0
        dm_plus  = []
        dm_minus = []
        trs      = []
        for i in range(1, len(highs)):
            h_diff = highs[i]  - highs[i - 1]
            l_diff = lows[i - 1] - lows[i]
            dm_plus.append( max(h_diff, 0) if h_diff > l_diff else 0)
            dm_minus.append(max(l_diff, 0) if l_diff > h_diff else 0)
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            ))
        dm_plus  = np.array(dm_plus,  dtype=float)
        dm_minus = np.array(dm_minus, dtype=float)
        trs      = np.array(trs,      dtype=float)

        # Wilder smooth
        def _smooth(arr: np.ndarray) -> np.ndarray:
            s = np.zeros(len(arr))
            s[period - 1] = np.sum(arr[:period])
            for i in range(period, len(arr)):
                s[i] = s[i - 1] - s[i - 1] / period + arr[i]
            return s

        atr_s     = _smooth(trs)
        dmp_s     = _smooth(dm_plus)
        dmm_s     = _smooth(dm_minus)
        di_plus   = np.divide(100 * dmp_s, atr_s, out=np.zeros_like(atr_s), where=atr_s > 0)
        di_minus  = np.divide(100 * dmm_s, atr_s, out=np.zeros_like(atr_s), where=atr_s > 0)
        di_sum    = di_plus + di_minus
        dx        = np.divide(100 * np.abs(di_plus - di_minus), di_sum, out=np.zeros_like(di_sum), where=di_sum > 0)
        # ADX = smoothed DX
        adx_arr = np.zeros(len(dx))
        adx_arr[period - 1] = np.mean(dx[:period])
        for i in range(period, len(dx)):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period
        return float(adx_arr[-1])

    @staticmethod
    def calculate_bollinger(closes: np.ndarray, period: int = 20,
                              std_mult: float = 2.0) -> Tuple[float, float, float]:
        """
        Bollinger Bands.
        Returns (mid, upper, lower).
        """
        closes = np.asarray(closes, dtype=float)
        if len(closes) < period:
            mid = float(closes[-1])
            return mid, mid, mid
        window = closes[-period:]
        mid    = float(np.mean(window))
        std    = float(np.std(window, ddof=1))
        return mid, mid + std_mult * std, mid - std_mult * std

    @staticmethod
    def calculate_effective_rr(
        direction: SignalDirection | str,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        tp2: float,
    ) -> float:
        """Calculate R:R from entry midpoint, matching aggregator enforcement."""
        direction_str = direction.value if hasattr(direction, "value") else str(direction)
        entry_mid = (entry_low + entry_high) / 2.0

        if direction_str == SignalDirection.LONG.value:
            risk = entry_mid - stop_loss
            reward = tp2 - entry_mid
        else:
            risk = stop_loss - entry_mid
            reward = entry_mid - tp2

        if risk <= 0 or reward <= 0:
            return 0.0
        return round(reward / risk, 2)

    # ── Compression / Squeeze Detection ──────────────────────────────

    @staticmethod
    def detect_bb_squeeze(closes: np.ndarray, period: int = 20,
                          lookback: int = 100) -> Dict[str, Any]:
        """
        Detect Bollinger Band squeeze (compression).

        Returns dict with:
          - is_squeeze: bool — current BB width is below 20th percentile
          - bandwidth_pctile: float 0-1 — where current width sits historically
          - compression_bars: int — consecutive bars below 30th pctile
          - bandwidth: float — current BB width as percentage of mid
        """
        closes = np.asarray(closes, dtype=float)
        result = {
            "is_squeeze": False,
            "bandwidth_pctile": 0.5,
            "compression_bars": 0,
            "bandwidth": 0.0,
        }
        if len(closes) < max(period, 30):
            return result

        # Calculate BB width series for the lookback
        n = min(lookback, len(closes) - period)
        widths = []
        for i in range(n):
            idx = len(closes) - n + i
            window = closes[idx - period + 1:idx + 1]
            if len(window) < period:
                continue
            mid = float(np.mean(window))
            std = float(np.std(window, ddof=1))
            bw = (2 * std * 2.0) / mid if mid > 0 else 0  # 2-sigma width / mid
            widths.append(bw)

        if len(widths) < 10:
            return result

        current_bw = widths[-1]
        sorted_bw = sorted(widths)
        pctile = float(np.searchsorted(sorted_bw, current_bw)) / len(sorted_bw)

        # Count consecutive compression bars (below 30th percentile)
        threshold_30 = sorted_bw[int(len(sorted_bw) * 0.30)]
        comp_bars = 0
        for bw in reversed(widths):
            if bw <= threshold_30:
                comp_bars += 1
            else:
                break

        result["is_squeeze"] = pctile < 0.20
        result["bandwidth_pctile"] = round(pctile, 3)
        result["compression_bars"] = comp_bars
        result["bandwidth"] = round(current_bw, 6)
        return result

    # ── Parabolic / Acceleration Detection ───────────────────────────

    @staticmethod
    def detect_parabolic(closes: np.ndarray, period: int = 10) -> Dict[str, Any]:
        """
        Detect parabolic price acceleration using ROC-of-ROC (second derivative).

        Returns dict with:
          - is_parabolic: bool — acceleration exceeds threshold
          - acceleration: float — rate of change of ROC (positive = accelerating up)
          - roc: float — current rate of change
          - direction: str — "UP", "DOWN", or "FLAT"
        """
        closes = np.asarray(closes, dtype=float)
        result = {
            "is_parabolic": False,
            "acceleration": 0.0,
            "roc": 0.0,
            "direction": "FLAT",
        }
        if len(closes) < period * 2 + 2:
            return result

        # ROC series: % change over `period` bars
        roc_series = []
        for i in range(period, len(closes)):
            prev = closes[i - period]
            if prev == 0:
                roc_series.append(0.0)
            else:
                roc_series.append((closes[i] - prev) / prev)

        if len(roc_series) < period + 1:
            return result

        current_roc = roc_series[-1]

        # ROC of ROC (acceleration)
        roc_of_roc = []
        for i in range(1, len(roc_series)):
            roc_of_roc.append(roc_series[i] - roc_series[i - 1])

        if not roc_of_roc:
            return result

        acceleration = roc_of_roc[-1]

        # Determine direction and parabolic threshold
        # Parabolic = acceleration consistently in one direction and exceeding 0.5% per bar
        _recent_accel = roc_of_roc[-3:] if len(roc_of_roc) >= 3 else roc_of_roc
        _avg_accel = float(np.mean(_recent_accel))
        _all_same_sign = all(a > 0 for a in _recent_accel) or all(a < 0 for a in _recent_accel)

        is_parabolic = _all_same_sign and abs(_avg_accel) > 0.005  # 0.5% acceleration per bar

        if current_roc > 0.01:
            direction = "UP"
        elif current_roc < -0.01:
            direction = "DOWN"
        else:
            direction = "FLAT"

        result["is_parabolic"] = is_parabolic
        result["acceleration"] = round(acceleration, 6)
        result["roc"] = round(current_roc, 6)
        result["direction"] = direction
        return result

    # ── Exhaustion Detection ─────────────────────────────────────────

    @staticmethod
    def detect_exhaustion(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        histogram: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Detect momentum exhaustion signals:
          - Histogram deceleration (MACD histogram shrinking)
          - Candle range contraction (consecutive smaller candles)
          - Volume climax (spike volume at extreme = capitulation)

        Returns dict with:
          - is_exhausted: bool — 2+ exhaustion signals present
          - signals: list of detected exhaustion signals
          - score: float 0-1 — exhaustion intensity
        """
        opens   = np.asarray(opens,   dtype=float)
        highs   = np.asarray(highs,   dtype=float)
        lows    = np.asarray(lows,    dtype=float)
        closes  = np.asarray(closes,  dtype=float)
        volumes = np.asarray(volumes, dtype=float)

        signals = []
        score = 0.0

        if len(closes) < 10:
            return {"is_exhausted": False, "signals": [], "score": 0.0}

        # 1. Histogram deceleration
        if histogram is not None and len(histogram) >= 4:
            h = np.asarray(histogram, dtype=float)
            # Check if absolute histogram is shrinking for 3+ bars
            if (abs(h[-1]) < abs(h[-2]) < abs(h[-3])):
                signals.append("histogram_deceleration")
                score += 0.3

        # 2. Candle range contraction
        ranges = highs - lows
        if len(ranges) >= 5:
            # 3 consecutive shrinking candle ranges
            if ranges[-1] < ranges[-2] < ranges[-3]:
                signals.append("range_contraction")
                score += 0.25
            # Tiny candles relative to recent average
            avg_range = float(np.mean(ranges[-20:]))
            if avg_range > 0 and ranges[-1] < avg_range * 0.4:
                signals.append("doji_exhaustion")
                score += 0.15

        # 3. Volume climax (spike volume at extreme)
        if len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            if avg_vol > 0:
                vol_ratio = volumes[-1] / avg_vol
                if vol_ratio > 3.0:
                    signals.append("volume_climax")
                    score += 0.35
                elif vol_ratio > 2.0:
                    signals.append("volume_spike")
                    score += 0.15

        # 4. Waning momentum: smaller bodies with bigger wicks
        if len(closes) >= 3:
            bodies = [abs(closes[-i] - opens[-i]) for i in range(1, 4)]
            total_ranges = [highs[-i] - lows[-i] for i in range(1, 4)]
            # Body shrinking while range stays same = indecision
            if (total_ranges[0] > 0 and total_ranges[1] > 0
                    and bodies[0] < bodies[1] * 0.6
                    and total_ranges[0] >= total_ranges[1] * 0.7):
                signals.append("waning_momentum")
                score += 0.2

        return {
            "is_exhausted": len(signals) >= 2,
            "signals": signals,
            "score": min(1.0, round(score, 3)),
        }

    # ── Signal validation ──────────────────────────────────────────────
    def validate_signal(self, sig: SignalResult) -> bool:
        """
        Validate signal geometry and minimum R:R.
        Returns True if the signal passes all checks.
        """
        try:
            if sig.entry_low >= sig.entry_high:
                logger.debug(f"validate_signal: {sig.symbol} entry zone inverted")
                return False

            if sig.direction == SignalDirection.LONG:
                if sig.stop_loss >= sig.entry_low:
                    logger.debug(f"validate_signal: {sig.symbol} LONG SL >= entry_low")
                    return False
                if sig.tp1 <= sig.entry_high:
                    logger.debug(f"validate_signal: {sig.symbol} LONG TP1 <= entry_high")
                    return False
            else:  # SHORT
                if sig.stop_loss <= sig.entry_high:
                    logger.debug(f"validate_signal: {sig.symbol} SHORT SL <= entry_high")
                    return False
                if sig.tp1 >= sig.entry_low:
                    logger.debug(f"validate_signal: {sig.symbol} SHORT TP1 >= entry_low")
                    return False

            min_rr = cfg_min_rr(sig.setup_class)
            if sig.rr_ratio < min_rr:
                logger.debug(
                    f"validate_signal: {sig.symbol} R:R {sig.rr_ratio:.2f} < "
                    f"min {min_rr:.2f} ({sig.setup_class})"
                )
                return False

            return True
        except Exception as e:
            logger.debug(f"validate_signal error: {e}")
            return False

    # ── Config helper ──────────────────────────────────────────────────
    def _get_cfg(self, key: str, default: Any = None) -> Any:
        """Get a value from this strategy's config section with a default fallback."""
        try:
            cfg_node = getattr(self, '_cfg', None)
            if cfg_node is None:
                return default
            val = getattr(cfg_node, key, None)
            if val is None:
                val = cfg_node.get(key, default) if hasattr(cfg_node, 'get') else default
            return val if val is not None else default
        except Exception:
            return default


# ── Confidence normalization ──────────────────────────────────────────────
# Each strategy uses a different base confidence and cap, making raw
# confidence values non-comparable across strategies.  This map records
# the observed [floor, cap] for each strategy so we can rescale to a
# common [NORM_FLOOR, NORM_CAP] range.
#
# The floor is the minimum confidence any strategy will produce (after
# the min() clamp inside each strategy) and the cap is the max.

_STRATEGY_CONFIDENCE_RANGE: Dict[str, Tuple[float, float]] = {
    # strategy_name: (floor, cap)
    "SmartMoneyConcepts":    (40, 95),
    "InstitutionalBreakout": (40, 95),
    "ExtremeReversal":       (40, 95),
    "MeanReversion":         (40, 92),
    "PriceAction":           (40, 93),
    "Momentum":              (40, 93),
    "MomentumContinuation":  (40, 93),
    "Ichimoku":              (40, 94),
    "IchimokuCloud":         (40, 94),
    "ElliottWave":           (40, 94),
    "FundingRateArb":        (40, 92),
    "RangeScalper":          (40, 88),
    "WyckoffAccDist":        (40, 95),
    "Wyckoff":               (40, 95),
    "HarmonicPattern":       (40, 88),
    "HarmonicDetector":      (40, 88),
    "GeometricPattern":      (40, 90),
    "GeometricPatterns":     (40, 90),
}

# Common normalised range — all strategies map to this after normalisation.
NORM_FLOOR = 40.0
NORM_CAP   = 95.0


def normalize_confidence(strategy_name: str, raw_confidence: float) -> float:
    """
    Rescale a strategy's raw confidence from its native [floor, cap]
    to the common [NORM_FLOOR, NORM_CAP] range so that a 75 from
    RangeScalper and a 75 from SMC carry equivalent conviction.

    If the strategy is unknown, the raw value is returned unchanged
    (safe default — avoids breaking new strategies).
    """
    bounds = _STRATEGY_CONFIDENCE_RANGE.get(strategy_name)
    if bounds is None:
        return raw_confidence

    src_floor, src_cap = bounds
    # Guard against division by zero (floor == cap)
    if src_cap <= src_floor:
        return raw_confidence

    # Linear rescale: [src_floor, src_cap] → [NORM_FLOOR, NORM_CAP]
    clamped = max(src_floor, min(src_cap, raw_confidence))
    ratio = (clamped - src_floor) / (src_cap - src_floor)
    return round(NORM_FLOOR + ratio * (NORM_CAP - NORM_FLOOR), 2)
