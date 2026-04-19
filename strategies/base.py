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
import hashlib
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

    # B-7: error-rate counter for parabolic_detector enrichment failures.
    # Incremented each time the enrichment except-branch fires; reset to zero
    # on a successful enrichment.  A WARNING is emitted at every
    # _PARABOLIC_ERROR_WARN_EVERY-th failure so persistent regressions in
    # analyzers.parabolic_detector are visible without spamming the log.
    _parabolic_error_count: int = 0
    _PARABOLIC_ERROR_WARN_EVERY: int = 5

    def __init__(self):
        self._indicator_cache: Dict[str, Any] = {}

    @classmethod
    def clear_indicator_cache(cls) -> None:
        """Clear the class-level indicator cache before processing a new symbol."""
        cls._class_indicator_cache.clear()

    @classmethod
    def _array_fingerprint(cls, arr: np.ndarray) -> str:
        """Stable content fingerprint for cache keys."""
        a = np.ascontiguousarray(np.asarray(arr, dtype=float))
        h = hashlib.blake2b(digest_size=16)
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
        return h.hexdigest()

    # ── Abstract interface ─────────────────────────────────────────────
    @abstractmethod
    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        """
        Analyse the symbol and return a SignalResult if a setup is found.
        Returns None if no setup qualifies.
        """

    # ── Multi-Timeframe Helpers ───────────────────────────────────────
    # Pattern: "4h direction + 1h zone + 15m trigger"
    # Higher TF sets the bias, middle TF identifies the zone, lowest TF times entry.

    @staticmethod
    def mtf_get_bias(ohlcv_dict: Dict, tf: str = "4h") -> Dict[str, Any]:
        """
        Determine higher-timeframe directional bias.

        Returns dict with:
          - bias: "BULLISH" / "BEARISH" / "NEUTRAL"
          - adx: float — trend strength
          - structure: "HH_HL" / "LH_LL" / "MIXED"
          - confidence: float 0-100
        """
        import pandas as pd
        result = {"bias": "NEUTRAL", "adx": 0.0, "structure": "MIXED", "confidence": 50.0}

        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 50:
            return result

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values

        # ADX for trend strength
        adx = BaseStrategy.calculate_adx(highs, lows, closes, period=14)
        result["adx"] = adx

        # Swing structure: HH/HL (bullish) vs LH/LL (bearish)
        swing_highs = []
        swing_lows = []
        # Scan from most recent to older pivots (i=2 => idx=-3 first), so
        # list index 0 is the most recent confirmed swing.
        for i in range(2, min(20, len(highs) - 1)):
            idx = -(i + 1)
            # Guard left-neighbor access at idx-1 for very short series.
            if idx - 1 < -len(highs):
                continue
            if highs[idx] > highs[idx - 1] and highs[idx] > highs[idx + 1]:
                swing_highs.append(highs[idx])
            if lows[idx] < lows[idx - 1] and lows[idx] < lows[idx + 1]:
                swing_lows.append(lows[idx])

        # Require at least 3 pivots so structure is not inferred from one bounce.
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            hh = swing_highs[0] > swing_highs[1] > swing_highs[2]  # Higher highs
            hl = swing_lows[0] > swing_lows[1] > swing_lows[2]     # Higher lows
            lh = swing_highs[0] < swing_highs[1] < swing_highs[2]  # Lower highs
            ll = swing_lows[0] < swing_lows[1] < swing_lows[2]     # Lower lows

            if hh and hl:
                result["structure"] = "HH_HL"
                result["bias"] = "BULLISH"
                result["confidence"] = min(90, 60 + adx * 0.5)
            elif lh and ll:
                result["structure"] = "LH_LL"
                result["bias"] = "BEARISH"
                result["confidence"] = min(90, 60 + adx * 0.5)
            else:
                result["structure"] = "MIXED"
                result["confidence"] = 40.0

        # ADX confirmation
        if adx > 25 and result["bias"] != "NEUTRAL":
            result["confidence"] = min(95, result["confidence"] + 10)
        elif adx < 15:
            result["bias"] = "NEUTRAL"
            result["confidence"] = max(30, result["confidence"] - 15)

        return result

    @staticmethod
    def mtf_find_zone(ohlcv_dict: Dict, tf: str = "1h", bias: str = "BULLISH") -> Optional[Dict[str, Any]]:
        """
        Find a supply/demand zone on the mid-timeframe aligned with HTF bias.

        Returns dict with:
          - zone_type: "demand" or "supply"
          - zone_high: float
          - zone_low: float
          - strength: float 0-1
        or None if no valid zone found.
        """
        import pandas as pd

        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 40:
            return None

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

        opens   = df["open"].values
        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values
        volumes = df["volume"].values

        atr = BaseStrategy.calculate_atr(highs, lows, closes, period=14)
        if atr == 0:
            return None

        current = closes[-1]
        avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1.0

        best_zone = None
        best_strength = 0.0

        # Look for order blocks (last opposing candle before impulse)
        for i in range(2, min(30, len(closes) - 2)):
            if bias == "BULLISH":
                # Bullish OB: bearish candle before bullish impulse
                ob_body = opens[-i] - closes[-i]
                if (ob_body > 0
                        and closes[-i + 1] > opens[-i + 1]
                        and (closes[-i + 1] - opens[-i + 1]) > ob_body * 0.5):
                    ob_high = opens[-i]
                    ob_low = closes[-i]
                    # Zone must be below current price (demand zone)
                    if ob_high < current and ob_high > current - atr * 3:
                        # Strength: recency + impulse size + volume
                        recency_bonus = max(0, 1 - i / 30)
                        impulse_size = (closes[-i + 1] - opens[-i + 1]) / atr
                        vol_bonus = min(0.3, (volumes[-i + 1] / avg_vol - 1) * 0.15) if avg_vol > 0 else 0
                        strength = min(1.0, recency_bonus * 0.4 + min(1, impulse_size / 3) * 0.4 + vol_bonus)
                        if strength > best_strength:
                            best_strength = strength
                            best_zone = {
                                "zone_type": "demand",
                                "zone_high": ob_high,
                                "zone_low": ob_low,
                                "strength": round(strength, 3),
                            }
            else:
                # Bearish OB: bullish candle before bearish impulse
                ob_body = closes[-i] - opens[-i]
                if (ob_body > 0
                        and closes[-i + 1] < opens[-i + 1]
                        and (opens[-i + 1] - closes[-i + 1]) > ob_body * 0.5):
                    ob_high = closes[-i]
                    ob_low = opens[-i]
                    if ob_high > current and ob_high < current + atr * 3:
                        recency_bonus = max(0, 1 - i / 30)
                        impulse_size = (opens[-i + 1] - closes[-i + 1]) / atr
                        vol_bonus = min(0.3, (volumes[-i + 1] / avg_vol - 1) * 0.15) if avg_vol > 0 else 0
                        strength = min(1.0, recency_bonus * 0.4 + min(1, impulse_size / 3) * 0.4 + vol_bonus)
                        if strength > best_strength:
                            best_strength = strength
                            best_zone = {
                                "zone_type": "supply",
                                "zone_high": ob_high,
                                "zone_low": ob_low,
                                "strength": round(strength, 3),
                            }

        return best_zone

    @staticmethod
    def mtf_check_trigger(
        ohlcv_dict: Dict,
        tf: str = "15m",
        direction: str = "LONG",
        zone: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Check for an entry trigger on the lowest timeframe.

        Looks for:
          - Price at/near the zone
          - Rejection candle (pin bar / engulfing)
          - Volume confirmation

        Returns dict with:
          - triggered: bool
          - trigger_type: str (e.g., "rejection_candle", "engulfing")
          - quality: float 0-1
        """
        import pandas as pd

        result = {"triggered": False, "trigger_type": "none", "quality": 0.0}

        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 20:
            return result

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

        opens   = df["open"].values
        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values
        volumes = df["volume"].values

        atr = BaseStrategy.calculate_atr(highs, lows, closes, period=14)
        if atr == 0:
            return result

        current = closes[-1]
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        body = abs(c - o)
        total = h - l

        # Zone proximity check
        if zone:
            if direction == "LONG":
                in_zone = l <= zone["zone_high"] + atr * 0.3
            else:
                in_zone = h >= zone["zone_low"] - atr * 0.3
            if not in_zone:
                return result

        quality = 0.0

        # Rejection candle (pin bar)
        if total > 0:
            if direction == "LONG":
                lower_wick = min(o, c) - l
                if lower_wick > body * 1.5 and lower_wick > total * 0.5:
                    result["triggered"] = True
                    result["trigger_type"] = "rejection_candle"
                    quality += 0.5
            else:
                upper_wick = h - max(o, c)
                if upper_wick > body * 1.5 and upper_wick > total * 0.5:
                    result["triggered"] = True
                    result["trigger_type"] = "rejection_candle"
                    quality += 0.5

        # Engulfing candle
        if len(opens) >= 2:
            prev_body = abs(closes[-2] - opens[-2])
            if body > prev_body * 1.3:
                if direction == "LONG" and c > o and closes[-2] < opens[-2]:
                    result["triggered"] = True
                    result["trigger_type"] = "bullish_engulfing"
                    quality += 0.6
                elif direction == "SHORT" and c < o and closes[-2] > opens[-2]:
                    result["triggered"] = True
                    result["trigger_type"] = "bearish_engulfing"
                    quality += 0.6

        # Volume confirmation
        avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1.0
        if avg_vol > 0 and volumes[-1] > avg_vol * 1.5:
            quality += 0.2

        result["quality"] = min(1.0, round(quality, 3))
        return result

    # ── Indicator helpers ──────────────────────────────────────────────
    @classmethod
    def calculate_rsi(cls, closes: np.ndarray, period: int = 14) -> float:
        """Wilder RSI. Returns the most recent RSI value (0-100)."""
        closes = np.asarray(closes, dtype=float)
        if len(closes) < period + 1:
            return 50.0
        cache_key = ("rsi", period, cls._array_fingerprint(closes))
        cached = cls._class_indicator_cache.get(cache_key)
        if cached is not None:
            return cached
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
        result = float(100.0 - (100.0 / (1.0 + rs)))
        cls._class_indicator_cache[cache_key] = result
        return result

    @classmethod
    def calculate_atr(cls, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                      period: int = 14) -> float:
        """Average True Range using Wilder's smoothing."""
        highs  = np.asarray(highs,  dtype=float)
        lows   = np.asarray(lows,   dtype=float)
        closes = np.asarray(closes, dtype=float)
        if len(highs) < period + 1:
            return 0.0
        cache_key = (
            "atr",
            period,
            cls._array_fingerprint(highs),
            cls._array_fingerprint(lows),
            cls._array_fingerprint(closes),
        )
        cached = cls._class_indicator_cache.get(cache_key)
        if cached is not None:
            return cached
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
        result = float(atr)
        cls._class_indicator_cache[cache_key] = result
        return result

    @classmethod
    def calculate_adx(cls, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                      period: int = 14) -> float:
        """Wilder ADX. Returns the most recent ADX value (0-100)."""
        highs  = np.asarray(highs,  dtype=float)
        lows   = np.asarray(lows,   dtype=float)
        closes = np.asarray(closes, dtype=float)
        if len(highs) < period * 2 + 1:
            return 0.0
        cache_key = (
            "adx",
            period,
            cls._array_fingerprint(highs),
            cls._array_fingerprint(lows),
            cls._array_fingerprint(closes),
        )
        cached = cls._class_indicator_cache.get(cache_key)
        if cached is not None:
            return cached
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
        result = float(adx_arr[-1])
        cls._class_indicator_cache[cache_key] = result
        return result

    @classmethod
    def calculate_bollinger(cls, closes: np.ndarray, period: int = 20,
                            std_mult: float = 2.0) -> Tuple[float, float, float]:
        """
        Bollinger Bands.
        Returns (mid, upper, lower).
        """
        closes = np.asarray(closes, dtype=float)
        if len(closes) < period:
            return float("nan"), float("nan"), float("nan")
        cache_key = ("bb", period, std_mult, cls._array_fingerprint(closes))
        cached = cls._class_indicator_cache.get(cache_key)
        if cached is not None:
            return cached
        window = closes[-period:]
        mid    = float(np.mean(window))
        std    = float(np.std(window, ddof=1))
        result = (mid, mid + std_mult * std, mid - std_mult * std)
        cls._class_indicator_cache[cache_key] = result
        return result

    @staticmethod
    def calculate_effective_rr(
        direction: SignalDirection | str,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        tp2: float,
    ) -> float:
        """
        Calculate R:R using worst-case fill — the deeper end of the entry zone.

        LONG fills at entry_high (top of zone = worst fill for buyer).
        SHORT fills at entry_low (bottom of zone = worst fill for seller).

        This is ~5-10% more conservative than the midpoint formula.
        The aggregator's own geometry check still uses entry_mid for enforcement,
        so signals with tight zones will have their rr_ratio overwritten downstream.
        """
        direction_str = direction.value if hasattr(direction, "value") else str(direction)

        if direction_str == SignalDirection.LONG.value:
            fill = entry_high       # worst-case fill for a limit LONG = top of zone
            risk = fill - stop_loss
            reward = tp2 - fill
        else:
            fill = entry_low        # worst-case fill for a limit SHORT = bottom of zone
            risk = stop_loss - fill
            reward = fill - tp2

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
    def detect_parabolic(
        closes: np.ndarray,
        period: int = 10,
        *,
        ohlcv: Optional[list] = None,
        direction: str = "LONG",
    ) -> Dict[str, Any]:
        """
        Detect parabolic price acceleration using ROC-of-ROC (second derivative).

        Returns dict with:
          - is_parabolic: bool — acceleration exceeds threshold
          - acceleration: float — rate of change of ROC (positive = accelerating up)
          - roc: float — current rate of change
          - direction: str — "UP", "DOWN", or "FLAT"
          - is_exhausted: bool — exhaustion pattern present (only when ohlcv passed)
          - exhaustion_signals: list[str] — names of exhaustion patterns
          - confidence_penalty: int — suggested penalty (negative) or bonus (positive)
            for the proposed `direction` (only when ohlcv passed)

        AUDIT FIX (parabolic_detector wiring): when caller provides the full
        OHLCV array, delegate to analyzers.parabolic_detector for the richer
        exhaustion + confidence_penalty output.  Previously parabolic_detector
        was a standalone module with no consumers despite providing strictly
        more information than this helper's closes-only analysis.
        """
        closes = np.asarray(closes, dtype=float)
        result = {
            "is_parabolic": False,
            "acceleration": 0.0,
            "roc": 0.0,
            "direction": "FLAT",
            # AUDIT FIX: always include the enrichment keys so early-return
            # paths (insufficient data, empty ROC series) don't surprise
            # callers that look for these fields.
            "is_exhausted": False,
            "exhaustion_signals": [],
            "confidence_penalty": 0,
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

        # Determine direction and parabolic threshold.
        # B-Q3: scale the threshold from the asset's own ROC-of-ROC distribution
        # rather than using a hardcoded 0.5% per bar.  A fixed 0.5% is extreme
        # for BTC 1h but trivial for a meme-coin on 15m.  Using 1.5× the
        # standard deviation of the full roc_of_roc series as the threshold
        # means we require acceleration that is "unusually large" relative to
        # the instrument's recent volatility.  The clamp [0.001, 0.05] prevents
        # the threshold from collapsing on perfectly smooth moves or exploding
        # on black-swan data.
        _recent_accel = roc_of_roc[-3:] if len(roc_of_roc) >= 3 else roc_of_roc
        _avg_accel = float(np.mean(_recent_accel))
        _all_same_sign = all(a > 0 for a in _recent_accel) or all(a < 0 for a in _recent_accel)

        if len(roc_of_roc) >= 5:
            _rr_std = float(np.std(roc_of_roc))
            # max(0.001, ...) handles the _rr_std==0 edge-case; the explicit guard
            # is not needed because 1.5*0 = 0 and max(0.001, 0) = 0.001.
            _dynamic_threshold = max(0.001, min(0.05, 1.5 * _rr_std))
        else:
            _dynamic_threshold = 0.005  # fallback for very short series

        is_parabolic = _all_same_sign and abs(_avg_accel) > _dynamic_threshold

        if current_roc > 0.01:
            price_direction = "UP"
        elif current_roc < -0.01:
            price_direction = "DOWN"
        else:
            price_direction = "FLAT"

        result["is_parabolic"] = is_parabolic
        result["acceleration"] = round(acceleration, 6)
        result["roc"] = round(current_roc, 6)
        result["direction"] = price_direction

        # ── Optional enrichment via analyzers.parabolic_detector ─────────
        # When caller supplies the full OHLCV array, run the richer analyzer
        # so callers can surface exhaustion patterns and the direction-aware
        # confidence penalty alongside the base parabolic fields.
        if ohlcv is not None:
            try:
                from analyzers.parabolic_detector import parabolic_detector as _pd
                # `direction` here is the trade direction (LONG/SHORT) used by
                # the analyzer to decide whether we're entering *into* or *against*
                # the parabolic move — not the price direction above.
                _full = _pd.analyze(ohlcv, direction=direction, roc_period=period)
                result["is_parabolic"] = bool(_full.is_parabolic)
                result["is_exhausted"] = bool(_full.is_exhausted)
                result["exhaustion_signals"] = list(_full.exhaustion_signals)
                result["confidence_penalty"] = int(_full.confidence_penalty)
                # Prefer analyzer's numbers if they're populated (keeps parity)
                if _full.roc or _full.acceleration:
                    result["roc"] = round(float(_full.roc), 6)
                    result["acceleration"] = round(float(_full.acceleration), 6)
                    result["direction"] = _full.direction
            except Exception:
                # Enrichment is best-effort — fall back to base result on any error.
                # B-7: track the error rate so persistent failures are visible.
                BaseStrategy._parabolic_error_count += 1
                _ec = BaseStrategy._parabolic_error_count
                if _ec % BaseStrategy._PARABOLIC_ERROR_WARN_EVERY == 0:
                    logger.warning(
                        "parabolic_detector enrichment has failed %d consecutive times — "
                        "check analyzers.parabolic_detector for regressions.",
                        _ec,
                        exc_info=True,
                    )
                else:
                    logger.debug("parabolic_detector enrichment skipped", exc_info=True)
            else:
                # Success path: reset error counter so we don't warn again until
                # the next run of consecutive failures.
                BaseStrategy._parabolic_error_count = 0

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
          - is_exhausted: bool — weighted score ≥ 0.50 (with histogram gate)
          - signals: list of detected exhaustion signals
          - score: float 0-1 — exhaustion intensity
        """
        # B-Q4 weight constants — named here for readability and ease of tuning.
        _W_HISTOGRAM_DECEL = 0.40   # strongest single signal; momentum slowing visibly
        _W_VOLUME_CLIMAX   = 0.35   # capitulation / blow-off top
        _W_RANGE_CONTRACT  = 0.20   # consecutive narrowing candles
        _W_VOLUME_SPIKE    = 0.15   # elevated but not extreme volume
        _W_WANING_MOMENTUM = 0.15   # small bodies with persistent range
        _W_DOJI_EXHAUSTION = 0.10   # candles well below recent average size

        opens   = np.asarray(opens,   dtype=float)
        highs   = np.asarray(highs,   dtype=float)
        lows    = np.asarray(lows,    dtype=float)
        closes  = np.asarray(closes,  dtype=float)
        volumes = np.asarray(volumes, dtype=float)

        signals = []
        score = 0.0

        if len(closes) < 10:
            return {"is_exhausted": False, "signals": [], "score": 0.0}

        # B-Q4: weighted voting rather than raw signal count.
        # Weights reflect the information content of each signal; a volume climax
        # at an extreme carries far more weight than simple range contraction.
        # Exhaustion is declared when the weighted score reaches 0.50 or higher.
        #
        # Gate rule: when a MACD histogram is supplied, its deceleration is treated
        # as a required gate — price-only signals alone are insufficient to confirm
        # momentum exhaustion if the histogram is available but NOT decelerating.

        # 1. Histogram deceleration (weight 0.40 — highest single-signal weight)
        histogram_provided = histogram is not None
        histogram_decelerating = False
        if histogram_provided and len(histogram) >= 4:
            h = np.asarray(histogram, dtype=float)
            if abs(h[-1]) < abs(h[-2]) < abs(h[-3]):
                signals.append("histogram_deceleration")
                score += _W_HISTOGRAM_DECEL
                histogram_decelerating = True

        # 2. Candle range contraction
        ranges = highs - lows
        if len(ranges) >= 5:
            if ranges[-1] < ranges[-2] < ranges[-3]:
                signals.append("range_contraction")
                score += _W_RANGE_CONTRACT
            # Tiny candles relative to recent average
            avg_range = float(np.mean(ranges[-20:]))
            if avg_range > 0 and ranges[-1] < avg_range * 0.4:
                signals.append("doji_exhaustion")
                score += _W_DOJI_EXHAUSTION

        # 3. Volume climax — spike at extreme
        if len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            if avg_vol > 0:
                vol_ratio = volumes[-1] / avg_vol
                if vol_ratio > 3.0:
                    signals.append("volume_climax")
                    score += _W_VOLUME_CLIMAX
                elif vol_ratio > 2.0:
                    signals.append("volume_spike")
                    score += _W_VOLUME_SPIKE

        # 4. Waning momentum: smaller bodies with bigger wicks
        if len(closes) >= 3:
            bodies = [abs(closes[-i] - opens[-i]) for i in range(1, 4)]
            total_ranges = [highs[-i] - lows[-i] for i in range(1, 4)]
            if (total_ranges[0] > 0 and total_ranges[1] > 0
                    and bodies[0] < bodies[1] * 0.6
                    and total_ranges[0] >= total_ranges[1] * 0.7):
                signals.append("waning_momentum")
                score += _W_WANING_MOMENTUM

        # Gate: when histogram data is provided but NOT decelerating, that is a
        # counter-signal to exhaustion — a still-expanding histogram means price
        # momentum is intact regardless of what structure signals say.  When no
        # histogram is passed (caller doesn't have MACD data), we cannot use it
        # to veto, so the gate passes by default.
        histogram_gate_ok = (not histogram_provided) or histogram_decelerating

        is_exhausted = score >= 0.50 and histogram_gate_ok

        return {
            "is_exhausted": is_exhausted,
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
    #
    # B-8: caps are set to each strategy's realistic output maximum based on
    # how many independent conditions must align.  High-conviction, multi-leg
    # strategies (SMC, Wyckoff, ExtremeReversal) can legitimately reach 95.
    # Single-dimension strategies (MeanReversion, FundingRateArb, RangeScalper)
    # rarely exceed 88-90 even in ideal setups.  Differentiating these caps
    # ensures the linear rescale below produces meaningfully different
    # normalized values rather than identity-mapping the majority of strategies.
    "SmartMoneyConcepts":    (40, 95),
    "InstitutionalBreakout": (40, 95),
    "ExtremeReversal":       (40, 95),
    "MeanReversion":         (40, 88),
    "PriceAction":           (40, 92),
    "Momentum":              (40, 90),
    "MomentumContinuation":  (40, 90),
    "Ichimoku":              (40, 91),
    "IchimokuCloud":         (40, 91),
    "ElliottWave":           (40, 91),
    "FundingRateArb":        (40, 88),
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

    B-8: strategies with (src_floor, src_cap) == (NORM_FLOOR, NORM_CAP) would
    receive identity mapping — no rescaling.  To ensure every registered
    strategy benefits from normalization, their caps are set to realistic
    output maxima (see _STRATEGY_CONFIDENCE_RANGE).  Strategies with cap < 95
    will have their scores scaled upward at the top end, making the full
    [40, 95] normalized range accessible even for strategies that inherently
    produce narrower raw ranges.

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

    # Short-circuit for the identity case (no work needed).
    if src_floor == NORM_FLOOR and src_cap == NORM_CAP:
        return round(max(NORM_FLOOR, min(NORM_CAP, raw_confidence)), 2)

    # Linear rescale: [src_floor, src_cap] → [NORM_FLOOR, NORM_CAP]
    clamped = max(src_floor, min(src_cap, raw_confidence))
    ratio = (clamped - src_floor) / (src_cap - src_floor)
    return round(NORM_FLOOR + ratio * (NORM_CAP - NORM_FLOOR), 2)
