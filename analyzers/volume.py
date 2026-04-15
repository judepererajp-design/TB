"""
TitanBot Pro — Volume Analyzer
================================
Analyzes volume structure to confirm or reject signals.

Modules:
  - VWAP with standard deviation bands
  - Volume Profile (POC, Value Area High/Low)
  - On-Balance Volume (OBV) divergence detection
  - Volume-weighted price levels
  - Accumulation/Distribution detection
  - Session volume comparison
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config.loader import cfg
from utils.formatting import fmt_price

logger = logging.getLogger(__name__)


@dataclass
class VolumeData:
    symbol: str
    vwap: float = 0.0
    vwap_upper_1: float = 0.0          # +1 sigma
    vwap_lower_1: float = 0.0          # -1 sigma
    vwap_upper_2: float = 0.0          # +2 sigma
    vwap_lower_2: float = 0.0          # -2 sigma
    price_vs_vwap: str = "AT"          # ABOVE | BELOW | AT
    poc: float = 0.0                   # Point of Control (highest volume price)
    vah: float = 0.0                   # Value Area High
    val: float = 0.0                   # Value Area Low
    price_in_value_area: bool = True
    volume_trend: str = "NEUTRAL"      # INCREASING | DECREASING | NEUTRAL
    volume_spike: bool = False
    volume_spike_mult: float = 1.0     # Current vol / average vol
    obv_divergence: str = "NONE"       # BULLISH | BEARISH | NONE
    accumulation: str = "NEUTRAL"      # ACCUMULATION | DISTRIBUTION | NEUTRAL
    score: float = 50.0
    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


@dataclass
class VolumeQualityResult:
    """Comprehensive volume quality assessment result."""
    quality_score: float = 50.0          # 0-100 composite quality
    quality_label: str = "MEDIUM"        # HIGH | MEDIUM | LOW
    magnitude_score: float = 50.0        # Volume size relative to average
    trend_score: float = 50.0            # Volume trend (increasing/decreasing)
    context_score: float = 50.0          # How well volume fits the trade type
    context_label: str = "NORMAL"        # EXHAUSTION | BREAKOUT | HEALTHY_PULLBACK etc.
    breadth_score: float = 50.0          # Participation breadth
    spread_score: float = 50.0           # Spread-adjusted quality
    spread_bps: float = 0.0              # Estimated spread proxy in basis points
    volume_trend: str = "STABLE"         # INCREASING | DECREASING | STABLE
    dry_volume: bool = False             # Price moved on no volume
    confidence_delta: int = 0            # Recommended confidence adjustment
    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


class VolumeAnalyzer:
    """
    Comprehensive volume analysis for signal confirmation.
    Volume is the most important confirmation tool — price moves
    without volume are suspect. Volume moves before price.
    """

    def __init__(self):
        self._vol_cfg = cfg.analyzers.volume
        self._vwap_window = self._vol_cfg.get('vwap_window', 24)
        self._profile_bins = self._vol_cfg.get('profile_bins', 100)
        self._value_area_pct = self._vol_cfg.get('value_area_pct', 0.70)

    def analyze(self, ohlcv: List, current_price: float) -> VolumeData:
        """
        Full volume analysis from OHLCV data.
        ohlcv: List of [timestamp, open, high, low, close, volume]
        """
        data = VolumeData(symbol="")

        if not ohlcv or len(ohlcv) < 20:
            return data

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # VWAP analysis
        self._calculate_vwap(df, data, current_price)

        # Volume Profile
        self._calculate_volume_profile(df, data, current_price)

        # Volume trend and spikes
        self._analyze_volume_trend(df, data)

        # OBV Divergence
        self._analyze_obv_divergence(df, data, current_price)

        # Accumulation/Distribution
        self._analyze_accumulation(df, data)

        # Final score
        data.score = self._calculate_score(data, current_price)

        return data

    def _calculate_vwap(self, df: pd.DataFrame, data: VolumeData, current_price: float):
        """
        Rolling VWAP with standard deviation bands.
        VWAP = sum(typical_price * volume) / sum(volume)
        """
        window = min(self._vwap_window, len(df))
        df_w = df.tail(window).copy()

        typical_price = (df_w['high'] + df_w['low'] + df_w['close']) / 3
        vol = df_w['volume']

        cumulative_vol = vol.cumsum()
        cumulative_tpvol = (typical_price * vol).cumsum()

        # Guard: if all volume is zero (e.g. synthetic/test data), skip VWAP
        last_cum_vol = float(cumulative_vol.iloc[-1])
        if last_cum_vol <= 0:
            return

        # Replace zero entries with NaN once so VWAP and std bands share the same
        # safe denominator without computing replace() twice.
        safe_cum_vol = cumulative_vol.replace(0, float('nan'))
        vwap_series = cumulative_tpvol / safe_cum_vol
        data.vwap = float(vwap_series.iloc[-1]) if np.isfinite(vwap_series.iloc[-1]) else 0.0

        # Standard deviation bands — guard against NaN from zero-volume bars
        variance = ((typical_price - vwap_series) ** 2 * vol).cumsum() / safe_cum_vol
        var_val = float(variance.iloc[-1])
        std = float(np.sqrt(var_val)) if np.isfinite(var_val) else 0.0

        data.vwap_upper_1 = data.vwap + std
        data.vwap_lower_1 = data.vwap - std
        data.vwap_upper_2 = data.vwap + 2 * std
        data.vwap_lower_2 = data.vwap - 2 * std

        # Price position relative to VWAP
        if current_price > data.vwap_upper_1:
            data.price_vs_vwap = "ABOVE"
            data.notes.append(f"📈 Price above VWAP ({fmt_price(data.vwap)})")
        elif current_price < data.vwap_lower_1:
            data.price_vs_vwap = "BELOW"
            data.notes.append(f"📉 Price below VWAP ({fmt_price(data.vwap)})")
        else:
            data.price_vs_vwap = "AT"

    def _calculate_volume_profile(
        self, df: pd.DataFrame, data: VolumeData, current_price: float
    ):
        """
        Volume Profile — builds a histogram of volume at each price level.
        POC (Point of Control) = price with highest volume = strongest magnet.
        Value Area = 70% of total volume = fair value zone.
        """
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        price_min = lows.min()
        price_max = highs.max()

        if price_max == price_min:
            return

        bins = self._profile_bins
        price_bins = np.linspace(price_min, price_max, bins + 1)
        volume_at_price = np.zeros(bins)

        for i in range(len(df)):
            if volumes[i] <= 0:
                continue
            # Distribute bar's volume across its price range
            bar_low = lows[i]
            bar_high = highs[i]
            bar_vol = volumes[i]

            # Find which bins this bar covers
            low_bin = max(0, int(np.searchsorted(price_bins, bar_low)) - 1)
            high_bin = min(bins - 1, int(np.searchsorted(price_bins, bar_high)))

            if low_bin >= high_bin:
                volume_at_price[low_bin] += bar_vol
            else:
                # Distribute evenly across covered bins
                n_bins = high_bin - low_bin
                per_bin = bar_vol / n_bins
                volume_at_price[low_bin:high_bin] += per_bin

        # POC = bin with maximum volume
        poc_bin = np.argmax(volume_at_price)
        data.poc = float((price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2)

        # Value Area — 70% of total volume centered around POC
        total_volume = volume_at_price.sum()
        target_volume = total_volume * self._value_area_pct

        va_volume = volume_at_price[poc_bin]
        low_idx = poc_bin
        high_idx = poc_bin

        while va_volume < target_volume:
            add_low = volume_at_price[low_idx - 1] if low_idx > 0 else 0
            add_high = volume_at_price[high_idx + 1] if high_idx < bins - 1 else 0

            if add_low > add_high and low_idx > 0:
                low_idx -= 1
                va_volume += add_low
            elif high_idx < bins - 1:
                high_idx += 1
                va_volume += add_high
            else:
                break

        data.val = float(price_bins[low_idx])
        data.vah = float(price_bins[min(high_idx + 1, bins)])
        data.price_in_value_area = data.val <= current_price <= data.vah

        # Annotate
        poc_dist = (current_price - data.poc) / data.poc * 100
        if abs(poc_dist) < 0.5:
            data.notes.append(f"🎯 Price at POC ({fmt_price(data.poc)}) — high interest level")
        elif current_price > data.vah:
            data.notes.append(f"⬆️ Price above Value Area — breakout zone")
        elif current_price < data.val:
            data.notes.append(f"⬇️ Price below Value Area — potential bounce")

    def _analyze_volume_trend(self, df: pd.DataFrame, data: VolumeData):
        """Detect volume trend and spikes"""
        volumes = df['volume'].values

        if len(volumes) < 20:
            return

        recent = volumes[-5:]
        avg_20 = np.mean(volumes[-20:])
        avg_5 = np.mean(recent)
        current_vol = volumes[-1]

        # Volume trend
        if avg_5 > avg_20 * 1.3:
            data.volume_trend = "INCREASING"
        elif avg_5 < avg_20 * 0.7:
            data.volume_trend = "DECREASING"
        else:
            data.volume_trend = "NEUTRAL"

        # Current bar volume spike
        data.volume_spike_mult = current_vol / avg_20 if avg_20 > 0 else 1.0
        data.volume_spike = data.volume_spike_mult >= 2.0

        if data.volume_spike:
            data.notes.append(
                f"🔥 Volume spike: {data.volume_spike_mult:.1f}x average"
            )

    def _analyze_obv_divergence(
        self, df: pd.DataFrame, data: VolumeData, current_price: float
    ):
        """
        On-Balance Volume divergence.
        OBV rising while price flat/falling = bullish divergence (buying pressure)
        OBV falling while price flat/rising = bearish divergence (selling into strength)
        """
        closes = df['close'].values
        volumes = df['volume'].values

        if len(closes) < 20:
            return

        # Calculate OBV
        obv = np.zeros(len(closes))
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]

        # Compare last 10 bars OBV trend vs price trend
        lookback = 10
        obv_recent = obv[-lookback:]
        price_recent = closes[-lookback:]

        obv_slope = np.polyfit(range(lookback), obv_recent, 1)[0]
        price_slope = np.polyfit(range(lookback), price_recent, 1)[0]

        # Normalize slopes by their respective standard deviations.
        # Use max() guard instead of tiny epsilon: if std is near zero (flat OBV or price),
        # the slope is also near zero so the divergence condition won't fire regardless.
        # Using a relative floor (1% of mean absolute value) avoids extreme blow-up on
        # near-flat series while staying meaningful for all price scales.
        obv_std_floor   = max(abs(np.mean(obv_recent)) * 0.01, 1e-6)
        price_std_floor = max(abs(np.mean(price_recent)) * 0.01, 1e-6)
        obv_norm   = obv_slope   / max(np.std(obv_recent),   obv_std_floor)
        price_norm = price_slope / max(np.std(price_recent), price_std_floor)

        if obv_norm > 0.5 and price_norm < -0.2:
            data.obv_divergence = "BULLISH"
            data.notes.append("✅ Bullish OBV divergence — buying pressure building")
        elif obv_norm < -0.5 and price_norm > 0.2:
            data.obv_divergence = "BEARISH"
            data.notes.append("⚠️ Bearish OBV divergence — selling into strength")
        else:
            data.obv_divergence = "NONE"

    def _analyze_accumulation(self, df: pd.DataFrame, data: VolumeData):
        """
        Chaikin Money Flow / simple accumulation detection.
        Accumulation: price consolidates, volume builds, buyers absorbing sellers
        Distribution: price elevated, volume builds, sellers offloading to buyers
        """
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        if len(closes) < 20:
            return

        lookback = 20

        # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        mfm_list = []
        for i in range(-lookback, 0):
            hl = highs[i] - lows[i]
            if hl > 0:
                mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl
            else:
                mfm = 0
            mfm_list.append(mfm * volumes[i])

        cmf = sum(mfm_list) / (sum(volumes[-lookback:]) + 1e-10)

        if cmf > 0.15:
            data.accumulation = "ACCUMULATION"
            data.notes.append("💚 Accumulation detected — smart money buying")
        elif cmf < -0.15:
            data.accumulation = "DISTRIBUTION"
            data.notes.append("🔴 Distribution detected — selling pressure")
        else:
            data.accumulation = "NEUTRAL"

    def _calculate_score(self, data: VolumeData, current_price: float) -> float:
        """Convert volume analysis to 0-100 score"""
        score = 50.0

        # VWAP position
        if data.price_vs_vwap == "ABOVE":
            score += 10
        elif data.price_vs_vwap == "BELOW":
            score -= 10

        # POC proximity (price at POC = high-interest = good entry zone)
        if data.poc > 0:
            poc_dist = abs(current_price - data.poc) / data.poc
            if poc_dist < 0.005:  # Within 0.5% of POC
                score += 8

        # Value area
        if data.price_in_value_area:
            score += 5  # In value area = fair value, good for reversals

        # Volume trend
        if data.volume_trend == "INCREASING":
            score += 8
        elif data.volume_trend == "DECREASING":
            score -= 5

        # Volume spike (confirmation)
        if data.volume_spike:
            score += 10 if data.volume_spike_mult >= 3 else 6

        # OBV divergence
        if data.obv_divergence == "BULLISH":
            score += 12
        elif data.obv_divergence == "BEARISH":
            score -= 12

        # Accumulation/Distribution
        if data.accumulation == "ACCUMULATION":
            score += 10
        elif data.accumulation == "DISTRIBUTION":
            score -= 10

        return max(0.0, min(100.0, score))

    def assess_volume_quality(
        self,
        ohlcv: List,
        current_price: float,
        direction: str = "",
        trade_type: str = "",
    ) -> 'VolumeQualityResult':
        """
        Comprehensive volume quality assessment.
        High volume ≠ always good. Context determines meaning.

        Parameters
        ----------
        ohlcv : list
            OHLCV data.
        current_price : float
        direction : str
            "LONG" or "SHORT" — for context-dependent evaluation.
        trade_type : str
            "breakout", "pullback", "range", "reversal" — affects how volume is interpreted.
        """
        from config.constants import VolumeQuality as VQ

        result = VolumeQualityResult()

        if not ohlcv or len(ohlcv) < 20:
            return result

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        if len(volumes) < 20:
            return result

        avg_vol_20 = float(np.mean(volumes[-20:]))
        current_vol = float(volumes[-1])

        if avg_vol_20 <= 0:
            return result

        vol_ratio = current_vol / avg_vol_20

        # ── 1. Magnitude score ─────────────────────────────────
        # Normalized volume relative to recent average
        magnitude = min(1.0, vol_ratio / 3.0)  # 3× average = perfect score
        result.magnitude_score = magnitude * 100

        # ── 2. Volume trend score ──────────────────────────────
        recent_5 = float(np.mean(volumes[-5:]))
        trend_ratio = recent_5 / avg_vol_20 if avg_vol_20 > 0 else 1.0
        if trend_ratio > 1.3:
            result.trend_score = 80.0
            result.volume_trend = "INCREASING"
        elif trend_ratio < 0.7:
            result.trend_score = 30.0
            result.volume_trend = "DECREASING"
        else:
            result.trend_score = 55.0
            result.volume_trend = "STABLE"

        # ── 3. Context score — how does volume fit the trade type? ──
        context_score = 50.0
        if vol_ratio >= VQ.CLIMACTIC_MULT:
            # Climactic volume: check if it's exhaustion or breakout
            bar_direction = closes[-1] - closes[-2] if len(closes) >= 2 else 0
            recent_trend = closes[-1] - closes[-5] if len(closes) >= 5 else 0

            is_reversal_bar = (
                (recent_trend > 0 and bar_direction < 0) or
                (recent_trend < 0 and bar_direction > 0)
            )

            if is_reversal_bar:
                result.context_label = VQ.CLIMACTIC_WITH_REVERSAL
                context_score = 25.0  # Exhaustion = bad for continuation
                result.notes.append(
                    f"⚠️ Climactic volume ({vol_ratio:.1f}×) + reversal bar = "
                    f"exhaustion signal — high volume is NOT bullish here"
                )
            else:
                result.context_label = VQ.CLIMACTIC_WITH_TREND
                context_score = 85.0  # Breakout = great
                result.notes.append(
                    f"🚀 Climactic volume ({vol_ratio:.1f}×) + trend bar = "
                    f"genuine breakout confirmation"
                )
        elif trade_type == "breakout" and vol_ratio < VQ.BREAKOUT_VOL_MIN_MULT:
            context_score = 30.0
            result.context_label = "WEAK_BREAKOUT"
            result.notes.append(
                f"⚠️ Breakout with only {vol_ratio:.1f}× volume — "
                f"need {VQ.BREAKOUT_VOL_MIN_MULT}× for confirmation"
            )
        elif trade_type == "pullback":
            # Pullback: DECREASING volume is healthy
            if result.volume_trend == "DECREASING":
                context_score = 75.0
                result.context_label = "HEALTHY_PULLBACK"
                result.notes.append(
                    "✅ Decreasing volume on pullback — healthy consolidation"
                )
            elif vol_ratio > VQ.RANGE_VOL_HIGH_WARNING:
                context_score = 35.0
                result.context_label = "HEAVY_PULLBACK"
                result.notes.append(
                    f"⚠️ High volume ({vol_ratio:.1f}×) on pullback — "
                    f"possible trend change, not just pullback"
                )
            else:
                context_score = 55.0
                result.context_label = "NORMAL"
        else:
            result.context_label = "NORMAL"
            context_score = 50.0

        result.context_score = context_score

        # ── 4. Breadth score ───────────────────────────────────
        # Approximate participation breadth using volume variance
        # High variance = whale-driven; low variance = broad participation
        if len(volumes) >= 10:
            vol_std = float(np.std(volumes[-10:]))
            vol_mean = float(np.mean(volumes[-10:]))
            cv = vol_std / max(vol_mean, 1e-10)  # Coefficient of variation
            if cv > 1.0:
                result.breadth_score = 30.0  # Very uneven = whale-driven
                result.notes.append("📊 Uneven volume distribution — whale-driven")
            elif cv < 0.3:
                result.breadth_score = 80.0  # Very even = broad participation
                result.notes.append("📊 Broad volume participation — healthy market")
            else:
                result.breadth_score = 55.0
        else:
            result.breadth_score = 50.0

        # ── 5. Spread-adjusted score (if data available) ──────
        # Use high-low range as proxy for spread
        if len(highs) >= 1 and len(lows) >= 1:
            typical_spread_bps = (
                float(np.mean(highs[-5:] - lows[-5:])) /
                max(current_price, 1e-10)
            ) * 10000  # Convert to bps
            result.spread_bps = typical_spread_bps
            if typical_spread_bps > VQ.SPREAD_WIDE_THRESHOLD_BPS:
                result.spread_score = 35.0
                result.notes.append(
                    f"⚠️ Wide spread ({typical_spread_bps:.0f} bps) — "
                    f"volume less reliable in illiquid conditions"
                )
            else:
                result.spread_score = 70.0
        else:
            result.spread_score = 50.0

        # ── 6. Dry volume detection ───────────────────────────
        if len(closes) >= 2:
            price_move_pct = abs(closes[-1] - closes[-2]) / max(closes[-2], 1e-10)
            atr_approx = float(np.mean(highs[-20:] - lows[-20:])) if len(highs) >= 20 else 0
            if atr_approx > 0:
                price_move_atr = abs(closes[-1] - closes[-2]) / atr_approx
                if (price_move_atr > VQ.DRY_VOL_PRICE_MOVE_ATR and
                        vol_ratio < VQ.DRY_VOL_THRESHOLD_MULT):
                    result.dry_volume = True
                    result.confidence_delta += VQ.DRY_VOL_PENALTY
                    result.notes.append(
                        f"⚠️ Dry volume move: price moved {price_move_atr:.2f} ATR "
                        f"on only {vol_ratio:.1f}× average volume — "
                        f"suspect sustainability"
                    )

        # ── 7. Intrabar delta estimation ──────────────────────
        # Approximate buy vs sell using close position within bar range
        # (close - low) / (high - low) = buy fraction (standard tick-rule proxy)
        delta_ratio = 0.5  # default neutral
        if len(highs) >= 1 and len(lows) >= 1 and len(closes) >= 1:
            hl_range = float(highs[-1] - lows[-1])
            if hl_range > 0:
                delta_ratio = float((closes[-1] - lows[-1]) / hl_range)
                if direction == "LONG" and delta_ratio >= VQ.DELTA_STRONG_THRESHOLD:
                    result.confidence_delta += VQ.DELTA_CONF_BONUS
                    result.notes.append(
                        f"📊 Strong buy delta ({delta_ratio:.0%}) — buyers in control"
                    )
                elif direction == "SHORT" and delta_ratio <= VQ.DELTA_WEAK_THRESHOLD:
                    result.confidence_delta += VQ.DELTA_CONF_BONUS
                    result.notes.append(
                        f"📊 Strong sell delta ({1 - delta_ratio:.0%}) — sellers in control"
                    )
                elif direction == "LONG" and delta_ratio <= VQ.DELTA_WEAK_THRESHOLD:
                    result.confidence_delta += VQ.DELTA_CONF_PENALTY
                    result.notes.append(
                        f"⚠️ Opposing delta ({delta_ratio:.0%} buy) — sellers dominating"
                    )
                elif direction == "SHORT" and delta_ratio >= VQ.DELTA_STRONG_THRESHOLD:
                    result.confidence_delta += VQ.DELTA_CONF_PENALTY
                    result.notes.append(
                        f"⚠️ Opposing delta ({delta_ratio:.0%} buy) — buyers dominating"
                    )

        # ── 8. Composite quality score ────────────────────────
        # Delta alignment score (0-100): how well delta supports the trade direction
        delta_alignment = 50.0  # neutral
        if direction == "LONG":
            delta_alignment = delta_ratio * 100.0
        elif direction == "SHORT":
            delta_alignment = (1.0 - delta_ratio) * 100.0

        quality = (
            result.magnitude_score * VQ.QUALITY_WEIGHT_MAGNITUDE * 0.92 +
            result.trend_score * VQ.QUALITY_WEIGHT_TREND * 0.92 +
            result.context_score * VQ.QUALITY_WEIGHT_CONTEXT * 0.92 +
            result.breadth_score * VQ.QUALITY_WEIGHT_BREADTH * 0.92 +
            result.spread_score * VQ.QUALITY_WEIGHT_SPREAD * 0.92 +
            delta_alignment * 0.08
        )
        result.quality_score = max(0.0, min(100.0, quality))

        # ── 9. Classify ──────────────────────────────────────
        if result.quality_score >= VQ.QUALITY_SCORE_HIGH:
            result.quality_label = "HIGH"
        elif result.quality_score >= VQ.QUALITY_SCORE_LOW:
            result.quality_label = "MEDIUM"
        else:
            result.quality_label = "LOW"
            result.confidence_delta -= 3  # Low quality = caution

        result.notes.append(
            f"Volume quality: {result.quality_label} ({result.quality_score:.0f}/100) — "
            f"mag={result.magnitude_score:.0f} trend={result.trend_score:.0f} "
            f"ctx={result.context_score:.0f} breadth={result.breadth_score:.0f}"
        )

        return result

    def confirm_breakout(self, data: VolumeData, direction: str) -> Tuple[bool, str]:
        """
        Confirm a breakout has genuine volume support.
        Returns (confirmed, reason)
        """
        if direction == "LONG":
            if data.volume_spike and data.volume_spike_mult >= 1.8:
                if data.price_vs_vwap == "ABOVE":
                    return True, "Volume breakout confirmed above VWAP"
                return True, f"Volume spike {data.volume_spike_mult:.1f}x on breakout"
            elif data.obv_divergence == "BULLISH":
                return True, "OBV divergence supports long breakout"
            else:
                return False, "Insufficient volume for long breakout"

        elif direction == "SHORT":
            if data.volume_spike and data.volume_spike_mult >= 1.8:
                if data.price_vs_vwap == "BELOW":
                    return True, "Volume breakdown confirmed below VWAP"
                return True, f"Volume spike {data.volume_spike_mult:.1f}x on breakdown"
            elif data.obv_divergence == "BEARISH":
                return True, "OBV divergence supports short breakdown"
            else:
                return False, "Insufficient volume for short breakdown"

        return False, "Unknown direction"


# ── Singleton ──────────────────────────────────────────────
volume_analyzer = VolumeAnalyzer()
