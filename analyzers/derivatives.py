"""
TitanBot Pro — Derivatives Analyzer
=====================================
Critical for perpetual futures trading.
Analyzes funding rates, open interest, and long/short ratios
to determine whether the crowd is positioned correctly —
and fade them when they're extreme.

Key insight: In perps, extreme funding = crowded positioning =
upcoming squeeze or liquidation cascade. This is the MOST important
edge in perpetual futures trading.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple
from dataclasses import dataclass

from config.loader import cfg
from data.api_client import api

logger = logging.getLogger(__name__)


@dataclass
class DerivativesData:
    symbol: str
    funding_rate: float = 0.0          # Current funding rate (%)
    funding_trend: str = "NEUTRAL"     # Level classifier: EXTREMELY_HIGH | HIGH | NEGATIVE | NEUTRAL
    # AUDIT FIX (funding delta): separate rate-of-change classifier.  Squeezes
    # come from acceleration, not absolute level — a flat-but-high funding is
    # already priced in, whereas a rapidly RISING funding signals crowd entry.
    funding_delta_trend: str = "FLAT"  # RISING | FALLING | FLAT
    open_interest: float = 0.0         # Current OI in USD
    oi_change_24h: float = 0.0         # OI change % in 24h
    oi_trend: str = "NEUTRAL"          # INCREASING | DECREASING | NEUTRAL
    long_short_ratio: float = 1.0      # L/S ratio (>1 = more longs)
    lsr_trend: str = "NEUTRAL"
    score: float = 50.0                # 0-100 score for signal integration
    signal_bias: str = "NEUTRAL"       # BULLISH | BEARISH | NEUTRAL | EXTREME_LONG | EXTREME_SHORT
    squeeze_risk: str = "LOW"          # LOW | MEDIUM | HIGH — short squeeze probability
    liquidation_risk: str = "LOW"      # Long liquidation risk
    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


class DerivativesAnalyzer:
    """
    Analyzes perpetual futures derivatives data for a given symbol.
    Used by the signal aggregator to validate/invalidate entries.
    """

    def __init__(self):
        self._der_cfg = cfg.analyzers.derivatives
        self._funding_threshold = self._der_cfg.get('funding_threshold', 0.01)
        self._oi_threshold = self._der_cfg.get('oi_change_threshold', 10)
        self._ls_extreme_high = self._der_cfg.get('ls_extreme_high', 2.0)
        self._ls_extreme_low = self._der_cfg.get('ls_extreme_low', 0.7)
        # AUDIT FIX: maintain per-symbol OI history to compute a real
        # 24h % change.  Previously `oi_change_24h` was populated from
        # ccxt's `openInterestAmount` (base-unit OI level), which is NOT
        # a 24h change and permanently left `oi_trend=NEUTRAL`, dropping
        # the OI contribution to score/validity.
        self._oi_history: Dict[str, Deque[Tuple[float, float]]] = {}
        # Retain ~26h of samples to survive a stale fetch or two.
        self._oi_history_max = 60
        # AUDIT FIX (funding delta): per-symbol short history of recent funding
        # rates used to classify rate-of-change (RISING / FALLING / FLAT).  Five
        # samples is enough to smooth a single outlier while still reacting
        # within ~an hour on a venue that polls at 8-minute intervals.
        self._funding_history: Dict[str, Deque[float]] = {}
        self._funding_history_max = 5
        # Delta threshold (in percentage-points, same scale as funding_rate).
        # 0.005 pp ≈ a clear regime shift on most perps whose typical funding
        # sits in ±0.01 %.  Keeps FLAT noisy in calm markets.
        self._funding_delta_threshold = 0.005

    async def analyze(self, symbol: str) -> DerivativesData:
        """
        Full derivatives analysis for a symbol.
        Returns a scored DerivativesData object.
        """
        data = DerivativesData(symbol=symbol)

        if not self._der_cfg.get('enabled', True):
            data.score = 50.0
            return data

        # Fetch all data concurrently
        funding, oi, lsr = await asyncio.gather(
            api.fetch_funding_rate(symbol),
            api.fetch_open_interest(symbol),
            api.fetch_long_short_ratio(symbol, period="1h"),
            return_exceptions=True
        )

        # Parse funding rate
        if isinstance(funding, dict) and funding:
            data.funding_rate = float(funding.get('fundingRate') or 0) * 100  # Convert to %
            data.funding_trend = self._classify_funding_trend(data.funding_rate)
            data.funding_delta_trend = self._classify_funding_delta(symbol, data.funding_rate)

        # Parse open interest
        # AUDIT FIX: compute a real 24h % change from in-memory OI history
        # instead of storing ccxt's `openInterestAmount` (base-unit OI level),
        # which made `oi_change_24h` static and disabled the OI trend logic.
        if isinstance(oi, dict) and oi:
            _raw_value = float(oi.get('openInterestValue') or 0)
            _raw_amount = float(oi.get('openInterestAmount') or 0)
            # Prefer USD notional (openInterestValue).  Some venues expose only
            # `openInterestAmount` in base units; fall back to that so we still
            # have a series.  Units only need to be consistent within a symbol
            # since we compute a ratio.
            _oi_level = _raw_value if _raw_value > 0 else _raw_amount
            data.open_interest = _raw_value if _raw_value > 0 else _raw_amount
            if _oi_level > 0:
                data.oi_change_24h = self._update_oi_history(symbol, _oi_level)
            else:
                data.oi_change_24h = 0.0
            data.oi_trend = self._classify_oi_trend(data.oi_change_24h)

        # Parse long/short ratio
        if isinstance(lsr, list) and lsr:
            lsr_data = lsr[0] if lsr else {}
            data.long_short_ratio = float(lsr_data.get('longShortRatio') or 1.0)
            data.lsr_trend = self._classify_lsr(data.long_short_ratio)

        # Classify overall signal bias
        data.signal_bias = self._classify_signal_bias(data)

        # Calculate squeeze risks
        data.squeeze_risk = self._assess_squeeze_risk(data)
        data.liquidation_risk = self._assess_liquidation_risk(data)

        # Calculate final score (0-100)
        data.score = self._calculate_score(data)

        return data

    def _update_oi_history(self, symbol: str, oi_level: float) -> float:
        """
        Append a new OI sample and return the % change vs the sample closest
        to 24 h ago.  Returns 0.0 until enough history has accumulated.

        Values are kept in whatever unit the exchange supplies; because we
        compute a ratio the unit cancels as long as it's consistent per symbol.
        """
        now = time.time()
        history = self._oi_history.setdefault(symbol, deque(maxlen=self._oi_history_max))
        history.append((now, oi_level))
        if len(history) < 2:
            return 0.0

        target_age = 24 * 3600
        cutoff = now - target_age
        # Find the sample with timestamp closest to (now - 24h), preferring
        # ones from BEFORE the cutoff so we're measuring an actual 24h window
        # rather than a shorter one.
        older = [s for s in history if s[0] <= cutoff]
        if older:
            baseline_ts, baseline_oi = older[-1]
        else:
            # Not yet 24h of history — use the oldest available sample, but
            # only report a change if the window is at least ~4h long to
            # avoid reacting to intraday noise as if it were 24h.
            baseline_ts, baseline_oi = history[0]
            if now - baseline_ts < 4 * 3600:
                return 0.0

        if baseline_oi <= 0:
            return 0.0
        return (oi_level - baseline_oi) / baseline_oi * 100.0

    def _classify_funding_trend(self, rate: float) -> str:
        """Classify funding rate"""
        threshold = self._funding_threshold
        if rate > threshold * 2:
            return "EXTREMELY_HIGH"  # Extreme longs paying → short fuel
        elif rate > threshold:
            return "HIGH"
        elif rate < -threshold:
            return "NEGATIVE"        # Shorts paying → long fuel
        else:
            return "NEUTRAL"

    def _classify_funding_delta(self, symbol: str, rate: float) -> str:
        """
        Classify funding rate-of-change across the last few polls.

        AUDIT FIX: the level classifier above tells us where funding IS.  Squeezes
        come from acceleration — a rapidly RISING funding rate means the crowd is
        piling in (squeeze fuel building), even if the absolute level still looks
        moderate.  Conversely, a rapidly FALLING funding rate means positioning
        is being unwound and the squeeze pressure is releasing.

        Uses a short per-symbol history (deque) and compares newest vs oldest
        sample.  Returns "FLAT" until we have at least 2 samples.
        """
        history = self._funding_history.setdefault(
            symbol, deque(maxlen=self._funding_history_max)
        )
        history.append(rate)
        if len(history) < 2:
            return "FLAT"
        delta = history[-1] - history[0]
        if delta > self._funding_delta_threshold:
            return "RISING"
        elif delta < -self._funding_delta_threshold:
            return "FALLING"
        return "FLAT"

    def _classify_oi_trend(self, oi_change: float) -> str:
        """Classify OI change"""
        threshold = self._oi_threshold
        if oi_change > threshold:
            return "STRONG_INCREASE"
        elif oi_change > threshold * 0.5:
            return "INCREASE"
        elif oi_change < -threshold:
            return "STRONG_DECREASE"
        elif oi_change < -threshold * 0.5:
            return "DECREASE"
        return "NEUTRAL"

    def _classify_lsr(self, ratio: float) -> str:
        """Classify long/short ratio"""
        if ratio > self._ls_extreme_high:
            return "EXTREME_LONG"    # Too many longs → squeeze risk
        elif ratio > 1.3:
            return "LONG_HEAVY"
        elif ratio < self._ls_extreme_low:
            return "EXTREME_SHORT"   # Too many shorts → short squeeze
        elif ratio < 0.9:
            return "SHORT_HEAVY"
        return "BALANCED"

    def _classify_signal_bias(self, data: DerivativesData) -> str:
        """
        Overall derivatives signal bias.
        This is the key derivative insight — when everyone is on one side,
        fade them.
        """
        funding_trend = data.funding_trend
        funding_delta = data.funding_delta_trend
        lsr = data.lsr_trend
        oi = data.oi_trend

        # Extreme long positioning → bearish bias (shorts will squeeze longs)
        if funding_trend == "EXTREMELY_HIGH" and lsr == "EXTREME_LONG":
            data.notes.append("⚠️ Extreme long positioning — squeeze risk HIGH")
            return "BEARISH"

        # Extreme short positioning → bullish bias (shorts will be squeezed)
        if funding_trend == "NEGATIVE" and lsr == "EXTREME_SHORT":
            data.notes.append("🚀 Extreme short positioning — short squeeze potential")
            return "BULLISH"

        # AUDIT FIX (funding delta wiring): rapidly RISING funding on an
        # already-HIGH level means the crowd is still piling long — fade
        # bias even if L/S hasn't hit EXTREME yet.  Symmetrically, rapidly
        # FALLING funding while NEGATIVE means shorts are piling in.
        if funding_trend == "HIGH" and funding_delta == "RISING" and lsr in ("LONG_HEAVY", "EXTREME_LONG"):
            data.notes.append("⚠️ Funding rising into HIGH — crowd still entering longs")
            return "BEARISH"
        if funding_trend == "NEGATIVE" and funding_delta == "FALLING" and lsr in ("SHORT_HEAVY", "EXTREME_SHORT"):
            data.notes.append("🚀 Funding falling while negative — shorts piling in")
            return "BULLISH"

        # OI rising + price rising = real momentum, support long
        if oi in ("INCREASE", "STRONG_INCREASE") and funding_trend == "NEUTRAL":
            return "BULLISH"

        # OI falling + funding high = longs exiting, caution
        if oi in ("DECREASE", "STRONG_DECREASE") and funding_trend == "HIGH":
            return "BEARISH"

        # Negative funding = someone paying to short = shorts think it goes lower
        # But negative funding can be a squeeze setup for smart money.
        # AUDIT FIX: don't classify negative funding as BULLISH during an OI
        # collapse (likely capitulation with both sides exiting) — require
        # OI to be holding or rising, or L/S to not be extreme-long.
        if funding_trend == "NEGATIVE":
            if oi in ("DECREASE", "STRONG_DECREASE") and lsr != "EXTREME_SHORT":
                data.notes.append(
                    "⚠️ Negative funding during OI decline — capitulation, "
                    "not squeeze fuel"
                )
                return "NEUTRAL"
            data.notes.append("💡 Negative funding — longs earning, squeeze potential")
            return "BULLISH"

        return "NEUTRAL"

    def _assess_squeeze_risk(self, data: DerivativesData) -> str:
        """
        Short squeeze risk — high when:
        - Many shorts (low L/S ratio)
        - Negative funding (shorts paying)
        - OI high (lots of positions to squeeze)
        - AUDIT FIX (funding delta wiring): funding rapidly falling toward
          zero/negative from a prior positive level = squeeze fuel building
          faster than the level classifier alone captures.
        """
        score = 0
        if data.lsr_trend == "EXTREME_SHORT":
            score += 2
        elif data.lsr_trend == "SHORT_HEAVY":
            score += 1
        if data.funding_trend == "NEGATIVE":
            score += 2
        if data.oi_trend in ("INCREASE", "STRONG_INCREASE"):
            score += 1
        # Rate-of-change adds conviction when it agrees with positioning
        if data.funding_delta_trend == "FALLING" and data.lsr_trend in ("SHORT_HEAVY", "EXTREME_SHORT"):
            score += 1

        if score >= 4:
            return "HIGH"
        elif score >= 2:
            return "MEDIUM"
        return "LOW"

    def _assess_liquidation_risk(self, data: DerivativesData) -> str:
        """
        Long liquidation risk — high when:
        - Many longs (high L/S ratio)
        - High positive funding
        - OI decreasing (longs running)
        """
        score = 0
        if data.lsr_trend == "EXTREME_LONG":
            score += 2
        elif data.lsr_trend == "LONG_HEAVY":
            score += 1
        if data.funding_trend == "EXTREMELY_HIGH":
            score += 2
        elif data.funding_trend == "HIGH":
            score += 1
        if data.oi_trend in ("DECREASE", "STRONG_DECREASE"):
            score += 1

        if score >= 4:
            return "HIGH"
        elif score >= 2:
            return "MEDIUM"
        return "LOW"

    def _calculate_score(self, data: DerivativesData) -> float:
        """
        Convert derivatives analysis to 0-100 score.
        50 = neutral, >50 = bullish lean, <50 = bearish lean.
        Used by signal aggregator to weight the derivatives component.
        """
        score = 50.0

        # Funding rate contribution
        funding = data.funding_rate
        threshold = self._funding_threshold

        if data.funding_trend == "NEGATIVE":
            # Negative funding = shorts paying = bullish lean
            score += min(20, abs(funding) / threshold * 10)
        elif data.funding_trend == "HIGH":
            # High funding = longs overextended = bearish lean
            score -= min(15, funding / threshold * 8)
        elif data.funding_trend == "EXTREMELY_HIGH":
            score -= min(25, funding / threshold * 10)

        # L/S ratio contribution
        if data.lsr_trend == "EXTREME_SHORT":
            score += 20  # Massive short squeeze potential
        elif data.lsr_trend == "SHORT_HEAVY":
            score += 10
        elif data.lsr_trend == "EXTREME_LONG":
            score -= 20  # Longs overextended
        elif data.lsr_trend == "LONG_HEAVY":
            score -= 10

        # OI contribution (OI up + neutral funding = genuine interest)
        if data.oi_trend == "STRONG_INCREASE" and data.funding_trend == "NEUTRAL":
            score += 8
        elif data.oi_trend == "STRONG_DECREASE":
            score -= 5

        # Squeeze/liquidation risk modifier
        if data.squeeze_risk == "HIGH":
            score += 10
        elif data.liquidation_risk == "HIGH":
            score -= 10

        return max(0.0, min(100.0, score))

    def assess_entry_validity(
        self, data: DerivativesData, direction: str
    ) -> Tuple[bool, float, list]:
        """
        Given a signal direction (LONG/SHORT), assess if derivatives
        support or oppose the trade.

        Returns:
            (is_valid, confidence_adjustment, warning_notes)
        """
        notes = []
        conf_adj = 0.0

        if direction == "LONG":
            # Check if funding is too high for longs
            if data.funding_trend == "EXTREMELY_HIGH":
                notes.append("⚠️ Extreme funding — longs paying heavily")
                conf_adj -= 12
            elif data.funding_trend == "HIGH":
                notes.append("⚡ High funding — monitor exit timing")
                conf_adj -= 5

            # Short squeeze = extra edge for longs
            if data.squeeze_risk == "HIGH":
                notes.append("🚀 Short squeeze fuel available")
                conf_adj += 8
            elif data.squeeze_risk == "MEDIUM":
                conf_adj += 4

            # OI rising = real buyers
            if data.oi_trend in ("INCREASE", "STRONG_INCREASE"):
                notes.append("✅ OI rising — real demand")
                conf_adj += 5

        elif direction == "SHORT":
            # Check if funding is too negative for shorts
            if data.funding_trend == "NEGATIVE":
                notes.append("⚠️ Negative funding — shorts paying, squeeze risk")
                conf_adj -= 8

            # High funding + many longs = short with the trend
            if data.funding_trend == "EXTREMELY_HIGH" and data.lsr_trend == "EXTREME_LONG":
                notes.append("🎯 Perfect fade setup — crowded longs")
                conf_adj += 12
            elif data.funding_trend == "HIGH":
                notes.append("✅ High funding supports short")
                conf_adj += 5

            # AUDIT FIX (symmetry): OI rising while funding is neutral means
            # new longs are entering — that's a headwind for a fresh short.
            # LONG side gets +5 for the same condition, so apply a matching
            # penalty here.
            if data.oi_trend in ("INCREASE", "STRONG_INCREASE") and \
               data.funding_trend == "NEUTRAL":
                notes.append("⚠️ OI rising with neutral funding — longs entering")
                conf_adj -= 5

            # Liquidation risk = extra edge for shorts
            if data.liquidation_risk == "HIGH":
                notes.append("💣 Long liquidation cascade risk")
                conf_adj += 8

        # Block obviously bad trades
        is_valid = conf_adj > -15  # Only block if very negative

        return is_valid, conf_adj, notes

    def format_for_telegram(self, data: DerivativesData) -> str:
        """Format derivatives data for Telegram signal card"""
        funding_str = f"{data.funding_rate:+.4f}%"
        oi_str = f"${data.open_interest/1e6:.1f}M" if data.open_interest > 0 else "N/A"
        lsr_str = f"{data.long_short_ratio:.2f}" if data.long_short_ratio > 0 else "N/A"

        emoji_map = {
            "BULLISH": "🟢",
            "BEARISH": "🔴",
            "NEUTRAL": "⚪",
            "EXTREME_LONG": "🔴",
            "EXTREME_SHORT": "🟢",
        }
        bias_emoji = emoji_map.get(data.signal_bias, "⚪")

        squeeze_emoji = {"HIGH": "🚀", "MEDIUM": "⚡", "LOW": "—"}.get(data.squeeze_risk, "—")

        return (
            f"📊 Funding: <b>{funding_str}</b> | "
            f"OI: {oi_str} ({data.oi_change_24h:+.1f}%) | "
            f"L/S: {lsr_str}\n"
            f"{bias_emoji} Derivatives bias: <b>{data.signal_bias}</b> | "
            f"Squeeze: {squeeze_emoji} {data.squeeze_risk}"
        )


# ── Singleton ──────────────────────────────────────────────
derivatives_analyzer = DerivativesAnalyzer()
