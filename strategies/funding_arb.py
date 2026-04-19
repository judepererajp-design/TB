"""
TitanBot Pro — Funding Rate Arbitrage Strategy
================================================
Detects overcrowded derivatives positions and fades them.

When funding is extremely positive (longs paying shorts heavily) → SHORT
When funding is extremely negative (shorts paying longs heavily) → LONG

Logic: Extreme funding creates crowded trades. When the market is too
one-sided, a reversion is inevitable as leveraged positions get squeezed.
Combined with OI spike for confirmation of overcrowding.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.loader import cfg
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp, compute_vol_percentile

logger = logging.getLogger(__name__)


def _normalize_threshold_pct(value: float) -> float:
    """
    Funding thresholds are expected in percentage points (e.g. 0.05 = 0.05%).
    Backward-compat: accept fractional form (e.g. 0.0005 = 0.05%) and convert.
    """
    v = float(value)
    if 0 < abs(v) < 0.001:
        return v * 100.0
    return v


class FundingRateArb(BaseStrategy):

    name = "FundingRateArb"
    description = "Fade overcrowded funding extremes with OI confirmation"

    # Funding is regime-independent — works in all market conditions
    VALID_REGIMES = {"BULL_TREND", "BEAR_TREND", "VOLATILE", "CHOPPY", "UNKNOWN"}

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.funding_arb

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            logger.debug(f"FundingRateArb.analyze {symbol}: {e}")
            return None

    async def _analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime context (not a gate — funding is regime-independent) ───
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception:
            regime = "UNKNOWN"

        tf = getattr(self._cfg, "timeframe", "1h")
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 40:
            return None

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values

        atr = self.calculate_atr(highs, lows, closes, period=14)
        if atr == 0:
            return None

        long_threshold  = _normalize_threshold_pct(
            getattr(self._cfg, "extreme_long_threshold", 0.05)
        )
        short_threshold = _normalize_threshold_pct(
            getattr(self._cfg, "extreme_short_threshold", -0.03)
        )
        min_oi_change   = getattr(self._cfg, "min_oi_change", 8.0)
        lookback_periods = getattr(self._cfg, "lookback_periods", 3)
        confidence_base = getattr(self._cfg, "confidence_base", 74)
        max_funding_age_sec = float(getattr(self._cfg, "max_funding_age_sec", 900))
        if long_threshold <= 0 or short_threshold >= 0:
            logger.warning(
                "FundingRateArb: invalid thresholds long=%s short=%s (expected +/-% units in percent)",
                long_threshold,
                short_threshold,
            )
            return None

        # ── Fetch funding rate from derivatives analyzer ───────────────────
        funding_rate_pct = 0.0
        oi_change = 0.0
        funding_delta_trend = "FLAT"
        funding_age_sec: Optional[float] = None
        try:
            from analyzers.derivatives import derivatives_analyzer
            deriv_data = await derivatives_analyzer.analyze(symbol)
            if deriv_data:
                # Derivatives analyzer outputs funding in percentage points
                # (e.g. 0.05 = 0.05% per funding cycle).
                funding_rate_pct = float(getattr(deriv_data, "funding_rate", 0.0) or 0.0)
                oi_change    = getattr(deriv_data, "oi_change_24h", 0.0) or 0.0
                funding_delta_trend = str(
                    getattr(deriv_data, "funding_delta_trend", "FLAT") or "FLAT"
                ).upper()
                funding_ts_ms = getattr(deriv_data, "funding_timestamp_ms", None)
                if funding_ts_ms:
                    funding_age_sec = max(0.0, time.time() - (float(funding_ts_ms) / 1000.0))
        except Exception as e:
            logger.debug(f"FundingRateArb: derivatives fetch failed for {symbol}: {e}")
            # Try raw_data fallback
            funding_rate_pct = 0.0
            oi_change    = 0.0

        # Sanity guard against obvious unit/corruption errors.
        if abs(funding_rate_pct) > 5.0:
            logger.warning(
                "FundingRateArb: abnormal funding value for %s: %s%% (skipping)",
                symbol,
                funding_rate_pct,
            )
            return None
        if funding_age_sec is not None and funding_age_sec > max_funding_age_sec:
            logger.debug(
                "FundingRateArb: stale funding for %s age=%.1fs (>%.1fs)",
                symbol,
                funding_age_sec,
                max_funding_age_sec,
            )
            return None

        # ── Determine if funding is extreme ───────────────────────────────
        if funding_rate_pct >= long_threshold:
            direction = "SHORT"   # Fade the overcrowded longs
        elif funding_rate_pct <= short_threshold:
            direction = "LONG"    # Fade the overcrowded shorts
        else:
            return None  # Funding not extreme enough

        # ── OI change confirmation ────────────────────────────────────────
        oi_confirms = abs(oi_change) >= min_oi_change

        # ── Confidence ────────────────────────────────────────────────────
        confidence = float(confidence_base)

        # Funding magnitude bonus: +1 per 0.01% above threshold
        if direction == "SHORT":
            excess = (funding_rate_pct - long_threshold) * 100
        else:
            excess = (abs(funding_rate_pct) - abs(short_threshold)) * 100

        confidence += min(15, excess * 1.0)

        if oi_confirms:
            confidence += 8

        # Funding trajectory: fading works better when extreme funding starts
        # unwinding, not when it is still accelerating in the crowded direction.
        if direction == "SHORT":
            if funding_delta_trend == "FALLING":
                confidence += 4
            elif funding_delta_trend == "RISING":
                confidence -= 8
        else:
            if funding_delta_trend == "RISING":
                confidence += 4
            elif funding_delta_trend == "FALLING":
                confidence -= 8

        # Regime counter-trend bonus (fading trend = slightly less confidence)
        if regime in ("CHOPPY", "VOLATILE"):
            confidence += 5   # Counter-trend setups work better in ranging markets

        # ── Recent swing levels for SL ────────────────────────────────────
        lookback_sl = min(20, len(highs) - 1)
        recent_high = float(np.max(highs[-lookback_sl:]))
        recent_low  = float(np.min(lows[-lookback_sl:]))
        current_price = closes[-1]

        # ── Entry / SL / TP ───────────────────────────────────────────────
        buf = atr * rp.entry_zone_tight
        vp = compute_vol_percentile(highs, lows, closes)

        if direction == "LONG":
            entry_low   = current_price - buf
            entry_high  = current_price + buf
            stop_loss   = recent_low - atr * rp.sl_atr_mult
            # FIX: were hardcoded 1.5/2.5/3.5 ATR — now uses rp.* so settings.yaml tuning applies
            # Wire timeframe-scaled TPs with volatility adjustment
            tp1         = entry_high + atr * rp.volatility_scaled_tp1(tf, vp)
            tp2         = entry_high + atr * rp.volatility_scaled_tp2(tf, vp)
            tp3         = entry_high + atr * rp.volatility_scaled_tp3(tf, vp)
        else:
            entry_low   = current_price - buf
            entry_high  = current_price + buf
            stop_loss   = recent_high + atr * rp.sl_atr_mult
            tp1         = entry_low - atr * rp.volatility_scaled_tp1(tf, vp)
            tp2         = entry_low - atr * rp.volatility_scaled_tp2(tf, vp)
            tp3         = entry_low - atr * rp.volatility_scaled_tp3(tf, vp)

        entry_mid = (entry_low + entry_high) / 2.0
        risk = (entry_mid - stop_loss) if direction == "LONG" else (stop_loss - entry_mid)
        if risk <= 0:
            return None
        rr_ratio = abs(tp2 - entry_mid) / risk

        confluence: List[str] = [
            f"✅ Funding rate: {funding_rate_pct:.4f}% ({'extreme LONG crowding' if direction == 'SHORT' else 'extreme SHORT crowding'})",
            f"   Threshold: {'>' + str(long_threshold) + '%' if direction == 'SHORT' else '<' + str(short_threshold) + '%'}",
        ]
        if oi_confirms:
            confluence.append(f"✅ OI change: {oi_change:.1f}% — crowding confirmed")
        else:
            confluence.append(f"⚠️ OI change: {oi_change:.1f}% (below {min_oi_change}% threshold)")
        if funding_age_sec is not None:
            confluence.append(f"🕒 Funding age: {funding_age_sec:.0f}s")
        confluence.append(f"📈 Funding trajectory: {funding_delta_trend}")
        confluence.append(f"📊 Counter-trend fade | Regime: {regime} | TF: {tf}")
        confluence.append(f"🎯 R:R {rr_ratio:.2f} | ATR: {fmt_price(atr)}")

        confidence = min(92, max(40, confidence))

        candidate = SignalResult(
            symbol=symbol,
            direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
            strategy=self.name,
            confidence=confidence,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            tp1=tp1, tp2=tp2, tp3=tp3,
            rr_ratio=rr_ratio,
            atr=atr,
            setup_class="intraday",
            timeframe=tf,
            analysis_timeframes=[tf],
            confluence=confluence,
            raw_data={
                "funding_rate": funding_rate_pct,
                "oi_change": oi_change,
                "oi_confirms": oi_confirms,
                "funding_delta_trend": funding_delta_trend,
                "funding_age_sec": funding_age_sec,
                "recent_high": recent_high,
                "recent_low": recent_low,
                "atr": atr,
                "regime": regime,
            },
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
FundingArbStrategy = FundingRateArb
