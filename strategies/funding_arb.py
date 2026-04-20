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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.loader import cfg
from config.constants import STRATEGY_VALID_REGIMES
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
    if abs(v) < 0.001 and v != 0:
        return v * 100.0
    return v


class FundingRateArb(BaseStrategy):

    name = "FundingRateArb"
    description = "Fade overcrowded funding extremes with OI confirmation"

    # Funding is regime-independent — works in all market conditions
    VALID_REGIMES = STRATEGY_VALID_REGIMES["FundingRateArb"]

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.funding_arb
        # Per-symbol cache of the previous funding delta for flip detection (B).
        # Audit P1: pair the delta with its last-seen wallclock time so a
        # long-running bot can prune symbols that have not been scanned for a
        # while (bounded by _PREV_DELTA_TTL_SEC / _PREV_DELTA_MAX_SIZE).
        self._prev_funding_delta: Dict[str, Tuple[float, float]] = {}

    # Prune stale entries more than 24 h old.  Also bound total size as a
    # defence against pathological symbol explosions.
    _PREV_DELTA_TTL_SEC: float = 24 * 60 * 60
    _PREV_DELTA_MAX_SIZE: int = 512

    def _prune_prev_delta(self, now: float) -> None:
        """Drop stale / overflow entries from the funding delta cache."""
        _ttl = self._PREV_DELTA_TTL_SEC
        if len(self._prev_funding_delta) > self._PREV_DELTA_MAX_SIZE:
            # Keep the most recent half when the cache overflows.
            _sorted = sorted(
                self._prev_funding_delta.items(),
                key=lambda kv: kv[1][1],
                reverse=True,
            )
            keep = _sorted[: self._PREV_DELTA_MAX_SIZE // 2]
            self._prev_funding_delta = {k: v for k, v in keep}
        # TTL sweep
        _stale = [k for k, (_, ts) in self._prev_funding_delta.items() if now - ts > _ttl]
        for k in _stale:
            self._prev_funding_delta.pop(k, None)

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            self._record_analyze_error(self.name, e, symbol)
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
        funding_delta_pct = 0.0
        funding_age_sec: Optional[float] = None
        squeeze_risk = "LOW"
        liquidation_risk = "LOW"
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
                funding_delta_pct = float(getattr(deriv_data, "funding_delta_pct", 0.0) or 0.0)
                squeeze_risk = str(getattr(deriv_data, "squeeze_risk", "LOW") or "LOW").upper()
                liquidation_risk = str(getattr(deriv_data, "liquidation_risk", "LOW") or "LOW").upper()
                funding_ts_ms = getattr(deriv_data, "funding_timestamp_ms", None)
                if funding_ts_ms:
                    funding_age_sec = max(0.0, time.time() - (float(funding_ts_ms) / 1000.0))
        except Exception as e:
            logger.debug(f"FundingRateArb: derivatives fetch failed for {symbol}: {e}")
            # Try raw_data fallback
            funding_rate_pct = 0.0
            oi_change    = 0.0
        # Sanity guard against obvious unit/corruption errors.  Even 1% funding
        # per 8h cycle would be historically extreme on any major exchange;
        # anything above that is almost certainly a unit-conversion bug
        # (fractional rate treated as percentage or similar).
        if abs(funding_rate_pct) > 1.0:
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

        # ── OI change confirmation (FA-Q3: scale threshold by ATR% so it is
        #    self-calibrating — a 8% OI move on BTC is a major event, but
        #    routine intraday noise on mid-cap alts with high ATR%).
        #    Fix #4: amplify the ATR scaling factor from 1× to 3× so the
        #    OI threshold is meaningfully sensitive to volatility; the
        #    original 1×(ATR/price) barely moved the bar (1–3% effect). ─────
        current_price = closes[-1]
        atr_frac = atr / current_price if current_price > 0 else 0.0
        effective_min_oi = min_oi_change * (1.0 + 3.0 * atr_frac)
        oi_confirms = abs(oi_change) >= effective_min_oi

        # ── Pre-compute normalized delta (used for confidence AND TP2 shrink) ──
        # Normalize by active_threshold so the same raw Δpp has proportional
        # meaning regardless of asset.  ≥1.0 = meaningful, <0.2 = noise.
        active_threshold = long_threshold if direction == "SHORT" else abs(short_threshold)
        normalized_delta = (
            funding_delta_pct / active_threshold
            if active_threshold > 0 and funding_delta_pct != 0.0
            else 0.0
        )

        # ── Acceleration flip (B) ─────────────────────────────────────────
        # A sign-flip in funding delta (e.g. rising → falling) often marks an
        # inflection point.  Reward when the flip aligns with the fade thesis:
        #   SHORT fade: prev_delta > 0 (funding rising) → now < 0 (falling)
        #   LONG  fade: prev_delta < 0 (funding falling) → now > 0 (rising)
        _flip_bonus = 0.0
        _prev_entry = self._prev_funding_delta.get(symbol)
        _prev_delta = _prev_entry[0] if _prev_entry is not None else None
        if (
            _prev_delta is not None
            and funding_delta_pct != 0.0
            and _prev_delta != 0.0
        ):
            _flipped = (funding_delta_pct > 0) != (_prev_delta > 0)
            if _flipped:
                _fade_aligned = (
                    (direction == "SHORT" and funding_delta_pct < 0) or
                    (direction == "LONG"  and funding_delta_pct > 0)
                )
                if _fade_aligned:
                    _flip_bonus = 4.0
        _now_ts = time.time()
        self._prev_funding_delta[symbol] = (funding_delta_pct, _now_ts)
        # Audit P1: bound cache size / TTL so long-running bots do not leak
        # memory as new symbols rotate into the scan pool.
        self._prune_prev_delta(_now_ts)

        # ── Confidence ────────────────────────────────────────────────────
        confidence = float(confidence_base)

        # Funding magnitude bonus: +1 per basis-point above threshold
        # (funding_rate_pct and long_threshold are in pp; *100 converts to 0.01%-steps)
        if direction == "SHORT":
            excess = (funding_rate_pct - long_threshold) * 100  # pp -> basis points
        else:
            excess = (abs(funding_rate_pct) - abs(short_threshold)) * 100  # pp -> basis points

        confidence += min(15, excess * 1.0)

        # Fix #6: Overcrowding exhaustion / blow-off guard — if funding is
        # already ≥ 3× the extreme threshold we may be entering at peak
        # squeeze risk (blow-off phase) where the trade is too late.
        # Penalise early to prevent chasing the very tail of a crowd extreme.
        if abs(funding_rate_pct) > active_threshold * 3:
            confidence -= 5

        # FA-Q1: Regime gate — counter-trend fades in a trending regime are
        # risky (2022 BTC bear had negative funding for months while price fell).
        # Apply a soft penalty unless funding is very extreme (≥ 2× threshold),
        # which provides enough evidence to override the trend filter.
        # Fix #3: scale the regime penalty with ADX so a weak trend applies a
        # lighter penalty and a strong trend applies a heavier one.
        adx = self.calculate_adx(highs, lows, closes, period=14)
        # Audit P1: clamp ADX to the mathematical [0, 100] range so
        # upstream computation bugs don't inflate _adx_factor.
        adx = max(0.0, min(100.0, float(adx)))
        # 40 is a conventional "strong trend" threshold on the 0-100 ADX scale;
        # at ADX=40 the full −10 applies, at ADX=20 only −5 applies.
        _adx_factor = min(1.5, adx / 40.0) if adx > 0 else 1.0
        if regime == "BULL_TREND" and direction == "SHORT":
            if funding_rate_pct < long_threshold * 2:
                confidence -= 10 * _adx_factor
        elif regime == "BEAR_TREND" and direction == "LONG":
            if abs(funding_rate_pct) < abs(short_threshold) * 2:
                confidence -= 10 * _adx_factor

        # FA-Q2: Funding trajectory adjustments.
        # Positive adjustments (unwind reward, acceleration flip, OI liq bonus)
        # are accumulated and capped so that correlated derivatives signals
        # cannot stack into an unrealistically inflated confidence (C).
        _pos_adj = 0.0

        # OI confirms (positive, accumulated under cap)
        if oi_confirms:
            _pos_adj += 8

        if funding_delta_pct != 0.0:
            if direction == "SHORT":
                # Fading extreme positive funding
                if funding_delta_pct < 0:  # Unwinding → reward the fade (positive)
                    _pos_adj += min(8, abs(normalized_delta) * 1.5)
                else:  # Continuation → penalize the fade (negative, direct)
                    confidence -= min(12, abs(normalized_delta) * 2.0)
            else:
                # Fading extreme negative funding
                if funding_delta_pct > 0:  # Unwinding → reward the fade (positive)
                    _pos_adj += min(8, abs(normalized_delta) * 1.5)
                else:  # Continuation → penalize the fade (negative, direct)
                    confidence -= min(12, abs(normalized_delta) * 2.0)

        # Acceleration flip bonus (B, positive, under cap)
        _pos_adj += _flip_bonus

        # Regime counter-trend bonus (regime-based, outside derivative cap)
        if regime in ("CHOPPY", "VOLATILE"):
            confidence += 5   # Counter-trend setups work better in ranging markets

        # FA-Q5: Graduated staleness penalty, amplified by volatility (E).
        # Old data is worse in fast markets; multiply base penalty by (1 + 2×ATR%).
        if funding_age_sec is not None:
            _vol_mult = 1.0 + 2.0 * atr_frac
            if funding_age_sec > 600:
                confidence -= 6 * _vol_mult
            elif funding_age_sec > 300:
                confidence -= 3 * _vol_mult

        # Liquidation context bonus — tiered (Fix #5), positive, under cap.
        if oi_confirms:
            if direction == "SHORT":
                if liquidation_risk == "HIGH":
                    _pos_adj += 6  # Severe crowding → full cascade risk
                elif liquidation_risk == "MEDIUM":
                    _pos_adj += 3  # Moderate crowding → partial cascade risk
            elif direction == "LONG":
                if squeeze_risk == "HIGH":
                    _pos_adj += 6  # Heavy short covering → strong squeeze
                elif squeeze_risk == "MEDIUM":
                    _pos_adj += 3  # Moderate short covering

        # Phase-2 FA-Q6: Payment-window freshness bonus.  The first 15 minutes
        # after a funding payment have the highest information value — the
        # just-paid cash flow is actively re-positioning leveraged books.
        # Reward fresh entries; do not apply a symmetric penalty (staleness
        # already handled by FA-Q5 above).
        if funding_age_sec is not None and funding_age_sec <= 900:
            _pos_adj += 3

        # Apply capped positive adjustment (C: prevents correlated bonuses from
        # inflating confidence — max combined derivative signal reward = +12).
        confidence += min(_pos_adj, 12.0)

        # ── Recent swing levels for SL ────────────────────────────────────
        # Audit P1: clamp to [1, 20] so the slice is never empty (len==1)
        # nor inverted — `highs[-0:]` returns the entire array, which silently
        # falls back to the global max/min rather than the recent swing.
        lookback_sl = max(1, min(20, len(highs) - 1))
        recent_high = float(np.max(highs[-lookback_sl:]))
        recent_low  = float(np.min(lows[-lookback_sl:]))

        # Entry lateness penalty (D) — if price is already > 1.2×ATR away from
        # the centre of the recent price range, the expected-fade move has
        # already partially occurred and we're entering late.
        recent_mid = (recent_high + recent_low) / 2.0
        if abs(current_price - recent_mid) > 1.2 * atr:
            confidence -= 3

        # ── Entry / SL / TP ───────────────────────────────────────────────
        buf = atr * rp.entry_zone_tight
        vp = compute_vol_percentile(highs, lows, closes)

        # FA-Q4: TP philosophy — TP1 is tactical (ATR-based, kept as-is).
        # TP2/TP3 are ATR-anchored proxies for funding normalization; the
        # actual exit target is when funding recedes to ≤ threshold * 0.5.
        # The 0.5 multiplier (= "half of the extreme threshold") serves as the
        # normalization proxy and is stored in raw_data as funding_exit_threshold
        # so runtime monitoring can trigger an early exit when funding normalizes.
        funding_exit_threshold = long_threshold * 0.5 if direction == "SHORT" else short_threshold * 0.5

        # Fix #1 → A: Continuous TP2 shrink — scaled by unwind strength.
        # The stronger the unwind already in progress, the smaller the
        # expected remaining move; scale from 100% (no unwind) to 70% (full
        # unwind, |normalized_delta| ≥ 1).
        # Formula: shrink_frac = min(0.30, 0.30 × |normalized_delta|)
        #          tp2_scale   = 1 − shrink_frac   (range: 0.70 – 1.00)
        _tp2_unwinding = (
            (direction == "SHORT" and funding_delta_pct < 0) or
            (direction == "LONG"  and funding_delta_pct > 0)
        )
        _nd_abs = abs(normalized_delta)
        _shrink_frac = min(0.30, 0.30 * _nd_abs) if _tp2_unwinding else 0.0
        _tp2_scale = 1.0 - _shrink_frac

        _tp2_unwinding_note = None
        if _shrink_frac > 0:
            _tp2_pct = round(_tp2_scale * 100)
            _tp2_unwinding_note = (
                f"⚡ Funding already unwinding (Δ{funding_delta_pct:+.4f}pp, "
                f"ñΔ={normalized_delta:+.2f}) "
                f"— TP2 scaled to {_tp2_pct}% of full range; "
                f"exit ≤ {funding_exit_threshold:.4f}%"
            )

        if direction == "LONG":
            entry_low   = current_price - buf
            entry_high  = current_price + buf
            stop_loss   = recent_low - atr * rp.sl_atr_mult
            tp1         = entry_high + atr * rp.volatility_scaled_tp1(tf, vp)
            _tp2_full   = entry_high + atr * rp.volatility_scaled_tp2(tf, vp)
            tp2         = entry_high + ((_tp2_full - entry_high) * _tp2_scale)
            tp3         = entry_high + atr * rp.volatility_scaled_tp3(tf, vp)
        else:
            entry_low   = current_price - buf
            entry_high  = current_price + buf
            stop_loss   = recent_high + atr * rp.sl_atr_mult
            tp1         = entry_low - atr * rp.volatility_scaled_tp1(tf, vp)
            _tp2_full   = entry_low - atr * rp.volatility_scaled_tp2(tf, vp)
            tp2         = entry_low - ((entry_low - _tp2_full) * _tp2_scale)
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
            confluence.append(
                f"✅ OI change: {oi_change:.1f}% — crowding confirmed"
                f" (min {effective_min_oi:.1f}%)"
            )
        else:
            confluence.append(
                f"⚠️ OI change: {oi_change:.1f}%"
                f" (below ATR-scaled min {effective_min_oi:.1f}%)"
            )
        if funding_age_sec is not None:
            confluence.append(f"🕒 Funding age: {funding_age_sec:.0f}s")
        confluence.append(
            f"📈 Funding trajectory: {funding_delta_trend}"
            f" (Δ{funding_delta_pct:+.4f}pp)"
        )
        if _tp2_unwinding_note:
            confluence.append(_tp2_unwinding_note)
        if regime in ("BULL_TREND", "BEAR_TREND"):
            confluence.append(
                f"⚠️ Counter-trend in {regime} — requires ≥2× threshold"
            )
        confluence.append(
            f"📊 Counter-trend fade | Regime: {regime} | TF: {tf}"
            f" | Exit signal: funding ≤ {funding_exit_threshold:.4f}%"
        )
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
                "funding_delta_pct": funding_delta_pct,
                "normalized_delta": normalized_delta,
                "funding_age_sec": funding_age_sec,
                "funding_exit_threshold": funding_exit_threshold,
                "effective_min_oi": effective_min_oi,
                "squeeze_risk": squeeze_risk,
                "liquidation_risk": liquidation_risk,
                "recent_high": recent_high,
                "recent_low": recent_low,
                "atr": atr,
                "adx": adx,
                "regime": regime,
                "tp2_scale": _tp2_scale,
                "pos_adj_capped": min(_pos_adj, 12.0),
                "flip_bonus": _flip_bonus,
            },
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
FundingArbStrategy = FundingRateArb
