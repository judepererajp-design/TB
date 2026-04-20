"""
TitanBot Pro — Range Scalper Strategy
=======================================
Dedicated chop/range-bound strategy that trades supply/demand zone bounces.

Only activates when chop_strength > 0.40 (regime is ranging).
Looks for:
  1. Price at range extremes (outer 20% of recent range)
  2. Rejection candle confirming zone reaction
  3. RSI divergence at extremes (oversold at demand, overbought at supply)
  4. Volume confirmation on the rejection

Targets: Range midpoint (equilibrium), NOT trend targets.
Stops: Tight — just outside the zone boundary.
R:R: Typically 1.5-2.5:1 (smaller moves, higher probability)
Hold time: Short — exits quickly at equilibrium.

This is how institutional traders profit during sideways markets:
  "Trade the edges, avoid the middle."
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from config.loader import cfg
from strategies.base import BaseStrategy, SignalResult, SignalDirection
from utils.formatting import fmt_price
from utils.risk_params import rp

logger = logging.getLogger(__name__)


class RangeScalperStrategy(BaseStrategy):

    name = "RangeScalper"
    description = "S/D zone bounce scalping during choppy/ranging markets"

    def __init__(self):
        super().__init__()
        try:
            self._cfg = cfg.strategies.range_scalper
        except Exception:
            self._cfg = None

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            self._record_analyze_error(self.name, e, symbol)
            return None

    async def _analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── 0. Check regime — only trade in ranging/choppy markets ──
        # RangeScalper is a mean-reversion strategy — it MUST NOT fire in:
        # - VOLATILE: high ATR, whipsaw candles, no mean to revert to
        # - BULL_TREND / BEAR_TREND: trending market eats range stops
        # - VOLATILE_PANIC: extreme conditions, emergency sizing only
        try:
            from analyzers.regime import regime_analyzer, Regime
            chop = regime_analyzer.chop_strength
            regime = regime_analyzer.regime
            
            # Block on trending/volatile regimes regardless of chop value
            _blocked_regimes = (Regime.VOLATILE, Regime.VOLATILE_PANIC, 
                                Regime.BULL_TREND, Regime.BEAR_TREND)
            if regime in _blocked_regimes:
                return None  # Range scalper kills P&L in trending/volatile conditions
            
            # Also block on insufficient chop (original check)
            if chop < 0.40:
                return None  # Not choppy enough — let trend strategies handle it
        except Exception:
            return None

        # ── Market state gate: compression warning ─────────────
        _ms_state = None
        _ms_compression_penalty = 0
        _ms_result = None
        try:
            from analyzers.market_state_engine import market_state_engine, MarketState
            _ms_result = await market_state_engine.get_state()
            _ms_state = _ms_result.state
            if _ms_state == MarketState.COMPRESSION:
                if _ms_result.compression_bars >= 8:
                    return None  # Range about to break — don't scalp
                elif _ms_result.compression_bars >= 4:
                    _ms_compression_penalty = -12  # Reduce confidence
            elif _ms_state in (MarketState.EXPANSION, MarketState.TRENDING):
                return None  # Market is trending/expanding — no range to scalp
        except Exception:
            pass

        # ── BTC breakout gate (RS-Q4) ─────────────────────────────
        # If BTC is breaking out / strongly trending, alt ranges are unstable.
        # Use the already-fetched MarketStateResult for BTC momentum data.
        # Hard reject at 2.5% 6h momentum; soft −6 confidence at 1.5% (graded).
        _btc_soft_penalty = 0
        if _ms_result is not None:
            _btc_momentum_abs = abs(_ms_result.btc_momentum_fast)
            _btc_bias = _ms_result.direction_bias
            _btc_breakout_thresh = 0.025  # 2.5% BTC 6h momentum = trending, not ranging
            _btc_soft_thresh = 0.015      # 1.5% — elevated momentum, degrade confidence
            if _btc_momentum_abs >= _btc_breakout_thresh and _btc_bias != "NEUTRAL":
                return None  # BTC strongly trending — alt ranges unreliable
            elif _btc_momentum_abs >= _btc_soft_thresh and _btc_bias != "NEUTRAL":
                # Convex penalty: rises from 6→10 as momentum approaches the hard block.
                # Risk increases nonlinearly as BTC trends toward breakout velocity.
                _btc_scale = (_btc_momentum_abs - _btc_soft_thresh) / (
                    _btc_breakout_thresh - _btc_soft_thresh
                )
                _btc_soft_penalty = int(6 + 4 * min(1.0, _btc_scale))

        # ── 1. Get candle data ────────────────────────────────────
        tf = '15m'  # Scalper uses 15m for entries
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 80:
            # Fallback to 1h
            tf = '1h'
            if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 50:
                return None

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

        closes  = df['close'].values
        highs   = df['high'].values
        lows    = df['low'].values
        opens   = df['open'].values
        volumes = df['volume'].values

        current = closes[-1]
        atr = self.calculate_atr(highs, lows, closes, 14)
        if atr == 0 or current == 0:
            return None

        # ── 2. Config params ──────────────────────────────────────
        lookback     = self._get_cfg('range_lookback', 40)
        edge_pct     = self._get_cfg('edge_pct', 0.20)      # Outer 20% of range
        edge_atr_floor = self._get_cfg('edge_atr_floor', 0.75)
        edge_atr_cap = self._get_cfg('edge_atr_cap', 3.0)
        rsi_ob       = self._get_cfg('rsi_overbought', 68)
        rsi_os       = self._get_cfg('rsi_oversold', 32)
        vol_mult     = self._get_cfg('vol_confirmation', 1.3)
        conf_base    = self._get_cfg('confidence_base', 62)
        min_range_atr = self._get_cfg('min_range_atr', 3.0)  # Range must be at least 3x ATR

        # ── 3. Detect range boundaries ────────────────────────────
        # Use swing-pivot percentiles instead of raw max/min.
        # A single wick spike 30 bars ago would anchor range_high far above
        # where price has actually been ranging, making the supply zone
        # unreachable and blocking all SHORT signals.
        # 90th/10th percentile of recent highs/lows filters out spike wicks.
        recent_highs = highs[-lookback:]
        recent_lows  = lows[-lookback:]

        range_high = float(np.percentile(recent_highs, 90))
        range_low  = float(np.percentile(recent_lows, 10))
        range_high_abs = float(np.max(recent_highs))
        range_low_abs = float(np.min(recent_lows))
        range_size = range_high - range_low

        if range_size < atr * min_range_atr:
            return None  # Range too tight — no room to scalp

        equilibrium = (range_high + range_low) / 2.0
        edge_band = range_size * edge_pct
        edge_band = max(edge_band, atr * edge_atr_floor)
        edge_band = min(edge_band, atr * edge_atr_cap, range_size * 0.45)

        supply_zone_start = range_high - edge_band
        demand_zone_start = range_low + edge_band

        # ── Range decay detection ─────────────────────────────────
        # Detects a range that is quietly tightening (compression) while
        # intrabar volatility is rising — a pre-breakout pattern that invalidates
        # range-scalping assumptions.  Compare older vs newer sub-window of the
        # lookback without external state; also checks ATR acceleration.
        _range_decaying = False
        _half = max(10, lookback // 2)
        if len(highs) >= lookback:
            _early_range = (
                float(np.percentile(highs[-lookback:-_half], 90))
                - float(np.percentile(lows[-lookback:-_half], 10))
            )
            _recent_range_local = (
                float(np.percentile(highs[-_half:], 90))
                - float(np.percentile(lows[-_half:], 10))
            )
            _atr_slow = self.calculate_atr(highs, lows, closes, min(30, len(closes) - 1))
            _range_decaying = (
                _early_range > 0
                and _recent_range_local < _early_range * 0.80  # range shrinking ≥20%
                and _atr_slow > 0
                and atr > _atr_slow * 0.95  # intrabar volatility not yet contracting
            )

        # ── 4. Determine if price is at a zone edge ───────────────
        direction = None
        zone_name = ""

        if current <= demand_zone_start:
            direction = "LONG"
            zone_name = "demand"
        elif current >= supply_zone_start:
            direction = "SHORT"
            zone_name = "supply"
        else:
            return None  # Price in mid-range — no trade

        # ── 5. Confluence checks ──────────────────────────────────
        confluence = []
        confidence = conf_base + _ms_compression_penalty - _btc_soft_penalty

        # a) Zone position — deeper in zone = higher confidence
        if direction == "LONG":
            depth = (demand_zone_start - current) / edge_band
        else:
            depth = (current - supply_zone_start) / edge_band
        depth = max(0, min(1, depth))

        confluence.append(
            f"✅ Price at range {zone_name} zone (depth: {depth:.0%})"
        )
        confidence += depth * 8  # Up to +8 for being deep in zone

        # Edge proximity sanity: even within the zone, the trade is only worthwhile
        # if price is close to the actual range wall.  If it's more than 1.2 ATR
        # away from the extreme, price is mid-band and timing is poor.
        if direction == "LONG":
            _distance_to_edge = current - range_low
        else:
            _distance_to_edge = range_high - current
        if _distance_to_edge > 1.2 * atr:
            confidence -= 4
            confluence.append(
                f"⚠️ Edge proximity: {_distance_to_edge / atr:.1f}× ATR from range wall (-4)"
            )

        # b) RSI confirmation
        rsi = self.calculate_rsi(closes, 14)
        if direction == "LONG" and rsi <= rsi_os:
            confluence.append(f"✅ RSI oversold: {rsi:.1f}")
            confidence += 8
        elif direction == "SHORT" and rsi >= rsi_ob:
            confluence.append(f"✅ RSI overbought: {rsi:.1f}")
            confidence += 8
        elif direction == "LONG" and rsi <= 40:
            confluence.append(f"📊 RSI low: {rsi:.1f}")
            confidence += 3
        elif direction == "SHORT" and rsi >= 60:
            confluence.append(f"📊 RSI high: {rsi:.1f}")
            confidence += 3

        # c) Rejection candle at zone
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        body = abs(c - o)
        total = h - l
        has_rejection = False

        if total > 0:
            if direction == "LONG":
                lower_wick = min(o, c) - l
                if lower_wick > body * 1.2 and lower_wick > total * 0.4:
                    confluence.append("✅ Rejection wick at demand zone")
                    confidence += 10
                    has_rejection = True
            else:
                upper_wick = h - max(o, c)
                if upper_wick > body * 1.2 and upper_wick > total * 0.4:
                    confluence.append("✅ Rejection wick at supply zone")
                    confidence += 10
                    has_rejection = True

        # d) Volume on rejection
        avg_vol = np.mean(volumes[-20:])
        cur_vol = volumes[-1]
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

        if vol_ratio >= vol_mult:
            confluence.append(f"✅ Volume: {vol_ratio:.1f}x avg")
            confidence += 5

        # e) Previous zone bounces (has this level held before?)
        bounce_count = self._count_zone_bounces(
            highs, lows, closes, direction, range_high, range_low, edge_band, lookback
        )
        if bounce_count >= 1:
            _bounce_bonus = min(6, bounce_count * 2)
            _bounce_label = f"✅ Zone tested {bounce_count}x — strong S/D level" if bounce_count >= 2 else "📊 Zone tested once before"
            confluence.append(_bounce_label)
            confidence += _bounce_bonus

        # f) ADX confirms no trend (must be low for range trading)
        adx = self.calculate_adx(highs, lows, closes, 14)
        if adx < 20:
            confluence.append(f"✅ ADX: {adx:.1f} — confirmed range-bound")
            confidence += 4
        elif adx < 25:
            confluence.append(f"📊 ADX: {adx:.1f} — weak trend")
            confidence += 2
        elif adx > 30:
            # Strong trend starting — abort range scalp
            return None

        # ── 6. Minimum confluence gate ────────────────────────────
        # Need at least: zone + one confirmation (rejection, RSI, or volume)
        if not has_rejection and rsi > rsi_os and rsi < rsi_ob and vol_ratio < vol_mult:
            return None  # No confirmation at all — skip

        # g) Range decay penalty — range tightening while vol rising = pre-breakout
        if _range_decaying:
            confidence -= 4
            confluence.append("⚠️ Range decay: structure compressing with rising vol (-4)")

        # ── 7. Calculate levels ───────────────────────────────────
        # Targets: midpoint (equilibrium), NOT trend continuation
        # Stops: just outside the zone boundary — tight

        # PHASE 2 FIX (P2-A): BOS check for range scalper.
        # If price has broken range structure (making new HL/LH beyond the range),
        # the range is invalid and we should not be fading it.
        bos_hold_bars = max(1, int(self._get_cfg('bos_hold_bars', 2)))
        bos_breach_pct = max(0.0, float(self._get_cfg('bos_breach_pct', 0.002)))
        if direction == "LONG":
            _bos_level = range_low * (1.0 - bos_breach_pct)
            if len(closes) >= bos_hold_bars and bool(np.all(closes[-bos_hold_bars:] < _bos_level)):
                confidence -= 15
                confluence.append(f"⚠️ BOS gate: {bos_hold_bars} closes below range low — scalp confidence reduced")
        elif direction == "SHORT":
            _bos_level = range_high * (1.0 + bos_breach_pct)
            if len(closes) >= bos_hold_bars and bool(np.all(closes[-bos_hold_bars:] > _bos_level)):
                confidence -= 15
                confluence.append(f"⚠️ BOS gate: {bos_hold_bars} closes above range high — scalp confidence reduced")

        # SL BUFFER FIX: was atr * sl_atr_mult * 0.4 — only 40% of normal.
        # On volatile alts (the majority of the 200+ scan pool), 40% of 1.2 ATR = 0.48 ATR.
        # Normal 5m noise on a micro-cap alt easily exceeds 0.5 ATR, triggering the stop
        # before the scalp can develop.  Raised to 0.65× (0.78 ATR) — still tighter than
        # trend strategies (1.2 ATR) but resilient to intrabar noise on thin instruments.
        sl_buffer = atr * rp.sl_atr_mult * 0.65

        if direction == "LONG":
            entry_low  = current - atr * rp.entry_zone_atr * 0.75
            entry_high = current + atr * rp.entry_zone_tight
            stop_loss  = range_low_abs - sl_buffer
            tp1        = current + (equilibrium - current) * 0.50    # Halfway to EQ
            tp2        = equilibrium                                   # Full EQ
            # Phase-2 RS-Q4: TP3 = EQ + min(0.5·ATR, 10% past-EQ).  The 10%
            # cap prevents a large ATR on a tight range from pushing TP3 near
            # the opposite edge (sell zone for LONG).  The ATR floor gives
            # a structurally meaningful target on ranges that are EQ-dominated
            # (where 10% of the upper half would be trivially small).
            tp3        = equilibrium + min(atr * 0.5, (range_high - equilibrium) * 0.10)
        else:
            entry_high = current + atr * rp.entry_zone_atr * 0.75
            entry_low  = current - atr * rp.entry_zone_tight
            stop_loss  = range_high_abs + sl_buffer
            tp1        = current - (current - equilibrium) * 0.50
            tp2        = equilibrium
            # Mirror: TP3 = EQ − min(0.5·ATR, 10% past-EQ).
            tp3        = equilibrium - min(atr * 0.5, (equilibrium - range_low) * 0.10)

        # ── 8. Risk/Reward check ──────────────────────────────────
        risk = abs(current - stop_loss)
        reward = abs(tp2 - current)
        rr = reward / risk if risk > 0 else 0

        # Range scalps can have lower R:R than trend trades (higher probability)
        min_rr = self._get_cfg('min_rr', 1.3)
        if rr < min_rr:
            return None

        # ── 9. Build signal ───────────────────────────────────────
        confluence.append(f"📐 Range: {self._fmt_p(range_low)}–{self._fmt_p(range_high)}")
        confluence.append(f"⚖️ Equilibrium: {self._fmt_p(equilibrium)}")
        confluence.append(f"🎯 Chop strength: {chop:.2f}")

        if _ms_compression_penalty:
            confluence.append(f"⚠️ Compression detected: range tightening ({_ms_compression_penalty})")

        if confidence < 45:
            return None
        confidence = min(88, max(45, confidence))

        signal = SignalResult(
            symbol=symbol,
            direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
            strategy=self.name,
            confidence=confidence,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            tp1=tp1, tp2=tp2, tp3=tp3,
            rr_ratio=round(rr, 2), atr=atr,
            setup_class="scalp", analysis_timeframes=[tf],
            confluence=confluence,
            raw_data={
                'range_high': range_high,
                'range_low': range_low,
                'equilibrium': equilibrium,
                'zone': zone_name,
                'depth': depth,
                'rsi': rsi,
                'adx': adx,
                'bounce_count': bounce_count,
                'chop_strength': chop,
                'market_state': getattr(_ms_state, 'value', None) if _ms_state else None,
            }
        )

        if not self.validate_signal(signal):
            return None

        return signal

    # ── Helpers ────────────────────────────────────────────────

    def _count_zone_bounces(
        self, highs, lows, closes, direction, range_high, range_low, edge_band, lookback
    ) -> int:
        """Count how many times price bounced off this zone in recent history"""
        count = 0
        # Audit P1: clamp lookback to array length so a misconfigured large
        # lookback (or short OHLCV window) cannot wrap Python negative indices
        # back into stale data at the start of the array.  We also need at
        # least 3 bars so the `i=-lookback → closes[i+1]` check is meaningful.
        _n = min(len(highs), len(lows), len(closes))
        if _n < 3:
            return 0
        effective_lookback = min(int(lookback), _n - 1)
        if effective_lookback < 2:
            return 0
        if direction == "LONG":
            zone_top = range_low + edge_band
            for i in range(-effective_lookback, -2):
                # Price entered demand zone then bounced up
                if lows[i] <= zone_top and closes[i + 1] > zone_top:
                    count += 1
        else:
            zone_bottom = range_high - edge_band
            for i in range(-effective_lookback, -2):
                # Price entered supply zone then bounced down
                if highs[i] >= zone_bottom and closes[i + 1] < zone_bottom:
                    count += 1
        return min(count, 5)  # Cap at 5

    def _get_cfg(self, key: str, default):
        """Safe config access with fallback"""
        if self._cfg is None:
            return default
        return getattr(self._cfg, key, default)

    @staticmethod
    def _fmt_p(price: float) -> str:
        """Quick price format for confluence strings"""
        return fmt_price(price)
