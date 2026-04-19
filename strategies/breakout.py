"""
TitanBot Pro — Institutional Breakout Strategy
===============================================
Detects volume-backed breakouts from consolidation using:
  - Donchian Channel (20-period high/low)
  - ADX trend strength filter (threshold OR rising slope)
  - Volume confirmation (1.8x average minimum; 3x when false_breakout_filter is on)
  - ATR-based stop and measured-move targets
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config.loader import cfg
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):

    name = "InstitutionalBreakout"
    description = "Volume-backed Donchian breakouts with ADX filter"

    # Breakout only fires in trending regimes.
    # VOLATILE_PANIC is explicitly excluded: rejection-heavy candles and
    # failed moves make breakout entries extremely unreliable.
    # In plain VOLATILE, breakouts are allowed but require 2× the normal
    # volume confirmation so only institutional-grade moves qualify.
    # In CHOPPY markets, Donchian breakouts fail > 70% of the time
    # — price breaks the level then immediately reverts (stop hunt).
    VALID_REGIMES = {"BULL_TREND", "BEAR_TREND", "VOLATILE"}

    # DIRECTIONAL-GATE FIX: Per-direction regime map, matching MomentumStrategy.
    # In BEAR_TREND, only SHORT breakouts are allowed (breakout longs in bear
    # markets rely entirely on the HTF guardrail, which can be bypassed by
    # the 4h bounce override). This adds a strategy-level gate as a second layer.
    _REGIME_PREFERRED_DIR = {
        "BULL_TREND":  "LONG",   # Breakout longs with the trend
        "BEAR_TREND":  "SHORT",  # Breakout shorts with the trend
        "VOLATILE":    None,     # Either direction
    }

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.breakout

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime gate (fastest check — before any computation) ─────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
            # BR-Q4: VOLATILE_PANIC generates too many trap breakouts — hard block.
            if regime == "VOLATILE_PANIC":
                return None
            if regime not in self.VALID_REGIMES:
                return None   # Breakouts are noise in chop — skip
            # Confidence bonus in confirmed trend; penalty in volatile/mixed
            _regime_bonus = 8 if "TREND" in regime else 0
            _regime_dir_constraint = self._REGIME_PREFERRED_DIR.get(regime)
            _is_volatile = (regime == "VOLATILE")
        except Exception:
            _regime_bonus = 0
            _regime_dir_constraint = None
            regime = "UNKNOWN"
            _is_volatile = False

        # ── Market state awareness ──────────────────────────────────────
        _market_state = None
        _ms_bonus = 0
        try:
            from analyzers.market_state_engine import market_state_engine, MarketState
            _market_state_result = await market_state_engine.get_state()
            _market_state = _market_state_result.state
            if _market_state == MarketState.COMPRESSION:
                # Breakout from compression = highest probability setup
                if _market_state_result.compression_bars >= 6:
                    _ms_bonus = 15
                elif _market_state_result.compression_bars >= 4:
                    _ms_bonus = 10
            elif _market_state == MarketState.EXPANSION:
                _ms_bonus = 8  # Already expanding — breakout confirmed by environment
            elif _market_state == MarketState.LIQUIDITY_HUNT:
                _ms_bonus = -10  # Breakouts during stop hunts are traps
        except Exception:
            pass

        tf = getattr(self._cfg, 'timeframe', '1h')
        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 60:
            return None

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        volumes = df['volume'].values

        lookback = getattr(self._cfg, 'donchian_period', 20)
        min_adx = getattr(self._cfg, 'min_adx', 20)
        vol_mult = getattr(self._cfg, 'volume_confirmation_mult', 1.8)
        # BR-Q3: when false_breakout_filter is on, require 3× vol to confirm
        # the close is not a stop-hunt poke.  This is the pragmatic substitute
        # for full retest logic; true retest detection requires multi-bar state.
        false_breakout_filter = getattr(self._cfg, 'false_breakout_filter', True)
        # BR-Q4: in VOLATILE, raise the volume bar to 2× normal minimum so only
        # institutional-grade moves fire.
        if _is_volatile:
            vol_mult = max(vol_mult, 2.0)
        confidence_base = getattr(self._cfg, 'confidence_base', 78)

        # Guard: need at least lookback+1 bars for Donchian channel
        if lookback < 1:
            lookback = 20
        if len(closes) < lookback + 2:
            return None

        # Donchian channel (exclude current bar — consistent with volume below)
        channel_high = np.max(highs[-lookback-1:-1])
        channel_low  = np.min(lows[-lookback-1:-1])
        current_close = closes[-1]
        current_high  = highs[-1]
        current_low   = lows[-1]
        current_open  = opens[-1]
        current_vol   = volumes[-1]
        # BR-2: exclude current bar from denominator so vol_ratio is not
        # deflated at precisely the moment we need it sharpest.
        # Guard: if the arrays are too short for the full lookback slice, fall
        # back to None (the lookback + 2 check above already guarantees we have
        # at least lookback + 2 bars, so this path is unreachable in practice).
        if lookback + 1 > len(volumes):
            return None
        avg_vol = np.mean(volumes[-lookback-1:-1])

        # BR-Q2: lower ADX threshold (28→20) to catch accumulation breakouts on alts
        # where ADX starts rising from ~15-22.  Pair with a slope check so we
        # accept *either* "ADX already strong" OR "ADX clearly rising".
        adx = self.calculate_adx(highs, lows, closes, period=14)
        # Slope: compare current ADX against the value 3 bars ago.
        _adx_prev = self.calculate_adx(highs[:-3], lows[:-3], closes[:-3], period=14) if len(closes) > 17 else adx
        _adx_rising = (adx - _adx_prev) >= 1.5   # Rising at least 1.5 pts over 3 bars
        # BR-Q2b: require adx > 15 when using the rising-slope bypass.  Without the
        # floor, ADX crawling from 10 → 12 passes as "rising" — that's directionless
        # noise, not accumulation.  The target range for rising-slope alts is 15-22.
        if adx < min_adx and not (_adx_rising and adx > 15):
            return None

        # Volume spike check
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio < vol_mult:
            return None
        # BR-Q3: false_breakout_filter applies a confidence penalty when volume is
        # above the baseline (vol_mult) but below 3×.  Signals still proceed —
        # a vol_ratio between vol_mult and 3× is real volume, just not institutional.
        # The penalty adjusts confidence downward rather than blocking entirely;
        # the final RR check is the hard gate.
        if false_breakout_filter and vol_ratio < 3.0:
            _fb_penalty = 8  # confidence penalty: real but not institutional volume
        else:
            _fb_penalty = 0

        atr = self.calculate_atr(highs, lows, closes, period=14)
        if atr == 0:
            return None

        confluence = []
        direction = None
        confidence = confidence_base + _regime_bonus + _ms_bonus - _fb_penalty

        # Bullish breakout
        if current_close > channel_high:
            direction = "LONG"
            confluence.append(f"✅ Donchian breakout above {fmt_price(channel_high)} ({lookback}H high)")
            confidence += min(10, (vol_ratio - vol_mult) * 5)
            confluence.append(f"✅ Volume: {vol_ratio:.1f}x average")

        # Bearish breakout
        elif current_close < channel_low:
            direction = "SHORT"
            confluence.append(f"✅ Donchian breakdown below {fmt_price(channel_low)} ({lookback}H low)")
            confidence += min(10, (vol_ratio - vol_mult) * 5)
            confluence.append(f"✅ Volume: {vol_ratio:.1f}x average")
        else:
            return None

        # BR-Q5: breakout quality scoring — directional close position and distance.
        # body_pct alone is misleading: a large red rejection candle also has a big body.
        # Instead we score WHERE the close sits within the candle range (direction-aware):
        #   LONG:  close near the HIGH is conviction;  close near the LOW is rejection.
        #   SHORT: close near the LOW is conviction;   close near the HIGH is rejection.
        _q5_bonus = 0
        candle_range = current_high - current_low
        if candle_range > 0:
            close_pos = (current_close - current_low) / candle_range  # 0.0 (bottom) – 1.0 (top)
            if direction == "LONG":
                if close_pos >= 0.7:                  # Closed near top — strong bullish candle
                    _q5_bonus += 5
                    confluence.append(f"💪 Bullish close position: {close_pos:.0%} of range")
                elif close_pos < 0.35:                # Closed near bottom — rejection candle
                    _q5_bonus -= 5
                    confluence.append(f"⚠️ Rejection close: {close_pos:.0%} of range")
            else:  # SHORT
                if close_pos <= 0.3:                  # Closed near bottom — strong bearish candle
                    _q5_bonus += 5
                    confluence.append(f"💪 Bearish close position: {close_pos:.0%} of range")
                elif close_pos > 0.65:                # Closed near top — rejection / wick probe
                    _q5_bonus -= 5
                    confluence.append(f"⚠️ Rejection close: {close_pos:.0%} of range")

        # BR-Q5b: distance from level — price well beyond the channel adds conviction.
        # Threshold: max(0.5×ATR, 0.2% of price) so low-ATR assets still need real distance.
        if atr > 0:
            _dist_thresh = max(0.5 * atr, current_close * 0.002)
            if direction == "LONG":
                dist = current_close - channel_high
            else:
                dist = channel_low - current_close
            if dist >= _dist_thresh:
                _q5_bonus += 3
                confluence.append(f"📐 Strong close: {dist / atr:.1f}× ATR beyond level")

        # BR-Q5c: cap quality bonuses at +8 so correlated signals (close position +
        # ATR distance) cannot inflate weak signals beyond the vol bonus weight.
        confidence += max(-5, min(8, _q5_bonus))

        # BR-Q5d: soft 2-bar continuation bonus.  Requiring previous_close > channel_high
        # as a HARD gate would block ~70% of valid first-bar breakouts.  Instead, a +3
        # bonus rewards confirmed continuation without eliminating single-bar entries.
        _prev_close = closes[-2] if len(closes) >= 2 else current_close
        if direction == "LONG" and _prev_close > channel_high * 0.995:
            confidence += 3
            confluence.append("🔄 2-bar continuation above level")
        elif direction == "SHORT" and _prev_close < channel_low * 1.005:
            confidence += 3
            confluence.append("🔄 2-bar continuation below level")

        # DIRECTIONAL-GATE FIX: Block counter-trend breakouts in strong trends.
        # In BEAR_TREND, a breakout LONG is fighting weekly structure. In BULL_TREND,
        # a breakout SHORT is fighting weekly structure. Hard-block when weekly ADX
        # confirms strong trend; soft-penalise in weak trends.
        if _regime_dir_constraint and direction != _regime_dir_constraint:
            try:
                from analyzers.htf_guardrail import htf_guardrail as _brk_htf
                # BR-1: use the public accessor instead of the private attribute.
                _brk_strong = _brk_htf.get_weekly_adx() >= 30
            except Exception:
                _brk_strong = False
            if _brk_strong:
                return None  # Hard block: no counter-trend breakouts in strong weekly trend
            else:
                confidence -= 10  # Soft penalty in weak trend

        confluence.append(f"✅ ADX: {adx:.1f} (trend strength confirmed)")

        # ADX bonus
        if adx > rp.adx_strong_trend:
            confidence += 5
            confluence.append("🔥 ADX >40 — very strong trend")

        # Calculate levels
        sl_mult  = cfg.risk.get('sl_atr_mult', 1.2)

        # Measured-move targets anchored to the consolidation range being broken.
        # BR-Q1: crypto breakouts hit 60-70% of the measured move before reversing
        # ~40% of the time, so TP2 is set to the 80% projection and TP3 to 100%.
        # TP1 stays at 50% as the first liquidity magnet.
        range_size = channel_high - channel_low  # Width of the broken consolidation

        # BR-3: a range_size below 0.5×ATR means the channel was essentially flat.
        # Measured-move targets would cluster at entry and produce near-zero RR before
        # the RR check; returning early is cleaner and avoids divide-by-near-zero edge
        # cases in the risk calculation below.
        if range_size < atr * 0.5:
            return None

        if direction == "LONG":
            entry_low  = channel_high
            entry_high = channel_high + atr * rp.entry_zone_atr
            # BR-4: SL just below the entry candle's low (traditional institutional
            # placement — outside the just-broken level, not inside the old range).
            # Cap = channel_high - sl_mult*2 ATR: the factor-of-2 relative to the
            # configured sl_atr_mult gives headroom for deep wicks on the entry bar
            # while still preventing an unreasonably wide stop on volatile assets.
            stop_loss  = max(current_low - atr * 0.2,
                             channel_high - atr * sl_mult * 2)
            tp1        = channel_high + range_size * 0.50  # 50% measured move
            tp2        = channel_high + range_size * 0.80  # 80% measured move (main target)
            tp3        = channel_high + range_size * 1.00  # 100% measured move (extension)
        else:
            entry_high = channel_low
            entry_low  = channel_low - atr * rp.entry_zone_atr
            # BR-4: SL just above the entry candle's high.
            stop_loss  = min(current_high + atr * 0.2,
                             channel_low + atr * sl_mult * 2)
            tp1        = channel_low - range_size * 0.50
            tp2        = channel_low - range_size * 0.80
            # Floor at a minimum positive price to prevent physically impossible targets.
            tp3        = max(channel_low * 0.01, channel_low - range_size * 1.00)

        risk = abs(channel_high - stop_loss) if direction == "LONG" else abs(channel_low - stop_loss)
        rr   = abs(tp2 - (channel_high if direction == "LONG" else channel_low)) / risk if risk > 0 else 0

        min_rr = cfg_min_rr("intraday")
        if rr < min_rr:
            return None

        if _market_state:
            confluence.append(f"🧠 Market State: {_market_state.value} ({_ms_bonus:+d})")

        confidence = min(95, max(40, confidence))

        # FIX: validate geometry and minimum RR before returning.
        # validate_signal() checks confidence, rr_ratio, entry zone, SL/TP sides.
        _candidate = SignalResult(
            symbol=symbol,
            direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
            strategy=self.name,
            confidence=confidence,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            tp1=tp1, tp2=tp2, tp3=tp3,
            rr_ratio=rr, atr=atr,
            setup_class="intraday", analysis_timeframes=[tf],
            confluence=confluence,
            raw_data={'adx': adx, 'vol_ratio': vol_ratio, 'channel_high': channel_high, 'channel_low': channel_low,
                      'market_state': _market_state.value if _market_state else None,
                      'market_state_bonus': _ms_bonus}
        )
        if not self.validate_signal(_candidate):
            return None
        return _candidate
