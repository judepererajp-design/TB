"""
TitanBot Pro — Institutional Breakout Strategy
===============================================
Detects volume-backed breakouts from consolidation using:
  - Donchian Channel (20-period high/low)
  - ADX trend strength filter
  - Volume confirmation (1.8x average minimum)
  - False breakout filter (retest logic)
  - ATR-based targets
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

    # Breakout only fires in trending or volatile regimes.
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
            if regime not in self.VALID_REGIMES:
                return None   # Breakouts are noise in chop — skip
            # Confidence bonus in confirmed trend; penalty in volatile/mixed
            _regime_bonus = 8 if "TREND" in regime else 0
            _regime_dir_constraint = self._REGIME_PREFERRED_DIR.get(regime)
        except Exception:
            _regime_bonus = 0
            _regime_dir_constraint = None
            regime = "UNKNOWN"

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
        volumes = df['volume'].values

        lookback = getattr(self._cfg, 'donchian_period', 20)
        min_adx = getattr(self._cfg, 'min_adx', 28)
        vol_mult = getattr(self._cfg, 'volume_confirmation_mult', 1.8)
        confidence_base = getattr(self._cfg, 'confidence_base', 78)

        # Guard: need at least lookback+1 bars for Donchian channel
        if lookback < 1:
            lookback = 20
        if len(closes) < lookback + 2:
            return None

        # Donchian channel (exclude current bar)
        channel_high = np.max(highs[-lookback-1:-1])
        channel_low  = np.min(lows[-lookback-1:-1])
        current_close = closes[-1]
        current_high  = highs[-1]
        current_low   = lows[-1]
        current_vol   = volumes[-1]
        avg_vol       = np.mean(volumes[-lookback:])

        # ADX filter
        adx = self.calculate_adx(highs, lows, closes, period=14)
        if adx < min_adx:
            return None

        # Volume spike check
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio < vol_mult:
            return None

        atr = self.calculate_atr(highs, lows, closes, period=14)
        if atr == 0:
            return None

        confluence = []
        direction = None
        confidence = confidence_base + _regime_bonus + _ms_bonus

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

        # DIRECTIONAL-GATE FIX: Block counter-trend breakouts in strong trends.
        # In BEAR_TREND, a breakout LONG is fighting weekly structure. In BULL_TREND,
        # a breakout SHORT is fighting weekly structure. Hard-block when weekly ADX
        # confirms strong trend; soft-penalise in weak trends.
        if _regime_dir_constraint and direction != _regime_dir_constraint:
            try:
                from analyzers.htf_guardrail import htf_guardrail as _brk_htf
                _brk_strong = _brk_htf._weekly_adx >= 30
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

        # STRUCTURAL-TP FIX (Breakout): Use measured move projection instead of ATR multiples.
        # Classic breakout target = channel_high + range_size (100% measured move).
        # TP2 = 100% measured move (most common institutional target).
        # TP1 = 50% of the measured move (first liquidity magnet).
        # TP3 = 150% extension (only reached in strong trends).
        # This anchors targets to the actual structure being broken, not to ATR drift.
        range_size = channel_high - channel_low  # Width of the broken consolidation

        if direction == "LONG":
            entry_low  = channel_high
            entry_high = channel_high + atr * rp.entry_zone_atr
            # V13: SL just below breakout level, not deep in old range
            # Use smaller of: ATR-based or 30% back into the range
            _atr_sl = channel_high - atr * sl_mult
            _range_sl = channel_high - range_size * 0.30
            stop_loss = max(_atr_sl, _range_sl)  # Whichever is CLOSER to breakout
            tp1        = channel_high + range_size * 0.50  # 50% measured move
            tp2        = channel_high + range_size * 1.00  # 100% measured move (main target)
            tp3        = channel_high + range_size * 1.50  # 150% extension
        else:
            entry_high = channel_low
            entry_low  = channel_low - atr * rp.entry_zone_atr  # Symmetric with LONG (was 0.5x)
            # V13: SL just above breakdown level
            _atr_sl = channel_low + atr * sl_mult
            _range_sl = channel_low + range_size * 0.30
            stop_loss = min(_atr_sl, _range_sl)
            tp1        = channel_low - range_size * 0.50
            tp2        = channel_low - range_size * 1.00
            # BUG-8 FIX: tp3 can go negative on low-price assets when range_size > channel_low/1.5.
            # Floor at a minimum positive price to prevent physically impossible targets.
            tp3        = max(channel_low * 0.01, channel_low - range_size * 1.50)

        risk = abs(channel_high - stop_loss) if direction == "LONG" else abs(channel_low - stop_loss)
        rr   = abs(tp2 - (channel_high if direction == "LONG" else channel_low)) / risk if risk > 0 else 0

        min_rr = cfg_min_rr("intraday")
        if rr < min_rr:
            return None

        if _market_state:
            confluence.append(f"🧠 Market State: {_market_state.value} ({'+' if _ms_bonus >= 0 else ''}{_ms_bonus})")

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
