"""
TitanBot Pro — Smart Money Concepts Strategy
=============================================
Detects institutional price action setups using SMC methodology:
  - Liquidity Sweep + Reversal (stop hunt detection)
  - Fair Value Gap (FVG / imbalance zones)
  - Order Block (last candle before impulse)
  - Break of Structure (BOS with momentum)

Killzone bonus applied during high-probability trading sessions.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.loader import cfg
from config.constants import Penalties, STRATEGY_VALID_REGIMES
from strategies.base import BaseStrategy, SignalResult, SignalDirection, cfg_min_rr
from utils.formatting import fmt_price
from utils.risk_params import rp, compute_vol_percentile

logger = logging.getLogger(__name__)


class SmartMoneyConcepts(BaseStrategy):

    name = "SmartMoneyConcepts"
    description = "Liquidity sweeps, FVG, order blocks, and BOS detection"

    # SMC works in all regimes — VOLATILE_PANIC is allowed here because
    # regime_thresholds already gates it to only SMC/Wyckoff/FundingRateArb
    # at max 2 signals/hr and 90 min_confidence.  Previously VOLATILE_PANIC
    # was missing, so the strategy ran fully then got blocked downstream
    # instead of short-circuiting early via the VALID_REGIMES check.
    VALID_REGIMES = STRATEGY_VALID_REGIMES["SmartMoneyConcepts"]

    # Direction-aware regime confidence: with-trend gets a boost, counter-trend
    # gets a penalty.  Previously flat +8 for both BULL/BEAR regardless of
    # direction — that inflated counter-trend LONGs in BEAR_TREND (production
    # evidence: SMC firing 2 LONGs, 0 SHORTs in BEAR_TREND, all killed).
    _REGIME_CONF_WITH_TREND = {
        "BULL_TREND":  +8,
        "BEAR_TREND":  +8,
        "VOLATILE":    +3,
        "VOLATILE_PANIC": +3,
        "CHOPPY":      -10,
        "UNKNOWN":     0,
    }
    _REGIME_CONF_COUNTER_TREND = {
        "BULL_TREND":  -12,   # SHORT in bull — high risk
        "BEAR_TREND":  -12,   # LONG in bear — high risk
        "VOLATILE":    +3,    # No trend to counter
        "VOLATILE_PANIC": +3,
        "CHOPPY":      -10,
        "UNKNOWN":     0,
    }
    _COUNTER_TREND_REVERSAL_SETUPS = {"LiquiditySweep"}

    def __init__(self):
        super().__init__()
        self._cfg = cfg.strategies.smc

    @classmethod
    def _allows_counter_trend_setup(
        cls,
        setup_type: str,
        has_choch: bool,
        choch_direction: str,
        direction: str,
    ) -> bool:
        """Allow only genuine reversal-style counter-trend SMC setups upstream."""
        if setup_type in cls._COUNTER_TREND_REVERSAL_SETUPS:
            return True
        expected_choch = "BULLISH" if direction == "LONG" else "BEARISH"
        return has_choch and choch_direction == expected_choch

    @classmethod
    def _local_poc(
        cls,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        lookback: int = 50,
        bins: int = 24,
    ) -> Optional[float]:
        """Volume-weighted Point of Control over the recent ``lookback`` bars.

        Phase-3 audit helper.  Stdlib + numpy only — safe under
        ``tests/conftest.py`` module mocks.  Returns ``None`` when the data
        is too sparse to form a meaningful histogram.

        Typical price (H+L+C)/3 is used as the per-bar price stand-in, which
        is both textbook-VP and cheap (no intrabar interpolation needed).
        """
        try:
            n = min(int(lookback), len(closes), len(volumes), len(highs), len(lows))
            if n < 15:  # too few bars — POC would be noise
                return None
            hi = np.asarray(highs[-n:], dtype=float)
            lo = np.asarray(lows[-n:], dtype=float)
            cl = np.asarray(closes[-n:], dtype=float)
            vol = np.asarray(volumes[-n:], dtype=float)
            typ = (hi + lo + cl) / 3.0
            p_max = float(np.max(typ))
            p_min = float(np.min(typ))
            if not (p_max > p_min) or np.sum(vol) <= 0.0:
                return None
            # np.histogram already bins and weights — one allocation, no loop.
            hist, edges = np.histogram(typ, bins=int(bins), range=(p_min, p_max), weights=vol)
            if not np.any(hist > 0):
                return None
            poc_bin = int(np.argmax(hist))
            return float((edges[poc_bin] + edges[poc_bin + 1]) / 2.0)
        except Exception:
            return None

    async def analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        try:
            return await self._analyze(symbol, ohlcv_dict)
        except Exception as e:
            self._record_analyze_error(self.name, e, symbol)
            return None

    async def _analyze(self, symbol: str, ohlcv_dict: Dict) -> Optional[SignalResult]:
        # ── Regime gate ───────────────────────────────────────────────────
        try:
            from analyzers.regime import regime_analyzer
            regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception as _rexc:
            # Audit P1: don't silently fail — log the first occurrence so an
            # analyzer breakage is visible instead of quietly zero'ing the
            # regime bonus for days.
            if not getattr(self, "_regime_import_warned", False):
                logger.warning("SMC: regime analyzer unavailable (%s) — defaulting to UNKNOWN", _rexc)
                self._regime_import_warned = True
            regime = "UNKNOWN"

        regime_bonus = 0  # deferred — computed after direction is known

        # ── Multi-Timeframe Analysis: 4h structure → 1h zones → 15m trigger ──
        _htf_bias = self.mtf_get_bias(ohlcv_dict, tf="4h")
        _mtf_zone = None
        _mtf_trigger = {"triggered": False, "trigger_type": "none", "quality": 0.0}
        _mtf_bonus = 0

        if _htf_bias["bias"] != "NEUTRAL" and _htf_bias["confidence"] > 55:
            _mtf_zone = self.mtf_find_zone(ohlcv_dict, tf="1h", bias=_htf_bias["bias"])
            if _mtf_zone:
                _trigger_dir = "LONG" if _htf_bias["bias"] == "BULLISH" else "SHORT"
                _mtf_trigger = self.mtf_check_trigger(
                    ohlcv_dict, tf="15m", direction=_trigger_dir, zone=_mtf_zone
                )
                if _mtf_trigger["triggered"]:
                    _mtf_bonus = int(_mtf_trigger["quality"] * 12)  # Up to +12

        # Primary timeframe for structure analysis
        tf = "4h"
        for candidate_tf in ("4h", "1h", "15m"):
            if candidate_tf in ohlcv_dict and len(ohlcv_dict[candidate_tf]) >= 60:
                tf = candidate_tf
                break
        else:
            return None

        if tf not in ohlcv_dict or len(ohlcv_dict[tf]) < 60:
            return None

        ohlcv = ohlcv_dict[tf]
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

        opens   = df["open"].values
        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values
        volumes = df["volume"].values

        atr = self.calculate_atr(highs, lows, closes, period=14)
        if atr == 0:
            return None

        confidence_base = getattr(self._cfg, "confidence_base", 75)
        lookback_swings = getattr(getattr(self._cfg, "structure", self._cfg), "lookback_swings", 20)
        sweep_lookback  = getattr(getattr(self._cfg, "sweeps", self._cfg), "lookback", 30)
        # Maximum bars between sweep and BOS for the sequencing bonus to fire at full value.
        # The bonus is scaled by recency: +4 at 1 bar, tapering to 0 at bos_sweep_max_lag+1.
        # Default 5: tight cause→effect on 15m/1h; still reasonable on 4h (= 20h window).
        bos_sweep_max_lag = getattr(getattr(self._cfg, "sweeps", self._cfg), "bos_sweep_max_lag", 5)
        # Define OB config at top-level so it's in scope for Priority 3, Priority 4,
        # and the stacked-confluence scan (which may run before Priority 3 if
        # a higher-priority setup already set direction).
        ob_min_impulse  = getattr(getattr(self._cfg, "order_blocks", self._cfg), "min_impulse_pct", 0.5)
        ob_max_age      = getattr(getattr(self._cfg, "order_blocks", self._cfg), "max_ob_age_bars", 20)

        # ── Killzone bonus ────────────────────────────────────────────────
        # FIX AUDIT-5: Use the last candle's timestamp instead of wall-clock
        # so backtests get deterministic killzone bonuses based on when the
        # candle formed, not when the backtest runs.
        _last_ts = df["ts"].iloc[-1]
        try:
            _candle_dt = datetime.fromtimestamp(float(_last_ts) / 1000, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            logger.warning(f"SMC {symbol}: failed to parse candle timestamp {_last_ts!r}, using wall-clock fallback")
            _candle_dt = datetime.now(timezone.utc)
        current_hour = _candle_dt.hour
        # SMC-7 FIX: The old 14-hour set covered more than half the day — not a killzone.
        # Read window config and expand to hour sets (end-exclusive), capping at ≤10 h total.
        # Classic ICT killzones: Asia open, London open, NY open, London close.
        _kz_cfg = getattr(self._cfg, "killzones", None)
        _killzone_bonus_val = int(getattr(_kz_cfg, "killzone_bonus", 10)) if _kz_cfg else 10
        _killzone_hours: set = set()
        if _kz_cfg:
            for _wname in ("asia_open", "london_open", "ny_open", "london_close"):
                _win = getattr(_kz_cfg, _wname, None)
                if _win and len(_win) >= 2:
                    _killzone_hours.update(range(int(_win[0]), int(_win[1])))
        if not _killzone_hours:
            # Fallback: 4 classic 2-hour ICT killzone windows (~8h total)
            _killzone_hours = {2, 3, 7, 8, 13, 14, 15, 16}
        killzone_bonus = _killzone_bonus_val if current_hour in _killzone_hours else 0

        # ── Identify swing highs/lows ─────────────────────────────────────
        swing_highs: List[float] = []
        swing_lows:  List[float] = []
        swing_high_points: List[Tuple[int, float]] = []
        swing_low_points: List[Tuple[int, float]] = []
        lb = min(lookback_swings, len(highs) - 3)
        # Audit P1: start at i=2 so idx=-3 and idx+1=-2 (last closed bar),
        # avoiding a right-neighbor comparison against the currently-forming
        # bar highs[-1]/lows[-1] (which flips as the bar develops).
        for i in range(2, lb):
            idx = -(i + 1)
            if highs[idx] > highs[idx - 1] and highs[idx] > highs[idx + 1]:
                swing_highs.append(highs[idx])
                swing_high_points.append((idx, highs[idx]))
            if lows[idx] < lows[idx - 1] and lows[idx] < lows[idx + 1]:
                swing_lows.append(lows[idx])
                swing_low_points.append((idx, lows[idx]))

        if not swing_highs or not swing_lows:
            return None

        key_swing_high = max(swing_highs[:5]) if swing_highs else highs[-20:-1].max()
        key_swing_low  = min(swing_lows[:5])  if swing_lows  else lows[-20:-1].min()
        # BOS SL FIX: for BOS stop-loss placement, use the NEAREST (highest) recent swing
        # low — the structural support just before the breakout.  min(swing_lows[:5]) gives
        # the deepest low, which produces excessively wide stops that kill RR on BOS entries.
        # We keep key_swing_low (deep) for liquidity sweep detection where we need the true
        # liquidity pool level, and add key_swing_low_near for BOS SL placement.
        key_swing_low_near  = max(swing_lows[:5])  if swing_lows  else lows[-20:-1].max()

        # SMC-1 FIX: BOS trigger should use the MOST RECENT swing high/low, not the
        # absolute highest/lowest of the last 5 swings.  Using min(swing_lows[:5]) means
        # price has to crash through ALL 5 swing lows before BOS fires — too extreme.
        # Use the most recently formed swing point (index 0 in the list, closest to now).
        bos_swing_high = swing_highs[0] if swing_highs else key_swing_high
        bos_swing_low  = swing_lows[0]  if swing_lows  else key_swing_low

        current_close = closes[-1]
        current_high  = highs[-1]
        current_low   = lows[-1]

        min_break_pct = getattr(getattr(self._cfg, "structure", self._cfg), "min_break_pct", 0.002)
        structure_trend = "UNKNOWN"
        has_choch = False
        choch_direction = ""
        # SMC-Q6 FIX: 2-point CHoCH (current vs prev) is noise — a single-pivot
        # false print fires a CHoCH.  Require 3 consecutive swing points that all
        # confirm the trend structure before declaring a structure shift.
        if len(swing_high_points) >= 3 and len(swing_low_points) >= 3:
            sh0, sh1, sh2 = swing_high_points[0][1], swing_high_points[1][1], swing_high_points[2][1]
            sl0, sl1, sl2 = swing_low_points[0][1],  swing_low_points[1][1],  swing_low_points[2][1]
            if sh0 > sh1 > sh2 and sl0 > sl1 > sl2:   # HH + HL pattern
                structure_trend = "BULLISH"
                if current_close < sl0 * (1.0 - min_break_pct):
                    has_choch = True
                    choch_direction = "BEARISH"
            elif sh0 < sh1 < sh2 and sl0 < sl1 < sl2:  # LH + LL pattern
                structure_trend = "BEARISH"
                if current_close > sh0 * (1.0 + min_break_pct):
                    has_choch = True
                    choch_direction = "BULLISH"
        elif len(swing_high_points) >= 2 and len(swing_low_points) >= 2:
            # Fallback when < 3 swing points: still detect trend, but no CHoCH
            _recent_high, _prev_high = swing_high_points[0][1], swing_high_points[1][1]
            _recent_low,  _prev_low  = swing_low_points[0][1],  swing_low_points[1][1]
            if _recent_high > _prev_high and _recent_low > _prev_low:
                structure_trend = "BULLISH"
            elif _recent_high < _prev_high and _recent_low < _prev_low:
                structure_trend = "BEARISH"

        direction:  Optional[str] = None
        setup_type: str = ""
        confidence: float = confidence_base + regime_bonus + killzone_bonus + _mtf_bonus
        entry_ref:  float = current_close
        sl_level:   Optional[float] = None   # SMC-5 FIX: use None sentinel (0.0 is a valid SL for sub-cent tokens)
        confluence: List[str] = []
        raw_data:   Dict = {}

        # ─────────────────────────────────────────────────────────────────
        # Priority 1: Liquidity Sweep + Reversal
        # ─────────────────────────────────────────────────────────────────
        sweep_lows  = lows[-sweep_lookback:-1]
        sweep_highs = highs[-sweep_lookback:-1]
        min_wick_pct = getattr(getattr(self._cfg, "sweeps", self._cfg), "min_wick_pct", 0.003)

        # Bullish sweep: wick below key swing low then close above it.
        # SMC-Q1 FIX: also accept the 2-bar sweep pattern where the prev bar
        # wicked below (and closed below) while current bar reclaims above —
        # institutional sweeps commonly complete over 2 bars.
        _single_sweep_bull = (
            current_low < key_swing_low
            and current_close > key_swing_low
            and (key_swing_low - current_low) / key_swing_low >= min_wick_pct
        )
        _two_bar_sweep_bull = (
            len(lows) >= 2
            and lows[-2] < key_swing_low
            and closes[-2] < key_swing_low    # prev bar DID NOT reclaim — true sweep-and-hold
            and current_close > key_swing_low
            and (key_swing_low - lows[-2]) / key_swing_low >= min_wick_pct
        )
        # Bearish sweep: wick above key swing high then close below it.
        _single_sweep_bear = (
            current_high > key_swing_high
            and current_close < key_swing_high
            and (current_high - key_swing_high) / key_swing_high >= min_wick_pct
        )
        _two_bar_sweep_bear = (
            len(highs) >= 2
            and highs[-2] > key_swing_high
            and closes[-2] > key_swing_high   # prev bar DID NOT reclaim below — true sweep
            and current_close < key_swing_high
            and (highs[-2] - key_swing_high) / key_swing_high >= min_wick_pct
        )

        if _single_sweep_bull or _two_bar_sweep_bull:
            _sweep_bar_low = current_low if _single_sweep_bull else lows[-2]
            direction  = "LONG"
            setup_type = "LiquiditySweep"
            confidence += 15
            sl_level   = _sweep_bar_low - atr * 0.3
            entry_ref  = current_close
            _sweep_desc = "single-bar" if _single_sweep_bull else "2-bar"
            confluence.append(f"✅ Bullish liquidity sweep below {fmt_price(key_swing_low)} — stop hunt confirmed ({_sweep_desc})")
            confluence.append(f"✅ Wick below swing low by {fmt_price(key_swing_low - _sweep_bar_low)}")
            raw_data["sweep_low"] = _sweep_bar_low
            raw_data["sweep_level"] = key_swing_low
            # Feature 4: Sweep strength — deeper institutional sweeps carry more weight.
            # A sweep that drives > 1 ATR below the liquidity level shows real intent.
            _sweep_depth = (key_swing_low - _sweep_bar_low) / atr
            if _sweep_depth > 1.0:
                confidence += 4
                confluence.append(f"💪 Deep sweep: {_sweep_depth:.2f}× ATR below level (+4)")
            raw_data["sweep_depth"] = round(_sweep_depth, 3)

        elif _single_sweep_bear or _two_bar_sweep_bear:
            _sweep_bar_high = current_high if _single_sweep_bear else highs[-2]
            direction  = "SHORT"
            setup_type = "LiquiditySweep"
            confidence += 15
            sl_level   = _sweep_bar_high + atr * 0.3
            entry_ref  = current_close
            _sweep_desc = "single-bar" if _single_sweep_bear else "2-bar"
            confluence.append(f"✅ Bearish liquidity sweep above {fmt_price(key_swing_high)} — stop hunt confirmed ({_sweep_desc})")
            confluence.append(f"✅ Wick above swing high by {fmt_price(_sweep_bar_high - key_swing_high)}")
            raw_data["sweep_high"] = _sweep_bar_high
            raw_data["sweep_level"] = key_swing_high
            # Feature 4: Sweep strength — deeper wicks indicate stronger institutional commitment.
            _sweep_depth = (_sweep_bar_high - key_swing_high) / atr
            if _sweep_depth > 1.0:
                confidence += 4
                confluence.append(f"💪 Deep sweep: {_sweep_depth:.2f}× ATR above level (+4)")
            raw_data["sweep_depth"] = round(_sweep_depth, 3)

        # ─────────────────────────────────────────────────────────────────
        # Priority 2: Fair Value Gap
        # ─────────────────────────────────────────────────────────────────
        # SMC-2 FIX: The original code used highs[-3]/lows[-1] (c1 closed, c3 LIVE).
        # Pattern detection on a live (incomplete) candle is unreliable — lows[-1]
        # keeps moving until the candle closes.  Use fully-closed candles only:
        # c1=[-4], c2=[-3], c3=[-2]; current bar [-1] is the potential retest.
        if direction is None and len(closes) >= 4:
            fvg_min_pct = getattr(getattr(self._cfg, "fvg", self._cfg), "min_size_pct", 0.002)
            c1_high, c3_low   = highs[-4], lows[-2]    # bullish FVG (all closed)
            c1_low,  c3_high  = lows[-4],  highs[-2]   # bearish FVG (all closed)

            # Bullish FVG: candle-1 high < candle-3 low (gap between c1 top and c3 bottom)
            if c1_high < c3_low and current_close > c1_high:
                fvg_mid = (c1_high + c3_low) / 2
                fvg_size = c3_low - c1_high
                # SMC-Q2 FIX: require price to be INSIDE the gap (not just approaching).
                # Old check: c1_high - atr*0.3 — allowed entry while price is still below
                # the gap bottom, which is outside the gap entirely.
                # New check: c1_high <= current_close <= fvg_mid + 0.3 ATR
                _in_gap = c1_high <= current_close <= fvg_mid + atr * 0.3
                if fvg_size / c1_high >= fvg_min_pct and _in_gap:
                    direction  = "LONG"
                    setup_type = "FairValueGap"
                    entry_ref  = fvg_mid
                    sl_level   = c1_high - atr * 0.3
                    confluence.append(f"✅ Bullish FVG: {fmt_price(c1_high)} → {fmt_price(c3_low)}")
                    confluence.append(f"   FVG midpoint: {fmt_price(fvg_mid)} | Price returned to gap")
                    raw_data["fvg_low"] = c1_high
                    raw_data["fvg_high"] = c3_low
                    # Feature 2: FVG fill depth — how deeply price has entered the gap.
                    # A shallow dip just past the gap edge is a weak retest; a deep fill
                    # (past the midpoint) shows real demand/supply absorption.
                    _fvg_gap = fvg_mid - c1_high
                    _fill_ratio = (current_close - c1_high) / _fvg_gap if _fvg_gap > 0 else 0.5
                    if _fill_ratio < 0.3:
                        confidence -= 3
                        confluence.append(f"⚠️ Shallow FVG fill ({_fill_ratio:.0%}) — weak retest (-3)")
                    elif _fill_ratio >= 0.6:
                        confidence += 3
                        confluence.append(f"💪 Deep FVG fill ({_fill_ratio:.0%}) — strong retest (+3)")
                    raw_data["fvg_fill_ratio"] = round(_fill_ratio, 3)

            # Bearish FVG: candle-1 low > candle-3 high
            elif c1_low > c3_high and current_close < c1_low:
                fvg_mid  = (c1_low + c3_high) / 2
                fvg_size = c1_low - c3_high
                # SMC-Q2 FIX: price must be inside the gap (not just approaching from below)
                _in_gap = fvg_mid - atr * 0.3 <= current_close <= c1_low
                if fvg_size / c1_low >= fvg_min_pct and _in_gap:
                    direction  = "SHORT"
                    setup_type = "FairValueGap"
                    entry_ref  = fvg_mid
                    sl_level   = c1_low + atr * 0.3
                    confluence.append(f"✅ Bearish FVG: {fmt_price(c3_high)} → {fmt_price(c1_low)}")
                    confluence.append(f"   FVG midpoint: {fmt_price(fvg_mid)} | Price returned to gap")
                    raw_data["fvg_high"] = c1_low
                    raw_data["fvg_low"] = c3_high
                    # Feature 2: FVG fill depth for bearish gap.
                    _fvg_gap = c1_low - fvg_mid
                    _fill_ratio = (c1_low - current_close) / _fvg_gap if _fvg_gap > 0 else 0.5
                    if _fill_ratio < 0.3:
                        confidence -= 3
                        confluence.append(f"⚠️ Shallow FVG fill ({_fill_ratio:.0%}) — weak retest (-3)")
                    elif _fill_ratio >= 0.6:
                        confidence += 3
                        confluence.append(f"💪 Deep FVG fill ({_fill_ratio:.0%}) — strong retest (+3)")
                    raw_data["fvg_fill_ratio"] = round(_fill_ratio, 3)

        # ─────────────────────────────────────────────────────────────────
        # Priority 3: Order Block
        # ─────────────────────────────────────────────────────────────────
        if direction is None:
            search_bars    = min(ob_max_age, len(closes) - 3)
            # Capture confidence before any OB-structural bonus/penalty so we can
            # apply time-decay only to the OB portion (killzone, MTF, regime are
            # earned independently and should not decay with OB age).
            _pre_ob_confidence = confidence
            _ob_decay_mult = 1.0  # sentinel: no decay unless a qualifying OB is found

            # SMC-3 FIX: original code divided impulse body by OB body.  A tiny doji OB
            # (0.05× ATR body) lets any noise impulse satisfy the ratio.  Divide by ATR
            # instead — impulse must represent a meaningful fraction of typical volatility.
            # SMC-4/Q4 FIX: instead of breaking on the first match (most recent OB), collect
            # all valid candidates and rank by impulse quality — the strongest institutional
            # signal wins over recency alone.
            bull_ob_candidates: List[Tuple[float, int, float, float]] = []  # (quality, i, ob_high, ob_low)
            bear_ob_candidates: List[Tuple[float, int, float, float]] = []

            for i in range(2, search_bars):
                # Bullish OB: last bearish candle before a bullish impulse
                _bull_ob_body = opens[-i] - closes[-i]
                if (_bull_ob_body > 0
                        and closes[-i + 1] > opens[-i + 1]
                        and (closes[-i + 1] - opens[-i + 1]) / atr >= ob_min_impulse):
                    ob_high = opens[-i]
                    ob_low  = closes[-i]
                    if ob_low <= current_close <= ob_high + atr * 0.5:
                        _quality = (closes[-i + 1] - opens[-i + 1]) / atr
                        bull_ob_candidates.append((_quality, i, ob_high, ob_low))

                # Bearish OB: last bullish candle before a bearish impulse
                _bear_ob_body = closes[-i] - opens[-i]
                if (_bear_ob_body > 0
                        and closes[-i + 1] < opens[-i + 1]
                        and (opens[-i + 1] - closes[-i + 1]) / atr >= ob_min_impulse):
                    ob_high = closes[-i]
                    ob_low  = opens[-i]
                    if ob_low - atr * 0.5 <= current_close <= ob_high:
                        _quality = (opens[-i + 1] - closes[-i + 1]) / atr
                        bear_ob_candidates.append((_quality, i, ob_high, ob_low))

            if bull_ob_candidates:
                bull_ob_candidates.sort(reverse=True)   # best quality first
                _, _oi, ob_high, ob_low = bull_ob_candidates[0]
                direction  = "LONG"
                setup_type = "OrderBlock"
                entry_ref  = (ob_high + ob_low) / 2
                sl_level   = ob_low - atr * 0.3
                confluence.append(f"✅ Bullish OB at {fmt_price(ob_low)}-{fmt_price(ob_high)} ({_oi} bars ago)")
                raw_data["ob_low"] = ob_low
                raw_data["ob_high"] = ob_high
                # Feature 1/SMC-Q5 (hybrid + recency): OB mitigation tracking.
                # Close inside OB = strong absorption (1.0); wick touch = partial (0.5).
                # Each touch is further weighted by recency: a tap 2 bars ago matters far
                # more than one 18 bars ago.  weight = max(0, 1 - bars_since / ob_max_age).
                # Threshold ≥ 2.0 calibrated so 2 recent closes OR ~4 recent wicks trigger.
                _ob_score = sum(
                    (1.0 if ob_low <= closes[-j] <= ob_high
                     else (0.5 if lows[-j] <= ob_high and highs[-j] >= ob_low else 0.0))
                    * max(0.0, 1.0 - j / ob_max_age)
                    for j in range(1, _oi)
                )
                if _ob_score >= 2.0:
                    confidence -= 5
                    confluence.append(f"⚠️ OB mitigated (score {_ob_score:.1f}) — diminished freshness (-5)")
                raw_data["ob_mitigation_score"] = round(_ob_score, 2)
                # Feature 6 (scoped): OB time decay applied only to the OB structural component.
                # External signals (killzone, MTF, regime) are earned independently — they should
                # not decay because the OB is old.  We track the pre-OB confidence and apply
                # decay to just the structural portion (confidence gained after OB priority fires).
                _age_factor = max(0.0, 1.0 - (_oi / ob_max_age))
                _ob_decay_mult = 0.8 + 0.2 * _age_factor
                raw_data["ob_age_factor"] = round(_ob_decay_mult, 3)
                # Phase-3 audit: OB-near-POC confluence — a bullish OB that sits
                # on or near the recent volume Point of Control is high-interest
                # institutional zone (price revisits where prior volume printed).
                _poc = self._local_poc(highs, lows, closes, volumes)
                if _poc is not None and atr > 0:
                    _overlap = ob_low <= _poc <= ob_high
                    _near    = abs(_poc - ((ob_low + ob_high) / 2.0)) <= atr * 0.5
                    if _overlap or _near:
                        confidence += 4
                        confluence.append(
                            f"✅ OB overlaps volume POC {fmt_price(_poc)} — institutional zone (+4)"
                        )
                        raw_data["ob_poc_overlap"] = True
                        raw_data["ob_poc_price"] = round(_poc, 8)
                    else:
                        raw_data["ob_poc_overlap"] = False
                        raw_data["ob_poc_price"] = round(_poc, 8)
            elif bear_ob_candidates:
                bear_ob_candidates.sort(reverse=True)
                _, _oi, ob_high, ob_low = bear_ob_candidates[0]
                direction  = "SHORT"
                setup_type = "OrderBlock"
                entry_ref  = (ob_high + ob_low) / 2
                sl_level   = ob_high + atr * 0.3
                confluence.append(f"✅ Bearish OB at {fmt_price(ob_low)}-{fmt_price(ob_high)} ({_oi} bars ago)")
                raw_data["ob_low"] = ob_low
                raw_data["ob_high"] = ob_high
                # Feature 1/SMC-Q5 (hybrid + recency): OB mitigation tracking (bearish OB).
                _ob_score = sum(
                    (1.0 if ob_low <= closes[-j] <= ob_high
                     else (0.5 if lows[-j] <= ob_high and highs[-j] >= ob_low else 0.0))
                    * max(0.0, 1.0 - j / ob_max_age)
                    for j in range(1, _oi)
                )
                if _ob_score >= 2.0:
                    confidence -= 5
                    confluence.append(f"⚠️ OB mitigated (score {_ob_score:.1f}) — diminished freshness (-5)")
                raw_data["ob_mitigation_score"] = round(_ob_score, 2)
                # Feature 6 (scoped): OB time decay — structural component only.
                _age_factor = max(0.0, 1.0 - (_oi / ob_max_age))
                _ob_decay_mult = 0.8 + 0.2 * _age_factor
                raw_data["ob_age_factor"] = round(_ob_decay_mult, 3)
                # Phase-3 audit: bearish OB-near-POC confluence (mirror of bullish).
                _poc = self._local_poc(highs, lows, closes, volumes)
                if _poc is not None and atr > 0:
                    _overlap = ob_low <= _poc <= ob_high
                    _near    = abs(_poc - ((ob_low + ob_high) / 2.0)) <= atr * 0.5
                    if _overlap or _near:
                        confidence += 4
                        confluence.append(
                            f"✅ OB overlaps volume POC {fmt_price(_poc)} — institutional zone (+4)"
                        )
                        raw_data["ob_poc_overlap"] = True
                        raw_data["ob_poc_price"] = round(_poc, 8)
                    else:
                        raw_data["ob_poc_overlap"] = False
                        raw_data["ob_poc_price"] = round(_poc, 8)

            # Apply OB time decay to structural portion only: keep _pre_ob_confidence
            # intact and scale only the delta (the OB-specific contribution).
            if direction is not None and _ob_decay_mult < 1.0:
                _ob_structural_delta = confidence - _pre_ob_confidence
                confidence = _pre_ob_confidence + _ob_structural_delta * _ob_decay_mult

        # ─────────────────────────────────────────────────────────────────
        # Priority 4: Break of Structure (with pullback confirmation)
        # ─────────────────────────────────────────────────────────────────
        # Classic BOS enters at the break level. Improved BOS waits for a
        # pullback to the OB/FVG that caused the break, then enters on the
        # retest — dramatically higher win rate.
        if direction is None:
            # SMC-1 FIX: BOS trigger uses bos_swing_high/low (most recent swing),
            # not key_swing_high/low (absolute max/min of 5 swings).  The absolute
            # extreme requires price to crash through ALL prior swings, which fires
            # far too rarely and misses valid BOS setups at intermediate structure.
            if (current_close > bos_swing_high
                    and (current_close - bos_swing_high) / bos_swing_high >= min_break_pct):
                # Bullish BOS detected — look for pullback to OB/FVG below
                _pullback_entry = None
                _pullback_sl = None
                _pullback_note = ""

                # Check for bullish OB below the breakout level (pullback target).
                # SMC-3 FIX applied here too: use ATR denominator for impulse check.
                for i in range(2, min(ob_max_age, len(closes) - 3)):
                    _bull_ob_body = opens[-i] - closes[-i]
                    if (_bull_ob_body > 0
                            and closes[-i + 1] > opens[-i + 1]
                            and (closes[-i + 1] - opens[-i + 1]) / atr >= ob_min_impulse):
                        ob_high = opens[-i]
                        ob_low = closes[-i]
                        # OB must be between current price and the BOS level
                        if ob_low <= bos_swing_high and ob_high >= current_close - atr * 1.5:
                            # Price has pulled back to or near this OB
                            if current_close <= ob_high + atr * 0.5:
                                _pullback_entry = (ob_high + ob_low) / 2
                                _pullback_sl = ob_low - atr * 0.3
                                _pullback_note = f"✅ BOS pullback to OB at {fmt_price(ob_low)}-{fmt_price(ob_high)}"
                                break

                # Check for bullish FVG below breakout level.
                # SMC-2 FIX applied here: start at i=4 so c3 is always a closed candle.
                if _pullback_entry is None and len(closes) >= 6:
                    for i in range(4, min(15, len(closes) - 2)):
                        c1h, c3l = highs[-i], lows[-i + 2]
                        if c1h < c3l:  # bullish FVG exists (all closed candles)
                            fvg_mid = (c1h + c3l) / 2
                            if c1h <= bos_swing_high and current_close <= c3l + atr * 0.3:
                                _pullback_entry = fvg_mid
                                _pullback_sl = c1h - atr * 0.3
                                _pullback_note = f"✅ BOS pullback to FVG at {fmt_price(c1h)}-{fmt_price(c3l)}"
                                break

                if _pullback_entry is not None:
                    direction  = "LONG"
                    setup_type = "BreakOfStructure"
                    entry_ref  = _pullback_entry
                    sl_level   = _pullback_sl
                    confidence += 8  # Pullback confirmation bonus
                    confluence.append(f"✅ BOS: Price broke above swing high {fmt_price(bos_swing_high)}")
                    confluence.append(_pullback_note)
                    confluence.append("✅ Pullback confirmation — higher win-rate BOS entry")
                    raw_data["bos_level"] = bos_swing_high
                    raw_data["bos_pullback"] = True
                else:
                    # No pullback found — use standard BOS entry (lower confidence)
                    direction  = "LONG"
                    setup_type = "BreakOfStructure"
                    entry_ref  = current_close
                    sl_level   = key_swing_low_near - atr * 0.3
                    confidence -= 5  # Penalty for chasing the break without pullback
                    confluence.append(f"✅ BOS: Price broke above swing high {fmt_price(bos_swing_high)}")
                    confluence.append("⚠️ No pullback to OB/FVG — chase entry")
                    raw_data["bos_level"] = bos_swing_high
                    raw_data["bos_pullback"] = False

            elif (current_close < bos_swing_low
                    and (bos_swing_low - current_close) / bos_swing_low >= min_break_pct):
                # Bearish BOS — look for pullback to bearish OB/FVG above
                _pullback_entry = None
                _pullback_sl = None
                _pullback_note = ""

                # SMC-3 FIX applied: use ATR denominator for impulse check.
                for i in range(2, min(ob_max_age, len(closes) - 3)):
                    _bear_ob_body = closes[-i] - opens[-i]
                    if (_bear_ob_body > 0
                            and closes[-i + 1] < opens[-i + 1]
                            and (opens[-i + 1] - closes[-i + 1]) / atr >= ob_min_impulse):
                        ob_high = closes[-i]
                        ob_low = opens[-i]
                        if ob_high >= bos_swing_low and ob_low <= current_close + atr * 1.5:
                            if current_close >= ob_low - atr * 0.5:
                                _pullback_entry = (ob_high + ob_low) / 2
                                _pullback_sl = ob_high + atr * 0.3
                                _pullback_note = f"✅ BOS pullback to OB at {fmt_price(ob_low)}-{fmt_price(ob_high)}"
                                break

                # SMC-2 FIX applied: start at i=4 for closed-candle FVG search.
                if _pullback_entry is None and len(closes) >= 6:
                    for i in range(4, min(15, len(closes) - 2)):
                        c1l, c3h = lows[-i], highs[-i + 2]
                        if c1l > c3h:  # bearish FVG exists (all closed candles)
                            fvg_mid = (c1l + c3h) / 2
                            if c1l >= bos_swing_low and current_close >= c3h - atr * 0.3:
                                _pullback_entry = fvg_mid
                                _pullback_sl = c1l + atr * 0.3
                                _pullback_note = f"✅ BOS pullback to FVG at {fmt_price(c3h)}-{fmt_price(c1l)}"
                                break

                if _pullback_entry is not None:
                    direction  = "SHORT"
                    setup_type = "BreakOfStructure"
                    entry_ref  = _pullback_entry
                    sl_level   = _pullback_sl
                    confidence += 8
                    confluence.append(f"✅ BOS: Price broke below swing low {fmt_price(bos_swing_low)}")
                    confluence.append(_pullback_note)
                    confluence.append("✅ Pullback confirmation — higher win-rate BOS entry")
                    raw_data["bos_level"] = bos_swing_low
                    raw_data["bos_pullback"] = True
                else:
                    direction  = "SHORT"
                    setup_type = "BreakOfStructure"
                    entry_ref  = current_close
                    sl_level   = key_swing_high + atr * 0.3
                    confidence -= 5
                    confluence.append(f"✅ BOS: Price broke below swing low {fmt_price(bos_swing_low)}")
                    confluence.append("⚠️ No pullback to OB/FVG — chase entry")
                    raw_data["bos_level"] = bos_swing_low
                    raw_data["bos_pullback"] = False

        if direction is None:
            return None

        # ── Feature 3: BOS strength scoring ──────────────────────────────
        # A barely-clearing BOS is noisy; a strong displacement (> 1.5× ATR
        # beyond the level) signals genuine institutional commitment.
        # Feature (R2-5): Overextension penalty — if price is already > 2.5× ATR
        # beyond the BOS level, chasing that displacement has poor expectancy.
        if setup_type == "BreakOfStructure":
            _bos_lv = raw_data.get(
                "bos_level",
                bos_swing_high if direction == "LONG" else bos_swing_low,
            )
            _bos_strength = abs(current_close - _bos_lv) / atr
            if _bos_strength > 2.5:
                confidence -= 4
                confluence.append(f"⚠️ BOS overextended: {_bos_strength:.2f}× ATR — chasing risk (-4)")
            elif _bos_strength > 1.5:
                confidence += 5
                confluence.append(f"💪 Strong BOS displacement: {_bos_strength:.2f}× ATR (+5)")
            elif _bos_strength < 0.5:
                confidence -= 3
                confluence.append(f"⚠️ Weak BOS displacement: {_bos_strength:.2f}× ATR (-3)")
            raw_data["bos_strength"] = round(_bos_strength, 3)

            # Feature (R2-6 / R3-2): Sweep→BOS sequencing bonus (recency-scaled).
            # Institutional flow: sweep out stops → accumulate/distribute → break structure.
            # The bonus scales with how recently the sweep occurred — a sweep 1 bar before
            # the BOS is a tight cause→effect; one 20 bars ago is much weaker evidence.
            # Max bonus: +4 at k=1. Scales linearly to 0 at k = bos_sweep_max_lag + 1.
            _bos_sweep_lag = None
            if direction == "LONG":
                for _k in range(1, min(sweep_lookback, len(lows) - 1)):
                    if (lows[-_k] < key_swing_low
                            and (key_swing_low - lows[-_k]) / key_swing_low >= min_wick_pct):
                        _bos_sweep_lag = _k
                        break
            else:
                for _k in range(1, min(sweep_lookback, len(highs) - 1)):
                    if (highs[-_k] > key_swing_high
                            and (highs[-_k] - key_swing_high) / key_swing_high >= min_wick_pct):
                        _bos_sweep_lag = _k
                        break
            _bos_sweep_seq = _bos_sweep_lag is not None
            if _bos_sweep_seq:
                _seq_scale = max(0.0, 1.0 - (_bos_sweep_lag - 1) / bos_sweep_max_lag)
                _seq_bonus = round(4 * _seq_scale)
                if _seq_bonus > 0:
                    confidence += _seq_bonus
                    confluence.append(
                        f"🔗 Sweep→BOS sequence ({_bos_sweep_lag}b ago): "
                        f"stop hunt preceded displacement (+{_seq_bonus})"
                    )
            raw_data["bos_sweep_sequence"] = _bos_sweep_seq
            raw_data["bos_sweep_lag"] = _bos_sweep_lag

        # ── Feature 5 (weighted): Stacked confluence bonus ───────────────
        # Structure-confirming factors (BOS, sweep) score 2 points each;
        # zone-based factors (FVG, OB) score 1 point each.  Threshold ≥ 4
        # requires at minimum two structure factors OR both structure + one zone.
        # This is equivalent to the old ≥3/4 binary count but correctly weights
        # the difference between institutional displacement and passive zones.
        _cf_bos = (
            (direction == "LONG" and current_close > bos_swing_high)
            or (direction == "SHORT" and current_close < bos_swing_low)
        )
        _cf_sweep = (
            (direction == "LONG" and (_single_sweep_bull or _two_bar_sweep_bull))
            or (direction == "SHORT" and (_single_sweep_bear or _two_bar_sweep_bear))
        )
        _cf_fvg = False
        _cf_fvg_high: float = 0.0
        _cf_fvg_low:  float = 0.0
        if len(closes) >= 4:
            if direction == "LONG" and highs[-4] < lows[-2]:
                _cf_fvg = True
                _cf_fvg_low  = highs[-4]   # gap bottom
                _cf_fvg_high = lows[-2]    # gap top
            elif direction == "SHORT" and lows[-4] > highs[-2]:
                _cf_fvg = True
                _cf_fvg_high = lows[-4]    # gap top
                _cf_fvg_low  = highs[-2]   # gap bottom
        # OB confluence: already confirmed if setup_type == OrderBlock, otherwise
        # run a lightweight scan to check whether a valid OB is near current price.
        _cf_ob = raw_data.get("ob_low") is not None
        _cf_ob_high = raw_data.get("ob_high", 0.0)
        _cf_ob_low  = raw_data.get("ob_low", 0.0)
        if not _cf_ob:
            _ob_scan_end = min(ob_max_age, len(closes) - 3)
            for _sci in range(2, _ob_scan_end):
                if direction == "LONG":
                    _ob_body = opens[-_sci] - closes[-_sci]
                    if (_ob_body > 0
                            and closes[-_sci + 1] > opens[-_sci + 1]
                            and (closes[-_sci + 1] - opens[-_sci + 1]) / atr >= ob_min_impulse
                            and closes[-_sci] <= current_close <= opens[-_sci] + atr * 0.5):
                        _cf_ob = True
                        _cf_ob_high = opens[-_sci]
                        _cf_ob_low  = closes[-_sci]
                        break
                else:
                    _ob_body = closes[-_sci] - opens[-_sci]
                    if (_ob_body > 0
                            and closes[-_sci + 1] < opens[-_sci + 1]
                            and (opens[-_sci + 1] - closes[-_sci + 1]) / atr >= ob_min_impulse
                            and opens[-_sci] - atr * 0.5 <= current_close <= closes[-_sci]):
                        _cf_ob = True
                        _cf_ob_high = closes[-_sci]
                        _cf_ob_low  = opens[-_sci]
                        break

        # Feature (R3-3): FVG + OB synergy — when the OB zone overlaps the FVG gap,
        # the combined POI (Point of Interest) is significantly more reliable than
        # either zone alone.  Overlap defined as the two ranges sharing any area.
        _fvg_ob_synergy = False
        if _cf_fvg and _cf_ob and _cf_ob_high > 0 and _cf_fvg_high > 0:
            if _cf_ob_low <= _cf_fvg_high and _cf_ob_high >= _cf_fvg_low:
                _fvg_ob_synergy = True
                confidence += 4
                confluence.append("🔗 OB inside FVG — high-probability POI confluence (+4)")
        raw_data["fvg_ob_synergy"] = _fvg_ob_synergy

        _smc_cf_score = (
            (2 if _cf_bos else 0)
            + (2 if _cf_sweep else 0)
            + (1 if _cf_fvg else 0)
            + (1 if _cf_ob else 0)
        )
        if _smc_cf_score >= 4:
            confidence += 6
            confluence.append(f"🔗 Stacked SMC confluence: score {_smc_cf_score}/6 (+6)")
        raw_data["smc_cf_score"] = _smc_cf_score

        # ── Direction-aware regime confidence ─────────────────────────────
        # With-trend (LONG in BULL, SHORT in BEAR) gets +8;
        # Counter-trend (LONG in BEAR, SHORT in BULL) gets -12.
        _is_with_trend = (
            (direction == "LONG" and regime == "BULL_TREND") or
            (direction == "SHORT" and regime == "BEAR_TREND")
        )
        if _is_with_trend or regime not in ("BULL_TREND", "BEAR_TREND"):
            regime_bonus = self._REGIME_CONF_WITH_TREND.get(regime, 0)
        else:
            if not self._allows_counter_trend_setup(setup_type, has_choch, choch_direction, direction):
                return None
            regime_bonus = self._REGIME_CONF_COUNTER_TREND.get(regime, 0)
        confidence += regime_bonus

        if sl_level is None:
            return None

        # ── Build entry zone around entry_ref ─────────────────────────────
        vp = compute_vol_percentile(highs, lows, closes)
        if direction == "LONG":
            entry_low  = entry_ref - atr * rp.entry_zone_tight
            entry_high = entry_ref + atr * rp.entry_zone_atr
            stop_loss  = min(sl_level, entry_low - atr * 0.1)
            risk       = entry_low - stop_loss
            # FIX: TPs were ATR-proportional from entry_high, but aggregator
            # verifies R/R from entry_mid — mismatch inflated R/R to 9-13R.
            # Now risk-proportional from entry_ref: tp_mult=2.8 → exactly 2.8R.
            risk_dist  = entry_ref - stop_loss
            tp1_calc   = entry_ref + risk_dist * rp.volatility_scaled_tp1(tf, vp)
            # Anchor TP1 to nearest swing high between entry and calculated TP1.
            _dz = atr * Penalties.TP1_SWING_DEAD_ZONE_ATR
            _n  = Penalties.TP1_MAX_SWING_LEVELS
            _tp1_resist = [sh for sh in swing_highs[:_n]
                          if entry_high + _dz < sh < tp1_calc]
            tp1        = min(_tp1_resist) if _tp1_resist else tp1_calc
            tp2        = entry_ref + risk_dist * rp.volatility_scaled_tp2(tf, vp)
            tp3        = entry_ref + risk_dist * rp.volatility_scaled_tp3(tf, vp)
        else:
            entry_high = entry_ref + atr * rp.entry_zone_tight
            entry_low  = entry_ref - atr * rp.entry_zone_atr
            stop_loss  = max(sl_level, entry_high + atr * 0.1)
            risk       = stop_loss - entry_high
            risk_dist  = stop_loss - entry_ref
            tp1_calc   = entry_ref - risk_dist * rp.volatility_scaled_tp1(tf, vp)
            # Anchor TP1 to nearest swing low between entry and calculated TP1.
            _dz = atr * Penalties.TP1_SWING_DEAD_ZONE_ATR
            _n  = Penalties.TP1_MAX_SWING_LEVELS
            _tp1_support = [sl for sl in swing_lows[:_n]
                           if tp1_calc < sl < entry_low - _dz]
            tp1        = max(_tp1_support) if _tp1_support else tp1_calc
            tp2        = entry_ref - risk_dist * rp.volatility_scaled_tp2(tf, vp)
            tp3        = entry_ref - risk_dist * rp.volatility_scaled_tp3(tf, vp)

        # X4: enforce tp1 < tp2 < tp3 ordering (tight volatility_scaled values can collapse)
        if direction == "LONG":
            tp2 = max(tp2, tp1 + atr * 0.1)
            tp3 = max(tp3, tp2 + atr * 0.1)
        else:
            tp2 = min(tp2, tp1 - atr * 0.1)
            tp3 = min(tp3, tp2 - atr * 0.1)

        if risk <= 0:
            return None

        rr_ratio = self.calculate_effective_rr(
            direction=direction,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            tp2=tp2,
        )

        # Confluence additions
        confluence.append(f"📊 Setup: {setup_type} | TF: {tf}")
        if has_choch:
            confluence.append(f"🧭 CHoCH: {choch_direction.title()} structure shift detected")
        if killzone_bonus:
            confluence.append(f"⏰ Killzone active (UTC {current_hour}h): +{killzone_bonus} confidence")
        confluence.append(f"📈 Regime: {regime} ({'+' if regime_bonus >= 0 else ''}{regime_bonus})")
        if _mtf_bonus > 0:
            confluence.append(
                f"🔄 MTF confirmed: 4h {_htf_bias['bias']} → "
                f"1h {_mtf_zone['zone_type'] if _mtf_zone else 'N/A'} zone → "
                f"15m {_mtf_trigger['trigger_type']} (+{_mtf_bonus})"
            )
        elif _htf_bias["bias"] != "NEUTRAL":
            confluence.append(f"🔄 HTF bias: {_htf_bias['bias']} (ADX {_htf_bias['adx']:.1f})")
        confluence.append(f"🎯 R:R {rr_ratio:.2f} | ATR: {fmt_price(atr)}")

        raw_data.update({
            "setup_type": setup_type,
            "atr": atr,
            "regime": regime,
            "key_swing_high": key_swing_high,
            "key_swing_low": key_swing_low,
            "key_swing_low_near": key_swing_low_near,
            "structure_trend": structure_trend,
            "has_choch": has_choch,
            "choch_direction": choch_direction,
            "has_bos": setup_type == "BreakOfStructure",
            "has_ob": setup_type == "OrderBlock",
            "has_fvg": setup_type == "FairValueGap",
            "has_sweep": setup_type == "LiquiditySweep",
            "mtf_htf_bias": _htf_bias["bias"],
            "mtf_zone": _mtf_zone["zone_type"] if _mtf_zone else None,
            "mtf_trigger": _mtf_trigger["trigger_type"],
            "mtf_bonus": _mtf_bonus,
        })

        confidence = min(95, max(40, confidence))

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
            setup_class="swing",
            timeframe=tf,
            analysis_timeframes=["4h", "1h", "15m"],
            confluence=confluence,
            raw_data=raw_data,
            regime=regime,
        )
        if not self.validate_signal(candidate):
            return None
        return candidate
SMCStrategy = SmartMoneyConcepts
