"""
TitanBot Pro — Smart Entry Zone Refiner
==========================================
Replaces fixed ATR offsets with market-structure-aware entry zones.

For each strategy type:
  - SMC: keeps OB zone entries (already structure-based)
  - Breakout: refines entry to retest of breakout level
  - Reversal: uses wick rejection zone
  - All: aligns entry edges with nearest S/R, VWAP band, or volume POC

Also adds "confirmation mode" for non-SMC strategies:
  Wait 1-2 bars for price to confirm direction (higher low / lower high)
  before triggering, reducing premature entries.

Integration point: called from engine._scan_symbol() AFTER confluence
scoring but BEFORE signal aggregation.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.formatting import fmt_price
from strategies.base import direction_str, is_long

logger = logging.getLogger(__name__)


class EntryRefiner:
    """
    Refines signal entry zones using structural levels.
    Dramatically reduces premature entries (the #2 source of losses).
    """

    def refine(
        self,
        signal,
        ohlcv_dict: Dict,
        strategy_name: str,
    ):
        """
        Adjust signal.entry_low and signal.entry_high based on
        market structure instead of raw ATR offsets.

        Modifies the signal in-place and returns it.
        Returns None if the setup invalidates during refinement.
        """
        try:
            # FIX #25: Pick best available timeframe in priority order.
            # Previously used entry_timeframe ('15m' for intraday) which many
            # strategies don't populate in ohlcv_dict — they only request 1h/4h.
            # This caused ATR to fall back to entry_high*0.002 (non-ATR zone).
            _preferred = ['15m', '1h', '30m', '4h']
            _sig_tf = getattr(signal, 'entry_timeframe', None) or getattr(signal, 'timeframe', None)
            if _sig_tf and _sig_tf in ohlcv_dict:
                tf = _sig_tf
            else:
                tf = next((t for t in _preferred if t in ohlcv_dict),
                          list(ohlcv_dict.keys())[0] if ohlcv_dict else '1h')
            bars = ohlcv_dict.get(tf, [])
            if len(bars) < 20:
                bars = next((v for v in ohlcv_dict.values() if len(v) >= 20), [])
            if len(bars) < 20:
                return signal  # No usable data

            highs = np.array([float(b[2]) for b in bars])
            lows = np.array([float(b[3]) for b in bars])
            closes = np.array([float(b[4]) for b in bars])
            volumes = np.array([float(b[5]) for b in bars])
            current_price = closes[-1]

            # ── Calculate structural levels ───────────────────────
            support, resistance = self._find_sr_levels(highs, lows, closes)
            vwap, vwap_upper, vwap_lower = self._calculate_vwap_bands(
                highs, lows, closes, volumes
            )
            vpoc = self._volume_poc(closes, volumes)

            # ── Strategy-specific refinement ──────────────────────
            strategy_lower = strategy_name.lower()

            if 'smc' in strategy_lower or 'smart_money' in strategy_lower:
                # SMC already uses OB zones — just add alignment bonus
                signal = self._align_with_structure(
                    signal, support, resistance, vwap, vwap_upper, vwap_lower, vpoc
                )

            elif 'breakout' in strategy_lower:
                signal = self._refine_breakout_entry(
                    signal, support, resistance, highs, lows, closes, current_price
                )

            elif 'reversal' in strategy_lower:
                signal = self._refine_reversal_entry(
                    signal, lows, highs, closes, current_price
                )

            else:
                # Generic refinement: align with VWAP/POC
                signal = self._align_with_structure(
                    signal, support, resistance, vwap, vwap_upper, vwap_lower, vpoc
                )

            # ── Confirmation check (non-SMC) ──────────────────────
            if 'smc' not in strategy_lower:
                confirmed = self._check_confirmation(
                    signal, highs, lows, closes
                )
                if not confirmed:
                    # Reduce confidence instead of blocking
                    signal.confidence = max(30, signal.confidence - 8)
                    signal.confluence.append("⚠️ Awaiting bar confirmation")
                else:
                    signal.confidence = min(95, signal.confidence + 3)
                    signal.confluence.append("✅ Price action confirmed")

            # ── FIX SL-REPAIR: Validate SL is still outside entry zone ──
            # After refinement the entry zone may have shifted so that the
            # strategy's original SL now falls INSIDE the new zone, causing
            # the aggregator's geometry check to reject with:
            #   "SL above entry zone for LONG" / "SL below entry zone for SHORT"
            # Push the SL one ATR-width beyond the zone edge when this happens.
            signal = self._repair_sl_if_inverted(signal, highs, lows)

            return signal

        except Exception as e:
            logger.error(f"Entry refinement error: {e}")
            return signal  # Return unmodified on error

    # ── Breakout-specific refinement ──────────────────────────

    def _refine_breakout_entry(
        self, signal, support, resistance, highs, lows, closes, current_price
    ):
        """
        For breakouts: entry should be at the RETEST of the breakout level,
        not the breakout candle close. This gives better R:R and reduces
        false breakout losses.
        """
        direction = direction_str(signal)

        if direction == "LONG":
            # Breakout level = the resistance that was broken
            # Entry zone = retest of that level from above (now support)
            breakout_level = signal.entry_low  # Usually set to channel_high
            
            # Look for nearest support near the breakout level
            nearest_support = self._nearest_level(
                support, breakout_level, tolerance_pct=0.005
            )
            
            if nearest_support:
                # Tighten entry to retest zone
                signal.entry_low = nearest_support * 0.999
                signal.entry_high = breakout_level * 1.002
                signal.confluence.append(
                    f"📐 Entry refined to retest zone: "
                    f"{fmt_price(signal.entry_low)}–{fmt_price(signal.entry_high)}"
                )
            else:
                # Use breakout level as bottom of entry
                signal.entry_low = breakout_level * 0.998
                signal.entry_high = breakout_level * 1.005

        elif direction == "SHORT":
            breakout_level = signal.entry_high
            nearest_resistance = self._nearest_level(
                resistance, breakout_level, tolerance_pct=0.005
            )

            if nearest_resistance:
                signal.entry_high = nearest_resistance * 1.001
                signal.entry_low = breakout_level * 0.998
                signal.confluence.append(
                    f"📐 Entry refined to retest zone: "
                    f"{fmt_price(signal.entry_low)}–{fmt_price(signal.entry_high)}"
                )

        # Recalculate R:R with tighter entry
        signal = self._recalculate_rr(signal)
        return signal

    # ── Reversal-specific refinement ──────────────────────────

    def _refine_reversal_entry(self, signal, lows, highs, closes, current_price):
        """
        For reversals: entry zone should be the wick rejection zone,
        not a fixed ATR offset around current price.
        """
        direction = direction_str(signal)

        if direction == "LONG":
            # Rejection zone = the area where wicks were rejected (recent lows)
            recent_low = float(np.min(lows[-5:]))
            recent_low_2 = float(np.min(lows[-3:]))
            wick_zone_low = recent_low
            wick_zone_high = recent_low_2

            if wick_zone_high > wick_zone_low:
                signal.entry_low = wick_zone_low * 1.001
                signal.entry_high = wick_zone_high * 1.002
                signal.confluence.append(
                    f"📐 Entry at wick rejection zone: "
                    f"{fmt_price(signal.entry_low)}–{fmt_price(signal.entry_high)}"
                )

        elif direction == "SHORT":
            recent_high = float(np.max(highs[-5:]))
            recent_high_2 = float(np.max(highs[-3:]))
            wick_zone_high = recent_high
            wick_zone_low = recent_high_2

            if wick_zone_high > wick_zone_low:
                signal.entry_high = wick_zone_high * 0.999
                signal.entry_low = wick_zone_low * 0.998
                signal.confluence.append(
                    f"📐 Entry at wick rejection zone: "
                    f"{fmt_price(signal.entry_low)}–{fmt_price(signal.entry_high)}"
                )

        signal = self._recalculate_rr(signal)
        return signal

    # ── Structure alignment (used by all strategies) ──────────

    def _align_with_structure(
        self, signal, support, resistance, vwap, vwap_upper, vwap_lower, vpoc
    ):
        """
        Snap entry zone edges to nearest structural level when close.
        Also add confluence notes for VWAP/POC alignment.
        """
        direction = direction_str(signal)
        entry_mid = (signal.entry_low + signal.entry_high) / 2
        if entry_mid <= 0:
            return signal  # Cannot align with zero/negative entry price
        zone_width = signal.entry_high - signal.entry_low
        snap_tolerance = zone_width * 1.5 if zone_width > 0 else entry_mid * 0.003

        # Minimum zone width floor: 0.2% of price.
        # Prevents zones collapsing to near-zero when snap lands on entry_low exactly
        # (observed on MOODENG: zone collapsed to $0.000029 = 0.005% because entry_low
        # was snapped to kijun-sen but entry_high wasn't adjusted to maintain width).
        _min_zone_pct = 0.002  # 0.2% minimum
        _min_zone_width = entry_mid * _min_zone_pct

        if direction == "LONG":
            # Check if entry zone aligns with support
            for s_level in support:
                if abs(s_level - signal.entry_low) < snap_tolerance:
                    # BUG-11 FIX: preserve zone width after snap
                    # Old code: only moved entry_low → entry_high stayed put → zone collapsed
                    old_width = signal.entry_high - signal.entry_low
                    signal.entry_low = s_level
                    signal.entry_high = s_level + max(old_width, _min_zone_width)
                    signal.confluence.append(f"📐 Entry snapped to support {fmt_price(s_level)}")
                    signal.confidence = min(95, signal.confidence + 3)
                    break

            # VWAP alignment
            if abs(vwap_lower - entry_mid) / entry_mid < 0.005:
                signal.confluence.append(f"📐 VWAP lower band alignment {fmt_price(vwap_lower)}")
                signal.confidence = min(95, signal.confidence + 4)

            # VPOC alignment
            if vpoc and abs(vpoc - entry_mid) / entry_mid < 0.005:
                signal.confluence.append(f"📐 Volume POC alignment {fmt_price(vpoc)}")
                signal.confidence = min(95, signal.confidence + 5)

        elif direction == "SHORT":
            for r_level in resistance:
                if abs(r_level - signal.entry_high) < snap_tolerance:
                    # BUG-11 FIX: preserve zone width after snap
                    old_width = signal.entry_high - signal.entry_low
                    signal.entry_high = r_level
                    signal.entry_low = r_level - max(old_width, _min_zone_width)
                    signal.confluence.append(f"📐 Entry snapped to resistance {fmt_price(r_level)}")
                    signal.confidence = min(95, signal.confidence + 3)
                    break

            if abs(vwap_upper - entry_mid) / entry_mid < 0.005:
                signal.confluence.append(f"📐 VWAP upper band alignment {fmt_price(vwap_upper)}")
                signal.confidence = min(95, signal.confidence + 4)

            if vpoc and abs(vpoc - entry_mid) / entry_mid < 0.005:
                signal.confluence.append(f"📐 Volume POC alignment {fmt_price(vpoc)}")
                signal.confidence = min(95, signal.confidence + 5)

        # Final safety net: enforce minimum zone width regardless of path taken
        if signal.entry_high - signal.entry_low < _min_zone_width:
            _center = (signal.entry_low + signal.entry_high) / 2
            signal.entry_low = _center - _min_zone_width / 2
            signal.entry_high = _center + _min_zone_width / 2

        return signal  # BUG FIX: was missing, causing signal=None in refine()

    # ── Confirmation check ────────────────────────────────────

    def _check_confirmation(self, signal, highs, lows, closes) -> bool:
        """
        Check if the last 1-2 bars confirm the signal direction.
        LONG: requires higher low (bar -1 low > bar -2 low)
        SHORT: requires lower high (bar -1 high < bar -2 high)
        """
        if len(closes) < 3:
            return True  # Can't check, assume confirmed

        direction = direction_str(signal)

        if direction == "LONG":
            # Higher low check
            return float(lows[-1]) >= float(lows[-2]) * 0.999
        else:
            # Lower high check
            return float(highs[-1]) <= float(highs[-2]) * 1.001

    # ── Structural level helpers ──────────────────────────────

    def _find_sr_levels(
        self, highs, lows, closes, lookback=50, tolerance=0.003
    ) -> Tuple[List[float], List[float]]:
        """Find support and resistance from swing points"""
        h = highs[-lookback:] if len(highs) > lookback else highs
        l = lows[-lookback:] if len(lows) > lookback else lows
        current = closes[-1]

        support = []
        resistance = []

        for i in range(2, len(h) - 2):
            # Swing high
            if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
                if h[i] > current * (1 + tolerance):
                    resistance.append(float(h[i]))
                elif h[i] < current * (1 - tolerance):
                    support.append(float(h[i]))

            # Swing low
            if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
                if l[i] < current * (1 - tolerance):
                    support.append(float(l[i]))
                elif l[i] > current * (1 + tolerance):
                    resistance.append(float(l[i]))

        # Cluster nearby levels
        support = self._cluster_levels(sorted(support, reverse=True), tolerance)
        resistance = self._cluster_levels(sorted(resistance), tolerance)

        return support[:5], resistance[:5]

    def _cluster_levels(self, levels, tolerance=0.003):
        """Merge levels within tolerance into averaged clusters"""
        if not levels:
            return []
        clusters = [[levels[0]]]
        for level in levels[1:]:
            if clusters[-1][-1] != 0 and abs(level - clusters[-1][-1]) / abs(clusters[-1][-1]) < tolerance:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        return [np.mean(c) for c in clusters]

    def _calculate_vwap_bands(self, highs, lows, closes, volumes, period=24):
        """
        Session VWAP with ±1σ bands.
        Uses last `period` bars as the session.
        """
        h = highs[-period:]
        l = lows[-period:]
        c = closes[-period:]
        v = volumes[-period:]

        # Typical price
        typical = (h + l + c) / 3
        cumvol = np.cumsum(v)
        cumtp_vol = np.cumsum(typical * v)

        if cumvol[-1] == 0:
            vwap = float(c[-1])
            return vwap, vwap, vwap

        vwap = float(cumtp_vol[-1] / cumvol[-1])

        # Standard deviation band
        sq_diff = (typical - vwap) ** 2
        variance = np.sum(sq_diff * v) / cumvol[-1]
        std = float(np.sqrt(variance))

        return vwap, vwap + std, vwap - std

    def _volume_poc(self, closes, volumes, bins=50) -> Optional[float]:
        """
        Volume Point of Control — price level with highest volume.
        Computed via histogram of close prices weighted by volume.
        """
        if len(closes) < 20:
            return None

        c = closes[-50:]
        v = volumes[-50:]

        price_range = float(np.max(c) - np.min(c))
        if price_range == 0:
            return float(c[-1])

        bin_edges = np.linspace(np.min(c), np.max(c), bins + 1)
        vol_profile = np.zeros(bins)

        for price, vol in zip(c, v):
            idx = int((price - bin_edges[0]) / (price_range / bins))
            idx = min(max(idx, 0), bins - 1)
            vol_profile[idx] += vol

        poc_idx = int(np.argmax(vol_profile))
        poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)
        return poc_price

    def _nearest_level(self, levels, target, tolerance_pct=0.005):
        """Find the nearest level within tolerance"""
        if not levels or target == 0:
            return None
        for level in levels:
            if abs(level - target) / abs(target) < tolerance_pct:
                return level
        return None

    def _recalculate_rr(self, signal):
        """Recalculate R:R ratio after entry refinement.
        BUG-5 FIX: Guard against tp2 being zero, None, or geometrically invalid
        (i.e. on the wrong side of entry_mid). A zero/invalid tp2 produces R:R=0
        which kills the signal at the R:R floor gate even when the setup is valid.
        """
        entry_mid = (signal.entry_low + signal.entry_high) / 2
        risk = abs(entry_mid - signal.stop_loss)
        if risk <= 1e-10:
            signal.rr_ratio = 0.0
            return signal
        # Validate tp2 is on the correct side of entry_mid before computing reward
        is_long = is_long(signal)
        tp2_valid = (
            signal.tp2 is not None and
            signal.tp2 > 0 and
            ((is_long and signal.tp2 > entry_mid) or
             (not is_long and signal.tp2 < entry_mid))
        )
        if tp2_valid:
            reward = abs(signal.tp2 - entry_mid)
            signal.rr_ratio = round(reward / risk, 2)
        else:
            signal.rr_ratio = 0.0
        return signal


    def _repair_sl_if_inverted(self, signal, highs, lows):
        """
        FIX SL-REPAIR: After entry refinement the entry zone can shift so the
        original stop-loss is inside the new zone.  The aggregator rejects
        these signals with 'SL above entry zone for LONG' or 'SL below entry
        zone for SHORT'.  This method detects inversion and pushes the SL
        just outside the zone using a small ATR-based buffer.
        """
        try:
            import numpy as _np
            direction = direction_str(signal)

            # Quick ATR estimate from recent bars (last 14 candles)
            _h = _np.array([float(x) for x in highs[-15:]])
            _l = _np.array([float(x) for x in lows[-15:]])
            _atr = float(_np.mean(_h[1:] - _l[1:])) if len(_h) > 1 else 0.0
            _buf = max(_atr * 0.5, signal.entry_low * 0.002)  # at least 0.2%

            if direction == "LONG":
                # SL must be BELOW entry_low
                if signal.stop_loss >= signal.entry_low:
                    old_sl = signal.stop_loss
                    signal.stop_loss = round(signal.entry_low - _buf, 8)
                    signal.confluence.append(
                        f"🔧 SL repaired (was inside entry zone): "
                        f"{old_sl:.6f} → {signal.stop_loss:.6f}"
                    )
                    # FIX: widening the SL shortens the effective risk distance,
                    # which changes R:R. Without this, the aggregator may size /
                    # gate the signal on a stale rr_ratio that no longer matches
                    # the repaired geometry.
                    signal = self._recalculate_rr(signal)
            else:
                # SL must be ABOVE entry_high
                if signal.stop_loss <= signal.entry_high:
                    old_sl = signal.stop_loss
                    signal.stop_loss = round(signal.entry_high + _buf, 8)
                    signal.confluence.append(
                        f"🔧 SL repaired (was inside entry zone): "
                        f"{old_sl:.6f} → {signal.stop_loss:.6f}"
                    )
                    signal = self._recalculate_rr(signal)
        except Exception:
            pass  # Non-fatal — aggregator geometry check is the safety net
        return signal


# Singleton
entry_refiner = EntryRefiner()
