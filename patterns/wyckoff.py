"""
TitanBot Pro — Wyckoff Analysis (patterns/wyckoff.py)
======================================================
Standalone Wyckoff schematic analyzer for the pattern detection subsystem.
Used by the candlestick/pattern layer — NOT directly by the strategy engine.

NOTE: This is separate from strategies/wyckoff.py, which is the full
WyckoffStrategy that extends BaseStrategy and generates SignalResult objects.
The engine imports strategies.wyckoff.WyckoffStrategy, not this file.

Detects Wyckoff accumulation and distribution schematics.

Wyckoff is the gold standard for understanding institutional
accumulation and distribution. Smart money follows a 5-phase plan:

ACCUMULATION (before markup):
  Phase A: Stopping the downtrend (PS, SC, AR, ST)
  Phase B: Building the cause (secondary tests)
  Phase C: Testing supply (Spring) ← HIGHEST EDGE ENTRY
  Phase D: Dominance of demand (SOS, LPS)
  Phase E: Markup begins

DISTRIBUTION (before markdown):
  Phase A: Stopping the uptrend (PSY, BC, AR, UT)
  Phase B: Building the cause (secondary tests)
  Phase C: Testing demand (UTAD) ← HIGHEST EDGE SHORT
  Phase D: Dominance of supply (SOW, LPSY)
  Phase E: Markdown begins

The Spring and UTAD are the highest probability entries
because they are the last manipulation before the real move.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.loader import cfg
from utils.formatting import fmt_price
from patterns._common import clamp_projection

logger = logging.getLogger(__name__)


class WyckoffPhase(str, Enum):
    UNKNOWN             = "UNKNOWN"
    ACCUMULATION_A      = "ACCUMULATION_A"
    ACCUMULATION_B      = "ACCUMULATION_B"
    ACCUMULATION_C      = "ACCUMULATION_C"  # Spring phase ← best entry
    ACCUMULATION_D      = "ACCUMULATION_D"  # Sign of Strength
    DISTRIBUTION_B      = "DISTRIBUTION_B"
    DISTRIBUTION_C      = "DISTRIBUTION_C"  # UTAD phase ← best short
    DISTRIBUTION_D      = "DISTRIBUTION_D"  # Sign of Weakness


@dataclass
class WyckoffResult:
    phase: WyckoffPhase
    confidence: float               # 0-100
    direction: str                  # LONG | SHORT
    entry_zone: Tuple[float, float]
    stop_loss: float
    key_level: float
    notes: List[str]
    spring_detected: bool = False
    utad_detected: bool   = False
    volume_confirms: bool = False
    # W-Q2: proper Wyckoff volume narrative flags
    volume_contraction_in_range: bool = False   # volume contracted during TR vs prior trend
    dryup_before_event: bool          = False   # quiet volume before Spring/UTAD/SOS
    # W-S1: structural geometry — exposed so strategies can independently validate the range
    range_high: float = 0.0         # upper boundary of detected trading range
    range_low:  float = 0.0         # lower boundary of detected trading range
    range_bars: int   = 0           # number of bars the range spans
    # P2-W1/W2: secondary-test + cause-effect governance lineage
    lps_confirmed: bool = False                  # LPS/LPSY retest held
    cause_effect_target: Optional[float] = None  # projected markup/markdown target


class WyckoffAnalyzer:
    """
    Detects Wyckoff phases and Spring/UTAD setups.
    Used by patterns module and can enhance SMC signals.
    """

    def __init__(self):
        self._cfg = cfg.patterns.wyckoff
        self._min_range_bars = getattr(self._cfg, 'min_range_bars', 20)
        self._vol_sensitivity = getattr(self._cfg, 'volume_sensitivity', 1.3)
        # P1-W15: max range scan window (was hard-coded 80)
        self._max_range_bars = getattr(self._cfg, 'max_range_bars', 120)
        # P2-W1: LPS / LPSY secondary-test bonus — after spring/UTAD, a
        # successful retest that holds above/below the key level is the
        # classical "Last Point of Support/Supply". Bumps confidence.
        self._lps_bonus = float(getattr(self._cfg, 'lps_bonus', 5.0))
        # P2-W2: emit the cause-effect projected target in raw_data & notes.
        self._emit_cause_effect = bool(getattr(self._cfg, 'emit_cause_effect_target', True))

    def analyze(self, ohlcv: List, timeframe: str = '4h') -> Optional[WyckoffResult]:
        """
        Full Wyckoff analysis. Returns a result if a valid phase is detected.
        """
        if not ohlcv or len(ohlcv) < self._min_range_bars + 20:
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df = df.astype({'open':float,'high':float,'low':float,'close':float,'volume':float})

        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        opens  = df['open'].values

        # P1-W9: climax baseline is median, not mean — extreme bars inflate the
        # mean and make subsequent climax detection harder.
        avg_volume = float(np.median(volumes)) if len(volumes) else 0.0

        # P1-W16: ATR for range/threshold scaling (used by Spring/UTAD)
        atr = _atr(highs, lows, closes, period=14)

        # ── Detect trading range ──────────────────────────────
        range_result = self._detect_trading_range(highs, lows, closes, volumes)
        if not range_result:
            return None

        range_high, range_low, range_start = range_result
        range_size = range_high - range_low
        range_bars = len(closes) - range_start  # bars since range began

        # W-Q2: volume narrative flags — computed once, shared across detectors
        vol_contraction  = self._check_range_volume_contraction(volumes, range_start)
        dryup_before_evt = self._check_dryup(volumes)

        # ── Detect Selling Climax (SC) or Buying Climax (BC) ─
        # (Results not used downstream currently but kept for future Phase-A work.)
        _ = self._find_climax(highs, lows, closes, volumes, opens, avg_volume,
                              direction='down', lookback=range_start)
        _ = self._find_climax(highs, lows, closes, volumes, opens, avg_volume,
                              direction='up', lookback=range_start)

        # ── Check for Spring ─────────────────────────────────
        spring = self._detect_spring(highs, lows, closes, volumes,
                                     range_low, range_size, avg_volume, atr)
        if spring:
            notes = [
                "✅ Wyckoff Spring detected — stop hunt below range",
                f"✅ Range low: {fmt_price(range_low)} (support)",
                "✅ Quick recovery above range low",
            ]
            # P1-W14: dry-up narrative bumps confidence, not just notes
            conf = 78.0
            if spring['volume_spike']:
                notes.append("✅ Volume climax on Spring — absorption confirmed")
                conf += 10.0
            if vol_contraction:
                notes.append("✅ Volume contracted inside range — Wyckoff narrative confirmed")
                conf += 4.0
            if dryup_before_evt:
                notes.append("✅ Volume dry-up before Spring — textbook setup")
                conf += 3.0
            # P2-W1: LPS (Last Point of Support) — bump confidence when a
            # successful retest has already printed after the spring.
            lps_ok = self._detect_lps(lows, closes, range_low, spring['bar'], atr)
            if lps_ok:
                notes.append("🎯 LPS confirmed — higher low retest held above range")
                conf += self._lps_bonus
            conf = min(95.0, conf)

            # P1-W10: Spring SL must tolerate typical undercut — use the deeper of
            # 25% of range size or 1.5 × ATR (was fixed 10% of range).
            _sl_buffer = max(range_size * 0.25, atr * 1.5) if atr > 0 else range_size * 0.25
            sl = min(spring['spring_low'], range_low) - _sl_buffer

            # P1-W10: entry zone — widen from 0.1% of price to ATR/range-based
            _ez_buffer = max(range_size * 0.02, atr * 0.25) if atr > 0 else range_size * 0.02
            entry_low  = range_low + _ez_buffer * 0.25
            entry_high = range_low + max(range_size * 0.15, _ez_buffer * 2)

            # P2-W2: cause-effect projected target
            ce_target = None
            if self._emit_cause_effect:
                ce_target = self._cause_effect_target(
                    "LONG", range_high, range_bars, range_size, atr
                )
                if ce_target:
                    notes.append(
                        f"🎯 Cause-effect target ≈ {fmt_price(ce_target)}"
                        f" (range_bars={range_bars})"
                    )

            return WyckoffResult(
                phase=WyckoffPhase.ACCUMULATION_C,
                confidence=conf,
                direction="LONG",
                entry_zone=(entry_low, entry_high),
                stop_loss=sl,
                key_level=range_low,
                notes=notes,
                spring_detected=True,
                volume_confirms=spring['volume_spike'],
                volume_contraction_in_range=vol_contraction,
                dryup_before_event=dryup_before_evt,
                range_high=range_high,
                range_low=range_low,
                range_bars=range_bars,
                lps_confirmed=lps_ok,
                cause_effect_target=ce_target,
            )

        # ── Check for UTAD (Upthrust After Distribution) ─────
        utad = self._detect_utad(highs, lows, closes, volumes,
                                 range_high, range_size, avg_volume, atr)
        if utad:
            notes = [
                "✅ Wyckoff UTAD detected — stop hunt above range",
                f"✅ Range high: {fmt_price(range_high)} (resistance)",
                "✅ Quick rejection below range high",
            ]
            conf = 76.0
            if utad['volume_spike']:
                notes.append("✅ Volume climax on UTAD — supply confirmed")
                conf += 10.0
            if vol_contraction:
                notes.append("✅ Volume contracted inside range — Wyckoff narrative confirmed")
                conf += 4.0
            if dryup_before_evt:
                notes.append("✅ Volume dry-up before UTAD — textbook setup")
                conf += 3.0
            # P2-W1: LPSY (Last Point of Supply)
            lpsy_ok = self._detect_lpsy(highs, closes, range_high, utad['bar'], atr)
            if lpsy_ok:
                notes.append("🎯 LPSY confirmed — lower high retest capped by range")
                conf += self._lps_bonus
            conf = min(95.0, conf)

            _sl_buffer = max(range_size * 0.25, atr * 1.5) if atr > 0 else range_size * 0.25
            sl = max(utad['utad_high'], range_high) + _sl_buffer
            _ez_buffer = max(range_size * 0.02, atr * 0.25) if atr > 0 else range_size * 0.02
            entry_high = range_high - _ez_buffer * 0.25
            entry_low  = range_high - max(range_size * 0.15, _ez_buffer * 2)

            ce_target = None
            if self._emit_cause_effect:
                ce_target = self._cause_effect_target(
                    "SHORT", range_low, range_bars, range_size, atr
                )
                if ce_target:
                    notes.append(
                        f"🎯 Cause-effect target ≈ {fmt_price(ce_target)}"
                        f" (range_bars={range_bars})"
                    )

            return WyckoffResult(
                phase=WyckoffPhase.DISTRIBUTION_C,
                confidence=conf,
                direction="SHORT",
                entry_zone=(entry_low, entry_high),
                stop_loss=sl,
                key_level=range_high,
                notes=notes,
                utad_detected=True,
                volume_confirms=utad['volume_spike'],
                volume_contraction_in_range=vol_contraction,
                dryup_before_event=dryup_before_evt,
                range_high=range_high,
                range_low=range_low,
                range_bars=range_bars,
                lps_confirmed=lpsy_ok,
                cause_effect_target=ce_target,
            )

        # ── Check for Sign of Strength (SOS) in Phase D ──────
        sos = self._detect_sos(closes, volumes, range_high, range_low, avg_volume)
        if sos:
            notes = [
                "✅ Wyckoff Sign of Strength — demand dominating",
                "✅ Phase D accumulation — markup approaching",
            ]
            conf = 70.0
            if vol_contraction:
                notes.append("✅ Volume contracted during range — institutional accumulation")
                conf += 5.0
            if dryup_before_evt:
                notes.append("✅ Volume dry-up → expansion on SOS confirmed")
                conf += 4.0
            return WyckoffResult(
                phase=WyckoffPhase.ACCUMULATION_D,
                confidence=min(95.0, conf),
                direction="LONG",
                entry_zone=(range_low + range_size * 0.3, range_low + range_size * 0.5),
                stop_loss=range_low - range_size * 0.05,
                key_level=range_high,
                notes=notes,
                volume_confirms=True,
                volume_contraction_in_range=vol_contraction,
                dryup_before_event=dryup_before_evt,
                range_high=range_high,
                range_low=range_low,
                range_bars=range_bars,
            )

        return None

    def _check_range_volume_contraction(
        self, volumes, range_start: int
    ) -> bool:
        """
        W-Q2: True when mean volume inside the trading range is meaningfully lower
        than mean volume in the prior trend segment.  This validates the Wyckoff
        narrative — accumulation/distribution requires volume to dry up as the smart
        money absorbs supply/demand quietly.
        """
        if range_start < 5:
            return False
        _pre_range_lookback = self._min_range_bars  # same as min range to stay consistent
        pre_range = volumes[:range_start][-_pre_range_lookback:]
        in_range  = volumes[range_start:]
        if len(pre_range) < 5 or len(in_range) < 5:
            return False
        return float(np.mean(in_range)) < float(np.mean(pre_range)) * 0.85

    def _check_dryup(self, volumes) -> bool:
        """
        W-Q2: True when volume dried up in the bars immediately preceding the most
        recent candles, followed by a volume pickup — the classic dry-up → expansion
        pattern that precedes legitimate Springs, UTADs, and SOS moves.
        Quiet window: bars -8 to -3 (relative to end); expansion window: last 3 bars.
        """
        if len(volumes) < 10:
            return False
        quiet   = float(np.mean(volumes[-8:-3]))
        expand  = float(np.mean(volumes[-3:]))
        overall = float(np.mean(volumes))
        return expand > quiet * 1.25 and quiet < overall * 0.9

    def _detect_trading_range(
        self, highs, lows, closes, volumes
    ) -> Optional[Tuple[float, float, int]]:
        """
        Detect a consolidation range (trading range).
        Returns (range_high, range_low, range_start_bar) or None.
        """
        min_bars = self._min_range_bars
        max_bars = self._max_range_bars   # P1-W15: configurable
        n = len(closes)

        # Slide window to find period of compression
        best_range = None
        best_compression = float('inf')

        for start in range(max(0, n - max_bars), n - min_bars):
            # P1-W6: use the SAME window for both the range measurement and
            # the in-range count. Previously the range was measured on
            # highs[start:n-5] but the in-range count used highs[start:],
            # which penalized ranges whose last 5 bars just broke out
            # (exactly the setup we want to catch).
            window_h = highs[start:n]
            window_l = lows[start:n]

            if len(window_h) < min_bars:
                continue

            range_h = float(np.max(window_h))
            range_l = float(np.min(window_l))
            range_size = range_h - range_l

            # P1-W7: stabilize compression anchor by using median close of window
            anchor = float(np.median(closes[start:n]))
            if anchor <= 0:
                continue
            compression = range_size / anchor

            # Look for 5-35% range compression
            if 0.05 <= compression <= 0.35:
                in_range = int(np.sum(
                    (window_h <= range_h * 1.02) &
                    (window_l >= range_l * 0.98)
                ))
                if in_range / len(window_h) >= 0.75:
                    if compression < best_compression:
                        best_compression = compression
                        best_range = (range_h, range_l, start)

        return best_range

    def _find_climax(
        self, highs, lows, closes, volumes, opens, avg_volume,
        direction: str, lookback: int
    ) -> Optional[Dict]:
        """Find a volume climax bar. `opens` is used to classify bar direction."""
        acc_cfg = cfg.patterns.wyckoff.get('accumulation', None) if hasattr(cfg.patterns.wyckoff, 'get') else None
        sc_mult = 2.0
        if acc_cfg is not None:
            sc_mult = getattr(acc_cfg, 'sc_volume_mult', 2.0) if hasattr(acc_cfg, 'sc_volume_mult') else 2.0

        # P1-W1: start at max(lookback, 1) — previous `max(lookback, 0)` allowed
        # i=0 where closes[i-1] wraps to the LAST bar, producing spurious climaxes.
        for i in range(max(lookback, 1), len(closes) - 2):
            if volumes[i] < avg_volume * sc_mult:
                continue
            # P1-W3: use opens[i] vs closes[i] (the bar's own direction)
            # rather than closes[i] vs closes[i-1] (the bar-to-bar delta).
            if direction == 'down':
                if closes[i] < opens[i]:
                    return {'bar': i, 'price': lows[i], 'volume': volumes[i]}
            else:
                if closes[i] > opens[i]:
                    return {'bar': i, 'price': highs[i], 'volume': volumes[i]}
        return None

    def _detect_lps(
        self, lows, closes, range_low: float, spring_bar: int, atr: float
    ) -> bool:
        """
        P2-W1: Last Point of Support (LPS) after a Spring.

        After the Spring prints, price should pull back toward range_low,
        make a HIGHER LOW (above the spring low), and resume the rally.
        We inspect bars AFTER spring_bar and before the current bar looking
        for this retest geometry.
        """
        n = len(lows)
        if n == 0 or spring_bar < 0 or spring_bar >= n - 2:
            return False
        # Require at least one bar between spring and current
        tail_lows = lows[spring_bar + 1 : n - 1]
        if len(tail_lows) == 0:
            return False
        pullback_low = float(min(tail_lows))
        spring_low = float(lows[spring_bar])
        # Must stay above spring low (HL) and above range_low minus small buffer
        tolerance = atr * 0.25 if atr > 0 else 0.0
        if pullback_low <= spring_low + tolerance:
            return False
        if pullback_low < range_low - tolerance:
            return False
        # Current bar should be above pullback low (rally resumed)
        current = float(closes[-1])
        return current > pullback_low

    def _detect_lpsy(
        self, highs, closes, range_high: float, utad_bar: int, atr: float
    ) -> bool:
        """P2-W1: mirror of _detect_lps for UTAD → Last Point of Supply."""
        n = len(highs)
        if n == 0 or utad_bar < 0 or utad_bar >= n - 2:
            return False
        tail_highs = highs[utad_bar + 1 : n - 1]
        if len(tail_highs) == 0:
            return False
        pullback_high = float(max(tail_highs))
        utad_high = float(highs[utad_bar])
        tolerance = atr * 0.25 if atr > 0 else 0.0
        if pullback_high >= utad_high - tolerance:
            return False
        if pullback_high > range_high + tolerance:
            return False
        current = float(closes[-1])
        return current < pullback_high

    def _cause_effect_target(
        self, direction: str, key_level: float, range_bars: int,
        range_size: float, atr: float,
    ) -> Optional[float]:
        """
        P2-W2: Wyckoff "cause and effect" projected target.

        Cause = time spent in the range (measured by bars, scaled by range
        size normalized to ATR).  Effect = magnitude of the subsequent move.

        target_distance = range_size × sqrt(range_bars / min_bars)

        The sqrt-scaling is a conservative variant of the point-and-figure
        "number of columns × box size" rule used in classical Wyckoff — it
        keeps very-long ranges from producing unreasonably distant targets.
        """
        if range_size <= 0 or range_bars <= 0 or key_level <= 0:
            return None
        try:
            import math
            scale = math.sqrt(max(range_bars / max(self._min_range_bars, 1), 1.0))
        except Exception:
            scale = 1.0
        raw_distance = range_size * scale
        # Phase-2: use shared projection clamp for consistency with geometric
        # patterns. clamp_projection caps at max(key_level * 0.5, 3 * atr).
        distance = clamp_projection(raw_distance, key_level, atr)
        if distance <= 0:
            return None
        if direction == "LONG":
            return key_level + distance
        return key_level - distance

    def _detect_spring(
        self, highs, lows, closes, volumes,
        range_low, range_size, avg_volume, atr
    ) -> Optional[Dict]:
        """
        Spring: Price briefly dips below the range low then recovers.
        P1-W5: thresholds are range/ATR-scaled, not fixed price percentages.
                On a tight range with range_size = 1% of price, a fixed 3%
                dip = 3× the range, which is absurd — fixed thresholds
                must be replaced with range-relative ones.
        P1-W8: recovery requires AND (bar closes above low) instead of OR.
        """
        # Dip depth: max of 10% of range or 0.5 ATR
        dip_threshold = max(range_size * 0.1, atr * 0.5) if atr > 0 else range_size * 0.1
        # Recovery threshold: bar closes at least 2% of range back above range_low
        recovery_threshold = max(range_size * 0.02, atr * 0.1) if atr > 0 else range_size * 0.02

        for i in range(max(0, len(lows) - 10), len(lows) - 1):
            # Price dips below range low by at least dip_threshold
            if lows[i] < range_low - dip_threshold * 0.5:
                # AND it recovers on the same or next bar
                recovered_same = closes[i] > range_low + recovery_threshold
                recovered_next = (i + 1 < len(closes) and
                                  closes[i + 1] > range_low + recovery_threshold)
                if recovered_same or recovered_next:
                    volume_spike = volumes[i] > avg_volume * self._vol_sensitivity
                    return {
                        'bar': i,
                        'spring_low': float(lows[i]),
                        'recovery_close': float(closes[i]),
                        'volume_spike': bool(volume_spike),
                    }
        return None

    def _detect_utad(
        self, highs, lows, closes, volumes,
        range_high, range_size, avg_volume, atr
    ) -> Optional[Dict]:
        """
        UTAD: Price briefly spikes above range high then fails.
        P1-W5: range/ATR-scaled thresholds (mirror of _detect_spring).
        P1-W8: AND-style recovery (bar closes back below range_high).
        """
        spike_threshold = max(range_size * 0.1, atr * 0.5) if atr > 0 else range_size * 0.1
        rej_threshold   = max(range_size * 0.02, atr * 0.1) if atr > 0 else range_size * 0.02

        for i in range(max(0, len(highs) - 10), len(highs) - 1):
            if highs[i] > range_high + spike_threshold * 0.5:
                rejected_same = closes[i] < range_high - rej_threshold
                rejected_next = (i + 1 < len(closes) and
                                 closes[i + 1] < range_high - rej_threshold)
                if rejected_same or rejected_next:
                    volume_spike = volumes[i] > avg_volume * self._vol_sensitivity
                    return {
                        'bar': i,
                        'utad_high': float(highs[i]),
                        'rejection_close': float(closes[i]),
                        'volume_spike': bool(volume_spike),
                    }
        return None

    def _detect_sos(
        self, closes, volumes, range_high, range_low, avg_volume
    ) -> bool:
        """
        Sign of Strength: CONFIRMED break above range_high on expansion volume,
        preceded by dry-up.  Previously, the gate was merely "price in upper
        40% of range" which fires on normal in-range bounces — a false SOS.
        """
        range_size = range_high - range_low
        if range_size <= 0:
            return False
        current = closes[-1]

        # P1-W13: require actual break above range_high, not just upper-40%
        if current < range_high:
            return False

        # Recent volume increasing vs older baseline
        if len(volumes) < 5:
            return False
        recent_avg = float(np.mean(volumes[-5:]))
        older_avg  = float(np.mean(volumes[-15:-5])) if len(volumes) >= 15 else float(np.mean(volumes))
        if not (recent_avg > older_avg * 1.2):
            return False

        # W-Q2: dry-up → expansion required — not just "recent > older"
        if len(volumes) >= 10:
            quiet_window = float(np.mean(volumes[-10:-5]))
            expansion    = float(np.mean(volumes[-5:]))
            dryup_to_expansion = expansion > quiet_window * 1.25
        else:
            dryup_to_expansion = True  # insufficient data; don't gate

        return dryup_to_expansion


# ── Helpers ───────────────────────────────────────────────────────────
def _atr(highs, lows, closes, period: int = 14) -> float:
    """
    Average True Range over trailing `period` bars.
    Defined in-module to keep this file standalone (no BaseStrategy dependency).
    Returns 0.0 for insufficient data.
    """
    try:
        if len(closes) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1]),
            )
            trs.append(tr)
        return float(np.mean(trs[-period:]))
    except Exception:
        return 0.0


# ── Module-level singleton (lazy) ─────────────────────────
_wyckoff_analyzer_instance: Optional[WyckoffAnalyzer] = None

def _get_wyckoff_analyzer() -> WyckoffAnalyzer:
    """Lazy accessor so cfg circular imports at module load don't crash."""
    global _wyckoff_analyzer_instance
    if _wyckoff_analyzer_instance is None:
        _wyckoff_analyzer_instance = WyckoffAnalyzer()
    return _wyckoff_analyzer_instance


class _LazyWyckoffAnalyzer:
    """Proxy so existing `wyckoff_analyzer.analyze(...)` callers keep working."""
    def __getattr__(self, name):
        return getattr(_get_wyckoff_analyzer(), name)


wyckoff_analyzer = _LazyWyckoffAnalyzer()
