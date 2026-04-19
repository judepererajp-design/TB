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


class WyckoffAnalyzer:
    """
    Detects Wyckoff phases and Spring/UTAD setups.
    Used by patterns module and can enhance SMC signals.
    """

    def __init__(self):
        self._cfg = cfg.patterns.wyckoff
        self._min_range_bars = getattr(self._cfg, 'min_range_bars', 20)
        self._vol_sensitivity = getattr(self._cfg, 'volume_sensitivity', 1.3)

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

        avg_volume = np.mean(volumes)

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
        sc_result = self._find_climax(highs, lows, closes, volumes, avg_volume,
                                       direction='down', lookback=range_start)
        bc_result = self._find_climax(highs, lows, closes, volumes, avg_volume,
                                       direction='up', lookback=range_start)

        current_price = float(closes[-1])

        # ── Check for Spring ─────────────────────────────────
        spring = self._detect_spring(highs, lows, closes, volumes, range_low, avg_volume)
        if spring:
            notes = [
                "✅ Wyckoff Spring detected — stop hunt below range",
                f"✅ Range low: {fmt_price(range_low)} (support)",
                "✅ Quick recovery above range low",
            ]
            if spring['volume_spike']:
                notes.append("✅ Volume climax on Spring — absorption confirmed")
            if vol_contraction:
                notes.append("✅ Volume contracted inside range — Wyckoff narrative confirmed")
            if dryup_before_evt:
                notes.append("✅ Volume dry-up before Spring — textbook setup")

            sl = range_low - (range_size * 0.1)  # Below spring low
            entry_low  = range_low * 1.001
            entry_high = range_low + range_size * 0.15

            return WyckoffResult(
                phase=WyckoffPhase.ACCUMULATION_C,
                confidence=78.0 + (10 if spring['volume_spike'] else 0),
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
            )

        # ── Check for UTAD (Upthrust After Distribution) ─────
        utad = self._detect_utad(highs, lows, closes, volumes, range_high, avg_volume)
        if utad:
            notes = [
                "✅ Wyckoff UTAD detected — stop hunt above range",
                f"✅ Range high: {fmt_price(range_high)} (resistance)",
                "✅ Quick rejection below range high",
            ]
            if utad['volume_spike']:
                notes.append("✅ Volume climax on UTAD — supply confirmed")
            if vol_contraction:
                notes.append("✅ Volume contracted inside range — Wyckoff narrative confirmed")
            if dryup_before_evt:
                notes.append("✅ Volume dry-up before UTAD — textbook setup")

            sl = range_high + (range_size * 0.1)
            entry_high = range_high * 0.999
            entry_low  = range_high - range_size * 0.15

            return WyckoffResult(
                phase=WyckoffPhase.DISTRIBUTION_C,
                confidence=76.0 + (10 if utad['volume_spike'] else 0),
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
            )

        # ── Check for Sign of Strength (SOS) in Phase D ──────
        sos = self._detect_sos(closes, volumes, range_high, range_low, avg_volume)
        if sos:
            notes = [
                "✅ Wyckoff Sign of Strength — demand dominating",
                "✅ Phase D accumulation — markup approaching",
            ]
            if vol_contraction:
                notes.append("✅ Volume contracted during range — institutional accumulation")
            if dryup_before_evt:
                notes.append("✅ Volume dry-up → expansion on SOS confirmed")
            return WyckoffResult(
                phase=WyckoffPhase.ACCUMULATION_D,
                confidence=70.0,
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
        n = len(closes)

        # Slide window to find period of compression
        best_range = None
        best_compression = float('inf')

        for start in range(max(0, n - 80), n - min_bars):
            window_h = highs[start:n-5]
            window_l = lows[start:n-5]

            if len(window_h) < min_bars:
                continue

            range_h = np.max(window_h)
            range_l = np.min(window_l)
            range_size = range_h - range_l

            # Relative range size — smaller = more compressed
            compression = range_size / closes[start]

            # Look for 10-30% range compression
            if 0.05 <= compression <= 0.35:
                # Check most bars stay within the range (not trending)
                in_range = np.sum(
                    (highs[start:] <= range_h * 1.02) &
                    (lows[start:]  >= range_l * 0.98)
                )
                if in_range / len(window_h) >= 0.75:  # 75% of bars in range
                    if compression < best_compression:
                        best_compression = compression
                        best_range = (range_h, range_l, start)

        return best_range

    def _find_climax(
        self, highs, lows, closes, volumes, avg_volume,
        direction: str, lookback: int
    ) -> Optional[Dict]:
        """Find a volume climax bar"""
        vol_threshold = avg_volume * 2.0  # SC/BC requires 2x+ volume
        acc_cfg = cfg.patterns.wyckoff.get('accumulation', {})
        sc_mult = getattr(acc_cfg, 'sc_volume_mult', 2.0)

        for i in range(max(lookback, 0), len(closes) - 2):
            if volumes[i] < avg_volume * sc_mult:
                continue
            if direction == 'down':
                # V17 FIX: `opens` was not passed to this method — use closes[i] < closes[i-1]
                # as a proxy for bearish candle (price dropped from prior bar)
                if closes[i] < closes[i-1]:
                    return {'bar': i, 'price': lows[i], 'volume': volumes[i]}
            else:
                if closes[i] > closes[i-1]:
                    return {'bar': i, 'price': highs[i], 'volume': volumes[i]}
        return None

    def _detect_spring(
        self, highs, lows, closes, volumes, range_low, avg_volume
    ) -> Optional[Dict]:
        """
        Spring: Price briefly dips below the range low then recovers.
        This is the bear trap — retail stops triggered, smart money buys.
        """
        acc_cfg = cfg.patterns.wyckoff.get('accumulation', {})
        spring_min = getattr(acc_cfg, 'spring_min_pct', -0.03)
        spring_rec = getattr(acc_cfg, 'spring_recovery_pct', 0.02)

        # Check last 10 bars for spring
        for i in range(max(0, len(lows) - 10), len(lows) - 1):
            # Price dips below range low
            if lows[i] < range_low * (1 + spring_min):  # e.g., 3% below range low
                # Then recovers back above range low
                if closes[i] > range_low * (1 + spring_rec) or closes[i+1] > range_low:
                    volume_spike = volumes[i] > avg_volume * self._vol_sensitivity
                    return {
                        'bar': i,
                        'spring_low': lows[i],
                        'recovery_close': closes[i],
                        'volume_spike': volume_spike
                    }
        return None

    def _detect_utad(
        self, highs, lows, closes, volumes, range_high, avg_volume
    ) -> Optional[Dict]:
        """
        UTAD: Price briefly spikes above range high then fails.
        The bull trap — retail breaks out, smart money distributes.
        """
        for i in range(max(0, len(highs) - 10), len(highs) - 1):
            if highs[i] > range_high * 1.01:  # Spike above range high
                if closes[i] < range_high or (i+1 < len(closes) and closes[i+1] < range_high):
                    volume_spike = volumes[i] > avg_volume * self._vol_sensitivity
                    return {
                        'bar': i,
                        'utad_high': highs[i],
                        'rejection_close': closes[i],
                        'volume_spike': volume_spike
                    }
        return None

    def _detect_sos(
        self, closes, volumes, range_high, range_low, avg_volume
    ) -> bool:
        """
        Sign of Strength: strong move up from range low toward range high on high volume,
        preceded by a quiet (dry-up) period.  The dry-up → expansion sequence confirms
        that demand is genuinely dominating, not just noise.
        """
        range_size = range_high - range_low
        current = closes[-1]

        # Price in middle-to-upper portion of range
        if current < range_low + range_size * 0.4:
            return False

        # Recent volume increasing vs older baseline
        if len(volumes) < 5:
            return False
        recent_avg = np.mean(volumes[-5:])
        older_avg  = np.mean(volumes[-15:-5]) if len(volumes) >= 15 else np.mean(volumes)
        if not (recent_avg > older_avg * 1.2):
            return False

        # W-Q2: dry-up → expansion required — not just "recent > older"
        # Look for a quiet window before the expansion (bars -10 to -5)
        if len(volumes) >= 10:
            quiet_window = float(np.mean(volumes[-10:-5]))
            expansion    = float(np.mean(volumes[-5:]))
            dryup_to_expansion = expansion > quiet_window * 1.25
        else:
            dryup_to_expansion = True  # insufficient data; don't gate

        return dryup_to_expansion


# ── Module-level convenience function ─────────────────────
wyckoff_analyzer = WyckoffAnalyzer()

# Type alias for import compatibility
from typing import Dict as _Dict
List = list
