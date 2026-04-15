"""
TitanBot Pro — Equilibrium Trading Module
============================================
Institutional concept: Price oscillates between premium and discount
zones around equilibrium (EQ). Smart money buys discount, sells premium.

Architecture:
  Regime Analyzer provides range_high, range_low, range_eq
  This module classifies price position and enforces directional rules:

  ┌─────────────────────────────────────────┐
  │  PREMIUM ZONE (above EQ)               │  ← SHORT only
  │  ┈┈┈┈┈┈┈┈┈┈ EQ (midpoint) ┈┈┈┈┈┈┈┈┈┈  │  ← No trade (dead zone)
  │  DISCOUNT ZONE (below EQ)              │  ← LONG only
  └─────────────────────────────────────────┘

Rules:
  - LONG signals only valid in DISCOUNT zone (below EQ)
  - SHORT signals only valid in PREMIUM zone (above EQ)
  - Mid-range signals (within 10% of EQ) suppressed entirely
  - Deeper in zone = higher confidence multiplier
  - Only enforced during CHOPPY regime (trends can trade anywhere)

This prevents the #1 retail mistake: buying at premium, selling at discount.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EQAssessment:
    """Result of equilibrium zone analysis"""
    zone: str               # 'premium', 'discount', 'eq_dead_zone', 'no_range'
    zone_depth: float       # 0.0 (at EQ) to 1.0 (at range extreme)
    range_high: float
    range_low: float
    equilibrium: float
    current_price: float
    confidence_mult: float  # Multiplier based on zone depth
    should_block: bool      # True if trade direction conflicts with zone
    reason: str             # Human-readable explanation


class EquilibriumAnalyzer:
    """
    Classifies price position relative to equilibrium and enforces
    premium/discount directional rules.
    """

    # Dead zone: suppress trades within ±10% of EQ (VOLATILE/TRENDING)
    # In CHOPPY regime reduced to ±7% — tighter range = more signals without losing the protection
    EQ_DEAD_ZONE_PCT = 0.10
    EQ_DEAD_ZONE_CHOPPY = 0.07

    # Only enforce EQ rules when chop is high enough
    MIN_CHOP_FOR_EQ_RULES = 0.40

    def assess(
        self,
        current_price: float,
        direction: str,            # "LONG" or "SHORT"
        symbol_range_high: float = 0.0,  # BUG-6 FIX: pass symbol's own range
        symbol_range_low: float = 0.0,   # instead of relying on BTC's range
    ) -> EQAssessment:
        """
        Assess whether a trade direction is valid given current price
        position relative to equilibrium.

        BUG-6 FIX: The original implementation used regime_analyzer.range_high/low/eq
        which are computed from BTC's last 20 4h closes (in USD). Comparing an altcoin's
        USD entry price against BTC's USD range boundaries is meaningless — when BTC is
        in a downtrend the range sits above current prices, so EVERY altcoin appears to be
        in "discount zone", blocking ALL short signals. In a bear market (weekly BEARISH,
        ADX=53) this deadlocks the entire bot: HTF guardrail blocks longs, EQ blocks shorts.

        Fix: caller passes the signal symbol's own recent range (from ohlcv_dict).
        If not provided, EQ rules are skipped (safe fallback). Also adds a hard bypass
        for confirmed bear/bull trends where EQ zone rules are counter-productive.

        Args:
            current_price: Current market price of the signal symbol
            direction: "LONG" or "SHORT"
            symbol_range_high: Recent range high for this symbol (from ohlcv_dict)
            symbol_range_low: Recent range low for this symbol (from ohlcv_dict)
        """
        from analyzers.regime import regime_analyzer, Regime

        chop = regime_analyzer.chop_strength
        current_regime = regime_analyzer.regime

        # ── Bypass EQ rules in confirmed trends ──────────────────────────────
        # In a clear trend the premium/discount concept doesn't apply — price
        # keeps making new highs/lows outside of any prior "range". Applying EQ
        # rules in BULL_TREND/BEAR_TREND reverses the logic (e.g. shorts always
        # look like they're in "premium" because price is at new lows vs old range).
        #
        # LOCAL RANGE AWARENESS: If the symbol's own range is tight (< 15%),
        # the symbol may be in accumulation/distribution even though BTC is
        # trending. In that case, apply a soft confidence penalty for
        # direction-zone mismatches (no hard block).
        if current_regime in (Regime.BULL_TREND, Regime.BEAR_TREND, Regime.VOLATILE):
            _sym_range_size = symbol_range_high - symbol_range_low
            _sym_range_pct = _sym_range_size / current_price if current_price > 0 else 1.0
            from config.constants import LocalRange as _LR
            if (
                symbol_range_high > symbol_range_low > 0
                and _sym_range_pct < _LR.RANGE_PCT_THRESHOLD
                and chop >= _LR.MIN_CHOP_FOR_LOCAL
            ):
                # Symbol is locally range-bound — apply soft EQ awareness
                _eq = (symbol_range_high + symbol_range_low) / 2
                _half = _sym_range_size / 2
                _pos = (current_price - _eq) / _half if _half > 1e-10 else 0
                _pos = max(-1.0, min(1.0, _pos))
                # Soft penalty: buying premium in trend or selling discount in trend
                _penalty_rate = _LR.SOFT_EQ_MAX_PENALTY
                _min_mult = 1.0 - _penalty_rate  # floor at 0.90
                _soft_mult = 1.0
                _soft_reason = f"Trend regime ({current_regime.value}) — EQ zone rules bypassed"
                if _pos > 0.20 and direction == "LONG":
                    # Buying in premium zone of a range-bound symbol
                    _soft_mult = max(_min_mult, 1.0 - abs(_pos) * _penalty_rate)
                    _soft_reason = (
                        f"⚖️ Soft EQ: LONG in premium ({_pos:+.2f}) of local range "
                        f"({_sym_range_pct*100:.1f}%) — {_soft_mult:.2f}× conf"
                    )
                elif _pos < -0.20 and direction == "SHORT":
                    # Selling in discount zone of a range-bound symbol
                    _soft_mult = max(_min_mult, 1.0 - abs(_pos) * _penalty_rate)
                    _soft_reason = (
                        f"⚖️ Soft EQ: SHORT in discount ({_pos:+.2f}) of local range "
                        f"({_sym_range_pct*100:.1f}%) — {_soft_mult:.2f}× conf"
                    )
                return EQAssessment(
                    zone='trending', zone_depth=abs(_pos),
                    range_high=symbol_range_high, range_low=symbol_range_low,
                    equilibrium=_eq, current_price=current_price,
                    confidence_mult=_soft_mult, should_block=False,
                    reason=_soft_reason,
                )
            # Symbol is NOT locally range-bound — full bypass (original behavior)
            return EQAssessment(
                zone='trending', zone_depth=0,
                range_high=symbol_range_high, range_low=symbol_range_low,
                equilibrium=0, current_price=current_price,
                confidence_mult=1.0, should_block=False,
                reason=f"Trend regime ({current_regime.value}) — EQ zone rules bypassed"
            )

        # ── Use symbol's own range, not BTC's ────────────────────────────────
        range_high = symbol_range_high
        range_low = symbol_range_low

        # If no valid symbol range passed, skip EQ rules (safe fallback)
        if range_high <= range_low or range_high == 0:
            return EQAssessment(
                zone='no_range', zone_depth=0, range_high=0, range_low=0,
                equilibrium=0, current_price=current_price,
                confidence_mult=1.0, should_block=False,
                reason="No symbol range available — EQ rules not applied"
            )

        if chop < self.MIN_CHOP_FOR_EQ_RULES:
            eq = (range_high + range_low) / 2
            return EQAssessment(
                zone='trending', zone_depth=0,
                range_high=range_high, range_low=range_low,
                equilibrium=eq, current_price=current_price,
                confidence_mult=1.0, should_block=False,
                reason=f"Trending (chop={chop:.2f}) — EQ rules relaxed"
            )

        range_size = range_high - range_low
        if range_size <= 0:
            return EQAssessment(
                zone='no_range', zone_depth=0, range_high=range_high,
                range_low=range_low, equilibrium=(range_high + range_low) / 2,
                current_price=current_price, confidence_mult=1.0,
                should_block=False, reason="Zero range"
            )

        # ── VWAP-anchored equilibrium ─────────────────────────────────────
        # The simple midpoint (range_high + range_low) / 2 ignores where
        # the most trading has actually occurred. VWAP is the volume-weighted
        # average price — it represents where the market has spent the most
        # time and traded the most volume. This is the true institutional EQ.
        #
        # If VWAP is unavailable, fall back to simple midpoint.
        try:
            from analyzers.regime import regime_analyzer
            vwap = getattr(regime_analyzer, '_vwap', 0)
            if vwap and range_low < vwap < range_high:
                eq = vwap  # VWAP is the true equilibrium
            else:
                eq = (range_high + range_low) / 2  # Fallback
        except Exception:
            eq = (range_high + range_low) / 2

        # Calculate position relative to EQ
        # +1.0 = at range_high, -1.0 = at range_low, 0 = at EQ
        half_range = range_size / 2
        if half_range < 1e-10:
            position = 0.0
        else:
            position = (current_price - eq) / half_range
        position = max(-1.0, min(1.0, position))

        # Dead zone check: within ±EQ_DEAD_ZONE_PCT of EQ
        # In CHOPPY regime use tighter zone (7% vs 10%) — more signals without losing protection
        _dead_zone_pct = self.EQ_DEAD_ZONE_PCT
        try:
            from analyzers.regime import regime_analyzer
            if regime_analyzer.regime.value == "CHOPPY":
                _dead_zone_pct = self.EQ_DEAD_ZONE_CHOPPY
        except Exception:
            pass
        if abs(position) < _dead_zone_pct * 2:
            return EQAssessment(
                zone='eq_dead_zone', zone_depth=abs(position),
                range_high=range_high, range_low=range_low,
                equilibrium=eq, current_price=current_price,
                confidence_mult=0.0, should_block=True,
                reason=f"Price at equilibrium dead zone ({position:+.2f}) — no trade"
            )

        # Determine zone
        if position > 0:
            zone = 'premium'
            zone_depth = min(1.0, position)
        else:
            zone = 'discount'
            zone_depth = min(1.0, abs(position))

        # Enforce directional rules
        should_block = False
        reason = ""
        confidence_mult = 1.0

        if zone == 'premium' and direction == "LONG":
            should_block = True
            reason = f"❌ LONG rejected in premium zone ({position:+.2f}) — buying at premium"
        elif zone == 'discount' and direction == "SHORT":
            should_block = True
            reason = f"❌ SHORT rejected in discount zone ({position:+.2f}) — selling at discount"
        elif zone == 'premium' and direction == "SHORT":
            confidence_mult = 1.0 + zone_depth * 0.10  # Up to +10%
            reason = f"✅ SHORT in premium zone ({position:+.2f}) — selling at premium"
        elif zone == 'discount' and direction == "LONG":
            confidence_mult = 1.0 + zone_depth * 0.10
            reason = f"✅ LONG in discount zone ({position:+.2f}) — buying at discount"

        return EQAssessment(
            zone=zone,
            zone_depth=zone_depth,
            range_high=range_high,
            range_low=range_low,
            equilibrium=eq,
            current_price=current_price,
            confidence_mult=confidence_mult,
            should_block=should_block,
            reason=reason,
        )

    def get_zone_label(self, price: float) -> str:
        """Quick label for display"""
        from analyzers.regime import regime_analyzer
        eq = regime_analyzer.range_eq
        if eq <= 0:
            return ""
        if price > eq * 1.02:
            return "PREMIUM"
        elif price < eq * 0.98:
            return "DISCOUNT"
        else:
            return "EQ"


# ── Singleton ──────────────────────────────────────────────
eq_analyzer = EquilibriumAnalyzer()
