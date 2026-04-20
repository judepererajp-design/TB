"""
TitanBot Pro — No-Trade Zone Engine
=====================================
Retail bots try to trade continuously.
Professional systems spend MOST time doing nothing.

This module provides automatic signal suppression when conditions
are unfavorable. Hard vetoes, not penalties.

Rules:
  1. ATR too low (below 60% of 30-day ATR) → disable breakout/momentum
  2. ATR too high (above 250% of 30-day ATR) → reduce to A+ only
  3. BTC momentum unstable (fast < -3% in 1h) → block altcoin longs
  4. Funding rate neutral (not extreme) → skip funding-based signals
  5. Price at liquidity midpoint (50% of recent range) → skip
  6. Session dead zone → reduce signal count
  7. Weekend session → reduce signal count

Each rule returns (blocked, reason, affected_strategies).
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from strategies.base import direction_str

logger = logging.getLogger(__name__)


@dataclass
class NoTradeDecision:
    """Result of no-trade zone evaluation."""
    hard_block: bool           # True = reject signal immediately
    confidence_penalty: int    # Soft penalty if not hard blocked
    reasons: List[str]
    affected_strategies: List[str]  # Empty = applies to all strategies

    @property
    def should_block_strategy(self) -> bool:
        return self.hard_block and bool(self.affected_strategies)

    def blocks(self, strategy: str) -> bool:
        if self.hard_block:
            if not self.affected_strategies:
                return True  # Universal block
            return strategy in self.affected_strategies
        return False


class NoTradeZoneEngine:
    """
    Evaluates all no-trade conditions and returns a combined decision.

    Usage:
        decision = no_trade_engine.evaluate(
            symbol=signal.symbol,
            strategy=signal.strategy,
            direction=direction_str(signal),
            vol_ratio=vol_ratio,
            btc_momentum_fast=btc_mom,
            funding_rate=funding,
            price=entry_mid,
            range_high=range_high,
            range_low=range_low,
        )
        if decision.blocks(signal.strategy):
            return  # Skip signal
    """

    # Strategies that are momentum/breakout based → suppressed in low-vol
    MOMENTUM_STRATEGIES = {
        "InstitutionalBreakout",
        "Momentum",
        "MomentumContinuation",
        "ElliottWave",
        "Ichimoku",
        "IchimokuCloud",
    }

    def evaluate(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        vol_ratio: float = 1.0,
        btc_momentum_fast: float = 0.0,
        btc_momentum_1h: float = 0.0,
        funding_rate: Optional[float] = None,
        price: float = 0.0,
        range_high: float = 0.0,
        range_low: float = 0.0,
        session: str = "NEUTRAL",
        adx: float = 20.0,
    ) -> NoTradeDecision:
        """
        Evaluate all no-trade rules and return combined decision.
        Returns the most restrictive applicable rule.
        """
        reasons: List[str] = []
        total_penalty: int = 0
        hard_blocks: List[Tuple[str, List[str]]] = []

        # ── Rule 1: Volatility too low (compression) ──────────────────
        # Gap 4 fix: interpolated penalty instead of a hard 0.60 cliff.
        # Momentum/breakout strategies are most affected; others get a lighter touch.
        # Hard block is preserved only at extreme compression (vol_ratio < 0.35).
        if vol_ratio < 0.35:
            # Extreme compression — breakout/momentum genuinely can't work
            if strategy in self.MOMENTUM_STRATEGIES:
                hard_blocks.append((
                    f"⛔ Extreme compression (ATR ratio={vol_ratio:.2f}) — "
                    f"breakout/momentum suppressed until expansion",
                    list(self.MOMENTUM_STRATEGIES)
                ))
            else:
                reasons.append(f"⚠️ Extreme compression (ratio={vol_ratio:.2f}) — heavy penalty")
                total_penalty += 15
        elif vol_ratio < 0.80:
            # Soft zone: interpolated penalty scales from 0 (at 0.80) to 12 (at 0.35)
            # penalty = clamp((0.80 - vol_ratio) / 0.45, 0, 1) * 12
            compression_factor = min(1.0, (0.80 - vol_ratio) / 0.45)
            if strategy in self.MOMENTUM_STRATEGIES:
                penalty = round(compression_factor * 15)
            else:
                penalty = round(compression_factor * 8)
            if penalty > 0:
                reasons.append(f"⚠️ Low vol (ratio={vol_ratio:.2f}) — confidence −{penalty}")
                total_penalty += penalty

        # ── Rule 2: Volatility too high (panic) ───────────────────────
        # Gap 4 fix: interpolated penalty from 2.0× up; hard block preserved at 3.0×+.
        elif vol_ratio > 3.0:
            hard_blocks.append((
                f"⛔ Extreme volatility (ATR ratio={vol_ratio:.2f}) — "
                f"only A+ setups should pass (apply externally)",
                []
            ))
            reasons.append(f"⚠️ Extreme vol={vol_ratio:.2f}")
        elif vol_ratio > 2.0:
            # Soft zone: interpolated penalty scales from 0 (at 2.0) to 12 (at 3.0)
            panic_factor = min(1.0, (vol_ratio - 2.0) / 1.0)
            penalty = round(panic_factor * 12)
            if penalty > 0:
                reasons.append(f"⚠️ Elevated vol (ratio={vol_ratio:.2f}) — confidence −{penalty}")
                total_penalty += penalty

        # ── Rule 3: BTC fast momentum crash → block altcoin longs ─────
        # Tightened: hard block at -3% (was -5%), soft zone -2% to -3%.
        # Rationale: when BTC drops 3% in 1h, altcoin longs almost never work.
        # Evidence: all 4 counter-trend longs lost during BTC -3.1% session.
        is_btc = "BTC" in symbol.upper()
        if not is_btc and direction == "LONG" and btc_momentum_fast < -0.02:
            if btc_momentum_fast <= -0.03:
                hard_blocks.append((
                    f"⛔ BTC dropping sharply ({btc_momentum_fast:.1%}) — "
                    f"altcoin LONG hard-blocked (BTC rule: ≤ -3%)",
                    []
                ))
            else:
                # -2% to -3%: interpolated penalty
                crash_factor = min(1.0, (abs(btc_momentum_fast) - 0.02) / 0.01)
                penalty = round(crash_factor * 18)
                reasons.append(
                    f"⚠️ BTC dropping ({btc_momentum_fast:.1%}) — "
                    f"altcoin long confidence −{penalty}"
                )
                total_penalty += penalty

        # ── Rule 4: BTC 1h momentum divergence ────────────────────────
        # Tightened: -1.5% 1h now triggers a hard block in BEAR_TREND regime.
        # Previously only applied a soft -12 penalty, which was insufficient.
        if not is_btc and direction == "LONG" and btc_momentum_1h < -0.015:
            # Check if we're in bear regime — if so, hard block
            try:
                from analyzers.regime import regime_analyzer as _ra_ntze
                _bear = _ra_ntze.regime.value in ("BEAR_TREND", "VOLATILE")
            except Exception:
                _bear = False
            if _bear and btc_momentum_1h < -0.02:
                hard_blocks.append((
                    f"⛔ BTC 1h negative ({btc_momentum_1h:.1%}) in BEAR_TREND — "
                    f"altcoin LONG blocked",
                    []
                ))
            else:
                reasons.append(
                    f"⚠️ BTC 1h negative ({btc_momentum_1h:.1%}) — "
                    f"altcoin long confidence reduced"
                )
                total_penalty += 15  # raised from 12

        # ── Rule 5: Funding rate neutral → suppress funding signals ───
        # FundingRateArb needs extreme funding to work
        if strategy == "FundingRateArb" and funding_rate is not None:
            abs_funding = abs(funding_rate)
            if abs_funding < 0.0005:  # 0.05% per 8h = neutral
                hard_blocks.append((
                    f"⛔ Funding neutral ({funding_rate:.4%}) — "
                    f"FundingRateArb suppressed",
                    ["FundingRateArb"]
                ))

        # ── Rule 6: Price at liquidity midpoint ───────────────────────
        # Mid-range trades have low probability — only trade outer zones
        if range_high > range_low and price > 0:
            range_size = range_high - range_low
            if range_size > 0:
                pct_in_range = (price - range_low) / range_size
                is_mid = 0.35 < pct_in_range < 0.65
                if is_mid and adx < 22:  # Only suppress if not trending
                    reasons.append(
                        f"⚠️ Price at range midpoint ({pct_in_range:.0%}) in low-ADX ({adx:.0f}) — "
                        f"reduced probability"
                    )
                    total_penalty += 10

        # ── Rule 7: Session dead zone → reduce signal count ───────────
        # Phase 9-12 audit fix: this rule was documented in the module
        # docstring but never implemented.  UTC 03:00-07:00 and weekends
        # are low-liquidity periods; signals are less reliable.
        _session_upper = session.upper() if session else "NEUTRAL"
        if _session_upper == "DEAD_ZONE":
            reasons.append(
                "⚠️ Dead-zone session (03-07 UTC) — low liquidity, reduced confidence"
            )
            total_penalty += 8
        elif _session_upper == "WEEKEND":
            reasons.append(
                "⚠️ Weekend session — thin liquidity, reduced confidence"
            )
            total_penalty += 5

        # ── Compile result ─────────────────────────────────────────────
        if hard_blocks:
            # Return the hardest block applicable to this strategy
            for block_reason, block_strats in hard_blocks:
                if not block_strats or strategy in block_strats:
                    return NoTradeDecision(
                        hard_block=True,
                        confidence_penalty=0,
                        reasons=[block_reason] + reasons,
                        affected_strategies=block_strats,
                    )

        return NoTradeDecision(
            hard_block=False,
            confidence_penalty=total_penalty,
            reasons=reasons,
            affected_strategies=[],
        )


# ── Singleton ────────────────────────────────────────────────────────────
no_trade_engine = NoTradeZoneEngine()
