"""
TitanBot Pro — Regime Transition Detector
==========================================
Most losses occur DURING regime transitions, not within regimes.
The market is most dangerous when switching between states:

  TREND → CHOP       (momentum fades, but signals keep firing)
  CHOP → EXPANSION   (breakout whipsaws before real move)
  EXPANSION → CHOP   (vol spike then dead)

Detection signals:
  - Rising volatility + flat trend = likely TREND → CHOP
  - Compression breakout probability = CHOP → EXPANSION
  - Rapid ATR collapse after expansion = EXPANSION → CHOP

Action:
  - During transition: reduce signal volume (confidence boost required)
  - Increase min_confidence threshold dynamically
  - Flag on Telegram message so trader is aware

Usage:
  from analyzers.regime_transition import transition_detector
  result = transition_detector.evaluate(
      current_regime=regime_name,
      chop_strength=chop_str,
      vol_ratio=vol_ratio,
      adx=adx,
  )
  if result.in_transition:
      signal.confluence.append(result.warning)
      adaptive_min += result.confidence_increase
"""

import logging
import time
from dataclasses import dataclass, replace
from typing import List, Optional

logger = logging.getLogger(__name__)

BREAKOUT_VOL_RATIO_THRESHOLD = 1.4
BREAKOUT_FADE_CYCLES = 2
TRANSITION_ESCALATION_MAX_CYCLES = 6
TRANSITION_ESCALATION_OFFSET = 1
TRANSITION_ESCALATION_STEP = 0.75
TRANSITION_RISK_ESCALATION_STEP = 0.04
TRANSITION_VOLUME_ESCALATION_STEP = 0.04
TREND_TRADE_TRANSITION_PENALTY_MULT = 0.70
MIN_TREND_TRANSITION_PENALTY = 1


@dataclass
class TransitionResult:
    in_transition: bool
    transition_type: str        # e.g., "TREND→CHOP"
    risk_level: float           # 0.0 to 1.0
    confidence_increase: int    # Extra confidence required (0–15)
    volume_reduction: float     # Multiply max_signals by this (0.5 to 1.0)
    warning: str
    cycles_in_transition: int = 0
    breakout_exemption_active: bool = False


class RegimeTransitionDetector:
    """
    Detects regime transitions and quantifies transition risk.
    Reduces signal volume during dangerous state changes.
    """

    def __init__(self):
        self._regime_history: List[tuple] = []  # (timestamp, regime, chop)
        self._adx_history: List[float] = []
        self._last_transition_type: Optional[str] = None
        self._transition_streak: int = 0
        self._breakout_exemption_active: bool = False
        self._breakout_fade_streak: int = 0

    def record(self, regime: str, chop_strength: float, adx: float):
        """Record current regime snapshot for history."""
        self._regime_history.append((time.time(), regime, chop_strength))
        self._adx_history.append(adx)
        # Keep 30 observations max
        if len(self._regime_history) > 30:
            self._regime_history.pop(0)
        if len(self._adx_history) > 30:
            self._adx_history.pop(0)

    def evaluate(
        self,
        current_regime: str,
        chop_strength: float,
        vol_ratio: float,
        adx: float,
    ) -> TransitionResult:
        """
        Evaluate current transition risk.

        Returns TransitionResult with risk level and recommendations.
        """
        def _build_transition_result(
            transition_type: str,
            risk_level: float,
            confidence_increase: int,
            volume_reduction: float,
            warning: str,
            breakout_exemption_active: bool = False,
        ) -> TransitionResult:
            if self._last_transition_type == transition_type:
                self._transition_streak += 1
            else:
                self._last_transition_type = transition_type
                self._transition_streak = 1

            escalation = min(
                TRANSITION_ESCALATION_MAX_CYCLES,
                max(0, self._transition_streak - TRANSITION_ESCALATION_OFFSET),
            )
            escalated_conf = confidence_increase + int(round(escalation * TRANSITION_ESCALATION_STEP))
            escalated_risk = min(0.95, risk_level + escalation * TRANSITION_RISK_ESCALATION_STEP)
            escalated_volume = max(0.5, volume_reduction - escalation * TRANSITION_VOLUME_ESCALATION_STEP)
            if escalation:
                warning = (
                    f"{warning} (transition persisting {self._transition_streak} cycles)"
                )

            return TransitionResult(
                in_transition=True,
                transition_type=transition_type,
                risk_level=escalated_risk,
                confidence_increase=escalated_conf,
                volume_reduction=escalated_volume,
                warning=warning,
                cycles_in_transition=self._transition_streak,
                breakout_exemption_active=breakout_exemption_active,
            )

        def _stable_result(breakout_exemption_active: bool = False) -> TransitionResult:
            self._last_transition_type = None
            self._transition_streak = 0
            return TransitionResult(
                in_transition=False,
                transition_type="STABLE",
                risk_level=0.0,
                confidence_increase=0,
                volume_reduction=1.0,
                warning="",
                cycles_in_transition=0,
                breakout_exemption_active=breakout_exemption_active,
            )

        # ── Check recent regime stability ───────────────────────────────
        if len(self._regime_history) >= 3:
            recent_regimes = [r for _, r, _ in self._regime_history[-5:]]
            unique_regimes = len(set(recent_regimes))
            regime_unstable = unique_regimes >= 3
        else:
            regime_unstable = False

        pattern = detect_transition_pattern(
            current_regime=current_regime,
            chop_strength=chop_strength,
            vol_ratio=vol_ratio,
            adx=adx,
            adx_history=self._adx_history,
        )
        if pattern:
            if pattern["transition_type"] == "CHOP→EXPANSION":
                self._breakout_exemption_active = True
                self._breakout_fade_streak = 0
            else:
                self._breakout_exemption_active = False
                self._breakout_fade_streak = 0
            return _build_transition_result(
                transition_type=pattern["transition_type"],
                risk_level=pattern["risk_level"],
                confidence_increase=pattern["confidence_increase"],
                volume_reduction=pattern["volume_reduction"],
                warning=pattern["warning"],
                breakout_exemption_active=self._breakout_exemption_active,
            )

        if self._breakout_exemption_active:
            if vol_ratio < BREAKOUT_VOL_RATIO_THRESHOLD:
                self._breakout_fade_streak += 1
                if self._breakout_fade_streak >= BREAKOUT_FADE_CYCLES:
                    self._breakout_exemption_active = False
                    self._breakout_fade_streak = 0
                    return _build_transition_result(
                        transition_type="EXPANSION→CHOP",
                        risk_level=0.60,
                        confidence_increase=7,
                        volume_reduction=0.7,
                        warning=(
                            "⚠️ Breakout expansion faded "
                            f"(vol_ratio={vol_ratio:.2f}) — breakout exemption expired"
                        ),
                    )
                return _stable_result(breakout_exemption_active=True)

            self._breakout_fade_streak = 0
            return _stable_result(breakout_exemption_active=True)

        # General instability (regime flipping frequently)
        self._breakout_exemption_active = False
        self._breakout_fade_streak = 0
        if regime_unstable:
            return _build_transition_result(
                transition_type="UNSTABLE",
                risk_level=0.70,
                confidence_increase=12,
                volume_reduction=0.6,
                warning=(
                    "⚠️ Regime unstable (multiple state changes recently) — "
                    "reduce signal frequency"
                ),
            )

        # No transition detected
        return _stable_result()


def has_active_breakout_exemption(
    market_transition_type: str,
    transition_result: Optional[TransitionResult],
) -> bool:
    """Breakout exemption stays active only while internal vol expansion persists."""
    return (
        market_transition_type == "breakout"
        and bool(getattr(transition_result, "breakout_exemption_active", False))
    )


def should_block_countertrend_pullback(
    trade_type: Optional[str],
    transition_result: Optional[TransitionResult],
    market_transition_type: str,
) -> bool:
    """Block only the dangerous transition/trade-type pairs."""
    if not trade_type or transition_result is None or not transition_result.in_transition:
        return False
    if has_active_breakout_exemption(market_transition_type, transition_result):
        return False
    if trade_type.startswith("LOCAL_CONTINUATION_"):
        return False
    if not trade_type.startswith("PULLBACK_"):
        return False
    return transition_result.transition_type in {"TREND→CHOP", "EXPANSION→CHOP", "UNSTABLE"}


def get_trade_type_transition_penalty(
    trade_type: Optional[str],
    transition_result: Optional[TransitionResult],
) -> int:
    """Scale transition penalties by trade type while keeping TREND trades non-zero."""
    if transition_result is None or not transition_result.in_transition:
        return 0

    base_penalty = transition_result.confidence_increase
    if not trade_type:
        return base_penalty
    if trade_type.startswith("TREND_"):
        return max(
            MIN_TREND_TRANSITION_PENALTY,
            int(round(base_penalty * TREND_TRADE_TRANSITION_PENALTY_MULT)),
        )
    return base_penalty


def detect_transition_pattern(
    current_regime: str,
    chop_strength: float,
    vol_ratio: float,
    adx: float,
    adx_history: Optional[List[float]] = None,
) -> Optional[dict]:
    """Detect the current transition pattern from internal market-state inputs only.

    Returns a transition descriptor dict for TREND→CHOP, CHOP→EXPANSION, or
    EXPANSION→CHOP when the ADX/chop/ATR-style inputs match those patterns.
    Returns None when no internal transition pattern is active.
    """
    adx_falling = False
    adx_rising = False
    if adx_history and len(adx_history) >= 5:
        adx_recent = adx_history[-5:]
        adx_change = adx_recent[-1] - adx_recent[0]
        adx_falling = adx_change < -5
        adx_rising = adx_change > 5

    if (adx < 25 and adx_falling
            and chop_strength > 0.45
            and 0.8 < vol_ratio < 1.3):
        return {
            "transition_type": "TREND→CHOP",
            "risk_level": 0.75,
            "confidence_increase": 10,
            "volume_reduction": 0.6,
            "warning": (
                "⚠️ Regime transition: TREND→CHOP detected "
                f"(ADX={adx:.0f}↓, chop={chop_strength:.2f}↑) — "
                "raise min confidence +10, reduce signal frequency"
            ),
        }

    if (chop_strength > 0.55
            and vol_ratio > 1.4
            and adx_rising):
        return {
            "transition_type": "CHOP→EXPANSION",
            "risk_level": 0.65,
            "confidence_increase": 8,
            "volume_reduction": 0.7,
            "warning": (
                "⚠️ Potential CHOP→EXPANSION transition "
                f"(vol_ratio={vol_ratio:.2f}↑, chop={chop_strength:.2f}) — "
                "breakout risk; wait for confirmation"
            ),
        }

    if (vol_ratio < 0.75
            and adx_falling
            and current_regime in ("VOLATILE", "BULL_TREND", "BEAR_TREND")):
        return {
            "transition_type": "EXPANSION→CHOP",
            "risk_level": 0.60,
            "confidence_increase": 7,
            "volume_reduction": 0.7,
            "warning": (
                "⚠️ EXPANSION→CHOP transition detected "
                f"(vol collapsed to {vol_ratio:.2f}x, ADX={adx:.0f}↓) — "
                "momentum strategies unreliable"
            ),
        }

    return None


def amplify_transition_with_news(
    transition_result: Optional[TransitionResult],
    regime_name: str,
    news_bias: str,
    news_score: float,
) -> Optional[TransitionResult]:
    """Amplify transition penalties when strong news bias reinforces the disruption.

    Only strong bullish/bearish net-news scores are considered. Amplification is
    applied to unstable or degrading transitions, with the strongest effect when
    the active news bias conflicts with the current bull/bear regime.
    """
    if transition_result is None or not transition_result.in_transition:
        return transition_result

    from config.constants import NewsIntelligence, RegimeTransition as RT

    if news_bias not in {"BULLISH", "BEARISH"}:
        return transition_result
    if abs(news_score) < min(
        abs(NewsIntelligence.NET_SCORE_BULLISH_THRESHOLD),
        abs(NewsIntelligence.NET_SCORE_BEARISH_THRESHOLD),
    ):
        return transition_result

    transition_type = transition_result.transition_type
    bias_conflicts_with_regime = (
        (regime_name == "BULL_TREND" and news_bias == "BEARISH")
        or (regime_name == "BEAR_TREND" and news_bias == "BULLISH")
    )
    if transition_type not in {"TREND→CHOP", "EXPANSION→CHOP", "UNSTABLE"}:
        return transition_result
    if not bias_conflicts_with_regime and transition_type != "UNSTABLE":
        return transition_result

    multiplier = min(
        RT.NEWS_AMPLIFIER_MAX_MULT,
        1.0 + abs(news_score) * RT.NEWS_AMPLIFIER_SENSITIVITY,
    )
    amplified_conf = max(
        transition_result.confidence_increase + 1,
        int(round(transition_result.confidence_increase * multiplier)),
    )
    amplified_risk = min(0.99, transition_result.risk_level * multiplier)
    amplified_volume = max(0.5, transition_result.volume_reduction - min(0.15, abs(news_score) * 0.1))
    amplified_warning = (
        f"{transition_result.warning} | {news_bias} news score {news_score:+.2f} amplified transition risk"
    )
    return replace(
        transition_result,
        risk_level=amplified_risk,
        confidence_increase=amplified_conf,
        volume_reduction=amplified_volume,
        warning=amplified_warning,
    )


# ── Singleton ──────────────────────────────────────────────────────────────
transition_detector = RegimeTransitionDetector()
