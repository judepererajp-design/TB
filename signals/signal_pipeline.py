"""
TitanBot Pro — Shared Signal Pipeline
=======================================
Common signal-processing gates used by both the live engine
(``core/engine.py``) and the backtest engine (``backtester/engine.py``).

Extracting these into a single module ensures that any bug-fix or
threshold change is applied in both contexts automatically.

Usage (backtester)::

    from signals.signal_pipeline import (
        apply_htf_hard_gate,
        apply_equilibrium_gate,
        apply_regime_filter_gate,
        passes_confidence_floor,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LOCAL_CONTINUATION_REVERSAL_STRENGTH = 2
PULLBACK_REVERSAL_STRENGTH = 3


# ── Result containers ──────────────────────────────────────────

@dataclass
class GateResult:
    """Outcome of a single pipeline gate."""
    blocked: bool = False
    reason: str = ""
    adjusted_confidence: Optional[float] = None


def has_reversal_confirmation(
    signal_direction: str,
    *,
    setup_context: Optional[dict] = None,
    execution_context: Optional[dict] = None,
) -> bool:
    """Return True when the signal carries explicit reversal/continuation evidence.

    This is used by both the live engine and the execution gate when deciding
    whether an opposite-direction follow-up signal is legitimate or just churn.
    """
    direction = direction_str(signal_direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return False

    structure = dict((setup_context or {}).get("structure") or {})
    pattern = dict((setup_context or {}).get("pattern") or {})
    trade = dict((execution_context or {}).get("trade") or {})

    choch_direction = direction_str(structure.get("choch_direction")).upper()
    if structure.get("choch"):
        if direction == "LONG" and choch_direction == "BULLISH":
            return True
        if direction == "SHORT" and choch_direction == "BEARISH":
            return True

    phase = str(pattern.get("wyckoff_phase") or "").lower()
    bullish_phases = {"accumulation", "markup", "reaccumulation"}
    bearish_phases = {"distribution", "markdown", "redistribution"}
    if direction == "LONG" and (phase in bullish_phases or pattern.get("spring_detected")):
        return True
    if direction == "SHORT" and (phase in bearish_phases or pattern.get("utad_detected")):
        return True

    trade_type = str(trade.get("type") or "").upper()
    try:
        trade_strength = int(trade.get("strength") or 0)
    except (TypeError, ValueError):
        trade_strength = 0

    if (
        trade_type == f"LOCAL_CONTINUATION_{direction}"
        and trade_strength >= LOCAL_CONTINUATION_REVERSAL_STRENGTH
    ):
        return True
    if (
        trade_type == f"PULLBACK_{direction}"
        and trade_strength >= PULLBACK_REVERSAL_STRENGTH
    ):
        return True

    return False


def evaluate_opposite_direction_conflict(
    *,
    new_direction: str,
    conflicting_direction: str,
    setup_context: Optional[dict] = None,
    execution_context: Optional[dict] = None,
    conflict_age_secs: Optional[float] = None,
    cooldown_secs: float = 0.0,
    conflict_is_active: bool = False,
) -> GateResult:
    """Block rapid/opposing flips unless the new signal shows real reversal evidence."""
    direction = direction_str(new_direction).upper()
    opposing = direction_str(conflicting_direction).upper()

    if not direction or not opposing or direction == opposing:
        return GateResult()

    if has_reversal_confirmation(
        direction,
        setup_context=setup_context,
        execution_context=execution_context,
    ):
        return GateResult()

    if conflict_is_active:
        return GateResult(
            blocked=True,
            reason=f"active {opposing} exposure still unresolved",
        )

    if conflict_age_secs is not None and conflict_age_secs < cooldown_secs:
        remaining_mins = max(1, int((cooldown_secs - conflict_age_secs) / 60))
        return GateResult(
            blocked=True,
            reason=(
                f"recent {opposing} signal still in cooldown "
                f"({remaining_mins}m remaining)"
            ),
        )

    return GateResult()


# ── Gate 1: HTF Guardrail Hard Block ──────────────────────────

def apply_htf_hard_gate(
    htf_guardrail: object,
    signal_direction: str,
    raw_confidence: float,
    strategy_name: str,
) -> GateResult:
    """Check the HTF guardrail's hard-block condition.

    Parameters
    ----------
    htf_guardrail :
        An instance exposing ``is_hard_blocked(signal_direction, raw_confidence, strategy_name)``.
    signal_direction : str
        ``"LONG"`` or ``"SHORT"``.
    raw_confidence : float
        Confidence score before any penalties.
    strategy_name : str
        Name of the strategy that generated the signal.

    Returns
    -------
    GateResult
        ``blocked=True`` if the HTF guardrail vetoes the signal.
    """
    try:
        blocked, reason = htf_guardrail.is_hard_blocked(
            signal_direction=signal_direction,
            raw_confidence=raw_confidence,
            strategy_name=strategy_name,
        )
        if blocked:
            return GateResult(blocked=True, reason=reason)
    except Exception as exc:
        logger.debug("HTF hard-gate error (skipped): %s", exc)

    return GateResult()


# ── Gate 2: Equilibrium Zone ──────────────────────────────────

def apply_equilibrium_gate(
    eq_analyzer: object,
    entry_mid: float,
    signal_direction: str,
    symbol_range_high: float = 0.0,
    symbol_range_low: float = 0.0,
) -> GateResult:
    """Assess whether the signal is inside an equilibrium zone.

    Returns ``blocked=True`` if the signal sits in the wrong zone
    (e.g. LONG at premium).  Otherwise ``adjusted_confidence`` carries
    the zone's multiplier-adjusted confidence.
    """
    try:
        eq_result = eq_analyzer.assess(
            entry_mid,
            signal_direction,
            symbol_range_high=symbol_range_high,
            symbol_range_low=symbol_range_low,
        )
        if eq_result.should_block:
            return GateResult(blocked=True, reason=eq_result.reason)
        return GateResult(
            adjusted_confidence=eq_result.confidence_mult,
            reason=eq_result.reason,
        )
    except Exception as exc:
        logger.debug("EQ gate error (skipped): %s", exc)
        return GateResult()


# ── Gate 3: Regime Threshold Filter ───────────────────────────

def apply_regime_filter_gate(
    confidence: float,
    signal_direction: str,
    strategy_name: str,
    regime_name: str,
    chop_strength: float = 0.0,
    htf_aligned: bool = False,
) -> GateResult:
    """Apply regime-based confidence adjustments.

    Wraps ``signals.regime_thresholds.apply_regime_filter`` and
    returns a :class:`GateResult` with the (possibly penalised)
    confidence in ``adjusted_confidence``.

    ``blocked=True`` only when the strategy is hard-blocked by the
    regime (e.g. "not allowed in VOLATILE_PANIC").
    """
    try:
        from signals.regime_thresholds import apply_regime_filter
        adj_conf, regime_blocked, reason = apply_regime_filter(
            confidence,
            signal_direction,
            strategy_name,
            regime_name,
            chop_strength=chop_strength,
            htf_aligned=htf_aligned,
        )
        if regime_blocked and "not allowed in" in reason:
            return GateResult(blocked=True, reason=reason)
        if regime_blocked:
            return GateResult(
                adjusted_confidence=max(30.0, adj_conf),
                reason=reason,
            )
        return GateResult(adjusted_confidence=adj_conf, reason=reason)
    except Exception as exc:
        logger.debug("Regime filter error (skipped): %s", exc)
        return GateResult(adjusted_confidence=confidence)


# ── Gate 4: Confidence Floor ──────────────────────────────────

def passes_confidence_floor(
    confidence: float,
    min_confidence: float = 72.0,
) -> bool:
    """Return True if the signal meets the minimum confidence threshold."""
    return confidence >= min_confidence


# ── Helper: extract direction string ──────────────────────────

def direction_str(direction: object) -> str:
    """Normalise an enum or plain string direction to a ``str``."""
    return getattr(direction, "value", str(direction))
