from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def _direction_value(direction: Any) -> str:
    return getattr(direction, "value", str(direction or "")).upper()


def _coalesce(value: Any, default: Any) -> Any:
    """Return default only for None so valid zero-values survive context building."""
    return default if value is None else value


@dataclass
class SetupContext:
    structure: Dict[str, Any] = field(default_factory=dict)
    pattern: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionContext:
    session: Dict[str, Any] = field(default_factory=dict)
    trigger: Dict[str, Any] = field(default_factory=dict)
    liquidity: Dict[str, Any] = field(default_factory=dict)
    whales: Dict[str, Any] = field(default_factory=dict)
    positioning: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, Any] = field(default_factory=dict)
    market: Dict[str, Any] = field(default_factory=dict)
    trade: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_setup_context(signal: Any) -> Dict[str, Any]:
    raw = getattr(signal, "raw_data", {}) or {}
    direction = _direction_value(getattr(signal, "direction", ""))
    setup_type = raw.get("setup_type") or raw.get("wyckoff_phase") or raw.get("target_wave") or signal.strategy

    order_block = None
    if raw.get("ob_low") is not None or raw.get("ob_high") is not None:
        order_block = {
            "low": raw.get("ob_low"),
            "high": raw.get("ob_high"),
        }

    fair_value_gap = None
    if raw.get("fvg_low") is not None or raw.get("fvg_high") is not None:
        fair_value_gap = {
            "low": raw.get("fvg_low"),
            "high": raw.get("fvg_high"),
        }

    liquidity_sweep = None
    if raw.get("sweep_level") is not None:
        liquidity_sweep = {
            "level": raw.get("sweep_level"),
            "sweep_low": raw.get("sweep_low"),
            "sweep_high": raw.get("sweep_high"),
        }

    elliott_wave = None
    if raw.get("target_wave") is not None:
        elliott_wave = {
            "target_wave": raw.get("target_wave"),
            "wave1_start": raw.get("wave1_start"),
            "wave1_end": raw.get("wave1_end"),
            "wave2_end": raw.get("wave2_end"),
            "projection": raw.get("tp_proj") or raw.get("w3_proj"),
        }

    setup = SetupContext(
        structure={
            "bias": direction,
            "trend": raw.get("structure_trend") or raw.get("regime") or getattr(signal, "regime", None) or "UNKNOWN",
            "bos": bool(raw.get("bos_level") is not None),
            "bos_level": raw.get("bos_level"),
            "choch": bool(raw.get("has_choch")),
            "choch_direction": raw.get("choch_direction") or "",
            "key_swing_high": raw.get("key_swing_high"),
            "key_swing_low": raw.get("key_swing_low"),
        },
        pattern={
            "setup_type": setup_type,
            "order_block": order_block,
            "fair_value_gap": fair_value_gap,
            "liquidity_sweep": liquidity_sweep,
            "wyckoff_phase": raw.get("wyckoff_phase"),
            "spring_detected": raw.get("spring_detected"),
            "utad_detected": raw.get("utad_detected"),
            "elliott_wave": elliott_wave,
        },
        location={
            "entry_low": getattr(signal, "entry_low", None),
            "entry_high": getattr(signal, "entry_high", None),
            "stop_loss": getattr(signal, "stop_loss", None),
            "timeframe": getattr(signal, "timeframe", None),
            "setup_class": getattr(signal, "setup_class", None),
            "entry_zone": raw.get("entry_zone"),
            "eq_zone": raw.get("eq_zone"),
            "eq_distance": raw.get("eq_distance"),
        },
        quality={
            "confidence": getattr(signal, "confidence", None),
            "rr_ratio": getattr(signal, "rr_ratio", None),
            "atr": raw.get("atr") or getattr(signal, "atr", None),
            "volume_confirms": raw.get("volume_confirms"),
            "wyckoff_confidence": raw.get("wyckoff_confidence"),
        },
        metadata={
            "symbol": getattr(signal, "symbol", ""),
            "strategy": getattr(signal, "strategy", ""),
            "regime": getattr(signal, "regime", None) or raw.get("regime"),
            "analysis_timeframes": list(getattr(signal, "analysis_timeframes", []) or []),
        },
    )
    return setup.to_dict()


def enrich_setup_context_with_eq(
    setup_context: Optional[Dict[str, Any]],
    *,
    eq_zone: str = "",
    eq_distance: float = 0.0,
) -> Dict[str, Any]:
    enriched = dict(setup_context or {})
    location = dict(enriched.get("location") or {})
    location["eq_zone"] = eq_zone
    location["entry_zone"] = eq_zone
    location["eq_distance"] = eq_distance
    enriched["location"] = location
    return enriched


def build_execution_context(
    signal: Any,
    *,
    session_name: str,
    is_killzone: bool,
    is_dead_zone: bool,
    is_weekend: bool,
    eq_zone: str,
    eq_zone_depth: float,
    volume_context: str,
    volume_score: float,
    derivatives_score: float = 50.0,
    sentiment_score: float = 50.0,
    volatility_regime: str = "",
    transition_type: str = "",
    transition_risk: float = 0.0,
    trade_type: str = "",
    trade_type_strength: int = 0,
) -> Dict[str, Any]:
    raw = getattr(signal, "raw_data", {}) or {}
    direction = _direction_value(getattr(signal, "direction", ""))
    whale_bias = (raw.get("whale_intent_bias") or "").upper()
    whale_buy_ratio = raw.get("whale_buy_ratio")
    if whale_buy_ratio is None:
        whale_conf = float(raw.get("whale_intent_confidence", 0.0) or 0.0)
        if whale_bias == "BULLISH":
            whale_buy_ratio = min(1.0, 0.5 + whale_conf * 0.5)
        elif whale_bias == "BEARISH":
            whale_buy_ratio = max(0.0, 0.5 - whale_conf * 0.5)
        else:
            whale_buy_ratio = 0.5
    whale_aligned = None
    if whale_bias in {"BULLISH", "BEARISH"} and direction in {"LONG", "SHORT"}:
        whale_aligned = (
            (direction == "LONG" and whale_bias == "BULLISH")
            or (direction == "SHORT" and whale_bias == "BEARISH")
        )

    execution = ExecutionContext(
        session={
            "name": session_name,
            "is_killzone": is_killzone,
            "is_dead_zone": is_dead_zone,
            "is_weekend": is_weekend,
        },
        trigger={
            "score": float(_coalesce(raw.get("trigger_quality_score"), 0.5)),
            "label": raw.get("trigger_quality_label", "MEDIUM"),
            "volume_context": raw.get("trigger_quality_volume_context", "NORMAL"),
        },
        liquidity={
            "spread_bps": float(_coalesce(raw.get("spread_bps"), 0.0)),
            "volume_context": volume_context,
            "volume_score": volume_score,
            "volume_quality_score": raw.get("volume_quality_score"),
            "volume_quality_label": raw.get("volume_quality_label"),
        },
        whales={
            "aligned": whale_aligned,
            "buy_ratio": float(_coalesce(whale_buy_ratio, 0.5)),
            "intent": raw.get("whale_intent"),
            "bias": whale_bias or "NEUTRAL",
            "confidence": raw.get("whale_intent_confidence"),
        },
        positioning={
            "derivatives_score": derivatives_score,
            "sentiment_score": sentiment_score,
            "funding_rate": raw.get("funding_rate"),
            "oi_change_24h": raw.get("oi_change_24h"),
            "contextual_overlay": raw.get("contextual_overlay"),
        },
        location={
            "eq_zone": eq_zone,
            "eq_distance": eq_zone_depth,
        },
        market={
            "volatility_regime": volatility_regime or "UNKNOWN",
            "transition_type": transition_type or "stable",
            "transition_risk": transition_risk,
        },
        trade={
            "direction": direction,
            "type": trade_type,
            "strength": trade_type_strength,
        },
    )
    return execution.to_dict()
