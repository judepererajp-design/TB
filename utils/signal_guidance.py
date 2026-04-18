from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from config.constants import Backtester


ENTRY_LATE_PCT: float = 2.0
DEAD_ZONE_SIZE_MULTIPLIER: float = 0.50


def _value(signal: Any, key: str, default=None):
    if isinstance(signal, Mapping):
        return signal.get(key, default)
    return getattr(signal, key, default)


def _direction_value(direction: Any) -> str:
    if hasattr(direction, "value"):
        return str(direction.value)
    return str(direction or "")


def _normalized_confluence(confluence: Optional[Iterable[Any]]) -> list[str]:
    if not confluence:
        return []
    return [str(item) for item in confluence]


def _entry_mid(signal: Any) -> Optional[float]:
    entry_low = _value(signal, "entry_low")
    entry_high = _value(signal, "entry_high")
    if entry_low is None or entry_high is None:
        return None
    return (float(entry_low) + float(entry_high)) / 2.0


def round_trip_friction_pct() -> float:
    return 2.0 * (Backtester.DEFAULT_COMMISSION_PCT + Backtester.DEFAULT_SLIPPAGE_PCT)


def predicted_round_trip_friction_pct(signal: Any) -> float:
    """Round-trip friction using the *predictive* slippage model when
    enough microstructure data is on the signal, else the static default.

    Returns the same units as ``round_trip_friction_pct`` (decimal fraction
    of price). Used by ``fee_adjusted_rr`` so the fee-adjusted RR gate
    rejects setups whose *predicted* (rather than historical-average)
    friction would eat the edge.
    """
    try:
        from analyzers.expected_slippage import estimate_slippage_pct_from_signal
        slip_pct = estimate_slippage_pct_from_signal(signal)
    except Exception:
        slip_pct = Backtester.DEFAULT_SLIPPAGE_PCT
    return 2.0 * (Backtester.DEFAULT_COMMISSION_PCT + slip_pct)


def _format_level(level: float) -> str:
    if level >= 1000:
        text = f"{level:.2f}"
    elif level >= 1:
        text = f"{level:.4f}"
    elif level >= 0.01:
        text = f"{level:.6f}"
    else:
        text = f"{level:.8f}"
    return text.rstrip("0").rstrip(".")


def fee_adjusted_rr(signal: Any) -> Optional[float]:
    entry_mid = _entry_mid(signal)
    stop_loss = _value(signal, "stop_loss")
    tp2 = _value(signal, "tp2")
    tp1 = _value(signal, "tp1")
    direction = _direction_value(_value(signal, "direction"))
    target = tp2 or tp1
    if not entry_mid or not stop_loss or not target or direction not in ("LONG", "SHORT"):
        return None

    stop_loss = float(stop_loss)
    target = float(target)
    # Use the predictive friction estimator: pulls spread/depth/vol off the
    # signal's raw_data when available, falls back to the static backtester
    # default otherwise. Same units as the original round-trip number.
    friction = entry_mid * predicted_round_trip_friction_pct(signal)
    if direction == "LONG":
        risk = entry_mid - stop_loss
        reward = target - entry_mid
    else:
        risk = stop_loss - entry_mid
        reward = entry_mid - target

    if risk <= 0 or reward <= 0:
        return None

    adjusted_reward = max(0.0, reward - friction)
    adjusted_risk = risk + friction
    if adjusted_risk <= 0:
        return None
    return adjusted_reward / adjusted_risk


def skip_level(signal: Any, late_pct: float = ENTRY_LATE_PCT) -> Optional[float]:
    direction = _direction_value(_value(signal, "direction"))
    entry_low = _value(signal, "entry_low")
    entry_high = _value(signal, "entry_high")
    if entry_low is None or entry_high is None:
        return None
    entry_low = float(entry_low)
    entry_high = float(entry_high)
    if direction == "LONG":
        return entry_high * (1.0 + late_pct / 100.0)
    if direction == "SHORT":
        return entry_low * (1.0 - late_pct / 100.0)
    return None


def skip_rule(signal: Any, late_pct: float = ENTRY_LATE_PCT) -> Optional[str]:
    direction = _direction_value(_value(signal, "direction"))
    level = skip_level(signal, late_pct=late_pct)
    if level is None or direction not in ("LONG", "SHORT"):
        return None
    comparator = "above" if direction == "LONG" else "below"
    return f"Skip if price trades {comparator} {_format_level(level)}"


def dead_zone_warning(
    signal: Any,
    *,
    confluence: Optional[Iterable[Any]] = None,
    current_session: Optional[str] = None,
) -> Optional[str]:
    conf = _normalized_confluence(confluence if confluence is not None else _value(signal, "confluence", []))
    session = str(current_session or _value(signal, "session", "") or "").upper()
    if session == "DEAD_ZONE":
        return "⚠️ Dead zone — reduce size by 50%"
    if any("dead zone" in item.lower() for item in conf):
        return "⚠️ Dead zone — reduce size by 50%"
    return None


def guidance_payload(
    signal: Any,
    *,
    confluence: Optional[Iterable[Any]] = None,
    current_session: Optional[str] = None,
) -> dict[str, Any]:
    entry_low = _value(signal, "entry_low")
    entry_high = _value(signal, "entry_high")
    warning = dead_zone_warning(signal, confluence=confluence, current_session=current_session)
    return {
        "expected_fill_low": float(entry_low) if entry_low is not None else None,
        "expected_fill_high": float(entry_high) if entry_high is not None else None,
        "expected_fill_mid": _entry_mid(signal),
        "fee_adjusted_rr": fee_adjusted_rr(signal),
        "round_trip_friction_pct": round_trip_friction_pct(),
        "skip_level": skip_level(signal),
        "skip_rule": skip_rule(signal),
        "session_warning": warning,
        "size_modifier": DEAD_ZONE_SIZE_MULTIPLIER if warning else 1.0,
    }
