
"""
Risk Governor
-------------
Adjusts risk exposure dynamically based on regime activity.
Safe layer: does NOT block trades, only scales risk multiplier.
"""

from governance.regime_tracker import compute_regime_stats


def get_risk_adjustment(current_regime: str) -> float:
    """Return risk multiplier based on regime participation."""

    stats = compute_regime_stats()

    if not stats or current_regime not in stats:
        return 1.0

    total_trades = sum(v["trades"] for v in stats.values())
    regime_trades = stats[current_regime]["trades"]

    if total_trades == 0:
        return 1.0

    participation = regime_trades / total_trades

    # Conservative adaptive scaling
    if participation > 0.5:
        return 1.1   # active profitable environment assumption
    elif participation < 0.2:
        return 0.7   # reduce exposure
    else:
        return 1.0
