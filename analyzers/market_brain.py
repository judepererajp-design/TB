
"""
Market Brain
------------
Central controller translating market state into execution behavior.
"""

def build_market_profile(market_state: dict) -> dict:
    """Translate the live intelligence bus into execution knobs.

    AUDIT FIX: if ``market_state`` is missing or lacks the core ``energy`` /
    ``orderbook`` sub-dicts (typical during an API outage where
    ``build_market_state`` returns ``{}``) we surface an ``UNKNOWN`` regime
    with a conservative 0.8 risk multiplier rather than silently returning
    NORMAL + 1.0× sizing.  Prior behaviour made cached-empty responses
    indistinguishable from a genuinely neutral market.
    """
    if not market_state or (
        "energy" not in market_state and "orderbook" not in market_state
    ):
        return {
            "regime": "UNKNOWN",
            "confirmation_required": 2,
            "risk_multiplier": 0.8,
        }

    compression = market_state.get("energy", {}).get("compression", False)
    imbalance = market_state.get("orderbook", {}).get("imbalance", 0)

    if compression:
        regime = "COMPRESSION"
        confirmations = 1
        risk_multiplier = 0.8

    elif abs(imbalance) > 0.25:
        regime = "EXPANSION"
        confirmations = 1
        risk_multiplier = 1.1

    else:
        regime = "NORMAL"
        confirmations = 2
        risk_multiplier = 1.0

    return {
        "regime": regime,
        "confirmation_required": confirmations,
        "risk_multiplier": risk_multiplier,
    }
