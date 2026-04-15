
"""
Market Brain
------------
Central controller translating market state into execution behavior.
"""

def build_market_profile(market_state: dict) -> dict:

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
