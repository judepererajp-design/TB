
"""
Execution Confidence
--------------------
Computes execution confidence score using:
- Energy score
- Liquidity reaction
- Orderbook imbalance
- Regime strength
"""

def compute_confidence(market_state: dict, profile: dict) -> int:

    score = 50  # baseline

    energy = market_state.get("energy", {}).get("energy_score", 0)
    if energy:
        score += int(energy * 20)

    imbalance = abs(market_state.get("orderbook", {}).get("imbalance", 0))
    if imbalance > 0.15:
        score += 10

    if (market_state.get("liquidity", {}).get("below_swept") or
            market_state.get("liquidity", {}).get("above_swept")):
        score += 10

    regime = profile.get("regime", "NORMAL")
    if regime == "COMPRESSION":
        score += 5
    elif regime == "EXPANSION":
        score += 8

    return max(0, min(100, score))
