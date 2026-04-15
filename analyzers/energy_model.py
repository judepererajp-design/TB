
"""
Energy Model
------------
Detects compression and expansion probability using:
- ATR contraction
- Range tightening
- Volume contraction
"""

import numpy as np

def compute_energy(highs, lows, closes, volumes):
    if len(closes) < 30:
        return {"compression": False, "energy_score": 0.0}

    ranges = highs - lows
    atr_recent = np.mean(ranges[-14:])
    atr_past = np.mean(ranges[-30:-14])

    vol_recent = np.mean(volumes[-10:])
    vol_past = np.mean(volumes[-30:-10])

    atr_contracting = atr_recent < atr_past * 0.8
    volume_contracting = vol_recent < vol_past * 0.85

    range_tight = np.std(closes[-15:]) < np.std(closes[-30:-15])

    compression = atr_contracting and volume_contracting and range_tight

    energy_score = float(
        (atr_past - atr_recent) / max(atr_past, 1e-6)
    )

    return {
        "compression": compression,
        "energy_score": max(0.0, min(1.0, energy_score))
    }
