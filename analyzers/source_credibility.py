"""
TitanBot Pro — Source Credibility Scoring
===========================================
Project Inheritance Feature 1.

Assigns a reliability multiplier (0.8–1.0) to each news source.
Applied per-headline in get_symbol_sentiment_score() to scale
sentiment weight based on source credibility.

Conservative baseline at launch — all sources in 0.80–1.00 range.
Future: dynamic scoring based on actual prediction accuracy.

Fallback: if source is unknown, uses DEFAULT_CREDIBILITY (0.85).
"""

import logging
from typing import Dict, Optional

from config.constants import NewsIntelligence
from config.feature_flags import ff
from config.shadow_mode import shadow_log

logger = logging.getLogger(__name__)


# ── Static baseline credibility scores ─────────────────────────
# Conservative range: 0.80–1.00. Extremes (0.6, 1.3) are reserved
# for the future dynamic system after 30 days of shadow mode data.
_SOURCE_SCORES: Dict[str, float] = {
    # Tier 1: Major crypto media (highest signal quality)
    "CoinTelegraph":    0.92,
    "CoinDesk":         0.95,
    "Decrypt":          0.93,
    "TheBlock":         0.96,
    "Blockworks":       0.94,
    # Tier 2: Good quality, slightly more noise
    "CryptoSlate":      0.88,
    "BeInCrypto":       0.86,
    "NewsBTC":          0.84,
    "AMBCrypto":        0.84,
    "CoinGape":         0.83,
    # Tier 3: Exchange blogs (biased toward own ecosystem)
    "BinanceBlog":      0.85,
    "BitcoinMagazine":  0.90,
    # Tier 4: Higher noise ratio
    "U.Today":          0.82,
    "CryptoBriefing":   0.86,
    "InsideBitcoins":   0.81,
    # API source
    "free-crypto-news": 0.85,
}

DEFAULT_CREDIBILITY: float = 0.85


def get_source_credibility(source: str) -> float:
    """
    Returns the credibility multiplier for a given news source.

    Args:
        source: Source name as it appears in RSS feed / API data.

    Returns:
        Float between SOURCE_CREDIBILITY_FLOOR (0.80) and
        SOURCE_CREDIBILITY_CEILING (1.00).
    """
    score = _SOURCE_SCORES.get(source, DEFAULT_CREDIBILITY)
    # Clamp to configured bounds
    return max(
        NewsIntelligence.SOURCE_CREDIBILITY_FLOOR,
        min(NewsIntelligence.SOURCE_CREDIBILITY_CEILING, score),
    )


def apply_source_credibility(
    source: str,
    raw_weight: float,
) -> float:
    """
    Apply source credibility multiplier to a headline weight.
    If feature flag is off, returns raw_weight unchanged.
    If in shadow mode, logs comparison but returns raw_weight.

    Args:
        source: The news source name.
        raw_weight: The original sentiment weight (e.g. 1.0, -1.0).

    Returns:
        Adjusted weight.
    """
    credibility = get_source_credibility(source)
    adjusted = raw_weight * credibility

    if ff.is_shadow("SOURCE_CREDIBILITY"):
        shadow_log("SOURCE_CREDIBILITY", {
            "source": source,
            "credibility": credibility,
            "live_weight": raw_weight,
            "shadow_weight": adjusted,
        })
        from config.fcn_logger import fcn_log
        fcn_log("SOURCE_CREDIBILITY", f"shadow | {source} cred={credibility:.2f} live_w={raw_weight:.3f} shadow_w={adjusted:.3f}")
        return raw_weight  # Shadow: don't affect live

    if ff.is_enabled("SOURCE_CREDIBILITY"):
        from config.fcn_logger import fcn_log
        fcn_log("SOURCE_CREDIBILITY", f"live | {source} cred={credibility:.2f} raw={raw_weight:.3f} → adjusted={adjusted:.3f}")
        return adjusted

    return raw_weight  # Feature off


def get_all_scores() -> Dict[str, float]:
    """Return a copy of all source credibility scores (for dashboards)."""
    return dict(_SOURCE_SCORES)
