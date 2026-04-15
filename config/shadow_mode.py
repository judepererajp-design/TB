"""
TitanBot Pro — Shadow Mode Logger
====================================
Runs new logic in parallel with the live pipeline, logs what the new
system WOULD have done vs what the old system DID, without affecting
live trading signals or confidence scores.

Usage:
    from config.shadow_mode import shadow_log
    shadow_log("SOURCE_CREDIBILITY", {
        "headline": title,
        "live_weight": 1.0,
        "shadow_weight": 0.85,
        "source": "CoinTelegraph",
    })
"""

import json
import logging
import time
from typing import Any, Dict

logger = logging.getLogger("titanbot.shadow")

# Dedicated shadow mode logger — writes to shadow_mode.log
_shadow_handler = None


def _ensure_handler():
    """Lazy-init the file handler for shadow mode logs."""
    global _shadow_handler
    if _shadow_handler is not None:
        return
    try:
        _shadow_handler = logging.FileHandler("logs/shadow_mode.log", mode="a")
        _shadow_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s")
        )
        logger.addHandler(_shadow_handler)
        logger.setLevel(logging.INFO)
    except Exception:
        # If we can't create the file handler, fall back to standard logging
        _shadow_handler = True  # Prevent retries


def shadow_log(feature: str, data: Dict[str, Any]):
    """
    Log a shadow mode comparison entry.

    Args:
        feature: The feature flag name (e.g. "SOURCE_CREDIBILITY")
        data: Dict with comparison data — must include both 'live' and
              'shadow' values so humans can compare.
    """
    _ensure_handler()
    entry = {
        "ts": time.time(),
        "feature": feature,
        **data,
    }
    logger.info(json.dumps(entry, default=str))
