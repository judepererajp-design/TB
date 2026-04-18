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
import logging.handlers
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("titanbot.shadow")

# Dedicated shadow mode logger — writes to shadow_mode.log
_shadow_handler = None
_setup_failed = False

SHADOW_LOG_FILE = "logs/shadow_mode.log"


def _ensure_handler():
    """Lazy-init the rotating file handler for shadow mode logs."""
    global _shadow_handler, _setup_failed
    if _shadow_handler is not None or _setup_failed:
        return
    try:
        Path(SHADOW_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            SHADOW_LOG_FILE,
            maxBytes=5_242_880,   # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Shadow entries are high-volume JSON; keep them out of the main log.
        logger.propagate = False
        _shadow_handler = handler
    except Exception:
        # Disable further attempts but leave the handler reference as None so
        # the sentinel can never be confused with a real handler object.
        _setup_failed = True


def shadow_log(feature: str, data: Dict[str, Any]):
    """
    Log a shadow mode comparison entry.

    Args:
        feature: The feature flag name (e.g. "SOURCE_CREDIBILITY")
        data: Dict with comparison data — must include both 'live' and
              'shadow' values so humans can compare.
    """
    _ensure_handler()
    if _shadow_handler is None:
        # Setup failed; skip silently rather than polluting the root logger.
        return
    entry = {
        "ts": time.time(),
        "feature": feature,
        **data,
    }
    logger.info(json.dumps(entry, default=str))
