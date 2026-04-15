"""
TitanBot Pro — FCN Modules Logger
====================================
Dedicated log file for all Project Inheritance (FCN) features.

Captures every decision, activation, and result from the 11 new modules
in one place so you can quickly verify they are working as intended.

Log file: logs/fcn_modules.log

Usage:
    from config.fcn_logger import fcn_log
    fcn_log("SOURCE_CREDIBILITY", "scored CoinDesk → 0.95", extra={"url": "..."})
    fcn_log("TIME_DECAY", "slow-burn λ=0.03 applied", weight=0.72)
"""

import json
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Any, Dict, Optional

# The dedicated logger — writes ONLY to logs/fcn_modules.log
_logger = logging.getLogger("titanbot.fcn")
_handler_installed = False

FCN_LOG_FILE = "logs/fcn_modules.log"


def _ensure_handler():
    """Lazy-init the file handler (called once on first log)."""
    global _handler_installed
    if _handler_installed:
        return
    try:
        Path(FCN_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            FCN_LOG_FILE,
            maxBytes=5_242_880,   # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handler.setLevel(logging.DEBUG)
        _logger.addHandler(handler)
        _logger.setLevel(logging.DEBUG)
        _logger.propagate = False   # don't echo to main titanbot.log / console
    except Exception:
        pass  # Fall back silently — don't crash the bot for a log setup issue
    _handler_installed = True


def fcn_log(
    feature: str,
    message: str,
    *,
    level: int = logging.INFO,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Log an event from an FCN (Project Inheritance) module.

    Args:
        feature:  Flag name, e.g. "SOURCE_CREDIBILITY", "TIME_DECAY"
        message:  Human-readable description of what happened
        level:    logging level (default INFO)
        extra:    Optional dict of structured data appended as JSON
        **kwargs: Additional key=value pairs merged into the JSON payload
    """
    _ensure_handler()

    payload = {}
    if extra:
        payload.update(extra)
    if kwargs:
        payload.update(kwargs)

    if payload:
        line = f"[{feature}] {message} | {json.dumps(payload, default=str)}"
    else:
        line = f"[{feature}] {message}"

    _logger.log(level, line)
