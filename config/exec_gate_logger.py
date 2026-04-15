"""
TitanBot Pro — Execution Gate Logger
======================================
Dedicated log file for every execution gate evaluation.

Captures every evaluate() call with the full factor table, composite score,
decision, and kill combo (if any) in one pasteable file.

Log file: logs/execution_gate.log

Usage:
    from config.exec_gate_logger import exec_gate_log
    exec_gate_log("BTCUSDT", "LONG", score=42.3, action="BLOCK",
                  factors={"session": 55, ...})
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional

# The dedicated logger — writes ONLY to logs/execution_gate.log
_logger = logging.getLogger("titanbot.exec_gate")
_handler_installed = False

EXEC_GATE_LOG_FILE = "logs/execution_gate.log"


def _ensure_handler():
    """Lazy-init the file handler (called once on first log)."""
    global _handler_installed
    if _handler_installed:
        return
    try:
        Path(EXEC_GATE_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            EXEC_GATE_LOG_FILE,
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


def exec_gate_log(
    symbol: str,
    direction: str,
    *,
    score: float = 0.0,
    action: str = "PASS",
    grade: str = "B",
    factors: Optional[Dict[str, float]] = None,
    bad_factors: Optional[list] = None,
    kill_combo: str = "",
    reason: str = "",
    penalty_mult: float = 1.0,
    level: int = logging.INFO,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Log an execution gate evaluation.

    Args:
        symbol:       Trading pair, e.g. "BTCUSDT"
        direction:    "LONG" or "SHORT"
        score:        Composite execution score (0-100)
        action:       "PASS", "PENALIZE", or "BLOCK"
        grade:        Alpha model grade (A+, A, B, C)
        factors:      Per-factor scores dict
        bad_factors:  List of factor names below threshold
        kill_combo:   Kill combo name if fired, empty otherwise
        reason:       Human-readable reason string
        penalty_mult: Confidence multiplier if penalized
        level:        Logging level (default INFO)
        extra:        Additional structured data
    """
    _ensure_handler()

    # Build the main log line
    action_icon = {"BLOCK": "⛔", "PENALIZE": "⚠️", "PASS": "✅"}.get(action, "❓")
    line = f"{action_icon} {symbol} {direction} | score={score:.1f} | {action}"

    if action == "PENALIZE":
        line += f" (×{penalty_mult:.2f})"

    line += f" | grade={grade}"

    if kill_combo:
        line += f" | KILL_COMBO={kill_combo}"

    if bad_factors:
        line += f" | bad=[{', '.join(bad_factors)}]"

    # Factor breakdown on same line for grep-ability
    if factors:
        parts = [f"{k}={v:.0f}" for k, v in factors.items()]
        line += f" | {' '.join(parts)}"

    if reason:
        line += f" | {reason}"

    # Append any extra structured data
    if extra:
        line += f" | {json.dumps(extra, default=str)}"

    _logger.log(level, line)
