
"""
Execution Journal
-----------------
Records contextual trade data for calibration and performance analysis.
Safe: does NOT affect trading logic.
"""

import json
import logging
import os
from datetime import datetime

JOURNAL_FILE = "execution_journal.jsonl"

logger = logging.getLogger(__name__)


def _ensure_file():
    if not os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "w") as f:
            pass


def log_execution(entry: dict):
    """Append execution record to journal."""
    try:
        _ensure_file()

        entry["timestamp"] = datetime.utcnow().isoformat()

        with open(JOURNAL_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

    except Exception as exc:
        # G-8 FIX: was a bare `except Exception: pass` — disk-full and
        # permission errors were silently swallowed, creating blind spots in
        # loss classification.  Log at WARNING so operators see the failure
        # without it ever disrupting trading logic.
        logger.warning("execution_journal.log_execution failed (non-fatal): %s", exc)
