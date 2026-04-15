
"""
Execution Journal
-----------------
Records contextual trade data for calibration and performance analysis.
Safe: does NOT affect trading logic.
"""

import json
import os
from datetime import datetime

JOURNAL_FILE = "execution_journal.jsonl"


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

    except Exception:
        # Journal must never break trading
        pass
