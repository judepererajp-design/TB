
"""
Regime Performance Tracker
---------------------------
Analyzes execution_journal.jsonl and computes performance by regime.
Non-blocking, safe analytics layer.
"""

import json
import os
from collections import defaultdict

JOURNAL_FILE = "execution_journal.jsonl"


def compute_regime_stats():
    if not os.path.exists(JOURNAL_FILE):
        return {}

    stats = defaultdict(lambda: {"trades": 0})

    try:
        with open(JOURNAL_FILE, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    regime = entry.get("regime", "UNKNOWN")
                    stats[regime]["trades"] += 1
                except Exception:
                    continue
    except Exception:
        return {}

    return dict(stats)
