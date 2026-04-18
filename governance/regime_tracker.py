
"""
Regime Performance Tracker
---------------------------
Analyzes execution_journal.jsonl and computes performance by regime.
Non-blocking, safe analytics layer.
"""

import json
import logging
import os
from collections import defaultdict

JOURNAL_FILE = "execution_journal.jsonl"

logger = logging.getLogger(__name__)


def compute_regime_stats():
    if not os.path.exists(JOURNAL_FILE):
        return {}

    # G-8 FIX: remove unnecessary defaultdict — stats is only populated inside
    # the try block and returned as a plain dict, so defaultdict adds no value
    # and the special repr could confuse downstream JSON serialisation.
    stats: dict = {}

    try:
        with open(JOURNAL_FILE, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    regime = entry.get("regime", "UNKNOWN")
                    if regime not in stats:
                        stats[regime] = {"trades": 0}
                    stats[regime]["trades"] += 1
                except Exception as exc:
                    # G-8 FIX: log malformed lines instead of silently skipping
                    logger.debug("regime_tracker: skipping malformed journal line: %s", exc)
                    continue
    except Exception as exc:
        # G-8 FIX: log file-read failures so operators know journal is unreadable
        logger.warning("regime_tracker.compute_regime_stats: failed to read journal: %s", exc)
        return {}

    return stats
