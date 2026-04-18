"""Tests for core.missed_fill_tracker (PR5 #4)."""
from __future__ import annotations

import time

import pytest

from core.missed_fill_tracker import MissedFillTracker


# ── Recording ─────────────────────────────────────────────────────

def test_record_missed_validates_inputs():
    t = MissedFillTracker()
    # bad direction
    assert t.record_missed(1, "X/USDT", "FLAT", 1, 2, 0.5, 3) is None
    # zero levels
    assert t.record_missed(2, "X/USDT", "LONG", 0, 2, 0.5, 3) is None
    assert t.record_missed(3, "X/USDT", "LONG", 1, 2, 0, 3) is None
    assert t.record_missed(4, "X/USDT", "LONG", 1, 2, 0.5, 0) is None


def test_record_missed_idempotent():
    t = MissedFillTracker()
    rec = t.record_missed(10, "BTC/USDT", "LONG", 100, 101, 99, 105)
    assert rec is not None
    again = t.record_missed(10, "BTC/USDT", "LONG", 100, 101, 99, 105)
    assert again is rec
    assert len(t._active) == 1


# ── Outcome resolution ────────────────────────────────────────────

def test_long_hits_tp1_after_late_entry():
    t = MissedFillTracker()
    t.record_missed(1, "BTC/USDT", "LONG",
                    entry_low=100, entry_high=101,
                    stop_loss=98, tp1=105)
    # Tick above the zone first (no entry yet)
    t.update_with_price("BTC/USDT", 102.0)
    rec = t.get_record(1)
    assert rec.outcome == "PENDING"
    assert rec.entered is False

    # Now price drops back into the zone — late entry
    finalised = t.update_with_price("BTC/USDT", 100.5)
    assert rec.entered is True
    assert finalised == []  # not yet resolved

    # Then TP1 fires
    finalised = t.update_with_price("BTC/USDT", 105.5)
    assert rec.outcome == "HIT_TP1"
    assert rec in finalised


def test_long_hits_sl_first_after_entry():
    t = MissedFillTracker()
    t.record_missed(2, "BTC/USDT", "LONG",
                    entry_low=100, entry_high=101,
                    stop_loss=98, tp1=105)
    t.update_with_price("BTC/USDT", 100.0)   # enter
    t.update_with_price("BTC/USDT", 97.0)    # SL
    rec = t.get_record(2)
    assert rec.outcome == "HIT_SL_FIRST"


def test_short_outcomes_mirror_long():
    t = MissedFillTracker()
    t.record_missed(3, "ETH/USDT", "SHORT",
                    entry_low=100, entry_high=101,
                    stop_loss=103, tp1=95)
    t.update_with_price("ETH/USDT", 100.5)   # enter
    t.update_with_price("ETH/USDT", 94.0)    # TP
    assert t.get_record(3).outcome == "HIT_TP1"

    t.record_missed(4, "ETH/USDT", "SHORT",
                    entry_low=100, entry_high=101,
                    stop_loss=103, tp1=95)
    t.update_with_price("ETH/USDT", 100.5)
    t.update_with_price("ETH/USDT", 104.0)   # SL
    assert t.get_record(4).outcome == "HIT_SL_FIRST"


def test_window_expires_with_never_entered():
    t = MissedFillTracker(eval_window_secs=1)
    rec = t.record_missed(5, "X/USDT", "LONG", 100, 101, 99, 105)
    # tick that never reaches the zone
    t.update_with_price("X/USDT", 95.0)
    assert rec.outcome == "PENDING"
    # Force window to elapse
    time.sleep(1.1)
    t.update_with_price("X/USDT", 95.0)
    assert rec.outcome == "NEVER_ENTERED"


def test_window_expires_after_entry_no_resolution():
    t = MissedFillTracker(eval_window_secs=1)
    rec = t.record_missed(6, "X/USDT", "LONG", 100, 101, 99, 105)
    t.update_with_price("X/USDT", 100.5)   # enter
    assert rec.entered is True
    # Tick stays inside the zone, neither TP nor SL
    time.sleep(1.1)
    t.update_with_price("X/USDT", 100.5)
    assert rec.outcome == "NO_DATA"


def test_purge_stale_finalises_inactive_records():
    t = MissedFillTracker(eval_window_secs=0)  # immediate
    t.record_missed(7, "X/USDT", "LONG", 100, 101, 99, 105)
    assert len(t._active) == 1
    purged = t.purge_stale()
    assert purged == 1
    assert len(t._active) == 0
    assert t.get_record(7).outcome == "NEVER_ENTERED"


# ── Stats / dashboard ─────────────────────────────────────────────

def test_get_stats_summary():
    t = MissedFillTracker(eval_window_secs=10_000)
    # 2 wins, 1 loss, 1 never-entered
    for sid, prices in [
        (1, [100.5, 105.5]),  # win
        (2, [100.5, 105.5]),  # win
        (3, [100.5, 97.0]),   # loss
    ]:
        t.record_missed(sid, "BTC/USDT", "LONG", 100, 101, 98, 105)
        for p in prices:
            t.update_with_price("BTC/USDT", p)

    # never-entered:
    t.record_missed(4, "ETH/USDT", "LONG", 200, 201, 198, 205)
    rec = t.get_record(4)
    rec.expired_at = time.time() - 10_001  # force window expiry
    t.update_with_price("ETH/USDT", 195.0)

    stats = t.get_stats(window_hours=24)
    assert stats["total"] == 4
    assert stats["would_have_won_pct"] == 50.0
    assert stats["would_have_lost_pct"] == 25.0
    assert stats["never_entered_pct"] == 25.0


def test_to_dict_snapshot():
    t = MissedFillTracker()
    t.record_missed(1, "X/USDT", "LONG", 100, 101, 99, 105)
    d = t.to_dict()
    assert d["active_count"] == 1
    assert d["finalised_count"] == 0
    assert "stats_24h" in d
