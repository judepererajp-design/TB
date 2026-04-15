"""Tests for signals.upgrade_tracker."""

import time
from types import SimpleNamespace

from signals.upgrade_tracker import (
    ScanSnapshot,
    SCAN_SNAPSHOT_TTL,
    UpgradeTracker,
)


def _make_scored(*, symbol="BTCUSDT", strategy="SMC", direction="LONG",
                 orderflow=75.0, confidence=68.0, confluence_count=4):
    signal = SimpleNamespace(
        symbol=symbol,
        strategy=strategy,
        direction=direction,
    )
    return SimpleNamespace(
        base_signal=signal,
        orderflow_score=orderflow,
        final_confidence=confidence,
        all_confluence=[f"c{i}" for i in range(confluence_count)],
    )


class TestUpgradeTrackerMomentum:

    def test_falls_back_to_base_proxy_without_previous_scan(self):
        tracker = UpgradeTracker()
        scored = _make_scored(orderflow=75.0, confluence_count=4)

        # Base proxy = avg(orderflow_norm=0.5, confluence_density=0.8)
        assert tracker._score_momentum(scored) == 0.65

    def test_recent_scan_growth_boosts_momentum(self):
        tracker = UpgradeTracker()
        key = "BTCUSDT|SMC|LONG"
        tracker._recent_scans[key] = ScanSnapshot(
            confluence_count=1,
            confidence=58.0,
            orderflow_score=55.0,
            seen_at=time.time(),
        )

        scored = _make_scored(orderflow=80.0, confidence=70.0, confluence_count=4)
        boosted = tracker._score_momentum(scored)

        assert boosted > 0.65

    def test_stale_previous_scan_is_ignored(self):
        tracker = UpgradeTracker()
        key = "BTCUSDT|SMC|LONG"
        tracker._recent_scans[key] = ScanSnapshot(
            confluence_count=1,
            confidence=58.0,
            orderflow_score=55.0,
            seen_at=time.time() - SCAN_SNAPSHOT_TTL - 1,
        )

        scored = _make_scored(orderflow=75.0, confluence_count=4)
        assert tracker._score_momentum(scored) == 0.65
