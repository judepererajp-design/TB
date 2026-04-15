"""Tests for core.extended_health."""

from core.extended_health import ExtendedHealthMonitor


class TestExtendedHealthMonitor:
    """Unit tests for ExtendedHealthMonitor."""

    def _make_monitor(self):
        return ExtendedHealthMonitor()

    # ── recording ───────────────────────────────────────────

    def test_record_db_latency(self):
        m = self._make_monitor()
        m.record_db_latency("insert_signal", 50.0)
        report = m.get_report()
        assert report["database"].total_calls == 1

    def test_record_api_latency(self):
        m = self._make_monitor()
        m.record_api_latency("fetch_ohlcv", 150.0)
        report = m.get_report()
        assert report["api"].total_calls == 1

    def test_record_telegram_latency(self):
        m = self._make_monitor()
        m.record_telegram_latency(200.0)
        report = m.get_report()
        assert report["telegram"].total_calls == 1

    # ── report computation ──────────────────────────────────

    def test_empty_report_returns_no_data(self):
        m = self._make_monitor()
        report = m.get_report()
        for sub in report.values():
            assert sub.status == "NO_DATA"

    def test_healthy_status(self):
        m = self._make_monitor()
        for _ in range(10):
            m.record_db_latency("read", 10.0, success=True)
        report = m.get_report()
        assert report["database"].status == "HEALTHY"

    def test_unhealthy_on_high_error_rate(self):
        m = self._make_monitor()
        for _ in range(10):
            m.record_api_latency("fetch", 50.0, success=False)
        report = m.get_report()
        assert report["api"].status == "UNHEALTHY"

    def test_degraded_on_slow_ops(self):
        m = self._make_monitor()
        # >20% of ops are slow (> 500ms threshold for DB)
        for _ in range(3):
            m.record_db_latency("fast", 10.0)
        for _ in range(7):
            m.record_db_latency("slow", 600.0)
        report = m.get_report()
        assert report["database"].status == "DEGRADED"

    def test_p95_latency_computed(self):
        m = self._make_monitor()
        for i in range(20):
            m.record_db_latency("op", float(i + 1))
        report = m.get_report()
        assert report["database"].p95_latency_ms > 0

    def test_avg_latency_computed(self):
        m = self._make_monitor()
        m.record_api_latency("call", 100.0)
        m.record_api_latency("call", 200.0)
        report = m.get_report()
        assert report["api"].avg_latency_ms == 150.0

    # ── convenience methods ─────────────────────────────────

    def test_get_summary_returns_status_strings(self):
        m = self._make_monitor()
        summary = m.get_summary()
        assert set(summary.keys()) == {"database", "api", "telegram"}
        for v in summary.values():
            assert isinstance(v, str)

    def test_is_healthy_with_no_data(self):
        m = self._make_monitor()
        assert m.is_healthy() is True

    def test_is_healthy_false_when_unhealthy(self):
        m = self._make_monitor()
        for _ in range(10):
            m.record_api_latency("fail", 50.0, success=False)
        assert m.is_healthy() is False
