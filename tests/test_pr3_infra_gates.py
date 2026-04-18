"""Tests for signals.exchange_health, signals.stablecoin_depeg, and
the regime-transition hysteresis helpers added to RegimeAnalyzer."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from config.constants import (
    ExchangeHealth as EHC,
    StablecoinDepeg as SDC,
    RegimeHysteresis as RH,
)


# ── Exchange health monitor ────────────────────────────────────

class _FakeAPI:
    def __init__(self, latency_ms=10.0, total_requests=100,
                 total_errors=0, is_healthy=True):
        self._latency_ms = latency_ms
        self._total_requests = total_requests
        self._total_errors = total_errors
        self.is_healthy = is_healthy

    def get_request_stats(self):
        return {
            "avg_latency_ms": self._latency_ms,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
        }


def _fresh_monitor():
    # Re-import to grab the singleton, then reset state
    from signals.exchange_health import exchange_health, HealthStatus
    exchange_health._consecutive_failures = 0
    exchange_health._last_recovery_ts = 0.0
    exchange_health._last_status = HealthStatus.HEALTHY
    return exchange_health


class TestExchangeHealthMonitor:
    def test_healthy_when_telemetry_clean(self):
        m = _fresh_monitor()
        with patch("data.api_client.api", _FakeAPI()):
            snap = m.check_health()
        assert snap.status.value == "HEALTHY"
        assert snap.should_block is False

    def test_unhealthy_on_consecutive_publish_failures(self):
        m = _fresh_monitor()
        for _ in range(EHC.UNHEALTHY_CONSEC_FAILS):
            m.record_failure()
        with patch("data.api_client.api", _FakeAPI()):
            snap = m.check_health()
        assert snap.status.value == "UNHEALTHY"
        assert snap.should_block is True

    def test_record_success_resets_failures(self):
        m = _fresh_monitor()
        m.record_failure()
        m.record_failure()
        m.record_success()
        with patch("data.api_client.api", _FakeAPI()):
            snap = m.check_health()
        # Reset counter; recovery grace window puts us in DEGRADED
        assert snap.status.value in ("HEALTHY", "DEGRADED")
        assert snap.should_block is False

    def test_unhealthy_on_high_latency(self):
        m = _fresh_monitor()
        with patch("data.api_client.api",
                   _FakeAPI(latency_ms=EHC.UNHEALTHY_AVG_LATENCY_MS + 1)):
            snap = m.check_health()
        assert snap.status.value == "UNHEALTHY"

    def test_unhealthy_when_api_flag_false(self):
        m = _fresh_monitor()
        # is_healthy=False is only honoured once the client has actually
        # made calls, so make sure total_requests > 0 in the fake.
        with patch("data.api_client.api",
                   _FakeAPI(total_requests=10, is_healthy=False)):
            snap = m.check_health()
        assert snap.status.value == "UNHEALTHY"

    def test_api_flag_ignored_at_cold_start(self):
        m = _fresh_monitor()
        # No requests yet → flag is informational only
        with patch("data.api_client.api",
                   _FakeAPI(total_requests=0, is_healthy=False)):
            snap = m.check_health()
        assert snap.status.value == "HEALTHY"

    def test_degraded_on_moderate_error_rate(self):
        m = _fresh_monitor()
        with patch("data.api_client.api",
                   _FakeAPI(total_requests=100,
                            total_errors=int(EHC.DEGRADED_ERROR_RATE * 100) + 1)):
            snap = m.check_health()
        assert snap.status.value == "DEGRADED"
        assert snap.should_block is False


# ── Stablecoin depeg guard ─────────────────────────────────────

def _fresh_depeg_guard():
    from signals.stablecoin_depeg import stablecoin_depeg_guard
    stablecoin_depeg_guard._cache.clear()
    stablecoin_depeg_guard._last_status.clear()
    return stablecoin_depeg_guard


class TestStablecoinDepegGuard:
    def test_extract_quote_with_slash(self):
        from signals.stablecoin_depeg import _extract_quote
        assert _extract_quote("BTC/USDT") == "USDT"
        assert _extract_quote("ETH/USDC:USDC") == "USDC"

    def test_extract_quote_concatenated(self):
        from signals.stablecoin_depeg import _extract_quote
        assert _extract_quote("BTCUSDT") == "USDT"
        assert _extract_quote("ETHUSDC") == "USDC"

    async def test_unmonitored_quote_is_healthy(self):
        g = _fresh_depeg_guard()
        snap = await g.check_quote("FOO")
        assert snap.status.value == "HEALTHY"

    async def test_fail_open_when_price_unavailable(self):
        g = _fresh_depeg_guard()

        async def _none_ticker(*_a, **_kw):
            return None

        with patch("data.api_client.api") as api_mock:
            api_mock.fetch_ticker = _none_ticker
            snap = await g.check_quote("USDT")
        assert snap.status.value == "HEALTHY"
        assert snap.should_block is False
        assert "fail-open" in snap.reason

    async def test_unhealthy_when_deviation_exceeds_threshold(self):
        g = _fresh_depeg_guard()
        # Probe pair USDC/USDT priced at 1.025 → USDT_usd = 1/1.025 ≈ 0.9756,
        # |dev|≈0.0244 ≥ UNHEALTHY (0.020).
        async def _fake_ticker(pair):
            return {"last": 1.025} if pair == "USDC/USDT" else None

        with patch("data.api_client.api") as api_mock:
            api_mock.fetch_ticker = _fake_ticker
            snap = await g.check_quote("USDT")
        assert snap.status.value == "UNHEALTHY"
        assert snap.should_block is True
        assert snap.deviation is not None
        assert snap.deviation >= SDC.UNHEALTHY_DEVIATION

    async def test_degraded_in_middle_band(self):
        g = _fresh_depeg_guard()
        # USDT_usd = 1/1.010 ≈ 0.9901, deviation ≈ 0.0099 → DEGRADED but not UNHEALTHY
        async def _fake_ticker(pair):
            return {"last": 1.010} if pair == "USDC/USDT" else None

        with patch("data.api_client.api") as api_mock:
            api_mock.fetch_ticker = _fake_ticker
            snap = await g.check_quote("USDT")
        assert snap.status.value == "DEGRADED"
        assert snap.should_block is False

    async def test_should_block_symbol_routes_to_quote(self):
        g = _fresh_depeg_guard()

        async def _fake_ticker(pair):
            return {"last": 1.030} if pair == "USDC/USDT" else None

        with patch("data.api_client.api") as api_mock:
            api_mock.fetch_ticker = _fake_ticker
            blocked = await g.should_block_symbol("BTC/USDT")
        assert blocked is True


# ── Regime hysteresis helpers ──────────────────────────────────

class TestRegimeHysteresis:
    def test_no_transition_returns_none(self):
        from analyzers.regime import RegimeAnalyzer
        r = RegimeAnalyzer()
        assert r.time_since_last_transition() is None
        assert r.is_recently_transitioned(RH.TRANSITION_HYSTERESIS_SECS) is False

    def test_recent_transition_detected(self):
        from analyzers.regime import RegimeAnalyzer
        r = RegimeAnalyzer()
        r._last_regime_change_at = time.time() - 30  # 30s ago
        elapsed = r.time_since_last_transition()
        assert elapsed is not None and 25 <= elapsed <= 60
        assert r.is_recently_transitioned(within_secs=600) is True
        assert r.is_recently_transitioned(within_secs=10) is False

    def test_old_transition_not_recent(self):
        from analyzers.regime import RegimeAnalyzer
        r = RegimeAnalyzer()
        r._last_regime_change_at = time.time() - 3600  # 1h ago
        assert r.is_recently_transitioned(within_secs=RH.TRANSITION_HYSTERESIS_SECS) is False
