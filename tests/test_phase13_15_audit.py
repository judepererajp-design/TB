"""
Tests for deep audits 13-15.

Covers:
  - BetaPosterior count uses dynamic default prior mass
  - Database save_signal persists ensemble_votes
  - AdaptiveWeightManager uses per-source ensemble_votes attribution
"""

import json
import sys
import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestBetaPosteriorCount:
    """Audit 13: count must derive from the current default prior mass."""

    def test_count_uses_runtime_default_prior_mass(self):
        from core.probability_engine import BetaPosterior

        fresh = BetaPosterior()
        prior_mass = fresh.alpha + fresh.beta

        posterior = BetaPosterior(
            alpha=fresh.alpha + 4,
            beta=fresh.beta + 6,
        )

        assert posterior.count == int((posterior.alpha + posterior.beta) - prior_mass)
        assert posterior.count == 10

    def test_count_clamps_below_prior_mass(self):
        from core.probability_engine import BetaPosterior

        posterior = BetaPosterior(alpha=1.0, beta=1.0)
        assert posterior.count == 0


class _FakeWriterCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def writer(self):
        return _FakeWriterCtx(self._conn)


class _FakeCursor:
    def __init__(self, lastrowid=123):
        self.lastrowid = lastrowid
        self.closed = False

    async def close(self):
        self.closed = True


class _FakeConn:
    def __init__(self):
        self.calls = []
        self.committed = False
        self.cursor = _FakeCursor()

    async def execute(self, sql, params):
        self.calls.append((sql, params))
        return self.cursor

    async def commit(self):
        self.committed = True


class TestSignalDatabaseEnsembleVotes:
    """Audit 14: ensemble_votes must survive save_signal()."""

    @pytest.mark.asyncio
    async def test_save_signal_persists_ensemble_votes_json(self):
        # aiosqlite is available as a real dependency (see requirements.txt).
        # Previously this used sys.modules.setdefault(..., MagicMock()) which
        # poisoned downstream tests that actually need the real module.
        import aiosqlite  # noqa: F401
        sys.modules.setdefault("data.connection_pool", MagicMock(ConnectionPool=MagicMock()))
        # Resolve relative to this test file so the test works regardless of
        # where the repository is checked out (was previously hard-coded to
        # /home/runner/work/Titanbot/Titanbot/...).
        module_path = Path(__file__).resolve().parents[1] / "data" / "database.py"
        spec = importlib.util.spec_from_file_location("audit_database_module", module_path)
        dbmod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(dbmod)
        Database = dbmod.Database

        db = object.__new__(Database)
        db._initialized = True
        db._pool = None
        db._lock = asyncio.Lock()
        conn = _FakeConn()

        db._pool = _FakePool(conn)
        signal_id = await db.save_signal({
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "strategy": "SmartMoneyConcepts",
            "confidence": 80,
            "entry_low": 100.0,
            "entry_high": 101.0,
            "stop_loss": 95.0,
            "tp1": 105.0,
            "tp2": 110.0,
            "tp3": 115.0,
            "rr_ratio": 2.5,
            "timeframe": "15m",
            "entry_timeframe": "15m",
            "regime": "BULL_TREND",
            "sector": "L1",
            "tier": 1,
            "ensemble_votes": {
                "cvd": {"value": 1, "weight": 1.8},
                "smart_money": {"value": -1, "weight": 2.0},
            },
        })

        assert signal_id == 123
        sql, params = conn.calls[0]
        assert "ensemble_votes" in sql
        assert params[-1] == json.dumps({
            "cvd": {"value": 1, "weight": 1.8},
            "smart_money": {"value": -1, "weight": 2.0},
        })


class TestAdaptiveWeightsPerSourceAttribution:
    """Audit 15: use persisted per-source votes instead of ensemble-only proxy."""

    def test_support_vs_oppose_votes_score_differently(self):
        from signals.adaptive_weights import AdaptiveWeightManager

        mgr = AdaptiveWeightManager()
        signals = [{
            "outcome": "LOSS",
            "pnl_r": -1.0,
            "ensemble_votes": {
                "cvd": {"value": 1, "weight": 1.8},
                "smart_money": {"value": -1, "weight": 2.0},
            },
        }] * 10

        cvd_score = mgr._compute_source_score("cvd", "BULL_TREND", signals)
        sm_score = mgr._compute_source_score("smart_money", "BULL_TREND", signals)

        assert cvd_score.win_rate == 0.0
        assert sm_score.win_rate == 1.0
        assert sm_score.avg_return == 1.0
        assert sm_score.score > cvd_score.score

    def test_json_string_ensemble_votes_are_supported(self):
        from signals.adaptive_weights import AdaptiveWeightManager

        mgr = AdaptiveWeightManager()
        signals = [{
            "outcome": "WIN",
            "pnl_r": 2.0,
            "ensemble_votes": json.dumps({
                "whale_flow": {"value": -1, "weight": 1.2},
            }),
        }] * 5

        score = mgr._compute_source_score("whale_flow", "VOLATILE", signals)

        assert score.win_rate == 0.0
        assert score.sample_count == 5

    def test_legacy_rows_without_ensemble_votes_fall_back(self):
        from signals.adaptive_weights import AdaptiveWeightManager

        mgr = AdaptiveWeightManager()
        signals = [
            {"outcome": "WIN", "pnl_r": 1.5},
            {"outcome": "LOSS", "pnl_r": -1.0},
        ] * 5

        score = mgr._compute_source_score("cvd", "NEUTRAL", signals)

        assert score.win_rate == 0.5
        assert score.sample_count == 10
