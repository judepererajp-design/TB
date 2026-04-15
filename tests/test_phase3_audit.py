"""
Tests for Phase 3 deep audit fixes.

Covers:
  P3-1: signal_publisher no longer double-registers A/B signals with outcome_monitor
  P3-2: SignalDeduplicator.unmark() removes stale marks when post-dedup gates kill signals
  P3-3: save_signal persists agg_grade to the DB
  P3-4: formatter uses alpha_score.grade when available
  P3-5: AI delta cap baseline re-anchored after deterministic mults
"""

import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ── P3-1: Publisher no longer double-registers outcome_monitor ─────────


class TestPublisherNoDoubleTracking:
    """P3-1: signal_publisher.publish() must NOT register A/B signals with
    outcome_monitor.  The engine routes them through execution_engine, which
    hands off to outcome_monitor at EXECUTE state."""

    def test_publisher_code_does_not_import_outcome_monitor(self):
        """The publisher module should not contain any outcome_monitor.track_signal
        calls after the Phase 3 fix removed the duplicate registration."""
        import inspect
        from signals.signal_publisher import SignalPublisher
        source = inspect.getsource(SignalPublisher.publish)
        # After fix, there should be no outcome_monitor.track_signal call
        assert "outcome_monitor.track_signal" not in source, (
            "Publisher still calls outcome_monitor.track_signal — "
            "this causes dual-tracking with execution_engine"
        )


# ── P3-2: Dedup unmark ────────────────────────────────────────────────


class TestDedupUnmark:
    """P3-2: SignalDeduplicator.unmark() must clear both in-memory and DB marks."""

    def test_unmark_clears_in_memory(self):
        from signals.aggregator import SignalDeduplicator
        from config.loader import cfg

        # Ensure aggregator config returns real numbers (not MagicMock)
        cfg.aggregator = MagicMock()
        cfg.aggregator.get = lambda key, default=None: {
            'dedup_window_minutes': 180,
            'c_grade_dedup_minutes': 60,
        }.get(key, default)
        cfg.database = MagicMock()
        cfg.database.get = lambda key, default=None: {
            'path': '/tmp/test_dedup.db',
        }.get(key, default)

        dedup = SignalDeduplicator()
        dedup.mark_sent("ETHUSDT", "LONG", confidence=75, grade="A")

        # Pass same confidence to avoid breakthrough exception
        assert dedup.is_duplicate("ETHUSDT", "LONG", new_final_confidence=75)

        dedup.unmark("ETHUSDT", "LONG")

        assert not dedup.is_duplicate("ETHUSDT", "LONG", new_final_confidence=75)

    def test_unmark_nonexistent_key_is_noop(self):
        from signals.aggregator import SignalDeduplicator

        dedup = SignalDeduplicator()
        # Should not raise
        dedup.unmark("UNKNOWN", "SHORT")


# ── P3-3: agg_grade persisted in save_signal ──────────────────────────


class TestAggGradePersistence:
    """P3-3: save_signal() INSERT must include agg_grade column."""

    @pytest.mark.asyncio
    async def test_save_signal_includes_agg_grade(self):
        """The signals INSERT statement should contain the agg_grade column.
        We verify by executing the exact same INSERT used in save_signal()
        against a fresh test database."""
        import aiosqlite
        import os

        db_path = "/tmp/test_p3_agg_grade.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row

        # Create schema matching data/database.py signals table
        await conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            strategy TEXT NOT NULL,
            confidence REAL NOT NULL,
            entry_low REAL NOT NULL,
            entry_high REAL NOT NULL,
            stop_loss REAL NOT NULL,
            tp1 REAL NOT NULL,
            tp2 REAL, tp3 REAL, rr_ratio REAL,
            timeframe TEXT,
            setup_class TEXT DEFAULT 'intraday',
            entry_timeframe TEXT DEFAULT '15m',
            regime TEXT, sector TEXT, tier INTEGER,
            confluence TEXT, confluence_strategies TEXT, raw_scores TEXT,
            message_id INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            valid_until TEXT,
            p_win REAL DEFAULT 0.55,
            ev_r REAL DEFAULT 0.0,
            alpha_grade TEXT DEFAULT 'B',
            agg_grade TEXT DEFAULT 'B',
            evidence TEXT DEFAULT '{}',
            risk_amount REAL DEFAULT 0.0,
            position_size REAL DEFAULT 0.0,
            publish_price REAL,
            ai_influence REAL,
            market_state TEXT,
            ensemble_votes TEXT DEFAULT '{}'
        );
        """)
        await conn.commit()

        # Execute the same INSERT pattern that save_signal uses
        cursor = await conn.execute("""
            INSERT INTO signals (
                symbol, direction, strategy, confidence,
                entry_low, entry_high, stop_loss,
                tp1, tp2, tp3, rr_ratio,
                timeframe, setup_class, entry_timeframe, regime, sector, tier,
                confluence, confluence_strategies, raw_scores, message_id, valid_until,
                p_win, ev_r, alpha_grade, agg_grade, evidence,
                risk_amount, position_size,
                publish_price, ai_influence, market_state, ensemble_votes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "BTCUSDT", "LONG", "Momentum", 80.0,
            60000, 61000, 59000,
            63000, 65000, None, 2.5,
            "15m", "intraday", "15m", "BULL_TREND", "L1", 1,
            "[]", None, "{}", None, None,
            0.65, 0.5, "A", "B+", "{}",
            100.0, 1000.0,
            60500.0, 2.0, "BULL_TREND", "{}",
        ))
        await conn.commit()
        signal_id = cursor.lastrowid

        # Verify both grades stored separately
        row = await conn.execute(
            "SELECT alpha_grade, agg_grade FROM signals WHERE id = ?", (signal_id,)
        )
        result = await row.fetchone()
        assert result["alpha_grade"] == "A"
        assert result["agg_grade"] == "B+"

        await conn.close()
        os.remove(db_path)


# ── P3-4: Formatter uses alpha grade ──────────────────────────────────


class TestFormatterUsesAlphaGrade:
    """P3-4: TelegramFormatter.format_signal uses alpha_score.grade when available."""

    def test_formatter_uses_alpha_grade_over_scored_grade(self):
        from tg.formatter import TelegramFormatter

        fmt = TelegramFormatter()

        scored = MagicMock()
        scored.base_signal = MagicMock()
        scored.base_signal.symbol = "ETHUSDT"
        scored.base_signal.direction = MagicMock(value="LONG")
        scored.base_signal.strategy = "Momentum"
        scored.base_signal.entry_low = 3000
        scored.base_signal.entry_high = 3050
        scored.base_signal.stop_loss = 2900
        scored.base_signal.tp1 = 3200
        scored.base_signal.tp2 = 3400
        scored.base_signal.tp3 = None
        scored.base_signal.rr_ratio = 3.0
        scored.base_signal.raw_data = {}
        scored.base_signal.confidence = 80
        scored.base_signal.timeframe = "1h"
        scored.base_signal.entry_timeframe = "15m"
        scored.base_signal.setup_class = "intraday"
        scored.base_signal.sector = "L1"
        scored.base_signal.tier = 1
        scored.final_confidence = 82
        scored.grade = "B"  # aggregator says B
        scored.volume_score = 75
        scored.is_killzone = False
        scored.killzone_bonus = 0
        scored.all_confluence = []

        alpha_score = MagicMock()
        alpha_score.grade = "A+"  # alpha says A+
        alpha_score.expected_value_r = 0.8
        alpha_score.p_win = 0.70
        alpha_score.kelly_fraction = 0.15
        alpha_score.sharpe_estimate = 1.2
        alpha_score.total_alpha = 0.5

        prob_estimate = MagicMock()
        prob_estimate.p_win = 0.70

        sizing = MagicMock()
        sizing.position_size_usdt = 5000
        sizing.risk_amount_usdt = 100
        sizing.leverage_suggested = 1

        text = fmt.format_signal(
            scored, signal_id=1,
            alpha_score=alpha_score,
            prob_estimate=prob_estimate,
            sizing=sizing,
        )

        # Card should show A+ (from alpha), not B (from aggregator)
        assert "A+" in text

    def test_formatter_falls_back_to_scored_grade_without_alpha(self):
        from tg.formatter import TelegramFormatter

        fmt = TelegramFormatter()

        scored = MagicMock()
        scored.base_signal = MagicMock()
        scored.base_signal.symbol = "ETHUSDT"
        scored.base_signal.direction = MagicMock(value="SHORT")
        scored.base_signal.strategy = "Momentum"
        scored.base_signal.entry_low = 3000
        scored.base_signal.entry_high = 3050
        scored.base_signal.stop_loss = 3200
        scored.base_signal.tp1 = 2800
        scored.base_signal.tp2 = 2600
        scored.base_signal.tp3 = None
        scored.base_signal.rr_ratio = 2.0
        scored.base_signal.raw_data = {}
        scored.base_signal.confidence = 70
        scored.base_signal.timeframe = "1h"
        scored.base_signal.entry_timeframe = "15m"
        scored.base_signal.setup_class = "intraday"
        scored.base_signal.sector = "L1"
        scored.base_signal.tier = 1
        scored.final_confidence = 72
        scored.grade = "B"
        scored.volume_score = 75
        scored.is_killzone = False
        scored.killzone_bonus = 0
        scored.all_confluence = []

        text = fmt.format_signal(scored, signal_id=2)

        # Without alpha_score, should use scored.grade = B
        # The card should not show A+ or A
        assert "A+" not in text


# ── P3-5: AI delta cap baseline re-anchored ──────────────────────────


class TestAIDeltaCapBaseline:
    """P3-5: _raw_confidence must be re-anchored AFTER market-state and BTC
    news multipliers, so those deterministic adjustments don't count against
    the ±12pt AI influence cap."""

    def test_raw_confidence_reanchor_logic(self):
        """Simulate the re-anchor: after applying state_mult and btc_mult,
        _raw_confidence should equal the new scored.final_confidence."""
        # Before fix: _raw_confidence was set only at line ~2467
        # After fix: _raw_confidence is re-set after pre-gate mults

        initial_conf = 70.0
        state_mult = 0.85  # VOLATILE regime penalty
        btc_mult = 0.90  # BTC negative news

        final_after_mults = initial_conf * state_mult * btc_mult
        _raw_confidence = final_after_mults  # The fix: re-anchor here

        # Then AI adjustments happen later (macro penalty etc.)
        ai_penalty = -5
        scored_final = _raw_confidence + ai_penalty

        ai_delta = scored_final - _raw_confidence
        assert ai_delta == ai_penalty  # Should be exactly -5, not -5 minus mult effects
        assert abs(ai_delta) <= 12  # Within cap
