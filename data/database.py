"""
TitanBot Pro — Database
=======================
SQLite with WAL mode for fast concurrent reads/writes on SSD.
Stores signals, outcomes, performance stats, and watchlist state.

Uses a connection pool (ConnectionPool) for concurrent read access while
serializing all writes through a single dedicated writer connection.

All writes are async and non-blocking. Schema auto-creates on first run.
"""

import asyncio
import aiosqlite
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from config.loader import cfg
from config.constants import Database as DBC
from data.connection_pool import ConnectionPool

logger = logging.getLogger(__name__)

def _safe_evidence(evidence: dict, max_bytes: int = 10_240) -> str:
    """FIX M7: Serialize evidence dict to JSON, truncating oversized blobs.
    Prevents feature_store bugs from writing megabyte evidence blobs to the DB."""
    try:
        raw = json.dumps(evidence)
        if len(raw.encode()) <= max_bytes:
            return raw
        # Oversized — keep only scalar values (drop any array/nested data)
        safe = {k: v for k, v in evidence.items() if isinstance(v, (bool, int, float, str, type(None)))}
        truncated = json.dumps(safe)
        logger.warning(f"Evidence blob truncated: {len(raw.encode())} → {len(truncated.encode())} bytes")
        return truncated
    except Exception as e:
        logger.debug(f"Evidence serialization failed (non-fatal): {e}")
        return "{}"




class Database:
    """
    Async SQLite database for TitanBot Pro.
    Uses WAL (Write-Ahead Logging) for concurrent access without locks.
    """

    _instance: Optional['Database'] = None

    def __new__(cls) -> 'Database':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._db_path = Path(cfg.database.path)
            self._pool: Optional[ConnectionPool] = None
            self._lock = asyncio.Lock()
            self._initialized = True

    @property
    def _conn(self) -> Optional[aiosqlite.Connection]:
        """Backward-compat shim: exposes the pool's writer connection.

        Existing code that references ``self._conn`` directly (e.g. migrations,
        schema creation) will transparently use the writer connection.
        """
        if self._pool is None:
            return None
        return self._pool._writer_conn

    async def _exec(self, sql: str, params=()) -> None:
        """Execute a write statement on the writer connection.

        In aiosqlite, every execute() returns a cursor object that sits on
        the background thread queue until closed. Bare 'await conn.execute()'
        without storing and closing the cursor causes the queue to fill up,
        making all subsequent execute() calls block indefinitely.
        This helper ensures the cursor is always closed immediately,
        even if execute() or an intermediate step raises an exception.
        """
        async with await self._pool._writer_conn.execute(sql, params):
            pass  # cursor is automatically closed by the context manager

    # Current schema version — increment when adding new migrations below.
    # SQLite's user_version pragma persists this value, so migrations only
    # run once instead of being attempted (and silently ignored) on every boot.
    _SCHEMA_VERSION = 9

    async def initialize(self):
        """Connect to database via connection pool and create schema"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._pool = ConnectionPool()
        await self._pool.initialize(str(self._db_path))

        # Backward-compat: self._lock now delegates to the pool's writer lock
        self._lock = self._pool._writer_lock

        logger.debug("DB: PRAGMAs set via pool, creating schema...")
        await self._create_schema()
        logger.debug("DB: Schema created, running retention cleanup...")
        # B8/I6: Run retention cleanup on every startup to keep DB small
        await self.run_retention_cleanup()  # FIX #27: retention set to 35d (was 30d = same as analysis window)
        # ── Safe column migrations — guarded by schema version to avoid
        # attempting every ALTER TABLE on every boot (no-op but noisy).
        await self._apply_migrations()

        logger.info(f"✅ Database initialized at {self._db_path}")

    async def _apply_migrations(self):
        """Run schema migrations that have not yet been applied.

        Uses SQLite's user_version pragma as a migration counter.
        Each block below is applied exactly once, then user_version is bumped.
        Adding a new migration: append to _migrations list and increment
        _SCHEMA_VERSION at the class level.
        """
        cur = await self._conn.execute("PRAGMA user_version")
        row = await cur.fetchone()
        await cur.close()
        current_version = int(row[0]) if row else 0

        if current_version >= self._SCHEMA_VERSION:
            return  # All migrations already applied

        # Version 1 → 2: outcome/signal column additions (previously run on every boot)
        _v1_migrations = [
            "ALTER TABLE outcomes ADD COLUMN pnl_pct REAL",
            "ALTER TABLE outcomes ADD COLUMN notes TEXT",
            "ALTER TABLE signals  ADD COLUMN user_taken INTEGER DEFAULT 0",
            "ALTER TABLE signals  ADD COLUMN user_notes TEXT",
        ]
        if current_version < 2:
            async with self._lock:
                for _mig in _v1_migrations:
                    try:
                        await self._conn.execute(_mig)
                    except Exception:
                        pass  # Column already exists — expected on existing DBs
                await self._conn.execute("PRAGMA user_version = 2")
                await self._conn.commit()
            logger.info("DB: schema migrated to version 2")

        # Version 2 → 3: AI influence tracking on signals
        # Enables per-signal attribution — answer "did AI adjustments help or hurt?"
        # by bucketing signals into no_ai / low / medium / high influence groups
        # and comparing win rate + avg R per bucket.
        _v2_migrations = [
            "ALTER TABLE signals ADD COLUMN ai_influence REAL DEFAULT NULL",
        ]
        if current_version < 3:
            async with self._lock:
                for _mig in _v2_migrations:
                    try:
                        await self._conn.execute(_mig)
                    except Exception:
                        pass  # Column already exists — safe to ignore
                await self._conn.execute("PRAGMA user_version = 3")
                await self._conn.commit()
            logger.info("DB: schema migrated to version 3")

        # Version 3 → 4: market state tracking on signals
        # Enables per-market-state win rate queries so the ParamTuner can make
        # statistically grounded adjustments to penalty magnitudes and multipliers.
        _v3_migrations = [
            "ALTER TABLE signals ADD COLUMN market_state TEXT DEFAULT NULL",
        ]
        if current_version < 4:
            async with self._lock:
                for _mig in _v3_migrations:
                    try:
                        await self._conn.execute(_mig)
                    except Exception:
                        pass  # Column already exists — safe to ignore
                await self._conn.execute("PRAGMA user_version = 4")
                await self._conn.commit()
            logger.info("DB: schema migrated to version 4")

        # Version 4 → 5: persistent strategy adaptive state + persistent risk state
        # strategy_persistence_v1 survives restarts (C1/C2 fix).
        # risk_state_v1 survives restarts (C3 fix).
        if current_version < 5:
            async with self._lock:
                _v4_stmts = [
                    """CREATE TABLE IF NOT EXISTS strategy_persistence_v1 (
                        strategy        TEXT PRIMARY KEY,
                        ewma_win_rate   REAL NOT NULL DEFAULT 0.5,
                        weight_mult     REAL NOT NULL DEFAULT 1.0,
                        is_disabled     INTEGER NOT NULL DEFAULT 0,
                        disabled_until  REAL NOT NULL DEFAULT 0,
                        disabled_at     REAL NOT NULL DEFAULT 0,
                        r_outcomes_json TEXT NOT NULL DEFAULT '[]',
                        updated_at      REAL NOT NULL,
                        version         INTEGER NOT NULL DEFAULT 1
                    )""",
                    """CREATE TABLE IF NOT EXISTS risk_state_v1 (
                        id              INTEGER PRIMARY KEY CHECK (id = 1),
                        daily_loss_pct  REAL NOT NULL DEFAULT 0,
                        peak_capital    REAL NOT NULL DEFAULT 0,
                        hard_kill       INTEGER NOT NULL DEFAULT 0,
                        resume_at       REAL,
                        consecutive     INTEGER NOT NULL DEFAULT 0,
                        updated_at      REAL NOT NULL,
                        version         INTEGER NOT NULL DEFAULT 1
                    )""",
                ]
                for _stmt in _v4_stmts:
                    try:
                        await self._conn.execute(_stmt)
                    except Exception:
                        pass
                # FIX: Was using self._SCHEMA_VERSION (7) — jumped version
                # from 4→7 instead of 4→5. Hardcode target like all other blocks.
                await self._conn.execute("PRAGMA user_version = 5")
                await self._conn.commit()
            logger.info("DB: schema migrated to version 5")

        if current_version < 6:
            async with self._lock:
                _v6_stmt = """CREATE TABLE IF NOT EXISTS tracked_signals_v1 (
                    signal_id               INTEGER PRIMARY KEY,
                    symbol                  TEXT NOT NULL,
                    direction               TEXT NOT NULL,
                    strategy                TEXT NOT NULL,
                    state                   TEXT NOT NULL,
                    entry_low               REAL NOT NULL,
                    entry_high              REAL NOT NULL,
                    entry_price             REAL,
                    stop_loss               REAL NOT NULL,
                    tp1                     REAL NOT NULL DEFAULT 0,
                    tp2                     REAL NOT NULL DEFAULT 0,
                    tp3                     REAL,
                    be_stop                 REAL,
                    trail_stop              REAL,
                    max_r                   REAL NOT NULL DEFAULT 0,
                    confidence              REAL NOT NULL DEFAULT 0,
                    grade                   TEXT NOT NULL DEFAULT 'B',
                    message_id              INTEGER,
                    created_at              REAL NOT NULL,
                    updated_at              REAL NOT NULL,
                    activated_at            REAL,
                    has_rejection_candle    INTEGER NOT NULL DEFAULT 0,
                    has_structure_shift     INTEGER NOT NULL DEFAULT 0,
                    has_momentum_expansion  INTEGER NOT NULL DEFAULT 0,
                    has_liquidity_reaction  INTEGER NOT NULL DEFAULT 0,
                    min_triggers            INTEGER NOT NULL DEFAULT 2,
                    setup_class             TEXT NOT NULL DEFAULT 'intraday',
                    rr_ratio                REAL NOT NULL DEFAULT 0,
                    trail_pct               REAL NOT NULL DEFAULT 0.40
                )"""
                try:
                    await self._conn.execute(_v6_stmt)
                except Exception:
                    pass
                await self._conn.execute("PRAGMA user_version = 6")
                await self._conn.commit()
            logger.info("DB: schema migrated to version 6")

        if current_version < 7:
            async with self._lock:
                # Version 6 → 7: execution analytics — entry_status column
                # Records whether price was IN_ZONE, LATE, or EXTENDED at the
                # moment the trade was entered. This is the single most important
                # field for diagnosing execution losses: a signal with correct
                # structure that entered LATE/EXTENDED has a fundamentally different
                # risk profile (worse RR) than an in-zone entry.
                _v7_migrations = [
                    "ALTER TABLE signals ADD COLUMN entry_status TEXT",
                    "ALTER TABLE tracked_signals_v1 ADD COLUMN entry_status TEXT",
                ]
                for _mig in _v7_migrations:
                    try:
                        await self._conn.execute(_mig)
                    except Exception:
                        pass  # Column already exists — safe to ignore
                await self._conn.execute("PRAGMA user_version = 7")
                await self._conn.commit()
            logger.info("DB: schema migrated to version 7")

        if current_version < 8:
            async with self._lock:
                _v8_migrations = [
                    "ALTER TABLE tracked_signals_v1 ADD COLUMN activated_at REAL",
                    "ALTER TABLE strategy_persistence_v1 ADD COLUMN disabled_at REAL NOT NULL DEFAULT 0",
                ]
                for _mig in _v8_migrations:
                    try:
                        await self._conn.execute(_mig)
                    except Exception:
                        pass
                await self._conn.execute("PRAGMA user_version = 8")
                await self._conn.commit()
            logger.info("DB: schema migrated to version 8")

        if current_version < 9:
            async with self._lock:
                _v9_migrations = [
                    "ALTER TABLE signals ADD COLUMN ensemble_votes TEXT DEFAULT '{}'",
                ]
                for _mig in _v9_migrations:
                    try:
                        await self._conn.execute(_mig)
                    except Exception:
                        pass
                await self._conn.execute("PRAGMA user_version = 9")
                await self._conn.commit()
            logger.info("DB: schema migrated to version 9")

    async def health_check(self) -> bool:
        """Verify the database connection is alive and responsive.

        Returns True if the connection is healthy, False otherwise.
        Callers can use this to decide whether to reconnect or raise an alert.
        """
        if self._pool is None:
            return False
        try:
            async with self._pool.reader() as conn:
                cur = await conn.execute("SELECT 1")
                await cur.fetchone()
                await cur.close()
            return True
        except Exception as e:
            logger.warning(f"DB health check failed: {e}")
            return False

    async def run_retention_cleanup(self, retain_days: int = DBC.RETENTION_DAYS):  # FIX #27: was 30d same as analysis window
        """
        B8/I6: Archive old data to keep the database fast and small.
        Keeps summary stats but removes raw signal/outcome rows older than retain_days.
        Runs on startup and can be called manually.
        """
        try:
            async with self._pool.writer() as conn:
                # ROOT CAUSE FIX: Do NOT use "async with conn.execute() as cur: return"
                # In aiosqlite, returning from inside an async-with-execute block leaves
                # the cursor object on the internal thread queue. Every subsequent
                # execute() call on this connection then blocks waiting for that orphaned
                # cursor to be cleaned up — causing the performance_tracker hang.
                # Always use: cur = await execute() / await cur.close() pattern instead.
                _cur = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='outcomes'"
                )
                _row = await _cur.fetchone()
                await _cur.close()
                if not _row:
                    logger.debug("DB retention: outcomes table not yet created, skipping")
                    return

                cutoff_expr = f"-{retain_days} days"

                _cur1 = await conn.execute(
                    """DELETE FROM signals
                       WHERE created_at < datetime('now', ?) AND outcome IS NOT NULL""",
                    (cutoff_expr,)
                )
                deleted_signals = _cur1.rowcount
                await _cur1.close()

                _cur2 = await conn.execute(
                    """DELETE FROM outcomes WHERE created_at < datetime('now', ?)""",
                    (cutoff_expr,)
                )
                deleted_outcomes = _cur2.rowcount
                await _cur2.close()

                await self._exec(
                    """DELETE FROM learning_state WHERE rowid NOT IN (
                        SELECT rowid FROM learning_state ORDER BY rowid DESC LIMIT 500
                    )"""
                )
                await self._exec(
                    """DELETE FROM events WHERE id NOT IN (
                        SELECT id FROM events ORDER BY id DESC LIMIT 1000
                    )"""
                )

                await conn.commit()
                if deleted_signals + deleted_outcomes > 100:
                    await self._exec("VACUUM")
                    logger.info(
                        f"🗄️  DB retention: removed {deleted_signals} signals, "
                        f"{deleted_outcomes} outcomes older than {retain_days}d. VACUUM run."
                    )
                elif deleted_signals + deleted_outcomes > 0:
                    logger.info(
                        f"🗄️  DB retention: removed {deleted_signals} signals, "
                        f"{deleted_outcomes} outcomes older than {retain_days}d."
                    )
        except Exception as e:
            logger.warning(f"DB retention cleanup failed (non-fatal): {e}")

    async def close(self):
        """Close all database connections via pool"""
        if self._pool:
            await self._pool.close()
            logger.info("Database closed")

    # ── Schema ──────────────────────────────────────────────

    async def _create_schema(self):
        """Create all tables if they don't exist"""
        schema = """
        -- Signals sent to Telegram
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,           -- LONG | SHORT
            strategy TEXT NOT NULL,
            confidence REAL NOT NULL,
            entry_low REAL NOT NULL,
            entry_high REAL NOT NULL,
            stop_loss REAL NOT NULL,
            tp1 REAL NOT NULL,
            tp2 REAL,
            tp3 REAL,
            rr_ratio REAL,
            timeframe TEXT,            -- back-compat (stores entry_timeframe)
            setup_class TEXT DEFAULT 'intraday',   -- 'swing' | 'intraday' | 'scalp'
            entry_timeframe TEXT DEFAULT '15m',
            regime TEXT,
            sector TEXT,
            tier INTEGER,
            confluence TEXT,                   -- JSON: list of confluence factors
            raw_scores TEXT,                   -- JSON: individual analyzer scores
            message_id INTEGER,                -- Telegram message ID
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            valid_until TEXT,
            -- Quant pipeline fields (Fix L1, L2, A1)
            p_win REAL DEFAULT 0.55,           -- Probability engine P(win)
            ev_r REAL DEFAULT 0.0,             -- Expected value in R
            alpha_grade TEXT DEFAULT 'B',      -- Alpha model grade
            agg_grade TEXT DEFAULT 'B',        -- Aggregator grade (confidence-based, shown in Telegram)
            evidence TEXT DEFAULT '{}',        -- JSON: evidence flags at signal time
            risk_amount REAL DEFAULT 0.0,      -- Risk in USDT (from portfolio_engine sizing)
            position_size REAL DEFAULT 0.0,    -- Position size in USDT (from portfolio_engine sizing)
            outcome TEXT,                      -- WIN | LOSS | BE | EXPIRED | REJECTED
            pnl_r REAL,                        -- Realized P&L in R multiples
            reject_reason TEXT,                -- Why signal was rejected (if applicable)
            confluence_strategies TEXT,        -- FIX 1D: JSON list of agreeing strategy names
            -- FIX 3A/3B: Fields needed for AI execution funnel analysis
            publish_price REAL,                -- Price at exact moment signal was published
            zone_reached INTEGER DEFAULT 0,    -- 1 if execution engine ever hit ENTRY_ZONE state
            zone_reached_at TEXT,              -- Timestamp when ENTRY_ZONE was first reached
            exec_state TEXT DEFAULT 'WATCHING', -- WATCHING | ALMOST | ENTRY_ZONE | EXECUTE | ACTIVE
            ensemble_votes TEXT DEFAULT '{}'   -- JSON: per-source ensemble vote breakdown
        );
        CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
        CREATE INDEX IF NOT EXISTS idx_signals_outcome ON signals(outcome);
        CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);

        -- User's actual trade outcomes (logged via Telegram buttons)
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER REFERENCES signals(id),
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            outcome TEXT,                      -- WIN | LOSS | BREAKEVEN | SKIPPED
            skip_reason TEXT,                  -- Why skipped (if skipped)
            pnl_pct REAL,                      -- % P&L
            pnl_r REAL,                        -- R multiple
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            strategy TEXT,                     -- Strategy that generated the signal
            closed_at TEXT,                    -- When outcome was recorded
            post_expiry_price REAL             -- FIX 3C: Price fetched when EXPIRED recorded (for direction accuracy)
        );
        -- FIX AUDIT-3: Prevent duplicate outcome rows for the same signal.
        -- Previously save_outcome() was a plain INSERT with no dedup, so repeated
        -- calls (e.g. outcome_monitor + Telegram button) created duplicate rows
        -- that inflated performance stats.
        CREATE UNIQUE INDEX IF NOT EXISTS idx_outcomes_signal_id ON outcomes(signal_id);

        -- Strategy performance stats (computed from outcomes)
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            period TEXT NOT NULL,              -- 7d | 30d | 90d | all
            signals_sent INTEGER DEFAULT 0,
            trades_taken INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            avg_pnl_r REAL DEFAULT 0,
            ewma_win_rate REAL DEFAULT 0.5,
            weight_adjustment REAL DEFAULT 1.0,
            last_updated TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(strategy, period)
        );

        -- Watchlist (stalker engine pre-signals)
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            score REAL NOT NULL,
            reasons TEXT,                      -- JSON: list of watch reasons
            detected_at TEXT NOT NULL DEFAULT (datetime('now')),
            expires_at TEXT,
            user_watching INTEGER DEFAULT 0,   -- 1 if user tapped Watch button
            telegram_message_id INTEGER
        );

        -- Symbol tier state
        CREATE TABLE IF NOT EXISTS symbol_tiers (
            symbol TEXT PRIMARY KEY,
            tier INTEGER NOT NULL DEFAULT 2,
            volume_24h REAL DEFAULT 0,
            last_promoted TEXT,
            promoted_from INTEGER,
            scan_count INTEGER DEFAULT 0,
            last_scan TEXT,
            activity_score REAL DEFAULT 0
        );

        -- Learning loop persistent state (Fix L7)
        CREATE TABLE IF NOT EXISTS learning_state (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- System events log (startup, errors, circuit breaker, etc.)
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            message TEXT,
            data TEXT,                         -- JSON extra data
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Circuit breaker state
        CREATE TABLE IF NOT EXISTS circuit_breaker (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            is_active INTEGER DEFAULT 0,
            reason TEXT,
            activated_at TEXT,
            resume_at TEXT,
            consecutive_losses INTEGER DEFAULT 0,
            losses_this_hour INTEGER DEFAULT 0
        );

        INSERT OR IGNORE INTO circuit_breaker (id) VALUES (1);

        -- Persistent strategy adaptive state (C1/C2 fix: survive restarts)
        -- version column allows future field additions without corrupt loads.
        CREATE TABLE IF NOT EXISTS strategy_persistence_v1 (
            strategy        TEXT PRIMARY KEY,
            ewma_win_rate   REAL NOT NULL DEFAULT 0.5,
            weight_mult     REAL NOT NULL DEFAULT 1.0,
            is_disabled     INTEGER NOT NULL DEFAULT 0,
            disabled_until  REAL NOT NULL DEFAULT 0,
            disabled_at     REAL NOT NULL DEFAULT 0,
            r_outcomes_json TEXT NOT NULL DEFAULT '[]',
            updated_at      REAL NOT NULL,
            version         INTEGER NOT NULL DEFAULT 1
        );

        -- Persistent circuit breaker risk state (C3 fix: survive restarts)
        -- Singleton row (id=1). version allows schema evolution.
        CREATE TABLE IF NOT EXISTS risk_state_v1 (
            id              INTEGER PRIMARY KEY CHECK (id = 1),
            daily_loss_pct  REAL NOT NULL DEFAULT 0,
            peak_capital    REAL NOT NULL DEFAULT 0,
            hard_kill       INTEGER NOT NULL DEFAULT 0,
            resume_at       REAL,
            consecutive     INTEGER NOT NULL DEFAULT 0,
            updated_at      REAL NOT NULL,
            version         INTEGER NOT NULL DEFAULT 1
        );
        -- C signal detection log (Phase 2 training data)
        -- Every C signal evaluated is logged here regardless of routing decision
        CREATE TABLE IF NOT EXISTS c_signal_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER REFERENCES signals(id),
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL,
            regime TEXT NOT NULL,
            context_key TEXT NOT NULL,          -- strategy|regime|direction
            confidence REAL NOT NULL,
            up_score REAL NOT NULL,
            phase1_up REAL NOT NULL,
            phase2_up REAL NOT NULL,
            phase2_weight REAL NOT NULL,
            sent_to_user INTEGER DEFAULT 0,     -- 1 if context sent to signals ch
            upgraded INTEGER,                   -- 1=yes, 0=no, NULL=pending
            upgrade_grade TEXT,                 -- B | A | A+
            upgrade_latency_min REAL,           -- minutes to upgrade
            expired INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Bayesian posteriors for upgrade probability (Phase 2)
        -- One row per context key (strategy|regime|direction)
        CREATE TABLE IF NOT EXISTS upgrade_posteriors (
            context_key TEXT PRIMARY KEY,
            alpha REAL NOT NULL DEFAULT 2.0,
            beta REAL NOT NULL DEFAULT 2.0,
            sample_count INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_c_signal_context ON c_signal_log(context_key);
        CREATE INDEX IF NOT EXISTS idx_c_signal_created ON c_signal_log(created_at);

        -- Indices for common queries
        -- DEDUP-FIX: Persist signal dedup state so restarts don't cause duplicate signals
        CREATE TABLE IF NOT EXISTS signal_dedup (
            symbol          TEXT NOT NULL,
            direction       TEXT NOT NULL,
            sent_at         REAL NOT NULL,
            final_confidence REAL DEFAULT 0,
            grade           TEXT DEFAULT 'B',
            PRIMARY KEY (symbol, direction)
        );

        -- PHASE 1 FIX (NO-RESTART): Persist open positions so bot can recover state on restart
        CREATE TABLE IF NOT EXISTS open_positions (
            signal_id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            strategy TEXT NOT NULL,
            entry_price REAL NOT NULL,
            size_usdt REAL NOT NULL,
            risk_usdt REAL NOT NULL,
            stop_loss REAL NOT NULL,
            tp1 REAL,
            tp2 REAL,
            tp3 REAL,
            rr_ratio REAL,
            sector TEXT DEFAULT '',
            opened_at TEXT DEFAULT (datetime('now')),
            message_id INTEGER,
            correlation_to_btc REAL DEFAULT 0.7,
            alpha_grade TEXT DEFAULT 'B',
            confidence REAL DEFAULT 0.0
        );
        CREATE TABLE IF NOT EXISTS card_state (
            signal_id  INTEGER PRIMARY KEY,
            message_id INTEGER,
            chat_id    INTEGER,
            symbol     TEXT,
            direction  TEXT,
            grade      TEXT,
            state      TEXT DEFAULT 'PENDING',
            confirmed  INTEGER DEFAULT 0,
            created_at REAL,
            updated_at REAL
        );

        -- Unified signal lifecycle persistence (schema v6).
        -- Covers both ExecutionEngine pre-fill states (WATCHING/ALMOST/EXECUTE)
        -- and OutcomeMonitor post-fill states (ACTIVE/TP1_HIT/BE_ACTIVE).
        -- Single source of truth that survives restarts.
        -- Rows are deleted when a signal reaches a terminal state.
        CREATE TABLE IF NOT EXISTS tracked_signals_v1 (
            signal_id               INTEGER PRIMARY KEY,
            symbol                  TEXT NOT NULL,
            direction               TEXT NOT NULL,
            strategy                TEXT NOT NULL,
            state                   TEXT NOT NULL,
            entry_low               REAL NOT NULL,
            entry_high              REAL NOT NULL,
            entry_price             REAL,
            stop_loss               REAL NOT NULL,
            tp1                     REAL NOT NULL DEFAULT 0,
            tp2                     REAL NOT NULL DEFAULT 0,
            tp3                     REAL,
            be_stop                 REAL,
            trail_stop              REAL,
            max_r                   REAL NOT NULL DEFAULT 0,
            confidence              REAL NOT NULL DEFAULT 0,
            grade                   TEXT NOT NULL DEFAULT 'B',
            message_id              INTEGER,
            created_at              REAL NOT NULL,
            updated_at              REAL NOT NULL,
            activated_at            REAL,
            has_rejection_candle    INTEGER NOT NULL DEFAULT 0,
            has_structure_shift     INTEGER NOT NULL DEFAULT 0,
            has_momentum_expansion  INTEGER NOT NULL DEFAULT 0,
            has_liquidity_reaction  INTEGER NOT NULL DEFAULT 0,
            min_triggers            INTEGER NOT NULL DEFAULT 2,
            setup_class             TEXT NOT NULL DEFAULT 'intraday',
            rr_ratio                REAL NOT NULL DEFAULT 0,
            trail_pct               REAL NOT NULL DEFAULT 0.40
        );

                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
        CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);
        CREATE INDEX IF NOT EXISTS idx_outcomes_signal ON outcomes(signal_id);
        CREATE INDEX IF NOT EXISTS idx_outcomes_created ON outcomes(created_at);
        """

        # FIX: Do NOT use executescript() — it demands an exclusive write lock and
        # hangs indefinitely when a .db-wal file from a prior session is present.
        # Instead split on ';' and execute each statement individually.
        for stmt in schema.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    await self._exec(stmt)
                except Exception as _schema_err:
                    # Ignore "already exists" errors — idempotent schema creation
                    if "already exists" not in str(_schema_err).lower():
                        logger.debug(f"Schema stmt skipped: {_schema_err}")
        await self._conn.commit()

        # ── Schema migrations (safe ALTER TABLE — idempotent) ──────────
        # These add columns that exist in the current schema but may not exist
        # in databases created by older versions of the bot.
        # ALTER TABLE ADD COLUMN is safe to run multiple times — SQLite ignores
        # "duplicate column" errors when caught, so we wrap each in try/except.
        migrations = [
            "ALTER TABLE signals ADD COLUMN setup_class TEXT DEFAULT 'intraday'",
            "ALTER TABLE signals ADD COLUMN entry_timeframe TEXT DEFAULT '15m'",
            "ALTER TABLE signals ADD COLUMN p_win REAL DEFAULT 0.55",
            "ALTER TABLE signals ADD COLUMN ev_r REAL DEFAULT 0.0",
            "ALTER TABLE signals ADD COLUMN alpha_grade TEXT DEFAULT 'B'",
            "ALTER TABLE open_positions ADD COLUMN correlation_to_btc REAL DEFAULT 0.7",
            "ALTER TABLE open_positions ADD COLUMN alpha_grade TEXT DEFAULT 'B'",
            "ALTER TABLE open_positions ADD COLUMN confidence REAL DEFAULT 0.0",
            "ALTER TABLE signals ADD COLUMN exec_state TEXT DEFAULT 'WATCHING'",
            "UPDATE signals SET exec_state = 'WATCHING' WHERE exec_state IS NULL",
            "ALTER TABLE signals ADD COLUMN evidence TEXT DEFAULT '{}'",
            "ALTER TABLE signals ADD COLUMN risk_amount REAL DEFAULT 0.0",
            "ALTER TABLE signals ADD COLUMN position_size REAL DEFAULT 0.0",
            "ALTER TABLE signals ADD COLUMN outcome TEXT",
            "ALTER TABLE signals ADD COLUMN pnl_r REAL",
            "ALTER TABLE signals ADD COLUMN reject_reason TEXT",
            "ALTER TABLE signals ADD COLUMN valid_until TEXT",
            "ALTER TABLE signals ADD COLUMN tier INTEGER",
            "ALTER TABLE signals ADD COLUMN confluence_strategies TEXT",
            # ── outcomes table itself (for DBs created before outcomes existed) ──
            "CREATE TABLE IF NOT EXISTS outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id INTEGER REFERENCES signals(id), symbol TEXT NOT NULL, direction TEXT NOT NULL, entry_price REAL, exit_price REAL, outcome TEXT, skip_reason TEXT, pnl_r REAL, strategy TEXT, regime TEXT, created_at REAL DEFAULT (unixepoch()), closed_at TEXT, post_expiry_price REAL)",
            "CREATE INDEX IF NOT EXISTS idx_outcomes_signal ON outcomes(signal_id)",
            "CREATE INDEX IF NOT EXISTS idx_outcomes_created ON outcomes(created_at)",
            "ALTER TABLE outcomes ADD COLUMN strategy TEXT",
            "ALTER TABLE outcomes ADD COLUMN pnl_r REAL",
            "ALTER TABLE signals ADD COLUMN user_taken INTEGER DEFAULT 0",
            "ALTER TABLE signals ADD COLUMN user_notes TEXT DEFAULT ''",
            "ALTER TABLE outcomes ADD COLUMN closed_at TEXT",
            # FIX 1D / 3A / 3B / 3C: New fields for AI execution funnel analysis
            "ALTER TABLE signals ADD COLUMN publish_price REAL",
            "ALTER TABLE signals ADD COLUMN zone_reached INTEGER DEFAULT 0",
            "ALTER TABLE signals ADD COLUMN zone_reached_at TEXT",
            "ALTER TABLE signals ADD COLUMN ensemble_votes TEXT DEFAULT '{}'",
            "ALTER TABLE outcomes ADD COLUMN post_expiry_price REAL",
            # Execution analytics layer — answers "why did this trade lose?"
            # Records actual fill price, timing, exit mechanics per trade.
            "ALTER TABLE signals ADD COLUMN entry_price REAL",
            "ALTER TABLE signals ADD COLUMN entry_time REAL",
            "ALTER TABLE signals ADD COLUMN exit_price REAL",
            "ALTER TABLE signals ADD COLUMN exit_reason TEXT",
            "ALTER TABLE signals ADD COLUMN max_r REAL DEFAULT 0.0",
            # PHASE 3 AUDIT FIX (P3-3): persist aggregator grade alongside alpha grade
            "ALTER TABLE signals ADD COLUMN agg_grade TEXT DEFAULT 'B'",
        ]
        for sql in migrations:
            try:
                await self._exec(sql)
            except Exception:
                pass  # Column already exists — safe to ignore
        await self._conn.commit()
        logger.debug(f"Schema migrations applied ({len(migrations)} checked)")

    # ── Signals ─────────────────────────────────────────────

    async def save_signal(self, signal: Dict[str, Any]) -> int:
        """Save a new signal and return its ID (Fix A1: includes quant columns)"""
        # FIX: validate critical fields before hitting SQLite so callers get a
        # clear error instead of a silent type mismatch on the read path.
        _direction = signal.get('direction')
        if _direction not in ('LONG', 'SHORT'):
            raise ValueError(f"save_signal: invalid direction {_direction!r} — must be 'LONG' or 'SHORT'")
        _confidence = signal.get('confidence')
        if _confidence is not None and not (0 <= float(_confidence) <= 100):
            raise ValueError(f"save_signal: confidence {_confidence} out of range [0, 100]")
        _t0 = time.time()
        try:
            async with self._pool.writer() as conn:
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
                    signal.get('symbol'),
                    signal.get('direction'),
                    signal.get('strategy'),
                    signal.get('confidence'),
                    signal.get('entry_low'),
                    signal.get('entry_high'),
                    signal.get('stop_loss'),
                    signal.get('tp1'),
                    signal.get('tp2'),
                    signal.get('tp3'),
                    signal.get('rr_ratio'),
                    signal.get('timeframe') or signal.get('entry_timeframe', '15m'),
                    signal.get('setup_class', 'intraday'),
                    signal.get('entry_timeframe', '15m'),
                    signal.get('regime'),
                    signal.get('sector'),
                    signal.get('tier'),
                    json.dumps(signal.get('confluence', [])),
                    # FIX 1D: confluence_strategies was set in engine.py but never saved
                    json.dumps(signal.get('confluence_strategies', [])) if signal.get('confluence_strategies') else None,
                    json.dumps(signal.get('raw_scores', {})),
                    signal.get('message_id'),
                    signal.get('valid_until'),
                    signal.get('p_win', 0.55),
                    signal.get('ev_r', 0.0),
                    signal.get('alpha_grade', 'B'),
                    signal.get('agg_grade', 'B'),  # PHASE 3 FIX (P3-3): aggregator grade
                    _safe_evidence(signal.get('evidence', {})),
                    signal.get('risk_amount', 0.0),
                    signal.get('position_size', 0.0),
                    # FIX 3A: publish_price — market price at exact moment signal saved
                    signal.get('publish_price'),
                    # AI influence: net confidence delta from AI context adjustments (post-MSE baseline)
                    signal.get('ai_influence'),
                    # Market state at signal time — enables per-state attribution by ParamTuner
                    signal.get('market_state'),
                    json.dumps(signal.get('ensemble_votes', {})),
                ))
                await conn.commit()
                _last_id = cursor.lastrowid
                await cursor.close()
                return _last_id
        finally:
            try:
                from core.diagnostic_engine import diagnostic_engine
                diagnostic_engine.record_db_operation(
                    "save_signal", (time.time() - _t0) * 1000, True
                )
            except Exception:
                pass

    async def update_signal_message_id(self, signal_id: int, message_id: int):
        """Update Telegram message ID after sending"""
        async with self._pool.writer() as conn:
            await self._exec(
                "UPDATE signals SET message_id = ? WHERE id = ?",
                (message_id, signal_id)
            )
            await conn.commit()

    async def get_signal(self, signal_id: int) -> Optional[Dict]:
        """Get a signal by ID"""
        async with self._pool.reader() as conn:
            cursor = await conn.execute(
                "SELECT * FROM signals WHERE id = ?", (signal_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            return dict(row) if row else None

    async def get_recent_signals(self, hours: int = DBC.DEFAULT_QUERY_WINDOW_HOURS, symbol: str = None, exclude_c_grade: bool = True) -> List[Dict]:
        """Get signals from the last N hours.
        BEH-8: excludes C grade by default so history/stats reflect real trade signals.
        """
        # FIX: limit was hardcoded at 50 regardless of hours requested.
        # 720h (30d) at 10 signals/day = 300 rows needed. Scale limit with window.
        dynamic_limit = max(DBC.QUERY_LIMIT_BASE, hours * DBC.QUERY_LIMIT_PER_HOUR)  # generous upper bound

        query = """
            SELECT * FROM signals
            WHERE created_at > datetime('now', ?)
        """
        params = [f'-{hours} hours']

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        # BEH-8: C grades are informational context signals, not trades
        if exclude_c_grade:
            query += " AND alpha_grade != 'C'"

        query += f" ORDER BY created_at DESC LIMIT {dynamic_limit}"
        async with self._pool.reader() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(r) for r in rows]

    async def get_recent_signals_all(self, hours: int = DBC.DEFAULT_QUERY_WINDOW_HOURS) -> List[Dict]:
        """Get ALL signals including C grade — for admin/debug use."""
        return await self.get_recent_signals(hours=hours, exclude_c_grade=False)

    async def get_signals_today(self, exclude_c_grade: bool = True) -> int:
        """Count actionable signals sent today. BEH-8: excludes C grade by default."""
        grade_filter = "AND alpha_grade != 'C'" if exclude_c_grade else ""
        async with self._pool.reader() as conn:
            cursor = await conn.execute(f"""
                SELECT COUNT(*) FROM signals
                WHERE date(created_at) = date('now') {grade_filter}
            """)
            try:
                row = await cursor.fetchone()
                return row[0] if row else 0
            finally:
                await cursor.close()

    # ── Outcomes ─────────────────────────────────────────────

    async def save_outcome(self, outcome: Dict[str, Any]) -> int:
        """Record a trade outcome (from Telegram button press).
        FIX AUDIT-3: Uses ON CONFLICT(signal_id) DO UPDATE so a second
        button-press (e.g. revised PnL) updates the existing row instead
        of being silently dropped.  COALESCE keeps prior values when the
        new payload omits a field."""
        async with self._pool.writer() as conn:
            cursor = await conn.execute("""
                INSERT INTO outcomes (
                    signal_id, symbol, direction,
                    entry_price, exit_price, outcome,
                    skip_reason, pnl_pct, pnl_r, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signal_id) DO UPDATE SET
                    outcome    = COALESCE(excluded.outcome,    outcome),
                    exit_price = COALESCE(excluded.exit_price, exit_price),
                    pnl_pct    = COALESCE(excluded.pnl_pct,   pnl_pct),
                    pnl_r      = COALESCE(excluded.pnl_r,     pnl_r),
                    skip_reason= COALESCE(excluded.skip_reason,skip_reason),
                    notes      = COALESCE(excluded.notes,      notes)
            """, (
                outcome.get('signal_id'),
                outcome.get('symbol'),
                outcome.get('direction'),
                outcome.get('entry_price'),
                outcome.get('exit_price'),
                outcome.get('outcome'),
                outcome.get('skip_reason'),
                outcome.get('pnl_pct'),
                outcome.get('pnl_r'),
                outcome.get('notes'),
            ))
            await conn.commit()
            _last_id = cursor.lastrowid
            await cursor.close()
            return _last_id

    async def update_signal_outcome(
        self,
        signal_id: int,
        outcome: str,
        pnl_r: float,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        max_r: float = 0.0,
    ):
        """
        Auto-record outcome from the OutcomeMonitor.
        Creates an outcomes row and updates signals.outcome + signals.pnl_r.
        Also writes execution analytics (exit_price, exit_reason, max_r) when provided.
        Uses COALESCE so that a later call without analytics won't overwrite values
        already written by an earlier call that had full analytics.
        """
        try:
            # Get signal info
            sig = await self.get_signal(signal_id)
            if not sig:
                return

            await self.save_outcome({
                'signal_id': signal_id,
                'symbol': sig.get('symbol', ''),
                'direction': sig.get('direction', ''),
                'outcome': outcome,
                'pnl_r': pnl_r,
                'notes': f'Auto-tracked by OutcomeMonitor',
            })

            # Update the signals table — COALESCE ensures analytics written by
            # outcome_monitor._complete() (which runs asynchronously) are never
            # overwritten by a subsequent call that lacks those values.
            async with self._pool.writer() as conn:
                await self._exec(
                    """UPDATE signals SET
                           outcome = ?,
                           pnl_r = ?,
                           exit_price = COALESCE(?, exit_price),
                           exit_reason = COALESCE(?, exit_reason),
                           max_r = CASE WHEN ? > COALESCE(max_r, 0) THEN ? ELSE COALESCE(max_r, 0) END
                       WHERE id = ?""",
                    (outcome, pnl_r, exit_price, exit_reason, max_r, max_r, signal_id)
                )
                await conn.commit()

            await self.log_event("AUTO_OUTCOME", f"Signal #{signal_id}: {outcome} ({pnl_r:+.1f}R)")
        except Exception as e:
            logger.error(f"Error saving auto outcome: {e}")

    async def update_signal_entry(self, signal_id: int, entry_price: float, entry_time: float,
                                  entry_status: Optional[str] = None) -> None:
        """Record actual fill price, entry timestamp, and entry status when price enters zone.
        Called by OutcomeMonitor when TrackedSignal transitions to ACTIVE.
        entry_status is IN_ZONE | LATE | EXTENDED — captures whether price was still
        inside the planned zone or had already overshot at the moment of entry.
        """
        try:
            async with self._pool.writer() as conn:
                await self._exec(
                    """UPDATE signals
                       SET entry_price  = COALESCE(entry_price, ?),
                           entry_time   = COALESCE(entry_time,  ?),
                           entry_status = COALESCE(entry_status, ?)
                       WHERE id = ?""",
                    (entry_price, entry_time, entry_status, signal_id),
                )
                await conn.commit()
        except Exception as e:
            logger.debug(f"update_signal_entry failed (non-fatal): {e}")

    async def write_zone_reached(self, signal_id: int) -> None:
        """FIX 3B: Record when execution engine first hits ENTRY_ZONE state.
        Called by execution_engine._check() on state transition to ENTRY_ZONE.
        Enables AI to distinguish 'zone never reached' from 'reached but trigger failed'.
        """
        try:
            async with self._pool.writer() as conn:
                await self._exec(
                    """UPDATE signals
                       SET zone_reached = 1,
                           zone_reached_at = COALESCE(zone_reached_at, datetime('now'))
                       WHERE id = ? AND zone_reached = 0""",
                    (signal_id,)
                )
                await conn.commit()
        except Exception as e:
            logger.debug(f"write_zone_reached error for signal {signal_id}: {e}")

    async def write_publish_price(self, signal_id: int, price: float) -> None:
        """FIX 3A: Backfill publish_price if not already set.
        Called by engine.py immediately after save_signal returns the ID.
        """
        try:
            if not price or price <= 0:
                return
            async with self._pool.writer() as conn:
                await self._exec(
                    "UPDATE signals SET publish_price = ? WHERE id = ? AND publish_price IS NULL",
                    (price, signal_id)
                )
                await conn.commit()
        except Exception as e:
            logger.debug(f"write_publish_price error for signal {signal_id}: {e}")

    async def write_post_expiry_price(self, signal_id: int, price: float) -> None:
        """FIX 3C: Record price at expiry time in outcomes table.
        Enables AI direction accuracy analysis on expired signals.
        """
        try:
            if not price or price <= 0:
                return
            async with self._pool.writer() as conn:
                await self._exec(
                    "UPDATE outcomes SET post_expiry_price = ? WHERE signal_id = ? AND post_expiry_price IS NULL",
                    (price, signal_id)
                )
                await conn.commit()
        except Exception as e:
            logger.debug(f"write_post_expiry_price error for signal {signal_id}: {e}")

    async def get_performance_stats(self, days: int = DBC.DEFAULT_PERFORMANCE_WINDOW_DAYS) -> Dict[str, Any]:
        """Get overall performance statistics"""
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN o.outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN o.outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN o.outcome = 'BREAKEVEN' THEN 1 ELSE 0 END) as breakevens,
                    SUM(CASE WHEN o.outcome = 'SKIPPED' THEN 1 ELSE 0 END) as skipped,
                    AVG(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN o.pnl_r END) as avg_r,
                    SUM(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN 1 ELSE 0 END) as trades_taken
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
            """, (f'-{days} days',))
            row = await cursor.fetchone()
            await cursor.close()
            return dict(row) if row else {}

    async def get_signal_quality_rows(self, days: int = DBC.DEFAULT_PERFORMANCE_WINDOW_DAYS) -> List[Dict[str, Any]]:
        """Recent signal rows enriched with outcome fields for fill/accuracy analytics."""
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT
                    s.id,
                    s.symbol,
                    s.direction,
                    s.publish_price,
                    s.entry_low,
                    s.entry_high,
                    s.entry_price,
                    s.entry_status,
                    s.max_r,
                    o.outcome,
                    o.pnl_r,
                    o.post_expiry_price
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
                ORDER BY s.created_at DESC
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(r) for r in rows]

    async def get_strategy_performance(self, days: int = DBC.DEFAULT_PERFORMANCE_WINDOW_DAYS) -> List[Dict]:
        """Get performance broken down by strategy"""
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT
                    s.strategy,
                    COUNT(*) as signals,
                    SUM(CASE WHEN o.outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN o.outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN o.outcome IN ('WIN','LOSS') THEN o.pnl_r END) as avg_r
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
                GROUP BY s.strategy
                ORDER BY wins DESC
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(r) for r in rows]

    async def get_sector_performance(self, days: int = DBC.DEFAULT_PERFORMANCE_WINDOW_DAYS) -> List[Dict]:
        """Get performance by sector"""
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT
                    s.sector,
                    COUNT(*) as signals,
                    SUM(CASE WHEN o.outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN o.outcome = 'LOSS' THEN 1 ELSE 0 END) as losses
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?) AND s.sector IS NOT NULL
                GROUP BY s.sector
                ORDER BY wins DESC
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(r) for r in rows]

    async def get_ai_attribution_stats(self, days: int = DBC.DEFAULT_PERFORMANCE_WINDOW_DAYS) -> List[Dict]:
        """Return win rate and avg R grouped by AI influence bucket.

        Buckets (based on net AI confidence delta stored on the signal):
          no_ai   — influence is NULL or 0        (AI had no effect)
          low     — |influence| in [1, 3]         (minor nudge)
          medium  — |influence| in [4, 8]         (moderate adjustment)
          high    — |influence| in [9, 12]        (near-cap effect)

        This answers the core question: "Does AI context help or hurt?"
        Compare win_rate and avg_r across buckets to validate or question
        the value of the AI adjustment layer.
        """
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT
                    CASE
                        WHEN s.ai_influence IS NULL OR ABS(s.ai_influence) < 1 THEN 'no_ai'
                        WHEN ABS(s.ai_influence) <= 3  THEN 'low'
                        WHEN ABS(s.ai_influence) <= 8  THEN 'medium'
                        ELSE                               'high'
                    END AS ai_bucket,
                    COUNT(*)                                                              AS signals,
                    SUM(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN 1 ELSE 0 END) AS trades,
                    SUM(CASE WHEN o.outcome = 'WIN'  THEN 1 ELSE 0 END)                  AS wins,
                    SUM(CASE WHEN o.outcome = 'LOSS' THEN 1 ELSE 0 END)                  AS losses,
                    AVG(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN o.pnl_r END) AS avg_r,
                    AVG(s.ai_influence)                                                   AS avg_influence
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
                GROUP BY ai_bucket
                ORDER BY
                    CASE ai_bucket
                        WHEN 'no_ai'  THEN 0
                        WHEN 'low'    THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'high'   THEN 3
                    END
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            await cursor.close()
        result = []
        for r in rows:
            row = dict(r)
            # Compute win rate (exclude breakevens from denominator)
            decisive = (row.get('wins') or 0) + (row.get('losses') or 0)
            row['win_rate'] = round((row.get('wins') or 0) / decisive, 3) if decisive else None
            result.append(row)
        return result

    async def get_performance_stats_by_market_state(self, days: int = DBC.DEFAULT_PERFORMANCE_WINDOW_DAYS) -> List[Dict]:
        """Return win rate, expectancy, and avg R grouped by market state (ParamTuner input).

        Requires the ``market_state`` column (schema v4) to be populated.
        Rows with a NULL market_state are excluded.  The ``win_rate`` field is
        computed as wins / (wins + losses) — breakevens are excluded from the
        denominator because they don't validate or invalidate a directional view.

        Additional fields returned for expectancy-aware tuning:
          ``avg_win_r``  — average R on winning trades (positive)
          ``avg_loss_r`` — average R on losing trades (negative)
          ``skipped``    — number of signals filtered out before execution
        These let the ParamTuner optimise for expectancy, not just win rate, and
        detect states where the filter is blocking too aggressively (opportunity cost).
        """
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT
                    s.market_state,
                    COUNT(*)                                                              AS signals,
                    SUM(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN 1 ELSE 0 END) AS trades,
                    SUM(CASE WHEN o.outcome = 'WIN'  THEN 1 ELSE 0 END)                  AS wins,
                    SUM(CASE WHEN o.outcome = 'LOSS' THEN 1 ELSE 0 END)                  AS losses,
                    AVG(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN o.pnl_r END) AS avg_r,
                    AVG(CASE WHEN o.outcome = 'WIN'  THEN o.pnl_r END)                   AS avg_win_r,
                    AVG(CASE WHEN o.outcome = 'LOSS' THEN o.pnl_r END)                   AS avg_loss_r,
                    SUM(CASE WHEN o.outcome = 'SKIPPED' THEN 1 ELSE 0 END)               AS skipped
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
                  AND s.market_state IS NOT NULL
                GROUP BY s.market_state
                ORDER BY signals DESC
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            await cursor.close()
        result = []
        for r in rows:
            row = dict(r)
            decisive = (row.get('wins') or 0) + (row.get('losses') or 0)
            row['win_rate'] = round((row.get('wins') or 0) / decisive, 3) if decisive else None
            # Compute expectancy = WR × avg_win_R + LossRate × avg_loss_R
            # avg_loss_r is already negative; result is expected R per trade
            wr = row['win_rate'] or 0.0
            avg_win  = row.get('avg_win_r')  or 0.0
            avg_loss = row.get('avg_loss_r') or 0.0
            row['expectancy'] = round(wr * avg_win + (1.0 - wr) * avg_loss, 3) if decisive else None
            result.append(row)
        return result

    async def get_pnl_attribution_summary(self, days: int = DBC.DEFAULT_PERFORMANCE_WINDOW_DAYS) -> Dict:
        """Return a structured P&L attribution breakdown across three dimensions.

        Answers the key question: **which component is generating alpha?**

        Returns a dict with three cross-tabs:

        ``by_market_state``  — same as get_performance_stats_by_market_state()
        ``by_ai_bucket``     — same as get_ai_attribution_stats()
        ``by_strategy``      — win rate, avg_r, and trade count per strategy

        Each row includes ``win_rate`` (wins / decisive trades), ``avg_r``,
        ``wins``, ``losses``, and ``signals``.  This lets a single Telegram
        report show all three attribution slices side-by-side.
        """
        async with self._pool.reader() as conn:
            cursor_s = await conn.execute("""
                SELECT
                    s.strategy,
                    COUNT(*)                                                              AS signals,
                    SUM(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN 1 ELSE 0 END) AS trades,
                    SUM(CASE WHEN o.outcome = 'WIN'  THEN 1 ELSE 0 END)                  AS wins,
                    SUM(CASE WHEN o.outcome = 'LOSS' THEN 1 ELSE 0 END)                  AS losses,
                    AVG(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN o.pnl_r END) AS avg_r
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
                  AND s.strategy IS NOT NULL
                GROUP BY s.strategy
                ORDER BY wins DESC
            """, (f'-{days} days',))
            by_strategy_rows = await cursor_s.fetchall()
            await cursor_s.close()

        def _add_wr(rows):
            result = []
            for r in rows:
                row = dict(r)
                decisive = (row.get('wins') or 0) + (row.get('losses') or 0)
                row['win_rate'] = round((row.get('wins') or 0) / decisive, 3) if decisive else None
                result.append(row)
            return result

        by_strategy = _add_wr(by_strategy_rows)
        by_state = await self.get_performance_stats_by_market_state(days=days)
        by_ai = await self.get_ai_attribution_stats(days=days)

        return {
            "days": days,
            "by_market_state": by_state,
            "by_ai_bucket": by_ai,
            "by_strategy": by_strategy,
        }

    # ── Watchlist ─────────────────────────────────────────────

    async def upsert_watchlist(self, symbol: str, score: float, reasons: List[str]):
        """Add or update a symbol on the watchlist"""
        async with self._pool.writer() as conn:
            await self._exec("""
                INSERT INTO watchlist (symbol, score, reasons, expires_at)
                VALUES (?, ?, ?, datetime('now', '+6 hours'))
                ON CONFLICT(symbol) DO UPDATE SET
                    score = excluded.score,
                    reasons = excluded.reasons,
                    detected_at = datetime('now'),
                    expires_at = datetime('now', '+6 hours')
            """, (symbol, score, json.dumps(reasons)))
            await conn.commit()

    async def get_watchlist(self) -> List[Dict]:
        """Get active watchlist symbols"""
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT * FROM watchlist
                WHERE expires_at > datetime('now')
                ORDER BY score DESC
                LIMIT 20
            """)
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(r) for r in rows]

    async def mark_user_watching(self, symbol: str, message_id: int):
        """Mark symbol as user-watched (pressed Watch button)"""
        async with self._pool.writer() as conn:
            await self._exec("""
                UPDATE watchlist SET user_watching = 1, telegram_message_id = ?
                WHERE symbol = ?
            """, (message_id, symbol))
            await conn.commit()

    async def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        async with self._pool.writer() as conn:
            await self._exec(
                "DELETE FROM watchlist WHERE symbol = ?", (symbol,)
            )
            await conn.commit()

    # ── Symbol tiers ─────────────────────────────────────────

    async def upsert_symbol_tier(self, symbol: str, tier: int, volume_24h: float):
        """Update symbol tier"""
        async with self._pool.writer() as conn:
            await self._exec("""
                INSERT INTO symbol_tiers (symbol, tier, volume_24h, last_scan)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(symbol) DO UPDATE SET
                    tier = excluded.tier,
                    volume_24h = excluded.volume_24h,
                    last_scan = datetime('now'),
                    scan_count = scan_count + 1
            """, (symbol, tier, volume_24h))
            await conn.commit()

    async def record_promotion(self, symbol: str, from_tier: int, to_tier: int):
        """Record a tier promotion"""
        async with self._pool.writer() as conn:
            await self._exec("""
                UPDATE symbol_tiers
                SET last_promoted = datetime('now'), promoted_from = ?
                WHERE symbol = ?
            """, (from_tier, symbol))
            await conn.commit()

    # ── Circuit Breaker ───────────────────────────────────────

    async def get_circuit_breaker_state(self) -> Dict:
        """Get current circuit breaker state"""
        async with self._pool.reader() as conn:
            cursor = await conn.execute(
                "SELECT * FROM circuit_breaker WHERE id = 1"
            )
            row = await cursor.fetchone()
            await cursor.close()
            return dict(row) if row else {}

    async def set_circuit_breaker(self, is_active: bool, reason: str = None, resume_at: str = None):
        """Update circuit breaker state"""
        async with self._pool.writer() as conn:
            await self._exec("""
                UPDATE circuit_breaker SET
                    is_active = ?,
                    reason = ?,
                    activated_at = CASE WHEN ? = 1 THEN datetime('now') ELSE activated_at END,
                    resume_at = ?
                WHERE id = 1
            """, (1 if is_active else 0, reason, 1 if is_active else 0, resume_at))
            await conn.commit()

    # ── Events ─────────────────────────────────────────────

    async def log_event(self, event_type: str, message: str, data: Dict = None):
        """Log a system event"""
        async with self._pool.writer() as conn:
            await self._exec("""
                INSERT INTO events (event_type, message, data)
                VALUES (?, ?, ?)
            """, (event_type, message, json.dumps(data) if data else None))
            await conn.commit()

    async def get_recent_events(self, limit: int = DBC.RECENT_EVENTS_DEFAULT_LIMIT) -> List[Dict]:
        """Get recent system events"""
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT * FROM events ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(r) for r in rows]

    # ── C Signal Log (Phase 2 training data) ────────────────────────

    async def log_c_signal(self, rec) -> int:
        """Log a C signal evaluation to DB for Phase 2 learning."""
        try:
            async with self._pool.writer() as conn:
                cursor = await conn.execute(
                    """INSERT INTO c_signal_log
                       (signal_id, symbol, strategy, direction, regime, context_key,
                        confidence, up_score, phase1_up, phase2_up, phase2_weight,
                        sent_to_user)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rec.signal_id, rec.symbol, rec.strategy, rec.direction,
                        rec.regime, rec.context_key, rec.confidence_at_detection,
                        rec.up_score, rec.phase1_up, rec.phase2_up, rec.phase2_weight,
                        1 if rec.sent_to_user else 0,
                    )
                )
                await conn.commit()
                _last_id = cursor.lastrowid
                await cursor.close()
                return _last_id
        except Exception as e:
            logger.warning(f"log_c_signal failed: {e}")
            return 0

    async def update_c_signal_outcome(self, signal_id: int,
                                       upgraded: bool,
                                       latency_min: Optional[float]):
        """Update C signal outcome after resolution."""
        try:
            async with self._pool.writer() as conn:
                await self._exec(
                    """UPDATE c_signal_log
                       SET upgraded=?, upgrade_latency_min=?,
                           expired=?
                       WHERE signal_id=?""",
                    (
                        1 if upgraded else 0,
                        latency_min,
                        0 if upgraded else 1,
                        signal_id,
                    )
                )
                await conn.commit()
        except Exception as e:
            logger.warning(f"update_c_signal_outcome failed: {e}")

    async def save_upgrade_posterior(self, context_key: str, posterior) -> None:
        """Persist a Bayesian posterior to DB."""
        try:
            async with self._pool.writer() as conn:
                await self._exec(
                    """INSERT INTO upgrade_posteriors
                       (context_key, alpha, beta, sample_count, last_updated)
                       VALUES (?,?,?,?,datetime('now'))
                       ON CONFLICT(context_key) DO UPDATE SET
                         alpha=excluded.alpha,
                         beta=excluded.beta,
                         sample_count=excluded.sample_count,
                         last_updated=excluded.last_updated""",
                    (context_key, posterior.alpha, posterior.beta, posterior.sample_count)
                )
                await conn.commit()
        except Exception as e:
            logger.warning(f"save_upgrade_posterior failed: {e}")

    async def load_upgrade_posterior(self, context_key: str) -> Optional[Dict]:
        """Load a single posterior from DB."""
        try:
            async with self._pool.reader() as conn:
                cursor = await conn.execute(
                    "SELECT alpha, beta FROM upgrade_posteriors WHERE context_key=?",
                    (context_key,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row:
                    return {'alpha': row[0], 'beta': row[1]}
        except Exception as e:
            logger.debug(f"load_upgrade_posterior failed: {e}")
        return None

    async def load_all_upgrade_posteriors(self) -> List[Dict]:
        """Load all posteriors at startup."""
        try:
            async with self._pool.reader() as conn:
                cursor = await conn.execute(
                    "SELECT context_key, alpha, beta FROM upgrade_posteriors"
                )
                rows = await cursor.fetchall()
                await cursor.close()
                return [
                    {'context_key': r[0], 'alpha': r[1], 'beta': r[2]}
                    for r in rows
                ]
        except Exception as e:
            logger.warning(f"load_all_upgrade_posteriors failed: {e}")
            return []

    async def get_upgrade_stats(self) -> Dict:
        """Aggregate C signal stats for /status and diagnostics."""
        try:
            async with self._pool.reader() as conn:
                cursor = await conn.execute(
                    f"""SELECT
                         COUNT(*) as total,
                         SUM(sent_to_user) as sent,
                         SUM(CASE WHEN upgraded=1 THEN 1 ELSE 0 END) as upgraded,
                         SUM(CASE WHEN expired=1 THEN 1 ELSE 0 END) as expired,
                         AVG(CASE WHEN upgraded=1 THEN upgrade_latency_min END) as avg_latency
                       FROM c_signal_log
                       WHERE created_at >= datetime('now', '-{DBC.DEFAULT_PERFORMANCE_WINDOW_DAYS} days')"""
                )
                row = await cursor.fetchone()
                await cursor.close()
            if row:
                total = row[0] or 0
                sent  = row[1] or 0
                upg   = row[2] or 0
                exp   = row[3] or 0
                lat   = row[4]
                return {
                    'total_30d':      total,
                    'sent_30d':       sent,
                    'suppressed_30d': total - sent,
                    'upgraded_30d':   upg,
                    'expired_30d':    exp,
                    'upgrade_rate':   f"{upg/sent*100:.1f}%" if sent > 0 else "—",
                    'avg_latency_min': f"{lat:.1f}" if lat else "—",
                }
        except Exception as e:
            logger.warning(f"get_upgrade_stats failed: {e}")
        return {}


    # ── Learning State (Fix L7) ──────────────────────────────

    async def save_learning_state(self, key: str, state: Dict) -> None:
        """Upsert learning loop state — single row per key, not append-only (Fix L7)"""
        try:
            async with self._pool.writer() as conn:
                await self._exec("""
                    INSERT INTO learning_state (key, value, updated_at)
                    VALUES (?, ?, datetime('now'))
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                """, (key, json.dumps(state)))
                await conn.commit()
        except Exception as e:
            logger.error(f"save_learning_state failed: {e}")

    async def load_learning_state(self, key: str) -> Optional[Dict]:
        """Load learning loop state by key (Fix L7)"""
        try:
            async with self._pool.reader() as conn:
                cursor = await conn.execute(
                    "SELECT value FROM learning_state WHERE key = ?", (key,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.debug(f"load_learning_state failed: {e}")
        return None

    async def save_rejected_signal(self, signal: Dict[str, Any], reject_reason: str) -> int:
        """Save a rejected signal to DB for analytics (Fix S5)"""
        try:
            async with self._pool.writer() as conn:
                cursor = await conn.execute("""
                    INSERT INTO signals (
                        symbol, direction, strategy, confidence,
                        entry_low, entry_high, stop_loss,
                        tp1, tp2, tp3, rr_ratio,
                        timeframe, regime, sector, tier,
                        confluence, raw_scores,
                        outcome, reject_reason,
                        p_win, ev_r, alpha_grade, evidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'REJECTED', ?, ?, ?, ?, ?)
                """, (
                    signal.get('symbol'), signal.get('direction'), signal.get('strategy'),
                    signal.get('confidence', 0),
                    signal.get('entry_low', 0), signal.get('entry_high', 0),
                    signal.get('stop_loss', 0), signal.get('tp1', 0),
                    signal.get('tp2'), signal.get('tp3'), signal.get('rr_ratio'),
                    signal.get('timeframe'), signal.get('regime'),
                    signal.get('sector'), signal.get('tier'),
                    json.dumps(signal.get('confluence', [])),
                    json.dumps(signal.get('raw_scores', {})),
                    reject_reason,
                    signal.get('p_win', 0.55), signal.get('ev_r', 0.0),
                    signal.get('alpha_grade', 'C'),
                    _safe_evidence(signal.get('evidence', {})),
                ))
                await conn.commit()
                _last_id = cursor.lastrowid
                await cursor.close()
                return _last_id
        except Exception as e:
            logger.debug(f"save_rejected_signal failed: {e}")
            return 0



    # ── Open Positions (PHASE 1 FIX: NO-RESTART) ──────────────

    async def save_open_position(self, signal_id: int, symbol: str, direction: str,
                                  strategy: str, entry_price: float, size_usdt: float,
                                  risk_usdt: float, stop_loss: float, tp1: float,
                                  tp2: float, tp3: float = None, rr_ratio: float = 0.0,
                                  sector: str = '', message_id: int = None,
                                  correlation_to_btc: float = 0.7,
                                  alpha_grade: str = 'B',
                                  confidence: float = 0.0, **kwargs) -> None:
        """Persist an open position to DB for recovery on restart."""
        try:
            async with self._pool.writer() as conn:
                await self._exec("""
                    INSERT OR REPLACE INTO open_positions
                    (signal_id, symbol, direction, strategy, entry_price, size_usdt,
                     risk_usdt, stop_loss, tp1, tp2, tp3, rr_ratio, sector, message_id,
                     correlation_to_btc, alpha_grade, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (signal_id, symbol, direction, strategy, entry_price, size_usdt,
                        risk_usdt, stop_loss, tp1, tp2, tp3, rr_ratio, sector, message_id,
                        correlation_to_btc,
                        alpha_grade,
                        confidence))
                await conn.commit()
        except Exception as e:
            logger.error(f"save_open_position failed: {e}")

    async def update_signal_exec_state(self, signal_id: int, exec_state: str) -> None:
        """Persist execution engine state transitions to DB so they survive restart."""
        try:
            async with self._pool.writer() as conn:
                await self._exec(
                    "UPDATE signals SET exec_state = ? WHERE id = ?",
                    (exec_state, signal_id)
                )
                await conn.commit()
        except Exception as e:
            logger.debug(f"update_signal_exec_state failed: {e}")

    async def close_open_position(self, signal_id: int) -> None:
        """Remove a position from the open_positions table on close."""
        try:
            async with self._pool.writer() as conn:
                await self._exec(
                    "DELETE FROM open_positions WHERE signal_id = ?", (signal_id,)
                )
                await conn.commit()
        except Exception as e:
            logger.error(f"close_open_position failed: {e}")

    async def load_open_positions(self) -> list:
        """Load all open positions on startup for recovery."""
        try:
            async with self._pool.reader() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM open_positions ORDER BY opened_at ASC"
                )
                rows = await cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                await cursor.close()
                return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            logger.error(f"load_open_positions failed: {e}")
            return []

    async def get_recent_signal_for_symbol(self, symbol: str, max_age_hours: int = DBC.SIGNAL_MAX_AGE_RECOVERY_HOURS) -> Optional[Dict]:
        """Find the most recent signal for a symbol (used in restart recovery)."""
        try:
            async with self._pool.reader() as conn:
                cursor = await conn.execute("""
                    SELECT * FROM signals
                    WHERE symbol = ? AND created_at > datetime('now', ?)
                      AND outcome IS NULL
                    ORDER BY created_at DESC LIMIT 1
                """, (symbol, f'-{max_age_hours} hours'))
                row = await cursor.fetchone()
                if row:
                    cols = [d[0] for d in cursor.description]
                    await cursor.close()
                    return dict(zip(cols, row))
                await cursor.close()
        except Exception as e:
            logger.debug(f"get_recent_signal_for_symbol failed: {e}")
        return None


    async def _cleanup_expired_dedup(self) -> None:
        """FIX-12: Remove expired entries to prevent unbounded growth."""
        try:
            async with self._pool.writer() as conn:
                await self._exec(
                    "DELETE FROM signal_dedup WHERE created_at < datetime('now', '-48 hours')"
                )
                await self._exec(
                    f"""DELETE FROM signals 
                       WHERE outcome IN ('REJECTED', 'EXPIRED') 
                       AND created_at < datetime('now', '-{DBC.CLEANUP_DELETED_SIGNALS_DAYS} days')"""
                )
                await conn.commit()
        except Exception as e:
            logger.debug(f"cleanup skipped: {e}")


    async def get_equity_curve(self, days: int = DBC.EQUITY_CURVE_DEFAULT_DAYS) -> list:
        """
        Returns daily cumulative R for equity curve chart.
        Each row: {date, daily_r, cumulative_r, win_count, loss_count}
        """
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT
                    date(s.created_at) as trade_date,
                    SUM(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN o.pnl_r ELSE 0 END) as daily_r,
                    SUM(CASE WHEN o.outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN o.outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    COUNT(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN 1 END) as trades
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
                  AND o.outcome IS NOT NULL
                GROUP BY date(s.created_at)
                ORDER BY trade_date ASC
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            await cursor.close()
        result = []
        cumulative = 0.0
        for row in rows:
            d = dict(row)
            cumulative += (d['daily_r'] or 0)
            d['cumulative_r'] = round(cumulative, 3)
            d['daily_r'] = round(d['daily_r'] or 0, 3)
            result.append(d)
        return result

    async def get_quant_metrics(self, days: int = DBC.DEFAULT_PERFORMANCE_WINDOW_DAYS) -> dict:
        """
        Returns Sharpe, Sortino, Calmar, max drawdown, profit factor,
        consecutive wins/losses, and R distribution for the performance page.
        """
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT o.pnl_r, o.outcome, s.created_at
                FROM signals s
                JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
                  AND o.outcome IN ('WIN', 'LOSS', 'BREAKEVEN')
                  AND o.pnl_r IS NOT NULL
                ORDER BY s.created_at ASC
            """, (f'-{days} days',))
            rows = [dict(r) for r in await cursor.fetchall()]
            await cursor.close()

        if not rows:
            return {}

        import math
        pnls = [r['pnl_r'] for r in rows]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total = len(pnls)
        win_count  = len(wins)
        loss_count = len(losses)
        avg_r = sum(pnls) / total if total else 0

        # Sharpe (R-based, no risk-free rate for crypto)
        if total > 1:
            mean = avg_r
            variance = sum((p - mean) ** 2 for p in pnls) / (total - 1)
            std = math.sqrt(variance) if variance > 0 else 0.001
            sharpe = (mean / std) * math.sqrt(252) if std > 0 else 0
        else:
            sharpe = 0

        # Sortino (downside deviation only)
        downside = [p for p in pnls if p < 0]
        if downside and total > 1:
            ds_var = sum(p**2 for p in downside) / total
            ds_std = math.sqrt(ds_var) if ds_var > 0 else 0.001
            sortino = (avg_r / ds_std) * math.sqrt(252) if ds_std > 0 else 0
        else:
            sortino = sharpe

        # Max Drawdown (peak-to-trough in R)
        peak = 0.0
        equity = 0.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        # Calmar = total R / max drawdown
        total_r = sum(pnls)
        calmar = abs(total_r / max_dd) if max_dd > 0 else 0

        # Profit factor = gross wins / abs(gross losses)
        gross_win  = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 0

        # Consecutive streaks
        max_consec_w = 0
        max_consec_l = 0
        cur_w = cur_l = 0
        for p in pnls:
            if p > 0:
                cur_w += 1; cur_l = 0
                max_consec_w = max(max_consec_w, cur_w)
            elif p < 0:
                cur_l += 1; cur_w = 0
                max_consec_l = max(max_consec_l, cur_l)
            else:
                cur_w = cur_l = 0

        # R distribution buckets: <-2, -2to-1, -1to0, 0to1, 1to2, 2to3, >3
        buckets = {'lt_-2': 0, '-2_-1': 0, '-1_0': 0, '0_1': 0, '1_2': 0, '2_3': 0, 'gt_3': 0}
        for p in pnls:
            if p < -2: buckets['lt_-2'] += 1
            elif p < -1: buckets['-2_-1'] += 1
            elif p < 0: buckets['-1_0'] += 1
            elif p < 1: buckets['0_1'] += 1
            elif p < 2: buckets['1_2'] += 1
            elif p < 3: buckets['2_3'] += 1
            else: buckets['gt_3'] += 1

        # VaR and CVaR (95% confidence, R-denominated)
        # Value at Risk: the loss threshold below which only 5% of trades fall
        # CVaR (Expected Shortfall): average of the worst 5% of outcomes
        sorted_pnls = sorted(pnls)
        var_idx = max(0, int(len(sorted_pnls) * 0.05) - 1)
        var_95  = round(sorted_pnls[var_idx], 3) if sorted_pnls else 0.0
        cvar_95 = round(
            sum(sorted_pnls[:var_idx + 1]) / (var_idx + 1), 3
        ) if var_idx >= 0 and sorted_pnls else 0.0

        # Omega ratio: probability-weighted gains / probability-weighted losses above threshold 0
        gains_omega  = sum(max(0, p) for p in pnls)
        losses_omega = sum(max(0, -p) for p in pnls)
        omega = round(gains_omega / losses_omega, 3) if losses_omega > 0 else 0.0

        return {
            'total_trades': total,
            'win_count': win_count,
            'loss_count': loss_count,
            'avg_r': round(avg_r, 3),
            'total_r': round(total_r, 3),
            'sharpe': round(sharpe, 2),
            'sortino': round(sortino, 2),
            'calmar': round(calmar, 2),
            'max_drawdown_r': round(max_dd, 3),
            'profit_factor': round(profit_factor, 2),
            'max_consec_wins': max_consec_w,
            'max_consec_losses': max_consec_l,
            'r_distribution': buckets,
            'avg_win_r': round(sum(wins)/len(wins), 3) if wins else 0,
            'avg_loss_r': round(sum(losses)/len(losses), 3) if losses else 0,
            # New institutional risk metrics
            'var_95': var_95,    # 95% VaR in R — e.g. -1.2 means "in 95% of trades you lose ≤1.2R"
            'cvar_95': cvar_95,  # CVaR: expected loss in worst 5% of trades
            'omega_ratio': omega, # >1.0 = gains dominate losses (target >1.5)
        }

    async def get_monthly_pnl(self, months: int = DBC.MONTHLY_PNL_DEFAULT_MONTHS) -> list:
        """
        Returns monthly P&L for the heatmap calendar.
        Each row: {year_month, total_r, wins, losses, trades}
        """
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT
                    strftime('%Y-%m', s.created_at) as month,
                    SUM(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN o.pnl_r ELSE 0 END) as total_r,
                    SUM(CASE WHEN o.outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN o.outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    COUNT(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN 1 END) as trades
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
                GROUP BY strftime('%Y-%m', s.created_at)
                ORDER BY month DESC
                LIMIT ?
            """, (f'-{months} months', months))
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(r) for r in rows]

    async def get_regime_performance(self, days: int = DBC.EQUITY_CURVE_DEFAULT_DAYS) -> list:
        """Performance breakdown by regime and direction."""
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT
                    s.regime,
                    s.direction,
                    COUNT(*) as signals,
                    SUM(CASE WHEN o.outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN o.outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN o.outcome IN ('WIN','LOSS') THEN o.pnl_r END) as avg_r,
                    SUM(CASE WHEN o.outcome IN ('WIN','LOSS','BREAKEVEN') THEN o.pnl_r ELSE 0 END) as total_r
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id
                WHERE s.created_at > datetime('now', ?)
                  AND s.regime IS NOT NULL
                GROUP BY s.regime, s.direction
                ORDER BY total_r DESC
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(r) for r in rows]

    async def get_signal_funnel(self, hours: int = DBC.DEFAULT_QUERY_WINDOW_HOURS) -> dict:
        """Signal pipeline funnel: raw → filtered → published → taken."""
        async with self._pool.reader() as conn:
            # Published signals
            cursor = await conn.execute("""
                SELECT COUNT(*) as published,
                       SUM(CASE WHEN alpha_grade = 'A+' THEN 1 ELSE 0 END) as grade_ap,
                       SUM(CASE WHEN alpha_grade = 'A'  THEN 1 ELSE 0 END) as grade_a,
                       SUM(CASE WHEN alpha_grade LIKE 'B%' THEN 1 ELSE 0 END) as grade_b,
                       SUM(CASE WHEN alpha_grade = 'C' THEN 1 ELSE 0 END) as grade_c,
                       SUM(CASE WHEN zone_reached = 1 THEN 1 ELSE 0 END) as zone_reached
                FROM signals
                WHERE created_at > datetime('now', ?)
            """, (f'-{hours} hours',))
            row = dict(await cursor.fetchone() or {})
            await cursor.close()

            # Taken (user logged via outcome buttons)
            cursor2 = await conn.execute("""
                SELECT COUNT(DISTINCT o.signal_id) as taken
                FROM outcomes o
                JOIN signals s ON o.signal_id = s.id
                WHERE s.created_at > datetime('now', ?)
                  AND o.outcome != 'SKIPPED'
            """, (f'-{hours} hours',))
            taken_row = dict(await cursor2.fetchone() or {})
            await cursor2.close()

        row['taken'] = taken_row.get('taken', 0)
        return row



    async def set_signal_note(self, signal_id: int, note: str) -> bool:
        """DASH-4: Save a trader note for a signal."""
        async with self._pool.writer() as conn:
            try:
                await self._exec("""
                    UPDATE signals SET user_notes = ? WHERE id = ?
                """, (note.strip()[:DBC.SIGNAL_NOTE_MAX_CHARS], signal_id))
                await conn.commit()
                return True
            except Exception as e:
                logger.error(f"set_signal_note error: {e}")
                return False

    async def get_signal_notes(self, limit: int = 50) -> list:
        """DASH-4: Get recent signals that have notes attached."""
        async with self._pool.reader() as conn:
            cursor = await conn.execute("""
                SELECT id, symbol, direction, strategy, alpha_grade,
                       created_at, user_notes as notes, outcome, pnl_r
                FROM signals
                WHERE user_notes IS NOT NULL AND user_notes != ''
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(r) for r in rows]

    async def get_strategy_direction_breakdown(self, hours: int = DBC.DEFAULT_QUERY_WINDOW_HOURS):
        """Strategy direction bias breakdown for Sentinel dashboard."""
        try:
            async with self._pool.reader() as conn:
                # D-6 FIX: signals.created_at is a TEXT ISO datetime column
                # (DEFAULT (datetime('now'))). Comparing against time.time() (a
                # REAL unix timestamp) produces a string vs float comparison that
                # silently returns 0 rows on most SQLite builds.  Use SQLite's
                # datetime arithmetic so both sides are the same type.
                cursor = await conn.execute("""
                    SELECT strategy,
                           SUM(CASE WHEN direction='LONG'  THEN 1 ELSE 0 END) AS long_count,
                           SUM(CASE WHEN direction='SHORT' THEN 1 ELSE 0 END) AS short_count
                    FROM signals
                    WHERE created_at > datetime('now', ?) AND strategy IS NOT NULL
                    GROUP BY strategy
                """, (f"-{hours} hours",))
                rows = await cursor.fetchall()
                await cursor.close()
                return [dict(r) for r in rows]
        except Exception:
            return []

    async def vacuum_if_needed(self) -> None:
        """Call weekly to reclaim space after deletions."""
        try:
            import os
            if os.path.getsize(self._db_path) > DBC.VACUUM_SIZE_THRESHOLD_BYTES:
                async with self._pool.writer():
                    await self._exec("VACUUM")
                logger.info("Database vacuumed")
        except Exception as e:
            logger.debug(f"VACUUM skipped: {e}")

    async def backup(self) -> bool:
        """
        D4-FIX: Create a daily rolling backup of the SQLite database.
        Uses SQLite's built-in online backup API (VACUUM INTO) so the backup
        is consistent even while writes are in progress (WAL mode safe).
        Keeps two backup slots: .bak and .bak.prev for recovery.

        Returns True on success, False on failure (non-fatal — logged only).
        """
        try:
            import os, shutil
            db_path = str(self._db_path)
            bak_path = db_path + '.bak'
            prev_path = db_path + '.bak.prev'

            # Validate that bak_path is a safe filesystem path before using it
            # in a SQL statement.  The path is derived entirely from self._db_path
            # (set at construction from config), but we defend against any quote
            # characters that would break or escape the SQL string literal.
            if "'" in bak_path or "\x00" in bak_path:
                raise ValueError(
                    f"Backup path contains unsafe characters: {bak_path!r}"
                )

            # Rotate: .bak → .bak.prev before overwriting
            if os.path.exists(bak_path):
                shutil.copy2(bak_path, prev_path)

            # SQLite VACUUM INTO creates a consistent point-in-time copy.
            # The path is embedded as a string literal because SQLite's VACUUM INTO
            # does not support bound parameters.  The quote-check above ensures it
            # is safe to interpolate.
            async with self._pool.writer():
                await self._exec(f"VACUUM INTO '{bak_path}'")

            bak_size_mb = os.path.getsize(bak_path) / (1024 * 1024)
            logger.info(
                f"✅ Database backup created: {bak_path} ({bak_size_mb:.1f} MB)"
            )
            return True
        except Exception as e:
            logger.warning(f"⚠️ Database backup failed (non-fatal): {e}")
            return False


# ── Singleton instance ─────────────────────────────────────

    async def save_card_state(self, signal_id: int, message_id: int,
                              chat_id: int, symbol: str, direction: str,
                              grade: str, state: str = "PENDING", confirmed: bool = False):
        """Persist active card state so button presses survive bot restarts."""
        import time as _t
        now = _t.time()
        async with self._pool.writer() as conn:
            # D-7 FIX: use _exec() so the aiosqlite cursor is always closed via
            # the context manager. Bare `await conn.execute(...)` without storing
            # and closing the cursor leaves the cursor on the background thread
            # queue, which causes all subsequent execute() calls on this connection
            # to block indefinitely after ~10–20 writes.
            await self._exec("""
                INSERT OR REPLACE INTO card_state
                (signal_id, message_id, chat_id, symbol, direction,
                 grade, state, confirmed, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,
                    COALESCE((SELECT created_at FROM card_state WHERE signal_id=?),?),?)
            """, (signal_id, message_id, chat_id, symbol, direction,
                  grade, state, 1 if confirmed else 0, signal_id, now, now))
            await conn.commit()

    async def load_active_card_states(self) -> list:
        """Reload card states on bot startup — restores button keyboards."""
        import time as _t
        cutoff = _t.time() - 86400 * 3  # last 3 days
        async with self._pool.reader() as conn:
            cur = await conn.execute("""
                SELECT signal_id, message_id, chat_id, symbol, direction,
                       grade, state, confirmed, created_at
                FROM card_state WHERE created_at > ?
                ORDER BY created_at DESC
            """, (cutoff,))
            rows = await cur.fetchall()
            await cur.close()
        cols = ['signal_id','message_id','chat_id','symbol','direction',
                'grade','state','confirmed','created_at']
        return [dict(zip(cols, r)) for r in rows]

    async def update_card_state(self, signal_id: int, state: str,
                                confirmed: bool = None):
        """Update card state when trade advances."""
        import time as _t
        async with self._pool.writer() as conn:
            # D-7 FIX: use _exec() to close cursor immediately (see save_card_state)
            if confirmed is not None:
                await self._exec(
                    "UPDATE card_state SET state=?,confirmed=?,updated_at=? WHERE signal_id=?",
                    (state, 1 if confirmed else 0, _t.time(), signal_id))
            else:
                await self._exec(
                    "UPDATE card_state SET state=?,updated_at=? WHERE signal_id=?",
                    (state, _t.time(), signal_id))
            await conn.commit()

    async def delete_card_state(self, signal_id: int):
        """Remove card state on signal close."""
        async with self._pool.writer() as conn:
            # D-7 FIX: use _exec() to close cursor immediately (see save_card_state)
            await self._exec(
                "DELETE FROM card_state WHERE signal_id=?", (signal_id,))
            await conn.commit()

    # ── Strategy persistence (C1/C2 fix) ─────────────────────

    async def save_strategy_state(
        self,
        strategy: str,
        ewma_win_rate: float,
        weight_mult: float,
        is_disabled: bool,
        disabled_until: float,
        disabled_at: float,
        r_outcomes_json: str,
    ) -> None:
        """Persist strategy adaptive state to survive restarts.

        Uses INSERT OR REPLACE inside a transaction so partial writes from
        crashes leave at most one stale row, never a corrupt half-row.
        Called by PerformanceTracker after every record_outcome() and
        suppress_strategy() call.
        """
        async with self._pool.writer() as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO strategy_persistence_v1
                   (strategy, ewma_win_rate, weight_mult, is_disabled,
                    disabled_until, disabled_at, r_outcomes_json, updated_at, version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    strategy,
                    ewma_win_rate,
                    weight_mult,
                    1 if is_disabled else 0,
                    disabled_until,
                    disabled_at,
                    r_outcomes_json,
                    time.time(),
                ),
            )
            await conn.commit()

    async def load_all_strategy_states(self) -> List[Dict]:
        """Load all persisted strategy adaptive states at startup.

        Returns a list of dicts with keys: strategy, ewma_win_rate,
        weight_mult, is_disabled, disabled_until, disabled_at, r_outcomes_json.
        Returns [] if table is empty or missing.
        """
        try:
            async with self._pool.reader() as conn:
                cur = await conn.execute(
                    """SELECT strategy, ewma_win_rate, weight_mult, is_disabled,
                              disabled_until, disabled_at, r_outcomes_json
                       FROM strategy_persistence_v1"""
                )
                rows = await cur.fetchall()
                await cur.close()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"load_all_strategy_states failed (non-fatal): {e}")
            return []

    # ── Risk state persistence (C3 fix) ──────────────────────

    async def save_risk_state(
        self,
        daily_loss_pct: float,
        peak_capital: float,
        hard_kill: bool,
        resume_at: Optional[float],
        consecutive: int,
    ) -> None:
        """Persist circuit breaker risk state to survive restarts.

        Uses INSERT OR REPLACE (singleton row id=1) inside a transaction.
        Called by CircuitBreaker after record_loss(), _trigger(),
        reset_daily(), and manually_resume().
        """
        async with self._pool.writer() as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO risk_state_v1
                   (id, daily_loss_pct, peak_capital, hard_kill, resume_at,
                    consecutive, updated_at, version)
                   VALUES (1, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    daily_loss_pct,
                    peak_capital,
                    1 if hard_kill else 0,
                    resume_at,
                    consecutive,
                    time.time(),
                ),
            )
            await conn.commit()

    async def load_risk_state(self) -> Optional[Dict]:
        """Load persisted circuit breaker state at startup.

        Returns a dict with keys: daily_loss_pct, peak_capital, hard_kill,
        resume_at, consecutive. Returns None if no row exists yet.
        """
        try:
            async with self._pool.reader() as conn:
                cur = await conn.execute(
                    """SELECT daily_loss_pct, peak_capital, hard_kill,
                              resume_at, consecutive
                       FROM risk_state_v1 WHERE id = 1"""
                )
                row = await cur.fetchone()
                await cur.close()
                return dict(row) if row else None
        except Exception as e:
            logger.warning(f"load_risk_state failed (non-fatal): {e}")
            return None


    async def save_tracked_signal(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        strategy: str,
        state: str,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        confidence: float,
        grade: str = 'B',
        message_id: Optional[int] = None,
        created_at: Optional[float] = None,
        tp1: float = 0.0,
        tp2: float = 0.0,
        tp3: Optional[float] = None,
        rr_ratio: float = 0.0,
        setup_class: str = 'intraday',
        min_triggers: int = 2,
        has_rejection_candle: bool = False,
        has_structure_shift: bool = False,
        has_momentum_expansion: bool = False,
        has_liquidity_reaction: bool = False,
        entry_price: Optional[float] = None,
        be_stop: Optional[float] = None,
        trail_stop: Optional[float] = None,
        max_r: float = 0.0,
        trail_pct: float = 0.40,
        activated_at: Optional[float] = None,
    ) -> None:
        """Persist a tracked signal (pre- or post-fill) to survive restarts.

        Called by ExecutionEngine on every state change and by OutcomeMonitor
        on entry, TP1, and trailing stop updates.  Uses INSERT OR REPLACE so
        the row is always up-to-date without any partial-write risk.
        """
        try:
            async with self._pool.writer() as conn:
                await conn.execute(
                    """INSERT OR REPLACE INTO tracked_signals_v1
                       (signal_id, symbol, direction, strategy, state,
                        entry_low, entry_high, entry_price, stop_loss,
                        tp1, tp2, tp3, be_stop, trail_stop, max_r,
                        confidence, grade, message_id, created_at, updated_at, activated_at,
                        has_rejection_candle, has_structure_shift,
                        has_momentum_expansion, has_liquidity_reaction,
                        min_triggers, setup_class, rr_ratio, trail_pct)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        signal_id, symbol, direction, strategy, state,
                        entry_low, entry_high, entry_price, stop_loss,
                        tp1, tp2, tp3, be_stop, trail_stop, max_r,
                        confidence, grade, message_id,
                        created_at if created_at is not None else time.time(),
                        time.time(),
                        activated_at,
                        1 if has_rejection_candle else 0,
                        1 if has_structure_shift else 0,
                        1 if has_momentum_expansion else 0,
                        1 if has_liquidity_reaction else 0,
                        min_triggers, setup_class, rr_ratio, trail_pct,
                    ),
                )
                await conn.commit()
        except Exception as e:
            logger.debug(f"save_tracked_signal failed (non-fatal): {e}")

    async def update_tracked_signal(self, signal_id: int, **fields) -> None:
        """Partially update a tracked_signals_v1 row.

        Accepts keyword arguments matching column names.  Always sets
        updated_at = now so freshness is visible in the dashboard.
        Called for lightweight updates (trailing stop ratchet, be_stop set)
        without rewriting every column.
        """
        if not fields:
            return
        try:
            fields['updated_at'] = time.time()
            # SQL injection protection: only allow known column names
            invalid_cols = set(fields.keys()) - DBC.TRACKED_SIGNAL_ALLOWED_COLUMNS
            if invalid_cols:
                logger.warning(f"update_tracked_signal: rejected invalid columns: {invalid_cols}")
                fields = {k: v for k, v in fields.items() if k in DBC.TRACKED_SIGNAL_ALLOWED_COLUMNS}
                if not fields:
                    return
            set_clause = ', '.join(f"{k} = ?" for k in fields)
            values = list(fields.values()) + [signal_id]
            async with self._pool.writer() as conn:
                await conn.execute(
                    f"UPDATE tracked_signals_v1 SET {set_clause} WHERE signal_id = ?",
                    values,
                )
                await conn.commit()
        except Exception as e:
            logger.debug(f"update_tracked_signal failed (non-fatal): {e}")

    async def load_tracked_signals(
        self, states: Optional[List[str]] = None
    ) -> List[Dict]:
        """Load tracked signals from DB, optionally filtered by state list.

        Called at startup before monitors start so both ExecutionEngine and
        OutcomeMonitor can restore their in-memory state from the last run.
        Returns [] on error so startup can proceed safely.
        """
        try:
            async with self._pool.reader() as conn:
                if states:
                    placeholders = ','.join('?' * len(states))
                    sql = (
                        f"SELECT * FROM tracked_signals_v1 "
                        f"WHERE state IN ({placeholders}) "
                        f"ORDER BY created_at ASC"
                    )
                    cursor = await conn.execute(sql, states)
                else:
                    cursor = await conn.execute(
                        "SELECT * FROM tracked_signals_v1 ORDER BY created_at ASC"
                    )
                rows = await cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                await cursor.close()
                return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            logger.warning(f"load_tracked_signals failed (non-fatal): {e}")
            return []

    async def delete_tracked_signal(self, signal_id: int) -> None:
        """Remove a signal from tracked_signals_v1 when it reaches a terminal state.

        Called by ExecutionEngine and OutcomeMonitor on EXPIRED, INVALIDATED,
        FILLED, and all COMPLETED_* outcomes so the table stays compact.
        """
        try:
            async with self._pool.writer() as conn:
                await conn.execute(
                    "DELETE FROM tracked_signals_v1 WHERE signal_id = ?",
                    (signal_id,),
                )
                await conn.commit()
        except Exception as e:
            logger.debug(f"delete_tracked_signal failed (non-fatal): {e}")


db = Database()
