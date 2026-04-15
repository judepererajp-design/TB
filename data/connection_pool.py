"""
TitanBot Pro — Connection Pool
===============================
Async SQLite connection pool for concurrent read access with serialized writes.

SQLite supports multiple concurrent readers via WAL mode, but only one writer
at a time.  This pool maintains a queue of reader connections and a single
dedicated writer connection behind an asyncio.Lock.

Usage:
    pool = ConnectionPool()
    await pool.initialize("data/titan.db")

    async with pool.reader() as conn:
        cur = await conn.execute("SELECT ...")
        ...

    async with pool.writer() as conn:
        await conn.execute("INSERT ...")
        await conn.commit()

    await pool.close()
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import aiosqlite

from config.constants import Database as DBC

logger = logging.getLogger(__name__)


async def _apply_pragmas(conn: aiosqlite.Connection) -> None:
    """Apply standard PRAGMA settings to a connection.

    Every connection (reader or writer) must use the same WAL-mode and
    performance PRAGMAs so that behaviour is identical regardless of
    which connection handles a given query.
    """
    for pragma in (
        "PRAGMA busy_timeout=10000",
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA cache_size=-32000",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA foreign_keys=ON",
    ):
        async with await conn.execute(pragma):
            pass
    await conn.commit()


class ConnectionPool:
    """Async SQLite connection pool: N readers + 1 writer.

    Reader connections are managed via an ``asyncio.Queue``.  The single
    writer connection is guarded by an ``asyncio.Lock`` so only one
    coroutine can write at a time (SQLite constraint).
    """

    def __init__(
        self,
        pool_reader_size: int = DBC.POOL_READER_SIZE,
        pool_writer_size: int = DBC.POOL_WRITER_SIZE,
        acquire_timeout: float = DBC.POOL_ACQUIRE_TIMEOUT_SECS,
    ):
        self._reader_size = pool_reader_size
        self._writer_size = min(pool_writer_size, 1)  # SQLite only supports 1 writer
        self._acquire_timeout = acquire_timeout

        self._reader_pool: Optional[asyncio.Queue] = None
        self._writer_conn: Optional[aiosqlite.Connection] = None
        self._writer_lock = asyncio.Lock()

        self._db_path: Optional[str] = None
        self._closed = True

        # Stats
        self._readers_created = 0
        self._reader_acquires = 0
        self._reader_releases = 0
        self._writer_acquires = 0
        self._initialized_at: Optional[float] = None

    # ── lifecycle ────────────────────────────────────────────

    async def initialize(self, db_path: str) -> None:
        """Create all connections and populate the reader queue."""
        self._db_path = db_path
        self._reader_pool = asyncio.Queue(maxsize=self._reader_size)

        # Writer connection
        self._writer_conn = await aiosqlite.connect(db_path)
        self._writer_conn.row_factory = aiosqlite.Row
        await _apply_pragmas(self._writer_conn)
        logger.debug("ConnectionPool: writer connection ready")

        # Reader connections
        for i in range(self._reader_size):
            conn = await aiosqlite.connect(db_path)
            conn.row_factory = aiosqlite.Row
            await _apply_pragmas(conn)
            await self._reader_pool.put(conn)
            self._readers_created += 1
        logger.debug(
            "ConnectionPool: %d reader connections ready", self._readers_created
        )

        self._closed = False
        self._initialized_at = time.monotonic()
        logger.info(
            "ConnectionPool initialized: %d readers, 1 writer — %s",
            self._reader_size,
            db_path,
        )

    async def close(self) -> None:
        """Close all connections in the pool."""
        if self._closed:
            return
        self._closed = True

        # Drain reader queue
        closed_readers = 0
        if self._reader_pool is not None:
            while not self._reader_pool.empty():
                try:
                    conn = self._reader_pool.get_nowait()
                    await conn.close()
                    closed_readers += 1
                except asyncio.QueueEmpty:
                    break
                except Exception as exc:
                    logger.debug("Error closing reader connection: %s", exc)

        # Close writer
        if self._writer_conn is not None:
            try:
                await self._writer_conn.close()
            except Exception as exc:
                logger.debug("Error closing writer connection: %s", exc)
            self._writer_conn = None

        logger.info(
            "ConnectionPool closed: %d readers, 1 writer", closed_readers
        )

    # ── reader management ────────────────────────────────────

    async def acquire_reader(self) -> aiosqlite.Connection:
        """Borrow a reader connection from the pool.

        Raises ``asyncio.TimeoutError`` if no connection becomes available
        within the configured timeout.
        """
        try:
            conn = await asyncio.wait_for(
                self._reader_pool.get(), timeout=self._acquire_timeout
            )
            self._reader_acquires += 1
            return conn
        except asyncio.TimeoutError:
            logger.error(
                "ConnectionPool: reader acquire timed out after %.1fs "
                "(pool size=%d, acquired=%d, released=%d)",
                self._acquire_timeout,
                self._reader_size,
                self._reader_acquires,
                self._reader_releases,
            )
            raise

    async def release_reader(self, conn: aiosqlite.Connection) -> None:
        """Return a reader connection to the pool."""
        self._reader_releases += 1
        await self._reader_pool.put(conn)

    @asynccontextmanager
    async def reader(self):
        """Async context manager: auto-acquire and release a reader connection."""
        conn = await self.acquire_reader()
        try:
            yield conn
        finally:
            await self.release_reader(conn)

    # ── writer management ────────────────────────────────────

    @asynccontextmanager
    async def writer(self):
        """Async context manager: acquire write lock and yield writer connection.

        Only one coroutine can hold the writer at a time (asyncio.Lock).
        Raises RuntimeError if the writer connection is not initialized.
        """
        async with self._writer_lock:
            self._writer_acquires += 1
            if self._writer_conn is None:
                raise RuntimeError(
                    "ConnectionPool writer connection is None — "
                    "pool not initialized or already closed"
                )
            yield self._writer_conn

    # ── health & stats ───────────────────────────────────────

    def pool_stats(self) -> Dict:
        """Return current pool health metrics for monitoring."""
        uptime = (
            round(time.monotonic() - self._initialized_at, 1)
            if self._initialized_at
            else 0.0
        )
        return {
            "reader_pool_size": self._reader_size,
            "readers_available": (
                self._reader_pool.qsize() if self._reader_pool else 0
            ),
            "readers_in_use": (
                self._reader_size
                - (self._reader_pool.qsize() if self._reader_pool else 0)
            ),
            "reader_acquires": self._reader_acquires,
            "reader_releases": self._reader_releases,
            "writer_acquires": self._writer_acquires,
            "writer_locked": self._writer_lock.locked(),
            "closed": self._closed,
            "uptime_secs": uptime,
        }
