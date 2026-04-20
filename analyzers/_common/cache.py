"""
analyzers._common.cache
=======================
Async-safe TTL cache with a three-state API.

Design goals
------------
1. **MISS vs EMPTY must be distinguishable.** A lot of audit findings
   came from code that returned ``{}`` or ``None`` for both "we never
   fetched this" and "we fetched it and the upstream returned nothing".
   The caller couldn't tell the difference and either re-fetched
   constantly or cached a lie. This cache distinguishes them explicitly
   via :class:`CacheState`.

2. **Single-flight.** Per-key :class:`asyncio.Lock` so a thundering
   herd doesn't trigger N parallel fetches for the same key. Losers
   await the winner's result.

3. **No background timer.** Entries are expired lazily on access. This
   makes the cache safe to use from any event loop (and cheap — no
   sweeper task, no shutdown coordination).

4. **Dependency-light.** stdlib only. Safe under the test-suite numpy
   mock.

Typical usage
-------------
::

    _cache: TTLCache[str, dict] = TTLCache(default_ttl=60.0)

    async def get_funding(symbol):
        entry = await _cache.get(symbol)
        if entry.state is CacheState.HIT:
            return entry.value
        async with _cache.lock(symbol):
            entry = await _cache.get(symbol)  # double-check pattern
            if entry.state is CacheState.HIT:
                return entry.value
            value = await _fetch_funding(symbol)
            await _cache.set(symbol, value)
            return value

Or the simpler ``get_or_fetch`` convenience:

::

    value = await _cache.get_or_fetch(symbol, _fetch_funding)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class CacheState(str, Enum):
    """Three-state cache hit semantics.

    HIT — key present and not expired; ``value`` is authoritative.
    MISS — key was never inserted (or was evicted / expired).
    EMPTY — key was set to a legitimately empty result (``None``,
    ``[]``, ``{}``). ``value`` carries whatever was inserted. Distinct
    from MISS so the caller doesn't re-fetch on every call.
    """

    HIT = "HIT"
    MISS = "MISS"
    EMPTY = "EMPTY"


# Sentinels for use by callers that want compact checks.
MISS = CacheState.MISS
EMPTY = CacheState.EMPTY


@dataclass(frozen=True)
class CacheEntry(Generic[V]):
    """Read-only snapshot returned by :meth:`TTLCache.get`.

    ``value`` is ``None`` only when ``state`` is :attr:`CacheState.MISS`.
    For :attr:`CacheState.EMPTY` it carries whatever "empty" value the
    producer inserted (``None``, ``[]``, ``{}``, …).
    """

    state: CacheState
    value: Optional[V] = None
    age_seconds: Optional[float] = None


def _is_empty_value(v: Any) -> bool:
    """What counts as "legitimately empty" for :class:`CacheState.EMPTY`.

    The distinction matters: an analyzer returning ``{}`` because the
    upstream legitimately has nothing to report is very different from
    an analyzer that hasn't been asked yet (``MISS``).
    """
    if v is None:
        return True
    if isinstance(v, (list, tuple, set, frozenset, dict, str, bytes)):
        return len(v) == 0
    return False


class TTLCache(Generic[K, V]):
    """Async-safe TTL cache with per-key single-flight locking.

    Parameters
    ----------
    default_ttl
        Default time-to-live in seconds. Can be overridden per ``set``.
    max_size
        Optional soft cap. When exceeded, the oldest entries are
        evicted on insert. ``None`` means unbounded (but entries still
        expire on TTL).
    clock
        Callable returning the current wall-clock time. Injected for
        determinism in tests.
    """

    def __init__(
        self,
        default_ttl: float = 60.0,
        max_size: Optional[int] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be > 0 or None")
        self._default_ttl = float(default_ttl)
        self._max_size = max_size
        self._clock = clock
        self._data: Dict[K, tuple[V, float, float, bool]] = {}
        # per-key locks for single-flight fetch. Created lazily.
        self._key_locks: Dict[K, asyncio.Lock] = {}
        # guard _key_locks creation itself from races.
        self._meta_lock = asyncio.Lock()
        # stats (read-only counters for diagnostics; don't rely on perfect atomicity)
        self.hits: int = 0
        self.misses: int = 0
        self.empties: int = 0
        self.evictions: int = 0

    # ---------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------

    async def get(self, key: K) -> CacheEntry[V]:
        """Return the current entry for ``key`` without fetching.

        Never raises — always returns a :class:`CacheEntry`.
        """
        now = self._clock()
        rec = self._data.get(key)
        if rec is None:
            self.misses += 1
            return CacheEntry(state=CacheState.MISS)
        value, inserted_at, expires_at, is_empty = rec
        if now >= expires_at:
            # Lazy eviction
            self._data.pop(key, None)
            self.misses += 1
            return CacheEntry(state=CacheState.MISS)
        age = now - inserted_at
        if is_empty:
            self.empties += 1
            return CacheEntry(state=CacheState.EMPTY, value=value, age_seconds=age)
        self.hits += 1
        return CacheEntry(state=CacheState.HIT, value=value, age_seconds=age)

    async def set(
        self,
        key: K,
        value: V,
        ttl: Optional[float] = None,
        *,
        force_empty: Optional[bool] = None,
    ) -> None:
        """Insert or replace ``key``.

        ``force_empty`` overrides the automatic empty-detection —
        producers that know their value is "no data" despite having a
        non-empty container can pass ``force_empty=True``.
        """
        _ttl = float(ttl) if ttl is not None else self._default_ttl
        if _ttl <= 0:
            raise ValueError("ttl must be > 0")
        now = self._clock()
        empty = force_empty if force_empty is not None else _is_empty_value(value)
        self._data[key] = (value, now, now + _ttl, bool(empty))
        if self._max_size is not None and len(self._data) > self._max_size:
            self._evict_oldest(len(self._data) - self._max_size)

    async def set_empty(
        self,
        key: K,
        value: Optional[V] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Insert an explicitly-empty marker (state will read back as EMPTY)."""
        await self.set(key, value, ttl=ttl, force_empty=True)  # type: ignore[arg-type]

    async def invalidate(self, key: K) -> bool:
        """Drop ``key`` from the cache. Returns ``True`` if it existed."""
        return self._data.pop(key, None) is not None

    async def clear(self) -> None:
        """Drop all entries."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: K) -> bool:
        rec = self._data.get(key)
        if rec is None:
            return False
        _, _, expires_at, _ = rec
        return self._clock() < expires_at

    # ---------------------------------------------------------------
    # Locking / single-flight
    # ---------------------------------------------------------------

    async def lock(self, key: K) -> asyncio.Lock:
        """Return the per-key :class:`asyncio.Lock` for single-flight fetching.

        Callers typically use::

            async with (await cache.lock(key)):
                entry = await cache.get(key)
                if entry.state is CacheState.HIT:
                    return entry.value
                ...
        """
        lock = self._key_locks.get(key)
        if lock is not None:
            return lock
        async with self._meta_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._key_locks[key] = lock
        return lock

    async def get_or_fetch(
        self,
        key: K,
        fetch: Callable[[], Awaitable[V]],
        ttl: Optional[float] = None,
        *,
        allow_empty: bool = True,
    ) -> CacheEntry[V]:
        """Return a cached entry or invoke ``fetch`` under a per-key lock.

        ``allow_empty=False`` causes legitimately-empty results *not* to
        be cached (next caller will re-fetch). Useful for values whose
        emptiness is always transient (e.g. a websocket that just
        reconnected).
        """
        entry = await self.get(key)
        if entry.state is CacheState.HIT or (
            entry.state is CacheState.EMPTY and allow_empty
        ):
            return entry
        lock = await self.lock(key)
        async with lock:
            # Double-check after acquiring the lock.
            entry = await self.get(key)
            if entry.state is CacheState.HIT or (
                entry.state is CacheState.EMPTY and allow_empty
            ):
                return entry
            value = await fetch()
            if _is_empty_value(value) and not allow_empty:
                # Don't cache — pass through as MISS-equivalent to caller.
                return CacheEntry(state=CacheState.MISS, value=value)
            await self.set(key, value, ttl=ttl)
            return await self.get(key)

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _evict_oldest(self, n: int) -> None:
        if n <= 0:
            return
        # Sorted by insertion time (index 1 of the tuple).
        ordered = sorted(self._data.items(), key=lambda kv: kv[1][1])
        for k, _ in ordered[:n]:
            self._data.pop(k, None)
            self.evictions += 1


__all__ = ["TTLCache", "CacheState", "CacheEntry", "MISS", "EMPTY"]
