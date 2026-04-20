"""
Tests for ``analyzers._common`` — the PR-1 cross-cutting infrastructure.

Only stdlib math is exercised (no numpy), so the global ``conftest``
mock of ``numpy`` does not affect these tests.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from analyzers._common import (
    CacheState,
    EMPTY,
    Freshness,
    FreshnessStatus,
    MISS,
    TTLCache,
    blom_percentile,
    clamp,
    is_finite,
    log_zscore,
    mad,
    mean,
    nan_to_num,
    percentile_rank,
    safe_div,
    staleness,
    stdev,
    zscore,
)


# ══════════════════════════════════════════════════════════════════
# freshness
# ══════════════════════════════════════════════════════════════════


class TestStaleness:
    def test_none_age_is_unknown(self):
        assert staleness(None, 60.0) is FreshnessStatus.UNKNOWN

    def test_negative_age_is_unknown(self):
        # Clock skew must not masquerade as fresh.
        assert staleness(-5.0, 60.0) is FreshnessStatus.UNKNOWN

    def test_nan_age_is_unknown(self):
        assert staleness(float("nan"), 60.0) is FreshnessStatus.UNKNOWN

    def test_non_numeric_age_is_unknown(self):
        assert staleness("abc", 60.0) is FreshnessStatus.UNKNOWN  # type: ignore[arg-type]

    def test_fresh_boundary_inclusive(self):
        assert staleness(60.0, 60.0) is FreshnessStatus.FRESH

    def test_aging_range(self):
        assert staleness(61.0, 60.0) is FreshnessStatus.AGING
        assert staleness(120.0, 60.0) is FreshnessStatus.AGING

    def test_stale_range(self):
        assert staleness(121.0, 60.0) is FreshnessStatus.STALE

    def test_zero_ttl_edge(self):
        assert staleness(0.0, 0.0) is FreshnessStatus.FRESH
        assert staleness(1.0, 0.0) is FreshnessStatus.STALE


class TestFreshness:
    def test_fresh_factory(self):
        f = Freshness.fresh("binance", ttl_seconds=30.0)
        assert f.status() is FreshnessStatus.FRESH
        assert f.is_fresh() and f.is_actionable()
        assert f.source == "binance"

    def test_from_timestamp_computes_age(self):
        f = Freshness.from_timestamp(produced_at=100.0, source="x", ttl_seconds=50.0, now=130.0)
        assert f.age_seconds == 30.0
        assert f.is_fresh()

    def test_from_timestamp_none_is_stale(self):
        f = Freshness.from_timestamp(produced_at=None, source="x")
        assert f.stale is True
        assert f.status() is FreshnessStatus.STALE

    def test_from_timestamp_clock_skew_clamped(self):
        # Data "produced in the future" should clamp to age 0, not lie.
        f = Freshness.from_timestamp(produced_at=200.0, source="x", ttl_seconds=10.0, now=100.0)
        assert f.age_seconds == 0.0
        assert f.is_fresh()

    def test_explicit_stale_overrides_young_age(self):
        f = Freshness(age_seconds=1.0, source="x", stale=True, ttl_seconds=60.0)
        assert f.status() is FreshnessStatus.STALE
        assert not f.is_actionable()

    def test_unknown_not_actionable(self):
        f = Freshness.unknown("y")
        assert f.status() is FreshnessStatus.STALE  # stale=True short-circuits
        # sanity: construct an UNKNOWN-age, non-stale one
        f2 = Freshness(age_seconds=None, source="y", stale=False, ttl_seconds=60.0)
        assert f2.status() is FreshnessStatus.UNKNOWN
        assert not f2.is_actionable()

    def test_to_dict_roundtrip(self):
        f = Freshness.fresh("binance", 30.0)
        d = f.to_dict()
        assert d["source"] == "binance"
        assert d["status"] == "FRESH"
        assert d["age_seconds"] == 0.0


# ══════════════════════════════════════════════════════════════════
# cache
# ══════════════════════════════════════════════════════════════════


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.new_event_loop().run_until_complete(coro)


class TestTTLCache:
    def test_miss_on_unset_key(self):
        async def go():
            c: TTLCache[str, int] = TTLCache(default_ttl=10.0)
            e = await c.get("x")
            assert e.state is CacheState.MISS
            assert e.value is None
            assert c.misses == 1

        _run(go())

    def test_hit_after_set(self):
        clock = [1000.0]

        async def go():
            c: TTLCache[str, int] = TTLCache(default_ttl=10.0, clock=lambda: clock[0])
            await c.set("x", 42)
            e = await c.get("x")
            assert e.state is CacheState.HIT
            assert e.value == 42
            assert e.age_seconds == 0.0

        _run(go())

    def test_expiry(self):
        clock = [1000.0]

        async def go():
            c: TTLCache[str, int] = TTLCache(default_ttl=10.0, clock=lambda: clock[0])
            await c.set("x", 42)
            clock[0] += 11.0
            e = await c.get("x")
            assert e.state is CacheState.MISS

        _run(go())

    def test_empty_distinguished_from_miss(self):
        async def go():
            c: TTLCache[str, dict] = TTLCache(default_ttl=10.0)
            await c.set("x", {})  # legitimately empty
            e = await c.get("x")
            assert e.state is CacheState.EMPTY
            assert e.value == {}
            # Sentinel identity check
            assert e.state is EMPTY
            assert e.state is not MISS

        _run(go())

    def test_force_empty_flag(self):
        async def go():
            c: TTLCache[str, list] = TTLCache(default_ttl=10.0)
            await c.set("x", [1, 2, 3], force_empty=True)
            e = await c.get("x")
            assert e.state is CacheState.EMPTY
            assert e.value == [1, 2, 3]  # the value is still there

        _run(go())

    def test_set_empty_helper(self):
        async def go():
            c: TTLCache[str, None] = TTLCache(default_ttl=10.0)
            await c.set_empty("x")
            e = await c.get("x")
            assert e.state is CacheState.EMPTY

        _run(go())

    def test_invalidate(self):
        async def go():
            c: TTLCache[str, int] = TTLCache(default_ttl=10.0)
            await c.set("x", 1)
            assert await c.invalidate("x") is True
            assert await c.invalidate("x") is False
            assert (await c.get("x")).state is CacheState.MISS

        _run(go())

    def test_clear(self):
        async def go():
            c: TTLCache[str, int] = TTLCache(default_ttl=10.0)
            await c.set("a", 1)
            await c.set("b", 2)
            await c.clear()
            assert len(c) == 0

        _run(go())

    def test_max_size_eviction(self):
        clock = [1000.0]

        async def go():
            c: TTLCache[str, int] = TTLCache(default_ttl=100.0, max_size=2, clock=lambda: clock[0])
            await c.set("a", 1)
            clock[0] += 1
            await c.set("b", 2)
            clock[0] += 1
            await c.set("c", 3)  # should evict "a"
            assert len(c) == 2
            assert (await c.get("a")).state is CacheState.MISS
            assert (await c.get("b")).state is CacheState.HIT
            assert (await c.get("c")).state is CacheState.HIT
            assert c.evictions >= 1

        _run(go())

    def test_single_flight_get_or_fetch(self):
        call_count = [0]

        async def fetch():
            call_count[0] += 1
            await asyncio.sleep(0.01)
            return 99

        async def go():
            c: TTLCache[str, int] = TTLCache(default_ttl=10.0)
            results = await asyncio.gather(
                c.get_or_fetch("x", fetch),
                c.get_or_fetch("x", fetch),
                c.get_or_fetch("x", fetch),
            )
            assert all(r.state in (CacheState.HIT, CacheState.EMPTY) for r in results)
            assert all(r.value == 99 for r in results)
            # Only one real fetch must have occurred.
            assert call_count[0] == 1

        _run(go())

    def test_get_or_fetch_allow_empty_false(self):
        async def fetch():
            return []  # empty

        async def go():
            c: TTLCache[str, list] = TTLCache(default_ttl=10.0)
            r1 = await c.get_or_fetch("x", fetch, allow_empty=False)
            assert r1.state is CacheState.MISS  # not cached
            # Nothing was persisted — next call re-fetches
            r2 = await c.get("x")
            assert r2.state is CacheState.MISS

        _run(go())

    def test_contains(self):
        clock = [1000.0]

        async def go():
            c: TTLCache[str, int] = TTLCache(default_ttl=5.0, clock=lambda: clock[0])
            await c.set("x", 1)
            assert "x" in c
            clock[0] += 10
            assert "x" not in c

        _run(go())

    def test_rejects_bad_ttl(self):
        with pytest.raises(ValueError):
            TTLCache(default_ttl=0)
        with pytest.raises(ValueError):
            TTLCache(default_ttl=10, max_size=0)


# ══════════════════════════════════════════════════════════════════
# stats
# ══════════════════════════════════════════════════════════════════


class TestPrimitives:
    def test_is_finite(self):
        assert is_finite(1.0)
        assert is_finite(0)
        assert not is_finite(float("nan"))
        assert not is_finite(float("inf"))
        assert not is_finite(-float("inf"))
        assert not is_finite(None)
        assert not is_finite("abc")

    def test_safe_div(self):
        assert safe_div(10, 2) == 5.0
        assert safe_div(1, 0) == 0.0
        assert safe_div(1, 0, default=-1.0) == -1.0
        # Near-zero denom guard
        assert safe_div(1, 1e-15) == 0.0
        # NaN propagation
        assert safe_div(float("nan"), 1) == 0.0

    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(11, 0, 10) == 10
        # Reversed bounds normalised
        assert clamp(5, 10, 0) == 5
        # NaN goes to lo
        assert clamp(float("nan"), 1, 2) == 1

    def test_nan_to_num(self):
        assert nan_to_num(1.5) == 1.5
        assert nan_to_num(float("nan")) == 0.0
        assert nan_to_num(None, default=-1.0) == -1.0


class TestAggregates:
    def test_mean_skips_nans(self):
        assert mean([1, 2, 3, float("nan"), None]) == 2.0

    def test_mean_empty_returns_default(self):
        assert mean([], default=-99.0) == -99.0

    def test_stdev_requires_two_samples(self):
        assert stdev([5.0]) == 0.0
        assert stdev([], default=-1.0) == -1.0

    def test_stdev_matches_known_value(self):
        # sample stdev of [2, 4, 4, 4, 5, 5, 7, 9] is 2.0 (well-known)
        assert abs(stdev([2, 4, 4, 4, 5, 5, 7, 9], ddof=1) - 2.138089935) < 1e-6
        assert abs(stdev([2, 4, 4, 4, 5, 5, 7, 9], ddof=0) - 2.0) < 1e-6

    def test_mad_handles_outlier(self):
        # A single outlier barely moves MAD, unlike stdev.
        baseline = [1, 2, 3, 4, 5]
        with_outlier = baseline + [1000]
        # scale=True default ⇒ MAD*1.4826; values are bounded
        assert mad(with_outlier) < stdev(with_outlier)

    def test_mad_empty(self):
        assert mad([]) == 0.0


class TestZScore:
    def test_basic(self):
        # Population mean 5, stdev ≈ 2.138
        pop = [2, 4, 4, 4, 5, 5, 7, 9]
        z = zscore(5, pop)
        assert abs(z) < 0.1  # x == mean-ish

    def test_empty_population(self):
        assert zscore(1.0, []) == 0.0

    def test_single_sample(self):
        assert zscore(1.0, [5.0]) == 0.0

    def test_mad_fallback_when_sigma_zero(self):
        # All values equal → σ=0, MAD=0. Should fall through to default.
        assert zscore(5.0, [5.0, 5.0, 5.0, 5.0]) == 0.0

    def test_clamped_to_bounds(self):
        # Extreme outlier — z must be clamped to ±10.
        pop = [0.0] * 10 + [0.001]  # tiny σ
        z = zscore(1e6, pop)
        assert -10.0 <= z <= 10.0

    def test_log_zscore_handles_power_law(self):
        # Volume-like distribution — the raw z of 10_000_000 vs small numbers
        # is enormous; log_z is bounded.
        pop = [100, 200, 150, 175, 225, 300, 250, 400, 350]
        lz = log_zscore(10_000_000, pop)
        assert -10.0 <= lz <= 10.0
        # Negative side
        lz_small = log_zscore(1, pop)
        assert -10.0 <= lz_small <= 10.0

    def test_log_zscore_floors_zero_and_negative(self):
        assert log_zscore(0, [1, 2, 3]) != float("-inf")
        assert log_zscore(-1, [1, 2, 3]) != float("-inf")


class TestPercentiles:
    def test_blom_empty(self):
        assert blom_percentile(1.0, []) is None
        assert blom_percentile(1.0, [], default=0.5) == 0.5

    def test_blom_monotonic(self):
        pop = list(range(100))
        p_lo = blom_percentile(5, pop)
        p_mid = blom_percentile(50, pop)
        p_hi = blom_percentile(95, pop)
        assert p_lo is not None and p_mid is not None and p_hi is not None
        assert p_lo < p_mid < p_hi
        # Bounded
        assert 0.0 <= p_lo <= 1.0
        assert 0.0 <= p_hi <= 1.0

    def test_blom_single_element(self):
        # With n=1, Blom formula yields (1.0 - 3/8) / (1 + 1/4) = 0.5
        assert blom_percentile(1.0, [1.0]) == 0.5

    def test_blom_nan_input(self):
        assert blom_percentile(float("nan"), [1, 2, 3]) is None

    def test_percentile_rank_ordinary(self):
        pop = list(range(10))     # 0..9
        assert percentile_rank(0, pop) == pytest.approx(0.1)
        assert percentile_rank(9, pop) == pytest.approx(1.0)
        assert percentile_rank(-1, pop) == pytest.approx(0.0)

    def test_percentile_rank_empty(self):
        assert percentile_rank(1.0, [], default=0.25) == 0.25
