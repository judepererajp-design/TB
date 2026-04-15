"""
TitanBot Pro — System Health Monitor
======================================
Catches the class of bugs that the Sentinel AI cannot see:
  - Signal data corruption (TP > 10x entry, SL on wrong side, etc.)
  - Silent asyncio exceptions (tasks dying with no log output)
  - Memory / task accumulation (price cache growing unboundedly, etc.)
  - OHLCV data quality (stale candles, zero prices, corrupt OHLCV rows)
  - Database schema drift (missing columns after update)
  - API rate limit accumulation (before it affects signal quality)

All violations are written to the standard logger as WARNING or CRITICAL
so the Sentinel AI can pick them up alongside signal data.

Architecture:
  HealthMonitor.validate_signal(signal, current_price)
      → called from engine before every publish
      → returns (ok: bool, reasons: list[str])

  HealthMonitor.validate_ohlcv(symbol, tf, candles)
      → called from engine after every fetch
      → returns (ok: bool, reasons: list[str])

  HealthMonitor._watchdog_loop()
      → runs every 5 minutes
      → logs task count, memory, cache sizes, rate limits

  HealthMonitor.record_rate_limit(provider)
      → called whenever a 429 is seen
      → logs warning when threshold crossed

  setup_asyncio_exception_handler()
      → called once at startup
      → logs every unhandled asyncio task exception
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────

# Signal sanity thresholds
MAX_TP_RATIO       = 5.0    # TP must not be more than 5x the entry price
MAX_SL_RATIO       = 3.0    # SL must not be more than 3x away from entry
MAX_RR_RATIO       = 25.0   # R/R above 25 is almost certainly wrong
MIN_RR_RATIO       = 0.3    # R/R below 0.3 is not worth trading
MAX_ENTRY_WIDTH    = 0.08   # Entry zone width must be < 8% of price
MIN_ENTRY_WIDTH    = 0.0001 # Entry zone width must be > 0.01% (no collapsed zones)

# OHLCV sanity thresholds
MAX_CANDLE_AGE_H   = 4.5    # FIX 1B: was 3.0h — Binance 4h candles naturally arrive
                              # at 3.0–3.1h after open, triggering 2,651 false-positive
                              # OHLCV_QUALITY_FAIL warnings per session. 4.5h is safe:
                              # a genuine 4h stale candle would be at 5h+ (missed close).
MIN_CANDLES        = 50     # Must have at least 50 candles

# Watchdog thresholds
MAX_ASYNCIO_TASKS  = 500    # Warn if asyncio tasks exceed this
MAX_PRICE_CACHE    = 1000   # Warn if price_cache subscribers exceed this
MAX_OUTCOME_ACTIVE = 200    # Warn if outcome_monitor active signals exceed this
WATCHDOG_INTERVAL  = 300    # Check every 5 minutes

# Rate limit thresholds (per provider per hour)
RATE_LIMIT_WARN    = 5      # Warn after 5 rate limits per hour
RATE_LIMIT_CRIT    = 15     # Critical after 15 per hour


class HealthMonitor:
    """
    Continuous self-monitoring for TitanBot Pro.
    Detects code-level and data-level problems the Sentinel AI cannot see.
    """

    def __init__(self):
        self._rate_limits: Dict[str, List[float]] = defaultdict(list)
        self._validation_failures = 0
        self._ohlcv_failures = 0
        self._watchdog_task: Optional[asyncio.Task] = None
        self._start_time = time.time()
        self._schema_checked = False

    # ── Signal validation ──────────────────────────────────────

    def validate_signal(
        self,
        symbol: str,
        direction: str,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: Optional[float],
        rr_ratio: float,
        strategy: str,
        current_price: float = 0.0,
    ) -> Tuple[bool, List[str]]:
        """
        Validate all price levels before a signal is published.
        Returns (ok, list_of_problems).
        If ok=False the engine should block the signal and log the problems.
        """
        problems = []
        entry_mid = (entry_low + entry_high) / 2
        if entry_mid <= 0:
            return False, ["entry_mid is zero or negative"]

        is_long = direction == "LONG"

        # ── 1. TP sanity: TP must not be more than MAX_TP_RATIO × entry ──
        for label, tp_val in [("TP1", tp1), ("TP2", tp2), ("TP3", tp3)]:
            if not tp_val or tp_val <= 0:
                continue
            dist = abs(tp_val - entry_mid)
            ratio = dist / entry_mid
            if ratio > MAX_TP_RATIO:
                problems.append(
                    f"{label}={tp_val:.4f} is {ratio:.0f}x entry ({entry_mid:.4f}) "
                    f"— likely BTC range corruption"
                )

        # ── 2. SL sanity: SL must not be more than MAX_SL_RATIO × risk away ──
        if stop_loss > 0:
            sl_dist = abs(stop_loss - entry_mid)
            sl_ratio = sl_dist / entry_mid
            if sl_ratio > MAX_SL_RATIO:
                problems.append(
                    f"SL={stop_loss:.4f} is {sl_ratio:.0f}x from entry ({entry_mid:.4f}) "
                    f"— likely range corruption"
                )

            # ── 3. SL must be on the correct side of entry ──
            if is_long and stop_loss > entry_low:
                problems.append(
                    f"LONG SL={stop_loss:.6f} is ABOVE entry_low={entry_low:.6f} "
                    f"— SL on wrong side"
                )
            elif not is_long and stop_loss < entry_high:
                problems.append(
                    f"SHORT SL={stop_loss:.6f} is BELOW entry_high={entry_high:.6f} "
                    f"— SL on wrong side"
                )

        # ── 4. TP must be on the correct side of entry ──
        if tp1 > 0:
            if is_long and tp1 <= entry_high:
                problems.append(
                    f"LONG TP1={tp1:.6f} is at or below entry_high={entry_high:.6f}"
                )
            elif not is_long and tp1 >= entry_low:
                problems.append(
                    f"SHORT TP1={tp1:.6f} is at or above entry_low={entry_low:.6f}"
                )

        # ── 5. R/R bounds ──
        if rr_ratio > MAX_RR_RATIO:
            problems.append(f"R/R={rr_ratio:.1f} exceeds maximum {MAX_RR_RATIO} — likely corrupted")
        elif rr_ratio < MIN_RR_RATIO and rr_ratio > 0:
            problems.append(f"R/R={rr_ratio:.2f} below minimum {MIN_RR_RATIO}")

        # ── 6. Entry zone width ──
        if entry_high > entry_low:
            zone_width_pct = (entry_high - entry_low) / entry_mid
            if zone_width_pct > MAX_ENTRY_WIDTH:
                problems.append(
                    f"Entry zone width {zone_width_pct*100:.1f}% > {MAX_ENTRY_WIDTH*100:.0f}% "
                    f"({entry_low:.6f}–{entry_high:.6f})"
                )
            elif zone_width_pct < MIN_ENTRY_WIDTH:
                problems.append(
                    f"Entry zone collapsed: width={zone_width_pct*100:.4f}% "
                    f"({entry_low:.6f}–{entry_high:.6f})"
                )
        elif entry_low >= entry_high:
            problems.append(
                f"Entry zone inverted: low={entry_low:.6f} >= high={entry_high:.6f}"
            )

        # ── 7. Current price sanity (if provided) ──
        if current_price > 0 and entry_mid > 0:
            price_ratio = abs(current_price - entry_mid) / entry_mid
            if price_ratio > 0.30:
                problems.append(
                    f"Entry zone {entry_mid:.6f} is {price_ratio*100:.0f}% from current price "
                    f"{current_price:.6f} — stale signal?"
                )

        # ── Log all problems ──
        if problems:
            self._validation_failures += 1
            for p in problems:
                logger.warning(
                    f"🚨 SIGNAL VALIDATION FAIL | {symbol} {direction} [{strategy}] | {p}"
                )
            # Report to diagnostic engine
            try:
                from core.diagnostic_engine import diagnostic_engine
                for p in problems:
                    diagnostic_engine.record_error(
                        module="health_monitor",
                        error_type="SIGNAL_VALIDATION",
                        message=p,
                        symbol=symbol,
                    )
            except Exception:
                pass

        return len(problems) == 0, problems

    # ── OHLCV validation ───────────────────────────────────────

    def validate_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        candles: list,
    ) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data quality before passing to strategies.
        Returns (ok, problems).
        """
        problems = []

        if not candles:
            return False, [f"Empty OHLCV for {symbol}/{timeframe}"]

        if len(candles) < MIN_CANDLES:
            problems.append(
                f"{symbol}/{timeframe}: only {len(candles)} candles (min {MIN_CANDLES})"
            )

        # Check last candle age
        try:
            last_ts = candles[-1][0]
            if last_ts > 1e12:  # milliseconds
                last_ts /= 1000
            age_h = (time.time() - last_ts) / 3600
            if age_h > MAX_CANDLE_AGE_H:
                problems.append(
                    f"{symbol}/{timeframe}: last candle is {age_h:.1f}h old "
                    f"(max {MAX_CANDLE_AGE_H}h) — stale data"
                )
        except Exception:
            problems.append(f"{symbol}/{timeframe}: cannot read last candle timestamp")

        # Check for zero/negative prices in recent candles
        try:
            for i, c in enumerate(candles[-10:]):
                o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
                if any(v <= 0 for v in [o, h, l, cl]):
                    problems.append(
                        f"{symbol}/{timeframe}: zero/negative price in candle[-{10-i}] "
                        f"o={o} h={h} l={l} c={cl}"
                    )
                    break
                if h < l:
                    problems.append(
                        f"{symbol}/{timeframe}: high={h} < low={l} in candle[-{10-i}] — corrupt"
                    )
                    break
                if h < o or h < cl or l > o or l > cl:
                    # Only flag extreme violations (5% outside range)
                    if max(abs(o-h)/h, abs(cl-h)/h, abs(l-o)/o if o>0 else 0) > 0.05:
                        problems.append(
                            f"{symbol}/{timeframe}: OHLC inconsistency in candle[-{10-i}]"
                        )
                        break
        except Exception as e:
            problems.append(f"{symbol}/{timeframe}: OHLCV parse error: {e}")

        if problems:
            self._ohlcv_failures += 1
            for p in problems:
                logger.warning(f"📊 OHLCV QUALITY FAIL | {p}")

        return len(problems) == 0, problems

    # ── Rate limit tracking ────────────────────────────────────

    def record_rate_limit(self, provider: str):
        """
        Call this whenever a 429 is received from any API.
        Automatically warns when rate limits accumulate.
        """
        now = time.time()
        # Keep only last hour and cap list at 1000 entries
        entries = self._rate_limits[provider]
        self._rate_limits[provider] = [
            t for t in entries[-1000:]
            if now - t < 3600
        ]
        self._rate_limits[provider].append(now)
        count = len(self._rate_limits[provider])

        if count == RATE_LIMIT_WARN:
            logger.warning(
                f"⚠️  RATE LIMIT | {provider}: {count} rate limits in last hour "
                f"— signal quality may be degrading"
            )
        elif count == RATE_LIMIT_CRIT:
            logger.critical(
                f"🚨 RATE LIMIT CRITICAL | {provider}: {count} rate limits in last hour "
                f"— API access severely throttled, signals unreliable"
            )
        elif count > RATE_LIMIT_CRIT and count % 5 == 0:
            logger.critical(
                f"🚨 RATE LIMIT STORM | {provider}: {count} rate limits in last hour"
            )

    # ── Watchdog ───────────────────────────────────────────────

    def start(self):
        """Start the watchdog background task."""
        try:
            loop = asyncio.get_running_loop()
            self._watchdog_task = loop.create_task(self._watchdog_loop())
            self._watchdog_task.add_done_callback(
                lambda t: t.exception() if not t.cancelled() else None
            )
            logger.info("🔍 Health monitor started (watchdog every 5min)")
        except RuntimeError:
            logger.warning("HealthMonitor.start(): no running loop — watchdog not started")

    def stop(self):
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()

    async def _watchdog_loop(self):
        """
        Runs every 5 minutes. Checks:
        - asyncio task count (leak detection)
        - memory usage (psutil if available)
        - price_cache subscriber count
        - outcome_monitor active signal count
        - rate limit summaries
        """
        await asyncio.sleep(60)  # Give bot time to fully start
        while True:
            try:
                await self._run_watchdog_checks()
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
            await asyncio.sleep(WATCHDOG_INTERVAL)

    async def _run_watchdog_checks(self):
        issues = []

        # ── Asyncio task count ──
        try:
            all_tasks = asyncio.all_tasks()
            task_count = len(all_tasks)
            if task_count > MAX_ASYNCIO_TASKS:
                issues.append(f"asyncio tasks={task_count} (max {MAX_ASYNCIO_TASKS}) — possible task leak")
                # Log names of tasks to help diagnose
                task_names = [t.get_name() for t in list(all_tasks)[:20]]
                logger.warning(f"🔍 WATCHDOG | Task leak sample: {task_names}")
            else:
                logger.debug(f"Watchdog: asyncio tasks={task_count}")
        except Exception as e:
            logger.debug(f"Watchdog task count error: {e}")

        # ── Memory usage ──
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / 1024 / 1024
            if mem_mb > 1500:
                issues.append(f"memory={mem_mb:.0f}MB — approaching limit for 8GB machine")
            elif mem_mb > 800:
                logger.info(f"🔍 WATCHDOG | Memory usage: {mem_mb:.0f}MB")
            else:
                logger.debug(f"Watchdog: memory={mem_mb:.0f}MB")
        except ImportError:
            pass  # psutil not installed — skip
        except Exception as e:
            logger.debug(f"Watchdog memory check error: {e}")

        # ── Price cache subscriber count ──
        try:
            from core.price_cache import price_cache
            sub_count = len(getattr(price_cache, '_subscribers', {}))
            if sub_count > MAX_PRICE_CACHE:
                issues.append(
                    f"price_cache subscribers={sub_count} (max {MAX_PRICE_CACHE}) "
                    f"— EXPIRED signals not unsubscribing"
                )
            else:
                logger.debug(f"Watchdog: price_cache subscribers={sub_count}")
        except Exception as e:
            logger.debug(f"Watchdog price_cache error: {e}")

        # ── Outcome monitor active signals ──
        try:
            from signals.outcome_monitor import outcome_monitor
            active_count = len(getattr(outcome_monitor, '_active', {}))
            if active_count > MAX_OUTCOME_ACTIVE:
                issues.append(
                    f"outcome_monitor active={active_count} (max {MAX_OUTCOME_ACTIVE}) "
                    f"— signals not completing (EXPIRED not firing on_outcome?)"
                )
            else:
                logger.debug(f"Watchdog: outcome_monitor active={active_count}")
        except Exception as e:
            logger.debug(f"Watchdog outcome_monitor error: {e}")

        # ── Rate limit summary ──
        try:
            now = time.time()
            for provider, times in self._rate_limits.items():
                recent = [t for t in times if now - t < 3600]
                if recent:
                    logger.debug(f"Watchdog: rate limits last 1h — {provider}={len(recent)}")
        except Exception:
            pass

        # ── Validation failure rate ──
        uptime_h = (time.time() - self._start_time) / 3600
        if uptime_h > 0.5 and self._validation_failures > 0:
            fail_rate = self._validation_failures / max(1, uptime_h)
            if fail_rate > 5:
                issues.append(
                    f"signal_validation_failures={self._validation_failures} "
                    f"({fail_rate:.1f}/h) — frequent TP/SL corruption"
                )

        # ── Database schema check (once per run) ──
        if not self._schema_checked:
            await self._check_db_schema()
            self._schema_checked = True

        # ── Disk space check ──
        try:
            import shutil
            from config.loader import cfg as _cfg
            db_path = _cfg.database.path if hasattr(_cfg, 'database') else 'titanbot.db'
            import os as _os
            _check_dir = _os.path.dirname(_os.path.abspath(db_path)) or '.'
            _disk = shutil.disk_usage(_check_dir)
            _free_gb = _disk.free / (1024 ** 3)
            _total_gb = _disk.total / (1024 ** 3)
            _used_pct = (_disk.used / _disk.total) * 100
            if _free_gb < 0.5:
                issues.append(
                    f"disk_space_free={_free_gb:.2f}GB ({_used_pct:.0f}% used) — "
                    f"CRITICAL: DB and WAL writes will fail soon"
                )
            elif _free_gb < 2.0:
                issues.append(
                    f"disk_space_free={_free_gb:.2f}GB ({_used_pct:.0f}% used) — "
                    f"low disk space, consider cleanup"
                )
            else:
                logger.debug(f"Watchdog: disk_free={_free_gb:.1f}GB / {_total_gb:.1f}GB")
        except Exception as e:
            logger.debug(f"Watchdog disk check error: {e}")

        # ── Log summary ──
        if issues:
            for issue in issues:
                logger.warning(f"🔍 WATCHDOG ALERT | {issue}")
                try:
                    from core.diagnostic_engine import diagnostic_engine
                    diagnostic_engine.record_error(
                        module="health_monitor.watchdog",
                        error_type="SYSTEM_HEALTH",
                        message=issue,
                    )
                except Exception:
                    pass
        else:
            uptime_h = (time.time() - self._start_time) / 3600
            logger.info(
                f"🔍 WATCHDOG OK | uptime={uptime_h:.1f}h | "
                f"validation_fails={self._validation_failures} | "
                f"ohlcv_fails={self._ohlcv_failures}"
            )

    async def _check_db_schema(self):
        """
        Verify the database has all expected columns.
        Logs a warning for any missing column so we catch schema drift
        after updates.
        """
        EXPECTED_SCHEMA = {
            "signals": [
                "id", "symbol", "direction", "strategy", "confidence",
                "entry_low", "entry_high", "stop_loss", "tp1", "tp2", "tp3",
                "rr_ratio", "alpha_grade", "outcome", "pnl_r", "created_at",
                "regime", "message_id",
            ],
            "outcomes": [
                "id", "signal_id", "outcome", "pnl_r", "closed_at",
            ],
        }
        try:
            from data.database import db
            for table, expected_cols in EXPECTED_SCHEMA.items():
                try:
                    cursor = await db._conn.execute(
                        f"PRAGMA table_info({table})"
                    )
                    rows = await cursor.fetchall()
                    await cursor.close()
                    actual_cols = {r[1] for r in rows}
                    missing = [c for c in expected_cols if c not in actual_cols]
                    if missing:
                        logger.warning(
                            f"🚨 DB SCHEMA | Table '{table}' missing columns: {missing} "
                            f"— may cause runtime errors"
                        )
                    else:
                        logger.debug(f"DB schema OK: {table} ({len(actual_cols)} columns)")
                except Exception as e:
                    logger.warning(f"DB schema check failed for {table}: {e}")
        except Exception as e:
            logger.debug(f"DB schema check skipped: {e}")


# ── Asyncio exception handler ──────────────────────────────────

def setup_asyncio_exception_handler():
    """
    Install a global asyncio exception handler so that unhandled
    exceptions in background tasks are logged instead of disappearing
    silently.

    This catches the class of bugs like:
      - Missing methods called on singleton objects
      - Import errors inside create_task() coroutines
      - Any await that raises inside a fire-and-forget task

    Without this, Python just prints to stderr and the task is gone.
    """
    def _handler(loop, context):
        exc = context.get("exception")
        task = context.get("task")
        msg = context.get("message", "Unknown asyncio error")
        task_name = task.get_name() if task else "unknown_task"

        if exc is not None:
            logger.error(
                f"🚨 UNHANDLED ASYNCIO EXCEPTION | task={task_name} | "
                f"{type(exc).__name__}: {exc}",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        else:
            logger.error(
                f"🚨 ASYNCIO ERROR | task={task_name} | {msg}"
            )

        # Also record in diagnostic engine
        try:
            from core.diagnostic_engine import diagnostic_engine
            diagnostic_engine.record_error(
                module=f"asyncio.{task_name}",
                error_type=type(exc).__name__ if exc else "ASYNCIO_ERROR",
                message=str(exc) if exc else msg,
            )
        except Exception:
            pass

    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_handler)
        logger.info("🔍 Asyncio exception handler installed — silent failures will now be logged")
    except RuntimeError:
        logger.warning("setup_asyncio_exception_handler(): no running loop")


# ── Singleton ──────────────────────────────────────────────────
health_monitor = HealthMonitor()
