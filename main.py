"""
TitanBot Pro — Entry Point
============================
Version: v3.0.0 (full audit overhaul)
Run with:  python main.py

Everything starts here. Single command, single process.
Press Ctrl+C to stop cleanly.
"""

import asyncio
import logging
import os
import sys
import signal
try:
    import fcntl
except ImportError:
    fcntl = None  # Windows doesn't have fcntl
from pathlib import Path
import time

# Force UTC timezone for all time operations
os.environ['TZ'] = 'UTC'
try:
    time.tzset()
except AttributeError:
    pass  # Windows doesn't have tzset

# Bootstrap logger for pre-setup messages (_acquire_pid_lock, caffeinate, etc.)
# setup_logging() replaces this with the full rotating-file configuration later.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_startup_logger = logging.getLogger('startup')


def setup_logging():
    """Configure logging — console + rotating main file + always-on errors file.

    Three outputs:
      1. Console          — INFO (or DEBUG when debug_mode: true)
      2. logs/titanbot.log — INFO (or DEBUG) rotating, main run log
      3. logs/errors.log   — WARNING/ERROR/CRITICAL ALWAYS, all modules,
                             with full stack traces.  This is the file to
                             commit to the repo for bug investigation.
      4. logs/debug.log    — Full DEBUG detail, written ONLY when
                             debug_mode: true is set in settings.yaml.
    """
    from logging.handlers import RotatingFileHandler
    from config.loader import cfg

    # debug_mode: true overrides log_level to DEBUG
    debug_mode = cfg.system.get('debug_mode', False)
    if debug_mode:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, cfg.system.get('log_level', 'INFO').upper(), logging.INFO)

    log_file    = cfg.system.get('log_file',       'logs/titanbot.log')
    error_file  = cfg.system.get('error_log_file', 'logs/errors.log')
    debug_file  = cfg.system.get('debug_log_file', 'logs/debug.log')
    trade_file  = cfg.system.get('trade_log_file', 'logs/trades.log')
    max_bytes   = cfg.system.get('log_max_bytes',   10_485_760)   # 10 MB
    backup_count = cfg.system.get('log_backup_count', 5)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # ── Formatters ────────────────────────────────────────────
    # Standard formatter — used on console and main log
    std_fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Detailed formatter — used on errors.log and debug.log; includes
    # source file + line number so every entry is traceable.
    detail_fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-35s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ── Handler 1: Console ────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(std_fmt)
    console.setLevel(log_level)

    # ── Handler 2: Main rotating file (titanbot.log) ──────────
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(std_fmt)
    file_handler.setLevel(log_level)

    # ── Handler 3: Errors-only file (errors.log) ─────────────
    # Always active regardless of log_level.  Captures WARNING, ERROR,
    # and CRITICAL from EVERY module — including noisy third-party libs
    # when they actually error.  Full stack traces are written via the
    # root logger's exc_info propagation.
    error_handler = RotatingFileHandler(
        error_file, maxBytes=5_242_880, backupCount=3   # 5 MB, 3 backups
    )
    error_handler.setFormatter(detail_fmt)
    error_handler.setLevel(logging.WARNING)

    # ── Root logger ───────────────────────────────────────────
    root = logging.getLogger()
    # Remove stale handlers left by basicConfig() so messages aren't duplicated
    # and DEBUG messages from third-party libs don't leak through the old handler.
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)   # root at DEBUG so all handlers can filter
    root.addHandler(console)
    root.addHandler(file_handler)
    root.addHandler(error_handler)

    # ── Handler 4: Full debug file (debug.log) ────────────────
    # Written ONLY when debug_mode: true — exhaustive trace of every
    # decision the bot makes.  Great for sharing with a developer.
    if debug_mode:
        debug_handler = RotatingFileHandler(
            debug_file, maxBytes=max_bytes, backupCount=backup_count
        )
        debug_handler.setFormatter(detail_fmt)
        debug_handler.setLevel(logging.DEBUG)
        root.addHandler(debug_handler)

    # ── Handler 5: Trade audit file (trades.log) ──────────────
    # Always active.  Receives only records from the "trades" logger
    # (utils/trade_logger.py) — one line per signal/trigger/trail/outcome.
    # Small file (signal-count-driven, not time-driven) — commit after a
    # debug run to trace logic errors: wrong TPs, weak trigger arming, etc.
    trade_handler = RotatingFileHandler(
        trade_file, maxBytes=5_242_880, backupCount=3   # 5 MB, 3 backups
    )
    trade_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M'
    ))
    trade_handler.setLevel(logging.INFO)
    trade_logger_inst = logging.getLogger("trades")
    trade_logger_inst.setLevel(logging.INFO)
    trade_logger_inst.addHandler(trade_handler)
    trade_logger_inst.propagate = False   # don't echo to main log / console

    # ── Handler 6: Governance audit file (governance.log) ─────
    # One line per strategy weight/EWMA change and disabled/re-enabled event.
    # Upload the whole logs/ folder instead of needing titanbot.db to diagnose
    # governance issues.
    governance_file = cfg.system.get('governance_log_file', 'logs/governance.log')
    governance_handler = RotatingFileHandler(
        governance_file, maxBytes=5_242_880, backupCount=3
    )
    governance_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    governance_handler.setLevel(logging.INFO)
    governance_logger_inst = logging.getLogger("governance")
    governance_logger_inst.setLevel(logging.INFO)
    governance_logger_inst.addHandler(governance_handler)
    governance_logger_inst.propagate = False

    # ── Handler 7: Risk audit file (risk.log) ─────────────────
    # One line per circuit-breaker trip, resume, daily reset, and loss record.
    risk_file = cfg.system.get('risk_log_file', 'logs/risk.log')
    risk_handler = RotatingFileHandler(
        risk_file, maxBytes=5_242_880, backupCount=3
    )
    risk_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    risk_handler.setLevel(logging.INFO)
    risk_logger_inst = logging.getLogger("risk")
    risk_logger_inst.setLevel(logging.INFO)
    risk_logger_inst.addHandler(risk_handler)
    risk_logger_inst.propagate = False

    # ── Handler 8: Params audit file (params.log) ─────────────
    # One line per adaptive-parameter adjustment (old → new value + context).
    params_file = cfg.system.get('params_log_file', 'logs/params.log')
    params_handler = RotatingFileHandler(
        params_file, maxBytes=5_242_880, backupCount=3
    )
    params_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    params_handler.setLevel(logging.INFO)
    params_logger_inst = logging.getLogger("params")
    params_logger_inst.setLevel(logging.INFO)
    params_logger_inst.addHandler(params_handler)
    params_logger_inst.propagate = False

    # ── Suppress noisy third-party libraries ──────────────────
    # These are suppressed on the CONSOLE and main log so they don't
    # flood normal output.  WARNING+ from them still reaches errors.log.
    noisy_libs = ['httpx', 'httpcore', 'telegram', 'telegram.ext', 'urllib3', 'asyncio', 'ccxt']
    for noisy_lib in noisy_libs:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    # ── Handler 9: FCN modules audit file (fcn_modules.log) ──
    # Dedicated log for ALL Project Inheritance (FCN) features.
    # One-stop file to verify the 11 new modules are working:
    # source credibility, clickbait filter, title dedup, headline evolution,
    # narrative tracker, fear & greed, time decay, on-chain↔news correlation,
    # whale intent, pump/dump detection, degradation engine.
    from config.fcn_logger import fcn_log, FCN_LOG_FILE
    fcn_log("STARTUP", "FCN modules logger initialised — all feature activity below")
    # Log current feature-flag states so the file is self-contained
    from config.feature_flags import ff
    for _flag, _state in sorted(ff.all_states().items()):
        fcn_log("FLAG_STATE", f"{_flag}: {_state}")

    logger = logging.getLogger('main')
    logger.info("📋 Logging active → %s (level: %s)", log_file,
                logging.getLevelName(log_level))
    logger.info("🚨 Error capture → %s (WARNING+ always)", error_file)
    logger.info("📒 Trade audit  → %s (every signal/outcome)", trade_file)
    logger.info("📊 Governance   → %s (strategy weights/disables)", governance_file)
    logger.info("🛡️  Risk         → %s (circuit breaker events)", risk_file)
    logger.info("⚙️  Params       → %s (adaptive param changes)", params_file)
    logger.info("🧩 FCN modules  → %s (Project Inheritance features)", FCN_LOG_FILE)
    if debug_mode:
        logger.info("🔧 DEBUG MODE — full trace → %s", debug_file)
    return logger


async def main():
    """Main async entry point"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("  TitanBot Pro v3.0.0")
    logger.info("  HyperTracker · CoinGecko Trending · RSS News edition")
    logger.info("  MacBook Pro 2013 / 8GB RAM / SSD Edition")
    logger.info("=" * 60)

    # ── Validate config ────────────────────────────────────────
    from config.loader import cfg
    logger.info("Validating configuration...")
    if not cfg.validate():
        logger.error("❌ Config validation failed. Check your .env file and settings.yaml")
        sys.exit(1)

    logger.info("✅ Config valid")

    # ── Import engine ──────────────────────────────────────────
    from core.engine import engine

    # ── Signal handlers for clean shutdown ─────────────────────
    # FIX: get_event_loop() deprecated in 3.10+. Inside async def main() we are
    # guaranteed to have a running loop, so get_running_loop() is correct here.
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("\nShutdown signal received...")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

    # ── Start engine ───────────────────────────────────────────
    # Layer 2: done-callback fires the instant the engine task completes
    # with an exception — no GC dependency — so it always reaches errors.log.
    def _engine_crash_handler(task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.critical(
                f"💀 ENGINE TASK DIED: {type(exc).__name__}: {exc}",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            # Trigger shutdown so main() doesn't hang forever
            shutdown_event.set()

    engine_task = asyncio.create_task(engine.start(), name="engine.start")
    engine_task.add_done_callback(_engine_crash_handler)

    try:
        # Layer 1: Race the engine task against the shutdown signal.
        # If engine.start() crashes, we detect it immediately instead
        # of blocking forever on shutdown_event.wait().
        shutdown_task = asyncio.create_task(
            shutdown_event.wait(), name="shutdown_wait"
        )
        done, pending = await asyncio.wait(
            [engine_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If engine_task finished first, it crashed — retrieve the exception
        # so it propagates through the normal logging pipeline.
        if engine_task in done and not engine_task.cancelled():
            exc = engine_task.exception()
            if exc:
                logger.critical(
                    f"🚨 Engine crashed during execution: {type(exc).__name__}: {exc}",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        # Clean up the pending task
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Initiating clean shutdown...")
        if not engine_task.done():
            engine_task.cancel()
            try:
                await engine_task
            except asyncio.CancelledError:
                pass
        await engine.stop()
        logger.info("✅ TitanBot Pro stopped cleanly.")



def _acquire_pid_lock():
    """
    Prevent two bot instances from running simultaneously.
    Two instances share the same Telegram channel but have separate in-memory dedup
    dicts — each sees every signal as new and both publish it, causing duplicates.

    FIX: On Windows the lock file is now always deleted on exit via atexit so a
    crashed/clean-exited process never blocks the next start permanently.
    """
    import atexit
    lock_path = Path("titanbot.lock")

    if fcntl is None:
        # Windows: simple PID file check (no advisory locking)
        if lock_path.exists():
            try:
                old_pid = int(lock_path.read_text().strip())
                import psutil
                if psutil.pid_exists(old_pid):
                    _startup_logger.error("❌ TitanBot is already running!")
                    _startup_logger.error("   Only one instance allowed. Use Ctrl+C to stop the existing instance.")
                    sys.exit(1)
            except (ValueError, ImportError):
                pass
        lock_path.write_text(str(os.getpid()))
        # FIX: register cleanup so the file is always removed on exit
        atexit.register(lambda: lock_path.unlink(missing_ok=True))
        return lock_path

    try:
        lock_file = open(lock_path, 'w')
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        # FIX: also register cleanup for Unix path — file is released by OS on
        # process exit but deleting it prevents confusing stale-file warnings.
        # Explicitly close the fd first so the flock is released before the path
        # is removed, preventing a brief window where the lock file exists but
        # the advisory lock is gone.
        def _unix_cleanup():
            try:
                lock_file.close()
            except Exception:
                pass
            lock_path.unlink(missing_ok=True)
        atexit.register(_unix_cleanup)
        return lock_file  # Keep reference alive to hold the lock
    except IOError:
        # Lock is held by another process.
        # Read PID from lock file to give a helpful message.
        try:
            old_pid = int(lock_path.read_text().strip())
            _startup_logger.error("❌ TitanBot is already running! (PID %d)", old_pid)
            _startup_logger.error("   Stop it first:  kill %d  or press Ctrl+C in its window.", old_pid)
        except Exception:
            _startup_logger.error("❌ TitanBot is already running!")
        _startup_logger.error("   Only one instance allowed — two instances = duplicate Telegram signals.")
        _startup_logger.error("   If you're sure no instance is running, delete titanbot.lock and retry.")
        sys.exit(1)

if __name__ == '__main__':
    _pid_lock = _acquire_pid_lock()
    # macOS stability: explicitly set event loop policy to avoid
    # kqueue issues with heavy async IO (Binance + Telegram polling)
    import platform
    if platform.system() == 'Darwin':
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    else:
        # Linux: try uvloop for performance, fall back to default
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            pass

    # D2-FIX: Honour prevent_sleep setting on macOS.
    # Without this, a MacBook screen-saver / sleep prevents all async IO from
    # running, silently dropping Telegram messages and exchange API calls.
    # caffeinate -i keeps the system awake as long as this process is alive.
    _caffeinate_proc = None
    try:
        from config.loader import cfg as _cfg_main
        if platform.system() == 'Darwin' and _cfg_main.system.get('prevent_sleep', False):
            import subprocess as _subprocess
            _caffeinate_proc = _subprocess.Popen(
                ['caffeinate', '-i', '-w', str(os.getpid())],
                stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL
            )
            _startup_logger.info("☕ Sleep prevention active (caffeinate running)")
    except Exception as _e:
        _startup_logger.warning("⚠️  prevent_sleep: could not start caffeinate (%s)", _e)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        if _caffeinate_proc is not None:
            try:
                _caffeinate_proc.terminate()
            except Exception:
                pass
