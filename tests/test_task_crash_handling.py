"""
TitanBot Pro — Task-Level Crash Handling Tests
================================================
Validates the 4 layers of defense that ensure task-level exceptions
always reach errors.log instead of being silently swallowed.

Layer 1: main.py uses asyncio.wait() to detect engine crashes immediately
Layer 2: done-callback on engine task fires on crash → logs + sets shutdown
Layer 3: engine.start() wraps _start_impl() with try/except → logs CRITICAL
Layer 4: _log_task_exception callback on all fire-and-forget background tasks
"""

import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Standalone _log_task_exception replica for testing
# (avoids importing core.engine which pulls in heavy deps)
# ---------------------------------------------------------------------------
_test_logger = logging.getLogger("core.engine")


def _log_task_exception(task: asyncio.Task):
    """Replica of Engine._log_task_exception for isolated testing."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _test_logger.error(
            f"🚨 BACKGROUND TASK DIED | {task.get_name()} | "
            f"{type(exc).__name__}: {exc}",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


# ---------------------------------------------------------------------------
# Layer 3: engine.start() wraps _start_impl() with try/except
# ---------------------------------------------------------------------------

class TestEngineStartWrapper:
    """Layer 3 — verify the start()/_start_impl() wrapping pattern works."""

    async def test_start_wrapper_logs_critical_on_crash(self, caplog):
        """If the inner impl raises, the wrapper logs CRITICAL and re-raises."""
        async def _start_impl():
            raise RuntimeError("exchange init exploded")

        async def _start_wrapper():
            """Replica of Engine.start() wrapping pattern."""
            try:
                await _start_impl()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _test_logger.critical(
                    f"💀 ENGINE FATAL — {type(e).__name__}: {e}",
                    exc_info=True,
                )
                raise

        with caplog.at_level(logging.CRITICAL, logger="core.engine"):
            with pytest.raises(RuntimeError, match="exchange init exploded"):
                await _start_wrapper()

        assert any("ENGINE FATAL" in r.message for r in caplog.records)
        assert any("RuntimeError" in r.message for r in caplog.records)

    async def test_start_wrapper_propagates_cancelled_error(self):
        """CancelledError must propagate without being logged as CRITICAL."""
        async def _start_impl():
            raise asyncio.CancelledError()

        async def _start_wrapper():
            try:
                await _start_impl()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _test_logger.critical(f"ENGINE FATAL: {e}", exc_info=True)
                raise

        with pytest.raises(asyncio.CancelledError):
            await _start_wrapper()


# ---------------------------------------------------------------------------
# Layer 4: _log_task_exception callback on background tasks
# ---------------------------------------------------------------------------

class TestLogTaskException:
    """Layer 4 — _log_task_exception fires for all background tasks."""

    async def test_callback_logs_error_on_exception(self, caplog):
        """When a background task raises, the callback logs the error."""
        async def _failing_task():
            raise ValueError("bad data from API")

        task = asyncio.create_task(_failing_task(), name="test_failing_task")
        # Await the task so it fully completes before we call the callback
        try:
            await task
        except ValueError:
            pass

        with caplog.at_level(logging.ERROR, logger="core.engine"):
            _log_task_exception(task)

        assert any("BACKGROUND TASK DIED" in r.message for r in caplog.records)
        assert any("test_failing_task" in r.message for r in caplog.records)
        assert any("ValueError" in r.message for r in caplog.records)

    async def test_callback_silent_on_cancellation(self, caplog):
        """Cancelled tasks should not trigger error logging."""
        async def _slow_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(_slow_task(), name="test_cancelled_task")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        with caplog.at_level(logging.ERROR, logger="core.engine"):
            _log_task_exception(task)

        assert not any("BACKGROUND TASK DIED" in r.message for r in caplog.records)

    async def test_callback_silent_on_clean_completion(self, caplog):
        """Tasks that complete successfully should not trigger error logging."""
        async def _ok_task():
            return 42

        task = asyncio.create_task(_ok_task(), name="test_ok_task")
        await task

        with caplog.at_level(logging.ERROR, logger="core.engine"):
            _log_task_exception(task)

        assert not any("BACKGROUND TASK DIED" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Layer 1 + 2: main.py engine crash detection
# ---------------------------------------------------------------------------

class TestMainEngineCrashDetection:
    """Layers 1+2 — main.py detects engine crash via asyncio.wait + callback."""

    async def test_engine_crash_triggers_shutdown_event(self):
        """
        Simulates what main.py does: create a task with the crash callback,
        then verify that the shutdown_event gets set when the task dies.
        """
        shutdown_event = asyncio.Event()

        def _engine_crash_handler(task):
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                shutdown_event.set()

        async def _crashing_engine():
            raise AttributeError("api._exchange")

        engine_task = asyncio.create_task(
            _crashing_engine(), name="engine.start"
        )
        engine_task.add_done_callback(_engine_crash_handler)

        # Use asyncio.wait to deterministically wait for the task to complete
        # and its done-callbacks to fire, instead of fragile sleep(0) loops.
        await asyncio.wait([engine_task], timeout=1.0)

        assert shutdown_event.is_set(), \
            "shutdown_event should be set when engine task crashes"

    async def test_asyncio_wait_detects_engine_crash(self):
        """
        asyncio.wait with FIRST_COMPLETED should return immediately
        when the engine task crashes, instead of blocking forever.
        """
        shutdown_event = asyncio.Event()

        async def _crashing_engine():
            raise RuntimeError("startup boom")

        engine_task = asyncio.create_task(
            _crashing_engine(), name="engine.start"
        )
        shutdown_task = asyncio.create_task(
            shutdown_event.wait(), name="shutdown_wait"
        )

        done, pending = await asyncio.wait(
            [engine_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        assert engine_task in done, \
            "engine_task should complete first (it crashed)"
        assert shutdown_task in pending, \
            "shutdown_task should still be pending"

        # Verify the exception is retrievable
        with pytest.raises(RuntimeError, match="startup boom"):
            engine_task.result()

        # Cleanup
        shutdown_task.cancel()
        try:
            await shutdown_task
        except asyncio.CancelledError:
            pass

    async def test_normal_shutdown_works(self):
        """
        When shutdown_event is set (e.g. SIGTERM), the wait should return
        with the engine task still running.
        """
        shutdown_event = asyncio.Event()

        async def _long_running_engine():
            await asyncio.sleep(100)

        engine_task = asyncio.create_task(
            _long_running_engine(), name="engine.start"
        )
        shutdown_task = asyncio.create_task(
            shutdown_event.wait(), name="shutdown_wait"
        )

        # Simulate SIGTERM
        shutdown_event.set()

        done, pending = await asyncio.wait(
            [engine_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        assert shutdown_task in done, \
            "shutdown_task should complete first (signal received)"
        assert engine_task in pending, \
            "engine_task should still be pending (running normally)"

        # Cleanup
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass
