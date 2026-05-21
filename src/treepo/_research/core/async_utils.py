"""
Async Utilities for OPS Framework.

This module provides helper functions for async operations, particularly
around proper task cleanup to prevent orphaned tasks from piling up.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import functools
import logging
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine, Iterable, List, Optional, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

# NOTE: We intentionally avoid asyncio.to_thread() because it uses the event loop's
# default executor. In this repo's runtime environment, shutting down the default
# executor during asyncio.run() teardown can hang indefinitely. By routing all
# thread offloads through a shared global executor, we avoid the default executor
# lifecycle entirely.
_GLOBAL_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXPLICIT_MAX_WORKERS: Optional[int] = None
_LOOP_HEARTBEATS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopHeartbeatState]" = weakref.WeakKeyDictionary()
_HEARTBEAT_INTERVAL_SECONDS = 0.05


class _LoopHeartbeatState:
    __slots__ = ("pending", "task")

    def __init__(self, pending: int, task: "asyncio.Task[None]") -> None:
        self.pending = int(pending)
        self.task = task


def _ensure_loop_heartbeat(loop: asyncio.AbstractEventLoop) -> _LoopHeartbeatState:
    """Ensure the loop wakes periodically while thread futures are pending.

    In this runtime environment, callbacks from threadpool futures can fail to
    wake the selector loop reliably (i.e., lost `call_soon_threadsafe` wakeups),
    causing awaits on `run_in_executor` to hang indefinitely when the loop has
    no scheduled timers. A lightweight heartbeat timer prevents the selector
    timeout from becoming infinite.
    """
    existing = _LOOP_HEARTBEATS.get(loop)
    if existing is not None and not existing.task.done():
        return existing

    async def _heartbeat() -> None:
        while True:
            state = _LOOP_HEARTBEATS.get(loop)
            if state is None or state.pending <= 0 or loop.is_closed():
                break
            await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        _LOOP_HEARTBEATS.pop(loop, None)

    task = loop.create_task(_heartbeat())
    state = _LoopHeartbeatState(pending=0, task=task)
    _LOOP_HEARTBEATS[loop] = state
    return state


def _fallback_max_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(4, min(32, cpu + 4))


def resolve_to_thread_max_workers() -> int:
    """Resolve the shared offload thread count.

    The default remains ``max(4, min(32, os.cpu_count() + 4))``. Set
    ``TT_TO_THREAD_MAX_WORKERS`` before the executor is created to override it,
    or call ``configure_to_thread_max_workers`` during benchmark setup.
    """
    if _EXPLICIT_MAX_WORKERS is not None:
        return int(_EXPLICIT_MAX_WORKERS)

    raw = os.environ.get("TT_TO_THREAD_MAX_WORKERS")
    if raw is not None and str(raw).strip():
        try:
            parsed = int(str(raw).strip())
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid TT_TO_THREAD_MAX_WORKERS=%r", raw)
    return _fallback_max_workers()


def _default_max_workers() -> int:
    return resolve_to_thread_max_workers()


def configure_to_thread_max_workers(max_workers: Optional[int]) -> None:
    """Set or clear the process-local shared offload thread cap.

    This is intended for benchmark/test setup before offloaded work is active.
    Passing ``None`` clears the explicit cap and falls back to the environment or
    default heuristic.
    """
    global _EXPLICIT_MAX_WORKERS, _GLOBAL_EXECUTOR

    if max_workers is not None:
        parsed = int(max_workers)
        if parsed <= 0:
            raise ValueError(f"max_workers must be positive or None, got {max_workers!r}")
        _EXPLICIT_MAX_WORKERS = parsed
    else:
        _EXPLICIT_MAX_WORKERS = None

    executor = _GLOBAL_EXECUTOR
    if executor is not None:
        _GLOBAL_EXECUTOR = None
        executor.shutdown(wait=False, cancel_futures=True)


def get_global_executor() -> ThreadPoolExecutor:
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        _GLOBAL_EXECUTOR = ThreadPoolExecutor(
            max_workers=_default_max_workers(),
            thread_name_prefix="thinkingtrees_to_thread",
        )

        def _shutdown() -> None:
            executor = _GLOBAL_EXECUTOR
            if executor is None:
                return
            executor.shutdown(wait=False)

        atexit.register(_shutdown)
    return _GLOBAL_EXECUTOR


async def to_thread(func: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
    """Run a sync callable in the shared global thread pool."""
    loop = asyncio.get_running_loop()
    heartbeat = _ensure_loop_heartbeat(loop)
    heartbeat.pending += 1
    bound = functools.partial(func, *args, **kwargs)
    future = loop.run_in_executor(get_global_executor(), bound)
    try:
        return await future
    finally:
        heartbeat.pending = max(0, int(heartbeat.pending) - 1)
        if heartbeat.pending <= 0:
            heartbeat.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat.task


async def gather_with_cleanup(
    coros: Iterable[Coroutine[Any, Any, Any]],
    return_exceptions: bool = True,
) -> List[Any]:
    """
    Gather coroutines with proper cleanup on cancellation.

    Unlike asyncio.gather, this function ensures all tasks are cancelled
    if the gather is cancelled (e.g., via Ctrl+C or timeout).

    Args:
        coros: Iterable of coroutines to run concurrently
        return_exceptions: If True, exceptions are returned as results.
                          If False, first exception is raised.

    Returns:
        List of results (or exceptions if return_exceptions=True)

    Example:
        # Safe gather that cleans up on cancellation
        results = await gather_with_cleanup([
            summarize_leaf(chunk) for chunk in chunks
        ])
    """
    # Convert coroutines to tasks
    tasks = [asyncio.ensure_future(c) for c in coros]

    if not tasks:
        return []

    try:
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    except asyncio.CancelledError:
        # Cancel all pending tasks
        cancelled_count = 0
        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1

        if cancelled_count > 0:
            logger.debug(f"Cancelled {cancelled_count} pending tasks due to cancellation")
            # Wait briefly for cancellation to propagate
            await asyncio.gather(*tasks, return_exceptions=True)

        # Re-raise the cancellation
        raise


async def cancel_tasks(tasks: Iterable[asyncio.Task], timeout: float = 5.0) -> int:
    """
    Cancel a collection of tasks and wait for them to complete.

    Args:
        tasks: Tasks to cancel
        timeout: Maximum time to wait for cancellation

    Returns:
        Number of tasks that were successfully cancelled
    """
    task_list = list(tasks)
    if not task_list:
        return 0

    # Cancel all tasks
    cancelled = 0
    for task in task_list:
        if not task.done():
            task.cancel()
            cancelled += 1

    if cancelled == 0:
        return 0

    # Wait for cancellation with timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(*task_list, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        remaining = sum(1 for t in task_list if not t.done())
        logger.warning(
            f"Timeout ({timeout}s) waiting for task cancellation. "
            f"{remaining}/{len(task_list)} tasks may still be running."
        )

    return cancelled
