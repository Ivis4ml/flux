"""
Asynchronous task runner for Flux.

Provides a background thread with an event loop for executing async tasks,
with support for pause/resume, task tracking, and graceful shutdown.

Based on AReaL's AsyncTaskRunner design with Flux-specific improvements.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

try:
    import uvloop

    HAS_UVLOOP = True
except ImportError:
    HAS_UVLOOP = False


T = TypeVar("T")


class TaskStatus(str, Enum):
    """Status of a task in the runner."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Information about a submitted task."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: Exception | None = None

    @property
    def elapsed_ms(self) -> float | None:
        """Time elapsed since task started, in milliseconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds() * 1000

    @property
    def wait_time_ms(self) -> float | None:
        """Time spent waiting before starting, in milliseconds."""
        if self.started_at is None:
            return None
        return (self.started_at - self.created_at).total_seconds() * 1000


@dataclass
class TaskResult:
    """Result of a completed task."""

    task_id: str
    success: bool
    result: Any = None
    error: Exception | None = None
    elapsed_ms: float = 0.0


class AsyncTaskRunner:
    """Background async task runner with thread-safe interface.

    Runs an asyncio event loop in a background thread, allowing synchronous
    code to submit async tasks and retrieve results.

    Features:
    - Thread-safe task submission and result retrieval
    - Pause/resume support for flow control
    - Task ID tracking for result correlation
    - Graceful shutdown with timeout
    - Optional uvloop for better performance

    Example:
        runner = AsyncTaskRunner(max_pending=100)
        runner.start()

        # Submit async task
        task_id = runner.submit(some_async_function(args))

        # Get result (blocks until ready)
        result = runner.get_result(timeout=10.0)

        runner.stop()
    """

    def __init__(
        self,
        max_pending: int = 1000,
        max_concurrent: int = 100,
        use_uvloop: bool = True,
        name: str = "async-runner",
    ) -> None:
        """Initialize the async task runner.

        Args:
            max_pending: Maximum number of pending tasks in queue.
            max_concurrent: Maximum concurrent tasks executing.
            use_uvloop: Whether to use uvloop (if available).
            name: Name for the runner (used in thread name).
        """
        self.max_pending = max_pending
        self.max_concurrent = max_concurrent
        self.use_uvloop = use_uvloop and HAS_UVLOOP
        self.name = name

        # Threading primitives
        self._lock = threading.Lock()
        self._submit_cv = threading.Condition(self._lock)
        self._result_cv = threading.Condition(self._lock)
        self._shutdown_event = threading.Event()
        self._ready_event = threading.Event()
        self._paused_event = threading.Event()

        # Task management
        self._pending_tasks: deque[tuple[str, Awaitable[Any]]] = deque()
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}
        self._completed_results: deque[TaskResult] = deque()
        self._task_info: dict[str, TaskInfo] = {}

        # Background thread
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Shutdown hooks (executed in reverse order)
        self._shutdown_hooks: list[Callable[[], Awaitable[None] | None]] = []

        # Exception from background thread
        self._background_exception: Exception | None = None

    @property
    def is_running(self) -> bool:
        """Whether the runner is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        """Whether the runner is paused."""
        return self._paused_event.is_set()

    @property
    def num_pending(self) -> int:
        """Number of pending tasks."""
        with self._lock:
            return len(self._pending_tasks)

    @property
    def num_running(self) -> int:
        """Number of running tasks."""
        with self._lock:
            return len(self._running_tasks)

    @property
    def num_completed(self) -> int:
        """Number of completed results waiting to be retrieved."""
        with self._lock:
            return len(self._completed_results)

    @property
    def capacity(self) -> int:
        """Remaining capacity for new tasks."""
        with self._lock:
            return max(
                0, self.max_pending - len(self._pending_tasks) - len(self._running_tasks)
            )

    def start(self, timeout: float = 10.0) -> None:
        """Start the background event loop thread.

        Args:
            timeout: Maximum time to wait for loop to be ready.

        Raises:
            RuntimeError: If runner is already running or fails to start.
        """
        if self.is_running:
            raise RuntimeError("Runner is already running")

        self._shutdown_event.clear()
        self._ready_event.clear()
        self._paused_event.clear()
        self._background_exception = None

        self._thread = threading.Thread(
            target=self._run_event_loop,
            name=f"{self.name}-thread",
            daemon=True,
        )
        self._thread.start()

        if not self._ready_event.wait(timeout=timeout):
            self._shutdown_event.set()
            raise RuntimeError(f"Event loop failed to start within {timeout}s")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the background event loop thread.

        Cancels all pending tasks and waits for running tasks to complete.

        Args:
            timeout: Maximum time to wait for shutdown.
        """
        if not self.is_running:
            return

        self._shutdown_event.set()

        # Wake up any waiting threads
        with self._submit_cv:
            self._submit_cv.notify_all()
        with self._result_cv:
            self._result_cv.notify_all()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Thread didn't stop cleanly
                pass
            self._thread = None

    def pause(self) -> None:
        """Pause task processing.

        Running tasks continue, but no new tasks will start.
        """
        self._paused_event.set()

    def resume(self) -> None:
        """Resume task processing."""
        self._paused_event.clear()
        with self._submit_cv:
            self._submit_cv.notify_all()

    def submit(
        self, coro: Awaitable[T], task_id: str | None = None
    ) -> str:
        """Submit an async task for execution.

        Args:
            coro: The coroutine to execute.
            task_id: Optional task ID (generated if not provided).

        Returns:
            The task ID.

        Raises:
            RuntimeError: If runner is not running or queue is full.
        """
        if not self.is_running:
            raise RuntimeError("Runner is not running")

        self._check_background_exception()

        if task_id is None:
            task_id = str(uuid.uuid4())

        with self._submit_cv:
            if len(self._pending_tasks) >= self.max_pending:
                raise RuntimeError("Task queue is full")

            task_info = TaskInfo(task_id=task_id)
            self._task_info[task_id] = task_info
            self._pending_tasks.append((task_id, coro))
            self._submit_cv.notify()

        return task_id

    def submit_wait(
        self, coro: Awaitable[T], task_id: str | None = None, timeout: float | None = None
    ) -> str:
        """Submit an async task, waiting if queue is full.

        Args:
            coro: The coroutine to execute.
            task_id: Optional task ID (generated if not provided).
            timeout: Maximum time to wait for queue space.

        Returns:
            The task ID.

        Raises:
            RuntimeError: If runner is not running.
            TimeoutError: If timeout expires before queue has space.
        """
        if not self.is_running:
            raise RuntimeError("Runner is not running")

        self._check_background_exception()

        if task_id is None:
            task_id = str(uuid.uuid4())

        with self._submit_cv:
            while len(self._pending_tasks) >= self.max_pending:
                if self._shutdown_event.is_set():
                    raise RuntimeError("Runner is shutting down")
                if not self._submit_cv.wait(timeout=timeout):
                    raise TimeoutError("Timed out waiting for queue space")
                self._check_background_exception()

            task_info = TaskInfo(task_id=task_id)
            self._task_info[task_id] = task_info
            self._pending_tasks.append((task_id, coro))
            self._submit_cv.notify()

        return task_id

    def get_result(self, timeout: float | None = None) -> TaskResult | None:
        """Get the next completed result.

        Args:
            timeout: Maximum time to wait for a result.

        Returns:
            TaskResult or None if timeout expires.

        Raises:
            RuntimeError: If background thread raised an exception.
        """
        self._check_background_exception()

        with self._result_cv:
            while len(self._completed_results) == 0:
                if self._shutdown_event.is_set() and len(self._running_tasks) == 0:
                    return None
                if not self._result_cv.wait(timeout=timeout):
                    return None
                self._check_background_exception()

            return self._completed_results.popleft()

    def get_results(
        self, max_results: int = 0, timeout: float | None = None
    ) -> list[TaskResult]:
        """Get multiple completed results.

        Args:
            max_results: Maximum results to return (0 = all available).
            timeout: Maximum time to wait for at least one result.

        Returns:
            List of TaskResults (may be empty if timeout expires).
        """
        self._check_background_exception()

        results = []
        with self._result_cv:
            # Wait for at least one result
            while len(self._completed_results) == 0:
                if self._shutdown_event.is_set() and len(self._running_tasks) == 0:
                    return results
                if not self._result_cv.wait(timeout=timeout):
                    return results
                self._check_background_exception()

            # Collect available results
            count = 0
            while self._completed_results:
                if max_results > 0 and count >= max_results:
                    break
                results.append(self._completed_results.popleft())
                count += 1

        return results

    def get_task_info(self, task_id: str) -> TaskInfo | None:
        """Get information about a task.

        Args:
            task_id: The task ID.

        Returns:
            TaskInfo or None if task not found.
        """
        with self._lock:
            return self._task_info.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if task was found and cancelled.
        """
        with self._lock:
            # Check pending tasks
            for i, (tid, _) in enumerate(self._pending_tasks):
                if tid == task_id:
                    del self._pending_tasks[i]
                    if task_id in self._task_info:
                        self._task_info[task_id].status = TaskStatus.CANCELLED
                        self._task_info[task_id].completed_at = datetime.now()
                    return True

            # Check running tasks
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                task.cancel()
                return True

        return False

    def add_shutdown_hook(
        self, hook: Callable[[], Awaitable[None] | None]
    ) -> None:
        """Add a shutdown hook to be called on stop.

        Hooks are called in reverse order (LIFO).

        Args:
            hook: Function to call on shutdown (can be async).
        """
        self._shutdown_hooks.append(hook)

    def _check_background_exception(self) -> None:
        """Check if background thread raised an exception."""
        if self._background_exception is not None:
            exc = self._background_exception
            self._background_exception = None
            raise exc

    def _run_event_loop(self) -> None:
        """Run the event loop in background thread."""
        try:
            # Set up event loop
            if self.use_uvloop:
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Signal ready
            self._ready_event.set()

            # Run main loop
            self._loop.run_until_complete(self._main_loop())

        except Exception as e:
            with self._lock:
                self._background_exception = e
            # Wake up waiting threads
            with self._result_cv:
                self._result_cv.notify_all()
        finally:
            # Run shutdown hooks
            if self._loop is not None:
                self._loop.run_until_complete(self._run_shutdown_hooks())
                self._loop.close()
            self._loop = None

    async def _main_loop(self) -> None:
        """Main event loop processing."""
        while not self._shutdown_event.is_set():
            # Process pending tasks
            await self._process_pending_tasks()

            # Small sleep to prevent busy loop
            await asyncio.sleep(0.001)

        # Cancel remaining tasks on shutdown
        await self._cancel_remaining_tasks()

    async def _process_pending_tasks(self) -> None:
        """Process pending tasks, starting new ones if capacity allows."""
        if self._paused_event.is_set():
            return

        tasks_to_start = []

        with self._lock:
            while (
                self._pending_tasks
                and len(self._running_tasks) < self.max_concurrent
            ):
                task_id, coro = self._pending_tasks.popleft()
                tasks_to_start.append((task_id, coro))

        # Start tasks outside lock
        for task_id, coro in tasks_to_start:
            await self._start_task(task_id, coro)

    async def _start_task(self, task_id: str, coro: Awaitable[Any]) -> None:
        """Start a task and track it."""
        with self._lock:
            if task_id in self._task_info:
                self._task_info[task_id].status = TaskStatus.RUNNING
                self._task_info[task_id].started_at = datetime.now()

        async def task_wrapper() -> Any:
            try:
                return await coro
            finally:
                pass

        task = asyncio.create_task(task_wrapper())

        with self._lock:
            self._running_tasks[task_id] = task

        # Add completion callback
        task.add_done_callback(
            lambda t: self._on_task_complete(task_id, t)
        )

    def _on_task_complete(self, task_id: str, task: asyncio.Task[Any]) -> None:
        """Handle task completion."""
        with self._result_cv:
            # Remove from running
            self._running_tasks.pop(task_id, None)

            # Create result
            try:
                if task.cancelled():
                    result = TaskResult(
                        task_id=task_id,
                        success=False,
                        error=asyncio.CancelledError(),
                    )
                    status = TaskStatus.CANCELLED
                elif task.exception() is not None:
                    result = TaskResult(
                        task_id=task_id,
                        success=False,
                        error=task.exception(),
                    )
                    status = TaskStatus.FAILED
                else:
                    result = TaskResult(
                        task_id=task_id,
                        success=True,
                        result=task.result(),
                    )
                    status = TaskStatus.COMPLETED
            except Exception as e:
                result = TaskResult(
                    task_id=task_id,
                    success=False,
                    error=e,
                )
                status = TaskStatus.FAILED

            # Update task info
            if task_id in self._task_info:
                info = self._task_info[task_id]
                info.status = status
                info.completed_at = datetime.now()
                info.result = result.result
                info.error = result.error
                if info.started_at is not None:
                    result.elapsed_ms = (
                        info.completed_at - info.started_at
                    ).total_seconds() * 1000

            # Add to results queue
            self._completed_results.append(result)
            self._result_cv.notify()

        # Notify submit waiters
        with self._submit_cv:
            self._submit_cv.notify()

    async def _cancel_remaining_tasks(self) -> None:
        """Cancel all remaining tasks on shutdown."""
        with self._lock:
            task_ids = list(self._running_tasks.keys())

        for task_id in task_ids:
            with self._lock:
                task = self._running_tasks.get(task_id)
            if task is not None:
                task.cancel()

        # Wait for cancellation
        with self._lock:
            tasks = list(self._running_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_shutdown_hooks(self) -> None:
        """Run shutdown hooks in reverse order."""
        for hook in reversed(self._shutdown_hooks):
            try:
                result = hook()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Ignore errors in shutdown hooks


class BatchTaskDispatcher:
    """Dispatcher that batches tasks with staleness-aware capacity control.

    Sits on top of AsyncTaskRunner and provides:
    - Producer-consumer pattern with background threads
    - Staleness-aware capacity gating
    - FIFO result ordering by submission time

    Example:
        dispatcher = BatchTaskDispatcher(
            runner=runner,
            capacity_fn=staleness_manager.get_capacity,
        )
        dispatcher.start()

        # Submit tasks
        dispatcher.submit(task_factory, args=(prompt,))

        # Get batch of results
        results = dispatcher.get_batch(batch_size=32, timeout=10.0)

        dispatcher.stop()
    """

    def __init__(
        self,
        runner: AsyncTaskRunner,
        capacity_fn: Callable[[], int] | None = None,
        max_queue_size: int = 10000,
    ) -> None:
        """Initialize the batch dispatcher.

        Args:
            runner: The underlying async task runner.
            capacity_fn: Function that returns current capacity (for staleness gating).
            max_queue_size: Maximum size of input queue.
        """
        self.runner = runner
        self.capacity_fn = capacity_fn or (lambda: runner.max_concurrent)
        self.max_queue_size = max_queue_size

        # Input queue
        self._input_queue: deque[
            tuple[str, Callable[..., Awaitable[Any]], tuple[Any, ...], dict[str, Any]]
        ] = deque()
        self._input_lock = threading.Lock()
        self._input_cv = threading.Condition(self._input_lock)

        # Result tracking with timestamps for ordering
        self._pending_results: dict[str, tuple[datetime, TaskResult | None]] = {}
        self._result_lock = threading.Lock()
        self._result_cv = threading.Condition(self._result_lock)

        # Control
        self._shutdown = threading.Event()
        self._producer_thread: threading.Thread | None = None
        self._consumer_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        """Whether the dispatcher is running."""
        return (
            self._producer_thread is not None
            and self._producer_thread.is_alive()
            and self._consumer_thread is not None
            and self._consumer_thread.is_alive()
        )

    def start(self) -> None:
        """Start the dispatcher threads."""
        if self.is_running:
            return

        self._shutdown.clear()

        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="dispatcher-producer",
            daemon=True,
        )
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            name="dispatcher-consumer",
            daemon=True,
        )

        self._producer_thread.start()
        self._consumer_thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the dispatcher threads."""
        self._shutdown.set()

        with self._input_cv:
            self._input_cv.notify_all()
        with self._result_cv:
            self._result_cv.notify_all()

        if self._producer_thread is not None:
            self._producer_thread.join(timeout=timeout / 2)
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=timeout / 2)

        self._producer_thread = None
        self._consumer_thread = None

    def submit(
        self,
        task_factory: Callable[..., Awaitable[T]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> str:
        """Submit a task to be dispatched.

        Args:
            task_factory: Factory function that creates the coroutine.
            args: Positional arguments for factory.
            kwargs: Keyword arguments for factory.
            task_id: Optional task ID.

        Returns:
            The task ID.
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        kwargs = kwargs or {}

        with self._input_cv:
            if len(self._input_queue) >= self.max_queue_size:
                raise RuntimeError("Input queue is full")

            self._input_queue.append((task_id, task_factory, args, kwargs))
            self._input_cv.notify()

        return task_id

    def get_batch(
        self,
        batch_size: int,
        timeout: float | None = None,
        min_batch_size: int = 1,
    ) -> list[TaskResult]:
        """Get a batch of completed results.

        Results are returned in submission order (FIFO).

        Args:
            batch_size: Target batch size.
            timeout: Maximum time to wait.
            min_batch_size: Minimum results to return.

        Returns:
            List of TaskResults in submission order.
        """
        results = []
        deadline = None
        if timeout is not None:
            deadline = datetime.now().timestamp() + timeout

        with self._result_cv:
            while len(results) < batch_size:
                # Collect ready results
                ready_ids = [
                    tid
                    for tid, (_, res) in self._pending_results.items()
                    if res is not None
                ]

                # Sort by submission time for FIFO ordering
                ready_ids.sort(
                    key=lambda tid: self._pending_results[tid][0]
                )

                for tid in ready_ids:
                    if len(results) >= batch_size:
                        break
                    _, result = self._pending_results.pop(tid)
                    if result is not None:
                        results.append(result)

                # Check if we have enough
                if len(results) >= min_batch_size:
                    break

                # Check shutdown
                if self._shutdown.is_set():
                    break

                # Wait for more results
                remaining = None
                if deadline is not None:
                    remaining = deadline - datetime.now().timestamp()
                    if remaining <= 0:
                        break

                self._result_cv.wait(timeout=remaining)

        return results

    def _producer_loop(self) -> None:
        """Producer thread: submits tasks to runner based on capacity."""
        while not self._shutdown.is_set():
            try:
                with self._input_cv:
                    # Wait for tasks in queue
                    while not self._input_queue and not self._shutdown.is_set():
                        self._input_cv.wait(timeout=0.1)

                    if self._shutdown.is_set():
                        break

                    # Check capacity
                    capacity = self.capacity_fn()
                    if capacity <= 0 or self.runner.capacity <= 0:
                        self._input_cv.wait(timeout=0.01)
                        continue

                    # Get task to submit
                    if not self._input_queue:
                        continue

                    task_id, factory, args, kwargs = self._input_queue.popleft()

                # Submit to runner (outside lock)
                try:
                    coro = factory(*args, **kwargs)
                    submitted_at = datetime.now()

                    self.runner.submit(coro, task_id=task_id)

                    # Track pending result
                    with self._result_lock:
                        self._pending_results[task_id] = (submitted_at, None)

                except RuntimeError:
                    # Queue full, put back
                    with self._input_cv:
                        self._input_queue.appendleft(
                            (task_id, factory, args, kwargs)
                        )

            except Exception:
                pass  # Continue on errors

    def _consumer_loop(self) -> None:
        """Consumer thread: collects results from runner."""
        while not self._shutdown.is_set():
            try:
                result = self.runner.get_result(timeout=0.1)
                if result is None:
                    continue

                with self._result_cv:
                    if result.task_id in self._pending_results:
                        submitted_at, _ = self._pending_results[result.task_id]
                        self._pending_results[result.task_id] = (
                            submitted_at,
                            result,
                        )
                    self._result_cv.notify_all()

                # Notify producer that capacity may have freed up
                with self._input_cv:
                    self._input_cv.notify()

            except Exception:
                pass  # Continue on errors
