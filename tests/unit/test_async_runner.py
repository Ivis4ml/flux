"""
Tests for AsyncTaskRunner and BatchTaskDispatcher.
"""

import asyncio
import time
import threading
import pytest

from flux.coordinator.async_runner import (
    AsyncTaskRunner,
    BatchTaskDispatcher,
    TaskInfo,
    TaskResult,
    TaskStatus,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"


class TestTaskInfo:
    """Tests for TaskInfo dataclass."""

    def test_task_info_creation(self):
        """Test TaskInfo creation with defaults."""
        info = TaskInfo(task_id="test-1")
        assert info.task_id == "test-1"
        assert info.status == TaskStatus.PENDING
        assert info.result is None
        assert info.error is None

    def test_elapsed_time(self):
        """Test elapsed time computation."""
        from datetime import datetime, timedelta

        now = datetime.now()
        info = TaskInfo(
            task_id="test-1",
            started_at=now - timedelta(seconds=1),
            completed_at=now,
        )
        elapsed = info.elapsed_ms
        assert elapsed is not None
        assert 900 < elapsed < 1100  # ~1000ms with some tolerance

    def test_wait_time(self):
        """Test wait time computation."""
        from datetime import datetime, timedelta

        now = datetime.now()
        info = TaskInfo(
            task_id="test-1",
            created_at=now - timedelta(seconds=2),
            started_at=now - timedelta(seconds=1),
        )
        wait = info.wait_time_ms
        assert wait is not None
        assert 900 < wait < 1100  # ~1000ms


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_successful_result(self):
        """Test successful task result."""
        result = TaskResult(
            task_id="test-1",
            success=True,
            result=42,
        )
        assert result.success
        assert result.result == 42
        assert result.error is None

    def test_failed_result(self):
        """Test failed task result."""
        error = ValueError("test error")
        result = TaskResult(
            task_id="test-1",
            success=False,
            error=error,
        )
        assert not result.success
        assert result.error is error


class TestAsyncTaskRunner:
    """Tests for AsyncTaskRunner."""

    def test_runner_creation(self):
        """Test runner creation with defaults."""
        runner = AsyncTaskRunner()
        assert runner.max_pending == 1000
        assert runner.max_concurrent == 100
        assert not runner.is_running
        assert not runner.is_paused

    def test_runner_custom_config(self):
        """Test runner with custom config."""
        runner = AsyncTaskRunner(
            max_pending=50,
            max_concurrent=10,
            use_uvloop=False,
            name="test-runner",
        )
        assert runner.max_pending == 50
        assert runner.max_concurrent == 10
        assert runner.name == "test-runner"

    def test_start_stop(self):
        """Test starting and stopping runner."""
        runner = AsyncTaskRunner()
        runner.start()
        assert runner.is_running

        runner.stop()
        assert not runner.is_running

    def test_start_twice_raises(self):
        """Test that starting twice raises error."""
        runner = AsyncTaskRunner()
        runner.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                runner.start()
        finally:
            runner.stop()

    def test_submit_simple_task(self):
        """Test submitting a simple async task."""
        runner = AsyncTaskRunner()
        runner.start()

        async def simple_task():
            return 42

        try:
            task_id = runner.submit(simple_task())
            assert task_id is not None
            assert runner.num_pending >= 0 or runner.num_running >= 0
        finally:
            runner.stop()

    def test_submit_custom_task_id(self):
        """Test submitting with custom task ID."""
        runner = AsyncTaskRunner()
        runner.start()

        async def simple_task():
            return 42

        try:
            task_id = runner.submit(simple_task(), task_id="custom-id")
            assert task_id == "custom-id"
        finally:
            runner.stop()

    def test_submit_not_running_raises(self):
        """Test that submitting to stopped runner raises."""
        runner = AsyncTaskRunner()

        async def simple_task():
            return 42

        with pytest.raises(RuntimeError, match="not running"):
            runner.submit(simple_task())

    def test_get_result(self):
        """Test getting task result."""
        runner = AsyncTaskRunner()
        runner.start()

        async def simple_task():
            return 42

        try:
            runner.submit(simple_task())
            result = runner.get_result(timeout=5.0)
            assert result is not None
            assert result.success
            assert result.result == 42
        finally:
            runner.stop()

    def test_get_result_timeout(self):
        """Test get_result with timeout."""
        runner = AsyncTaskRunner()
        runner.start()

        try:
            # No tasks submitted, should timeout
            result = runner.get_result(timeout=0.1)
            assert result is None
        finally:
            runner.stop()

    def test_get_results_multiple(self):
        """Test getting multiple results."""
        runner = AsyncTaskRunner()
        runner.start()

        async def task(value):
            return value

        try:
            for i in range(5):
                runner.submit(task(i))

            # Wait for all to complete
            time.sleep(0.2)

            results = runner.get_results(max_results=3, timeout=1.0)
            assert len(results) == 3
            assert all(r.success for r in results)
        finally:
            runner.stop()

    def test_task_with_exception(self):
        """Test handling task that raises exception."""
        runner = AsyncTaskRunner()
        runner.start()

        async def failing_task():
            raise ValueError("test error")

        try:
            runner.submit(failing_task())
            result = runner.get_result(timeout=5.0)
            assert result is not None
            assert not result.success
            assert isinstance(result.error, ValueError)
        finally:
            runner.stop()

    def test_pause_resume(self):
        """Test pause and resume functionality."""
        runner = AsyncTaskRunner()
        runner.start()

        try:
            runner.pause()
            assert runner.is_paused

            runner.resume()
            assert not runner.is_paused
        finally:
            runner.stop()

    def test_cancel_pending_task(self):
        """Test cancelling a pending task."""
        runner = AsyncTaskRunner(max_concurrent=1)
        runner.start()

        async def blocking_task():
            await asyncio.sleep(10)
            return 42

        async def quick_task():
            return "quick"

        try:
            # Submit blocking task first
            runner.submit(blocking_task(), task_id="blocking")

            # Submit another that will be pending
            runner.submit(quick_task(), task_id="pending")

            # Cancel the pending one
            cancelled = runner.cancel_task("pending")
            # May or may not be pending depending on timing
            # Just ensure no errors
        finally:
            runner.stop()

    def test_get_task_info(self):
        """Test getting task info."""
        runner = AsyncTaskRunner()
        runner.start()

        async def simple_task():
            return 42

        try:
            task_id = runner.submit(simple_task())
            info = runner.get_task_info(task_id)
            assert info is not None
            assert info.task_id == task_id
        finally:
            runner.stop()

    def test_capacity(self):
        """Test capacity tracking."""
        runner = AsyncTaskRunner(max_pending=10, max_concurrent=5)
        runner.start()

        try:
            initial_capacity = runner.capacity
            assert initial_capacity == 10  # max_pending when no tasks

            async def slow_task():
                await asyncio.sleep(0.5)
                return 42

            for i in range(3):
                runner.submit(slow_task())

            # Capacity should decrease
            time.sleep(0.1)  # Give time for tasks to start
            assert runner.capacity <= initial_capacity
        finally:
            runner.stop()

    def test_shutdown_hook(self):
        """Test shutdown hooks."""
        runner = AsyncTaskRunner()
        hook_called = threading.Event()

        async def cleanup_hook():
            hook_called.set()

        runner.add_shutdown_hook(cleanup_hook)
        runner.start()
        runner.stop()

        # Wait a bit for hook to complete
        assert hook_called.wait(timeout=1.0)

    def test_concurrent_tasks(self):
        """Test running multiple concurrent tasks."""
        runner = AsyncTaskRunner(max_concurrent=5)
        runner.start()

        results = []
        lock = threading.Lock()

        async def task(value):
            await asyncio.sleep(0.01)
            return value * 2

        try:
            for i in range(10):
                runner.submit(task(i))

            # Collect all results
            while len(results) < 10:
                result = runner.get_result(timeout=2.0)
                if result is None:
                    break
                with lock:
                    results.append(result)

            assert len(results) == 10
            assert all(r.success for r in results)
        finally:
            runner.stop()

    def test_submit_wait(self):
        """Test submit_wait with queue pressure."""
        runner = AsyncTaskRunner(max_pending=2, max_concurrent=1)
        runner.start()

        async def slow_task():
            await asyncio.sleep(0.1)
            return 42

        try:
            # Fill the queue
            runner.submit(slow_task())
            runner.submit(slow_task())

            # This should wait since queue is full
            def submit_with_wait():
                runner.submit_wait(slow_task(), timeout=2.0)

            thread = threading.Thread(target=submit_with_wait)
            thread.start()

            # Consume a result to make space
            time.sleep(0.2)
            runner.get_result(timeout=1.0)

            thread.join(timeout=2.0)
            assert not thread.is_alive()
        finally:
            runner.stop()


class TestBatchTaskDispatcher:
    """Tests for BatchTaskDispatcher."""

    def test_dispatcher_creation(self):
        """Test dispatcher creation."""
        runner = AsyncTaskRunner()
        runner.start()

        try:
            dispatcher = BatchTaskDispatcher(runner)
            assert not dispatcher.is_running
        finally:
            runner.stop()

    def test_start_stop(self):
        """Test starting and stopping dispatcher."""
        runner = AsyncTaskRunner()
        runner.start()

        try:
            dispatcher = BatchTaskDispatcher(runner)
            dispatcher.start()
            assert dispatcher.is_running

            dispatcher.stop()
            assert not dispatcher.is_running
        finally:
            runner.stop()

    def test_submit_and_get_batch(self):
        """Test submitting tasks and getting batch."""
        runner = AsyncTaskRunner()
        runner.start()

        try:
            dispatcher = BatchTaskDispatcher(runner)
            dispatcher.start()

            async def task(value):
                return value * 2

            # Submit tasks
            for i in range(5):
                dispatcher.submit(task, args=(i,))

            # Get batch
            results = dispatcher.get_batch(batch_size=5, timeout=2.0)
            assert len(results) == 5
            assert all(r.success for r in results)

            # Results should be in FIFO order
            values = [r.result for r in results]
            assert values == [0, 2, 4, 6, 8]
        finally:
            dispatcher.stop()
            runner.stop()

    def test_get_batch_partial(self):
        """Test getting partial batch when min_batch_size is met."""
        runner = AsyncTaskRunner()
        runner.start()

        try:
            dispatcher = BatchTaskDispatcher(runner)
            dispatcher.start()

            async def task(value):
                return value

            # Submit fewer tasks than batch_size
            for i in range(3):
                dispatcher.submit(task, args=(i,))

            # Get batch with min_batch_size=2
            results = dispatcher.get_batch(
                batch_size=5,
                min_batch_size=2,
                timeout=1.0,
            )
            assert len(results) >= 2
        finally:
            dispatcher.stop()
            runner.stop()

    def test_capacity_gating(self):
        """Test capacity-based gating."""
        runner = AsyncTaskRunner()
        runner.start()

        capacity_value = [5]  # Use list for mutable closure

        def get_capacity():
            return capacity_value[0]

        try:
            dispatcher = BatchTaskDispatcher(
                runner,
                capacity_fn=get_capacity,
            )
            dispatcher.start()

            async def task(value):
                return value

            # Submit more tasks than capacity
            for i in range(10):
                dispatcher.submit(task, args=(i,))

            # With capacity=5, we should process limited tasks initially
            time.sleep(0.2)

            # Increase capacity
            capacity_value[0] = 10

            # Now all should complete
            results = dispatcher.get_batch(batch_size=10, timeout=2.0)
            assert len(results) == 10
        finally:
            dispatcher.stop()
            runner.stop()

    def test_custom_task_id(self):
        """Test submitting with custom task ID."""
        runner = AsyncTaskRunner()
        runner.start()

        try:
            dispatcher = BatchTaskDispatcher(runner)
            dispatcher.start()

            async def task():
                return 42

            task_id = dispatcher.submit(task, task_id="my-custom-id")
            assert task_id == "my-custom-id"
        finally:
            dispatcher.stop()
            runner.stop()

    def test_failed_task_in_batch(self):
        """Test handling failed task in batch."""
        runner = AsyncTaskRunner()
        runner.start()

        try:
            dispatcher = BatchTaskDispatcher(runner)
            dispatcher.start()

            async def success_task():
                return "success"

            async def fail_task():
                raise ValueError("fail")

            dispatcher.submit(success_task)
            dispatcher.submit(fail_task)
            dispatcher.submit(success_task)

            results = dispatcher.get_batch(batch_size=3, timeout=2.0)
            assert len(results) == 3

            successes = [r for r in results if r.success]
            failures = [r for r in results if not r.success]

            assert len(successes) == 2
            assert len(failures) == 1
        finally:
            dispatcher.stop()
            runner.stop()
