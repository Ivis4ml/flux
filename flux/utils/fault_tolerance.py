"""
Fault tolerance utilities for Flux.

Provides graceful shutdown handling, retry logic, and recovery mechanisms.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ShutdownReason(Enum):
    """Reason for shutdown."""

    USER_INTERRUPT = "user_interrupt"
    SIGNAL = "signal"
    ERROR = "error"
    TIMEOUT = "timeout"
    PROGRAMMATIC = "programmatic"


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff.
        jitter: Whether to add random jitter to delays.
        retry_on: Exception types to retry on (None = all).
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] | None = None

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay,
        )

        if self.jitter:
            import random
            delay = delay * (0.5 + random.random())

        return delay

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if should retry for given exception.

        Args:
            exception: The exception that occurred.
            attempt: Current attempt number.

        Returns:
            True if should retry.
        """
        if attempt >= self.max_retries:
            return False

        if self.retry_on is None:
            return True

        return isinstance(exception, self.retry_on)


def with_retry(
    config: RetryConfig | None = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[F], F]:
    """Decorator for automatic retry with exponential backoff.

    Args:
        config: RetryConfig to use (overrides other params).
        max_retries: Maximum retry attempts.
        base_delay: Initial delay between retries.
        on_retry: Callback called on each retry.

    Returns:
        Decorated function with retry logic.

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        def flaky_operation():
            ...
    """
    if config is None:
        config = RetryConfig(max_retries=max_retries, base_delay=base_delay)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e, attempt):
                        raise

                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for "
                        f"{func.__name__} after {delay:.2f}s: {e}"
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e, attempt):
                        raise

                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for "
                        f"{func.__name__} after {delay:.2f}s: {e}"
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


@dataclass
class ShutdownState:
    """State of graceful shutdown."""

    requested: bool = False
    reason: ShutdownReason | None = None
    timestamp: float | None = None
    error: Exception | None = None


class GracefulShutdown:
    """Manages graceful shutdown with signal handling.

    Features:
    - Signal handling (SIGINT, SIGTERM)
    - Async cleanup callbacks
    - Timeout-based forced shutdown
    - Thread-safe shutdown coordination

    Example:
        shutdown = GracefulShutdown(timeout=30.0)

        # Register cleanup handlers
        shutdown.register_cleanup(save_checkpoint)
        shutdown.register_cleanup(close_connections)

        # In main loop
        while not shutdown.is_requested:
            train_step()

        # Or use context manager
        with shutdown:
            while not shutdown.is_requested:
                train_step()
    """

    def __init__(
        self,
        timeout: float = 30.0,
        install_handlers: bool = True,
    ) -> None:
        """Initialize graceful shutdown handler.

        Args:
            timeout: Timeout for graceful shutdown in seconds.
            install_handlers: Whether to install signal handlers.
        """
        self.timeout = timeout
        self._state = ShutdownState()
        self._cleanup_handlers: list[Callable[[], Any]] = []
        self._async_cleanup_handlers: list[Callable[[], Any]] = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._original_handlers: dict[int, Any] = {}

        if install_handlers:
            self._install_handlers()

    @property
    def is_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._state.requested

    @property
    def reason(self) -> ShutdownReason | None:
        """Get shutdown reason if requested."""
        return self._state.reason

    def request_shutdown(
        self,
        reason: ShutdownReason = ShutdownReason.PROGRAMMATIC,
        error: Exception | None = None,
    ) -> None:
        """Request graceful shutdown.

        Args:
            reason: Reason for shutdown.
            error: Optional error that triggered shutdown.
        """
        with self._lock:
            if self._state.requested:
                return

            self._state.requested = True
            self._state.reason = reason
            self._state.timestamp = time.time()
            self._state.error = error
            self._event.set()

        logger.info(f"Graceful shutdown requested: {reason.value}")

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for shutdown to be requested.

        Args:
            timeout: Maximum time to wait.

        Returns:
            True if shutdown was requested, False if timeout.
        """
        return self._event.wait(timeout)

    def register_cleanup(
        self,
        handler: Callable[[], Any],
        *,
        is_async: bool = False,
    ) -> None:
        """Register a cleanup handler.

        Args:
            handler: Cleanup function to call on shutdown.
            is_async: Whether handler is async.
        """
        if is_async:
            self._async_cleanup_handlers.append(handler)
        else:
            self._cleanup_handlers.append(handler)

    def run_cleanup(self) -> list[Exception]:
        """Run all cleanup handlers.

        Returns:
            List of exceptions from failed handlers.
        """
        errors: list[Exception] = []

        # Run sync handlers
        for handler in reversed(self._cleanup_handlers):
            try:
                handler()
            except Exception as e:
                logger.error(f"Cleanup handler failed: {e}")
                errors.append(e)

        return errors

    async def run_cleanup_async(self) -> list[Exception]:
        """Run all cleanup handlers including async ones.

        Returns:
            List of exceptions from failed handlers.
        """
        errors: list[Exception] = []

        # Run sync handlers first
        errors.extend(self.run_cleanup())

        # Run async handlers
        for handler in reversed(self._async_cleanup_handlers):
            try:
                result = handler()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Async cleanup handler failed: {e}")
                errors.append(e)

        return errors

    def _install_handlers(self) -> None:
        """Install signal handlers."""
        signals = [signal.SIGINT, signal.SIGTERM]

        # On Windows, SIGTERM doesn't exist
        if sys.platform == "win32":
            signals = [signal.SIGINT]

        for sig in signals:
            try:
                self._original_handlers[sig] = signal.signal(
                    sig, self._signal_handler
                )
            except (OSError, ValueError):
                # Signal handling may not be available in all contexts
                pass

    def _uninstall_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name}")

        if self._state.requested:
            # Second signal - force exit
            logger.warning("Forced exit due to repeated signal")
            sys.exit(1)

        self.request_shutdown(ShutdownReason.SIGNAL)

    def __enter__(self) -> "GracefulShutdown":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        if exc_type is not None:
            self.request_shutdown(ShutdownReason.ERROR, error=exc_val)

        self.run_cleanup()
        self._uninstall_handlers()


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered

    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

        @breaker
        def call_external_service():
            ...
    """

    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening.
            recovery_timeout: Time to wait before half-open.
            half_open_max_calls: Test calls in half-open state.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        """Get current state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == self.State.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = self.State.HALF_OPEN
                    self._success_count = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.State.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.State.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
            elif self._state == self.State.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.State.HALF_OPEN:
                self._state = self.State.OPEN
                logger.info("Circuit breaker: HALF_OPEN -> OPEN")
            elif self._state == self.State.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.State.OPEN
                    logger.info("Circuit breaker: CLOSED -> OPEN")

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        state = self.state
        if state == self.State.CLOSED:
            return True
        elif state == self.State.HALF_OPEN:
            return True
        else:
            return False

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with circuit breaker."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.allow_request():
                raise CircuitBreakerOpen(
                    f"Circuit breaker is open for {func.__name__}"
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.allow_request():
                raise CircuitBreakerOpen(
                    f"Circuit breaker is open for {func.__name__}"
                )

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""

    pass


@dataclass
class HealthState:
    """Health state of a component."""

    healthy: bool = True
    last_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    last_error: str | None = None


class HealthMonitor:
    """Monitors health of multiple components.

    Example:
        monitor = HealthMonitor()

        # Register health checks
        monitor.register("database", check_db_connection)
        monitor.register("cache", check_cache_connection)

        # Check health
        health = monitor.check_all()
        if not health["healthy"]:
            handle_unhealthy(health)
    """

    def __init__(
        self,
        unhealthy_threshold: int = 3,
        check_timeout: float = 5.0,
    ) -> None:
        """Initialize health monitor.

        Args:
            unhealthy_threshold: Consecutive failures before unhealthy.
            check_timeout: Timeout for health checks.
        """
        self.unhealthy_threshold = unhealthy_threshold
        self.check_timeout = check_timeout

        self._checks: dict[str, Callable[[], bool]] = {}
        self._states: dict[str, HealthState] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        check: Callable[[], bool],
    ) -> None:
        """Register a health check.

        Args:
            name: Name of the component.
            check: Function returning True if healthy.
        """
        with self._lock:
            self._checks[name] = check
            self._states[name] = HealthState()

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
            self._states.pop(name, None)

    def check(self, name: str) -> HealthState:
        """Check health of a specific component.

        Args:
            name: Name of the component.

        Returns:
            HealthState of the component.
        """
        if name not in self._checks:
            return HealthState(healthy=False, last_error="Not registered")

        check_fn = self._checks[name]
        state = self._states[name]

        try:
            healthy = check_fn()
            state.healthy = healthy
            state.last_check = time.time()

            if healthy:
                state.consecutive_failures = 0
                state.last_error = None
            else:
                state.consecutive_failures += 1
                state.last_error = "Check returned False"

        except Exception as e:
            state.healthy = False
            state.last_check = time.time()
            state.consecutive_failures += 1
            state.last_error = str(e)

        # Mark as unhealthy if threshold exceeded
        if state.consecutive_failures >= self.unhealthy_threshold:
            state.healthy = False

        return state

    def check_all(self) -> dict[str, Any]:
        """Check health of all registered components.

        Returns:
            Dictionary with overall health and component details.
        """
        results: dict[str, HealthState] = {}

        for name in self._checks:
            results[name] = self.check(name)

        overall_healthy = all(s.healthy for s in results.values())

        return {
            "healthy": overall_healthy,
            "timestamp": time.time(),
            "components": {
                name: {
                    "healthy": state.healthy,
                    "last_check": state.last_check,
                    "failures": state.consecutive_failures,
                    "error": state.last_error,
                }
                for name, state in results.items()
            },
        }

    async def check_all_async(self) -> dict[str, Any]:
        """Check all components asynchronously.

        Returns:
            Dictionary with overall health and component details.
        """
        # For now, run checks in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check_all)
