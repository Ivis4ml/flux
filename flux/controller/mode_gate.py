"""
Mode Gate for Flux adaptive async training.

Provides a state machine controlling sync/async transitions based on:
- Staleness thresholds (from PID controller)
- Capacity limits (from StalenessManager)
- Backpressure signals (buffer watermark, GPU utilization)

The Mode Gate acts as the central decision point for async training,
ensuring training stability while maximizing throughput.

State Machine:
    ASYNC_RUNNING (default)
        │
        ├──(staleness > threshold)──► SYNC_BARRIER
        │                                   │
        │                                   ├──(in_flight drained)──► ASYNC_RUNNING
        │                                   │
        ├──(capacity exhausted)─────────────┼───► THROTTLED
        │   (buffer > high watermark)───────┘           │
        │                                               │
        └──(capacity restored)──────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from flux.controller.staleness import StalenessManager

logger = logging.getLogger(__name__)


class AsyncMode(Enum):
    """Async training mode states."""

    SYNC_BARRIER = auto()    # Waiting for all in-flight to complete
    ASYNC_RUNNING = auto()   # Normal async operation
    THROTTLED = auto()       # Capacity exhausted, backpressure active


@dataclass
class ModeGateState:
    """Snapshot of Mode Gate state for external inspection."""

    mode: AsyncMode
    reason: str
    staleness: float
    capacity: int
    in_flight: int
    buffer_fill_ratio: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"ModeGateState(mode={self.mode.name}, reason={self.reason}, "
            f"staleness={self.staleness:.3f}, capacity={self.capacity}, "
            f"in_flight={self.in_flight})"
        )


@dataclass
class ModeGateConfig:
    """Configuration for Mode Gate thresholds."""

    # Staleness threshold to trigger SYNC_BARRIER
    staleness_threshold: float = 0.3

    # Capacity threshold (0 = exhausted)
    capacity_low_watermark: int = 0

    # Buffer fill ratio to trigger THROTTLED
    buffer_high_watermark: float = 0.9

    # Minimum time in SYNC_BARRIER before allowing transition back
    min_barrier_duration_ms: float = 100.0

    # Hysteresis: staleness must drop below this to exit SYNC_BARRIER
    staleness_recovery_threshold: float | None = None  # Default: 80% of threshold

    # Enable logging of state transitions
    log_transitions: bool = True

    def __post_init__(self) -> None:
        if self.staleness_recovery_threshold is None:
            self.staleness_recovery_threshold = self.staleness_threshold * 0.8


class ModeGate:
    """
    State machine controlling sync/async transitions.

    Integrates:
    - Staleness thresholds (from PID controller)
    - Capacity limits (from StalenessManager)
    - Backpressure signals (buffer watermark, GPU util)

    Usage:
        gate = ModeGate(config=ModeGateConfig(staleness_threshold=0.3))

        # In training loop
        gate_state = gate.evaluate(
            staleness=staleness_manager.current_staleness,
            capacity=staleness_manager.get_capacity(),
            buffer_fill_ratio=trajectory_buffer.fill_ratio,
            in_flight=staleness_manager.stats.total_in_flight,
        )

        if gate_state.mode == AsyncMode.SYNC_BARRIER:
            await gate.enforce_barrier(wait_for_in_flight)
            await sync_weights()

        if gate.can_submit_rollout():
            submit_rollouts()

    Thread Safety:
        All public methods are thread-safe.
    """

    def __init__(
        self,
        config: ModeGateConfig | None = None,
        staleness_manager: "StalenessManager | None" = None,
    ) -> None:
        """Initialize the Mode Gate.

        Args:
            config: Mode Gate configuration.
            staleness_manager: Optional staleness manager for integrated queries.
        """
        self.config = config or ModeGateConfig()
        self.staleness_manager = staleness_manager

        # Current state
        self._current_mode = AsyncMode.ASYNC_RUNNING
        self._current_reason = "initialized"

        # State tracking
        self._in_flight_count = 0
        self._last_state_change = time.time()

        # Barrier management
        self._barrier_event: asyncio.Event | None = None
        self._barrier_lock = asyncio.Lock()

        # Thread safety
        self._lock = threading.Lock()

        # Transition history for diagnostics
        self._transition_history: list[tuple[float, AsyncMode, AsyncMode, str]] = []

    @property
    def current_mode(self) -> AsyncMode:
        """Current mode."""
        with self._lock:
            return self._current_mode

    @property
    def current_state(self) -> ModeGateState:
        """Get current state snapshot."""
        with self._lock:
            return ModeGateState(
                mode=self._current_mode,
                reason=self._current_reason,
                staleness=self._last_staleness if hasattr(self, "_last_staleness") else 0.0,
                capacity=self._last_capacity if hasattr(self, "_last_capacity") else 0,
                in_flight=self._in_flight_count,
            )

    def evaluate(
        self,
        staleness: float,
        capacity: int,
        buffer_fill_ratio: float,
        in_flight: int,
    ) -> ModeGateState:
        """
        Evaluate current conditions and determine mode.

        Called after each training step or rollout completion.

        Args:
            staleness: Current staleness metric (0-1 range typically).
            capacity: Remaining capacity for new rollouts.
            buffer_fill_ratio: How full the trajectory buffer is (0-1).
            in_flight: Number of rollouts currently in flight.

        Returns:
            ModeGateState with current mode and reason.
        """
        with self._lock:
            self._in_flight_count = in_flight
            self._last_staleness = staleness
            self._last_capacity = capacity

            previous_mode = self._current_mode

            # State transition logic with priorities
            new_mode, reason = self._compute_next_mode(
                staleness, capacity, buffer_fill_ratio, in_flight
            )

            # Apply hysteresis for stability
            if self._should_apply_hysteresis(previous_mode, new_mode, staleness):
                new_mode = previous_mode
                reason = f"hysteresis_hold({reason})"

            # Record transition
            if new_mode != previous_mode:
                self._transition(previous_mode, new_mode, reason)

            self._current_mode = new_mode
            self._current_reason = reason

            return ModeGateState(
                mode=new_mode,
                reason=reason,
                staleness=staleness,
                capacity=capacity,
                in_flight=in_flight,
                buffer_fill_ratio=buffer_fill_ratio,
            )

    def _compute_next_mode(
        self,
        staleness: float,
        capacity: int,
        buffer_fill_ratio: float,
        in_flight: int,
    ) -> tuple[AsyncMode, str]:
        """Compute next mode based on conditions.

        Priority order:
        1. Capacity exhausted -> THROTTLED
        2. Staleness too high -> SYNC_BARRIER
        3. Buffer approaching full -> THROTTLED
        4. Default -> ASYNC_RUNNING
        """
        # Priority 1: Capacity exhausted
        if capacity <= self.config.capacity_low_watermark:
            return AsyncMode.THROTTLED, "capacity_exhausted"

        # Priority 2: Staleness too high
        if staleness > self.config.staleness_threshold:
            return AsyncMode.SYNC_BARRIER, f"staleness={staleness:.3f}"

        # Priority 3: Buffer approaching full
        if buffer_fill_ratio > self.config.buffer_high_watermark:
            return AsyncMode.THROTTLED, f"buffer_full={buffer_fill_ratio:.2f}"

        # Priority 4: In SYNC_BARRIER, check if we can exit
        if self._current_mode == AsyncMode.SYNC_BARRIER:
            # Stay in barrier if in_flight > 0 or staleness not recovered
            if in_flight > 0:
                return AsyncMode.SYNC_BARRIER, f"draining_in_flight={in_flight}"
            if staleness > self.config.staleness_recovery_threshold:
                return AsyncMode.SYNC_BARRIER, f"staleness_recovering={staleness:.3f}"

        # Default: Normal async operation
        return AsyncMode.ASYNC_RUNNING, "normal"

    def _should_apply_hysteresis(
        self,
        previous_mode: AsyncMode,
        new_mode: AsyncMode,
        staleness: float,
    ) -> bool:
        """Determine if hysteresis should prevent mode change.

        Prevents rapid oscillation between modes.
        """
        # Don't apply hysteresis when entering SYNC_BARRIER (safety critical)
        if new_mode == AsyncMode.SYNC_BARRIER:
            return False

        # Apply hysteresis when exiting SYNC_BARRIER
        if previous_mode == AsyncMode.SYNC_BARRIER and new_mode == AsyncMode.ASYNC_RUNNING:
            # Must have been in SYNC_BARRIER for minimum duration
            elapsed_ms = (time.time() - self._last_state_change) * 1000
            if elapsed_ms < self.config.min_barrier_duration_ms:
                return True

        return False

    def _transition(self, from_mode: AsyncMode, to_mode: AsyncMode, reason: str) -> None:
        """Record a state transition."""
        now = time.time()
        self._transition_history.append((now, from_mode, to_mode, reason))

        # Keep history bounded
        if len(self._transition_history) > 1000:
            self._transition_history = self._transition_history[-500:]

        self._last_state_change = now

        if self.config.log_transitions:
            logger.info(
                f"ModeGate transition: {from_mode.name} -> {to_mode.name} ({reason})"
            )

    def can_submit_rollout(self) -> bool:
        """Check if new rollouts can be submitted.

        Returns:
            True if in ASYNC_RUNNING mode.
        """
        with self._lock:
            return self._current_mode == AsyncMode.ASYNC_RUNNING

    def should_wait_for_barrier(self) -> bool:
        """Check if training should wait for sync barrier.

        Returns:
            True if in SYNC_BARRIER mode.
        """
        with self._lock:
            return self._current_mode == AsyncMode.SYNC_BARRIER

    def is_throttled(self) -> bool:
        """Check if system is in throttled state.

        Returns:
            True if in THROTTLED mode.
        """
        with self._lock:
            return self._current_mode == AsyncMode.THROTTLED

    async def enforce_barrier(
        self,
        wait_for_in_flight: Callable[[], asyncio.Future] | None = None,
        timeout: float = 30.0,
    ) -> bool:
        """
        Block until all in-flight rollouts complete.

        Args:
            wait_for_in_flight: Async function that waits for in-flight to drain.
                If None, uses internal polling.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if barrier completed successfully, False if timeout.
        """
        if self.current_mode != AsyncMode.SYNC_BARRIER:
            return True

        async with self._barrier_lock:
            start_time = time.time()

            if wait_for_in_flight is not None:
                # Use provided wait function
                try:
                    await asyncio.wait_for(wait_for_in_flight(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Barrier timeout after {timeout}s")
                    return False
            else:
                # Poll in_flight count
                while time.time() - start_time < timeout:
                    with self._lock:
                        if self._in_flight_count == 0:
                            break
                    await asyncio.sleep(0.1)
                else:
                    logger.warning(f"Barrier timeout, {self._in_flight_count} still in flight")
                    return False

            # Barrier completed - transition back to ASYNC_RUNNING
            with self._lock:
                if self._current_mode == AsyncMode.SYNC_BARRIER:
                    self._transition(
                        AsyncMode.SYNC_BARRIER,
                        AsyncMode.ASYNC_RUNNING,
                        "barrier_completed",
                    )
                    self._current_mode = AsyncMode.ASYNC_RUNNING
                    self._current_reason = "barrier_completed"

            return True

    def update_threshold(self, new_threshold: float) -> None:
        """Update staleness threshold dynamically.

        Called by PID controller to adjust threshold based on async ratio.

        Args:
            new_threshold: New staleness threshold.
        """
        with self._lock:
            old_threshold = self.config.staleness_threshold
            self.config.staleness_threshold = new_threshold
            # Update recovery threshold proportionally
            self.config.staleness_recovery_threshold = new_threshold * 0.8

            if self.config.log_transitions and abs(new_threshold - old_threshold) > 0.01:
                logger.debug(
                    f"ModeGate threshold updated: {old_threshold:.3f} -> {new_threshold:.3f}"
                )

    def force_mode(self, mode: AsyncMode, reason: str = "forced") -> None:
        """Force a specific mode (for testing/debugging).

        Args:
            mode: Mode to force.
            reason: Reason for forcing.
        """
        with self._lock:
            if self._current_mode != mode:
                self._transition(self._current_mode, mode, f"forced:{reason}")
            self._current_mode = mode
            self._current_reason = f"forced:{reason}"

    def reset(self) -> None:
        """Reset to initial state."""
        with self._lock:
            self._current_mode = AsyncMode.ASYNC_RUNNING
            self._current_reason = "reset"
            self._in_flight_count = 0
            self._last_state_change = time.time()
            self._transition_history.clear()

    def get_diagnostics(self) -> dict[str, float | int | str]:
        """Get diagnostic information.

        Returns:
            Dict with mode gate metrics.
        """
        with self._lock:
            time_in_mode = time.time() - self._last_state_change
            return {
                "mode": self._current_mode.name,
                "reason": self._current_reason,
                "in_flight": self._in_flight_count,
                "staleness_threshold": self.config.staleness_threshold,
                "time_in_mode_s": time_in_mode,
                "transition_count": len(self._transition_history),
            }

    def get_transition_history(
        self, last_n: int | None = None
    ) -> list[dict[str, float | str]]:
        """Get recent state transitions.

        Args:
            last_n: Number of recent transitions (None = all).

        Returns:
            List of transition records.
        """
        with self._lock:
            history = self._transition_history
            if last_n is not None:
                history = history[-last_n:]

            return [
                {
                    "timestamp": ts,
                    "from_mode": from_m.name,
                    "to_mode": to_m.name,
                    "reason": reason,
                }
                for ts, from_m, to_m, reason in history
            ]


class ModeGateIntegration:
    """
    Integration helper for ModeGate with coordinator.

    Provides convenience methods for common coordinator patterns.

    Usage:
        integration = ModeGateIntegration(
            mode_gate=gate,
            staleness_manager=staleness_mgr,
        )

        # In coordinator loop
        state = await integration.check_and_enforce()

        if state.mode == AsyncMode.ASYNC_RUNNING:
            submit_rollouts()
    """

    def __init__(
        self,
        mode_gate: ModeGate,
        staleness_manager: "StalenessManager",
        trajectory_buffer_fill_fn: Callable[[], float] | None = None,
    ) -> None:
        """Initialize integration.

        Args:
            mode_gate: The Mode Gate instance.
            staleness_manager: Staleness manager for metrics.
            trajectory_buffer_fill_fn: Function returning buffer fill ratio (0-1).
        """
        self.mode_gate = mode_gate
        self.staleness_manager = staleness_manager
        self._buffer_fill_fn = trajectory_buffer_fill_fn or (lambda: 0.0)

    async def check_and_enforce(
        self,
        wait_for_in_flight: Callable[[], asyncio.Future] | None = None,
    ) -> ModeGateState:
        """Evaluate mode and enforce barrier if needed.

        Convenience method that combines evaluate() and enforce_barrier().

        Args:
            wait_for_in_flight: Async function to wait for in-flight rollouts.

        Returns:
            Current ModeGateState after any enforcement.
        """
        # Gather metrics
        staleness = self.staleness_manager.current_staleness
        capacity = self.staleness_manager.get_capacity()
        stats = self.staleness_manager.stats
        buffer_fill = self._buffer_fill_fn()

        # Evaluate
        state = self.mode_gate.evaluate(
            staleness=staleness,
            capacity=capacity,
            buffer_fill_ratio=buffer_fill,
            in_flight=stats.total_in_flight,
        )

        # Enforce barrier if needed
        if state.mode == AsyncMode.SYNC_BARRIER:
            await self.mode_gate.enforce_barrier(wait_for_in_flight)
            # Re-evaluate after barrier
            state = self.mode_gate.evaluate(
                staleness=staleness,
                capacity=capacity,
                buffer_fill_ratio=buffer_fill,
                in_flight=0,  # After barrier, in_flight should be 0
            )

        return state

    def get_rollout_slots(self) -> int:
        """Get number of rollout slots available.

        Returns 0 if not in ASYNC_RUNNING mode.

        Returns:
            Number of rollouts that can be submitted.
        """
        if not self.mode_gate.can_submit_rollout():
            return 0
        return self.staleness_manager.get_capacity()
