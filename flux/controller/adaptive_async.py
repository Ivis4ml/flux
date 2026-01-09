"""
Adaptive async controller for Flux.

Implements a PID controller that dynamically adjusts the sync/async ratio
based on measured staleness, achieving both synchronous stability and
asynchronous efficiency.

Key innovation: Uses control theory (PID) to maintain target staleness,
automatically finding the optimal balance between sync (stable but slow)
and async (fast but potentially unstable).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from flux.core.config import AdaptiveAsyncConfig
from flux.core.types import AsyncDecision, StalenessMetrics, TrainingPhase
from flux.controller.staleness import StalenessManager


@dataclass
class PIDState:
    """Internal state for PID controller."""

    # Error terms
    error: float = 0.0           # Current error (target - actual)
    integral: float = 0.0        # Accumulated error
    derivative: float = 0.0      # Rate of error change
    previous_error: float = 0.0  # Previous error for derivative

    # Output
    output: float = 0.0          # Controller output

    # Anti-windup
    integral_min: float = -10.0
    integral_max: float = 10.0


@dataclass
class ControllerRecord:
    """Record of controller state for analysis."""

    timestamp: datetime
    staleness: float
    async_ratio: float
    error: float
    integral: float
    derivative: float
    output: float
    decision: AsyncDecision


class AdaptiveAsyncController:
    """PID controller for adaptive sync/async ratio.

    The controller maintains a target staleness level by adjusting the
    async ratio (fraction of async operations vs sync barriers).

    When staleness is below target:
    - Increase async ratio (more overlap, higher throughput)

    When staleness is above target:
    - Decrease async ratio (more sync, better stability)

    PID Control:
        output = kp * error + ki * integral + kd * derivative
        async_ratio += output

    Where:
        - error = target_staleness - current_staleness
        - integral = accumulated error over time
        - derivative = rate of error change

    Example:
        controller = AdaptiveAsyncController(
            config=AdaptiveAsyncConfig(
                target_staleness=0.15,
                kp=0.1,
                ki=0.01,
                kd=0.05,
            ),
            staleness_manager=staleness_mgr,
        )

        # Each training step
        decision = controller.step(current_staleness=0.12)
        if decision.should_sync:
            trigger_sync_barrier()

        # Use decision.async_ratio for scheduling
    """

    def __init__(
        self,
        config: AdaptiveAsyncConfig | None = None,
        staleness_manager: StalenessManager | None = None,
        initial_async_ratio: float | None = None,
    ) -> None:
        """Initialize the adaptive async controller.

        Args:
            config: Adaptive async configuration.
            staleness_manager: Optional staleness manager for capacity queries.
            initial_async_ratio: Starting async ratio (defaults to midpoint).
        """
        self.config = config or AdaptiveAsyncConfig()
        self.staleness_manager = staleness_manager

        # PID state
        self._pid = PIDState()

        # Current async ratio
        if initial_async_ratio is not None:
            self._async_ratio = initial_async_ratio
        else:
            # Start at midpoint
            self._async_ratio = (
                self.config.min_async_ratio + self.config.max_async_ratio
            ) / 2

        # Thread safety
        self._lock = threading.Lock()

        # History for analysis
        self._history: deque[ControllerRecord] = deque(maxlen=1000)

        # Steps since last sync
        self._steps_since_sync = 0

        # Training phase for adaptive behavior
        self._training_phase = TrainingPhase.WARMUP

        # EMA of staleness for smooth control
        self._ema_staleness = self.config.target_staleness

    @property
    def async_ratio(self) -> float:
        """Current async ratio."""
        with self._lock:
            return self._async_ratio

    @property
    def target_staleness(self) -> float:
        """Target staleness level."""
        return self.config.target_staleness

    def step(
        self,
        staleness_metrics: StalenessMetrics | None = None,
        current_staleness: float | None = None,
    ) -> AsyncDecision:
        """Perform one controller step and get decision.

        Args:
            staleness_metrics: Full staleness metrics (preferred).
            current_staleness: Scalar staleness value (alternative).

        Returns:
            AsyncDecision with current async_ratio and sync recommendation.
        """
        # Get staleness value
        if staleness_metrics is not None:
            staleness = staleness_metrics.combined_staleness
        elif current_staleness is not None:
            staleness = current_staleness
        elif self.staleness_manager is not None:
            staleness = self.staleness_manager.current_staleness
        else:
            staleness = 0.0

        with self._lock:
            # Update EMA
            alpha = self.config.ema_alpha
            self._ema_staleness = alpha * staleness + (1 - alpha) * self._ema_staleness

            # Compute PID terms
            error = self.config.target_staleness - self._ema_staleness

            # Integral with anti-windup
            self._pid.integral += error
            self._pid.integral = max(
                self._pid.integral_min,
                min(self._pid.integral_max, self._pid.integral),
            )

            # Derivative
            self._pid.derivative = error - self._pid.previous_error
            self._pid.previous_error = error

            # PID output
            output = (
                self.config.kp * error
                + self.config.ki * self._pid.integral
                + self.config.kd * self._pid.derivative
            )

            self._pid.error = error
            self._pid.output = output

            # Update async ratio
            self._async_ratio += output

            # Clamp to valid range
            self._async_ratio = max(
                self.config.min_async_ratio,
                min(self.config.max_async_ratio, self._async_ratio),
            )

            # Adjust for training phase
            adjusted_ratio = self._adjust_for_phase(self._async_ratio)

            # Increment step counter before checking (so sync triggers after N steps)
            self._steps_since_sync += 1

            # Determine if sync is needed
            should_sync = self._should_sync(staleness)

            # Get capacity if staleness manager available
            capacity = 0
            if self.staleness_manager is not None:
                capacity = self.staleness_manager.get_capacity()

            decision = AsyncDecision(
                async_ratio=adjusted_ratio,
                should_sync=should_sync,
                staleness_estimate=staleness,
                capacity_remaining=capacity,
            )

            # Record for analysis
            self._history.append(
                ControllerRecord(
                    timestamp=datetime.now(),
                    staleness=staleness,
                    async_ratio=adjusted_ratio,
                    error=error,
                    integral=self._pid.integral,
                    derivative=self._pid.derivative,
                    output=output,
                    decision=decision,
                )
            )

            return decision

    def _should_sync(self, current_staleness: float) -> bool:
        """Determine if a sync barrier should be triggered.

        Args:
            current_staleness: Current staleness measurement.

        Returns:
            Whether to trigger sync.
        """
        # Check staleness threshold
        threshold = self.config.target_staleness + self.config.tolerance
        if current_staleness > threshold:
            self._steps_since_sync = 0
            if self.staleness_manager is not None:
                self.staleness_manager.record_sync()
            return True

        # Check max steps without sync
        if self._steps_since_sync >= self.config.max_steps_without_sync:
            self._steps_since_sync = 0
            if self.staleness_manager is not None:
                self.staleness_manager.record_sync()
            return True

        # Check from staleness manager if available
        if self.staleness_manager is not None:
            if self.staleness_manager.should_sync():
                self._steps_since_sync = 0
                self.staleness_manager.record_sync()
                return True

        return False

    def _adjust_for_phase(self, async_ratio: float) -> float:
        """Adjust async ratio based on training phase.

        Early training should be more synchronous for stability.
        Late training can be more async for efficiency.

        Args:
            async_ratio: Base async ratio from PID.

        Returns:
            Adjusted async ratio.
        """
        if self._training_phase == TrainingPhase.WARMUP:
            # Very conservative during warmup
            return max(self.config.min_async_ratio, async_ratio * 0.5)
        elif self._training_phase == TrainingPhase.EARLY:
            # Slightly conservative
            return max(self.config.min_async_ratio, async_ratio * 0.7)
        elif self._training_phase == TrainingPhase.MID:
            # Normal operation
            return async_ratio
        else:  # LATE
            # Can be more aggressive
            return min(
                self.config.max_async_ratio,
                async_ratio * 1.1,
            )

    def set_training_phase(self, phase: TrainingPhase) -> None:
        """Update the training phase.

        Args:
            phase: New training phase.
        """
        with self._lock:
            self._training_phase = phase

    def record_sync(self) -> None:
        """Record that a sync barrier was executed."""
        with self._lock:
            self._steps_since_sync = 0
            # Call staleness_manager inside lock to ensure thread safety
            if self.staleness_manager is not None:
                self.staleness_manager.record_sync()

    def reset(self) -> None:
        """Reset controller state."""
        with self._lock:
            self._pid = PIDState()
            self._async_ratio = (
                self.config.min_async_ratio + self.config.max_async_ratio
            ) / 2
            self._steps_since_sync = 0
            self._ema_staleness = self.config.target_staleness
            self._history.clear()

    def get_diagnostics(self) -> dict[str, float]:
        """Get diagnostic information about controller state.

        Returns:
            Dict with controller metrics.
        """
        with self._lock:
            return {
                "async_ratio": self._async_ratio,
                "ema_staleness": self._ema_staleness,
                "target_staleness": self.config.target_staleness,
                "pid_error": self._pid.error,
                "pid_integral": self._pid.integral,
                "pid_derivative": self._pid.derivative,
                "pid_output": self._pid.output,
                "steps_since_sync": self._steps_since_sync,
                "training_phase": self._training_phase.value,
            }

    def get_history(self, last_n: int | None = None) -> list[ControllerRecord]:
        """Get controller history.

        Args:
            last_n: Number of recent records (None = all).

        Returns:
            List of controller records.
        """
        with self._lock:
            if last_n is None:
                return list(self._history)
            return list(self._history)[-last_n:]

    def analyze_convergence(
        self, window_size: int = 50
    ) -> dict[str, float]:
        """Analyze controller convergence.

        Args:
            window_size: Number of steps for analysis window.

        Returns:
            Dict with convergence metrics.
        """
        with self._lock:
            if len(self._history) < window_size:
                return {
                    "converged": False,
                    "error_mean": 0.0,
                    "error_std": 0.0,
                    "ratio_mean": 0.0,
                    "ratio_std": 0.0,
                }

            recent = list(self._history)[-window_size:]
            errors = [r.error for r in recent]
            ratios = [r.async_ratio for r in recent]

            error_mean = sum(errors) / len(errors)
            error_std = (
                sum((e - error_mean) ** 2 for e in errors) / len(errors)
            ) ** 0.5

            ratio_mean = sum(ratios) / len(ratios)
            ratio_std = (
                sum((r - ratio_mean) ** 2 for r in ratios) / len(ratios)
            ) ** 0.5

            # Consider converged if error is small and ratio is stable
            converged = (
                abs(error_mean) < self.config.tolerance
                and error_std < self.config.tolerance
                and ratio_std < 0.1
            )

            return {
                "converged": converged,
                "error_mean": error_mean,
                "error_std": error_std,
                "ratio_mean": ratio_mean,
                "ratio_std": ratio_std,
            }


class AdaptiveAsyncScheduler:
    """High-level scheduler using adaptive async control.

    Combines AdaptiveAsyncController with StalenessManager to provide
    a complete scheduling interface for the training loop.

    Example:
        scheduler = AdaptiveAsyncScheduler(
            config=config.adaptive_async,
            batch_size=config.batch_size,
        )

        # In training loop
        while not done:
            # Check if we can submit more rollouts
            if scheduler.can_submit():
                submit_rollout()
                scheduler.on_rollout_submitted()

            # When batch is ready
            staleness = compute_staleness(batch)
            decision = scheduler.step(staleness)

            if decision.should_sync:
                sync_weights()

            train_step(batch)
            scheduler.on_batch_consumed()
    """

    def __init__(
        self,
        config: AdaptiveAsyncConfig | None = None,
        batch_size: int = 32,
        version_provider: Callable[[], int] | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            config: Adaptive async configuration.
            batch_size: Training batch size.
            version_provider: Function returning current policy version.
        """
        self.config = config or AdaptiveAsyncConfig()
        self.batch_size = batch_size

        # Create staleness manager
        self.staleness_manager = StalenessManager(
            config=self.config,
            version_provider=version_provider,
            batch_size=batch_size,
        )

        # Create controller
        self.controller = AdaptiveAsyncController(
            config=self.config,
            staleness_manager=self.staleness_manager,
        )

    @property
    def async_ratio(self) -> float:
        """Current async ratio."""
        return self.controller.async_ratio

    def can_submit(self) -> bool:
        """Check if more rollouts can be submitted.

        Returns:
            True if capacity is available.
        """
        return self.staleness_manager.get_capacity() > 0

    def get_capacity(self) -> int:
        """Get remaining capacity for rollouts.

        Returns:
            Number of rollouts that can be submitted.
        """
        return self.staleness_manager.get_capacity()

    def step(
        self,
        staleness_metrics: StalenessMetrics | None = None,
        current_staleness: float | None = None,
        **kwargs,
    ) -> AsyncDecision:
        """Perform one scheduler step.

        Args:
            staleness_metrics: Staleness from current batch.
            current_staleness: Scalar staleness value (alternative to metrics).
            **kwargs: Additional args for staleness computation (kl_divergence, etc).

        Returns:
            AsyncDecision for this step.
        """
        # If scalar staleness provided, use controller directly
        if current_staleness is not None:
            return self.controller.step(current_staleness=current_staleness)

        # Update staleness manager if metrics provided
        if staleness_metrics is None and kwargs:
            staleness_metrics = self.staleness_manager.compute_staleness(**kwargs)

        # Get controller decision
        return self.controller.step(staleness_metrics=staleness_metrics)

    def on_rollout_enqueued(self, count: int = 1) -> None:
        """Record rollouts added to queue."""
        self.staleness_manager.on_rollout_enqueued(count)

    def on_rollout_submitted(self, count: int = 1) -> None:
        """Record rollouts submitted for generation."""
        self.staleness_manager.on_rollout_submitted(count)

    def on_rollout_completed(self, count: int = 1) -> None:
        """Record rollouts completed successfully."""
        self.staleness_manager.on_rollout_accepted(count)

    def on_rollout_failed(self, count: int = 1) -> None:
        """Record rollouts that failed."""
        self.staleness_manager.on_rollout_rejected(count)

    def on_batch_consumed(self, count: int | None = None) -> None:
        """Record training batch consumption."""
        self.staleness_manager.on_batch_consumed(count)

    def set_training_phase(self, phase: TrainingPhase) -> None:
        """Update training phase."""
        self.controller.set_training_phase(phase)

    def reset(self) -> None:
        """Reset scheduler state."""
        self.controller.reset()
        self.staleness_manager.reset_stats()

    def get_diagnostics(self) -> dict[str, float | int]:
        """Get scheduler diagnostics.

        Returns:
            Dict with scheduler metrics.
        """
        diagnostics = self.controller.get_diagnostics()
        stats = self.staleness_manager.stats

        diagnostics.update({
            "rollouts_enqueued": stats.enqueued,
            "rollouts_running": stats.running,
            "rollouts_accepted": stats.accepted,
            "rollouts_rejected": stats.rejected,
            "capacity": self.staleness_manager.get_capacity(),
        })

        return diagnostics
