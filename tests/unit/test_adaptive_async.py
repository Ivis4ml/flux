"""
Tests for AdaptiveAsyncController and AdaptiveAsyncScheduler.
"""

import pytest
import torch

from flux.core.config import AdaptiveAsyncConfig
from flux.core.types import AsyncDecision, StalenessMetrics, TrainingPhase
from flux.controller.adaptive_async import (
    AdaptiveAsyncController,
    AdaptiveAsyncScheduler,
    ControllerRecord,
    PIDState,
)
from flux.controller.staleness import StalenessManager


class TestPIDState:
    """Tests for PIDState dataclass."""

    def test_default_values(self):
        """Test default PID state values."""
        state = PIDState()
        assert state.error == 0.0
        assert state.integral == 0.0
        assert state.derivative == 0.0
        assert state.previous_error == 0.0
        assert state.output == 0.0

    def test_integral_bounds(self):
        """Test integral bounds."""
        state = PIDState()
        assert state.integral_min == -10.0
        assert state.integral_max == 10.0


class TestAsyncDecision:
    """Tests for AsyncDecision dataclass."""

    def test_creation(self):
        """Test decision creation."""
        decision = AsyncDecision(
            async_ratio=0.5,
            should_sync=False,
            staleness_estimate=0.1,
        )
        assert decision.async_ratio == 0.5
        assert not decision.should_sync

    def test_async_ratio_clamping(self):
        """Test that async_ratio is clamped."""
        decision = AsyncDecision(async_ratio=1.5, should_sync=False)
        assert decision.async_ratio == 1.0

        decision = AsyncDecision(async_ratio=-0.5, should_sync=False)
        assert decision.async_ratio == 0.0


class TestAdaptiveAsyncController:
    """Tests for AdaptiveAsyncController."""

    def test_creation_defaults(self):
        """Test controller creation with defaults."""
        controller = AdaptiveAsyncController()
        assert controller.target_staleness == 0.15
        # Initial ratio should be midpoint of [0.1, 0.9] = 0.5
        assert controller.async_ratio == 0.5

    def test_creation_custom_config(self):
        """Test controller with custom config."""
        config = AdaptiveAsyncConfig(
            target_staleness=0.2,
            min_async_ratio=0.2,
            max_async_ratio=0.8,
        )
        controller = AdaptiveAsyncController(config=config)
        assert controller.target_staleness == 0.2
        # Initial should be midpoint of [0.2, 0.8] = 0.5
        assert controller.async_ratio == 0.5

    def test_creation_initial_ratio(self):
        """Test controller with explicit initial ratio."""
        controller = AdaptiveAsyncController(initial_async_ratio=0.7)
        assert controller.async_ratio == 0.7

    def test_step_returns_decision(self):
        """Test that step returns AsyncDecision."""
        controller = AdaptiveAsyncController()
        decision = controller.step(current_staleness=0.1)

        assert isinstance(decision, AsyncDecision)
        assert 0.0 <= decision.async_ratio <= 1.0

    def test_step_with_staleness_metrics(self):
        """Test step with StalenessMetrics object."""
        controller = AdaptiveAsyncController()
        metrics = StalenessMetrics(combined_staleness=0.1)
        decision = controller.step(staleness_metrics=metrics)

        assert isinstance(decision, AsyncDecision)

    def test_step_increases_ratio_when_staleness_low(self):
        """Test that low staleness increases async ratio."""
        config = AdaptiveAsyncConfig(
            target_staleness=0.15,
            kp=0.5,  # High gain for faster response
            ki=0.0,
            kd=0.0,
            ema_alpha=1.0,  # No smoothing
        )
        controller = AdaptiveAsyncController(
            config=config,
            initial_async_ratio=0.5,
        )

        # Low staleness (below target) should increase ratio
        initial_ratio = controller.async_ratio
        for _ in range(5):
            controller.step(current_staleness=0.05)  # Well below target

        assert controller.async_ratio > initial_ratio

    def test_step_decreases_ratio_when_staleness_high(self):
        """Test that high staleness decreases async ratio."""
        config = AdaptiveAsyncConfig(
            target_staleness=0.15,
            kp=0.5,  # High gain for faster response
            ki=0.0,
            kd=0.0,
            ema_alpha=1.0,  # No smoothing
        )
        controller = AdaptiveAsyncController(
            config=config,
            initial_async_ratio=0.5,
        )

        # High staleness (above target) should decrease ratio
        initial_ratio = controller.async_ratio
        for _ in range(5):
            controller.step(current_staleness=0.3)  # Well above target

        assert controller.async_ratio < initial_ratio

    def test_ratio_clamped_to_bounds(self):
        """Test that ratio is clamped to configured bounds."""
        config = AdaptiveAsyncConfig(
            min_async_ratio=0.2,
            max_async_ratio=0.8,
            kp=1.0,  # Very high gain
            ki=0.0,
            kd=0.0,
        )
        controller = AdaptiveAsyncController(config=config)

        # Push toward maximum
        for _ in range(20):
            controller.step(current_staleness=0.0)  # Very low staleness

        assert controller.async_ratio <= 0.8

        # Push toward minimum
        for _ in range(50):
            controller.step(current_staleness=1.0)  # Very high staleness

        assert controller.async_ratio >= 0.2

    def test_should_sync_high_staleness(self):
        """Test should_sync when staleness exceeds threshold."""
        config = AdaptiveAsyncConfig(
            target_staleness=0.15,
            tolerance=0.05,
        )
        controller = AdaptiveAsyncController(config=config)

        # Below threshold
        decision = controller.step(current_staleness=0.15)
        assert not decision.should_sync

        # Above threshold (0.15 + 0.05 = 0.20)
        decision = controller.step(current_staleness=0.25)
        assert decision.should_sync

    def test_should_sync_max_steps(self):
        """Test should_sync after max steps without sync."""
        config = AdaptiveAsyncConfig(
            max_steps_without_sync=5,
            target_staleness=0.5,  # High target
            tolerance=0.5,  # Maximum allowed tolerance
        )
        controller = AdaptiveAsyncController(config=config)

        # Steps 1-4 should not sync (staleness 0.1 is well below 0.5+0.5=1.0 threshold)
        for i in range(4):
            decision = controller.step(current_staleness=0.1)
            # First 4 steps should not sync
            assert not decision.should_sync, f"Step {i+1} should not sync"

        # Step 5 should sync (reaches max_steps_without_sync)
        decision = controller.step(current_staleness=0.1)
        assert decision.should_sync, "Step 5 should trigger sync"

    def test_record_sync_resets_counter(self):
        """Test that record_sync resets step counter."""
        config = AdaptiveAsyncConfig(max_steps_without_sync=5)
        controller = AdaptiveAsyncController(config=config)

        for _ in range(3):
            controller.step(current_staleness=0.1)

        controller.record_sync()

        # Should need 5 more steps before sync
        for _ in range(4):
            decision = controller.step(current_staleness=0.1)
            assert not decision.should_sync

    def test_training_phase_adjustment(self):
        """Test async ratio adjustment based on training phase."""
        config = AdaptiveAsyncConfig(
            min_async_ratio=0.1,
            max_async_ratio=0.9,
            kp=0.0,  # No PID adjustment to isolate phase effect
            ki=0.0,
            kd=0.0,
        )

        # Test MID phase (normal operation)
        controller = AdaptiveAsyncController(
            config=config,
            initial_async_ratio=0.5,
        )
        controller.set_training_phase(TrainingPhase.MID)
        decision_mid = controller.step(current_staleness=0.15)
        ratio_mid = decision_mid.async_ratio

        # Test WARMUP phase (should be more conservative)
        controller_warmup = AdaptiveAsyncController(
            config=config,
            initial_async_ratio=0.5,
        )
        controller_warmup.set_training_phase(TrainingPhase.WARMUP)
        decision_warmup = controller_warmup.step(current_staleness=0.15)
        ratio_warmup = decision_warmup.async_ratio

        # Warmup should be more conservative (lower ratio)
        assert ratio_warmup < ratio_mid, f"warmup={ratio_warmup} should be < mid={ratio_mid}"

        # Test LATE phase (should allow higher ratio)
        controller_late = AdaptiveAsyncController(
            config=config,
            initial_async_ratio=0.5,
        )
        controller_late.set_training_phase(TrainingPhase.LATE)
        decision_late = controller_late.step(current_staleness=0.15)
        ratio_late = decision_late.async_ratio

        # Late should allow higher ratio
        assert ratio_late >= ratio_mid, f"late={ratio_late} should be >= mid={ratio_mid}"

    def test_reset(self):
        """Test controller reset."""
        controller = AdaptiveAsyncController()

        # Make some changes
        for _ in range(10):
            controller.step(current_staleness=0.1)

        controller.reset()

        # Should be back to initial state
        assert controller.async_ratio == 0.5
        diagnostics = controller.get_diagnostics()
        assert diagnostics["pid_error"] == 0.0
        assert diagnostics["pid_integral"] == 0.0

    def test_get_diagnostics(self):
        """Test getting diagnostic info."""
        controller = AdaptiveAsyncController()
        controller.step(current_staleness=0.1)

        diagnostics = controller.get_diagnostics()

        assert "async_ratio" in diagnostics
        assert "ema_staleness" in diagnostics
        assert "target_staleness" in diagnostics
        assert "pid_error" in diagnostics
        assert "pid_integral" in diagnostics
        assert "pid_derivative" in diagnostics
        assert "pid_output" in diagnostics

    def test_get_history(self):
        """Test getting controller history."""
        controller = AdaptiveAsyncController()

        for i in range(5):
            controller.step(current_staleness=0.1 * i)

        history = controller.get_history()
        assert len(history) == 5
        assert all(isinstance(r, ControllerRecord) for r in history)

        # Test last_n
        recent = controller.get_history(last_n=3)
        assert len(recent) == 3

    def test_pid_convergence(self):
        """Test PID controller convergence to target staleness.

        This is the key verification test for Phase 3.
        The controller should converge to maintain target staleness.
        """
        config = AdaptiveAsyncConfig(
            target_staleness=0.15,
            tolerance=0.03,
            kp=0.2,
            ki=0.02,
            kd=0.05,
            min_async_ratio=0.1,
            max_async_ratio=0.9,
            ema_alpha=0.3,
        )
        controller = AdaptiveAsyncController(config=config)

        # Simulate staleness that responds to async ratio
        # Higher async ratio -> higher staleness
        def simulate_staleness(async_ratio: float) -> float:
            # Simple model: staleness increases with async ratio
            base_staleness = async_ratio * 0.3
            noise = 0.01  # Small noise
            return min(1.0, max(0.0, base_staleness + noise))

        # Run for many iterations
        for _ in range(200):
            staleness = simulate_staleness(controller.async_ratio)
            controller.step(current_staleness=staleness)

        # Analyze convergence
        convergence = controller.analyze_convergence(window_size=50)

        # Check that controller has stabilized
        # The ratio should be around 0.5 (which gives staleness ~0.15)
        assert convergence["ratio_std"] < 0.15, "Ratio should be stable"
        assert abs(convergence["error_mean"]) < 0.1, "Error should be small"

    def test_pid_integral_windup(self):
        """Test that integral windup is prevented."""
        config = AdaptiveAsyncConfig(
            kp=0.0,
            ki=1.0,  # High integral gain
            kd=0.0,
        )
        controller = AdaptiveAsyncController(config=config)

        # Push integral very high
        for _ in range(100):
            controller.step(current_staleness=0.0)  # Large positive error

        diagnostics = controller.get_diagnostics()
        # Integral should be bounded
        assert diagnostics["pid_integral"] <= 10.0

    def test_with_staleness_manager(self):
        """Test controller with staleness manager integration."""
        staleness_manager = StalenessManager(batch_size=32)
        controller = AdaptiveAsyncController(
            staleness_manager=staleness_manager,
        )

        # Simulate some rollouts
        staleness_manager.on_rollout_enqueued(20)

        decision = controller.step(current_staleness=0.1)

        # Should have capacity info from manager
        assert decision.capacity_remaining >= 0


class TestAdaptiveAsyncScheduler:
    """Tests for AdaptiveAsyncScheduler."""

    def test_creation(self):
        """Test scheduler creation."""
        scheduler = AdaptiveAsyncScheduler()
        assert scheduler.batch_size == 32
        assert scheduler.async_ratio == 0.5

    def test_creation_custom_config(self):
        """Test scheduler with custom config."""
        config = AdaptiveAsyncConfig(target_staleness=0.2)
        scheduler = AdaptiveAsyncScheduler(
            config=config,
            batch_size=64,
        )
        assert scheduler.batch_size == 64
        assert scheduler.controller.target_staleness == 0.2

    def test_can_submit(self):
        """Test can_submit check."""
        scheduler = AdaptiveAsyncScheduler(batch_size=32)
        assert scheduler.can_submit()  # Initially should have capacity

    def test_get_capacity(self):
        """Test get_capacity."""
        scheduler = AdaptiveAsyncScheduler(batch_size=32)
        capacity = scheduler.get_capacity()
        assert capacity > 0

    def test_step(self):
        """Test scheduler step."""
        scheduler = AdaptiveAsyncScheduler()
        decision = scheduler.step(current_staleness=0.1)
        assert isinstance(decision, AsyncDecision)

    def test_step_with_kwargs(self):
        """Test scheduler step with staleness kwargs."""
        scheduler = AdaptiveAsyncScheduler()
        decision = scheduler.step(kl_divergence=0.05, version_gap=1.0)
        assert isinstance(decision, AsyncDecision)

    def test_rollout_tracking(self):
        """Test rollout tracking methods."""
        scheduler = AdaptiveAsyncScheduler(batch_size=32)

        scheduler.on_rollout_enqueued(10)
        scheduler.on_rollout_submitted(5)
        scheduler.on_rollout_completed(3)
        scheduler.on_rollout_failed(1)

        diagnostics = scheduler.get_diagnostics()
        assert diagnostics["rollouts_enqueued"] == 5  # 10 - 5
        assert diagnostics["rollouts_running"] == 1  # 5 - 3 - 1
        assert diagnostics["rollouts_accepted"] == 3
        assert diagnostics["rollouts_rejected"] == 1

    def test_on_batch_consumed(self):
        """Test batch consumption tracking."""
        scheduler = AdaptiveAsyncScheduler(batch_size=32)

        scheduler.on_rollout_enqueued(50)
        scheduler.on_rollout_submitted(50)
        scheduler.on_rollout_completed(50)

        scheduler.on_batch_consumed()

        diagnostics = scheduler.get_diagnostics()
        assert diagnostics["rollouts_accepted"] == 18  # 50 - 32

    def test_set_training_phase(self):
        """Test setting training phase."""
        scheduler = AdaptiveAsyncScheduler()
        scheduler.set_training_phase(TrainingPhase.LATE)

        # Phase should affect ratio
        decision = scheduler.step(current_staleness=0.15)
        assert isinstance(decision, AsyncDecision)

    def test_reset(self):
        """Test scheduler reset."""
        scheduler = AdaptiveAsyncScheduler()

        scheduler.on_rollout_enqueued(20)
        scheduler.step(current_staleness=0.1)

        scheduler.reset()

        diagnostics = scheduler.get_diagnostics()
        assert diagnostics["rollouts_enqueued"] == 0

    def test_get_diagnostics(self):
        """Test getting scheduler diagnostics."""
        scheduler = AdaptiveAsyncScheduler()
        scheduler.step(current_staleness=0.1)

        diagnostics = scheduler.get_diagnostics()

        # Should include controller diagnostics
        assert "async_ratio" in diagnostics
        assert "pid_error" in diagnostics

        # Should include rollout stats
        assert "rollouts_enqueued" in diagnostics
        assert "rollouts_running" in diagnostics
        assert "capacity" in diagnostics

    def test_version_provider(self):
        """Test with custom version provider."""
        version = [0]

        def get_version():
            return version[0]

        scheduler = AdaptiveAsyncScheduler(
            version_provider=get_version,
            batch_size=32,
        )

        # Initial capacity
        cap1 = scheduler.get_capacity()

        # Increase version
        version[0] = 5
        cap2 = scheduler.get_capacity()

        # Higher version should allow more capacity
        assert cap2 > cap1


class TestControllerRecord:
    """Tests for ControllerRecord dataclass."""

    def test_creation(self):
        """Test record creation."""
        from datetime import datetime

        decision = AsyncDecision(async_ratio=0.5, should_sync=False)
        record = ControllerRecord(
            timestamp=datetime.now(),
            staleness=0.1,
            async_ratio=0.5,
            error=0.05,
            integral=0.1,
            derivative=-0.01,
            output=0.02,
            decision=decision,
        )
        assert record.staleness == 0.1
        assert record.async_ratio == 0.5
        assert record.error == 0.05


class TestPIDControllerBehavior:
    """Detailed tests for PID controller behavior."""

    def test_proportional_response(self):
        """Test proportional term response."""
        config = AdaptiveAsyncConfig(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            ema_alpha=1.0,
        )
        controller = AdaptiveAsyncController(
            config=config,
            initial_async_ratio=0.5,
        )

        # Small error
        controller.step(current_staleness=0.14)  # error = 0.15 - 0.14 = 0.01
        diag1 = controller.get_diagnostics()

        controller.reset()

        # Large error
        controller.step(current_staleness=0.05)  # error = 0.15 - 0.05 = 0.10
        diag2 = controller.get_diagnostics()

        # Larger error should produce larger output
        assert abs(diag2["pid_output"]) > abs(diag1["pid_output"])

    def test_integral_accumulation(self):
        """Test integral term accumulation."""
        config = AdaptiveAsyncConfig(
            kp=0.0,
            ki=1.0,
            kd=0.0,
            ema_alpha=1.0,
        )
        controller = AdaptiveAsyncController(config=config)

        # Constant positive error
        for _ in range(5):
            controller.step(current_staleness=0.10)  # error = 0.05

        diagnostics = controller.get_diagnostics()
        # Integral should have accumulated
        assert diagnostics["pid_integral"] > 0

    def test_derivative_response(self):
        """Test derivative term response to rapid changes."""
        config = AdaptiveAsyncConfig(
            kp=0.0,
            ki=0.0,
            kd=1.0,
            ema_alpha=1.0,
        )
        controller = AdaptiveAsyncController(config=config)

        # First step establishes previous error
        controller.step(current_staleness=0.15)

        # Second step with same staleness (derivative should be 0)
        controller.step(current_staleness=0.15)
        diag1 = controller.get_diagnostics()
        assert diag1["pid_derivative"] == 0.0

        # Third step with different staleness
        controller.step(current_staleness=0.10)
        diag2 = controller.get_diagnostics()
        assert diag2["pid_derivative"] != 0.0

    def test_system_stability(self):
        """Test that PID doesn't cause oscillations."""
        config = AdaptiveAsyncConfig(
            target_staleness=0.15,
            kp=0.1,
            ki=0.01,
            kd=0.05,
            ema_alpha=0.2,
        )
        controller = AdaptiveAsyncController(config=config)

        # Run with constant staleness
        ratios = []
        for _ in range(100):
            controller.step(current_staleness=0.15)  # Exact target
            ratios.append(controller.async_ratio)

        # Check for stability (low variance in later iterations)
        later_ratios = ratios[-50:]
        variance = sum((r - sum(later_ratios) / len(later_ratios)) ** 2 for r in later_ratios) / len(later_ratios)

        assert variance < 0.01, "Controller should be stable at target"
