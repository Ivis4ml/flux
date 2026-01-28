"""
Tests for ModeGate state machine.
"""

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

import pytest

from flux.controller.mode_gate import (
    AsyncMode,
    ModeGate,
    ModeGateConfig,
    ModeGateIntegration,
    ModeGateState,
)


class TestAsyncMode:
    """Tests for AsyncMode enum."""

    def test_enum_values(self):
        """Test AsyncMode has expected values."""
        assert AsyncMode.SYNC_BARRIER is not None
        assert AsyncMode.ASYNC_RUNNING is not None
        assert AsyncMode.THROTTLED is not None

    def test_enum_names(self):
        """Test AsyncMode names."""
        assert AsyncMode.SYNC_BARRIER.name == "SYNC_BARRIER"
        assert AsyncMode.ASYNC_RUNNING.name == "ASYNC_RUNNING"
        assert AsyncMode.THROTTLED.name == "THROTTLED"


class TestModeGateState:
    """Tests for ModeGateState dataclass."""

    def test_creation(self):
        """Test ModeGateState creation."""
        state = ModeGateState(
            mode=AsyncMode.ASYNC_RUNNING,
            reason="normal",
            staleness=0.1,
            capacity=100,
            in_flight=5,
        )

        assert state.mode == AsyncMode.ASYNC_RUNNING
        assert state.reason == "normal"
        assert state.staleness == 0.1
        assert state.capacity == 100
        assert state.in_flight == 5
        assert state.buffer_fill_ratio == 0.0

    def test_str_representation(self):
        """Test string representation."""
        state = ModeGateState(
            mode=AsyncMode.ASYNC_RUNNING,
            reason="normal",
            staleness=0.1,
            capacity=100,
            in_flight=5,
        )

        s = str(state)
        assert "ASYNC_RUNNING" in s
        assert "normal" in s

    def test_with_buffer_fill(self):
        """Test with buffer_fill_ratio."""
        state = ModeGateState(
            mode=AsyncMode.THROTTLED,
            reason="buffer_full",
            staleness=0.1,
            capacity=100,
            in_flight=5,
            buffer_fill_ratio=0.95,
        )

        assert state.buffer_fill_ratio == 0.95


class TestModeGateConfig:
    """Tests for ModeGateConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModeGateConfig()

        assert config.staleness_threshold == 0.3
        assert config.capacity_low_watermark == 0
        assert config.buffer_high_watermark == 0.9
        assert config.min_barrier_duration_ms == 100.0
        assert config.log_transitions is True

    def test_recovery_threshold_auto(self):
        """Test that recovery threshold is auto-computed."""
        config = ModeGateConfig(staleness_threshold=0.5)

        # Recovery threshold should be 80% of staleness threshold
        assert config.staleness_recovery_threshold == 0.4

    def test_custom_recovery_threshold(self):
        """Test custom recovery threshold."""
        config = ModeGateConfig(
            staleness_threshold=0.5,
            staleness_recovery_threshold=0.3,
        )

        assert config.staleness_recovery_threshold == 0.3


class TestModeGate:
    """Tests for ModeGate class."""

    def test_creation(self):
        """Test ModeGate creation."""
        gate = ModeGate()

        assert gate.current_mode == AsyncMode.ASYNC_RUNNING
        assert gate.can_submit_rollout() is True
        assert gate.should_wait_for_barrier() is False
        assert gate.is_throttled() is False

    def test_creation_with_config(self):
        """Test ModeGate with custom config."""
        config = ModeGateConfig(
            staleness_threshold=0.2,
            capacity_low_watermark=10,
        )
        gate = ModeGate(config=config)

        assert gate.config.staleness_threshold == 0.2
        assert gate.config.capacity_low_watermark == 10

    def test_evaluate_normal_operation(self):
        """Test evaluate returns ASYNC_RUNNING for normal conditions."""
        gate = ModeGate()

        state = gate.evaluate(
            staleness=0.1,  # Low staleness
            capacity=100,  # Good capacity
            buffer_fill_ratio=0.5,  # Buffer not full
            in_flight=10,
        )

        assert state.mode == AsyncMode.ASYNC_RUNNING
        assert state.reason == "normal"

    def test_evaluate_high_staleness(self):
        """Test evaluate returns SYNC_BARRIER for high staleness."""
        config = ModeGateConfig(staleness_threshold=0.3)
        gate = ModeGate(config=config)

        state = gate.evaluate(
            staleness=0.5,  # High staleness (> 0.3)
            capacity=100,
            buffer_fill_ratio=0.5,
            in_flight=10,
        )

        assert state.mode == AsyncMode.SYNC_BARRIER
        assert "staleness" in state.reason

    def test_evaluate_capacity_exhausted(self):
        """Test evaluate returns THROTTLED for exhausted capacity."""
        config = ModeGateConfig(capacity_low_watermark=0)
        gate = ModeGate(config=config)

        state = gate.evaluate(
            staleness=0.1,
            capacity=0,  # No capacity
            buffer_fill_ratio=0.5,
            in_flight=10,
        )

        assert state.mode == AsyncMode.THROTTLED
        assert state.reason == "capacity_exhausted"

    def test_evaluate_buffer_full(self):
        """Test evaluate returns THROTTLED for full buffer."""
        config = ModeGateConfig(buffer_high_watermark=0.9)
        gate = ModeGate(config=config)

        state = gate.evaluate(
            staleness=0.1,
            capacity=100,
            buffer_fill_ratio=0.95,  # Buffer nearly full (> 0.9)
            in_flight=10,
        )

        assert state.mode == AsyncMode.THROTTLED
        assert "buffer_full" in state.reason

    def test_evaluate_priority_capacity_over_staleness(self):
        """Test that capacity exhaustion takes priority over staleness."""
        config = ModeGateConfig(
            staleness_threshold=0.3,
            capacity_low_watermark=0,
        )
        gate = ModeGate(config=config)

        state = gate.evaluate(
            staleness=0.5,  # High staleness
            capacity=0,  # No capacity (higher priority)
            buffer_fill_ratio=0.5,
            in_flight=10,
        )

        assert state.mode == AsyncMode.THROTTLED
        assert state.reason == "capacity_exhausted"

    def test_sync_barrier_drains_in_flight(self):
        """Test that SYNC_BARRIER waits for in-flight to drain."""
        config = ModeGateConfig(staleness_threshold=0.3)
        gate = ModeGate(config=config)

        # Enter SYNC_BARRIER due to high staleness
        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.current_mode == AsyncMode.SYNC_BARRIER

        # Staleness dropped but still have in-flight
        state = gate.evaluate(
            staleness=0.1,  # Low staleness now
            capacity=100,
            buffer_fill_ratio=0.5,
            in_flight=5,  # Still have in-flight
        )

        # Should stay in SYNC_BARRIER until in_flight = 0
        assert state.mode == AsyncMode.SYNC_BARRIER
        assert "draining" in state.reason

    def test_sync_barrier_exit_on_drain(self):
        """Test that SYNC_BARRIER exits when in-flight drained."""
        config = ModeGateConfig(
            staleness_threshold=0.3,
            min_barrier_duration_ms=0,  # Disable for test
        )
        gate = ModeGate(config=config)

        # Enter SYNC_BARRIER
        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.current_mode == AsyncMode.SYNC_BARRIER

        # Wait a bit to satisfy hysteresis
        time.sleep(0.01)

        # Staleness recovered and in-flight drained
        state = gate.evaluate(
            staleness=0.1,  # Below recovery threshold (0.24)
            capacity=100,
            buffer_fill_ratio=0.5,
            in_flight=0,  # All drained
        )

        assert state.mode == AsyncMode.ASYNC_RUNNING

    def test_can_submit_rollout(self):
        """Test can_submit_rollout method."""
        gate = ModeGate()

        # ASYNC_RUNNING - can submit
        gate.evaluate(staleness=0.1, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.can_submit_rollout() is True

        # SYNC_BARRIER - cannot submit
        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.can_submit_rollout() is False

        # THROTTLED - cannot submit
        gate.evaluate(staleness=0.1, capacity=0, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.can_submit_rollout() is False

    def test_should_wait_for_barrier(self):
        """Test should_wait_for_barrier method."""
        gate = ModeGate()

        gate.evaluate(staleness=0.1, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.should_wait_for_barrier() is False

        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.should_wait_for_barrier() is True

    def test_is_throttled(self):
        """Test is_throttled method."""
        gate = ModeGate()

        gate.evaluate(staleness=0.1, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.is_throttled() is False

        gate.evaluate(staleness=0.1, capacity=0, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.is_throttled() is True

    def test_update_threshold(self):
        """Test dynamic threshold update."""
        gate = ModeGate()

        gate.update_threshold(0.5)

        assert gate.config.staleness_threshold == 0.5
        assert gate.config.staleness_recovery_threshold == 0.4  # 80% of 0.5

    def test_force_mode(self):
        """Test forcing a specific mode."""
        gate = ModeGate()

        gate.force_mode(AsyncMode.THROTTLED, reason="testing")

        assert gate.current_mode == AsyncMode.THROTTLED
        assert "forced" in gate.current_state.reason

    def test_reset(self):
        """Test reset method."""
        gate = ModeGate()

        # Enter SYNC_BARRIER
        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.current_mode == AsyncMode.SYNC_BARRIER

        gate.reset()

        assert gate.current_mode == AsyncMode.ASYNC_RUNNING
        assert gate.current_state.reason == "reset"

    def test_get_diagnostics(self):
        """Test get_diagnostics method."""
        gate = ModeGate()
        gate.evaluate(staleness=0.1, capacity=100, buffer_fill_ratio=0.5, in_flight=10)

        diag = gate.get_diagnostics()

        assert diag["mode"] == "ASYNC_RUNNING"
        assert "reason" in diag
        assert diag["in_flight"] == 10
        assert diag["staleness_threshold"] == 0.3
        assert "time_in_mode_s" in diag
        assert "transition_count" in diag

    def test_transition_history(self):
        """Test transition history tracking."""
        config = ModeGateConfig(min_barrier_duration_ms=0)
        gate = ModeGate(config=config)

        # Make some transitions
        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)  # -> SYNC_BARRIER
        gate.evaluate(staleness=0.1, capacity=100, buffer_fill_ratio=0.5, in_flight=0)  # -> ASYNC_RUNNING
        gate.evaluate(staleness=0.1, capacity=0, buffer_fill_ratio=0.5, in_flight=0)  # -> THROTTLED

        history = gate.get_transition_history()

        assert len(history) >= 3  # At least 3 transitions
        assert all("from_mode" in h for h in history)
        assert all("to_mode" in h for h in history)
        assert all("reason" in h for h in history)

    def test_transition_history_last_n(self):
        """Test getting last N transitions."""
        config = ModeGateConfig(min_barrier_duration_ms=0)
        gate = ModeGate(config=config)

        # Make several transitions
        for i in range(5):
            gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
            gate.evaluate(staleness=0.1, capacity=100, buffer_fill_ratio=0.5, in_flight=0)

        history = gate.get_transition_history(last_n=3)
        assert len(history) == 3

    def test_hysteresis_min_barrier_duration(self):
        """Test hysteresis prevents rapid exit from SYNC_BARRIER."""
        config = ModeGateConfig(
            staleness_threshold=0.3,
            min_barrier_duration_ms=100,  # 100ms minimum
        )
        gate = ModeGate(config=config)

        # Enter SYNC_BARRIER
        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.current_mode == AsyncMode.SYNC_BARRIER

        # Immediately try to exit (should be blocked by hysteresis)
        state = gate.evaluate(
            staleness=0.1,  # Low staleness
            capacity=100,
            buffer_fill_ratio=0.5,
            in_flight=0,  # Drained
        )

        # Should still be in SYNC_BARRIER due to hysteresis
        assert state.mode == AsyncMode.SYNC_BARRIER
        assert "hysteresis" in state.reason

    def test_current_state_property(self):
        """Test current_state property."""
        gate = ModeGate()
        gate.evaluate(staleness=0.1, capacity=50, buffer_fill_ratio=0.3, in_flight=5)

        state = gate.current_state

        assert isinstance(state, ModeGateState)
        assert state.mode == AsyncMode.ASYNC_RUNNING
        assert state.in_flight == 5

    @pytest.mark.asyncio
    async def test_enforce_barrier_with_wait_fn(self):
        """Test enforce_barrier with custom wait function."""
        gate = ModeGate()
        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)
        assert gate.current_mode == AsyncMode.SYNC_BARRIER

        wait_called = False

        async def mock_wait():
            nonlocal wait_called
            wait_called = True
            await asyncio.sleep(0.01)

        success = await gate.enforce_barrier(wait_for_in_flight=mock_wait, timeout=1.0)

        assert success is True
        assert wait_called is True
        assert gate.current_mode == AsyncMode.ASYNC_RUNNING

    @pytest.mark.asyncio
    async def test_enforce_barrier_timeout(self):
        """Test enforce_barrier timeout."""
        gate = ModeGate()
        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)

        async def slow_wait():
            await asyncio.sleep(10.0)  # Way too long

        success = await gate.enforce_barrier(wait_for_in_flight=slow_wait, timeout=0.1)

        assert success is False

    @pytest.mark.asyncio
    async def test_enforce_barrier_polling(self):
        """Test enforce_barrier with polling (no wait function)."""
        gate = ModeGate()
        gate.evaluate(staleness=0.5, capacity=100, buffer_fill_ratio=0.5, in_flight=10)

        # Simulate in_flight draining
        async def drain_in_flight():
            await asyncio.sleep(0.05)
            with gate._lock:
                gate._in_flight_count = 0

        # Start draining in background
        asyncio.create_task(drain_in_flight())

        success = await gate.enforce_barrier(timeout=1.0)

        assert success is True
        assert gate.current_mode == AsyncMode.ASYNC_RUNNING

    @pytest.mark.asyncio
    async def test_enforce_barrier_not_in_barrier_mode(self):
        """Test enforce_barrier returns immediately if not in SYNC_BARRIER."""
        gate = ModeGate()
        assert gate.current_mode == AsyncMode.ASYNC_RUNNING

        success = await gate.enforce_barrier(timeout=0.1)

        assert success is True  # Immediately returns True

    def test_thread_safety(self):
        """Test thread safety of ModeGate."""
        import threading

        gate = ModeGate()
        errors = []

        def evaluate_worker():
            try:
                for i in range(100):
                    gate.evaluate(
                        staleness=0.1 + (i % 5) * 0.1,
                        capacity=100 - i % 10,
                        buffer_fill_ratio=0.5,
                        in_flight=i % 20,
                    )
            except Exception as e:
                errors.append(e)

        def check_worker():
            try:
                for _ in range(100):
                    gate.can_submit_rollout()
                    gate.current_mode
                    gate.get_diagnostics()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=evaluate_worker),
            threading.Thread(target=check_worker),
            threading.Thread(target=evaluate_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestModeGateIntegration:
    """Tests for ModeGateIntegration helper."""

    @pytest.fixture
    def mock_staleness_manager(self):
        """Create mock staleness manager."""
        mock = MagicMock()
        mock.current_staleness = 0.1
        mock.get_capacity.return_value = 100
        mock.stats.total_in_flight = 5
        return mock

    def test_creation(self, mock_staleness_manager):
        """Test ModeGateIntegration creation."""
        gate = ModeGate()
        integration = ModeGateIntegration(
            mode_gate=gate,
            staleness_manager=mock_staleness_manager,
        )

        assert integration.mode_gate is gate
        assert integration.staleness_manager is mock_staleness_manager

    def test_get_rollout_slots_async_running(self, mock_staleness_manager):
        """Test get_rollout_slots in ASYNC_RUNNING mode."""
        gate = ModeGate()
        integration = ModeGateIntegration(
            mode_gate=gate,
            staleness_manager=mock_staleness_manager,
        )

        slots = integration.get_rollout_slots()
        assert slots == 100  # From mock capacity

    def test_get_rollout_slots_not_running(self, mock_staleness_manager):
        """Test get_rollout_slots when not in ASYNC_RUNNING."""
        gate = ModeGate()
        gate.force_mode(AsyncMode.THROTTLED)

        integration = ModeGateIntegration(
            mode_gate=gate,
            staleness_manager=mock_staleness_manager,
        )

        slots = integration.get_rollout_slots()
        assert slots == 0

    @pytest.mark.asyncio
    async def test_check_and_enforce_normal(self, mock_staleness_manager):
        """Test check_and_enforce in normal operation."""
        gate = ModeGate()
        integration = ModeGateIntegration(
            mode_gate=gate,
            staleness_manager=mock_staleness_manager,
        )

        state = await integration.check_and_enforce()

        assert state.mode == AsyncMode.ASYNC_RUNNING

    @pytest.mark.asyncio
    async def test_check_and_enforce_barrier(self, mock_staleness_manager):
        """Test check_and_enforce triggers barrier."""
        mock_staleness_manager.current_staleness = 0.5  # High staleness
        mock_staleness_manager.stats.total_in_flight = 10

        gate = ModeGate()
        integration = ModeGateIntegration(
            mode_gate=gate,
            staleness_manager=mock_staleness_manager,
        )

        wait_called = False

        async def mock_wait():
            nonlocal wait_called
            wait_called = True

        # First call should trigger barrier
        state = await integration.check_and_enforce(wait_for_in_flight=mock_wait)

        assert wait_called is True

    def test_custom_buffer_fill_fn(self, mock_staleness_manager):
        """Test with custom buffer fill function."""
        gate = ModeGate()

        buffer_fill_fn = MagicMock(return_value=0.8)

        integration = ModeGateIntegration(
            mode_gate=gate,
            staleness_manager=mock_staleness_manager,
            trajectory_buffer_fill_fn=buffer_fill_fn,
        )

        # Should call the buffer fill function
        slots = integration.get_rollout_slots()
        # Note: slots are based on staleness_manager.get_capacity()
        assert slots == 100
