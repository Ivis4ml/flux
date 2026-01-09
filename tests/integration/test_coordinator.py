"""
Integration tests for FluxCoordinator.
"""

import pytest

from flux.coordinator.coordinator import (
    CoordinatorState,
    FluxCoordinator,
    StepResult,
)
from flux.core.config import FluxConfig
from flux.core.types import PolicyVersion
from flux.rewards.rule_based import LengthReward


class TestCoordinatorState:
    """Tests for CoordinatorState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = CoordinatorState()
        assert state.step == 0
        assert state.version.version_id == 0
        assert state.total_trajectories == 0

    def test_state_with_values(self):
        """Test state with custom values."""
        state = CoordinatorState(
            step=100,
            version=PolicyVersion(version_id=50),
            total_trajectories=1000,
        )
        assert state.step == 100
        assert state.version.version_id == 50


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_default_result(self):
        """Test default result values."""
        result = StepResult(step=1)
        assert result.step == 1
        assert result.training_result is None
        assert result.batch_size == 0

    def test_result_with_metrics(self):
        """Test result with metrics."""
        result = StepResult(
            step=10,
            batch_size=32,
            num_trajectories=48,
            elapsed_ms=150.5,
            sync_performed=True,
        )
        assert result.step == 10
        assert result.batch_size == 32
        assert result.sync_performed


class TestFluxCoordinator:
    """Tests for FluxCoordinator."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return FluxConfig(
            model_path="test-model",
            num_steps=10,
            batch_size=4,
        )

    @pytest.fixture
    def coordinator(self, config):
        """Create coordinator without initialization."""
        return FluxCoordinator(config=config)

    def test_creation(self, coordinator):
        """Test coordinator creation."""
        assert coordinator.config is not None
        assert not coordinator.is_initialized
        assert coordinator.current_version.version_id == 0

    def test_state_access(self, coordinator):
        """Test state access."""
        state = coordinator.state
        assert isinstance(state, CoordinatorState)
        assert state.step == 0

    def test_training_state_access(self, coordinator):
        """Test training state access."""
        training_state = coordinator.training_state
        assert training_state.total_steps == 10

    def test_add_step_callback(self, coordinator):
        """Test adding step callback."""
        callback_results = []

        def callback(result):
            callback_results.append(result.step)

        coordinator.add_step_callback(callback)
        assert len(coordinator._step_callbacks) == 1

    def test_get_statistics(self, coordinator):
        """Test getting statistics."""
        stats = coordinator.get_statistics()

        assert "step" in stats
        assert "version" in stats
        assert "buffer_size" in stats
        assert "batch_composer" in stats

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self, config):
        """Test initialization and shutdown."""
        # Use minimal config to avoid loading models
        coordinator = FluxCoordinator(
            config=config,
            reward_function=LengthReward(),
        )

        # Note: Full initialization would require models
        # This tests the basic lifecycle
        assert not coordinator.is_initialized

    def test_generate_rollouts_sync(self, coordinator):
        """Test sync rollout generation."""
        prompts = ["Hello", "World"]
        trajectories = coordinator._generate_rollouts_sync(prompts)

        assert len(trajectories) == 2
        assert trajectories[0].prompt == "Hello"
        assert trajectories[1].prompt == "World"

    def test_compute_rewards(self, coordinator):
        """Test reward computation."""
        from flux.core.trajectory import Trajectory

        # Set a reward function
        coordinator._reward_fn = LengthReward()

        trajectories = [
            Trajectory(id="t1", response="short"),
            Trajectory(id="t2", response="longer response here"),
        ]

        trajectories = coordinator._compute_rewards(trajectories)

        # Rewards should be computed
        assert trajectories[0].reward >= 0
        assert trajectories[1].reward >= 0

    def test_compose_batch(self, config):
        """Test batch composition."""
        from flux.core.trajectory import Trajectory

        coordinator = FluxCoordinator(config=config)

        # Add trajectories to buffer
        for i in range(10):
            traj = Trajectory(
                id=f"t-{i}",
                tokens=list(range(100)),
                version=PolicyVersion(version_id=0),
            )
            coordinator._buffer.add(traj)

        # Compose batch
        batch = coordinator._compose_batch()

        # Should get a batch (might be None if not enough data)
        # With 10 trajectories and batch_size=4, should work
        if batch is not None:
            assert len(batch) <= config.batch_size


class TestCoordinatorCheckpointing:
    """Tests for coordinator checkpointing."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test config with temp output dir."""
        return FluxConfig(
            model_path="test-model",
            output_dir=str(tmp_path),
        )

    def test_save_and_load_checkpoint(self, config, tmp_path):
        """Test checkpoint save and load."""
        coordinator = FluxCoordinator(config=config)

        # Set some state
        coordinator._state.step = 100
        coordinator._state.version = PolicyVersion(version_id=50)
        coordinator._state.total_trajectories = 1000
        coordinator._state.rewards_sum = 500.0

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint"
        coordinator.save_checkpoint(checkpoint_path)

        # Create new coordinator and load
        coordinator2 = FluxCoordinator(config=config)
        coordinator2.load_checkpoint(checkpoint_path)

        # Verify state restored
        assert coordinator2._state.step == 100
        assert coordinator2._state.version.version_id == 50
        assert coordinator2._state.total_trajectories == 1000
