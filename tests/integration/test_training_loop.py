"""
Integration tests for the training loop.
"""

import pytest
from pathlib import Path

from flux.core.config import FluxConfig
from flux.core.trajectory import Trajectory
from flux.core.types import PolicyVersion
from flux.rewards.rule_based import LengthReward
from flux.trainer import FluxTrainer, TrainingResult


class TestFluxTrainer:
    """Tests for FluxTrainer."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test config."""
        return FluxConfig(
            model_path="test-model",
            output_dir=str(tmp_path),
            num_steps=5,
            batch_size=4,
        )

    @pytest.fixture
    def prompts(self):
        """Create test prompts."""
        return [
            "What is 2+2?",
            "Explain Python",
            "Write a poem",
            "Tell me a joke",
            "Describe the weather",
        ] * 10  # Repeat for enough data

    def test_trainer_creation(self, config):
        """Test trainer creation."""
        trainer = FluxTrainer(config)

        assert trainer.config == config
        assert not trainer.is_initialized
        assert trainer.coordinator is None

    def test_add_callback(self, config):
        """Test adding callbacks."""
        trainer = FluxTrainer(config)

        def my_callback(state, metrics):
            pass

        trainer.add_callback(my_callback)
        assert len(trainer._callbacks) == 1

    def test_add_step_callback(self, config):
        """Test adding step callbacks."""
        trainer = FluxTrainer(config)

        def step_callback(result):
            pass

        trainer.add_step_callback(step_callback)
        assert len(trainer._step_callbacks) == 1

    def test_prepare_prompts_list(self, config):
        """Test prompt preparation from list."""
        trainer = FluxTrainer(config)

        prompts = ["Hello", "World"]
        prepared = trainer._prepare_prompts(prompts)

        assert prepared == ["Hello", "World"]

    def test_prepare_prompts_dict(self, config):
        """Test prompt preparation from dict list."""
        trainer = FluxTrainer(config)

        prompts = [
            {"prompt": "Hello"},
            {"text": "World"},
        ]
        prepared = trainer._prepare_prompts(prompts)

        assert prepared == ["Hello", "World"]

    def test_prepare_prompts_none(self, config):
        """Test prompt preparation with None."""
        trainer = FluxTrainer(config)

        prepared = trainer._prepare_prompts(None)
        assert prepared == []

    def test_context_manager(self, config):
        """Test context manager usage."""
        with FluxTrainer(config) as trainer:
            # Should be initialized inside context
            pass
        # Should be torn down after context

    def test_get_statistics(self, config):
        """Test getting statistics."""
        trainer = FluxTrainer(config)
        stats = trainer.get_statistics()

        assert "training_state" in stats
        assert "initialized" in stats
        assert stats["initialized"] == False


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_default_result(self):
        """Test default values."""
        result = TrainingResult()
        assert result.total_steps == 0
        assert result.final_loss == 0.0
        assert result.reward_history == []

    def test_result_with_values(self):
        """Test with custom values."""
        result = TrainingResult(
            total_steps=1000,
            final_loss=0.5,
            total_samples=5000,
            total_time_seconds=300.0,
        )
        assert result.total_steps == 1000
        assert result.samples_per_second == 0.0  # Not computed

    def test_to_dict(self):
        """Test serialization."""
        result = TrainingResult(
            total_steps=100,
            final_loss=0.3,
        )
        d = result.to_dict()

        assert d["total_steps"] == 100
        assert d["final_loss"] == 0.3


class TestTrainingIntegration:
    """Integration tests for full training flow."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test config."""
        return FluxConfig(
            model_path="test-model",
            output_dir=str(tmp_path),
            num_steps=3,
            batch_size=2,
        )

    def test_trajectory_flow(self, config):
        """Test trajectory through the system."""
        # Create trajectory
        traj = Trajectory(
            id="test-1",
            prompt="Hello",
            response="Hi there!",
            tokens=[1, 2, 3, 4],
            log_probs=[-0.1, -0.2, -0.1, -0.15],
            version=PolicyVersion(version_id=0),
        )

        # Compute reward
        reward_fn = LengthReward()
        output = reward_fn.compute_reward(traj)
        traj.reward = output.reward

        # Verify trajectory is complete
        assert traj.prompt == "Hello"
        assert traj.reward >= 0

    def test_batch_composition_flow(self, config):
        """Test batch composition through the system."""
        from flux.training.batch_composer import SmartBatchComposer
        from flux.core.trajectory import TrajectoryBuffer

        # Create trajectories
        buffer = TrajectoryBuffer()
        for i in range(10):
            traj = Trajectory(
                id=f"t-{i}",
                tokens=list(range(100 + i * 10)),
                version=PolicyVersion(version_id=i % 3),
            )
            buffer.add(traj)

        # Compose batches
        composer = SmartBatchComposer(
            config=config.batch_composer,
            batch_size=config.batch_size,
        )

        trajectories = buffer.sample(
            n=8,
            current_version=2,
            stratified=True,
        )

        batches = list(composer.compose_batches(
            trajectories=trajectories,
            current_version=2,
        ))

        # Should have batches
        assert len(batches) >= 1

    def test_reward_composition(self):
        """Test composite reward flow."""
        from flux.rewards.base import CompositeReward

        composite = CompositeReward(
            rewards=[
                (LengthReward(), 0.5),
                (LengthReward(reward_type="log"), 0.5),
            ],
        )

        traj = Trajectory(
            id="test",
            response="A reasonable length response",
            response_length=28,
        )

        output = composite.compute_reward(traj)
        assert output.reward >= 0
        assert "components" in output.metadata
