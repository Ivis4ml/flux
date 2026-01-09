"""
Pytest configuration and fixtures for Flux tests.
"""

import pytest
import torch

from flux.core.config import (
    AdaptiveAsyncConfig,
    AlgorithmConfig,
    FluxConfig,
    RolloutConfig,
    SGLangConfig,
)
from flux.core.trajectory import Trajectory, TrajectoryBuffer
from flux.core.types import PolicyVersion, TrainingState


@pytest.fixture
def sample_config() -> FluxConfig:
    """Create a sample FluxConfig for testing."""
    return FluxConfig(
        model_path="test-model",
        output_dir="./test_outputs",
        learning_rate=1e-6,
        batch_size=4,
        num_steps=100,
    )


@pytest.fixture
def sample_adaptive_config() -> AdaptiveAsyncConfig:
    """Create a sample AdaptiveAsyncConfig for testing."""
    return AdaptiveAsyncConfig(
        target_staleness=0.15,
        tolerance=0.05,
        min_async_ratio=0.1,
        max_async_ratio=0.9,
    )


@pytest.fixture
def sample_trajectory() -> Trajectory:
    """Create a sample trajectory for testing."""
    return Trajectory(
        id="test_traj_001",
        prompt="What is 2 + 2?",
        response="2 + 2 = 4",
        tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        attention_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        loss_mask=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        log_probs=[-0.1, -0.2, -0.15, -0.1, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05],
        behavior_log_probs=[-0.1, -0.2, -0.15, -0.1, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05],
        reward=1.0,
        prompt_length=4,
        response_length=6,
        version=PolicyVersion(version_id=1),
    )


@pytest.fixture
def sample_trajectories() -> list[Trajectory]:
    """Create multiple sample trajectories for testing."""
    trajectories = []
    for i in range(5):
        traj = Trajectory(
            id=f"test_traj_{i:03d}",
            prompt=f"Prompt {i}",
            response=f"Response {i}",
            tokens=list(range(i * 10, (i + 1) * 10)),
            attention_mask=[1] * 10,
            loss_mask=[0] * 4 + [1] * 6,
            log_probs=[-0.1] * 10,
            behavior_log_probs=[-0.1] * 10,
            reward=float(i),
            prompt_length=4,
            response_length=6,
            version=PolicyVersion(version_id=i),
        )
        trajectories.append(traj)
    return trajectories


@pytest.fixture
def sample_buffer(sample_trajectories: list[Trajectory]) -> TrajectoryBuffer:
    """Create a sample trajectory buffer for testing."""
    buffer = TrajectoryBuffer(max_size=100, max_staleness=5)
    buffer.add_batch(sample_trajectories)
    return buffer


@pytest.fixture
def sample_training_state() -> TrainingState:
    """Create a sample training state for testing."""
    return TrainingState(
        global_step=50,
        epoch=1,
        step_in_epoch=50,
        total_steps=1000,
    )


@pytest.fixture
def device() -> torch.device:
    """Get appropriate device for tests."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# Markers for test categories
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
