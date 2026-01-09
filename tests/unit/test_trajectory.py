"""
Unit tests for Flux trajectory module.
"""

import pytest
import torch

from flux.core.trajectory import (
    PartialTrajectory,
    Trajectory,
    TrajectoryBatch,
    TrajectoryBuffer,
    VersionSegment,
)
from flux.core.types import PolicyVersion, TrajectoryStatus


class TestVersionSegment:
    """Tests for VersionSegment."""

    def test_creation(self) -> None:
        """Test segment creation."""
        segment = VersionSegment(
            start=0,
            end=100,
            version=PolicyVersion(version_id=1),
        )
        assert segment.start == 0
        assert segment.end == 100
        assert segment.length == 100


class TestTrajectory:
    """Tests for Trajectory."""

    def test_basic_creation(self) -> None:
        """Test basic trajectory creation."""
        traj = Trajectory(
            id="test_001",
            prompt="Hello",
            response="World",
        )
        assert traj.id == "test_001"
        assert traj.status == TrajectoryStatus.PENDING

    def test_total_length(self) -> None:
        """Test total length calculation."""
        traj = Trajectory(
            tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        assert traj.total_length == 10

    def test_version_gap(self) -> None:
        """Test version gap calculation."""
        traj = Trajectory(
            version=PolicyVersion(version_id=5),
        )
        assert traj.get_version_gap(10) == 5
        assert traj.get_version_gap(PolicyVersion(version_id=10)) == 5

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        traj = Trajectory(
            id="test_001",
            tokens=[1, 2, 3],
            reward=1.5,
            version=PolicyVersion(version_id=3),
        )
        d = traj.to_dict()
        assert d["id"] == "test_001"
        assert d["tokens"] == [1, 2, 3]
        assert d["reward"] == 1.5
        assert d["version"] == 3

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        d = {
            "id": "test_001",
            "tokens": [1, 2, 3],
            "reward": 1.5,
            "version": 3,
        }
        traj = Trajectory.from_dict(d)
        assert traj.id == "test_001"
        assert traj.tokens == [1, 2, 3]
        assert traj.version.version_id == 3

    def test_version_boundaries(self) -> None:
        """Test version boundary detection."""
        # No segments = no boundaries
        traj = Trajectory()
        assert traj.has_version_boundaries is False

        # Multiple segments = has boundaries
        traj.version_segments = [
            VersionSegment(0, 50, PolicyVersion(1)),
            VersionSegment(50, 100, PolicyVersion(2)),
        ]
        assert traj.has_version_boundaries is True


class TestPartialTrajectory:
    """Tests for PartialTrajectory."""

    def test_can_continue(self) -> None:
        """Test continuation eligibility."""
        partial = PartialTrajectory(
            status=TrajectoryStatus.ABORTED,
            continuation_count=0,
        )
        assert partial.can_continue() is True

        partial.continuation_count = 3
        assert partial.can_continue() is False

    def test_prepare_continuation(self) -> None:
        """Test continuation prompt preparation."""
        partial = PartialTrajectory(
            prompt="Hello ",
            response="World",
        )
        continuation = partial.prepare_continuation()
        assert continuation == "Hello World"


class TestTrajectoryBatch:
    """Tests for TrajectoryBatch."""

    def test_empty_batch(self) -> None:
        """Test empty batch properties."""
        batch = TrajectoryBatch()
        assert len(batch) == 0
        assert batch.max_length == 0

    def test_batch_from_trajectories(
        self, sample_trajectories: list[Trajectory]
    ) -> None:
        """Test batch creation from trajectories."""
        batch = TrajectoryBatch(trajectories=sample_trajectories)
        assert len(batch) == 5
        assert batch.batch_size == 5

    def test_to_tensors(self, sample_trajectories: list[Trajectory]) -> None:
        """Test conversion to tensors."""
        batch = TrajectoryBatch(trajectories=sample_trajectories)
        tensors = batch.to_tensors(device="cpu", pad_token_id=0)

        assert "input_ids" in tensors
        assert "attention_mask" in tensors
        assert "rewards" in tensors

        assert tensors["input_ids"].shape[0] == 5  # batch size
        assert tensors["rewards"].shape == (5,)

    def test_padding_ratio(self, sample_trajectory: Trajectory) -> None:
        """Test padding ratio calculation."""
        # Create trajectories of different lengths
        short = Trajectory(tokens=[1, 2, 3])
        long = Trajectory(tokens=list(range(10)))

        batch = TrajectoryBatch(trajectories=[short, long])
        ratio = batch.compute_padding_ratio()

        # 7 padding tokens out of 20 total = 0.35
        assert ratio == pytest.approx(0.35)

    def test_version_stats(self, sample_trajectories: list[Trajectory]) -> None:
        """Test version statistics."""
        batch = TrajectoryBatch(trajectories=sample_trajectories)
        stats = batch.get_version_stats()

        assert "mean_version" in stats
        assert "version_spread" in stats
        assert stats["min_version"] == 0
        assert stats["max_version"] == 4


class TestTrajectoryBuffer:
    """Tests for TrajectoryBuffer."""

    def test_add_trajectory(self, sample_trajectory: Trajectory) -> None:
        """Test adding trajectory to buffer."""
        buffer = TrajectoryBuffer()
        buffer.add(sample_trajectory)
        assert len(buffer) == 1

    def test_add_batch(self, sample_trajectories: list[Trajectory]) -> None:
        """Test adding multiple trajectories."""
        buffer = TrajectoryBuffer()
        buffer.add_batch(sample_trajectories)
        assert len(buffer) == 5

    def test_get_available(self, sample_buffer: TrajectoryBuffer) -> None:
        """Test getting available trajectories."""
        # Version 5 means trajectories with version >= 0 are available
        available = sample_buffer.get_available(current_version=5)
        assert len(available) == 5

        # With max_staleness=2, only versions 3, 4 should be available
        available = sample_buffer.get_available(current_version=5, max_staleness=2)
        assert len(available) == 2

    def test_sample(self, sample_buffer: TrajectoryBuffer) -> None:
        """Test sampling from buffer."""
        samples = sample_buffer.sample(n=3, current_version=10)
        assert len(samples) <= 3

    def test_stratified_sample(self, sample_buffer: TrajectoryBuffer) -> None:
        """Test stratified sampling by staleness."""
        samples = sample_buffer.sample(
            n=3, current_version=10, stratified=True
        )
        assert len(samples) <= 3

    def test_remove_stale(self, sample_buffer: TrajectoryBuffer) -> None:
        """Test removing stale trajectories."""
        initial_size = len(sample_buffer)
        # Current version 10, max_staleness 5 -> keep version >= 5
        # Our trajectories are version 0-4, so all should be removed
        removed = sample_buffer.remove_stale(current_version=10)
        assert removed == initial_size
        assert len(sample_buffer) == 0

    def test_partial_buffer(self) -> None:
        """Test partial trajectory buffer."""
        buffer = TrajectoryBuffer()
        partial = PartialTrajectory(
            id="partial_001",
            status=TrajectoryStatus.ABORTED,
        )
        buffer.add_partial(partial)

        partials = buffer.get_partials_for_prompts(["partial_001"])
        assert "partial_001" in partials

    def test_stats(self, sample_buffer: TrajectoryBuffer) -> None:
        """Test buffer statistics."""
        stats = sample_buffer.get_stats()
        assert "size" in stats
        assert "mean_staleness" in stats
        assert stats["size"] == 5

    def test_max_size(self) -> None:
        """Test buffer max size limit."""
        buffer = TrajectoryBuffer(max_size=3)
        for i in range(5):
            buffer.add(Trajectory(id=f"traj_{i}"))
        assert len(buffer) == 3  # Oldest removed
