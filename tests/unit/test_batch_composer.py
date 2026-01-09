"""
Tests for SmartBatchComposer.
"""

import pytest

from flux.core.config import BatchComposerConfig
from flux.core.trajectory import Trajectory, TrajectoryBuffer
from flux.core.types import PolicyVersion
from flux.training.batch_composer import (
    BatchIterator,
    LengthBucket,
    SmartBatchComposer,
    StalenessStratum,
)


class TestLengthBucket:
    """Tests for LengthBucket dataclass."""

    def test_creation(self):
        """Test bucket creation."""
        bucket = LengthBucket(min_length=0, max_length=512)
        assert bucket.min_length == 0
        assert bucket.max_length == 512
        assert bucket.size == 0
        assert bucket.is_empty

    def test_add_trajectory(self):
        """Test adding trajectories to bucket."""
        bucket = LengthBucket(min_length=0, max_length=512)
        traj = Trajectory(id="test-1", tokens=list(range(100)))
        bucket.add(traj)
        assert bucket.size == 1
        assert not bucket.is_empty

    def test_sample(self):
        """Test sampling from bucket."""
        bucket = LengthBucket(min_length=0, max_length=512)
        for i in range(10):
            bucket.add(Trajectory(id=f"test-{i}", tokens=list(range(100))))

        sampled = bucket.sample(5)
        assert len(sampled) == 5

        # Sample more than available
        sampled = bucket.sample(20)
        assert len(sampled) == 10

    def test_clear(self):
        """Test clearing bucket."""
        bucket = LengthBucket(min_length=0, max_length=512)
        for i in range(5):
            bucket.add(Trajectory(id=f"test-{i}"))
        bucket.clear()
        assert bucket.is_empty


class TestStalenessStratum:
    """Tests for StalenessStratum dataclass."""

    def test_creation(self):
        """Test stratum creation."""
        stratum = StalenessStratum(min_staleness=0, max_staleness=2)
        assert stratum.min_staleness == 0
        assert stratum.max_staleness == 2
        assert stratum.size == 0

    def test_add_trajectory(self):
        """Test adding to stratum."""
        stratum = StalenessStratum(min_staleness=0, max_staleness=2)
        stratum.trajectories.append(Trajectory(id="test-1"))
        assert stratum.size == 1


class TestSmartBatchComposer:
    """Tests for SmartBatchComposer."""

    def _make_trajectory(
        self, id: str, length: int, version: int
    ) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(
            id=id,
            tokens=list(range(length)),
            response="x" * length,
            response_length=length,
            version=PolicyVersion(version_id=version),
        )

    def test_creation_defaults(self):
        """Test composer creation with defaults."""
        composer = SmartBatchComposer()
        assert composer.batch_size == 32
        assert composer.current_version == 0

    def test_creation_with_config(self):
        """Test composer with custom config."""
        config = BatchComposerConfig(
            use_length_bucketing=True,
            use_staleness_balancing=True,
            length_bucket_boundaries=(256, 512, 1024),
        )
        composer = SmartBatchComposer(config=config, batch_size=16)
        assert composer.batch_size == 16

    def test_bucket_creation(self):
        """Test length buckets are created from config."""
        config = BatchComposerConfig(
            length_bucket_boundaries=(512, 1024),
        )
        composer = SmartBatchComposer(config=config)

        # Should have 3 buckets: [0-512), [512-1024), [1024-inf)
        assert len(composer._buckets) == 3
        assert composer._buckets[0].min_length == 0
        assert composer._buckets[0].max_length == 512
        assert composer._buckets[1].min_length == 512
        assert composer._buckets[1].max_length == 1024

    def test_get_bucket_for_length(self):
        """Test finding correct bucket for length."""
        config = BatchComposerConfig(
            length_bucket_boundaries=(512, 1024),
        )
        composer = SmartBatchComposer(config=config)

        bucket = composer._get_bucket_for_length(100)
        assert bucket.max_length == 512

        bucket = composer._get_bucket_for_length(700)
        assert bucket.min_length == 512

        bucket = composer._get_bucket_for_length(2000)
        assert bucket.min_length == 1024

    def test_compose_batch_simple(self):
        """Test basic batch composition."""
        composer = SmartBatchComposer(batch_size=4)
        trajectories = [
            self._make_trajectory(f"t-{i}", 100 + i * 10, 0)
            for i in range(4)
        ]

        batch = composer.compose_batch(trajectories)
        assert len(batch) == 4

    def test_compose_batches_no_bucketing(self):
        """Test batch composition without bucketing."""
        config = BatchComposerConfig(
            use_length_bucketing=False,
            use_staleness_balancing=False,
            use_curriculum=False,
        )
        composer = SmartBatchComposer(config=config, batch_size=4)

        trajectories = [
            self._make_trajectory(f"t-{i}", 100, 0)
            for i in range(10)
        ]

        batches = list(composer.compose_batches(trajectories))

        # Should have 3 batches (10 / 4 = 2.5 -> 3)
        assert len(batches) == 3
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4
        assert len(batches[2]) == 2

    def test_compose_batches_with_bucketing(self):
        """Test batch composition with length bucketing."""
        config = BatchComposerConfig(
            use_length_bucketing=True,
            use_staleness_balancing=False,
            use_curriculum=False,
            length_bucket_boundaries=(256, 512),
        )
        composer = SmartBatchComposer(config=config, batch_size=4)

        # Create trajectories with different lengths
        trajectories = [
            self._make_trajectory("short-1", 100, 0),
            self._make_trajectory("short-2", 150, 0),
            self._make_trajectory("medium-1", 300, 0),
            self._make_trajectory("medium-2", 400, 0),
            self._make_trajectory("long-1", 600, 0),
        ]

        batches = list(composer.compose_batches(trajectories))

        # Should have separate batches per bucket
        assert len(batches) >= 1

    def test_compose_batches_with_staleness_balancing(self):
        """Test staleness balancing."""
        config = BatchComposerConfig(
            use_length_bucketing=False,
            use_staleness_balancing=True,
            use_curriculum=False,
            staleness_strata=3,
        )
        composer = SmartBatchComposer(config=config, batch_size=8)
        composer.current_version = 5

        # Create trajectories with different versions
        trajectories = []
        for v in range(3):
            for i in range(4):
                trajectories.append(
                    self._make_trajectory(f"t-{v}-{i}", 100, v)
                )

        batches = list(composer.compose_batches(trajectories, current_version=5))
        assert len(batches) >= 1

    def test_compose_batches_with_curriculum(self):
        """Test curriculum ordering."""
        config = BatchComposerConfig(
            use_length_bucketing=False,
            use_staleness_balancing=False,
            use_curriculum=True,
            curriculum_randomness_decay=1.0,
        )
        composer = SmartBatchComposer(config=config, batch_size=4)

        trajectories = [
            self._make_trajectory("easy", 50, 0),
            self._make_trajectory("medium", 500, 0),
            self._make_trajectory("hard", 1000, 0),
        ]
        trajectories[0].reward = 0.9  # Easy (high reward)
        trajectories[1].reward = 0.5  # Medium
        trajectories[2].reward = 0.1  # Hard (low reward)

        batches = list(composer.compose_batches(trajectories))
        assert len(batches) >= 1

        # Curriculum step should increment
        assert composer._curriculum_step == 1

    def test_set_difficulty_function(self):
        """Test custom difficulty function."""
        composer = SmartBatchComposer()

        def custom_difficulty(traj):
            return traj.reward  # Higher reward = harder

        composer.set_difficulty_function(custom_difficulty)
        assert composer._difficulty_fn is not None

    def test_compose_from_buffer(self):
        """Test composing from trajectory buffer."""
        composer = SmartBatchComposer(batch_size=4)

        buffer = TrajectoryBuffer(max_size=100)
        for i in range(10):
            buffer.add(self._make_trajectory(f"t-{i}", 100, 0))

        batches = list(composer.compose_from_buffer(
            buffer=buffer,
            current_version=0,
            n_batches=2,
        ))

        assert len(batches) >= 1

    def test_get_statistics(self):
        """Test getting composer statistics."""
        composer = SmartBatchComposer(batch_size=16)

        stats = composer.get_statistics()

        assert stats["batch_size"] == 16
        assert "buckets" in stats
        assert "strata" in stats
        assert "config" in stats

    def test_reset_curriculum(self):
        """Test resetting curriculum state."""
        composer = SmartBatchComposer()
        composer._curriculum_step = 100

        composer.reset_curriculum()
        assert composer._curriculum_step == 0

    def test_version_update(self):
        """Test version update."""
        composer = SmartBatchComposer()
        assert composer.current_version == 0

        composer.current_version = 10
        assert composer.current_version == 10


class TestBatchIterator:
    """Tests for BatchIterator."""

    def _make_trajectory(self, id: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(
            id=id,
            tokens=list(range(100)),
            version=PolicyVersion(version_id=0),
        )

    def test_creation(self):
        """Test iterator creation."""
        composer = SmartBatchComposer(batch_size=4)
        buffer = TrajectoryBuffer()
        for i in range(10):
            buffer.add(self._make_trajectory(f"t-{i}"))

        iterator = BatchIterator(
            composer=composer,
            buffer=buffer,
            current_version=0,
            num_batches=5,
        )

        assert iterator.num_batches == 5

    def test_iteration(self):
        """Test iterating over batches."""
        composer = SmartBatchComposer(batch_size=4)
        buffer = TrajectoryBuffer()
        for i in range(20):
            buffer.add(self._make_trajectory(f"t-{i}"))

        iterator = BatchIterator(
            composer=composer,
            buffer=buffer,
            current_version=0,
            num_batches=3,
        )

        count = 0
        for batch in iterator:
            count += 1
            if count >= 3:
                break

        assert count == 3

    def test_update_version(self):
        """Test version update."""
        composer = SmartBatchComposer()
        buffer = TrajectoryBuffer()
        iterator = BatchIterator(
            composer=composer,
            buffer=buffer,
            current_version=0,
        )

        iterator.update_version(10)
        assert iterator.current_version == 10
