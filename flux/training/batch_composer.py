"""
Smart batch composition for efficient training.

This module provides intelligent batching strategies that minimize padding waste,
balance staleness for stable training, and support curriculum learning.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from flux.core.config import BatchComposerConfig
from flux.core.trajectory import Trajectory, TrajectoryBatch, TrajectoryBuffer
from flux.core.types import PolicyVersion


logger = logging.getLogger(__name__)


@dataclass
class LengthBucket:
    """A bucket for trajectories of similar length."""

    min_length: int
    max_length: int | float  # Can be float("inf") for unbounded upper limit
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.trajectories)

    @property
    def is_empty(self) -> bool:
        return len(self.trajectories) == 0

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to this bucket."""
        self.trajectories.append(trajectory)

    def clear(self) -> None:
        """Clear all trajectories from bucket."""
        self.trajectories.clear()

    def sample(self, n: int) -> list[Trajectory]:
        """Sample n trajectories from this bucket."""
        if n >= len(self.trajectories):
            return list(self.trajectories)
        return random.sample(self.trajectories, n)


@dataclass
class StalenessStratum:
    """A stratum for trajectories with similar staleness."""

    min_staleness: int
    max_staleness: int | float  # Can be float("inf") for unbounded upper limit
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.trajectories)


@dataclass
class CompositionStats:
    """Statistics from batch composition."""

    total_trajectories: int = 0
    batches_created: int = 0
    padding_ratio: float = 0.0
    mean_staleness: float = 0.0
    staleness_std: float = 0.0
    length_efficiency: float = 0.0  # 1 - padding_ratio
    curriculum_score: float = 0.0


class SmartBatchComposer:
    """Intelligent batch composition with multiple optimization strategies.

    Features:
    - Length bucketing: Groups trajectories by length to minimize padding waste
    - Staleness balancing: Stratified sampling to balance fresh and stale data
    - Curriculum ordering: Sorts by difficulty with decaying randomness

    Example:
        composer = SmartBatchComposer(
            config=BatchComposerConfig(
                use_length_bucketing=True,
                use_staleness_balancing=True,
                use_curriculum=True,
            ),
            batch_size=32,
        )

        for batch in composer.compose_batches(trajectories, current_version=10):
            train_step(batch)
    """

    def __init__(
        self,
        config: BatchComposerConfig | None = None,
        batch_size: int = 32,
        current_version: int = 0,
    ) -> None:
        """Initialize the batch composer.

        Args:
            config: Batch composition configuration.
            batch_size: Target batch size.
            current_version: Current policy version for staleness computation.
        """
        self.config = config or BatchComposerConfig()
        self.batch_size = batch_size
        self._current_version = current_version

        # Initialize length buckets
        self._buckets = self._create_length_buckets()

        # Initialize staleness strata
        self._strata = self._create_staleness_strata()

        # Curriculum state
        self._curriculum_step = 0
        self._difficulty_fn: Callable[[Trajectory], float] | None = None

        # Statistics
        self._stats = CompositionStats()

    @property
    def current_version(self) -> int:
        """Current policy version."""
        return self._current_version

    @current_version.setter
    def current_version(self, value: int) -> None:
        """Update current policy version."""
        self._current_version = value

    def _create_length_buckets(self) -> list[LengthBucket]:
        """Create length buckets from configuration."""
        boundaries = list(self.config.length_bucket_boundaries)
        buckets = []

        # First bucket: 0 to first boundary
        if boundaries:
            buckets.append(LengthBucket(min_length=0, max_length=boundaries[0]))

            # Middle buckets
            for i in range(len(boundaries) - 1):
                buckets.append(LengthBucket(
                    min_length=boundaries[i],
                    max_length=boundaries[i + 1],
                ))

            # Last bucket: last boundary to infinity
            buckets.append(LengthBucket(
                min_length=boundaries[-1],
                max_length=float("inf"),
            ))
        else:
            # Single bucket if no boundaries
            buckets.append(LengthBucket(min_length=0, max_length=float("inf")))

        return buckets

    def _create_staleness_strata(self) -> list[StalenessStratum]:
        """Create staleness strata for stratified sampling."""
        num_strata = self.config.staleness_strata
        strata = []

        for i in range(num_strata):
            strata.append(StalenessStratum(
                min_staleness=i,
                max_staleness=i + 1 if i < num_strata - 1 else float("inf"),
            ))

        return strata

    def _get_bucket_for_length(self, length: int) -> LengthBucket:
        """Find the appropriate bucket for a given length."""
        for bucket in self._buckets:
            if bucket.min_length <= length < bucket.max_length:
                return bucket
        # Default to last bucket
        return self._buckets[-1]

    def _get_stratum_for_staleness(self, staleness: int) -> StalenessStratum:
        """Find the appropriate stratum for a given staleness."""
        for stratum in self._strata:
            if stratum.min_staleness <= staleness < stratum.max_staleness:
                return stratum
        # Default to last stratum
        return self._strata[-1]

    def set_difficulty_function(
        self,
        fn: Callable[[Trajectory], float],
    ) -> None:
        """Set custom difficulty function for curriculum learning.

        Args:
            fn: Function that takes a trajectory and returns difficulty score (0-1).
        """
        self._difficulty_fn = fn

    def _compute_difficulty(self, trajectory: Trajectory) -> float:
        """Compute difficulty score for a trajectory.

        Default heuristic uses:
        - Response length (longer = harder)
        - Reward magnitude (lower reward = harder)
        """
        if self._difficulty_fn is not None:
            return self._difficulty_fn(trajectory)

        # Default difficulty heuristic
        length_score = min(1.0, trajectory.total_length / 2048)
        reward_score = 1.0 - min(1.0, max(0.0, (trajectory.reward + 1) / 2))

        return 0.6 * length_score + 0.4 * reward_score

    def _bucket_trajectories(
        self,
        trajectories: list[Trajectory],
    ) -> None:
        """Distribute trajectories into length buckets."""
        # Clear existing buckets
        for bucket in self._buckets:
            bucket.clear()

        # Distribute trajectories
        for traj in trajectories:
            bucket = self._get_bucket_for_length(traj.total_length)
            bucket.add(traj)

    def _stratify_by_staleness(
        self,
        trajectories: list[Trajectory],
    ) -> None:
        """Distribute trajectories into staleness strata."""
        # Clear existing strata
        for stratum in self._strata:
            stratum.trajectories.clear()

        # Distribute trajectories
        for traj in trajectories:
            staleness = traj.get_version_gap(self._current_version)
            stratum = self._get_stratum_for_staleness(staleness)
            stratum.trajectories.append(traj)

    def _sample_balanced_staleness(
        self,
        n: int,
    ) -> list[Trajectory]:
        """Sample trajectories with balanced staleness distribution."""
        result = []
        non_empty_strata = [s for s in self._strata if s.size > 0]

        if not non_empty_strata:
            return result

        # Calculate per-stratum quota
        per_stratum = max(1, n // len(non_empty_strata))

        for stratum in non_empty_strata:
            sample_size = min(per_stratum, stratum.size)
            sampled = random.sample(stratum.trajectories, sample_size)
            result.extend(sampled)

        # Fill remaining quota
        remaining = n - len(result)
        if remaining > 0:
            all_remaining = []
            for stratum in non_empty_strata:
                for traj in stratum.trajectories:
                    if traj not in result:
                        all_remaining.append(traj)

            if all_remaining:
                extra = random.sample(all_remaining, min(remaining, len(all_remaining)))
                result.extend(extra)

        return result[:n]

    def _apply_curriculum_ordering(
        self,
        trajectories: list[Trajectory],
    ) -> list[Trajectory]:
        """Apply curriculum learning ordering with decaying randomness.

        Early in training: More randomness, mix of difficulties
        Later in training: More structured, progressive difficulty
        """
        if not trajectories:
            return trajectories

        # Compute difficulty for each trajectory
        difficulties = [(traj, self._compute_difficulty(traj)) for traj in trajectories]

        # Sort by difficulty
        difficulties.sort(key=lambda x: x[1])

        # Compute randomness factor (decays over curriculum steps)
        decay = self.config.curriculum_randomness_decay
        randomness = 1.0 / (1.0 + decay * self._curriculum_step)

        # Apply randomness through partial shuffling
        if randomness > 0:
            sorted_trajs = [t for t, _ in difficulties]

            # Swap pairs with probability proportional to randomness
            for i in range(len(sorted_trajs) - 1):
                if random.random() < randomness:
                    j = random.randint(i, len(sorted_trajs) - 1)
                    sorted_trajs[i], sorted_trajs[j] = sorted_trajs[j], sorted_trajs[i]

            return sorted_trajs

        return [t for t, _ in difficulties]

    def compose_batch(
        self,
        trajectories: list[Trajectory],
    ) -> TrajectoryBatch:
        """Compose a single batch from trajectories.

        Args:
            trajectories: Trajectories to batch.

        Returns:
            TrajectoryBatch ready for training.
        """
        return TrajectoryBatch(trajectories=trajectories)

    def compose_batches(
        self,
        trajectories: list[Trajectory],
        current_version: int | None = None,
        shuffle: bool = True,
    ) -> Iterator[TrajectoryBatch]:
        """Compose multiple batches with smart composition strategies.

        Args:
            trajectories: All available trajectories.
            current_version: Current policy version (updates internal state).
            shuffle: Whether to shuffle within buckets.

        Yields:
            TrajectoryBatch objects ready for training.
        """
        if current_version is not None:
            self._current_version = current_version

        if not trajectories:
            return

        working_trajectories = list(trajectories)

        # Apply composition strategies
        if self.config.use_staleness_balancing:
            self._stratify_by_staleness(working_trajectories)
            working_trajectories = self._sample_balanced_staleness(len(working_trajectories))

        if self.config.use_curriculum:
            working_trajectories = self._apply_curriculum_ordering(working_trajectories)
            self._curriculum_step += 1

        if self.config.use_length_bucketing:
            # Bucket and yield batches from each bucket
            self._bucket_trajectories(working_trajectories)

            for bucket in self._buckets:
                if bucket.is_empty:
                    continue

                bucket_trajs = list(bucket.trajectories)
                if shuffle:
                    random.shuffle(bucket_trajs)

                # Yield batches from this bucket
                for i in range(0, len(bucket_trajs), self.batch_size):
                    batch_trajs = bucket_trajs[i:i + self.batch_size]
                    if batch_trajs:
                        yield self.compose_batch(batch_trajs)
        else:
            # No bucketing - just yield batches
            if shuffle:
                random.shuffle(working_trajectories)

            for i in range(0, len(working_trajectories), self.batch_size):
                batch_trajs = working_trajectories[i:i + self.batch_size]
                if batch_trajs:
                    yield self.compose_batch(batch_trajs)

    def compose_from_buffer(
        self,
        buffer: TrajectoryBuffer,
        current_version: PolicyVersion | int,
        n_batches: int = 1,
    ) -> Iterator[TrajectoryBatch]:
        """Compose batches from a trajectory buffer.

        Args:
            buffer: Trajectory buffer to sample from.
            current_version: Current policy version.
            n_batches: Number of batches to compose.

        Yields:
            TrajectoryBatch objects.
        """
        version_id = current_version.version_id if isinstance(
            current_version, PolicyVersion
        ) else current_version
        self._current_version = version_id

        # Sample trajectories for n_batches
        total_needed = n_batches * self.batch_size
        trajectories = buffer.sample(
            n=total_needed,
            current_version=current_version,
            stratified=self.config.use_staleness_balancing,
        )

        # Use compose_batches for the rest
        yield from self.compose_batches(trajectories, shuffle=True)

    def get_statistics(self) -> dict[str, Any]:
        """Get composition statistics.

        Returns:
            Dictionary of composition statistics.
        """
        bucket_stats = []
        for bucket in self._buckets:
            bucket_stats.append({
                "range": f"{bucket.min_length}-{bucket.max_length}",
                "size": bucket.size,
            })

        stratum_stats = []
        for stratum in self._strata:
            stratum_stats.append({
                "range": f"{stratum.min_staleness}-{stratum.max_staleness}",
                "size": stratum.size,
            })

        return {
            "batch_size": self.batch_size,
            "current_version": self._current_version,
            "curriculum_step": self._curriculum_step,
            "buckets": bucket_stats,
            "strata": stratum_stats,
            "config": {
                "use_length_bucketing": self.config.use_length_bucketing,
                "use_staleness_balancing": self.config.use_staleness_balancing,
                "use_curriculum": self.config.use_curriculum,
            },
        }

    def reset_curriculum(self) -> None:
        """Reset curriculum learning state."""
        self._curriculum_step = 0


class BatchIterator:
    """Iterator wrapper for batch composition with prefetching.

    Supports automatic batch prefetching for efficient training.
    """

    def __init__(
        self,
        composer: SmartBatchComposer,
        buffer: TrajectoryBuffer,
        current_version: PolicyVersion | int,
        num_batches: int | None = None,
        prefetch: int = 2,
    ) -> None:
        """Initialize batch iterator.

        Args:
            composer: SmartBatchComposer instance.
            buffer: Trajectory buffer to sample from.
            current_version: Current policy version.
            num_batches: Maximum number of batches (None for infinite).
            prefetch: Number of batches to prefetch.
        """
        self.composer = composer
        self.buffer = buffer
        self.current_version = current_version
        self.num_batches = num_batches
        self.prefetch = prefetch

        self._batch_count = 0

    def __iter__(self) -> Iterator[TrajectoryBatch]:
        """Iterate over batches."""
        self._batch_count = 0
        return self

    def __next__(self) -> TrajectoryBatch:
        """Get next batch."""
        if self.num_batches is not None and self._batch_count >= self.num_batches:
            raise StopIteration

        # Get a single batch
        batches = list(self.composer.compose_from_buffer(
            buffer=self.buffer,
            current_version=self.current_version,
            n_batches=1,
        ))

        if not batches:
            raise StopIteration

        self._batch_count += 1
        return batches[0]

    def update_version(self, version: PolicyVersion | int) -> None:
        """Update the current policy version.

        Args:
            version: New policy version.
        """
        self.current_version = version
