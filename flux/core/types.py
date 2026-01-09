"""
Core type definitions for Flux.

This module defines the fundamental data types used throughout the Flux framework,
including training state, policy versioning, async decisions, and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import torch


class TrainingPhase(str, Enum):
    """Training phase for curriculum and adaptive behavior.

    Different phases may have different:
    - Sync/async ratios (early: more sync, late: more async)
    - Temperature schedules
    - Batch composition strategies
    """

    WARMUP = "warmup"      # Initial warmup, high sync ratio
    EARLY = "early"        # Early training (0-20%), policy changing rapidly
    MID = "mid"            # Mid training (20-70%), balanced
    LATE = "late"          # Late training (70-100%), can tolerate more async


class TrajectoryStatus(str, Enum):
    """Status of a trajectory in the pipeline."""

    PENDING = "pending"        # Waiting to be processed
    GENERATING = "generating"  # Currently being generated
    COMPLETED = "completed"    # Successfully completed
    TRUNCATED = "truncated"    # Hit max length
    ABORTED = "aborted"        # Aborted (long-tail)
    FAILED = "failed"          # Failed with error


class SyncStrategy(str, Enum):
    """Weight synchronization strategy."""

    FULL = "full"              # Full weight sync
    DELTA = "delta"            # Delta compression
    LAZY = "lazy"              # Sync only when needed


@dataclass
class PolicyVersion:
    """Tracks policy version for staleness computation.

    Each training step increments the version. Trajectories generated with
    older versions are considered "stale" and may require importance correction.
    """

    version_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PolicyVersion):
            return self.version_id == other.version_id
        if isinstance(other, int):
            return self.version_id == other
        return False

    def __lt__(self, other: PolicyVersion | int) -> bool:
        if isinstance(other, PolicyVersion):
            return self.version_id < other.version_id
        return self.version_id < other

    def __hash__(self) -> int:
        return hash(self.version_id)


@dataclass
class TrainingState:
    """Current state of the training process.

    Tracks global step, epoch, and phase information for adaptive behavior.
    """

    global_step: int = 0
    epoch: int = 0
    step_in_epoch: int = 0
    total_steps: int = 0
    phase: TrainingPhase = TrainingPhase.WARMUP

    # Policy version tracking
    current_version: PolicyVersion = field(
        default_factory=lambda: PolicyVersion(version_id=0)
    )

    # Progress metrics
    samples_seen: int = 0
    tokens_processed: int = 0

    @property
    def progress(self) -> float:
        """Training progress as a fraction [0, 1]."""
        if self.total_steps == 0:
            return 0.0
        return min(1.0, self.global_step / self.total_steps)

    def get_phase(self) -> TrainingPhase:
        """Determine training phase based on progress."""
        progress = self.progress
        if progress < 0.05:
            return TrainingPhase.WARMUP
        elif progress < 0.20:
            return TrainingPhase.EARLY
        elif progress < 0.70:
            return TrainingPhase.MID
        else:
            return TrainingPhase.LATE

    def step(self) -> None:
        """Advance training by one step."""
        self.global_step += 1
        self.step_in_epoch += 1
        self.current_version = PolicyVersion(version_id=self.global_step)
        self.phase = self.get_phase()

    def next_epoch(self) -> None:
        """Start a new epoch."""
        self.epoch += 1
        self.step_in_epoch = 0


@dataclass
class AsyncDecision:
    """Decision from the adaptive async controller.

    Determines how much async overlap to use and whether to sync.
    """

    async_ratio: float              # Fraction of async operations [0, 1]
    should_sync: bool               # Whether to trigger sync barrier
    sync_subset: list[str] | None = None  # Specific servers to sync (None = all)

    # Diagnostic info
    staleness_estimate: float = 0.0
    capacity_remaining: int = 0

    def __post_init__(self) -> None:
        # Clamp async_ratio to valid range
        self.async_ratio = max(0.0, min(1.0, self.async_ratio))


@dataclass
class StalenessMetrics:
    """Metrics for measuring data staleness.

    Staleness indicates how much the policy has changed since the data was generated.
    High staleness may require importance weight correction or sync barriers.
    """

    # Individual staleness components
    kl_divergence: float = 0.0           # KL(π_current || π_behavior)
    importance_weight_variance: float = 0.0  # Var(π_current/π_behavior)
    version_gap: float = 0.0             # current_version - data_version

    # Combined staleness score [0, 1]
    combined_staleness: float = 0.0

    # Normalization factors (for computing combined)
    kl_normalizer: float = 0.1
    iw_normalizer: float = 2.0
    max_version_gap: int = 5

    def compute_combined(self) -> float:
        """Compute combined staleness from components.

        Uses weighted combination:
        staleness = 0.4 * kl + 0.3 * iw_var + 0.3 * version_gap
        """
        kl_contrib = min(1.0, self.kl_divergence / self.kl_normalizer)
        iw_contrib = min(1.0, self.importance_weight_variance / self.iw_normalizer)
        version_contrib = min(1.0, self.version_gap / self.max_version_gap)

        self.combined_staleness = (
            0.4 * kl_contrib +
            0.3 * iw_contrib +
            0.3 * version_contrib
        )
        return self.combined_staleness


@dataclass
class BatchMetrics:
    """Metrics collected during a training batch.

    Used for logging, adaptive control, and debugging.
    """

    # Loss components
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    total_loss: float = 0.0

    # Reward statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0

    # Advantage statistics
    mean_advantage: float = 0.0
    std_advantage: float = 0.0

    # KL and entropy
    mean_kl: float = 0.0
    mean_entropy: float = 0.0

    # Importance weight statistics (for off-policy)
    mean_importance_weight: float = 1.0
    max_importance_weight: float = 1.0
    clipped_ratio: float = 0.0  # Fraction of clipped importance weights

    # Staleness
    staleness: StalenessMetrics = field(default_factory=StalenessMetrics)

    # Batch info
    batch_size: int = 0
    sequence_length: int = 0
    num_tokens: int = 0
    padding_ratio: float = 0.0  # Fraction of padding tokens

    # Timing
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "loss/policy": self.policy_loss,
            "loss/value": self.value_loss,
            "loss/entropy": self.entropy_loss,
            "loss/total": self.total_loss,
            "reward/mean": self.mean_reward,
            "reward/std": self.std_reward,
            "reward/min": self.min_reward,
            "reward/max": self.max_reward,
            "advantage/mean": self.mean_advantage,
            "advantage/std": self.std_advantage,
            "kl/mean": self.mean_kl,
            "entropy/mean": self.mean_entropy,
            "importance_weight/mean": self.mean_importance_weight,
            "importance_weight/max": self.max_importance_weight,
            "importance_weight/clipped_ratio": self.clipped_ratio,
            "staleness/combined": self.staleness.combined_staleness,
            "staleness/kl": self.staleness.kl_divergence,
            "staleness/iw_variance": self.staleness.importance_weight_variance,
            "staleness/version_gap": self.staleness.version_gap,
            "batch/size": self.batch_size,
            "batch/seq_length": self.sequence_length,
            "batch/num_tokens": self.num_tokens,
            "batch/padding_ratio": self.padding_ratio,
            "time/forward_ms": self.forward_time_ms,
            "time/backward_ms": self.backward_time_ms,
            "time/total_ms": self.total_time_ms,
        }


@dataclass
class RolloutMetrics:
    """Metrics collected during rollout generation."""

    # Counts
    num_completed: int = 0
    num_aborted: int = 0
    num_failed: int = 0
    num_truncated: int = 0

    # Length statistics
    mean_prompt_length: float = 0.0
    mean_response_length: float = 0.0
    max_response_length: int = 0

    # Timing
    mean_generation_time_ms: float = 0.0
    total_generation_time_ms: float = 0.0

    # APRIL metrics
    oversample_ratio: float = 1.0
    abort_ratio: float = 0.0
    reuse_ratio: float = 0.0

    # GPU utilization
    gpu_utilization: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of successful completions."""
        total = self.num_completed + self.num_aborted + self.num_failed
        if total == 0:
            return 0.0
        return self.num_completed / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "rollout/completed": self.num_completed,
            "rollout/aborted": self.num_aborted,
            "rollout/failed": self.num_failed,
            "rollout/truncated": self.num_truncated,
            "rollout/success_rate": self.success_rate,
            "rollout/mean_prompt_length": self.mean_prompt_length,
            "rollout/mean_response_length": self.mean_response_length,
            "rollout/max_response_length": self.max_response_length,
            "rollout/mean_time_ms": self.mean_generation_time_ms,
            "rollout/total_time_ms": self.total_generation_time_ms,
            "april/oversample_ratio": self.oversample_ratio,
            "april/abort_ratio": self.abort_ratio,
            "april/reuse_ratio": self.reuse_ratio,
            "gpu/utilization": self.gpu_utilization,
        }


# Type aliases for clarity
Tensor = torch.Tensor
DeviceType = str | torch.device
