"""
Trajectory data structures for Flux.

This module defines the core data structures for handling trajectories (rollouts)
in the RLHF pipeline. Inspired by Slime's Sample class and AReaL's trajectory dict.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import torch

from flux.core.types import PolicyVersion, TrajectoryStatus

if TYPE_CHECKING:
    from flux.training.base import GPUBatch


@dataclass
class VersionSegment:
    """Segment of a trajectory generated with a specific policy version.

    Used when a trajectory spans multiple policy updates (partial rollout reuse).
    """

    start: int              # Start token index (inclusive)
    end: int                # End token index (exclusive)
    version: PolicyVersion  # Policy version for this segment

    @property
    def length(self) -> int:
        """Number of tokens in this segment."""
        return self.end - self.start


@dataclass
class Trajectory:
    """A single trajectory (prompt + response) in the RLHF pipeline.

    Core data structure for storing generated responses along with all metadata
    needed for training, including log probabilities, rewards, and version info.
    """

    # Identifiers
    id: str = ""                           # Unique trajectory ID
    group_id: str | None = None            # Group ID for multi-response sampling

    # Prompt
    prompt: str = ""                       # Original prompt text
    prompt_tokens: list[int] = field(default_factory=list)
    prompt_length: int = 0

    # Response
    response: str = ""                     # Generated response text
    response_tokens: list[int] = field(default_factory=list)
    response_length: int = 0

    # Combined sequence (prompt + response)
    tokens: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)  # 1 for response tokens only

    # Log probabilities (for policy gradient)
    log_probs: list[float] = field(default_factory=list)          # Current policy
    behavior_log_probs: list[float] = field(default_factory=list)  # Behavior policy

    # Rewards
    reward: float = 0.0                    # Scalar reward
    token_rewards: list[float] = field(default_factory=list)  # Token-level rewards

    # Value estimates (for critic-based methods)
    values: list[float] = field(default_factory=list)

    # Advantages and returns (computed during training)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    # Version tracking for staleness
    version: PolicyVersion = field(default_factory=lambda: PolicyVersion(version_id=0))
    version_segments: list[VersionSegment] = field(default_factory=list)

    # Status
    status: TrajectoryStatus = TrajectoryStatus.PENDING

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    generation_time_ms: float = 0.0

    @property
    def total_length(self) -> int:
        """Total sequence length (prompt + response)."""
        return len(self.tokens) if self.tokens else self.prompt_length + self.response_length

    @property
    def has_version_boundaries(self) -> bool:
        """Whether this trajectory spans multiple policy versions."""
        return len(self.version_segments) > 1

    @property
    def is_complete(self) -> bool:
        """Whether the trajectory is complete."""
        return self.status == TrajectoryStatus.COMPLETED

    def get_version_gap(self, current_version: PolicyVersion | int) -> int:
        """Compute version gap from current policy."""
        current_id = current_version.version_id if isinstance(current_version, PolicyVersion) else current_version
        return current_id - self.version.version_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for batching."""
        return {
            "id": self.id,
            "tokens": self.tokens,
            "attention_mask": self.attention_mask,
            "loss_mask": self.loss_mask,
            "log_probs": self.log_probs,
            "behavior_log_probs": self.behavior_log_probs,
            "reward": self.reward,
            "token_rewards": self.token_rewards,
            "values": self.values,
            "advantages": self.advantages,
            "returns": self.returns,
            "version": self.version.version_id,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        """Create trajectory from dictionary."""
        version = PolicyVersion(version_id=data.get("version", 0))
        return cls(
            id=data.get("id", ""),
            tokens=data.get("tokens", []),
            attention_mask=data.get("attention_mask", []),
            loss_mask=data.get("loss_mask", []),
            log_probs=data.get("log_probs", []),
            behavior_log_probs=data.get("behavior_log_probs", []),
            reward=data.get("reward", 0.0),
            token_rewards=data.get("token_rewards", []),
            values=data.get("values", []),
            advantages=data.get("advantages", []),
            returns=data.get("returns", []),
            version=version,
            prompt_length=data.get("prompt_length", 0),
            response_length=data.get("response_length", 0),
            status=TrajectoryStatus.COMPLETED,
        )


@dataclass
class PartialTrajectory(Trajectory):
    """A partially completed trajectory that can be continued.

    Used for APRIL strategy where long-tail generations are aborted
    and can be resumed in later rollouts.
    """

    # Continuation info
    continuation_prompt: str = ""          # Prompt for continuation
    start_rollout_id: str | None = None    # Original rollout ID
    continuation_count: int = 0            # Number of times continued

    # Partial generation state
    kv_cache_ref: Any = None               # Reference to KV cache (if available)

    def can_continue(self, max_continuations: int = 3) -> bool:
        """Check if this trajectory can be continued."""
        return (
            self.status == TrajectoryStatus.ABORTED and
            self.continuation_count < max_continuations
        )

    def prepare_continuation(self) -> str:
        """Prepare prompt for continuation."""
        # Concatenate original prompt + partial response
        return self.prompt + self.response


@dataclass
class TrajectoryBatch:
    """Batched trajectories for efficient training.

    Handles padding and creates tensors for training.
    """

    trajectories: list[Trajectory] = field(default_factory=list)

    # Batched tensors (computed on demand)
    _input_ids: torch.Tensor | None = None
    _attention_mask: torch.Tensor | None = None
    _loss_mask: torch.Tensor | None = None
    _log_probs: torch.Tensor | None = None
    _behavior_log_probs: torch.Tensor | None = None
    _rewards: torch.Tensor | None = None
    _advantages: torch.Tensor | None = None
    _returns: torch.Tensor | None = None
    _versions: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    @property
    def batch_size(self) -> int:
        return len(self.trajectories)

    @property
    def max_length(self) -> int:
        if not self.trajectories:
            return 0
        return max(t.total_length for t in self.trajectories)

    @property
    def tokens(self) -> list[list[int]]:
        """Token IDs for each trajectory."""
        return [t.tokens for t in self.trajectories]

    @property
    def rewards(self) -> list[float]:
        """Scalar rewards for each trajectory."""
        return [t.reward for t in self.trajectories]

    @property
    def token_rewards(self) -> list[list[float]]:
        """Per-token rewards for each trajectory."""
        return [t.token_rewards for t in self.trajectories]

    @property
    def loss_mask(self) -> list[list[int]]:
        """Loss mask for each trajectory."""
        return [t.loss_mask for t in self.trajectories]

    @property
    def behavior_log_probs(self) -> list[list[float]]:
        """Behavior policy log probs for each trajectory."""
        return [t.behavior_log_probs for t in self.trajectories]

    @property
    def num_tokens(self) -> int:
        """Total tokens across all trajectories."""
        return sum(t.total_length for t in self.trajectories)

    def to_tensors(
        self,
        device: str | torch.device = "cpu",
        pad_token_id: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Convert batch to padded tensors.

        Returns:
            Dictionary with batched tensors ready for training.
        """
        if not self.trajectories:
            return {}

        batch_size = len(self.trajectories)
        max_len = self.max_length

        # Initialize tensors
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.float)
        log_probs = torch.zeros((batch_size, max_len), dtype=torch.float)
        behavior_log_probs = torch.zeros((batch_size, max_len), dtype=torch.float)
        rewards = torch.zeros(batch_size, dtype=torch.float)
        advantages = torch.zeros((batch_size, max_len), dtype=torch.float)
        returns = torch.zeros((batch_size, max_len), dtype=torch.float)
        versions = torch.zeros(batch_size, dtype=torch.long)

        # Fill tensors
        for i, traj in enumerate(self.trajectories):
            seq_len = len(traj.tokens)
            input_ids[i, :seq_len] = torch.tensor(traj.tokens)
            attention_mask[i, :seq_len] = 1

            loss_mask_list = traj.loss_mask
            if not loss_mask_list and seq_len > 0:
                prompt_len = max(0, min(traj.prompt_length, seq_len))
                response_len = max(0, min(traj.response_length, seq_len - prompt_len))
                if response_len > 0:
                    loss_mask_list = [0] * prompt_len + [1] * response_len
                    if len(loss_mask_list) < seq_len:
                        loss_mask_list.extend([1] * (seq_len - len(loss_mask_list)))
                else:
                    loss_mask_list = [1] * seq_len

            if loss_mask_list:
                loss_mask[i, :len(loss_mask_list)] = torch.tensor(
                    loss_mask_list,
                    dtype=torch.float,
                )

            if traj.log_probs:
                log_probs[i, :len(traj.log_probs)] = torch.tensor(traj.log_probs)

            if traj.behavior_log_probs:
                behavior_log_probs[i, :len(traj.behavior_log_probs)] = torch.tensor(
                    traj.behavior_log_probs
                )

            rewards[i] = traj.reward

            if traj.advantages:
                advantages[i, :len(traj.advantages)] = torch.tensor(traj.advantages)

            if traj.returns:
                returns[i, :len(traj.returns)] = torch.tensor(traj.returns)

            versions[i] = traj.version.version_id

        # Move to device
        self._input_ids = input_ids.to(device)
        self._attention_mask = attention_mask.to(device)
        self._loss_mask = loss_mask.to(device)
        self._log_probs = log_probs.to(device)
        self._behavior_log_probs = behavior_log_probs.to(device)
        self._rewards = rewards.to(device)
        self._advantages = advantages.to(device)
        self._returns = returns.to(device)
        self._versions = versions.to(device)

        return {
            "input_ids": self._input_ids,
            "attention_mask": self._attention_mask,
            "loss_mask": self._loss_mask,
            "log_probs": self._log_probs,
            "behavior_log_probs": self._behavior_log_probs,
            "rewards": self._rewards,
            "advantages": self._advantages,
            "returns": self._returns,
            "versions": self._versions,
        }

    def as_gpu_batch(
        self,
        device: str | torch.device = "cpu",
        pad_token_id: int = 0,
        current_version: int = 0,
    ) -> "GPUBatch":
        """Convert batch to a GPUBatch for training backends.

        Args:
            device: Target device for tensors.
            pad_token_id: Token ID used for padding.
            current_version: Current policy version for computing version gaps.

        Returns:
            GPUBatch with all tensors on the specified device.

        Raises:
            ValueError: If batch is empty.
        """
        from flux.training.base import GPUBatch

        if not self.trajectories:
            raise ValueError("Cannot create GPUBatch from empty TrajectoryBatch")

        tensors = self.to_tensors(device=device, pad_token_id=pad_token_id)

        # Compute version_gaps from versions and current_version
        version_gaps = (current_version - tensors["versions"]).to(torch.long)

        return GPUBatch(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            behavior_log_probs=tensors["behavior_log_probs"],
            rewards=tensors["rewards"],
            version_gaps=version_gaps,
            loss_mask=tensors["loss_mask"],
            advantages=tensors["advantages"],
            returns=tensors["returns"],
        )

    def compute_padding_ratio(self) -> float:
        """Compute ratio of padding tokens."""
        if not self.trajectories:
            return 0.0

        total_tokens = len(self.trajectories) * self.max_length
        actual_tokens = sum(t.total_length for t in self.trajectories)
        padding_tokens = total_tokens - actual_tokens

        return padding_tokens / total_tokens if total_tokens > 0 else 0.0

    def get_version_stats(self) -> dict[str, float]:
        """Get statistics about trajectory versions."""
        if not self.trajectories:
            return {"mean_version": 0, "min_version": 0, "max_version": 0, "version_spread": 0}

        versions = [t.version.version_id for t in self.trajectories]
        return {
            "mean_version": sum(versions) / len(versions),
            "min_version": min(versions),
            "max_version": max(versions),
            "version_spread": max(versions) - min(versions),
        }


class TrajectoryBuffer:
    """Buffer for storing trajectories with staleness-aware management.

    Supports FIFO and priority-based retrieval, with staleness tracking
    for adaptive batch composition.
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_staleness: int = 5,
    ):
        """Initialize trajectory buffer.

        Args:
            max_size: Maximum number of trajectories to store.
            max_staleness: Maximum version gap before trajectories are discarded.
        """
        self.max_size = max_size
        self.max_staleness = max_staleness

        self._buffer: deque[Trajectory] = deque(maxlen=max_size)
        self._partial_buffer: deque[PartialTrajectory] = deque(maxlen=max_size // 10)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer."""
        self._buffer.append(trajectory)

    def add_batch(self, trajectories: list[Trajectory]) -> None:
        """Add multiple trajectories."""
        for traj in trajectories:
            self.add(traj)

    def add_partial(self, partial: PartialTrajectory) -> None:
        """Add a partial trajectory for potential reuse."""
        self._partial_buffer.append(partial)

    def get_available(
        self,
        current_version: PolicyVersion | int,
        max_staleness: int | None = None,
    ) -> list[Trajectory]:
        """Get all available trajectories within staleness limit.

        Args:
            current_version: Current policy version for staleness computation.
            max_staleness: Override default max staleness.

        Returns:
            List of trajectories within staleness limit.
        """
        max_stale = max_staleness if max_staleness is not None else self.max_staleness
        current_id = current_version.version_id if isinstance(current_version, PolicyVersion) else current_version

        return [
            t for t in self._buffer
            if t.get_version_gap(current_id) <= max_stale
        ]

    def sample(
        self,
        n: int,
        current_version: PolicyVersion | int,
        stratified: bool = True,
    ) -> list[Trajectory]:
        """Sample trajectories from buffer.

        Args:
            n: Number of trajectories to sample.
            current_version: Current policy version.
            stratified: If True, stratify by staleness.

        Returns:
            List of sampled trajectories.
        """
        available = self.get_available(current_version)
        if not available:
            return []

        if not stratified or len(available) <= n:
            # Simple sampling
            import random
            return random.sample(available, min(n, len(available)))

        # Stratified sampling by staleness
        current_id = current_version.version_id if isinstance(current_version, PolicyVersion) else current_version

        # Group by staleness
        staleness_groups: dict[int, list[Trajectory]] = {}
        for traj in available:
            staleness = traj.get_version_gap(current_id)
            if staleness not in staleness_groups:
                staleness_groups[staleness] = []
            staleness_groups[staleness].append(traj)

        # Sample proportionally from each group
        import random
        result = []
        per_group = max(1, n // len(staleness_groups))

        for group in staleness_groups.values():
            sampled = random.sample(group, min(per_group, len(group)))
            result.extend(sampled)

        # Fill remaining
        remaining = n - len(result)
        if remaining > 0:
            all_remaining = [t for t in available if t not in result]
            if all_remaining:
                result.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))

        return result[:n]

    def get_partials_for_prompts(
        self,
        prompt_ids: list[str],
    ) -> dict[str, PartialTrajectory]:
        """Get partial trajectories that can continue given prompts.

        Args:
            prompt_ids: List of prompt IDs to match.

        Returns:
            Dict mapping prompt_id to matching partial trajectory.
        """
        result = {}
        for partial in self._partial_buffer:
            if partial.id in prompt_ids and partial.can_continue():
                result[partial.id] = partial
        return result

    def remove_stale(self, current_version: PolicyVersion | int) -> int:
        """Remove trajectories that are too stale.

        Returns:
            Number of trajectories removed.
        """
        current_id = current_version.version_id if isinstance(current_version, PolicyVersion) else current_version

        original_len = len(self._buffer)
        self._buffer = deque(
            (t for t in self._buffer if t.get_version_gap(current_id) <= self.max_staleness),
            maxlen=self.max_size,
        )
        return original_len - len(self._buffer)

    def clear(self) -> None:
        """Clear all trajectories."""
        self._buffer.clear()
        self._partial_buffer.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        if not self._buffer:
            return {
                "size": 0,
                "partial_size": len(self._partial_buffer),
                "mean_staleness": 0,
                "max_staleness": 0,
            }

        versions = [t.version.version_id for t in self._buffer]
        max_version = max(versions) if versions else 0
        staleness_values = [max_version - v for v in versions]

        return {
            "size": len(self._buffer),
            "partial_size": len(self._partial_buffer),
            "mean_staleness": sum(staleness_values) / len(staleness_values),
            "max_staleness": max(staleness_values),
            "min_version": min(versions),
            "max_version": max_version,
        }
