"""
Base reward function interface for Flux.

This module defines the abstract base class for reward functions,
supporting both synchronous and asynchronous reward computation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from flux.core.trajectory import Trajectory


logger = logging.getLogger(__name__)


@dataclass
class RewardOutput:
    """Output from reward computation.

    Attributes:
        reward: Scalar reward value.
        token_rewards: Optional per-token rewards for dense supervision.
        metadata: Additional reward metadata (e.g., breakdown by component).
    """

    reward: float
    token_rewards: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate reward output."""
        if not isinstance(self.reward, (int, float)):
            raise ValueError(f"Reward must be numeric, got {type(self.reward)}")


@dataclass
class RewardBatchOutput:
    """Output from batch reward computation.

    Attributes:
        rewards: List of scalar rewards.
        token_rewards: List of per-token reward lists.
        metadata: Batch-level metadata.
    """

    rewards: list[float] = field(default_factory=list)
    token_rewards: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return len(self.rewards)

    def mean_reward(self) -> float:
        """Compute mean reward."""
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)


class RewardFunction(ABC):
    """Abstract base class for reward functions.

    Reward functions compute reward signals for trajectories, which are
    used to train the policy. Flux supports multiple reward function types:

    - Rule-based: Fast, deterministic rewards based on heuristics
    - Model-based: Neural network reward models
    - Hybrid: Combination of rule and model rewards

    Example:
        class MyRewardFunction(RewardFunction):
            def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
                # Custom reward logic
                score = analyze_response(trajectory.response)
                return RewardOutput(reward=score)
    """

    def __init__(self, name: str = "base", scale: float = 1.0, clip: float | None = None):
        """Initialize reward function.

        Args:
            name: Name of this reward function.
            scale: Scale factor to apply to rewards.
            clip: Optional clip value for rewards (clips to [-clip, clip]).
        """
        self.name = name
        self.scale = scale
        self.clip = clip

    @abstractmethod
    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute reward for a single trajectory.

        Args:
            trajectory: Trajectory to compute reward for.

        Returns:
            RewardOutput with reward value and metadata.
        """
        pass

    def compute_rewards(self, trajectories: list[Trajectory]) -> RewardBatchOutput:
        """Compute rewards for a batch of trajectories.

        Default implementation calls compute_reward for each trajectory.
        Override for more efficient batch processing.

        Args:
            trajectories: List of trajectories.

        Returns:
            RewardBatchOutput with all rewards.
        """
        rewards = []
        token_rewards = []

        for traj in trajectories:
            output = self.compute_reward(traj)
            rewards.append(output.reward)
            token_rewards.append(output.token_rewards)

        return RewardBatchOutput(
            rewards=rewards,
            token_rewards=token_rewards,
        )

    def _apply_scale_and_clip(self, reward: float) -> float:
        """Apply scaling and clipping to reward.

        Args:
            reward: Raw reward value.

        Returns:
            Scaled and optionally clipped reward.
        """
        reward = reward * self.scale
        if self.clip is not None:
            reward = max(-self.clip, min(self.clip, reward))
        return reward

    def __call__(self, trajectory: Trajectory) -> RewardOutput:
        """Compute reward with scaling and clipping.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with processed reward.
        """
        output = self.compute_reward(trajectory)
        output.reward = self._apply_scale_and_clip(output.reward)
        return output


class CompositeReward(RewardFunction):
    """Combine multiple reward functions with weights.

    Example:
        reward_fn = CompositeReward(
            rewards=[
                (LengthReward(), 0.2),
                (FormatReward(), 0.3),
                (CorrectnessReward(), 0.5),
            ]
        )
    """

    def __init__(
        self,
        rewards: list[tuple[RewardFunction, float]],
        name: str = "composite",
        normalize: bool = True,
    ):
        """Initialize composite reward.

        Args:
            rewards: List of (reward_function, weight) tuples.
            name: Name for this composite reward.
            normalize: Whether to normalize weights to sum to 1.
        """
        super().__init__(name=name)
        self.reward_functions = rewards
        self.normalize = normalize

        if normalize:
            total_weight = sum(w for _, w in rewards)
            if total_weight > 0:
                self.weights = [w / total_weight for _, w in rewards]
            else:
                self.weights = [1.0 / len(rewards)] * len(rewards)
        else:
            self.weights = [w for _, w in rewards]

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute weighted sum of all reward components.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with combined reward.
        """
        total_reward = 0.0
        metadata = {"components": {}}

        for (reward_fn, _), weight in zip(self.reward_functions, self.weights):
            output = reward_fn.compute_reward(trajectory)
            component_reward = output.reward * weight
            total_reward += component_reward
            metadata["components"][reward_fn.name] = {
                "raw": output.reward,
                "weighted": component_reward,
                "weight": weight,
            }

        return RewardOutput(reward=total_reward, metadata=metadata)


class FunctionReward(RewardFunction):
    """Wrapper to create a reward function from a callable.

    Example:
        def my_scorer(trajectory):
            return len(trajectory.response) / 100

        reward_fn = FunctionReward(my_scorer, name="length_scorer")
    """

    def __init__(
        self,
        fn: Callable[[Trajectory], float],
        name: str = "function",
        scale: float = 1.0,
        clip: float | None = None,
    ):
        """Initialize function-based reward.

        Args:
            fn: Function that takes a Trajectory and returns a float.
            name: Name for this reward function.
            scale: Scale factor for rewards.
            clip: Optional clip value.
        """
        super().__init__(name=name, scale=scale, clip=clip)
        self.fn = fn

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute reward using the wrapped function.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with computed reward.
        """
        try:
            reward = self.fn(trajectory)
            return RewardOutput(reward=float(reward))
        except Exception as e:
            logger.warning(f"Reward function {self.name} failed: {e}")
            return RewardOutput(reward=0.0, metadata={"error": str(e)})
