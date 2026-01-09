"""
Flux: An Adaptive Post-Training Framework for Large Language Models
====================================================================

Flux is a next-generation reinforcement learning post-training framework
that treats sync/async, colocated/separated, and on/off-policy as continuous
spectrums that can be dynamically optimized during training.

Key Features:
- Adaptive Async: Dynamically adjusts sync/async ratio based on staleness
- Native-First Design: Direct Megatron + SGLang integration
- Unified Importance Correction: Handles staleness, trajectory inconsistency, and replay
- Speculative Sync: Predicts long-tail rollouts, starts training early
- Multi-Dimensional Adaptation: Temperature, batch composition, compute ratio all adaptive

Example:
    >>> from flux import FluxTrainer, FluxConfig
    >>> config = FluxConfig(model_path="meta-llama/Llama-3-8B")
    >>> trainer = FluxTrainer(config)
    >>> await trainer.fit(prompts, num_steps=10000)
"""

from flux.version import __version__, __version_info__

# Core types
from flux.core.config import (
    AdaptiveAsyncConfig,
    AlgorithmConfig,
    BatchComposerConfig,
    FluxConfig,
    MegatronConfig,
    RewardConfig,
    RolloutConfig,
    SGLangConfig,
    WeightSyncConfig,
)
from flux.core.types import (
    AsyncDecision,
    BatchMetrics,
    PolicyVersion,
    StalenessMetrics,
    TrainingPhase,
    TrainingState,
)
from flux.core.trajectory import (
    PartialTrajectory,
    Trajectory,
    TrajectoryBatch,
    TrajectoryBuffer,
    VersionSegment,
)

# Main trainer
from flux.trainer import FluxTrainer, TrainingResult

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Config
    "FluxConfig",
    "AdaptiveAsyncConfig",
    "AlgorithmConfig",
    "BatchComposerConfig",
    "MegatronConfig",
    "RewardConfig",
    "RolloutConfig",
    "SGLangConfig",
    "WeightSyncConfig",
    # Types
    "AsyncDecision",
    "BatchMetrics",
    "PolicyVersion",
    "StalenessMetrics",
    "TrainingPhase",
    "TrainingState",
    # Trajectory
    "Trajectory",
    "TrajectoryBatch",
    "TrajectoryBuffer",
    "PartialTrajectory",
    "VersionSegment",
    # Trainer
    "FluxTrainer",
    "TrainingResult",
]
