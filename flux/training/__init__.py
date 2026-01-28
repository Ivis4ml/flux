"""Flux training module.

This module provides the training infrastructure for Flux, including:
- Native trainer contract (TrainingBackend ABC, GPUBatch)
- Algorithm registry for extensible RL algorithms
- Built-in algorithms (PPO, GRPO, REINFORCE, DPO, DAPO, GSPO, RLOO)
- Batch composition utilities
- Training backends (Transformers, Megatron)
"""

from flux.training.algorithms import (
    ADV_ESTIMATOR_REGISTRY,
    POLICY_LOSS_REGISTRY,
    AdvantageEstimator,
    register_adv_estimator,
    register_policy_loss,
    get_adv_estimator_fn,
    get_policy_loss_fn,
)
from flux.training.base import (
    GPUBatch,
    TrainStepResult,
    TrainingBackend,
    TrainingBackendBase,
    TrainingBackendType,
    create_training_backend,
)
from flux.training.batch_composer import (
    BatchIterator,
    CompositionStats,
    LengthBucket,
    SmartBatchComposer,
    StalenessStratum,
)
from flux.training.megatron_engine import (
    MegatronEngine,
    ModelState,
    TrainingEngine,
    TrainingStep,
)
from flux.training.backends import TransformersBackend

__all__ = [
    # Native trainer contract
    "GPUBatch",
    "TrainStepResult",
    "TrainingBackend",
    "TrainingBackendBase",
    "TrainingBackendType",
    "create_training_backend",
    # Algorithms
    "ADV_ESTIMATOR_REGISTRY",
    "POLICY_LOSS_REGISTRY",
    "AdvantageEstimator",
    "register_adv_estimator",
    "register_policy_loss",
    "get_adv_estimator_fn",
    "get_policy_loss_fn",
    # Batch composition
    "BatchIterator",
    "CompositionStats",
    "LengthBucket",
    "SmartBatchComposer",
    "StalenessStratum",
    # Training backends
    "TransformersBackend",
    # Legacy training engines
    "MegatronEngine",
    "ModelState",
    "TrainingEngine",
    "TrainingStep",
]
