"""Flux rollout module.

This module provides rollout generation components:
- SGLangClient: HTTP client for SGLang inference servers
- StreamingRolloutManager: APRIL-based rollout generation
- LengthPredictor: Output length prediction for scheduling
- PartialTrajectoryBuffer: Storage for incomplete trajectories
"""

from flux.rollout.sglang_client import (
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    SGLangClient,
    SGLangClientPool,
)
from flux.rollout.manager import (
    RolloutBatch,
    RolloutRequest,
    StreamingRolloutManager,
)
from flux.rollout.length_predictor import (
    LengthObservation,
    LengthPrediction,
    LengthPredictor,
)
from flux.rollout.partial_buffer import (
    PartialEntry,
    PartialTrajectoryBuffer,
)

__all__ = [
    # SGLang client
    "GenerationRequest",
    "GenerationResult",
    "GenerationStatus",
    "SGLangClient",
    "SGLangClientPool",
    # Rollout manager
    "RolloutBatch",
    "RolloutRequest",
    "StreamingRolloutManager",
    # Length prediction
    "LengthObservation",
    "LengthPrediction",
    "LengthPredictor",
    # Partial buffer
    "PartialEntry",
    "PartialTrajectoryBuffer",
]
