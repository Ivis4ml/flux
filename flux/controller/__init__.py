"""Flux controller module.

This module provides control plane components for Flux:
- Importance correction for off-policy data
- Staleness management and tracking
- Adaptive async control with PID
- Mode Gate for sync/async state machine
"""

from flux.controller.importance import (
    UnifiedImportanceCorrection,
    compute_importance_weights,
    compute_staleness_decay,
    compute_trajectory_consistency,
)
from flux.controller.staleness import (
    RolloutStats,
    StalenessManager,
    StalenessRecord,
)
from flux.controller.adaptive_async import (
    AdaptiveAsyncController,
    AdaptiveAsyncScheduler,
    ControllerRecord,
    PIDState,
)
from flux.controller.mode_gate import (
    AsyncMode,
    ModeGate,
    ModeGateConfig,
    ModeGateIntegration,
    ModeGateState,
)

__all__ = [
    # Importance correction
    "UnifiedImportanceCorrection",
    "compute_importance_weights",
    "compute_staleness_decay",
    "compute_trajectory_consistency",
    # Staleness management
    "RolloutStats",
    "StalenessManager",
    "StalenessRecord",
    # Adaptive async control
    "AdaptiveAsyncController",
    "AdaptiveAsyncScheduler",
    "ControllerRecord",
    "PIDState",
    # Mode Gate
    "AsyncMode",
    "ModeGate",
    "ModeGateConfig",
    "ModeGateIntegration",
    "ModeGateState",
]
