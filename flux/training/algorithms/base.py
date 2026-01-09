"""Algorithm registry for Flux.

This module provides a VERL-style registry pattern for extensible RL algorithms.
Algorithms are registered via decorators and retrieved via config strings.

Design principles:
- Algorithm-agnostic interface: standard signatures for all algorithms
- No inheritance: pure functions with decorators
- Config-driven: algorithm selection via strings enables A/B testing
- Off-policy support: optional importance weights parameter

Registry pattern:
    ADV_ESTIMATOR_REGISTRY: Maps names to advantage estimator functions
    POLICY_LOSS_REGISTRY: Maps names to policy loss functions

Standard interfaces:
    Advantage estimators: (rewards, mask, **kwargs) -> (advantages, returns)
    Policy loss functions: (old_logp, logp, advantages, mask, **kwargs) -> (loss, metrics)
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, TypeAlias

import torch


class AdvantageEstimator(str, Enum):
    """Built-in advantage estimator algorithms.

    Use string values for config-driven algorithm selection.
    Enum provides type safety and prevents typos.
    """

    # Value-based
    GAE = "gae"

    # Group-based
    GRPO = "grpo"

    # REINFORCE variants
    REINFORCE = "reinforce"
    REINFORCE_BASELINE = "reinforce_baseline"

    # Leave-one-out
    RLOO = "rloo"

    # Direct preference
    DPO = "dpo"

    # Decoupled async
    DAPO = "dapo"

    # Group stability
    GSPO = "gspo"


# Type aliases for algorithm functions
AdvEstimatorFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor],  # rewards, mask
    tuple[torch.Tensor, torch.Tensor],  # (advantages, returns)
]

PolicyLossFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  # old_logp, logp, adv, mask
    tuple[torch.Tensor, dict[str, Any]],  # (loss, metrics)
]

# Global registries
ADV_ESTIMATOR_REGISTRY: dict[str, AdvEstimatorFn] = {}
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_adv_estimator(
    name: str | AdvantageEstimator,
) -> Callable[[AdvEstimatorFn], AdvEstimatorFn]:
    """Register an advantage estimator function.

    Can be used with string or AdvantageEstimator enum:
        @register_adv_estimator("my_estimator")
        @register_adv_estimator(AdvantageEstimator.GRPO)

    The registered function must have signature:
        def fn(rewards: Tensor, mask: Tensor, **kwargs) -> tuple[Tensor, Tensor]

    Returns (advantages, returns) tensors of same shape as input.

    Args:
        name: Algorithm name (string or enum)

    Returns:
        Decorator that registers the function
    """

    def decorator(fn: AdvEstimatorFn) -> AdvEstimatorFn:
        key = name.value if isinstance(name, AdvantageEstimator) else name

        if key in ADV_ESTIMATOR_REGISTRY:
            existing = ADV_ESTIMATOR_REGISTRY[key]
            if existing is not fn:
                raise ValueError(
                    f"Advantage estimator '{key}' already registered with {existing.__name__}. "
                    f"Cannot re-register with {fn.__name__}."
                )

        ADV_ESTIMATOR_REGISTRY[key] = fn
        return fn

    return decorator


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function.

    The registered function must have signature:
        def fn(
            old_log_prob: Tensor,
            log_prob: Tensor,
            advantages: Tensor,
            response_mask: Tensor,
            **kwargs
        ) -> tuple[Tensor, dict[str, Any]]

    Returns (loss, metrics) where loss is a scalar and metrics is a dict.

    Args:
        name: Loss function name

    Returns:
        Decorator that registers the function
    """

    def decorator(fn: PolicyLossFn) -> PolicyLossFn:
        if name in POLICY_LOSS_REGISTRY:
            existing = POLICY_LOSS_REGISTRY[name]
            if existing is not fn:
                raise ValueError(
                    f"Policy loss '{name}' already registered with {existing.__name__}. "
                    f"Cannot re-register with {fn.__name__}."
                )

        POLICY_LOSS_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name: str | AdvantageEstimator) -> AdvEstimatorFn:
    """Get a registered advantage estimator function.

    Args:
        name: Algorithm name (string or enum)

    Returns:
        The registered advantage estimator function

    Raises:
        ValueError: If name is not registered
    """
    key = name.value if isinstance(name, AdvantageEstimator) else name

    if key not in ADV_ESTIMATOR_REGISTRY:
        available = list(ADV_ESTIMATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown advantage estimator: '{key}'. "
            f"Available: {available}"
        )

    return ADV_ESTIMATOR_REGISTRY[key]


def get_policy_loss_fn(name: str) -> PolicyLossFn:
    """Get a registered policy loss function.

    Args:
        name: Loss function name

    Returns:
        The registered policy loss function

    Raises:
        ValueError: If name is not registered
    """
    if name not in POLICY_LOSS_REGISTRY:
        available = list(POLICY_LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown policy loss: '{name}'. "
            f"Available: {available}"
        )

    return POLICY_LOSS_REGISTRY[name]


def list_adv_estimators() -> list[str]:
    """List all registered advantage estimator names.

    Returns:
        List of registered algorithm names
    """
    return list(ADV_ESTIMATOR_REGISTRY.keys())


def list_policy_losses() -> list[str]:
    """List all registered policy loss names.

    Returns:
        List of registered loss function names
    """
    return list(POLICY_LOSS_REGISTRY.keys())


# Utility functions for algorithm implementations


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute masked mean of a tensor.

    Args:
        tensor: Input tensor
        mask: Binary mask (1 for valid, 0 for invalid)
        dim: Dimension to reduce. If None, reduces all dimensions.

    Returns:
        Masked mean value
    """
    if dim is None:
        return (tensor * mask).sum() / mask.sum().clamp(min=1)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)


def masked_var(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    unbiased: bool = True,
) -> torch.Tensor:
    """Compute masked variance of a tensor.

    Args:
        tensor: Input tensor
        mask: Binary mask
        dim: Dimension to reduce
        unbiased: Use Bessel's correction

    Returns:
        Masked variance value
    """
    mean = masked_mean(tensor, mask, dim=dim)
    if dim is not None:
        mean = mean.unsqueeze(dim)
    diff_sq = ((tensor - mean) ** 2) * mask

    if dim is None:
        n = mask.sum()
    else:
        n = mask.sum(dim=dim)

    if unbiased:
        n = (n - 1).clamp(min=1)

    if dim is None:
        return diff_sq.sum() / n
    return diff_sq.sum(dim=dim) / n.clamp(min=1)


def masked_whiten(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Whiten (normalize) a tensor using masked statistics.

    Args:
        tensor: Input tensor
        mask: Binary mask
        eps: Small value for numerical stability

    Returns:
        Whitened tensor: (x - mean) / (std + eps)
    """
    mean = masked_mean(tensor, mask)
    var = masked_var(tensor, mask, unbiased=False)
    std = var.sqrt()
    return (tensor - mean) / (std + eps)


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """Aggregate loss matrix to scalar.

    Args:
        loss_mat: Per-token loss matrix (batch, seq_len)
        loss_mask: Binary mask for valid tokens
        loss_agg_mode: Aggregation mode
            - "token-mean": Mean over all valid tokens
            - "seq-mean": Mean of per-sequence means
            - "sum": Sum over all valid tokens

    Returns:
        Aggregated scalar loss
    """
    if loss_agg_mode == "token-mean":
        return masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean":
        # Per-sequence mean, then batch mean
        seq_loss = masked_mean(loss_mat, loss_mask, dim=-1)
        return seq_loss.mean()
    elif loss_agg_mode == "sum":
        return (loss_mat * loss_mask).sum()
    else:
        raise ValueError(f"Unknown loss aggregation mode: {loss_agg_mode}")
