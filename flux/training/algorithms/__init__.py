"""Flux algorithm registry and built-in algorithms.

This module provides:
- Registry pattern for advantage estimators and policy loss functions
- Built-in algorithms: PPO, GRPO, REINFORCE, DPO, DAPO, GSPO, RLOO

Example usage:
    # Use built-in algorithm
    from flux.training.algorithms import get_adv_estimator_fn, get_policy_loss_fn

    adv_fn = get_adv_estimator_fn("grpo")
    advantages, returns = adv_fn(rewards, mask, index=group_ids)

    loss_fn = get_policy_loss_fn("ppo")
    loss, metrics = loss_fn(old_logp, logp, advantages, mask)

    # Register custom algorithm
    from flux.training.algorithms import register_adv_estimator, register_policy_loss

    @register_adv_estimator("my_estimator")
    def my_advantage_fn(rewards, mask, **kwargs):
        advantages = ...
        returns = ...
        return advantages, returns

    @register_policy_loss("my_loss")
    def my_loss_fn(old_logp, logp, advantages, mask, **kwargs):
        loss = ...
        metrics = {"my_metric": value}
        return loss, metrics
"""

from flux.training.algorithms.base import (
    ADV_ESTIMATOR_REGISTRY,
    POLICY_LOSS_REGISTRY,
    AdvantageEstimator,
    AdvEstimatorFn,
    PolicyLossFn,
    register_adv_estimator,
    register_policy_loss,
    get_adv_estimator_fn,
    get_policy_loss_fn,
    list_adv_estimators,
    list_policy_losses,
)

# Import built-in algorithms to trigger registration
from flux.training.algorithms import ppo
from flux.training.algorithms import grpo
from flux.training.algorithms import reinforce
from flux.training.algorithms import dpo
from flux.training.algorithms import dapo
from flux.training.algorithms import gspo
from flux.training.algorithms import rloo

__all__ = [
    # Registry
    "ADV_ESTIMATOR_REGISTRY",
    "POLICY_LOSS_REGISTRY",
    # Types
    "AdvantageEstimator",
    "AdvEstimatorFn",
    "PolicyLossFn",
    # Registration decorators
    "register_adv_estimator",
    "register_policy_loss",
    # Retrieval functions
    "get_adv_estimator_fn",
    "get_policy_loss_fn",
    "list_adv_estimators",
    "list_policy_losses",
]
