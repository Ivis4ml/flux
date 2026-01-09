#!/usr/bin/env python3
"""
Custom Algorithm Example

This example demonstrates how to create custom RL algorithms
using Flux's registry pattern.

Usage:
    python examples/custom_algorithm.py
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F

from flux.training.algorithms.base import (
    get_adv_estimator,
    get_policy_loss,
    list_adv_estimators,
    list_policy_losses,
    register_adv_estimator,
    register_policy_loss,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Custom Advantage Estimator
# =============================================================================


@register_adv_estimator("awac")
def awac_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AWAC-style advantage estimation with exponential weighting.

    Advantage-Weighted Actor Critic computes advantages using GAE
    and then applies exponential weighting for the policy update.

    Args:
        rewards: Reward tensor [batch, seq_len]
        values: Value estimates [batch, seq_len]
        mask: Valid token mask [batch, seq_len]
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: Exponentially weighted advantages
        returns: TD(lambda) returns
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device

    # Compute GAE advantages
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, device=device)

    # Backward pass for GAE
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = torch.zeros(batch_size, device=device)
        else:
            next_value = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_value * mask[:, t] - values[:, t]
        gae = delta + gamma * lam * mask[:, t] * gae
        advantages[:, t] = gae
        returns[:, t] = gae + values[:, t]

    # Apply exponential weighting (AWAC style)
    # Higher advantages get exponentially higher weights
    temperature = kwargs.get("awac_temperature", 1.0)
    exp_advantages = torch.exp(advantages / temperature)

    # Normalize within batch
    exp_advantages = exp_advantages / (exp_advantages.sum() + 1e-8)
    exp_advantages = exp_advantages * mask.sum()  # Rescale

    return exp_advantages, returns


@register_adv_estimator("reward_weighted")
def reward_weighted_advantage(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    baseline: str = "mean",
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple reward-weighted advantage with configurable baseline.

    Args:
        rewards: Reward tensor [batch, seq_len]
        mask: Valid token mask [batch, seq_len]
        baseline: Baseline type ("mean", "median", "none")

    Returns:
        advantages: Reward-based advantages
        returns: Same as rewards (no bootstrapping)
    """
    # Compute per-sequence rewards
    seq_rewards = (rewards * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # Compute baseline
    if baseline == "mean":
        baseline_value = seq_rewards.mean()
    elif baseline == "median":
        baseline_value = seq_rewards.median()
    else:
        baseline_value = 0.0

    # Compute advantages
    advantages = seq_rewards - baseline_value

    # Expand to sequence length
    advantages = advantages.unsqueeze(1).expand_as(rewards)

    # Normalize
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages * mask, rewards


# =============================================================================
# Custom Policy Loss
# =============================================================================


@register_policy_loss("soft_ppo")
def soft_ppo_loss(
    old_logp: torch.Tensor,
    logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio: float = 0.2,
    soft_temperature: float = 0.5,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict]:
    """Soft PPO loss with temperature-controlled clipping.

    Instead of hard clipping, uses soft sigmoid-based clipping
    for smoother gradients.

    Args:
        old_logp: Log probs from behavior policy
        logp: Log probs from current policy
        advantages: Advantage estimates
        mask: Valid token mask
        clip_ratio: Clipping ratio
        soft_temperature: Temperature for soft clipping

    Returns:
        loss: Scalar loss
        metrics: Dictionary of metrics
    """
    # Compute importance ratio
    ratio = torch.exp(logp - old_logp)

    # Soft clipping using sigmoid
    # Maps ratio to [1-clip, 1+clip] smoothly
    clip_low = 1.0 - clip_ratio
    clip_high = 1.0 + clip_ratio

    # Sigmoid-based soft clip
    soft_ratio = clip_low + (clip_high - clip_low) * torch.sigmoid(
        (ratio - 1.0) / soft_temperature
    )

    # Use soft ratio where it would clip, original otherwise
    clipped_ratio = torch.where(
        (ratio < clip_low) | (ratio > clip_high),
        soft_ratio,
        ratio,
    )

    # Compute losses
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages

    # Take minimum (pessimistic)
    loss = -torch.min(surr1, surr2)

    # Apply mask and reduce
    loss = (loss * mask).sum() / mask.sum().clamp(min=1)

    # Compute metrics
    with torch.no_grad():
        clip_fraction = ((ratio < clip_low) | (ratio > clip_high)).float().mean()
        approx_kl = 0.5 * ((logp - old_logp) ** 2).mean()

    metrics = {
        "loss": loss.item(),
        "clip_fraction": clip_fraction.item(),
        "approx_kl": approx_kl.item(),
        "mean_ratio": ratio.mean().item(),
    }

    return loss, metrics


@register_policy_loss("awr")
def awr_loss(
    old_logp: torch.Tensor,
    logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict]:
    """Advantage-Weighted Regression loss.

    Simple behavior cloning weighted by advantages.
    No clipping, relies on advantage weighting for stability.

    Args:
        old_logp: Log probs from behavior policy (unused)
        logp: Log probs from current policy
        advantages: Advantage estimates (should be positive weights)
        mask: Valid token mask
        temperature: Temperature for advantage weighting

    Returns:
        loss: Scalar loss
        metrics: Dictionary of metrics
    """
    # Compute weights from advantages
    weights = F.softmax(advantages / temperature, dim=0)

    # Weighted negative log likelihood
    loss = -(logp * weights * mask).sum() / mask.sum().clamp(min=1)

    metrics = {
        "loss": loss.item(),
        "mean_weight": weights.mean().item(),
        "max_weight": weights.max().item(),
    }

    return loss, metrics


@register_policy_loss("dual_clip")
def dual_clip_loss(
    old_logp: torch.Tensor,
    logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio: float = 0.2,
    dual_clip_coef: float = 3.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict]:
    """Dual-clip PPO loss for additional stability.

    Adds a second clipping term when advantages are negative
    to prevent the policy from moving too far in either direction.

    Args:
        old_logp: Log probs from behavior policy
        logp: Log probs from current policy
        advantages: Advantage estimates
        mask: Valid token mask
        clip_ratio: Primary clip ratio
        dual_clip_coef: Coefficient for dual clipping

    Returns:
        loss: Scalar loss
        metrics: Dictionary of metrics
    """
    ratio = torch.exp(logp - old_logp)

    clip_low = 1.0 - clip_ratio
    clip_high = 1.0 + clip_ratio

    # Standard PPO clipped objective
    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages

    # Dual clip for negative advantages
    dual_clip_value = dual_clip_coef * advantages
    surr3 = torch.where(
        advantages < 0,
        torch.max(surr1, dual_clip_value),
        surr1,
    )

    # Take minimum of all objectives
    loss = -torch.min(torch.min(surr1, surr2), surr3)
    loss = (loss * mask).sum() / mask.sum().clamp(min=1)

    with torch.no_grad():
        clip_fraction = ((ratio < clip_low) | (ratio > clip_high)).float().mean()

    metrics = {
        "loss": loss.item(),
        "clip_fraction": clip_fraction.item(),
        "mean_ratio": ratio.mean().item(),
    }

    return loss, metrics


# =============================================================================
# Demo
# =============================================================================


def demo_custom_algorithms():
    """Demonstrate custom algorithm usage."""

    logger.info("Custom Algorithm Example")
    logger.info("=" * 50)

    # List available algorithms
    logger.info("\n--- Available Advantage Estimators ---")
    for name in list_adv_estimators():
        logger.info(f"  - {name}")

    logger.info("\n--- Available Policy Losses ---")
    for name in list_policy_losses():
        logger.info(f"  - {name}")

    # Create test data
    batch_size, seq_len = 4, 10
    device = "cpu"

    rewards = torch.randn(batch_size, seq_len, device=device)
    values = torch.randn(batch_size, seq_len, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)
    mask[:, -2:] = 0  # Mask last 2 tokens

    old_logp = torch.randn(batch_size, seq_len, device=device) - 1
    logp = old_logp + torch.randn_like(old_logp) * 0.1

    # Test custom advantage estimators
    logger.info("\n--- Testing Advantage Estimators ---")

    # AWAC advantage
    awac_fn = get_adv_estimator("awac")
    adv, ret = awac_fn(rewards, values, mask, awac_temperature=0.5)
    logger.info(f"AWAC advantages: mean={adv.mean():.4f}, std={adv.std():.4f}")

    # Reward-weighted advantage
    rw_fn = get_adv_estimator("reward_weighted")
    adv, ret = rw_fn(rewards, mask, baseline="mean")
    logger.info(f"Reward-weighted advantages: mean={adv.mean():.4f}, std={adv.std():.4f}")

    # Test custom policy losses
    logger.info("\n--- Testing Policy Losses ---")

    # Soft PPO
    soft_ppo_fn = get_policy_loss("soft_ppo")
    loss, metrics = soft_ppo_fn(old_logp, logp, adv, mask, soft_temperature=0.5)
    logger.info(f"Soft PPO: loss={loss:.4f}, metrics={metrics}")

    # AWR
    awr_fn = get_policy_loss("awr")
    # AWR needs positive weights
    pos_adv = F.relu(adv) + 0.01
    loss, metrics = awr_fn(old_logp, logp, pos_adv, mask)
    logger.info(f"AWR: loss={loss:.4f}, metrics={metrics}")

    # Dual clip
    dual_fn = get_policy_loss("dual_clip")
    loss, metrics = dual_fn(old_logp, logp, adv, mask)
    logger.info(f"Dual-clip: loss={loss:.4f}, metrics={metrics}")

    # Example: Using in training config
    logger.info("\n--- Training Configuration Example ---")
    logger.info("""
To use custom algorithms in training:

```yaml
# config.yaml
algorithm:
  name: "soft_ppo"
  advantage_estimator: "awac"
  clip_ratio: 0.2
  soft_temperature: 0.5
  awac_temperature: 1.0
```

Or in Python:

```python
from flux import FluxConfig, FluxTrainer

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    algorithm={
        "name": "soft_ppo",
        "advantage_estimator": "awac",
        "clip_ratio": 0.2,
        "soft_temperature": 0.5,
    },
)

trainer = FluxTrainer(config)
```
""")


def demo_algorithm_composition():
    """Show how to compose algorithms."""

    logger.info("\n--- Algorithm Composition ---")

    # Get existing components
    grpo_adv = get_adv_estimator("grpo")
    ppo_loss = get_policy_loss("ppo")

    # Create a hybrid: GRPO advantages + PPO loss
    logger.info("Hybrid: GRPO advantage estimation + PPO policy loss")

    # Test data
    batch_size, seq_len = 8, 10
    rewards = torch.randn(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)
    old_logp = torch.randn(batch_size, seq_len) - 1
    logp = old_logp + torch.randn_like(old_logp) * 0.1

    # Compute GRPO advantages
    adv, ret = grpo_adv(rewards, mask, group_size=4)
    logger.info(f"GRPO advantages computed: shape={adv.shape}")

    # Apply PPO loss
    loss, metrics = ppo_loss(old_logp, logp, adv, mask)
    logger.info(f"PPO loss with GRPO advantages: {loss:.4f}")


if __name__ == "__main__":
    demo_custom_algorithms()
    demo_algorithm_composition()
    logger.info("\n" + "=" * 50)
    logger.info("Custom algorithm examples complete!")
