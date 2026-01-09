"""PPO (Proximal Policy Optimization) algorithm.

Implements:
- GAE (Generalized Advantage Estimation) for advantage computation
- Clipped surrogate objective for policy loss

Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

from __future__ import annotations

from typing import Any

import torch

from flux.training.algorithms.base import (
    AdvantageEstimator,
    agg_loss,
    masked_mean,
    masked_whiten,
    register_adv_estimator,
    register_policy_loss,
)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    values: torch.Tensor | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    whiten: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE (Generalized Advantage Estimation).

    GAE provides a smooth interpolation between TD(0) and Monte Carlo
    estimation, controlled by the lambda parameter.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        values: Value function estimates (batch, seq_len). If None, uses zeros.
        gamma: Discount factor (default: 0.99)
        lam: GAE lambda for bias-variance tradeoff (default: 0.95)
        whiten: Whether to whiten advantages (default: True)

    Returns:
        advantages: GAE advantages (batch, seq_len)
        returns: Value targets (batch, seq_len)
    """
    device = token_level_rewards.device
    batch_size, seq_len = token_level_rewards.shape

    # Use zero values if not provided (reduces to MC estimation)
    if values is None:
        values = torch.zeros_like(token_level_rewards)

    with torch.no_grad():
        # Initialize for backward pass
        advantages = torch.zeros_like(token_level_rewards)
        lastgaelam = torch.zeros(batch_size, device=device)
        nextvalues = torch.zeros(batch_size, device=device)

        # Backward pass over sequence
        for t in reversed(range(seq_len)):
            mask_t = response_mask[:, t]

            # TD error: delta = r + gamma * V(s') - V(s)
            delta = (
                token_level_rewards[:, t]
                + gamma * nextvalues * mask_t
                - values[:, t]
            )

            # GAE: A = delta + gamma * lambda * A'
            lastgaelam = delta + gamma * lam * lastgaelam * mask_t
            advantages[:, t] = lastgaelam

            # Update for next iteration
            nextvalues = values[:, t] * mask_t + nextvalues * (1 - mask_t)
            lastgaelam = lastgaelam * mask_t

        # Returns = Advantages + Values
        returns = advantages + values

        # Optionally whiten advantages for training stability
        if whiten:
            advantages = masked_whiten(advantages, response_mask)

    return advantages, returns


@register_policy_loss("ppo")
def compute_ppo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    loss_agg_mode: str = "token-mean",
    importance_weights: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute PPO clipped surrogate loss.

    The PPO-clip objective prevents too large policy updates:
        L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

    Args:
        old_log_prob: Log probabilities from behavior policy (batch, seq_len)
        log_prob: Log probabilities from current policy (batch, seq_len)
        advantages: Advantage estimates (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        clip_ratio: Clipping parameter epsilon (default: 0.2)
        clip_ratio_low: Lower clip bound (default: same as clip_ratio)
        clip_ratio_high: Upper clip bound (default: same as clip_ratio)
        loss_agg_mode: Loss aggregation mode (default: "token-mean")
        importance_weights: Optional off-policy importance weights (batch,) or (batch, seq_len)

    Returns:
        loss: Scalar policy loss
        metrics: Dict with KL divergence and clip fraction
    """
    # Use symmetric clipping by default
    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio

    # Compute log ratio with numerical stability
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)

    # Probability ratio
    ratio = torch.exp(log_ratio)

    # PPO clipped objective
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
    )
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply optional importance weights for off-policy correction
    if importance_weights is not None:
        if importance_weights.dim() == 1:
            importance_weights = importance_weights.unsqueeze(-1)
        pg_losses = pg_losses * importance_weights

    # Aggregate loss
    loss = agg_loss(pg_losses, response_mask, loss_agg_mode)

    # Compute metrics
    with torch.no_grad():
        # Approximate KL divergence: KL ≈ -log_ratio (first-order approximation)
        approx_kl = masked_mean(-log_ratio, response_mask)

        # Clip fraction: how often clipping is active
        clipped = torch.gt(pg_losses2, pg_losses1).float()
        clip_frac = masked_mean(clipped, response_mask)

        # Entropy approximation (useful for monitoring)
        entropy = masked_mean(-log_prob, response_mask)

    metrics = {
        "actor/ppo_kl": approx_kl.item(),
        "actor/clip_frac": clip_frac.item(),
        "actor/entropy": entropy.item(),
        "actor/ratio_mean": masked_mean(ratio, response_mask).item(),
        "actor/ratio_std": (
            masked_mean((ratio - ratio.mean()) ** 2, response_mask).sqrt().item()
        ),
    }

    return loss, metrics


@register_policy_loss("ppo_kl")
def compute_ppo_kl_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    kl_coef: float = 0.1,
    target_kl: float = 0.01,
    loss_agg_mode: str = "token-mean",
    importance_weights: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute PPO with KL penalty instead of clipping.

    Alternative to clipped objective that adds KL divergence penalty:
        L = -ratio * A + kl_coef * KL(old || new)

    Args:
        old_log_prob: Log probabilities from behavior policy
        log_prob: Log probabilities from current policy
        advantages: Advantage estimates
        response_mask: Binary mask for valid tokens
        kl_coef: Coefficient for KL penalty (default: 0.1)
        target_kl: Target KL for adaptive coefficient (default: 0.01)
        loss_agg_mode: Loss aggregation mode
        importance_weights: Optional off-policy importance weights

    Returns:
        loss: Scalar policy loss
        metrics: Dict with KL and other metrics
    """
    # Log ratio
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Policy gradient loss
    pg_losses = -advantages * ratio

    # KL divergence penalty
    kl_div = -log_ratio  # Approximate KL

    # Combined loss
    total_losses = pg_losses + kl_coef * kl_div

    # Apply importance weights
    if importance_weights is not None:
        if importance_weights.dim() == 1:
            importance_weights = importance_weights.unsqueeze(-1)
        total_losses = total_losses * importance_weights

    loss = agg_loss(total_losses, response_mask, loss_agg_mode)

    # Metrics
    with torch.no_grad():
        approx_kl = masked_mean(kl_div, response_mask)
        entropy = masked_mean(-log_prob, response_mask)

    metrics = {
        "actor/ppo_kl": approx_kl.item(),
        "actor/entropy": entropy.item(),
        "actor/kl_coef": kl_coef,
        "actor/ratio_mean": masked_mean(ratio, response_mask).item(),
    }

    return loss, metrics
