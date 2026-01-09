"""REINFORCE algorithm and variants.

Implements:
- Basic REINFORCE (Monte Carlo policy gradient)
- REINFORCE with baseline
- REINFORCE++ (baseline + entropy bonus)

Reference: Williams "Simple Statistical Gradient-Following Algorithms" (1992)
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


@register_adv_estimator(AdvantageEstimator.REINFORCE)
def compute_reinforce_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    whiten: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute REINFORCE (Monte Carlo) returns.

    Basic REINFORCE computes returns as discounted sum of future rewards:
        G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}

    This is the simplest advantage estimator with high variance but no bias.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        gamma: Discount factor (default: 1.0, no discounting)
        whiten: Whether to whiten returns (default: True)

    Returns:
        advantages: Discounted returns (batch, seq_len)
        returns: Same as advantages (no separate value function)
    """
    device = token_level_rewards.device
    batch_size, seq_len = token_level_rewards.shape

    with torch.no_grad():
        # Compute returns via backward pass
        returns = torch.zeros_like(token_level_rewards)
        future_return = torch.zeros(batch_size, device=device)

        for t in reversed(range(seq_len)):
            mask_t = response_mask[:, t]
            future_return = token_level_rewards[:, t] + gamma * future_return * mask_t
            returns[:, t] = future_return

        # For basic REINFORCE, advantage = return
        advantages = returns.clone()

        # Optionally whiten for stability
        if whiten:
            advantages = masked_whiten(advantages, response_mask)

    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_BASELINE)
def compute_reinforce_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    baseline: torch.Tensor | None = None,
    gamma: float = 1.0,
    whiten: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute REINFORCE with baseline.

    Subtracts a baseline from returns to reduce variance:
        A_t = G_t - b

    The baseline can be a learned value function or running average.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        baseline: Baseline values (batch, seq_len) or scalar. If None, uses
            batch mean as baseline.
        gamma: Discount factor (default: 1.0)
        whiten: Whether to whiten advantages (default: True)

    Returns:
        advantages: Returns minus baseline (batch, seq_len)
        returns: Discounted returns (batch, seq_len)
    """
    # First compute returns
    advantages, returns = compute_reinforce_advantage(
        token_level_rewards, response_mask, gamma=gamma, whiten=False
    )

    with torch.no_grad():
        # Subtract baseline
        if baseline is None:
            # Use batch mean as simple baseline
            baseline = masked_mean(returns, response_mask)

        if isinstance(baseline, (int, float)):
            baseline = torch.tensor(baseline, device=returns.device)

        if baseline.dim() == 0:
            # Scalar baseline
            advantages = returns - baseline
        else:
            # Per-token baseline
            advantages = returns - baseline

        # Optionally whiten
        if whiten:
            advantages = masked_whiten(advantages, response_mask)

    return advantages, returns


@register_adv_estimator("reinforce_plus_plus")
def compute_reinforce_pp_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    optimal_baseline: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute REINFORCE++ advantage with optimal baseline.

    REINFORCE++ uses the optimal constant baseline that minimizes variance:
        b* = E[G * ||grad log pi||^2] / E[||grad log pi||^2]

    For LLMs where we can't easily compute gradient norms, we approximate
    with the variance-minimizing baseline for each token position.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        gamma: Discount factor (default: 1.0)
        optimal_baseline: Use position-wise optimal baseline (default: True)

    Returns:
        advantages: Returns minus optimal baseline (batch, seq_len)
        returns: Discounted returns (batch, seq_len)
    """
    device = token_level_rewards.device
    batch_size, seq_len = token_level_rewards.shape

    # First compute returns
    _, returns = compute_reinforce_advantage(
        token_level_rewards, response_mask, gamma=gamma, whiten=False
    )

    with torch.no_grad():
        if optimal_baseline:
            # Compute position-wise optimal baseline
            # For each position, use mean return at that position as baseline
            baseline = torch.zeros(seq_len, device=device)
            for t in range(seq_len):
                mask_t = response_mask[:, t]
                if mask_t.sum() > 0:
                    baseline[t] = masked_mean(returns[:, t], mask_t)

            # Subtract position-wise baseline
            advantages = returns - baseline.unsqueeze(0)
        else:
            # Global mean baseline
            baseline = masked_mean(returns, response_mask)
            advantages = returns - baseline

        # Apply mask and whiten
        advantages = advantages * response_mask
        advantages = masked_whiten(advantages, response_mask)

    return advantages, returns


@register_policy_loss("reinforce")
def compute_reinforce_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    entropy_coef: float = 0.0,
    importance_weights: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE policy gradient loss.

    Basic policy gradient: L = -log(pi(a|s)) * A

    Args:
        old_log_prob: Log probabilities from behavior policy (unused but kept
            for interface compatibility)
        log_prob: Log probabilities from current policy (batch, seq_len)
        advantages: Advantage/return estimates (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        loss_agg_mode: Loss aggregation mode (default: "token-mean")
        entropy_coef: Coefficient for entropy bonus (default: 0.0)
        importance_weights: Optional importance weights for off-policy

    Returns:
        loss: Scalar policy loss
        metrics: Dict with metrics
    """
    # Policy gradient loss: -log_prob * advantage
    pg_losses = -log_prob * advantages

    # Optional entropy bonus
    entropy = -log_prob
    if entropy_coef > 0:
        pg_losses = pg_losses - entropy_coef * entropy

    # Apply importance weights if provided
    if importance_weights is not None:
        if importance_weights.dim() == 1:
            importance_weights = importance_weights.unsqueeze(-1)
        pg_losses = pg_losses * importance_weights

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode)

    # Metrics
    with torch.no_grad():
        avg_entropy = masked_mean(entropy, response_mask)
        adv_mean = masked_mean(advantages, response_mask)
        adv_std = masked_mean((advantages - adv_mean) ** 2, response_mask).sqrt()

    metrics = {
        "actor/entropy": avg_entropy.item(),
        "actor/adv_mean": adv_mean.item(),
        "actor/adv_std": adv_std.item(),
        "actor/log_prob_mean": masked_mean(log_prob, response_mask).item(),
    }

    return loss, metrics
