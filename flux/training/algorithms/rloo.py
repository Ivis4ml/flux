"""RLOO (Reinforced Leave-One-Out) algorithm.

Implements:
- Leave-one-out baseline estimation
- Unbiased advantage estimation without a learned value function

RLOO uses leave-one-out cross-validation to estimate the baseline,
providing unbiased gradients without requiring a separate value network.

Reference: Kool et al. "Buy 4 REINFORCE Samples, Get a Baseline for Free!" (2019)
           Ahmadian et al. "Back to Basics: Revisiting REINFORCE Style Optimization" (2024)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from flux.training.algorithms.base import (
    AdvantageEstimator,
    agg_loss,
    masked_mean,
    masked_whiten,
    register_adv_estimator,
    register_policy_loss,
)


@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | torch.Tensor | None = None,
    whiten: bool = True,
    eps: float = 1e-6,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RLOO (Leave-One-Out) advantage.

    For each sample, the baseline is the mean of all OTHER samples in its group:
        b_i = (sum_{j != i} r_j) / (n - 1)
        A_i = r_i - b_i

    This provides an unbiased baseline without a learned value function.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        index: Group indices. Samples with same index use LOO within group.
            If None, uses entire batch as one group.
        whiten: Whether to whiten advantages (default: True)
        eps: Numerical stability term

    Returns:
        advantages: LOO advantages (batch, seq_len)
        returns: Raw returns (batch, seq_len)
    """
    device = token_level_rewards.device
    batch_size, seq_len = token_level_rewards.shape

    # Sum rewards over sequence for outcome scores
    scores = (token_level_rewards * response_mask).sum(dim=-1)

    with torch.no_grad():
        if index is None:
            index = np.zeros(batch_size, dtype=np.int64)
        elif isinstance(index, torch.Tensor):
            index = index.cpu().numpy()

        # Group samples by index
        id_to_indices: dict[Any, list[int]] = defaultdict(list)
        for i in range(batch_size):
            id_to_indices[index[i]].append(i)

        advantages = torch.zeros(batch_size, device=device)

        for group_id, group_indices in id_to_indices.items():
            group_size = len(group_indices)
            group_scores = scores[group_indices]

            if group_size == 1:
                # Single sample: no LOO possible, use zero advantage
                advantages[group_indices[0]] = 0.0
            else:
                # LOO baseline: mean of all others
                total_score = group_scores.sum()
                for i, idx in enumerate(group_indices):
                    # Baseline = mean of all others = (total - self) / (n - 1)
                    loo_baseline = (total_score - scores[idx]) / (group_size - 1)
                    advantages[idx] = scores[idx] - loo_baseline

        # Expand to sequence length
        advantages_seq = advantages.unsqueeze(-1) * response_mask
        returns = scores.unsqueeze(-1) * response_mask

        if whiten:
            advantages_seq = masked_whiten(advantages_seq, response_mask)

    return advantages_seq, returns


@register_adv_estimator("rloo_vectorized")
def compute_rloo_advantage_vectorized(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int = 4,
    whiten: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized RLOO for structured batches.

    Assumes batch is organized as consecutive groups of `group_size` samples
    for the same prompt.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
            batch_size must be divisible by group_size
        response_mask: Binary mask for valid tokens
        group_size: Number of responses per prompt (default: 4)
        whiten: Whether to whiten advantages

    Returns:
        advantages: LOO advantages (batch, seq_len)
        returns: Raw returns (batch, seq_len)
    """
    batch_size, seq_len = token_level_rewards.shape

    if batch_size % group_size != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by group_size {group_size}"
        )

    num_groups = batch_size // group_size

    # Outcome scores
    scores = (token_level_rewards * response_mask).sum(dim=-1)

    with torch.no_grad():
        # Reshape to (num_groups, group_size)
        scores_grouped = scores.view(num_groups, group_size)

        # Compute LOO baselines efficiently
        # For each sample i: baseline_i = (sum - score_i) / (n - 1)
        group_sums = scores_grouped.sum(dim=1, keepdim=True)  # (num_groups, 1)

        # LOO baseline for each sample
        loo_baselines = (group_sums - scores_grouped) / (group_size - 1)

        # Advantages
        advantages_grouped = scores_grouped - loo_baselines

        # Reshape back
        advantages = advantages_grouped.view(batch_size)
        advantages_seq = advantages.unsqueeze(-1) * response_mask

        returns = scores.unsqueeze(-1) * response_mask

        if whiten:
            advantages_seq = masked_whiten(advantages_seq, response_mask)

    return advantages_seq, returns


@register_adv_estimator("rloo_token")
def compute_rloo_token_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | torch.Tensor | None = None,
    gamma: float = 1.0,
    whiten: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level RLOO with discounted returns.

    Applies LOO baseline at each token position using the
    returns-to-go from that position.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
        response_mask: Binary mask for valid tokens
        index: Group indices for LOO grouping
        gamma: Discount factor (default: 1.0)
        whiten: Whether to whiten advantages

    Returns:
        advantages: Token-level LOO advantages (batch, seq_len)
        returns: Discounted returns (batch, seq_len)
    """
    device = token_level_rewards.device
    batch_size, seq_len = token_level_rewards.shape

    with torch.no_grad():
        # Compute returns-to-go
        returns = torch.zeros_like(token_level_rewards)
        future_return = torch.zeros(batch_size, device=device)

        for t in reversed(range(seq_len)):
            mask_t = response_mask[:, t]
            future_return = token_level_rewards[:, t] + gamma * future_return * mask_t
            returns[:, t] = future_return

        if index is None:
            index = np.zeros(batch_size, dtype=np.int64)
        elif isinstance(index, torch.Tensor):
            index = index.cpu().numpy()

        # Group by index
        id_to_indices: dict[Any, list[int]] = defaultdict(list)
        for i in range(batch_size):
            id_to_indices[index[i]].append(i)

        advantages = torch.zeros_like(returns)

        # Compute LOO at each token position
        for group_id, group_indices in id_to_indices.items():
            group_size = len(group_indices)

            if group_size == 1:
                advantages[group_indices[0]] = 0.0
            else:
                group_returns = returns[group_indices]  # (group_size, seq_len)
                group_mask = response_mask[group_indices]

                for t in range(seq_len):
                    # LOO at position t
                    returns_t = group_returns[:, t]
                    mask_t = group_mask[:, t]

                    valid_count = mask_t.sum()
                    if valid_count > 1:
                        total = (returns_t * mask_t).sum()
                        for i, idx in enumerate(group_indices):
                            if mask_t[i] > 0:
                                loo_baseline = (total - returns_t[i]) / (valid_count - 1)
                                advantages[idx, t] = returns_t[i] - loo_baseline

        advantages = advantages * response_mask

        if whiten:
            advantages = masked_whiten(advantages, response_mask)

    return advantages, returns


@register_policy_loss("rloo")
def compute_rloo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    use_clipping: bool = True,
    loss_agg_mode: str = "token-mean",
    importance_weights: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute RLOO policy loss.

    Can use either vanilla REINFORCE or PPO-style clipping with RLOO advantages.

    Args:
        old_log_prob: Log probabilities from behavior policy (batch, seq_len)
        log_prob: Log probabilities from current policy (batch, seq_len)
        advantages: RLOO advantage estimates (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        clip_ratio: Clipping ratio for PPO-style loss (default: 0.2)
        use_clipping: Whether to use PPO clipping (default: True)
        loss_agg_mode: Loss aggregation mode (default: "token-mean")
        importance_weights: Optional importance weights

    Returns:
        loss: Scalar policy loss
        metrics: Dict with metrics
    """
    if use_clipping:
        # PPO-style clipping with RLOO advantages
        log_ratio = log_prob - old_log_prob
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pg_losses = torch.maximum(pg_loss1, pg_loss2)

        clip_frac = masked_mean(
            torch.ne(ratio, torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)).float(),
            response_mask,
        )
    else:
        # Vanilla REINFORCE with RLOO advantages
        pg_losses = -log_prob * advantages
        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
        clip_frac = torch.tensor(0.0)

    # Apply importance weights
    if importance_weights is not None:
        if importance_weights.dim() == 1:
            importance_weights = importance_weights.unsqueeze(-1)
        pg_losses = pg_losses * importance_weights

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode)

    # Metrics
    with torch.no_grad():
        approx_kl = masked_mean(-log_ratio, response_mask)
        entropy = masked_mean(-log_prob, response_mask)
        adv_mean = masked_mean(advantages, response_mask)
        adv_std = masked_mean((advantages - adv_mean) ** 2, response_mask).sqrt()

    metrics = {
        "actor/rloo_kl": approx_kl.item(),
        "actor/clip_frac": clip_frac.item() if isinstance(clip_frac, torch.Tensor) else clip_frac,
        "actor/entropy": entropy.item(),
        "actor/adv_mean": adv_mean.item(),
        "actor/adv_std": adv_std.item(),
        "actor/ratio_mean": masked_mean(ratio, response_mask).item(),
    }

    return loss, metrics
