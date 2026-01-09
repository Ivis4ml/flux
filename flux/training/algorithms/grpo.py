"""GRPO (Group Relative Policy Optimization) algorithm.

Implements:
- Group-based advantage normalization
- Relative rewards within prompt groups

GRPO normalizes rewards within groups of responses to the same prompt,
providing more stable training signal for RL with LLMs.

Reference: DeepSeek-AI "DeepSeek-R1" (2024)
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
    register_adv_estimator,
    register_policy_loss,
)


@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | torch.Tensor | None = None,
    normalize_by_std: bool = True,
    eps: float = 1e-6,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GRPO (Group Relative Policy Optimization) advantage.

    GRPO computes advantages by:
    1. Grouping samples by prompt ID
    2. Computing mean/std within each group
    3. Normalizing rewards: (r - mean) / (std + eps)
    4. Replicating the normalized score across all tokens

    This provides relative advantage compared to other responses to the
    same prompt, reducing variance from prompt-specific reward scales.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        index: Group indices (batch,). Samples with same index are grouped.
            If None, each sample is its own group (falls back to vanilla).
        normalize_by_std: Whether to divide by group std (default: True)
        eps: Small value for numerical stability (default: 1e-6)

    Returns:
        advantages: Group-normalized advantages (batch, seq_len)
        returns: Same as advantages for GRPO (no value function)
    """
    device = token_level_rewards.device
    batch_size, seq_len = token_level_rewards.shape

    # Sum rewards over sequence to get outcome scores
    scores = (token_level_rewards * response_mask).sum(dim=-1)  # (batch,)

    with torch.no_grad():
        # Convert index to numpy for grouping
        if index is None:
            # No grouping: each sample is its own group
            index = np.arange(batch_size)
        elif isinstance(index, torch.Tensor):
            index = index.cpu().numpy()

        # Group scores by prompt ID
        id_to_scores: dict[Any, list[tuple[int, torch.Tensor]]] = defaultdict(list)
        for i in range(batch_size):
            id_to_scores[index[i]].append((i, scores[i]))

        # Compute group statistics and normalize
        normalized_scores = torch.zeros(batch_size, device=device)

        for group_id, group_items in id_to_scores.items():
            indices = [item[0] for item in group_items]
            group_scores = torch.stack([item[1] for item in group_items])

            if len(group_items) == 1:
                # Single sample: use zero advantage (no relative comparison)
                normalized_scores[indices[0]] = 0.0
            else:
                # Compute group statistics
                group_mean = group_scores.mean()
                group_std = group_scores.std()

                # Normalize within group
                for idx, score in zip(indices, group_scores):
                    if normalize_by_std:
                        normalized_scores[idx] = (score - group_mean) / (
                            group_std + eps
                        )
                    else:
                        normalized_scores[idx] = score - group_mean

        # Expand to sequence length (same advantage for all tokens)
        advantages = normalized_scores.unsqueeze(-1) * response_mask

    # For GRPO, returns = advantages (no value function)
    return advantages, advantages.clone()


@register_policy_loss("grpo")
def compute_grpo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    beta: float = 0.01,
    loss_agg_mode: str = "token-mean",
    importance_weights: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute GRPO policy loss.

    Combines PPO-style clipping with KL regularization:
        L = -clip(ratio, 1-eps, 1+eps) * A + beta * KL(old || new)

    Args:
        old_log_prob: Log probabilities from behavior policy (batch, seq_len)
        log_prob: Log probabilities from current policy (batch, seq_len)
        advantages: Advantage estimates (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        clip_ratio: Clipping parameter epsilon (default: 0.2)
        beta: KL penalty coefficient (default: 0.01)
        loss_agg_mode: Loss aggregation mode (default: "token-mean")
        importance_weights: Optional off-policy importance weights

    Returns:
        loss: Scalar policy loss
        metrics: Dict with metrics
    """
    # Log ratio
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # Policy gradient loss with clipping
    # For positive advantages, use min; for negative, use max
    # This is equivalent to: -min(ratio * A, clip(ratio) * A) for A > 0
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_losses = torch.where(
        advantages >= 0,
        torch.maximum(pg_loss1, pg_loss2),
        torch.minimum(pg_loss1, pg_loss2),
    )

    # KL penalty
    kl_div = -log_ratio  # Approximate KL

    # Combined loss
    total_losses = pg_losses + beta * kl_div

    # Apply importance weights
    if importance_weights is not None:
        if importance_weights.dim() == 1:
            importance_weights = importance_weights.unsqueeze(-1)
        total_losses = total_losses * importance_weights

    loss = agg_loss(total_losses, response_mask, loss_agg_mode)

    # Metrics
    with torch.no_grad():
        approx_kl = masked_mean(kl_div, response_mask)
        clipped = torch.ne(ratio, clipped_ratio).float()
        clip_frac = masked_mean(clipped, response_mask)
        entropy = masked_mean(-log_prob, response_mask)

        # Advantage statistics
        adv_mean = masked_mean(advantages, response_mask)
        adv_std = (
            masked_mean((advantages - adv_mean) ** 2, response_mask).sqrt()
        )

    metrics = {
        "actor/grpo_kl": approx_kl.item(),
        "actor/clip_frac": clip_frac.item(),
        "actor/entropy": entropy.item(),
        "actor/adv_mean": adv_mean.item(),
        "actor/adv_std": adv_std.item(),
        "actor/ratio_mean": masked_mean(ratio, response_mask).item(),
    }

    return loss, metrics


@register_adv_estimator("grpo_vectorized")
def compute_grpo_advantage_vectorized(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int = 4,
    normalize_by_std: bool = True,
    eps: float = 1e-6,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GRPO advantage computation.

    Assumes batch is structured as consecutive groups of `group_size` responses
    to the same prompt. This is more efficient than the general GRPO when
    batches are pre-organized.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
            batch_size must be divisible by group_size
        response_mask: Binary mask for valid tokens
        group_size: Number of responses per prompt (default: 4)
        normalize_by_std: Whether to divide by group std
        eps: Numerical stability term

    Returns:
        advantages: Group-normalized advantages
        returns: Same as advantages
    """
    batch_size, seq_len = token_level_rewards.shape

    if batch_size % group_size != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by group_size {group_size}"
        )

    num_groups = batch_size // group_size

    # Sum rewards over sequence
    scores = (token_level_rewards * response_mask).sum(dim=-1)  # (batch,)

    with torch.no_grad():
        # Reshape to (num_groups, group_size)
        scores_grouped = scores.view(num_groups, group_size)

        # Compute group statistics
        group_mean = scores_grouped.mean(dim=1, keepdim=True)  # (num_groups, 1)
        group_std = scores_grouped.std(dim=1, keepdim=True)  # (num_groups, 1)

        # Normalize within groups
        if normalize_by_std:
            normalized = (scores_grouped - group_mean) / (group_std + eps)
        else:
            normalized = scores_grouped - group_mean

        # Reshape back to (batch,)
        normalized_scores = normalized.view(batch_size)

        # Expand to sequence length
        advantages = normalized_scores.unsqueeze(-1) * response_mask

    return advantages, advantages.clone()
