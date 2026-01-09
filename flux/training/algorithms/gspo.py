"""GSPO (Group Stability Policy Optimization) algorithm.

Implements:
- Group-aware stability constraints
- Adaptive clipping based on group variance
- Entropy regularization with group normalization

GSPO extends GRPO with stability-focused optimizations to prevent
catastrophic forgetting and ensure consistent improvements across
response groups.

Reference: Inspired by DeepSeek and stability research in RL
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


@register_adv_estimator(AdvantageEstimator.GSPO)
def compute_gspo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | torch.Tensor | None = None,
    stability_weight: float = 0.1,
    normalize_by_std: bool = True,
    eps: float = 1e-6,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GSPO advantage with stability regularization.

    GSPO modifies GRPO advantages to encourage stability:
        A_gspo = A_grpo * stability_factor

    where stability_factor penalizes groups with high variance.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        index: Group indices for normalization
        stability_weight: Weight for stability regularization (default: 0.1)
        normalize_by_std: Whether to normalize by group std
        eps: Numerical stability term

    Returns:
        advantages: Stability-weighted advantages (batch, seq_len)
        returns: Same as advantages
    """
    device = token_level_rewards.device
    batch_size, seq_len = token_level_rewards.shape

    # Sum rewards over sequence
    scores = (token_level_rewards * response_mask).sum(dim=-1)

    with torch.no_grad():
        if index is None:
            index = np.arange(batch_size)
        elif isinstance(index, torch.Tensor):
            index = index.cpu().numpy()

        # Group statistics
        id_to_scores: dict[Any, list[tuple[int, torch.Tensor]]] = defaultdict(list)
        for i in range(batch_size):
            id_to_scores[index[i]].append((i, scores[i]))

        normalized_scores = torch.zeros(batch_size, device=device)
        stability_factors = torch.ones(batch_size, device=device)

        # Global statistics for stability reference
        global_mean = scores.mean()
        global_std = scores.std() + eps

        for group_id, group_items in id_to_scores.items():
            indices = [item[0] for item in group_items]
            group_scores = torch.stack([item[1] for item in group_items])

            if len(group_items) == 1:
                normalized_scores[indices[0]] = 0.0
                stability_factors[indices[0]] = 1.0
            else:
                group_mean = group_scores.mean()
                group_std = group_scores.std() + eps

                # Normalize within group
                for idx, score in zip(indices, group_scores):
                    if normalize_by_std:
                        normalized_scores[idx] = (score - group_mean) / group_std
                    else:
                        normalized_scores[idx] = score - group_mean

                # Stability factor: penalize high-variance groups
                # Higher group variance -> lower stability factor
                relative_variance = (group_std / global_std) ** 2
                stability = 1.0 / (1.0 + stability_weight * relative_variance)

                for idx in indices:
                    stability_factors[idx] = stability

        # Apply stability factors
        normalized_scores = normalized_scores * stability_factors

        # Expand to sequence
        advantages = normalized_scores.unsqueeze(-1) * response_mask

    return advantages, advantages.clone()


@register_policy_loss("gspo")
def compute_gspo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    adaptive_clip: bool = True,
    beta: float = 0.01,
    entropy_coef: float = 0.01,
    stability_coef: float = 0.1,
    loss_agg_mode: str = "token-mean",
    importance_weights: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute GSPO loss with stability regularization.

    GSPO combines:
    1. Adaptive clipping based on advantage magnitude
    2. KL penalty for policy stability
    3. Entropy bonus for exploration
    4. Stability regularization to prevent catastrophic updates

    Args:
        old_log_prob: Log probabilities from behavior policy (batch, seq_len)
        log_prob: Log probabilities from current policy (batch, seq_len)
        advantages: Advantage estimates (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        clip_ratio: Base clipping ratio (default: 0.2)
        adaptive_clip: Use advantage-based adaptive clipping (default: True)
        beta: KL penalty coefficient (default: 0.01)
        entropy_coef: Entropy bonus coefficient (default: 0.01)
        stability_coef: Stability regularization weight (default: 0.1)
        loss_agg_mode: Loss aggregation mode (default: "token-mean")
        importance_weights: Optional importance weights

    Returns:
        loss: Scalar policy loss
        metrics: Dict with metrics
    """
    # Log ratio
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    if adaptive_clip:
        # Adaptive clipping: tighter bounds for larger advantages
        with torch.no_grad():
            adv_norm = advantages.abs() / (advantages.abs().max() + 1e-8)
            # Higher advantage -> tighter clip
            adaptive_eps = clip_ratio * (1.0 - 0.5 * adv_norm)

        clip_low = 1.0 - adaptive_eps
        clip_high = 1.0 + adaptive_eps
    else:
        clip_low = 1.0 - clip_ratio
        clip_high = 1.0 + clip_ratio

    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

    # Policy gradient loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_loss1, pg_loss2)

    # KL divergence penalty
    kl_div = -log_ratio

    # Entropy bonus
    entropy = -log_prob

    # Stability regularization: penalize large ratio deviations
    stability_loss = (ratio - 1.0) ** 2

    # Combined loss
    total_losses = (
        pg_losses
        + beta * kl_div
        - entropy_coef * entropy
        + stability_coef * stability_loss
    )

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
        avg_entropy = masked_mean(entropy, response_mask)
        avg_stability = masked_mean(stability_loss, response_mask)

    metrics = {
        "actor/gspo_kl": approx_kl.item(),
        "actor/clip_frac": clip_frac.item(),
        "actor/entropy": avg_entropy.item(),
        "actor/stability_loss": avg_stability.item(),
        "actor/ratio_mean": masked_mean(ratio, response_mask).item(),
        "actor/ratio_std": (
            masked_mean((ratio - ratio.mean()) ** 2, response_mask).sqrt().item()
        ),
    }

    return loss, metrics


@register_policy_loss("gspo_conservative")
def compute_gspo_conservative_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.1,
    trust_region: float = 0.05,
    loss_agg_mode: str = "token-mean",
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Conservative GSPO with hard trust region constraint.

    More aggressive stability constraint that truncates gradients
    outside a trust region.
    """
    # Log ratio
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Trust region: zero out contributions outside region
    in_trust_region = (ratio >= 1.0 - trust_region) & (ratio <= 1.0 + trust_region)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # Policy gradient loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_loss1, pg_loss2)

    # Zero out losses outside trust region
    pg_losses = pg_losses * in_trust_region.float()

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode)

    # Metrics
    with torch.no_grad():
        trust_frac = masked_mean(in_trust_region.float(), response_mask)
        clip_frac = masked_mean(torch.ne(ratio, clipped_ratio).float(), response_mask)

    metrics = {
        "actor/trust_frac": trust_frac.item(),
        "actor/clip_frac": clip_frac.item(),
        "actor/ratio_mean": masked_mean(ratio, response_mask).item(),
    }

    return loss, metrics
