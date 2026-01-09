"""DAPO (Decoupled Async Policy Optimization) algorithm.

Implements:
- Decoupled clipping for upper and lower bounds
- Dynamic sampling for efficient exploration
- Asynchronous-friendly advantage estimation

DAPO is designed for asynchronous RL settings where samples may be
off-policy. It uses decoupled clipping ratios to handle the asymmetry
between over- and under-estimation.

Reference: Yu et al. "DAPO: Decoupled Async Policy Optimization" (2024)
"""

from __future__ import annotations

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


@register_adv_estimator(AdvantageEstimator.DAPO)
def compute_dapo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | torch.Tensor | None = None,
    version_gap: torch.Tensor | None = None,
    staleness_decay: float = 0.99,
    normalize_by_std: bool = True,
    eps: float = 1e-6,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DAPO advantage with staleness adjustment.

    DAPO extends GRPO with staleness-aware advantage scaling:
        A_dapo = A_grpo * staleness_decay^version_gap

    This reduces the influence of stale samples in asynchronous training.

    Args:
        token_level_rewards: Per-token rewards (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        index: Group indices for GRPO-style normalization
        version_gap: Policy version gap for each sample (batch,)
        staleness_decay: Decay factor per version gap (default: 0.99)
        normalize_by_std: Whether to normalize by group std
        eps: Numerical stability term

    Returns:
        advantages: Staleness-adjusted advantages (batch, seq_len)
        returns: Same as advantages
    """
    from flux.training.algorithms.grpo import compute_grpo_advantage

    # First compute GRPO advantages
    advantages, returns = compute_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        normalize_by_std=normalize_by_std,
        eps=eps,
    )

    # Apply staleness decay if version gap provided
    if version_gap is not None:
        with torch.no_grad():
            if isinstance(version_gap, np.ndarray):
                version_gap = torch.from_numpy(version_gap).to(advantages.device)

            # Staleness decay: weight = decay^gap
            staleness_weight = staleness_decay ** version_gap.float()
            staleness_weight = staleness_weight.unsqueeze(-1)  # (batch, 1)

            # Scale advantages by staleness weight
            advantages = advantages * staleness_weight
            returns = returns * staleness_weight

    return advantages, returns


@register_policy_loss("dapo")
def compute_dapo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    beta: float = 0.01,
    dynamic_sampling: bool = True,
    temperature: float = 1.0,
    loss_agg_mode: str = "token-mean",
    importance_weights: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute DAPO loss with decoupled clipping.

    DAPO uses different clipping bounds for positive and negative advantages:
        - Positive advantages: clip(ratio, 1-eps_low, 1+eps_high)
        - Negative advantages: clip(ratio, 1-eps_high, 1+eps_low)

    This asymmetric clipping helps handle the different effects of
    over- and under-estimation in asynchronous settings.

    Args:
        old_log_prob: Log probabilities from behavior policy (batch, seq_len)
        log_prob: Log probabilities from current policy (batch, seq_len)
        advantages: Advantage estimates (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        clip_ratio_low: Lower clipping bound (default: 0.2)
        clip_ratio_high: Upper clipping bound (default: 0.28)
        beta: KL penalty coefficient (default: 0.01)
        dynamic_sampling: Apply importance weighting (default: True)
        temperature: Softmax temperature for sampling weights (default: 1.0)
        loss_agg_mode: Loss aggregation mode (default: "token-mean")
        importance_weights: Optional explicit importance weights

    Returns:
        loss: Scalar policy loss
        metrics: Dict with metrics
    """
    # Log ratio
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Decoupled clipping based on advantage sign
    pos_mask = advantages >= 0
    neg_mask = ~pos_mask

    # For positive advantages: encourage higher probability
    pos_clipped = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    # For negative advantages: discourage higher probability
    neg_clipped = torch.clamp(ratio, 1.0 - clip_ratio_high, 1.0 + clip_ratio_low)

    clipped_ratio = torch.where(pos_mask, pos_clipped, neg_clipped)

    # Policy gradient loss
    pg_loss_unclipped = -advantages * ratio
    pg_loss_clipped = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_loss_unclipped, pg_loss_clipped)

    # KL penalty
    kl_div = -log_ratio

    # Combined loss
    total_losses = pg_losses + beta * kl_div

    # Dynamic sampling: weight samples by advantage magnitude
    if dynamic_sampling and importance_weights is None:
        with torch.no_grad():
            # Compute sample-level importance from advantage magnitude
            adv_magnitude = advantages.abs().sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
            sample_weights = torch.softmax(adv_magnitude / temperature, dim=0)
            # Normalize to have mean 1
            sample_weights = sample_weights * sample_weights.numel()
            importance_weights = sample_weights

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

        pos_clip_frac = masked_mean(clipped * pos_mask.float(), response_mask)
        neg_clip_frac = masked_mean(clipped * neg_mask.float(), response_mask)

    metrics = {
        "actor/dapo_kl": approx_kl.item(),
        "actor/clip_frac": clip_frac.item(),
        "actor/pos_clip_frac": pos_clip_frac.item(),
        "actor/neg_clip_frac": neg_clip_frac.item(),
        "actor/entropy": entropy.item(),
        "actor/ratio_mean": masked_mean(ratio, response_mask).item(),
    }

    return loss, metrics


@register_policy_loss("dapo_token")
def compute_dapo_token_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    token_level_clip: bool = True,
    loss_agg_mode: str = "token-mean",
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute DAPO loss with token-level clipping.

    Variant of DAPO that applies decoupled clipping at each token position
    based on the token-level advantage sign.

    This provides finer-grained control but may be noisier.
    """
    # Log ratio
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    if token_level_clip:
        # Token-level decoupled clipping
        pos_mask = advantages >= 0

        clipped_ratio = torch.where(
            pos_mask,
            torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high),
            torch.clamp(ratio, 1.0 - clip_ratio_high, 1.0 + clip_ratio_low),
        )
    else:
        # Sequence-level: use sign of summed advantage
        seq_adv = (advantages * response_mask).sum(dim=-1, keepdim=True)
        pos_mask = seq_adv >= 0

        clipped_ratio = torch.where(
            pos_mask,
            torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high),
            torch.clamp(ratio, 1.0 - clip_ratio_high, 1.0 + clip_ratio_low),
        )

    # Loss
    pg_loss_unclipped = -advantages * ratio
    pg_loss_clipped = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_loss_unclipped, pg_loss_clipped)

    loss = agg_loss(pg_losses, response_mask, loss_agg_mode)

    # Metrics
    with torch.no_grad():
        approx_kl = masked_mean(-log_ratio, response_mask)
        clip_frac = masked_mean(torch.ne(ratio, clipped_ratio).float(), response_mask)

    metrics = {
        "actor/dapo_kl": approx_kl.item(),
        "actor/clip_frac": clip_frac.item(),
    }

    return loss, metrics
