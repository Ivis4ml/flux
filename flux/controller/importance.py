"""Unified importance correction for off-policy data.

This module provides importance weight computation for handling off-policy
data in asynchronous RL training. It combines:
1. Standard importance sampling weights (behavior vs current policy)
2. Staleness decay based on policy version gap
3. Trajectory consistency weighting

Reference: AReaL patterns for staleness-aware importance weighting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from flux.core.types import PolicyVersion


@dataclass
class ImportanceCorrectionConfig:
    """Configuration for importance correction.

    Formula (per-trajectory):
        log_ratio = current_logprobs - behavior_logprobs         # [seq_len]
        mean_log_ratio = (log_ratio * mask).sum() / seq_len      # scalar
        base_weight = exp(clip(mean_log_ratio, -20, 20))         # scalar

        staleness_weight = staleness_decay ** version_gap        # default decay = 0.99

        importance_weight = base_weight * staleness_weight
        importance_weight = clip(importance_weight, min=0.2, max=5.0)

        # Normalize to preserve gradient scale
        importance_weight = importance_weight * (batch_size / sum(importance_weight))

    Attributes:
        staleness_decay: Decay factor per version gap (default: 0.99)
        max_importance_weight: Maximum importance weight for clipping (default: 5.0)
        min_importance_weight: Minimum importance weight for clipping (default: 0.2)
        use_staleness: Whether to apply staleness-based decay (default: True)
        use_consistency: Whether to weight by trajectory consistency (default: True)
        consistency_threshold: Min consistency to include sample (default: 0.5)
        normalize_weights: Whether to normalize weights to sum to batch size (default: True)
    """

    staleness_decay: float = 0.99
    max_importance_weight: float = 5.0
    min_importance_weight: float = 0.2
    use_staleness: bool = True
    use_consistency: bool = True
    consistency_threshold: float = 0.5
    normalize_weights: bool = True


def compute_importance_weights(
    behavior_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    version_gap: torch.Tensor | None = None,
    staleness_decay: float = 0.99,
    max_weight: float = 5.0,
    min_weight: float = 0.2,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute importance sampling weights for off-policy correction.

    The importance weight is:
        w = pi_current(a|s) / pi_behavior(a|s) * staleness_decay^version_gap

    For sequence models, we compute per-sequence weights as the product
    (or mean of log ratios) over tokens.

    Args:
        behavior_log_probs: Log probs under behavior policy (batch, seq_len)
        current_log_probs: Log probs under current policy (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        version_gap: Policy version difference for each sample (batch,)
        staleness_decay: Decay factor per version gap (default: 0.99)
        max_weight: Maximum weight after clipping (default: 5.0)
        min_weight: Minimum weight after clipping (default: 0.2)
        normalize: Normalize weights to sum to batch size (default: True)

    Returns:
        importance_weights: Per-sample importance weights (batch,)
    """
    # Compute log importance ratio per token
    log_ratio = current_log_probs - behavior_log_probs

    # Sum log ratios over sequence (product of ratios)
    seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
    log_importance = (log_ratio * response_mask).sum(dim=-1)

    # Convert to importance weight
    # Use mean log ratio to prevent extreme weights for long sequences
    mean_log_ratio = log_importance / seq_lengths
    importance_weight = torch.exp(mean_log_ratio.clamp(min=-20.0, max=20.0))

    # Apply staleness decay
    if version_gap is not None:
        staleness_weight = staleness_decay ** version_gap.float()
        importance_weight = importance_weight * staleness_weight

    # Clip weights for stability
    importance_weight = importance_weight.clamp(min=min_weight, max=max_weight)

    # Optionally normalize to maintain expected gradient magnitude
    if normalize:
        importance_weight = importance_weight * (
            importance_weight.numel() / importance_weight.sum().clamp(min=1e-8)
        )

    return importance_weight


def compute_staleness_decay(
    version_gap: torch.Tensor,
    decay_rate: float = 0.99,
    min_weight: float = 0.1,
) -> torch.Tensor:
    """Compute staleness-based decay weights.

    Older samples (larger version gap) receive lower weight:
        weight = decay_rate^version_gap

    Args:
        version_gap: Policy version difference for each sample (batch,)
        decay_rate: Decay factor per version (default: 0.99)
        min_weight: Minimum weight to maintain (default: 0.1)

    Returns:
        staleness_weights: Per-sample staleness weights (batch,)
    """
    weights = decay_rate ** version_gap.float()
    return weights.clamp(min=min_weight)


def compute_trajectory_consistency(
    behavior_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    method: str = "kl",
) -> torch.Tensor:
    """Compute trajectory consistency score.

    Measures how consistent a trajectory is with the current policy.
    High consistency indicates the trajectory is still relevant for training.

    Args:
        behavior_log_probs: Log probs under behavior policy (batch, seq_len)
        current_log_probs: Log probs under current policy (batch, seq_len)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        method: Consistency method - "kl", "ratio", or "cosine" (default: "kl")

    Returns:
        consistency_scores: Per-sample consistency in [0, 1] (batch,)
    """
    if method == "kl":
        # KL divergence-based consistency
        # Higher KL = lower consistency
        kl_per_token = behavior_log_probs - current_log_probs  # Approximate KL
        seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
        mean_kl = (kl_per_token * response_mask).sum(dim=-1) / seq_lengths

        # Convert to consistency score (higher is better)
        consistency = torch.exp(-mean_kl.abs())

    elif method == "ratio":
        # Ratio-based consistency
        # Ratios close to 1 = high consistency
        log_ratio = current_log_probs - behavior_log_probs
        mean_log_ratio = (log_ratio * response_mask).sum(dim=-1) / response_mask.sum(
            dim=-1
        ).clamp(min=1)
        ratio = torch.exp(mean_log_ratio.clamp(min=-10.0, max=10.0))

        # Consistency = 1 - |ratio - 1| / max_deviation
        deviation = (ratio - 1.0).abs()
        consistency = 1.0 / (1.0 + deviation)

    elif method == "cosine":
        # Cosine similarity of log prob distributions
        behavior_norm = (behavior_log_probs * response_mask).pow(2).sum(dim=-1).sqrt()
        current_norm = (current_log_probs * response_mask).pow(2).sum(dim=-1).sqrt()
        dot_product = (behavior_log_probs * current_log_probs * response_mask).sum(
            dim=-1
        )

        consistency = dot_product / (behavior_norm * current_norm).clamp(min=1e-8)
        consistency = (consistency + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

    else:
        raise ValueError(f"Unknown consistency method: {method}")

    return consistency.clamp(min=0.0, max=1.0)


@dataclass
class UnifiedImportanceCorrection:
    """Unified importance correction for off-policy RL.

    Combines multiple correction factors:
    1. Importance sampling: pi_new / pi_old
    2. Staleness decay: decay^version_gap
    3. Trajectory consistency: similarity to current policy

    Example usage:
        corrector = UnifiedImportanceCorrection()
        weights = corrector.compute_weights(
            behavior_log_probs=old_logp,
            current_log_probs=new_logp,
            response_mask=mask,
            version_gap=gaps,
        )
        # Use weights in loss computation
        loss = (pg_loss * weights.unsqueeze(-1)).mean()
    """

    config: ImportanceCorrectionConfig = field(
        default_factory=ImportanceCorrectionConfig
    )

    def compute_weights(
        self,
        behavior_log_probs: torch.Tensor,
        current_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        version_gap: torch.Tensor | None = None,
        current_version: int | None = None,
        behavior_versions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute unified importance correction weights.

        Args:
            behavior_log_probs: Log probs under behavior policy (batch, seq_len)
            current_log_probs: Log probs under current policy (batch, seq_len)
            response_mask: Binary mask for valid tokens (batch, seq_len)
            version_gap: Explicit version gaps (batch,). If None, computed from
                current_version and behavior_versions.
            current_version: Current policy version (int)
            behavior_versions: Version when each sample was collected (batch,)

        Returns:
            weights: Per-sample importance weights (batch,)
        """
        batch_size = behavior_log_probs.shape[0]
        device = behavior_log_probs.device

        # Compute version gap if not provided
        if version_gap is None and current_version is not None:
            if behavior_versions is not None:
                version_gap = current_version - behavior_versions.float()
            else:
                version_gap = torch.zeros(batch_size, device=device)

        # 1. Base importance sampling weight
        weights = compute_importance_weights(
            behavior_log_probs=behavior_log_probs,
            current_log_probs=current_log_probs,
            response_mask=response_mask,
            version_gap=version_gap if self.config.use_staleness else None,
            staleness_decay=self.config.staleness_decay,
            max_weight=self.config.max_importance_weight,
            min_weight=self.config.min_importance_weight,
            normalize=False,  # Normalize at the end
        )

        # 2. Trajectory consistency weighting
        if self.config.use_consistency:
            consistency = compute_trajectory_consistency(
                behavior_log_probs=behavior_log_probs,
                current_log_probs=current_log_probs,
                response_mask=response_mask,
                method="kl",
            )

            # Zero out samples below consistency threshold
            consistency_mask = consistency >= self.config.consistency_threshold
            weights = weights * consistency_mask.float()

            # Optionally weight by consistency
            weights = weights * consistency

        # 3. Normalize to maintain gradient scale
        if self.config.normalize_weights:
            total_weight = weights.sum().clamp(min=1e-8)
            weights = weights * (batch_size / total_weight)

        return weights

    def filter_samples(
        self,
        behavior_log_probs: torch.Tensor,
        current_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        version_gap: torch.Tensor | None = None,
        min_weight: float = 0.1,
    ) -> torch.Tensor:
        """Return mask of samples to include in training.

        Filters out samples with very low importance weights or consistency.

        Args:
            behavior_log_probs: Log probs under behavior policy
            current_log_probs: Log probs under current policy
            response_mask: Binary mask for valid tokens
            version_gap: Policy version gaps
            min_weight: Minimum weight to include sample (default: 0.1)

        Returns:
            include_mask: Boolean mask (batch,) of samples to include
        """
        weights = self.compute_weights(
            behavior_log_probs=behavior_log_probs,
            current_log_probs=current_log_probs,
            response_mask=response_mask,
            version_gap=version_gap,
        )

        return weights >= min_weight

    def get_metrics(
        self,
        weights: torch.Tensor,
        version_gap: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute metrics about importance weights.

        Args:
            weights: Computed importance weights (batch,)
            version_gap: Policy version gaps (batch,)

        Returns:
            metrics: Dict of importance correction metrics
        """
        metrics = {
            "importance/mean_weight": weights.mean().item(),
            "importance/std_weight": weights.std().item(),
            "importance/min_weight": weights.min().item(),
            "importance/max_weight": weights.max().item(),
            "importance/effective_samples": (
                weights.sum() ** 2 / (weights ** 2).sum()
            ).item(),
        }

        if version_gap is not None:
            metrics["importance/mean_version_gap"] = version_gap.float().mean().item()
            metrics["importance/max_version_gap"] = version_gap.max().item()

        return metrics
