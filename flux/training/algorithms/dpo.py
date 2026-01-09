"""DPO (Direct Preference Optimization) algorithm.

Implements:
- DPO loss for preference-based training
- DPO advantage estimator (implicit reward)

DPO directly optimizes the policy to match human preferences without
explicit reward modeling. It's an offline algorithm that requires
paired preference data.

Reference: Rafailov et al. "Direct Preference Optimization" (2023)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from flux.training.algorithms.base import (
    AdvantageEstimator,
    agg_loss,
    masked_mean,
    register_adv_estimator,
    register_policy_loss,
)


@register_adv_estimator(AdvantageEstimator.DPO)
def compute_dpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    chosen_log_probs: torch.Tensor | None = None,
    rejected_log_probs: torch.Tensor | None = None,
    ref_chosen_log_probs: torch.Tensor | None = None,
    ref_rejected_log_probs: torch.Tensor | None = None,
    beta: float = 0.1,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DPO implicit advantage.

    DPO defines implicit rewards as:
        r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))

    The advantage is the difference in implicit rewards between chosen and
    rejected responses.

    Args:
        token_level_rewards: Per-token rewards (unused, kept for interface)
        response_mask: Binary mask for valid tokens (batch, seq_len)
        chosen_log_probs: Log probs of chosen responses (batch, seq_len)
        rejected_log_probs: Log probs of rejected responses (batch, seq_len)
        ref_chosen_log_probs: Reference model log probs for chosen
        ref_rejected_log_probs: Reference model log probs for rejected
        beta: Temperature parameter (default: 0.1)

    Returns:
        advantages: Implicit reward difference (batch, seq_len)
        returns: Same as advantages
    """
    # DPO requires preference pairs
    if chosen_log_probs is None or rejected_log_probs is None:
        # Fall back to using rewards directly
        return token_level_rewards, token_level_rewards

    device = token_level_rewards.device
    batch_size, seq_len = token_level_rewards.shape

    with torch.no_grad():
        # Compute log ratios
        chosen_ratio = chosen_log_probs
        rejected_ratio = rejected_log_probs

        if ref_chosen_log_probs is not None:
            chosen_ratio = chosen_log_probs - ref_chosen_log_probs
        if ref_rejected_log_probs is not None:
            rejected_ratio = rejected_log_probs - ref_rejected_log_probs

        # Implicit rewards
        chosen_rewards = beta * chosen_ratio
        rejected_rewards = beta * rejected_ratio

        # Sum over sequence for outcome-level rewards
        chosen_reward = (chosen_rewards * response_mask).sum(dim=-1, keepdim=True)
        rejected_reward = (rejected_rewards * response_mask).sum(dim=-1, keepdim=True)

        # Advantage is the reward margin
        advantage = chosen_reward - rejected_reward

        # Expand to sequence length
        advantages = advantage.expand(-1, seq_len) * response_mask

    return advantages, advantages.clone()


@register_policy_loss("dpo")
def compute_dpo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    chosen_log_probs: torch.Tensor | None = None,
    rejected_log_probs: torch.Tensor | None = None,
    ref_chosen_log_probs: torch.Tensor | None = None,
    ref_rejected_log_probs: torch.Tensor | None = None,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_agg_mode: str = "seq-mean",
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute DPO loss.

    DPO loss directly optimizes the policy to prefer chosen over rejected:
        L = -log(sigmoid(beta * (log_pi(y_w) - log_pi_ref(y_w) -
                                 log_pi(y_l) + log_pi_ref(y_l))))

    Args:
        old_log_prob: Unused (kept for interface)
        log_prob: Unused (kept for interface)
        advantages: Unused (DPO computes its own)
        response_mask: Binary mask for valid tokens (batch/2, seq_len)
            First half is chosen, second half is rejected for paired data
        chosen_log_probs: Log probs of chosen responses under policy
        rejected_log_probs: Log probs of rejected responses under policy
        ref_chosen_log_probs: Log probs under reference model (chosen)
        ref_rejected_log_probs: Log probs under reference model (rejected)
        beta: Temperature parameter (default: 0.1)
        label_smoothing: Label smoothing for soft labels (default: 0.0)
        loss_agg_mode: Loss aggregation mode (default: "seq-mean")

    Returns:
        loss: Scalar DPO loss
        metrics: Dict with accuracy, reward margins, etc.
    """
    if chosen_log_probs is None or rejected_log_probs is None:
        raise ValueError("DPO loss requires chosen_log_probs and rejected_log_probs")

    # Sum log probs over sequence
    chosen_logps = (chosen_log_probs * response_mask).sum(dim=-1)
    rejected_logps = (rejected_log_probs * response_mask).sum(dim=-1)

    # Reference model log probs (if provided)
    if ref_chosen_log_probs is not None:
        ref_chosen_logps = (ref_chosen_log_probs * response_mask).sum(dim=-1)
    else:
        ref_chosen_logps = torch.zeros_like(chosen_logps)

    if ref_rejected_log_probs is not None:
        ref_rejected_logps = (ref_rejected_log_probs * response_mask).sum(dim=-1)
    else:
        ref_rejected_logps = torch.zeros_like(rejected_logps)

    # DPO reward margin
    pi_logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)

    # DPO loss with optional label smoothing
    if label_smoothing > 0:
        # Soft labels: target = 1 - eps for chosen, eps for rejected
        target = 1.0 - label_smoothing
        losses = (
            -F.logsigmoid(logits) * target
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        losses = -F.logsigmoid(logits)

    loss = losses.mean()

    # Metrics
    with torch.no_grad():
        # Implicit rewards
        chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (rejected_logps - ref_rejected_logps)
        reward_margin = chosen_rewards - rejected_rewards

        # Accuracy: how often policy prefers chosen
        accuracy = (logits > 0).float().mean()

    metrics = {
        "dpo/loss": loss.item(),
        "dpo/accuracy": accuracy.item(),
        "dpo/chosen_reward": chosen_rewards.mean().item(),
        "dpo/rejected_reward": rejected_rewards.mean().item(),
        "dpo/reward_margin": reward_margin.mean().item(),
        "dpo/logits_mean": logits.mean().item(),
        "dpo/logits_std": logits.std().item(),
    }

    return loss, metrics


@register_policy_loss("ipo")
def compute_ipo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    chosen_log_probs: torch.Tensor | None = None,
    rejected_log_probs: torch.Tensor | None = None,
    ref_chosen_log_probs: torch.Tensor | None = None,
    ref_rejected_log_probs: torch.Tensor | None = None,
    beta: float = 0.1,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute IPO (Identity Preference Optimization) loss.

    IPO uses a squared loss instead of log-sigmoid:
        L = (logits - 1/(2*beta))^2

    This can be more stable than DPO in some cases.

    Reference: Azar et al. "A General Theoretical Paradigm to Understand
               Learning from Human Feedback" (2023)
    """
    if chosen_log_probs is None or rejected_log_probs is None:
        raise ValueError("IPO loss requires chosen_log_probs and rejected_log_probs")

    # Sum log probs
    chosen_logps = (chosen_log_probs * response_mask).sum(dim=-1)
    rejected_logps = (rejected_log_probs * response_mask).sum(dim=-1)

    if ref_chosen_log_probs is not None:
        ref_chosen_logps = (ref_chosen_log_probs * response_mask).sum(dim=-1)
    else:
        ref_chosen_logps = torch.zeros_like(chosen_logps)

    if ref_rejected_log_probs is not None:
        ref_rejected_logps = (ref_rejected_log_probs * response_mask).sum(dim=-1)
    else:
        ref_rejected_logps = torch.zeros_like(rejected_logps)

    # Log ratios
    pi_logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = pi_logratios - ref_logratios

    # IPO loss: squared distance from target margin
    target = 1.0 / (2.0 * beta)
    losses = (logits - target) ** 2

    loss = losses.mean()

    # Metrics
    with torch.no_grad():
        accuracy = (logits > 0).float().mean()
        chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (rejected_logps - ref_rejected_logps)

    metrics = {
        "ipo/loss": loss.item(),
        "ipo/accuracy": accuracy.item(),
        "ipo/chosen_reward": chosen_rewards.mean().item(),
        "ipo/rejected_reward": rejected_rewards.mean().item(),
        "ipo/logits_mean": logits.mean().item(),
    }

    return loss, metrics
