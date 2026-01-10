---
title: Add a Custom Algorithm
description: Create and register your own RL algorithm
---

# Add a Custom Algorithm

Learn how to implement custom algorithms using Flux's registry pattern.

## Overview

Flux algorithms have two parts:
1. **Advantage Estimator**: Computes advantages from rewards
2. **Policy Loss**: Computes loss from advantages

## Step 1: Create Advantage Estimator

```python
# my_algorithm.py
import torch
from flux.training.algorithms.base import register_adv_estimator

@register_adv_estimator("my_advantage")
def compute_my_advantage(
    token_level_rewards: torch.Tensor,  # (batch, seq_len)
    response_mask: torch.Tensor,        # (batch, seq_len)
    gamma: float = 1.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute custom advantages.
    
    Returns:
        advantages: (batch, seq_len)
        returns: (batch, seq_len)
    """
    # Sum rewards per sequence
    rewards = (token_level_rewards * response_mask).sum(dim=-1)
    
    # Normalize (simple baseline)
    advantages = rewards - rewards.mean()
    advantages = advantages.unsqueeze(-1) * response_mask
    
    returns = advantages.clone()
    return advantages, returns
```

## Step 2: Create Policy Loss

```python
from flux.training.algorithms.base import (
    register_policy_loss,
    masked_mean,
    agg_loss,
)

@register_policy_loss("my_loss")
def compute_my_loss(
    old_log_prob: torch.Tensor,     # (batch, seq_len)
    log_prob: torch.Tensor,         # (batch, seq_len)
    advantages: torch.Tensor,       # (batch, seq_len)
    response_mask: torch.Tensor,    # (batch, seq_len)
    clip_ratio: float = 0.2,
    importance_weights: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute custom policy loss.
    
    Returns:
        loss: Scalar loss tensor
        metrics: Dict of metrics to log
    """
    # Compute ratio
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio.clamp(-20, 20))
    
    # Clipped objective
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    loss = -torch.min(ratio * advantages, clipped * advantages)
    
    # Apply importance weights if provided
    if importance_weights is not None:
        loss = loss * importance_weights.unsqueeze(-1)
    
    # Aggregate
    loss = agg_loss(loss, response_mask, mode="token-mean")
    
    # Metrics
    with torch.no_grad():
        metrics = {
            "loss": loss.item(),
            "ratio_mean": masked_mean(ratio, response_mask).item(),
            "clip_frac": masked_mean(
                (ratio != clipped).float(), response_mask
            ).item(),
        }
    
    return loss, metrics
```

## Step 3: Register and Use

```python
# Import to register
import my_algorithm

# Use in config
from flux import FluxConfig

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    algorithm={
        "name": "my_loss",
        "adv_estimator": "my_advantage",
        "clip_ratio": 0.2,
    }
)
```

Or in YAML:

```yaml
algorithm:
  name: my_loss
  adv_estimator: my_advantage
  clip_ratio: 0.2
```

## Complete Example

```python
# custom_reinforce.py
import torch
from flux.training.algorithms.base import (
    register_adv_estimator,
    register_policy_loss,
    masked_mean,
)

@register_adv_estimator("reinforce_baseline")
def reinforce_with_baseline(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    baseline: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """REINFORCE with moving average baseline."""
    rewards = (token_level_rewards * response_mask).sum(dim=-1)
    advantages = rewards - baseline
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages.clone()

@register_policy_loss("reinforce_pg")
def reinforce_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_coef: float = 0.01,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Vanilla REINFORCE with entropy bonus."""
    # Policy gradient
    pg_loss = -log_prob * advantages.detach()
    
    # Entropy bonus
    entropy = -log_prob
    
    # Combined
    loss = masked_mean(pg_loss - entropy_coef * entropy, response_mask)
    
    metrics = {
        "pg_loss": masked_mean(pg_loss, response_mask).item(),
        "entropy": masked_mean(entropy, response_mask).item(),
    }
    
    return loss, metrics
```

## Tips

1. **Use `masked_mean`** for correct averaging over variable lengths
2. **Clamp log ratios** to prevent numerical issues
3. **Return useful metrics** for debugging
4. **Test with small batches** before full training

## See Also

- [Algorithms API](../api/algorithms.md)
- [Algorithms Overview](../algorithms/index.md)
