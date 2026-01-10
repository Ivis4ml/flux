---
title: Algorithms API
description: Algorithm registry and extension points
---

# Algorithms API

Flux uses a registry pattern for algorithms, making it easy to add custom algorithms.

## Registry Functions

### @register_adv_estimator

Register a custom advantage estimator.

```python
from flux.training.algorithms.base import register_adv_estimator

@register_adv_estimator("my_advantage")
def compute_my_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages and returns."""
    advantages = ...
    returns = ...
    return advantages, returns
```

### @register_policy_loss

Register a custom policy loss function.

```python
from flux.training.algorithms.base import register_policy_loss

@register_policy_loss("my_loss")
def compute_my_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute loss and metrics."""
    loss = ...
    metrics = {"loss": loss.item()}
    return loss, metrics
```

## Using Custom Algorithms

```yaml
algorithm:
  name: my_loss
  adv_estimator: my_advantage
```

## Built-in Estimators

| Name | Description |
|:-----|:------------|
| `grpo` | Group relative normalization |
| `grpo_vectorized` | Vectorized GRPO |
| `gae` | Generalized Advantage Estimation |
| `reinforce` | Simple reward-based |
| `rloo` | Leave-one-out baseline |

## Built-in Losses

| Name | Description |
|:-----|:------------|
| `grpo` | GRPO with clipping + KL |
| `ppo` | PPO clipped surrogate |
| `dpo` | Direct preference optimization |
| `reinforce` | Vanilla policy gradient |
| `dapo` | Decoupled clipping |

## Helper Functions

```python
from flux.training.algorithms.base import (
    masked_mean,    # Mean over masked positions
    agg_loss,       # Aggregate loss with mode
    get_adv_estimator,   # Get estimator by name
    get_policy_loss,     # Get loss by name
)
```

## See Also

- [Algorithms Guide](../algorithms/index.md)
- [Custom Algorithms](../algorithms/custom.md)
