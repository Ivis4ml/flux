---
title: Custom Algorithms
description: Create custom RL algorithms for Flux
---

# Custom Algorithms

Learn how to implement custom algorithms in Flux.

## Registry Pattern

```python
from flux.training.algorithms.base import (
    register_adv_estimator,
    register_policy_loss,
)

@register_adv_estimator("my_advantage")
def compute_my_advantage(rewards, mask, **kwargs):
    advantages = rewards - rewards.mean()
    returns = rewards
    return advantages, returns

@register_policy_loss("my_loss")
def compute_my_loss(old_logp, logp, advantages, mask, **kwargs):
    ratio = torch.exp(logp - old_logp)
    loss = -(ratio * advantages * mask).sum() / mask.sum()
    return loss, {"loss": loss.item()}
```

## Usage

```yaml
algorithm:
  name: my_loss
  advantage_estimator: my_advantage
```

## See Also

- [Algorithms Overview](index.md)
- [How-to: Custom Algorithm](../how-to/custom-algorithm.md)
