---
title: Algorithm Configuration
description: RL algorithm settings
---

# Algorithm Configuration

Configure the RL algorithm.

## Basic Settings

```yaml
algorithm:
  name: grpo              # Algorithm name
  clip_range: 0.2         # PPO clip range
  entropy_coef: 0.01      # Entropy bonus
  value_coef: 0.5         # Value loss weight
  max_grad_norm: 1.0      # Gradient clipping
```

## Available Algorithms

| Name | Description |
|:-----|:------------|
| `grpo` | Group Relative Policy Optimization (default) |
| `ppo` | Proximal Policy Optimization |
| `dpo` | Direct Preference Optimization |
| `reinforce` | Vanilla REINFORCE |
| `dapo` | Decoupled PPO |
| `rloo` | Leave-One-Out baseline |

## Custom Algorithms

```yaml
algorithm:
  name: my_custom_loss
  adv_estimator: my_advantage
  clip_range: 0.2
```

## See Also

- [Algorithms Guide](../algorithms/index.md)
- [Custom Algorithms](../how-to/custom-algorithm.md)
