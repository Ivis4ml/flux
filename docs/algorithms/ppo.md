---
title: PPO
description: Proximal Policy Optimization algorithm
---

# PPO (Proximal Policy Optimization)

The classic, battle-tested algorithm for RLHF training.

## Overview

PPO prevents large policy updates using a clipped surrogate objective, ensuring stable training even with noisy rewards.

## Loss Function

$$
L^{CLIP} = -\mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

Where:
- $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ is the probability ratio
- $A_t$ is the advantage estimate
- $\epsilon$ is the clip range (default: 0.2)

## Configuration

```yaml
algorithm:
  name: ppo
  clip_range: 0.2         # Clipping parameter
  clip_range_vf: null     # Value function clip (optional)
  entropy_coef: 0.01      # Entropy bonus
  value_coef: 0.5         # Value loss weight
  kl_coef: 0.0            # KL penalty (optional)
  kl_target: 0.01         # Target KL for adaptive
  gae_lambda: 0.95        # GAE lambda
  gamma: 1.0              # Discount factor
```

## When to Use

**Best for:**
- Maximum training stability
- When you have a value function
- General-purpose RLHF

**Compared to GRPO:**
- More stable but slower
- Requires value function training
- Better for single-sample generation

## Usage

```python
from flux import FluxConfig, FluxTrainer

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    algorithm="ppo",
    algorithm_config={
        "clip_range": 0.2,
        "entropy_coef": 0.01,
        "kl_coef": 0.1,
    }
)

trainer = FluxTrainer(config)
trainer.fit(prompts="data.jsonl")
```

## Key Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `clip_range` | `0.2` | Clipping for ratio |
| `entropy_coef` | `0.01` | Entropy bonus |
| `kl_coef` | `0.0` | KL penalty weight |
| `gae_lambda` | `0.95` | GAE parameter |

## See Also

- [GRPO](grpo.md) - More sample-efficient alternative
- [Algorithms Overview](index.md)
