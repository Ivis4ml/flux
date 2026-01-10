---
title: REINFORCE
description: Vanilla policy gradient algorithm
---

# REINFORCE

The simplest policy gradient algorithm - good for baselines and debugging.

## Overview

REINFORCE uses reward-weighted log probabilities to update the policy. Simple but high variance.

## Loss Function

$$
L = -\log \pi_\theta(a|s) \cdot (R - b)
$$

Where:
- $\pi_\theta(a|s)$ = policy probability
- $R$ = total reward
- $b$ = baseline (reduces variance)

## Configuration

```yaml
algorithm:
  name: reinforce
  baseline: moving_average   # none, mean, moving_average
  baseline_decay: 0.99       # For moving average
  entropy_coef: 0.01         # Entropy bonus
```

## When to Use

**Best for:**
- Simple baselines
- Debugging reward functions
- Educational purposes

**Limitations:**
- High variance
- Slower convergence than PPO/GRPO
- No clipping protection

## Usage

```python
from flux import FluxConfig, FluxTrainer

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    algorithm="reinforce",
    algorithm_config={
        "baseline": "moving_average",
        "entropy_coef": 0.01,
    }
)

trainer = FluxTrainer(config)
trainer.fit(prompts="data.jsonl")
```

## Key Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `baseline` | `"mean"` | Baseline type |
| `baseline_decay` | `0.99` | EMA decay for moving avg |
| `entropy_coef` | `0.01` | Entropy bonus |

## See Also

- [PPO](ppo.md) - More stable alternative
- [RLOO](rloo.md) - Variance reduction with LOO
- [Algorithms Overview](index.md)
