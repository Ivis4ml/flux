---
title: DAPO
description: Decoupled Clip and Dynamic Sampling
---

# DAPO (Decoupled Clip and Dynamic Sampling)

Advanced PPO variant with separate clipping for positive/negative advantages.

## Overview

DAPO improves on PPO by:
1. **Decoupled clipping**: Different clip ranges for positive vs negative advantages
2. **Dynamic sampling**: Adjusts sampling based on advantage magnitude
3. **Token-level loss**: Per-token weighting

## Key Innovations

### Decoupled Clipping

```
For positive advantages (good actions):
  clip_high = 1 + ε_high  (allow more increase)
  
For negative advantages (bad actions):
  clip_low = 1 - ε_low    (allow more decrease)
```

This asymmetry lets the model learn more from positive examples.

## Configuration

```yaml
algorithm:
  name: dapo
  clip_ratio_low: 0.2      # Clip for negative advantages
  clip_ratio_high: 0.28    # Clip for positive advantages
  dynamic_sampling: true   # Enable dynamic sampling
  token_level_loss: true   # Per-token weighting
  entropy_coef: 0.01
```

## When to Use

**Best for:**
- High variance reward functions
- When PPO training is unstable
- Fine-grained control needed

**Compared to PPO:**
- More stable with noisy rewards
- Better sample efficiency
- Slightly more complex

## Usage

```python
from flux import FluxConfig, FluxTrainer

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    algorithm="dapo",
    algorithm_config={
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.28,
        "dynamic_sampling": True,
    }
)

trainer = FluxTrainer(config)
trainer.fit(prompts="data.jsonl")
```

## Key Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `clip_ratio_low` | `0.2` | Clip for negative adv |
| `clip_ratio_high` | `0.28` | Clip for positive adv |
| `dynamic_sampling` | `true` | Enable dynamic sampling |
| `token_level_loss` | `true` | Per-token weighting |

## See Also

- [PPO](ppo.md) - Simpler baseline
- [GRPO](grpo.md) - Group-based alternative
- [Algorithms Overview](index.md)
