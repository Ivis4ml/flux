---
title: RLOO
description: REINFORCE Leave-One-Out baseline
---

# RLOO (REINFORCE Leave-One-Out)

REINFORCE with a leave-one-out baseline for variance reduction.

## Overview

RLOO uses the average reward of other samples in the same group as the baseline, reducing variance without requiring a value function.

## Advantage Formula

$$
A_i = r_i - \frac{1}{n-1} \sum_{j \neq i} r_j
$$

For each sample, the baseline is the mean of all *other* samples' rewards.

## Why Leave-One-Out?

- **Unbiased**: Unlike mean baseline, LOO is unbiased
- **Low variance**: Uses group information
- **Simple**: No value function needed

## Configuration

```yaml
algorithm:
  name: rloo
  num_samples: 4         # Samples per prompt
  entropy_coef: 0.01     # Entropy bonus
```

## When to Use

**Best for:**
- Multi-sample generation
- When you want REINFORCE simplicity with lower variance
- No value function training

**Compared to GRPO:**
- Similar sample efficiency
- Simpler than GRPO (no clipping)
- Higher variance than GRPO

## Usage

```python
from flux import FluxConfig, FluxTrainer

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    algorithm="rloo",
    algorithm_config={
        "num_samples": 4,
        "entropy_coef": 0.01,
    }
)

trainer = FluxTrainer(config)
trainer.fit(prompts="data.jsonl")
```

## Key Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `num_samples` | `4` | Samples per prompt |
| `entropy_coef` | `0.01` | Entropy bonus |

## Mathematical Details

For a group of $n$ samples with rewards $r_1, ..., r_n$:

$$
\text{baseline}_i = \frac{1}{n-1} \sum_{j \neq i} r_j = \frac{n \cdot \bar{r} - r_i}{n-1}
$$

This is more efficient than naively computing $n$ separate means.

## See Also

- [REINFORCE](reinforce.md) - Basic version
- [GRPO](grpo.md) - Normalized version
- [Algorithms Overview](index.md)
