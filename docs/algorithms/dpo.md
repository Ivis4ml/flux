---
title: DPO
description: Direct Preference Optimization
---

# DPO (Direct Preference Optimization)

Train directly from preference pairs without a reward model.

## Overview

DPO bypasses reward modeling by directly optimizing the policy using preference pairs (chosen vs rejected responses).

## Loss Function

$$
L_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)
$$

Where:
- $y_w$ = chosen (winning) response
- $y_l$ = rejected (losing) response
- $\beta$ = temperature parameter
- $\pi_{ref}$ = reference (original) policy

## Configuration

```yaml
algorithm:
  name: dpo
  beta: 0.1               # Temperature parameter
  reference_free: false   # Use reference model
  label_smoothing: 0.0    # Optional smoothing
```

## Data Format

DPO requires preference pairs:

```json
{
  "prompt": "Explain quantum computing.",
  "chosen": "Quantum computing uses qubits...",
  "rejected": "I don't really understand quantum stuff..."
}
```

## When to Use

**Best for:**
- When you have preference data
- Simpler training pipeline
- No reward model needed

**Compared to GRPO/PPO:**
- Simpler setup
- No rollout generation
- Requires paired preferences

## Usage

```python
from flux import FluxConfig, FluxTrainer

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    algorithm="dpo",
    algorithm_config={
        "beta": 0.1,
    }
)

trainer = FluxTrainer(config)
trainer.fit(
    prompts="preferences.jsonl",
    data_format="preferences",
)
```

## Key Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `beta` | `0.1` | Temperature (higher = stronger) |
| `reference_free` | `false` | Skip reference model |
| `label_smoothing` | `0.0` | Smooth labels |

## See Also

- [DPO Tutorial](../tutorials/dpo-training.md)
- [Algorithms Overview](index.md)
