---
title: Training Configuration
description: Learning rate, batch size, and optimization settings
---

# Training Configuration

Core training parameters.

## Basic Settings

```yaml
learning_rate: 1.0e-6         # Learning rate
batch_size: 32                # Trajectories per batch
gradient_accumulation_steps: 4
num_steps: 10000              # Total training steps
warmup_steps: 100             # LR warmup steps
```

## Optimizer Settings

```yaml
weight_decay: 0.0
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1.0e-8
```

## Guidelines

| Model Size | Learning Rate | Batch Size |
|:-----------|:--------------|:-----------|
| < 1B | 1e-5 to 5e-6 | 16-32 |
| 1-10B | 5e-6 to 1e-6 | 32-64 |
| 10-70B | 1e-6 to 5e-7 | 64-128 |
| > 70B | 5e-7 to 1e-7 | 128+ |

## See Also

- [Configuration Reference](reference.md)
