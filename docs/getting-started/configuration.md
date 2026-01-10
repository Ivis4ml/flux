---
title: Configuration Basics
description: Learn the basics of Flux configuration
---

# Configuration Basics

A quick introduction to configuring Flux for your training needs.

---

## Configuration Methods

Flux supports three ways to configure training:

=== "YAML File (Recommended)"

    ```yaml title="config.yaml"
    model_path: Qwen/Qwen3-8B
    num_steps: 1000
    batch_size: 32
    algorithm: grpo
    ```

    ```python
    config = FluxConfig.from_yaml("config.yaml")
    ```

=== "Python Code"

    ```python
    config = FluxConfig(
        model_path="Qwen/Qwen3-8B",
        num_steps=1000,
        batch_size=32,
        algorithm="grpo",
    )
    ```

=== "Environment Variables"

    ```bash
    export FLUX_MODEL_PATH="Qwen/Qwen3-8B"
    export FLUX_NUM_STEPS=1000
    flux train --prompts data.jsonl
    ```

---

## Essential Settings

### Model Settings

```yaml
model_path: Qwen/Qwen3-8B    # HuggingFace model ID or local path
output_dir: ./outputs         # Where to save checkpoints
```

### Training Settings

```yaml
num_steps: 1000              # Total training steps
batch_size: 32               # Samples per training batch
learning_rate: 1.0e-6        # Learning rate (lower for larger models)
seed: 42                     # Random seed for reproducibility
```

### Algorithm Settings

```yaml
algorithm:
  name: grpo                 # Algorithm: ppo, grpo, dpo, reinforce
  group_size: 4              # For GRPO: responses per prompt
  clip_ratio: 0.2            # For PPO: clipping range
```

---

## Adaptive Async Settings

The key to Flux's efficiency:

```yaml
adaptive_async:
  target_staleness: 0.15     # Target staleness level (0-1)
  min_async_ratio: 0.1       # Minimum async (never fully sync)
  max_async_ratio: 0.9       # Maximum async (never fully async)
  kp: 0.1                    # PID proportional gain
  ki: 0.01                   # PID integral gain
  kd: 0.05                   # PID derivative gain
```

### Quick Tuning Guide

| Goal | `target_staleness` | `max_async_ratio` |
|:-----|:-------------------|:------------------|
| Maximum stability | 0.05-0.1 | 0.3-0.5 |
| Balanced (default) | 0.15 | 0.7 |
| Maximum throughput | 0.3-0.4 | 0.9 |

---

## SGLang Settings

Configure the inference server connection:

```yaml
sglang:
  base_url: http://localhost:8000
  timeout: 60                # Request timeout (seconds)
  max_retries: 3             # Retry count on failure
```

---

## Rollout Settings

Control how responses are generated:

```yaml
rollout:
  max_length: 2048           # Maximum response length
  temperature: 0.8           # Sampling temperature
  top_p: 0.95                # Nucleus sampling
  top_k: 50                  # Top-k sampling (-1 to disable)

  april:                     # APRIL strategy settings
    oversample_ratio: 1.5    # Oversample factor
    batch_timeout: 30.0      # Timeout for batch completion
```

---

## Checkpoint Settings

Control saving and loading:

```yaml
checkpoint:
  save_steps: 500            # Save every N steps
  max_checkpoints: 5         # Maximum checkpoints to keep
  keep_best: 3               # Best checkpoints to keep
  save_optimizer: true       # Include optimizer state
```

---

## Logging Settings

```yaml
logging:
  log_level: INFO            # DEBUG, INFO, WARNING, ERROR
  log_steps: 10              # Log every N steps
  wandb_project: null        # W&B project name (optional)
  tensorboard_dir: null      # TensorBoard directory (optional)
```

---

## Complete Example

```yaml title="config.yaml"
# Complete configuration example
model_path: Qwen/Qwen3-8B
output_dir: ./outputs

# Training
num_steps: 5000
batch_size: 32
learning_rate: 1.0e-6
seed: 42

# SGLang
sglang:
  base_url: http://localhost:8000
  timeout: 60

# Adaptive async
adaptive_async:
  target_staleness: 0.15
  min_async_ratio: 0.1
  max_async_ratio: 0.7

# Algorithm
algorithm:
  name: grpo
  group_size: 4

# Rollout
rollout:
  max_length: 2048
  temperature: 0.8
  april:
    oversample_ratio: 1.5

# Checkpoints
checkpoint:
  save_steps: 500
  max_checkpoints: 5

# Logging
logging:
  log_steps: 10
  log_level: INFO
```

---

## Validation

Flux validates configuration at load time:

```python
from flux import FluxConfig
from pydantic import ValidationError

try:
    config = FluxConfig(num_steps=-1)  # Invalid!
except ValidationError as e:
    print(f"Error: {e}")
```

---

## Next Steps

- **[Full Configuration Reference](../configuration/reference.md)** - All options
- **[Example Configs](../configuration/examples.md)** - Ready-to-use templates
- **[First Training Run](first-training.md)** - Put it into practice
