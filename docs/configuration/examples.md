---
title: Example Configurations
description: Ready-to-use configuration templates
---

# Example Configurations

Copy and modify these templates for common scenarios.

## Stable Training (Default)

```yaml
model_path: Qwen/Qwen3-8B
output_dir: ./outputs

num_steps: 5000
batch_size: 32
learning_rate: 1.0e-6

adaptive_async:
  target_staleness: 0.15
  min_async_ratio: 0.1
  max_async_ratio: 0.7

algorithm:
  name: grpo
  clip_range: 0.2

sglang:
  base_url: http://localhost:8000
```

## High Throughput

```yaml
model_path: Qwen/Qwen3-8B
output_dir: ./outputs

num_steps: 10000
batch_size: 64
learning_rate: 5.0e-7

adaptive_async:
  target_staleness: 0.25
  max_async_ratio: 0.9

rollout:
  oversample_ratio: 2.0
  batch_timeout: 20.0
```

## Maximum Stability

```yaml
model_path: Qwen/Qwen3-8B
output_dir: ./outputs

num_steps: 5000
batch_size: 16
learning_rate: 5.0e-7

adaptive_async:
  target_staleness: 0.05
  max_async_ratio: 0.3

algorithm:
  name: ppo
  clip_range: 0.1
  kl_coef: 0.2
```

## DPO Training

```yaml
model_path: Qwen/Qwen3-8B
output_dir: ./outputs/dpo

num_steps: 1000
batch_size: 8
learning_rate: 5.0e-7

algorithm:
  name: dpo
  beta: 0.1

adaptive_async:
  enabled: false
```

## Multi-GPU (8x H100)

```yaml
model_path: Qwen/Qwen3-72B
output_dir: ./outputs

num_steps: 10000
batch_size: 128
learning_rate: 1.0e-6

megatron:
  tp_size: 4
  dp_size: 2
  bf16: true
  use_flash_attention: true

adaptive_async:
  target_staleness: 0.15
```

## See Also

- [Configuration Reference](reference.md)
- [Getting Started](../getting-started/index.md)
