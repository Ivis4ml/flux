---
title: FluxConfig API
description: Configuration classes reference
---

# FluxConfig

Main configuration class for Flux trainer.

## Overview

FluxConfig uses Pydantic for validation and serialization. All configs are immutable by default.

## Basic Usage

```python
from flux import FluxConfig

# From Python
config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    num_steps=1000,
    batch_size=32,
)

# From YAML
config = FluxConfig.from_yaml("config.yaml")

# Save to YAML
config.to_yaml("output.yaml")
```

## FluxConfig

Top-level configuration combining all sub-configurations.

### Core Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model_path` | `str` | *required* | Model path or HuggingFace ID |
| `model_type` | `str` | `"llama"` | Model architecture type |
| `output_dir` | `str` | `"./outputs"` | Output directory |

### Training Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `learning_rate` | `float` | `1e-6` | Learning rate |
| `batch_size` | `int` | `32` | Batch size |
| `gradient_accumulation_steps` | `int` | `4` | Gradient accumulation |
| `num_steps` | `int` | `10000` | Total training steps |
| `warmup_steps` | `int` | `100` | Warmup steps |
| `weight_decay` | `float` | `0.0` | Weight decay |

### Logging Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `log_interval` | `int` | `10` | Steps between logs |
| `checkpoint_interval` | `int` | `1000` | Steps between checkpoints |
| `eval_interval` | `int` | `500` | Steps between evaluations |
| `seed` | `int` | `42` | Random seed |
| `wandb_project` | `str \| None` | `None` | W&B project name |

### Sub-Configurations

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `adaptive_async` | `AdaptiveAsyncConfig` | Adaptive async settings |
| `rollout` | `RolloutConfig` | Rollout generation settings |
| `batch_composer` | `BatchComposerConfig` | Batch composition settings |
| `weight_sync` | `WeightSyncConfig` | Weight sync settings |
| `algorithm` | `AlgorithmConfig` | Algorithm settings |
| `reward` | `RewardConfig` | Reward settings |
| `sglang` | `SGLangConfig` | SGLang server settings |
| `megatron` | `MegatronConfig` | Megatron settings |

## AdaptiveAsyncConfig

Controls the adaptive async controller.

```python
adaptive_async = AdaptiveAsyncConfig(
    target_staleness=0.15,    # Target staleness level
    min_async_ratio=0.1,      # Minimum async ratio
    max_async_ratio=0.9,      # Maximum async ratio
    kp=0.1,                   # PID proportional gain
    ki=0.01,                  # PID integral gain
    kd=0.05,                  # PID derivative gain
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `target_staleness` | `0.15` | Target staleness [0, 1] |
| `tolerance` | `0.05` | Acceptable deviation |
| `min_async_ratio` | `0.1` | Minimum async ratio |
| `max_async_ratio` | `0.9` | Maximum async ratio |
| `kp`, `ki`, `kd` | `0.1, 0.01, 0.05` | PID gains |
| `max_steps_without_sync` | `50` | Force sync after N steps |
| `ema_alpha` | `0.1` | EMA smoothing factor |

## RolloutConfig

Controls rollout generation and APRIL strategy.

```python
rollout = RolloutConfig(
    max_tokens=2048,
    temperature=1.0,
    oversample_ratio=1.5,     # APRIL: oversample
    batch_timeout=30.0,       # APRIL: abort timeout
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `max_tokens` | `2048` | Max tokens per response |
| `temperature` | `1.0` | Sampling temperature |
| `top_p` | `1.0` | Nucleus sampling |
| `top_k` | `-1` | Top-k sampling |
| `oversample_ratio` | `1.5` | APRIL oversample ratio |
| `batch_timeout` | `30.0` | Batch timeout (seconds) |

## AlgorithmConfig

Controls the RL algorithm.

```python
algorithm = AlgorithmConfig(
    name="grpo",
    clip_range=0.2,
    entropy_coef=0.01,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `name` | `"grpo"` | Algorithm name |
| `clip_range` | `0.2` | PPO clip range |
| `entropy_coef` | `0.01` | Entropy bonus |
| `value_coef` | `0.5` | Value loss coefficient |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `gamma` | `1.0` | Discount factor |
| `gae_lambda` | `0.95` | GAE lambda |

## SGLangConfig

SGLang server connection settings.

```python
sglang = SGLangConfig(
    base_url="http://localhost:8000",
    timeout=60.0,
    max_retries=3,
)
```

## MegatronConfig

Megatron distributed training settings.

```python
megatron = MegatronConfig(
    tp_size=4,      # Tensor parallelism
    pp_size=1,      # Pipeline parallelism
    dp_size=2,      # Data parallelism
    bf16=True,
)
```

## Methods

### from_yaml() / to_yaml()

```python
config = FluxConfig.from_yaml("config.yaml")
config.to_yaml("output.yaml")
```

### to_dict() / from_dict()

```python
data = config.to_dict()
config = FluxConfig.from_dict(data)
```

### get_effective_batch_size()

```python
effective_bs = config.get_effective_batch_size()
# = batch_size * gradient_accumulation_steps
```

### get_total_gpus()

```python
total_gpus = config.get_total_gpus()
# = tp_size * pp_size * dp_size
```

## See Also

- [Configuration Guide](../configuration/index.md)
- [Example Configs](../configuration/examples.md)
