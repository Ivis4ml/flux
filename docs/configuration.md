# Flux Configuration Guide

Flux uses a hierarchical configuration system based on Pydantic. All configuration is validated at load time with sensible defaults.

## Loading Configuration

### From YAML

```python
from flux import FluxConfig

config = FluxConfig.from_yaml("configs/qwen3-8b.yaml")
```

### From Python

```python
from flux import FluxConfig

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    output_dir="./outputs",
    num_steps=1000,
)
```

### From Environment

```bash
export FLUX_MODEL_PATH="Qwen/Qwen3-8B"
export FLUX_OUTPUT_DIR="./outputs"
flux train --prompts data/prompts.json
```

## Core Settings

### Basic Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to model or HuggingFace model ID |
| `output_dir` | str | "./outputs" | Output directory for checkpoints |
| `num_steps` | int | 1000 | Total training steps |
| `batch_size` | int | 32 | Training batch size |
| `learning_rate` | float | 1e-6 | Learning rate |
| `seed` | int | 42 | Random seed |

```yaml
model_path: "Qwen/Qwen3-8B"
output_dir: "./outputs"
num_steps: 5000
batch_size: 32
learning_rate: 1.0e-6
seed: 42
```

## Adaptive Async Configuration

Controls the dynamic sync/async ratio adjustment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_staleness` | float | 0.5 | Target staleness level (0=fresh, 1=stale) |
| `min_async_ratio` | float | 0.0 | Minimum async ratio |
| `max_async_ratio` | float | 1.0 | Maximum async ratio |
| `kp` | float | 0.1 | Proportional gain |
| `ki` | float | 0.01 | Integral gain |
| `kd` | float | 0.05 | Derivative gain |
| `update_frequency` | int | 10 | Steps between ratio updates |

```yaml
adaptive_async:
  target_staleness: 0.5
  min_async_ratio: 0.0
  max_async_ratio: 0.8
  kp: 0.1
  ki: 0.01
  kd: 0.05
  update_frequency: 10
```

### Understanding Staleness

- **0.0**: Completely on-policy (synchronous)
- **0.5**: Moderate staleness (balanced)
- **1.0**: Heavily off-policy (high async)

Lower staleness = more stable but slower training.

## Rollout Configuration

Controls generation (inference) behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_length` | int | 2048 | Maximum generation length |
| `temperature` | float | 1.0 | Sampling temperature |
| `top_p` | float | 1.0 | Nucleus sampling parameter |
| `top_k` | int | -1 | Top-k sampling (-1 = disabled) |
| `num_return_sequences` | int | 1 | Sequences per prompt |

```yaml
rollout:
  max_length: 2048
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  num_return_sequences: 4
```

### APRIL Settings

APRIL (Asynchronous Policy Rollout with Importance-weighted Learning):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `oversample_ratio` | float | 1.5 | Oversample factor for rollouts |
| `batch_timeout` | float | 30.0 | Timeout for batch completion |
| `partial_reuse_threshold` | float | 0.3 | Threshold for partial trajectory reuse |

```yaml
rollout:
  april:
    oversample_ratio: 1.5
    batch_timeout: 30.0
    partial_reuse_threshold: 0.3
```

## Batch Composer Configuration

Controls how trajectories are grouped into batches.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `length_bucket_boundaries` | list | [256, 512, 1024] | Length bucket boundaries |
| `staleness_balance_weight` | float | 0.3 | Weight for staleness balancing |
| `curriculum_enabled` | bool | true | Enable curriculum learning |
| `curriculum_decay` | float | 0.995 | Curriculum randomness decay |

```yaml
batch_composer:
  length_bucket_boundaries: [256, 512, 1024, 2048]
  staleness_balance_weight: 0.3
  curriculum_enabled: true
  curriculum_decay: 0.995
```

## Algorithm Configuration

Select and configure the RL algorithm.

### Available Algorithms

- `ppo`: Proximal Policy Optimization
- `grpo`: Group Relative Policy Optimization (default)
- `dpo`: Direct Preference Optimization
- `reinforce`: Basic REINFORCE
- `dapo`: Decoupled clip and dynamic sampling
- `rloo`: Leave-One-Out baseline

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "grpo" | Algorithm name |
| `clip_ratio` | float | 0.2 | PPO clip ratio |
| `entropy_coef` | float | 0.01 | Entropy bonus coefficient |
| `value_coef` | float | 0.5 | Value loss coefficient |
| `max_grad_norm` | float | 1.0 | Gradient clipping |

```yaml
algorithm:
  name: "grpo"
  clip_ratio: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 1.0
```

### PPO-specific

```yaml
algorithm:
  name: "ppo"
  clip_ratio: 0.2
  kl_penalty: 0.1
  target_kl: 0.01
```

### GRPO-specific

```yaml
algorithm:
  name: "grpo"
  group_size: 4
  baseline: "mean"  # or "leave_one_out"
```

## Weight Sync Configuration

Controls how weights are synchronized between training and inference.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | "full" | Sync method: "full", "delta", "snapshot" |
| `interval` | int | 1 | Steps between syncs |
| `compression` | bool | false | Enable delta compression |
| `snapshot_dir` | str | null | Directory for snapshots |

```yaml
weight_sync:
  method: "delta"
  interval: 1
  compression: true
  snapshot_dir: "./snapshots"
```

## Checkpoint Configuration

Controls checkpoint saving and loading.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_steps` | int | 500 | Steps between saves |
| `max_checkpoints` | int | 5 | Maximum checkpoints to keep |
| `keep_best` | int | 3 | Best checkpoints to keep |
| `save_optimizer` | bool | true | Save optimizer state |

```yaml
checkpoint:
  save_steps: 500
  max_checkpoints: 5
  keep_best: 3
  save_optimizer: true
```

## Logging Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | str | "INFO" | Logging level |
| `log_steps` | int | 10 | Steps between logs |
| `wandb_project` | str | null | W&B project name |
| `tensorboard_dir` | str | null | TensorBoard log directory |

```yaml
logging:
  log_level: "INFO"
  log_steps: 10
  wandb_project: "flux-training"
  tensorboard_dir: "./logs"
```

## Distributed Configuration

For multi-GPU and multi-node training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `world_size` | int | 1 | Total number of processes |
| `tensor_parallel` | int | 1 | Tensor parallelism degree |
| `pipeline_parallel` | int | 1 | Pipeline parallelism degree |
| `data_parallel` | int | 1 | Data parallelism degree |

```yaml
distributed:
  world_size: 8
  tensor_parallel: 4
  pipeline_parallel: 1
  data_parallel: 2
```

## Complete Example

```yaml
# Full configuration example
model_path: "Qwen/Qwen3-8B"
output_dir: "./outputs/qwen3-8b-rlhf"
num_steps: 10000
batch_size: 32
learning_rate: 1.0e-6
seed: 42

adaptive_async:
  target_staleness: 0.4
  min_async_ratio: 0.1
  max_async_ratio: 0.7
  kp: 0.1
  ki: 0.01
  kd: 0.05

rollout:
  max_length: 2048
  temperature: 0.8
  top_p: 0.95
  april:
    oversample_ratio: 1.5
    batch_timeout: 30.0

batch_composer:
  length_bucket_boundaries: [256, 512, 1024, 2048]
  staleness_balance_weight: 0.3
  curriculum_enabled: true

algorithm:
  name: "grpo"
  clip_ratio: 0.2
  entropy_coef: 0.01
  max_grad_norm: 1.0

weight_sync:
  method: "delta"
  interval: 2
  compression: true

checkpoint:
  save_steps: 500
  max_checkpoints: 5
  keep_best: 3

logging:
  log_level: "INFO"
  log_steps: 10
  wandb_project: "flux-qwen3"

distributed:
  world_size: 8
  tensor_parallel: 4
  data_parallel: 2
```

## Configuration Validation

Flux validates all configuration at load time:

```python
from flux import FluxConfig

# This will raise ValidationError if invalid
try:
    config = FluxConfig(
        model_path="invalid",
        num_steps=-1,  # Invalid!
    )
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Environment Variables

Override any config with environment variables:

```bash
# Format: FLUX_<SECTION>_<PARAMETER>
export FLUX_NUM_STEPS=2000
export FLUX_ADAPTIVE_ASYNC_TARGET_STALENESS=0.3
export FLUX_ALGORITHM_NAME=ppo
```

## Next Steps

- [Algorithm Guide](algorithms.md) - Deep dive into RL algorithms
- [API Documentation](api.md) - Full API reference
- [Examples](../examples/) - Working code examples
