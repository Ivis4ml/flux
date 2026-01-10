---
title: Configuration Reference
description: Complete configuration reference
---

# Configuration Reference

Complete reference for all FluxConfig parameters.

## FluxConfig (Top-Level)

```yaml
# Model
model_path: Qwen/Qwen3-8B     # Required: Model path or HF ID
model_type: llama             # Model architecture
output_dir: ./outputs         # Output directory

# Training
learning_rate: 1.0e-6         # Learning rate
batch_size: 32                # Batch size
gradient_accumulation_steps: 4
num_steps: 10000              # Total training steps
warmup_steps: 100             # LR warmup steps
weight_decay: 0.0             # Weight decay

# Adam optimizer
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1.0e-8

# Logging
log_interval: 10              # Steps between logs
checkpoint_interval: 1000     # Steps between checkpoints
eval_interval: 500            # Steps between evals
seed: 42                      # Random seed

# W&B (optional)
wandb_project: null           # W&B project name
wandb_run_name: null          # W&B run name
```

## adaptive_async

```yaml
adaptive_async:
  target_staleness: 0.15      # Target staleness [0, 1]
  tolerance: 0.05             # Acceptable deviation
  min_async_ratio: 0.1        # Never fully sync
  max_async_ratio: 0.9        # Never fully async
  
  # PID controller
  kp: 0.1                     # Proportional gain
  ki: 0.01                    # Integral gain
  kd: 0.05                    # Derivative gain
  
  # Staleness computation
  kl_normalizer: 0.1
  iw_normalizer: 2.0
  max_version_gap: 5
  kl_weight: 0.4              # Must sum to 1.0
  iw_weight: 0.3
  version_weight: 0.3
  
  # Sync control
  max_steps_without_sync: 50
  ema_alpha: 0.1
```

## rollout

```yaml
rollout:
  max_tokens: 2048            # Max tokens per response
  temperature: 1.0            # Sampling temperature
  top_p: 1.0                  # Nucleus sampling
  top_k: -1                   # Top-k (-1 disabled)
  
  # APRIL strategy
  oversample_ratio: 1.5       # Oversample factor
  min_yield_size: 8           # Min batch before yield
  batch_timeout: 30.0         # Timeout (seconds)
  
  # Partial reuse
  use_length_prediction: true
  partial_reuse_threshold: 0.5
  partial_buffer_max_factor: 2.0
```

## algorithm

```yaml
algorithm:
  name: grpo                  # ppo, grpo, dpo, reinforce, dapo, rloo
  clip_range: 0.2             # PPO clip range
  clip_range_vf: null         # Value function clip (optional)
  entropy_coef: 0.01          # Entropy bonus
  value_coef: 0.5             # Value loss coefficient
  max_grad_norm: 1.0          # Gradient clipping
  gamma: 1.0                  # Discount factor
  gae_lambda: 0.95            # GAE lambda
  normalize_advantages: true
  kl_coef: 0.0                # KL penalty
  kl_target: null             # Adaptive KL target
  adv_estimator: null         # Override advantage estimator
  policy_loss: null           # Override policy loss
```

## reward

```yaml
reward:
  reward_type: rule_based     # rule_based, model_based, hybrid
  reward_model_path: null     # Path to reward model
  rule_functions: []          # List of rule function names
  reward_scale: 1.0           # Scale factor
  reward_clip: 10.0           # Clip to [-clip, clip]
  baseline_type: mean         # none, mean, per_token
  kl_penalty_coef: 0.0        # KL penalty in reward
```

## sglang

```yaml
sglang:
  base_url: http://localhost:8000
  num_servers: 1              # Number of server instances
  server_urls: null           # Explicit URL list (overrides above)
  timeout: 60.0               # Request timeout
  max_retries: 3              # Retry count
  use_streaming: true         # Streaming responses
```

## megatron

```yaml
megatron:
  tp_size: 1                  # Tensor parallelism
  pp_size: 1                  # Pipeline parallelism
  dp_size: 1                  # Data parallelism
  sequence_parallel: false
  activation_checkpointing: true
  fp16: false
  bf16: true
  use_flash_attention: true
  accumulate_allreduce_grads_in_fp32: true
```

## weight_sync

```yaml
weight_sync:
  method: delta               # full, delta, per_tensor
  sync_interval: 1            # Steps between syncs
  use_cuda_ipc: true          # CUDA IPC for same node
  sparsity_threshold: 1.0e-6
  sparsity_target: 0.3
  snapshot_interval: 10
  max_snapshots: 5
  quantize: false
  quantize_bits: 16           # 8 or 16
```

## batch_composer

```yaml
batch_composer:
  use_length_bucketing: true
  use_staleness_balancing: true
  use_curriculum: true
  length_bucket_boundaries: [512, 1024, 2048]
  staleness_strata: 4
  curriculum_randomness_decay: 1.0
```

## importance_correction

```yaml
importance_correction:
  enabled: true
  staleness_decay: 0.99       # Decay per version gap
  weight_min: 0.2             # Min importance weight
  weight_max: 5.0             # Max importance weight
  log_clip_min: -20.0
  log_clip_max: 20.0
  normalize: true             # Normalize weights
```
