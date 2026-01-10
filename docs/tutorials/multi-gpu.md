---
title: Multi-GPU Training
description: Scale your training across multiple GPUs
tags:
  - tutorial
  - intermediate
  - distributed
---

# Multi-GPU Training

Learn how to scale Flux training across multiple GPUs on a single node.

**Time**: 45 minutes
**Prerequisites**: [Basic RLHF Training](basic-rlhf.md), Multiple GPUs available

---

## Overview

In this tutorial, you'll learn:

- [x] GPU allocation strategies
- [x] Configuring tensor parallelism
- [x] Configuring data parallelism
- [x] Optimizing throughput
- [x] Monitoring GPU utilization

---

## GPU Allocation Strategies

Flux supports different GPU allocation patterns:

### Colocated (Recommended for Single Node)

Training and inference share GPUs via time-slicing:

```
┌─────────────────────────────────────┐
│  GPUs 0-7: Training + Inference     │
│  (time-shared)                      │
└─────────────────────────────────────┘
```

### Separated

Dedicated GPUs for each workload:

```
┌─────────────────────────────────────┐
│  GPUs 0-3: Training (Megatron)      │
│  GPUs 4-7: Inference (SGLang)       │
└─────────────────────────────────────┘
```

---

## Setup: 8 GPU Example

### Step 1: Start SGLang Server

For the separated approach:

```bash
# Start SGLang on GPUs 4-7 with tensor parallelism
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --port 8000 \
    --tp 4
```

### Step 2: Configure Training

```yaml title="config-8gpu.yaml"
model_path: Qwen/Qwen3-8B
output_dir: ./outputs

# Training settings
num_steps: 5000
batch_size: 64  # Larger batch with more GPUs
learning_rate: 1.0e-6

# SGLang (on GPUs 4-7)
sglang:
  base_url: http://localhost:8000

# Distributed training (on GPUs 0-3)
distributed:
  world_size: 4
  tensor_parallel: 2  # TP=2
  data_parallel: 2    # DP=2

# Adaptive async
adaptive_async:
  target_staleness: 0.15
  max_async_ratio: 0.8

algorithm:
  name: grpo
  group_size: 8  # More samples per prompt
```

### Step 3: Launch Training

```bash
# Training on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    -m flux.cli train \
    --config config-8gpu.yaml \
    --prompts data/prompts.jsonl
```

---

## Parallelism Strategies

### Tensor Parallelism (TP)

Splits model layers across GPUs:

```
┌─────────────────────────────────────────┐
│  TP=4: Each layer split across 4 GPUs   │
│                                         │
│  GPU 0: Layer weights [0:25%]           │
│  GPU 1: Layer weights [25:50%]          │
│  GPU 2: Layer weights [50:75%]          │
│  GPU 3: Layer weights [75:100%]         │
└─────────────────────────────────────────┘
```

**When to use**: Large models that don't fit on one GPU

```yaml
distributed:
  tensor_parallel: 4  # Split layers across 4 GPUs
```

### Data Parallelism (DP)

Replicates model, splits data:

```
┌─────────────────────────────────────────┐
│  DP=4: Each GPU processes different data │
│                                         │
│  GPU 0: Batch samples [0:8]             │
│  GPU 1: Batch samples [8:16]            │
│  GPU 2: Batch samples [16:24]           │
│  GPU 3: Batch samples [24:32]           │
└─────────────────────────────────────────┘
```

**When to use**: Increase throughput when model fits on one GPU

```yaml
distributed:
  data_parallel: 4  # Process 4x more data
```

### Combined (TP + DP)

```yaml
# 8 GPUs: TP=2 × DP=4
distributed:
  world_size: 8
  tensor_parallel: 2
  data_parallel: 4
```

---

## Configuration Guide

### Model Size to GPU Mapping

| Model Size | Min GPUs | Recommended TP | Recommended DP |
|:-----------|:---------|:---------------|:---------------|
| 7-8B | 1 | 1 | 1-8 |
| 13-14B | 1-2 | 1-2 | 1-4 |
| 30-34B | 2-4 | 2-4 | 1-2 |
| 65-72B | 4-8 | 4-8 | 1 |

### Batch Size Scaling

Scale batch size with data parallelism:

```yaml
# Single GPU
batch_size: 8

# 4 GPUs with DP=4
batch_size: 32  # 8 × 4

# 8 GPUs with DP=8
batch_size: 64  # 8 × 8
```

---

## Monitoring

### GPU Utilization

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or use nvidia-smi dmon
nvidia-smi dmon -s u -d 1
```

### Training Metrics

```python
# Log distributed metrics
@trainer.add_step_callback
def log_distributed_metrics(result):
    print(f"Step {result.step}: "
          f"throughput={result.samples_per_second:.1f} samples/s, "
          f"gpu_util={result.metrics.get('gpu_utilization', 0):.1f}%")
```

---

## Troubleshooting

??? warning "Out of Memory (OOM)"

    **Solutions:**
    - Increase tensor parallelism
    - Reduce batch size per GPU
    - Enable gradient checkpointing

    ```yaml
    distributed:
      tensor_parallel: 4  # Increase TP
    training:
      gradient_checkpointing: true
    ```

??? warning "Slow Training"

    **Check:**
    - GPU utilization (should be >70%)
    - Network bandwidth between GPUs
    - Batch size too small

    ```bash
    # Check NVLink topology
    nvidia-smi topo -m
    ```

??? warning "NCCL Errors"

    **Solutions:**
    ```bash
    # Set NCCL debug
    export NCCL_DEBUG=INFO

    # Try different NCCL algorithms
    export NCCL_ALGO=Ring
    ```

---

## Performance Tips

1. **Use NVLink** when available for faster GPU-GPU communication
2. **Match TP to NVLink pairs** for optimal bandwidth
3. **Increase batch size** with more GPUs
4. **Use gradient accumulation** if batch size is limited by memory

```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 4  # Effective batch = 128
```

---

## Next Steps

- **[Adaptive Async in Practice](adaptive-async.md)** - Fine-tune async control
- **[Production Deployment](production.md)** - Scale to multiple nodes
- **[Performance Optimization](../how-to/performance.md)** - Advanced tuning
