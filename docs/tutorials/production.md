---
title: Production Deployment
description: Deploy Flux training at scale with monitoring and fault tolerance
tags:
  - tutorial
  - advanced
  - production
---

# Production Deployment

Learn how to deploy Flux for production-scale training with monitoring, fault tolerance, and best practices.

**Time**: 90 minutes
**Prerequisites**: All previous tutorials

---

## Overview

Production deployment covers:

- [x] Multi-node setup
- [x] Monitoring and alerting
- [x] Fault tolerance
- [x] Checkpointing strategies
- [x] Resource optimization

---

## Multi-Node Architecture

### Recommended Topology

```
┌─────────────────────────────────────────────────────────────┐
│  Node 1: Coordinator + Training                              │
│  GPUs 0-7: Megatron (TP=4, DP=2)                            │
├─────────────────────────────────────────────────────────────┤
│  Node 2: Training                                            │
│  GPUs 0-7: Megatron (TP=4, DP=2)                            │
├─────────────────────────────────────────────────────────────┤
│  Node 3-4: Inference                                         │
│  GPUs 0-7: SGLang servers (TP=4 × 2)                        │
└─────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml title="production-config.yaml"
model_path: Qwen/Qwen3-72B
output_dir: /shared/outputs

# Multi-node training
distributed:
  world_size: 16  # 2 nodes × 8 GPUs
  tensor_parallel: 4
  pipeline_parallel: 1
  data_parallel: 4
  master_addr: node1
  master_port: 29500

# Multiple SGLang servers
sglang:
  servers:
    - url: http://node3:8000
      weight: 1.0
    - url: http://node4:8000
      weight: 1.0
  load_balance: round_robin
  health_check_interval: 30

# Training
num_steps: 50000
batch_size: 256
learning_rate: 1.0e-6
gradient_accumulation_steps: 4

# Robust async settings
adaptive_async:
  target_staleness: 0.15
  max_async_ratio: 0.7

# Checkpointing
checkpoint:
  save_steps: 1000
  save_on_interrupt: true
  distributed_checkpoint: true
  output_dir: /shared/checkpoints
```

---

## Monitoring

### Prometheus Metrics

```yaml title="monitoring.yaml"
logging:
  prometheus:
    enabled: true
    port: 9090

  metrics:
    - name: flux_training_loss
      type: gauge
    - name: flux_staleness
      type: gauge
    - name: flux_async_ratio
      type: gauge
    - name: flux_throughput
      type: gauge
    - name: flux_gpu_utilization
      type: gauge
```

### Start Metrics Server

```python
from flux.utils import MetricsExporter

# In training script
with MetricsExporter(port=9090):
    trainer.fit(prompts="data.jsonl")
```

### Grafana Dashboard

```json
{
  "panels": [
    {
      "title": "Training Loss",
      "targets": [{"expr": "flux_training_loss"}]
    },
    {
      "title": "Staleness vs Target",
      "targets": [
        {"expr": "flux_staleness"},
        {"expr": "flux_target_staleness"}
      ]
    },
    {
      "title": "GPU Utilization",
      "targets": [{"expr": "flux_gpu_utilization"}]
    }
  ]
}
```

### Alerting Rules

```yaml
groups:
  - name: flux_alerts
    rules:
      - alert: HighStaleness
        expr: flux_staleness > 0.4
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Training staleness too high"

      - alert: LowThroughput
        expr: flux_throughput < 100
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Training throughput degraded"

      - alert: TrainingStalled
        expr: increase(flux_training_step[5m]) == 0
        labels:
          severity: critical
        annotations:
          summary: "Training has stalled"
```

---

## Fault Tolerance

### Checkpoint Strategy

```yaml
checkpoint:
  # Regular saves
  save_steps: 1000
  max_checkpoints: 10
  keep_best: 3

  # Emergency saves
  save_on_interrupt: true   # SIGINT/SIGTERM
  save_on_error: true       # Exceptions

  # Distributed
  distributed_checkpoint: true
  async_save: true          # Non-blocking saves
```

### Automatic Recovery

```python
from flux import FluxTrainer, FluxConfig
from flux.utils import GracefulShutdown

config = FluxConfig.from_yaml("production-config.yaml")
trainer = FluxTrainer(config)

# Setup graceful shutdown
with GracefulShutdown(timeout=60) as shutdown:
    shutdown.register_cleanup(trainer.save_checkpoint)

    # Resume from latest checkpoint if exists
    if trainer.checkpoint_exists():
        trainer.load_checkpoint("latest")
        print(f"Resumed from step {trainer.current_step}")

    # Train with interruption handling
    try:
        trainer.fit(prompts="data.jsonl")
    except KeyboardInterrupt:
        print("Interrupted, saving checkpoint...")
        trainer.save_checkpoint("interrupted")
```

### SGLang Failover

```yaml
sglang:
  servers:
    - url: http://node3:8000
      weight: 1.0
    - url: http://node4:8000
      weight: 1.0
    - url: http://node5:8000  # Backup
      weight: 0.0              # Only used on failover

  failover:
    enabled: true
    health_check_interval: 10
    max_retries: 3
    cooldown: 60
```

---

## Resource Optimization

### Memory Optimization

```yaml
training:
  # Gradient checkpointing
  gradient_checkpointing: true

  # Mixed precision
  mixed_precision: bf16

  # Optimizer memory
  optimizer:
    name: adamw
    fused: true           # Fused CUDA kernels
    foreach: true         # Batched operations
```

### Network Optimization

```yaml
distributed:
  # NCCL tuning
  nccl_timeout: 1800

  # Overlap communication
  overlap_grad_reduce: true
  overlap_param_gather: true

  # Bucket size for gradient reduction
  bucket_cap_mb: 25
```

### I/O Optimization

```yaml
data:
  # Data loading
  num_workers: 4
  prefetch_factor: 2
  pin_memory: true

  # Sharding
  shard_data: true
  drop_last: true
```

---

## Launch Scripts

### SLURM Script

```bash title="submit.slurm"
#!/bin/bash
#SBATCH --job-name=flux-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out

# Load modules
module load cuda/12.4
module load nccl

# Set environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# Launch training
srun python -m flux.cli train \
    --config production-config.yaml \
    --prompts /data/prompts.jsonl
```

### Docker Compose

```yaml title="docker-compose.yaml"
version: "3.8"

services:
  trainer:
    image: fluxrlhf/flux:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 8
    volumes:
      - ./data:/data
      - ./outputs:/outputs
    command: >
      flux train
        --config /data/config.yaml
        --prompts /data/prompts.jsonl

  sglang:
    image: lmsysorg/sglang:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 4
    ports:
      - "8000:8000"
    command: >
      python -m sglang.launch_server
        --model-path Qwen/Qwen3-8B
        --port 8000
        --tp 4

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

---

## Checklist

Before production deployment:

- [ ] Test on small scale first
- [ ] Validate checkpoint save/restore
- [ ] Set up monitoring and alerts
- [ ] Configure failover for SGLang
- [ ] Test graceful shutdown
- [ ] Document configuration
- [ ] Set up log aggregation
- [ ] Plan for model versioning

---

## Next Steps

- **[How-to: Multi-Node](../how-to/multi-node.md)** - Detailed multi-node guide
- **[How-to: Monitoring](../how-to/monitoring.md)** - Prometheus/Grafana setup
- **[How-to: Performance](../how-to/performance.md)** - Optimization tips
