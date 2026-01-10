---
title: Optimize Performance
description: Get maximum throughput from your hardware
---

# Optimize Performance

Tips for maximizing training throughput.

## Measure Baseline

```python
@trainer.add_step_callback
def log_perf(result):
    print(f"Throughput: {result.metrics.get('throughput', 0):.1f} samples/s")
```

## GPU Utilization

```bash
watch -n 1 nvidia-smi
# Target: >80% GPU utilization
```

## Key Optimizations

### 1. Increase Batch Size

```yaml
batch_size: 64
gradient_accumulation_steps: 4  # Effective: 256
```

### 2. Tune Async Ratio

```yaml
adaptive_async:
  target_staleness: 0.2
  max_async_ratio: 0.8
```

### 3. Enable Flash Attention

```yaml
megatron:
  use_flash_attention: true
```

### 4. Use BF16

```yaml
megatron:
  bf16: true
  fp16: false
```

### 5. Optimize Weight Sync

```yaml
weight_sync:
  method: delta
  use_cuda_ipc: true
```

### 6. Tune APRIL

```yaml
rollout:
  oversample_ratio: 1.5
  batch_timeout: 20.0
```

## Profiling

```python
from torch.profiler import profile

with profile() as prof:
    trainer.fit(prompts, num_steps=10)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## See Also

- [Multi-GPU Training](../tutorials/multi-gpu.md)
- [Production Deployment](../tutorials/production.md)
