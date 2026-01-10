---
title: Debug Training Issues
description: Diagnose and fix common training problems
---

# Debug Training Issues

Common issues and how to fix them.

## Loss Not Decreasing

**Symptoms:** Loss stays flat or increases

**Causes & Solutions:**

| Cause | Solution |
|:------|:---------|
| Learning rate too high | Reduce by 5-10x |
| Learning rate too low | Increase by 2-5x |
| Reward function broken | Check reward distribution |
| Batch size too small | Increase or use grad accumulation |

**Debugging:**

```python
@trainer.add_step_callback
def debug_loss(result):
    print(f"Step {result.step}:")
    print(f"  Loss: {result.metrics.get('loss', 'N/A')}")
    print(f"  Grad norm: {result.metrics.get('grad_norm', 'N/A')}")
```

## High Staleness

**Symptoms:** `staleness > 0.3` consistently

**Solutions:**

```yaml
adaptive_async:
  max_async_ratio: 0.5  # Reduce async
  target_staleness: 0.1  # Lower target

weight_sync:
  sync_interval: 1  # Sync every step
```

## Out of Memory

**Symptoms:** CUDA OOM

**Solutions:**

1. Reduce batch size
2. Enable gradient checkpointing
3. Use smaller model
4. Increase tensor parallelism

```yaml
batch_size: 16  # Reduce
megatron:
  activation_checkpointing: true
  tp_size: 2  # Increase TP
```

## NaN in Loss

**Symptoms:** Loss becomes NaN

**Solutions:**

1. Reduce learning rate
2. Add gradient clipping
3. Check for log(0) in rewards

```yaml
algorithm:
  max_grad_norm: 0.5  # Tighter clipping
```

## Model Degrades

**Symptoms:** Outputs get worse over training

**Solutions:**

1. Add KL penalty
2. Reduce learning rate
3. Check for reward hacking

```yaml
algorithm:
  kl_coef: 0.1  # Add KL penalty
```

## Slow Training

**Symptoms:** Low throughput

**Check:**

```bash
watch -n 1 nvidia-smi  # GPU utilization
```

**Solutions:**

1. Increase async ratio
2. Use more workers
3. Check network bottlenecks

## See Also

- [Performance Guide](performance.md)
- [Monitoring Guide](monitoring.md)
