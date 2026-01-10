---
title: Weight Synchronization
description: Efficient weight transfer between training and inference
---

# Weight Synchronization

Flux synchronizes weights between training (Megatron) and inference (SGLang) efficiently.

## Sync Methods

| Method | Latency | Use Case |
|:-------|:--------|:---------|
| CUDA IPC | ~10ms | Same node |
| NCCL | ~100ms | Cross-node |
| HTTP | ~1s | Fallback |

## Delta Compression

Only transfer changed weights:

```yaml
weight_sync:
  method: delta
  compression: true
  snapshot_interval: 10
```

Typical compression: 60-80% bandwidth reduction.
