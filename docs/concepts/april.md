---
title: APRIL Strategy
description: Active Partial Rollout for efficient generation
---

# APRIL Strategy

**A**ctive **P**artial **R**ollout for **I**efficient generation with **L**ong-tail handling.

## Strategy

1. **Oversample**: Generate more rollouts than needed
2. **Abort**: Cancel long-running generations  
3. **Reuse**: Save and reuse partial trajectories

## Configuration

```yaml
rollout:
  april:
    oversample_ratio: 1.5
    batch_timeout: 30.0
    partial_reuse_threshold: 0.5
```

## Benefits

- Reduces waiting for slow generations
- Improves GPU utilization
- Maintains training throughput
