---
title: Adaptive Async Configuration
description: Staleness targets and PID controller settings
---

# Adaptive Async Configuration

Configure the adaptive async controller.

## Basic Settings

```yaml
adaptive_async:
  target_staleness: 0.15   # Target staleness [0, 1]
  min_async_ratio: 0.1     # Never fully sync
  max_async_ratio: 0.9     # Never fully async
```

## PID Controller

```yaml
adaptive_async:
  kp: 0.1     # Proportional gain
  ki: 0.01    # Integral gain
  kd: 0.05    # Derivative gain
```

## Staleness Weights

```yaml
adaptive_async:
  kl_weight: 0.4       # KL divergence weight
  iw_weight: 0.3       # Importance weight variance
  version_weight: 0.3  # Version gap weight
  # Must sum to 1.0
```

## Tuning Guide

| Goal | target_staleness | max_async_ratio |
|:-----|:-----------------|:----------------|
| Maximum stability | 0.05-0.1 | 0.3-0.5 |
| Balanced | 0.15 | 0.7 |
| Maximum throughput | 0.25-0.3 | 0.9 |

## See Also

- [Adaptive Async Concept](../concepts/adaptive-async.md)
- [Adaptive Async Tutorial](../tutorials/adaptive-async.md)
