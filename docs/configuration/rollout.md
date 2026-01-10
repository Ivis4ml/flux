---
title: Rollout Configuration
description: Generation settings and APRIL strategy
---

# Rollout Configuration

Control how responses are generated.

## Generation Settings

```yaml
rollout:
  max_tokens: 2048     # Maximum response length
  temperature: 1.0     # Sampling temperature
  top_p: 1.0           # Nucleus sampling
  top_k: -1            # Top-k (-1 to disable)
```

## APRIL Strategy

Active Partial Rollout with Long-tail handling.

```yaml
rollout:
  oversample_ratio: 1.5    # Generate 1.5x prompts
  batch_timeout: 30.0      # Abort after 30s
  partial_reuse_threshold: 0.5  # Reuse if >50% done
```

## See Also

- [APRIL Strategy](../concepts/april.md)
- [Configuration Reference](reference.md)
