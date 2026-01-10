---
title: Staleness & Importance Correction
description: Understanding and correcting for off-policy data
---

# Staleness & Importance Correction

Learn how Flux measures staleness and corrects for off-policy data.

## Staleness Metrics

Flux tracks three types of staleness:

1. **KL Divergence**: Policy drift
2. **Importance Weight Variance**: Distribution shift
3. **Version Gap**: Training steps elapsed

## Combined Staleness

$$
\text{staleness} = 0.4 \cdot \text{KL} + 0.3 \cdot \text{IW} + 0.3 \cdot \text{version}
$$

## Importance Correction

Importance weights correct for off-policy data:

$$
w = \frac{\pi_{current}(a|s)}{\pi_{behavior}(a|s)} \cdot \gamma^{\text{version\_gap}}
$$

See [Adaptive Async](adaptive-async.md) for how this integrates with training.
