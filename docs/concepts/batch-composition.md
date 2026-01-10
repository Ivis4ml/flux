---
title: Smart Batch Composition
description: Intelligent batching strategies for optimal training
---

# Smart Batch Composition

Flux uses smart batching to optimize training efficiency.

## Strategies

### 1. Length Bucketing
Groups similar-length sequences to minimize padding.

### 2. Staleness Balancing
Stratified sampling ensures balanced staleness distribution.

### 3. Curriculum Learning
Progressive difficulty ordering (easy → hard).

## Configuration

```yaml
batch_composer:
  length_bucket_boundaries: [256, 512, 1024, 2048]
  staleness_balance_weight: 0.3
  curriculum_enabled: true
  curriculum_decay: 0.995
```
