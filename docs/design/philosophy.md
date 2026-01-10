---
title: Design Philosophy
description: The principles that shaped Flux
---

# Design Philosophy

The guiding principles behind Flux's architecture.

## Core Principles

### 1. Adaptive by Default

Every parameter that could benefit from adaptation should be adaptive.

- Sync/async ratio adapts to staleness
- Learning rate can adapt to KL divergence
- Batch composition adapts to trajectory distribution

### 2. Native Performance

Direct integration with optimized systems, no abstraction overhead.

- **Megatron-LM**: Industry-standard distributed training
- **SGLang**: State-of-the-art inference serving
- **CUDA IPC**: Zero-copy weight transfer

### 3. Simple Codebase

Less than 5,000 lines of core framework code.

- Easy to understand
- Easy to debug
- Easy to extend

## What We Learned From Others

### From VERL

- Robust synchronous training patterns
- HybridFlow controller design
- Checkpoint/resume logic

### From AReaL

- Asynchronous training architecture
- Staleness-aware PPO
- High GPU utilization techniques

### From Slime

- SGLang-native integration
- APRIL strategy
- Simple HTTP coordination

## Trade-offs We Made

### Simplicity over Flexibility

- Fixed Megatron + SGLang stack
- Single deployment topology per run
- Configuration over code

**Why**: Easier to understand, debug, maintain

### Adaptive over Fixed

- PID controller for async ratio
- Dynamic batch composition
- Automatic staleness correction

**Why**: Optimal settings change during training

## See Also

- [Technical Specification](specification.md)
- [Framework Comparison](comparison.md)
