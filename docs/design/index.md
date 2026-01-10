---
title: Design Documentation
description: Technical design and architecture of Flux
---

# Design Documentation

Deep dive into the technical design and architecture decisions behind Flux.

---

## Overview

Flux is designed with three core principles:

1. **Adaptive by Default** - Every parameter that could benefit from adaptation should be adaptive
2. **Native Performance** - Direct integration with Megatron and SGLang, no abstraction overhead
3. **Simple Codebase** - Less than 5,000 lines of core framework code

---

## Documentation

<div class="grid cards" markdown>

-   :material-compass:{ .lg .middle } **Design Philosophy**

    ---

    The principles and decisions that shaped Flux

    [:octicons-arrow-right-24: Philosophy](philosophy.md)

-   :material-file-document-outline:{ .lg .middle } **Technical Specification**

    ---

    Detailed technical specification document

    [:octicons-arrow-right-24: Specification](specification.md)

-   :material-compare:{ .lg .middle } **Framework Comparison**

    ---

    How Flux compares to VERL, AReaL, and Slime

    [:octicons-arrow-right-24: Comparison](comparison.md)

-   :material-road:{ .lg .middle } **Roadmap**

    ---

    Future development plans

    [:octicons-arrow-right-24: Roadmap](roadmap.md)

</div>

---

## Key Design Decisions

### Why Not Ray?

After analyzing VERL, AReaL, and Slime, we concluded that Ray adds unnecessary overhead for LLM training:

| Ray Provides | LLM Training Needs | Mismatch |
|:-------------|:-------------------|:---------|
| Task-level scheduling | NCCL collectives | Ray doesn't understand NCCL |
| Actor lifecycle | Fine-grained GPU control | Ray treats GPU as "count" |
| Flexible placement | Fixed TP/PP/DP config | Changing requires restart |
| Object Store | CUDA IPC / NCCL | Serialization is slow |

**Result**: Both VERL and Slime bypass Ray for critical paths. If you're bypassing the framework, why use it?

### Why Adaptive Async?

The sync vs async choice is typically presented as binary, but it's actually a spectrum:

```
Sync ◄──────────────────────────────────────────────► Async
      │                                              │
      │  Low throughput                High staleness │
      │  High stability               Low stability  │
      │                                              │
      └──────────────────────────────────────────────┘
```

**Key insight**: The optimal position on this spectrum changes during training:

- **Early training**: Policy changing fast → need fresh data → more sync
- **Mid training**: Balanced exploration/exploitation → balanced async
- **Late training**: Fine-tuning → stable policy → can tolerate more async

### Why Unified Importance Correction?

Multiple sources of distribution shift:

1. **Staleness**: Data from old policy versions
2. **Trajectory inconsistency**: Mixed versions within trajectory
3. **Replay**: Data reused from buffer

All are forms of the same problem: $\pi_{current} \neq \pi_{behavior}$

**Solution**: Single importance weight formula that handles all cases:

$$w = \frac{\pi_{current}(a|s)}{\pi_{behavior}(a|s)} \cdot \gamma^{version\_gap}$$

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                  Layer 3: Adaptive Control                   │
│  • PID Controller    • Smart Batch Composer                  │
│  • Staleness Monitor • APRIL Manager                         │
├─────────────────────────────────────────────────────────────┤
│                 Layer 2: Coordinator                         │
│  • asyncio event loop • ZeroMQ/gRPC comm                     │
│  • Weight sync orchestration • Checkpoint management         │
├─────────────────────────────────────────────────────────────┤
│                Layer 1: Native Engines                       │
│  • Megatron-LM (training)  • SGLang (inference)              │
│  • CUDA IPC (weight sync)  • NCCL (gradients)                │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Native Execution Engines

Direct integration with battle-tested systems:

- **Megatron-LM**: Industry-standard distributed training
- **SGLang**: State-of-the-art LLM serving
- **CUDA IPC**: Zero-copy weight transfer

### Layer 2: Lightweight Coordinator

Minimal orchestration layer:

- Pure Python asyncio (no Ray overhead)
- ZeroMQ for local communication
- gRPC for remote communication

### Layer 3: Adaptive Control Plane

The "brain" of Flux:

- PID controller for async ratio
- Staleness measurement and monitoring
- Smart batch composition

---

## Performance Targets

| Metric | Target | Measurement |
|:-------|:-------|:------------|
| GPU Utilization | > 80% | nvidia-smi average |
| Throughput | 2x VERL | samples/hour |
| Staleness | Mean < 0.2 | Combined metric |
| KL Blow-up | < 5% of runs | Detection rate |
| Scaling | > 85% at 64 GPUs | Linear efficiency |

---

## Trade-offs

### Simplicity vs Flexibility

We chose simplicity:

- Fixed Megatron + SGLang stack (no pluggable backends)
- Single deployment topology per run
- Configuration over code for most use cases

**Benefit**: Easier to understand, debug, and maintain

### Stability vs Throughput

We chose adaptive:

- Neither pure sync nor pure async
- Target staleness as control variable
- Automatic adjustment based on training dynamics

**Benefit**: Best of both worlds for most workloads

---

## Further Reading

- [Full Specification](specification.md) - Complete technical details
- [Framework Comparison](comparison.md) - Detailed comparison with alternatives
- [Roadmap](roadmap.md) - Future development plans
