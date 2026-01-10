---
title: Architecture Overview
description: Flux system architecture and design
---

# Architecture Overview

Flux is designed with a three-layer architecture for maximum performance and flexibility.

## Three-Layer Design

```mermaid
graph TB
    subgraph Layer3["Layer 3: Adaptive Control"]
        AC[Adaptive Async Controller]
        SM[Staleness Monitor]
        BC[Batch Composer]
    end
    
    subgraph Layer2["Layer 2: Coordinator"]
        CO[FluxCoordinator]
        WS[Weight Sync]
    end
    
    subgraph Layer1["Layer 1: Native Engines"]
        ME[Megatron Training]
        SG[SGLang Inference]
    end
    
    Layer3 --> Layer2
    Layer2 --> Layer1
```

## Components

### Layer 1: Native Engines
- **Megatron-LM**: Distributed training with 3D parallelism
- **SGLang**: High-performance inference server

### Layer 2: Coordinator
- Orchestrates training loop
- Manages weight synchronization
- Handles checkpointing

### Layer 3: Adaptive Control
- PID controller for async ratio
- Smart batch composition
- Staleness measurement

For detailed design, see [Technical Specification](../design/specification.md).
