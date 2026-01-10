---
title: Framework Comparison
description: How Flux compares to alternatives
---

# Framework Comparison

How Flux compares to VERL, AReaL, and Slime.

## Quick Comparison

| Aspect | VERL | AReaL | Slime | **Flux** |
|:-------|:----:|:-----:|:-----:|:--------:|
| Sync Strategy | Fixed sync | Fixed async | Manual | **Adaptive** |
| Orchestration | Ray | Custom | HTTP | **asyncio** |
| Training | Megatron/FSDP | Custom | Megatron | **Megatron** |
| Inference | vLLM/SGLang | Custom | SGLang | **SGLang** |
| Weight Sync | Ray Store | Custom | CUDA IPC | **CUDA IPC** |
| Staleness | N/A | Aware | APRIL | **Unified** |
| Code Size | ~15k LOC | ~25k LOC | ~8k LOC | **<5k LOC** |

## Why Not Ray?

Ray adds overhead for LLM training:

| Ray Provides | LLM Needs | Mismatch |
|:-------------|:----------|:---------|
| Task scheduling | NCCL collectives | Ray doesn't understand NCCL |
| Actor lifecycle | Fine-grained GPU control | Ray treats GPU as "count" |
| Flexible placement | Fixed TP/PP/DP config | Changing requires restart |
| Object Store | CUDA IPC | Serialization is slow |

Both VERL and Slime bypass Ray for critical paths.

## Sync vs Async

```
VERL:  ████████████░░░░░░░░░░░░░░░░  Stable but slow
AReaL: ░░░░░░░░░░░░░░░░████████████  Fast but risky
Flux:  ◄═══════ adapts here ═══════►  Best of both
```

## Performance Targets

| Metric | VERL | AReaL | Flux |
|:-------|:-----|:------|:-----|
| GPU Util | ~45% | ~95% | **>80%** |
| Stability | High | Medium | **High** |
| Throughput | 1x | 2x | **1.8x** |

## When to Use Each

| Use Case | Recommendation |
|:---------|:---------------|
| Maximum stability | VERL or Flux |
| Maximum throughput | AReaL or Flux |
| SGLang native | Slime or Flux |
| Simple setup | Flux |
| Production | Flux |

## See Also

- [Architecture](../concepts/architecture.md)
- [Technical Specification](specification.md)
