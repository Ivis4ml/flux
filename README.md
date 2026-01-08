# 🌊 Flux

**An Adaptive Post-Training Framework for Large Language Models**

> *"The best of all worlds"* — Synchronous stability + Asynchronous efficiency + Native simplicity

---

## Why Flux?

Existing RLHF frameworks force you to choose:

| Framework | Trade-off |
|-----------|-----------|
| **VERL** | Stable but slow (GPU bubbles from synchronous training) |
| **AReaL** | Fast but unstable (staleness from full async) |
| **Slime** | Simple but less flexible |

**Flux takes a different approach**: Instead of binary choices, we treat everything as a **continuous spectrum** that adapts during training.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Sync ◄─────── Flux adapts in real-time ───────► Async        │
│                                                                 │
│   • Early training: More sync (stability)                       │
│   • Mid training: Balanced                                      │
│   • Late training: More async (speed)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 🎯 Adaptive Async Control

Flux automatically adjusts the sync/async ratio based on training dynamics:

```python
# Flux maintains staleness within a target range
# Not too stale (unstable) and not too fresh (slow)
controller = AdaptiveAsyncController(
    target_staleness=0.15,  # Sweet spot
    tolerance=0.05
)
```

### ⚡ Native Performance

No Ray. No unnecessary abstraction layers. Direct integration with:

- **Megatron-LM** for training (TP, PP, DP, EP, CP)
- **SGLang** for inference (continuous batching, FP8)
- **CUDA IPC** for weight sync (zero-copy)

### 🧠 Smart Batching

Flux optimizes every batch:

- **Length-aware packing**: Minimize padding waste
- **Staleness balancing**: Reduce importance weight variance
- **Curriculum ordering**: Easy → Hard as training progresses

### 🔄 APRIL Strategy

Active Partial Rollout for handling long-tail generations:

```
Standard approach:     Wait for slowest → GPU idle
                       ████░░░░░░░░░░░░░░░░

APRIL approach:        Oversample, abort long-tail, reuse partials
                       ████████████████████
```

## Quick Start

```python
from flux import FluxTrainer, FluxConfig

# Configure
config = FluxConfig(
    model_path="meta-llama/Llama-3-8B",
    sglang_url="http://localhost:8000",
    
    # Adaptive settings
    target_staleness=0.15,
    min_async_ratio=0.1,
    max_async_ratio=0.9,
)

# Train
trainer = FluxTrainer(config)
trainer.fit(
    prompts=train_prompts,
    num_steps=10000,
    eval_prompts=eval_prompts,
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Adaptive Control Plane                       │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │
│  │ Adaptive  │ │  Smart    │ │ Staleness │ │Speculative│       │
│  │  Async    │ │  Batch    │ │  Monitor  │ │  Sync     │       │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │
├─────────────────────────────────────────────────────────────────┤
│                   Lightweight Coordinator                        │
│              (asyncio + ZeroMQ, no Ray)                         │
├─────────────────────────────────────────────────────────────────┤
│                   Native Execution Engines                       │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   Megatron-LM       │◄──►│      SGLang         │            │
│  │   (Training)        │    │    (Inference)      │            │
│  └─────────────────────┘    └─────────────────────┘            │
│                    CUDA IPC Weight Sync                         │
└─────────────────────────────────────────────────────────────────┘
```

## Performance

Preliminary targets (to be validated):

| Metric | VERL | AReaL | Flux |
|--------|------|-------|------|
| GPU Utilization | ~45% | ~95% | **~85%** |
| Training Stability | ★★★★★ | ★★★☆☆ | **★★★★☆** |
| Throughput | 1.0x | 1.8x | **2.0x** |

## Design Philosophy

### 1. "Continuous Spectrum, Not Binary Choice"

Every hyperparameter that could benefit from adaptation should be adaptive:
- Sync/async ratio
- Temperature
- Batch composition
- Compute allocation

### 2. "Native First"

Use the best existing tools directly:
- Don't wrap Megatron, integrate with it
- Don't wrap SGLang, call its HTTP API
- Don't use Ray, write simple Python

### 3. "Simple > Clever"

- < 5000 lines of core code
- No magic, explicit control flow
- Easy to debug and extend

## Comparison with Other Frameworks

| Aspect | VERL | AReaL | Slime | **Flux** |
|--------|------|-------|-------|----------|
| Sync Strategy | Fixed sync | Fixed async | Both | **Adaptive** |
| Orchestration | Ray | Custom | HTTP | **asyncio** |
| Training Backend | Megatron/FSDP | Custom | Megatron | **Megatron** |
| Inference Backend | vLLM/SGLang | Custom | SGLang | **SGLang** |
| Weight Sync | Ray Object Store | Custom | CUDA IPC | **CUDA IPC** |
| Staleness Handling | N/A (sync) | Staleness-aware PPO | APRIL | **Unified correction** |
| Code Complexity | Medium | High | Low | **Low** |

## Roadmap

- [x] Design specification
- [x] Core component skeleton
- [ ] Phase 1: Foundation (Megatron + SGLang integration)
- [ ] Phase 2: Adaptive components
- [ ] Phase 3: Optimizations
- [ ] Phase 4: Production readiness

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{flux2025,
  author = {Xin},
  title = {Flux: An Adaptive Post-Training Framework for Large Language Models},
  year = {2025},
  url = {https://github.com/xxx/flux}
}
```

## License

Apache 2.0

---

<p align="center">
  <i>Flux: Where stability meets efficiency</i>
</p>
