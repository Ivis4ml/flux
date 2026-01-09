<div align="center">

<img src="docs/flux-concept.png" alt="Flux" width="320">

<h2>Adaptive Post-Training Framework for LLMs</h2>

<p>
<strong>The best of all worlds</strong> — Synchronous stability + Asynchronous efficiency + Native simplicity
</p>

<p>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
<a href="https://github.com/flux-team/flux/stargazers"><img src="https://img.shields.io/github/stars/flux-team/flux" alt="GitHub Stars"></a>
</p>

<p>
<a href="#installation">Installation</a> •
<a href="#quick-start">Quick Start</a> •
<a href="#key-features">Features</a> •
<a href="#architecture">Architecture</a> •
<a href="docs/flux-design-spec.md">Design Doc</a>
</p>

</div>

---

## Overview

Flux is a flexible and efficient reinforcement learning framework for LLM post-training (RLHF). Unlike existing frameworks that force a binary choice between synchronous stability and asynchronous throughput, **Flux adaptively adjusts the sync/async ratio in real-time** based on measured staleness.

**Key Insight**: Sync vs Async is NOT a binary choice. Flux operates anywhere on this spectrum, adapting in real-time to maximize both stability and throughput.

```
Sync ◄────────────────────────────────────────────────────► Async

     VERL        ████████████░░░░░░░░░░░░░░░░░░  Stable but slow
     AReaL       ░░░░░░░░░░░░░░░░░░████████████  Fast but risky
     Flux        ◄═══════ adapts here ═══════►  Best of both
```

## Comparison

| Aspect | VERL | AReaL | Slime | **Flux** |
|:-------|:----:|:-----:|:-----:|:--------:|
| Sync Strategy | Fixed sync | Fixed async | Both | **Adaptive** |
| Orchestration | Ray | Custom | HTTP | **asyncio** |
| Training Backend | Megatron/FSDP | Custom | Megatron | **Megatron** |
| Inference Backend | vLLM/SGLang | Custom | SGLang | **SGLang** |
| Weight Sync | Ray Object Store | Custom | CUDA IPC | **CUDA IPC** |
| Staleness Handling | N/A | Staleness-aware PPO | APRIL | **Unified** |
| Code Complexity | ~15k LOC | ~25k LOC | ~8k LOC | **<5k LOC** |

---

## Key Features

### Adaptive Async Control

Flux uses a PID controller to dynamically adjust the sync/async ratio based on measured staleness:

- **Early training**: More synchronous (policy changing rapidly)
- **Late training**: More asynchronous (policy stable, maximize throughput)

The controller monitors three staleness signals: KL divergence, importance weight variance, and version gap.

### APRIL Strategy

Active Partial Rollout for efficient generation:

| Step | Description |
|:-----|:------------|
| **Oversample** | Generate 1.5× prompts to have buffer |
| **Abort** | Cancel long-tail generations after timeout |
| **Reuse** | Save partial trajectories for continuation |

### Smart Batch Composition

- **Length bucketing** — Group similar lengths to minimize padding
- **Staleness balancing** — Stratified sampling to reduce importance weight variance
- **Curriculum ordering** — Easy → hard progression as training proceeds

### Zero-Copy Weight Sync

CUDA IPC for same-node weight transfer between Megatron and SGLang:
- No serialization overhead
- Delta compression for incremental updates
- Lazy sync (only when inference needs fresh weights)

---

## Supported Algorithms

| Algorithm | Type | Description |
|:----------|:-----|:------------|
| **PPO** | On-policy | Clipped surrogate objective |
| **GRPO** | On-policy | Group Relative Policy Optimization (default) |
| **DPO** | Preference | Direct Preference Optimization |
| **REINFORCE** | On-policy | Basic policy gradient |
| **DAPO** | On-policy | Decoupled clip and dynamic sampling |
| **GSPO** | On-policy | Group Stability Policy Optimization |
| **RLOO** | On-policy | Leave-One-Out baseline estimator |

All algorithms benefit from Flux's unified importance correction for off-policy data.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.0+ with NCCL
- [SGLang](https://github.com/sgl-project/sglang) server
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (for distributed training)

### Install from Source

```bash
git clone https://github.com/flux-team/flux.git
cd flux

# Basic installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# Full installation (includes all dependencies)
pip install -e ".[all]"
```

---

## Quick Start

### 1. Start SGLang Server

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 8000
```

### 2. Run Training

```python
from flux import FluxTrainer, FluxConfig, AdaptiveAsyncConfig, SGLangConfig

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    sglang=SGLangConfig(base_url="http://localhost:8000"),
    adaptive_async=AdaptiveAsyncConfig(
        target_staleness=0.15,
        min_async_ratio=0.1,
        max_async_ratio=0.9,
    ),
    learning_rate=1e-6,
    batch_size=32,
    num_steps=10000,
)

trainer = FluxTrainer(config)
trainer.fit(prompts, eval_prompts=eval_prompts)
```

### 3. Using YAML Configuration

```bash
flux train --config configs/qwen3-8b-8gpu.yaml --prompts data/prompts.jsonl
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FLUX ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           Layer 3: Adaptive Control Plane                 │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │  │
│  │  │  Adaptive   │ │   Smart     │ │  Staleness  │         │  │
│  │  │   Async     │ │   Batch     │ │   Monitor   │         │  │
│  │  │ Controller  │ │  Composer   │ │             │         │  │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘         │  │
│  └─────────┼───────────────┼───────────────┼─────────────────┘  │
│            └───────────────┼───────────────┘                    │
│                            │                                    │
│  ┌─────────────────────────▼─────────────────────────────────┐  │
│  │          Layer 2: Lightweight Coordinator                 │  │
│  │                  (asyncio + ZeroMQ)                       │  │
│  └─────────────────────────┬─────────────────────────────────┘  │
│                            │                                    │
│            ┌───────────────┴───────────────┐                    │
│            │                               │                    │
│  ┌─────────▼─────────┐           ┌─────────▼─────────┐          │
│  │   Layer 1a:       │           │   Layer 1b:       │          │
│  │   Megatron-LM     │◄─────────►│   SGLang          │          │
│  │   (Training)      │ CUDA IPC  │   (Inference)     │          │
│  └───────────────────┘           └───────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
flux/
├── core/                    # Core abstractions
│   ├── config.py           # Pydantic configuration classes
│   ├── types.py            # Type definitions
│   └── trajectory.py       # Trajectory data structures
├── controller/              # Adaptive control plane
│   ├── adaptive_async.py   # PID-based async ratio controller
│   ├── staleness.py        # Staleness measurement
│   └── importance.py       # Importance weight correction
├── rollout/                 # Rollout generation
│   ├── manager.py          # Streaming rollout with APRIL
│   └── sglang_client.py    # SGLang HTTP client
├── training/                # Training engine
│   ├── megatron_engine.py  # Megatron-LM integration
│   ├── batch_composer.py   # Smart batch composition
│   └── algorithms/         # PPO, GRPO, etc.
├── sync/                    # Weight synchronization
│   ├── weight_sync.py      # Sync manager
│   └── cuda_ipc.py         # Zero-copy CUDA IPC
└── coordinator/             # Lightweight coordinator
    └── coordinator.py      # Main asyncio coordinator
```

---

## Design Philosophy

### 1. Continuous Spectrum, Not Binary Choice

Flux operates anywhere on the sync-async spectrum, adapting in real-time based on training dynamics.

### 2. Native First

- **Direct Megatron-LM integration** — not wrapped
- **SGLang HTTP API** — simple and efficient
- **Pure asyncio** — no Ray abstraction layer

### 3. Simple > Clever

- < 5000 lines of core code
- No magic, explicit control flow
- Easy to debug and extend

---

## Documentation

| Resource | Description |
|:---------|:------------|
| [Design Specification](docs/flux-design-spec.md) | Detailed architecture and design decisions |
| [Project Structure](docs/flux-project-structure.md) | Implementation status and roadmap |
| API Reference | Coming soon |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest                           # all tests
pytest tests/unit/               # unit tests only
pytest -m "not slow"             # skip slow tests

# Code quality
ruff check . && black --check . && mypy flux/
```

---

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

```bash
# Setup development environment
pip install -e ".[dev]"

# Run all checks before commit
ruff check . && black --check . && mypy flux/ && pytest
```

---

## Citation

```bibtex
@software{flux2025,
  title  = {Flux: An Adaptive Post-Training Framework for LLMs},
  year   = {2025},
  url    = {https://github.com/flux-team/flux}
}
```

## License

Apache 2.0

---

<div align="center">
<strong>Flux: Where stability meets efficiency</strong>
</div>
