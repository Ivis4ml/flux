<div align="center">

<img src="docs/assets/flux-concept.png" alt="Flux" width="480">

<h2>Adaptive Post-Training Framework for LLMs</h2>

<p>
<strong>The best of all worlds</strong> — Synchronous stability + Asynchronous efficiency + Native simplicity
</p>

<p>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
<a href="https://ivis4ml.github.io/flux/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="Documentation"></a>
</p>

<p>
<a href="#installation">Installation</a> •
<a href="#quick-start">Quick Start</a> •
<a href="#key-features">Features</a> •
<a href="#architecture">Architecture</a> •
<a href="https://ivis4ml.github.io/flux/">Documentation</a>
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

| Aspect | [VERL](https://github.com/volcengine/verl) | [AReaL](https://github.com/inclusionAI/AReaL) | [Slime](https://github.com/THUDM/slime) | **Flux** |
|:-------|:----:|:-----:|:-----:|:--------:|
| Sync Strategy | Fixed sync | Fixed async | Both modes | **Adaptive** |
| Orchestration | Ray | Custom | HTTP | **asyncio** |
| Training Backend | Megatron/FSDP | Custom | Megatron | **Megatron** |
| Inference Backend | vLLM/SGLang | Custom | SGLang | **SGLang** |
| Weight Sync | Ray Object Store | Custom | CUDA IPC | **CUDA IPC + NCCL** |
| Staleness Handling | N/A | Staleness-aware PPO | APRIL | **Unified** |
| Code Complexity | ~15k LOC | ~25k LOC | ~8k LOC | **<5k LOC**<sup>†</sup> |

<sup>†</sup> Core runtime only; excludes third-party integrations (Megatron, SGLang) and test code.

**Framework Notes:**
- **VERL** (ByteDance/HKU): HybridFlow controller with Ray-based orchestration, stable but has GPU bubble time
- **AReaL** (Ant Group/Tsinghua): Fully async with importance weight correction, high throughput but potential instability
- **Slime** (THUDM/Zhipu): SGLang-native with APRIL strategy, simple but less flexible dataflow

---

## Key Features

### Adaptive Async Control

Flux uses a **PID controller** to dynamically adjust the sync/async ratio based on measured staleness.

#### What does "async_ratio" control?

The `async_ratio` (float in `[0.1, 0.9]`) controls the **maximum proportion of off-policy data** allowed in training batches:

| async_ratio | Behavior |
|:------------|:---------|
| 0.1 (more sync) | Sync barrier triggered frequently; training waits for fresh rollouts |
| 0.5 (balanced) | Mixed fresh/stale data; moderate overlap between rollout and training |
| 0.9 (more async) | Training proceeds with older data; rollouts run independently |

The controller output affects three subsystems:

1. **Sync Policy**: When `staleness > target + tolerance`, trigger a sync barrier (wait for all in-flight rollouts)
2. **Buffer Policy**: Capacity limit = `(max_version_gap + current_version + 1) × batch_size - in_flight`
3. **Batch Policy**: Stratified sampling balances fresh vs stale trajectories based on `async_ratio`

#### Control Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Flux Control Loop                               │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   1. ROLLOUT GENERATION                                         │
    │      SGLang generates responses → Trajectory Buffer             │
    │      (tracks: behavior_logprobs, policy_version)                │
    │                                                                 │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │   2. STALENESS MEASUREMENT                                      │
    │      For each batch, compute:                                   │
    │        • KL divergence (π_current vs π_behavior)                │
    │        • Importance weight variance                             │
    │        • Version gap (current_version - trajectory_version)     │
    │      → combined_staleness = weighted sum (see formula below)    │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │   3. PID CONTROLLER UPDATE                                      │
    │      error = target_staleness - EMA(staleness)                  │
    │      output = kp*error + ki*∫error + kd*d(error)/dt             │
    │      async_ratio = clip(async_ratio + output, [0.1, 0.9])       │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │   4. SYNC DECISION                                              │
    │      should_sync = (staleness > threshold) OR                   │
    │                    (steps_since_sync > max_steps) OR            │
    │                    (capacity_remaining <= 0)                    │
    │                                                                 │
    │      If should_sync:                                            │
    │        → Wait for rollouts, sync weights to SGLang              │
    │      Else:                                                      │
    │        → Continue async, submit more rollouts                   │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │   5. BATCH COMPOSITION                                          │
    │      Smart composer creates batch:                              │
    │        • Length bucketing (minimize padding)                    │
    │        • Staleness balancing (stratified by version_gap)        │
    │        • Curriculum ordering (easy→hard as training proceeds)   │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │   6. TRAINING STEP                                              │
    │      • Apply importance correction weights                      │
    │      • Compute policy loss (PPO/GRPO/etc.)                      │
    │      • Gradient step, increment policy_version                  │
    │      • Mark weights dirty → lazy sync to SGLang                 │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
                               └──────────────► Loop back to step 1
```

### Staleness Measurement

Flux monitors **three staleness signals** to quantify how much the policy has drifted since data was generated:

#### Signal Definitions

| Signal | Formula | Computed Over | Unit | Danger Threshold |
|:-------|:--------|:--------------|:-----|:-----------------|
| **KL Divergence** | `D_KL(π_behavior ‖ π_current) = E_behavior[log π_behavior - log π_current]` | Token-level, averaged over response tokens | nats | > 0.1 |
| **IW Variance** | `Var(w)` where `w = exp(mean_token(log π_current - log π_behavior))` | Per-trajectory, then variance across batch | unitless | > 2.0 |
| **Version Gap** | `current_version - trajectory_version` | Per-trajectory, averaged over batch | integer (steps) | > 5 |

#### Combined Staleness Score

```python
# Normalize each component to [0, 1]
kl_contrib     = min(1, kl_divergence / kl_normalizer)           # default kl_normalizer = 0.1
iw_contrib     = min(1, iw_variance / iw_normalizer)             # default iw_normalizer = 2.0
version_contrib = min(1, version_gap / max_version_gap)          # default max_version_gap = 5

# Weighted sum (heuristic defaults, can be tuned)
combined_staleness = 0.4 * kl_contrib + 0.3 * iw_contrib + 0.3 * version_contrib
```

The combined staleness is smoothed via EMA (`α=0.1`) before feeding into the PID controller.

### Unified Importance Correction

Flux provides **algorithm-agnostic off-policy correction** that works with any RL algorithm. This corrects for the distribution shift between the behavior policy (which generated the data) and the current policy (which is being trained).

#### Correction Formula

```python
# Per-trajectory importance weight
log_ratio = current_logprobs - behavior_logprobs           # [batch, seq_len]
mean_log_ratio = (log_ratio * mask).sum(dim=-1) / seq_len  # Average over tokens
base_weight = exp(clip(mean_log_ratio, -20, 20))           # [batch]

# Staleness decay (older data gets lower weight)
staleness_weight = staleness_decay ** version_gap          # default decay = 0.99

# Final importance weight
importance_weight = base_weight * staleness_weight
importance_weight = clip(importance_weight, min=0.2, max=5.0)

# Normalize to preserve gradient scale
importance_weight = importance_weight * (batch_size / sum(importance_weight))
```

#### How it integrates with algorithms

| Algorithm | Native Clip | Importance Correction | Interaction |
|:----------|:------------|:---------------------|:------------|
| **PPO** | `clip(ratio, 1-ε, 1+ε)` | Per-sample weight | PPO clip applies to ratio, IW multiplies the clipped loss. No conflict—IW adjusts *magnitude*, PPO clip bounds *direction*. |
| **GRPO** | Group normalization | Per-sample weight | IW applied before group normalization; stale samples contribute less to group baseline. |
| **DPO** | N/A | Per-pair weight | IW downweights stale preference pairs; fresh pairs dominate loss. |
| **REINFORCE** | N/A | Per-sample weight | Direct multiplication with policy gradient. |

For on-policy algorithms (PPO, GRPO), importance correction is **optional but beneficial** when using async training. For truly on-policy training (async_ratio → 0), IW approaches 1.0 for all samples.

### APRIL Strategy

**A**ctive **P**artial **R**ollout for e**I**fficient generation with **L**ong-tail handling:

| Step | Description | Parameters | Behavior |
|:-----|:------------|:-----------|:---------|
| **Oversample** | Generate more prompts than needed | `oversample_ratio=1.5` | Submit 1.5× prompts to buffer against long-tail |
| **Abort** | Cancel long-running generations | `batch_timeout=30s` | After timeout, abort remaining and yield completed |
| **Reuse** | Save partial trajectories | `partial_reuse_threshold=0.5` | If >50% tokens generated, save prefix for continuation |

**Abort Rule**: A generation is aborted when:
- `elapsed_time > batch_timeout`, AND
- `completed_count >= target_count`

**Partial Reuse Consistency**:
- Partial trajectories store: `(prompt, generated_prefix, behavior_logprobs, version)`
- On continuation: new tokens use current policy, prefix tokens keep original logprobs
- Version annotation: `trajectory.version_segments = [(0, prefix_len, old_version), (prefix_len, total_len, new_version)]`
- Consistency weighting: each segment weighted by `staleness_decay ** segment_version_gap`

**Buffer Management**:
- Partial buffer max size: `2 × batch_size` (FIFO eviction)
- Oversample pool max size: `oversample_ratio × prompt_queue_size`

### Smart Batch Composition

#### Strategy Interface

```python
class SmartBatchComposer:
    def compose_batches(
        self,
        trajectories: List[Trajectory],
        current_version: int,
    ) -> Iterator[TrajectoryBatch]:
        """
        Yields batches optimized for:
        1. Minimal padding (length bucketing)
        2. Balanced staleness (stratified sampling)
        3. Curriculum progression (difficulty ordering)
        """
```

#### Length Bucketing

| Bucket | Length Range | Purpose |
|:-------|:-------------|:--------|
| short | 0-512 tokens | Fast iteration, minimal padding |
| medium | 512-1024 tokens | Balanced |
| long | 1024-2048 tokens | Complex reasoning |
| very_long | 2048+ tokens | Full context tasks |

Trajectories are grouped by bucket; batches drawn from single bucket to minimize padding waste.

#### Staleness Balancing

Stratified sampling by `version_gap` buckets:

```python
strata = [
    {"version_gap": 0, "trajectories": [...]},      # Fresh (on-policy)
    {"version_gap": 1, "trajectories": [...]},      # 1 step stale
    {"version_gap": 2, "trajectories": [...]},      # 2 steps stale
    {"version_gap": 3+, "trajectories": [...]},     # 3+ steps stale
]
# Sample proportionally from each stratum to balance batch
```

#### Curriculum Ordering

Difficulty signal sources (configurable):
- **Length**: `difficulty = min(1, total_length / 2048)`
- **Reward**: `difficulty = 1 - (reward + 1) / 2` (lower reward = harder)
- **Custom**: `composer.set_difficulty_function(my_fn)`

Ordering with decaying randomness:
- Early training: `randomness = 1.0` (fully shuffled)
- Late training: `randomness → 0` (strict easy→hard ordering)
- Decay: `randomness = 1 / (1 + decay_rate × curriculum_step)`

### Weight Synchronization

#### Sync Methods by Deployment

| Deployment | Method | Mechanism | Latency |
|:-----------|:-------|:----------|:--------|
| **Same node** (colocated) | CUDA IPC | Zero-copy GPU-to-GPU via IPC handles | ~10ms |
| **Same node** (separate process) | CUDA IPC | Shared memory handles | ~10ms |
| **Cross-node** | NCCL broadcast | Distributed all-reduce pattern | ~100ms |
| **Cross-node** (fallback) | HTTP + torch.save | Serialized state_dict transfer | ~1s+ |

**Current Implementation Status**:
- ✅ Same-node CUDA IPC (implemented)
- ✅ HTTP-based state_dict sync (implemented)
- ✅ Delta compression (implemented)
- 🚧 Multi-node NCCL broadcast (planned)
- 🚧 Checkpoint-based sync (planned)

#### Delta Compression

When enabled (`method="delta"`):
1. Baseline snapshot taken every `snapshot_interval` steps
2. Delta = `current_weights - baseline_weights`
3. Sparse encoding if `sparsity_ratio < sparsity_target`
4. Optional quantization (8/16-bit)

Typical compression: 60-80% bandwidth reduction for fine-tuning workloads.

---

## Supported Algorithms

| Algorithm | Type | Off-policy Support | Required Interface |
|:----------|:-----|:-------------------|:-------------------|
| **PPO** | On-policy | ✅ Optional (IW correction) | `log_prob`, `value`, `advantage` |
| **GRPO** | On-policy | ✅ Optional (IW correction) | `log_prob`, `group_rewards` |
| **DPO** | Preference | ✅ Recommended | `log_prob`, `ref_log_prob`, `preference_pairs` |
| **REINFORCE** | On-policy | ✅ Optional | `log_prob`, `reward` |
| **DAPO** | On-policy | ✅ Optional | `log_prob`, `dynamic_clip_range` |
| **GSPO** | On-policy | ✅ Optional | `log_prob`, `group_stability_term` |
| **RLOO** | On-policy | ✅ Optional | `log_prob`, `leave_one_out_baseline` |

**Algorithm Extension Interface**:

```python
@register_adv_estimator("my_algorithm")
def compute_my_advantage(rewards, mask, **kwargs) -> tuple[Tensor, Tensor]:
    """Return (advantages, returns)"""
    ...

@register_policy_loss("my_loss")
def compute_my_loss(old_logp, logp, adv, mask, **kwargs) -> tuple[Tensor, dict]:
    """Return (loss, metrics_dict)"""
    ...
```

The framework handles staleness, importance correction, and batching automatically.

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

### 1. Prepare Data

Flux expects prompts in JSONL format:

```json
{"prompt": "Explain quantum computing in simple terms."}
{"prompt": "Write a Python function to compute Fibonacci numbers."}
{"prompt": "What are the benefits of exercise?"}
```

### 2. Start SGLang Server

```bash
# Single GPU
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 8000

# Multi-GPU (tensor parallel)
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 8000 --tp 4
```

### 3. Run Training

```python
from flux import FluxTrainer, FluxConfig
from flux.core.config import AdaptiveAsyncConfig, SGLangConfig

config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    sglang=SGLangConfig(base_url="http://localhost:8000"),
    adaptive_async=AdaptiveAsyncConfig(
        target_staleness=0.15,       # Target staleness level
        min_async_ratio=0.1,         # Never fully synchronous
        max_async_ratio=0.9,         # Never fully asynchronous
        kp=0.1, ki=0.01, kd=0.05,    # PID controller gains
    ),
    learning_rate=1e-6,
    batch_size=32,
    num_steps=10000,
)

trainer = FluxTrainer(config)
trainer.fit(
    prompts="data/prompts.jsonl",
    eval_prompts="data/eval_prompts.jsonl",
)
```

### 4. Expected Log Output

```
[Step 100] loss=0.342 | staleness=0.12 | async_ratio=0.45 | throughput=1.2k tok/s
[Step 200] loss=0.298 | staleness=0.14 | async_ratio=0.52 | throughput=1.4k tok/s
[Step 300] loss=0.267 | staleness=0.16 | async_ratio=0.48 (sync triggered)
[Step 400] loss=0.241 | staleness=0.13 | async_ratio=0.55 | throughput=1.5k tok/s
...
[Step 1000] loss=0.156 | staleness=0.15 | async_ratio=0.62 | throughput=1.8k tok/s
                                           ↑ Controller stabilizes around target
```

### 5. Using YAML Configuration

```bash
flux train --config configs/qwen3-8b-8gpu.yaml --prompts data/prompts.jsonl
```

Example config (`configs/qwen3-8b-8gpu.yaml`):

```yaml
model_path: Qwen/Qwen3-8B

sglang:
  base_url: http://localhost:8000

adaptive_async:
  target_staleness: 0.15
  min_async_ratio: 0.1
  max_async_ratio: 0.9
  kp: 0.1
  ki: 0.01
  kd: 0.05

training:
  learning_rate: 1e-6
  batch_size: 32
  gradient_accumulation_steps: 4

algorithm: grpo  # or ppo, dpo, reinforce
```

---

## Architecture

<p align="center">
  <img src="docs/assets/architecture.png" alt="Flux Architecture" width="600">
</p>

### Deployment Topology

| Mode | Training | Inference | Weight Sync | Recommended For |
|:-----|:---------|:----------|:------------|:----------------|
| **Colocated** | GPU 0-3 | GPU 4-7 | CUDA IPC | Single node, low latency |
| **Separated** | Node A | Node B | NCCL/HTTP | Multi-node, high throughput |
| **Hybrid** | GPU 0-5 | GPU 6-7 + Node B | Mixed | Large-scale production |

**Default recommendation**: Colocated on single node for simplicity; separated for 64+ GPU training.

### Coordinator Scope

The `FluxCoordinator` is a **single-machine orchestrator** (not a distributed control plane):

- Runs on rank 0 of the training process
- Uses `asyncio` event loop for non-blocking I/O
- Coordinates via ZeroMQ (local) or HTTP (remote) to SGLang servers
- Checkpoint/resume handled by Megatron's distributed checkpointing

**Fault Tolerance** (current):
- Timeout on rollout requests (default: 60s)
- Retry on transient HTTP failures (3 attempts)
- Graceful degradation: skip failed rollouts, continue training

**Fault Tolerance** (planned):
- SGLang server health monitoring and failover
- Training checkpoint on staleness spike
- Automatic rollout worker restart

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

- < 5000 lines of core runtime code (excludes backends and tests)
- No magic, explicit control flow
- Easy to debug and extend

---

## Performance Targets

| Metric | Target | Measurement Method |
|:-------|:-------|:-------------------|
| **GPU Utilization** | > 80% | nvidia-smi dmon average |
| **Throughput** | 2× VERL baseline | samples/hour on same hardware |
| **Staleness** | Mean < 0.2, Max < 0.4 | Combined staleness metric |
| **KL Blow-up** | < 5% of runs | KL divergence spike detection |
| **Scaling Efficiency** | > 85% at 64 GPUs | Throughput ratio vs linear |

*Note: These are design targets. Benchmarks in progress.*

---

## Documentation

**Full documentation:** [https://ivis4ml.github.io/flux/](https://ivis4ml.github.io/flux/)

| Resource | Description |
|:---------|:------------|
| [Getting Started](https://ivis4ml.github.io/flux/getting-started/) | Installation and quick start guide |
| [Tutorials](https://ivis4ml.github.io/flux/tutorials/) | Step-by-step training tutorials |
| [Algorithms](https://ivis4ml.github.io/flux/algorithms/) | PPO, GRPO, DPO, and custom algorithms |
| [Configuration](https://ivis4ml.github.io/flux/configuration/) | Full configuration reference |
| [API Reference](https://ivis4ml.github.io/flux/api/) | Complete API documentation |

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
