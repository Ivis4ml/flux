# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Guiding Principles

**Flux combines the best of existing RL frameworks:**
- **VERL**: Synchronous stability, robust training convergence
- **AReaL**: Asynchronous efficiency, high GPU utilization
- **Slime**: SGLang-native simplicity, APRIL strategy

**All development must prioritize:**
1. **Highest Quality**: Production-ready code, stable training, comprehensive testing
2. **Scalability**: Designed for 64+ GPU distributed training from the start
3. **Extensibility**: Algorithm-agnostic design supporting any RL algorithm (on-policy, off-policy, hybrid, future methods)

When implementing features, always consider: Will this work at scale? Is this algorithm-agnostic? Can future RL methods plug into this?

## Development Tracking

See **[TODO.md](TODO.md)** for the complete development plan and progress tracking.

## Reference Repositories

During development, study these frameworks to learn best practices:

**VERL** - /Users/xxzhou/OSS/verl
- https://github.com/volcengine/verl
- Learn: HybridFlow controller design, 3D-HybridEngine for Megatron integration
- Learn: Robust synchronous training patterns, checkpoint/resume logic
- Key files: `verl/trainer/`, `verl/workers/`, `verl/utils/`
- Avoid: Ray overhead, rigid sync-only approach

**AReaL** - /Users/xxzhou/OSS/AReaL
- https://github.com/inclusionAI/AReaL
- Learn: Fully asynchronous training architecture, staleness-aware PPO
- Learn: High GPU utilization techniques, importance weight correction
- Key files: `areal/algorithms/`, `areal/trainer/`
- Avoid: Training instability from unbounded staleness

**Slime** -/Users/xxzhou/OSS/slime
- https://github.com/THUDM/slime
- Learn: SGLang-native integration, APRIL strategy implementation
- Learn: CUDA IPC weight sync, simple HTTP-based coordination
- Key files: `slime/rollout/`, `slime/sync/`
- Avoid: Less flexible dataflow patterns

**When implementing a new component**, check how each framework handles it:
1. How does VERL ensure stability?
2. How does AReaL maximize throughput?
3. How does Slime keep it simple?

Then design Flux's approach to combine all three strengths.

## Project Overview

Flux is an adaptive post-training (RLHF) framework for LLMs. It dynamically adjusts the sync/async training ratio based on measured staleness, achieving both synchronous stability and asynchronous efficiency.

**Core design:**
- **Adaptive Async**: PID controller maintains target staleness, auto-adjusting sync/async ratio
- **Native-First**: Direct Megatron-LM + SGLang integration without Ray abstraction layer
- **APRIL Strategy**: Oversample rollouts, abort long-tail generations, reuse partial trajectories
- **Unified Importance Correction**: Algorithm-agnostic off-policy correction that works with any RL algorithm

## Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest                           # all tests
pytest tests/unit/               # unit tests only
pytest -m "not slow"             # skip slow tests
pytest tests/unit/test_adaptive_async.py::TestAdaptiveController  # single test class
pytest -k "test_staleness"       # run tests matching pattern

# Code quality
ruff check .                     # linting
ruff check --fix .               # auto-fix lint issues
black .                          # format code
mypy flux/                       # type checking

# All checks before commit
ruff check . && black --check . && mypy flux/ && pytest
```

## Architecture

Three-layer architecture:

1. **Native Execution Engines** (`flux/training/`, `flux/rollout/`)
   - `megatron_engine.py`: Training with Megatron-LM (3D parallelism)
   - `sglang_client.py`: Inference via SGLang HTTP API
   - Weight sync via CUDA IPC for zero-copy transfers (`flux/sync/`)

2. **Lightweight Coordinator** (`flux/coordinator/`)
   - asyncio event loop + ZeroMQ/gRPC communication
   - Orchestrates training steps, weight syncs, checkpoints

3. **Adaptive Control Plane** (`flux/controller/`)
   - `adaptive_async.py`: PID controller adjusting async ratio based on staleness
   - `staleness.py`: Computes staleness from KL divergence, importance weight variance, version gap
   - `importance.py`: Unified importance correction for off-policy data

## Key Abstractions

- **FluxConfig** (`flux/core/config.py`): Pydantic-based nested configuration. All configs are immutable.
- **Trajectory** (`flux/core/trajectory.py`): Generated response with behavior log-probs, version info
- **TrajectoryBuffer**: Holds trajectories with staleness metadata for smart batch composition
- **SmartBatchComposer** (`flux/training/batch_composer.py`): Groups by length, balances staleness, applies curriculum

## Configuration

Configs use Pydantic with strict validation. Key sub-configs:
- `AdaptiveAsyncConfig`: target_staleness, PID gains (kp, ki, kd), async ratio bounds
- `RolloutConfig`: APRIL params (oversample_ratio, batch_timeout, partial_reuse_threshold)
- `BatchComposerConfig`: length bucketing, staleness balancing, curriculum learning
- `WeightSyncConfig`: delta compression, CUDA IPC, snapshot intervals

Load from YAML: `FluxConfig.from_yaml("configs/qwen3-8b-8gpu.yaml")`

## Algorithms

Located in `flux/training/algorithms/`. Designed for extensibility using VERL-style registry pattern.

**Registry pattern** in `base.py`:
- `@register_adv_estimator("name")` - Register advantage estimator functions
- `@register_policy_loss("name")` - Register policy loss functions
- Algorithms selected via config string, not class inheritance

**Built-in algorithms**:
- PPO: Clipped surrogate objective (on-policy)
- GRPO: Group Relative Policy Optimization (default)
- DPO: Direct Preference Optimization
- REINFORCE: Basic policy gradient
- DAPO: Decoupled clip and dynamic sampling
- GSPO: Group Stability Policy Optimization
- RLOO: Leave-One-Out baseline estimator

**Adding new algorithms**:
```python
@register_adv_estimator("my_algorithm")
def compute_my_advantage(rewards, mask, **kwargs) -> tuple[Tensor, Tensor]:
    # Your advantage computation
    return advantages, returns

@register_policy_loss("my_loss")
def compute_my_loss(old_logp, logp, adv, mask, **kwargs) -> tuple[Tensor, dict]:
    # Your loss computation
    return loss, metrics
```

The framework handles staleness, importance correction, and batching - algorithms just compute loss.

## Development Environment

**Primary dev server**: 8x NVIDIA H100 80GB HBM3
- CUDA 12.4, Driver 550.144.03
- ~640GB total GPU memory available

**Environment setup on dev server**:
```bash
# Create conda environment
conda create -n flux python=3.11 -y
conda activate flux

# Install PyTorch with CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install Flux with dev dependencies
cd /path/to/flux
pip install -e ".[dev]"

# Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.version.cuda}')"
```

**SGLang setup for inference testing**:
```bash
# Install SGLang
pip install "sglang[all]"

# Launch server on GPU 0 (for testing)
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 8000 --device cuda:0

# Multi-GPU inference (TP=4)
python -m sglang.launch_server --model-path Qwen/Qwen3-72B --port 8000 --tp 4
```

**GPU allocation strategy for development**:
- GPU 0: SGLang inference server
- GPUs 1-7: Megatron training (TP=4, DP=1 or TP=2, DP=3)
- For unit tests: Single GPU sufficient, use `CUDA_VISIBLE_DEVICES=1`

## Testing Conventions

**Test structure**:
- `tests/unit/`: Fast isolated tests (no GPU required for most)
- `tests/integration/`: Tests requiring multiple components
- `tests/e2e/`: Full training loop tests (marked slow, requires GPU)

**Running tests**:
```bash
# Unit tests only (fast, no GPU)
pytest tests/unit/ -v

# Integration tests (may need GPU)
CUDA_VISIBLE_DEVICES=1 pytest tests/integration/ -v

# E2E tests (slow, needs multi-GPU)
pytest tests/e2e/ -v -m "e2e"

# Skip slow tests during development
pytest -m "not slow"

# Run specific test with output
pytest tests/unit/test_config.py -v -s
```

**Test markers**:
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Requires multiple components
- `@pytest.mark.e2e`: End-to-end training tests
- `@pytest.mark.gpu`: Requires GPU

Uses pytest-asyncio with `asyncio_mode = "auto"`
