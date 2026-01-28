---
title: Changelog
description: Version history and release notes
---

# Changelog

All notable changes to Flux are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### Native Trainer Contract (v0.2 Architecture)
- **`TrainingBackend` ABC** (`flux/training/base.py`): Abstract base class for all training backends
  - GPU-direct assumptions: all tensors already on target device
  - Async-safe: `train_step()` can be called from asyncio event loop
  - Version tracking: increments after each successful train step
- **`GPUBatch`** dataclass: Frozen, device-owned tensor batch for training
  - Required tensors: `input_ids`, `attention_mask`, `behavior_log_probs`, `rewards`, `version_gaps`
  - Optional tensors: `loss_mask`, `token_rewards`, `ref_log_probs`, `values`, `advantages`, `returns`
  - Validation and device transfer methods
- **`TrainStepResult`** dataclass: Standardized return type with loss, metrics, and timing info
- **`create_training_backend()` factory**: Creates backend from config enum

#### Training Backends
- **`TransformersBackend`** (`flux/training/backends/transformers.py`): HuggingFace Transformers-based backend
  - Suitable for development, single-GPU, and multi-GPU with DDP
  - Supports Flash Attention 2, gradient checkpointing
  - PPO-style clipped surrogate loss implementation
  - Checkpoint save/load support
- **`MegatronEngine` refactoring**: Now implements both `TrainingBackend` and legacy `TrainingEngine` interfaces
  - Dual interface support for backward compatibility
  - GPU-direct batch handling via `GPUBatch`
  - Importance weight computation on GPU

#### Mode Gate (Sync/Async State Machine)
- **`AsyncMode`** enum: `SYNC_BARRIER`, `ASYNC_RUNNING`, `THROTTLED`
- **`ModeGate`** class (`flux/controller/mode_gate.py`): State machine controlling sync/async transitions
  - Priority-based state transitions (capacity > staleness > buffer fill)
  - Hysteresis to prevent rapid oscillation
  - Barrier enforcement with timeout
- **`ModeGateConfig`** dataclass: Configuration for thresholds and watermarks
- **`ModeGateState`** dataclass: Current state with reason and metrics
- **`ModeGateIntegration`** helper: Integration with staleness manager and trajectory buffer

#### Documentation
- Initial documentation website with MkDocs Material
- Comprehensive tutorials and how-to guides
- API reference documentation
- Updated architecture documentation with new components
- Training backend and Mode Gate API documentation

### Changed
- Reorganized documentation structure for better navigation
- Updated `flux/training/__init__.py` with new exports
- Updated `flux/controller/__init__.py` with ModeGate exports

### Fixed
- Documentation links and cross-references

---

## [0.1.0] - 2025-01-XX

### Added

#### Core Features
- **Adaptive Async Controller**: PID-based dynamic sync/async ratio adjustment
- **Staleness Measurement**: KL divergence, importance weight variance, version gap tracking
- **Unified Importance Correction**: Algorithm-agnostic off-policy correction
- **APRIL Strategy**: Oversample, abort long-tail, reuse partial trajectories
- **Smart Batch Composer**: Length bucketing, staleness balancing, curriculum learning

#### Algorithms
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Policy Optimization)
- DPO (Direct Preference Optimization)
- REINFORCE with configurable baselines
- DAPO (Decoupled clip and dynamic sampling)
- RLOO (Leave-One-Out baseline)
- GSPO (Group Stability Policy Optimization)
- Registry pattern for custom algorithms

#### Infrastructure
- Megatron-LM integration for distributed training
- SGLang HTTP client for inference
- CUDA IPC weight synchronization
- Delta compression for efficient weight transfer
- Checkpoint management with best model tracking
- Prometheus metrics export

#### Configuration
- Pydantic-based hierarchical configuration
- YAML configuration file support
- Environment variable overrides
- Configuration validation

#### CLI
- `flux train` - Run training
- `flux test` - Test configuration
- `flux generate` - Generate samples
- `flux info` - System information

#### Reward Functions
- LengthReward
- FormatReward
- KeywordReward
- CompositeReward
- FunctionReward
- RewardModel (neural)
- LLMJudge

### Infrastructure
- Project scaffolding and CI/CD setup
- pytest test suite with unit and integration tests
- Type hints throughout codebase
- Ruff linting and Black formatting

---

## Version History

| Version | Date | Highlights |
|:--------|:-----|:-----------|
| 0.1.0 | 2025-01 | Initial release |

---

## Upgrade Guides

### Upgrading to 0.1.0

This is the initial release. No upgrade path required.

---

## Deprecation Policy

- **Minor versions** (0.x.0): May include breaking changes during initial development (pre-1.0)
- **Patch versions** (0.0.x): Bug fixes and documentation only
- **Major versions** (x.0.0): May include breaking changes after 1.0

Deprecated features will be marked in documentation and emit warnings for at least one minor version before removal.

---

## Release Schedule

We aim to release:
- **Patch releases**: As needed for bug fixes
- **Minor releases**: Monthly with new features
- **Major releases**: When significant changes warrant

---

## Contributing

See [Contributing Guide](contributing/index.md) for how to contribute to Flux.
