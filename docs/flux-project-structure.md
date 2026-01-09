# Flux Project Structure

```
flux/
├── README.md                          # User-facing documentation
├── DESIGN.md                          # Design specification (detailed)
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # Package configuration
├── setup.py                           # Installation script
│
├── flux/                              # Main package
│   ├── __init__.py
│   ├── version.py
│   │
│   ├── core/                          # Core abstractions
│   │   ├── __init__.py
│   │   ├── config.py                  # Configuration dataclasses
│   │   ├── types.py                   # Type definitions
│   │   ├── trajectory.py              # Trajectory data structure
│   │   └── metrics.py                 # Metrics collection
│   │
│   ├── controller/                    # Adaptive control plane
│   │   ├── __init__.py
│   │   ├── adaptive_async.py          # Adaptive async controller
│   │   ├── staleness.py               # Staleness measurement
│   │   ├── importance.py              # Importance weight correction
│   │   └── scheduler.py               # Training scheduler
│   │
│   ├── rollout/                       # Rollout management
│   │   ├── __init__.py
│   │   ├── manager.py                 # Streaming rollout manager
│   │   ├── sglang_client.py           # SGLang HTTP client
│   │   ├── length_predictor.py        # Output length prediction
│   │   └── partial_buffer.py          # Partial trajectory buffer
│   │
│   ├── training/                      # Training engine
│   │   ├── __init__.py
│   │   ├── megatron_engine.py         # Megatron integration
│   │   ├── algorithms/                # RL algorithms
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Base algorithm class
│   │   │   ├── ppo.py                 # PPO implementation
│   │   │   └── grpo.py                # GRPO implementation
│   │   └── batch_composer.py          # Smart batch composition
│   │
│   ├── sync/                          # Weight synchronization
│   │   ├── __init__.py
│   │   ├── weight_sync.py             # Weight sync manager
│   │   ├── cuda_ipc.py                # CUDA IPC utilities
│   │   └── delta_compression.py       # Delta compression
│   │
│   ├── coordinator/                   # Lightweight coordinator
│   │   ├── __init__.py
│   │   ├── coordinator.py             # Main coordinator
│   │   ├── communication.py           # ZeroMQ/gRPC communication
│   │   └── checkpoint.py              # Checkpoint management
│   │
│   ├── rewards/                       # Reward computation
│   │   ├── __init__.py
│   │   ├── base.py                    # Base reward class
│   │   ├── rule_based.py              # Rule-based rewards
│   │   └── model_based.py             # Model-based rewards
│   │
│   └── trainer.py                     # Main FluxTrainer class
│
├── configs/                           # Example configurations
│   ├── qwen3-8b-8gpu.yaml
│   ├── qwen3-72b-64gpu.yaml
│   └── qwen3-moe-128gpu.yaml
│
├── scripts/                           # Utility scripts
│   ├── launch.py                      # Multi-node launcher
│   ├── convert_checkpoint.py          # Checkpoint conversion
│   └── benchmark.py                   # Benchmarking script
│
├── tests/                             # Test suite
│   ├── unit/
│   │   ├── test_adaptive_async.py
│   │   ├── test_importance.py
│   │   └── test_batch_composer.py
│   ├── integration/
│   │   ├── test_training_loop.py
│   │   └── test_weight_sync.py
│   └── e2e/
│       └── test_full_training.py
│
├── examples/                          # Usage examples
│   ├── basic_training.py
│   ├── custom_reward.py
│   ├── multi_objective.py
│   └── distributed_training.py
│
├── benchmarks/                        # Benchmark scripts
│   ├── throughput/
│   ├── scalability/
│   └── comparison/
│
└── docs/                              # Documentation
    ├── getting_started.md
    ├── configuration.md
    ├── api_reference.md
    └── design_decisions.md
```

## Key Files Implementation Status

### Phase 1 (Foundation)

| File | Status | Description |
|------|--------|-------------|
| `flux/core/config.py` | 🔴 TODO | Configuration dataclasses |
| `flux/core/types.py` | 🔴 TODO | Type definitions |
| `flux/core/trajectory.py` | 🔴 TODO | Trajectory data structure |
| `flux/rollout/sglang_client.py` | 🔴 TODO | SGLang HTTP client |
| `flux/training/megatron_engine.py` | 🔴 TODO | Megatron integration |
| `flux/sync/weight_sync.py` | 🔴 TODO | Basic weight sync |
| `flux/coordinator/coordinator.py` | 🔴 TODO | Main coordinator |
| `flux/trainer.py` | 🔴 TODO | FluxTrainer class |

### Phase 2 (Adaptive)

| File | Status | Description |
|------|--------|-------------|
| `flux/controller/adaptive_async.py` | 🔴 TODO | Adaptive async controller |
| `flux/controller/staleness.py` | 🔴 TODO | Staleness measurement |
| `flux/controller/importance.py` | 🔴 TODO | Importance correction |
| `flux/rollout/manager.py` | 🔴 TODO | Streaming rollout |
| `flux/training/batch_composer.py` | 🔴 TODO | Smart batching |

### Phase 3 (Optimization)

| File | Status | Description |
|------|--------|-------------|
| `flux/rollout/length_predictor.py` | 🔴 TODO | Length prediction |
| `flux/rollout/partial_buffer.py` | 🔴 TODO | Partial trajectory buffer |
| `flux/sync/delta_compression.py` | 🔴 TODO | Delta compression |
| `flux/sync/cuda_ipc.py` | 🔴 TODO | CUDA IPC utilities |
