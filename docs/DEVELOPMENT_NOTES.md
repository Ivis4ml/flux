# Flux Development Notes

This document provides a comprehensive record of the Flux framework development process, documenting design decisions, implementation details, and lessons learned.

## Project Overview

**Flux** is an adaptive post-training (RLHF) framework for Large Language Models. It combines the best aspects of existing frameworks:

- **VERL**: Synchronous stability, robust training convergence
- **AReaL**: Asynchronous efficiency, high GPU utilization
- **Slime**: SGLang-native simplicity, APRIL strategy

The framework dynamically adjusts the sync/async training ratio based on measured staleness, achieving both synchronous stability and asynchronous efficiency.

## Development Timeline

### Phase 1: Core Infrastructure

**Goal**: Establish foundational types, configuration, and basic abstractions.

**Components Created**:
1. `flux/core/types.py` - Core type definitions (PolicyVersion, TrajectoryID, etc.)
2. `flux/core/config.py` - Pydantic-based configuration system
3. `flux/core/trajectory.py` - Trajectory and TrajectoryBuffer classes
4. `flux/core/constants.py` - Framework constants

**Key Design Decisions**:
- Used Pydantic for configuration validation with strict typing
- Implemented PolicyVersion with monotonic ordering for staleness tracking
- TrajectoryBuffer supports stratified sampling by policy version

**Tests**: 65 tests covering all core components

### Phase 2: RL Algorithms

**Goal**: Implement algorithm-agnostic infrastructure supporting multiple RL methods.

**Components Created**:
1. `flux/training/algorithms/base.py` - Registry pattern for algorithms
2. `flux/training/algorithms/advantages.py` - Advantage estimators (GAE, GRPO, RLOO, etc.)
3. `flux/training/algorithms/losses.py` - Policy losses (PPO, REINFORCE, DPO, etc.)
4. `flux/training/algorithms/optimizer.py` - PolicyGradientOptimizer orchestrator
5. `flux/training/algorithms/reference.py` - Reference policy management
6. `flux/training/metrics.py` - Training metrics aggregation

**Key Design Decisions**:
- **Registry Pattern**: Algorithms are registered via decorators (`@register_adv_estimator`, `@register_policy_loss`)
- **Composable Design**: Advantage estimators and policy losses are independent and composable
- **Algorithm Support**: PPO, GRPO, DPO, REINFORCE, DAPO, RLOO, GSPO all implemented
- **Importance Correction**: Built-in support for off-policy data with importance sampling

**Tests**: 137 tests (72 new) covering all algorithms

### Phase 3: Async Infrastructure

**Goal**: Build asynchronous task execution and staleness management.

**Components Created**:
1. `flux/coordinator/async_runner.py` - AsyncTaskRunner for background task execution
2. `flux/coordinator/async_runner.py` - BatchTaskDispatcher for staleness-aware batching
3. `flux/controller/staleness.py` - StalenessManager for computing staleness metrics
4. `flux/controller/adaptive_async.py` - AdaptiveAsyncScheduler with PID controller
5. `flux/controller/importance.py` - ImportanceWeightComputer for off-policy correction

**Key Design Decisions**:
- **PID Controller**: Uses PID control theory to adjust async ratio based on staleness error
- **Multi-metric Staleness**: Combines version gap, KL divergence, and importance weight variance
- **Exponential Moving Average**: Smoothed staleness tracking for stability
- **Task Priorities**: Support for task prioritization in async queue

**Staleness Computation**:
```python
staleness = (
    version_gap_weight * normalized_version_gap +
    kl_weight * normalized_kl +
    iw_var_weight * normalized_iw_variance
)
```

**PID Control**:
```python
error = target_staleness - current_staleness
adjustment = kp * error + ki * integral + kd * derivative
async_ratio = clamp(async_ratio + adjustment, min_ratio, max_ratio)
```

**Tests**: 225 tests (88 new) covering async infrastructure

### Phase 4: SGLang Integration

**Goal**: Integrate with SGLang for efficient inference and implement weight synchronization.

**Components Created**:
1. `flux/rollout/sglang_client.py` - SGLangClient for HTTP-based inference
2. `flux/rollout/rollout_manager.py` - RolloutManager orchestrating generation
3. `flux/sync/delta_sync.py` - DeltaCompressor for efficient weight updates
4. `flux/sync/weight_manager.py` - WeightSyncManager coordinating synchronization

**Key Design Decisions**:
- **HTTP-based Communication**: Simple REST API for SGLang interaction
- **Delta Compression**: Only transmit changed weights with configurable threshold
- **Quantization Support**: 8-bit and 16-bit quantization for bandwidth reduction
- **Async Generation**: Non-blocking rollout generation with callbacks

**APRIL Strategy**:
- Oversample rollouts (generate 1.5x needed)
- Abort long-tail generations (timeout-based)
- Reuse partial trajectories (above threshold)

**Tests**: 326 tests (101 new) covering SGLang integration

### Phase 5: Coordinator & Trainer

**Goal**: Tie everything together into a complete training loop.

**Components Created**:
1. `flux/training/batch_composer.py` - SmartBatchComposer for intelligent batching
2. `flux/rewards/base.py` - RewardFunction base class and CompositeReward
3. `flux/rewards/rule_based.py` - Rule-based rewards (Length, Format, Keyword, etc.)
4. `flux/rewards/model_based.py` - Neural reward models (RewardModel, LLMJudge, PRM)
5. `flux/coordinator/coordinator.py` - FluxCoordinator orchestrating training
6. `flux/trainer.py` - FluxTrainer high-level API
7. `flux/cli.py` - Command-line interface

**Key Design Decisions**:
- **Smart Batching**: Length bucketing + staleness balancing + curriculum learning
- **Modular Rewards**: Base class with scaling/clipping, composite rewards for combinations
- **Training Loop**: Orchestrates rollout → reward → batch → train → sync cycle
- **Context Manager**: FluxTrainer supports `with` statement for resource management

**Batch Composition Strategy**:
1. Group trajectories by length buckets
2. Stratified sampling across policy versions
3. Curriculum ordering (difficulty-based)
4. Decaying randomness for exploration

**Tests**: 414 tests (88 new) covering coordinator and trainer

### Phase 6: Production Readiness

**Goal**: Add utilities for production deployment.

**Components Created**:
1. `flux/utils/checkpoint.py` - CheckpointManager with atomic saves, registry, best tracking
2. `flux/utils/fault_tolerance.py` - GracefulShutdown, RetryConfig, CircuitBreaker
3. `flux/utils/monitoring.py` - Prometheus metrics, health checks, HTTP exporter
4. `docs/` - Comprehensive documentation (getting started, config, algorithms, API)
5. `examples/` - Example scripts (basic, custom reward, custom algorithm)
6. `configs/` - Production configurations (qwen3-8b-8gpu, qwen3-72b-64gpu)

**Key Design Decisions**:
- **Atomic Checkpoints**: Temp directory + rename prevents corruption
- **Circuit Breaker**: Prevents cascade failures in distributed systems
- **Prometheus Export**: Standard /metrics endpoint for monitoring
- **Graceful Shutdown**: Signal handling with cleanup callbacks

**Final Test Count**: 485 tests (71 new)

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         FluxTrainer                              │
│  - High-level API                                                │
│  - Context manager                                               │
│  - Callbacks                                                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FluxCoordinator                            │
│  - Orchestrates training loop                                    │
│  - Manages state and versioning                                  │
│  - Coordinates async/sync                                        │
└─────────────────────────────────────────────────────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ RolloutManager│     │ SmartBatch    │     │ Algorithm     │
│ + SGLangClient│     │ Composer      │     │ Optimizer     │
└───────────────┘     └───────────────┘     └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Trajectory    │     │ Adaptive      │     │ Weight        │
│ Buffer        │     │ Async Sched   │     │ SyncManager   │
└───────────────┘     └───────────────┘     └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Staleness Manager   │
                    │   + PID Controller    │
                    └───────────────────────┘
```

## Key Algorithms

### Adaptive Async Control

The heart of Flux is the adaptive async controller:

1. **Measure Staleness**: Combine version gap, KL divergence, importance weight variance
2. **Compute Error**: `error = target - current`
3. **PID Adjustment**: `adjustment = kp*error + ki*∫error + kd*d(error)/dt`
4. **Update Ratio**: `async_ratio = clamp(ratio + adjustment, min, max)`
5. **Apply**: Determine sync points based on ratio

### GRPO (Group Relative Policy Optimization)

Default algorithm with excellent sample efficiency:

```python
# Generate multiple responses per prompt
responses = generate(prompt, n=group_size)

# Compute rewards
rewards = [reward_fn(r) for r in responses]

# Normalize within group
advantages = (rewards - mean(rewards)) / std(rewards)

# Policy gradient with clipping
loss = -min(ratio * adv, clip(ratio) * adv)
```

### Smart Batch Composition

Intelligent batching for efficient training:

1. **Length Buckets**: Group sequences by length to minimize padding
2. **Staleness Stratification**: Mix fresh and stale data for stability
3. **Curriculum Learning**: Start random, progressively order by difficulty
4. **Dynamic Sizing**: Adjust batch size based on sequence lengths

## Lessons Learned

### 1. Configuration Management
Pydantic proved excellent for configuration:
- Type validation catches errors early
- Default values reduce boilerplate
- Nested configs keep things organized
- YAML/JSON serialization is automatic

### 2. Registry Pattern
The algorithm registry pattern (`@register_adv_estimator`) enables:
- Easy extension without modifying core code
- Runtime algorithm selection via config strings
- Clear separation between algorithm logic and orchestration

### 3. Staleness vs Throughput Trade-off
Key insight: Pure async training is unstable, pure sync is slow.
- Target staleness of 0.3-0.5 works well in practice
- PID gains need tuning per workload (start conservative)
- Version gap is the most reliable staleness signal

### 4. Testing Strategy
Comprehensive testing was essential:
- Unit tests for individual components
- Integration tests for component interactions
- Mock objects for external dependencies (SGLang, models)
- Async testing requires careful event loop management

### 5. PyTorch 2.6 Compatibility
`torch.load` now defaults to `weights_only=True`:
- Must use `weights_only=False` for checkpoint dicts with metadata
- Consider separating metadata from model weights

## File Structure

```
flux/
├── core/                    # Core types and config
│   ├── types.py
│   ├── config.py
│   ├── trajectory.py
│   └── constants.py
├── training/                # Training infrastructure
│   ├── algorithms/          # RL algorithms
│   │   ├── base.py
│   │   ├── advantages.py
│   │   ├── losses.py
│   │   └── optimizer.py
│   ├── batch_composer.py
│   └── metrics.py
├── coordinator/             # Training orchestration
│   ├── coordinator.py
│   └── async_runner.py
├── controller/              # Adaptive control
│   ├── adaptive_async.py
│   ├── staleness.py
│   └── importance.py
├── rollout/                 # Inference
│   ├── sglang_client.py
│   └── rollout_manager.py
├── sync/                    # Weight synchronization
│   ├── delta_sync.py
│   └── weight_manager.py
├── rewards/                 # Reward functions
│   ├── base.py
│   ├── rule_based.py
│   └── model_based.py
├── utils/                   # Utilities
│   ├── checkpoint.py
│   ├── fault_tolerance.py
│   └── monitoring.py
├── trainer.py               # High-level API
└── cli.py                   # Command-line interface

tests/
├── unit/                    # Unit tests
└── integration/             # Integration tests

docs/
├── getting-started.md
├── configuration.md
├── algorithms.md
└── api.md

examples/
├── basic_training.py
├── custom_reward.py
└── custom_algorithm.py

configs/
├── qwen3-8b-8gpu.yaml
└── qwen3-72b-64gpu.yaml
```

## Statistics

- **Total Lines of Code**: ~10,000 (excluding tests)
- **Test Coverage**: 485 tests
- **Modules**: 25+ Python modules
- **Algorithms**: 7 built-in (PPO, GRPO, DPO, REINFORCE, DAPO, RLOO, GSPO)
- **Reward Functions**: 10+ built-in

## Future Directions

1. **Multi-node Training**: Full distributed coordinator implementation
2. **Megatron Integration**: Direct 3D parallelism support
3. **CUDA IPC**: Zero-copy weight transfer
4. **Online RL**: Continuous learning without restarting
5. **Multi-turn Training**: Conversation-level optimization

## References

- [VERL](https://github.com/volcengine/verl) - Synchronous training patterns
- [AReaL](https://github.com/inclusionAI/AReaL) - Asynchronous architecture
- [Slime](https://github.com/THUDM/slime) - APRIL strategy
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [GRPO Paper](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization
