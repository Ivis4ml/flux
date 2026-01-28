---
title: FluxCoordinator API
description: Low-level coordinator for custom training loops
---

# FluxCoordinator

Low-level API for custom training loops and advanced use cases.

## Overview

`FluxCoordinator` is the core orchestration layer that `FluxTrainer` uses internally. Use it when you need fine-grained control over:

- Training step execution
- Weight synchronization timing
- Async/sync decisions
- Custom batching logic

## Basic Usage

```python
from flux.coordinator import FluxCoordinator

coordinator = FluxCoordinator(
    config=config,
    reward_function=my_reward,
)

await coordinator.initialize()

async for result in coordinator.run_training_async(prompts, num_steps=1000):
    print(f"Step {result.step}: loss={result.training_result.loss}")
    
await coordinator.shutdown()
```

## Synchronous Usage

```python
for result in coordinator.run_training(prompts, num_steps):
    # Process each step result
    pass
```

## StepResult

Returned for each training step:

```python
@dataclass
class StepResult:
    step: int
    training_result: TrainingStepResult | None
    staleness_metrics: StalenessMetrics | None
    async_decision: AsyncDecision | None
    batch_size: int
    num_trajectories: int
    elapsed_ms: float
    metrics: dict[str, float]
```

## Key Methods

### initialize() / shutdown()

```python
await coordinator.initialize()  # Set up all components
await coordinator.shutdown()    # Clean shutdown
```

### run_training() / run_training_async()

Main training loop, sync and async versions.

### save_checkpoint() / load_checkpoint()

```python
coordinator.save_checkpoint("path/to/checkpoint")
coordinator.load_checkpoint("path/to/checkpoint")
```

### get_statistics()

```python
stats = coordinator.get_statistics()
# Returns comprehensive training statistics
```

---

## Mode Gate

The `ModeGate` is a state machine that controls sync/async training transitions.

### AsyncMode Enum

```python
from flux.controller import AsyncMode

class AsyncMode(Enum):
    SYNC_BARRIER = auto()    # Waiting for all in-flight rollouts
    ASYNC_RUNNING = auto()   # Normal async operation
    THROTTLED = auto()       # Capacity exhausted, backpressure active
```

### ModeGateState

```python
@dataclass
class ModeGateState:
    mode: AsyncMode
    reason: str
    staleness: float
    capacity: int
    in_flight: int
    buffer_fill_ratio: float = 0.0
```

### ModeGate Class

```python
from flux.controller import ModeGate, ModeGateConfig

config = ModeGateConfig(
    staleness_threshold=0.3,
    capacity_low_watermark=0,
    buffer_high_watermark=0.9,
    min_barrier_duration_ms=100.0,
)

gate = ModeGate(config)

# Evaluate current state
state = gate.evaluate(
    staleness=0.25,
    capacity=10,
    buffer_fill_ratio=0.5,
    in_flight=5,
)

# Check if new rollouts can be submitted
if gate.can_submit_rollout():
    submit_rollout()

# Enforce sync barrier (async)
await gate.enforce_barrier(wait_for_in_flight_fn, timeout=30.0)
```

### ModeGateIntegration

Helper class for coordinator integration:

```python
from flux.controller import ModeGateIntegration

integration = ModeGateIntegration(
    mode_gate=gate,
    staleness_manager=staleness_mgr,
    trajectory_buffer=buffer,
)

# Check and enforce mode transitions
state = await integration.check_and_enforce(wait_for_in_flight)

# Get available rollout slots
slots = integration.get_rollout_slots()
```

### State Transitions

| Current State | Condition | New State | Action |
|:--------------|:----------|:----------|:-------|
| ASYNC_RUNNING | `staleness > threshold` | SYNC_BARRIER | Wait for in-flight |
| ASYNC_RUNNING | `capacity <= 0` | THROTTLED | Pause rollouts |
| ASYNC_RUNNING | `buffer > 90%` | THROTTLED | Pause rollouts |
| SYNC_BARRIER | In-flight drained | ASYNC_RUNNING | Resume |
| THROTTLED | Capacity recovered | ASYNC_RUNNING | Resume |

---

## See Also

- [FluxTrainer](trainer.md) - High-level API
- [Architecture](../concepts/architecture.md) - System design
