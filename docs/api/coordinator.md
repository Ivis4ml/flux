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

## See Also

- [FluxTrainer](trainer.md) - High-level API
- [Architecture](../concepts/architecture.md) - System design
