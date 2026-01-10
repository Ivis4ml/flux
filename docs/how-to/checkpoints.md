---
title: Resume from Checkpoint
description: Save and restore training state
---

# Resume from Checkpoint

Manage checkpoints for long training runs.

## Automatic Checkpoints

```yaml
checkpoint_interval: 1000  # Save every 1000 steps
```

## Manual Checkpoints

```python
trainer.save_checkpoint("checkpoints/my-checkpoint")
trainer.load_checkpoint("checkpoints/my-checkpoint")
```

## Resume Training

```python
from flux import FluxTrainer, FluxConfig

config = FluxConfig.from_yaml("config.yaml")
trainer = FluxTrainer(config)

# Resume from latest
trainer.load_checkpoint("checkpoints/latest")

# Continue training
result = trainer.fit(prompts, num_steps=5000)
```

## CheckpointManager

```python
from flux.utils import CheckpointManager

manager = CheckpointManager(
    output_dir="./checkpoints",
    max_checkpoints=10,  # Keep last 10
    keep_best=3,         # Keep 3 best
)

# Save with metrics
manager.save(state_dict, step=1000, metrics={"loss": 0.5})

# Load best
best = manager.load_best(metric="loss", mode="min")

# Load latest
latest = manager.load_latest()
```

## Checkpoint Contents

- Model weights
- Optimizer state
- Training step
- RNG states
- Config

## Distributed Checkpoints

```yaml
checkpoint:
  distributed_checkpoint: true  # Shard across GPUs
```

## See Also

- [Production Deployment](../tutorials/production.md)
