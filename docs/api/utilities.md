---
title: Utilities API
description: Helper classes and functions
---

# Utilities API

Helper classes for checkpointing, monitoring, and fault tolerance.

## CheckpointManager

Manages checkpoint saving and loading.

```python
from flux.utils import CheckpointManager

manager = CheckpointManager(
    output_dir="./checkpoints",
    max_checkpoints=10,
    keep_best=3,
)

# Save
manager.save(state_dict, step=1000, metrics={"loss": 0.5})

# Load latest
state_dict = manager.load_latest()

# Load best
state_dict = manager.load_best(metric="loss", mode="min")
```

## GracefulShutdown

Handles signals for clean shutdown.

```python
from flux.utils import GracefulShutdown

with GracefulShutdown(timeout=60) as shutdown:
    shutdown.register_cleanup(trainer.save_checkpoint)
    
    while not shutdown.should_exit:
        trainer.step()
```

## @with_retry

Retry decorator for fault tolerance.

```python
from flux.utils import with_retry

@with_retry(max_retries=3, delay=1.0, backoff=2.0)
def flaky_operation():
    ...
```

## MetricsRegistry

Prometheus metrics export.

```python
from flux.utils import MetricsRegistry

metrics = MetricsRegistry()
metrics.register_gauge("training_loss")
metrics.update("training_loss", 0.5)

# Export for Prometheus scraping
metrics.export(port=9090)
```

## See Also

- [How-to: Checkpoints](../how-to/checkpoints.md)
- [How-to: Monitoring](../how-to/monitoring.md)
