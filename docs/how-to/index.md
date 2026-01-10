---
title: How-to Guides
description: Practical guides for common Flux tasks
---

# How-to Guides

Task-oriented guides that help you accomplish specific goals with Flux.

---

## Available Guides

<div class="grid cards" markdown>

-   :material-puzzle-plus:{ .lg .middle } **Add a Custom Algorithm**

    ---

    Create and register your own RL algorithm

    [:octicons-arrow-right-24: Guide](custom-algorithm.md)

-   :material-star:{ .lg .middle } **Implement Custom Rewards**

    ---

    Build reward functions for your specific task

    [:octicons-arrow-right-24: Guide](custom-rewards.md)

-   :material-server-network:{ .lg .middle } **Scale to Multiple Nodes**

    ---

    Distribute training across multiple machines

    [:octicons-arrow-right-24: Guide](multi-node.md)

-   :material-bug:{ .lg .middle } **Debug Training Issues**

    ---

    Diagnose and fix common training problems

    [:octicons-arrow-right-24: Guide](debugging.md)

-   :material-speedometer:{ .lg .middle } **Optimize Performance**

    ---

    Get maximum throughput from your hardware

    [:octicons-arrow-right-24: Guide](performance.md)

-   :material-chart-line:{ .lg .middle } **Monitor with Prometheus**

    ---

    Set up comprehensive training monitoring

    [:octicons-arrow-right-24: Guide](monitoring.md)

-   :material-content-save:{ .lg .middle } **Resume from Checkpoint**

    ---

    Save and restore training state

    [:octicons-arrow-right-24: Guide](checkpoints.md)

</div>

---

## Quick Answers

### How do I change the learning rate during training?

```python
# Use a scheduler
from flux import FluxConfig

config = FluxConfig(
    learning_rate=1e-6,
    lr_scheduler="cosine",
    lr_scheduler_config={
        "warmup_steps": 100,
        "min_lr": 1e-7,
    }
)
```

### How do I use multiple reward functions?

```python
from flux.rewards import CompositeReward, LengthReward, KeywordReward

reward = CompositeReward([
    (LengthReward(target=200), 0.3),
    (KeywordReward(required=["answer"]), 0.7),
])
```

### How do I save checkpoints more frequently?

```yaml
checkpoint:
  save_steps: 100  # Save every 100 steps
  max_checkpoints: 10
  keep_best: 3
```

### How do I use a different model for inference?

```yaml
model_path: "Qwen/Qwen3-8B"  # Training model

sglang:
  model_path: "Qwen/Qwen3-8B-Instruct"  # Inference model (optional)
  base_url: http://localhost:8000
```

### How do I adjust the async ratio manually?

```yaml
adaptive_async:
  enabled: false  # Disable adaptive control
  fixed_async_ratio: 0.5  # Use fixed ratio
```

---

## By Category

### Training

| Guide | Description |
|:------|:------------|
| [Custom Algorithm](custom-algorithm.md) | Add your own RL algorithm |
| [Custom Rewards](custom-rewards.md) | Build task-specific rewards |
| [Debugging](debugging.md) | Fix training issues |
| [Checkpoints](checkpoints.md) | Save and restore training |

### Scaling

| Guide | Description |
|:------|:------------|
| [Multi-Node](multi-node.md) | Distribute across machines |
| [Performance](performance.md) | Optimize throughput |

### Operations

| Guide | Description |
|:------|:------------|
| [Monitoring](monitoring.md) | Set up Prometheus/Grafana |

---

## Contributing Guides

Have a guide to share? We welcome contributions!

1. Fork the repository
2. Add your guide to `docs/how-to/`
3. Update this index
4. Submit a pull request

See [Contributing](../contributing/index.md) for details.
