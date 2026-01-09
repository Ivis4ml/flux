# Getting Started with Flux

Flux is an adaptive post-training (RLHF) framework for Large Language Models. It dynamically adjusts the sync/async training ratio based on measured staleness, achieving both synchronous stability and asynchronous efficiency.

## Installation

### From Source

```bash
git clone https://github.com/your-org/flux.git
cd flux
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.0+ (for GPU training)
- SGLang (for inference)

## Quick Start

### Basic Training

```python
from flux import FluxConfig, FluxTrainer
from flux.rewards import LengthReward

# Create configuration
config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    output_dir="./outputs",
    num_steps=1000,
    batch_size=32,
)

# Create trainer with reward function
trainer = FluxTrainer(
    config=config,
    reward_function=LengthReward(target_length=200),
)

# Training data
prompts = [
    "Explain quantum computing",
    "Write a poem about AI",
    "Describe machine learning",
    # ... more prompts
]

# Train
result = trainer.fit(prompts=prompts)
print(f"Training complete: {result.total_steps} steps")
```

### Using CLI

```bash
# Start training
flux train --config configs/qwen3-8b.yaml --prompts data/prompts.json

# Test configuration
flux test --config configs/qwen3-8b.yaml

# Generate samples
flux generate --model outputs/checkpoint-1000 --prompt "Hello world"

# Show system info
flux info
```

## Core Concepts

### Adaptive Async Training

Flux uses a PID controller to dynamically adjust the sync/async ratio:

- **Synchronous**: More stable, better convergence
- **Asynchronous**: Higher throughput, better GPU utilization

The controller monitors staleness (policy version gap, KL divergence) and adjusts the ratio to maintain a target staleness level.

### Trajectories

A trajectory represents one generation + reward cycle:

```python
from flux.core import Trajectory

trajectory = Trajectory(
    id="traj-001",
    prompt="What is 2+2?",
    response="2+2 equals 4.",
    tokens=[1, 2, 3, 4, 5],
    log_probs=[-0.1, -0.2, -0.15, -0.1, -0.12],
    reward=0.8,
)
```

### Reward Functions

Flux supports multiple reward function types:

```python
from flux.rewards import (
    LengthReward,
    CompositeReward,
    FunctionReward,
)

# Simple length-based reward
reward = LengthReward(target_length=200)

# Composite reward combining multiple signals
composite = CompositeReward(
    rewards=[
        (LengthReward(), 0.3),
        (FormatReward(), 0.3),
        (KeywordReward(required=["python"]), 0.4),
    ],
)

# Custom function reward
custom = FunctionReward(
    fn=lambda traj: 1.0 if "answer" in traj.response.lower() else 0.0
)
```

### Batch Composition

The SmartBatchComposer groups trajectories intelligently:

- **Length bucketing**: Groups similar-length sequences
- **Staleness balancing**: Mixes fresh and slightly stale data
- **Curriculum learning**: Progressively harder examples

## Configuration

Flux uses a hierarchical configuration system:

```yaml
# config.yaml
model_path: "Qwen/Qwen3-8B"
output_dir: "./outputs"

# Training settings
num_steps: 5000
batch_size: 32
learning_rate: 1.0e-6

# Adaptive async settings
adaptive_async:
  target_staleness: 0.5
  min_async_ratio: 0.0
  max_async_ratio: 0.8
  kp: 0.1
  ki: 0.01
  kd: 0.05

# Rollout settings
rollout:
  max_length: 2048
  temperature: 0.7
  top_p: 0.9

# Algorithm settings
algorithm:
  name: "grpo"
  clip_ratio: 0.2
```

See [Configuration Guide](configuration.md) for all options.

## Training Workflow

1. **Initialize**: Load model, configure training
2. **Generate Rollouts**: Produce responses from prompts
3. **Compute Rewards**: Score responses
4. **Update Policy**: Train on reward-weighted samples
5. **Sync Weights**: Update inference model (async)
6. **Repeat**: Continue until convergence

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Prompts    │────>│  Generate   │────>│   Reward    │
└─────────────┘     │  Rollouts   │     │   Scoring   │
                    └─────────────┘     └──────┬──────┘
                                               │
                                               v
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Output    │<────│   Policy    │<────│   Batch     │
│   Model     │     │   Update    │     │  Compose    │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Next Steps

- [Configuration Reference](configuration.md)
- [Algorithm Guide](algorithms.md)
- [API Documentation](api.md)
- [Example Scripts](../examples/)
