---
title: Fine-tuning with DPO
description: Use Direct Preference Optimization for preference learning
tags:
  - tutorial
  - intermediate
  - dpo
---

# Fine-tuning with DPO

Learn how to use Direct Preference Optimization (DPO) for training with preference data.

**Time**: 30 minutes
**Prerequisites**: [Basic RLHF Training](basic-rlhf.md), Preference dataset

---

## Overview

DPO (Direct Preference Optimization) is an alternative to reward-model-based RLHF. Instead of training a reward model, DPO directly optimizes the policy using preference pairs.

In this tutorial, you'll learn:

- [x] What DPO is and when to use it
- [x] Preparing preference data
- [x] Configuring DPO training
- [x] Running and evaluating DPO

---

## When to Use DPO

| Use DPO when... | Use GRPO/PPO when... |
|:----------------|:---------------------|
| You have preference pairs | You have a reward model |
| Simple setup preferred | Maximum flexibility needed |
| Stable training important | Higher throughput needed |

---

## Preparing Preference Data

DPO requires pairs of (chosen, rejected) responses:

```python title="prepare_dpo_data.py"
import json

# Preference pairs
preferences = [
    {
        "prompt": "Explain quantum computing.",
        "chosen": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, unlike classical bits which are either 0 or 1. This allows quantum computers to solve certain problems much faster.",
        "rejected": "Quantum computing is complicated. It uses quantum stuff to compute things faster. I don't really understand it myself."
    },
    {
        "prompt": "Write a professional email.",
        "chosen": "Dear Mr. Smith,\n\nThank you for your inquiry. I would be happy to schedule a meeting at your earliest convenience.\n\nBest regards,\nJohn",
        "rejected": "hey john whats up, yeah we can meet whenever lol"
    },
    # Add more pairs...
]

# Save to JSONL
with open("preferences.jsonl", "w") as f:
    for p in preferences:
        f.write(json.dumps(p) + "\n")

print(f"Created {len(preferences)} preference pairs")
```

---

## Configuration

```yaml title="dpo-config.yaml"
model_path: Qwen/Qwen3-8B
output_dir: ./outputs/dpo

sglang:
  base_url: http://localhost:8000

# Training settings
num_steps: 1000
batch_size: 8
learning_rate: 5.0e-7  # Lower LR for DPO

# DPO algorithm
algorithm:
  name: dpo
  beta: 0.1              # Temperature parameter
  reference_free: false  # Use reference model
  label_smoothing: 0.0   # Optional smoothing

# DPO doesn't need async (no rollouts)
adaptive_async:
  enabled: false

checkpoint:
  save_steps: 200
```

---

## Running DPO Training

```python title="train_dpo.py"
from flux import FluxConfig, FluxTrainer

# Load config
config = FluxConfig.from_yaml("dpo-config.yaml")

# Create trainer
trainer = FluxTrainer(config)

# Train with preference data
result = trainer.fit(
    prompts="preferences.jsonl",
    data_format="preferences",  # Indicates DPO format
)

print(f"Training complete!")
print(f"Final loss: {result.final_loss:.4f}")
```

---

## DPO Loss Function

DPO optimizes the following objective:

$$
\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)
$$

Where:
- $y_w$ = chosen (winning) response
- $y_l$ = rejected (losing) response
- $\beta$ = temperature (controls strength)
- $\pi_{ref}$ = reference model (original)

---

## Key Parameters

### Beta (Temperature)

Controls how strongly to prefer chosen over rejected:

| Beta | Effect |
|:-----|:-------|
| 0.01 | Very weak preference learning |
| 0.1 | Standard (recommended) |
| 0.5 | Strong preference learning |
| 1.0 | Very strong, may be unstable |

```yaml
algorithm:
  name: dpo
  beta: 0.1  # Start here
```

### Reference-Free DPO

Skip the reference model (simpler but less stable):

```yaml
algorithm:
  name: dpo
  reference_free: true
```

---

## Monitoring DPO Training

```python
@trainer.add_step_callback
def log_dpo_metrics(result):
    metrics = result.metrics
    print(f"Step {result.step}: "
          f"loss={metrics['loss']:.4f}, "
          f"chosen_reward={metrics.get('chosen_reward', 0):.3f}, "
          f"rejected_reward={metrics.get('rejected_reward', 0):.3f}, "
          f"reward_margin={metrics.get('reward_margin', 0):.3f}")
```

### What to Look For

| Metric | Good Sign | Warning Sign |
|:-------|:----------|:-------------|
| `loss` | Decreasing | Stuck or increasing |
| `reward_margin` | Increasing (> 0) | Negative or decreasing |
| `chosen_reward` | Higher than rejected | Similar to rejected |

---

## Evaluation

After training, compare chosen vs rejected preference:

```python
from flux import FluxTrainer

trainer = FluxTrainer(config)
trainer.load_checkpoint("outputs/dpo/best")

# Test prompts
test_prompt = "Explain machine learning."

# Generate response
response = trainer.generate(test_prompt)
print(f"Trained model: {response}")
```

---

## Troubleshooting

??? warning "Loss not decreasing"

    **Solutions:**
    - Increase beta (0.1 → 0.2)
    - Reduce learning rate
    - Check data quality (clear chosen/rejected distinction)

??? warning "Model degrades"

    **Solutions:**
    - Add KL regularization
    - Use reference model (reference_free=false)
    - Reduce training steps

??? warning "Overfitting to preferences"

    **Solutions:**
    - Add more diverse preference pairs
    - Reduce number of epochs
    - Use label smoothing

    ```yaml
    algorithm:
      label_smoothing: 0.1
    ```

---

## DPO vs RLHF Comparison

| Aspect | DPO | GRPO/PPO |
|:-------|:----|:---------|
| Data needed | Preference pairs | Prompts + reward model |
| Training complexity | Lower | Higher |
| Compute cost | Lower | Higher (needs rollouts) |
| Flexibility | Lower | Higher |
| Stability | Higher | Depends on reward |

---

## Next Steps

- **[Custom Rewards](custom-rewards.md)** - For RLHF-style training
- **[Algorithms Guide](../algorithms/dpo.md)** - DPO deep dive
- **[Production Deployment](production.md)** - Scale up
