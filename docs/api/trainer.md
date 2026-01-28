---
title: FluxTrainer API
description: High-level training interface
---

# FluxTrainer

The main entry point for RLHF training with Flux.

## Overview

`FluxTrainer` orchestrates the complete RLHF training pipeline:

1. Rollout generation via SGLang
2. Reward computation
3. Advantage estimation
4. Policy optimization via Megatron
5. Weight synchronization

## Basic Usage

```python
from flux import FluxTrainer, FluxConfig

config = FluxConfig(model_path="Qwen/Qwen3-8B")
trainer = FluxTrainer(config)
result = trainer.fit(prompts, num_steps=10000)
```

## Class Definition

```python
class FluxTrainer:
    def __init__(
        self,
        config: FluxConfig,
        training_engine: MegatronEngine | None = None,
        sglang_client: SGLangClient | None = None,
        reward_function: RewardFunction | None = None,
    ) -> None:
        """Initialize FluxTrainer.

        Args:
            config: FluxConfig with all training settings.
            training_engine: Optional pre-configured training engine.
            sglang_client: Optional pre-configured SGLang client.
            reward_function: Optional custom reward function.
        """
```

## Properties

| Property | Type | Description |
|:---------|:-----|:------------|
| `state` | `TrainingState` | Current training state |
| `coordinator` | `FluxCoordinator` | Underlying coordinator |
| `is_initialized` | `bool` | Whether trainer is set up |

## Methods

### fit()

Run the complete training loop.

```python
def fit(
    self,
    prompts: PromptsType,
    num_steps: int | None = None,
    eval_prompts: PromptsType | None = None,
    eval_interval: int = 500,
    checkpoint_interval: int = 1000,
    callbacks: list[CallbackType] | None = None,
) -> TrainingResult:
```

**Parameters:**

- `prompts`: Training prompts (list of strings or Dataset)
- `num_steps`: Total training steps (overrides config)
- `eval_prompts`: Optional evaluation prompts
- `eval_interval`: Steps between evaluations
- `checkpoint_interval`: Steps between checkpoints
- `callbacks`: Optional list of callbacks

**Returns:** `TrainingResult` with final metrics and paths

### training_loop()

Iterator for custom training loops.

```python
def training_loop(
    self,
    prompts: list[str],
    num_steps: int | None = None,
) -> Iterator[StepResult]:
```

**Example:**

```python
trainer.setup()
for step_result in trainer.training_loop(prompts):
    print(f"Step {step_result.step}: loss={step_result.training_result.loss:.4f}")
    if step_result.step % 1000 == 0:
        trainer.save_checkpoint(f"checkpoint-{step_result.step}")
trainer.teardown()
```

### setup() / teardown()

Manual lifecycle control.

```python
trainer.setup()     # Initialize all components
# ... training ...
trainer.teardown()  # Shutdown components
```

### save_checkpoint() / load_checkpoint()

```python
trainer.save_checkpoint("path/to/checkpoint")
trainer.load_checkpoint("path/to/checkpoint")
```

### evaluate()

Run evaluation on given prompts.

```python
metrics = trainer.evaluate(eval_prompts)
# Returns: {"mean_reward": 0.75, "num_samples": 100}
```

### add_callback() / add_step_callback()

Register callbacks for training events.

```python
@trainer.add_step_callback
def log_metrics(result):
    print(f"Step {result.step}: {result.metrics}")
```

## TrainingResult

Returned by `fit()`:

```python
@dataclass
class TrainingResult:
    final_reward: float = 0.0
    final_loss: float = 0.0
    total_steps: int = 0
    total_samples: int = 0
    total_tokens: int = 0
    checkpoint_path: str | None = None
    log_dir: str | None = None
    reward_history: list[float]
    loss_history: list[float]
    total_time_seconds: float = 0.0
    samples_per_second: float = 0.0
```

## Context Manager

```python
with FluxTrainer(config) as trainer:
    result = trainer.fit(prompts)
# Automatic teardown on exit
```

---

## Training Backends

Flux supports multiple training backends via the `TrainingBackend` abstraction.

### TrainingBackend ABC

The base interface for all training backends:

```python
from flux.training import TrainingBackend, GPUBatch, TrainStepResult

class TrainingBackend(Protocol):
    @property
    def version(self) -> int:
        """Current policy version (incremented after each train_step)."""
        ...

    @property
    def device(self) -> torch.device:
        """Device this backend operates on."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether backend has been initialized."""
        ...

    def initialize(self, config: Any) -> None:
        """Initialize backend with config. Called once before training."""
        ...

    def train_step(self, batch: GPUBatch) -> TrainStepResult:
        """Execute one training step."""
        ...

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Get model weights for sync. Returns CPU tensors."""
        ...

    def set_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load model weights."""
        ...
```

### GPUBatch

Frozen, device-owned tensor batch for training:

```python
@dataclass(frozen=True)
class GPUBatch:
    # Required tensors
    input_ids: torch.Tensor          # [B, seq]
    attention_mask: torch.Tensor     # [B, seq]
    behavior_log_probs: torch.Tensor # [B, seq]
    rewards: torch.Tensor            # [B] or [B, seq]
    version_gaps: torch.Tensor       # [B]

    # Optional tensors
    loss_mask: torch.Tensor | None = None
    token_rewards: torch.Tensor | None = None
    ref_log_probs: torch.Tensor | None = None
    values: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
```

### TrainStepResult

Result from a single training step:

```python
@dataclass
class TrainStepResult:
    loss: float
    version: int
    metrics: dict[str, float]
    grad_norm: float = 0.0
    batch_size: int = 0
    num_tokens: int = 0
    throughput_tokens_per_sec: float = 0.0
```

### Available Backends

#### TransformersBackend

HuggingFace Transformers-based backend for development:

```python
from flux.training.backends import TransformersBackend
from flux.training.backends.transformers import TransformersConfig

config = TransformersConfig(
    model_path="Qwen/Qwen3-8B",
    learning_rate=1e-6,
    torch_dtype="bfloat16",
    gradient_checkpointing=True,
    use_flash_attention=True,
)

backend = TransformersBackend()
backend.initialize(config)
```

#### MegatronEngine

Megatron-LM backend for production with 3D parallelism:

```python
from flux.training import MegatronEngine

engine = MegatronEngine(config)
engine.initialize(training_config)

# Supports both new GPUBatch and legacy TrajectoryBatch interfaces
result = engine.train_step(gpu_batch)
```

### Backend Factory

Create backends from config:

```python
from flux.training.base import create_training_backend, TrainingBackendType

# In config
config.training_backend = TrainingBackendType.TRANSFORMERS

# Create backend
backend = create_training_backend(config)
```

---

## See Also

- [FluxConfig](config.md) - Configuration options
- [FluxCoordinator](coordinator.md) - Lower-level API
- [Getting Started](../getting-started/quickstart.md) - Quick start guide
