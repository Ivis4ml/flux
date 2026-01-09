# Flux API Reference

## Core Classes

### FluxConfig

Main configuration class for Flux training.

```python
from flux import FluxConfig

config = FluxConfig(
    model_path: str,               # Required: Model path or HF model ID
    output_dir: str = "./outputs", # Output directory
    num_steps: int = 1000,         # Total training steps
    batch_size: int = 32,          # Training batch size
    learning_rate: float = 1e-6,   # Learning rate
    seed: int = 42,                # Random seed
)
```

**Methods**:
- `from_yaml(path: str) -> FluxConfig`: Load from YAML file
- `to_yaml(path: str) -> None`: Save to YAML file
- `validate() -> None`: Validate configuration

### FluxTrainer

High-level trainer interface.

```python
from flux import FluxTrainer
from flux.rewards import LengthReward

trainer = FluxTrainer(
    config: FluxConfig,
    reward_function: RewardFunction = None,
)
```

**Methods**:

```python
# Main training method
result = trainer.fit(
    prompts: list[str] | list[dict],  # Training prompts
    eval_prompts: list[str] = None,   # Evaluation prompts
)

# Save/load checkpoints
trainer.save_checkpoint(path: str)
trainer.load_checkpoint(path: str)

# Evaluation
metrics = trainer.evaluate(prompts: list[str])

# Statistics
stats = trainer.get_statistics()

# Add callbacks
trainer.add_callback(fn: Callable)
trainer.add_step_callback(fn: Callable)
```

**Context Manager**:
```python
with FluxTrainer(config) as trainer:
    result = trainer.fit(prompts)
```

### TrainingResult

Result from training.

```python
@dataclass
class TrainingResult:
    total_steps: int           # Steps completed
    final_loss: float          # Final loss value
    total_samples: int         # Samples processed
    total_time_seconds: float  # Training duration
    reward_history: list       # Reward history
    loss_history: list         # Loss history
    samples_per_second: float  # Throughput
    final_metrics: dict        # Final metrics
```

## Coordinator

### FluxCoordinator

Orchestrates the training loop.

```python
from flux.coordinator import FluxCoordinator

coordinator = FluxCoordinator(
    config: FluxConfig,
    reward_function: RewardFunction = None,
)
```

**Properties**:
- `state: CoordinatorState`: Current state
- `current_version: PolicyVersion`: Current policy version
- `is_initialized: bool`: Initialization status

**Methods**:
```python
# Initialize (call before training)
await coordinator.initialize()

# Run training steps
async for result in coordinator.run(prompts):
    print(f"Step {result.step}: loss={result.loss}")

# Manual step
result = await coordinator.step(prompts)

# Shutdown
await coordinator.shutdown()
```

### CoordinatorState

```python
@dataclass
class CoordinatorState:
    step: int                    # Current step
    version: PolicyVersion       # Policy version
    total_trajectories: int      # Total generated
    rewards_sum: float           # Cumulative rewards
    loss_sum: float              # Cumulative loss
```

## Trajectories

### Trajectory

Represents a single generation.

```python
from flux.core import Trajectory

traj = Trajectory(
    id: str,                     # Unique identifier
    prompt: str = "",            # Input prompt
    response: str = "",          # Generated response
    tokens: list[int] = [],      # Token IDs
    log_probs: list[float] = [], # Log probabilities
    reward: float = 0.0,         # Computed reward
    version: PolicyVersion = None, # Generation version
)
```

**Properties**:
- `total_length: int`: Total tokens
- `response_length: int`: Response length
- `mean_log_prob: float`: Mean log probability

### TrajectoryBuffer

Buffer for storing trajectories.

```python
from flux.core import TrajectoryBuffer

buffer = TrajectoryBuffer(max_size: int = 10000)
```

**Methods**:
```python
# Add trajectory
buffer.add(trajectory: Trajectory)
buffer.add_batch(trajectories: list[Trajectory])

# Sample trajectories
samples = buffer.sample(
    n: int,
    current_version: int = 0,
    stratified: bool = False,
)

# Get/clear
all_trajs = buffer.get_all()
buffer.clear()
```

### PolicyVersion

Tracks policy versions for staleness.

```python
from flux.core import PolicyVersion

version = PolicyVersion(
    version_id: int = 0,
    timestamp: float = None,
)
```

## Rewards

### RewardFunction (Base)

```python
from flux.rewards import RewardFunction

class MyReward(RewardFunction):
    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        score = self._compute_score(trajectory)
        return RewardOutput(reward=score)
```

### RewardOutput

```python
@dataclass
class RewardOutput:
    reward: float                # Scalar reward
    metadata: dict = {}          # Additional info
```

### Built-in Rewards

```python
from flux.rewards import (
    LengthReward,
    FormatReward,
    KeywordReward,
    MathReward,
    CodeReward,
    CompositeReward,
    FunctionReward,
)

# Length-based reward
length_reward = LengthReward(
    target_length: int = 200,
    reward_type: str = "linear",  # or "gaussian", "log"
)

# Format checking
format_reward = FormatReward(
    required_sections: list[str] = [],
    forbidden_patterns: list[str] = [],
)

# Keyword reward
keyword_reward = KeywordReward(
    required_keywords: list[str] = [],
    bonus_keywords: list[str] = [],
    penalty_keywords: list[str] = [],
)

# Composite reward
composite = CompositeReward(
    rewards: list[tuple[RewardFunction, float]],  # (reward, weight)
)

# Custom function
custom = FunctionReward(
    fn: Callable[[Trajectory], float],
)
```

### Model-based Rewards

```python
from flux.rewards import RewardModel, LLMJudge

# Neural reward model
reward_model = RewardModel(
    model_path: str,
    device: str = "cuda",
)

# LLM-as-judge
llm_judge = LLMJudge(
    model_path: str,
    prompt_template: str,
)
```

## Batch Composition

### SmartBatchComposer

```python
from flux.training import SmartBatchComposer

composer = SmartBatchComposer(
    config: BatchComposerConfig,
    batch_size: int = 32,
)
```

**Methods**:
```python
# Compose batches from trajectories
batches = composer.compose_batches(
    trajectories: list[Trajectory],
    current_version: int,
)

# Iterator for continuous training
iterator = composer.batch_iterator(
    buffer: TrajectoryBuffer,
    current_version: int,
    min_batch_size: int = 1,
)
```

## Adaptive Async

### AdaptiveAsyncScheduler

```python
from flux.controller import AdaptiveAsyncScheduler

scheduler = AdaptiveAsyncScheduler(
    config: AdaptiveAsyncConfig,
    batch_size: int = 32,
)
```

**Methods**:
```python
# Get current async ratio
ratio = scheduler.get_async_ratio()

# Update with staleness measurement
scheduler.update(staleness: float)

# Check if should sync
should_sync = scheduler.should_sync(step: int)
```

### StalenessManager

```python
from flux.controller import StalenessManager

manager = StalenessManager()
```

**Methods**:
```python
# Compute staleness for batch
staleness = manager.compute_batch_staleness(
    trajectories: list[Trajectory],
    current_version: PolicyVersion,
)

# Individual metrics
version_gap = manager.compute_version_gap(traj, current)
kl_div = manager.compute_kl_divergence(old_logp, new_logp)
iw_variance = manager.compute_iw_variance(weights)
```

## Utilities

### CheckpointManager

```python
from flux.utils import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir: str,
    max_checkpoints: int = 5,
    keep_best: int = 3,
)
```

**Methods**:
```python
# Save checkpoint
metadata = manager.save(
    step: int,
    model_state: dict,
    optimizer_state: dict = None,
    metrics: dict = None,
)

# Load checkpoint
state = manager.load(checkpoint_id: str)
state = manager.load_latest()
state = manager.load_best(metric: str, higher_is_better: bool)

# List/delete
checkpoints = manager.list_checkpoints(tags: list = None)
manager.delete(checkpoint_id: str)
```

### GracefulShutdown

```python
from flux.utils import GracefulShutdown

shutdown = GracefulShutdown(timeout: float = 30.0)
```

**Usage**:
```python
with GracefulShutdown() as shutdown:
    shutdown.register_cleanup(save_checkpoint)

    while not shutdown.is_requested:
        train_step()
```

### Retry Decorator

```python
from flux.utils import with_retry, RetryConfig

@with_retry(max_retries=3, base_delay=1.0)
def flaky_operation():
    ...

# With config
config = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    exponential_base=2.0,
    retry_on=(ConnectionError, TimeoutError),
)

@with_retry(config)
async def async_operation():
    ...
```

### Metrics

```python
from flux.utils import (
    MetricsRegistry,
    MetricsExporter,
    FluxMetrics,
)

# Create registry
registry = MetricsRegistry()

# Create metrics
counter = registry.counter("requests_total", "Total requests")
gauge = registry.gauge("temperature", "Current temperature")
histogram = registry.histogram("latency_seconds", "Request latency")

# Use metrics
counter.inc()
gauge.set(25.0)
histogram.observe(0.5)

# Export
print(registry.export_prometheus())

# HTTP server
with MetricsExporter(registry, port=9090):
    # Metrics available at http://localhost:9090/metrics
    train()
```

## CLI

```bash
# Training
flux train --config config.yaml --prompts prompts.json

# Testing
flux test --config config.yaml

# Generation
flux generate --model path/to/model --prompt "Hello"

# System info
flux info
```

## Type Annotations

```python
from flux.core.types import (
    PromptsType,      # list[str] | list[dict]
    MetricsDict,      # dict[str, float]
    StateDict,        # dict[str, torch.Tensor]
    CallbackType,     # Callable[[CoordinatorState, MetricsDict], None]
    StepCallbackType, # Callable[[StepResult], None]
)
```

## Exceptions

```python
from flux.utils import CircuitBreakerOpen

try:
    result = call_service()
except CircuitBreakerOpen:
    # Service is unavailable
    use_fallback()
```

## Constants

```python
from flux.core.constants import (
    DEFAULT_BATCH_SIZE,      # 32
    DEFAULT_LEARNING_RATE,   # 1e-6
    DEFAULT_NUM_STEPS,       # 1000
    MAX_SEQUENCE_LENGTH,     # 4096
)
```
