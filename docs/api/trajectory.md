---
title: Trajectory API
description: Data structures for rollouts and batches
---

# Trajectory API

Core data structures for handling trajectories (rollouts) in the RLHF pipeline.

## Trajectory

A single trajectory (prompt + response) with all metadata for training.

### Definition

```python
@dataclass
class Trajectory:
    # Identifiers
    id: str = ""
    group_id: str | None = None  # For multi-response sampling

    # Prompt
    prompt: str = ""
    prompt_tokens: list[int] = field(default_factory=list)
    prompt_length: int = 0

    # Response
    response: str = ""
    response_tokens: list[int] = field(default_factory=list)
    response_length: int = 0

    # Combined sequence
    tokens: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)

    # Log probabilities
    log_probs: list[float] = field(default_factory=list)
    behavior_log_probs: list[float] = field(default_factory=list)

    # Rewards
    reward: float = 0.0
    token_rewards: list[float] = field(default_factory=list)

    # Training data
    values: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    # Version tracking
    version: PolicyVersion
    version_segments: list[VersionSegment] = field(default_factory=list)

    # Status and metadata
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    generation_time_ms: float = 0.0
```

### Properties

| Property | Type | Description |
|:---------|:-----|:------------|
| `total_length` | `int` | Total tokens (prompt + response) |
| `has_version_boundaries` | `bool` | Spans multiple policy versions |
| `is_complete` | `bool` | Status is COMPLETED |

### Methods

```python
# Get staleness relative to current policy
gap = trajectory.get_version_gap(current_version)

# Convert to dict for batching
data = trajectory.to_dict()

# Create from dict
trajectory = Trajectory.from_dict(data)
```

## TrajectoryBatch

Batched trajectories with padding for training.

### Usage

```python
batch = TrajectoryBatch(trajectories=list_of_trajectories)

# Properties
batch.batch_size      # Number of trajectories
batch.max_length      # Maximum sequence length
batch.num_tokens      # Total tokens
batch.rewards         # List of scalar rewards

# Convert to tensors
tensors = batch.to_tensors(device="cuda", pad_token_id=0)
# Returns: {
#   "input_ids": Tensor,
#   "attention_mask": Tensor,
#   "loss_mask": Tensor,
#   "log_probs": Tensor,
#   "behavior_log_probs": Tensor,
#   "rewards": Tensor,
#   "advantages": Tensor,
#   "returns": Tensor,
#   "versions": Tensor,
# }

# Statistics
padding_ratio = batch.compute_padding_ratio()
version_stats = batch.get_version_stats()
```

## TrajectoryBuffer

Buffer for storing trajectories with staleness-aware management.

### Usage

```python
buffer = TrajectoryBuffer(
    max_size=10000,
    max_staleness=5,
)

# Add trajectories
buffer.add(trajectory)
buffer.add_batch(trajectories)

# Get available (within staleness limit)
available = buffer.get_available(current_version)

# Stratified sampling by staleness
sample = buffer.sample(n=32, current_version=version, stratified=True)

# Remove stale trajectories
removed_count = buffer.remove_stale(current_version)

# Statistics
stats = buffer.get_stats()
# {"size": 100, "mean_staleness": 1.5, "max_staleness": 3}
```

## PartialTrajectory

For APRIL strategy - aborted generations that can be continued.

```python
class PartialTrajectory(Trajectory):
    continuation_prompt: str = ""
    start_rollout_id: str | None = None
    continuation_count: int = 0
    kv_cache_ref: Any = None

    def can_continue(self, max_continuations: int = 3) -> bool: ...
    def prepare_continuation(self) -> str: ...
```

## PolicyVersion

Tracks policy versions for staleness computation.

```python
@dataclass
class PolicyVersion:
    version_id: int = 0
    timestamp: float = 0.0
    step: int = 0
```

## TrajectoryStatus

```python
class TrajectoryStatus(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"
```

## See Also

- [Batch Composition](../concepts/batch-composition.md)
- [Staleness](../concepts/staleness.md)
- [APRIL Strategy](../concepts/april.md)
