# Flux Architecture Evolution Plan

**Goal**: Transform Flux into a general, native-trainer, GPU-direct, config-driven RL framework.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FluxTrainer (User API)                          │
│   config-driven • callbacks • CLI                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FluxCoordinator (Stable Core)                       │
│   orchestration only • no direct execution • swappable backends via config   │
└─────────────────────────────────────────────────────────────────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐
│   Mode Gate     │    │  First-Class    │    │   Trajectory Store          │
│  (sync/async    │    │   Services      │    │   ├─ Hot Buffer (GPU)       │
│   state machine)│    │   ├─ Reference  │    │   └─ Cold Store (CPU/disk)  │
│                 │    │   ├─ Critic     │    │       with version gating   │
│                 │    │   └─ Reward     │    │                             │
└─────────────────┘    └─────────────────┘    └─────────────────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Backend Adapter Layer                              │
│  ┌──────────────────────┐              ┌──────────────────────┐             │
│  │  Training Backends   │              │  Rollout Backends    │             │
│  │  ├─ MegatronAdapter  │              │  ├─ SGLangAdapter    │             │
│  │  ├─ FSDPAdapter      │              │  ├─ vLLMAdapter      │             │
│  │  └─ TransformersAdpt │              │  └─ (future)         │             │
│  └──────────────────────┘              └──────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Transport Layer                                  │
│  ┌──────────────────────┐              ┌──────────────────────┐             │
│  │  RPC Control         │              │  Weight/Event Bus    │             │
│  │  (batch requests,    │              │  (broadcast updates, │             │
│  │   health checks)     │              │   version ACK)       │             │
│  │  ZMQ ROUTER/DEALER   │              │  ZMQ PUB/SUB or NCCL │             │
│  └──────────────────────┘              └──────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Native Trainer Contract + Config Model

### 1.1 Trainer API Specification

```python
# flux/training/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol
import torch

@dataclass(frozen=True)
class GPUBatch:
    """Tensorized batch - all tensors already on target device."""
    input_ids: torch.Tensor          # [B, seq]
    attention_mask: torch.Tensor     # [B, seq]
    behavior_log_probs: torch.Tensor # [B, seq]
    rewards: torch.Tensor            # [B] or [B, seq]
    version_gaps: torch.Tensor       # [B]

    # Optional
    token_rewards: torch.Tensor | None = None
    ref_log_probs: torch.Tensor | None = None


@dataclass
class TrainStepResult:
    """Result from a single training step."""
    loss: float
    version: int
    metrics: dict[str, float]
    grad_norm: float = 0.0


class TrainingBackend(ABC):
    """
    Native trainer contract.

    GPU-direct assumptions:
    - All tensors in GPUBatch are already on the correct device
    - Backend owns device allocation (no host copies in hot path)
    - Async-safe: train_step() can be called from asyncio event loop
    """

    @abstractmethod
    def initialize(self, config: 'TrainingConfig') -> None:
        """Initialize backend with config. Called once."""
        pass

    @abstractmethod
    def train_step(self, batch: GPUBatch) -> TrainStepResult:
        """
        Execute one training step.

        Contract:
        - batch tensors are on self.device
        - returns before any async gradient sync completes (if applicable)
        - increments internal version counter
        """
        pass

    @abstractmethod
    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Get model weights for sync. Returns CPU tensors."""
        pass

    @abstractmethod
    def set_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load model weights."""
        pass

    @property
    @abstractmethod
    def version(self) -> int:
        """Current policy version (incremented after each train_step)."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device this backend operates on."""
        pass
```

### 1.2 Config-First Backend Selection

```python
# flux/core/config.py (additions)

class TrainingBackendType(str, Enum):
    MEGATRON = "megatron"
    FSDP = "fsdp"
    TRANSFORMERS = "transformers"

class RolloutBackendType(str, Enum):
    SGLANG = "sglang"
    VLLM = "vllm"

class FluxConfig(BaseModel):
    # Backend selection - no code changes needed
    training_backend: TrainingBackendType = TrainingBackendType.TRANSFORMERS
    rollout_backend: RolloutBackendType = RolloutBackendType.SGLANG

    # Backend-specific configs (only relevant one is used)
    megatron: MegatronConfig = MegatronConfig()
    fsdp: FSDPConfig = FSDPConfig()
    sglang: SGLangConfig = SGLangConfig()

    # ... rest of config
```

### 1.3 Backend Factory

```python
# flux/training/backends/__init__.py

def create_training_backend(config: FluxConfig) -> TrainingBackend:
    """Factory function - backend selected by config."""
    match config.training_backend:
        case TrainingBackendType.MEGATRON:
            from .megatron import MegatronBackend
            return MegatronBackend(config.megatron)
        case TrainingBackendType.FSDP:
            from .fsdp import FSDPBackend
            return FSDPBackend(config.fsdp)
        case TrainingBackendType.TRANSFORMERS:
            from .transformers import TransformersBackend
            return TransformersBackend(config.model_path)
```

---

## Phase 2: Core Architecture Refinements

### 2.1 Mode Gate (Sync/Async State Machine)

```python
# flux/controller/mode_gate.py

from enum import Enum, auto
from dataclasses import dataclass
import asyncio

class AsyncMode(Enum):
    """Async training mode states."""
    SYNC_BARRIER = auto()    # Waiting for all in-flight to complete
    ASYNC_RUNNING = auto()   # Normal async operation
    THROTTLED = auto()       # Capacity exhausted, backpressure active


@dataclass
class ModeGateState:
    mode: AsyncMode
    reason: str
    staleness: float
    capacity: int
    in_flight: int


class ModeGate:
    """
    State machine controlling sync/async transitions.

    Integrates:
    - Staleness thresholds (from PID controller)
    - Capacity limits (from StalenessManager)
    - Backpressure signals (buffer watermark, GPU util)
    """

    def __init__(
        self,
        staleness_threshold: float = 0.3,
        capacity_low_watermark: int = 0,
        buffer_high_watermark: float = 0.9,
    ):
        self._staleness_threshold = staleness_threshold
        self._capacity_low_watermark = capacity_low_watermark
        self._buffer_high_watermark = buffer_high_watermark

        self._current_mode = AsyncMode.ASYNC_RUNNING
        self._barrier_event: asyncio.Event | None = None
        self._in_flight_count = 0

    def evaluate(
        self,
        staleness: float,
        capacity: int,
        buffer_fill_ratio: float,
        in_flight: int,
    ) -> ModeGateState:
        """
        Evaluate current state and determine mode.

        Called after each training step or rollout completion.
        """
        self._in_flight_count = in_flight

        # Priority 1: Capacity exhausted
        if capacity <= self._capacity_low_watermark:
            self._current_mode = AsyncMode.THROTTLED
            return ModeGateState(
                mode=AsyncMode.THROTTLED,
                reason="capacity_exhausted",
                staleness=staleness,
                capacity=capacity,
                in_flight=in_flight,
            )

        # Priority 2: Staleness too high
        if staleness > self._staleness_threshold:
            self._current_mode = AsyncMode.SYNC_BARRIER
            return ModeGateState(
                mode=AsyncMode.SYNC_BARRIER,
                reason=f"staleness={staleness:.3f} > {self._staleness_threshold}",
                staleness=staleness,
                capacity=capacity,
                in_flight=in_flight,
            )

        # Priority 3: Buffer approaching full
        if buffer_fill_ratio > self._buffer_high_watermark:
            self._current_mode = AsyncMode.THROTTLED
            return ModeGateState(
                mode=AsyncMode.THROTTLED,
                reason=f"buffer_fill={buffer_fill_ratio:.2f}",
                staleness=staleness,
                capacity=capacity,
                in_flight=in_flight,
            )

        # Default: Continue async
        self._current_mode = AsyncMode.ASYNC_RUNNING
        return ModeGateState(
            mode=AsyncMode.ASYNC_RUNNING,
            reason="normal",
            staleness=staleness,
            capacity=capacity,
            in_flight=in_flight,
        )

    async def enforce_barrier(self, wait_for_in_flight: callable) -> None:
        """
        Block until all in-flight rollouts complete.

        Args:
            wait_for_in_flight: Async function that waits for in-flight to drain
        """
        if self._current_mode != AsyncMode.SYNC_BARRIER:
            return

        await wait_for_in_flight()
        self._current_mode = AsyncMode.ASYNC_RUNNING

    def can_submit_rollout(self) -> bool:
        """Check if new rollouts can be submitted."""
        return self._current_mode == AsyncMode.ASYNC_RUNNING
```

### 2.2 First-Class Services

```python
# flux/services/base.py

class ServiceBase(ABC):
    """Base class for first-class services (Reference, Critic, Reward)."""

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        pass


# flux/services/reference.py

class ReferenceService(ServiceBase):
    """
    Reference policy service.

    Manages reference model for KL penalty computation.
    Can be:
    - Shared with actor (frozen copy)
    - Separate model (different GPU)
    - Remote service (HTTP)
    """

    def __init__(self, config: ReferenceConfig):
        self._config = config
        self._model = None
        self._frozen_version: int | None = None

    async def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reference log probs for KL penalty."""
        pass

    async def update_reference(self, state_dict: dict, version: int) -> None:
        """Update reference model (for periodic refresh)."""
        pass


# flux/services/reward.py

class RewardService(ServiceBase):
    """
    Reward computation service.

    Supports:
    - Rule-based rewards (fast, CPU)
    - Model-based rewards (GPU, batched)
    - Composite rewards (weighted combination)
    """

    async def compute_rewards(
        self,
        prompts: list[str],
        responses: list[str],
        batch_size: int = 32,
    ) -> list[float]:
        """Compute rewards for prompt-response pairs."""
        pass
```

### 2.3 Trajectory Store (Hot + Cold)

```python
# flux/core/trajectory_store.py

from collections import deque
from dataclasses import dataclass
import asyncio

@dataclass
class StoreTier:
    """Configuration for a storage tier."""
    max_size: int
    device: str  # "cuda", "cpu", "disk"
    eviction_policy: str = "fifo"  # "fifo", "lru", "version"


class TrajectoryStore:
    """
    Two-tier trajectory storage with version gating.

    Hot Buffer (GPU):
    - Recent trajectories for immediate training
    - Fast access, limited capacity
    - Evicts to cold store when full

    Cold Store (CPU/disk):
    - Older trajectories for replay
    - Large capacity, slower access
    - Version-gated: only trajectories within max_version_gap
    """

    def __init__(
        self,
        hot_config: StoreTier,
        cold_config: StoreTier,
        max_version_gap: int = 5,
    ):
        self._hot_buffer: deque = deque(maxlen=hot_config.max_size)
        self._cold_store: deque = deque(maxlen=cold_config.max_size)
        self._max_version_gap = max_version_gap
        self._current_version = 0
        self._lock = asyncio.Lock()

    async def add(self, trajectory: 'Trajectory') -> None:
        """Add trajectory to hot buffer."""
        async with self._lock:
            if len(self._hot_buffer) >= self._hot_buffer.maxlen:
                # Evict oldest to cold store
                evicted = self._hot_buffer.popleft()
                self._cold_store.append(evicted)
            self._hot_buffer.append(trajectory)

    async def sample_batch(
        self,
        batch_size: int,
        current_version: int,
        staleness_strategy: str = "stratified",
    ) -> list['Trajectory']:
        """
        Sample batch with version gating.

        Trajectories with version_gap > max_version_gap are excluded.
        """
        self._current_version = current_version

        async with self._lock:
            # Filter by version gap
            valid_hot = [
                t for t in self._hot_buffer
                if current_version - t.version.version_id <= self._max_version_gap
            ]
            valid_cold = [
                t for t in self._cold_store
                if current_version - t.version.version_id <= self._max_version_gap
            ]

            # Combine and sample
            all_valid = valid_hot + valid_cold

            if staleness_strategy == "stratified":
                return self._stratified_sample(all_valid, batch_size, current_version)
            else:
                # Random sample
                import random
                return random.sample(all_valid, min(batch_size, len(all_valid)))

    def _stratified_sample(
        self,
        trajectories: list,
        batch_size: int,
        current_version: int,
    ) -> list:
        """Sample with staleness stratification."""
        # Group by version gap
        strata: dict[int, list] = {}
        for t in trajectories:
            gap = current_version - t.version.version_id
            strata.setdefault(gap, []).append(t)

        # Sample proportionally from each stratum
        result = []
        per_stratum = max(1, batch_size // len(strata)) if strata else 0

        for gap in sorted(strata.keys()):
            sample_size = min(per_stratum, len(strata[gap]))
            result.extend(random.sample(strata[gap], sample_size))

        return result[:batch_size]

    async def gc_stale(self, current_version: int) -> int:
        """Garbage collect trajectories beyond max_version_gap."""
        async with self._lock:
            before = len(self._hot_buffer) + len(self._cold_store)

            self._hot_buffer = deque(
                (t for t in self._hot_buffer
                 if current_version - t.version.version_id <= self._max_version_gap),
                maxlen=self._hot_buffer.maxlen,
            )
            self._cold_store = deque(
                (t for t in self._cold_store
                 if current_version - t.version.version_id <= self._max_version_gap),
                maxlen=self._cold_store.maxlen,
            )

            after = len(self._hot_buffer) + len(self._cold_store)
            return before - after
```

---

## Phase 3: Data Model Upgrade

### 3.1 Enhanced TrajectoryBatch

```python
# flux/core/trajectory.py (additions)

@dataclass
class TrajectoryBatch:
    trajectories: list[Trajectory]

    # Cached tensors (lazily computed)
    _cached_tensors: dict[str, torch.Tensor] | None = None
    _device: torch.device | None = None

    @property
    def batch_size(self) -> int:
        return len(self.trajectories)

    @property
    def max_length(self) -> int:
        return max(len(t.tokens) for t in self.trajectories) if self.trajectories else 0

    # === New DataProto-like operations ===

    def pad_to(self, max_length: int, pad_value: int = 0) -> 'TrajectoryBatch':
        """Pad all sequences to max_length (returns new batch)."""
        padded = []
        for t in self.trajectories:
            if len(t.tokens) < max_length:
                padded.append(t.pad_to(max_length, pad_value))
            else:
                padded.append(t.truncate_to(max_length))
        return TrajectoryBatch(trajectories=padded)

    def concat(self, other: 'TrajectoryBatch') -> 'TrajectoryBatch':
        """Concatenate with another batch (auto-pads to max length)."""
        combined = self.trajectories + other.trajectories
        max_len = max(len(t.tokens) for t in combined)
        return TrajectoryBatch(trajectories=combined).pad_to(max_len)

    def slice(self, indices: list[int]) -> 'TrajectoryBatch':
        """Extract subset by indices."""
        return TrajectoryBatch(
            trajectories=[self.trajectories[i] for i in indices]
        )

    def to_device(self, device: torch.device) -> 'TrajectoryBatch':
        """Move cached tensors to device."""
        if self._cached_tensors is None:
            self._cached_tensors = self._build_tensors(device)
        elif self._device != device:
            self._cached_tensors = {
                k: v.to(device) for k, v in self._cached_tensors.items()
            }
        self._device = device
        return self

    def as_gpu_batch(self, device: torch.device) -> 'GPUBatch':
        """
        Convert to frozen GPU batch for training.

        This is the primary interface for TrainingBackend.train_step().
        """
        self.to_device(device)
        tensors = self._cached_tensors

        return GPUBatch(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            behavior_log_probs=tensors["behavior_log_probs"],
            rewards=tensors["rewards"],
            version_gaps=tensors["version_gaps"],
            token_rewards=tensors.get("token_rewards"),
            ref_log_probs=tensors.get("ref_log_probs"),
        )

    def _build_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Build padded tensors from trajectories."""
        max_len = self.max_length
        batch_size = self.batch_size

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float, device=device)
        behavior_log_probs = torch.zeros(batch_size, max_len, dtype=torch.float, device=device)
        rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
        version_gaps = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i, t in enumerate(self.trajectories):
            seq_len = len(t.tokens)
            input_ids[i, :seq_len] = torch.tensor(t.tokens, device=device)
            attention_mask[i, :seq_len] = 1.0
            if t.log_probs:
                behavior_log_probs[i, :len(t.log_probs)] = torch.tensor(t.log_probs, device=device)
            rewards[i] = t.reward if t.reward is not None else 0.0
            version_gaps[i] = t.version.version_id if t.version else 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "behavior_log_probs": behavior_log_probs,
            "rewards": rewards,
            "version_gaps": version_gaps,
        }
```

---

## Phase 4: Transport Layer

### 4.1 Split RPC vs Event Bus

```python
# flux/transport/base.py

class RPCChannel(ABC):
    """
    RPC control channel.

    For: batch requests, health checks, configuration.
    Pattern: Request-Reply
    """

    @abstractmethod
    async def call(self, method: str, params: dict) -> dict:
        """Make RPC call and wait for response."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass


class EventBus(ABC):
    """
    Weight/Event broadcast channel.

    For: weight updates, version notifications, shutdown signals.
    Pattern: Publish-Subscribe
    """

    @abstractmethod
    async def publish(self, topic: str, data: bytes) -> None:
        """Publish event to topic."""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: callable) -> None:
        """Subscribe to topic with handler."""
        pass


# flux/transport/zmq_impl.py

class ZMQRPCChannel(RPCChannel):
    """ZMQ ROUTER/DEALER implementation."""

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._socket = None

    async def call(self, method: str, params: dict) -> dict:
        # ROUTER/DEALER request-reply
        pass


class ZMQEventBus(EventBus):
    """ZMQ PUB/SUB implementation."""

    def __init__(self, pub_endpoint: str, sub_endpoint: str):
        self._pub_endpoint = pub_endpoint
        self._sub_endpoint = sub_endpoint

    async def publish(self, topic: str, data: bytes) -> None:
        # PUB socket
        pass

    async def subscribe(self, topic: str, handler: callable) -> None:
        # SUB socket
        pass
```

### 4.2 Decision: ZMQ-Only (Clean Up)

For now, we commit to **ZMQ-only** and update docs accordingly.

**Action Items:**
- [ ] Remove HTTP fallback references from `flux/core/config.py`
- [ ] Update `flux/coordinator/communication.py` docstrings
- [ ] Add clear documentation: "ZMQ required for coordinator communication; SGLang uses separate HTTP"

---

## Phase 5: Weight Sync Roadmap

### 5.1 Version ACK Protocol

```python
# flux/sync/weight_sync.py (additions)

@dataclass
class SyncResult:
    version: int
    success: bool
    acked_nodes: list[str]
    failed_nodes: list[str]
    elapsed_ms: float


class WeightSyncManager:
    async def sync_with_ack(
        self,
        state_dict: dict[str, torch.Tensor],
        version: int,
        timeout: float = 30.0,
    ) -> SyncResult:
        """
        Sync weights and wait for ACK from all inference servers.

        Protocol:
        1. Push weights to all servers
        2. Poll /get_weight_version on each server
        3. Wait until all servers report matching version
        4. Return result with success/failure list
        """
        start = time.time()

        # Step 1: Push weights
        push_results = await self._push_to_all(state_dict, version)

        # Step 2: Wait for ACK
        acked = []
        failed = []

        deadline = start + timeout
        while time.time() < deadline:
            for server in self._servers:
                if server in acked:
                    continue

                try:
                    server_version = await self._get_version(server)
                    if server_version >= version:
                        acked.append(server)
                except Exception:
                    pass

            if len(acked) == len(self._servers):
                break

            await asyncio.sleep(0.5)

        # Servers that didn't ACK
        failed = [s for s in self._servers if s not in acked]

        return SyncResult(
            version=version,
            success=len(failed) == 0,
            acked_nodes=acked,
            failed_nodes=failed,
            elapsed_ms=(time.time() - start) * 1000,
        )
```

### 5.2 Sync Strategy Matrix

| Deployment | Strategy | Implementation | Status |
|------------|----------|----------------|--------|
| Same node, colocated | CUDA IPC | `ColocatedWeightSync` | ✅ Done |
| Same node, separate process | SHM + mmap | `ColocatedWeightSync.SHM` | ✅ Done |
| Cross-node | HTTP + delta | `WeightSyncManager.HTTP` | ✅ Done |
| Cross-node, high perf | NCCL broadcast | `NCCLWeightSync` | 🚧 Planned |

---

## Phase 6: Simplified Training Loop

### 6.1 Slime-Style Clarity

```python
# flux/coordinator/coordinator.py (refactored)

class FluxCoordinator:
    async def run_training_loop(self):
        """
        Main training loop with explicit async injection points.

        Slime-style clarity:
        - Sequential structure
        - Explicit mode gate checks
        - Clear async boundaries
        """
        for step in range(self.config.num_steps):
            # === 1. Mode Gate Check ===
            gate_state = self.mode_gate.evaluate(
                staleness=self.staleness_manager.current_staleness,
                capacity=self.staleness_manager.get_capacity(),
                buffer_fill_ratio=self.trajectory_store.fill_ratio,
                in_flight=self.staleness_manager.stats.total_in_flight,
            )

            if gate_state.mode == AsyncMode.SYNC_BARRIER:
                await self.mode_gate.enforce_barrier(self._wait_for_in_flight)
                await self._sync_weights()

            # === 2. Submit Rollouts (if capacity allows) ===
            if self.mode_gate.can_submit_rollout():
                self._submit_rollouts(self.config.rollout_batch_size)

            # === 3. Collect Completed Rollouts ===
            trajectories = await self._collect_completed(timeout=0.1)
            if trajectories:
                await self.trajectory_store.add_batch(trajectories)

            # === 4. Sample Training Batch ===
            batch = await self.trajectory_store.sample_batch(
                batch_size=self.config.batch_size,
                current_version=self.training_backend.version,
            )

            if len(batch) < self.config.min_batch_size:
                continue  # Not enough data

            # === 5. Train Step ===
            gpu_batch = TrajectoryBatch(batch).as_gpu_batch(self.training_backend.device)
            result = self.training_backend.train_step(gpu_batch)

            # === 6. Update Staleness ===
            self.staleness_manager.compute_staleness(
                version_gap=gpu_batch.version_gaps.float().mean().item()
            )

            # === 7. PID Controller Update ===
            decision = self.adaptive_controller.update(
                self.staleness_manager.current_staleness
            )
            self.mode_gate.update_threshold(decision.async_ratio)

            # === 8. Conditional Weight Sync ===
            if self.staleness_manager.should_sync():
                sync_result = await self._sync_weights_with_ack()
                self.staleness_manager.record_sync()

            # === 9. Logging ===
            if step % self.config.log_interval == 0:
                self._log_step(step, result, gate_state)
```

---

## Implementation Checklist

### Phase 1: Native Trainer Contract
- [x] Define `TrainingBackend` ABC in `flux/training/base.py`
- [x] Define `GPUBatch` dataclass
- [x] Add `TrainingBackendType` to config
- [x] Implement `create_training_backend()` factory
- [x] Implement `TransformersBackend` (simplest)
- [x] Refactor `MegatronEngine` to implement `TrainingBackend`

### Phase 2: Core Architecture
- [x] Implement `ModeGate` state machine
- [ ] Add `ReferenceService` stub
- [ ] Add `RewardService` with existing reward functions
- [ ] Implement `TrajectoryStore` with hot/cold tiers

### Phase 3: Data Model
- [ ] Add `pad_to()`, `concat()`, `slice()` to `TrajectoryBatch`
- [ ] Add `to_device()` and `as_gpu_batch()`
- [ ] Update tests

### Phase 4: Transport
- [ ] Decide: ZMQ-only (clean docs) or implement HTTP fallback
- [ ] Split `CommunicationManager` into `RPCChannel` + `EventBus`
- [ ] Update docs to reflect transport decision

### Phase 5: Weight Sync
- [ ] Add `sync_with_ack()` method
- [ ] Add version polling to `SGLangClient`
- [ ] Document sync strategy matrix

### Phase 6: Training Loop
- [ ] Refactor `FluxCoordinator.run_training_loop()` for clarity
- [ ] Integrate `ModeGate`
- [ ] Add explicit async injection points

### Phase 7: Documentation
- [x] Update architecture diagram
- [x] Document new abstractions
- [x] Update CLAUDE.md with new patterns

---

## Migration Path

1. **v0.2**: Add `TrainingBackend` ABC, keep `MegatronEngine` as default
2. **v0.3**: Add `ModeGate`, `TrajectoryStore`
3. **v0.4**: Add `GPUBatch`, refactor data flow
4. **v0.5**: Transport cleanup, weight sync ACK
5. **v1.0**: Full general RL framework
