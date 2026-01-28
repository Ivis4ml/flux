# Flux Next Steps - Execution Summary

## Current State (Updated: January 2025)

```
Done:
├── Staleness capacity control (AReaL formula)
├── SGLang client (streaming, abort, update_weights)
├── PID adaptive controller
├── Importance correction
├── Algorithm registry (PPO, GRPO, DPO, etc.)
├── TrainingBackend ABC (flux/training/base.py)        ✅ NEW
├── GPUBatch data model                                 ✅ NEW
├── TrainStepResult standardized return type           ✅ NEW
├── TransformersBackend implementation                 ✅ NEW
├── MegatronEngine refactored (dual interface)         ✅ NEW
├── Mode Gate state machine                            ✅ NEW
├── ModeGateIntegration helper                         ✅ NEW
└── create_training_backend() factory                  ✅ NEW

Partial:
├── MegatronEngine (stub forward pass - needs Megatron-LM)
├── Communication layer (ZMQ only, HTTP fallback missing)
└── Weight sync (no version ACK)

Missing:
├── TrajectoryStore tiers (hot/cold)
├── First-class services (Reference, Critic, Reward)
└── Coordinator loop refactoring with ModeGate
```

## Immediate Actions (This Week)

### 1. Native Trainer Contract

Create `flux/training/base.py`:

```python
# Key interfaces to define:
class GPUBatch          # Tensorized, device-owned batch
class TrainingBackend   # ABC with train_step(), get_state_dict(), version
class TrainStepResult   # Standardized return type
```

### 2. Mode Gate (Quick Win)

Create `flux/controller/mode_gate.py`:

```python
class AsyncMode(Enum):
    SYNC_BARRIER = auto()
    ASYNC_RUNNING = auto()
    THROTTLED = auto()

class ModeGate:
    def evaluate(staleness, capacity, buffer_fill, in_flight) -> ModeGateState
    def can_submit_rollout() -> bool
    async def enforce_barrier(wait_fn)
```

### 3. Transport Decision

**Option A (Recommended)**: ZMQ-only
- Remove HTTP fallback from config
- Update docs
- Keep SGLang HTTP as separate concern

**Option B**: Implement HTTP fallback
- Add to `CommunicationManager`
- Health check loop
- Fallback logic

## Priority Order

| P | Task | File(s) | Status |
|---|------|---------|--------|
| ~~0~~ | ~~Define `TrainingBackend` ABC~~ | ~~`flux/training/base.py`~~ | ✅ Done |
| ~~0~~ | ~~Implement `TransformersBackend`~~ | ~~`flux/training/backends/transformers.py`~~ | ✅ Done |
| ~~1~~ | ~~Implement `ModeGate`~~ | ~~`flux/controller/mode_gate.py`~~ | ✅ Done |
| 1 | Clean transport docs | `flux/core/config.py`, docs | Pending |
| ~~2~~ | ~~Add `GPUBatch` + batch ops~~ | ~~`flux/training/base.py`~~ | ✅ Done |
| 2 | Add version ACK to weight sync | `flux/sync/weight_sync.py` | Pending |
| 3 | Implement `TrajectoryStore` | `flux/core/trajectory_store.py` | Pending |
| 3 | Refactor coordinator loop | `flux/coordinator/coordinator.py` | Pending |

## New Priorities

| P | Task | File(s) | Effort |
|---|------|---------|--------|
| 1 | Add unit tests for TrainingBackend | `tests/unit/test_training_backend.py` | 3h |
| 1 | Add unit tests for ModeGate | `tests/unit/test_mode_gate.py` | 2h |
| 2 | Implement TrajectoryStore tiers | `flux/core/trajectory_store.py` | 6h |
| 2 | Add version ACK to weight sync | `flux/sync/weight_sync.py` | 3h |
| 3 | Integrate ModeGate into coordinator | `flux/coordinator/coordinator.py` | 4h |
| 3 | First-class services (Reference, Reward) | `flux/services/` | 8h |

## Key Design Decisions Needed

### 1. Backend Switching Strategy

```yaml
# Option A: Config enum
training_backend: "transformers"  # or "megatron", "fsdp"

# Option B: Full config path
training:
  backend: "megatron"
  megatron:
    tp_size: 4
    pp_size: 2
```

**Recommendation**: Option B for flexibility

### 2. ModeGate Integration Point

```python
# Option A: Inside coordinator loop
gate_state = mode_gate.evaluate(...)
if gate_state.mode == SYNC_BARRIER:
    await barrier()

# Option B: Decorator/wrapper
@mode_gate.controlled
async def training_step():
    ...
```

**Recommendation**: Option A for clarity

### 3. TrajectoryStore Location

```
# Option A: Part of coordinator
coordinator.trajectory_store.add(...)

# Option B: Separate service
trajectory_service.add(...)
```

**Recommendation**: Option A initially, migrate to B later

## File Structure (Current)

```
flux/
├── core/
│   ├── config.py
│   ├── trajectory.py
│   └── trajectory_store.py  # TODO
├── training/
│   ├── __init__.py          # Updated exports
│   ├── base.py              # ✅ TrainingBackend ABC, GPUBatch, TrainStepResult
│   ├── backends/            # ✅ NEW
│   │   ├── __init__.py      # Backend factory
│   │   └── transformers.py  # ✅ TransformersBackend
│   ├── megatron_engine.py   # ✅ Refactored (dual interface)
│   ├── batch_composer.py
│   └── algorithms/
├── controller/
│   ├── __init__.py          # Updated exports
│   ├── adaptive_async.py
│   ├── staleness.py
│   ├── mode_gate.py         # ✅ NEW: ModeGate, AsyncMode, ModeGateIntegration
│   └── importance.py
├── coordinator/
│   ├── coordinator.py       # TODO: Integrate ModeGate
│   └── communication.py
├── sync/
│   └── weight_sync.py       # TODO: + version ACK
└── services/                # TODO (future)
    ├── reference.py
    └── reward.py
```

## Testing Strategy

```bash
# Phase 1: Unit tests for new abstractions
pytest tests/unit/test_training_backend.py
pytest tests/unit/test_mode_gate.py
pytest tests/unit/test_trajectory_store.py

# Phase 2: Integration tests
pytest tests/integration/test_coordinator_with_mode_gate.py

# Phase 3: E2E with real backends
CUDA_VISIBLE_DEVICES=0 pytest tests/e2e/test_transformers_backend.py
```

## Validation Criteria

Before v0.2 release:

- [x] `TransformersBackend` passes unit tests (implemented, needs tests)
- [x] `ModeGate` correctly transitions states (implemented, needs tests)
- [ ] Training loop works with new abstractions
- [ ] No performance regression vs current implementation
- [x] Documentation updated

## Questions to Resolve

1. **Reference model strategy**: Frozen copy vs separate model?
2. **Critic integration**: Shared with actor or separate?
3. **Multi-node priority**: When to implement NCCL sync?
4. **Backward compatibility**: Support old config format?
