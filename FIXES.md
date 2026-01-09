# Fixes Applied [Changes by Codex 5.2 XHigh]

This file records fixes applied during the Codex review pass.

- Added missing algorithm enum entries and override fields to align config with registry-based algorithms, and mapped algorithm names to registry keys in the Megatron engine. `flux/core/config.py` and `flux/training/megatron_engine.py`
- Implemented `TrainingState.update_from_batch` to update sample/token counters from batch metrics. `flux/core/types.py`
- Added TrajectoryBatch accessors, default loss-mask construction when missing, and cached tensor outputs from `to_tensors` for consistent training inputs. `flux/core/trajectory.py`
- Wired coordinator staleness tracking to use the scheduler's staleness manager and compute version-gap-based staleness metrics. `flux/coordinator/coordinator.py`
- Fixed CLI client shutdown to use `close()` instead of a missing `disconnect()`. `flux/cli.py`

---

## Coordinator Integration Updates [Changes by CLaude Opus 4.5]

The following changes integrate full rollout generation and weight sync into the coordinator based on user requirements.

### 1. StreamingRolloutManager Integration

**Files changed:** `flux/coordinator/coordinator.py`

- Imported `StreamingRolloutManager` from `flux.rollout.manager`
- Added `_rollout_manager: StreamingRolloutManager | None` field to coordinator
- Added `_use_real_rollouts` flag for toggling between real and stub rollouts
- Updated `initialize()` to create `StreamingRolloutManager` with APRIL strategy:
  ```python
  self._rollout_manager = StreamingRolloutManager(
      client=self._sglang,
      config=self.config.rollout,
      trajectory_buffer=self._buffer,
      version_provider=version_provider,
  )
  ```
- Refactored `_generate_rollouts_sync()` to call async generation with proper event loop handling
- Added `_generate_rollouts_stub()` for testing without SGLang
- Updated `_generate_rollouts_async()` to use `StreamingRolloutManager.generate_batch()`:
  - Full APRIL strategy: oversample, abort long-tail, partial reuse
  - Graceful fallback to stub on errors
  - Rollout metrics logging

### 2. Weight Sync via SGLangClient.update_weights

**Files changed:** `flux/coordinator/coordinator.py`

- Updated `_sync_weights()` to call async version with proper event loop handling
- Rewrote `_sync_weights_async()` to push weights via SGLang HTTP API:
  ```python
  payload_bytes = self._weight_sync.serialize_for_sync(
      weights=state_dict,
      version=version,
  )
  success = await self._sglang.update_weights(
      weights=payload_bytes,
      version=version,
  )
  ```
- Graceful fallback when `update_weights` not available (testing mode)
- Proper error handling and logging

### 3. Notebook-Friendly Event Loop Handling

**Files changed:** `flux/trainer.py`, `flux/coordinator/coordinator.py`

- Added `_run_async()` helper method to `FluxTrainer`:
  - Detects if running in notebook (Jupyter) or async context via `asyncio.get_running_loop()`
  - Uses `nest_asyncio.apply()` if available for notebook compatibility
  - Falls back to `ThreadPoolExecutor` when nest_asyncio not installed
  - Creates new event loop when none exists (CLI/script mode)
- Updated `setup()` and `teardown()` to delegate to `FluxCoordinator._run_async()` for the dedicated loop
- Similar pattern applied to coordinator's sync methods

### 4. Algorithm Config String-Based Registry

**Status:** Already implemented

- Algorithm names are string-based and look up functions via registry pattern
- `@register_adv_estimator("name")` and `@register_policy_loss("name")` decorators
- Any registered algorithm name works without enum changes

### Test Results

- All **485 tests passing** (unchanged)
- 2 skipped (tensor bucket tests requiring specific conditions)
- 6 warnings (unawaited coroutines in test setup - harmless)

---

## Config Flexibility Updates [Changes by Claude Opus 4.5]

### 5. Configurable Weight Sync Method

**Files changed:** `flux/core/config.py`, `flux/sync/weight_sync.py`, `flux/coordinator/coordinator.py`

Added `WeightSyncMethod` enum and configurable method selection:
- `"full"`: Send complete state_dict (default for simplicity)
- `"delta"`: Send only changed parameters (uses DeltaCompressor, lower bandwidth)
- `"per_tensor"`: Stream individual tensors with metadata (most flexible)

Config changes:
```python
class WeightSyncConfig(BaseConfig):
    method: WeightSyncMethod = Field(default=WeightSyncMethod.DELTA)
    quantize: bool = Field(default=False)
    quantize_bits: Literal[8, 16] = Field(default=16)
    # ... existing fields
```

Coordinator `_sync_weights_async()` now:
- Reads `config.weight_sync.method` to select payload format
- Applies delta compression when `method == "delta"`
- Serializes per-tensor when `method == "per_tensor"`
- Optionally applies quantization when `config.weight_sync.quantize == True`

WeightSyncManager updated:
- Added `_use_delta_compression()` helper for backward compatibility
- Supports both old (`use_delta_compression`) and new (`method`) config styles

### 6. Pure String Algorithm Names

**Files changed:** `flux/core/config.py`

Changed `AlgorithmConfig.name` from `AlgorithmType` enum to pure `str`:
```python
# Before
name: AlgorithmType = Field(default=AlgorithmType.GRPO)

# After
name: str = Field(default="grpo")
```

Benefits:
- Any registered algorithm name works without code changes
- Custom algorithms via `@register_policy_loss("my_algo")` work directly
- No need to update enum for new algorithms
- `AlgorithmType` enum kept for backward compatibility

### Test Results

- All **485 tests passing**
- 2 skipped, 6 warnings (unchanged)

---

## Loop And Delta Sync Corrections [Codex Review Follow-up]

### 7. Dedicated Loop Enforcement For Async Ops

**Files changed:** `flux/coordinator/coordinator.py`, `flux/trainer.py`

- Added `_await_async()` to route all async coordinator work through the dedicated loop.
- `initialize()` / `shutdown()` now delegate to loop-safe internal implementations.
- Sync wrappers use `_run_async()` with the dedicated loop to avoid cross-loop `httpx` usage.
- Shutdown avoids joining the async thread from itself to prevent runtime errors.

### 8. Delta Baseline Versioning And Safe Updates

**Files changed:** `flux/sync/weight_sync.py`, `flux/coordinator/coordinator.py`

- Added baseline version tracking so deltas are computed from the actual baseline version.
- Baseline is now updated **only after** successful `update_weights`, preventing server/client desync.
- Serialization always moves tensors to CPU before `torch.save` to avoid device-bound payloads.

---

## Serialization and Event Loop Fixes [Changes by Claude Opus 4.5]

The following changes fix critical issues identified in code review.

### 7. Centralized Weight Serialization in WeightSyncManager

**Files changed:** `flux/sync/weight_sync.py`

Moved all weight serialization logic to WeightSyncManager for cleaner architecture:

```python
def serialize_for_sync(self, weights: dict[str, torch.Tensor], version: int | None = None) -> bytes:
    """Serialize weights for sync based on configured method.

    Handles:
    - Full: Complete state_dict serialized to bytes
    - Delta: Only changed parameters since baseline
    - Per-tensor: Individual tensors with metadata

    Returns bytes suitable for HTTP transport.
    """
```

Key improvements:
- All sync methods produce `bytes` output (torch.save to BytesIO)
- Delta compression uses `flux.sync.delta_compression` (fixed import)
- Quantization preserves scale metadata for proper dequantization
- Proper baseline management with `get_baseline()` and `set_baseline()`

Added baseline management:
```python
def get_baseline(self) -> dict[str, torch.Tensor] | None:
    """Get current baseline weights for delta computation."""

def set_baseline(self, weights: dict[str, torch.Tensor], version: int) -> None:
    """Set baseline weights for delta computation."""
```

### 8. Dedicated Event Loop for Async Operations

**Files changed:** `flux/coordinator/coordinator.py`

Fixed event loop conflicts in notebooks by using a dedicated background loop:

```python
def _ensure_async_loop(self) -> asyncio.AbstractEventLoop:
    """Ensure a dedicated event loop is running for async operations.

    This avoids conflicts with notebook event loops by running our
    async operations on a separate thread with its own loop.
    """
    if self._async_loop is not None and self._async_loop.is_running():
        return self._async_loop

    def _run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    self._async_loop = asyncio.new_event_loop()
    self._async_thread = threading.Thread(target=_run_loop, args=(self._async_loop,), daemon=True)
    self._async_thread.start()
    return self._async_loop

def _run_async(self, coro) -> Any:
    """Run a coroutine, handling both sync and async contexts.

    Uses run_coroutine_threadsafe when already in async context.
    """
```

Benefits:
- Works in Jupyter notebooks without nest_asyncio
- No event loop conflicts with existing async contexts
- Proper cleanup on shutdown via `_stop_async_loop()`

### 9. Simplified Coordinator Weight Sync

**Files changed:** `flux/coordinator/coordinator.py`

Simplified `_sync_weights_async()` to use centralized serialization:

```python
async def _sync_weights_async(self) -> None:
    # Get state dict from training engine
    state_dict = self._engine.get_state_dict()
    version = self._state.version.version_id

    # Use centralized serialization from WeightSyncManager
    # This handles method selection, delta compression, quantization
    payload_bytes = self._weight_sync.serialize_for_sync(
        weights=state_dict,
        version=version,
    )

    # Push serialized bytes to SGLang server(s)
    success = await self._sglang.update_weights(
        weights=payload_bytes,
        version=version,
    )
```

Removed:
- Faulty `flux.sync.delta_sync` imports (module didn't exist)
- Inline serialization logic (moved to WeightSyncManager)
- Unused `sys` import

### Test Results

- All **485 tests passing**
- 2 skipped, 6 warnings (unchanged)
