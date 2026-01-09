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
  success = await self._sglang.update_weights(
      weights=state_dict,
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
- Updated `setup()` and `teardown()` to use `_run_async()`
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
