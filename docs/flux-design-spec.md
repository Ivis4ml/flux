# Flux: An Adaptive Post-Training Framework for Large Language Models

> **"The best of all worlds"** — Synchronous stability + Asynchronous efficiency + Native simplicity

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Motivation](#motivation)
3. [Design Goals & Non-Goals](#design-goals--non-goals)
4. [Key Insights](#key-insights)
5. [Architecture Overview](#architecture-overview)
6. [Core Components](#core-components)
7. [API Design](#api-design)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Evaluation Plan](#evaluation-plan)
10. [Related Work](#related-work)

---

## Executive Summary

**Flux** is a next-generation reinforcement learning post-training framework for Large Language Models. Unlike existing frameworks that force binary choices (synchronous vs. asynchronous, colocated vs. separated), Flux treats these as **continuous spectrums** that can be dynamically optimized during training.

### Key Innovations

| Innovation | Description | Benefit |
|------------|-------------|---------|
| **Adaptive Async** | Dynamically adjusts sync/async ratio based on staleness | ~85% GPU utilization with sync-level stability |
| **Native-First Design** | Direct Megatron + SGLang integration, no Ray overhead | Maximum performance, minimal abstraction |
| **Unified Importance Correction** | Handles staleness, trajectory inconsistency, and replay | Principled off-policy correction |
| **Speculative Sync** | Predicts long-tail rollouts, starts training early | Reduces bubble time by ~40% |
| **Multi-Dimensional Adaptation** | Temperature, batch composition, compute ratio all adaptive | Every hyperparameter becomes a learnable schedule |

### Performance Targets

```
                          VERL        AReaL       Flux (Target)
GPU Utilization           ~45%        ~95%        ~85%
Training Stability        ★★★★★       ★★★☆☆       ★★★★☆
Code Complexity           Medium      High        Low
Throughput (samples/hr)   1.0x        1.8x        2.0x
```

---

## Motivation

### The Current Landscape

Three major RLHF frameworks have emerged, each with distinct trade-offs:

| Framework | Organization | Key Design | Strength | Weakness |
|-----------|--------------|------------|----------|----------|
| **VERL** | ByteDance + HKU | Synchronous, Ray-based | Stable training | GPU bubbles |
| **AReaL** | Ant Group + Tsinghua | Fully asynchronous | High throughput | Staleness issues |
| **Slime** | THUDM (Zhipu) | SGLang-native | Simple, performant | Less flexible dataflow |

### The Problem: False Dichotomies

Existing frameworks present **binary choices** that are actually **continuous spectrums**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Binary Choices → Continuous Spectrums                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Sync vs Async            →  Adaptive async ratio (0% to 100%)          │
│  2. Colocate vs Separate     →  Dynamic GPU placement                       │
│  3. On-policy vs Off-policy  →  Importance-weighted mixture                 │
│  4. Fixed temperature        →  Per-prompt, per-stage adaptive T           │
│  5. Fixed batch composition  →  Curriculum-aware sampling                   │
│  6. Fixed compute ratio      →  Elastic gen/train allocation               │
│                                                                             │
│  Key Insight: The optimal configuration CHANGES during training             │
│               There is no single best static configuration                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Not Ray?

After analyzing VERL, AReaL, and Slime, we conclude that **Ray adds unnecessary overhead** for LLM training:

| Ray Provides | LLM Training Actually Needs | Mismatch |
|--------------|----------------------------|----------|
| Task-level scheduling | NCCL collective communication | Ray doesn't understand NCCL topology |
| Actor lifecycle management | Fine-grained GPU memory control | Ray treats GPU as a "resource count" |
| Flexible placement | Fixed TP/PP/DP configuration | Changing parallelism requires restart |
| Object Store for data transfer | CUDA IPC / NCCL for weights | Ray serialization is slow for large tensors |

**Evidence**: Both VERL and Slime end up bypassing Ray's abstractions for critical paths (weight sync, gradient communication). If you're bypassing the framework, why use it?

---

## Design Goals & Non-Goals

### Goals

1. **Adaptive by Default**: Every "hyperparameter" that could benefit from adaptation should be adaptive
2. **Native Performance**: Use Megatron and SGLang directly, no wrapper overhead
3. **Simple Codebase**: < 5000 lines of core framework code (excluding backends)
4. **Production Ready**: Support 64+ GPU training with fault tolerance
5. **Research Friendly**: Easy to experiment with new algorithms

### Non-Goals

1. **General-purpose distributed computing**: We focus on LLM post-training only
2. **Support for all training backends**: We choose Megatron + SGLang, not PyTorch DDP
3. **Backward compatibility with Ray APIs**: Clean break from Ray paradigm
4. **Real-time serving**: Training framework only, use SGLang directly for serving

---

## Key Insights

### Insight 1: Staleness is Measurable and Controllable

```python
# Staleness can be quantified in real-time
def measure_staleness(batch):
    """
    Staleness = how much has the policy changed since data was generated?
    
    Metrics:
    1. KL divergence: D_KL(π_current || π_behavior)
    2. Importance weight variance: Var(π_current(a|s) / π_behavior(a|s))
    3. Version gap: current_version - trajectory_version
    """
    kl = compute_kl_divergence(current_policy, batch.behavior_policy)
    iw_var = compute_importance_weight_variance(batch)
    version_gap = current_version - batch.mean_version
    
    # Weighted combination (can be learned)
    staleness = 0.5 * kl + 0.3 * iw_var + 0.2 * normalize(version_gap)
    return staleness
```

**Implication**: Instead of choosing sync (staleness=0) or async (staleness=high), we can **target a specific staleness level** and adapt our async ratio to maintain it.

### Insight 2: Long-Tail Rollouts are Predictable

```python
# Output length is correlated with prompt characteristics
class LengthPredictor:
    """
    Features predictive of output length:
    1. Prompt length
    2. Task type (math → long CoT, chat → short)
    3. Historical length for similar prompts
    4. Prompt complexity score
    """
    
    def predict(self, prompt):
        features = self.extract_features(prompt)
        predicted_length = self.model(features)
        confidence = self.compute_confidence(features)
        return predicted_length, confidence
```

**Implication**: We can **schedule short prompts first**, and **speculatively start training** before all rollouts complete.

### Insight 3: Training Phases Have Different Needs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Training Phase → Optimal Configuration                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Early Training (0-20% steps)                                      │
│  ├─ High exploration needed → High temperature, high gen ratio              │
│  ├─ Policy changing rapidly → More sync, less tolerance for staleness       │
│  └─ Diverse data needed → Random batch composition                          │
│                                                                             │
│  Phase 2: Mid Training (20-70% steps)                                       │
│  ├─ Balanced explore/exploit → Medium temperature                           │
│  ├─ Policy stabilizing → Can tolerate more async                            │
│  └─ Curriculum emerging → Difficulty-aware batching                         │
│                                                                             │
│  Phase 3: Late Training (70-100% steps)                                     │
│  ├─ Exploitation focus → Low temperature                                    │
│  ├─ Fine-tuning → More training, less generation                            │
│  └─ Hard examples focus → Priority sampling of failures                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Flux Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Layer 3: Adaptive Control Plane                    │  │
│  │                                                                       │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │  │
│  │  │   Adaptive   │ │   Elastic    │ │   Smart      │ │  Speculative │ │  │
│  │  │   Async      │ │   Compute    │ │   Batch      │ │  Prefetch    │ │  │
│  │  │   Controller │ │   Scheduler  │ │   Composer   │ │  Manager     │ │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                            Metrics & Decisions                              │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │                  Layer 2: Lightweight Coordinator                     │  │
│  │                                                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │                      FluxCoordinator                              │ │  │
│  │  │                                                                   │ │  │
│  │  │  - Async event loop (asyncio)                                    │ │  │
│  │  │  - ZeroMQ/gRPC for communication                                 │ │  │
│  │  │  - Metrics collection & aggregation                              │ │  │
│  │  │  - Weight sync orchestration                                     │ │  │
│  │  │  - Checkpoint management                                         │ │  │
│  │  │                                                                   │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                           Control Commands                                  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │                   Layer 1: Native Execution Engines                   │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────┐    ┌─────────────────────────┐          │  │
│  │  │   Megatron-LM Engine    │    │    SGLang Engine        │          │  │
│  │  │                         │    │                         │          │  │
│  │  │  - 3D Parallelism       │    │  - Server-based         │          │  │
│  │  │  - FSDP / Megatron-Core │    │  - Continuous batching  │          │  │
│  │  │  - Mixed precision      │    │  - FP8 / DeepEP         │          │  │
│  │  │  - Gradient checkpointing│   │  - Speculative decode   │          │  │
│  │  │                         │    │                         │          │  │
│  │  └────────────┬────────────┘    └────────────┬────────────┘          │  │
│  │               │                              │                        │  │
│  │               └──────────┬───────────────────┘                        │  │
│  │                          │                                            │  │
│  │                 ┌────────▼────────┐                                   │  │
│  │                 │  CUDA IPC Sync  │                                   │  │
│  │                 │  (Zero-Copy)    │                                   │  │
│  │                 └─────────────────┘                                   │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Training Step Flow                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Coordinator requests rollouts from SGLang                               │
│     │                                                                       │
│     ▼                                                                       │
│  2. SGLang generates responses (streaming, with APRIL for long-tail)        │
│     │                                                                       │
│     ├──► As responses complete, send to Data Buffer                         │
│     │                                                                       │
│     ▼                                                                       │
│  3. Adaptive Controller monitors staleness                                  │
│     │                                                                       │
│     ├─── If staleness LOW:  continue async, request more rollouts           │
│     ├─── If staleness HIGH: trigger sync barrier, wait for rollouts         │
│     │                                                                       │
│     ▼                                                                       │
│  4. Smart Batch Composer creates training batch                             │
│     │                                                                       │
│     ├──► Group by length (reduce padding)                                   │
│     ├──► Balance staleness (stratified sampling)                            │
│     ├──► Curriculum ordering (easy → hard)                                  │
│     │                                                                       │
│     ▼                                                                       │
│  5. Megatron executes training step                                         │
│     │                                                                       │
│     ├──► Compute importance-corrected loss                                  │
│     ├──► Gradient accumulation + sync                                       │
│     ├──► Optimizer step                                                     │
│     │                                                                       │
│     ▼                                                                       │
│  6. Weight Sync Manager updates SGLang                                      │
│     │                                                                       │
│     ├──► Lazy sync (mark dirty, sync on demand)                             │
│     ├──► Delta compression (only send changes)                              │
│     ├──► CUDA IPC (zero-copy transfer)                                      │
│     │                                                                       │
│     ▼                                                                       │
│  7. Metrics logged, loop continues                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Adaptive Async Controller

The brain of Flux's dynamic scheduling.

```python
class AdaptiveAsyncController:
    """
    Dynamically adjusts the sync/async ratio based on training dynamics.
    
    Key idea: Maintain staleness within a target range, not a fixed sync/async mode.
    """
    
    def __init__(self, config: AdaptiveAsyncConfig):
        self.target_staleness = config.target_staleness  # e.g., 0.15
        self.staleness_tolerance = config.tolerance       # e.g., 0.05
        
        # Adaptive parameters
        self.async_ratio = 0.5  # Start balanced
        self.sync_interval = 10  # Steps between forced sync
        
        # Exponential moving averages
        self.staleness_ema = 0.0
        self.throughput_ema = 0.0
        
        # PID-like controller for smooth adaptation
        self.integral_error = 0.0
        self.prev_error = 0.0
        
    def update(self, batch_metrics: BatchMetrics) -> AsyncDecision:
        """
        Called after each training step to decide next action.
        
        Returns:
            AsyncDecision with:
            - async_ratio: float in [0, 1]
            - should_sync: bool
            - sync_subset: Optional[List[WorkerId]] (partial sync)
        """
        # Update staleness estimate
        current_staleness = self._compute_staleness(batch_metrics)
        self.staleness_ema = 0.9 * self.staleness_ema + 0.1 * current_staleness
        
        # PID control for async ratio
        error = self.target_staleness - self.staleness_ema
        self.integral_error += error
        derivative = error - self.prev_error
        self.prev_error = error
        
        # PID output
        adjustment = (
            self.config.kp * error +
            self.config.ki * self.integral_error +
            self.config.kd * derivative
        )
        
        # Update async ratio
        self.async_ratio = np.clip(
            self.async_ratio + adjustment,
            self.config.min_async_ratio,  # Never fully sync (waste)
            self.config.max_async_ratio   # Never fully async (unstable)
        )
        
        # Decide if sync needed
        should_sync = (
            self.staleness_ema > self.target_staleness + self.staleness_tolerance or
            self.steps_since_sync > self.sync_interval
        )
        
        return AsyncDecision(
            async_ratio=self.async_ratio,
            should_sync=should_sync,
            sync_subset=self._select_sync_subset() if should_sync else None
        )
    
    def _compute_staleness(self, metrics: BatchMetrics) -> float:
        """Multi-metric staleness computation."""
        # Metric 1: KL divergence (requires forward pass on ref model)
        kl_staleness = metrics.kl_divergence / self.config.kl_normalizer
        
        # Metric 2: Importance weight variance
        iw_variance = metrics.importance_weight_variance
        iw_staleness = np.log1p(iw_variance) / self.config.iw_normalizer
        
        # Metric 3: Version gap (lightweight)
        version_staleness = metrics.mean_version_gap / self.config.max_version_gap
        
        # Weighted combination
        staleness = (
            0.4 * kl_staleness +
            0.3 * iw_staleness +
            0.3 * version_staleness
        )
        
        return float(np.clip(staleness, 0, 1))
```

### 2. Streaming Rollout Manager

Handles generation with APRIL (Active Partial Rollout) strategy.

```python
class StreamingRolloutManager:
    """
    Manages rollout generation with:
    - Streaming output (don't wait for all to complete)
    - APRIL strategy (oversample, abort long-tail, reuse partials)
    - Length prediction (schedule short prompts first)
    """
    
    def __init__(self, sglang_client: SGLangClient, config: RolloutConfig):
        self.client = sglang_client
        self.config = config
        
        # APRIL components
        self.length_predictor = LengthPredictor()
        self.partial_buffer = PartialTrajectoryBuffer()
        
        # Metrics
        self.completion_times = deque(maxlen=1000)
        
    async def generate_batch(
        self,
        prompts: List[Prompt],
        target_count: int,
        temperature: float
    ) -> AsyncIterator[List[Trajectory]]:
        """
        Streaming batch generation with APRIL.
        
        Yields trajectories as they complete, allowing training to start early.
        """
        # Step 1: Predict lengths and sort (short first)
        predictions = self.length_predictor.batch_predict(prompts)
        sorted_prompts = self._sort_by_predicted_length(prompts, predictions)
        
        # Step 2: Check partial buffer for reusable trajectories
        reusable = self.partial_buffer.get_continuable(sorted_prompts)
        new_prompts = [p for p in sorted_prompts if p.id not in reusable]
        
        # Step 3: Oversample new prompts
        oversample_count = int(len(new_prompts) * self.config.oversample_ratio)
        to_generate = new_prompts[:oversample_count]
        
        # Step 4: Launch all generations (non-blocking)
        futures = {}
        for prompt in to_generate:
            future = asyncio.create_task(
                self.client.generate(
                    prompt=prompt.text,
                    temperature=temperature,
                    max_tokens=self.config.max_tokens
                )
            )
            futures[future] = prompt
        
        # Also continue partial trajectories
        for prompt_id, partial in reusable.items():
            future = asyncio.create_task(
                self.client.continue_generation(partial)
            )
            futures[future] = partial.prompt
        
        # Step 5: Collect results (streaming)
        completed = []
        completed_ids = set()
        
        while futures and len(completed) < target_count:
            # Wait for any completion
            done, pending = await asyncio.wait(
                futures.keys(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=self.config.batch_timeout
            )
            
            for future in done:
                prompt = futures.pop(future)
                try:
                    result = future.result()
                    trajectory = self._create_trajectory(prompt, result)
                    
                    if prompt.id not in completed_ids:
                        completed.append(trajectory)
                        completed_ids.add(prompt.id)
                        
                        # Yield batch when enough collected
                        if len(completed) >= self.config.min_yield_size:
                            yield completed
                            completed = []
                            
                except asyncio.TimeoutError:
                    # Save partial for later
                    partial = future.get_partial_result()
                    if partial:
                        self.partial_buffer.add(prompt.id, partial)
            
            futures = {f: p for f, p in futures.items() if f in pending}
        
        # Step 6: Abort remaining (long-tail) and save partials
        for future, prompt in futures.items():
            future.cancel()
            partial = await self._get_partial_safe(future)
            if partial:
                self.partial_buffer.add(prompt.id, partial)
        
        # Yield any remaining
        if completed:
            yield completed
```

### 3. Unified Importance Correction

Handles all sources of off-policy data.

```python
class UnifiedImportanceCorrection:
    """
    Computes importance weights that correct for:
    1. Staleness (data from old policy versions)
    2. Trajectory inconsistency (mixed versions within trajectory)
    3. Replay (data reused from buffer)
    
    Key insight: These are all forms of distribution shift, handle uniformly.
    """
    
    def __init__(self, config: ImportanceCorrectionConfig):
        self.config = config
        self.version_tracker = PolicyVersionTracker()
        
    def compute_weights(
        self,
        trajectories: List[Trajectory],
        current_policy: Policy
    ) -> torch.Tensor:
        """
        Compute per-token importance weights for a batch.
        
        Returns:
            Tensor of shape [batch, seq_len] with importance weights
        """
        weights = []
        
        for traj in trajectories:
            traj_weights = self._compute_trajectory_weights(traj, current_policy)
            weights.append(traj_weights)
        
        # Stack and clip
        weights = torch.stack(weights)
        weights = torch.clamp(weights, self.config.min_weight, self.config.max_weight)
        
        # Optional: normalize to reduce variance
        if self.config.normalize_weights:
            weights = weights / weights.mean()
        
        return weights
    
    def _compute_trajectory_weights(
        self,
        trajectory: Trajectory,
        current_policy: Policy
    ) -> torch.Tensor:
        """Compute weights for a single trajectory."""
        
        # Base importance weight: π_current / π_behavior
        with torch.no_grad():
            current_logprobs = current_policy.log_prob(
                trajectory.tokens,
                trajectory.attention_mask
            )
        
        log_ratio = current_logprobs - trajectory.behavior_logprobs
        base_weight = torch.exp(log_ratio)
        
        # Staleness correction
        staleness = self.version_tracker.current_version - trajectory.policy_version
        staleness_factor = self.config.staleness_decay ** staleness
        
        # Trajectory consistency correction
        # If trajectory was generated across multiple policy versions
        if trajectory.has_version_boundaries:
            consistency_weights = self._compute_consistency_weights(trajectory)
        else:
            consistency_weights = torch.ones_like(base_weight)
        
        # Combine
        final_weight = base_weight * staleness_factor * consistency_weights
        
        return final_weight
    
    def _compute_consistency_weights(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Handle trajectories generated across policy version boundaries.
        
        Strategy: Weight each segment by how "fresh" its generating policy was.
        """
        weights = torch.ones(trajectory.length)
        
        for segment in trajectory.version_segments:
            version_gap = self.version_tracker.current_version - segment.version
            segment_weight = self.config.staleness_decay ** version_gap
            weights[segment.start:segment.end] = segment_weight
        
        return weights
```

### 4. Smart Batch Composer

Creates optimized training batches.

```python
class SmartBatchComposer:
    """
    Composes training batches with multiple optimizations:
    1. Length-aware packing (minimize padding)
    2. Staleness balancing (stratified sampling)
    3. Curriculum ordering (progressive difficulty)
    4. Diversity enforcement (avoid repetition)
    """
    
    def __init__(self, config: BatchComposerConfig):
        self.config = config
        self.difficulty_estimator = DifficultyEstimator()
        
    def compose_batch(
        self,
        buffer: TrajectoryBuffer,
        batch_size: int,
        training_progress: float
    ) -> TrainingBatch:
        """
        Compose an optimized training batch.
        
        Args:
            buffer: Available trajectories
            batch_size: Target batch size (in sequences)
            training_progress: Float in [0, 1] indicating training progress
            
        Returns:
            TrainingBatch with selected trajectories and metadata
        """
        candidates = buffer.get_available()
        
        # Step 1: Length-based bucketing
        length_buckets = self._bucket_by_length(candidates)
        
        # Step 2: Select bucket based on current needs
        # Early training: more diversity (mix buckets)
        # Late training: more efficiency (single bucket)
        bucket_probs = self._compute_bucket_probs(length_buckets, training_progress)
        
        selected = []
        remaining_size = batch_size
        
        while remaining_size > 0 and any(length_buckets.values()):
            # Sample bucket
            bucket_id = self._sample_bucket(bucket_probs, length_buckets)
            bucket = length_buckets[bucket_id]
            
            if not bucket:
                continue
            
            # Step 3: Within bucket, stratified sampling by staleness
            to_sample = min(remaining_size, len(bucket))
            sampled = self._stratified_sample_by_staleness(bucket, to_sample)
            
            selected.extend(sampled)
            remaining_size -= len(sampled)
            
            # Remove sampled from bucket
            for t in sampled:
                bucket.remove(t)
        
        # Step 4: Order by difficulty (curriculum)
        if self.config.use_curriculum and training_progress > 0.1:
            selected = self._curriculum_order(selected, training_progress)
        
        # Step 5: Create batch with efficient packing
        batch = self._pack_batch(selected)
        
        return batch
    
    def _bucket_by_length(
        self,
        trajectories: List[Trajectory]
    ) -> Dict[str, List[Trajectory]]:
        """Group trajectories by length for efficient padding."""
        buckets = defaultdict(list)
        
        for traj in trajectories:
            # Bucket by length ranges: 0-512, 512-1024, 1024-2048, 2048+
            length = traj.total_length
            if length <= 512:
                bucket_id = "short"
            elif length <= 1024:
                bucket_id = "medium"
            elif length <= 2048:
                bucket_id = "long"
            else:
                bucket_id = "very_long"
            
            buckets[bucket_id].append(traj)
        
        return buckets
    
    def _stratified_sample_by_staleness(
        self,
        candidates: List[Trajectory],
        n: int
    ) -> List[Trajectory]:
        """
        Sample to balance staleness distribution.
        
        Goal: Each batch should have similar staleness distribution
              to reduce importance weight variance.
        """
        # Group by staleness
        staleness_groups = defaultdict(list)
        for traj in candidates:
            staleness_bucket = int(traj.staleness)  # 0, 1, 2, ...
            staleness_groups[staleness_bucket].append(traj)
        
        # Sample proportionally from each group
        selected = []
        samples_per_group = n // max(len(staleness_groups), 1)
        
        for group_id in sorted(staleness_groups.keys()):
            group = staleness_groups[group_id]
            to_take = min(samples_per_group, len(group))
            selected.extend(random.sample(group, to_take))
        
        # Fill remaining randomly
        remaining = [t for t in candidates if t not in selected]
        if len(selected) < n and remaining:
            additional = min(n - len(selected), len(remaining))
            selected.extend(random.sample(remaining, additional))
        
        return selected[:n]
    
    def _curriculum_order(
        self,
        trajectories: List[Trajectory],
        progress: float
    ) -> List[Trajectory]:
        """
        Order by difficulty for curriculum learning.
        
        Early training: More random (explore)
        Late training: Easy → Hard (exploit)
        """
        # Compute difficulties
        for traj in trajectories:
            if traj.difficulty is None:
                traj.difficulty = self.difficulty_estimator.estimate(traj)
        
        # Sort by difficulty
        sorted_trajs = sorted(trajectories, key=lambda t: t.difficulty)
        
        # Mix in some randomness (less as training progresses)
        randomness = 1.0 - progress
        
        if randomness > 0.1:
            # Shuffle within local windows
            window_size = max(1, int(len(sorted_trajs) * randomness))
            for i in range(0, len(sorted_trajs), window_size):
                window = sorted_trajs[i:i+window_size]
                random.shuffle(window)
                sorted_trajs[i:i+window_size] = window
        
        return sorted_trajs
```

### 5. Weight Sync Manager

Efficient weight synchronization between training and inference.

```python
class WeightSyncManager:
    """
    Manages weight synchronization between Megatron (training) and SGLang (inference).
    
    Optimizations:
    1. Lazy sync: Only sync when inference actually needs new weights
    2. Delta compression: Send only changed parameters
    3. CUDA IPC: Zero-copy transfer on same node
    4. Async broadcast: Don't block training
    """
    
    def __init__(
        self,
        megatron_model: MegatronModel,
        sglang_servers: List[SGLangServer],
        config: WeightSyncConfig
    ):
        self.megatron = megatron_model
        self.servers = sglang_servers
        self.config = config
        
        # Version tracking
        self.current_version = 0
        self.server_versions = {s.id: 0 for s in sglang_servers}
        
        # Delta compression state
        self.weight_snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
        self.snapshot_interval = config.snapshot_interval
        
        # CUDA IPC handles
        self.ipc_handles: Dict[str, Any] = {}
        
    def mark_updated(self):
        """Called after each training step."""
        self.current_version += 1
        
        # Periodic snapshot for delta compression
        if self.current_version % self.snapshot_interval == 0:
            self._take_snapshot()
    
    def sync_server(self, server_id: str, blocking: bool = False):
        """
        Sync weights to a specific server.
        
        Args:
            server_id: Target server
            blocking: If True, wait for sync to complete
        """
        server_version = self.server_versions[server_id]
        
        if server_version == self.current_version:
            return  # Already up to date
        
        # Choose sync method
        if self._should_use_delta(server_version):
            future = self._sync_delta(server_id, server_version)
        else:
            future = self._sync_full(server_id)
        
        if blocking:
            future.result()
        
        self.server_versions[server_id] = self.current_version
    
    def sync_all(self, blocking: bool = False):
        """Sync all servers."""
        futures = []
        for server in self.servers:
            future = asyncio.create_task(
                self._async_sync_server(server.id)
            )
            futures.append(future)
        
        if blocking:
            asyncio.get_event_loop().run_until_complete(
                asyncio.gather(*futures)
            )
    
    async def _sync_delta(self, server_id: str, from_version: int):
        """Send only parameter deltas."""
        server = self._get_server(server_id)
        
        # Find closest snapshot
        snapshot_version = self._find_closest_snapshot(from_version)
        snapshot = self.weight_snapshots[snapshot_version]
        
        # Compute delta
        current_weights = self._get_current_weights()
        delta = {}
        
        for name, current in current_weights.items():
            if name in snapshot:
                diff = current - snapshot[name]
                
                # Sparsify if beneficial
                if self._should_sparsify(diff):
                    delta[name] = self._sparsify(diff)
                else:
                    delta[name] = diff
        
        # Send delta
        await server.apply_weight_delta(delta)
    
    async def _sync_full(self, server_id: str):
        """Full weight sync using CUDA IPC."""
        server = self._get_server(server_id)
        
        if server.is_local:
            # Same node: use CUDA IPC (zero-copy)
            handles = self._get_ipc_handles()
            await server.load_weights_from_ipc(handles)
        else:
            # Different node: serialize and send
            weights = self._get_current_weights()
            await server.load_weights(weights)
    
    def _get_ipc_handles(self) -> Dict[str, Any]:
        """Get CUDA IPC handles for zero-copy transfer."""
        if not self.ipc_handles or self._handles_stale():
            self.ipc_handles = {}
            
            for name, param in self.megatron.named_parameters():
                # Create IPC handle
                handle = param.data.storage()._share_cuda_()
                self.ipc_handles[name] = {
                    'handle': handle,
                    'shape': param.shape,
                    'dtype': param.dtype
                }
        
        return self.ipc_handles
    
    def _should_sparsify(self, diff: torch.Tensor) -> bool:
        """Check if delta is sparse enough to benefit from sparse encoding."""
        nonzero_ratio = (diff.abs() > self.config.sparsity_threshold).float().mean()
        return nonzero_ratio < self.config.sparsity_target
    
    def _sparsify(self, diff: torch.Tensor) -> SparseTensor:
        """Convert dense delta to sparse representation."""
        mask = diff.abs() > self.config.sparsity_threshold
        indices = mask.nonzero()
        values = diff[mask]
        return SparseTensor(indices=indices, values=values, shape=diff.shape)
```

---

## API Design

### User-Facing API

```python
# flux/trainer.py

class FluxTrainer:
    """
    Main entry point for Flux training.
    
    Example:
        trainer = FluxTrainer(config)
        trainer.fit(train_prompts, num_steps=10000)
    """
    
    def __init__(self, config: FluxConfig):
        """
        Initialize Flux trainer.
        
        Args:
            config: FluxConfig with all settings
        """
        pass
    
    def fit(
        self,
        prompts: Union[List[str], Dataset],
        num_steps: int,
        eval_prompts: Optional[List[str]] = None,
        eval_interval: int = 500,
        checkpoint_interval: int = 1000,
        callbacks: Optional[List[Callback]] = None
    ) -> TrainingResult:
        """
        Run training loop.
        
        Args:
            prompts: Training prompts
            num_steps: Total training steps
            eval_prompts: Optional evaluation prompts
            eval_interval: Steps between evaluations
            checkpoint_interval: Steps between checkpoints
            callbacks: Optional callbacks for custom logic
            
        Returns:
            TrainingResult with metrics and final model path
        """
        pass
    
    def save_checkpoint(self, path: str):
        """Save current state to checkpoint."""
        pass
    
    def load_checkpoint(self, path: str):
        """Load state from checkpoint."""
        pass


# Configuration
@dataclass
class FluxConfig:
    """Main configuration for Flux trainer."""
    
    # Model configs
    model: ModelConfig
    
    # Backend configs
    megatron: MegatronConfig
    sglang: SGLangConfig
    
    # Algorithm configs
    algorithm: AlgorithmConfig  # PPO, GRPO, etc.
    reward: RewardConfig
    
    # Adaptive configs
    adaptive_async: AdaptiveAsyncConfig
    batch_composer: BatchComposerConfig
    
    # Training configs
    learning_rate: float = 1e-6
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Logging
    log_interval: int = 10
    wandb_project: Optional[str] = None


@dataclass
class AdaptiveAsyncConfig:
    """Configuration for adaptive async controller."""
    
    target_staleness: float = 0.15
    tolerance: float = 0.05
    
    min_async_ratio: float = 0.1
    max_async_ratio: float = 0.9
    
    # PID controller gains
    kp: float = 0.1
    ki: float = 0.01
    kd: float = 0.05
    
    # Staleness computation
    staleness_decay: float = 0.95
    kl_normalizer: float = 0.1
    iw_normalizer: float = 2.0
```

### Extending Flux

```python
# Custom reward function
class CustomReward(RewardFunction):
    def __init__(self, model_path: str):
        self.model = load_reward_model(model_path)
    
    def compute(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        return self.model.score(prompts, responses)


# Custom algorithm
class CustomAlgorithm(RLAlgorithm):
    def compute_loss(
        self,
        batch: TrainingBatch,
        weights: torch.Tensor
    ) -> torch.Tensor:
        # Your custom loss computation
        pass


# Using custom components
config = FluxConfig(
    algorithm=AlgorithmConfig(
        name="custom",
        impl=CustomAlgorithm(...)
    ),
    reward=RewardConfig(
        impl=CustomReward("path/to/reward/model")
    )
)
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

| Task | Description | Priority |
|------|-------------|----------|
| Project setup | Repository, CI/CD, documentation structure | P0 |
| Megatron integration | Basic training loop with Megatron | P0 |
| SGLang integration | HTTP client for generation | P0 |
| Weight sync | CUDA IPC implementation | P0 |
| Basic coordinator | Async event loop, metrics collection | P0 |

**Milestone**: Basic synchronous training working (VERL-equivalent)

### Phase 2: Adaptive Components (Weeks 4-6)

| Task | Description | Priority |
|------|-------------|----------|
| Staleness measurement | KL, importance weight variance | P0 |
| Adaptive async controller | PID-based control | P0 |
| Streaming rollout | APRIL strategy | P1 |
| Smart batch composer | Length bucketing, stratified sampling | P1 |
| Delta weight sync | Compression, sparse encoding | P2 |

**Milestone**: Adaptive async working, showing GPU utilization improvement

### Phase 3: Optimizations (Weeks 7-9)

| Task | Description | Priority |
|------|-------------|----------|
| Length predictor | ML model for output length prediction | P1 |
| Speculative sync | Early training start | P1 |
| Curriculum learning | Difficulty estimation, ordering | P2 |
| Multi-objective rewards | Dynamic reward composition | P2 |
| Partial trajectory reuse | Buffer management for APRIL | P2 |

**Milestone**: Full feature set, competitive with AReaL throughput

### Phase 4: Production Readiness (Weeks 10-12)

| Task | Description | Priority |
|------|-------------|----------|
| Fault tolerance | Checkpoint/restore, worker restart | P0 |
| Multi-node support | Cross-node weight sync | P0 |
| Monitoring | Prometheus metrics, Grafana dashboards | P1 |
| Documentation | User guide, API docs, examples | P0 |
| Benchmarks | Comparison with VERL, AReaL, Slime | P1 |

**Milestone**: Production-ready release

---

## Evaluation Plan

### Metrics

| Category | Metric | Target |
|----------|--------|--------|
| **Throughput** | Samples/hour | 2x VERL |
| **Efficiency** | GPU utilization | > 80% |
| **Stability** | Reward variance | ≤ VERL |
| **Quality** | Final reward | ≥ VERL |
| **Scalability** | Scaling efficiency at 64 GPUs | > 85% |

### Benchmarks

1. **Controlled comparison**: Same model, same data, same hardware
   - Models: Qwen3-8B, Qwen3-32B, Qwen3-72B
   - Data: OpenHermes, UltraChat, Infinity-Instruct
   - Hardware: 8x H100, 64x H100

2. **Ablation studies**:
   - Adaptive async vs fixed async ratio
   - APRIL vs standard rollout
   - Smart batching vs random batching
   - Curriculum vs random ordering

3. **Real-world tasks**:
   - Math reasoning (GSM8K, MATH)
   - Code generation (HumanEval, MBPP)
   - General chat (MT-Bench, AlpacaEval)

---

## Related Work

### Training Frameworks

| Framework | Paper/Repo | Key Contribution |
|-----------|------------|------------------|
| VERL | [HybridFlow (EuroSys'25)](https://arxiv.org/abs/2409.19256) | Hybrid controller, 3D-HybridEngine |
| AReaL | [arXiv:2505.24298](https://arxiv.org/abs/2505.24298) | Fully async, staleness-aware PPO |
| Slime | [THUDM/slime](https://github.com/THUDM/slime) | SGLang-native, APRIL strategy |
| OpenRLHF | [GitHub](https://github.com/OpenRLHF/OpenRLHF) | Easy-to-use RLHF |
| TRL | [Hugging Face](https://github.com/huggingface/trl) | Transformers integration |

### Algorithms

| Algorithm | Paper | Key Idea |
|-----------|-------|----------|
| PPO | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) | Clipped surrogate objective |
| GRPO | [Shao et al., 2024](https://arxiv.org/abs/2402.03300) | Group relative policy optimization |
| DPO | [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) | Direct preference optimization |
| DAPO | [Yu et al., 2025](https://arxiv.org/abs/2503.14476) | Decoupled clip and dynamic sampling |

### Systems

| System | Description |
|--------|-------------|
| Megatron-LM | NVIDIA's distributed training library |
| SGLang | Fast LLM serving with RadixAttention |
| vLLM | PagedAttention for efficient serving |
| DeepSpeed | Microsoft's distributed training library |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Staleness** | How much the policy has changed since data was generated |
| **Async ratio** | Fraction of training steps that don't wait for all rollouts |
| **APRIL** | Active Partial Rollout - oversample, abort long-tail, reuse partials |
| **Importance weight** | π_current(a\|s) / π_behavior(a\|s), corrects for distribution shift |
| **Colocate** | Training and inference on same GPUs (time-shared) |
| **Separate** | Training and inference on different GPUs (space-shared) |

---

## Appendix B: Configuration Reference

See `configs/` directory for example configurations:

- `configs/qwen3-8b-8gpu.yaml`: Basic 8B model on 8 GPUs
- `configs/qwen3-72b-64gpu.yaml`: 72B model on 64 GPUs
- `configs/qwen3-moe-128gpu.yaml`: MoE model on 128 GPUs

---

## License

Apache 2.0

---

## Citation

```bibtex
@software{flux2025,
  author = {Xin},
  title = {Flux: An Adaptive Post-Training Framework for Large Language Models},
  year = {2025},
  url = {https://github.com/xxx/flux}
}
```
