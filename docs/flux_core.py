"""
Flux: Core Components Implementation Skeleton
=============================================

This file contains the initial implementation of Flux's core components.
Each class is designed to be self-contained and testable.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

# Note: torch imports would be needed in actual implementation
# import torch
# import torch.distributed as dist


# =============================================================================
# Core Types and Configurations
# =============================================================================

@dataclass
class FluxConfig:
    """Main configuration for Flux trainer."""
    
    # Model
    model_path: str
    model_type: str = "qwen3"
    
    # Training
    learning_rate: float = 1e-6
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    num_steps: int = 10000
    
    # Generation
    max_tokens: int = 2048
    temperature: float = 1.0
    
    # Adaptive async
    target_staleness: float = 0.15
    staleness_tolerance: float = 0.05
    min_async_ratio: float = 0.1
    max_async_ratio: float = 0.9
    
    # Rollout
    oversample_ratio: float = 1.5
    min_yield_size: int = 8
    batch_timeout: float = 30.0
    
    # Weight sync
    sync_interval: int = 1
    use_delta_compression: bool = True
    sparsity_threshold: float = 1e-6
    
    # Importance correction
    staleness_decay: float = 0.95
    min_weight: float = 0.1
    max_weight: float = 10.0
    
    # SGLang
    sglang_url: str = "http://localhost:8000"
    sglang_num_servers: int = 1
    
    # Megatron
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    
    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 1000
    eval_interval: int = 500


@dataclass
class Trajectory:
    """Represents a single rollout trajectory."""
    
    id: str
    prompt: str
    response: str
    tokens: List[int]
    
    # Probabilities
    behavior_logprobs: List[float]
    
    # Metadata
    policy_version: int
    generation_time: float
    
    # Optional fields
    reward: Optional[float] = None
    advantages: Optional[List[float]] = None
    difficulty: Optional[float] = None
    
    # Version boundaries (for trajectories spanning policy updates)
    version_segments: List[Tuple[int, int, int]] = field(default_factory=list)

    # Staleness is derived from policy_version vs current version (see get_version_gap in trajectory.py)

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def total_length(self) -> int:
        return len(self.prompt) + len(self.response)

    @property
    def has_version_boundaries(self) -> bool:
        return len(self.version_segments) > 1


@dataclass
class TrainingBatch:
    """A batch ready for training."""
    
    trajectories: List[Trajectory]
    
    # Computed fields
    importance_weights: Optional[Any] = None  # torch.Tensor
    padding_mask: Optional[Any] = None        # torch.Tensor
    
    @property
    def size(self) -> int:
        return len(self.trajectories)
    
    @property
    def total_tokens(self) -> int:
        return sum(t.length for t in self.trajectories)


@dataclass
class BatchMetrics:
    """Metrics from a training batch."""
    
    loss: float
    reward_mean: float
    reward_std: float
    kl_divergence: float
    importance_weight_variance: float
    mean_version_gap: float
    
    # Throughput
    samples_per_second: float
    tokens_per_second: float
    
    # GPU utilization (if available)
    gpu_utilization: Optional[float] = None


@dataclass
class AsyncDecision:
    """Decision from the adaptive async controller."""
    
    async_ratio: float
    should_sync: bool
    sync_subset: Optional[List[str]] = None  # Worker IDs to sync


# =============================================================================
# Adaptive Async Controller
# =============================================================================

class AdaptiveAsyncController:
    """
    Dynamically adjusts the sync/async ratio based on training dynamics.
    
    Key idea: Maintain staleness within a target range, not a fixed sync/async mode.
    
    Uses PID control for smooth adaptation.
    """
    
    def __init__(self, config: FluxConfig):
        self.config = config
        self.target_staleness = config.target_staleness
        self.tolerance = config.staleness_tolerance
        
        # Adaptive state
        self.async_ratio = 0.5  # Start balanced
        
        # PID controller state
        self.integral_error = 0.0
        self.prev_error = 0.0
        
        # PID gains (can be tuned)
        self.kp = 0.1
        self.ki = 0.01
        self.kd = 0.05
        
        # Staleness tracking
        self.staleness_ema = 0.0
        self.staleness_history = deque(maxlen=100)
        
        # Sync tracking
        self.steps_since_sync = 0
        self.max_steps_without_sync = 50
        
    def update(self, metrics: BatchMetrics) -> AsyncDecision:
        """
        Update controller and return decision for next step.
        
        Args:
            metrics: Metrics from the last training batch
            
        Returns:
            AsyncDecision with async_ratio and sync decision
        """
        # Update staleness estimate
        current_staleness = self._compute_staleness(metrics)
        self.staleness_ema = 0.9 * self.staleness_ema + 0.1 * current_staleness
        self.staleness_history.append(current_staleness)
        
        # PID control
        error = self.target_staleness - self.staleness_ema
        self.integral_error = np.clip(
            self.integral_error + error,
            -1.0, 1.0  # Anti-windup
        )
        derivative = error - self.prev_error
        self.prev_error = error
        
        # Compute adjustment
        adjustment = (
            self.kp * error +
            self.ki * self.integral_error +
            self.kd * derivative
        )
        
        # Update async ratio
        self.async_ratio = np.clip(
            self.async_ratio + adjustment,
            self.config.min_async_ratio,
            self.config.max_async_ratio
        )
        
        # Decide if sync needed
        self.steps_since_sync += 1
        should_sync = self._should_sync()
        
        if should_sync:
            self.steps_since_sync = 0
        
        return AsyncDecision(
            async_ratio=self.async_ratio,
            should_sync=should_sync,
            sync_subset=None  # Could implement partial sync
        )
    
    def _compute_staleness(self, metrics: BatchMetrics) -> float:
        """Compute staleness from multiple metrics."""
        # Normalize each metric
        kl_staleness = min(metrics.kl_divergence / 0.1, 1.0)
        iw_staleness = min(np.log1p(metrics.importance_weight_variance) / 2.0, 1.0)
        version_staleness = min(metrics.mean_version_gap / 5.0, 1.0)
        
        # Weighted combination
        staleness = (
            0.4 * kl_staleness +
            0.3 * iw_staleness +
            0.3 * version_staleness
        )
        
        return float(np.clip(staleness, 0, 1))
    
    def _should_sync(self) -> bool:
        """Determine if a sync barrier is needed."""
        # Sync if staleness too high
        if self.staleness_ema > self.target_staleness + self.tolerance:
            return True
        
        # Periodic sync
        if self.steps_since_sync >= self.max_steps_without_sync:
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, float]:
        """Get controller statistics for logging."""
        return {
            "async_ratio": self.async_ratio,
            "staleness_ema": self.staleness_ema,
            "staleness_std": np.std(self.staleness_history) if self.staleness_history else 0,
            "steps_since_sync": self.steps_since_sync,
        }


# =============================================================================
# Unified Importance Correction
# =============================================================================

class UnifiedImportanceCorrection:
    """
    Computes importance weights that correct for all sources of distribution shift.
    
    Handles:
    1. Staleness (data from old policy versions)
    2. Trajectory inconsistency (mixed versions within trajectory)
    3. Replay (data reused from buffer)
    """
    
    def __init__(self, config: FluxConfig):
        self.config = config
        self.staleness_decay = config.staleness_decay
        self.min_weight = config.min_weight
        self.max_weight = config.max_weight
        
        # Version tracking
        self.current_version = 0
        
    def increment_version(self):
        """Called after each training step."""
        self.current_version += 1
    
    def compute_weights(
        self,
        trajectories: List[Trajectory],
        current_logprobs: List[List[float]]
    ) -> List[List[float]]:
        """
        Compute per-token importance weights.
        
        Args:
            trajectories: List of trajectories
            current_logprobs: Log probs under current policy
            
        Returns:
            List of weight lists, one per trajectory
        """
        all_weights = []
        
        for traj, curr_lp in zip(trajectories, current_logprobs):
            weights = self._compute_trajectory_weights(traj, curr_lp)
            all_weights.append(weights)
        
        return all_weights
    
    def _compute_trajectory_weights(
        self,
        trajectory: Trajectory,
        current_logprobs: List[float]
    ) -> List[float]:
        """Compute weights for a single trajectory."""
        
        # Base importance weight: exp(log π_current - log π_behavior)
        log_ratios = [
            curr - behav 
            for curr, behav in zip(current_logprobs, trajectory.behavior_logprobs)
        ]
        base_weights = [np.exp(lr) for lr in log_ratios]
        
        # Staleness correction
        staleness = self.current_version - trajectory.policy_version
        staleness_factor = self.staleness_decay ** staleness
        
        # Trajectory consistency correction
        if trajectory.has_version_boundaries:
            consistency_weights = self._compute_consistency_weights(trajectory)
        else:
            consistency_weights = [1.0] * len(base_weights)
        
        # Combine and clip
        final_weights = []
        for bw, cw in zip(base_weights, consistency_weights):
            w = bw * staleness_factor * cw
            w = np.clip(w, self.min_weight, self.max_weight)
            final_weights.append(w)
        
        return final_weights
    
    def _compute_consistency_weights(self, trajectory: Trajectory) -> List[float]:
        """Handle trajectories with version boundaries."""
        weights = [1.0] * trajectory.length
        
        for start, end, version in trajectory.version_segments:
            version_gap = self.current_version - version
            segment_weight = self.staleness_decay ** version_gap
            for i in range(start, min(end, trajectory.length)):
                weights[i] = segment_weight
        
        return weights


# =============================================================================
# Streaming Rollout Manager
# =============================================================================

class StreamingRolloutManager:
    """
    Manages rollout generation with streaming and APRIL strategy.
    
    Features:
    - Streaming output (yield as trajectories complete)
    - APRIL: Oversample, abort long-tail, reuse partials
    - Length prediction for scheduling
    """
    
    def __init__(self, config: FluxConfig):
        self.config = config
        self.sglang_url = config.sglang_url
        
        # APRIL state
        self.partial_buffer: Dict[str, Dict] = {}
        
        # Length prediction (simple moving average for now)
        self.length_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Statistics
        self.completion_times = deque(maxlen=1000)
        
    async def generate_batch(
        self,
        prompts: List[Dict[str, Any]],
        target_count: int,
        temperature: float,
        policy_version: int
    ) -> AsyncIterator[List[Trajectory]]:
        """
        Generate trajectories with streaming and APRIL.
        
        Yields batches of trajectories as they complete.
        """
        # Sort by predicted length (short first)
        sorted_prompts = self._sort_by_predicted_length(prompts)
        
        # Oversample
        oversample_count = min(
            int(target_count * self.config.oversample_ratio),
            len(sorted_prompts)
        )
        to_generate = sorted_prompts[:oversample_count]
        
        # Launch all generations
        tasks = {}
        for prompt in to_generate:
            task = asyncio.create_task(
                self._generate_single(prompt, temperature, policy_version)
            )
            tasks[task] = prompt
        
        # Collect results (streaming)
        completed = []
        completed_ids = set()
        
        while tasks and len(completed_ids) < target_count:
            done, pending = await asyncio.wait(
                tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=self.config.batch_timeout
            )
            
            for task in done:
                prompt = tasks.pop(task)
                
                try:
                    trajectory = task.result()
                    
                    if prompt['id'] not in completed_ids:
                        completed.append(trajectory)
                        completed_ids.add(prompt['id'])
                        
                        # Update length history
                        self._update_length_history(prompt, trajectory)
                        
                        # Yield when enough collected
                        if len(completed) >= self.config.min_yield_size:
                            yield completed
                            completed = []
                            
                except Exception as e:
                    print(f"Generation failed for {prompt['id']}: {e}")
            
            tasks = {t: p for t, p in tasks.items() if t in pending}
        
        # Cancel remaining (long-tail)
        for task, prompt in tasks.items():
            task.cancel()
            # Could save partial results here
        
        # Yield remaining
        if completed:
            yield completed
    
    async def _generate_single(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        policy_version: int
    ) -> Trajectory:
        """Generate a single trajectory."""
        start_time = time.time()
        
        # In real implementation, call SGLang HTTP API
        # response = await self.sglang_client.generate(...)
        
        # Placeholder
        response = {
            'text': '[Generated response]',
            'tokens': [1, 2, 3],
            'logprobs': [-0.1, -0.2, -0.3]
        }
        
        generation_time = time.time() - start_time
        self.completion_times.append(generation_time)
        
        return Trajectory(
            id=f"{prompt['id']}_{policy_version}",
            prompt=prompt['text'],
            response=response['text'],
            tokens=response['tokens'],
            behavior_logprobs=response['logprobs'],
            policy_version=policy_version,
            generation_time=generation_time
        )
    
    def _sort_by_predicted_length(
        self,
        prompts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort prompts by predicted output length (short first)."""
        def predict_length(prompt):
            history = self.length_history.get(prompt.get('type', 'default'))
            if history:
                return np.mean(history)
            return len(prompt['text']) * 2  # Simple heuristic
        
        return sorted(prompts, key=predict_length)
    
    def _update_length_history(self, prompt: Dict, trajectory: Trajectory):
        """Update length prediction history."""
        prompt_type = prompt.get('type', 'default')
        self.length_history[prompt_type].append(trajectory.length)


# =============================================================================
# Smart Batch Composer
# =============================================================================

class SmartBatchComposer:
    """
    Creates optimized training batches with:
    - Length-aware packing
    - Staleness balancing
    - Curriculum ordering
    """
    
    def __init__(self, config: FluxConfig):
        self.config = config
        
    def compose_batch(
        self,
        candidates: List[Trajectory],
        batch_size: int,
        training_progress: float
    ) -> TrainingBatch:
        """
        Compose an optimized training batch.
        
        Args:
            candidates: Available trajectories
            batch_size: Target number of trajectories
            training_progress: Float in [0, 1]
            
        Returns:
            TrainingBatch with selected trajectories
        """
        if len(candidates) <= batch_size:
            return TrainingBatch(trajectories=candidates)
        
        # Step 1: Bucket by length
        buckets = self._bucket_by_length(candidates)
        
        # Step 2: Sample with staleness balancing
        selected = self._stratified_sample(buckets, batch_size)
        
        # Step 3: Curriculum ordering (if enabled)
        if training_progress > 0.1:
            selected = self._curriculum_order(selected, training_progress)
        
        return TrainingBatch(trajectories=selected)
    
    def _bucket_by_length(
        self,
        trajectories: List[Trajectory]
    ) -> Dict[str, List[Trajectory]]:
        """Group by length ranges."""
        buckets = defaultdict(list)

        for traj in trajectories:
            bucket_id = self._get_length_bucket(traj.total_length)
            buckets[bucket_id].append(traj)

        return buckets

    def _get_length_bucket(self, length: int) -> str:
        """Determine bucket ID for a given length."""
        if length <= 512:
            return "short"
        if length <= 1024:
            return "medium"
        if length <= 2048:
            return "long"
        return "very_long"
    
    def _stratified_sample(
        self,
        buckets: Dict[str, List[Trajectory]],
        n: int
    ) -> List[Trajectory]:
        """Sample with staleness balancing."""
        all_trajs = [traj for bucket in buckets.values() for traj in bucket]

        if len(all_trajs) <= n:
            return all_trajs

        # Group by staleness
        staleness_groups = defaultdict(list)
        for traj in all_trajs:
            staleness_groups[int(traj.staleness)].append(traj)

        # Sample proportionally from each staleness group
        selected = []
        samples_per_group = n // max(len(staleness_groups), 1)

        for group_id in sorted(staleness_groups.keys()):
            group = staleness_groups[group_id]
            to_take = min(samples_per_group, len(group))
            selected.extend(np.random.choice(group, to_take, replace=False).tolist())

        # Fill remaining slots from unused trajectories
        selected_ids = {id(traj) for traj in selected}
        remaining = [t for t in all_trajs if id(t) not in selected_ids]
        if len(selected) < n and remaining:
            additional = min(n - len(selected), len(remaining))
            selected.extend(np.random.choice(remaining, additional, replace=False).tolist())

        return selected[:n]
    
    def _curriculum_order(
        self,
        trajectories: List[Trajectory],
        progress: float
    ) -> List[Trajectory]:
        """Order by difficulty for curriculum learning."""
        # Estimate difficulty for trajectories that don't have it set
        for traj in trajectories:
            if traj.difficulty is None:
                traj.difficulty = traj.length / 2048

        # Sort by difficulty
        sorted_trajs = sorted(trajectories, key=lambda t: t.difficulty)

        # Add randomness that decreases as training progresses
        randomness = 1.0 - progress
        if randomness > 0.1:
            window_size = max(1, int(len(sorted_trajs) * randomness))
            for i in range(0, len(sorted_trajs), window_size):
                window = sorted_trajs[i:i + window_size]
                np.random.shuffle(window)
                sorted_trajs[i:i + window_size] = window

        return sorted_trajs


# =============================================================================
# Weight Sync Manager
# =============================================================================

class WeightSyncManager:
    """
    Manages weight synchronization between training and inference.
    
    Features:
    - Lazy sync (mark dirty, sync on demand)
    - Delta compression
    - CUDA IPC for same-node sync
    """
    
    def __init__(self, config: FluxConfig):
        self.config = config
        
        # Version tracking
        self.current_version = 0
        self.server_versions: Dict[str, int] = {}
        
        # Delta compression state
        self.weight_snapshots: Dict[int, Dict[str, Any]] = {}
        self.snapshot_interval = 10
        
    def mark_updated(self):
        """Called after each training step."""
        self.current_version += 1
        
        # Take snapshot periodically
        if self.current_version % self.snapshot_interval == 0:
            self._take_snapshot()
    
    def sync_server(self, server_id: str, weights: Dict[str, Any]):
        """
        Sync weights to a specific server.
        
        In real implementation, this would:
        1. Check if delta sync is possible
        2. Compute delta or use full weights
        3. Send via CUDA IPC or network
        """
        server_version = self.server_versions.get(server_id, 0)
        
        if server_version == self.current_version:
            return  # Already up to date
        
        if self.config.use_delta_compression and self._can_delta_sync(server_version):
            delta = self._compute_delta(server_version, weights)
            self._send_delta(server_id, delta)
        else:
            self._send_full(server_id, weights)
        
        self.server_versions[server_id] = self.current_version
    
    def _take_snapshot(self):
        """Take a weight snapshot for delta compression."""
        # In real implementation, would clone current weights
        self.weight_snapshots[self.current_version] = {}

        # Remove snapshots older than 50 versions
        min_version = self.current_version - 50
        self.weight_snapshots = {
            v: s for v, s in self.weight_snapshots.items() if v >= min_version
        }
    
    def _can_delta_sync(self, from_version: int) -> bool:
        """Check if delta sync is possible from given version."""
        return from_version in self.weight_snapshots
    
    def _compute_delta(
        self,
        from_version: int,
        current_weights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute weight delta from snapshot."""
        snapshot = self.weight_snapshots[from_version]
        delta = {}
        
        for name, current in current_weights.items():
            if name in snapshot:
                # diff = current - snapshot[name]
                # In real implementation, would compute actual diff
                delta[name] = current  # Placeholder
        
        return delta
    
    def _send_delta(self, server_id: str, delta: Dict[str, Any]):
        """Send delta to server."""
        # In real implementation, would send via network/IPC
        pass
    
    def _send_full(self, server_id: str, weights: Dict[str, Any]):
        """Send full weights to server."""
        # In real implementation, would send via network/IPC
        pass


# =============================================================================
# Flux Coordinator
# =============================================================================

class FluxCoordinator:
    """
    Lightweight coordinator for orchestrating training.
    
    Responsibilities:
    - Manage training loop
    - Coordinate rollouts and training
    - Collect metrics
    - Handle checkpointing
    """
    
    def __init__(self, config: FluxConfig):
        self.config = config
        
        # Components
        self.async_controller = AdaptiveAsyncController(config)
        self.importance_correction = UnifiedImportanceCorrection(config)
        self.rollout_manager = StreamingRolloutManager(config)
        self.batch_composer = SmartBatchComposer(config)
        self.weight_sync = WeightSyncManager(config)
        
        # State
        self.current_step = 0
        self.trajectory_buffer: List[Trajectory] = []
        
        # Metrics
        self.metrics_history: List[BatchMetrics] = []
        
    async def train_step(self, prompts: List[Dict[str, Any]]) -> BatchMetrics:
        """
        Execute one training step.
        
        Args:
            prompts: Prompts for this step
            
        Returns:
            BatchMetrics from training
        """
        policy_version = self.importance_correction.current_version
        
        # Generate trajectories (streaming)
        async for trajectories in self.rollout_manager.generate_batch(
            prompts,
            target_count=self.config.batch_size,
            temperature=self.config.temperature,
            policy_version=policy_version
        ):
            # Add to buffer
            for traj in trajectories:
                traj.staleness = self.importance_correction.current_version - traj.policy_version
            self.trajectory_buffer.extend(trajectories)
            
            # Check if we should start training
            decision = self.async_controller.update(self._get_placeholder_metrics())
            
            if decision.should_sync or len(self.trajectory_buffer) >= self.config.batch_size:
                break
        
        # Compose training batch
        progress = self.current_step / self.config.num_steps
        batch = self.batch_composer.compose_batch(
            self.trajectory_buffer,
            self.config.batch_size,
            progress
        )
        
        # Compute importance weights
        # In real implementation, would get current logprobs from model
        current_logprobs = [[0.0] * t.length for t in batch.trajectories]
        weights = self.importance_correction.compute_weights(
            batch.trajectories,
            current_logprobs
        )
        batch.importance_weights = weights
        
        # Training step (placeholder)
        metrics = self._execute_training_step(batch)
        
        # Update state
        self.importance_correction.increment_version()
        self.weight_sync.mark_updated()
        self.current_step += 1
        
        # Clear used trajectories from buffer
        used_ids = {t.id for t in batch.trajectories}
        self.trajectory_buffer = [t for t in self.trajectory_buffer if t.id not in used_ids]
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _execute_training_step(self, batch: TrainingBatch) -> BatchMetrics:
        """Execute actual training step (placeholder)."""
        # In real implementation, would call Megatron
        return BatchMetrics(
            loss=0.5,
            reward_mean=0.0,
            reward_std=1.0,
            kl_divergence=0.01,
            importance_weight_variance=0.1,
            mean_version_gap=1.0,
            samples_per_second=100.0,
            tokens_per_second=10000.0
        )
    
    def _get_placeholder_metrics(self) -> BatchMetrics:
        """Get placeholder metrics for controller."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return self._create_empty_metrics()

    def _create_empty_metrics(self) -> BatchMetrics:
        """Create empty metrics with all values set to zero."""
        return BatchMetrics(
            loss=0.0,
            reward_mean=0.0,
            reward_std=0.0,
            kl_divergence=0.0,
            importance_weight_variance=0.0,
            mean_version_gap=0.0,
            samples_per_second=0.0,
            tokens_per_second=0.0,
        )


# =============================================================================
# Main Trainer
# =============================================================================

class FluxTrainer:
    """
    Main entry point for Flux training.
    
    Example:
        trainer = FluxTrainer(config)
        trainer.fit(prompts, num_steps=10000)
    """
    
    def __init__(self, config: FluxConfig):
        self.config = config
        self.coordinator = FluxCoordinator(config)
        
    async def fit(
        self,
        prompts: List[Dict[str, Any]],
        num_steps: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """Run training loop."""
        num_steps = num_steps or self.config.num_steps
        callbacks = callbacks or []

        for step in range(num_steps):
            batch_prompts = self._sample_prompts(prompts, self.config.batch_size)
            metrics = await self.coordinator.train_step(batch_prompts)

            if step % self.config.log_interval == 0:
                self._log_metrics(step, metrics)

            for callback in callbacks:
                callback(step, metrics)

            if step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_{step}")
    
    def _sample_prompts(
        self,
        prompts: List[Dict[str, Any]],
        n: int
    ) -> List[Dict[str, Any]]:
        """Sample prompts for a batch."""
        indices = np.random.choice(len(prompts), min(n, len(prompts)), replace=False)
        return [prompts[i] for i in indices]
    
    def _log_metrics(self, step: int, metrics: BatchMetrics):
        """Log training metrics."""
        stats = self.coordinator.async_controller.get_stats()
        print(
            f"Step {step}: "
            f"loss={metrics.loss:.4f}, "
            f"reward={metrics.reward_mean:.4f}, "
            f"async_ratio={stats['async_ratio']:.2f}, "
            f"staleness={stats['staleness_ema']:.3f}"
        )
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        # In real implementation, would save model weights, optimizer state, etc.
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        # In real implementation, would load model weights, optimizer state, etc.
        print(f"Checkpoint loaded from {path}")


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of Flux."""
    
    # Configuration
    config = FluxConfig(
        model_path="Qwen/Qwen3-8B",
        learning_rate=1e-6,
        batch_size=32,
        num_steps=1000,
        target_staleness=0.15,
    )
    
    # Sample prompts
    prompts = [
        {"id": f"prompt_{i}", "text": f"Solve this math problem: {i} + {i} = ?", "type": "math"}
        for i in range(1000)
    ]
    
    # Train
    trainer = FluxTrainer(config)
    await trainer.fit(prompts, num_steps=100)


if __name__ == "__main__":
    asyncio.run(main())
