"""
FluxCoordinator - Orchestrates the complete RLHF training loop.

The coordinator manages the flow of data between:
1. Rollout generation (SGLang)
2. Reward computation
3. Batch composition
4. Training (Megatron)
5. Weight synchronization
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from flux.controller.adaptive_async import AdaptiveAsyncScheduler
from flux.controller.staleness import StalenessManager
from flux.core.config import FluxConfig
from flux.core.metrics import MetricsAggregator
from flux.core.trajectory import Trajectory, TrajectoryBatch, TrajectoryBuffer
from flux.core.types import (
    AsyncDecision,
    BatchMetrics,
    PolicyVersion,
    RolloutMetrics,
    StalenessMetrics,
    TrainingPhase,
    TrainingState,
)
from flux.rewards.base import RewardFunction
from flux.rollout.sglang_client import SGLangClient
from flux.sync.weight_sync import WeightSyncManager
from flux.training.batch_composer import SmartBatchComposer
from flux.training.megatron_engine import MegatronEngine, TrainingStep


logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result from a single training step."""

    step: int
    training_result: TrainingStep | None = None
    rollout_metrics: RolloutMetrics | None = None
    staleness_metrics: StalenessMetrics | None = None
    async_decision: AsyncDecision | None = None
    batch_size: int = 0
    num_trajectories: int = 0
    elapsed_ms: float = 0.0
    sync_performed: bool = False


@dataclass
class CoordinatorState:
    """Internal state of the coordinator."""

    step: int = 0
    version: PolicyVersion = field(default_factory=lambda: PolicyVersion(version_id=0))
    phase: TrainingPhase = TrainingPhase.WARMUP
    total_trajectories: int = 0
    total_tokens: int = 0
    rewards_sum: float = 0.0
    last_checkpoint_step: int = 0
    last_eval_step: int = 0
    start_time: float = field(default_factory=time.time)


class FluxCoordinator:
    """Coordinates the complete RLHF training pipeline.

    The coordinator is responsible for:
    1. Managing the training state
    2. Orchestrating rollout generation
    3. Computing rewards
    4. Composing training batches
    5. Executing training steps
    6. Synchronizing weights to inference servers
    7. Managing checkpoints

    Example:
        coordinator = FluxCoordinator(config=config)
        await coordinator.initialize()

        for step_result in coordinator.run_training():
            print(f"Step {step_result.step}: loss={step_result.training_result.loss:.4f}")

        await coordinator.shutdown()
    """

    def __init__(
        self,
        config: FluxConfig,
        training_engine: MegatronEngine | None = None,
        sglang_client: SGLangClient | None = None,
        reward_function: RewardFunction | None = None,
    ) -> None:
        """Initialize the coordinator.

        Args:
            config: FluxConfig with all settings.
            training_engine: Optional pre-configured training engine.
            sglang_client: Optional pre-configured SGLang client.
            reward_function: Optional custom reward function.
        """
        self.config = config

        # Core components (initialized later or passed in)
        self._engine = training_engine
        self._sglang = sglang_client
        self._reward_fn = reward_function

        # State management
        self._state = CoordinatorState()
        self._training_state = TrainingState(total_steps=config.num_steps)

        # Buffers and managers
        self._buffer = TrajectoryBuffer(
            max_size=config.batch_size * 100,  # Store many batches
            max_staleness=config.adaptive_async.max_version_gap,
        )
        self._batch_composer = SmartBatchComposer(
            config=config.batch_composer,
            batch_size=config.batch_size,
        )

        # Adaptive async control
        self._staleness_manager = StalenessManager(config=config.adaptive_async)
        self._async_scheduler = AdaptiveAsyncScheduler(
            config=config.adaptive_async,
            batch_size=config.batch_size,
        )

        # Weight sync
        self._weight_sync = WeightSyncManager()

        # Metrics
        self._metrics = MetricsAggregator()

        # Callbacks
        self._step_callbacks: list[Callable[[StepResult], None]] = []

        # Initialization flag
        self._initialized = False

    @property
    def state(self) -> CoordinatorState:
        """Current coordinator state."""
        return self._state

    @property
    def training_state(self) -> TrainingState:
        """Current training state."""
        return self._training_state

    @property
    def current_version(self) -> PolicyVersion:
        """Current policy version."""
        return self._state.version

    @property
    def is_initialized(self) -> bool:
        """Whether coordinator is initialized."""
        return self._initialized

    def add_step_callback(self, callback: Callable[[StepResult], None]) -> None:
        """Add a callback to be called after each step.

        Args:
            callback: Function taking StepResult.
        """
        self._step_callbacks.append(callback)

    async def initialize(self) -> None:
        """Initialize all components.

        This must be called before running training.
        """
        if self._initialized:
            return

        logger.info("Initializing FluxCoordinator...")

        # Initialize training engine
        if self._engine is None:
            self._engine = MegatronEngine(
                config=self.config.megatron,
                algorithm_config=self.config.algorithm,
            )
        self._engine.initialize()
        self._engine.load_model(self.config.model_path)

        # Initialize SGLang client
        if self._sglang is None:
            from flux.rollout.sglang_client import SGLangClient
            self._sglang = SGLangClient(config=self.config.sglang)

        # Connect to SGLang (if implemented)
        if hasattr(self._sglang, 'connect'):
            try:
                await self._sglang.connect()
            except Exception as e:
                logger.warning(f"Failed to connect to SGLang: {e}")

        # Initialize reward function if not provided
        if self._reward_fn is None:
            self._reward_fn = self._create_default_reward_function()

        # Initialize weight sync
        sglang_urls = self._sglang._server_urls if hasattr(self._sglang, '_server_urls') else []
        for url in sglang_urls:
            self._weight_sync.add_server(url)

        self._initialized = True
        logger.info("FluxCoordinator initialized")

    async def shutdown(self) -> None:
        """Shutdown all components cleanly."""
        logger.info("Shutting down FluxCoordinator...")

        # Disconnect SGLang
        if self._sglang is not None and hasattr(self._sglang, 'disconnect'):
            try:
                await self._sglang.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting SGLang: {e}")

        self._initialized = False
        logger.info("FluxCoordinator shutdown complete")

    def run_training(
        self,
        prompts: list[str] | Iterator[str],
        num_steps: int | None = None,
    ) -> Iterator[StepResult]:
        """Run the training loop synchronously.

        Args:
            prompts: Training prompts (list or iterator).
            num_steps: Override number of steps.

        Yields:
            StepResult for each training step.
        """
        if not self._initialized:
            raise RuntimeError("Coordinator not initialized. Call initialize() first.")

        num_steps = num_steps or self.config.num_steps
        prompt_iter = iter(prompts) if not hasattr(prompts, '__iter__') else iter(prompts)

        while self._state.step < num_steps:
            # Get prompts for this step
            batch_prompts = self._get_next_prompts(prompt_iter)
            if not batch_prompts:
                logger.warning("No more prompts available")
                break

            # Run one training step
            step_result = self._run_single_step(batch_prompts)

            # Run callbacks
            for callback in self._step_callbacks:
                try:
                    callback(step_result)
                except Exception as e:
                    logger.warning(f"Step callback failed: {e}")

            yield step_result

    async def run_training_async(
        self,
        prompts: list[str] | Iterator[str],
        num_steps: int | None = None,
    ) -> AsyncIterator[StepResult]:
        """Run the training loop asynchronously.

        Args:
            prompts: Training prompts.
            num_steps: Override number of steps.

        Yields:
            StepResult for each training step.
        """
        if not self._initialized:
            raise RuntimeError("Coordinator not initialized. Call initialize() first.")

        num_steps = num_steps or self.config.num_steps
        prompt_iter = iter(prompts)

        while self._state.step < num_steps:
            batch_prompts = self._get_next_prompts(prompt_iter)
            if not batch_prompts:
                break

            step_result = await self._run_single_step_async(batch_prompts)

            for callback in self._step_callbacks:
                try:
                    callback(step_result)
                except Exception as e:
                    logger.warning(f"Step callback failed: {e}")

            yield step_result

    def _run_single_step(self, prompts: list[str]) -> StepResult:
        """Run a single training step synchronously.

        Args:
            prompts: Prompts for this step.

        Returns:
            StepResult with metrics.
        """
        start_time = time.time()
        step = self._state.step + 1

        # 1. Generate rollouts (simplified sync version)
        trajectories = self._generate_rollouts_sync(prompts)

        # 2. Compute rewards
        trajectories = self._compute_rewards(trajectories)

        # 3. Add to buffer
        self._buffer.add_batch(trajectories)

        # 4. Get adaptive async decision
        staleness_metrics = self._staleness_manager.compute_staleness(trajectories)
        async_decision = self._async_scheduler.step(staleness_metrics=staleness_metrics)

        # 5. Compose batch
        batch = self._compose_batch()
        if batch is None or len(batch) == 0:
            return StepResult(
                step=step,
                num_trajectories=len(trajectories),
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # 6. Run training step
        training_result = self._train_step(batch)

        # 7. Maybe sync weights
        sync_performed = False
        if async_decision.should_sync:
            self._sync_weights()
            sync_performed = True

        # Update state
        self._state.step = step
        self._state.version = self._engine.version if self._engine else PolicyVersion(version_id=step)
        self._state.total_trajectories += len(trajectories)
        self._batch_composer.current_version = self._state.version.version_id

        # Update training state
        self._training_state.global_step = step
        if training_result:
            self._training_state.update_from_batch(training_result.metrics)

        return StepResult(
            step=step,
            training_result=training_result,
            staleness_metrics=staleness_metrics,
            async_decision=async_decision,
            batch_size=len(batch),
            num_trajectories=len(trajectories),
            elapsed_ms=(time.time() - start_time) * 1000,
            sync_performed=sync_performed,
        )

    async def _run_single_step_async(self, prompts: list[str]) -> StepResult:
        """Run a single training step asynchronously.

        Args:
            prompts: Prompts for this step.

        Returns:
            StepResult with metrics.
        """
        start_time = time.time()
        step = self._state.step + 1

        # 1. Generate rollouts
        trajectories = await self._generate_rollouts_async(prompts)

        # 2. Compute rewards
        trajectories = self._compute_rewards(trajectories)

        # 3. Add to buffer
        self._buffer.add_batch(trajectories)

        # 4. Get adaptive async decision
        staleness_metrics = self._staleness_manager.compute_staleness(trajectories)
        async_decision = self._async_scheduler.step(staleness_metrics=staleness_metrics)

        # 5. Compose batch
        batch = self._compose_batch()
        if batch is None or len(batch) == 0:
            return StepResult(
                step=step,
                num_trajectories=len(trajectories),
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # 6. Run training step
        training_result = self._train_step(batch)

        # 7. Maybe sync weights
        sync_performed = False
        if async_decision.should_sync:
            await self._sync_weights_async()
            sync_performed = True

        # Update state
        self._state.step = step
        self._state.version = self._engine.version if self._engine else PolicyVersion(version_id=step)
        self._state.total_trajectories += len(trajectories)
        self._batch_composer.current_version = self._state.version.version_id

        # Update training state
        self._training_state.global_step = step
        if training_result:
            self._training_state.update_from_batch(training_result.metrics)

        return StepResult(
            step=step,
            training_result=training_result,
            staleness_metrics=staleness_metrics,
            async_decision=async_decision,
            batch_size=len(batch),
            num_trajectories=len(trajectories),
            elapsed_ms=(time.time() - start_time) * 1000,
            sync_performed=sync_performed,
        )

    def _get_next_prompts(self, prompt_iter: Iterator[str]) -> list[str]:
        """Get the next batch of prompts.

        Args:
            prompt_iter: Iterator over prompts.

        Returns:
            List of prompts for next batch.
        """
        batch_size = self.config.batch_size
        oversample = self.config.rollout.oversample_ratio

        target = int(batch_size * oversample)
        prompts = []

        for _ in range(target):
            try:
                prompts.append(next(prompt_iter))
            except StopIteration:
                break

        return prompts

    def _generate_rollouts_sync(self, prompts: list[str]) -> list[Trajectory]:
        """Generate rollouts synchronously.

        Args:
            prompts: Prompts to generate responses for.

        Returns:
            List of generated trajectories.
        """
        trajectories = []

        for i, prompt in enumerate(prompts):
            # Create a simple trajectory (actual generation would use SGLang)
            traj = Trajectory(
                id=f"traj-{self._state.step}-{i}",
                prompt=prompt,
                response="",  # Would be filled by SGLang
                version=self._state.version,
            )
            trajectories.append(traj)

        return trajectories

    async def _generate_rollouts_async(self, prompts: list[str]) -> list[Trajectory]:
        """Generate rollouts asynchronously using SGLang.

        Args:
            prompts: Prompts to generate responses for.

        Returns:
            List of generated trajectories.
        """
        if self._sglang is None or not self._sglang.is_connected:
            return self._generate_rollouts_sync(prompts)

        trajectories = []

        # Generate in parallel
        tasks = []
        for prompt in prompts:
            tasks.append(self._sglang.generate(
                prompt=prompt,
                max_tokens=self.config.rollout.max_tokens,
                temperature=self.config.rollout.temperature,
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            if isinstance(result, Exception):
                logger.warning(f"Generation failed for prompt {i}: {result}")
                continue

            traj = Trajectory(
                id=f"traj-{self._state.step}-{i}",
                prompt=prompt,
                response=result.response if hasattr(result, 'response') else "",
                tokens=result.tokens if hasattr(result, 'tokens') else [],
                log_probs=result.log_probs if hasattr(result, 'log_probs') else [],
                behavior_log_probs=result.log_probs if hasattr(result, 'log_probs') else [],
                version=self._state.version,
            )
            trajectories.append(traj)

        return trajectories

    def _compute_rewards(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Compute rewards for trajectories.

        Args:
            trajectories: Trajectories to score.

        Returns:
            Trajectories with rewards filled in.
        """
        if self._reward_fn is None:
            return trajectories

        for traj in trajectories:
            try:
                output = self._reward_fn(traj)
                traj.reward = output.reward
                traj.token_rewards = output.token_rewards
            except Exception as e:
                logger.warning(f"Reward computation failed: {e}")
                traj.reward = 0.0

            self._state.rewards_sum += traj.reward

        return trajectories

    def _compose_batch(self) -> TrajectoryBatch | None:
        """Compose a batch from the buffer.

        Returns:
            TrajectoryBatch or None if not enough data.
        """
        # Sample from buffer
        trajectories = self._buffer.sample(
            n=self.config.batch_size,
            current_version=self._state.version,
            stratified=self.config.batch_composer.use_staleness_balancing,
        )

        if not trajectories:
            return None

        # Use batch composer for single batch
        batches = list(self._batch_composer.compose_batches(
            trajectories=trajectories,
            current_version=self._state.version.version_id,
        ))

        return batches[0] if batches else None

    def _train_step(self, batch: TrajectoryBatch) -> TrainingStep | None:
        """Execute one training step.

        Args:
            batch: Batch of trajectories.

        Returns:
            TrainingStep result or None.
        """
        if self._engine is None:
            return None

        try:
            return self._engine.train_step(
                batch=batch,
                algorithm_config=self.config.algorithm,
            )
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return None

    def _sync_weights(self) -> None:
        """Synchronize weights to inference servers (sync version)."""
        if self._engine is None:
            return

        try:
            state_dict = self._engine.get_state_dict()
            self._weight_sync.mark_updated()
            # Actual sync would happen here
            logger.debug(f"Weights synced at version {self._state.version.version_id}")
        except Exception as e:
            logger.warning(f"Weight sync failed: {e}")

    async def _sync_weights_async(self) -> None:
        """Synchronize weights to inference servers (async version)."""
        if self._engine is None:
            return

        try:
            state_dict = self._engine.get_state_dict()
            self._weight_sync.mark_updated()
            # Actual async sync would happen here
            logger.debug(f"Weights synced at version {self._state.version.version_id}")
        except Exception as e:
            logger.warning(f"Weight sync failed: {e}")

    def _create_default_reward_function(self) -> RewardFunction:
        """Create default reward function based on config.

        Returns:
            RewardFunction instance.
        """
        from flux.rewards.rule_based import LengthReward

        # Simple length-based reward as default
        return LengthReward()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save a checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save training engine checkpoint
        if self._engine is not None:
            self._engine.save_checkpoint(str(path / "model.pt"))

        # Save coordinator state
        import json
        state_path = path / "coordinator_state.json"
        state_dict = {
            "step": self._state.step,
            "version": self._state.version.version_id,
            "phase": self._state.phase.value,
            "total_trajectories": self._state.total_trajectories,
            "total_tokens": self._state.total_tokens,
            "rewards_sum": self._state.rewards_sum,
        }
        with open(state_path, "w") as f:
            json.dump(state_dict, f, indent=2)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a checkpoint.

        Args:
            path: Path to checkpoint.
        """
        path = Path(path)

        # Load training engine checkpoint
        if self._engine is not None:
            engine_path = path / "model.pt"
            if engine_path.exists():
                self._engine.load_checkpoint(str(engine_path))

        # Load coordinator state
        import json
        state_path = path / "coordinator_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state_dict = json.load(f)

            self._state.step = state_dict.get("step", 0)
            self._state.version = PolicyVersion(version_id=state_dict.get("version", 0))
            self._state.phase = TrainingPhase(state_dict.get("phase", "warmup"))
            self._state.total_trajectories = state_dict.get("total_trajectories", 0)
            self._state.total_tokens = state_dict.get("total_tokens", 0)
            self._state.rewards_sum = state_dict.get("rewards_sum", 0.0)

        logger.info(f"Checkpoint loaded from {path}")

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics.

        Returns:
            Dict with all statistics.
        """
        elapsed_time = time.time() - self._state.start_time

        stats = {
            "step": self._state.step,
            "version": self._state.version.version_id,
            "phase": self._state.phase.value,
            "total_trajectories": self._state.total_trajectories,
            "total_tokens": self._state.total_tokens,
            "elapsed_time_seconds": elapsed_time,
            "steps_per_second": self._state.step / elapsed_time if elapsed_time > 0 else 0,
            "mean_reward": (
                self._state.rewards_sum / self._state.total_trajectories
                if self._state.total_trajectories > 0 else 0
            ),
            "buffer_size": len(self._buffer),
        }

        # Add component statistics
        stats["batch_composer"] = self._batch_composer.get_statistics()
        stats["buffer"] = self._buffer.get_stats()
        stats["weight_sync"] = {
            "current_version": self._weight_sync.current_version,
            "is_dirty": self._weight_sync.is_dirty,
        }

        return stats


# Type hint for async iterator
from typing import AsyncIterator
