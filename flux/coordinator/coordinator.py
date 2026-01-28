"""
FluxCoordinator - Orchestrates the complete RLHF training loop.

The coordinator manages the flow of data between:
1. Rollout generation (SGLang via StreamingRolloutManager with APRIL)
2. Reward computation
3. Batch composition
4. Training (Megatron)
5. Weight synchronization (via SGLangClient.update_weights)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

import torch

from flux.controller.adaptive_async import AdaptiveAsyncScheduler
from flux.controller.mode_gate import AsyncMode, ModeGate, ModeGateConfig
from flux.coordinator.communication import CommunicationManager
from flux.core.config import FluxConfig
from flux.core.metrics import MetricsAggregator
from flux.core.trajectory import Trajectory, TrajectoryBatch, TrajectoryBuffer
from flux.core.types import (
    AsyncDecision,
    PolicyVersion,
    RolloutMetrics,
    StalenessMetrics,
    TrainingPhase,
    TrainingState,
)
from flux.rewards.base import RewardFunction
from flux.rollout.manager import StreamingRolloutManager
from flux.rollout.sglang_client import SGLangClient
from flux.sync.weight_sync import WeightSyncManager
from flux.training.base import TrainingBackend, TrainingBackendBase, create_training_backend
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
        training_engine: MegatronEngine | TrainingBackendBase | None = None,
        sglang_client: SGLangClient | None = None,
        reward_function: RewardFunction | None = None,
    ) -> None:
        """Initialize the coordinator.

        Args:
            config: FluxConfig with all settings.
            training_engine: Optional pre-configured training engine (MegatronEngine or TrainingBackend).
            sglang_client: Optional pre-configured SGLang client.
            reward_function: Optional custom reward function.
        """
        self.config = config

        # Core components (initialized later or passed in)
        self._engine: MegatronEngine | TrainingBackendBase | None = training_engine
        self._sglang = sglang_client
        self._reward_fn = reward_function
        self._rollout_manager: StreamingRolloutManager | None = None

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
        version_provider = lambda: self._state.version.version_id
        self._async_scheduler = AdaptiveAsyncScheduler(
            config=config.adaptive_async,
            batch_size=config.batch_size,
            version_provider=version_provider,
        )
        self._staleness_manager = self._async_scheduler.staleness_manager

        # Mode gate for sync/async transitions
        self._mode_gate = ModeGate(
            config=ModeGateConfig(
                staleness_threshold=config.adaptive_async.staleness_threshold
                if hasattr(config.adaptive_async, 'staleness_threshold')
                else 0.3,
                capacity_low_watermark=config.adaptive_async.capacity_low_watermark
                if hasattr(config.adaptive_async, 'capacity_low_watermark')
                else 0,
                buffer_high_watermark=config.adaptive_async.buffer_high_watermark
                if hasattr(config.adaptive_async, 'buffer_high_watermark')
                else 0.9,
            ),
            staleness_manager=self._staleness_manager,
        )

        # Weight sync
        self._weight_sync = WeightSyncManager()

        # Metrics
        self._metrics = MetricsAggregator()

        # Callbacks
        self._step_callbacks: list[Callable[[StepResult], None]] = []

        # Communication layer (ZeroMQ/gRPC)
        self._comm_manager: CommunicationManager | None = None

        # Initialization flag
        self._initialized = False

        # Flag for using real rollouts vs stub (for testing)
        self._use_real_rollouts = config.rollout.use_real_rollouts if hasattr(config.rollout, 'use_real_rollouts') else True

        # Dedicated event loop for async operations (to avoid conflicts with notebooks)
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None

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

    def _ensure_async_loop(self) -> asyncio.AbstractEventLoop:
        """Ensure a dedicated event loop is running for async operations.

        This avoids conflicts with notebook event loops by running our
        async operations on a separate thread with its own loop.

        Returns:
            The dedicated event loop.
        """
        if self._async_loop is not None and self._async_loop.is_running():
            return self._async_loop

        def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=_run_loop,
            args=(self._async_loop,),
            daemon=True,
            name="flux-async-loop",
        )
        self._async_thread.start()
        return self._async_loop

    def _run_async(self, coro) -> Any:
        """Run a coroutine, handling both sync and async contexts.

        Uses a dedicated event loop on a background thread to avoid
        conflicts with notebook/REPL event loops.

        Args:
            coro: Coroutine to run.

        Returns:
            Result from the coroutine.
        """
        bg_loop = self._ensure_async_loop()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is bg_loop:
            raise RuntimeError("Cannot call _run_async from the coordinator async loop")

        future = asyncio.run_coroutine_threadsafe(coro, bg_loop)
        return future.result()

    async def _await_async(self, coro) -> Any:
        """Await a coroutine on the dedicated async loop when needed."""
        bg_loop = self._ensure_async_loop()
        running = asyncio.get_running_loop()

        if running is bg_loop:
            return await coro

        future = asyncio.run_coroutine_threadsafe(coro, bg_loop)
        return await asyncio.wrap_future(future)

    def _stop_async_loop(self) -> None:
        """Stop the dedicated async loop."""
        if self._async_loop is not None and self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            if self._async_thread is not None and threading.current_thread() is not self._async_thread:
                self._async_thread.join(timeout=5.0)
        self._async_loop = None
        self._async_thread = None

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
        await self._await_async(self._initialize_impl())

    async def _initialize_impl(self) -> None:
        """Initialize components on the dedicated async loop."""
        if self._initialized:
            return

        logger.info("Initializing FluxCoordinator...")

        # Initialize training engine based on config.training_backend
        if self._engine is None:
            from flux.core.config import TrainingBackendType
            backend_type = self.config.training_backend

            if backend_type == TrainingBackendType.MEGATRON:
                # Use MegatronEngine directly (legacy path)
                self._engine = MegatronEngine(
                    config=self.config.megatron,
                    algorithm_config=self.config.algorithm,
                )
                self._engine.initialize()
                self._engine.load_model(self.config.model_path)
            else:
                # Use TrainingBackend ABC via factory
                self._engine = create_training_backend(self.config)
                self._engine.initialize(self.config)
        elif isinstance(self._engine, MegatronEngine):
            self._engine.initialize()
            self._engine.load_model(self.config.model_path)
        else:
            # TrainingBackendBase instance passed in
            if not self._engine.is_initialized:
                self._engine.initialize(self.config)

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

        # Initialize StreamingRolloutManager with APRIL strategy
        version_provider = lambda: self._state.version
        self._rollout_manager = StreamingRolloutManager(
            client=self._sglang,
            config=self.config.rollout,
            trajectory_buffer=self._buffer,
            version_provider=version_provider,
        )
        logger.info("StreamingRolloutManager initialized with APRIL strategy")

        # Initialize reward function if not provided
        if self._reward_fn is None:
            self._reward_fn = self._create_default_reward_function()

        # Initialize weight sync with SGLang servers
        sglang_urls = self._sglang._server_urls if hasattr(self._sglang, '_server_urls') else []
        for url in sglang_urls:
            self._weight_sync.add_server(url)

        # Initialize ZeroMQ/gRPC communication layer (optional, for distributed workers)
        comm_config = self.config.communication
        try:
            self._comm_manager = CommunicationManager(
                use_zmq=comm_config.use_zmq,
                zmq_router_addr=comm_config.zmq_router_addr,
                zmq_pub_addr=comm_config.zmq_pub_addr,
                identity=comm_config.identity,
            )
            await self._comm_manager.start()
            logger.info("Communication manager initialized (ZMQ: %s)", self._comm_manager.use_zmq)
        except Exception as e:
            logger.warning(f"Failed to initialize communication manager: {e}")
            self._comm_manager = None

        self._initialized = True
        logger.info("FluxCoordinator initialized")

    async def shutdown(self) -> None:
        """Shutdown all components cleanly."""
        await self._await_async(self._shutdown_impl())

    async def _shutdown_impl(self) -> None:
        """Shutdown components on the dedicated async loop."""
        logger.info("Shutting down FluxCoordinator...")

        # Stop communication manager
        if self._comm_manager is not None:
            try:
                await self._comm_manager.stop()
            except Exception as e:
                logger.warning(f"Error stopping communication manager: {e}")
            self._comm_manager = None

        # Disconnect SGLang
        if self._sglang is not None and hasattr(self._sglang, 'close'):
            try:
                await self._sglang.close()
            except Exception as e:
                logger.warning(f"Error closing SGLang: {e}")

        # Stop the dedicated async loop
        self._stop_async_loop()

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
            # Get current async_ratio for scheduling
            current_async_ratio = self._async_scheduler.async_ratio

            # Get prompts for this step (scaled by async_ratio)
            batch_prompts = self._get_next_prompts(prompt_iter, async_ratio=current_async_ratio)
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
            # Get current async_ratio for scheduling
            current_async_ratio = self._async_scheduler.async_ratio

            # Get prompts for this step (scaled by async_ratio)
            batch_prompts = self._get_next_prompts(prompt_iter, async_ratio=current_async_ratio)
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
        staleness_metrics = self._compute_staleness_metrics(trajectories)
        async_decision = self._async_scheduler.step(staleness_metrics=staleness_metrics)

        # 4b. Evaluate ModeGate state based on current metrics
        buffer_fill_ratio = len(self._buffer) / self._buffer.max_size if self._buffer.max_size > 0 else 0.0
        mode_state = self._mode_gate.evaluate(
            staleness=staleness_metrics.combined_staleness,
            capacity=self._async_scheduler.get_capacity() or 0,
            buffer_fill_ratio=buffer_fill_ratio,
            in_flight=self._staleness_manager.stats.total_in_flight,
        )

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

        # Track batch consumption for capacity throttling
        if training_result is not None:
            self._async_scheduler.on_batch_consumed(len(batch))

        # 7. Maybe sync weights (check both async_decision and ModeGate)
        sync_performed = False
        if async_decision.should_sync or mode_state.mode == AsyncMode.SYNC_BARRIER:
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
        staleness_metrics = self._compute_staleness_metrics(trajectories)
        async_decision = self._async_scheduler.step(staleness_metrics=staleness_metrics)

        # 4b. Evaluate ModeGate state based on current metrics
        buffer_fill_ratio = len(self._buffer) / self._buffer.max_size if self._buffer.max_size > 0 else 0.0
        mode_state = self._mode_gate.evaluate(
            staleness=staleness_metrics.combined_staleness,
            capacity=self._async_scheduler.get_capacity() or 0,
            buffer_fill_ratio=buffer_fill_ratio,
            in_flight=self._staleness_manager.stats.total_in_flight,
        )

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

        # Track batch consumption for capacity throttling
        if training_result is not None:
            self._async_scheduler.on_batch_consumed(len(batch))

        # 7. Maybe sync weights (check both async_decision and ModeGate)
        sync_performed = False
        if async_decision.should_sync or mode_state.mode == AsyncMode.SYNC_BARRIER:
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

    def _get_next_prompts(
        self,
        prompt_iter: Iterator[str],
        async_ratio: float | None = None,
    ) -> list[str]:
        """Get the next batch of prompts, scaled by async_ratio and capacity.

        Uses adaptive async scheduling and ModeGate to control rollout submission:
        - Higher async_ratio = more aggressive oversampling
        - Respects capacity limits to prevent excessive staleness
        - Respects ModeGate state (no rollouts in SYNC_BARRIER or THROTTLED modes)

        Args:
            prompt_iter: Iterator over prompts.
            async_ratio: Current async ratio from adaptive scheduler (0-1).

        Returns:
            List of prompts for next batch.
        """
        batch_size = self.config.batch_size
        base_oversample = self.config.rollout.oversample_ratio

        # Check ModeGate - don't submit if in SYNC_BARRIER or THROTTLED
        if not self._mode_gate.can_submit_rollout():
            # Still need to make progress, submit minimum batch
            logger.debug(
                f"ModeGate blocking rollouts (mode={self._mode_gate.current_mode.name}), "
                f"submitting minimum batch"
            )
            target = batch_size
        else:
            # Scale oversampling by async_ratio
            # High async_ratio (close to 1) = more aggressive oversampling
            # Low async_ratio (close to 0) = conservative, closer to batch_size
            if async_ratio is not None:
                # Interpolate between 1.0 (no oversample) and full oversample
                scaled_oversample = 1.0 + (base_oversample - 1.0) * async_ratio
            else:
                scaled_oversample = base_oversample

            target = int(batch_size * scaled_oversample)

        # Check capacity - don't submit if buffer is too full/stale
        capacity = self._async_scheduler.get_capacity()
        if capacity is not None and capacity > 0:
            # Limit to available capacity
            target = min(target, capacity)
        elif capacity == 0:
            # No capacity - still submit minimum batch for progress
            target = min(target, batch_size)

        prompts = []
        for _ in range(target):
            try:
                prompts.append(next(prompt_iter))
            except StopIteration:
                break

        # Track enqueued prompts for capacity throttling
        if prompts:
            self._async_scheduler.on_rollout_enqueued(len(prompts))

        return prompts

    def _generate_rollouts_sync(self, prompts: list[str]) -> list[Trajectory]:
        """Generate rollouts synchronously.

        Uses the dedicated async loop to run async rollout generation.
        Falls back to stub if real rollouts not available.

        Args:
            prompts: Prompts to generate responses for.

        Returns:
            List of generated trajectories.
        """
        # Mark as submitted (moved from enqueued to running)
        if prompts:
            self._async_scheduler.on_rollout_submitted(len(prompts))

        if not self._use_real_rollouts or self._rollout_manager is None:
            trajectories = self._generate_rollouts_stub(prompts)
        else:
            trajectories = self._run_async(self._generate_rollouts_async_impl(prompts))

        # Track completed/failed rollouts
        completed = len(trajectories)
        failed = len(prompts) - completed
        if completed > 0:
            self._async_scheduler.on_rollout_completed(completed)
        if failed > 0:
            self._async_scheduler.on_rollout_failed(failed)

        return trajectories

    def _generate_rollouts_stub(self, prompts: list[str]) -> list[Trajectory]:
        """Generate stub trajectories for testing.

        Args:
            prompts: Prompts to generate responses for.

        Returns:
            List of stub trajectories.
        """
        trajectories = []
        for i, prompt in enumerate(prompts):
            traj = Trajectory(
                id=f"traj-{self._state.step}-{i}",
                prompt=prompt,
                response="",
                version=self._state.version,
            )
            trajectories.append(traj)
        return trajectories

    async def _generate_rollouts_async(self, prompts: list[str]) -> list[Trajectory]:
        """Generate rollouts asynchronously using the dedicated async loop."""
        # Mark as submitted (moved from enqueued to running)
        if prompts:
            self._async_scheduler.on_rollout_submitted(len(prompts))

        trajectories = await self._await_async(self._generate_rollouts_async_impl(prompts))

        # Track completed/failed rollouts
        completed = len(trajectories)
        failed = len(prompts) - completed
        if completed > 0:
            self._async_scheduler.on_rollout_completed(completed)
        if failed > 0:
            self._async_scheduler.on_rollout_failed(failed)

        return trajectories

    async def _generate_rollouts_async_impl(self, prompts: list[str]) -> list[Trajectory]:
        """Generate rollouts asynchronously using StreamingRolloutManager.

        Uses APRIL strategy: oversample, abort long-tail, partial reuse.

        Args:
            prompts: Prompts to generate responses for.

        Returns:
            List of generated trajectories.
        """
        if not self._use_real_rollouts or self._rollout_manager is None:
            return self._generate_rollouts_stub(prompts)

        if self._sglang is None or not self._sglang.is_connected:
            logger.warning("SGLang not connected, using stub rollouts")
            return self._generate_rollouts_stub(prompts)

        try:
            # Use StreamingRolloutManager with APRIL strategy
            batch = await self._rollout_manager.generate_batch(
                prompts=prompts,
                target_count=self.config.batch_size,
                timeout=self.config.rollout.batch_timeout,
            )

            # Log rollout metrics
            logger.debug(
                f"Rollout batch: {batch.completed} completed, "
                f"{batch.aborted} aborted, {batch.reused} reused, "
                f"success_rate={batch.success_rate:.2%}"
            )

            return batch.trajectories

        except Exception as e:
            logger.error(f"Rollout generation failed: {e}")
            return self._generate_rollouts_stub(prompts)

    def _compute_rewards(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Compute rewards for trajectories.

        Args:
            trajectories: Trajectories to score.

        Returns:
            Trajectories with rewards filled in.
        """
        if self._reward_fn is None:
            return trajectories

        failed_count = 0
        for traj in trajectories:
            try:
                output = self._reward_fn(traj)
                traj.reward = output.reward
                traj.token_rewards = output.token_rewards
            except Exception as e:
                failed_count += 1
                logger.warning(
                    f"Reward computation failed for trajectory {traj.id}: {e}. "
                    "Setting reward to 0.0 (this may affect training stability)."
                )
                traj.reward = 0.0
                # Store failure info in metadata for debugging
                traj.metadata["reward_computation_failed"] = True
                traj.metadata["reward_computation_error"] = str(e)

            self._state.rewards_sum += traj.reward

        if failed_count > 0:
            failure_rate = failed_count / len(trajectories)
            if failure_rate > 0.1:  # More than 10% failures
                logger.error(
                    f"High reward computation failure rate: {failure_rate:.1%} "
                    f"({failed_count}/{len(trajectories)} trajectories). "
                    "Check reward function configuration."
                )

        return trajectories

    def _compute_staleness_metrics(
        self, trajectories: list[Trajectory]
    ) -> StalenessMetrics:
        """Compute staleness metrics from trajectory versions and log probs.

        Uses full staleness computation including:
        - KL divergence between current and behavior policies
        - Importance weight variance
        - Version gap

        Falls back to version-gap-only computation if:
        - Log probs unavailable
        - log_probs == behavior_log_probs (would give meaningless 0 KL/IW)
        """
        if not trajectories:
            return self._staleness_manager.compute_staleness(version_gap=0.0)

        # Try to compute full staleness with KL and IW variance
        # This requires both behavior and current log probs, and they must differ
        has_behavior_logprobs = all(
            traj.behavior_log_probs is not None and len(traj.behavior_log_probs) > 0
            for traj in trajectories
        )
        has_current_logprobs = all(
            traj.log_probs is not None and len(traj.log_probs) > 0
            for traj in trajectories
        )

        # Check if log_probs differ from behavior_log_probs
        # If they're the same, KL and IW variance would be 0 (meaningless)
        has_distinct_logprobs = False
        if has_behavior_logprobs and has_current_logprobs:
            for traj in trajectories:
                if traj.log_probs != traj.behavior_log_probs:
                    has_distinct_logprobs = True
                    break

        if has_behavior_logprobs and has_current_logprobs and has_distinct_logprobs:
            try:
                # Collect log probs and versions
                current_logprobs_list = []
                behavior_logprobs_list = []
                versions = []
                masks = []

                for traj in trajectories:
                    if traj.log_probs is not None and traj.behavior_log_probs is not None:
                        # Convert to tensors if needed
                        current_lp = traj.log_probs
                        behavior_lp = traj.behavior_log_probs

                        if isinstance(current_lp, list):
                            current_lp = torch.tensor(current_lp)
                        if isinstance(behavior_lp, list):
                            behavior_lp = torch.tensor(behavior_lp)

                        current_logprobs_list.append(current_lp)
                        behavior_logprobs_list.append(behavior_lp)
                        versions.append(traj.version.version_id)

                        # Use loss_mask if available, else ones
                        if hasattr(traj, 'loss_mask') and traj.loss_mask is not None:
                            mask = traj.loss_mask
                            if isinstance(mask, list):
                                mask = torch.tensor(mask, dtype=torch.float)
                            masks.append(mask)
                        else:
                            masks.append(torch.ones_like(current_lp))

                if current_logprobs_list:
                    # Pad sequences to same length for batch processing
                    max_len = max(lp.shape[-1] for lp in current_logprobs_list)

                    def pad_to_length(tensor: torch.Tensor, length: int) -> torch.Tensor:
                        if tensor.shape[-1] >= length:
                            return tensor[..., :length]
                        pad_size = length - tensor.shape[-1]
                        return torch.nn.functional.pad(tensor, (0, pad_size), value=0.0)

                    current_batch = torch.stack([
                        pad_to_length(lp, max_len) for lp in current_logprobs_list
                    ])
                    behavior_batch = torch.stack([
                        pad_to_length(lp, max_len) for lp in behavior_logprobs_list
                    ])
                    mask_batch = torch.stack([
                        pad_to_length(m, max_len) for m in masks
                    ])

                    # Compute full staleness with KL and IW variance
                    return self._staleness_manager.compute_staleness_from_trajectories(
                        current_logprobs=current_batch,
                        behavior_logprobs=behavior_batch,
                        trajectory_versions=versions,
                        mask=mask_batch,
                    )

            except Exception as e:
                logger.warning(f"Failed to compute full staleness, falling back to version gap: {e}")

        # Fallback: version-gap only computation
        gaps = [
            traj.get_version_gap(self._state.version) for traj in trajectories
        ]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        return self._staleness_manager.compute_staleness(version_gap=avg_gap)

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
        """Synchronize weights to inference servers (sync version).

        Uses the dedicated async loop to call the async version.
        """
        if self._engine is None:
            return

        self._run_async(self._sync_weights_async_impl())

    async def _sync_weights_async(self) -> None:
        """Synchronize weights to inference servers via the dedicated async loop."""
        await self._await_async(self._sync_weights_async_impl())

    async def _sync_weights_async_impl(self) -> None:
        """Synchronize weights to inference servers via SGLangClient.update_weights.

        Uses centralized WeightSyncManager.serialize_for_sync() which handles:
        - "full": Complete state_dict serialized to bytes
        - "delta": Only changed parameters since baseline
        - "per_tensor": Individual tensors with metadata

        All payload formats are serialized to bytes for HTTP transport.
        """
        if self._engine is None:
            return

        if self._sglang is None:
            logger.warning("SGLang client not available for weight sync")
            return

        try:
            # Get state dict from training engine
            state_dict = self._engine.get_state_dict()
            version = self._state.version.version_id

            # Use centralized serialization from WeightSyncManager
            # This handles method selection, delta compression, quantization
            payload_bytes = self._weight_sync.serialize_for_sync(
                weights=state_dict,
                version=version,
            )

            # Determine sync method
            method = getattr(self.config.weight_sync, 'method', 'full')
            if hasattr(method, 'value'):
                method = method.value

            # Push serialized bytes to SGLang server(s) via HTTP
            http_success = False
            if hasattr(self._sglang, 'update_weights'):
                http_success = await self._sglang.update_weights(
                    weights=payload_bytes,
                    version=version,
                )
                if http_success:
                    self._weight_sync.mark_updated()
                    if method == "delta":
                        self._weight_sync.set_baseline(state_dict, version)
                    logger.info(
                        f"Weights synced to SGLang at version {version} "
                        f"(method={method}, {len(payload_bytes)} bytes)"
                    )
                else:
                    logger.warning(f"Weight sync to SGLang returned False at version {version}")
            else:
                # Fallback for testing without full SGLang
                self._weight_sync.mark_updated()
                http_success = True
                logger.debug(f"Weight sync marked (SGLang update_weights not available) at version {version}")

            # Also broadcast via ZeroMQ if communication manager is available
            # This enables distributed workers to receive weight updates
            if self._comm_manager is not None and http_success:
                try:
                    await self._comm_manager.broadcast_weights(
                        weights=state_dict,
                        version=version,
                    )
                    logger.debug(f"Weight update broadcast via ZMQ at version {version}")
                except Exception as zmq_err:
                    logger.warning(f"ZMQ weight broadcast failed: {zmq_err}")

        except Exception as e:
            logger.error(f"Weight sync failed: {e}")

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
        stats["mode_gate"] = self._mode_gate.get_diagnostics()

        return stats
