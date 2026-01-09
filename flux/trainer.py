"""
Main FluxTrainer class for RLHF training.

This module provides the high-level training API for Flux.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Union

from flux.coordinator.coordinator import FluxCoordinator, StepResult
from flux.core.config import FluxConfig
from flux.core.metrics import MetricsAggregator, MetricsLogger
from flux.core.trajectory import Trajectory, TrajectoryBuffer
from flux.core.types import BatchMetrics, TrainingState
from flux.rewards.base import RewardFunction
from flux.rollout.sglang_client import SGLangClient
from flux.training.megatron_engine import MegatronEngine


logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result from a training run."""

    # Final metrics
    final_reward: float = 0.0
    final_loss: float = 0.0
    total_steps: int = 0
    total_samples: int = 0
    total_tokens: int = 0

    # Paths
    checkpoint_path: str | None = None
    log_dir: str | None = None

    # History
    reward_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)

    # Timing
    total_time_seconds: float = 0.0
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_reward": self.final_reward,
            "final_loss": self.final_loss,
            "total_steps": self.total_steps,
            "total_samples": self.total_samples,
            "total_tokens": self.total_tokens,
            "checkpoint_path": self.checkpoint_path,
            "total_time_seconds": self.total_time_seconds,
            "samples_per_second": self.samples_per_second,
            "tokens_per_second": self.tokens_per_second,
        }


# Type alias for prompts
PromptType = Union[str, dict[str, Any]]
PromptsType = Union[list[PromptType], Any]  # Any for Dataset-like objects

# Type alias for callbacks
CallbackType = Callable[[TrainingState, BatchMetrics], None]
StepCallbackType = Callable[[StepResult], None]


class FluxTrainer:
    """Main entry point for Flux training.

    FluxTrainer orchestrates the complete RLHF training pipeline:
    1. Rollout generation via SGLang
    2. Reward computation
    3. Advantage estimation
    4. Policy optimization via Megatron
    5. Weight synchronization

    Example:
        >>> from flux import FluxTrainer, FluxConfig
        >>> config = FluxConfig(model_path="Qwen/Qwen3-8B")
        >>> trainer = FluxTrainer(config)
        >>> result = trainer.fit(prompts, num_steps=10000)

    For more control:
        >>> trainer = FluxTrainer(config)
        >>> trainer.setup()  # Initialize components
        >>>
        >>> for step_result in trainer.training_loop(prompts):
        >>>     print(f"Step {step_result.step}: loss={step_result.training_result.loss:.4f}")
        >>>     if step_result.step % 1000 == 0:
        >>>         trainer.save_checkpoint(f"checkpoint-{step_result.step}")
        >>>
        >>> trainer.teardown()
    """

    def __init__(
        self,
        config: FluxConfig,
        training_engine: MegatronEngine | None = None,
        sglang_client: SGLangClient | None = None,
        reward_function: RewardFunction | None = None,
    ) -> None:
        """Initialize FluxTrainer.

        Args:
            config: FluxConfig with all training settings.
            training_engine: Optional pre-configured training engine.
            sglang_client: Optional pre-configured SGLang client.
            reward_function: Optional custom reward function.
        """
        self.config = config
        self._state = TrainingState(total_steps=config.num_steps)
        self._buffer = TrajectoryBuffer()
        self._logger: MetricsLogger | None = None
        self._metrics = MetricsAggregator()
        self._callbacks: list[CallbackType] = []
        self._step_callbacks: list[StepCallbackType] = []

        # External components (can be passed in)
        self._training_engine = training_engine
        self._sglang_client = sglang_client
        self._reward_function = reward_function

        # Coordinator (created in setup)
        self._coordinator: FluxCoordinator | None = None

        # Initialization flag
        self._initialized = False

    @property
    def state(self) -> TrainingState:
        """Current training state."""
        return self._state

    @property
    def coordinator(self) -> FluxCoordinator | None:
        """Get the underlying coordinator."""
        return self._coordinator

    @property
    def is_initialized(self) -> bool:
        """Whether trainer is set up and ready."""
        return self._initialized

    def add_callback(self, callback: CallbackType) -> None:
        """Add a callback to be called after each training step.

        Args:
            callback: Function taking (TrainingState, BatchMetrics) as arguments.
        """
        self._callbacks.append(callback)

    def add_step_callback(self, callback: StepCallbackType) -> None:
        """Add a callback for step-level events.

        Args:
            callback: Function taking StepResult.
        """
        self._step_callbacks.append(callback)

    def setup(self) -> None:
        """Initialize all components for training.

        Must be called before fit() or training_loop().
        Can be called explicitly for more control over initialization timing.
        """
        if self._initialized:
            return

        logger.info("Setting up FluxTrainer...")

        # Create coordinator
        self._coordinator = FluxCoordinator(
            config=self.config,
            training_engine=self._training_engine,
            sglang_client=self._sglang_client,
            reward_function=self._reward_function,
        )

        # Initialize coordinator (sync wrapper for async init)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(self._coordinator.initialize())

        # Add step callbacks to coordinator
        for callback in self._step_callbacks:
            self._coordinator.add_step_callback(callback)

        # Initialize logger
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._logger = MetricsLogger(
            console=True,
            console_interval=self.config.log_interval,
            tensorboard_dir=str(output_dir / "tensorboard"),
            wandb_project=self.config.wandb_project,
            wandb_run_name=self.config.wandb_run_name,
            wandb_config=self.config.to_dict(),
        )

        self._initialized = True
        logger.info("FluxTrainer setup complete")

    def teardown(self) -> None:
        """Shutdown all components.

        Should be called after training is complete.
        """
        if not self._initialized:
            return

        logger.info("Tearing down FluxTrainer...")

        if self._coordinator is not None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(self._coordinator.shutdown())
            self._coordinator = None

        if self._logger is not None:
            self._logger.close()
            self._logger = None

        self._initialized = False
        logger.info("FluxTrainer teardown complete")

    def fit(
        self,
        prompts: PromptsType,
        num_steps: int | None = None,
        eval_prompts: PromptsType | None = None,
        eval_interval: int = 500,
        checkpoint_interval: int = 1000,
        callbacks: list[CallbackType] | None = None,
    ) -> TrainingResult:
        """Run the training loop.

        Args:
            prompts: Training prompts (list of strings or Dataset).
            num_steps: Total training steps (overrides config if provided).
            eval_prompts: Optional evaluation prompts.
            eval_interval: Steps between evaluations.
            checkpoint_interval: Steps between checkpoints.
            callbacks: Optional list of callbacks.

        Returns:
            TrainingResult with final metrics and paths.
        """
        # Update state if num_steps provided
        if num_steps is not None:
            self._state.total_steps = num_steps

        # Add callbacks
        if callbacks:
            for cb in callbacks:
                self.add_callback(cb)

        # Setup if not already done
        if not self._initialized:
            self.setup()

        # Convert prompts to list if needed
        prompt_list = self._prepare_prompts(prompts)
        eval_list = self._prepare_prompts(eval_prompts) if eval_prompts else None

        # Run training loop
        start_time = time.time()
        result = TrainingResult()
        result.log_dir = str(Path(self.config.output_dir) / "tensorboard")

        try:
            for step_result in self.training_loop(prompt_list, num_steps):
                # Update result
                result.total_steps = step_result.step
                result.total_samples += step_result.num_trajectories

                if step_result.training_result:
                    result.loss_history.append(step_result.training_result.loss)
                    result.final_loss = step_result.training_result.loss

                # Log metrics
                if self._logger and step_result.step % self.config.log_interval == 0:
                    self._log_step(step_result)

                # Run legacy callbacks
                if step_result.training_result:
                    self._run_callbacks(step_result.training_result.metrics)

                # Checkpoint
                if step_result.step % checkpoint_interval == 0:
                    checkpoint_path = self._save_checkpoint(step_result.step)
                    result.checkpoint_path = checkpoint_path

                # Evaluation
                if eval_list and step_result.step % eval_interval == 0:
                    eval_metrics = self.evaluate(eval_list)
                    result.reward_history.append(eval_metrics.get("mean_reward", 0.0))
                    result.final_reward = eval_metrics.get("mean_reward", 0.0)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            # Finalize result
            result.total_time_seconds = time.time() - start_time
            if result.total_time_seconds > 0:
                result.samples_per_second = (
                    result.total_samples / result.total_time_seconds
                )

            # Final checkpoint
            if result.total_steps > 0:
                result.checkpoint_path = self._save_checkpoint(result.total_steps)

            # Teardown
            self.teardown()

        return result

    def training_loop(
        self,
        prompts: list[str],
        num_steps: int | None = None,
    ) -> Iterator[StepResult]:
        """Run the training loop, yielding results for each step.

        This provides more control than fit() for custom training loops.

        Args:
            prompts: List of training prompts.
            num_steps: Number of steps to run.

        Yields:
            StepResult for each training step.
        """
        if self._coordinator is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        yield from self._coordinator.run_training(
            prompts=prompts,
            num_steps=num_steps or self._state.total_steps,
        )

    async def training_loop_async(
        self,
        prompts: list[str],
        num_steps: int | None = None,
    ):
        """Async version of training_loop.

        Args:
            prompts: List of training prompts.
            num_steps: Number of steps to run.

        Yields:
            StepResult for each training step.
        """
        if self._coordinator is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        async for result in self._coordinator.run_training_async(
            prompts=prompts,
            num_steps=num_steps or self._state.total_steps,
        ):
            yield result

    def save_checkpoint(self, path: str | Path) -> None:
        """Save current training state to checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        if self._coordinator is None:
            raise RuntimeError("Trainer not initialized.")

        self._coordinator.save_checkpoint(path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training state from checkpoint.

        Args:
            path: Path to checkpoint.
        """
        if not self._initialized:
            self.setup()

        if self._coordinator is not None:
            self._coordinator.load_checkpoint(path)
            # Sync state
            self._state.global_step = self._coordinator.state.step

    def evaluate(self, prompts: PromptsType) -> dict[str, float]:
        """Run evaluation on given prompts.

        Args:
            prompts: Prompts to evaluate on.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self._coordinator is None:
            raise RuntimeError("Trainer not initialized.")

        # Prepare prompts
        prompt_list = self._prepare_prompts(prompts)

        # Generate and score
        total_reward = 0.0
        num_samples = 0

        for prompt in prompt_list[:100]:  # Limit eval samples
            # Simple evaluation - generate and compute reward
            trajectories = self._coordinator._generate_rollouts_sync([prompt])
            trajectories = self._coordinator._compute_rewards(trajectories)

            for traj in trajectories:
                total_reward += traj.reward
                num_samples += 1

        return {
            "mean_reward": total_reward / max(1, num_samples),
            "num_samples": num_samples,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive training statistics.

        Returns:
            Dictionary of all training statistics.
        """
        stats = {
            "training_state": {
                "global_step": self._state.global_step,
                "total_steps": self._state.total_steps,
                "samples_seen": self._state.samples_seen,
                "tokens_processed": self._state.tokens_processed,
            },
            "initialized": self._initialized,
        }

        if self._coordinator:
            stats["coordinator"] = self._coordinator.get_statistics()

        if self._metrics:
            stats["metrics"] = self._metrics.get_summary()

        return stats

    def _prepare_prompts(self, prompts: PromptsType) -> list[str]:
        """Convert prompts to list of strings.

        Args:
            prompts: Prompts in various formats.

        Returns:
            List of prompt strings.
        """
        if prompts is None:
            return []

        if isinstance(prompts, list):
            # Handle list of strings or dicts
            result = []
            for p in prompts:
                if isinstance(p, str):
                    result.append(p)
                elif isinstance(p, dict):
                    # Extract prompt from dict (common Dataset format)
                    result.append(p.get("prompt", p.get("text", str(p))))
                else:
                    result.append(str(p))
            return result

        # Handle Dataset-like objects
        if hasattr(prompts, "__iter__"):
            return self._prepare_prompts(list(prompts))

        return [str(prompts)]

    def _log_step(self, step_result: StepResult) -> None:
        """Log step metrics.

        Args:
            step_result: Result from training step.
        """
        if self._logger is None:
            return

        metrics = {}

        if step_result.training_result:
            metrics["loss"] = step_result.training_result.loss
            metrics["grad_norm"] = step_result.training_result.grad_norm
            metrics["throughput"] = step_result.training_result.throughput

        if step_result.staleness_metrics:
            metrics["staleness"] = step_result.staleness_metrics.staleness
            metrics["kl_divergence"] = step_result.staleness_metrics.kl_divergence

        if step_result.async_decision:
            metrics["async_ratio"] = step_result.async_decision.async_ratio

        metrics["step_time_ms"] = step_result.elapsed_ms
        metrics["batch_size"] = step_result.batch_size

        self._logger.log(metrics, step=step_result.step)

    def _run_callbacks(self, metrics: BatchMetrics) -> None:
        """Run all registered callbacks.

        Args:
            metrics: Metrics from training step.
        """
        for callback in self._callbacks:
            try:
                callback(self._state, metrics)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")

    def _save_checkpoint(self, step: int) -> str:
        """Save checkpoint at given step.

        Args:
            step: Current step number.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_path = checkpoint_dir / f"step-{step}"

        try:
            self.save_checkpoint(checkpoint_path)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

        return str(checkpoint_path)

    def __enter__(self) -> "FluxTrainer":
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.teardown()
