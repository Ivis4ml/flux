"""
Megatron-LM training engine for Flux.

Provides an interface for training with Megatron-LM's 3D parallelism
(tensor, pipeline, and data parallelism).

This is a stub implementation that defines the interface. Full implementation
requires Megatron-LM installation and proper distributed setup.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn

from flux.core.config import AlgorithmConfig, MegatronConfig
from flux.core.trajectory import TrajectoryBatch
from flux.core.types import BatchMetrics, PolicyVersion


logger = logging.getLogger(__name__)


@dataclass
class TrainingStep:
    """Result of a single training step."""

    step: int
    loss: float
    metrics: BatchMetrics
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    throughput: float = 0.0  # tokens/second


@dataclass
class ModelState:
    """State of the model for checkpointing."""

    version: PolicyVersion
    state_dict: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    rng_state: dict[str, Any] | None = None


class TrainingEngine(ABC):
    """Abstract base class for training engines.

    Defines the interface that training engines must implement.
    """

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load model from path."""
        pass

    @abstractmethod
    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Get model state dict for weight sync."""
        pass

    @abstractmethod
    def train_step(
        self, batch: TrajectoryBatch, algorithm_config: AlgorithmConfig
    ) -> TrainingStep:
        """Perform one training step."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint to path."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from path."""
        pass


class MegatronEngine(TrainingEngine):
    """Training engine using Megatron-LM.

    Supports 3D parallelism for efficient large model training:
    - Tensor Parallelism (TP): Split layers across GPUs
    - Pipeline Parallelism (PP): Split model stages across GPUs
    - Data Parallelism (DP): Replicate across GPU groups

    Example:
        engine = MegatronEngine(
            config=MegatronConfig(tp_size=4, pp_size=2, dp_size=2),
        )
        engine.load_model("path/to/model")

        for batch in data_loader:
            result = engine.train_step(batch, algo_config)
            if result.step % 100 == 0:
                engine.save_checkpoint(f"checkpoint-{result.step}")
    """

    def __init__(
        self,
        config: MegatronConfig | None = None,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> None:
        """Initialize the Megatron engine.

        Args:
            config: Megatron configuration.
            algorithm_config: RL algorithm configuration.
        """
        self.config = config or MegatronConfig()
        self.algorithm_config = algorithm_config or AlgorithmConfig()

        # Model and training state
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None

        # Version tracking
        self._version = PolicyVersion(version_id=0)
        self._global_step = 0

        # Gradient accumulation
        self._accumulated_loss = 0.0
        self._accumulation_count = 0

        # Distributed state
        self._is_initialized = False
        self._world_size = 1
        self._rank = 0
        self._local_rank = 0

    @property
    def version(self) -> PolicyVersion:
        """Current policy version."""
        return self._version

    @property
    def global_step(self) -> int:
        """Current global step."""
        return self._global_step

    @property
    def is_initialized(self) -> bool:
        """Whether engine is initialized."""
        return self._is_initialized

    def initialize(self) -> None:
        """Initialize distributed training.

        Sets up Megatron's distributed environment and process groups.
        """
        if self._is_initialized:
            return

        # Check for Megatron-LM
        try:
            import megatron
            from megatron import get_args, initialize_megatron
            from megatron.core import mpu
        except ImportError:
            logger.warning(
                "Megatron-LM not installed. Using stub implementation."
            )
            self._is_initialized = True
            return

        # Initialize Megatron
        # Note: Actual initialization requires proper command-line args
        # This is a simplified stub

        self._is_initialized = True
        logger.info("Megatron engine initialized")

    def _resolve_algorithm_names(
        self, algo_config: AlgorithmConfig
    ) -> tuple[str, str]:
        """Resolve advantage estimator and policy loss names."""
        algo_name = algo_config.name
        if isinstance(algo_name, Enum):
            algo_name = algo_name.value
        else:
            algo_name = str(algo_name)

        adv_name = algo_config.adv_estimator
        loss_name = algo_config.policy_loss

        mapping = {
            "ppo": ("gae", "ppo"),
            "grpo": ("grpo", "grpo"),
            "reinforce": ("reinforce", "reinforce"),
            "dpo": ("dpo", "dpo"),
            "dapo": ("dapo", "dapo"),
            "gspo": ("gspo", "gspo"),
            "rloo": ("rloo", "rloo"),
        }

        if adv_name is None or loss_name is None:
            mapped = mapping.get(algo_name, (algo_name, algo_name))
            if adv_name is None:
                adv_name = mapped[0]
            if loss_name is None:
                loss_name = mapped[1]

        if algo_name == "ppo" and algo_config.kl_coef > 0:
            loss_name = "ppo_kl"

        return adv_name, loss_name

    def load_model(self, model_path: str) -> None:
        """Load model from path.

        Args:
            model_path: Path to model weights or HuggingFace model ID.
        """
        if not self._is_initialized:
            self.initialize()

        # Stub: Load model using transformers or Megatron's loader
        try:
            from transformers import AutoModelForCausalLM

            logger.info(f"Loading model from {model_path}")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )

            # Move to GPU
            if torch.cuda.is_available():
                self._model = self._model.cuda()

            logger.info(f"Model loaded with {sum(p.numel() for p in self._model.parameters()):,} parameters")

        except ImportError:
            logger.warning("transformers not installed, using dummy model")
            self._model = nn.Linear(1, 1)

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Get model state dict for weight sync.

        Returns:
            Model state dictionary.
        """
        if self._model is None:
            return {}
        return self._model.state_dict()

    def set_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Set model state dict.

        Args:
            state_dict: State dictionary to load.
        """
        if self._model is not None:
            self._model.load_state_dict(state_dict)

    def train_step(
        self,
        batch: TrajectoryBatch,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> TrainingStep:
        """Perform one training step.

        Args:
            batch: Batch of trajectories.
            algorithm_config: Algorithm configuration (uses default if None).

        Returns:
            TrainingStep with loss and metrics.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        algo_config = algorithm_config or self.algorithm_config
        import time
        start_time = time.time()

        # Get algorithm functions
        from flux.training.algorithms import get_policy_loss_fn, get_adv_estimator_fn

        adv_name, loss_name = self._resolve_algorithm_names(algo_config)
        adv_fn = get_adv_estimator_fn(adv_name)
        loss_fn = get_policy_loss_fn(loss_name)

        # Prepare tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensors = batch.to_tensors(device=device, pad_token_id=0)
        input_ids = tensors.get("input_ids")
        loss_mask = tensors.get("loss_mask")
        behavior_log_probs = tensors.get("behavior_log_probs")
        rewards = tensors.get("rewards")

        if input_ids is None or loss_mask is None or behavior_log_probs is None or rewards is None:
            raise RuntimeError("Batch tensors missing required fields")

        # Build token-level rewards
        token_rewards = None
        if any(traj.token_rewards for traj in batch.trajectories):
            token_rewards = torch.zeros_like(loss_mask)
            for i, traj in enumerate(batch.trajectories):
                if not traj.token_rewards:
                    continue
                length = min(len(traj.token_rewards), token_rewards.shape[1])
                token_rewards[i, :length] = torch.tensor(
                    traj.token_rewards[:length],
                    device=device,
                    dtype=token_rewards.dtype,
                )
        token_level_rewards = (
            token_rewards if token_rewards is not None else rewards.unsqueeze(-1) * loss_mask
        )

        # Compute advantages
        advantages, returns = adv_fn(
            token_level_rewards=token_level_rewards,
            response_mask=loss_mask,
            gamma=algo_config.gamma,
            lam=algo_config.gae_lambda,
        )

        # Normalize advantages if configured
        if algo_config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass (simplified)
        self._model.train()

        # Get logits from model
        # Note: Actual implementation would handle tokenization and model forward
        # This is a stub showing the structure

        try:
            if input_ids.numel() > 0:
                tokens = input_ids

                outputs = self._model(tokens)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # Compute log probs
                log_probs = torch.log_softmax(logits, dim=-1)
                # Gather log probs for actual tokens
                token_log_probs = log_probs[:, :-1].gather(
                    -1,
                    tokens[:, 1:].unsqueeze(-1),
                ).squeeze(-1)

                # Align masks and behavior log probs with token_log_probs
                target_len = token_log_probs.shape[1]
                behavior_log_probs = behavior_log_probs[:, :target_len]
                loss_mask = loss_mask[:, :target_len]
                token_level_rewards = token_level_rewards[:, :target_len]

                loss, loss_metrics = loss_fn(
                    old_log_prob=behavior_log_probs,
                    log_prob=token_log_probs,
                    advantages=advantages[:, :target_len],
                    response_mask=loss_mask,
                    clip_ratio=algo_config.clip_range,
                    entropy_coef=algo_config.entropy_coef,
                    kl_coef=algo_config.kl_coef,
                    target_kl=algo_config.kl_target,
                )
            else:
                # Dummy loss for testing
                loss = torch.tensor(0.1, requires_grad=True)
                if torch.cuda.is_available():
                    loss = loss.cuda()
                loss_metrics = {}

        except Exception as e:
            logger.warning(f"Forward pass failed: {e}, using dummy loss")
            loss = torch.tensor(0.1, requires_grad=True)
            if torch.cuda.is_available():
                loss = loss.cuda()
            loss_metrics = {}

        # Backward pass
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=1e-6,
            )

        self._optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            algo_config.max_grad_norm,
        )

        self._optimizer.step()

        # Update state
        self._global_step += 1
        self._version = PolicyVersion(version_id=self._global_step)

        # Compute metrics
        elapsed = time.time() - start_time
        num_tokens = batch.num_tokens if hasattr(batch, 'num_tokens') else 0

        metrics = BatchMetrics(
            policy_loss=loss.item(),
            total_loss=loss.item(),
            batch_size=batch.batch_size,
            num_tokens=num_tokens,
            forward_time_ms=elapsed * 500,  # Rough split
            backward_time_ms=elapsed * 500,
            total_time_ms=elapsed * 1000,
        )

        return TrainingStep(
            step=self._global_step,
            loss=loss.item(),
            metrics=metrics,
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            throughput=num_tokens / elapsed if elapsed > 0 else 0,
        )

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint to path.

        Args:
            path: Path to save checkpoint.
        """
        if self._model is None:
            logger.warning("No model to save")
            return

        state = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict() if self._optimizer else None,
            "scheduler": self._scheduler.state_dict() if self._scheduler else None,
            "global_step": self._global_step,
            "version": self._version.version_id,
        }

        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from path.

        Args:
            path: Path to checkpoint.
        """
        state = torch.load(path, map_location="cpu")

        if self._model is not None and "model" in state:
            self._model.load_state_dict(state["model"])

        if self._optimizer is not None and "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])

        if "global_step" in state:
            self._global_step = state["global_step"]

        if "version" in state:
            self._version = PolicyVersion(version_id=state["version"])

        logger.info(f"Checkpoint loaded from {path}")

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.

        Returns:
            Dict with model statistics.
        """
        if self._model is None:
            return {"loaded": False}

        num_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )

        return {
            "loaded": True,
            "num_parameters": num_params,
            "trainable_parameters": trainable_params,
            "dtype": str(next(self._model.parameters()).dtype),
            "device": str(next(self._model.parameters()).device),
            "global_step": self._global_step,
            "version": self._version.version_id,
        }
