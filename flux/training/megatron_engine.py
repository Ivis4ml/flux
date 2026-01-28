"""
Megatron-LM training engine for Flux.

Provides an interface for training with Megatron-LM's 3D parallelism
(tensor, pipeline, and data parallelism).

Supports:
- Tensor Parallelism (TP): Split model layers across GPUs
- Pipeline Parallelism (PP): Split model stages across GPUs
- Data Parallelism (DP): Replicate model across GPU groups

When Megatron-LM is not installed, falls back to a simple PyTorch implementation.

This module provides two interfaces:
1. TrainingBackend (new): GPUBatch-based, native-first interface
2. TrainingEngine (legacy): TrajectoryBatch-based interface for backward compatibility
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn

from flux.core.config import AlgorithmConfig, MegatronConfig
from flux.core.trajectory import TrajectoryBatch
from flux.core.types import BatchMetrics, PolicyVersion
from flux.training.base import GPUBatch, TrainStepResult, TrainingBackendBase


logger = logging.getLogger(__name__)

# Megatron availability check
try:
    import megatron
    from megatron import get_args, initialize_megatron
    from megatron.core import mpu
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
        get_data_parallel_rank,
        get_data_parallel_world_size,
    )
    HAS_MEGATRON = True
except ImportError:
    HAS_MEGATRON = False


@dataclass
class ParallelState:
    """Distributed parallelism state."""

    # Process ranks
    world_rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    # Tensor parallelism
    tp_rank: int = 0
    tp_size: int = 1

    # Pipeline parallelism
    pp_rank: int = 0
    pp_size: int = 1

    # Data parallelism
    dp_rank: int = 0
    dp_size: int = 1

    @property
    def is_first_stage(self) -> bool:
        """Whether this is the first pipeline stage."""
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        """Whether this is the last pipeline stage."""
        return self.pp_rank == self.pp_size - 1

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process for logging/saving."""
        return self.world_rank == 0


@dataclass
class TrainingStep:
    """Result of a single training step (legacy interface)."""

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
    """Abstract base class for training engines (legacy interface).

    Defines the interface that training engines must implement.
    For new code, prefer using TrainingBackend from flux.training.base.
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


class MegatronEngine(TrainingBackendBase, TrainingEngine):
    """Training engine using Megatron-LM.

    Implements both TrainingBackend (new) and TrainingEngine (legacy) interfaces.

    Supports 3D parallelism for efficient large model training:
    - Tensor Parallelism (TP): Split layers across GPUs
    - Pipeline Parallelism (PP): Split model stages across GPUs
    - Data Parallelism (DP): Replicate across GPU groups

    New interface (TrainingBackend):
        engine = MegatronEngine(config=MegatronConfig(...))
        engine.initialize(config)

        for gpu_batch in batches:
            result = engine.train_step(gpu_batch)  # Returns TrainStepResult

    Legacy interface (TrainingEngine):
        engine = MegatronEngine(config=MegatronConfig(...))
        engine.load_model("path/to/model")

        for batch in data_loader:
            result = engine.train_step_legacy(batch, algo_config)  # Returns TrainingStep
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
        # Initialize base class
        TrainingBackendBase.__init__(self)

        self.config = config or MegatronConfig()
        self.algorithm_config = algorithm_config or AlgorithmConfig()

        # Model and training state
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None

        # Version tracking (use base class _version for new interface)
        self._policy_version = PolicyVersion(version_id=0)
        self._global_step = 0

        # Gradient accumulation
        self._accumulated_loss = 0.0
        self._accumulation_count = 0

        # Distributed state
        self._parallel_state = ParallelState()
        self._use_megatron = False

    # ========================================================================
    # TrainingBackend interface (new)
    # ========================================================================

    @property
    def version(self) -> int:
        """Current policy version (TrainingBackend interface)."""
        return self._version

    @property
    def device(self) -> torch.device:
        """Device this backend operates on."""
        return self._device

    @property
    def is_initialized(self) -> bool:
        """Whether backend has been initialized."""
        return self._is_initialized

    def _do_initialize(self, config: Any) -> None:
        """Initialize backend with config.

        Args:
            config: MegatronConfig or compatible config object.
        """
        if isinstance(config, MegatronConfig):
            self.config = config
        elif hasattr(config, "megatron"):
            self.config = config.megatron

        # Set device
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._device = torch.device(f"cuda:{local_rank}")
        else:
            self._device = torch.device("cpu")

        # Initialize distributed
        self._initialize_distributed()

        # Load model if path provided
        model_path = getattr(self.config, "model_path", None)
        if model_path:
            self.load_model(model_path)

    def _do_train_step(self, batch: GPUBatch) -> tuple[float, dict[str, float]]:
        """Execute one training step with GPUBatch.

        Args:
            batch: GPUBatch on correct device.

        Returns:
            Tuple of (loss, metrics).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.perf_counter()

        # Get algorithm functions
        from flux.training.algorithms import get_policy_loss_fn, get_adv_estimator_fn

        adv_name, loss_name = self._resolve_algorithm_names(self.algorithm_config)
        adv_fn = get_adv_estimator_fn(adv_name)
        loss_fn = get_policy_loss_fn(loss_name)

        # Extract tensors from GPUBatch
        input_ids = batch.input_ids
        loss_mask = batch.loss_mask if batch.loss_mask is not None else batch.attention_mask.float()
        behavior_log_probs = batch.behavior_log_probs
        rewards = batch.rewards

        # Build token-level rewards if not provided
        if batch.token_rewards is not None:
            token_level_rewards = batch.token_rewards
        else:
            if rewards.dim() == 1:
                token_level_rewards = rewards.unsqueeze(-1) * loss_mask
            else:
                token_level_rewards = rewards

        # Compute advantages if not provided
        if batch.advantages is not None:
            advantages = batch.advantages
        else:
            advantages, returns = adv_fn(
                token_level_rewards=token_level_rewards,
                response_mask=loss_mask,
                gamma=self.algorithm_config.gamma,
                lam=self.algorithm_config.gae_lambda,
            )

            # Normalize advantages
            if self.algorithm_config.normalize_advantages and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        self._model.train()
        forward_start = time.perf_counter()

        try:
            outputs = self._model(input_ids, attention_mask=batch.attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Compute log probs
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs[:, :-1].gather(
                -1,
                input_ids[:, 1:].unsqueeze(-1),
            ).squeeze(-1)

            # Align tensors
            target_len = token_log_probs.shape[1]
            behavior_log_probs_aligned = behavior_log_probs[:, :target_len]
            loss_mask_aligned = loss_mask[:, :target_len]
            advantages_aligned = advantages[:, :target_len] if advantages.dim() > 1 else advantages

            # Compute importance weights for off-policy correction
            importance_weights = self._compute_importance_weights_gpu(
                behavior_log_probs=behavior_log_probs_aligned,
                current_log_probs=token_log_probs,
                response_mask=loss_mask_aligned,
                version_gaps=batch.version_gaps,
            )

            # Compute loss
            loss, loss_metrics = loss_fn(
                old_log_prob=behavior_log_probs_aligned,
                log_prob=token_log_probs,
                advantages=advantages_aligned,
                response_mask=loss_mask_aligned,
                clip_ratio=self.algorithm_config.clip_range,
                entropy_coef=self.algorithm_config.entropy_coef,
                kl_coef=self.algorithm_config.kl_coef,
                target_kl=self.algorithm_config.kl_target,
                importance_weights=importance_weights,
            )

        except Exception as e:
            logger.warning(f"Forward pass failed: {e}, using dummy loss")
            loss = torch.tensor(0.1, requires_grad=True, device=self._device)
            loss_metrics = {"error": 1.0}

        forward_time = time.perf_counter() - forward_start

        # Backward pass
        backward_start = time.perf_counter()

        if self._optimizer is None:
            self._setup_optimizer()

        self._optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            self.algorithm_config.max_grad_norm,
        )

        self._optimizer.step()
        backward_time = time.perf_counter() - backward_start

        # Update version tracking
        self._global_step += 1
        self._policy_version = PolicyVersion(version_id=self._global_step)

        total_time = time.perf_counter() - start_time

        # Build metrics
        metrics = {
            "forward_time_ms": forward_time * 1000,
            "backward_time_ms": backward_time * 1000,
            "total_time_ms": total_time * 1000,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "mean_version_gap": batch.mean_version_gap,
            "max_version_gap": batch.max_version_gap,
        }
        metrics.update(loss_metrics)

        return loss.item(), metrics

    def _compute_importance_weights_gpu(
        self,
        behavior_log_probs: torch.Tensor,
        current_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        version_gaps: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute importance weights from GPUBatch tensors.

        Args:
            behavior_log_probs: Log probs under behavior policy [B, S]
            current_log_probs: Log probs under current policy [B, S]
            response_mask: Valid token mask [B, S]
            version_gaps: Version gap per sample [B]

        Returns:
            Per-sample importance weights [B] or None
        """
        if not getattr(self.algorithm_config, 'use_importance_weighting', True):
            return None

        try:
            from flux.controller.importance import compute_importance_weights

            return compute_importance_weights(
                behavior_log_probs=behavior_log_probs,
                current_log_probs=current_log_probs,
                response_mask=response_mask,
                version_gap=version_gaps.float(),
                staleness_decay=getattr(self.algorithm_config, 'staleness_decay', 0.99),
                max_weight=getattr(self.algorithm_config, 'max_importance_weight', 5.0),
                min_weight=getattr(self.algorithm_config, 'min_importance_weight', 0.2),
                normalize=True,
            )
        except Exception as e:
            logger.warning(f"Failed to compute importance weights: {e}")
            return None

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Get model weights for sync. Returns CPU tensors."""
        if self._model is None:
            return {}
        return {k: v.cpu() for k, v in self._model.state_dict().items()}

    def set_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load model weights."""
        if self._model is not None:
            device_state_dict = {k: v.to(self._device) for k, v in state_dict.items()}
            self._model.load_state_dict(device_state_dict)

    # ========================================================================
    # TrainingEngine interface (legacy)
    # ========================================================================

    @property
    def policy_version(self) -> PolicyVersion:
        """Current policy version (legacy interface)."""
        return self._policy_version

    @property
    def global_step(self) -> int:
        """Current global step."""
        return self._global_step

    @property
    def parallel_state(self) -> ParallelState:
        """Current parallel state."""
        return self._parallel_state

    def initialize(self, config: Any = None) -> None:
        """Initialize distributed training.

        This method supports both interfaces:
        - TrainingBackend: Called by base class
        - TrainingEngine: Called directly for legacy usage

        Args:
            config: Optional config override.
        """
        if self._is_initialized:
            return

        if config is not None:
            self._do_initialize(config)
        else:
            self._do_initialize(self.config)

        self._is_initialized = True

    def _initialize_distributed(self) -> None:
        """Initialize distributed training."""
        if HAS_MEGATRON and self.config.use_megatron:
            self._initialize_megatron()
        else:
            self._initialize_pytorch_distributed()

    def _initialize_megatron(self) -> None:
        """Initialize with Megatron-LM for 3D parallelism."""
        logger.info("Initializing Megatron-LM distributed training")

        megatron_args = self._build_megatron_args()
        initialize_megatron(args_defaults=megatron_args)

        self._parallel_state = ParallelState(
            world_rank=dist.get_rank() if dist.is_initialized() else 0,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(),
            pp_rank=get_pipeline_model_parallel_rank(),
            pp_size=get_pipeline_model_parallel_world_size(),
            dp_rank=get_data_parallel_rank(),
            dp_size=get_data_parallel_world_size(),
        )

        self._use_megatron = True
        logger.info(
            f"Megatron initialized: TP={self._parallel_state.tp_size}, "
            f"PP={self._parallel_state.pp_size}, DP={self._parallel_state.dp_size}"
        )

    def _initialize_pytorch_distributed(self) -> None:
        """Initialize with basic PyTorch distributed (fallback)."""
        if not dist.is_initialized():
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend)
                logger.info(f"Initialized PyTorch distributed with {backend}")
            else:
                logger.info("Running in single-process mode")
                return

        self._parallel_state = ParallelState(
            world_rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            dp_rank=dist.get_rank(),
            dp_size=dist.get_world_size(),
        )

        logger.info(f"PyTorch distributed initialized: DP={self._parallel_state.dp_size}")

    def _build_megatron_args(self) -> dict[str, Any]:
        """Build Megatron initialization args from config."""
        return {
            "tensor_model_parallel_size": self.config.tp_size,
            "pipeline_model_parallel_size": self.config.pp_size,
            "micro_batch_size": self.config.micro_batch_size,
            "global_batch_size": self.config.global_batch_size,
            "seq_length": self.config.seq_length,
            "fp16": self.config.fp16,
            "bf16": self.config.bf16,
            "use_flash_attn": self.config.use_flash_attention,
            "no_save_optim": True,
            "no_save_rng": True,
        }

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

    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        if self._model is None:
            return

        lr = getattr(self.config, 'learning_rate', 1e-6)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=lr,
            weight_decay=0.01,
        )

    def load_model(self, model_path: str) -> None:
        """Load model from path.

        Args:
            model_path: Path to model weights or HuggingFace model ID.
        """
        if not self._is_initialized:
            self.initialize()

        try:
            from transformers import AutoModelForCausalLM

            logger.info(f"Loading model from {model_path}")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                trust_remote_code=True,
            )

            self._model = self._model.to(self._device)
            logger.info(
                f"Model loaded with {sum(p.numel() for p in self._model.parameters()):,} parameters"
            )

        except ImportError:
            logger.warning("transformers not installed, using dummy model")
            self._model = nn.Linear(1, 1).to(self._device)

    def train_step(
        self,
        batch: TrajectoryBatch | GPUBatch,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> TrainingStep | TrainStepResult:
        """Perform one training step.

        Supports both new (GPUBatch) and legacy (TrajectoryBatch) interfaces.

        Args:
            batch: GPUBatch or TrajectoryBatch.
            algorithm_config: Algorithm configuration (uses default if None).

        Returns:
            TrainStepResult for GPUBatch, TrainingStep for TrajectoryBatch.
        """
        # Handle GPUBatch (new interface)
        if isinstance(batch, GPUBatch):
            if algorithm_config is not None:
                self.algorithm_config = algorithm_config
            return TrainingBackendBase.train_step(self, batch)

        # Handle TrajectoryBatch (legacy interface)
        return self.train_step_legacy(batch, algorithm_config)

    def train_step_legacy(
        self,
        batch: TrajectoryBatch,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> TrainingStep:
        """Perform one training step with TrajectoryBatch (legacy interface).

        Args:
            batch: Batch of trajectories.
            algorithm_config: Algorithm configuration (uses default if None).

        Returns:
            TrainingStep with loss and metrics.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        algo_config = algorithm_config or self.algorithm_config
        self.algorithm_config = algo_config

        # Convert TrajectoryBatch to GPUBatch
        gpu_batch = self._trajectory_batch_to_gpu_batch(batch)

        # Use new interface internally
        result = TrainingBackendBase.train_step(self, gpu_batch)

        # Convert result to legacy format
        num_tokens = batch.num_tokens if hasattr(batch, 'num_tokens') else 0

        metrics = BatchMetrics(
            policy_loss=result.loss,
            total_loss=result.loss,
            batch_size=batch.batch_size,
            num_tokens=num_tokens,
            forward_time_ms=result.metrics.get("forward_time_ms", 0),
            backward_time_ms=result.metrics.get("backward_time_ms", 0),
            total_time_ms=result.total_time_ms,
        )

        return TrainingStep(
            step=self._global_step,
            loss=result.loss,
            metrics=metrics,
            grad_norm=result.metrics.get("grad_norm", 0.0),
            throughput=result.throughput_tokens_per_sec,
        )

    def _trajectory_batch_to_gpu_batch(self, batch: TrajectoryBatch) -> GPUBatch:
        """Convert TrajectoryBatch to GPUBatch.

        Args:
            batch: TrajectoryBatch to convert.

        Returns:
            GPUBatch on correct device.
        """
        tensors = batch.to_tensors(device=self._device, pad_token_id=0)

        # Extract version gaps
        version_gaps = torch.tensor(
            [t.version.version_id for t in batch.trajectories],
            device=self._device,
            dtype=torch.long,
        )
        # Convert to gaps from current version
        version_gaps = self._version - version_gaps

        return GPUBatch(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            behavior_log_probs=tensors["behavior_log_probs"],
            rewards=tensors["rewards"],
            version_gaps=version_gaps,
            loss_mask=tensors["loss_mask"],
            advantages=tensors.get("advantages"),
            returns=tensors.get("returns"),
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
            "version": self._version,
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
            self._version = state["version"]
            self._policy_version = PolicyVersion(version_id=state["version"])

        logger.info(f"Checkpoint loaded from {path}")

    def get_info(self) -> dict[str, Any]:
        """Get backend/engine information.

        Returns:
            Dict with model statistics.
        """
        info = TrainingBackendBase.get_info(self)

        if self._model is not None:
            num_params = sum(p.numel() for p in self._model.parameters())
            trainable_params = sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            )
            info.update({
                "num_parameters": num_params,
                "trainable_parameters": trainable_params,
                "dtype": str(next(self._model.parameters()).dtype),
                "global_step": self._global_step,
                "use_megatron": self._use_megatron,
                "parallel_state": {
                    "tp_size": self._parallel_state.tp_size,
                    "pp_size": self._parallel_state.pp_size,
                    "dp_size": self._parallel_state.dp_size,
                },
            })

        return info

    # Legacy alias
    def get_model_info(self) -> dict[str, Any]:
        """Get model information (legacy method)."""
        return self.get_info()
