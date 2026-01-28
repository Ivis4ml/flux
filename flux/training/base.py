"""
Native trainer contract for Flux.

This module defines the core abstractions for training backends:
- GPUBatch: Frozen, device-owned tensor batch for training
- TrainStepResult: Standardized result from train_step()
- TrainingBackend: Abstract base class for training implementations

Design principles:
1. GPU-Direct: All tensors in GPUBatch are already on target device
2. Native-First: Backends own device allocation, no host copies in hot path
3. Async-Safe: train_step() can be called from asyncio event loop
4. Algorithm-Agnostic: Backend handles forward/backward, algorithms are separate

Usage:
    # Create backend from config
    backend = create_training_backend(config)
    backend.initialize(config.training)

    # Training loop
    for batch in trajectory_store.sample_batches():
        gpu_batch = batch.as_gpu_batch(backend.device)
        result = backend.train_step(gpu_batch)

        if result.version % sync_interval == 0:
            sync_weights(backend.get_state_dict())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from flux.core.config import TrainingBackendType

if TYPE_CHECKING:
    from flux.core.config import FluxConfig


@dataclass(frozen=True)
class GPUBatch:
    """
    Tensorized batch for training - all tensors already on target device.

    This is the primary input format for TrainingBackend.train_step().
    Immutable (frozen) to prevent accidental modification during training.

    Attributes:
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        behavior_log_probs: Log probs from behavior policy [batch_size, seq_len]
        rewards: Scalar rewards [batch_size] or token-level [batch_size, seq_len]
        version_gaps: Policy version gap for each sample [batch_size]
        loss_mask: Mask for loss computation [batch_size, seq_len]

    Optional attributes for specific algorithms:
        token_rewards: Per-token rewards [batch_size, seq_len]
        ref_log_probs: Reference policy log probs [batch_size, seq_len]
        values: Value estimates [batch_size, seq_len]
        advantages: Pre-computed advantages [batch_size, seq_len]
        returns: Pre-computed returns [batch_size, seq_len]
    """

    # Required tensors
    input_ids: torch.Tensor                # [B, seq]
    attention_mask: torch.Tensor           # [B, seq]
    behavior_log_probs: torch.Tensor       # [B, seq]
    rewards: torch.Tensor                  # [B] or [B, seq]
    version_gaps: torch.Tensor             # [B]

    # Loss mask (required for proper loss computation)
    loss_mask: torch.Tensor | None = None  # [B, seq] - 1 for response tokens

    # Optional tensors
    token_rewards: torch.Tensor | None = None      # [B, seq]
    ref_log_probs: torch.Tensor | None = None      # [B, seq]
    values: torch.Tensor | None = None             # [B, seq]
    advantages: torch.Tensor | None = None         # [B, seq]
    returns: torch.Tensor | None = None            # [B, seq]

    @property
    def batch_size(self) -> int:
        """Number of samples in batch."""
        return self.input_ids.shape[0]

    @property
    def seq_len(self) -> int:
        """Sequence length (after padding)."""
        return self.input_ids.shape[1]

    @property
    def device(self) -> torch.device:
        """Device where tensors reside."""
        return self.input_ids.device

    @property
    def num_tokens(self) -> int:
        """Total number of non-padding tokens."""
        return int(self.attention_mask.sum().item())

    @property
    def num_loss_tokens(self) -> int:
        """Number of tokens contributing to loss."""
        if self.loss_mask is None:
            return self.num_tokens
        return int(self.loss_mask.sum().item())

    @property
    def mean_version_gap(self) -> float:
        """Average version gap across batch."""
        return float(self.version_gaps.float().mean().item())

    @property
    def max_version_gap(self) -> int:
        """Maximum version gap in batch."""
        return int(self.version_gaps.max().item())

    def validate(self) -> None:
        """Validate tensor shapes and device consistency.

        Raises:
            ValueError: If shapes don't match or tensors are on different devices.
        """
        B, S = self.input_ids.shape
        device = self.device

        # Check required tensor shapes
        if self.attention_mask.shape != (B, S):
            raise ValueError(
                f"attention_mask shape {self.attention_mask.shape} != expected {(B, S)}"
            )
        if self.behavior_log_probs.shape != (B, S):
            raise ValueError(
                f"behavior_log_probs shape {self.behavior_log_probs.shape} != expected {(B, S)}"
            )
        if self.rewards.shape[0] != B:
            raise ValueError(
                f"rewards batch size {self.rewards.shape[0]} != expected {B}"
            )
        if self.version_gaps.shape != (B,):
            raise ValueError(
                f"version_gaps shape {self.version_gaps.shape} != expected {(B,)}"
            )

        # Check device consistency
        tensors = [
            ("attention_mask", self.attention_mask),
            ("behavior_log_probs", self.behavior_log_probs),
            ("rewards", self.rewards),
            ("version_gaps", self.version_gaps),
        ]

        for name, tensor in tensors:
            if tensor.device != device:
                raise ValueError(
                    f"{name} on {tensor.device}, expected {device}"
                )

        # Check optional tensors
        optional_tensors = [
            ("loss_mask", self.loss_mask),
            ("token_rewards", self.token_rewards),
            ("ref_log_probs", self.ref_log_probs),
            ("values", self.values),
            ("advantages", self.advantages),
            ("returns", self.returns),
        ]

        for name, tensor in optional_tensors:
            if tensor is not None:
                if tensor.device != device:
                    raise ValueError(f"{name} on {tensor.device}, expected {device}")
                if tensor.shape[0] != B:
                    raise ValueError(
                        f"{name} batch size {tensor.shape[0]} != expected {B}"
                    )

    def to(self, device: torch.device | str) -> "GPUBatch":
        """Move all tensors to specified device.

        Note: Returns a new GPUBatch since this class is frozen.

        Args:
            device: Target device.

        Returns:
            New GPUBatch with tensors on target device.
        """
        device = torch.device(device) if isinstance(device, str) else device

        return GPUBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            behavior_log_probs=self.behavior_log_probs.to(device),
            rewards=self.rewards.to(device),
            version_gaps=self.version_gaps.to(device),
            loss_mask=self.loss_mask.to(device) if self.loss_mask is not None else None,
            token_rewards=self.token_rewards.to(device) if self.token_rewards is not None else None,
            ref_log_probs=self.ref_log_probs.to(device) if self.ref_log_probs is not None else None,
            values=self.values.to(device) if self.values is not None else None,
            advantages=self.advantages.to(device) if self.advantages is not None else None,
            returns=self.returns.to(device) if self.returns is not None else None,
        )

    @classmethod
    def from_tensors(
        cls,
        tensors: dict[str, torch.Tensor],
        device: torch.device | str | None = None,
    ) -> "GPUBatch":
        """Create GPUBatch from dictionary of tensors.

        Args:
            tensors: Dict with tensor names as keys.
            device: Optional device to move tensors to.

        Returns:
            New GPUBatch instance.
        """
        if device is not None:
            device = torch.device(device) if isinstance(device, str) else device
            tensors = {k: v.to(device) for k, v in tensors.items()}

        return cls(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            behavior_log_probs=tensors["behavior_log_probs"],
            rewards=tensors["rewards"],
            version_gaps=tensors["version_gaps"],
            loss_mask=tensors.get("loss_mask"),
            token_rewards=tensors.get("token_rewards"),
            ref_log_probs=tensors.get("ref_log_probs"),
            values=tensors.get("values"),
            advantages=tensors.get("advantages"),
            returns=tensors.get("returns"),
        )


@dataclass
class TrainStepResult:
    """Result from a single training step.

    Contains loss, metrics, and state for logging and checkpointing.
    """

    # Core results
    loss: float
    version: int
    metrics: dict[str, float] = field(default_factory=dict)

    # Gradient info
    grad_norm: float = 0.0
    grad_norm_clipped: bool = False

    # Timing
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    optimizer_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Batch info
    batch_size: int = 0
    num_tokens: int = 0
    throughput_tokens_per_sec: float = 0.0

    # Optional detailed metrics
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    kl_divergence: float | None = None
    clip_fraction: float | None = None
    approx_kl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "loss": self.loss,
            "version": self.version,
            "grad_norm": self.grad_norm,
            "batch_size": self.batch_size,
            "num_tokens": self.num_tokens,
            "throughput": self.throughput_tokens_per_sec,
        }
        result.update(self.metrics)

        # Add optional metrics if present
        optional = [
            ("policy_loss", self.policy_loss),
            ("value_loss", self.value_loss),
            ("entropy", self.entropy),
            ("kl_divergence", self.kl_divergence),
            ("clip_fraction", self.clip_fraction),
            ("approx_kl", self.approx_kl),
        ]
        for name, value in optional:
            if value is not None:
                result[name] = value

        return result


@runtime_checkable
class TrainingBackend(Protocol):
    """
    Protocol for training backends.

    GPU-direct assumptions:
    - All tensors in GPUBatch are already on the correct device
    - Backend owns device allocation (no host copies in hot path)
    - Async-safe: train_step() can be called from asyncio event loop

    Implementations should follow these conventions:
    1. initialize() is called once before training
    2. train_step() is the hot path - optimize for throughput
    3. get_state_dict() returns CPU tensors for weight sync
    4. version increments after each successful train_step()
    """

    @property
    def version(self) -> int:
        """Current policy version (incremented after each train_step)."""
        ...

    @property
    def device(self) -> torch.device:
        """Device this backend operates on."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether backend has been initialized."""
        ...

    def initialize(self, config: Any) -> None:
        """Initialize backend with config. Called once before training."""
        ...

    def train_step(self, batch: GPUBatch) -> TrainStepResult:
        """
        Execute one training step.

        Contract:
        - batch tensors are on self.device
        - returns after optimizer step completes
        - increments internal version counter on success
        - raises on failure (does not increment version)

        Args:
            batch: GPUBatch with tensors on correct device.

        Returns:
            TrainStepResult with loss and metrics.
        """
        ...

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Get model weights for sync. Returns CPU tensors."""
        ...

    def set_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load model weights."""
        ...


class TrainingBackendBase(ABC):
    """
    Abstract base class for training backends.

    Provides common functionality and enforces the TrainingBackend contract.
    Subclasses must implement the abstract methods.

    Example subclass:
        class TransformersBackend(TrainingBackendBase):
            def _do_initialize(self, config):
                self._model = AutoModelForCausalLM.from_pretrained(...)
                self._optimizer = AdamW(self._model.parameters(), ...)

            def _do_train_step(self, batch):
                outputs = self._model(batch.input_ids, ...)
                loss = compute_loss(outputs, batch)
                loss.backward()
                self._optimizer.step()
                return loss.item(), metrics
    """

    def __init__(self) -> None:
        """Initialize base backend state."""
        self._version = 0
        self._device: torch.device = torch.device("cpu")
        self._is_initialized = False

    @property
    def version(self) -> int:
        """Current policy version."""
        return self._version

    @property
    def device(self) -> torch.device:
        """Device this backend operates on."""
        return self._device

    @property
    def is_initialized(self) -> bool:
        """Whether backend has been initialized."""
        return self._is_initialized

    def initialize(self, config: Any) -> None:
        """Initialize backend with config.

        Args:
            config: Backend-specific configuration.
        """
        if self._is_initialized:
            return

        self._do_initialize(config)
        self._is_initialized = True

    def train_step(self, batch: GPUBatch) -> TrainStepResult:
        """Execute one training step.

        Args:
            batch: GPUBatch with tensors on correct device.

        Returns:
            TrainStepResult with loss and metrics.

        Raises:
            RuntimeError: If backend not initialized.
            ValueError: If batch is on wrong device.
        """
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Validate batch device
        if batch.device != self._device:
            raise ValueError(
                f"Batch on {batch.device}, expected {self._device}. "
                "Use batch.to(backend.device) before train_step()."
            )

        import time
        start_time = time.perf_counter()

        # Delegate to subclass - returns (loss, metrics, optimizer_stepped)
        loss, metrics, optimizer_stepped = self._do_train_step(batch)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Only increment version when optimizer step actually happened
        # This prevents version drift during gradient accumulation
        if optimizer_stepped:
            self._version += 1

        return TrainStepResult(
            loss=loss,
            version=self._version,
            metrics=metrics,
            batch_size=batch.batch_size,
            num_tokens=batch.num_tokens,
            total_time_ms=elapsed_ms,
            throughput_tokens_per_sec=(
                batch.num_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            ),
        )

    @abstractmethod
    def _do_initialize(self, config: Any) -> None:
        """Subclass initialization hook.

        Args:
            config: Backend-specific configuration.
        """
        pass

    @abstractmethod
    def _do_train_step(self, batch: GPUBatch) -> tuple[float, dict[str, float], bool]:
        """Subclass training step hook.

        Args:
            batch: GPUBatch with tensors on correct device.

        Returns:
            Tuple of (loss, metrics_dict, optimizer_stepped).
            optimizer_stepped should be True only when the optimizer.step() was called
            (i.e., after gradient accumulation is complete). This controls version increment.
        """
        pass

    @abstractmethod
    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Get model weights for sync. Returns CPU tensors."""
        pass

    @abstractmethod
    def set_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load model weights."""
        pass

    def get_info(self) -> dict[str, Any]:
        """Get backend information for logging.

        Returns:
            Dict with backend stats.
        """
        return {
            "backend_type": self.__class__.__name__,
            "version": self._version,
            "device": str(self._device),
            "is_initialized": self._is_initialized,
        }


def create_training_backend(config: "FluxConfig") -> TrainingBackendBase:
    """Factory function to create training backend from config.

    Args:
        config: Flux configuration with training_backend field.

    Returns:
        Initialized TrainingBackend instance.

    Raises:
        ValueError: If backend type is not supported.
    """
    backend_type = config.training_backend

    # Handle string values (from YAML configs)
    if isinstance(backend_type, str):
        backend_type = TrainingBackendType(backend_type)

    match backend_type:
        case TrainingBackendType.MEGATRON:
            from flux.training.megatron_engine import MegatronEngine
            # Wrap MegatronEngine in adapter (TODO: refactor MegatronEngine)
            raise NotImplementedError(
                "MegatronBackend adapter not yet implemented. "
                "Use MegatronEngine directly for now."
            )

        case TrainingBackendType.FSDP:
            raise NotImplementedError("FSDPBackend not yet implemented")

        case TrainingBackendType.TRANSFORMERS:
            from flux.training.backends.transformers import TransformersBackend
            return TransformersBackend()

        case TrainingBackendType.DEEPSPEED:
            raise NotImplementedError("DeepSpeedBackend not yet implemented")

        case _:
            raise ValueError(f"Unknown backend type: {backend_type}")
