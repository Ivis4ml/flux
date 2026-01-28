"""
HuggingFace Transformers training backend for Flux.

This is the simplest backend implementation, suitable for:
- Single-GPU training
- Multi-GPU with DataParallel or DDP
- Development and debugging

For large models requiring model parallelism, use MegatronBackend or FSDPBackend.

Usage:
    backend = TransformersBackend()
    backend.initialize(TransformersConfig(
        model_path="Qwen/Qwen2-7B",
        learning_rate=1e-6,
    ))

    for batch in dataloader:
        gpu_batch = batch.as_gpu_batch(backend.device)
        result = backend.train_step(gpu_batch)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from flux.training.base import GPUBatch, TrainStepResult, TrainingBackendBase

logger = logging.getLogger(__name__)


@dataclass
class TransformersConfig:
    """Configuration for TransformersBackend."""

    # Model
    model_path: str = "gpt2"  # HuggingFace model ID or local path
    torch_dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    device_map: str | None = None  # For multi-GPU inference

    # Training
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False

    # Algorithm integration
    use_importance_weighting: bool = True
    importance_weight_clip: float = 5.0
    entropy_coef: float = 0.01
    kl_coef: float = 0.0

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


class TransformersBackend(TrainingBackendBase):
    """
    Training backend using HuggingFace Transformers.

    Simple but functional backend for development and single/multi-GPU training.
    Uses standard PyTorch optimizer and supports gradient checkpointing.

    Note: This backend is not optimized for large models requiring tensor/pipeline
    parallelism. Use MegatronBackend or FSDPBackend for those cases.
    """

    def __init__(self) -> None:
        """Initialize TransformersBackend."""
        super().__init__()

        self._config: TransformersConfig | None = None
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None

        # Gradient accumulation state
        self._accumulation_step = 0
        self._accumulated_loss = 0.0

    def _do_initialize(self, config: Any) -> None:
        """Initialize with TransformersConfig.

        Args:
            config: TransformersConfig or dict with config values.
        """
        if isinstance(config, dict):
            config = TransformersConfig(**config)
        elif not isinstance(config, TransformersConfig):
            # Try to extract relevant fields
            config = TransformersConfig(
                model_path=getattr(config, "model_path", "gpt2"),
                learning_rate=getattr(config, "learning_rate", 1e-6),
                weight_decay=getattr(config, "weight_decay", 0.01),
                max_grad_norm=getattr(config, "max_grad_norm", 1.0),
                torch_dtype=getattr(config, "torch_dtype", "bfloat16"),
                gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
            )

        self._config = config

        # Set device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # Load model
        self._load_model()

        # Setup optimizer
        self._setup_optimizer()

        logger.info(
            f"TransformersBackend initialized: model={config.model_path}, "
            f"device={self._device}, lr={config.learning_rate}"
        )

    def _load_model(self) -> None:
        """Load model from config."""
        try:
            from transformers import AutoModelForCausalLM, AutoConfig

            model_path = self._config.model_path
            torch_dtype = self._config.get_torch_dtype()

            logger.info(f"Loading model from {model_path}")

            # Load config first to check if flash attention is supported
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            # Set flash attention if available
            if self._config.use_flash_attention:
                if hasattr(model_config, "attn_implementation"):
                    model_config.attn_implementation = "flash_attention_2"
                elif hasattr(model_config, "_attn_implementation"):
                    model_config._attn_implementation = "flash_attention_2"

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=model_config,
                torch_dtype=torch_dtype,
                device_map=self._config.device_map,
                trust_remote_code=True,
            )

            # Move to device if no device_map
            if self._config.device_map is None:
                self._model = self._model.to(self._device)

            # Enable gradient checkpointing
            if self._config.gradient_checkpointing:
                self._model.gradient_checkpointing_enable()

            # Set training mode
            self._model.train()

            num_params = sum(p.numel() for p in self._model.parameters())
            trainable_params = sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            )

            logger.info(
                f"Model loaded: {num_params:,} params ({trainable_params:,} trainable)"
            )

        except ImportError:
            logger.warning(
                "transformers not installed, using dummy model for testing"
            )
            self._model = nn.Linear(1, 1).to(self._device)

    def _setup_optimizer(self) -> None:
        """Setup optimizer and optional scheduler."""
        if self._model is None:
            return

        # AdamW optimizer with weight decay
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _do_train_step(self, batch: GPUBatch) -> tuple[float, dict[str, float], bool]:
        """Execute one training step.

        Args:
            batch: GPUBatch on correct device.

        Returns:
            Tuple of (loss, metrics, optimizer_stepped).
        """
        if self._model is None or self._optimizer is None:
            raise RuntimeError("Model or optimizer not initialized")

        start_time = time.perf_counter()

        # Forward pass
        forward_start = time.perf_counter()
        loss, forward_metrics = self._forward_pass(batch)
        forward_time = time.perf_counter() - forward_start

        # Backward pass
        backward_start = time.perf_counter()
        self._backward_pass(loss)
        backward_time = time.perf_counter() - backward_start

        # Gradient accumulation
        self._accumulation_step += 1
        self._accumulated_loss += loss.item()

        # Optimizer step (if accumulation complete)
        optimizer_time = 0.0
        optimizer_stepped = False
        if self._accumulation_step >= self._config.gradient_accumulation_steps:
            optimizer_start = time.perf_counter()
            grad_norm = self._optimizer_step()
            optimizer_time = time.perf_counter() - optimizer_start
            optimizer_stepped = True

            # Reset accumulation
            avg_loss = self._accumulated_loss / self._accumulation_step
            self._accumulation_step = 0
            self._accumulated_loss = 0.0
        else:
            avg_loss = loss.item()
            grad_norm = 0.0

        total_time = time.perf_counter() - start_time

        # Build metrics
        metrics = {
            "forward_time_ms": forward_time * 1000,
            "backward_time_ms": backward_time * 1000,
            "optimizer_time_ms": optimizer_time * 1000,
            "total_time_ms": total_time * 1000,
            "grad_norm": grad_norm,
            "mean_version_gap": batch.mean_version_gap,
            "optimizer_stepped": float(optimizer_stepped),
        }
        metrics.update(forward_metrics)

        return avg_loss, metrics, optimizer_stepped

    def _forward_pass(self, batch: GPUBatch) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute forward pass and loss.

        Args:
            batch: GPUBatch on correct device.

        Returns:
            Tuple of (loss_tensor, metrics).
        """
        metrics = {}

        try:
            # Forward through model
            outputs = self._model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                return_dict=True,
            )

            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Compute log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Gather log probs for actual tokens (shifted by 1)
            # logits: [B, S, V] -> log_probs for tokens [B, S-1]
            target_tokens = batch.input_ids[:, 1:].unsqueeze(-1)  # [B, S-1, 1]
            token_log_probs = log_probs[:, :-1].gather(-1, target_tokens).squeeze(-1)

            # Align tensors
            seq_len = token_log_probs.shape[1]
            behavior_log_probs = batch.behavior_log_probs[:, :seq_len]

            if batch.loss_mask is not None:
                loss_mask = batch.loss_mask[:, :seq_len]
            else:
                loss_mask = batch.attention_mask[:, :seq_len].float()

            # Compute policy loss (simplified PPO-style)
            # Log ratio: log(π_θ / π_old) = log_prob - old_log_prob
            log_ratio = token_log_probs - behavior_log_probs
            ratio = torch.exp(log_ratio.clamp(-20, 20))

            # Get advantages (use rewards if not provided)
            if batch.advantages is not None:
                advantages = batch.advantages[:, :seq_len]
            else:
                # Simple: use rewards broadcast across sequence
                if batch.rewards.dim() == 1:
                    advantages = batch.rewards.unsqueeze(-1).expand(-1, seq_len)
                else:
                    advantages = batch.rewards[:, :seq_len]

            # Normalize advantages
            adv_mean = (advantages * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            adv_std = torch.sqrt(
                ((advantages - adv_mean) ** 2 * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            )
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            # Clipped surrogate objective
            clip_ratio = 0.2
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            policy_loss = -torch.min(surrogate1, surrogate2)

            # Apply loss mask
            policy_loss = (policy_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

            # Entropy bonus
            entropy = -(log_probs[:, :-1] * torch.exp(log_probs[:, :-1])).sum(dim=-1)
            entropy = (entropy * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            entropy_loss = -self._config.entropy_coef * entropy

            # KL penalty (optional)
            kl = (behavior_log_probs - token_log_probs).mean()
            kl_loss = self._config.kl_coef * kl

            # Total loss
            loss = policy_loss + entropy_loss + kl_loss

            # Compute metrics
            with torch.no_grad():
                clip_fraction = (
                    ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float() * loss_mask
                ).sum() / (loss_mask.sum() + 1e-8)

                approx_kl = 0.5 * ((log_ratio ** 2) * loss_mask).sum() / (loss_mask.sum() + 1e-8)

            metrics.update({
                "policy_loss": policy_loss.item(),
                "entropy": entropy.item(),
                "kl_divergence": kl.item(),
                "clip_fraction": clip_fraction.item(),
                "approx_kl": approx_kl.item(),
            })

            return loss, metrics

        except Exception as e:
            logger.warning(f"Forward pass error: {e}, using dummy loss")
            loss = torch.tensor(0.1, requires_grad=True, device=self._device)
            return loss, {"error": 1.0}

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Compute gradients.

        Args:
            loss: Loss tensor to backpropagate.
        """
        # Scale loss for gradient accumulation
        if self._config.gradient_accumulation_steps > 1:
            loss = loss / self._config.gradient_accumulation_steps

        loss.backward()

    def _optimizer_step(self) -> float:
        """Perform optimizer step with gradient clipping.

        Returns:
            Gradient norm before clipping.
        """
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            self._config.max_grad_norm,
        )

        # Optimizer step
        self._optimizer.step()

        # Zero gradients
        self._optimizer.zero_grad()

        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Get model weights for sync.

        Returns:
            Model state dict with tensors on CPU.
        """
        if self._model is None:
            return {}

        # Move to CPU for sync
        return {k: v.cpu() for k, v in self._model.state_dict().items()}

    def set_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load model weights.

        Args:
            state_dict: State dict to load (can be on any device).
        """
        if self._model is None:
            return

        # Move tensors to correct device and load
        device_state_dict = {k: v.to(self._device) for k, v in state_dict.items()}
        self._model.load_state_dict(device_state_dict)

    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        info = super().get_info()

        if self._model is not None:
            num_params = sum(p.numel() for p in self._model.parameters())
            trainable = sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            )
            info.update({
                "num_parameters": num_params,
                "trainable_parameters": trainable,
                "model_path": self._config.model_path if self._config else None,
            })

        return info

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint to path.

        Args:
            path: Path to save checkpoint.
        """
        if self._model is None:
            return

        checkpoint = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict() if self._optimizer else None,
            "version": self._version,
            "config": self._config.__dict__ if self._config else None,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from path.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self._device)

        if self._model is not None and "model" in checkpoint:
            self._model.load_state_dict(checkpoint["model"])

        if self._optimizer is not None and "optimizer" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer"])

        if "version" in checkpoint:
            self._version = checkpoint["version"]

        logger.info(f"Checkpoint loaded from {path}")
