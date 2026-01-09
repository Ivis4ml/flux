"""
Checkpoint management for Flux.

Provides robust checkpoint save/load with versioning, cleanup, and recovery.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch


logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        step: Training step when checkpoint was created.
        version: Policy version at checkpoint time.
        timestamp: When the checkpoint was created.
        metrics: Optional metrics at checkpoint time.
        tags: Optional tags for filtering/searching.
    """

    checkpoint_id: str
    step: int
    version: int
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "tags": self.tags,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            step=data["step"],
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=data.get("metrics", {}),
            tags=data.get("tags", []),
            path=data.get("path", ""),
        )


@dataclass
class CheckpointState:
    """Complete checkpoint state.

    Attributes:
        metadata: Checkpoint metadata.
        model_state: Model state dict.
        optimizer_state: Optimizer state dict.
        scheduler_state: LR scheduler state dict.
        rng_state: Random number generator states.
        extra: Any additional state to save.
    """

    metadata: CheckpointMetadata
    model_state: dict[str, torch.Tensor] | None = None
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    rng_state: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manages checkpoint lifecycle with versioning and cleanup.

    Features:
    - Automatic cleanup of old checkpoints
    - Checkpoint registry with metadata
    - Atomic saves to prevent corruption
    - Best checkpoint tracking
    - Resume from latest or specific checkpoint

    Example:
        manager = CheckpointManager(
            checkpoint_dir="./checkpoints",
            max_checkpoints=5,
        )

        # Save checkpoint
        manager.save(
            step=1000,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            metrics={"loss": 0.5, "reward": 0.8},
        )

        # Load latest
        state = manager.load_latest()

        # Load best by metric
        state = manager.load_best(metric="reward", higher_is_better=True)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
        keep_best: int = 3,
        save_optimizer: bool = True,
        save_rng: bool = True,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
            max_checkpoints: Maximum number of checkpoints to keep.
            keep_best: Number of best checkpoints to always keep.
            save_optimizer: Whether to save optimizer state.
            save_rng: Whether to save RNG state.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.save_optimizer = save_optimizer
        self.save_rng = save_rng

        # Registry of all checkpoints
        self._registry: dict[str, CheckpointMetadata] = {}
        self._best_checkpoints: dict[str, str] = {}  # metric -> checkpoint_id

        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self._load_registry()

    @property
    def num_checkpoints(self) -> int:
        """Number of checkpoints in registry."""
        return len(self._registry)

    @property
    def latest_checkpoint(self) -> CheckpointMetadata | None:
        """Get metadata for latest checkpoint."""
        if not self._registry:
            return None
        return max(self._registry.values(), key=lambda m: m.step)

    def save(
        self,
        step: int,
        model_state: dict[str, torch.Tensor],
        optimizer_state: dict[str, Any] | None = None,
        scheduler_state: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CheckpointMetadata:
        """Save a checkpoint.

        Args:
            step: Current training step.
            model_state: Model state dict.
            optimizer_state: Optional optimizer state.
            scheduler_state: Optional scheduler state.
            metrics: Optional metrics for best checkpoint tracking.
            tags: Optional tags for this checkpoint.
            extra: Optional extra state to save.

        Returns:
            CheckpointMetadata for the saved checkpoint.
        """
        checkpoint_id = f"step-{step}-{int(time.time())}"
        checkpoint_path = self.checkpoint_dir / checkpoint_id

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            step=step,
            version=step,  # Assuming version = step
            metrics=metrics or {},
            tags=tags or [],
            path=str(checkpoint_path),
        )

        # Prepare state
        state = {
            "metadata": metadata.to_dict(),
            "model_state": model_state,
        }

        if self.save_optimizer and optimizer_state is not None:
            state["optimizer_state"] = optimizer_state

        if scheduler_state is not None:
            state["scheduler_state"] = scheduler_state

        if self.save_rng:
            state["rng_state"] = self._capture_rng_state()

        if extra:
            state["extra"] = extra

        # Atomic save
        self._atomic_save(checkpoint_path, state)

        # Update registry
        self._registry[checkpoint_id] = metadata
        self._update_best_checkpoints(metadata)
        self._save_registry()

        # Cleanup old checkpoints
        self._cleanup()

        logger.info(f"Saved checkpoint: {checkpoint_id}")
        return metadata

    def load(self, checkpoint_id: str) -> CheckpointState:
        """Load a specific checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to load.

        Returns:
            CheckpointState with all saved state.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        if checkpoint_id not in self._registry:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        metadata = self._registry[checkpoint_id]
        checkpoint_path = Path(metadata.path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file missing: {checkpoint_path}")

        state = torch.load(
            checkpoint_path / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,  # We save metadata dicts, not just weights
        )

        return CheckpointState(
            metadata=metadata,
            model_state=state.get("model_state"),
            optimizer_state=state.get("optimizer_state"),
            scheduler_state=state.get("scheduler_state"),
            rng_state=state.get("rng_state"),
            extra=state.get("extra", {}),
        )

    def load_latest(self) -> CheckpointState | None:
        """Load the latest checkpoint.

        Returns:
            CheckpointState or None if no checkpoints exist.
        """
        if not self._registry:
            return None

        latest = self.latest_checkpoint
        if latest is None:
            return None

        return self.load(latest.checkpoint_id)

    def load_best(
        self,
        metric: str,
        higher_is_better: bool = True,
    ) -> CheckpointState | None:
        """Load the best checkpoint for a metric.

        Args:
            metric: Metric name to optimize.
            higher_is_better: Whether higher values are better.

        Returns:
            CheckpointState or None if no matching checkpoints.
        """
        candidates = [
            m for m in self._registry.values()
            if metric in m.metrics
        ]

        if not candidates:
            return None

        if higher_is_better:
            best = max(candidates, key=lambda m: m.metrics[metric])
        else:
            best = min(candidates, key=lambda m: m.metrics[metric])

        return self.load(best.checkpoint_id)

    def list_checkpoints(
        self,
        tags: list[str] | None = None,
    ) -> list[CheckpointMetadata]:
        """List all checkpoints, optionally filtered by tags.

        Args:
            tags: Optional tags to filter by.

        Returns:
            List of checkpoint metadata.
        """
        checkpoints = list(self._registry.values())

        if tags:
            checkpoints = [
                c for c in checkpoints
                if any(t in c.tags for t in tags)
            ]

        return sorted(checkpoints, key=lambda m: m.step)

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete.

        Returns:
            True if deleted, False if not found.
        """
        if checkpoint_id not in self._registry:
            return False

        metadata = self._registry[checkpoint_id]
        checkpoint_path = Path(metadata.path)

        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)

        del self._registry[checkpoint_id]
        self._save_registry()

        logger.info(f"Deleted checkpoint: {checkpoint_id}")
        return True

    def restore_rng_state(self, state: CheckpointState) -> None:
        """Restore RNG state from checkpoint.

        Args:
            state: CheckpointState containing rng_state.
        """
        if state.rng_state is None:
            return

        rng = state.rng_state
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "torch_cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        if "numpy" in rng:
            import numpy as np
            np.random.set_state(rng["numpy"])
        if "random" in rng:
            import random
            random.setstate(rng["random"])

    def _atomic_save(self, path: Path, state: dict[str, Any]) -> None:
        """Save checkpoint atomically to prevent corruption."""
        path.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")

        try:
            temp_path.mkdir(parents=True, exist_ok=True)
            torch.save(state, temp_path / "checkpoint.pt")

            # Atomic rename
            if path.exists():
                shutil.rmtree(path)
            temp_path.rename(path)

        except Exception as e:
            if temp_path.exists():
                shutil.rmtree(temp_path)
            raise e

    def _capture_rng_state(self) -> dict[str, Any]:
        """Capture all RNG states."""
        state = {
            "torch": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()

        try:
            import numpy as np
            state["numpy"] = np.random.get_state()
        except ImportError:
            pass

        try:
            import random
            state["random"] = random.getstate()
        except ImportError:
            pass

        return state

    def _update_best_checkpoints(self, metadata: CheckpointMetadata) -> None:
        """Update best checkpoint tracking."""
        for metric, value in metadata.metrics.items():
            current_best = self._best_checkpoints.get(metric)
            if current_best is None:
                self._best_checkpoints[metric] = metadata.checkpoint_id
            else:
                current_meta = self._registry.get(current_best)
                if current_meta and value > current_meta.metrics.get(metric, float("-inf")):
                    self._best_checkpoints[metric] = metadata.checkpoint_id

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self._registry) <= self.max_checkpoints:
            return

        # Sort by step
        sorted_checkpoints = sorted(
            self._registry.values(),
            key=lambda m: m.step,
        )

        # Identify checkpoints to keep
        keep_ids = set()

        # Keep latest
        for cp in sorted_checkpoints[-self.max_checkpoints:]:
            keep_ids.add(cp.checkpoint_id)

        # Keep best
        for metric, cp_id in self._best_checkpoints.items():
            keep_ids.add(cp_id)

        # Delete old checkpoints
        to_delete = [
            cp.checkpoint_id
            for cp in sorted_checkpoints
            if cp.checkpoint_id not in keep_ids
        ]

        for cp_id in to_delete:
            self.delete(cp_id)

    def _load_registry(self) -> None:
        """Load checkpoint registry from disk."""
        registry_path = self.checkpoint_dir / "registry.json"

        if not registry_path.exists():
            return

        try:
            with open(registry_path) as f:
                data = json.load(f)

            self._registry = {
                cp_id: CheckpointMetadata.from_dict(meta)
                for cp_id, meta in data.get("checkpoints", {}).items()
            }
            self._best_checkpoints = data.get("best", {})

        except Exception as e:
            logger.warning(f"Failed to load checkpoint registry: {e}")

    def _save_registry(self) -> None:
        """Save checkpoint registry to disk."""
        registry_path = self.checkpoint_dir / "registry.json"

        data = {
            "checkpoints": {
                cp_id: meta.to_dict()
                for cp_id, meta in self._registry.items()
            },
            "best": self._best_checkpoints,
        }

        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)
