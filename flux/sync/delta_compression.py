"""
Delta compression for efficient weight synchronization.

Implements techniques to reduce the amount of data transferred
when synchronizing model weights:
1. Snapshot management: Track baseline weights for delta computation
2. Sparse delta encoding: Only transfer significantly changed weights
3. Quantization: Reduce precision for smaller transfers
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch


logger = logging.getLogger(__name__)


@dataclass
class DeltaStats:
    """Statistics about a delta computation."""

    total_params: int = 0
    changed_params: int = 0
    total_elements: int = 0
    changed_elements: int = 0
    compression_ratio: float = 0.0
    sparsity: float = 0.0


@dataclass
class WeightSnapshot:
    """Snapshot of model weights at a specific version."""

    version: int
    weights: dict[str, torch.Tensor]
    created_at: float = 0.0  # timestamp

    def __post_init__(self) -> None:
        import time
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class CompressedDelta:
    """Compressed representation of weight changes."""

    from_version: int
    to_version: int
    changed_params: dict[str, torch.Tensor]  # Only changed parameters
    sparse_indices: dict[str, torch.Tensor] | None = None  # For sparse encoding
    sparse_values: dict[str, torch.Tensor] | None = None
    stats: DeltaStats = field(default_factory=DeltaStats)


class SnapshotManager:
    """Manages weight snapshots for delta compression.

    Keeps track of baseline weight snapshots at regular intervals.
    Used to compute deltas from the nearest snapshot.

    Example:
        manager = SnapshotManager(max_snapshots=5)

        # Take snapshot at training step
        manager.take_snapshot(model.state_dict(), version=100)

        # Later, compute delta from nearest snapshot
        delta = manager.compute_delta(current_weights, from_version=95)
    """

    def __init__(
        self,
        max_snapshots: int = 5,
        snapshot_interval: int = 10,
    ) -> None:
        """Initialize the snapshot manager.

        Args:
            max_snapshots: Maximum number of snapshots to keep.
            snapshot_interval: Recommended interval between snapshots.
        """
        self.max_snapshots = max_snapshots
        self.snapshot_interval = snapshot_interval

        # Ordered dict for LRU-style eviction
        self._snapshots: OrderedDict[int, WeightSnapshot] = OrderedDict()

    @property
    def num_snapshots(self) -> int:
        """Number of stored snapshots."""
        return len(self._snapshots)

    @property
    def versions(self) -> list[int]:
        """List of snapshot versions."""
        return list(self._snapshots.keys())

    def take_snapshot(
        self,
        weights: dict[str, torch.Tensor],
        version: int,
        clone: bool = True,
    ) -> WeightSnapshot:
        """Take a snapshot of current weights.

        Args:
            weights: Model state dict.
            version: Version number for this snapshot.
            clone: If True, clone tensors to avoid reference issues.

        Returns:
            The created snapshot.
        """
        # Clone weights if requested
        if clone:
            weights = {k: v.clone().cpu() for k, v in weights.items()}
        else:
            weights = {k: v.cpu() for k, v in weights.items()}

        snapshot = WeightSnapshot(version=version, weights=weights)
        self._snapshots[version] = snapshot

        # Evict oldest if over capacity
        while len(self._snapshots) > self.max_snapshots:
            self._snapshots.popitem(last=False)

        return snapshot

    def get_snapshot(self, version: int) -> WeightSnapshot | None:
        """Get a specific snapshot.

        Args:
            version: Version number.

        Returns:
            Snapshot or None if not found.
        """
        return self._snapshots.get(version)

    def get_nearest_snapshot(self, version: int) -> WeightSnapshot | None:
        """Get the nearest snapshot at or before a version.

        Args:
            version: Target version.

        Returns:
            Nearest snapshot or None if no suitable snapshot.
        """
        nearest = None
        nearest_version = -1

        for v, snapshot in self._snapshots.items():
            if v <= version and v > nearest_version:
                nearest = snapshot
                nearest_version = v

        return nearest

    def compute_delta(
        self,
        current_weights: dict[str, torch.Tensor],
        from_version: int,
        threshold: float = 1e-6,
    ) -> CompressedDelta | None:
        """Compute delta from a snapshot to current weights.

        Args:
            current_weights: Current model weights.
            from_version: Version to compute delta from.
            threshold: Minimum change magnitude to include.

        Returns:
            CompressedDelta or None if no suitable snapshot.
        """
        snapshot = self.get_nearest_snapshot(from_version)
        if snapshot is None:
            return None

        return compute_weight_delta(
            baseline=snapshot.weights,
            current=current_weights,
            from_version=snapshot.version,
            to_version=from_version,
            threshold=threshold,
        )

    def should_snapshot(self, version: int) -> bool:
        """Check if a snapshot should be taken at this version.

        Args:
            version: Current version.

        Returns:
            True if snapshot should be taken.
        """
        if not self._snapshots:
            return True

        latest_version = max(self._snapshots.keys())
        return (version - latest_version) >= self.snapshot_interval

    def clear(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()


def compute_weight_delta(
    baseline: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    from_version: int,
    to_version: int,
    threshold: float = 1e-6,
) -> CompressedDelta:
    """Compute delta between two weight dicts.

    Args:
        baseline: Baseline weights.
        current: Current weights.
        from_version: Version of baseline.
        to_version: Version of current.
        threshold: Minimum change magnitude.

    Returns:
        CompressedDelta containing only changed parameters.
    """
    changed_params = {}
    stats = DeltaStats()

    total_elements = 0
    changed_elements = 0

    for name, current_tensor in current.items():
        total_elements += current_tensor.numel()
        stats.total_params += 1

        if name not in baseline:
            # New parameter, include fully
            changed_params[name] = current_tensor
            changed_elements += current_tensor.numel()
            stats.changed_params += 1
            continue

        baseline_tensor = baseline[name]

        # Compute difference
        diff = current_tensor.cpu() - baseline_tensor.cpu()
        max_diff = diff.abs().max().item()

        if max_diff > threshold:
            # Parameter has changed, include delta
            changed_params[name] = diff.to(current_tensor.device)
            changed_elements += current_tensor.numel()
            stats.changed_params += 1

    stats.total_elements = total_elements
    stats.changed_elements = changed_elements
    stats.sparsity = 1.0 - (changed_elements / max(1, total_elements))
    stats.compression_ratio = changed_elements / max(1, total_elements)

    return CompressedDelta(
        from_version=from_version,
        to_version=to_version,
        changed_params=changed_params,
        stats=stats,
    )


def apply_delta(
    baseline: dict[str, torch.Tensor],
    delta: CompressedDelta,
) -> dict[str, torch.Tensor]:
    """Apply a delta to baseline weights.

    Args:
        baseline: Baseline weights.
        delta: Delta to apply.

    Returns:
        Updated weights.
    """
    result = {}

    for name, baseline_tensor in baseline.items():
        if name in delta.changed_params:
            # Apply delta
            result[name] = baseline_tensor + delta.changed_params[name]
        else:
            # Keep baseline
            result[name] = baseline_tensor

    # Add any new parameters not in baseline
    for name in delta.changed_params:
        if name not in result:
            result[name] = delta.changed_params[name]

    return result


class SparseEncoder:
    """Encodes weight deltas as sparse tensors.

    For highly sparse deltas, storing only non-zero indices and values
    can significantly reduce transfer size.

    Example:
        encoder = SparseEncoder(sparsity_threshold=0.5)

        # Encode sparse delta
        indices, values = encoder.encode(delta_tensor)

        # Decode back to dense
        dense = encoder.decode(indices, values, shape)
    """

    def __init__(
        self,
        sparsity_threshold: float = 0.3,
        value_threshold: float = 1e-6,
    ) -> None:
        """Initialize the sparse encoder.

        Args:
            sparsity_threshold: Minimum sparsity to use sparse encoding.
            value_threshold: Minimum absolute value to keep.
        """
        self.sparsity_threshold = sparsity_threshold
        self.value_threshold = value_threshold

    def should_encode_sparse(self, tensor: torch.Tensor) -> bool:
        """Check if a tensor should be sparse-encoded.

        Args:
            tensor: The tensor to check.

        Returns:
            True if sparse encoding would be beneficial.
        """
        non_zero = (tensor.abs() > self.value_threshold).sum().item()
        sparsity = 1.0 - (non_zero / tensor.numel())
        return sparsity >= self.sparsity_threshold

    def encode(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a tensor as sparse indices and values.

        Args:
            tensor: Dense tensor to encode.

        Returns:
            Tuple of (indices, values) tensors.
        """
        flat = tensor.view(-1)
        mask = flat.abs() > self.value_threshold
        indices = torch.nonzero(mask, as_tuple=True)[0]
        values = flat[mask]
        return indices, values

    def decode(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Decode sparse indices and values to dense tensor.

        Args:
            indices: Flat indices of non-zero elements.
            values: Values at those indices.
            shape: Target tensor shape.
            device: Target device.

        Returns:
            Dense tensor.
        """
        numel = 1
        for s in shape:
            numel *= s

        flat = torch.zeros(numel, dtype=values.dtype, device=device or values.device)
        flat[indices] = values
        return flat.view(shape)

    def encode_dict(
        self, tensors: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, tuple]]:
        """Encode multiple tensors, using sparse where beneficial.

        Args:
            tensors: Dict of name -> tensor.

        Returns:
            Tuple of (indices_dict, values_dict, shapes_dict).
        """
        indices_dict = {}
        values_dict = {}
        shapes_dict = {}

        for name, tensor in tensors.items():
            if self.should_encode_sparse(tensor):
                indices, values = self.encode(tensor)
                indices_dict[name] = indices
                values_dict[name] = values
            else:
                # Store flat tensor directly
                indices_dict[name] = torch.arange(tensor.numel())
                values_dict[name] = tensor.view(-1)

            shapes_dict[name] = tuple(tensor.shape)

        return indices_dict, values_dict, shapes_dict

    def decode_dict(
        self,
        indices_dict: dict[str, torch.Tensor],
        values_dict: dict[str, torch.Tensor],
        shapes_dict: dict[str, tuple],
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode multiple sparse tensors.

        Args:
            indices_dict: Dict of name -> indices.
            values_dict: Dict of name -> values.
            shapes_dict: Dict of name -> shape.
            device: Target device.

        Returns:
            Dict of name -> dense tensor.
        """
        tensors = {}
        for name in indices_dict:
            tensors[name] = self.decode(
                indices_dict[name],
                values_dict[name],
                shapes_dict[name],
                device,
            )
        return tensors


class QuantizedEncoder:
    """Encodes weight deltas with reduced precision.

    Quantizes float32 deltas to lower precision formats
    to reduce transfer size.
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
    ) -> None:
        """Initialize the quantized encoder.

        Args:
            bits: Number of bits for quantization (8 or 16).
            symmetric: If True, use symmetric quantization.
        """
        self.bits = bits
        self.symmetric = symmetric

        if bits == 8:
            self.qtype = torch.int8
            self.qmin = -128
            self.qmax = 127
        elif bits == 16:
            self.qtype = torch.int16
            self.qmin = -32768
            self.qmax = 32767
        else:
            raise ValueError(f"Unsupported bits: {bits}")

    def encode(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Quantize a tensor.

        Args:
            tensor: Float tensor to quantize.

        Returns:
            Tuple of (quantized tensor, scale factor).
        """
        if self.symmetric:
            abs_max = tensor.abs().max().item()
            scale = abs_max / self.qmax if abs_max > 0 else 1.0
        else:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            scale = (max_val - min_val) / (self.qmax - self.qmin)
            scale = scale if scale > 0 else 1.0

        quantized = (tensor / scale).round().clamp(self.qmin, self.qmax).to(self.qtype)
        return quantized, scale

    def decode(
        self,
        quantized: torch.Tensor,
        scale: float,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantize a tensor.

        Args:
            quantized: Quantized tensor.
            scale: Scale factor from encoding.
            dtype: Target dtype.

        Returns:
            Dequantized float tensor.
        """
        return quantized.to(dtype) * scale

    def encode_dict(
        self, tensors: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Quantize multiple tensors.

        Args:
            tensors: Dict of name -> tensor.

        Returns:
            Tuple of (quantized dict, scales dict).
        """
        quantized = {}
        scales = {}
        for name, tensor in tensors.items():
            q, s = self.encode(tensor)
            quantized[name] = q
            scales[name] = s
        return quantized, scales

    def decode_dict(
        self,
        quantized: dict[str, torch.Tensor],
        scales: dict[str, float],
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Dequantize multiple tensors.

        Args:
            quantized: Dict of quantized tensors.
            scales: Dict of scale factors.
            dtype: Target dtype.

        Returns:
            Dict of dequantized tensors.
        """
        tensors = {}
        for name, q in quantized.items():
            tensors[name] = self.decode(q, scales[name], dtype)
        return tensors
