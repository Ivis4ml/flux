"""
CUDA IPC utilities for zero-copy tensor transfer.

Provides utilities for sharing GPU tensors between processes
on the same machine without copying through CPU memory.

CUDA IPC (Inter-Process Communication) allows sharing GPU memory
between processes via IPC handles.
"""

from __future__ import annotations

import logging
import os
import struct
from dataclasses import dataclass, field
from typing import Any

import torch


logger = logging.getLogger(__name__)


@dataclass
class TensorMetadata:
    """Metadata for a tensor to be transferred."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: int  # CUDA device index
    numel: int
    nbytes: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "device": self.device,
            "numel": self.numel,
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorMetadata":
        """Create from dictionary."""
        dtype_map = {
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.int64": torch.int64,
            "torch.int32": torch.int32,
            "torch.int8": torch.int8,
            "torch.uint8": torch.uint8,
            "torch.bool": torch.bool,
        }
        return cls(
            name=data["name"],
            shape=tuple(data["shape"]),
            dtype=dtype_map.get(data["dtype"], torch.float32),
            device=data["device"],
            numel=data["numel"],
            nbytes=data["nbytes"],
        )


@dataclass
class IPCHandle:
    """Handle for CUDA IPC tensor sharing."""

    handle: bytes  # CUDA IPC memory handle
    metadata: TensorMetadata
    event_handle: bytes | None = None  # Optional sync event

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        import base64
        return {
            "handle": base64.b64encode(self.handle).decode("utf-8"),
            "metadata": self.metadata.to_dict(),
            "event_handle": (
                base64.b64encode(self.event_handle).decode("utf-8")
                if self.event_handle
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IPCHandle":
        """Create from dictionary."""
        import base64
        return cls(
            handle=base64.b64decode(data["handle"]),
            metadata=TensorMetadata.from_dict(data["metadata"]),
            event_handle=(
                base64.b64decode(data["event_handle"])
                if data.get("event_handle")
                else None
            ),
        )


def is_cuda_available() -> bool:
    """Check if CUDA is available for IPC."""
    return torch.cuda.is_available()


def get_device_count() -> int:
    """Get number of CUDA devices."""
    if not is_cuda_available():
        return 0
    return torch.cuda.device_count()


def get_ipc_handle(tensor: torch.Tensor) -> IPCHandle:
    """Get IPC handle for a CUDA tensor.

    The tensor must be on CUDA and contiguous.

    Args:
        tensor: The tensor to share.

    Returns:
        IPCHandle that can be used to reconstruct the tensor.

    Raises:
        ValueError: If tensor is not on CUDA or not contiguous.
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA device")
    if not tensor.is_contiguous():
        raise ValueError("Tensor must be contiguous")

    # Get the IPC handle
    storage = tensor.storage()
    handle = storage._share_cuda_()[1]  # Returns (device, handle, size, offset)

    # Create metadata
    metadata = TensorMetadata(
        name="",  # Will be set by caller
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device.index or 0,
        numel=tensor.numel(),
        nbytes=tensor.element_size() * tensor.numel(),
    )

    return IPCHandle(handle=handle, metadata=metadata)


def open_ipc_handle(handle: IPCHandle, device: int | None = None) -> torch.Tensor:
    """Reconstruct a tensor from an IPC handle.

    Args:
        handle: The IPC handle from get_ipc_handle.
        device: Target CUDA device (uses handle's device if None).

    Returns:
        Reconstructed tensor.
    """
    device = device if device is not None else handle.metadata.device

    # Reconstruct storage from IPC handle
    storage = torch.cuda.storage._UntypedStorage._new_shared_cuda(
        device,
        handle.handle,
        handle.metadata.nbytes,
        0,  # offset
    )

    # Create tensor from storage
    tensor = torch.tensor([], dtype=handle.metadata.dtype, device=f"cuda:{device}")
    tensor.set_(storage, 0, handle.metadata.shape)

    return tensor


def share_tensor_dict(
    tensors: dict[str, torch.Tensor],
    device: int | None = None,
) -> dict[str, IPCHandle]:
    """Share multiple tensors via IPC.

    Args:
        tensors: Dict of name -> tensor.
        device: Target device (uses tensor's device if None).

    Returns:
        Dict of name -> IPCHandle.
    """
    handles = {}
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            # Move to CUDA if needed
            target_device = device if device is not None else 0
            tensor = tensor.to(f"cuda:{target_device}")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        handle = get_ipc_handle(tensor)
        handle.metadata.name = name
        handles[name] = handle

    return handles


def reconstruct_tensor_dict(
    handles: dict[str, IPCHandle],
    device: int | None = None,
) -> dict[str, torch.Tensor]:
    """Reconstruct tensors from IPC handles.

    Args:
        handles: Dict of name -> IPCHandle.
        device: Target device (uses handle's device if None).

    Returns:
        Dict of name -> tensor.
    """
    tensors = {}
    for name, handle in handles.items():
        tensors[name] = open_ipc_handle(handle, device)
    return tensors


class TensorBucket:
    """Bucket for batching multiple small tensors into one transfer.

    Flattens multiple named tensors into a single contiguous buffer
    for more efficient IPC transfer.

    Example:
        bucket = TensorBucket()
        bucket.add("layer1.weight", tensor1)
        bucket.add("layer1.bias", tensor2)

        # Get single flattened buffer
        flat_tensor, metadata = bucket.flatten()

        # Transfer via IPC
        handle = get_ipc_handle(flat_tensor)

        # Reconstruct on receiving end
        reconstructed = bucket.unflatten(flat_tensor, metadata)
    """

    def __init__(self, device: int = 0) -> None:
        """Initialize the tensor bucket.

        Args:
            device: CUDA device for the bucket.
        """
        self.device = device
        self._tensors: dict[str, torch.Tensor] = {}
        self._flattened: torch.Tensor | None = None
        self._metadata: list[TensorMetadata] = []

    def add(self, name: str, tensor: torch.Tensor) -> None:
        """Add a tensor to the bucket.

        Args:
            name: Tensor name.
            tensor: The tensor to add.
        """
        self._tensors[name] = tensor
        self._flattened = None  # Invalidate cache

    def add_dict(self, tensors: dict[str, torch.Tensor]) -> None:
        """Add multiple tensors.

        Args:
            tensors: Dict of name -> tensor.
        """
        for name, tensor in tensors.items():
            self.add(name, tensor)

    def flatten(self) -> tuple[torch.Tensor, list[TensorMetadata]]:
        """Flatten all tensors into a single buffer.

        Returns:
            Tuple of (flattened tensor, metadata list).
        """
        if self._flattened is not None:
            return self._flattened, self._metadata

        # Calculate total size
        total_numel = sum(t.numel() for t in self._tensors.values())

        # Use float32 as common dtype for the buffer
        self._flattened = torch.empty(
            total_numel,
            dtype=torch.float32,
            device=f"cuda:{self.device}",
        )

        # Copy tensors into buffer
        self._metadata = []
        offset = 0

        for name, tensor in self._tensors.items():
            numel = tensor.numel()
            flat_view = tensor.view(-1).to(torch.float32)
            self._flattened[offset : offset + numel] = flat_view

            self._metadata.append(
                TensorMetadata(
                    name=name,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    device=self.device,
                    numel=numel,
                    nbytes=tensor.element_size() * numel,
                )
            )
            offset += numel

        return self._flattened, self._metadata

    def unflatten(
        self,
        flat_tensor: torch.Tensor,
        metadata: list[TensorMetadata],
    ) -> dict[str, torch.Tensor]:
        """Unflatten a buffer back into individual tensors.

        Args:
            flat_tensor: The flattened buffer.
            metadata: Metadata describing the tensors.

        Returns:
            Dict of name -> tensor.
        """
        tensors = {}
        offset = 0

        for meta in metadata:
            chunk = flat_tensor[offset : offset + meta.numel]
            tensor = chunk.view(meta.shape).to(meta.dtype)
            tensors[meta.name] = tensor
            offset += meta.numel

        return tensors

    def clear(self) -> None:
        """Clear the bucket."""
        self._tensors.clear()
        self._flattened = None
        self._metadata.clear()

    def __len__(self) -> int:
        """Number of tensors in bucket."""
        return len(self._tensors)


class IPCServer:
    """Server for handling IPC tensor requests.

    Listens for tensor requests and provides IPC handles
    for efficient tensor sharing.

    Note: This is a simplified implementation. Production use
    would require proper synchronization and error handling.
    """

    def __init__(self, device: int = 0) -> None:
        """Initialize the IPC server.

        Args:
            device: CUDA device for tensors.
        """
        self.device = device
        self._tensors: dict[str, torch.Tensor] = {}
        self._handles: dict[str, IPCHandle] = {}

    def register_tensor(self, name: str, tensor: torch.Tensor) -> IPCHandle:
        """Register a tensor for sharing.

        Args:
            name: Name for the tensor.
            tensor: The tensor to share.

        Returns:
            IPC handle for the tensor.
        """
        # Ensure tensor is on correct device and contiguous
        if not tensor.is_cuda or tensor.device.index != self.device:
            tensor = tensor.to(f"cuda:{self.device}")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        self._tensors[name] = tensor
        handle = get_ipc_handle(tensor)
        handle.metadata.name = name
        self._handles[name] = handle

        return handle

    def get_handle(self, name: str) -> IPCHandle | None:
        """Get IPC handle for a tensor.

        Args:
            name: Tensor name.

        Returns:
            IPCHandle or None if not found.
        """
        return self._handles.get(name)

    def update_tensor(self, name: str, tensor: torch.Tensor) -> IPCHandle | None:
        """Update a registered tensor with new data.

        Note: This copies data into the existing buffer to preserve
        the IPC handle. Shape must match.

        Args:
            name: Tensor name.
            tensor: New tensor data.

        Returns:
            Updated handle or None if name not found.
        """
        if name not in self._tensors:
            return None

        existing = self._tensors[name]
        if existing.shape != tensor.shape:
            raise ValueError(f"Shape mismatch: {existing.shape} vs {tensor.shape}")

        existing.copy_(tensor)
        return self._handles.get(name)

    def unregister_tensor(self, name: str) -> None:
        """Unregister a tensor.

        Args:
            name: Tensor name.
        """
        self._tensors.pop(name, None)
        self._handles.pop(name, None)

    def clear(self) -> None:
        """Clear all registered tensors."""
        self._tensors.clear()
        self._handles.clear()


class IPCClient:
    """Client for receiving tensors via IPC.

    Connects to an IPC server and reconstructs tensors
    from IPC handles.
    """

    def __init__(self, device: int = 0) -> None:
        """Initialize the IPC client.

        Args:
            device: Target CUDA device for tensors.
        """
        self.device = device
        self._tensors: dict[str, torch.Tensor] = {}

    def receive_tensor(self, handle: IPCHandle) -> torch.Tensor:
        """Receive a tensor from an IPC handle.

        Args:
            handle: The IPC handle.

        Returns:
            Reconstructed tensor.
        """
        tensor = open_ipc_handle(handle, self.device)
        name = handle.metadata.name
        if name:
            self._tensors[name] = tensor
        return tensor

    def receive_dict(
        self, handles: dict[str, IPCHandle]
    ) -> dict[str, torch.Tensor]:
        """Receive multiple tensors.

        Args:
            handles: Dict of name -> handle.

        Returns:
            Dict of name -> tensor.
        """
        return reconstruct_tensor_dict(handles, self.device)

    def get_tensor(self, name: str) -> torch.Tensor | None:
        """Get a previously received tensor.

        Args:
            name: Tensor name.

        Returns:
            Tensor or None if not found.
        """
        return self._tensors.get(name)

    def clear(self) -> None:
        """Clear received tensors."""
        self._tensors.clear()
