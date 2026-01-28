"""
Weight synchronization manager for Flux.

Coordinates weight updates between training and inference engines.
Supports both colocated (same machine, CUDA IPC) and distributed
(cross-machine, NCCL) weight synchronization.

Based on Slime's dual-path weight sync with Flux-specific enhancements.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import torch

from flux.core.config import WeightSyncConfig
from flux.core.types import PolicyVersion
from flux.sync.delta_compression import (
    CompressedDelta,
    QuantizedEncoder,
    SnapshotManager,
    compute_weight_delta,
)


logger = logging.getLogger(__name__)


class SyncStrategy(str, Enum):
    """Weight synchronization strategy."""

    FULL = "full"        # Full weight transfer
    DELTA = "delta"      # Only changed weights
    LAZY = "lazy"        # Sync only when needed


class SyncStatus(str, Enum):
    """Status of a sync operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SyncRequest:
    """Request to sync weights to a server."""

    request_id: str
    server_id: str
    version: int
    strategy: SyncStrategy
    created_at: datetime = field(default_factory=datetime.now)
    status: SyncStatus = SyncStatus.PENDING
    completed_at: datetime | None = None
    error: str | None = None


@dataclass
class SyncMetrics:
    """Metrics for weight synchronization."""

    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    bytes_transferred: int = 0
    mean_sync_time_ms: float = 0.0
    last_sync_time_ms: float = 0.0
    last_sync_version: int = 0


class WeightSyncManager:
    """Manages weight synchronization to inference servers.

    Features:
    - Lazy sync: Only sync when weights have been updated
    - Version tracking: Ensures servers have correct weight version
    - Delta compression: Only transfer changed weights (optional)
    - Batch sync: Update multiple servers concurrently
    - Colocated optimization: Use CUDA IPC for same-machine transfers

    Example:
        manager = WeightSyncManager(
            config=WeightSyncConfig(method="delta"),
            server_urls=["http://localhost:8000", "http://localhost:8001"],
        )

        # Mark weights as updated
        manager.mark_updated()

        # Sync to all servers
        await manager.sync_all()

        # Or sync to specific server
        await manager.sync_server("http://localhost:8000")
    """

    def __init__(
        self,
        config: WeightSyncConfig | None = None,
        server_urls: list[str] | None = None,
        weight_provider: Callable[[], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        """Initialize the weight sync manager.

        Args:
            config: Weight synchronization configuration.
            server_urls: List of inference server URLs.
            weight_provider: Function that returns current model weights.
        """
        self.config = config or WeightSyncConfig()
        self._server_urls = server_urls or []
        self._weight_provider = weight_provider

        # Version tracking
        self._current_version = 0
        self._server_versions: dict[str, int] = {}  # server_url -> version
        self._weights_dirty = False

        # Snapshots for delta computation (legacy)
        self._snapshots: dict[int, dict[str, torch.Tensor]] = {}
        self._snapshot_versions: list[int] = []

        # Snapshot manager for baseline tracking
        self._snapshot_manager = SnapshotManager(
            max_snapshots=self.config.max_snapshots,
            snapshot_interval=self.config.snapshot_interval,
        )

        # Current baseline for delta computation
        self._baseline: dict[str, torch.Tensor] | None = None
        self._baseline_version: int | None = None

        # Pending sync requests
        self._pending_syncs: dict[str, SyncRequest] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Metrics
        self._metrics = SyncMetrics()

        # HTTP client for sync
        self._http_client: Any = None

        # Initialize server versions
        for url in self._server_urls:
            self._server_versions[url] = 0

    @property
    def current_version(self) -> int:
        """Current weight version."""
        return self._current_version

    @property
    def is_dirty(self) -> bool:
        """Whether weights have been updated since last sync."""
        return self._weights_dirty

    def _use_delta_compression(self) -> bool:
        """Check if delta compression should be used.

        Supports both old (use_delta_compression) and new (method) config styles.
        """
        # New style: check method field
        if hasattr(self.config, 'method'):
            method = self.config.method
            if isinstance(method, Enum):
                method = method.value
            return method == "delta"
        # Old style: check use_delta_compression field
        if hasattr(self.config, 'use_delta_compression'):
            return self.config.use_delta_compression
        return False

    def get_baseline(self) -> dict[str, torch.Tensor] | None:
        """Get current baseline weights for delta computation.

        Returns:
            Baseline weights or None if not set.
        """
        with self._lock:
            return self._baseline

    def get_baseline_version(self) -> int | None:
        """Get the version of the current baseline weights."""
        with self._lock:
            return self._baseline_version

    def set_baseline(self, weights: dict[str, torch.Tensor], version: int) -> None:
        """Set baseline weights for delta computation.

        Args:
            weights: Model state dict to use as baseline.
            version: Version number for this baseline.
        """
        with self._lock:
            # Clone to CPU to avoid GPU memory issues
            self._baseline = {k: v.clone().cpu() for k, v in weights.items()}
            self._baseline_version = version
            # Also take a snapshot for versioned delta
            self._snapshot_manager.take_snapshot(weights, version)

    def serialize_for_sync(
        self,
        weights: dict[str, torch.Tensor],
        version: int | None = None,
    ) -> bytes:
        """Serialize weights for sync based on configured method.

        This is the centralized serialization point. It handles:
        - Full: Complete state_dict serialized to bytes
        - Delta: Only changed parameters since baseline
        - Per-tensor: Individual tensors with metadata

        The output is always bytes suitable for HTTP transport.

        Args:
            weights: Current model state dict.
            version: Weight version (uses current if not provided).

        Returns:
            Serialized bytes ready for HTTP transport.
        """
        import io

        version = version if version is not None else self._current_version
        method = getattr(self.config, 'method', 'full')
        if isinstance(method, Enum):
            method = method.value

        weights_cpu = {k: v.detach().cpu() for k, v in weights.items()}
        payload: dict[str, Any] = {
            "version": version,
            "method": method,
        }

        if method == "delta":
            # Delta compression - only changed parameters
            baseline = self.get_baseline()
            baseline_version = self.get_baseline_version()
            if baseline is not None and baseline_version is not None and baseline_version < version:
                delta = compute_weight_delta(
                    baseline=baseline,
                    current=weights_cpu,
                    from_version=baseline_version,
                    to_version=version,
                    threshold=self.config.sparsity_threshold,
                )
                payload["weights"] = delta.changed_params
                payload["from_version"] = delta.from_version
                payload["baseline_version"] = baseline_version
                payload["stats"] = {
                    "changed_params": delta.stats.changed_params,
                    "total_params": delta.stats.total_params,
                    "compression_ratio": delta.stats.compression_ratio,
                }
                logger.debug(
                    f"Delta sync: {delta.stats.changed_params}/{delta.stats.total_params} "
                    f"params changed, {delta.stats.compression_ratio:.1%} ratio"
                )
            else:
                # No baseline, fall back to full
                payload["weights"] = weights_cpu
                payload["method"] = "full"
                logger.debug("No baseline for delta, using full sync")

        elif method == "per_tensor":
            # Per-tensor with shapes and dtypes for validation
            tensor_meta = {}
            for name, tensor in weights_cpu.items():
                tensor_meta[name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                }
            payload["weights"] = weights_cpu
            payload["tensor_meta"] = tensor_meta

        else:
            # Full state_dict (default)
            payload["weights"] = weights_cpu

        # Apply quantization if configured
        if getattr(self.config, 'quantize', False):
            bits = getattr(self.config, 'quantize_bits', 16)
            encoder = QuantizedEncoder(bits=bits)
            if "weights" in payload and isinstance(payload["weights"], dict):
                quantized, scales = encoder.encode_dict(payload["weights"])
                payload["weights"] = quantized
                payload["quantization"] = {
                    "bits": bits,
                    "scales": scales,
                }
                if "tensor_meta" in payload:
                    for name, meta in payload["tensor_meta"].items():
                        meta["orig_dtype"] = meta.get("dtype")
                        meta["dtype"] = str(quantized[name].dtype)
                logger.debug(f"Applied {bits}-bit quantization")

        # Serialize to bytes using torch.save
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    def add_server(self, server_url: str) -> None:
        """Add a server to sync to.

        Args:
            server_url: URL of the inference server.
        """
        with self._lock:
            if server_url not in self._server_versions:
                self._server_urls.append(server_url)
                self._server_versions[server_url] = 0

    def remove_server(self, server_url: str) -> None:
        """Remove a server from sync list.

        Args:
            server_url: URL of the server to remove.
        """
        with self._lock:
            if server_url in self._server_urls:
                self._server_urls.remove(server_url)
            self._server_versions.pop(server_url, None)

    def mark_updated(self) -> int:
        """Mark weights as updated after training step.

        Returns:
            New version number.
        """
        with self._lock:
            self._current_version += 1
            self._weights_dirty = True
            self._metrics.last_sync_version = self._current_version

            # Take snapshot if needed for delta compression
            if self._use_delta_compression():
                self._maybe_take_snapshot()

            return self._current_version

    def _maybe_take_snapshot(self) -> None:
        """Take weight snapshot if at snapshot interval."""
        if (
            self._weight_provider is not None
            and self._current_version % self.config.snapshot_interval == 0
        ):
            weights = self._weight_provider()
            # Deep copy to avoid reference issues
            snapshot = {k: v.clone() for k, v in weights.items()}
            self._snapshots[self._current_version] = snapshot
            self._snapshot_versions.append(self._current_version)

            # Clean up old snapshots
            while len(self._snapshot_versions) > self.config.max_snapshots:
                old_version = self._snapshot_versions.pop(0)
                self._snapshots.pop(old_version, None)

    def needs_sync(self, server_url: str | None = None) -> bool:
        """Check if any server needs sync.

        Args:
            server_url: Specific server to check (checks all if None).

        Returns:
            True if sync is needed.
        """
        if not self._weights_dirty:
            return False

        with self._lock:
            if server_url:
                server_version = self._server_versions.get(server_url, 0)
                return server_version < self._current_version

            # Check all servers
            for url in self._server_urls:
                if self._server_versions.get(url, 0) < self._current_version:
                    return True

        return False

    def get_stale_servers(self) -> list[str]:
        """Get list of servers that need sync.

        Returns:
            List of server URLs that are behind current version.
        """
        with self._lock:
            stale = []
            for url in self._server_urls:
                if self._server_versions.get(url, 0) < self._current_version:
                    stale.append(url)
            return stale

    async def sync_server(
        self,
        server_url: str,
        strategy: SyncStrategy | None = None,
        force: bool = False,
    ) -> bool:
        """Sync weights to a specific server.

        Args:
            server_url: URL of the server to sync.
            strategy: Sync strategy (uses config default if None).
            force: If True, sync even if server is up to date.

        Returns:
            True if sync was successful.
        """
        with self._lock:
            server_version = self._server_versions.get(server_url, 0)
            if not force and server_version >= self._current_version:
                return True  # Already up to date

        strategy = strategy or SyncStrategy.FULL
        if self._use_delta_compression():
            strategy = SyncStrategy.DELTA

        start_time = time.time()

        try:
            # Get weight data
            if self._weight_provider is None:
                raise RuntimeError("No weight provider configured")

            weights = self._weight_provider()

            # Compute delta if using delta strategy
            if strategy == SyncStrategy.DELTA:
                weights = self._compute_delta(weights, server_version)

            # Serialize weights
            weight_data = self._serialize_weights(weights)

            # Send to server
            success = await self._send_weights(server_url, weight_data)

            if success:
                with self._lock:
                    self._server_versions[server_url] = self._current_version
                    self._metrics.successful_syncs += 1
                    self._metrics.bytes_transferred += len(weight_data)
            else:
                self._metrics.failed_syncs += 1

            # Update metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.total_syncs += 1
            self._metrics.last_sync_time_ms = elapsed_ms
            self._update_mean_sync_time(elapsed_ms)

            return success

        except Exception as e:
            logger.error(f"Failed to sync weights to {server_url}: {e}")
            self._metrics.failed_syncs += 1
            return False

    async def sync_all(
        self,
        strategy: SyncStrategy | None = None,
        concurrent: bool = True,
    ) -> dict[str, bool]:
        """Sync weights to all servers.

        Args:
            strategy: Sync strategy to use.
            concurrent: If True, sync to all servers concurrently.

        Returns:
            Dict mapping server URL to success status.
        """
        stale_servers = self.get_stale_servers()
        if not stale_servers:
            return {}

        results: dict[str, bool] = {}

        if concurrent:
            tasks = [
                self.sync_server(url, strategy)
                for url in stale_servers
            ]
            sync_results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, result in zip(stale_servers, sync_results):
                if isinstance(result, Exception):
                    results[url] = False
                else:
                    results[url] = result
        else:
            for url in stale_servers:
                results[url] = await self.sync_server(url, strategy)

        # Mark as not dirty if all syncs succeeded
        with self._lock:
            if all(results.values()):
                self._weights_dirty = False

        return results

    def _compute_delta(
        self,
        current_weights: dict[str, torch.Tensor],
        from_version: int,
    ) -> dict[str, torch.Tensor]:
        """Compute delta from snapshot to current weights.

        Args:
            current_weights: Current model weights.
            from_version: Version to compute delta from.

        Returns:
            Dict with only changed weights (as deltas).
        """
        # Find nearest snapshot
        snapshot_version = None
        for v in reversed(self._snapshot_versions):
            if v <= from_version:
                snapshot_version = v
                break

        if snapshot_version is None or snapshot_version not in self._snapshots:
            # No snapshot available, return full weights
            return current_weights

        snapshot = self._snapshots[snapshot_version]
        delta = {}

        for name, current in current_weights.items():
            if name not in snapshot:
                # New parameter, include full tensor
                delta[name] = current
            else:
                diff = current - snapshot[name]
                # Only include if changed significantly
                if diff.abs().max() > self.config.sparsity_threshold:
                    delta[name] = diff

        return delta

    def _serialize_weights(self, weights: dict[str, torch.Tensor]) -> bytes:
        """Serialize weights to bytes for transfer.

        Args:
            weights: Dict of parameter name to tensor.

        Returns:
            Serialized bytes.
        """
        import io
        buffer = io.BytesIO()
        torch.save(weights, buffer)
        return buffer.getvalue()

    async def _send_weights(self, server_url: str, weight_data: bytes) -> bool:
        """Send weight data to server via HTTP.

        Args:
            server_url: Target server URL.
            weight_data: Serialized weight data.

        Returns:
            True if successful.
        """
        try:
            import httpx

            if self._http_client is None:
                self._http_client = httpx.AsyncClient(timeout=60.0)

            endpoint = f"{server_url}/update_weights"
            payload = {
                "weights": base64.b64encode(weight_data).decode("utf-8"),
                "weight_version": str(self._current_version),
            }

            response = await self._http_client.post(endpoint, json=payload)
            response.raise_for_status()
            return True

        except ImportError:
            logger.error("httpx not installed, cannot sync weights via HTTP")
            return False
        except Exception as e:
            logger.error(f"HTTP weight sync failed: {e}")
            return False

    def _update_mean_sync_time(self, new_time_ms: float) -> None:
        """Update running mean of sync time."""
        if self._metrics.total_syncs == 1:
            self._metrics.mean_sync_time_ms = new_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self._metrics.mean_sync_time_ms = (
                alpha * new_time_ms
                + (1 - alpha) * self._metrics.mean_sync_time_ms
            )

    def get_server_version(self, server_url: str) -> int:
        """Get the weight version on a specific server.

        Args:
            server_url: Server URL.

        Returns:
            Version number on that server.
        """
        with self._lock:
            return self._server_versions.get(server_url, 0)

    def get_metrics(self) -> SyncMetrics:
        """Get sync metrics.

        Returns:
            SyncMetrics object.
        """
        return SyncMetrics(
            total_syncs=self._metrics.total_syncs,
            successful_syncs=self._metrics.successful_syncs,
            failed_syncs=self._metrics.failed_syncs,
            bytes_transferred=self._metrics.bytes_transferred,
            mean_sync_time_ms=self._metrics.mean_sync_time_ms,
            last_sync_time_ms=self._metrics.last_sync_time_ms,
            last_sync_version=self._current_version,
        )

    async def close(self) -> None:
        """Close the sync manager and release resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


@dataclass
class EngineHandle:
    """Handle for a registered inference engine.

    Supports different sync protocols:
    - ipc: Direct CUDA IPC tensor sharing (fastest, same machine)
    - shm: Shared memory with CPU serialization
    - callback: Custom callback function for weight updates
    """

    engine_id: str
    protocol: str = "ipc"  # "ipc", "shm", or "callback"
    device: int = 0  # Target CUDA device
    shared_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    callback: Callable[[dict[str, torch.Tensor], int], None] | None = None
    shm_path: str | None = None  # Path for shared memory file


class ColocatedWeightSync:
    """Weight sync for same-machine inference engines.

    Uses CUDA IPC for efficient GPU-to-GPU tensor transfer
    without going through CPU memory. Falls back to CPU shared
    memory when IPC is not available.

    Sync Protocols:
    1. **IPC (CUDA Inter-Process Communication)**:
       - Zero-copy GPU-to-GPU transfer
       - Requires pre-allocated shared buffers
       - Best for same-machine, multi-process setups

    2. **SHM (Shared Memory)**:
       - CPU-based shared memory file
       - Works across processes without CUDA IPC support
       - Higher latency but more compatible

    3. **Callback**:
       - Custom callback for in-process weight updates
       - Used when engine is in same process (e.g., SGLang subprocess)

    Example:
        sync = ColocatedWeightSync(weight_provider=model.state_dict)

        # Register engine with IPC protocol
        sync.register_engine(
            engine_id="sglang-0",
            protocol="ipc",
            device=0,
        )

        # Or with callback for in-process updates
        sync.register_engine(
            engine_id="local",
            protocol="callback",
            callback=engine.update_weights,
        )

        # Sync weights
        synced = sync.sync_weights()
    """

    def __init__(
        self,
        weight_provider: Callable[[], dict[str, torch.Tensor]] | None = None,
        use_cuda_ipc: bool = True,
        default_device: int = 0,
    ) -> None:
        """Initialize colocated weight sync.

        Args:
            weight_provider: Function returning current weights.
            use_cuda_ipc: Whether to use CUDA IPC (falls back to CPU if False).
            default_device: Default CUDA device for new engines.
        """
        self._weight_provider = weight_provider
        self._use_cuda_ipc = use_cuda_ipc and torch.cuda.is_available()
        self._default_device = default_device

        # Registered engines
        self._engines: dict[str, EngineHandle] = {}

        # Version tracking
        self._version = 0

        # Shared buffer initialization state
        self._buffers_initialized = False

        # Metrics
        self._sync_count = 0
        self._total_bytes_synced = 0

    @property
    def version(self) -> int:
        """Current weight version."""
        return self._version

    def register_engine(
        self,
        engine_id: str,
        protocol: str = "ipc",
        device: int | None = None,
        callback: Callable[[dict[str, torch.Tensor], int], None] | None = None,
        shm_path: str | None = None,
    ) -> EngineHandle:
        """Register an inference engine for weight sync.

        Args:
            engine_id: Unique identifier for the engine.
            protocol: Sync protocol ("ipc", "shm", or "callback").
            device: Target CUDA device (uses default if None).
            callback: Callback function for "callback" protocol.
            shm_path: Shared memory path for "shm" protocol.

        Returns:
            The created EngineHandle.

        Raises:
            ValueError: If protocol is invalid or required args missing.
        """
        if protocol not in ("ipc", "shm", "callback"):
            raise ValueError(f"Invalid protocol: {protocol}")
        if protocol == "callback" and callback is None:
            raise ValueError("callback protocol requires callback function")
        if protocol == "shm" and shm_path is None:
            shm_path = f"/dev/shm/flux_weights_{engine_id}"

        handle = EngineHandle(
            engine_id=engine_id,
            protocol=protocol,
            device=device if device is not None else self._default_device,
            callback=callback,
            shm_path=shm_path,
        )
        self._engines[engine_id] = handle
        return handle

    def unregister_engine(self, engine_id: str) -> None:
        """Unregister an inference engine.

        Args:
            engine_id: Engine identifier to remove.
        """
        if engine_id in self._engines:
            handle = self._engines.pop(engine_id)
            # Clean up shared buffers
            handle.shared_buffers.clear()

    def sync_weights(self, engine_ids: list[str] | None = None) -> int:
        """Sync weights to registered engines.

        Args:
            engine_ids: Specific engines to sync (all if None).

        Returns:
            Number of engines synced.
        """
        if self._weight_provider is None:
            raise RuntimeError("No weight provider configured")

        weights = self._weight_provider()
        self._version += 1

        target_engines = engine_ids or list(self._engines.keys())
        synced = 0

        for engine_id in target_engines:
            if engine_id not in self._engines:
                continue

            handle = self._engines[engine_id]

            try:
                if handle.protocol == "ipc" and self._use_cuda_ipc:
                    self._sync_via_ipc(handle, weights)
                elif handle.protocol == "callback" and handle.callback is not None:
                    self._sync_via_callback(handle, weights)
                else:
                    self._sync_via_cpu(handle, weights)
                synced += 1
                self._sync_count += 1
            except Exception as e:
                logger.error(f"Failed to sync to engine {engine_id}: {e}")

        return synced

    def _init_shared_buffers(
        self, handle: EngineHandle, weights: dict[str, torch.Tensor]
    ) -> None:
        """Initialize shared buffers for IPC sync.

        Pre-allocates GPU memory that can be shared via IPC handles.
        The inference engine can then map these buffers to its address space.
        """
        device = f"cuda:{handle.device}"

        for name, tensor in weights.items():
            if name not in handle.shared_buffers:
                # Allocate contiguous buffer on target device
                buffer = torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device=device,
                )
                handle.shared_buffers[name] = buffer

    def _sync_via_ipc(
        self, handle: EngineHandle, weights: dict[str, torch.Tensor]
    ) -> None:
        """Sync weights using CUDA IPC.

        Uses pre-allocated shared buffers for zero-copy transfer.
        The inference engine reads from these buffers directly.

        Args:
            handle: Engine handle with shared buffers.
            weights: Current model weights.
        """
        # Initialize buffers on first sync
        if not handle.shared_buffers:
            self._init_shared_buffers(handle, weights)

        device = f"cuda:{handle.device}"
        bytes_synced = 0

        # Copy weights to shared buffers
        for name, tensor in weights.items():
            if name in handle.shared_buffers:
                buffer = handle.shared_buffers[name]
                # Ensure shapes match (may differ after model changes)
                if buffer.shape != tensor.shape:
                    buffer = torch.empty(
                        tensor.shape,
                        dtype=tensor.dtype,
                        device=device,
                    )
                    handle.shared_buffers[name] = buffer

                # Copy to shared buffer (non-blocking for efficiency)
                buffer.copy_(tensor.to(device), non_blocking=True)
                bytes_synced += tensor.numel() * tensor.element_size()
            else:
                # New parameter, allocate buffer
                buffer = tensor.to(device).contiguous()
                handle.shared_buffers[name] = buffer
                bytes_synced += tensor.numel() * tensor.element_size()

        # Synchronize to ensure copies complete
        torch.cuda.synchronize(handle.device)

        self._total_bytes_synced += bytes_synced
        logger.debug(
            f"IPC sync to {handle.engine_id}: {bytes_synced / 1024 / 1024:.2f} MB"
        )

    def _sync_via_callback(
        self, handle: EngineHandle, weights: dict[str, torch.Tensor]
    ) -> None:
        """Sync weights via callback function.

        Used for in-process weight updates where the engine
        can directly receive the weight tensors.

        Args:
            handle: Engine handle with callback.
            weights: Current model weights.
        """
        if handle.callback is None:
            raise RuntimeError(f"No callback for engine {handle.engine_id}")

        # Move weights to target device if needed
        device_weights = {}

        if torch.cuda.is_available():
            device = f"cuda:{handle.device}"
            for name, tensor in weights.items():
                if tensor.device.type == "cuda" and tensor.device.index == handle.device:
                    device_weights[name] = tensor
                else:
                    device_weights[name] = tensor.to(device)
        else:
            # CPU-only mode: just use the weights as-is
            for name, tensor in weights.items():
                device_weights[name] = tensor.cpu() if tensor.is_cuda else tensor

        # Call the update callback with weights and version
        handle.callback(device_weights, self._version)

        logger.debug(f"Callback sync to {handle.engine_id}: version {self._version}")

    def _sync_via_cpu(
        self, handle: EngineHandle, weights: dict[str, torch.Tensor]
    ) -> None:
        """Sync weights via CPU shared memory.

        Serializes weights to a shared memory file that can be
        read by other processes.

        Args:
            handle: Engine handle with shm_path.
            weights: Current model weights.
        """
        import io
        import os

        if handle.shm_path is None:
            handle.shm_path = f"/dev/shm/flux_weights_{handle.engine_id}"

        # Serialize weights to CPU
        cpu_weights = {k: v.detach().cpu() for k, v in weights.items()}

        # Create payload with version
        payload = {
            "version": self._version,
            "weights": cpu_weights,
        }

        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()

        # Write to shared memory file
        # Use a temp file and atomic rename for consistency
        tmp_path = f"{handle.shm_path}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.replace(tmp_path, handle.shm_path)
        except OSError:
            # Fall back to regular file if /dev/shm not available
            fallback_path = f"/tmp/flux_weights_{handle.engine_id}"
            with open(fallback_path, "wb") as f:
                f.write(data)
            handle.shm_path = fallback_path
            logger.warning(f"Using fallback path: {fallback_path}")

        self._total_bytes_synced += len(data)
        logger.debug(
            f"SHM sync to {handle.engine_id}: {len(data) / 1024 / 1024:.2f} MB"
        )

    def get_ipc_handles(
        self, engine_id: str
    ) -> dict[str, Any] | None:
        """Get IPC handles for an engine's shared buffers.

        The inference engine can use these handles to map the
        shared GPU memory to its address space.

        Args:
            engine_id: Engine identifier.

        Returns:
            Dict of parameter name to IPC handle, or None if not found.
        """
        from flux.sync.cuda_ipc import get_ipc_handle

        if engine_id not in self._engines:
            return None

        handle = self._engines[engine_id]
        if not handle.shared_buffers:
            return None

        ipc_handles = {}
        for name, tensor in handle.shared_buffers.items():
            if tensor.is_cuda and tensor.is_contiguous():
                ipc_handles[name] = get_ipc_handle(tensor)

        return ipc_handles

    def get_metrics(self) -> dict[str, Any]:
        """Get sync metrics.

        Returns:
            Dict with sync statistics.
        """
        return {
            "version": self._version,
            "sync_count": self._sync_count,
            "total_bytes_synced": self._total_bytes_synced,
            "total_mb_synced": self._total_bytes_synced / 1024 / 1024,
            "num_engines": len(self._engines),
            "use_cuda_ipc": self._use_cuda_ipc,
        }
