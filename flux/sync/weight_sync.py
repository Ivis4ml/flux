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


class ColocatedWeightSync:
    """Weight sync for same-machine inference engines.

    Uses CUDA IPC for efficient GPU-to-GPU tensor transfer
    without going through CPU memory.

    Example:
        sync = ColocatedWeightSync(
            weight_provider=model.state_dict,
            engine_handles=[(engine_id, ipc_handle), ...],
        )

        # Update weights
        sync.sync_weights()
    """

    def __init__(
        self,
        weight_provider: Callable[[], dict[str, torch.Tensor]] | None = None,
        use_cuda_ipc: bool = True,
    ) -> None:
        """Initialize colocated weight sync.

        Args:
            weight_provider: Function returning current weights.
            use_cuda_ipc: Whether to use CUDA IPC (falls back to CPU if False).
        """
        self._weight_provider = weight_provider
        self._use_cuda_ipc = use_cuda_ipc and torch.cuda.is_available()

        # Registered engines
        self._engines: dict[str, Any] = {}  # engine_id -> handle

        # Version tracking
        self._version = 0

    def register_engine(self, engine_id: str, handle: Any) -> None:
        """Register an inference engine for weight sync.

        Args:
            engine_id: Unique identifier for the engine.
            handle: IPC handle or reference to the engine.
        """
        self._engines[engine_id] = handle

    def unregister_engine(self, engine_id: str) -> None:
        """Unregister an inference engine.

        Args:
            engine_id: Engine identifier to remove.
        """
        self._engines.pop(engine_id, None)

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
                if self._use_cuda_ipc:
                    self._sync_via_ipc(handle, weights)
                else:
                    self._sync_via_cpu(handle, weights)
                synced += 1
            except Exception as e:
                logger.error(f"Failed to sync to engine {engine_id}: {e}")

        return synced

    def _sync_via_ipc(
        self, handle: Any, weights: dict[str, torch.Tensor]
    ) -> None:
        """Sync weights using CUDA IPC.

        Note: Actual implementation depends on engine interface.
        """
        # This is a placeholder for CUDA IPC implementation
        # Real implementation would use torch.cuda.ipc_collect/reconstruct
        raise NotImplementedError("CUDA IPC sync not implemented")

    def _sync_via_cpu(
        self, handle: Any, weights: dict[str, torch.Tensor]
    ) -> None:
        """Sync weights via CPU memory.

        Note: Actual implementation depends on engine interface.
        """
        # This is a placeholder for CPU-based sync
        # Real implementation would serialize and transfer
        raise NotImplementedError("CPU sync not implemented")
