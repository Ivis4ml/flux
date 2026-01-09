"""
Tests for weight synchronization and delta compression.
"""

import pytest
import torch

from flux.core.config import WeightSyncConfig
from flux.sync.weight_sync import (
    SyncMetrics,
    SyncRequest,
    SyncStatus,
    SyncStrategy,
    WeightSyncManager,
)
from flux.sync.delta_compression import (
    CompressedDelta,
    DeltaStats,
    QuantizedEncoder,
    SnapshotManager,
    SparseEncoder,
    WeightSnapshot,
    apply_delta,
    compute_weight_delta,
)
from flux.sync.cuda_ipc import (
    TensorBucket,
    TensorMetadata,
    is_cuda_available,
)


class TestSyncStrategy:
    """Tests for SyncStrategy enum."""

    def test_strategy_values(self):
        """Test all strategy values exist."""
        assert SyncStrategy.FULL == "full"
        assert SyncStrategy.DELTA == "delta"
        assert SyncStrategy.LAZY == "lazy"


class TestSyncStatus:
    """Tests for SyncStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert SyncStatus.PENDING == "pending"
        assert SyncStatus.IN_PROGRESS == "in_progress"
        assert SyncStatus.COMPLETED == "completed"
        assert SyncStatus.FAILED == "failed"


class TestWeightSyncManager:
    """Tests for WeightSyncManager."""

    def test_creation_defaults(self):
        """Test manager creation with defaults."""
        manager = WeightSyncManager()
        assert manager.current_version == 0
        assert not manager.is_dirty
        assert not manager.needs_sync()

    def test_creation_with_servers(self):
        """Test manager with server URLs."""
        manager = WeightSyncManager(
            server_urls=["http://localhost:8000", "http://localhost:8001"],
        )
        assert len(manager._server_urls) == 2
        assert manager.get_server_version("http://localhost:8000") == 0

    def test_add_remove_server(self):
        """Test adding and removing servers."""
        manager = WeightSyncManager()
        assert len(manager._server_urls) == 0

        manager.add_server("http://localhost:8000")
        assert len(manager._server_urls) == 1

        manager.remove_server("http://localhost:8000")
        assert len(manager._server_urls) == 0

    def test_mark_updated(self):
        """Test marking weights as updated."""
        manager = WeightSyncManager()
        assert manager.current_version == 0
        assert not manager.is_dirty

        version = manager.mark_updated()
        assert version == 1
        assert manager.current_version == 1
        assert manager.is_dirty

    def test_needs_sync(self):
        """Test needs_sync detection."""
        manager = WeightSyncManager(
            server_urls=["http://localhost:8000"],
        )

        # Initially no sync needed
        assert not manager.needs_sync()

        # After update, sync is needed
        manager.mark_updated()
        assert manager.needs_sync()
        assert manager.needs_sync("http://localhost:8000")

    def test_get_stale_servers(self):
        """Test getting stale server list."""
        manager = WeightSyncManager(
            server_urls=["http://server1:8000", "http://server2:8000"],
        )

        # Initially no stale servers
        assert manager.get_stale_servers() == []

        # After update, all servers are stale
        manager.mark_updated()
        stale = manager.get_stale_servers()
        assert len(stale) == 2

    def test_get_metrics(self):
        """Test getting sync metrics."""
        manager = WeightSyncManager()
        metrics = manager.get_metrics()

        assert isinstance(metrics, SyncMetrics)
        assert metrics.total_syncs == 0
        assert metrics.successful_syncs == 0


class TestSnapshotManager:
    """Tests for SnapshotManager."""

    def test_creation(self):
        """Test snapshot manager creation."""
        manager = SnapshotManager(max_snapshots=5, snapshot_interval=10)
        assert manager.num_snapshots == 0
        assert manager.max_snapshots == 5

    def test_take_snapshot(self):
        """Test taking a snapshot."""
        manager = SnapshotManager()
        weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }

        snapshot = manager.take_snapshot(weights, version=1)
        assert snapshot.version == 1
        assert "layer1.weight" in snapshot.weights
        assert manager.num_snapshots == 1

    def test_get_snapshot(self):
        """Test getting a specific snapshot."""
        manager = SnapshotManager()
        weights = {"weight": torch.randn(5, 5)}

        manager.take_snapshot(weights, version=10)
        manager.take_snapshot(weights, version=20)

        snapshot = manager.get_snapshot(10)
        assert snapshot is not None
        assert snapshot.version == 10

        snapshot = manager.get_snapshot(15)
        assert snapshot is None

    def test_get_nearest_snapshot(self):
        """Test getting nearest snapshot."""
        manager = SnapshotManager()
        weights = {"weight": torch.randn(5, 5)}

        manager.take_snapshot(weights, version=10)
        manager.take_snapshot(weights, version=20)
        manager.take_snapshot(weights, version=30)

        # Nearest to 25 is 20
        snapshot = manager.get_nearest_snapshot(25)
        assert snapshot is not None
        assert snapshot.version == 20

        # Nearest to 35 is 30
        snapshot = manager.get_nearest_snapshot(35)
        assert snapshot.version == 30

    def test_max_snapshots_eviction(self):
        """Test that old snapshots are evicted."""
        manager = SnapshotManager(max_snapshots=3)
        weights = {"weight": torch.randn(5, 5)}

        for i in range(5):
            manager.take_snapshot(weights, version=i * 10)

        assert manager.num_snapshots == 3
        # Oldest (0, 10) should be evicted
        assert manager.get_snapshot(0) is None
        assert manager.get_snapshot(10) is None
        assert manager.get_snapshot(20) is not None

    def test_should_snapshot(self):
        """Test snapshot interval check."""
        manager = SnapshotManager(snapshot_interval=10)

        # No snapshots yet, should take one
        assert manager.should_snapshot(0)

        weights = {"weight": torch.randn(5, 5)}
        manager.take_snapshot(weights, version=0)

        # Not enough steps
        assert not manager.should_snapshot(5)

        # Enough steps
        assert manager.should_snapshot(10)


class TestDeltaComputation:
    """Tests for delta computation."""

    def test_compute_delta_no_change(self):
        """Test delta with no changes."""
        baseline = {"weight": torch.zeros(10, 10)}
        current = {"weight": torch.zeros(10, 10)}

        delta = compute_weight_delta(baseline, current, 0, 1)

        assert len(delta.changed_params) == 0
        assert delta.stats.changed_params == 0

    def test_compute_delta_with_changes(self):
        """Test delta with changes."""
        baseline = {"weight": torch.zeros(10, 10)}
        current = {"weight": torch.ones(10, 10)}

        delta = compute_weight_delta(baseline, current, 0, 1)

        assert len(delta.changed_params) == 1
        assert "weight" in delta.changed_params
        assert delta.stats.changed_params == 1

    def test_compute_delta_new_param(self):
        """Test delta with new parameter."""
        baseline = {"weight": torch.zeros(10, 10)}
        current = {
            "weight": torch.zeros(10, 10),
            "bias": torch.ones(10),
        }

        delta = compute_weight_delta(baseline, current, 0, 1)

        assert "bias" in delta.changed_params

    def test_apply_delta(self):
        """Test applying delta to baseline."""
        baseline = {"weight": torch.zeros(10, 10)}
        delta_tensor = torch.ones(10, 10)

        delta = CompressedDelta(
            from_version=0,
            to_version=1,
            changed_params={"weight": delta_tensor},
        )

        result = apply_delta(baseline, delta)
        assert torch.allclose(result["weight"], torch.ones(10, 10))


class TestSparseEncoder:
    """Tests for SparseEncoder."""

    def test_encode_sparse_tensor(self):
        """Test encoding a sparse tensor."""
        encoder = SparseEncoder(sparsity_threshold=0.5)

        # Create sparse tensor (mostly zeros)
        tensor = torch.zeros(100)
        tensor[10] = 1.0
        tensor[50] = 2.0

        indices, values = encoder.encode(tensor)

        assert len(indices) == 2
        assert len(values) == 2
        assert 10 in indices.tolist()
        assert 50 in indices.tolist()

    def test_decode_sparse_tensor(self):
        """Test decoding a sparse tensor."""
        encoder = SparseEncoder()

        # Encode
        tensor = torch.zeros(100)
        tensor[10] = 1.0
        tensor[50] = 2.0
        indices, values = encoder.encode(tensor)

        # Decode
        decoded = encoder.decode(indices, values, (100,))

        assert decoded[10] == 1.0
        assert decoded[50] == 2.0
        assert decoded[0] == 0.0

    def test_should_encode_sparse(self):
        """Test sparsity threshold check."""
        encoder = SparseEncoder(sparsity_threshold=0.5)

        # Dense tensor (not sparse enough)
        dense = torch.ones(100)
        assert not encoder.should_encode_sparse(dense)

        # Sparse tensor
        sparse = torch.zeros(100)
        sparse[:10] = 1.0  # 10% non-zero = 90% sparse
        assert encoder.should_encode_sparse(sparse)


class TestQuantizedEncoder:
    """Tests for QuantizedEncoder."""

    def test_encode_8bit(self):
        """Test 8-bit quantization."""
        encoder = QuantizedEncoder(bits=8)
        tensor = torch.randn(100)

        quantized, scale = encoder.encode(tensor)

        assert quantized.dtype == torch.int8
        assert scale > 0

    def test_encode_16bit(self):
        """Test 16-bit quantization."""
        encoder = QuantizedEncoder(bits=16)
        tensor = torch.randn(100)

        quantized, scale = encoder.encode(tensor)

        assert quantized.dtype == torch.int16

    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip."""
        encoder = QuantizedEncoder(bits=8)
        tensor = torch.randn(100)

        quantized, scale = encoder.encode(tensor)
        decoded = encoder.decode(quantized, scale)

        # Should be close but not exact due to quantization
        assert decoded.shape == tensor.shape
        # Allow some error from quantization
        assert torch.allclose(decoded, tensor, atol=scale * 2)

    def test_encode_dict(self):
        """Test encoding multiple tensors."""
        encoder = QuantizedEncoder(bits=8)
        tensors = {
            "weight": torch.randn(10, 10),
            "bias": torch.randn(10),
        }

        quantized, scales = encoder.encode_dict(tensors)

        assert "weight" in quantized
        assert "bias" in quantized
        assert "weight" in scales
        assert "bias" in scales


class TestTensorBucket:
    """Tests for TensorBucket."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bucket_add_and_flatten(self):
        """Test adding tensors and flattening."""
        bucket = TensorBucket(device=0)

        bucket.add("weight", torch.randn(10, 10).cuda())
        bucket.add("bias", torch.randn(10).cuda())

        flat, metadata = bucket.flatten()

        assert flat.numel() == 110  # 100 + 10
        assert len(metadata) == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bucket_unflatten(self):
        """Test unflattening back to tensors."""
        bucket = TensorBucket(device=0)

        original_weight = torch.randn(10, 10).cuda()
        original_bias = torch.randn(10).cuda()

        bucket.add("weight", original_weight)
        bucket.add("bias", original_bias)

        flat, metadata = bucket.flatten()
        tensors = bucket.unflatten(flat, metadata)

        assert "weight" in tensors
        assert "bias" in tensors
        assert tensors["weight"].shape == (10, 10)
        assert tensors["bias"].shape == (10,)


class TestTensorMetadata:
    """Tests for TensorMetadata."""

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        metadata = TensorMetadata(
            name="weight",
            shape=(10, 10),
            dtype=torch.float32,
            device=0,
            numel=100,
            nbytes=400,
        )

        data = metadata.to_dict()
        restored = TensorMetadata.from_dict(data)

        assert restored.name == metadata.name
        assert restored.shape == metadata.shape
        assert restored.dtype == metadata.dtype
        assert restored.numel == metadata.numel


class TestWeightSnapshot:
    """Tests for WeightSnapshot."""

    def test_creation(self):
        """Test snapshot creation."""
        weights = {"weight": torch.randn(10, 10)}
        snapshot = WeightSnapshot(version=5, weights=weights)

        assert snapshot.version == 5
        assert "weight" in snapshot.weights
        assert snapshot.created_at > 0


class TestDeltaStats:
    """Tests for DeltaStats."""

    def test_default_values(self):
        """Test default statistics."""
        stats = DeltaStats()
        assert stats.total_params == 0
        assert stats.changed_params == 0
        assert stats.compression_ratio == 0.0


class TestSyncRequest:
    """Tests for SyncRequest."""

    def test_creation(self):
        """Test sync request creation."""
        request = SyncRequest(
            request_id="sync-1",
            server_id="http://localhost:8000",
            version=5,
            strategy=SyncStrategy.DELTA,
        )
        assert request.request_id == "sync-1"
        assert request.status == SyncStatus.PENDING


class TestSyncMetrics:
    """Tests for SyncMetrics."""

    def test_default_values(self):
        """Test default metrics."""
        metrics = SyncMetrics()
        assert metrics.total_syncs == 0
        assert metrics.successful_syncs == 0
        assert metrics.bytes_transferred == 0
