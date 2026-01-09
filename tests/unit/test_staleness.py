"""
Tests for StalenessManager.
"""

import pytest
import torch

from flux.core.config import AdaptiveAsyncConfig
from flux.core.types import BatchMetrics, StalenessMetrics
from flux.controller.staleness import (
    RolloutStats,
    StalenessManager,
    StalenessRecord,
)


class TestRolloutStats:
    """Tests for RolloutStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = RolloutStats()
        assert stats.enqueued == 0
        assert stats.running == 0
        assert stats.accepted == 0
        assert stats.rejected == 0

    def test_total_in_flight(self):
        """Test total_in_flight property."""
        stats = RolloutStats(enqueued=5, running=3, accepted=10, rejected=2)
        assert stats.total_in_flight == 18  # 5 + 3 + 10

    def test_total_submitted(self):
        """Test total_submitted property."""
        stats = RolloutStats(enqueued=5, running=3, accepted=10, rejected=2)
        assert stats.total_submitted == 20  # 5 + 3 + 10 + 2


class TestStalenessMetricsComputation:
    """Tests for staleness metrics computation."""

    def test_compute_combined_default_weights(self):
        """Test combined staleness computation with default weights."""
        metrics = StalenessMetrics(
            kl_divergence=0.05,
            importance_weight_variance=1.0,
            version_gap=2.5,
        )
        combined = metrics.compute_combined()

        # kl_contrib = min(1, 0.05/0.1) = 0.5
        # iw_contrib = min(1, 1.0/2.0) = 0.5
        # version_contrib = min(1, 2.5/5) = 0.5
        # combined = 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5 = 0.5
        assert abs(combined - 0.5) < 0.01

    def test_compute_combined_clamped(self):
        """Test combined staleness is clamped to [0, 1]."""
        metrics = StalenessMetrics(
            kl_divergence=1.0,
            importance_weight_variance=10.0,
            version_gap=20.0,
        )
        combined = metrics.compute_combined()
        # All contributions maxed at 1.0
        assert combined == 1.0

    def test_compute_combined_zero(self):
        """Test combined staleness when all zero."""
        metrics = StalenessMetrics(
            kl_divergence=0.0,
            importance_weight_variance=0.0,
            version_gap=0.0,
        )
        combined = metrics.compute_combined()
        assert combined == 0.0


class TestStalenessManager:
    """Tests for StalenessManager."""

    def test_creation_default(self):
        """Test manager creation with defaults."""
        manager = StalenessManager()
        assert manager.current_staleness == 0.0
        assert manager.steps_since_sync == 0
        assert manager.batch_size == 32

    def test_creation_custom_config(self):
        """Test manager with custom config."""
        config = AdaptiveAsyncConfig(
            target_staleness=0.2,
            kl_normalizer=0.2,
        )
        manager = StalenessManager(config=config, batch_size=64)
        assert manager.config.target_staleness == 0.2
        assert manager.batch_size == 64

    def test_compute_staleness_explicit_values(self):
        """Test computing staleness with explicit values."""
        manager = StalenessManager()
        metrics = manager.compute_staleness(
            kl_divergence=0.05,
            importance_weights=torch.tensor([1.0, 1.1, 0.9, 1.0]),
            version_gap=1.0,
        )

        assert metrics.kl_divergence == 0.05
        assert metrics.version_gap == 1.0
        assert metrics.combined_staleness >= 0.0
        assert metrics.combined_staleness <= 1.0

    def test_compute_staleness_from_batch_metrics(self):
        """Test computing staleness from BatchMetrics."""
        manager = StalenessManager()

        batch_metrics = BatchMetrics(
            mean_kl=0.02,
            mean_importance_weight=1.0,
            max_importance_weight=1.5,
            staleness=StalenessMetrics(version_gap=2.0),
        )

        metrics = manager.compute_staleness(batch_metrics)

        assert metrics.kl_divergence == 0.02
        assert metrics.version_gap == 2.0

    def test_compute_staleness_updates_ema(self):
        """Test that computing staleness updates EMA."""
        config = AdaptiveAsyncConfig(ema_alpha=0.5)
        manager = StalenessManager(config=config)

        initial = manager.current_staleness
        assert initial == 0.0

        # Compute with high staleness
        manager.compute_staleness(kl_divergence=0.1, version_gap=5.0)

        # EMA should have updated
        assert manager.current_staleness > 0.0

    def test_compute_staleness_updates_steps(self):
        """Test that computing staleness increments step counter."""
        manager = StalenessManager()
        assert manager.steps_since_sync == 0

        manager.compute_staleness(kl_divergence=0.01)
        assert manager.steps_since_sync == 1

        manager.compute_staleness(kl_divergence=0.01)
        assert manager.steps_since_sync == 2

    def test_compute_staleness_from_trajectories(self):
        """Test computing staleness directly from trajectory data."""
        version_counter = [5]

        def get_version():
            return version_counter[0]

        manager = StalenessManager(version_provider=get_version)

        current_logprobs = torch.tensor([[-0.1, -0.2, -0.3], [-0.15, -0.25, -0.1]])
        behavior_logprobs = torch.tensor([[-0.12, -0.22, -0.28], [-0.14, -0.24, -0.12]])
        trajectory_versions = [3, 4]
        mask = torch.ones_like(current_logprobs)

        metrics = manager.compute_staleness_from_trajectories(
            current_logprobs=current_logprobs,
            behavior_logprobs=behavior_logprobs,
            trajectory_versions=trajectory_versions,
            mask=mask,
        )

        # Version gap should be (5-3 + 5-4)/2 = 1.5
        assert abs(metrics.version_gap - 1.5) < 0.01
        assert metrics.kl_divergence >= 0.0

    def test_get_capacity(self):
        """Test capacity computation."""
        manager = StalenessManager(batch_size=32)
        manager._max_staleness = 3

        # Initial capacity with version 0
        capacity = manager.get_capacity(current_version=0)
        # capacity = (3 + 0 + 1) * 32 - 0 = 128
        assert capacity == 128

    def test_get_capacity_with_in_flight(self):
        """Test capacity with in-flight rollouts."""
        manager = StalenessManager(batch_size=32)
        manager._max_staleness = 3

        # Add some in-flight rollouts
        manager.on_rollout_enqueued(20)
        manager.on_rollout_submitted(10)  # 10 still enqueued, 10 running
        manager.on_rollout_accepted(5)  # 10 enqueued, 5 running, 5 accepted

        capacity = manager.get_capacity(current_version=0)
        # in_flight = 10 + 5 + 5 = 20
        # capacity = (3 + 0 + 1) * 32 - 20 = 108
        assert capacity == 108

    def test_get_capacity_version_provider(self):
        """Test capacity with version provider."""
        version_counter = [2]

        def get_version():
            return version_counter[0]

        manager = StalenessManager(
            version_provider=get_version,
            batch_size=32,
        )
        manager._max_staleness = 3

        capacity = manager.get_capacity()
        # capacity = (3 + 2 + 1) * 32 - 0 = 192
        assert capacity == 192

    def test_should_sync_staleness_threshold(self):
        """Test should_sync based on staleness threshold."""
        config = AdaptiveAsyncConfig(
            target_staleness=0.1,
            tolerance=0.05,
            ema_alpha=1.0,  # No smoothing for test
        )
        manager = StalenessManager(config=config)

        # Low staleness - should not sync
        manager.compute_staleness(kl_divergence=0.01, version_gap=0.5)
        assert not manager.should_sync()

        # High staleness (> 0.1 + 0.05) - should sync
        manager.compute_staleness(kl_divergence=0.2, version_gap=5.0)
        assert manager.should_sync()

    def test_should_sync_max_steps(self):
        """Test should_sync based on max steps without sync."""
        config = AdaptiveAsyncConfig(
            max_steps_without_sync=3,
            target_staleness=1.0,  # Very high to not trigger on staleness
        )
        manager = StalenessManager(config=config)

        # Steps 1 and 2 should not sync
        manager.compute_staleness(kl_divergence=0.01)
        assert not manager.should_sync()
        manager.compute_staleness(kl_divergence=0.01)
        assert not manager.should_sync()

        # Step 3 should trigger sync
        manager.compute_staleness(kl_divergence=0.01)
        assert manager.should_sync()

    def test_record_sync_resets_counter(self):
        """Test that record_sync resets step counter."""
        manager = StalenessManager()
        manager.compute_staleness(kl_divergence=0.01)
        manager.compute_staleness(kl_divergence=0.01)
        assert manager.steps_since_sync == 2

        manager.record_sync()
        assert manager.steps_since_sync == 0

    def test_rollout_tracking_callbacks(self):
        """Test rollout tracking callbacks."""
        manager = StalenessManager()

        manager.on_rollout_enqueued(5)
        stats = manager.stats
        assert stats.enqueued == 5
        assert stats.running == 0
        assert stats.accepted == 0

        manager.on_rollout_submitted(3)
        stats = manager.stats
        assert stats.enqueued == 2
        assert stats.running == 3
        assert stats.accepted == 0

        manager.on_rollout_accepted(2)
        stats = manager.stats
        assert stats.enqueued == 2
        assert stats.running == 1
        assert stats.accepted == 2

        manager.on_rollout_rejected(1)
        stats = manager.stats
        assert stats.running == 0
        assert stats.rejected == 1

    def test_on_batch_consumed(self):
        """Test batch consumption tracking."""
        manager = StalenessManager(batch_size=32)
        manager.on_rollout_enqueued(50)
        manager.on_rollout_submitted(50)
        manager.on_rollout_accepted(50)

        manager.on_batch_consumed()  # Default batch_size
        stats = manager.stats
        assert stats.accepted == 18  # 50 - 32

        manager.on_batch_consumed(count=10)
        stats = manager.stats
        assert stats.accepted == 8  # 18 - 10

    def test_reset_stats(self):
        """Test resetting rollout stats."""
        manager = StalenessManager()
        manager.on_rollout_enqueued(10)
        manager.on_rollout_submitted(5)
        manager.on_rollout_accepted(3)

        manager.reset_stats()
        stats = manager.stats
        assert stats.enqueued == 0
        assert stats.running == 0
        assert stats.accepted == 0

    def test_get_staleness_history(self):
        """Test getting staleness history."""
        manager = StalenessManager()

        for i in range(5):
            manager.compute_staleness(kl_divergence=0.01 * i, version_gap=float(i))

        history = manager.get_staleness_history()
        assert len(history) == 5

        # Test last_n
        recent = manager.get_staleness_history(last_n=3)
        assert len(recent) == 3

    def test_get_average_staleness(self):
        """Test getting average staleness."""
        config = AdaptiveAsyncConfig(ema_alpha=1.0)  # No smoothing
        manager = StalenessManager(config=config)

        # Add records with known staleness
        for i in range(10):
            manager.compute_staleness(kl_divergence=0.01, version_gap=1.0)

        avg = manager.get_average_staleness(last_n=5)
        assert avg > 0.0

    def test_get_version_gap_stats(self):
        """Test getting version gap statistics."""
        manager = StalenessManager()

        # Add records with known version gaps
        for gap in [1.0, 2.0, 3.0, 2.0, 2.0]:
            manager.compute_staleness(kl_divergence=0.01, version_gap=gap)

        stats = manager.get_version_gap_stats()
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["mean"] == 2.0

    def test_thread_safety(self):
        """Test thread safety of staleness manager."""
        import threading

        manager = StalenessManager()
        errors = []

        def enqueue_worker():
            try:
                for _ in range(100):
                    manager.on_rollout_enqueued(1)
            except Exception as e:
                errors.append(e)

        def submit_worker():
            try:
                for _ in range(100):
                    manager.on_rollout_submitted(1)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=enqueue_worker),
            threading.Thread(target=submit_worker),
            threading.Thread(target=enqueue_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestStalenessRecord:
    """Tests for StalenessRecord dataclass."""

    def test_creation(self):
        """Test record creation."""
        from datetime import datetime

        record = StalenessRecord(
            timestamp=datetime.now(),
            staleness=StalenessMetrics(combined_staleness=0.5),
            version=10,
            batch_size=32,
        )
        assert record.version == 10
        assert record.batch_size == 32
        assert record.staleness.combined_staleness == 0.5
