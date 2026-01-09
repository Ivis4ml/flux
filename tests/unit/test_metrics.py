"""
Unit tests for Flux metrics module.
"""

import pytest

from flux.core.metrics import (
    ConsoleLogger,
    MetricsAggregator,
    MetricsLogger,
    MetricsSnapshot,
    RunningStatistics,
)
from flux.core.types import BatchMetrics, RolloutMetrics


class TestRunningStatistics:
    """Tests for RunningStatistics."""

    def test_empty(self) -> None:
        """Test empty statistics."""
        stats = RunningStatistics()
        assert stats.count == 0
        assert stats.mean == 0.0

    def test_single_value(self) -> None:
        """Test with single value."""
        stats = RunningStatistics()
        stats.update(5.0)
        assert stats.count == 1
        assert stats.mean == 5.0
        assert stats.min_val == 5.0
        assert stats.max_val == 5.0

    def test_multiple_values(self) -> None:
        """Test with multiple values."""
        stats = RunningStatistics()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stats.update(v)

        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0
        assert stats.sum == 15.0

    def test_variance(self) -> None:
        """Test variance calculation."""
        stats = RunningStatistics()
        # Variance of [1, 2, 3, 4, 5] = 2.0
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stats.update(v)
        assert stats.variance == pytest.approx(2.0)

    def test_update_batch(self) -> None:
        """Test batch update."""
        stats = RunningStatistics()
        stats.update_batch([1.0, 2.0, 3.0])
        assert stats.count == 3
        assert stats.mean == 2.0

    def test_reset(self) -> None:
        """Test reset."""
        stats = RunningStatistics()
        stats.update_batch([1.0, 2.0, 3.0])
        stats.reset()
        assert stats.count == 0


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot."""

    def test_creation(self) -> None:
        """Test snapshot creation."""
        snapshot = MetricsSnapshot(
            step=100,
            timestamp=1234567890.0,
            metrics={"loss": 0.5, "reward": 1.0},
        )
        assert snapshot.step == 100
        assert snapshot["loss"] == 0.5

    def test_setitem(self) -> None:
        """Test setting metrics."""
        snapshot = MetricsSnapshot(step=0, timestamp=0.0)
        snapshot["new_metric"] = 42.0
        assert snapshot["new_metric"] == 42.0

    def test_to_dict(self) -> None:
        """Test conversion to dict."""
        snapshot = MetricsSnapshot(
            step=100,
            timestamp=0.0,
            metrics={"loss": 0.5},
        )
        d = snapshot.to_dict()
        assert d["step"] == 100
        assert d["loss"] == 0.5


class TestMetricsAggregator:
    """Tests for MetricsAggregator."""

    def test_update_single(self) -> None:
        """Test updating single metric."""
        agg = MetricsAggregator()
        agg.update("loss", 0.5)
        assert agg.get_mean("loss") == 0.5
        assert agg.get_latest("loss") == 0.5

    def test_update_dict(self) -> None:
        """Test updating from dictionary."""
        agg = MetricsAggregator()
        agg.update_dict({"loss": 0.5, "reward": 1.0})
        assert agg.get_mean("loss") == 0.5
        assert agg.get_mean("reward") == 1.0

    def test_update_batch_metrics(self) -> None:
        """Test updating from BatchMetrics."""
        agg = MetricsAggregator()
        metrics = BatchMetrics(policy_loss=0.5, mean_reward=1.0)
        agg.update_batch_metrics(metrics)
        assert agg.get_mean("loss/policy") == 0.5

    def test_update_rollout_metrics(self) -> None:
        """Test updating from RolloutMetrics."""
        agg = MetricsAggregator()
        metrics = RolloutMetrics(num_completed=100)
        agg.update_rollout_metrics(metrics)
        assert agg.get_mean("rollout/completed") == 100

    def test_get_all_means(self) -> None:
        """Test getting all mean values."""
        agg = MetricsAggregator()
        agg.update("a", 1.0)
        agg.update("b", 2.0)
        means = agg.get_all_means()
        assert means["a"] == 1.0
        assert means["b"] == 2.0

    def test_get_summary(self) -> None:
        """Test getting summary statistics."""
        agg = MetricsAggregator()
        agg.update("loss", 0.5)
        agg.update("loss", 0.7)
        summary = agg.get_summary()
        assert "loss" in summary
        assert summary["loss"]["count"] == 2

    def test_reset(self) -> None:
        """Test reset."""
        agg = MetricsAggregator()
        agg.update("loss", 0.5)
        agg.reset()
        assert agg.get_mean("loss") == 0.0


class TestConsoleLogger:
    """Tests for ConsoleLogger."""

    def test_creation(self) -> None:
        """Test logger creation."""
        logger = ConsoleLogger(log_interval=5)
        assert logger.log_interval == 5

    def test_log_interval(self, capsys: pytest.CaptureFixture) -> None:
        """Test log interval filtering."""
        logger = ConsoleLogger(log_interval=2)

        logger.log({"loss": 0.5}, step=1)  # Not logged (step 1)
        captured = capsys.readouterr()
        assert captured.out == ""

        logger.log({"loss": 0.5}, step=2)  # Logged (step 2)
        captured = capsys.readouterr()
        assert "loss" in captured.out


class TestMetricsLogger:
    """Tests for MetricsLogger."""

    def test_console_only(self) -> None:
        """Test console-only logging."""
        logger = MetricsLogger(
            console=True,
            console_interval=1,
            tensorboard_dir=None,
            wandb_project=None,
        )
        assert logger.aggregator is not None

    def test_log_and_aggregate(self) -> None:
        """Test logging with aggregation."""
        logger = MetricsLogger(console=False)
        logger.log({"loss": 0.5}, step=1)
        logger.log({"loss": 0.7}, step=2)

        means = logger.get_aggregated_metrics()
        assert means["loss"] == 0.6  # Average of 0.5 and 0.7

    def test_log_batch_metrics(self) -> None:
        """Test logging BatchMetrics."""
        logger = MetricsLogger(console=False)
        metrics = BatchMetrics(policy_loss=0.5)
        logger.log_batch_metrics(metrics, step=1)

        means = logger.get_aggregated_metrics()
        assert "loss/policy" in means

    def test_step(self) -> None:
        """Test internal step counter."""
        logger = MetricsLogger(console=False)
        assert logger._current_step == 0
        logger.step()
        assert logger._current_step == 1

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        with MetricsLogger(console=False) as logger:
            logger.log({"loss": 0.5}, step=1)
        # Should not raise on close
