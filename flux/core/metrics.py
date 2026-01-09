"""
Metrics collection and logging for Flux.

This module provides utilities for collecting, aggregating, and logging
training metrics to various backends (console, TensorBoard, W&B).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

from flux.core.types import BatchMetrics, RolloutMetrics


class LoggerBackend(Protocol):
    """Protocol for logging backends."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics at a given step."""
        ...

    def close(self) -> None:
        """Close the logger."""
        ...


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of metrics."""

    step: int
    timestamp: float
    metrics: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.metrics.get(key, 0.0)

    def __setitem__(self, key: str, value: float) -> None:
        self.metrics[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            **self.metrics,
        }


class RunningStatistics:
    """Compute running mean, variance, min, max efficiently."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # For Welford's algorithm
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.sum = 0.0

    def update(self, value: float) -> None:
        """Update statistics with a new value using Welford's algorithm."""
        self.count += 1
        self.sum += value
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def update_batch(self, values: list[float]) -> None:
        """Update with multiple values."""
        for v in values:
            self.update(v)

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return self.variance ** 0.5

    def get_stats(self) -> dict[str, float]:
        """Get all statistics."""
        return {
            "count": float(self.count),
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val if self.count > 0 else 0.0,
            "max": self.max_val if self.count > 0 else 0.0,
            "sum": self.sum,
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.sum = 0.0


class MetricsAggregator:
    """Aggregate metrics over multiple steps with running statistics.

    Supports hierarchical metric names (e.g., "loss/policy", "reward/mean").
    """

    def __init__(self) -> None:
        self._stats: dict[str, RunningStatistics] = defaultdict(RunningStatistics)
        self._latest: dict[str, float] = {}
        self._start_time = time.time()

    def update(self, name: str, value: float) -> None:
        """Update a single metric."""
        self._stats[name].update(value)
        self._latest[name] = value

    def update_dict(self, metrics: dict[str, float]) -> None:
        """Update multiple metrics from a dictionary."""
        for name, value in metrics.items():
            self.update(name, value)

    def update_batch_metrics(self, batch_metrics: BatchMetrics) -> None:
        """Update from BatchMetrics object."""
        self.update_dict(batch_metrics.to_dict())

    def update_rollout_metrics(self, rollout_metrics: RolloutMetrics) -> None:
        """Update from RolloutMetrics object."""
        self.update_dict(rollout_metrics.to_dict())

    def get_mean(self, name: str) -> float:
        """Get mean value for a metric."""
        if name not in self._stats:
            return 0.0
        return self._stats[name].mean

    def get_latest(self, name: str) -> float:
        """Get latest value for a metric."""
        return self._latest.get(name, 0.0)

    def get_stats(self, name: str) -> dict[str, float]:
        """Get all statistics for a metric."""
        if name not in self._stats:
            return {}
        return self._stats[name].get_stats()

    def get_all_means(self) -> dict[str, float]:
        """Get mean values for all metrics."""
        return {name: stats.mean for name, stats in self._stats.items()}

    def get_all_latest(self) -> dict[str, float]:
        """Get latest values for all metrics."""
        return dict(self._latest)

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all metrics."""
        return {name: stats.get_stats() for name, stats in self._stats.items()}

    def elapsed_time(self) -> float:
        """Time since aggregator creation in seconds."""
        return time.time() - self._start_time

    def reset(self) -> None:
        """Reset all aggregated metrics."""
        self._stats.clear()
        self._latest.clear()
        self._start_time = time.time()

    def reset_metric(self, name: str) -> None:
        """Reset a specific metric."""
        if name in self._stats:
            self._stats[name].reset()
        if name in self._latest:
            del self._latest[name]


class ConsoleLogger:
    """Simple console logger for metrics."""

    def __init__(self, prefix: str = "", log_interval: int = 10) -> None:
        self.prefix = prefix
        self.log_interval = log_interval
        self._step_count = 0

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to console."""
        self._step_count += 1
        if self._step_count % self.log_interval != 0:
            return

        # Format metrics for display
        parts = [f"step={step}"]

        # Group by prefix
        grouped: dict[str, list[str]] = defaultdict(list)
        for key, value in sorted(metrics.items()):
            if "/" in key:
                prefix, name = key.rsplit("/", 1)
                if isinstance(value, float):
                    grouped[prefix].append(f"{name}={value:.4f}")
                else:
                    grouped[prefix].append(f"{name}={value}")
            else:
                if isinstance(value, float):
                    parts.append(f"{key}={value:.4f}")
                else:
                    parts.append(f"{key}={value}")

        # Add grouped metrics
        for prefix, items in sorted(grouped.items()):
            parts.append(f"{prefix}=[{', '.join(items)}]")

        print(f"{self.prefix}[{', '.join(parts)}]")

    def close(self) -> None:
        """No cleanup needed for console logger."""
        pass


class TensorBoardLogger:
    """TensorBoard logger for metrics."""

    def __init__(self, log_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self._enabled = True
        except ImportError:
            print("Warning: tensorboard not installed, TensorBoard logging disabled")
            self.writer = None
            self._enabled = False

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to TensorBoard."""
        if not self._enabled or self.writer is None:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class WandBLogger:
    """Weights & Biases logger for metrics."""

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import wandb
            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config,
            )
            self._enabled = True
        except ImportError:
            print("Warning: wandb not installed, W&B logging disabled")
            self._wandb = None
            self._run = None
            self._enabled = False

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to W&B."""
        if not self._enabled:
            return

        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        """Finish W&B run."""
        if self._run is not None:
            self._run.finish()


class MetricsLogger:
    """Unified metrics logger supporting multiple backends.

    Example:
        logger = MetricsLogger(
            console=True,
            tensorboard_dir="logs/run1",
            wandb_project="my_project",
        )
        logger.log({"loss": 0.5, "reward/mean": 1.0}, step=100)
    """

    def __init__(
        self,
        console: bool = True,
        console_interval: int = 10,
        tensorboard_dir: str | None = None,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        self._backends: list[LoggerBackend] = []

        if console:
            self._backends.append(ConsoleLogger(log_interval=console_interval))

        if tensorboard_dir:
            self._backends.append(TensorBoardLogger(tensorboard_dir))

        if wandb_project:
            self._backends.append(WandBLogger(
                project=wandb_project,
                run_name=wandb_run_name,
                config=wandb_config,
            ))

        self._aggregator = MetricsAggregator()
        self._current_step = 0

    @property
    def aggregator(self) -> MetricsAggregator:
        """Access the metrics aggregator."""
        return self._aggregator

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        aggregate: bool = True,
    ) -> None:
        """Log metrics to all backends.

        Args:
            metrics: Dictionary of metric names to values.
            step: Training step (uses internal counter if None).
            aggregate: Whether to also aggregate metrics.
        """
        if step is None:
            step = self._current_step
        else:
            self._current_step = step

        # Aggregate if requested
        if aggregate:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._aggregator.update(name, float(value))

        # Log to all backends
        for backend in self._backends:
            backend.log(metrics, step)

    def log_batch_metrics(self, batch_metrics: BatchMetrics, step: int | None = None) -> None:
        """Log BatchMetrics object."""
        self.log(batch_metrics.to_dict(), step=step)

    def log_rollout_metrics(self, rollout_metrics: RolloutMetrics, step: int | None = None) -> None:
        """Log RolloutMetrics object."""
        self.log(rollout_metrics.to_dict(), step=step)

    def get_aggregated_metrics(self) -> dict[str, float]:
        """Get mean values of all aggregated metrics."""
        return self._aggregator.get_all_means()

    def step(self) -> None:
        """Increment internal step counter."""
        self._current_step += 1

    def reset_aggregator(self) -> None:
        """Reset the metrics aggregator."""
        self._aggregator.reset()

    def close(self) -> None:
        """Close all backends."""
        for backend in self._backends:
            backend.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
