"""
Monitoring utilities for Flux.

Provides metrics collection, health checks, and Prometheus-style export.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Type of metric."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with labels."""

    name: str
    type: MetricType
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    help_text: str = ""


class Counter:
    """A monotonically increasing counter.

    Example:
        requests = Counter("http_requests_total", "Total HTTP requests")
        requests.inc()
        requests.inc(labels={"method": "GET", "status": "200"})
    """

    def __init__(self, name: str, help_text: str = "") -> None:
        """Initialize counter.

        Args:
            name: Metric name.
            help_text: Description of the metric.
        """
        self.name = name
        self.help_text = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment counter.

        Args:
            amount: Amount to increment by.
            labels: Optional labels for this increment.
        """
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] += amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get counter value.

        Args:
            labels: Labels to get value for.

        Returns:
            Current counter value.
        """
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values[key]

    def collect(self) -> list[MetricValue]:
        """Collect all values for export."""
        values = []
        with self._lock:
            for key, value in self._values.items():
                values.append(
                    MetricValue(
                        name=self.name,
                        type=MetricType.COUNTER,
                        value=value,
                        labels=dict(key),
                        help_text=self.help_text,
                    )
                )
        return values


class Gauge:
    """A metric that can go up and down.

    Example:
        temperature = Gauge("temperature_celsius", "Current temperature")
        temperature.set(23.5)
        temperature.inc(0.5)
        temperature.dec(1.0)
    """

    def __init__(self, name: str, help_text: str = "") -> None:
        """Initialize gauge.

        Args:
            name: Metric name.
            help_text: Description of the metric.
        """
        self.name = name
        self.help_text = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set gauge value.

        Args:
            value: Value to set.
            labels: Optional labels.
        """
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment gauge.

        Args:
            amount: Amount to increment by.
            labels: Optional labels.
        """
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Decrement gauge.

        Args:
            amount: Amount to decrement by.
            labels: Optional labels.
        """
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] -= amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get gauge value."""
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values[key]

    def collect(self) -> list[MetricValue]:
        """Collect all values for export."""
        values = []
        with self._lock:
            for key, value in self._values.items():
                values.append(
                    MetricValue(
                        name=self.name,
                        type=MetricType.GAUGE,
                        value=value,
                        labels=dict(key),
                        help_text=self.help_text,
                    )
                )
        return values


class Histogram:
    """A histogram for measuring distributions.

    Example:
        latency = Histogram(
            "request_latency_seconds",
            "Request latency",
            buckets=[0.1, 0.5, 1.0, 5.0],
        )
        latency.observe(0.3)
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        help_text: str = "",
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        """Initialize histogram.

        Args:
            name: Metric name.
            help_text: Description of the metric.
            buckets: Bucket boundaries.
        """
        self.name = name
        self.help_text = help_text
        self.buckets = tuple(sorted(buckets or self.DEFAULT_BUCKETS))

        # Per-label data
        self._data: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _get_data(self, labels: dict[str, str] | None) -> dict[str, Any]:
        """Get or create data for labels."""
        key = tuple(sorted((labels or {}).items()))
        if key not in self._data:
            self._data[key] = {
                "buckets": {b: 0 for b in self.buckets},
                "sum": 0.0,
                "count": 0,
            }
        return self._data[key]

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation.

        Args:
            value: Value to observe.
            labels: Optional labels.
        """
        with self._lock:
            data = self._get_data(labels)
            data["sum"] += value
            data["count"] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

    def collect(self) -> list[MetricValue]:
        """Collect all values for export."""
        values = []
        with self._lock:
            for key, data in self._data.items():
                labels = dict(key)
                cumulative = 0

                for bucket in self.buckets:
                    cumulative += data["buckets"][bucket]
                    bucket_labels = {**labels, "le": str(bucket)}
                    values.append(
                        MetricValue(
                            name=f"{self.name}_bucket",
                            type=MetricType.HISTOGRAM,
                            value=cumulative,
                            labels=bucket_labels,
                            help_text=self.help_text,
                        )
                    )

                # +Inf bucket
                values.append(
                    MetricValue(
                        name=f"{self.name}_bucket",
                        type=MetricType.HISTOGRAM,
                        value=data["count"],
                        labels={**labels, "le": "+Inf"},
                        help_text=self.help_text,
                    )
                )

                # Sum and count
                values.append(
                    MetricValue(
                        name=f"{self.name}_sum",
                        type=MetricType.HISTOGRAM,
                        value=data["sum"],
                        labels=labels,
                        help_text=self.help_text,
                    )
                )
                values.append(
                    MetricValue(
                        name=f"{self.name}_count",
                        type=MetricType.HISTOGRAM,
                        value=data["count"],
                        labels=labels,
                        help_text=self.help_text,
                    )
                )

        return values


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HealthCheck:
    """Health check for a component.

    Example:
        health = HealthCheck("database")

        @health.check
        def check_db():
            # Return True if healthy, False otherwise
            return db.ping()

        result = health.run()
    """

    def __init__(self, name: str, timeout: float = 5.0) -> None:
        """Initialize health check.

        Args:
            name: Component name.
            timeout: Check timeout in seconds.
        """
        self.name = name
        self.timeout = timeout
        self._check_fn: Callable[[], bool | HealthCheckResult] | None = None
        self._last_result: HealthCheckResult | None = None

    def check(self, fn: Callable[[], bool | HealthCheckResult]) -> Callable[[], bool | HealthCheckResult]:
        """Decorator to register check function.

        Args:
            fn: Check function returning bool or HealthCheckResult.

        Returns:
            The original function.
        """
        self._check_fn = fn
        return fn

    def run(self) -> HealthCheckResult:
        """Run the health check.

        Returns:
            HealthCheckResult with status.
        """
        if self._check_fn is None:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="No check function registered",
            )

        try:
            result = self._check_fn()

            if isinstance(result, HealthCheckResult):
                self._last_result = result
                return result

            if result:
                self._last_result = HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message=f"{self.name} is healthy",
                )
            else:
                self._last_result = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"{self.name} check failed",
                )

            return self._last_result

        except Exception as e:
            self._last_result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"{self.name} check error: {e}",
                details={"error": str(e)},
            )
            return self._last_result

    @property
    def last_result(self) -> HealthCheckResult | None:
        """Get last check result."""
        return self._last_result


class MetricsRegistry:
    """Registry for all metrics.

    Example:
        registry = MetricsRegistry()
        requests = registry.counter("http_requests_total", "Total requests")
        latency = registry.histogram("latency_seconds", "Request latency")

        # Export all metrics
        text = registry.export_prometheus()
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._health_checks: dict[str, HealthCheck] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help_text: str = "") -> Counter:
        """Get or create a counter.

        Args:
            name: Metric name.
            help_text: Description.

        Returns:
            Counter instance.
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, help_text)
            return self._metrics[name]  # type: ignore

    def gauge(self, name: str, help_text: str = "") -> Gauge:
        """Get or create a gauge.

        Args:
            name: Metric name.
            help_text: Description.

        Returns:
            Gauge instance.
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, help_text)
            return self._metrics[name]  # type: ignore

    def histogram(
        self,
        name: str,
        help_text: str = "",
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """Get or create a histogram.

        Args:
            name: Metric name.
            help_text: Description.
            buckets: Bucket boundaries.

        Returns:
            Histogram instance.
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, help_text, buckets)
            return self._metrics[name]  # type: ignore

    def health_check(self, name: str, timeout: float = 5.0) -> HealthCheck:
        """Get or create a health check.

        Args:
            name: Component name.
            timeout: Check timeout.

        Returns:
            HealthCheck instance.
        """
        with self._lock:
            if name not in self._health_checks:
                self._health_checks[name] = HealthCheck(name, timeout)
            return self._health_checks[name]

    def collect(self) -> list[MetricValue]:
        """Collect all metric values.

        Returns:
            List of all metric values.
        """
        values = []
        with self._lock:
            for metric in self._metrics.values():
                values.extend(metric.collect())
        return values

    def check_health(self) -> dict[str, HealthCheckResult]:
        """Run all health checks.

        Returns:
            Dictionary of component name to result.
        """
        results = {}
        with self._lock:
            for name, check in self._health_checks.items():
                results[name] = check.run()
        return results

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-format metrics string.
        """
        lines = []
        values = self.collect()

        # Group by metric name
        by_name: dict[str, list[MetricValue]] = defaultdict(list)
        for v in values:
            base_name = v.name.rsplit("_", 1)[0] if v.name.endswith(("_bucket", "_sum", "_count")) else v.name
            by_name[base_name].append(v)

        for name, metric_values in sorted(by_name.items()):
            if not metric_values:
                continue

            first = metric_values[0]
            if first.help_text:
                lines.append(f"# HELP {name} {first.help_text}")
            lines.append(f"# TYPE {name} {first.type.value}")

            for v in metric_values:
                if v.labels:
                    label_str = ",".join(
                        f'{k}="{v}"' for k, v in sorted(v.labels.items())
                    )
                    lines.append(f"{v.name}{{{label_str}}} {v.value}")
                else:
                    lines.append(f"{v.name} {v.value}")

        return "\n".join(lines) + "\n"

    def export_json(self) -> str:
        """Export metrics as JSON.

        Returns:
            JSON string of metrics.
        """
        values = self.collect()
        data = [
            {
                "name": v.name,
                "type": v.type.value,
                "value": v.value,
                "labels": v.labels,
                "timestamp": v.timestamp,
            }
            for v in values
        ]
        return json.dumps(data, indent=2)


class MetricsExporter:
    """HTTP server for exposing metrics.

    Example:
        registry = MetricsRegistry()
        exporter = MetricsExporter(registry, port=9090)
        exporter.start()  # Non-blocking

        # ... do work ...

        exporter.stop()
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        host: str = "0.0.0.0",
        port: int = 9090,
    ) -> None:
        """Initialize exporter.

        Args:
            registry: Metrics registry to export.
            host: Host to bind to.
            port: Port to listen on.
        """
        self.registry = registry
        self.host = host
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the metrics server."""
        handler = self._create_handler()
        self._server = HTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True
        self._thread.start()
        logger.info(f"Metrics server started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Metrics server stopped")

    def _create_handler(self) -> type[BaseHTTPRequestHandler]:
        """Create HTTP request handler class."""
        registry = self.registry

        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/metrics":
                    content = registry.export_prometheus()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content.encode("utf-8"))

                elif self.path == "/health":
                    results = registry.check_health()
                    healthy = all(
                        r.status == HealthStatus.HEALTHY for r in results.values()
                    )

                    data = {
                        "healthy": healthy,
                        "checks": {
                            name: {
                                "status": r.status.value,
                                "message": r.message,
                                "timestamp": r.timestamp,
                            }
                            for name, r in results.items()
                        },
                    }
                    content = json.dumps(data, indent=2)

                    self.send_response(200 if healthy else 503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content.encode("utf-8"))

                elif self.path == "/ready":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"OK")

                elif self.path == "/live":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"OK")

                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                # Suppress default logging
                pass

        return MetricsHandler

    def __enter__(self) -> "MetricsExporter":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.stop()


# Global default registry
_default_registry: MetricsRegistry | None = None


def get_registry() -> MetricsRegistry:
    """Get the default metrics registry.

    Returns:
        Default MetricsRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricsRegistry()
    return _default_registry


# Flux-specific metrics
class FluxMetrics:
    """Pre-defined metrics for Flux training.

    Example:
        metrics = FluxMetrics()
        metrics.training_step_total.inc()
        metrics.batch_size.set(32)
        metrics.step_duration.observe(0.5)
    """

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        """Initialize Flux metrics.

        Args:
            registry: Registry to use (default: global registry).
        """
        self.registry = registry or get_registry()

        # Training metrics
        self.training_step_total = self.registry.counter(
            "flux_training_step_total",
            "Total number of training steps",
        )
        self.training_samples_total = self.registry.counter(
            "flux_training_samples_total",
            "Total number of training samples",
        )
        self.training_tokens_total = self.registry.counter(
            "flux_training_tokens_total",
            "Total number of tokens processed",
        )

        # Rollout metrics
        self.rollout_total = self.registry.counter(
            "flux_rollout_total",
            "Total number of rollouts generated",
        )
        self.rollout_aborted_total = self.registry.counter(
            "flux_rollout_aborted_total",
            "Total number of aborted rollouts",
        )
        self.rollout_latency = self.registry.histogram(
            "flux_rollout_latency_seconds",
            "Rollout generation latency",
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )

        # Batch metrics
        self.batch_size = self.registry.gauge(
            "flux_batch_size",
            "Current batch size",
        )
        self.batch_staleness = self.registry.gauge(
            "flux_batch_staleness",
            "Current batch staleness",
        )

        # Model metrics
        self.policy_version = self.registry.gauge(
            "flux_policy_version",
            "Current policy version",
        )
        self.weight_sync_total = self.registry.counter(
            "flux_weight_sync_total",
            "Total number of weight syncs",
        )
        self.weight_sync_latency = self.registry.histogram(
            "flux_weight_sync_latency_seconds",
            "Weight sync latency",
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        )

        # Reward metrics
        self.reward_mean = self.registry.gauge(
            "flux_reward_mean",
            "Mean reward",
        )
        self.reward_std = self.registry.gauge(
            "flux_reward_std",
            "Reward standard deviation",
        )

        # Loss metrics
        self.loss = self.registry.gauge(
            "flux_loss",
            "Current loss value",
        )
        self.policy_loss = self.registry.gauge(
            "flux_policy_loss",
            "Policy loss component",
        )
        self.value_loss = self.registry.gauge(
            "flux_value_loss",
            "Value loss component",
        )

        # Async ratio metrics
        self.async_ratio = self.registry.gauge(
            "flux_async_ratio",
            "Current async ratio",
        )
        self.staleness_estimate = self.registry.gauge(
            "flux_staleness_estimate",
            "Estimated staleness",
        )

        # Resource metrics
        self.gpu_memory_used = self.registry.gauge(
            "flux_gpu_memory_used_bytes",
            "GPU memory used",
        )
        self.gpu_utilization = self.registry.gauge(
            "flux_gpu_utilization_percent",
            "GPU utilization percentage",
        )

        # Health checks
        self.model_health = self.registry.health_check("model")
        self.rollout_health = self.registry.health_check("rollout")
        self.buffer_health = self.registry.health_check("buffer")
