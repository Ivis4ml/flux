"""Flux utilities module.

This module provides utility functions and classes:
- Checkpoint management
- Fault tolerance
- Monitoring
"""

from flux.utils.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointState,
)
from flux.utils.fault_tolerance import (
    CircuitBreaker,
    CircuitBreakerOpen,
    GracefulShutdown,
    HealthMonitor,
    RetryConfig,
    ShutdownReason,
    with_retry,
)
from flux.utils.monitoring import (
    Counter,
    FluxMetrics,
    Gauge,
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    Histogram,
    MetricType,
    MetricsExporter,
    MetricsRegistry,
    get_registry,
)

__all__ = [
    # Checkpoint
    "CheckpointManager",
    "CheckpointMetadata",
    "CheckpointState",
    # Fault tolerance
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "GracefulShutdown",
    "HealthMonitor",
    "RetryConfig",
    "ShutdownReason",
    "with_retry",
    # Monitoring
    "Counter",
    "FluxMetrics",
    "Gauge",
    "HealthCheck",
    "HealthCheckResult",
    "HealthStatus",
    "Histogram",
    "MetricType",
    "MetricsExporter",
    "MetricsRegistry",
    "get_registry",
]
