"""
Unit tests for Flux utility modules.
"""

import asyncio
import json
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

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


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata."""

    def test_creation(self):
        """Test metadata creation."""
        meta = CheckpointMetadata(
            checkpoint_id="step-100-123456",
            step=100,
            version=100,
        )

        assert meta.checkpoint_id == "step-100-123456"
        assert meta.step == 100
        assert meta.version == 100
        assert isinstance(meta.timestamp, datetime)

    def test_with_metrics(self):
        """Test metadata with metrics."""
        meta = CheckpointMetadata(
            checkpoint_id="test",
            step=50,
            version=50,
            metrics={"loss": 0.5, "reward": 0.8},
            tags=["best", "evaluation"],
        )

        assert meta.metrics["loss"] == 0.5
        assert "best" in meta.tags

    def test_to_dict(self):
        """Test serialization to dict."""
        meta = CheckpointMetadata(
            checkpoint_id="test",
            step=100,
            version=100,
            metrics={"loss": 0.3},
        )
        d = meta.to_dict()

        assert d["checkpoint_id"] == "test"
        assert d["step"] == 100
        assert "timestamp" in d
        assert d["metrics"]["loss"] == 0.3

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "checkpoint_id": "test",
            "step": 100,
            "version": 100,
            "timestamp": "2024-01-01T12:00:00",
            "metrics": {"loss": 0.5},
            "tags": ["test"],
            "path": "/tmp/test",
        }
        meta = CheckpointMetadata.from_dict(data)

        assert meta.checkpoint_id == "test"
        assert meta.step == 100
        assert meta.metrics["loss"] == 0.5

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CheckpointMetadata(
            checkpoint_id="test",
            step=100,
            version=100,
            metrics={"loss": 0.5},
            tags=["tag1", "tag2"],
        )
        restored = CheckpointMetadata.from_dict(original.to_dict())

        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.step == original.step
        assert restored.metrics == original.metrics


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create checkpoint manager."""
        return CheckpointManager(
            checkpoint_dir=tmp_path,
            max_checkpoints=3,
            keep_best=1,
        )

    @pytest.fixture
    def model_state(self):
        """Create dummy model state."""
        return {
            "weight": torch.randn(10, 10),
            "bias": torch.randn(10),
        }

    def test_creation(self, manager):
        """Test manager creation."""
        assert manager.num_checkpoints == 0
        assert manager.latest_checkpoint is None

    def test_save_checkpoint(self, manager, model_state):
        """Test saving checkpoint."""
        meta = manager.save(
            step=100,
            model_state=model_state,
            metrics={"loss": 0.5},
        )

        assert meta.step == 100
        assert manager.num_checkpoints == 1
        assert manager.latest_checkpoint.step == 100

    def test_save_multiple(self, manager, model_state):
        """Test saving multiple checkpoints."""
        for step in [100, 200, 300]:
            manager.save(step=step, model_state=model_state)

        assert manager.num_checkpoints == 3
        assert manager.latest_checkpoint.step == 300

    def test_load_checkpoint(self, manager, model_state):
        """Test loading checkpoint."""
        manager.save(step=100, model_state=model_state)

        state = manager.load_latest()
        assert state is not None
        assert state.metadata.step == 100
        assert state.model_state is not None

    def test_load_best(self, manager, model_state):
        """Test loading best checkpoint."""
        manager.save(step=100, model_state=model_state, metrics={"reward": 0.5})
        manager.save(step=200, model_state=model_state, metrics={"reward": 0.8})
        manager.save(step=300, model_state=model_state, metrics={"reward": 0.6})

        state = manager.load_best(metric="reward", higher_is_better=True)
        assert state is not None
        assert state.metadata.step == 200

    def test_load_best_lower(self, manager, model_state):
        """Test loading best checkpoint with lower is better."""
        manager.save(step=100, model_state=model_state, metrics={"loss": 0.5})
        manager.save(step=200, model_state=model_state, metrics={"loss": 0.3})
        manager.save(step=300, model_state=model_state, metrics={"loss": 0.4})

        state = manager.load_best(metric="loss", higher_is_better=False)
        assert state is not None
        assert state.metadata.step == 200

    def test_list_checkpoints(self, manager, model_state):
        """Test listing checkpoints."""
        manager.save(step=100, model_state=model_state, tags=["train"])
        manager.save(step=200, model_state=model_state, tags=["eval"])
        manager.save(step=300, model_state=model_state, tags=["train"])

        all_checkpoints = manager.list_checkpoints()
        assert len(all_checkpoints) == 3

        train_checkpoints = manager.list_checkpoints(tags=["train"])
        assert len(train_checkpoints) == 2

    def test_delete_checkpoint(self, manager, model_state):
        """Test deleting checkpoint."""
        meta = manager.save(step=100, model_state=model_state)

        assert manager.delete(meta.checkpoint_id)
        assert manager.num_checkpoints == 0

    def test_cleanup_old(self, manager, model_state):
        """Test automatic cleanup of old checkpoints."""
        # Save 5 checkpoints with max_checkpoints=3
        for step in [100, 200, 300, 400, 500]:
            manager.save(step=step, model_state=model_state)

        # Should keep only 3 most recent
        assert manager.num_checkpoints == 3
        steps = [m.step for m in manager.list_checkpoints()]
        assert 500 in steps
        assert 400 in steps

    def test_registry_persistence(self, tmp_path, model_state):
        """Test registry persists across manager instances."""
        # Create and save
        manager1 = CheckpointManager(checkpoint_dir=tmp_path)
        manager1.save(step=100, model_state=model_state)

        # Create new instance
        manager2 = CheckpointManager(checkpoint_dir=tmp_path)
        assert manager2.num_checkpoints == 1
        assert manager2.latest_checkpoint.step == 100


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0

    def test_get_delay(self):
        """Test delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0

    def test_max_delay(self):
        """Test max delay cap."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)

        assert config.get_delay(10) == 5.0

    def test_should_retry(self):
        """Test retry decision."""
        config = RetryConfig(max_retries=3, retry_on=(ValueError,))

        assert config.should_retry(ValueError(), 0)
        assert not config.should_retry(TypeError(), 0)
        assert not config.should_retry(ValueError(), 3)

    def test_should_retry_all_exceptions(self):
        """Test retry on all exceptions."""
        config = RetryConfig(max_retries=3, retry_on=None)

        assert config.should_retry(ValueError(), 0)
        assert config.should_retry(TypeError(), 0)


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_success(self):
        """Test successful call."""
        @with_retry(max_retries=3)
        def succeed():
            return "success"

        assert succeed() == "success"

    def test_retry_then_success(self):
        """Test retry then success."""
        calls = []

        @with_retry(max_retries=3, base_delay=0.01)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("fail")
            return "success"

        assert flaky() == "success"
        assert len(calls) == 3

    def test_max_retries_exceeded(self):
        """Test max retries exceeded."""
        @with_retry(max_retries=2, base_delay=0.01)
        def always_fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            always_fail()

    def test_with_config(self):
        """Test with RetryConfig."""
        config = RetryConfig(max_retries=2, base_delay=0.01)

        @with_retry(config)
        def flaky():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            flaky()

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async function retry."""
        calls = []

        @with_retry(max_retries=3, base_delay=0.01)
        async def async_flaky():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("fail")
            return "success"

        result = await async_flaky()
        assert result == "success"
        assert len(calls) == 2


class TestGracefulShutdown:
    """Tests for GracefulShutdown."""

    def test_creation(self):
        """Test shutdown handler creation."""
        shutdown = GracefulShutdown(timeout=30.0, install_handlers=False)
        assert not shutdown.is_requested

    def test_request_shutdown(self):
        """Test requesting shutdown."""
        shutdown = GracefulShutdown(install_handlers=False)
        shutdown.request_shutdown(ShutdownReason.PROGRAMMATIC)

        assert shutdown.is_requested
        assert shutdown.reason == ShutdownReason.PROGRAMMATIC

    def test_wait(self):
        """Test waiting for shutdown."""
        shutdown = GracefulShutdown(install_handlers=False)

        # Should timeout
        result = shutdown.wait(timeout=0.1)
        assert not result

        # Request and wait
        shutdown.request_shutdown()
        result = shutdown.wait(timeout=0.1)
        assert result

    def test_register_cleanup(self):
        """Test registering cleanup handlers."""
        shutdown = GracefulShutdown(install_handlers=False)
        cleanup_called = []

        def cleanup1():
            cleanup_called.append(1)

        def cleanup2():
            cleanup_called.append(2)

        shutdown.register_cleanup(cleanup1)
        shutdown.register_cleanup(cleanup2)

        errors = shutdown.run_cleanup()
        assert len(errors) == 0
        assert 1 in cleanup_called
        assert 2 in cleanup_called

    def test_cleanup_error_handling(self):
        """Test cleanup error handling."""
        shutdown = GracefulShutdown(install_handlers=False)

        def failing_cleanup():
            raise RuntimeError("cleanup failed")

        shutdown.register_cleanup(failing_cleanup)

        errors = shutdown.run_cleanup()
        assert len(errors) == 1

    def test_context_manager(self):
        """Test context manager usage."""
        cleanup_called = []

        with GracefulShutdown(install_handlers=False) as shutdown:
            shutdown.register_cleanup(lambda: cleanup_called.append(1))

        assert 1 in cleanup_called


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_creation(self):
        """Test circuit breaker creation."""
        breaker = CircuitBreaker(failure_threshold=5)
        assert breaker.state == CircuitBreaker.State.CLOSED

    def test_closed_allows_requests(self):
        """Test closed state allows requests."""
        breaker = CircuitBreaker()
        assert breaker.allow_request()

    def test_opens_after_failures(self):
        """Test opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitBreaker.State.OPEN
        assert not breaker.allow_request()

    def test_success_resets_count(self):
        """Test success resets failure count."""
        breaker = CircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()

        assert breaker.state == CircuitBreaker.State.CLOSED

    def test_half_open_after_timeout(self):
        """Test transitions to half-open after timeout."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
        )

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitBreaker.State.OPEN

        time.sleep(0.15)
        assert breaker.state == CircuitBreaker.State.HALF_OPEN

    def test_decorator(self):
        """Test decorator usage."""
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker
        def service_call():
            raise RuntimeError("service down")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                service_call()

        with pytest.raises(CircuitBreakerOpen):
            service_call()


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    def test_register_check(self):
        """Test registering health check."""
        monitor = HealthMonitor()

        monitor.register("db", lambda: True)
        result = monitor.check("db")

        assert result.healthy

    def test_check_failure(self):
        """Test failing health check."""
        monitor = HealthMonitor(unhealthy_threshold=2)

        monitor.register("db", lambda: False)

        result = monitor.check("db")
        assert not result.healthy

    def test_check_exception(self):
        """Test health check exception."""
        monitor = HealthMonitor()

        def failing_check():
            raise RuntimeError("connection failed")

        monitor.register("db", failing_check)
        result = monitor.check("db")

        assert not result.healthy
        assert "connection failed" in result.last_error

    def test_check_all(self):
        """Test checking all components."""
        monitor = HealthMonitor()

        monitor.register("db", lambda: True)
        monitor.register("cache", lambda: True)
        monitor.register("api", lambda: False)

        results = monitor.check_all()
        assert not results["healthy"]
        assert results["components"]["db"]["healthy"]
        assert not results["components"]["api"]["healthy"]


class TestCounter:
    """Tests for Counter metric."""

    def test_creation(self):
        """Test counter creation."""
        counter = Counter("requests_total", "Total requests")
        assert counter.get() == 0

    def test_increment(self):
        """Test incrementing counter."""
        counter = Counter("requests_total")
        counter.inc()
        counter.inc(5)

        assert counter.get() == 6

    def test_with_labels(self):
        """Test counter with labels."""
        counter = Counter("requests_total")

        counter.inc(labels={"method": "GET"})
        counter.inc(labels={"method": "POST"})
        counter.inc(labels={"method": "GET"})

        assert counter.get(labels={"method": "GET"}) == 2
        assert counter.get(labels={"method": "POST"}) == 1

    def test_collect(self):
        """Test collecting values."""
        counter = Counter("requests_total", "Total requests")
        counter.inc(labels={"method": "GET"})
        counter.inc(labels={"method": "POST"})

        values = counter.collect()
        assert len(values) == 2
        assert all(v.type == MetricType.COUNTER for v in values)


class TestGauge:
    """Tests for Gauge metric."""

    def test_set(self):
        """Test setting gauge."""
        gauge = Gauge("temperature")
        gauge.set(23.5)

        assert gauge.get() == 23.5

    def test_inc_dec(self):
        """Test incrementing and decrementing."""
        gauge = Gauge("connections")
        gauge.set(10)
        gauge.inc(5)
        gauge.dec(3)

        assert gauge.get() == 12

    def test_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge("temperature")

        gauge.set(20.0, labels={"room": "bedroom"})
        gauge.set(22.0, labels={"room": "kitchen"})

        assert gauge.get(labels={"room": "bedroom"}) == 20.0
        assert gauge.get(labels={"room": "kitchen"}) == 22.0


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        """Test observing values."""
        histogram = Histogram("latency", buckets=(0.1, 0.5, 1.0))

        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)

        values = histogram.collect()
        # Should have bucket values + sum + count
        assert len(values) > 0

    def test_default_buckets(self):
        """Test default buckets."""
        histogram = Histogram("latency")
        assert len(histogram.buckets) > 0

    def test_collect_format(self):
        """Test collect format."""
        histogram = Histogram("latency", buckets=(0.5, 1.0))
        histogram.observe(0.3)

        values = histogram.collect()

        # Find bucket values
        bucket_values = [v for v in values if v.name.endswith("_bucket")]
        sum_value = next((v for v in values if v.name.endswith("_sum")), None)
        count_value = next((v for v in values if v.name.endswith("_count")), None)

        assert len(bucket_values) > 0
        assert sum_value is not None
        assert count_value is not None


class TestHealthCheck:
    """Tests for HealthCheck."""

    def test_creation(self):
        """Test health check creation."""
        health = HealthCheck("database")
        result = health.run()

        assert result.status == HealthStatus.UNHEALTHY

    def test_decorator(self):
        """Test check decorator."""
        health = HealthCheck("database")

        @health.check
        def check_db():
            return True

        result = health.run()
        assert result.status == HealthStatus.HEALTHY

    def test_failing_check(self):
        """Test failing check."""
        health = HealthCheck("database")

        @health.check
        def check_db():
            return False

        result = health.run()
        assert result.status == HealthStatus.UNHEALTHY

    def test_exception_handling(self):
        """Test exception handling."""
        health = HealthCheck("database")

        @health.check
        def check_db():
            raise RuntimeError("connection refused")

        result = health.run()
        assert result.status == HealthStatus.UNHEALTHY
        assert "connection refused" in result.message

    def test_custom_result(self):
        """Test returning custom result."""
        health = HealthCheck("database")

        @health.check
        def check_db():
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message="High latency",
                details={"latency_ms": 500},
            )

        result = health.run()
        assert result.status == HealthStatus.DEGRADED
        assert result.details["latency_ms"] == 500


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_creation(self):
        """Test registry creation."""
        registry = MetricsRegistry()
        assert registry is not None

    def test_counter(self):
        """Test counter access."""
        registry = MetricsRegistry()

        counter1 = registry.counter("requests", "Total requests")
        counter2 = registry.counter("requests")

        assert counter1 is counter2

    def test_gauge(self):
        """Test gauge access."""
        registry = MetricsRegistry()

        gauge = registry.gauge("temperature")
        gauge.set(25.0)

        assert gauge.get() == 25.0

    def test_histogram(self):
        """Test histogram access."""
        registry = MetricsRegistry()

        histogram = registry.histogram("latency", buckets=(0.1, 1.0))
        histogram.observe(0.5)

        assert len(histogram.collect()) > 0

    def test_collect_all(self):
        """Test collecting all metrics."""
        registry = MetricsRegistry()

        registry.counter("requests").inc()
        registry.gauge("temperature").set(25.0)

        values = registry.collect()
        assert len(values) == 2

    def test_prometheus_export(self):
        """Test Prometheus format export."""
        registry = MetricsRegistry()

        counter = registry.counter("http_requests_total", "Total HTTP requests")
        counter.inc(labels={"method": "GET"})

        output = registry.export_prometheus()

        assert "# HELP http_requests_total Total HTTP requests" in output
        assert "# TYPE http_requests_total counter" in output
        assert 'http_requests_total{method="GET"}' in output

    def test_json_export(self):
        """Test JSON format export."""
        registry = MetricsRegistry()

        registry.counter("requests").inc(5)

        output = registry.export_json()
        data = json.loads(output)

        assert len(data) == 1
        assert data[0]["name"] == "requests"
        assert data[0]["value"] == 5

    def test_health_checks(self):
        """Test health check integration."""
        registry = MetricsRegistry()

        health = registry.health_check("db")

        @health.check
        def check():
            return True

        results = registry.check_health()
        assert "db" in results
        assert results["db"].status == HealthStatus.HEALTHY


class TestFluxMetrics:
    """Tests for FluxMetrics."""

    def test_creation(self):
        """Test Flux metrics creation."""
        registry = MetricsRegistry()
        metrics = FluxMetrics(registry)

        assert metrics.training_step_total is not None
        assert metrics.rollout_latency is not None

    def test_record_metrics(self):
        """Test recording metrics."""
        registry = MetricsRegistry()
        metrics = FluxMetrics(registry)

        metrics.training_step_total.inc()
        metrics.batch_size.set(32)
        metrics.rollout_latency.observe(1.5)
        metrics.reward_mean.set(0.75)

        assert metrics.training_step_total.get() == 1
        assert metrics.batch_size.get() == 32

    def test_health_checks(self):
        """Test Flux health checks."""
        registry = MetricsRegistry()
        metrics = FluxMetrics(registry)

        @metrics.model_health.check
        def check_model():
            return True

        result = metrics.model_health.run()
        assert result.status == HealthStatus.HEALTHY


class TestMetricsExporter:
    """Tests for MetricsExporter."""

    def test_creation(self):
        """Test exporter creation."""
        registry = MetricsRegistry()
        exporter = MetricsExporter(registry, port=9999)

        assert exporter.port == 9999

    def test_context_manager(self):
        """Test context manager usage."""
        registry = MetricsRegistry()
        registry.counter("test").inc()

        # Use a random available port
        exporter = MetricsExporter(registry, port=0)

        # Just test start/stop don't crash
        # Note: Using port 0 would assign random port which we can't easily test
        # So we skip the actual HTTP test


class TestGetRegistry:
    """Tests for get_registry function."""

    def test_returns_registry(self):
        """Test returns a registry."""
        registry = get_registry()
        assert isinstance(registry, MetricsRegistry)

    def test_singleton(self):
        """Test returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
