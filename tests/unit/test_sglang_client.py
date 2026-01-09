"""
Tests for SGLang client components.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from flux.core.config import RolloutConfig, SGLangConfig
from flux.rollout.sglang_client import (
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    SGLangClient,
    SGLangClientPool,
)


class TestGenerationStatus:
    """Tests for GenerationStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert GenerationStatus.PENDING == "pending"
        assert GenerationStatus.GENERATING == "generating"
        assert GenerationStatus.COMPLETED == "completed"
        assert GenerationStatus.TRUNCATED == "truncated"
        assert GenerationStatus.ABORTED == "aborted"
        assert GenerationStatus.FAILED == "failed"


class TestGenerationRequest:
    """Tests for GenerationRequest dataclass."""

    def test_creation_defaults(self):
        """Test request creation with defaults."""
        request = GenerationRequest(
            request_id="test-1",
            prompt="Hello, world!",
        )
        assert request.request_id == "test-1"
        assert request.prompt == "Hello, world!"
        assert request.max_tokens == 2048
        assert request.temperature == 1.0
        assert request.return_logprobs is True

    def test_creation_with_tokens(self):
        """Test request with token IDs."""
        request = GenerationRequest(
            request_id="test-1",
            prompt=[1, 2, 3, 4, 5],
            max_tokens=100,
        )
        assert request.prompt == [1, 2, 3, 4, 5]
        assert request.max_tokens == 100


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_successful_result(self):
        """Test successful generation result."""
        result = GenerationResult(
            request_id="test-1",
            status=GenerationStatus.COMPLETED,
            prompt="Hello",
            response="Hello! How can I help?",
            tokens=[1, 2, 3, 4, 5],
            log_probs=[-0.1, -0.2, -0.1, -0.15, -0.1],
            finish_reason="stop",
            start_time=1000.0,
            end_time=1001.5,
            completion_tokens=5,
        )
        assert result.status == GenerationStatus.COMPLETED
        assert result.elapsed_ms == 1500.0
        assert result.tokens_per_second == pytest.approx(5 / 1.5, rel=0.01)

    def test_failed_result(self):
        """Test failed generation result."""
        result = GenerationResult(
            request_id="test-1",
            status=GenerationStatus.FAILED,
            prompt="Hello",
            error="Connection timeout",
        )
        assert result.status == GenerationStatus.FAILED
        assert result.error == "Connection timeout"
        assert result.elapsed_ms == 0.0

    def test_truncated_result(self):
        """Test truncated (max tokens) result."""
        result = GenerationResult(
            request_id="test-1",
            status=GenerationStatus.TRUNCATED,
            prompt="Hello",
            response="Long response...",
            finish_reason="length",
        )
        assert result.status == GenerationStatus.TRUNCATED


class TestSGLangClient:
    """Tests for SGLangClient."""

    def test_client_creation_defaults(self):
        """Test client creation with defaults."""
        client = SGLangClient()
        assert client.config.base_url == "http://localhost:8000"
        assert not client.is_connected
        assert client.num_servers == 1

    def test_client_custom_config(self):
        """Test client with custom config."""
        config = SGLangConfig(
            base_url="http://localhost:9000",
            num_servers=2,
            timeout=30.0,
        )
        client = SGLangClient(config=config)
        assert len(client._server_urls) == 2
        assert "http://localhost:9000" in client._server_urls
        assert "http://localhost:9001" in client._server_urls

    def test_client_explicit_urls(self):
        """Test client with explicit server URLs."""
        config = SGLangConfig(
            server_urls=["http://host1:8000", "http://host2:8000"],
        )
        client = SGLangClient(config=config)
        assert client._server_urls == ["http://host1:8000", "http://host2:8000"]

    def test_round_robin_server_selection(self):
        """Test round-robin server selection."""
        config = SGLangConfig(
            server_urls=["http://server1:8000", "http://server2:8000"],
        )
        client = SGLangClient(config=config)

        url1 = client._get_next_server_url()
        url2 = client._get_next_server_url()
        url3 = client._get_next_server_url()

        assert url1 == "http://server1:8000"
        assert url2 == "http://server2:8000"
        assert url3 == "http://server1:8000"  # Wraps around

    def test_request_id_generation(self):
        """Test unique request ID generation."""
        client = SGLangClient()
        id1 = client._generate_request_id()
        id2 = client._generate_request_id()
        assert id1 != id2
        assert id1.startswith("flux-")
        assert id2.startswith("flux-")

    def test_get_metrics(self):
        """Test getting client metrics."""
        client = SGLangClient()
        metrics = client.get_metrics()

        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "weight_version" in metrics

    @pytest.mark.asyncio
    async def test_generate_not_connected_raises(self):
        """Test that generate raises when not connected."""
        client = SGLangClient()
        with pytest.raises(RuntimeError, match="not connected"):
            await client.generate("Hello")


class TestSGLangClientPool:
    """Tests for SGLangClientPool."""

    def test_pool_creation(self):
        """Test pool creation."""
        pool = SGLangClientPool(
            server_urls=["http://localhost:8000", "http://localhost:8001"],
            clients_per_server=2,
        )
        assert pool.num_clients == 0  # Not connected yet

    def test_pool_client_round_robin(self):
        """Test pool client selection."""
        pool = SGLangClientPool(
            server_urls=["http://localhost:8000"],
            clients_per_server=1,
        )
        # Mock clients
        mock_client = MagicMock()
        pool._clients = [mock_client]

        client = pool._get_next_client()
        assert client == mock_client
