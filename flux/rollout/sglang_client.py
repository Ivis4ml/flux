"""
SGLang client for Flux.

Provides an async HTTP client for communicating with SGLang inference servers.
Supports generation, request abortion, weight updates, and health checks.

Based on Slime's SGLang integration with Flux-specific enhancements.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from flux.core.config import RolloutConfig, SGLangConfig
from flux.core.types import PolicyVersion


logger = logging.getLogger(__name__)


class GenerationStatus(str, Enum):
    """Status of a generation request."""

    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    TRUNCATED = "truncated"  # Hit max_new_tokens
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class GenerationRequest:
    """Request for text generation."""

    request_id: str
    prompt: str | list[int]  # Text prompt or token IDs
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    return_logprobs: bool = True
    skip_special_tokens: bool = False

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    weight_version: int = 0


@dataclass
class GenerationResult:
    """Result of a generation request."""

    request_id: str
    status: GenerationStatus
    prompt: str | list[int]
    response: str = ""
    tokens: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    finish_reason: str = ""

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Weight version when generated
    weight_version: int = 0

    # Error info
    error: str | None = None

    @property
    def elapsed_ms(self) -> float:
        """Time to generate in milliseconds."""
        if self.end_time > 0 and self.start_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        """Generation speed in tokens per second."""
        elapsed = self.elapsed_ms / 1000
        if elapsed > 0 and self.completion_tokens > 0:
            return self.completion_tokens / elapsed
        return 0.0


class SGLangClient:
    """Async HTTP client for SGLang inference servers.

    Provides methods for:
    - Text generation with sampling parameters
    - Request abortion for long-tail cutting
    - Weight updates from training
    - Health monitoring

    Example:
        client = SGLangClient(config=SGLangConfig(base_url="http://localhost:8000"))
        await client.connect()

        # Generate text
        result = await client.generate(
            prompt="Hello, world!",
            max_tokens=100,
            temperature=0.7,
        )

        # Abort a request
        await client.abort_request(request_id)

        # Update weights
        await client.update_weights(weight_data, version=1)

        await client.close()
    """

    def __init__(
        self,
        config: SGLangConfig | None = None,
        rollout_config: RolloutConfig | None = None,
    ) -> None:
        """Initialize the SGLang client.

        Args:
            config: SGLang server configuration.
            rollout_config: Rollout configuration for generation params.
        """
        if not HAS_HTTPX:
            raise ImportError("httpx is required for SGLangClient. Install with: pip install httpx")

        self.config = config or SGLangConfig()
        self.rollout_config = rollout_config or RolloutConfig()

        # Server URLs
        self._server_urls = self._get_server_urls()
        self._current_server_idx = 0

        # HTTP client
        self._client: httpx.AsyncClient | None = None

        # Request tracking
        self._pending_requests: dict[str, GenerationRequest] = {}
        self._request_counter = 0

        # Server state
        self._connected = False
        self._weight_version = 0

        # Metrics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._aborted_requests = 0

    def _get_server_urls(self) -> list[str]:
        """Get list of server URLs from config."""
        if self.config.server_urls:
            return list(self.config.server_urls)
        # Generate URLs from base_url and num_servers
        base = self.config.base_url.rstrip("/")
        if self.config.num_servers == 1:
            return [base]
        # Assume ports increment from base
        # e.g., http://localhost:8000 -> [http://localhost:8000, http://localhost:8001, ...]
        import urllib.parse
        parsed = urllib.parse.urlparse(base)
        port = parsed.port or 8000
        urls = []
        for i in range(self.config.num_servers):
            new_port = port + i
            new_netloc = f"{parsed.hostname}:{new_port}"
            new_url = urllib.parse.urlunparse(
                (parsed.scheme, new_netloc, parsed.path, "", "", "")
            )
            urls.append(new_url)
        return urls

    @property
    def is_connected(self) -> bool:
        """Whether client is connected."""
        return self._connected and self._client is not None

    @property
    def weight_version(self) -> int:
        """Current weight version on server."""
        return self._weight_version

    @property
    def num_servers(self) -> int:
        """Number of server URLs."""
        return len(self._server_urls)

    async def connect(self) -> None:
        """Connect to SGLang server(s)."""
        if self._client is not None:
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )

        # Verify connection
        healthy = await self.health_check()
        if not healthy:
            await self._client.aclose()
            self._client = None
            raise ConnectionError("Failed to connect to SGLang server")

        self._connected = True
        logger.info(f"Connected to SGLang server(s): {self._server_urls}")

    async def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._connected = False

    async def __aenter__(self) -> "SGLangClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def _get_next_server_url(self) -> str:
        """Get next server URL in round-robin fashion."""
        url = self._server_urls[self._current_server_idx]
        self._current_server_idx = (self._current_server_idx + 1) % len(self._server_urls)
        return url

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        return f"flux-{self._request_counter}-{int(time.time() * 1000)}"

    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        return_logprobs: bool = True,
        request_id: str | None = None,
    ) -> GenerationResult:
        """Generate text completion.

        Args:
            prompt: Text prompt or list of token IDs.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            stop: Stop sequences.
            stop_token_ids: Stop token IDs.
            return_logprobs: Whether to return log probabilities.
            request_id: Optional request ID (generated if not provided).

        Returns:
            GenerationResult with response and metadata.
        """
        if not self.is_connected:
            raise RuntimeError("Client not connected. Call connect() first.")

        # Use defaults from config
        max_tokens = max_tokens or self.rollout_config.max_tokens
        temperature = temperature if temperature is not None else self.rollout_config.temperature
        top_p = top_p if top_p is not None else self.rollout_config.top_p
        top_k = top_k if top_k is not None else self.rollout_config.top_k

        # Create request
        request_id = request_id or self._generate_request_id()
        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            stop_token_ids=stop_token_ids,
            return_logprobs=return_logprobs,
            weight_version=self._weight_version,
        )

        self._pending_requests[request_id] = request
        self._total_requests += 1

        try:
            result = await self._send_generate_request(request)
            self._successful_requests += 1
            return result
        except Exception as e:
            self._failed_requests += 1
            return GenerationResult(
                request_id=request_id,
                status=GenerationStatus.FAILED,
                prompt=prompt,
                error=str(e),
            )
        finally:
            self._pending_requests.pop(request_id, None)

    async def _send_generate_request(
        self, request: GenerationRequest
    ) -> GenerationResult:
        """Send generation request to server."""
        url = self._get_next_server_url()
        endpoint = f"{url}/generate"

        # Build payload
        sampling_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_new_tokens": request.max_tokens,
            "skip_special_tokens": request.skip_special_tokens,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }

        if request.stop:
            sampling_params["stop"] = request.stop
        if request.stop_token_ids:
            sampling_params["stop_token_ids"] = request.stop_token_ids

        payload: dict[str, Any] = {
            "sampling_params": sampling_params,
            "return_logprob": request.return_logprobs,
        }

        # Add prompt (text or token IDs)
        if isinstance(request.prompt, str):
            payload["text"] = request.prompt
        else:
            payload["input_ids"] = request.prompt

        start_time = time.time()

        # Send request with retry
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as e:
                if attempt == self.config.max_retries:
                    raise
                logger.warning(f"HTTP error {e.response.status_code}, retrying ({attempt + 1}/{self.config.max_retries})")
                await asyncio.sleep(1.0)
            except httpx.RequestError as e:
                if attempt == self.config.max_retries:
                    raise
                logger.warning(f"Request error: {e}, retrying ({attempt + 1}/{self.config.max_retries})")
                await asyncio.sleep(1.0)
            except (httpx.TimeoutException, ConnectionError, OSError) as e:
                # Handle network-level errors specifically
                if attempt == self.config.max_retries:
                    raise
                logger.warning(f"Network error: {e}, retrying ({attempt + 1}/{self.config.max_retries})")
                await asyncio.sleep(1.0)

        end_time = time.time()

        # Parse response
        return self._parse_generate_response(request, data, start_time, end_time)

    def _parse_generate_response(
        self,
        request: GenerationRequest,
        data: dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> GenerationResult:
        """Parse generation response from server."""
        # Handle different response formats
        text = data.get("text", "")
        tokens = data.get("output_ids", data.get("output_token_ids", []))
        meta_info = data.get("meta_info", {})
        finish_reason = meta_info.get("finish_reason", {})

        # Get log probs if available
        log_probs = []
        if "meta_info" in data and "output_token_logprobs" in data["meta_info"]:
            log_probs = data["meta_info"]["output_token_logprobs"]

        # Determine status from finish reason
        finish_type = finish_reason.get("type", "stop") if isinstance(finish_reason, dict) else str(finish_reason)
        status_map = {
            "length": GenerationStatus.TRUNCATED,
            "abort": GenerationStatus.ABORTED,
        }
        status = status_map.get(finish_type, GenerationStatus.COMPLETED)

        return GenerationResult(
            request_id=request.request_id,
            status=status,
            prompt=request.prompt,
            response=text,
            tokens=tokens,
            log_probs=log_probs,
            finish_reason=finish_type,
            start_time=start_time,
            end_time=end_time,
            prompt_tokens=meta_info.get("prompt_tokens", 0),
            completion_tokens=len(tokens) if tokens else len(text.split()),
            weight_version=request.weight_version,
        )

    async def generate_batch(
        self,
        prompts: list[str | list[int]],
        **kwargs,
    ) -> list[GenerationResult]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of prompts.
            **kwargs: Additional generation parameters.

        Returns:
            List of GenerationResults.
        """
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_stream(
        self,
        prompt: str | list[int],
        **kwargs,
    ) -> AsyncIterator[GenerationResult]:
        """Stream generation results token by token.

        Uses Server-Sent Events (SSE) for real-time token streaming.

        Args:
            prompt: The prompt.
            **kwargs: Generation parameters.

        Yields:
            Partial GenerationResults as tokens are generated.
        """
        if not self.config.use_streaming:
            # Fallback to non-streaming
            result = await self.generate(prompt, **kwargs)
            yield result
            return

        import json

        url = self._get_next_server_url()
        endpoint = f"{url}/generate"
        request_id = kwargs.pop("request_id", f"stream-{time.time()}")

        # Use rollout_config for generation params (SGLangConfig doesn't have these)
        request_data: dict[str, Any] = {
            "sampling_params": {
                "max_new_tokens": kwargs.get("max_tokens", self.rollout_config.max_tokens),
                "temperature": kwargs.get("temperature", self.rollout_config.temperature),
                "top_p": kwargs.get("top_p", self.rollout_config.top_p),
            },
            "stream": True,
            "rid": request_id,
            "return_logprob": kwargs.get("return_logprobs", True),
        }

        # Add prompt (text or token IDs)
        if isinstance(prompt, str):
            request_data["text"] = prompt
        else:
            request_data["input_ids"] = prompt

        # Track for abort capability
        self._pending_requests[request_id] = request_data
        start_time = time.time()

        try:
            async with self._client.stream(
                "POST",
                endpoint,
                json=request_data,
                timeout=self.config.timeout,
            ) as response:
                response.raise_for_status()

                # Accumulated state for streaming
                accumulated_text = ""
                accumulated_tokens: list[int] = []
                accumulated_log_probs: list[float] = []
                finish_reason = None

                # Parse SSE events
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # SSE format: "data: {...}"
                    if line.startswith("data:"):
                        data_str = line[5:].strip()

                        # Check for end of stream
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Extract token info from SSE event
                        # SGLang typically sends: {"text": "token", "meta_info": {...}}
                        if "text" in data:
                            new_text = data["text"]
                            accumulated_text += new_text

                        if "meta_info" in data:
                            meta = data["meta_info"]
                            if "output_token_ids" in meta:
                                accumulated_tokens = meta["output_token_ids"]
                            if "output_token_logprobs" in meta:
                                accumulated_log_probs = meta["output_token_logprobs"]
                            if "finish_reason" in meta:
                                finish_reason = meta["finish_reason"]

                        # Yield partial result
                        current_time = time.time()

                        partial_result = GenerationResult(
                            request_id=request_id,
                            status=GenerationStatus.GENERATING,
                            prompt=prompt,
                            response=accumulated_text,
                            tokens=accumulated_tokens,
                            log_probs=accumulated_log_probs if accumulated_log_probs else [],
                            finish_reason=finish_reason if finish_reason else "",
                            start_time=start_time,
                            end_time=current_time,
                            prompt_tokens=len(prompt.split()) if isinstance(prompt, str) else len(prompt),
                            completion_tokens=len(accumulated_tokens),
                            weight_version=self._weight_version,
                        )
                        yield partial_result

                # Final result
                end_time = time.time()
                status = GenerationStatus.COMPLETED
                if finish_reason == "length":
                    status = GenerationStatus.TRUNCATED

                final_result = GenerationResult(
                    request_id=request_id,
                    status=status,
                    prompt=prompt,
                    response=accumulated_text,
                    tokens=accumulated_tokens,
                    log_probs=accumulated_log_probs if accumulated_log_probs else [],
                    finish_reason=finish_reason if finish_reason else "",
                    start_time=start_time,
                    end_time=end_time,
                    prompt_tokens=len(prompt.split()) if isinstance(prompt, str) else len(prompt),
                    completion_tokens=len(accumulated_tokens),
                    weight_version=self._weight_version,
                )
                self._successful_requests += 1
                yield final_result

        except httpx.TimeoutException:
            logger.warning(f"Streaming request {request_id} timed out")
            yield GenerationResult(
                request_id=request_id,
                status=GenerationStatus.FAILED,
                prompt=prompt,
                response="",
                tokens=[],
                error="Timeout",
                start_time=start_time,
                end_time=time.time(),
            )
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield GenerationResult(
                request_id=request_id,
                status=GenerationStatus.FAILED,
                prompt=prompt,
                response="",
                tokens=[],
                error=str(e),
                start_time=start_time,
                end_time=time.time(),
            )
        finally:
            self._pending_requests.pop(request_id, None)

    async def abort_request(self, request_id: str) -> bool:
        """Abort a pending generation request.

        Args:
            request_id: The request ID to abort.

        Returns:
            True if abort was successful.
        """
        if request_id not in self._pending_requests:
            return False

        url = self._get_next_server_url()
        endpoint = f"{url}/abort"

        try:
            response = await self._client.post(
                endpoint,
                json={"rid": request_id},
            )
            response.raise_for_status()
            self._aborted_requests += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to abort request {request_id}: {e}")
            return False

    async def abort_all(self) -> int:
        """Abort all pending requests.

        Returns:
            Number of requests aborted.
        """
        count = 0
        for request_id in list(self._pending_requests.keys()):
            if await self.abort_request(request_id):
                count += 1
        return count

    async def health_check(self, server_url: str | None = None) -> bool:
        """Check if server is healthy.

        Args:
            server_url: Specific server URL to check (checks all if None).

        Returns:
            True if server(s) are healthy.
        """
        if self._client is None:
            return False

        urls = [server_url] if server_url else self._server_urls

        for url in urls:
            try:
                endpoint = f"{url}/health_generate"
                response = await self._client.get(endpoint, timeout=5.0)
                if response.status_code != 200:
                    return False
            except Exception:
                return False

        return True

    async def get_weight_version(self, server_url: str | None = None) -> int:
        """Get current weight version from server.

        Args:
            server_url: Specific server URL (uses first if None).

        Returns:
            Current weight version.
        """
        url = server_url or self._server_urls[0]
        endpoint = f"{url}/get_weight_version"

        try:
            response = await self._client.get(endpoint)
            response.raise_for_status()
            return int(response.text)
        except Exception as e:
            logger.warning(f"Failed to get weight version: {e}")
            return self._weight_version

    async def update_weights(
        self,
        weights: bytes | dict[str, Any],
        version: int | None = None,
        server_url: str | None = None,
    ) -> bool:
        """Update model weights on server.

        Args:
            weights: Serialized weight data (bytes) or weight dict.
            version: New weight version.
            server_url: Specific server (updates all if None).

        Returns:
            True if update was successful.
        """
        urls = [server_url] if server_url else self._server_urls

        if version is not None:
            self._weight_version = version

        for url in urls:
            try:
                endpoint = f"{url}/update_weights"

                if isinstance(weights, bytes):
                    # Binary data
                    payload = {
                        "weights": base64.b64encode(weights).decode("utf-8"),
                        "weight_version": str(self._weight_version),
                    }
                else:
                    # Dict data
                    payload = {
                        "weights": weights,
                        "weight_version": str(self._weight_version),
                    }

                response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Failed to update weights on {url}: {e}")
                return False

        logger.info(f"Updated weights to version {self._weight_version} on {len(urls)} server(s)")
        return True

    async def flush_cache(self, server_url: str | None = None) -> bool:
        """Flush KV cache on server.

        Args:
            server_url: Specific server (flushes all if None).

        Returns:
            True if flush was successful.
        """
        urls = [server_url] if server_url else self._server_urls

        for url in urls:
            try:
                endpoint = f"{url}/flush_cache"
                response = await self._client.get(endpoint)
                response.raise_for_status()
            except Exception as e:
                logger.warning(f"Failed to flush cache on {url}: {e}")
                return False

        return True

    def get_metrics(self) -> dict[str, Any]:
        """Get client metrics.

        Returns:
            Dict with request statistics.
        """
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "aborted_requests": self._aborted_requests,
            "pending_requests": len(self._pending_requests),
            "weight_version": self._weight_version,
            "num_servers": self.num_servers,
        }


class SGLangClientPool:
    """Pool of SGLang clients for load balancing.

    Manages multiple clients across different servers for improved throughput.

    Example:
        pool = SGLangClientPool(
            server_urls=["http://localhost:8000", "http://localhost:8001"],
            clients_per_server=2,
        )
        await pool.connect()

        results = await pool.generate_batch(prompts)

        await pool.close()
    """

    def __init__(
        self,
        server_urls: list[str],
        clients_per_server: int = 1,
        config: SGLangConfig | None = None,
    ) -> None:
        """Initialize the client pool.

        Args:
            server_urls: List of server URLs.
            clients_per_server: Number of clients per server.
            config: Base configuration.
        """
        self._server_urls = server_urls
        self._clients_per_server = clients_per_server
        self._base_config = config or SGLangConfig()
        self._clients: list[SGLangClient] = []
        self._client_idx = 0

    @property
    def num_clients(self) -> int:
        """Total number of clients."""
        return len(self._clients)

    async def connect(self) -> None:
        """Connect all clients."""
        for url in self._server_urls:
            for _ in range(self._clients_per_server):
                config = SGLangConfig(
                    base_url=url,
                    num_servers=1,
                    timeout=self._base_config.timeout,
                    max_retries=self._base_config.max_retries,
                    use_streaming=self._base_config.use_streaming,
                )
                client = SGLangClient(config=config)
                await client.connect()
                self._clients.append(client)

    async def close(self) -> None:
        """Close all clients."""
        for client in self._clients:
            await client.close()
        self._clients.clear()

    async def __aenter__(self) -> "SGLangClientPool":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _get_next_client(self) -> SGLangClient:
        """Get next client in round-robin fashion."""
        client = self._clients[self._client_idx]
        self._client_idx = (self._client_idx + 1) % len(self._clients)
        return client

    async def generate(self, prompt: str | list[int], **kwargs) -> GenerationResult:
        """Generate using next available client."""
        client = self._get_next_client()
        return await client.generate(prompt, **kwargs)

    async def generate_batch(
        self,
        prompts: list[str | list[int]],
        **kwargs,
    ) -> list[GenerationResult]:
        """Generate batch across pool."""
        # Distribute across clients
        tasks = []
        for i, prompt in enumerate(prompts):
            client = self._clients[i % len(self._clients)]
            tasks.append(client.generate(prompt, **kwargs))
        return await asyncio.gather(*tasks)

    async def update_weights(self, weights: bytes | dict[str, Any], version: int) -> bool:
        """Update weights on all servers."""
        for client in self._clients:
            if not await client.update_weights(weights, version):
                return False
        return True

    async def health_check(self) -> bool:
        """Check health of all clients."""
        for client in self._clients:
            if not await client.health_check():
                return False
        return True
