"""
Streaming rollout manager for Flux.

Implements the APRIL strategy for efficient rollout generation:
- Abort: Cancel long-running generations to cut long-tail latency
- Partial: Save incomplete trajectories for later reuse
- Reuse: Continue from partial trajectories when possible
- Iterate: Stream results as they complete
- Learn: Adapt generation parameters based on history

Based on Slime's rollout manager with Flux-specific enhancements.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from flux.core.config import RolloutConfig
from flux.core.trajectory import PartialTrajectory, Trajectory, TrajectoryBuffer
from flux.core.types import PolicyVersion, RolloutMetrics, TrajectoryStatus
from flux.rollout.length_predictor import LengthPredictor
from flux.rollout.partial_buffer import PartialTrajectoryBuffer
from flux.rollout.sglang_client import (
    GenerationResult,
    GenerationStatus,
    SGLangClient,
)


logger = logging.getLogger(__name__)


@dataclass
class RolloutRequest:
    """Request for rollout generation."""

    request_id: str
    prompt: str
    prompt_tokens: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # For continuation from partial trajectory
    partial_trajectory: PartialTrajectory | None = None
    continuation_prompt: str | None = None

    # Timing
    submitted_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class RolloutBatch:
    """Batch of rollout results."""

    trajectories: list[Trajectory]
    partial_trajectories: list[PartialTrajectory]
    failed_requests: list[tuple[str, str]]  # (request_id, error)

    # Metrics
    total_requested: int = 0
    completed: int = 0
    truncated: int = 0
    aborted: int = 0
    failed: int = 0
    reused: int = 0

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        """Total time in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def success_rate(self) -> float:
        """Fraction of successful completions."""
        total = self.completed + self.aborted + self.failed
        if total == 0:
            return 0.0
        return self.completed / total


class StreamingRolloutManager:
    """Manages streaming rollout generation with APRIL strategy.

    APRIL Strategy:
    1. **Oversample**: Request more prompts than needed to account for filtering
    2. **Abort**: Cancel long-running requests to meet batch timeout
    3. **Partial Reuse**: Save aborted trajectories for continuation
    4. **Stream**: Yield results as they complete, don't wait for full batch

    Example:
        manager = StreamingRolloutManager(
            client=sglang_client,
            config=RolloutConfig(
                oversample_ratio=1.5,
                batch_timeout=30.0,
                partial_reuse_threshold=64,
            ),
        )

        # Generate with streaming
        async for trajectory in manager.generate_stream(prompts, target_count=32):
            process(trajectory)

        # Or batch generation
        batch = await manager.generate_batch(prompts, target_count=32)
    """

    def __init__(
        self,
        client: SGLangClient,
        config: RolloutConfig | None = None,
        trajectory_buffer: TrajectoryBuffer | None = None,
        version_provider: Callable[[], PolicyVersion] | None = None,
        length_predictor: LengthPredictor | None = None,
        partial_buffer: PartialTrajectoryBuffer | None = None,
    ) -> None:
        """Initialize the rollout manager.

        Args:
            client: SGLang client for generation.
            config: Rollout configuration.
            trajectory_buffer: Buffer for storing completed trajectories.
            version_provider: Function returning current policy version.
            length_predictor: Predictor for output length scheduling.
            partial_buffer: Buffer for storing partial trajectories.
        """
        self.client = client
        self.config = config or RolloutConfig()
        self.trajectory_buffer = trajectory_buffer
        self.version_provider = version_provider or (lambda: PolicyVersion(version_id=0))

        # Length predictor for smart scheduling (APRIL Learn)
        self._length_predictor = length_predictor or LengthPredictor(
            default_length=256,
            min_length=16,
            max_length=getattr(self.config, 'max_tokens', 4096),
            history_size=1000,
        )

        # Partial trajectory buffer with priority-based retrieval (APRIL Reuse)
        self._partial_buffer = partial_buffer or PartialTrajectoryBuffer(
            max_size=1000,
            max_staleness=getattr(self.config, 'max_partial_staleness', 5),
            min_length_threshold=self.config.get_min_partial_tokens(),
        )

        # Request tracking
        self._active_requests: dict[str, RolloutRequest] = {}
        self._request_counter = 0

        # Metrics
        self._total_generated = 0
        self._total_aborted = 0
        self._total_reused = 0
        self._metrics_history: deque[RolloutMetrics] = deque(maxlen=100)

    @property
    def num_partial(self) -> int:
        """Number of partial trajectories in buffer."""
        return self._partial_buffer.size

    @property
    def length_predictor(self) -> LengthPredictor:
        """Access to the length predictor for external configuration."""
        return self._length_predictor

    @property
    def partial_buffer(self) -> PartialTrajectoryBuffer:
        """Access to the partial trajectory buffer."""
        return self._partial_buffer

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        return f"rollout-{self._request_counter}-{int(time.time() * 1000)}"

    async def generate_batch(
        self,
        prompts: list[str],
        target_count: int | None = None,
        timeout: float | None = None,
        filter_fn: Callable[[Trajectory], bool] | None = None,
        sort_by_length: bool = True,
    ) -> RolloutBatch:
        """Generate a batch of trajectories.

        Uses APRIL strategy:
        1. Oversample to get more candidates
        2. Sort by predicted length (short first) for faster batch completion
        3. Start generation with partial trajectory reuse
        4. Abort long-tail after timeout
        5. Filter and collect results
        6. Learn from observed lengths

        Args:
            prompts: List of prompts to generate from.
            target_count: Target number of successful trajectories.
            timeout: Maximum time to wait for batch completion.
            filter_fn: Optional filter function for trajectories.
            sort_by_length: If True, sort prompts by predicted length (default True).

        Returns:
            RolloutBatch with results and metrics.
        """
        target_count = target_count or len(prompts)
        timeout = timeout or self.config.batch_timeout

        # Apply oversampling
        oversample_count = int(target_count * self.config.oversample_ratio)
        selected_prompts = prompts[:oversample_count]

        # Sort prompts by predicted length (APRIL Learn)
        # Shorter outputs first -> faster batch completion, earlier abort for long-tail
        if sort_by_length and len(selected_prompts) > 1:
            sorted_pairs = self._length_predictor.sort_by_length(
                selected_prompts, ascending=True
            )
            selected_prompts = [p for p, _ in sorted_pairs]

        # Get current policy version for staleness checks
        version = self.version_provider()
        current_version_id = version.version_id

        # Try to reuse partial trajectories first (APRIL Reuse)
        reused_count = 0
        requests = []

        for prompt in selected_prompts:
            request = RolloutRequest(
                request_id=self._generate_request_id(),
                prompt=prompt,
            )

            # Check for partial trajectory match using priority-based buffer
            partial = self._partial_buffer.find_match(
                prompt,
                current_version=current_version_id,
                remove=True,  # Remove matched partial from buffer
            )
            if partial is not None:
                request.partial_trajectory = partial
                request.continuation_prompt = partial.response
                reused_count += 1
                self._total_reused += 1

            requests.append(request)
            self._active_requests[request.request_id] = request

        # Start generation
        start_time = time.time()

        tasks = []
        for request in requests:
            task = self._generate_single(request, version)
            tasks.append(task)

        # Wait with timeout
        results: list[tuple[RolloutRequest, GenerationResult | None]] = []
        try:
            done, pending = await asyncio.wait(
                [asyncio.create_task(t) for t in tasks],
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            for task in done:
                results.append(task.result())

            # Abort pending requests
            if pending:
                for task in pending:
                    task.cancel()
                await self._abort_pending_requests()

        except asyncio.CancelledError:
            await self._abort_pending_requests()
            raise

        end_time = time.time()

        # Process results
        return self._process_batch_results(
            results, target_count, filter_fn, reused_count, start_time, end_time, version
        )

    async def _generate_single(
        self,
        request: RolloutRequest,
        version: PolicyVersion,
    ) -> tuple[RolloutRequest, GenerationResult | None]:
        """Generate a single trajectory."""
        request.started_at = datetime.now()

        try:
            # Build prompt (with continuation if available)
            prompt = request.prompt
            if request.continuation_prompt:
                prompt = request.prompt + request.continuation_prompt

            result = await self.client.generate(
                prompt=prompt,
                request_id=request.request_id,
            )

            request.completed_at = datetime.now()
            return (request, result)

        except Exception as e:
            logger.error(f"Generation failed for {request.request_id}: {e}")
            request.completed_at = datetime.now()
            return (request, None)

        finally:
            self._active_requests.pop(request.request_id, None)

    async def _abort_pending_requests(self) -> int:
        """Abort all pending requests."""
        count = 0
        for request_id in list(self._active_requests.keys()):
            if await self.client.abort_request(request_id):
                count += 1
        self._total_aborted += count
        return count

    def _process_batch_results(
        self,
        results: list[tuple[RolloutRequest, GenerationResult | None]],
        target_count: int,
        filter_fn: Callable[[Trajectory], bool] | None,
        reused_count: int,
        start_time: float,
        end_time: float,
        version: PolicyVersion,
    ) -> RolloutBatch:
        """Process generation results into a batch.

        Also updates length predictor with observed lengths (APRIL Learn).
        """
        trajectories = []
        partial_trajectories = []
        failed_requests = []

        completed = 0
        truncated = 0
        aborted = 0
        failed = 0

        for request, result in results:
            if result is None:
                failed += 1
                failed_requests.append((request.request_id, "Generation failed"))
                continue

            if result.status == GenerationStatus.FAILED:
                failed += 1
                failed_requests.append((request.request_id, result.error or "Unknown error"))
                continue

            if result.status == GenerationStatus.ABORTED:
                aborted += 1
                # Save as partial if long enough (APRIL Partial)
                min_tokens = self.config.get_min_partial_tokens()
                if len(result.tokens) >= min_tokens:
                    partial = self._create_partial_trajectory(request, result, version)
                    partial_trajectories.append(partial)
                    # Use priority-based buffer instead of simple deque
                    self._partial_buffer.add(partial)
                continue

            # Create trajectory
            trajectory = self._create_trajectory(request, result, version)

            # Update length predictor with observed output length (APRIL Learn)
            self._length_predictor.observe(
                prompt=request.prompt,
                actual_length=result.completion_tokens,
            )

            # Apply filter
            if filter_fn and not filter_fn(trajectory):
                continue

            trajectories.append(trajectory)

            if result.status == GenerationStatus.TRUNCATED:
                truncated += 1
            else:
                completed += 1

            # Stop if we have enough
            if len(trajectories) >= target_count:
                break

        self._total_generated += len(trajectories)

        # Record metrics
        metrics = RolloutMetrics(
            num_completed=completed,
            num_aborted=aborted,
            num_failed=failed,
            num_truncated=truncated,
            mean_response_length=sum(len(t.response) for t in trajectories) / max(1, len(trajectories)),
            oversample_ratio=self.config.oversample_ratio,
            abort_ratio=aborted / max(1, completed + aborted + failed),
            reuse_ratio=reused_count / max(1, len(results)),
            total_generation_time_ms=(end_time - start_time) * 1000,
        )
        self._metrics_history.append(metrics)

        return RolloutBatch(
            trajectories=trajectories,
            partial_trajectories=partial_trajectories,
            failed_requests=failed_requests,
            total_requested=len(results),
            completed=completed,
            truncated=truncated,
            aborted=aborted,
            failed=failed,
            reused=reused_count,
            start_time=start_time,
            end_time=end_time,
        )

    def _create_trajectory(
        self,
        request: RolloutRequest,
        result: GenerationResult,
        version: PolicyVersion,
    ) -> Trajectory:
        """Create a Trajectory from generation result."""
        status_map = {
            GenerationStatus.TRUNCATED: TrajectoryStatus.TRUNCATED,
            GenerationStatus.ABORTED: TrajectoryStatus.ABORTED,
            GenerationStatus.FAILED: TrajectoryStatus.FAILED,
        }
        status = status_map.get(result.status, TrajectoryStatus.COMPLETED)

        return Trajectory(
            id=request.request_id,
            prompt=request.prompt,
            response=result.response,
            tokens=result.tokens,
            attention_mask=[1] * len(result.tokens),
            loss_mask=[0] * result.prompt_tokens + [1] * result.completion_tokens,
            log_probs=result.log_probs,
            behavior_log_probs=result.log_probs,  # Same at generation time
            reward=0.0,  # To be filled by reward model
            prompt_length=result.prompt_tokens,
            response_length=result.completion_tokens,
            version=version,
            status=status,
            metadata={
                "weight_version": result.weight_version,
                "finish_reason": result.finish_reason,
                "generation_time_ms": result.elapsed_ms,
            },
        )

    def _create_partial_trajectory(
        self,
        request: RolloutRequest,
        result: GenerationResult,
        version: PolicyVersion,
    ) -> PartialTrajectory:
        """Create a PartialTrajectory from aborted generation."""
        return PartialTrajectory(
            id=request.request_id,
            prompt=request.prompt,
            response=result.response,
            tokens=result.tokens,
            log_probs=result.log_probs,
            version=version,
            metadata={
                "aborted_at": datetime.now().isoformat(),
                "weight_version": result.weight_version,
            },
        )

    async def generate_stream(
        self,
        prompts: list[str],
        target_count: int | None = None,
        min_yield_size: int | None = None,
        sort_by_length: bool = True,
    ):
        """Generate trajectories with streaming results.

        Yields batches as they complete rather than waiting for all.
        Uses APRIL strategy with length prediction and partial reuse.

        Args:
            prompts: List of prompts.
            target_count: Target number of trajectories.
            min_yield_size: Minimum batch size before yielding.
            sort_by_length: If True, sort prompts by predicted length.

        Yields:
            Lists of Trajectories as they complete.
        """
        target_count = target_count or len(prompts)
        min_yield_size = min_yield_size or self.config.min_yield_size

        # Apply oversampling
        oversample_count = int(target_count * self.config.oversample_ratio)
        selected_prompts = prompts[:oversample_count]

        # Sort prompts by predicted length (APRIL Learn)
        if sort_by_length and len(selected_prompts) > 1:
            sorted_pairs = self._length_predictor.sort_by_length(
                selected_prompts, ascending=True
            )
            selected_prompts = [p for p, _ in sorted_pairs]

        version = self.version_provider()
        current_version_id = version.version_id
        pending_results = []
        yielded_count = 0

        # Create tasks for all prompts with partial reuse
        tasks = []
        request_map: dict[str, RolloutRequest] = {}  # Track requests for length observation

        for prompt in selected_prompts:
            request = RolloutRequest(
                request_id=self._generate_request_id(),
                prompt=prompt,
            )

            # Check for partial trajectory match (APRIL Reuse)
            partial = self._partial_buffer.find_match(
                prompt,
                current_version=current_version_id,
                remove=True,
            )
            if partial is not None:
                request.partial_trajectory = partial
                request.continuation_prompt = partial.response
                self._total_reused += 1

            self._active_requests[request.request_id] = request
            request_map[request.request_id] = request
            task = asyncio.create_task(self._generate_single(request, version))
            tasks.append(task)

        # Process as they complete
        for coro in asyncio.as_completed(tasks):
            try:
                request, result = await coro

                if result is not None and result.status in (
                    GenerationStatus.COMPLETED,
                    GenerationStatus.TRUNCATED,
                ):
                    trajectory = self._create_trajectory(request, result, version)
                    pending_results.append(trajectory)

                    # Update length predictor (APRIL Learn)
                    self._length_predictor.observe(
                        prompt=request.prompt,
                        actual_length=result.completion_tokens,
                    )

                elif result is not None and result.status == GenerationStatus.ABORTED:
                    # Save aborted as partial (APRIL Partial)
                    min_tokens = self.config.get_min_partial_tokens()
                    if len(result.tokens) >= min_tokens:
                        partial = self._create_partial_trajectory(request, result, version)
                        self._partial_buffer.add(partial)

                # Yield when we have enough
                if len(pending_results) >= min_yield_size:
                    yield pending_results
                    yielded_count += len(pending_results)
                    pending_results = []

                # Stop if we've yielded enough
                if yielded_count >= target_count:
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await self._abort_pending_requests()
                    break

            except asyncio.CancelledError:
                continue
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")

        # Yield remaining
        if pending_results:
            yield pending_results

    def get_metrics(self) -> RolloutMetrics:
        """Get aggregated rollout metrics."""
        if not self._metrics_history:
            return RolloutMetrics()

        # Aggregate recent metrics
        recent = list(self._metrics_history)[-10:]

        return RolloutMetrics(
            num_completed=sum(m.num_completed for m in recent),
            num_aborted=sum(m.num_aborted for m in recent),
            num_failed=sum(m.num_failed for m in recent),
            num_truncated=sum(m.num_truncated for m in recent),
            mean_response_length=sum(m.mean_response_length for m in recent) / len(recent),
            oversample_ratio=self.config.oversample_ratio,
            abort_ratio=sum(m.abort_ratio for m in recent) / len(recent),
            reuse_ratio=sum(m.reuse_ratio for m in recent) / len(recent),
        )

    def clear_partial_buffer(self) -> int:
        """Clear partial trajectory buffer.

        Returns:
            Number of cleared trajectories.
        """
        return self._partial_buffer.clear()

    def cleanup_stale_partials(self) -> int:
        """Remove stale partial trajectories.

        Removes partials that are too old relative to current policy version.

        Returns:
            Number of removed trajectories.
        """
        current_version = self.version_provider().version_id
        return self._partial_buffer.cleanup_stale(current_version)

    def get_partial_buffer_stats(self) -> dict[str, Any]:
        """Get statistics about the partial trajectory buffer.

        Returns:
            Dict with buffer statistics including size, match rate, etc.
        """
        return self._partial_buffer.get_statistics()

    def get_length_predictor_stats(self) -> dict[str, float]:
        """Get statistics about the length predictor.

        Returns:
            Dict with predictor statistics including mean length, etc.
        """
        return self._length_predictor.get_statistics()
