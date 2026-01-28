"""
Staleness management for Flux.

Provides staleness computation and capacity control for adaptive async training.
Staleness measures how much the policy has changed since data was generated.

Based on AReaL's staleness management with enhanced metrics tracking.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import torch

from flux.core.config import AdaptiveAsyncConfig
from flux.core.types import BatchMetrics, PolicyVersion, StalenessMetrics


@dataclass
class RolloutStats:
    """Statistics for tracking rollout pipeline state.

    Used by StalenessManager to track how many rollouts are in various stages.
    """

    enqueued: int = 0      # Pending in input queue
    running: int = 0       # Currently generating
    accepted: int = 0      # Successfully completed
    rejected: int = 0      # Failed/aborted

    @property
    def total_in_flight(self) -> int:
        """Total rollouts not yet consumed by training."""
        return self.enqueued + self.running + self.accepted

    @property
    def total_submitted(self) -> int:
        """Total rollouts submitted (including rejected)."""
        return self.enqueued + self.running + self.accepted + self.rejected


@dataclass
class StalenessRecord:
    """Record of staleness measurement for history tracking."""

    timestamp: datetime
    staleness: StalenessMetrics
    version: int
    batch_size: int = 0


class StalenessManager:
    """Manages staleness computation and capacity control.

    The staleness manager tracks:
    1. Per-batch staleness metrics (KL, importance weight variance, version gap)
    2. Rolling statistics for adaptive control
    3. Capacity for staleness-aware async scheduling

    Signal Definitions:
    - KL Divergence: D_KL(π_behavior || π_current), token-level averaged. Danger threshold: > 0.1
    - IW Variance: Var(w) where w = π_current/π_behavior, per-trajectory then batch. Danger: > 2.0
    - Version Gap: current_version - trajectory_version, averaged. Danger threshold: > 5

    Staleness Formula (with configurable weights from AdaptiveAsyncConfig):
        kl_contrib = min(1, kl_divergence / kl_normalizer)       # default kl_normalizer = 0.1
        iw_contrib = min(1, iw_variance / iw_normalizer)         # default iw_normalizer = 2.0
        version_contrib = min(1, version_gap / max_version_gap)  # default max_version_gap = 5

        combined = kl_weight * kl_contrib + iw_weight * iw_contrib + version_weight * version_contrib
        # default weights: 0.4, 0.3, 0.3 (configurable in AdaptiveAsyncConfig)

    The combined staleness is smoothed via EMA (alpha=0.1) before being used.

    Capacity is computed to ensure rollouts won't be too stale when consumed:
        capacity = (max_version_gap + current_version + 1) * batch_size - in_flight

    Example:
        manager = StalenessManager(
            config=AdaptiveAsyncConfig(target_staleness=0.15),
            version_provider=lambda: trainer.current_version,
        )

        # After each batch
        staleness = manager.compute_staleness(batch_metrics)

        # Before submitting rollouts
        if manager.get_capacity() > 0:
            submit_rollout()
    """

    def __init__(
        self,
        config: AdaptiveAsyncConfig | None = None,
        version_provider: Callable[[], int] | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the staleness manager.

        Args:
            config: Adaptive async configuration.
            version_provider: Function that returns current policy version.
            batch_size: Number of trajectories per training batch.
        """
        self.config = config or AdaptiveAsyncConfig()
        self.version_provider = version_provider or (lambda: 0)
        self.batch_size = batch_size

        # Thread safety
        self._lock = threading.Lock()

        # Rollout tracking
        self._stats = RolloutStats()

        # Staleness history
        self._history: deque[StalenessRecord] = deque(maxlen=100)

        # EMA of staleness for smooth tracking
        self._ema_staleness = 0.0

        # Steps since last sync
        self._steps_since_sync = 0

        # Maximum staleness allowed (in version gaps)
        self._max_staleness = self.config.max_version_gap

    @property
    def current_staleness(self) -> float:
        """Current EMA staleness value."""
        with self._lock:
            return self._ema_staleness

    @property
    def steps_since_sync(self) -> int:
        """Number of steps since last sync."""
        with self._lock:
            return self._steps_since_sync

    @property
    def stats(self) -> RolloutStats:
        """Current rollout statistics (copy)."""
        with self._lock:
            return RolloutStats(
                enqueued=self._stats.enqueued,
                running=self._stats.running,
                accepted=self._stats.accepted,
                rejected=self._stats.rejected,
            )

    def compute_staleness(
        self,
        batch_metrics: BatchMetrics | None = None,
        *,
        kl_divergence: float | None = None,
        importance_weights: torch.Tensor | None = None,
        version_gap: float | None = None,
    ) -> StalenessMetrics:
        """Compute staleness metrics for a batch.

        Can be called with either:
        1. A BatchMetrics object containing staleness info
        2. Individual metric values

        Args:
            batch_metrics: Batch metrics containing KL and importance weights.
            kl_divergence: KL divergence between current and behavior policy.
            importance_weights: Importance weight tensor for IW variance.
            version_gap: Average version gap of trajectories in batch.

        Returns:
            StalenessMetrics with computed combined staleness.
        """
        metrics = StalenessMetrics(
            kl_normalizer=self.config.kl_normalizer,
            iw_normalizer=self.config.iw_normalizer,
            max_version_gap=self.config.max_version_gap,
            # Use configurable weights from config
            kl_weight=self.config.kl_weight,
            iw_weight=self.config.iw_weight,
            version_weight=self.config.version_weight,
        )

        # Extract values from batch_metrics if provided
        if batch_metrics is not None:
            metrics.kl_divergence = batch_metrics.mean_kl
            metrics.importance_weight_variance = self._compute_iw_variance_from_batch(
                batch_metrics
            )
            metrics.version_gap = batch_metrics.staleness.version_gap

        # Override with explicit values if provided
        if kl_divergence is not None:
            metrics.kl_divergence = kl_divergence
        if importance_weights is not None:
            metrics.importance_weight_variance = self._compute_iw_variance(
                importance_weights
            )
        if version_gap is not None:
            metrics.version_gap = version_gap

        # Compute combined staleness using configurable weights
        metrics.compute_combined()

        # Update EMA and history
        with self._lock:
            alpha = self.config.ema_alpha
            self._ema_staleness = (
                alpha * metrics.combined_staleness
                + (1 - alpha) * self._ema_staleness
            )

            self._history.append(
                StalenessRecord(
                    timestamp=datetime.now(),
                    staleness=metrics,
                    version=self.version_provider(),
                    batch_size=batch_metrics.batch_size if batch_metrics else 0,
                )
            )

            self._steps_since_sync += 1

        return metrics

    def compute_staleness_from_trajectories(
        self,
        current_logprobs: torch.Tensor,
        behavior_logprobs: torch.Tensor,
        trajectory_versions: list[int],
        mask: torch.Tensor | None = None,
    ) -> StalenessMetrics:
        """Compute staleness directly from trajectory data.

        This is more accurate than using batch_metrics as it computes
        the metrics directly from the log probabilities.

        Args:
            current_logprobs: Log probs under current policy [batch, seq].
            behavior_logprobs: Log probs under behavior policy [batch, seq].
            trajectory_versions: Version when each trajectory was generated.
            mask: Optional mask for valid tokens [batch, seq].

        Returns:
            StalenessMetrics with computed values.
        """
        if mask is None:
            mask = torch.ones_like(current_logprobs)

        # Check for empty mask to avoid division by zero
        mask_sum = mask.sum()
        if mask_sum == 0:
            # No valid tokens - return version-gap-only staleness
            current_version = self.version_provider()
            version_gaps = [current_version - v for v in trajectory_versions]
            avg_version_gap = sum(version_gaps) / len(version_gaps) if version_gaps else 0.0
            return self.compute_staleness(version_gap=avg_version_gap)

        # Compute KL divergence: E_q[log q - log p]
        # where q = behavior, p = current
        # KL(behavior || current) = E_behavior[log behavior - log current]
        log_ratio = behavior_logprobs - current_logprobs
        kl = (log_ratio * mask).sum() / mask_sum
        # KL divergence is always non-negative; use abs to handle numerical issues
        kl_divergence = abs(kl.item())

        # Compute importance weights and variance
        # w = current / behavior = exp(log_current - log_behavior)
        log_importance = current_logprobs - behavior_logprobs
        # Sum log probs per sequence (assuming independent tokens)
        seq_log_importance = (log_importance * mask).sum(dim=-1)
        # Clamp to avoid overflow in exp
        seq_log_importance = seq_log_importance.clamp(-20, 20)
        importance_weights = torch.exp(seq_log_importance)
        iw_variance = importance_weights.var().item() if importance_weights.numel() > 1 else 0.0

        # Compute version gap
        current_version = self.version_provider()
        version_gaps = [current_version - v for v in trajectory_versions]
        avg_version_gap = sum(version_gaps) / len(version_gaps) if version_gaps else 0.0

        return self.compute_staleness(
            kl_divergence=kl_divergence,
            importance_weights=importance_weights,
            version_gap=avg_version_gap,
        )

    def _compute_iw_variance(self, importance_weights: torch.Tensor) -> float:
        """Compute variance of importance weights."""
        if importance_weights.numel() == 0:
            return 0.0
        return importance_weights.var().item()

    def _compute_iw_variance_from_batch(self, batch_metrics: BatchMetrics) -> float:
        """Estimate IW variance from batch metrics.

        If we don't have the raw importance weights, estimate variance from
        the mean and max values.
        """
        mean_iw = batch_metrics.mean_importance_weight
        max_iw = batch_metrics.max_importance_weight

        if max_iw <= mean_iw:
            return 0.0

        # Rough estimate: assume weights are roughly uniform between 1 and max
        # Variance of uniform distribution [a, b] is (b-a)^2 / 12
        # Here we use [1, max_iw] as rough bounds
        range_estimate = max_iw - 1.0
        return (range_estimate ** 2) / 12.0

    def get_capacity(self, current_version: int | None = None) -> int:
        """Get remaining capacity for new rollouts.

        Capacity is computed to ensure that rollouts won't exceed max_staleness
        when they are consumed by training.

        Formula:
            capacity = (max_staleness + current_version + 1) * batch_size - in_flight

        Args:
            current_version: Override for current policy version.

        Returns:
            Number of additional rollouts that can be submitted.
        """
        if current_version is None:
            current_version = self.version_provider()

        with self._lock:
            in_flight = self._stats.total_in_flight

            # Maximum samples we can have in flight while staying within staleness
            max_samples = (
                self._max_staleness + current_version + 1
            ) * self.batch_size

            capacity = max_samples - in_flight
            return max(0, capacity)

    def should_sync(self) -> bool:
        """Determine if a sync barrier should be triggered.

        Returns True if:
        1. Staleness exceeds target + tolerance
        2. Too many steps since last sync

        Returns:
            Whether to trigger sync.
        """
        with self._lock:
            threshold = self.config.target_staleness + self.config.tolerance
            return (
                self._ema_staleness > threshold
                or self._steps_since_sync >= self.config.max_steps_without_sync
            )

    def record_sync(self) -> None:
        """Record that a sync barrier was triggered."""
        with self._lock:
            self._steps_since_sync = 0

    # Rollout tracking callbacks
    def on_rollout_enqueued(self, count: int = 1) -> None:
        """Record that rollouts were added to the queue."""
        with self._lock:
            self._stats.enqueued += count

    def on_rollout_submitted(self, count: int = 1) -> None:
        """Record that rollouts moved from queue to running."""
        with self._lock:
            self._stats.enqueued = max(0, self._stats.enqueued - count)
            self._stats.running += count

    def on_rollout_accepted(self, count: int = 1) -> None:
        """Record that rollouts completed successfully."""
        with self._lock:
            self._stats.running = max(0, self._stats.running - count)
            self._stats.accepted += count

    def on_rollout_rejected(self, count: int = 1) -> None:
        """Record that rollouts failed or were aborted."""
        with self._lock:
            self._stats.running = max(0, self._stats.running - count)
            self._stats.rejected += count

    def on_batch_consumed(self, count: int | None = None) -> None:
        """Record that a training batch consumed rollouts.

        Args:
            count: Number of trajectories consumed (defaults to batch_size).
        """
        if count is None:
            count = self.batch_size

        with self._lock:
            self._stats.accepted = max(0, self._stats.accepted - count)

    def reset_stats(self) -> None:
        """Reset rollout statistics."""
        with self._lock:
            self._stats = RolloutStats()

    def get_staleness_history(
        self, last_n: int | None = None
    ) -> list[StalenessRecord]:
        """Get staleness history.

        Args:
            last_n: Number of recent records to return (None = all).

        Returns:
            List of staleness records.
        """
        with self._lock:
            if last_n is None:
                return list(self._history)
            return list(self._history)[-last_n:]

    def get_average_staleness(self, last_n: int = 10) -> float:
        """Get average staleness over recent batches.

        Args:
            last_n: Number of recent batches to average.

        Returns:
            Average combined staleness.
        """
        with self._lock:
            recent = list(self._history)[-last_n:]
            if not recent:
                return 0.0
            return sum(r.staleness.combined_staleness for r in recent) / len(recent)

    def get_version_gap_stats(self) -> dict[str, float]:
        """Get statistics about version gaps in recent history.

        Returns:
            Dict with min, max, mean, std of version gaps.
        """
        with self._lock:
            if not self._history:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

            gaps = [r.staleness.version_gap for r in self._history]
            return {
                "min": min(gaps),
                "max": max(gaps),
                "mean": sum(gaps) / len(gaps),
                "std": (
                    sum((g - sum(gaps) / len(gaps)) ** 2 for g in gaps) / len(gaps)
                ) ** 0.5,
            }
