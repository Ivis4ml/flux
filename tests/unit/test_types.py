"""
Unit tests for Flux type definitions.
"""

import pytest

from flux.core.types import (
    AsyncDecision,
    BatchMetrics,
    PolicyVersion,
    RolloutMetrics,
    StalenessMetrics,
    TrainingPhase,
    TrainingState,
)


class TestPolicyVersion:
    """Tests for PolicyVersion."""

    def test_creation(self) -> None:
        """Test basic version creation."""
        v = PolicyVersion(version_id=5)
        assert v.version_id == 5
        assert v.timestamp is not None

    def test_equality(self) -> None:
        """Test version equality."""
        v1 = PolicyVersion(version_id=5)
        v2 = PolicyVersion(version_id=5)
        v3 = PolicyVersion(version_id=6)

        assert v1 == v2
        assert v1 != v3
        assert v1 == 5  # Compare with int

    def test_comparison(self) -> None:
        """Test version comparison."""
        v1 = PolicyVersion(version_id=5)
        v2 = PolicyVersion(version_id=10)

        assert v1 < v2
        assert v1 < 10

    def test_hash(self) -> None:
        """Test version hashing for dict keys."""
        v = PolicyVersion(version_id=5)
        d = {v: "test"}
        assert d[v] == "test"


class TestTrainingState:
    """Tests for TrainingState."""

    def test_default_state(self) -> None:
        """Test default training state."""
        state = TrainingState()
        assert state.global_step == 0
        assert state.epoch == 0
        assert state.phase == TrainingPhase.WARMUP

    def test_progress_calculation(self) -> None:
        """Test progress calculation."""
        state = TrainingState(global_step=500, total_steps=1000)
        assert state.progress == 0.5

    def test_phase_detection(self) -> None:
        """Test automatic phase detection."""
        # Warmup phase (0-5%)
        state = TrainingState(global_step=2, total_steps=100)
        assert state.get_phase() == TrainingPhase.WARMUP

        # Early phase (5-20%)
        state = TrainingState(global_step=10, total_steps=100)
        assert state.get_phase() == TrainingPhase.EARLY

        # Mid phase (20-70%)
        state = TrainingState(global_step=50, total_steps=100)
        assert state.get_phase() == TrainingPhase.MID

        # Late phase (70-100%)
        state = TrainingState(global_step=80, total_steps=100)
        assert state.get_phase() == TrainingPhase.LATE

    def test_step(self) -> None:
        """Test step advancement."""
        state = TrainingState(total_steps=100)
        assert state.global_step == 0

        state.step()
        assert state.global_step == 1
        assert state.current_version.version_id == 1

    def test_next_epoch(self) -> None:
        """Test epoch advancement."""
        state = TrainingState()
        state.step_in_epoch = 100

        state.next_epoch()
        assert state.epoch == 1
        assert state.step_in_epoch == 0


class TestAsyncDecision:
    """Tests for AsyncDecision."""

    def test_creation(self) -> None:
        """Test decision creation."""
        decision = AsyncDecision(
            async_ratio=0.5,
            should_sync=False,
        )
        assert decision.async_ratio == 0.5
        assert decision.should_sync is False

    def test_ratio_clamping(self) -> None:
        """Test async ratio is clamped to [0, 1]."""
        decision = AsyncDecision(async_ratio=1.5, should_sync=False)
        assert decision.async_ratio == 1.0

        decision = AsyncDecision(async_ratio=-0.5, should_sync=False)
        assert decision.async_ratio == 0.0


class TestStalenessMetrics:
    """Tests for StalenessMetrics."""

    def test_default_values(self) -> None:
        """Test default staleness is zero."""
        metrics = StalenessMetrics()
        assert metrics.kl_divergence == 0.0
        assert metrics.combined_staleness == 0.0

    def test_combined_computation(self) -> None:
        """Test combined staleness computation."""
        metrics = StalenessMetrics(
            kl_divergence=0.05,
            importance_weight_variance=1.0,
            version_gap=2.0,
        )
        combined = metrics.compute_combined()

        # Should be weighted combination
        assert 0.0 <= combined <= 1.0
        assert metrics.combined_staleness == combined

    def test_normalization(self) -> None:
        """Test staleness normalization."""
        # High values should be clamped
        metrics = StalenessMetrics(
            kl_divergence=1.0,  # > kl_normalizer
            importance_weight_variance=10.0,  # > iw_normalizer
            version_gap=10,  # > max_version_gap
        )
        combined = metrics.compute_combined()
        assert combined == 1.0  # All components maxed out


class TestBatchMetrics:
    """Tests for BatchMetrics."""

    def test_default_values(self) -> None:
        """Test default metric values."""
        metrics = BatchMetrics()
        assert metrics.policy_loss == 0.0
        assert metrics.mean_importance_weight == 1.0

    def test_to_dict(self) -> None:
        """Test conversion to logging dictionary."""
        metrics = BatchMetrics(
            policy_loss=0.5,
            mean_reward=1.0,
            batch_size=32,
        )
        d = metrics.to_dict()

        assert "loss/policy" in d
        assert d["loss/policy"] == 0.5
        assert d["reward/mean"] == 1.0
        assert d["batch/size"] == 32


class TestRolloutMetrics:
    """Tests for RolloutMetrics."""

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        metrics = RolloutMetrics(
            num_completed=80,
            num_aborted=10,
            num_failed=10,
        )
        assert metrics.success_rate == 0.8

    def test_success_rate_zero(self) -> None:
        """Test success rate with no completions."""
        metrics = RolloutMetrics()
        assert metrics.success_rate == 0.0

    def test_to_dict(self) -> None:
        """Test conversion to logging dictionary."""
        metrics = RolloutMetrics(
            num_completed=100,
            mean_response_length=50.0,
        )
        d = metrics.to_dict()

        assert "rollout/completed" in d
        assert d["rollout/completed"] == 100
