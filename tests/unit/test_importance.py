"""Unit tests for Flux importance correction module."""

import pytest
import torch

from flux.controller.importance import (
    ImportanceCorrectionConfig,
    UnifiedImportanceCorrection,
    compute_importance_weights,
    compute_staleness_decay,
    compute_trajectory_consistency,
)


# ============================================================================
# Importance Weight Computation Tests
# ============================================================================


class TestImportanceWeights:
    """Tests for importance weight computation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample log probability data."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        return {
            "behavior_logp": torch.randn(batch_size, seq_len) - 2,
            "current_logp": torch.randn(batch_size, seq_len) - 2,
            "mask": torch.ones(batch_size, seq_len),
            "version_gap": torch.tensor([0, 1, 3, 5]),
        }

    def test_importance_weights_shape(self, sample_data):
        """Importance weights should have correct shape."""
        weights = compute_importance_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
        )

        assert weights.shape == (4,)  # One weight per sample

    def test_importance_weights_positive(self, sample_data):
        """Importance weights should be positive."""
        weights = compute_importance_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
        )

        assert (weights > 0).all()

    def test_importance_weights_clipping(self, sample_data):
        """Importance weights should be clipped to bounds."""
        # Create extreme difference
        behavior_logp = torch.zeros(4, 10)
        current_logp = torch.ones(4, 10) * 10  # Very high ratio

        weights = compute_importance_weights(
            behavior_log_probs=behavior_logp,
            current_log_probs=current_logp,
            response_mask=sample_data["mask"],
            max_weight=5.0,
            min_weight=0.2,
        )

        assert (weights <= 5.0).all()
        assert (weights >= 0.2).all()

    def test_importance_weights_staleness_decay(self, sample_data):
        """Staleness should decrease weights."""
        # Same log probs, different staleness
        weights_no_staleness = compute_importance_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["behavior_logp"],  # Same as behavior
            response_mask=sample_data["mask"],
            version_gap=None,
            normalize=False,
        )

        weights_with_staleness = compute_importance_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["behavior_logp"],
            response_mask=sample_data["mask"],
            version_gap=sample_data["version_gap"],
            staleness_decay=0.9,
            normalize=False,
        )

        # Sample with version_gap=5 should have lower weight
        assert weights_with_staleness[3] < weights_no_staleness[3]
        # Sample with version_gap=0 should have same weight
        assert torch.isclose(weights_with_staleness[0], weights_no_staleness[0])

    def test_importance_weights_normalization(self, sample_data):
        """Normalized weights should sum to batch size."""
        weights = compute_importance_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
            normalize=True,
        )

        assert torch.isclose(weights.sum(), torch.tensor(4.0), atol=1e-5)

    def test_importance_weights_respects_mask(self, sample_data):
        """Importance weights should only use masked tokens."""
        # Create partial mask
        mask = sample_data["mask"].clone()
        mask[:, 5:] = 0

        weights = compute_importance_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=mask,
        )

        assert not torch.isnan(weights).any()


# ============================================================================
# Staleness Decay Tests
# ============================================================================


class TestStalenessDecay:
    """Tests for staleness decay computation."""

    def test_staleness_decay_shape(self):
        """Staleness decay should return correct shape."""
        version_gap = torch.tensor([0, 1, 2, 3])
        weights = compute_staleness_decay(version_gap, decay_rate=0.99)

        assert weights.shape == (4,)

    def test_staleness_decay_values(self):
        """Staleness decay should decrease with gap."""
        version_gap = torch.tensor([0, 1, 2, 3])
        weights = compute_staleness_decay(version_gap, decay_rate=0.9)

        # Check decreasing order
        assert (weights[:-1] >= weights[1:]).all()

        # Check specific values
        assert torch.isclose(weights[0], torch.tensor(1.0))  # 0.9^0 = 1
        assert torch.isclose(weights[1], torch.tensor(0.9))  # 0.9^1 = 0.9
        assert torch.isclose(weights[2], torch.tensor(0.81))  # 0.9^2 = 0.81

    def test_staleness_decay_minimum(self):
        """Staleness decay should respect minimum weight."""
        version_gap = torch.tensor([100])  # Very stale
        weights = compute_staleness_decay(
            version_gap, decay_rate=0.9, min_weight=0.1
        )

        assert weights[0] >= 0.1


# ============================================================================
# Trajectory Consistency Tests
# ============================================================================


class TestTrajectoryConsistency:
    """Tests for trajectory consistency computation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        return {
            "behavior_logp": torch.randn(batch_size, seq_len) - 2,
            "current_logp": torch.randn(batch_size, seq_len) - 2,
            "mask": torch.ones(batch_size, seq_len),
        }

    def test_consistency_shape(self, sample_data):
        """Consistency should return correct shape."""
        consistency = compute_trajectory_consistency(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
        )

        assert consistency.shape == (4,)

    def test_consistency_range(self, sample_data):
        """Consistency should be in [0, 1]."""
        for method in ["kl", "ratio", "cosine"]:
            consistency = compute_trajectory_consistency(
                behavior_log_probs=sample_data["behavior_logp"],
                current_log_probs=sample_data["current_logp"],
                response_mask=sample_data["mask"],
                method=method,
            )

            assert (consistency >= 0).all()
            assert (consistency <= 1).all()

    def test_consistency_identical_policies(self, sample_data):
        """Identical policies should have high consistency."""
        consistency = compute_trajectory_consistency(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["behavior_logp"],  # Same as behavior
            response_mask=sample_data["mask"],
            method="kl",
        )

        # Should be close to 1
        assert (consistency > 0.9).all()

    def test_consistency_different_policies(self, sample_data):
        """Very different policies should have low consistency."""
        # Create very different policies
        behavior_logp = torch.zeros(4, 10)
        current_logp = torch.ones(4, 10) * -10  # Very different

        consistency = compute_trajectory_consistency(
            behavior_log_probs=behavior_logp,
            current_log_probs=current_logp,
            response_mask=sample_data["mask"],
            method="kl",
        )

        # Should be lower
        assert (consistency < 0.5).all()

    def test_consistency_invalid_method(self, sample_data):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown consistency method"):
            compute_trajectory_consistency(
                behavior_log_probs=sample_data["behavior_logp"],
                current_log_probs=sample_data["current_logp"],
                response_mask=sample_data["mask"],
                method="invalid",
            )


# ============================================================================
# UnifiedImportanceCorrection Tests
# ============================================================================


class TestUnifiedImportanceCorrection:
    """Tests for unified importance correction class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        return {
            "behavior_logp": torch.randn(batch_size, seq_len) - 2,
            "current_logp": torch.randn(batch_size, seq_len) - 2,
            "mask": torch.ones(batch_size, seq_len),
            "version_gap": torch.tensor([0, 1, 3, 5]),
        }

    def test_default_config(self):
        """Default config should be reasonable."""
        corrector = UnifiedImportanceCorrection()

        assert corrector.config.staleness_decay == 0.99
        assert corrector.config.max_importance_weight == 5.0
        assert corrector.config.use_staleness is True

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = ImportanceCorrectionConfig(
            staleness_decay=0.95,
            max_importance_weight=3.0,
            use_staleness=False,
        )
        corrector = UnifiedImportanceCorrection(config=config)

        assert corrector.config.staleness_decay == 0.95
        assert corrector.config.max_importance_weight == 3.0
        assert corrector.config.use_staleness is False

    def test_compute_weights(self, sample_data):
        """Should compute unified weights correctly."""
        corrector = UnifiedImportanceCorrection()

        weights = corrector.compute_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
            version_gap=sample_data["version_gap"],
        )

        assert weights.shape == (4,)
        assert (weights > 0).all()

    def test_compute_weights_with_version_info(self, sample_data):
        """Should compute version gap from version info."""
        corrector = UnifiedImportanceCorrection()

        weights = corrector.compute_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
            current_version=10,
            behavior_versions=torch.tensor([10, 9, 7, 5]),
        )

        assert weights.shape == (4,)

    def test_compute_weights_no_staleness(self, sample_data):
        """Should work without staleness correction."""
        config = ImportanceCorrectionConfig(use_staleness=False)
        corrector = UnifiedImportanceCorrection(config=config)

        weights = corrector.compute_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
            version_gap=sample_data["version_gap"],
        )

        # All samples should have similar base weights (no staleness penalty)
        assert weights.shape == (4,)

    def test_compute_weights_no_consistency(self, sample_data):
        """Should work without consistency weighting."""
        config = ImportanceCorrectionConfig(use_consistency=False)
        corrector = UnifiedImportanceCorrection(config=config)

        weights = corrector.compute_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
        )

        assert weights.shape == (4,)

    def test_filter_samples(self, sample_data):
        """Should correctly filter low-weight samples."""
        corrector = UnifiedImportanceCorrection()

        # Create data where some samples should be filtered
        behavior_logp = torch.zeros(4, 10)
        current_logp = torch.zeros(4, 10)
        current_logp[0] = -10  # Very different policy for first sample

        include_mask = corrector.filter_samples(
            behavior_log_probs=behavior_logp,
            current_log_probs=current_logp,
            response_mask=sample_data["mask"],
            min_weight=0.5,
        )

        assert include_mask.shape == (4,)
        assert include_mask.dtype == torch.bool

    def test_get_metrics(self, sample_data):
        """Should compute metrics correctly."""
        corrector = UnifiedImportanceCorrection()

        weights = corrector.compute_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
            version_gap=sample_data["version_gap"],
        )

        metrics = corrector.get_metrics(weights, sample_data["version_gap"])

        assert "importance/mean_weight" in metrics
        assert "importance/std_weight" in metrics
        assert "importance/effective_samples" in metrics
        assert "importance/mean_version_gap" in metrics

    def test_effective_samples_metric(self, sample_data):
        """Effective samples should be <= actual samples."""
        corrector = UnifiedImportanceCorrection()

        weights = corrector.compute_weights(
            behavior_log_probs=sample_data["behavior_logp"],
            current_log_probs=sample_data["current_logp"],
            response_mask=sample_data["mask"],
        )

        metrics = corrector.get_metrics(weights)

        # Effective samples should be <= batch size
        assert metrics["importance/effective_samples"] <= 4.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestImportanceCorrectionIntegration:
    """Integration tests for importance correction with algorithms."""

    def test_with_ppo_loss(self):
        """Importance weights should work with PPO loss."""
        from flux.training.algorithms import get_policy_loss_fn

        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        behavior_logp = torch.randn(batch_size, seq_len) - 2
        current_logp = torch.randn(batch_size, seq_len) - 2
        mask = torch.ones(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)

        # Compute importance weights
        corrector = UnifiedImportanceCorrection()
        weights = corrector.compute_weights(
            behavior_log_probs=behavior_logp,
            current_log_probs=current_logp,
            response_mask=mask,
        )

        # Use with PPO loss
        loss_fn = get_policy_loss_fn("ppo")
        loss, metrics = loss_fn(
            old_log_prob=behavior_logp,
            log_prob=current_logp,
            advantages=advantages,
            response_mask=mask,
            importance_weights=weights,
        )

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_weights_affect_loss_gradient(self):
        """Importance weights should affect loss magnitude."""
        from flux.training.algorithms import get_policy_loss_fn

        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        behavior_logp = torch.randn(batch_size, seq_len) - 2
        current_logp = torch.randn(batch_size, seq_len) - 2
        mask = torch.ones(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)

        loss_fn = get_policy_loss_fn("ppo")

        # Loss without weights
        loss_no_weights, _ = loss_fn(
            old_log_prob=behavior_logp,
            log_prob=current_logp,
            advantages=advantages,
            response_mask=mask,
        )

        # Loss with reduced weights
        weights = torch.ones(batch_size) * 0.5
        loss_with_weights, _ = loss_fn(
            old_log_prob=behavior_logp,
            log_prob=current_logp,
            advantages=advantages,
            response_mask=mask,
            importance_weights=weights,
        )

        # Weighted loss should be ~half magnitude
        assert torch.isclose(loss_with_weights, loss_no_weights * 0.5, rtol=0.1)
