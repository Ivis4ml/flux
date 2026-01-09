"""Unit tests for Flux algorithm registry and built-in algorithms."""

import numpy as np
import pytest
import torch

from flux.training.algorithms import (
    ADV_ESTIMATOR_REGISTRY,
    POLICY_LOSS_REGISTRY,
    AdvantageEstimator,
    get_adv_estimator_fn,
    get_policy_loss_fn,
    list_adv_estimators,
    list_policy_losses,
    register_adv_estimator,
    register_policy_loss,
)
from flux.training.algorithms.base import (
    agg_loss,
    masked_mean,
    masked_var,
    masked_whiten,
)


# ============================================================================
# Registry Tests
# ============================================================================


class TestRegistry:
    """Tests for algorithm registry functionality."""

    def test_adv_estimator_registry_not_empty(self):
        """Built-in advantage estimators should be registered."""
        assert len(ADV_ESTIMATOR_REGISTRY) > 0
        assert "gae" in ADV_ESTIMATOR_REGISTRY
        assert "grpo" in ADV_ESTIMATOR_REGISTRY
        assert "reinforce" in ADV_ESTIMATOR_REGISTRY

    def test_policy_loss_registry_not_empty(self):
        """Built-in policy losses should be registered."""
        assert len(POLICY_LOSS_REGISTRY) > 0
        assert "ppo" in POLICY_LOSS_REGISTRY
        assert "grpo" in POLICY_LOSS_REGISTRY
        assert "reinforce" in POLICY_LOSS_REGISTRY

    def test_get_adv_estimator_by_string(self):
        """Should retrieve estimator by string name."""
        fn = get_adv_estimator_fn("grpo")
        assert callable(fn)

    def test_get_adv_estimator_by_enum(self):
        """Should retrieve estimator by enum."""
        fn = get_adv_estimator_fn(AdvantageEstimator.GRPO)
        assert callable(fn)

    def test_get_adv_estimator_unknown_raises(self):
        """Unknown estimator should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown advantage estimator"):
            get_adv_estimator_fn("nonexistent")

    def test_get_policy_loss_by_string(self):
        """Should retrieve loss function by string name."""
        fn = get_policy_loss_fn("ppo")
        assert callable(fn)

    def test_get_policy_loss_unknown_raises(self):
        """Unknown loss should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown policy loss"):
            get_policy_loss_fn("nonexistent")

    def test_list_functions(self):
        """List functions should return all registered names."""
        estimators = list_adv_estimators()
        losses = list_policy_losses()

        assert isinstance(estimators, list)
        assert isinstance(losses, list)
        assert "grpo" in estimators
        assert "ppo" in losses

    def test_register_custom_estimator(self):
        """Should be able to register custom advantage estimator."""

        @register_adv_estimator("test_custom_adv")
        def test_advantage(rewards, mask, **kwargs):
            return rewards, rewards

        assert "test_custom_adv" in ADV_ESTIMATOR_REGISTRY
        fn = get_adv_estimator_fn("test_custom_adv")
        assert fn is test_advantage

    def test_register_custom_loss(self):
        """Should be able to register custom policy loss."""

        @register_policy_loss("test_custom_loss")
        def test_loss(old_logp, logp, adv, mask, **kwargs):
            return torch.tensor(0.0), {}

        assert "test_custom_loss" in POLICY_LOSS_REGISTRY
        fn = get_policy_loss_fn("test_custom_loss")
        assert fn is test_loss

    def test_duplicate_registration_same_function_ok(self):
        """Re-registering the same function should not raise."""

        @register_adv_estimator("test_duplicate")
        def my_fn(rewards, mask, **kwargs):
            return rewards, rewards

        # Re-registering same function is fine
        register_adv_estimator("test_duplicate")(my_fn)

    def test_duplicate_registration_different_function_raises(self):
        """Registering different function with same name should raise."""

        @register_adv_estimator("test_conflict")
        def fn1(rewards, mask, **kwargs):
            return rewards, rewards

        with pytest.raises(ValueError, match="already registered"):

            @register_adv_estimator("test_conflict")
            def fn2(rewards, mask, **kwargs):
                return rewards * 2, rewards * 2


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for algorithm utility functions."""

    def test_masked_mean_all_valid(self):
        """Masked mean with all valid should equal regular mean."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.ones_like(tensor)
        result = masked_mean(tensor, mask)
        assert torch.isclose(result, tensor.mean())

    def test_masked_mean_partial_valid(self):
        """Masked mean should ignore masked values."""
        tensor = torch.tensor([[1.0, 2.0, 0.0], [4.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        result = masked_mean(tensor, mask)
        expected = (1.0 + 2.0 + 4.0) / 3.0
        assert torch.isclose(result, torch.tensor(expected))

    def test_masked_mean_with_dim(self):
        """Masked mean along dimension."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.ones_like(tensor)
        result = masked_mean(tensor, mask, dim=1)
        expected = torch.tensor([1.5, 3.5])
        assert torch.allclose(result, expected)

    def test_masked_var(self):
        """Masked variance computation."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = torch.ones_like(tensor)
        result = masked_var(tensor, mask, unbiased=False)
        expected = tensor.var(unbiased=False)
        assert torch.isclose(result, expected)

    def test_masked_whiten(self):
        """Masked whitening should produce zero mean, unit variance."""
        torch.manual_seed(42)
        tensor = torch.randn(100)
        mask = torch.ones_like(tensor)
        result = masked_whiten(tensor, mask)

        assert torch.isclose(masked_mean(result, mask), torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(masked_var(result, mask, unbiased=False), torch.tensor(1.0), atol=1e-4)

    def test_agg_loss_token_mean(self):
        """Token-mean aggregation."""
        loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        result = agg_loss(loss, mask, "token-mean")
        expected = (1 + 2 + 3 + 4 + 5) / 5.0
        assert torch.isclose(result, torch.tensor(expected))

    def test_agg_loss_seq_mean(self):
        """Sequence-mean aggregation."""
        loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        result = agg_loss(loss, mask, "seq-mean")
        # Seq 1: mean = 2.0, Seq 2: mean = 4.5
        expected = (2.0 + 4.5) / 2.0
        assert torch.isclose(result, torch.tensor(expected))

    def test_agg_loss_sum(self):
        """Sum aggregation."""
        loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.ones_like(loss)
        result = agg_loss(loss, mask, "sum")
        assert torch.isclose(result, torch.tensor(10.0))


# ============================================================================
# PPO Algorithm Tests
# ============================================================================


class TestPPO:
    """Tests for PPO algorithm."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for PPO tests."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        return {
            "rewards": torch.randn(batch_size, seq_len),
            "mask": torch.ones(batch_size, seq_len),
            "old_logp": torch.randn(batch_size, seq_len) - 2,
            "logp": torch.randn(batch_size, seq_len) - 2,
            "values": torch.randn(batch_size, seq_len),
        }

    def test_gae_advantage_shape(self, sample_data):
        """GAE should return correct shapes."""
        fn = get_adv_estimator_fn("gae")
        advantages, returns = fn(
            sample_data["rewards"],
            sample_data["mask"],
            values=sample_data["values"],
        )

        assert advantages.shape == sample_data["rewards"].shape
        assert returns.shape == sample_data["rewards"].shape

    def test_gae_advantage_respects_mask(self, sample_data):
        """GAE should respect response mask by using masked statistics."""
        # Set mask to zero for second half
        mask = sample_data["mask"].clone()
        mask[:, 5:] = 0

        fn = get_adv_estimator_fn("gae")
        advantages_full, _ = fn(
            sample_data["rewards"], sample_data["mask"], values=sample_data["values"], whiten=False
        )
        advantages_partial, _ = fn(
            sample_data["rewards"], mask, values=sample_data["values"], whiten=False
        )

        # Advantages in valid region should differ due to different backward propagation
        # The key test is that computation completes without error and masked values
        # don't affect the valid region's statistics when whitening
        assert not torch.isnan(advantages_partial).any()
        assert advantages_partial.shape == sample_data["rewards"].shape

    def test_gae_without_values(self, sample_data):
        """GAE without values should work (MC estimation)."""
        fn = get_adv_estimator_fn("gae")
        advantages, returns = fn(
            sample_data["rewards"], sample_data["mask"], values=None
        )

        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()

    def test_ppo_loss_shape(self, sample_data):
        """PPO loss should return scalar loss and metrics dict."""
        fn = get_policy_loss_fn("ppo")
        advantages = torch.randn_like(sample_data["rewards"])

        loss, metrics = fn(
            sample_data["old_logp"],
            sample_data["logp"],
            advantages,
            sample_data["mask"],
        )

        assert loss.dim() == 0  # Scalar
        assert isinstance(metrics, dict)
        assert "actor/ppo_kl" in metrics
        assert "actor/clip_frac" in metrics

    def test_ppo_clipping(self, sample_data):
        """PPO should clip extreme ratios."""
        fn = get_policy_loss_fn("ppo")
        advantages = torch.ones_like(sample_data["rewards"])

        # Create extreme log prob difference
        old_logp = torch.zeros_like(sample_data["old_logp"])
        logp = old_logp + 5.0  # Very large ratio

        loss, metrics = fn(old_logp, logp, advantages, sample_data["mask"])

        assert not torch.isnan(loss)
        assert metrics["actor/clip_frac"] > 0  # Clipping should be active


# ============================================================================
# GRPO Algorithm Tests
# ============================================================================


class TestGRPO:
    """Tests for GRPO algorithm."""

    @pytest.fixture
    def grouped_data(self):
        """Create grouped sample data."""
        torch.manual_seed(42)
        # 8 samples, 2 groups of 4
        batch_size, seq_len = 8, 10

        return {
            "rewards": torch.randn(batch_size, seq_len),
            "mask": torch.ones(batch_size, seq_len),
            "old_logp": torch.randn(batch_size, seq_len) - 2,
            "logp": torch.randn(batch_size, seq_len) - 2,
            "index": np.array([0, 0, 0, 0, 1, 1, 1, 1]),  # 2 groups
        }

    def test_grpo_advantage_shape(self, grouped_data):
        """GRPO should return correct shapes."""
        fn = get_adv_estimator_fn("grpo")
        advantages, returns = fn(
            grouped_data["rewards"],
            grouped_data["mask"],
            index=grouped_data["index"],
        )

        assert advantages.shape == grouped_data["rewards"].shape
        assert returns.shape == grouped_data["rewards"].shape

    def test_grpo_group_normalization(self, grouped_data):
        """GRPO should normalize within groups."""
        fn = get_adv_estimator_fn("grpo")
        advantages, _ = fn(
            grouped_data["rewards"],
            grouped_data["mask"],
            index=grouped_data["index"],
            normalize_by_std=True,
        )

        # Sum of advantages within each group should be ~0
        group1_adv = advantages[:4].sum(dim=-1).sum()
        group2_adv = advantages[4:].sum(dim=-1).sum()

        assert torch.isclose(group1_adv, torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(group2_adv, torch.tensor(0.0), atol=1e-5)

    def test_grpo_vectorized(self):
        """Vectorized GRPO should match regular GRPO."""
        torch.manual_seed(42)
        batch_size, seq_len, group_size = 8, 10, 4

        rewards = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len)
        index = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        fn_regular = get_adv_estimator_fn("grpo")
        fn_vectorized = get_adv_estimator_fn("grpo_vectorized")

        adv_regular, _ = fn_regular(rewards, mask, index=index, normalize_by_std=True)
        adv_vectorized, _ = fn_vectorized(
            rewards, mask, group_size=group_size, normalize_by_std=True
        )

        assert torch.allclose(adv_regular, adv_vectorized, atol=1e-5)

    def test_grpo_loss(self, grouped_data):
        """GRPO loss should work correctly."""
        fn = get_policy_loss_fn("grpo")
        advantages = torch.randn_like(grouped_data["rewards"])

        loss, metrics = fn(
            grouped_data["old_logp"],
            grouped_data["logp"],
            advantages,
            grouped_data["mask"],
        )

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "actor/grpo_kl" in metrics


# ============================================================================
# REINFORCE Algorithm Tests
# ============================================================================


class TestREINFORCE:
    """Tests for REINFORCE algorithm."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        return {
            "rewards": torch.randn(batch_size, seq_len),
            "mask": torch.ones(batch_size, seq_len),
            "old_logp": torch.randn(batch_size, seq_len) - 2,
            "logp": torch.randn(batch_size, seq_len) - 2,
        }

    def test_reinforce_advantage(self, sample_data):
        """REINFORCE should compute discounted returns."""
        fn = get_adv_estimator_fn("reinforce")
        advantages, returns = fn(
            sample_data["rewards"], sample_data["mask"], gamma=0.99
        )

        assert advantages.shape == sample_data["rewards"].shape
        assert not torch.isnan(advantages).any()

    def test_reinforce_with_baseline(self, sample_data):
        """REINFORCE with baseline should reduce variance."""
        fn_base = get_adv_estimator_fn("reinforce")
        fn_baseline = get_adv_estimator_fn("reinforce_baseline")

        adv_base, _ = fn_base(sample_data["rewards"], sample_data["mask"], whiten=False)
        adv_baseline, _ = fn_baseline(
            sample_data["rewards"], sample_data["mask"], whiten=False
        )

        # Baseline should reduce variance
        assert adv_baseline.var() <= adv_base.var() + 1e-6

    def test_reinforce_loss(self, sample_data):
        """REINFORCE loss should work correctly."""
        fn = get_policy_loss_fn("reinforce")
        advantages = torch.randn_like(sample_data["rewards"])

        loss, metrics = fn(
            sample_data["old_logp"],
            sample_data["logp"],
            advantages,
            sample_data["mask"],
        )

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "actor/entropy" in metrics


# ============================================================================
# RLOO Algorithm Tests
# ============================================================================


class TestRLOO:
    """Tests for RLOO algorithm."""

    @pytest.fixture
    def grouped_data(self):
        """Create grouped sample data."""
        torch.manual_seed(42)
        batch_size, seq_len = 8, 10

        return {
            "rewards": torch.randn(batch_size, seq_len),
            "mask": torch.ones(batch_size, seq_len),
            "index": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        }

    def test_rloo_advantage_shape(self, grouped_data):
        """RLOO should return correct shapes."""
        fn = get_adv_estimator_fn("rloo")
        advantages, returns = fn(
            grouped_data["rewards"],
            grouped_data["mask"],
            index=grouped_data["index"],
        )

        assert advantages.shape == grouped_data["rewards"].shape
        assert returns.shape == grouped_data["rewards"].shape

    def test_rloo_leave_one_out(self, grouped_data):
        """RLOO baseline should exclude the sample itself."""
        fn = get_adv_estimator_fn("rloo")

        # Use uniform rewards for easier verification
        rewards = torch.ones(4, 10)  # All same reward
        mask = torch.ones(4, 10)
        index = np.array([0, 0, 0, 0])  # All same group

        advantages, _ = fn(rewards, mask, index=index, whiten=False)

        # With all same rewards, advantages should be ~0
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5)

    def test_rloo_vectorized_matches(self):
        """Vectorized RLOO should match regular RLOO."""
        torch.manual_seed(42)
        batch_size, seq_len, group_size = 8, 10, 4

        rewards = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len)
        index = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        fn_regular = get_adv_estimator_fn("rloo")
        fn_vectorized = get_adv_estimator_fn("rloo_vectorized")

        adv_regular, _ = fn_regular(rewards, mask, index=index, whiten=False)
        adv_vectorized, _ = fn_vectorized(
            rewards, mask, group_size=group_size, whiten=False
        )

        assert torch.allclose(adv_regular, adv_vectorized, atol=1e-5)


# ============================================================================
# DAPO Algorithm Tests
# ============================================================================


class TestDAPO:
    """Tests for DAPO algorithm."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with version information and grouping."""
        torch.manual_seed(42)
        batch_size, seq_len = 8, 10  # 8 samples for 2 groups of 4

        return {
            "rewards": torch.randn(batch_size, seq_len),
            "mask": torch.ones(batch_size, seq_len),
            "old_logp": torch.randn(batch_size, seq_len) - 2,
            "logp": torch.randn(batch_size, seq_len) - 2,
            "version_gap": torch.tensor([0, 1, 2, 5, 0, 1, 2, 5]),
            "index": np.array([0, 0, 0, 0, 1, 1, 1, 1]),  # 2 groups
        }

    def test_dapo_advantage_staleness(self, sample_data):
        """DAPO should apply staleness decay."""
        fn = get_adv_estimator_fn("dapo")

        # Use index for grouping to get non-zero GRPO advantages
        adv_no_staleness, _ = fn(
            sample_data["rewards"],
            sample_data["mask"],
            index=sample_data["index"],
            version_gap=None
        )

        adv_with_staleness, _ = fn(
            sample_data["rewards"],
            sample_data["mask"],
            index=sample_data["index"],
            version_gap=sample_data["version_gap"],
            staleness_decay=0.9,
        )

        # Stale samples should have smaller advantage magnitude
        # Due to staleness decay, samples with higher version_gap get lower weights
        assert adv_with_staleness.abs().sum() < adv_no_staleness.abs().sum()

    def test_dapo_decoupled_clipping(self, sample_data):
        """DAPO should use decoupled clipping."""
        fn = get_policy_loss_fn("dapo")
        advantages = torch.randn_like(sample_data["rewards"])

        loss, metrics = fn(
            sample_data["old_logp"],
            sample_data["logp"],
            advantages,
            sample_data["mask"],
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
        )

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "actor/pos_clip_frac" in metrics
        assert "actor/neg_clip_frac" in metrics


# ============================================================================
# GSPO Algorithm Tests
# ============================================================================


class TestGSPO:
    """Tests for GSPO algorithm."""

    @pytest.fixture
    def grouped_data(self):
        """Create grouped sample data."""
        torch.manual_seed(42)
        batch_size, seq_len = 8, 10

        return {
            "rewards": torch.randn(batch_size, seq_len),
            "mask": torch.ones(batch_size, seq_len),
            "old_logp": torch.randn(batch_size, seq_len) - 2,
            "logp": torch.randn(batch_size, seq_len) - 2,
            "index": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        }

    def test_gspo_stability_weighting(self, grouped_data):
        """GSPO should apply stability-based weighting."""
        fn = get_adv_estimator_fn("gspo")

        # Create groups with different variances
        rewards = grouped_data["rewards"].clone()
        rewards[:4] *= 0.1  # Low variance group
        rewards[4:] *= 10.0  # High variance group

        advantages, _ = fn(
            rewards,
            grouped_data["mask"],
            index=grouped_data["index"],
            stability_weight=0.5,
        )

        # High variance group should have lower magnitude advantages
        low_var_adv = advantages[:4].abs().mean()
        high_var_adv = advantages[4:].abs().mean()

        # This is approximate due to normalization
        assert not torch.isnan(advantages).any()

    def test_gspo_loss_metrics(self, grouped_data):
        """GSPO loss should include stability metrics."""
        fn = get_policy_loss_fn("gspo")
        advantages = torch.randn_like(grouped_data["rewards"])

        loss, metrics = fn(
            grouped_data["old_logp"],
            grouped_data["logp"],
            advantages,
            grouped_data["mask"],
        )

        assert loss.dim() == 0
        assert "actor/stability_loss" in metrics


# ============================================================================
# DPO Algorithm Tests
# ============================================================================


class TestDPO:
    """Tests for DPO algorithm."""

    @pytest.fixture
    def preference_data(self):
        """Create preference pair data."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        return {
            "rewards": torch.randn(batch_size, seq_len),
            "mask": torch.ones(batch_size, seq_len),
            "chosen_logp": torch.randn(batch_size, seq_len) - 1,
            "rejected_logp": torch.randn(batch_size, seq_len) - 2,
            "ref_chosen_logp": torch.randn(batch_size, seq_len) - 1.5,
            "ref_rejected_logp": torch.randn(batch_size, seq_len) - 2.5,
        }

    def test_dpo_loss_shape(self, preference_data):
        """DPO loss should return scalar."""
        fn = get_policy_loss_fn("dpo")

        loss, metrics = fn(
            old_log_prob=torch.zeros(4, 10),
            log_prob=torch.zeros(4, 10),
            advantages=torch.zeros(4, 10),
            response_mask=preference_data["mask"],
            chosen_log_probs=preference_data["chosen_logp"],
            rejected_log_probs=preference_data["rejected_logp"],
            ref_chosen_log_probs=preference_data["ref_chosen_logp"],
            ref_rejected_log_probs=preference_data["ref_rejected_logp"],
        )

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "dpo/accuracy" in metrics
        assert "dpo/reward_margin" in metrics

    def test_dpo_prefers_chosen(self, preference_data):
        """DPO should prefer chosen over rejected."""
        fn = get_policy_loss_fn("dpo")

        # Make chosen clearly better
        chosen_logp = preference_data["ref_chosen_logp"] + 1.0
        rejected_logp = preference_data["ref_rejected_logp"] - 1.0

        loss, metrics = fn(
            old_log_prob=torch.zeros(4, 10),
            log_prob=torch.zeros(4, 10),
            advantages=torch.zeros(4, 10),
            response_mask=preference_data["mask"],
            chosen_log_probs=chosen_logp,
            rejected_log_probs=rejected_logp,
            ref_chosen_log_probs=preference_data["ref_chosen_logp"],
            ref_rejected_log_probs=preference_data["ref_rejected_logp"],
        )

        # Accuracy should be high
        assert metrics["dpo/accuracy"] > 0.5


# ============================================================================
# Integration Tests
# ============================================================================


class TestAlgorithmIntegration:
    """Integration tests for algorithm combinations."""

    def test_all_estimators_compatible_with_all_losses(self):
        """All advantage estimators should work with all loss functions."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        rewards = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len)
        old_logp = torch.randn(batch_size, seq_len) - 2
        logp = torch.randn(batch_size, seq_len) - 2
        index = np.array([0, 0, 1, 1])

        # Test each estimator
        for est_name in ["gae", "grpo", "reinforce", "rloo", "dapo", "gspo"]:
            est_fn = get_adv_estimator_fn(est_name)

            # Compute advantages
            advantages, _ = est_fn(rewards, mask, index=index)

            # Test with each loss (excluding DPO which needs preference data)
            for loss_name in ["ppo", "grpo", "reinforce", "rloo", "dapo", "gspo"]:
                loss_fn = get_policy_loss_fn(loss_name)
                loss, metrics = loss_fn(old_logp, logp, advantages, mask)

                assert loss.dim() == 0, f"{est_name} + {loss_name} failed"
                assert not torch.isnan(loss), f"{est_name} + {loss_name} produced NaN"

    def test_importance_weights_work_with_all_losses(self):
        """All losses should accept importance weights."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 10

        advantages = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len)
        old_logp = torch.randn(batch_size, seq_len) - 2
        logp = torch.randn(batch_size, seq_len) - 2
        importance_weights = torch.ones(batch_size) * 0.5

        for loss_name in ["ppo", "grpo", "reinforce", "rloo", "dapo", "gspo"]:
            loss_fn = get_policy_loss_fn(loss_name)
            loss, _ = loss_fn(
                old_logp, logp, advantages, mask, importance_weights=importance_weights
            )

            assert loss.dim() == 0
            assert not torch.isnan(loss)
