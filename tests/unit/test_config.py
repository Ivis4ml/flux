"""
Unit tests for Flux configuration module.
"""

import pytest
from pydantic import ValidationError

from flux.core.config import (
    AdaptiveAsyncConfig,
    AlgorithmConfig,
    AlgorithmType,
    BatchComposerConfig,
    FluxConfig,
    MegatronConfig,
    RewardConfig,
    RolloutConfig,
    SGLangConfig,
    WeightSyncConfig,
)


class TestFluxConfig:
    """Tests for FluxConfig."""

    def test_basic_creation(self) -> None:
        """Test basic config creation with required fields."""
        config = FluxConfig(model_path="test-model")
        assert config.model_path == "test-model"
        assert config.learning_rate == 1e-6  # default
        assert config.batch_size == 32  # default

    def test_nested_configs(self) -> None:
        """Test that nested configs are properly initialized."""
        config = FluxConfig(model_path="test-model")
        assert isinstance(config.adaptive_async, AdaptiveAsyncConfig)
        assert isinstance(config.rollout, RolloutConfig)
        assert isinstance(config.algorithm, AlgorithmConfig)

    def test_custom_nested_config(self) -> None:
        """Test custom nested config values."""
        config = FluxConfig(
            model_path="test-model",
            adaptive_async=AdaptiveAsyncConfig(target_staleness=0.2),
            rollout=RolloutConfig(max_tokens=4096),
        )
        assert config.adaptive_async.target_staleness == 0.2
        assert config.rollout.max_tokens == 4096

    def test_effective_batch_size(self) -> None:
        """Test effective batch size calculation."""
        config = FluxConfig(
            model_path="test-model",
            batch_size=8,
            gradient_accumulation_steps=4,
        )
        assert config.get_effective_batch_size() == 32

    def test_total_gpus(self) -> None:
        """Test total GPU calculation."""
        config = FluxConfig(
            model_path="test-model",
            megatron=MegatronConfig(tp_size=2, pp_size=2, dp_size=4),
        )
        assert config.get_total_gpus() == 16

    def test_to_dict(self) -> None:
        """Test config serialization to dict."""
        config = FluxConfig(model_path="test-model")
        d = config.to_dict()
        assert d["model_path"] == "test-model"
        assert "adaptive_async" in d

    def test_immutable(self) -> None:
        """Test that config is immutable."""
        config = FluxConfig(model_path="test-model")
        with pytest.raises(ValidationError):
            config.model_path = "new-model"


class TestAdaptiveAsyncConfig:
    """Tests for AdaptiveAsyncConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AdaptiveAsyncConfig()
        assert config.target_staleness == 0.15
        assert config.min_async_ratio == 0.1
        assert config.max_async_ratio == 0.9

    def test_validation_staleness_range(self) -> None:
        """Test staleness must be in [0, 1]."""
        with pytest.raises(ValidationError):
            AdaptiveAsyncConfig(target_staleness=1.5)

        with pytest.raises(ValidationError):
            AdaptiveAsyncConfig(target_staleness=-0.1)

    def test_validation_ratio_order(self) -> None:
        """Test min_async_ratio must be <= max_async_ratio."""
        with pytest.raises(ValidationError):
            AdaptiveAsyncConfig(min_async_ratio=0.9, max_async_ratio=0.1)

    def test_pid_gains(self) -> None:
        """Test PID controller gains."""
        config = AdaptiveAsyncConfig(kp=0.2, ki=0.02, kd=0.1)
        assert config.kp == 0.2
        assert config.ki == 0.02
        assert config.kd == 0.1


class TestRolloutConfig:
    """Tests for RolloutConfig."""

    def test_default_values(self) -> None:
        """Test default APRIL settings."""
        config = RolloutConfig()
        assert config.oversample_ratio == 1.5
        assert config.use_length_prediction is True

    def test_validation_temperature(self) -> None:
        """Test temperature must be in valid range."""
        config = RolloutConfig(temperature=0.7)
        assert config.temperature == 0.7

        with pytest.raises(ValidationError):
            RolloutConfig(temperature=-0.1)


class TestAlgorithmConfig:
    """Tests for AlgorithmConfig."""

    def test_default_algorithm(self) -> None:
        """Test default algorithm is GRPO."""
        config = AlgorithmConfig()
        assert config.name == AlgorithmType.GRPO

    def test_ppo_config(self) -> None:
        """Test PPO-specific settings."""
        config = AlgorithmConfig(
            name=AlgorithmType.PPO,
            clip_range=0.2,
            entropy_coef=0.01,
        )
        assert config.name == AlgorithmType.PPO
        assert config.clip_range == 0.2


class TestMegatronConfig:
    """Tests for MegatronConfig."""

    def test_parallelism_settings(self) -> None:
        """Test parallelism configuration."""
        config = MegatronConfig(tp_size=4, pp_size=2, dp_size=8)
        assert config.tp_size == 4
        assert config.pp_size == 2
        assert config.dp_size == 8

    def test_precision_validation(self) -> None:
        """Test that fp16 and bf16 cannot both be enabled."""
        with pytest.raises(ValidationError):
            MegatronConfig(fp16=True, bf16=True)

    def test_default_precision(self) -> None:
        """Test default precision is bf16."""
        config = MegatronConfig()
        assert config.bf16 is True
        assert config.fp16 is False


class TestSGLangConfig:
    """Tests for SGLangConfig."""

    def test_default_url(self) -> None:
        """Test default SGLang URL."""
        config = SGLangConfig()
        assert config.base_url == "http://localhost:8000"

    def test_custom_urls(self) -> None:
        """Test custom server URLs."""
        config = SGLangConfig(
            server_urls=["http://host1:8000", "http://host2:8000"]
        )
        assert len(config.server_urls) == 2

    def test_url_validation(self) -> None:
        """Test URL validation."""
        with pytest.raises(ValidationError):
            SGLangConfig(server_urls=["invalid-url"])
