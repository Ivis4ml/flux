"""
Tests for TransformersBackend training backend.
"""

import pytest
import torch
import tempfile
import os

from flux.training.backends.transformers import TransformersBackend, TransformersConfig
from flux.training.base import GPUBatch, TrainStepResult


class TestTransformersConfig:
    """Tests for TransformersConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TransformersConfig()

        assert config.model_path == "gpt2"
        assert config.torch_dtype == "bfloat16"
        assert config.learning_rate == 1e-6
        assert config.weight_decay == 0.01
        assert config.max_grad_norm == 1.0
        assert config.gradient_accumulation_steps == 1
        assert config.use_flash_attention is True
        assert config.gradient_checkpointing is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TransformersConfig(
            model_path="Qwen/Qwen3-8B",
            learning_rate=5e-7,
            torch_dtype="float16",
            gradient_checkpointing=True,
        )

        assert config.model_path == "Qwen/Qwen3-8B"
        assert config.learning_rate == 5e-7
        assert config.torch_dtype == "float16"
        assert config.gradient_checkpointing is True

    def test_get_torch_dtype(self):
        """Test get_torch_dtype conversion."""
        config = TransformersConfig(torch_dtype="float32")
        assert config.get_torch_dtype() == torch.float32

        config = TransformersConfig(torch_dtype="float16")
        assert config.get_torch_dtype() == torch.float16

        config = TransformersConfig(torch_dtype="bfloat16")
        assert config.get_torch_dtype() == torch.bfloat16

    def test_get_torch_dtype_unknown(self):
        """Test get_torch_dtype with unknown type defaults to bfloat16."""
        config = TransformersConfig(torch_dtype="unknown")
        assert config.get_torch_dtype() == torch.bfloat16

    def test_algorithm_params(self):
        """Test algorithm-related parameters."""
        config = TransformersConfig(
            use_importance_weighting=True,
            importance_weight_clip=3.0,
            entropy_coef=0.02,
            kl_coef=0.1,
        )

        assert config.use_importance_weighting is True
        assert config.importance_weight_clip == 3.0
        assert config.entropy_coef == 0.02
        assert config.kl_coef == 0.1


class TestTransformersBackendCreation:
    """Tests for TransformersBackend creation and initialization."""

    def test_creation(self):
        """Test backend creation."""
        backend = TransformersBackend()

        assert not backend.is_initialized
        assert backend.version == 0

    def test_initialization_with_dict(self):
        """Test initialization with dict config."""
        backend = TransformersBackend()
        config = {
            "model_path": "gpt2",
            "learning_rate": 1e-5,
        }

        backend.initialize(config)

        assert backend.is_initialized
        assert backend._config.learning_rate == 1e-5

    def test_initialization_with_config(self):
        """Test initialization with TransformersConfig."""
        backend = TransformersBackend()
        config = TransformersConfig(
            model_path="gpt2",
            learning_rate=2e-6,
        )

        backend.initialize(config)

        assert backend.is_initialized
        assert backend._config.learning_rate == 2e-6

    def test_device_selection(self):
        """Test device selection."""
        backend = TransformersBackend()
        backend.initialize({"model_path": "gpt2"})

        if torch.cuda.is_available():
            assert backend.device == torch.device("cuda")
        else:
            assert backend.device == torch.device("cpu")


class TestTransformersBackendTraining:
    """Tests for TransformersBackend training functionality."""

    @pytest.fixture
    def backend(self, device):
        """Create initialized backend."""
        backend = TransformersBackend()
        # Use a minimal config for testing
        config = TransformersConfig(
            model_path="gpt2",
            learning_rate=1e-5,
            gradient_accumulation_steps=1,
        )
        backend.initialize(config)
        return backend

    @pytest.fixture
    def sample_batch(self, device):
        """Create sample GPUBatch for testing."""
        batch_size = 2
        seq_len = 32

        return GPUBatch(
            input_ids=torch.randint(0, 1000, (batch_size, seq_len), device=device),
            attention_mask=torch.ones(batch_size, seq_len, device=device),
            behavior_log_probs=torch.randn(batch_size, seq_len, device=device) - 2.0,
            rewards=torch.randn(batch_size, device=device),
            version_gaps=torch.zeros(batch_size, dtype=torch.long, device=device),
            loss_mask=torch.ones(batch_size, seq_len, device=device),
        )

    @pytest.mark.slow
    def test_train_step_basic(self, backend, sample_batch):
        """Test basic training step."""
        # Move batch to backend device
        batch = sample_batch.to(backend.device)

        result = backend.train_step(batch)

        assert isinstance(result, TrainStepResult)
        assert result.loss >= 0.0
        assert result.version == 1
        assert backend.version == 1

    @pytest.mark.slow
    def test_train_step_metrics(self, backend, sample_batch):
        """Test training step produces metrics."""
        batch = sample_batch.to(backend.device)

        result = backend.train_step(batch)

        # Check timing metrics
        assert "forward_time_ms" in result.metrics
        assert "backward_time_ms" in result.metrics

    @pytest.mark.slow
    def test_version_increments(self, backend, sample_batch):
        """Test version increments on each step."""
        batch = sample_batch.to(backend.device)

        for i in range(3):
            result = backend.train_step(batch)
            assert result.version == i + 1
            assert backend.version == i + 1

    @pytest.mark.slow
    def test_get_state_dict(self, backend, sample_batch):
        """Test getting state dict."""
        batch = sample_batch.to(backend.device)
        backend.train_step(batch)

        state_dict = backend.get_state_dict()

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        # All tensors should be on CPU
        for key, tensor in state_dict.items():
            assert tensor.device == torch.device("cpu")

    @pytest.mark.slow
    def test_set_state_dict(self, backend, sample_batch):
        """Test loading state dict."""
        batch = sample_batch.to(backend.device)
        backend.train_step(batch)

        # Get state dict
        state_dict = backend.get_state_dict()

        # Modify and reload
        for key in state_dict:
            state_dict[key] = state_dict[key] + 0.1

        backend.set_state_dict(state_dict)

        # Verify loaded (get_state_dict should return similar values)
        new_state_dict = backend.get_state_dict()
        # Just verify it doesn't raise

    @pytest.mark.slow
    def test_get_info(self, backend, sample_batch):
        """Test get_info method."""
        info = backend.get_info()

        assert info["backend_type"] == "TransformersBackend"
        assert info["is_initialized"] is True
        assert "num_parameters" in info
        assert "trainable_parameters" in info
        assert info["model_path"] == "gpt2"


class TestTransformersBackendCheckpointing:
    """Tests for checkpoint save/load."""

    @pytest.fixture
    def backend(self, device):
        """Create initialized backend."""
        backend = TransformersBackend()
        config = TransformersConfig(model_path="gpt2")
        backend.initialize(config)
        return backend

    @pytest.mark.slow
    def test_save_checkpoint(self, backend, device, tmp_path):
        """Test saving checkpoint."""
        # Do a training step to update state
        batch = GPUBatch(
            input_ids=torch.randint(0, 1000, (2, 32), device=device),
            attention_mask=torch.ones(2, 32, device=device),
            behavior_log_probs=torch.randn(2, 32, device=device) - 2.0,
            rewards=torch.randn(2, device=device),
            version_gaps=torch.zeros(2, dtype=torch.long, device=device),
        )
        batch = batch.to(backend.device)
        backend.train_step(batch)

        checkpoint_path = str(tmp_path / "checkpoint.pt")
        backend.save_checkpoint(checkpoint_path)

        assert os.path.exists(checkpoint_path)

    @pytest.mark.slow
    def test_load_checkpoint(self, backend, device, tmp_path):
        """Test loading checkpoint."""
        # Do a training step
        batch = GPUBatch(
            input_ids=torch.randint(0, 1000, (2, 32), device=device),
            attention_mask=torch.ones(2, 32, device=device),
            behavior_log_probs=torch.randn(2, 32, device=device) - 2.0,
            rewards=torch.randn(2, device=device),
            version_gaps=torch.zeros(2, dtype=torch.long, device=device),
        )
        batch = batch.to(backend.device)
        backend.train_step(batch)
        backend.train_step(batch)

        # Save checkpoint
        checkpoint_path = str(tmp_path / "checkpoint.pt")
        backend.save_checkpoint(checkpoint_path)

        # Create new backend and load
        new_backend = TransformersBackend()
        new_backend.initialize(TransformersConfig(model_path="gpt2"))
        new_backend.load_checkpoint(checkpoint_path)

        assert new_backend.version == 2


class TestTransformersBackendGradientAccumulation:
    """Tests for gradient accumulation."""

    @pytest.mark.slow
    def test_gradient_accumulation(self, device):
        """Test gradient accumulation steps."""
        backend = TransformersBackend()
        config = TransformersConfig(
            model_path="gpt2",
            gradient_accumulation_steps=2,
        )
        backend.initialize(config)

        batch = GPUBatch(
            input_ids=torch.randint(0, 1000, (2, 32), device=device),
            attention_mask=torch.ones(2, 32, device=device),
            behavior_log_probs=torch.randn(2, 32, device=device) - 2.0,
            rewards=torch.randn(2, device=device),
            version_gaps=torch.zeros(2, dtype=torch.long, device=device),
        )
        batch = batch.to(backend.device)

        # First step - accumulation (no optimizer step yet)
        result1 = backend.train_step(batch)

        # Second step - should do optimizer step
        result2 = backend.train_step(batch)

        # With gradient_accumulation_steps=2:
        # - First step: accumulation only, version stays 0
        # - Second step: optimizer steps, version becomes 1
        assert result1.version == 0  # No optimizer step yet
        assert result2.version == 1  # Optimizer stepped


class TestTransformersBackendEdgeCases:
    """Tests for edge cases and error handling."""

    def test_not_initialized_train_step(self, device):
        """Test train_step without initialization raises error."""
        backend = TransformersBackend()

        batch = GPUBatch(
            input_ids=torch.zeros(2, 16, dtype=torch.long, device=device),
            attention_mask=torch.ones(2, 16, device=device),
            behavior_log_probs=torch.zeros(2, 16, device=device),
            rewards=torch.zeros(2, device=device),
            version_gaps=torch.zeros(2, dtype=torch.long, device=device),
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            backend.train_step(batch)

    def test_not_initialized_get_state_dict(self):
        """Test get_state_dict without initialization returns empty."""
        backend = TransformersBackend()

        state_dict = backend.get_state_dict()
        assert state_dict == {}

    @pytest.mark.slow
    def test_with_advantages(self, device):
        """Test training with pre-computed advantages."""
        backend = TransformersBackend()
        config = TransformersConfig(model_path="gpt2")
        backend.initialize(config)

        batch_size = 2
        seq_len = 32

        batch = GPUBatch(
            input_ids=torch.randint(0, 1000, (batch_size, seq_len), device=device),
            attention_mask=torch.ones(batch_size, seq_len, device=device),
            behavior_log_probs=torch.randn(batch_size, seq_len, device=device) - 2.0,
            rewards=torch.randn(batch_size, device=device),
            version_gaps=torch.zeros(batch_size, dtype=torch.long, device=device),
            advantages=torch.randn(batch_size, seq_len, device=device),
        )
        batch = batch.to(backend.device)

        result = backend.train_step(batch)
        assert result.loss >= 0.0

    @pytest.mark.slow
    def test_with_varying_batch_sizes(self, device):
        """Test training with different batch sizes."""
        backend = TransformersBackend()
        config = TransformersConfig(model_path="gpt2")
        backend.initialize(config)

        for batch_size in [1, 2, 4]:
            batch = GPUBatch(
                input_ids=torch.randint(0, 1000, (batch_size, 32), device=device),
                attention_mask=torch.ones(batch_size, 32, device=device),
                behavior_log_probs=torch.randn(batch_size, 32, device=device) - 2.0,
                rewards=torch.randn(batch_size, device=device),
                version_gaps=torch.zeros(batch_size, dtype=torch.long, device=device),
            )
            batch = batch.to(backend.device)

            result = backend.train_step(batch)
            assert result.batch_size == batch_size


class TestTransformersBackendFromConfig:
    """Tests for creating backend from FluxConfig."""

    def test_create_from_flux_config(self):
        """Test creating backend via factory."""
        from flux.core.config import FluxConfig
        from flux.training.base import create_training_backend, TrainingBackendType

        config = FluxConfig(
            model_path="test-model",
            output_dir="./test_outputs",
            training_backend=TrainingBackendType.TRANSFORMERS,
        )

        backend = create_training_backend(config)

        assert isinstance(backend, TransformersBackend)
