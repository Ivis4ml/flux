"""
Tests for TrainingBackend ABC, GPUBatch, and TrainStepResult.
"""

import pytest
import torch

from flux.training.base import (
    GPUBatch,
    TrainStepResult,
    TrainingBackendBase,
    TrainingBackendType,
    create_training_backend,
)


class TestGPUBatch:
    """Tests for GPUBatch dataclass."""

    @pytest.fixture
    def sample_tensors(self, device):
        """Create sample tensors for testing."""
        batch_size = 4
        seq_len = 16
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device=device),
            "attention_mask": torch.ones(batch_size, seq_len, device=device),
            "behavior_log_probs": torch.randn(batch_size, seq_len, device=device),
            "rewards": torch.randn(batch_size, device=device),
            "version_gaps": torch.randint(0, 5, (batch_size,), device=device),
        }

    def test_creation(self, sample_tensors, device):
        """Test GPUBatch creation."""
        batch = GPUBatch(**sample_tensors)
        assert batch.batch_size == 4
        assert batch.seq_len == 16
        assert batch.device == device

    def test_frozen(self, sample_tensors):
        """Test that GPUBatch is frozen (immutable)."""
        batch = GPUBatch(**sample_tensors)
        with pytest.raises(AttributeError):
            batch.input_ids = torch.zeros(1, 1)

    def test_num_tokens(self, device):
        """Test num_tokens property."""
        batch_size = 4
        seq_len = 16
        # Create attention mask with some padding
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        attention_mask[:, 10:] = 0  # Last 6 tokens are padding

        batch = GPUBatch(
            input_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
            attention_mask=attention_mask,
            behavior_log_probs=torch.zeros(batch_size, seq_len, device=device),
            rewards=torch.zeros(batch_size, device=device),
            version_gaps=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

        assert batch.num_tokens == 4 * 10  # 4 samples * 10 non-padding tokens

    def test_num_loss_tokens(self, device):
        """Test num_loss_tokens property."""
        batch_size = 4
        seq_len = 16
        loss_mask = torch.zeros(batch_size, seq_len, device=device)
        loss_mask[:, 5:10] = 1  # Only tokens 5-9 contribute to loss

        batch = GPUBatch(
            input_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
            attention_mask=torch.ones(batch_size, seq_len, device=device),
            behavior_log_probs=torch.zeros(batch_size, seq_len, device=device),
            rewards=torch.zeros(batch_size, device=device),
            version_gaps=torch.zeros(batch_size, dtype=torch.long, device=device),
            loss_mask=loss_mask,
        )

        assert batch.num_loss_tokens == 4 * 5  # 4 samples * 5 loss tokens

    def test_mean_version_gap(self, device):
        """Test mean_version_gap property."""
        batch = GPUBatch(
            input_ids=torch.zeros(4, 16, dtype=torch.long, device=device),
            attention_mask=torch.ones(4, 16, device=device),
            behavior_log_probs=torch.zeros(4, 16, device=device),
            rewards=torch.zeros(4, device=device),
            version_gaps=torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device),
        )

        assert batch.mean_version_gap == 2.5

    def test_max_version_gap(self, device):
        """Test max_version_gap property."""
        batch = GPUBatch(
            input_ids=torch.zeros(4, 16, dtype=torch.long, device=device),
            attention_mask=torch.ones(4, 16, device=device),
            behavior_log_probs=torch.zeros(4, 16, device=device),
            rewards=torch.zeros(4, device=device),
            version_gaps=torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device),
        )

        assert batch.max_version_gap == 4

    def test_validate_success(self, sample_tensors):
        """Test validation passes for correct batch."""
        batch = GPUBatch(**sample_tensors)
        batch.validate()  # Should not raise

    def test_validate_shape_mismatch(self, device):
        """Test validation fails for shape mismatch."""
        batch = GPUBatch(
            input_ids=torch.zeros(4, 16, dtype=torch.long, device=device),
            attention_mask=torch.ones(4, 8, device=device),  # Wrong shape
            behavior_log_probs=torch.zeros(4, 16, device=device),
            rewards=torch.zeros(4, device=device),
            version_gaps=torch.zeros(4, dtype=torch.long, device=device),
        )

        with pytest.raises(ValueError, match="attention_mask shape"):
            batch.validate()

    def test_validate_device_mismatch(self):
        """Test validation fails for device mismatch."""
        # Only test if CUDA is available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch = GPUBatch(
            input_ids=torch.zeros(4, 16, dtype=torch.long, device="cuda"),
            attention_mask=torch.ones(4, 16, device="cpu"),  # Wrong device
            behavior_log_probs=torch.zeros(4, 16, device="cuda"),
            rewards=torch.zeros(4, device="cuda"),
            version_gaps=torch.zeros(4, dtype=torch.long, device="cuda"),
        )

        with pytest.raises(ValueError, match="attention_mask on"):
            batch.validate()

    def test_to_device(self, sample_tensors, device):
        """Test moving batch to a device."""
        batch = GPUBatch(**sample_tensors)
        cpu_batch = batch.to("cpu")

        assert cpu_batch.device == torch.device("cpu")
        assert cpu_batch.input_ids.device == torch.device("cpu")
        assert cpu_batch.attention_mask.device == torch.device("cpu")

    def test_from_tensors(self, sample_tensors, device):
        """Test creating GPUBatch from dict."""
        batch = GPUBatch.from_tensors(sample_tensors, device=device)

        assert batch.batch_size == 4
        assert batch.seq_len == 16
        assert batch.device == device

    def test_optional_tensors(self, device):
        """Test GPUBatch with optional tensors."""
        batch_size = 4
        seq_len = 16

        batch = GPUBatch(
            input_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
            attention_mask=torch.ones(batch_size, seq_len, device=device),
            behavior_log_probs=torch.zeros(batch_size, seq_len, device=device),
            rewards=torch.zeros(batch_size, device=device),
            version_gaps=torch.zeros(batch_size, dtype=torch.long, device=device),
            advantages=torch.randn(batch_size, seq_len, device=device),
            returns=torch.randn(batch_size, seq_len, device=device),
            ref_log_probs=torch.randn(batch_size, seq_len, device=device),
        )

        assert batch.advantages is not None
        assert batch.returns is not None
        assert batch.ref_log_probs is not None
        batch.validate()  # Should not raise


class TestTrainStepResult:
    """Tests for TrainStepResult dataclass."""

    def test_creation(self):
        """Test TrainStepResult creation."""
        result = TrainStepResult(
            loss=0.5,
            version=10,
            metrics={"policy_loss": 0.3, "entropy": 0.1},
        )

        assert result.loss == 0.5
        assert result.version == 10
        assert result.metrics["policy_loss"] == 0.3

    def test_to_dict(self):
        """Test conversion to dict."""
        result = TrainStepResult(
            loss=0.5,
            version=10,
            metrics={"custom": 1.0},
            grad_norm=0.8,
            batch_size=32,
            num_tokens=512,
            throughput_tokens_per_sec=10000.0,
            policy_loss=0.3,
            entropy=0.1,
        )

        d = result.to_dict()

        assert d["loss"] == 0.5
        assert d["version"] == 10
        assert d["grad_norm"] == 0.8
        assert d["batch_size"] == 32
        assert d["custom"] == 1.0  # From metrics
        assert d["policy_loss"] == 0.3
        assert d["entropy"] == 0.1

    def test_optional_metrics(self):
        """Test TrainStepResult with optional metrics."""
        result = TrainStepResult(
            loss=0.5,
            version=10,
            metrics={},
            kl_divergence=0.02,
            clip_fraction=0.15,
            approx_kl=0.01,
        )

        d = result.to_dict()
        assert d["kl_divergence"] == 0.02
        assert d["clip_fraction"] == 0.15
        assert d["approx_kl"] == 0.01


class TestTrainingBackendType:
    """Tests for TrainingBackendType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert TrainingBackendType.MEGATRON.value == "megatron"
        assert TrainingBackendType.FSDP.value == "fsdp"
        assert TrainingBackendType.TRANSFORMERS.value == "transformers"
        assert TrainingBackendType.DEEPSPEED.value == "deepspeed"

    def test_from_string(self):
        """Test creating from string."""
        backend_type = TrainingBackendType("transformers")
        assert backend_type == TrainingBackendType.TRANSFORMERS


class DummyBackend(TrainingBackendBase):
    """Dummy backend for testing TrainingBackendBase."""

    def _do_initialize(self, config):
        self._dummy_model = True

    def _do_train_step(self, batch):
        loss = 0.5
        metrics = {"dummy_metric": 1.0}
        optimizer_stepped = True  # Dummy always "steps"
        return loss, metrics, optimizer_stepped

    def get_state_dict(self):
        return {"dummy_weight": torch.tensor([1.0])}

    def set_state_dict(self, state_dict):
        pass


class TestTrainingBackendBase:
    """Tests for TrainingBackendBase ABC."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = DummyBackend()
        assert not backend.is_initialized
        assert backend.version == 0
        assert backend.device == torch.device("cpu")

        backend.initialize({})
        assert backend.is_initialized

    def test_double_initialization(self):
        """Test that double initialization is idempotent."""
        backend = DummyBackend()
        backend.initialize({})
        backend.initialize({})  # Should not raise
        assert backend.is_initialized

    def test_train_step_not_initialized(self, device):
        """Test train_step fails if not initialized."""
        backend = DummyBackend()
        batch = GPUBatch(
            input_ids=torch.zeros(4, 16, dtype=torch.long, device=device),
            attention_mask=torch.ones(4, 16, device=device),
            behavior_log_probs=torch.zeros(4, 16, device=device),
            rewards=torch.zeros(4, device=device),
            version_gaps=torch.zeros(4, dtype=torch.long, device=device),
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            backend.train_step(batch)

    def test_train_step_wrong_device(self):
        """Test train_step fails if batch on wrong device."""
        # Only test if CUDA is available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        backend = DummyBackend()
        backend.initialize({})
        # Backend defaults to CPU

        batch = GPUBatch(
            input_ids=torch.zeros(4, 16, dtype=torch.long, device="cuda"),
            attention_mask=torch.ones(4, 16, device="cuda"),
            behavior_log_probs=torch.zeros(4, 16, device="cuda"),
            rewards=torch.zeros(4, device="cuda"),
            version_gaps=torch.zeros(4, dtype=torch.long, device="cuda"),
        )

        with pytest.raises(ValueError, match="expected cpu"):
            backend.train_step(batch)

    def test_train_step_success(self, device):
        """Test successful train_step."""
        backend = DummyBackend()
        backend._device = device
        backend.initialize({})

        batch = GPUBatch(
            input_ids=torch.zeros(4, 16, dtype=torch.long, device=device),
            attention_mask=torch.ones(4, 16, device=device),
            behavior_log_probs=torch.zeros(4, 16, device=device),
            rewards=torch.zeros(4, device=device),
            version_gaps=torch.zeros(4, dtype=torch.long, device=device),
        )

        result = backend.train_step(batch)

        assert isinstance(result, TrainStepResult)
        assert result.loss == 0.5
        assert result.version == 1
        assert backend.version == 1
        assert "dummy_metric" in result.metrics

    def test_version_increments(self, device):
        """Test that version increments on each train_step."""
        backend = DummyBackend()
        backend._device = device
        backend.initialize({})

        batch = GPUBatch(
            input_ids=torch.zeros(4, 16, dtype=torch.long, device=device),
            attention_mask=torch.ones(4, 16, device=device),
            behavior_log_probs=torch.zeros(4, 16, device=device),
            rewards=torch.zeros(4, device=device),
            version_gaps=torch.zeros(4, dtype=torch.long, device=device),
        )

        for i in range(5):
            result = backend.train_step(batch)
            assert result.version == i + 1
            assert backend.version == i + 1

    def test_get_info(self):
        """Test get_info method."""
        backend = DummyBackend()
        backend.initialize({})

        info = backend.get_info()
        assert info["backend_type"] == "DummyBackend"
        assert info["version"] == 0
        assert info["is_initialized"] is True


class TestCreateTrainingBackend:
    """Tests for create_training_backend factory."""

    def test_create_transformers_backend(self):
        """Test creating TransformersBackend."""
        from flux.core.config import FluxConfig

        config = FluxConfig(
            model_path="test-model",
            output_dir="./test_outputs",
            training_backend=TrainingBackendType.TRANSFORMERS,
        )
        backend = create_training_backend(config)

        from flux.training.backends import TransformersBackend
        assert isinstance(backend, TransformersBackend)

    def test_create_megatron_backend_not_implemented(self):
        """Test that MegatronBackend raises NotImplementedError."""
        from flux.core.config import FluxConfig

        config = FluxConfig(
            model_path="test-model",
            output_dir="./test_outputs",
            training_backend=TrainingBackendType.MEGATRON,
        )

        with pytest.raises(NotImplementedError):
            create_training_backend(config)

    def test_create_fsdp_backend_not_implemented(self):
        """Test that FSDPBackend raises NotImplementedError."""
        from flux.core.config import FluxConfig

        config = FluxConfig(
            model_path="test-model",
            output_dir="./test_outputs",
            training_backend=TrainingBackendType.FSDP,
        )

        with pytest.raises(NotImplementedError):
            create_training_backend(config)

    def test_create_deepspeed_backend_not_implemented(self):
        """Test that DeepSpeedBackend raises NotImplementedError."""
        from flux.core.config import FluxConfig

        config = FluxConfig(
            model_path="test-model",
            output_dir="./test_outputs",
            training_backend=TrainingBackendType.DEEPSPEED,
        )

        with pytest.raises(NotImplementedError):
            create_training_backend(config)

    def test_create_from_string(self):
        """Test creating backend from string type."""
        from flux.core.config import FluxConfig

        config = FluxConfig(
            model_path="test-model",
            output_dir="./test_outputs",
            training_backend="transformers",
        )
        backend = create_training_backend(config)

        from flux.training.backends import TransformersBackend
        assert isinstance(backend, TransformersBackend)
