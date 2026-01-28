"""
Training backend implementations for Flux.

This package contains concrete implementations of the TrainingBackend ABC:
- TransformersBackend: HuggingFace Transformers-based backend (default)
- MegatronBackend: Megatron-LM with 3D parallelism (planned)
- FSDPBackend: PyTorch FSDP for efficient sharding (planned)
- DeepSpeedBackend: DeepSpeed ZeRO integration (planned)

Usage:
    from flux.training.base import create_training_backend

    backend = create_training_backend(config)
    backend.initialize(config.training)
"""

from flux.training.backends.transformers import TransformersBackend

__all__ = [
    "TransformersBackend",
]
