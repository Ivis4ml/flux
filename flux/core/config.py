"""
Configuration dataclasses for Flux.

All configurations use Pydantic for validation and serialization support.
Configurations are immutable by default to prevent accidental modifications.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AlgorithmType(str, Enum):
    """Supported RL algorithms."""

    PPO = "ppo"
    GRPO = "grpo"
    DPO = "dpo"
    DAPO = "dapo"
    GSPO = "gspo"
    RLOO = "rloo"
    REINFORCE = "reinforce"
    CUSTOM = "custom"


class RewardType(str, Enum):
    """Supported reward computation types."""

    RULE_BASED = "rule_based"
    MODEL_BASED = "model_based"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class WeightSyncMethod(str, Enum):
    """Weight synchronization methods."""

    FULL = "full"  # Send complete state_dict
    DELTA = "delta"  # Send only changed parameters
    PER_TENSOR = "per_tensor"  # Stream individual tensors


class BaseConfig(BaseModel):
    """Base configuration class with common settings."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


class AdaptiveAsyncConfig(BaseConfig):
    """Configuration for the adaptive async controller.

    The controller dynamically adjusts sync/async ratio based on measured staleness,
    using a PID controller for smooth adaptation.

    The async_ratio (float in [min_async_ratio, max_async_ratio]) controls the
    maximum proportion of off-policy data allowed in training batches:
    - 0.1 (more sync): Sync barrier triggered frequently; training waits for fresh rollouts
    - 0.5 (balanced): Mixed fresh/stale data; moderate overlap between rollout and training
    - 0.9 (more async): Training proceeds with older data; rollouts run independently

    Attributes:
        target_staleness: Target staleness level (0 = fully sync, 1 = very stale).
        tolerance: Acceptable deviation from target staleness before triggering sync.
        min_async_ratio: Minimum async ratio (never fully sync, wastes GPU).
        max_async_ratio: Maximum async ratio (never fully async, unstable).
        kp: Proportional gain for PID controller.
        ki: Integral gain for PID controller.
        kd: Derivative gain for PID controller.
        kl_normalizer: Normalization factor for KL divergence in staleness computation.
        iw_normalizer: Normalization factor for importance weight variance.
        max_version_gap: Maximum expected version gap for normalization.
        kl_weight: Weight for KL contribution in combined staleness (default 0.4).
        iw_weight: Weight for IW variance contribution in combined staleness (default 0.3).
        version_weight: Weight for version gap contribution in combined staleness (default 0.3).
        max_steps_without_sync: Force sync after this many steps.
        ema_alpha: Smoothing factor for exponential moving average of staleness.
    """

    target_staleness: float = Field(default=0.15, ge=0.0, le=1.0)
    tolerance: float = Field(default=0.05, ge=0.0, le=0.5)
    min_async_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    max_async_ratio: float = Field(default=0.9, ge=0.0, le=1.0)

    # PID controller gains
    kp: float = Field(default=0.1, ge=0.0)
    ki: float = Field(default=0.01, ge=0.0)
    kd: float = Field(default=0.05, ge=0.0)

    # Staleness computation normalization
    kl_normalizer: float = Field(default=0.1, gt=0.0)
    iw_normalizer: float = Field(default=2.0, gt=0.0)
    max_version_gap: int = Field(default=5, ge=1)

    # Staleness combination weights (must sum to 1.0)
    kl_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    iw_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    version_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Sync control
    max_steps_without_sync: int = Field(default=50, ge=1)
    ema_alpha: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ratios(self) -> "AdaptiveAsyncConfig":
        """Ensure min_async_ratio <= max_async_ratio and weights sum to 1.0."""
        if self.min_async_ratio > self.max_async_ratio:
            raise ValueError("min_async_ratio must be <= max_async_ratio")
        weight_sum = self.kl_weight + self.iw_weight + self.version_weight
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Staleness weights must sum to 1.0, got {weight_sum}")
        return self


class RolloutConfig(BaseConfig):
    """Configuration for rollout generation.

    APRIL Strategy Parameters:
    - oversample_ratio: Generate more prompts than needed to buffer against long-tail
    - batch_timeout: After timeout, abort remaining and yield completed
    - partial_reuse_threshold: If this fraction of max_tokens generated, save prefix

    Abort Rule: A generation is aborted when:
    - elapsed_time > batch_timeout, AND
    - completed_count >= target_count

    Attributes:
        max_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        oversample_ratio: Ratio of extra prompts to sample for APRIL strategy (default 1.5).
        min_yield_size: Minimum batch size before yielding trajectories.
        batch_timeout: Timeout in seconds for batch completion (default 30s).
        use_length_prediction: Whether to use length prediction for scheduling.
        partial_reuse_threshold: Fraction of max_tokens to save as partial (default 0.5).
        partial_buffer_max_factor: Partial buffer max size as factor of batch_size (default 2).
        oversample_pool_max_factor: Oversample pool max size as factor of prompt queue (default 1.5).
    """

    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)

    # APRIL strategy
    oversample_ratio: float = Field(default=1.5, ge=1.0, le=3.0)
    min_yield_size: int = Field(default=8, ge=1)
    batch_timeout: float = Field(default=30.0, gt=0.0)

    # Length prediction
    use_length_prediction: bool = Field(default=True)

    # Partial trajectory reuse
    partial_reuse_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    partial_buffer_max_factor: float = Field(default=2.0, ge=1.0)
    oversample_pool_max_factor: float = Field(default=1.5, ge=1.0)

    def get_min_partial_tokens(self) -> int:
        """Get minimum tokens for partial trajectory reuse."""
        return int(self.max_tokens * self.partial_reuse_threshold)


class BatchComposerConfig(BaseConfig):
    """Configuration for smart batch composition.

    Length Bucketing:
    - Buckets: [0, 512), [512, 1024), [1024, 2048), [2048, ∞)
    - Trajectories grouped by bucket; batches drawn from single bucket

    Staleness Balancing:
    - Stratified sampling by version_gap buckets
    - Sample proportionally from each stratum to balance batch

    Curriculum Ordering:
    - Difficulty signal: length + reward based heuristic (configurable)
    - Early training: randomness = 1.0 (fully shuffled)
    - Late training: randomness → 0 (strict easy→hard ordering)

    Attributes:
        use_length_bucketing: Whether to group by length for efficient padding.
        use_staleness_balancing: Whether to balance staleness distribution.
        use_curriculum: Whether to use curriculum learning ordering.
        length_bucket_boundaries: Token count boundaries for length buckets.
        staleness_strata: Number of staleness strata for stratified sampling.
        curriculum_randomness_decay: Decay rate for curriculum randomness.
    """

    use_length_bucketing: bool = Field(default=True)
    use_staleness_balancing: bool = Field(default=True)
    use_curriculum: bool = Field(default=True)

    # Creates buckets: [0, 512), [512, 1024), [1024, 2048), [2048, ∞)
    length_bucket_boundaries: tuple[int, ...] = Field(default=(512, 1024, 2048))

    # Strata: version_gap 0, 1, 2, 3+ (4 strata total)
    staleness_strata: int = Field(default=4, ge=1)

    # randomness = 1 / (1 + decay_rate × curriculum_step)
    curriculum_randomness_decay: float = Field(default=1.0, ge=0.0)


class ImportanceCorrectionConfig(BaseConfig):
    """Configuration for unified importance weight correction.

    Importance correction adjusts for distribution shift between behavior policy
    (which generated the data) and current policy (being trained). This enables
    off-policy learning with staleness-aware weighting.

    Formula:
        log_ratio = current_logprobs - behavior_logprobs
        mean_log_ratio = mean(log_ratio * mask)
        base_weight = exp(clip(mean_log_ratio, log_clip_min, log_clip_max))
        staleness_weight = staleness_decay ** version_gap
        importance_weight = clip(base_weight * staleness_weight, weight_min, weight_max)
        importance_weight = importance_weight * (batch_size / sum(importance_weight))

    Attributes:
        enabled: Whether to apply importance correction.
        staleness_decay: Decay factor per version gap (default 0.99).
        weight_min: Minimum importance weight (default 0.2).
        weight_max: Maximum importance weight (default 5.0).
        log_clip_min: Minimum log ratio before exp (default -20).
        log_clip_max: Maximum log ratio before exp (default 20).
        normalize: Whether to normalize weights to preserve gradient scale.
    """

    enabled: bool = Field(default=True)
    staleness_decay: float = Field(default=0.99, ge=0.0, le=1.0)
    weight_min: float = Field(default=0.2, ge=0.0)
    weight_max: float = Field(default=5.0, gt=0.0)
    log_clip_min: float = Field(default=-20.0)
    log_clip_max: float = Field(default=20.0)
    normalize: bool = Field(default=True)


class WeightSyncConfig(BaseConfig):
    """Configuration for weight synchronization.

    Attributes:
        method: Sync method - "full" (complete state_dict), "delta" (changed params only),
                or "per_tensor" (stream individual tensors).
        sync_interval: Steps between weight syncs (0 = sync every step).
        use_cuda_ipc: Whether to use CUDA IPC for same-node transfers.
        sparsity_threshold: Threshold for sparse delta encoding (delta mode).
        sparsity_target: Target sparsity ratio for using sparse encoding.
        snapshot_interval: Steps between weight snapshots for delta computation.
        max_snapshots: Maximum number of snapshots to keep.
        quantize: Whether to quantize weights before transfer (reduces bandwidth).
        quantize_bits: Quantization bit width (8 or 16).
    """

    method: WeightSyncMethod = Field(default=WeightSyncMethod.DELTA)
    sync_interval: int = Field(default=1, ge=0)
    use_cuda_ipc: bool = Field(default=True)
    sparsity_threshold: float = Field(default=1e-6, ge=0.0)
    sparsity_target: float = Field(default=0.3, ge=0.0, le=1.0)
    snapshot_interval: int = Field(default=10, ge=1)
    max_snapshots: int = Field(default=5, ge=1)
    quantize: bool = Field(default=False)
    quantize_bits: Literal[8, 16] = Field(default=16)


class AlgorithmConfig(BaseConfig):
    """Configuration for RL algorithm.

    Attributes:
        name: Algorithm name - any string that maps to a registered algorithm.
              Built-in: "ppo", "grpo", "dpo", "reinforce", "dapo", "gspo", "rloo".
              Custom algorithms can be registered via @register_policy_loss decorator.
        clip_range: PPO clip range for policy ratio.
        clip_range_vf: PPO clip range for value function.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        gamma: Discount factor for returns.
        gae_lambda: GAE lambda for advantage estimation.
        normalize_advantages: Whether to normalize advantages.
        kl_coef: KL divergence coefficient.
        kl_target: Target KL divergence for adaptive KL.
        adv_estimator: Override for advantage estimator registry key.
        policy_loss: Override for policy loss registry key.
    """

    name: str = Field(default="grpo")
    clip_range: float = Field(default=0.2, ge=0.0, le=1.0)
    clip_range_vf: float | None = Field(default=None, ge=0.0, le=1.0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    value_coef: float = Field(default=0.5, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    gamma: float = Field(default=1.0, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    normalize_advantages: bool = Field(default=True)
    kl_coef: float = Field(default=0.0, ge=0.0)
    kl_target: float | None = Field(default=None, ge=0.0)
    adv_estimator: str | None = Field(default=None)
    policy_loss: str | None = Field(default=None)


class RewardConfig(BaseConfig):
    """Configuration for reward computation.

    Attributes:
        reward_type: Type of reward computation.
        reward_model_path: Path to reward model (for model-based).
        rule_functions: List of rule function names (for rule-based).
        reward_scale: Scale factor for rewards.
        reward_clip: Clip rewards to [-clip, clip].
        baseline_type: Type of baseline for variance reduction.
        kl_penalty_coef: Coefficient for KL penalty in reward.
    """

    reward_type: RewardType = Field(default=RewardType.RULE_BASED)
    reward_model_path: str | None = Field(default=None)
    rule_functions: list[str] = Field(default_factory=list)
    reward_scale: float = Field(default=1.0, gt=0.0)
    reward_clip: float | None = Field(default=10.0, gt=0.0)
    baseline_type: Literal["none", "mean", "per_token"] = Field(default="mean")
    kl_penalty_coef: float = Field(default=0.0, ge=0.0)


class SGLangConfig(BaseConfig):
    """Configuration for SGLang inference servers.

    Attributes:
        base_url: Base URL for SGLang server.
        num_servers: Number of SGLang server instances.
        server_urls: Explicit list of server URLs (overrides base_url/num_servers).
        timeout: Request timeout in seconds.
        max_retries: Maximum retries for failed requests.
        use_streaming: Whether to use streaming responses.
    """

    base_url: str = Field(default="http://localhost:8000")
    num_servers: int = Field(default=1, ge=1)
    server_urls: list[str] | None = Field(default=None)
    timeout: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    use_streaming: bool = Field(default=True)

    @field_validator("server_urls")
    @classmethod
    def validate_server_urls(cls, v: list[str] | None) -> list[str] | None:
        """Validate server URLs if provided."""
        if v is not None:
            for url in v:
                if not url.startswith(("http://", "https://")):
                    raise ValueError(f"Invalid URL: {url}")
        return v


class MegatronConfig(BaseConfig):
    """Configuration for Megatron training backend.

    Attributes:
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        dp_size: Data parallel size.
        sequence_parallel: Whether to use sequence parallelism.
        activation_checkpointing: Whether to use gradient checkpointing.
        fp16: Whether to use FP16 mixed precision.
        bf16: Whether to use BF16 mixed precision.
        use_flash_attention: Whether to use Flash Attention.
        accumulate_allreduce_grads_in_fp32: Accumulate gradients in FP32.
    """

    tp_size: int = Field(default=1, ge=1)
    pp_size: int = Field(default=1, ge=1)
    dp_size: int = Field(default=1, ge=1)
    sequence_parallel: bool = Field(default=False)
    activation_checkpointing: bool = Field(default=True)
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=True)
    use_flash_attention: bool = Field(default=True)
    accumulate_allreduce_grads_in_fp32: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_precision(self) -> "MegatronConfig":
        """Ensure only one precision mode is enabled."""
        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16")
        return self


class FluxConfig(BaseConfig):
    """Main configuration for Flux trainer.

    This is the top-level configuration that combines all sub-configurations.

    Attributes:
        model_path: Path to the model or model name.
        model_type: Type of model architecture.
        output_dir: Directory for outputs (checkpoints, logs).
        learning_rate: Learning rate for optimizer.
        batch_size: Number of trajectories per training batch.
        gradient_accumulation_steps: Steps to accumulate gradients.
        num_steps: Total number of training steps.
        warmup_steps: Number of warmup steps for learning rate.
        weight_decay: Weight decay for optimizer.
        adam_beta1: Adam beta1 parameter.
        adam_beta2: Adam beta2 parameter.
        adam_epsilon: Adam epsilon parameter.
        log_interval: Steps between logging.
        checkpoint_interval: Steps between checkpoints.
        eval_interval: Steps between evaluations.
        seed: Random seed for reproducibility.
        wandb_project: W&B project name (None to disable).
        wandb_run_name: W&B run name.
    """

    # Model
    model_path: str
    model_type: str = Field(default="llama")
    output_dir: str = Field(default="./outputs")

    # Training
    learning_rate: float = Field(default=1e-6, gt=0.0)
    batch_size: int = Field(default=32, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_steps: int = Field(default=10000, ge=1)
    warmup_steps: int = Field(default=100, ge=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, le=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, le=1.0)
    adam_epsilon: float = Field(default=1e-8, gt=0.0)

    # Logging
    log_interval: int = Field(default=10, ge=1)
    checkpoint_interval: int = Field(default=1000, ge=1)
    eval_interval: int = Field(default=500, ge=1)

    # Reproducibility
    seed: int = Field(default=42)

    # W&B
    wandb_project: str | None = Field(default=None)
    wandb_run_name: str | None = Field(default=None)

    # Sub-configurations
    adaptive_async: AdaptiveAsyncConfig = Field(default_factory=AdaptiveAsyncConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    batch_composer: BatchComposerConfig = Field(default_factory=BatchComposerConfig)
    weight_sync: WeightSyncConfig = Field(default_factory=WeightSyncConfig)
    importance_correction: ImportanceCorrectionConfig = Field(
        default_factory=ImportanceCorrectionConfig
    )
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    sglang: SGLangConfig = Field(default_factory=SGLangConfig)
    megatron: MegatronConfig = Field(default_factory=MegatronConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FluxConfig":
        """Load configuration from YAML file with nested config support."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    def get_total_gpus(self) -> int:
        """Calculate total number of GPUs needed."""
        return self.megatron.tp_size * self.megatron.pp_size * self.megatron.dp_size
