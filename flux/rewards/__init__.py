"""Flux rewards module.

This module provides reward computation for RLHF training:
- Base reward function interface
- Rule-based rewards (length, format, keywords)
- Model-based rewards (reward models, LLM-as-judge)
"""

from flux.rewards.base import (
    CompositeReward,
    FunctionReward,
    RewardBatchOutput,
    RewardFunction,
    RewardOutput,
)
from flux.rewards.rule_based import (
    CodeReward,
    FormatReward,
    KeywordReward,
    LengthPenalty,
    LengthReward,
    MathReward,
    RepetitionPenalty,
    StructuredOutputReward,
    create_reward_function,
)
from flux.rewards.model_based import (
    ChatRewardModel,
    LLMJudge,
    ProcessRewardModel,
    RewardModel,
    RewardModelConfig,
)

__all__ = [
    # Base
    "CompositeReward",
    "FunctionReward",
    "RewardBatchOutput",
    "RewardFunction",
    "RewardOutput",
    # Rule-based
    "CodeReward",
    "FormatReward",
    "KeywordReward",
    "LengthPenalty",
    "LengthReward",
    "MathReward",
    "RepetitionPenalty",
    "StructuredOutputReward",
    "create_reward_function",
    # Model-based
    "ChatRewardModel",
    "LLMJudge",
    "ProcessRewardModel",
    "RewardModel",
    "RewardModelConfig",
]
