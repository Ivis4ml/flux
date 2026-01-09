"""
Model-based reward functions for Flux.

This module provides neural network-based reward functions, including
reward models trained on preference data and LLM-as-judge approaches.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from flux.core.trajectory import Trajectory
from flux.rewards.base import RewardBatchOutput, RewardFunction, RewardOutput


logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration for reward models.

    Attributes:
        model_path: Path to reward model or HuggingFace model ID.
        device: Device to run model on.
        dtype: Data type for model.
        max_length: Maximum sequence length.
        batch_size: Batch size for inference.
    """

    model_path: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_length: int = 2048
    batch_size: int = 8


class RewardModel(RewardFunction):
    """Base class for neural network reward models.

    Subclasses should implement _forward() to compute reward from model output.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        name: str = "reward_model",
    ):
        """Initialize reward model.

        Args:
            config: Reward model configuration.
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.config = config
        self._model: nn.Module | None = None
        self._tokenizer: Any = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded."""
        return self._is_loaded

    def load(self) -> None:
        """Load the reward model.

        Override this method for custom model loading.
        """
        if self._is_loaded:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading reward model from {self.config.model_path}")

            # Determine dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
            )

            # Load model
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

            # Move to device
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                self._model = self._model.to(self.config.device)

            self._model.eval()
            self._is_loaded = True

            logger.info(f"Reward model loaded: {sum(p.numel() for p in self._model.parameters()):,} parameters")

        except ImportError:
            logger.warning("transformers not installed, using dummy reward model")
            self._is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load reward model: {e}")
            raise

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute reward for a single trajectory.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with model reward.
        """
        if not self._is_loaded:
            self.load()

        if self._model is None or self._tokenizer is None:
            # Fallback for when transformers is not installed
            return RewardOutput(reward=0.0, metadata={"error": "model_not_loaded"})

        # Prepare input
        text = self._format_input(trajectory)

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        )

        # Move to device
        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Extract reward
        reward = self._extract_reward(outputs)

        return RewardOutput(
            reward=reward,
            metadata={"model": self.config.model_path},
        )

    def compute_rewards(self, trajectories: list[Trajectory]) -> RewardBatchOutput:
        """Compute rewards for a batch of trajectories.

        Args:
            trajectories: List of trajectories.

        Returns:
            RewardBatchOutput with all rewards.
        """
        if not self._is_loaded:
            self.load()

        if self._model is None or self._tokenizer is None:
            return RewardBatchOutput(
                rewards=[0.0] * len(trajectories),
                metadata={"error": "model_not_loaded"},
            )

        # Prepare inputs
        texts = [self._format_input(traj) for traj in trajectories]

        # Process in batches
        all_rewards = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
            )

            # Move to device
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Extract rewards
            batch_rewards = self._extract_batch_rewards(outputs)
            all_rewards.extend(batch_rewards)

        return RewardBatchOutput(rewards=all_rewards)

    def _format_input(self, trajectory: Trajectory) -> str:
        """Format trajectory as input for reward model.

        Args:
            trajectory: Trajectory to format.

        Returns:
            Formatted text string.
        """
        # Default format: prompt + response
        return f"{trajectory.prompt}\n\n{trajectory.response}"

    def _extract_reward(self, outputs: Any) -> float:
        """Extract scalar reward from model outputs.

        Args:
            outputs: Model outputs.

        Returns:
            Scalar reward value.
        """
        if hasattr(outputs, "logits"):
            # Classification model
            logits = outputs.logits
            if logits.shape[-1] == 1:
                return logits[0, 0].item()
            else:
                # Use softmax for multi-class
                probs = torch.softmax(logits, dim=-1)
                return probs[0, -1].item()  # Assume last class is "good"
        return 0.0

    def _extract_batch_rewards(self, outputs: Any) -> list[float]:
        """Extract rewards from batched model outputs.

        Args:
            outputs: Batched model outputs.

        Returns:
            List of scalar rewards.
        """
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            if logits.shape[-1] == 1:
                return logits[:, 0].tolist()
            else:
                probs = torch.softmax(logits, dim=-1)
                return probs[:, -1].tolist()
        return [0.0] * len(outputs)


class ChatRewardModel(RewardModel):
    """Reward model for chat-formatted inputs.

    Uses chat template for formatting prompts and responses.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        system_prompt: str | None = None,
        name: str = "chat_reward_model",
    ):
        """Initialize chat reward model.

        Args:
            config: Reward model configuration.
            system_prompt: Optional system prompt.
            name: Name for this reward function.
        """
        super().__init__(config=config, name=name)
        self.system_prompt = system_prompt

    def _format_input(self, trajectory: Trajectory) -> str:
        """Format trajectory using chat template.

        Args:
            trajectory: Trajectory to format.

        Returns:
            Chat-formatted text.
        """
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": trajectory.prompt})
        messages.append({"role": "assistant", "content": trajectory.response})

        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass

        # Fallback format
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)


class LLMJudge(RewardFunction):
    """LLM-as-a-judge reward function.

    Uses a language model to score response quality through prompting.
    """

    def __init__(
        self,
        judge_url: str = "http://localhost:8000",
        judge_prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        name: str = "llm_judge",
    ):
        """Initialize LLM judge.

        Args:
            judge_url: URL of the LLM server (SGLang/vLLM compatible).
            judge_prompt: Custom judge prompt template.
            max_tokens: Maximum tokens for judge response.
            temperature: Temperature for judge sampling.
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.judge_url = judge_url
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.judge_prompt = judge_prompt or self._default_judge_prompt()

    def _default_judge_prompt(self) -> str:
        """Get default judge prompt template."""
        return """You are an expert judge. Rate the quality of the assistant's response on a scale of 1-10.

<question>
{prompt}
</question>

<response>
{response}
</response>

Provide your rating as a single number between 1 and 10. Only output the number, nothing else.

Rating:"""

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute reward using LLM judge.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with judge reward.
        """
        # Format judge prompt
        judge_input = self.judge_prompt.format(
            prompt=trajectory.prompt,
            response=trajectory.response,
        )

        # Call judge (sync version)
        try:
            import httpx

            response = httpx.post(
                f"{self.judge_url}/generate",
                json={
                    "prompt": judge_input,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
            judge_text = result.get("text", "")

            # Parse rating
            rating = self._parse_rating(judge_text)
            # Normalize to 0-1
            reward = (rating - 1) / 9.0

            return RewardOutput(
                reward=reward,
                metadata={
                    "raw_rating": rating,
                    "judge_response": judge_text,
                },
            )

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return RewardOutput(reward=0.0, metadata={"error": str(e)})

    async def compute_reward_async(self, trajectory: Trajectory) -> RewardOutput:
        """Async version of compute_reward.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with judge reward.
        """
        judge_input = self.judge_prompt.format(
            prompt=trajectory.prompt,
            response=trajectory.response,
        )

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.judge_url}/generate",
                    json={
                        "prompt": judge_input,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()
                judge_text = result.get("text", "")

            rating = self._parse_rating(judge_text)
            reward = (rating - 1) / 9.0

            return RewardOutput(
                reward=reward,
                metadata={
                    "raw_rating": rating,
                    "judge_response": judge_text,
                },
            )

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return RewardOutput(reward=0.0, metadata={"error": str(e)})

    def _parse_rating(self, text: str) -> float:
        """Parse rating from judge response.

        Args:
            text: Judge response text.

        Returns:
            Parsed rating (1-10).
        """
        import re

        # Try to find a number
        numbers = re.findall(r'\b([1-9]|10)\b', text)
        if numbers:
            return float(numbers[0])

        # Default to middle rating
        return 5.0


class ProcessRewardModel(RewardModel):
    """Process Reward Model (PRM) for step-by-step scoring.

    Scores each step in a reasoning chain, useful for math and coding tasks.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        step_separator: str = "\n\n",
        aggregate: str = "min",
        name: str = "prm",
    ):
        """Initialize process reward model.

        Args:
            config: Reward model configuration.
            step_separator: Separator between reasoning steps.
            aggregate: How to aggregate step rewards ("min", "mean", "last").
            name: Name for this reward function.
        """
        super().__init__(config=config, name=name)
        self.step_separator = step_separator
        self.aggregate = aggregate

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute reward for each step and aggregate.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with aggregated process reward.
        """
        if not self._is_loaded:
            self.load()

        if self._model is None or self._tokenizer is None:
            return RewardOutput(reward=0.0, metadata={"error": "model_not_loaded"})

        # Split response into steps
        steps = trajectory.response.split(self.step_separator)
        step_rewards = []

        # Compute reward for each step
        cumulative_response = ""
        for step in steps:
            if not step.strip():
                continue

            cumulative_response += step + self.step_separator

            # Create temporary trajectory for this prefix
            step_traj = Trajectory(
                prompt=trajectory.prompt,
                response=cumulative_response.strip(),
            )

            # Compute reward for this step
            output = super().compute_reward(step_traj)
            step_rewards.append(output.reward)

        if not step_rewards:
            return RewardOutput(reward=0.0, metadata={"num_steps": 0})

        # Aggregate step rewards
        if self.aggregate == "min":
            final_reward = min(step_rewards)
        elif self.aggregate == "mean":
            final_reward = sum(step_rewards) / len(step_rewards)
        elif self.aggregate == "last":
            final_reward = step_rewards[-1]
        else:
            final_reward = sum(step_rewards) / len(step_rewards)

        return RewardOutput(
            reward=final_reward,
            token_rewards=[],  # Could provide per-step rewards here
            metadata={
                "num_steps": len(step_rewards),
                "step_rewards": step_rewards,
                "aggregate": self.aggregate,
            },
        )
