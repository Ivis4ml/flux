"""
Rule-based reward functions for Flux.

This module provides fast, deterministic reward functions based on heuristics
that don't require a neural network. Useful for format compliance, length
control, and basic quality signals.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from flux.core.trajectory import Trajectory
from flux.rewards.base import RewardFunction, RewardOutput


logger = logging.getLogger(__name__)


# =============================================================================
# Length-based rewards
# =============================================================================


class LengthReward(RewardFunction):
    """Reward based on response length.

    Can reward longer responses, shorter responses, or target a specific length.
    """

    def __init__(
        self,
        target_length: int | None = None,
        min_length: int = 0,
        max_length: int | None = None,
        reward_type: str = "linear",
        name: str = "length",
    ):
        """Initialize length reward.

        Args:
            target_length: Target length for maximum reward (None for no target).
            min_length: Minimum length for any positive reward.
            max_length: Maximum length (longer gets 0 reward).
            reward_type: "linear", "log", or "target".
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.target_length = target_length
        self.min_length = min_length
        self.max_length = max_length
        self.reward_type = reward_type

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute length-based reward.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with length reward.
        """
        length = trajectory.response_length or len(trajectory.response)

        # Check bounds
        if length < self.min_length:
            return RewardOutput(reward=0.0, metadata={"length": length, "reason": "too_short"})

        if self.max_length is not None and length > self.max_length:
            return RewardOutput(reward=0.0, metadata={"length": length, "reason": "too_long"})

        # Compute reward based on type
        if self.reward_type == "target" and self.target_length is not None:
            # Gaussian-like reward centered on target
            diff = abs(length - self.target_length) / self.target_length
            reward = max(0.0, 1.0 - diff)
        elif self.reward_type == "log":
            # Logarithmic reward (diminishing returns for length)
            import math
            reward = math.log(1 + length) / 10.0
        else:
            # Linear reward (normalized by max_length or 2048)
            normalizer = self.max_length or 2048
            reward = min(1.0, length / normalizer)

        return RewardOutput(reward=reward, metadata={"length": length})


class LengthPenalty(RewardFunction):
    """Penalize responses that are too long or too short."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 2048,
        penalty_scale: float = 0.1,
        name: str = "length_penalty",
    ):
        """Initialize length penalty.

        Args:
            min_length: Minimum acceptable length.
            max_length: Maximum acceptable length.
            penalty_scale: Scale factor for penalty.
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.min_length = min_length
        self.max_length = max_length
        self.penalty_scale = penalty_scale

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute length penalty.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with length penalty (0 if within bounds, negative otherwise).
        """
        length = trajectory.response_length or len(trajectory.response)

        if length < self.min_length:
            # Penalty for being too short
            penalty = (self.min_length - length) / self.min_length * self.penalty_scale
            return RewardOutput(reward=-penalty, metadata={"length": length, "violation": "too_short"})

        if length > self.max_length:
            # Penalty for being too long
            penalty = (length - self.max_length) / self.max_length * self.penalty_scale
            return RewardOutput(reward=-penalty, metadata={"length": length, "violation": "too_long"})

        return RewardOutput(reward=0.0, metadata={"length": length})


# =============================================================================
# Format-based rewards
# =============================================================================


class FormatReward(RewardFunction):
    """Reward for following specified output format."""

    def __init__(
        self,
        required_patterns: list[str] | None = None,
        forbidden_patterns: list[str] | None = None,
        case_sensitive: bool = False,
        name: str = "format",
    ):
        """Initialize format reward.

        Args:
            required_patterns: Regex patterns that must be present.
            forbidden_patterns: Regex patterns that must not be present.
            case_sensitive: Whether pattern matching is case-sensitive.
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.required_patterns = required_patterns or []
        self.forbidden_patterns = forbidden_patterns or []
        self.case_sensitive = case_sensitive

        # Compile patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        self._required = [re.compile(p, flags) for p in self.required_patterns]
        self._forbidden = [re.compile(p, flags) for p in self.forbidden_patterns]

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute format compliance reward.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with format reward.
        """
        response = trajectory.response
        metadata = {"required_matches": [], "forbidden_violations": []}

        # Check required patterns
        required_score = 1.0
        for i, pattern in enumerate(self._required):
            if pattern.search(response):
                metadata["required_matches"].append(i)
            else:
                required_score -= 1.0 / len(self._required)

        # Check forbidden patterns
        forbidden_penalty = 0.0
        for i, pattern in enumerate(self._forbidden):
            if pattern.search(response):
                forbidden_penalty += 1.0 / max(1, len(self._forbidden))
                metadata["forbidden_violations"].append(i)

        reward = max(0.0, required_score - forbidden_penalty)
        return RewardOutput(reward=reward, metadata=metadata)


class StructuredOutputReward(RewardFunction):
    """Reward for structured outputs (JSON, code blocks, etc.)."""

    def __init__(
        self,
        expected_format: str = "json",
        strict: bool = False,
        name: str = "structured_output",
    ):
        """Initialize structured output reward.

        Args:
            expected_format: Expected format ("json", "code", "markdown", "xml").
            strict: Whether to require valid parsing.
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.expected_format = expected_format
        self.strict = strict

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute structured output reward.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with structure reward.
        """
        response = trajectory.response
        metadata = {"format": self.expected_format}

        if self.expected_format == "json":
            reward, meta = self._check_json(response)
        elif self.expected_format == "code":
            reward, meta = self._check_code_blocks(response)
        elif self.expected_format == "markdown":
            reward, meta = self._check_markdown(response)
        elif self.expected_format == "xml":
            reward, meta = self._check_xml(response)
        else:
            reward, meta = 0.0, {"error": f"Unknown format: {self.expected_format}"}

        metadata.update(meta)
        return RewardOutput(reward=reward, metadata=metadata)

    def _check_json(self, text: str) -> tuple[float, dict]:
        """Check for valid JSON."""
        import json

        # Look for JSON in response
        json_pattern = re.compile(r'\{[^{}]*\}|\[[^\[\]]*\]', re.DOTALL)
        matches = json_pattern.findall(text)

        if not matches:
            return 0.0, {"valid_json": False, "reason": "no_json_found"}

        # Try to parse
        for match in matches:
            try:
                json.loads(match)
                return 1.0, {"valid_json": True}
            except json.JSONDecodeError:
                continue

        if self.strict:
            return 0.0, {"valid_json": False, "reason": "parse_failed"}
        return 0.5, {"valid_json": False, "reason": "json_like_found"}

    def _check_code_blocks(self, text: str) -> tuple[float, dict]:
        """Check for code blocks."""
        code_block_pattern = re.compile(r'```\w*\n[\s\S]*?```')
        matches = code_block_pattern.findall(text)

        if matches:
            return 1.0, {"code_blocks": len(matches)}
        return 0.0, {"code_blocks": 0}

    def _check_markdown(self, text: str) -> tuple[float, dict]:
        """Check for markdown formatting."""
        indicators = [
            (r'^#{1,6}\s', 0.2),  # Headers
            (r'\*\*[^*]+\*\*', 0.1),  # Bold
            (r'\*[^*]+\*', 0.1),  # Italic
            (r'^\s*[-*+]\s', 0.2),  # Lists
            (r'```', 0.2),  # Code blocks
            (r'\[.+\]\(.+\)', 0.2),  # Links
        ]

        score = 0.0
        found = []
        for pattern, weight in indicators:
            if re.search(pattern, text, re.MULTILINE):
                score += weight
                found.append(pattern)

        return min(1.0, score), {"markdown_features": len(found)}

    def _check_xml(self, text: str) -> tuple[float, dict]:
        """Check for XML structure."""
        xml_pattern = re.compile(r'<[^>]+>.*?</[^>]+>', re.DOTALL)
        matches = xml_pattern.findall(text)

        if matches:
            return 1.0, {"xml_tags": len(matches)}
        return 0.0, {"xml_tags": 0}


# =============================================================================
# Content-based rewards
# =============================================================================


class KeywordReward(RewardFunction):
    """Reward based on presence of keywords."""

    def __init__(
        self,
        required_keywords: list[str] | None = None,
        bonus_keywords: list[str] | None = None,
        penalty_keywords: list[str] | None = None,
        case_sensitive: bool = False,
        name: str = "keyword",
    ):
        """Initialize keyword reward.

        Args:
            required_keywords: Keywords that must be present.
            bonus_keywords: Keywords that give bonus reward.
            penalty_keywords: Keywords that reduce reward.
            case_sensitive: Whether matching is case-sensitive.
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.required_keywords = required_keywords or []
        self.bonus_keywords = bonus_keywords or []
        self.penalty_keywords = penalty_keywords or []
        self.case_sensitive = case_sensitive

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute keyword-based reward.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with keyword reward.
        """
        response = trajectory.response
        if not self.case_sensitive:
            response = response.lower()

        metadata = {"required_found": [], "bonus_found": [], "penalty_found": []}

        # Check required keywords
        required_score = 0.0
        if self.required_keywords:
            for kw in self.required_keywords:
                check_kw = kw if self.case_sensitive else kw.lower()
                if check_kw in response:
                    required_score += 1.0 / len(self.required_keywords)
                    metadata["required_found"].append(kw)
        else:
            required_score = 1.0  # No required keywords means full score

        # Check bonus keywords
        bonus_score = 0.0
        if self.bonus_keywords:
            for kw in self.bonus_keywords:
                check_kw = kw if self.case_sensitive else kw.lower()
                if check_kw in response:
                    bonus_score += 0.1  # 0.1 bonus per keyword
                    metadata["bonus_found"].append(kw)

        # Check penalty keywords
        penalty = 0.0
        for kw in self.penalty_keywords:
            check_kw = kw if self.case_sensitive else kw.lower()
            if check_kw in response:
                penalty += 0.2  # 0.2 penalty per keyword
                metadata["penalty_found"].append(kw)

        reward = max(0.0, required_score + bonus_score - penalty)
        return RewardOutput(reward=min(1.0, reward), metadata=metadata)


class RepetitionPenalty(RewardFunction):
    """Penalize repetitive content in responses."""

    def __init__(
        self,
        ngram_size: int = 3,
        max_repetition_ratio: float = 0.3,
        name: str = "repetition_penalty",
    ):
        """Initialize repetition penalty.

        Args:
            ngram_size: Size of n-grams to check for repetition.
            max_repetition_ratio: Maximum acceptable repetition ratio.
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.ngram_size = ngram_size
        self.max_repetition_ratio = max_repetition_ratio

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute repetition penalty.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with repetition penalty.
        """
        response = trajectory.response
        words = response.split()

        if len(words) < self.ngram_size:
            return RewardOutput(reward=0.0, metadata={"repetition_ratio": 0.0})

        # Extract n-grams
        ngrams = []
        for i in range(len(words) - self.ngram_size + 1):
            ngram = " ".join(words[i:i + self.ngram_size])
            ngrams.append(ngram)

        if not ngrams:
            return RewardOutput(reward=0.0, metadata={"repetition_ratio": 0.0})

        # Calculate repetition ratio
        unique_ngrams = set(ngrams)
        repetition_ratio = 1.0 - len(unique_ngrams) / len(ngrams)

        # Convert to penalty
        if repetition_ratio > self.max_repetition_ratio:
            excess = repetition_ratio - self.max_repetition_ratio
            penalty = min(1.0, excess * 2)  # Scale penalty
            return RewardOutput(reward=-penalty, metadata={
                "repetition_ratio": repetition_ratio,
                "unique_ngrams": len(unique_ngrams),
                "total_ngrams": len(ngrams),
            })

        return RewardOutput(reward=0.0, metadata={"repetition_ratio": repetition_ratio})


# =============================================================================
# Task-specific rewards
# =============================================================================


class MathReward(RewardFunction):
    """Reward for mathematical correctness.

    Simple pattern-based check for math expressions and common formats.
    """

    def __init__(
        self,
        check_boxed: bool = True,
        check_equation: bool = True,
        name: str = "math",
    ):
        """Initialize math reward.

        Args:
            check_boxed: Check for \\boxed{} format.
            check_equation: Check for equation-like content.
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.check_boxed = check_boxed
        self.check_equation = check_equation

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute math format reward.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with math format reward.
        """
        response = trajectory.response
        metadata = {}
        score = 0.0

        # Check for boxed answer
        if self.check_boxed:
            boxed_pattern = re.compile(r'\\boxed\{[^}]+\}')
            if boxed_pattern.search(response):
                score += 0.5
                metadata["has_boxed"] = True
            else:
                metadata["has_boxed"] = False

        # Check for equation-like content
        if self.check_equation:
            equation_patterns = [
                r'[0-9]+\s*[+\-*/=]\s*[0-9]+',  # Basic arithmetic
                r'\\frac\{',  # Fractions
                r'\\sqrt\{',  # Square roots
                r'[a-z]\s*=\s*[0-9]',  # Variable assignment
            ]
            has_equation = any(re.search(p, response) for p in equation_patterns)
            if has_equation:
                score += 0.5
                metadata["has_equation"] = True
            else:
                metadata["has_equation"] = False

        return RewardOutput(reward=score, metadata=metadata)


class CodeReward(RewardFunction):
    """Reward for code quality and correctness signals."""

    def __init__(
        self,
        language: str | None = None,
        check_syntax: bool = False,
        name: str = "code",
    ):
        """Initialize code reward.

        Args:
            language: Expected programming language.
            check_syntax: Whether to check syntax (requires additional setup).
            name: Name for this reward function.
        """
        super().__init__(name=name)
        self.language = language
        self.check_syntax = check_syntax

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute code reward.

        Args:
            trajectory: Trajectory to score.

        Returns:
            RewardOutput with code reward.
        """
        response = trajectory.response
        metadata = {}
        score = 0.0

        # Check for code blocks
        code_block_pattern = re.compile(r'```(\w*)\n([\s\S]*?)```')
        matches = code_block_pattern.findall(response)

        if matches:
            score += 0.3
            metadata["code_blocks"] = len(matches)

            # Check language if specified
            if self.language:
                lang_match = any(
                    lang.lower() == self.language.lower()
                    for lang, _ in matches
                )
                if lang_match:
                    score += 0.2
                    metadata["language_match"] = True

            # Check for common code patterns
            for _, code in matches:
                if self._has_code_structure(code):
                    score += 0.2
                    break
        else:
            metadata["code_blocks"] = 0

        # Check for inline code
        inline_code = re.findall(r'`[^`]+`', response)
        if inline_code:
            score += 0.1
            metadata["inline_code"] = len(inline_code)

        return RewardOutput(reward=min(1.0, score), metadata=metadata)

    def _has_code_structure(self, code: str) -> bool:
        """Check if code has typical structural elements."""
        patterns = [
            r'def\s+\w+\s*\(',  # Python function
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'import\s+',  # Import statement
            r'return\s+',  # Return statement
            r'if\s+.*:',  # Conditional
            r'for\s+.*:',  # Loop
        ]
        return any(re.search(p, code) for p in patterns)


# =============================================================================
# Utility functions
# =============================================================================


def create_reward_function(
    reward_type: str,
    **kwargs: Any,
) -> RewardFunction:
    """Factory function to create reward functions by name.

    Args:
        reward_type: Type of reward function.
        **kwargs: Arguments for the reward function.

    Returns:
        RewardFunction instance.

    Raises:
        ValueError: If reward_type is unknown.
    """
    reward_types = {
        "length": LengthReward,
        "length_penalty": LengthPenalty,
        "format": FormatReward,
        "structured_output": StructuredOutputReward,
        "keyword": KeywordReward,
        "repetition_penalty": RepetitionPenalty,
        "math": MathReward,
        "code": CodeReward,
    }

    if reward_type not in reward_types:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available types: {list(reward_types.keys())}"
        )

    return reward_types[reward_type](**kwargs)
