"""
Length predictor for rollout scheduling.

Predicts the output length for prompts to enable efficient scheduling:
- Short prompts scheduled first for lower latency
- Long prompts grouped together to reduce padding waste
- Adaptive predictions based on historical data

Based on empirical observation that prompt features correlate with output length.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class LengthPrediction:
    """Prediction of output length."""

    prompt: str
    predicted_length: int
    confidence: float  # [0, 1]
    prompt_features: dict[str, float] = field(default_factory=dict)


@dataclass
class LengthObservation:
    """Observed output length for learning."""

    prompt: str
    actual_length: int
    prompt_length: int
    prompt_features: dict[str, float] = field(default_factory=dict)


class LengthPredictor:
    """Predicts output length from prompt features.

    Uses a combination of:
    1. Simple heuristics based on prompt characteristics
    2. Historical statistics for similar prompts
    3. Optional learned model for fine-grained prediction

    Features used:
    - Prompt length (tokens/characters)
    - Question indicators (?, "what", "how", etc.)
    - Code indicators (```, function, def, etc.)
    - List indicators (numbered lists, bullets)

    Example:
        predictor = LengthPredictor()

        # Predict lengths
        predictions = predictor.predict_batch(prompts)

        # Sort by predicted length for scheduling
        sorted_prompts = sorted(
            zip(prompts, predictions),
            key=lambda x: x[1].predicted_length
        )

        # Update with actual lengths
        predictor.observe(prompt, actual_length)
    """

    def __init__(
        self,
        default_length: int = 256,
        min_length: int = 16,
        max_length: int = 4096,
        history_size: int = 1000,
        use_learned_model: bool = False,
    ) -> None:
        """Initialize the length predictor.

        Args:
            default_length: Default prediction when no features match.
            min_length: Minimum predicted length.
            max_length: Maximum predicted length.
            history_size: Number of observations to keep for statistics.
            use_learned_model: Whether to use a learned model.
        """
        self.default_length = default_length
        self.min_length = min_length
        self.max_length = max_length
        self.history_size = history_size
        self.use_learned_model = use_learned_model

        # Historical observations
        self._history: deque[LengthObservation] = deque(maxlen=history_size)

        # Running statistics
        self._mean_length = float(default_length)
        self._std_length = float(default_length / 2)
        self._total_observed = 0

        # Feature-based statistics
        self._feature_stats: dict[str, tuple[float, float, int]] = {}  # mean, var, count

        # Learned model (optional)
        self._model: nn.Module | None = None
        if use_learned_model:
            self._model = self._create_model()

    def _create_model(self) -> nn.Module:
        """Create a simple MLP for length prediction."""

        class LengthMLP(nn.Module):
            def __init__(self, input_dim: int = 8):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Softplus(),  # Ensure positive output
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        return LengthMLP()

    def predict(self, prompt: str) -> LengthPrediction:
        """Predict output length for a single prompt.

        Args:
            prompt: The input prompt.

        Returns:
            LengthPrediction with predicted length and confidence.
        """
        features = self._extract_features(prompt)
        predicted_length = self._predict_from_features(features)
        confidence = self._compute_confidence(features)

        return LengthPrediction(
            prompt=prompt,
            predicted_length=predicted_length,
            confidence=confidence,
            prompt_features=features,
        )

    def predict_batch(self, prompts: list[str]) -> list[LengthPrediction]:
        """Predict output lengths for a batch of prompts.

        Args:
            prompts: List of input prompts.

        Returns:
            List of LengthPredictions.
        """
        return [self.predict(prompt) for prompt in prompts]

    def observe(self, prompt: str, actual_length: int) -> None:
        """Record an observed output length.

        Used to improve predictions over time.

        Args:
            prompt: The input prompt.
            actual_length: The actual output length.
        """
        features = self._extract_features(prompt)

        observation = LengthObservation(
            prompt=prompt,
            actual_length=actual_length,
            prompt_length=len(prompt),
            prompt_features=features,
        )
        self._history.append(observation)

        # Update statistics
        self._update_statistics(features, actual_length)

    def observe_batch(
        self, prompts: list[str], actual_lengths: list[int]
    ) -> None:
        """Record multiple observations.

        Args:
            prompts: List of prompts.
            actual_lengths: List of actual output lengths.
        """
        for prompt, length in zip(prompts, actual_lengths):
            self.observe(prompt, length)

    def _extract_features(self, prompt: str) -> dict[str, float]:
        """Extract features from prompt for prediction."""
        features = {}

        # Length features
        features["char_length"] = len(prompt)
        features["word_count"] = len(prompt.split())
        features["line_count"] = prompt.count("\n") + 1

        # Question indicators
        question_words = ["what", "how", "why", "when", "where", "who", "which", "explain"]
        features["is_question"] = float(
            "?" in prompt or
            any(prompt.lower().startswith(w) for w in question_words)
        )

        # Code indicators
        code_patterns = ["```", "def ", "function", "class ", "import ", "return "]
        features["has_code"] = float(any(p in prompt for p in code_patterns))

        # List/structured output indicators
        list_patterns = ["1.", "2.", "- ", "* ", "list", "enumerate"]
        features["wants_list"] = float(any(p in prompt.lower() for p in list_patterns))

        # Summary/brief indicators
        brief_patterns = ["briefly", "short", "concise", "summarize", "tldr"]
        features["wants_brief"] = float(any(p in prompt.lower() for p in brief_patterns))

        # Long/detailed indicators
        long_patterns = ["detail", "explain", "elaborate", "comprehensive", "thorough"]
        features["wants_detail"] = float(any(p in prompt.lower() for p in long_patterns))

        # Math/reasoning indicators
        math_patterns = ["calculate", "solve", "proof", "derive", "equation"]
        features["has_math"] = float(any(p in prompt.lower() for p in math_patterns))

        return features

    def _predict_from_features(self, features: dict[str, float]) -> int:
        """Predict length from extracted features."""
        # Start with base prediction
        base_length = self._mean_length

        # Adjust based on prompt length
        char_length = features.get("char_length", 100)
        length_ratio = min(2.0, char_length / 100)  # Longer prompts -> longer outputs
        predicted = base_length * (0.8 + 0.4 * length_ratio)

        # Adjust based on content type
        if features.get("is_question", 0):
            predicted *= 1.2  # Questions often need longer answers

        if features.get("has_code", 0):
            predicted *= 1.5  # Code generation tends to be longer

        if features.get("wants_list", 0):
            predicted *= 1.3  # Lists are longer

        if features.get("wants_brief", 0):
            predicted *= 0.5  # User wants brevity

        if features.get("wants_detail", 0):
            predicted *= 1.5  # User wants detail

        if features.get("has_math", 0):
            predicted *= 1.4  # Math problems need steps

        # Use feature statistics if available
        for feature_name, value in features.items():
            if value > 0 and feature_name in self._feature_stats:
                mean, var, count = self._feature_stats[feature_name]
                if count >= 10:
                    # Blend with feature-specific mean
                    weight = min(0.5, count / 100)
                    predicted = (1 - weight) * predicted + weight * mean

        # Clamp to valid range
        predicted = max(self.min_length, min(self.max_length, int(predicted)))

        return predicted

    def _compute_confidence(self, features: dict[str, float]) -> float:
        """Compute confidence in the prediction."""
        # Base confidence from observation count
        base_confidence = min(0.8, self._total_observed / 100)

        # Adjust based on feature matches
        matched_features = 0
        for feature_name, value in features.items():
            if value > 0 and feature_name in self._feature_stats:
                _, _, count = self._feature_stats[feature_name]
                if count >= 10:
                    matched_features += 1

        feature_confidence = min(0.2, matched_features * 0.05)

        return base_confidence + feature_confidence

    def _update_statistics(
        self, features: dict[str, float], actual_length: int
    ) -> None:
        """Update running statistics with new observation."""
        self._total_observed += 1

        # Update global statistics (Welford's algorithm)
        delta = actual_length - self._mean_length
        self._mean_length += delta / self._total_observed
        delta2 = actual_length - self._mean_length
        self._std_length = math.sqrt(
            (self._std_length ** 2 * (self._total_observed - 1) + delta * delta2)
            / self._total_observed
        )

        # Update feature-specific statistics
        for feature_name, value in features.items():
            if value > 0:
                if feature_name not in self._feature_stats:
                    self._feature_stats[feature_name] = (actual_length, 0.0, 1)
                else:
                    old_mean, old_var, count = self._feature_stats[feature_name]
                    count += 1
                    delta = actual_length - old_mean
                    new_mean = old_mean + delta / count
                    delta2 = actual_length - new_mean
                    new_var = (old_var * (count - 1) + delta * delta2) / count
                    self._feature_stats[feature_name] = (new_mean, new_var, count)

    def sort_by_length(
        self, prompts: list[str], ascending: bool = True
    ) -> list[tuple[str, LengthPrediction]]:
        """Sort prompts by predicted output length.

        Args:
            prompts: List of prompts.
            ascending: If True, shortest first; if False, longest first.

        Returns:
            List of (prompt, prediction) tuples sorted by length.
        """
        predictions = self.predict_batch(prompts)
        sorted_pairs = sorted(
            zip(prompts, predictions),
            key=lambda x: x[1].predicted_length,
            reverse=not ascending,
        )
        return sorted_pairs

    def bucket_by_length(
        self,
        prompts: list[str],
        bucket_boundaries: tuple[int, ...] = (256, 512, 1024, 2048),
    ) -> dict[str, list[tuple[str, LengthPrediction]]]:
        """Bucket prompts by predicted length.

        Args:
            prompts: List of prompts.
            bucket_boundaries: Length boundaries for buckets.

        Returns:
            Dict mapping bucket name to list of (prompt, prediction) tuples.
        """
        predictions = self.predict_batch(prompts)
        buckets: dict[str, list[tuple[str, LengthPrediction]]] = {
            f"<{bucket_boundaries[0]}": []
        }
        for i in range(len(bucket_boundaries) - 1):
            buckets[f"{bucket_boundaries[i]}-{bucket_boundaries[i+1]}"] = []
        buckets[f">{bucket_boundaries[-1]}"] = []

        for prompt, pred in zip(prompts, predictions):
            length = pred.predicted_length
            if length < bucket_boundaries[0]:
                bucket = f"<{bucket_boundaries[0]}"
            elif length > bucket_boundaries[-1]:
                bucket = f">{bucket_boundaries[-1]}"
            else:
                for i in range(len(bucket_boundaries) - 1):
                    if bucket_boundaries[i] <= length < bucket_boundaries[i + 1]:
                        bucket = f"{bucket_boundaries[i]}-{bucket_boundaries[i+1]}"
                        break

            buckets[bucket].append((prompt, pred))

        return buckets

    def get_statistics(self) -> dict[str, float]:
        """Get predictor statistics.

        Returns:
            Dict with prediction statistics.
        """
        return {
            "mean_length": self._mean_length,
            "std_length": self._std_length,
            "total_observed": self._total_observed,
            "num_features_tracked": len(self._feature_stats),
        }
