"""
Tests for reward functions.
"""

import pytest

from flux.core.trajectory import Trajectory
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


class TestRewardOutput:
    """Tests for RewardOutput dataclass."""

    def test_creation(self):
        """Test reward output creation."""
        output = RewardOutput(reward=0.5)
        assert output.reward == 0.5
        assert output.token_rewards == []
        assert output.metadata == {}

    def test_creation_with_metadata(self):
        """Test with metadata."""
        output = RewardOutput(
            reward=0.8,
            token_rewards=[0.1, 0.2, 0.3],
            metadata={"reason": "good"},
        )
        assert output.reward == 0.8
        assert len(output.token_rewards) == 3
        assert output.metadata["reason"] == "good"


class TestRewardBatchOutput:
    """Tests for RewardBatchOutput dataclass."""

    def test_creation(self):
        """Test batch output creation."""
        output = RewardBatchOutput(rewards=[0.5, 0.6, 0.7])
        assert output.batch_size == 3

    def test_mean_reward(self):
        """Test mean reward calculation."""
        output = RewardBatchOutput(rewards=[0.4, 0.5, 0.6])
        assert output.mean_reward() == pytest.approx(0.5, rel=0.01)

    def test_empty_batch(self):
        """Test empty batch."""
        output = RewardBatchOutput()
        assert output.batch_size == 0
        assert output.mean_reward() == 0.0


class TestLengthReward:
    """Tests for LengthReward."""

    def _make_trajectory(self, response: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(
            id="test",
            prompt="Hello",
            response=response,
            response_length=len(response),
        )

    def test_linear_reward(self):
        """Test linear length reward."""
        reward_fn = LengthReward(reward_type="linear")
        traj = self._make_trajectory("x" * 100)

        output = reward_fn.compute_reward(traj)
        assert 0 <= output.reward <= 1
        assert output.metadata["length"] == 100

    def test_log_reward(self):
        """Test log length reward."""
        reward_fn = LengthReward(reward_type="log")
        traj = self._make_trajectory("x" * 100)

        output = reward_fn.compute_reward(traj)
        assert output.reward > 0

    def test_target_reward(self):
        """Test target length reward."""
        reward_fn = LengthReward(
            reward_type="target",
            target_length=100,
        )

        # Exactly at target
        traj = self._make_trajectory("x" * 100)
        output = reward_fn.compute_reward(traj)
        assert output.reward == pytest.approx(1.0, rel=0.01)

        # Far from target
        traj = self._make_trajectory("x" * 200)
        output = reward_fn.compute_reward(traj)
        assert output.reward < 1.0

    def test_min_length_filter(self):
        """Test minimum length filter."""
        reward_fn = LengthReward(min_length=50)
        traj = self._make_trajectory("x" * 10)

        output = reward_fn.compute_reward(traj)
        assert output.reward == 0.0
        assert output.metadata["reason"] == "too_short"

    def test_max_length_filter(self):
        """Test maximum length filter."""
        reward_fn = LengthReward(max_length=100)
        traj = self._make_trajectory("x" * 200)

        output = reward_fn.compute_reward(traj)
        assert output.reward == 0.0
        assert output.metadata["reason"] == "too_long"


class TestLengthPenalty:
    """Tests for LengthPenalty."""

    def _make_trajectory(self, length: int) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(
            id="test",
            response="x" * length,
            response_length=length,
        )

    def test_no_penalty_in_range(self):
        """Test no penalty when in range."""
        penalty = LengthPenalty(min_length=10, max_length=100)
        traj = self._make_trajectory(50)

        output = penalty.compute_reward(traj)
        assert output.reward == 0.0

    def test_penalty_too_short(self):
        """Test penalty for too short responses."""
        penalty = LengthPenalty(min_length=50, max_length=100)
        traj = self._make_trajectory(10)

        output = penalty.compute_reward(traj)
        assert output.reward < 0
        assert output.metadata["violation"] == "too_short"

    def test_penalty_too_long(self):
        """Test penalty for too long responses."""
        penalty = LengthPenalty(min_length=10, max_length=50)
        traj = self._make_trajectory(100)

        output = penalty.compute_reward(traj)
        assert output.reward < 0
        assert output.metadata["violation"] == "too_long"


class TestFormatReward:
    """Tests for FormatReward."""

    def _make_trajectory(self, response: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(id="test", response=response)

    def test_required_patterns_present(self):
        """Test with required patterns present."""
        reward_fn = FormatReward(
            required_patterns=[r"\d+", r"answer"],
        )
        traj = self._make_trajectory("The answer is 42")

        output = reward_fn.compute_reward(traj)
        assert output.reward == 1.0

    def test_required_patterns_missing(self):
        """Test with required patterns missing."""
        reward_fn = FormatReward(
            required_patterns=[r"\d+", r"answer"],
        )
        traj = self._make_trajectory("No numbers here")

        output = reward_fn.compute_reward(traj)
        assert output.reward < 1.0

    def test_forbidden_patterns(self):
        """Test with forbidden patterns."""
        reward_fn = FormatReward(
            forbidden_patterns=[r"error", r"fail"],
        )
        traj = self._make_trajectory("There was an error")

        output = reward_fn.compute_reward(traj)
        assert output.reward < 1.0


class TestStructuredOutputReward:
    """Tests for StructuredOutputReward."""

    def _make_trajectory(self, response: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(id="test", response=response)

    def test_json_valid(self):
        """Test valid JSON detection."""
        reward_fn = StructuredOutputReward(expected_format="json")
        traj = self._make_trajectory('Here is the data: {"key": "value"}')

        output = reward_fn.compute_reward(traj)
        assert output.reward == 1.0
        assert output.metadata["valid_json"] == True

    def test_json_invalid(self):
        """Test invalid JSON detection."""
        reward_fn = StructuredOutputReward(expected_format="json")
        traj = self._make_trajectory("No JSON here")

        output = reward_fn.compute_reward(traj)
        assert output.reward < 1.0

    def test_code_blocks(self):
        """Test code block detection."""
        reward_fn = StructuredOutputReward(expected_format="code")
        traj = self._make_trajectory("```python\nprint('hello')\n```")

        output = reward_fn.compute_reward(traj)
        assert output.reward == 1.0
        assert output.metadata["code_blocks"] > 0

    def test_markdown(self):
        """Test markdown detection."""
        reward_fn = StructuredOutputReward(expected_format="markdown")
        traj = self._make_trajectory("# Header\n\n**Bold text**")

        output = reward_fn.compute_reward(traj)
        assert output.reward > 0


class TestKeywordReward:
    """Tests for KeywordReward."""

    def _make_trajectory(self, response: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(id="test", response=response)

    def test_required_keywords(self):
        """Test required keywords."""
        reward_fn = KeywordReward(
            required_keywords=["python", "code"],
        )
        traj = self._make_trajectory("Here is some python code")

        output = reward_fn.compute_reward(traj)
        assert output.reward == 1.0

    def test_missing_required_keywords(self):
        """Test missing required keywords."""
        reward_fn = KeywordReward(
            required_keywords=["python", "code"],
        )
        traj = self._make_trajectory("Just some text")

        output = reward_fn.compute_reward(traj)
        assert output.reward < 1.0

    def test_bonus_keywords(self):
        """Test bonus keywords increase reward (capped at 1.0)."""
        reward_fn = KeywordReward(
            bonus_keywords=["excellent", "perfect"],
        )
        traj = self._make_trajectory("This is excellent work, perfect!")

        output = reward_fn.compute_reward(traj)
        # Reward is capped at 1.0, but bonus keywords should be detected
        assert output.reward == 1.0
        assert "excellent" in output.metadata["bonus_found"]
        assert "perfect" in output.metadata["bonus_found"]

    def test_penalty_keywords(self):
        """Test penalty keywords."""
        reward_fn = KeywordReward(
            penalty_keywords=["bad", "wrong"],
        )
        traj = self._make_trajectory("This is bad and wrong")

        output = reward_fn.compute_reward(traj)
        assert output.reward < 1.0


class TestRepetitionPenalty:
    """Tests for RepetitionPenalty."""

    def _make_trajectory(self, response: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(id="test", response=response)

    def test_no_repetition(self):
        """Test no penalty for unique content."""
        penalty = RepetitionPenalty(ngram_size=3)
        traj = self._make_trajectory("This is unique content with no repeats")

        output = penalty.compute_reward(traj)
        assert output.reward >= 0

    def test_high_repetition(self):
        """Test penalty for repetitive content."""
        penalty = RepetitionPenalty(ngram_size=3, max_repetition_ratio=0.2)
        # Highly repetitive text
        traj = self._make_trajectory("hello world hello world hello world hello world")

        output = penalty.compute_reward(traj)
        assert output.reward < 0


class TestMathReward:
    """Tests for MathReward."""

    def _make_trajectory(self, response: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(id="test", response=response)

    def test_boxed_answer(self):
        """Test boxed answer detection."""
        reward_fn = MathReward(check_boxed=True)
        traj = self._make_trajectory("The answer is \\boxed{42}")

        output = reward_fn.compute_reward(traj)
        assert output.reward >= 0.5
        assert output.metadata["has_boxed"] == True

    def test_equation(self):
        """Test equation detection."""
        reward_fn = MathReward(check_equation=True)
        traj = self._make_trajectory("We have x = 5 and 2 + 2 = 4")

        output = reward_fn.compute_reward(traj)
        assert output.reward >= 0.5
        assert output.metadata["has_equation"] == True


class TestCodeReward:
    """Tests for CodeReward."""

    def _make_trajectory(self, response: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(id="test", response=response)

    def test_with_code_blocks(self):
        """Test code block detection."""
        reward_fn = CodeReward()
        traj = self._make_trajectory("```python\ndef foo():\n    pass\n```")

        output = reward_fn.compute_reward(traj)
        assert output.reward > 0
        assert output.metadata["code_blocks"] > 0

    def test_with_language_match(self):
        """Test language matching."""
        reward_fn = CodeReward(language="python")
        traj = self._make_trajectory("```python\nprint('hello')\n```")

        output = reward_fn.compute_reward(traj)
        assert output.reward > 0.5


class TestCompositeReward:
    """Tests for CompositeReward."""

    def _make_trajectory(self, response: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(
            id="test",
            response=response,
            response_length=len(response),
        )

    def test_weighted_combination(self):
        """Test weighted combination of rewards."""
        composite = CompositeReward(
            rewards=[
                (LengthReward(), 0.5),
                (KeywordReward(required_keywords=["test"]), 0.5),
            ],
            normalize=True,
        )

        traj = self._make_trajectory("This is a test response")
        output = composite.compute_reward(traj)

        assert "components" in output.metadata
        assert len(output.metadata["components"]) == 2


class TestFunctionReward:
    """Tests for FunctionReward."""

    def _make_trajectory(self, response: str) -> Trajectory:
        """Helper to create trajectories."""
        return Trajectory(id="test", response=response)

    def test_custom_function(self):
        """Test custom reward function."""
        def my_scorer(traj):
            return len(traj.response) / 100

        reward_fn = FunctionReward(my_scorer, name="custom")
        traj = self._make_trajectory("x" * 50)

        output = reward_fn.compute_reward(traj)
        assert output.reward == 0.5

    def test_function_with_scale(self):
        """Test scaling."""
        def my_scorer(traj):
            return 1.0

        reward_fn = FunctionReward(my_scorer, scale=2.0)
        traj = self._make_trajectory("test")

        output = reward_fn(traj)  # Uses __call__ which applies scale
        assert output.reward == 2.0

    def test_function_with_clip(self):
        """Test clipping."""
        def my_scorer(traj):
            return 100.0

        reward_fn = FunctionReward(my_scorer, clip=1.0)
        traj = self._make_trajectory("test")

        output = reward_fn(traj)
        assert output.reward == 1.0


class TestCreateRewardFunction:
    """Tests for create_reward_function factory."""

    def test_create_length(self):
        """Test creating length reward."""
        fn = create_reward_function("length")
        assert isinstance(fn, LengthReward)

    def test_create_format(self):
        """Test creating format reward."""
        fn = create_reward_function("format", required_patterns=[r"\d+"])
        assert isinstance(fn, FormatReward)

    def test_create_unknown_raises(self):
        """Test unknown type raises error."""
        with pytest.raises(ValueError, match="Unknown reward type"):
            create_reward_function("unknown_type")
