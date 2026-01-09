#!/usr/bin/env python3
"""
Custom Reward Function Example

This example demonstrates how to create custom reward functions
for specialized training objectives.

Usage:
    python examples/custom_reward.py
"""

import logging
import re
from dataclasses import dataclass

from flux import FluxConfig, FluxTrainer
from flux.core.trajectory import Trajectory
from flux.rewards import (
    CompositeReward,
    FunctionReward,
    KeywordReward,
    LengthReward,
    RewardFunction,
    RewardOutput,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Custom Reward Class
# =============================================================================


class StructuredResponseReward(RewardFunction):
    """Reward for well-structured responses with clear sections.

    This reward function checks for:
    - Numbered lists or bullet points
    - Clear section headers
    - Appropriate length
    - Conclusion/summary section
    """

    def __init__(
        self,
        min_sections: int = 2,
        require_conclusion: bool = True,
        scale: float = 1.0,
    ):
        """Initialize structured response reward.

        Args:
            min_sections: Minimum number of sections required.
            require_conclusion: Whether to require a conclusion.
            scale: Reward scaling factor.
        """
        super().__init__(scale=scale)
        self.min_sections = min_sections
        self.require_conclusion = require_conclusion

    def _compute_raw_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute reward for structured responses."""
        response = trajectory.response
        score = 0.0
        metadata = {}

        # Check for numbered lists (1., 2., etc.)
        numbered_pattern = r"^\d+\.\s"
        numbered_items = len(re.findall(numbered_pattern, response, re.MULTILINE))
        metadata["numbered_items"] = numbered_items

        # Check for bullet points
        bullet_pattern = r"^[-*•]\s"
        bullet_items = len(re.findall(bullet_pattern, response, re.MULTILINE))
        metadata["bullet_items"] = bullet_items

        # Check for section headers (lines ending with :)
        header_pattern = r"^[A-Z][^:]+:\s*$"
        headers = len(re.findall(header_pattern, response, re.MULTILINE))
        metadata["headers"] = headers

        # Check for markdown headers
        md_header_pattern = r"^#{1,3}\s"
        md_headers = len(re.findall(md_header_pattern, response, re.MULTILINE))
        metadata["md_headers"] = md_headers

        # Total structure elements
        total_structure = numbered_items + bullet_items + headers + md_headers
        metadata["total_structure"] = total_structure

        # Score based on structure
        if total_structure >= self.min_sections:
            score += 0.5
        elif total_structure > 0:
            score += 0.25

        # Check for conclusion
        conclusion_patterns = [
            r"in conclusion",
            r"to summarize",
            r"in summary",
            r"overall",
            r"finally",
        ]
        has_conclusion = any(
            re.search(pattern, response.lower())
            for pattern in conclusion_patterns
        )
        metadata["has_conclusion"] = has_conclusion

        if has_conclusion or not self.require_conclusion:
            score += 0.3

        # Bonus for good length
        word_count = len(response.split())
        metadata["word_count"] = word_count
        if 100 <= word_count <= 500:
            score += 0.2

        return RewardOutput(
            reward=min(score, 1.0),  # Cap at 1.0
            metadata=metadata,
        )


class CodeQualityReward(RewardFunction):
    """Reward for code quality in responses.

    Checks for:
    - Proper code blocks
    - Comments
    - Function/class definitions
    - Docstrings
    """

    def __init__(self, language: str = "python", scale: float = 1.0):
        super().__init__(scale=scale)
        self.language = language

    def _compute_raw_reward(self, trajectory: Trajectory) -> RewardOutput:
        response = trajectory.response
        score = 0.0
        metadata = {}

        # Check for code blocks
        code_block_pattern = r"```[\s\S]*?```"
        code_blocks = re.findall(code_block_pattern, response)
        metadata["code_blocks"] = len(code_blocks)

        if code_blocks:
            score += 0.3

            # Extract code content
            code_content = "\n".join(code_blocks)

            # Check for function definitions
            if self.language == "python":
                functions = len(re.findall(r"def \w+\(", code_content))
                classes = len(re.findall(r"class \w+", code_content))
                docstrings = len(re.findall(r'"""[\s\S]*?"""', code_content))
                comments = len(re.findall(r"#.*$", code_content, re.MULTILINE))

                metadata["functions"] = functions
                metadata["classes"] = classes
                metadata["docstrings"] = docstrings
                metadata["comments"] = comments

                # Score components
                if functions > 0 or classes > 0:
                    score += 0.3
                if docstrings > 0:
                    score += 0.2
                if comments > 0:
                    score += 0.1

        # Check for explanation text outside code
        non_code = re.sub(code_block_pattern, "", response)
        if len(non_code.split()) > 20:
            score += 0.1
            metadata["has_explanation"] = True

        return RewardOutput(
            reward=min(score, 1.0),
            metadata=metadata,
        )


# =============================================================================
# Using Function Rewards
# =============================================================================


def helpfulness_score(trajectory: Trajectory) -> float:
    """Simple helpfulness heuristic.

    Checks for helpful indicators like:
    - Direct answers
    - Step-by-step guidance
    - Examples
    """
    response = trajectory.response.lower()
    score = 0.0

    # Direct answer indicators
    if any(phrase in response for phrase in ["here's", "here is", "the answer"]):
        score += 0.2

    # Step-by-step indicators
    if any(phrase in response for phrase in ["step", "first", "then", "next"]):
        score += 0.2

    # Example indicators
    if any(phrase in response for phrase in ["for example", "such as", "e.g."]):
        score += 0.2

    # Question acknowledgment
    if any(phrase in response for phrase in ["you asked", "your question"]):
        score += 0.1

    # Offers further help
    if any(phrase in response for phrase in ["let me know", "feel free", "hope this"]):
        score += 0.1

    # Length bonus (not too short, not too long)
    words = len(response.split())
    if 50 <= words <= 300:
        score += 0.2

    return min(score, 1.0)


# =============================================================================
# Main Example
# =============================================================================


def main():
    logger.info("Custom Reward Functions Example")
    logger.info("=" * 50)

    # Create test trajectories
    trajectories = [
        Trajectory(
            id="t1",
            prompt="Explain machine learning",
            response="""Machine learning is a subset of artificial intelligence.

**Key Concepts:**
1. Training data is used to build models
2. Models learn patterns from data
3. Predictions are made on new data

In conclusion, machine learning enables computers to learn from experience.""",
        ),
        Trajectory(
            id="t2",
            prompt="Write a Python function",
            response='''Here's a Python function to calculate factorial:

```python
def factorial(n: int) -> int:
    """Calculate factorial of n.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

This recursive function handles the base case (n <= 1) and
recursively multiplies n by factorial(n-1).''',
        ),
        Trajectory(
            id="t3",
            prompt="What is 2+2?",
            response="4",
        ),
    ]

    # Test individual rewards
    logger.info("\n--- Testing Individual Rewards ---")

    # Structured response reward
    struct_reward = StructuredResponseReward(min_sections=2)
    for traj in trajectories:
        output = struct_reward.compute_reward(traj)
        logger.info(f"Structured Reward for '{traj.prompt[:30]}...': {output.reward:.2f}")
        logger.info(f"  Metadata: {output.metadata}")

    # Code quality reward
    code_reward = CodeQualityReward(language="python")
    for traj in trajectories:
        output = code_reward.compute_reward(traj)
        logger.info(f"Code Quality for '{traj.prompt[:30]}...': {output.reward:.2f}")

    # Function-based reward
    help_reward = FunctionReward(fn=helpfulness_score)
    for traj in trajectories:
        output = help_reward.compute_reward(traj)
        logger.info(f"Helpfulness for '{traj.prompt[:30]}...': {output.reward:.2f}")

    # Test composite reward
    logger.info("\n--- Testing Composite Reward ---")

    composite = CompositeReward(
        rewards=[
            (StructuredResponseReward(), 0.3),
            (CodeQualityReward(), 0.3),
            (FunctionReward(fn=helpfulness_score), 0.2),
            (LengthReward(target_length=100), 0.2),
        ],
    )

    for traj in trajectories:
        output = composite.compute_reward(traj)
        logger.info(f"Composite for '{traj.prompt[:30]}...': {output.reward:.2f}")
        logger.info(f"  Components: {output.metadata.get('components', {})}")

    # Example training setup
    logger.info("\n--- Training Setup Example ---")

    config = FluxConfig(
        model_path="Qwen/Qwen3-8B",
        output_dir="./outputs/custom_reward",
        num_steps=1000,
        batch_size=32,
    )

    # Create sophisticated reward for coding tasks
    coding_reward = CompositeReward(
        rewards=[
            (CodeQualityReward(language="python"), 0.4),
            (StructuredResponseReward(require_conclusion=False), 0.2),
            (FunctionReward(fn=helpfulness_score), 0.2),
            (KeywordReward(
                required_keywords=["def", "return"],
                bonus_keywords=["example", "usage"],
            ), 0.2),
        ],
    )

    logger.info(f"Created coding reward with {len(coding_reward.rewards)} components")
    logger.info("Ready for training with custom reward function!")

    # Note: Actual training would require GPU and model
    # trainer = FluxTrainer(config, reward_function=coding_reward)
    # result = trainer.fit(prompts=coding_prompts)


if __name__ == "__main__":
    main()
