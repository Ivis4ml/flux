---
title: Rewards API
description: Reward functions and utilities
---

# Rewards API

Reward functions define what "good" means for your training objective.

## RewardFunction Base Class

```python
from flux.rewards import RewardFunction, RewardOutput

class RewardFunction(ABC):
    @abstractmethod
    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        """Compute reward for a trajectory."""
        pass

@dataclass
class RewardOutput:
    reward: float                    # Scalar reward [0, 1]
    token_rewards: list[float] = []  # Optional per-token rewards
    metadata: dict = {}              # Optional metadata
```

## Built-in Reward Functions

### LengthReward

Rewards responses of target length.

```python
from flux.rewards import LengthReward

reward = LengthReward(
    target_length=200,      # Target word count
    tolerance=50,           # Acceptable deviation
    reward_type="gaussian", # "gaussian" or "linear"
)
```

### FormatReward

Rewards structured responses.

```python
from flux.rewards import FormatReward

reward = FormatReward(
    required_sections=["Introduction", "Conclusion"],
    forbidden_patterns=["I don't know", "As an AI"],
)
```

### KeywordReward

Rewards keyword presence.

```python
from flux.rewards import KeywordReward

reward = KeywordReward(
    required_keywords=["because", "therefore"],
    bonus_keywords=["example"],
    penalty_keywords=["maybe"],
)
```

### FunctionReward

Quick custom reward from a function.

```python
from flux.rewards import FunctionReward

def my_scorer(trajectory):
    return 1.0 if len(trajectory.response) > 100 else 0.5

reward = FunctionReward(fn=my_scorer)
```

### CompositeReward

Combine multiple rewards.

```python
from flux.rewards import CompositeReward

reward = CompositeReward([
    (LengthReward(target=150), 0.3),   # 30% weight
    (KeywordReward(required=["because"]), 0.4),
    (FormatReward(required_sections=["Answer"]), 0.3),
])
```

## Model-Based Rewards

### RewardModel

Neural network reward model.

```python
from flux.rewards import RewardModel

reward = RewardModel(
    model_path="OpenAssistant/reward-model-deberta",
    device="cuda",
)
```

### LLMJudge

LLM-as-judge reward.

```python
from flux.rewards import LLMJudge

reward = LLMJudge(
    judge_model="gpt-4",
    prompt_template="Rate this response...",
    api_key="...",
)
```

## Creating Custom Rewards

```python
from flux.rewards import RewardFunction, RewardOutput
from flux.core.trajectory import Trajectory

class QualityReward(RewardFunction):
    def __init__(self, weights: dict = None):
        self.weights = weights or {
            "length": 0.3,
            "structure": 0.3,
            "clarity": 0.4,
        }

    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        response = trajectory.response
        
        # Compute sub-scores
        length_score = self._score_length(response)
        structure_score = self._score_structure(response)
        clarity_score = self._score_clarity(response)
        
        # Weighted combination
        score = (
            self.weights["length"] * length_score +
            self.weights["structure"] * structure_score +
            self.weights["clarity"] * clarity_score
        )
        
        return RewardOutput(
            reward=score,
            metadata={
                "length": length_score,
                "structure": structure_score,
                "clarity": clarity_score,
            }
        )
    
    def _score_length(self, text: str) -> float:
        words = len(text.split())
        return min(1.0, words / 200)
    
    def _score_structure(self, text: str) -> float:
        has_structure = any(x in text for x in ["1.", "First", "•"])
        return 1.0 if has_structure else 0.3
    
    def _score_clarity(self, text: str) -> float:
        # Simplified clarity heuristic
        sentences = text.count(".")
        if sentences < 2:
            return 0.3
        avg_words = len(text.split()) / sentences
        return 1.0 if 10 <= avg_words <= 25 else 0.5
```

## Best Practices

1. **Normalize to [0, 1]**: Keeps gradients stable
2. **Avoid sparse rewards**: Provide gradual feedback
3. **Log metadata**: Helps debugging
4. **Test thoroughly**: Verify reward distribution

## See Also

- [Custom Rewards Tutorial](../tutorials/custom-rewards.md)
- [How-to: Custom Rewards](../how-to/custom-rewards.md)
