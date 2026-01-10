---
title: Implement Custom Rewards
description: Build reward functions for your specific task
---

# Implement Custom Rewards

Create reward functions tailored to your training objectives.

## Basic Pattern

```python
from flux.rewards import RewardFunction, RewardOutput
from flux.core.trajectory import Trajectory

class MyReward(RewardFunction):
    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        response = trajectory.response
        score = self._compute_score(response)
        return RewardOutput(reward=score)
    
    def _compute_score(self, text: str) -> float:
        # Your scoring logic
        return 0.5
```

## Example: Quality Reward

```python
class QualityReward(RewardFunction):
    def __init__(self):
        self.good_words = ["because", "therefore", "example"]
        self.bad_words = ["maybe", "perhaps", "idk"]
    
    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        response = trajectory.response.lower()
        score = 0.5  # Base score
        
        # Reward good words
        for word in self.good_words:
            if word in response:
                score += 0.1
        
        # Penalize bad words
        for word in self.bad_words:
            if word in response:
                score -= 0.15
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return RewardOutput(
            reward=score,
            metadata={"raw_score": score}
        )
```

## Combining Rewards

```python
from flux.rewards import CompositeReward

reward = CompositeReward([
    (QualityReward(), 0.5),
    (LengthReward(target=200), 0.3),
    (FormatReward(), 0.2),
])
```

## Using External Models

```python
from transformers import pipeline

class SentimentReward(RewardFunction):
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")
    
    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        result = self.classifier(trajectory.response[:512])[0]
        score = result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]
        return RewardOutput(reward=score)
```

## Per-Token Rewards

```python
class TokenLevelReward(RewardFunction):
    def compute_reward(self, trajectory: Trajectory) -> RewardOutput:
        response = trajectory.response
        tokens = response.split()
        
        token_rewards = []
        for token in tokens:
            if token in self.good_tokens:
                token_rewards.append(0.1)
            else:
                token_rewards.append(0.0)
        
        return RewardOutput(
            reward=sum(token_rewards),
            token_rewards=token_rewards,
        )
```

## Best Practices

1. **Normalize to [0, 1]** - Keeps training stable
2. **Avoid sparse rewards** - Provide gradual feedback
3. **Log metadata** - Helps debugging
4. **Test distribution** - Check reward statistics

## See Also

- [Custom Rewards Tutorial](../tutorials/custom-rewards.md)
- [Rewards API](../api/rewards.md)
