---
title: Code Style
description: Coding standards and conventions
---

# Code Style

Coding standards for Flux contributions.

## Formatting

We use:
- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

```bash
# Format code
black .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Type check
mypy flux/
```

## Python Style

### Type Hints

```python
# Good
def process(items: list[str], count: int = 10) -> dict[str, int]:
    ...

# Bad
def process(items, count=10):
    ...
```

### Docstrings

```python
def compute_reward(
    trajectory: Trajectory,
    threshold: float = 0.5,
) -> RewardOutput:
    """Compute reward for trajectory.
    
    Args:
        trajectory: Input trajectory.
        threshold: Minimum score threshold.
        
    Returns:
        RewardOutput with computed reward.
        
    Raises:
        ValueError: If trajectory is invalid.
    """
```

### Naming

- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

## Pre-commit

All checks run automatically on commit:

```bash
pre-commit run --all-files
```

## See Also

- [Testing Guide](testing.md)
