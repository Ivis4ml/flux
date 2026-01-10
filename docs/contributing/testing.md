---
title: Testing
description: How to write and run tests
---

# Testing

Guidelines for testing Flux code.

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=flux --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Specific file
pytest tests/unit/test_config.py -v
```

## Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Multi-component tests
└── e2e/           # Full training tests (slow)
```

## Writing Tests

```python
import pytest
from flux import FluxConfig

class TestFluxConfig:
    def test_default_values(self):
        config = FluxConfig(model_path="test")
        assert config.batch_size == 32
        assert config.learning_rate == 1e-6
    
    def test_validation_error(self):
        with pytest.raises(ValueError):
            FluxConfig(model_path="test", num_steps=-1)
    
    @pytest.mark.slow
    def test_large_config(self):
        # This test takes a while
        ...
```

## Markers

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.integration` - Integration tests

## See Also

- [Code Style](code-style.md)
