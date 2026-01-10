---
title: Development Setup
description: Set up your development environment
---

# Development Setup

Set up Flux for local development.

## Prerequisites

- Python 3.10+
- CUDA 12.0+ (optional for GPU development)
- Git

## Clone Repository

```bash
git clone https://github.com/flux-team/flux.git
cd flux
```

## Create Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
```

## Install Dependencies

```bash
# Dev dependencies
pip install -e ".[dev,test,docs]"

# Pre-commit hooks
pre-commit install
```

## Verify Installation

```bash
# Run tests
pytest tests/unit/ -v

# Run linting
ruff check .

# Type checking
mypy flux/
```

## Development Workflow

1. Create a branch: `git checkout -b feature/my-feature`
2. Make changes
3. Run checks: `ruff check . && pytest`
4. Commit: `git commit -m "feat: add feature"`
5. Push and create PR

## See Also

- [Code Style](code-style.md)
- [Testing Guide](testing.md)
