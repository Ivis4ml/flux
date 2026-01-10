---
title: Contributing
description: How to contribute to Flux
---

# Contributing to Flux

Thank you for your interest in contributing to Flux! This guide will help you get started.

---

## Ways to Contribute

<div class="grid cards" markdown>

-   :material-bug:{ .lg .middle } **Report Bugs**

    ---

    Found a bug? Let us know!

    [:octicons-arrow-right-24: Report Bug](https://github.com/flux-team/flux/issues/new?template=bug_report.md)

-   :material-lightbulb:{ .lg .middle } **Suggest Features**

    ---

    Have an idea? We'd love to hear it!

    [:octicons-arrow-right-24: Feature Request](https://github.com/flux-team/flux/issues/new?template=feature_request.md)

-   :material-code-tags:{ .lg .middle } **Submit Code**

    ---

    Fix bugs or add features

    [:octicons-arrow-right-24: Development Guide](setup.md)

-   :material-file-document:{ .lg .middle } **Improve Docs**

    ---

    Help us improve documentation

    [:octicons-arrow-right-24: Documentation Guide](documentation.md)

</div>

---

## Quick Start

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/flux.git
cd flux
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dev dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/my-bug-fix
```

### 4. Make Changes

```bash
# Write code...
# Write tests...
# Update docs...

# Run checks
ruff check .
black .
mypy flux/
pytest
```

### 5. Submit Pull Request

```bash
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature
```

Then open a PR on GitHub!

---

## Development Guides

| Guide | Description |
|:------|:------------|
| [Development Setup](setup.md) | Detailed environment setup |
| [Code Style](code-style.md) | Coding standards and conventions |
| [Testing](testing.md) | How to write and run tests |
| [Documentation](documentation.md) | How to contribute to docs |

---

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Please be respectful and inclusive.

### In Short

- Be welcoming and inclusive
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

---

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines (`ruff check . && black --check .`)
- [ ] All tests pass (`pytest`)
- [ ] Type hints are correct (`mypy flux/`)
- [ ] Documentation is updated if needed
- [ ] Commit messages follow conventions

### PR Title Format

```
<type>: <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- test: Testing
- chore: Maintenance
```

Examples:

- `feat: add REINFORCE algorithm`
- `fix: correct staleness calculation`
- `docs: add tutorial for custom rewards`

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Docs updated
- [ ] Changelog updated (if applicable)
```

---

## Issue Guidelines

### Bug Reports

Include:

- Flux version (`flux info`)
- Python version
- GPU/CUDA info
- Minimal reproduction code
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:

- Use case description
- Proposed solution
- Alternative approaches considered
- Willingness to implement

---

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/flux-team/flux/discussions)
- **Bugs**: Open an [Issue](https://github.com/flux-team/flux/issues)
- **Chat**: Join our [Discord](https://discord.gg/flux-rlhf)

---

## Recognition

Contributors are recognized in:

- [CONTRIBUTORS.md](https://github.com/flux-team/flux/blob/main/CONTRIBUTORS.md)
- Release notes
- Documentation credits

Thank you for making Flux better!
