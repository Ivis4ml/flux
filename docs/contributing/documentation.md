---
title: Documentation
description: How to contribute to docs
---

# Documentation

Guide for contributing to Flux documentation.

## Running Locally

```bash
# Install deps
pip install -r docs/requirements.txt

# Serve locally
mkdocs serve
# Open http://localhost:8000
```

## File Structure

```
docs/
├── index.md              # Landing page
├── getting-started/      # Getting started guides
├── tutorials/            # Step-by-step tutorials
├── concepts/             # Core concepts
├── algorithms/           # Algorithm guides
├── configuration/        # Config reference
├── api/                  # API reference
├── how-to/               # Task-oriented guides
├── design/               # Design docs
└── contributing/         # This section
```

## Writing Guidelines

### Use Clear Headings

```markdown
# Page Title
## Section
### Subsection
```

### Code Examples

```markdown
\`\`\`python
from flux import FluxConfig
config = FluxConfig(model_path="Qwen/Qwen3-8B")
\`\`\`
```

### Admonitions

```markdown
!!! note "Title"
    Content here

!!! warning
    Important warning
```

### Links

```markdown
[Link text](../path/to/page.md)
```

## See Also

- [MkDocs Material Docs](https://squidfunk.github.io/mkdocs-material/)
