---
title: Getting Started
description: Get up and running with Flux in minutes
---

# Getting Started with Flux

Welcome to Flux! This guide will help you get up and running with the adaptive post-training framework for LLMs.

## What is Flux?

Flux is an **adaptive post-training framework** that combines the best of synchronous and asynchronous RLHF training. Instead of forcing you to choose between stability and efficiency, Flux dynamically adjusts based on your training dynamics.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **5 min setup**

    ---

    Get Flux installed and configured in under 5 minutes

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Run your first training in 10 lines of code

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-school:{ .lg .middle } **First Training**

    ---

    Complete walkthrough of your first training run

    [:octicons-arrow-right-24: First Training](first-training.md)

-   :material-cog:{ .lg .middle } **Configuration**

    ---

    Learn the basics of Flux configuration

    [:octicons-arrow-right-24: Configuration](configuration.md)

</div>

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+**
- **CUDA 12.0+** with compatible GPU (H100/A100 recommended)
- **16GB+ GPU memory** (for 8B models)
- **SGLang** installed for inference

---

## Installation Overview

=== "pip (Recommended)"

    ```bash
    pip install flux-rlhf
    ```

=== "From Source"

    ```bash
    git clone https://github.com/flux-team/flux.git
    cd flux
    pip install -e ".[dev]"
    ```

=== "Docker"

    ```bash
    docker pull fluxrlhf/flux:latest
    docker run --gpus all -it fluxrlhf/flux
    ```

[:octicons-arrow-right-24: Full Installation Guide](installation.md)

---

## Quick Example

```python
from flux import FluxConfig, FluxTrainer

# Configure training
config = FluxConfig(
    model_path="Qwen/Qwen3-8B",
    num_steps=1000,
    batch_size=32,
    algorithm="grpo",
)

# Create trainer and run
trainer = FluxTrainer(config)
trainer.fit(prompts="data/prompts.jsonl")
```

[:octicons-arrow-right-24: Detailed Quick Start](quickstart.md)

---

## Learning Path

New to Flux? Follow this recommended learning path:

```mermaid
graph LR
    A[Installation] --> B[Quick Start]
    B --> C[First Training]
    C --> D[Configuration]
    D --> E[Tutorials]
    E --> F[Advanced Topics]
```

1. **[Installation](installation.md)** - Set up your environment
2. **[Quick Start](quickstart.md)** - Run a minimal example
3. **[First Training](first-training.md)** - Complete training walkthrough
4. **[Configuration](configuration.md)** - Understand configuration options
5. **[Tutorials](../tutorials/index.md)** - Deep-dive tutorials
6. **[Concepts](../concepts/index.md)** - Core concepts explained

---

## Need Help?

<div class="grid cards" markdown>

-   :material-forum:{ .lg .middle } **Community**

    ---

    Ask questions and get help from the community

    [:octicons-arrow-right-24: Discord](https://discord.gg/flux-rlhf)

-   :material-bug:{ .lg .middle } **Issues**

    ---

    Report bugs or request features

    [:octicons-arrow-right-24: GitHub Issues](https://github.com/flux-team/flux/issues)

-   :material-book-open-page-variant:{ .lg .middle } **FAQ**

    ---

    Common questions and answers

    [:octicons-arrow-right-24: FAQ](#faq)

</div>

---

## FAQ

??? question "What GPUs are supported?"

    Flux supports NVIDIA GPUs with CUDA 12.0+. We recommend:

    - **Development**: RTX 4090, A100 (40GB)
    - **Production**: H100 (80GB), A100 (80GB)

    Minimum 16GB VRAM for 8B models, 80GB for 70B models.

??? question "Can I use Flux with vLLM instead of SGLang?"

    Currently, Flux is optimized for SGLang integration. vLLM support is planned for future releases.

??? question "How does Flux compare to TRL?"

    TRL is excellent for simple RLHF setups. Flux is designed for:

    - Large-scale distributed training (64+ GPUs)
    - Maximum GPU utilization through adaptive async
    - Native Megatron integration for 3D parallelism

    If you're training on 1-8 GPUs, TRL may be simpler. For scale, choose Flux.

??? question "Is Flux production-ready?"

    Flux is under active development. Core features are stable, but we recommend testing thoroughly before production deployment.

---

## Next Steps

Ready to dive in? Start with the [Installation Guide](installation.md).
