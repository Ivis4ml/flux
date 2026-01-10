---
title: Installation
description: Install Flux and its dependencies
---

# Installation

This guide covers installing Flux and all required dependencies.

## Requirements

### Hardware

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| GPU | RTX 3090 (24GB) | H100 (80GB) |
| GPU Memory | 16GB | 80GB |
| System RAM | 32GB | 128GB |
| Storage | 50GB | 500GB SSD |

### Software

| Dependency | Version | Notes |
|:-----------|:--------|:------|
| Python | 3.10+ | 3.11 recommended |
| CUDA | 12.0+ | 12.4 recommended |
| PyTorch | 2.0+ | Auto-installed |
| SGLang | 0.3+ | For inference |

---

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Basic installation
pip install flux-rlhf

# With all optional dependencies
pip install "flux-rlhf[all]"

# Verify installation
python -c "import flux; print(flux.__version__)"
```

### Method 2: From Source

Best for development or getting the latest features.

```bash
# Clone the repository
git clone https://github.com/flux-team/flux.git
cd flux

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation
flux info
```

### Method 3: Docker

For isolated, reproducible environments.

```bash
# Pull the latest image
docker pull fluxrlhf/flux:latest

# Run with GPU support
docker run --gpus all -it \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    fluxrlhf/flux:latest

# Or use docker-compose
docker-compose up -d
```

### Method 4: Conda

```bash
# Create conda environment
conda create -n flux python=3.11 -y
conda activate flux

# Install PyTorch with CUDA
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia

# Install Flux
pip install flux-rlhf
```

---

## Installing Dependencies

### SGLang (Required for Inference)

SGLang is required for efficient LLM inference during rollout generation.

```bash
# Install SGLang
pip install "sglang[all]"

# Verify SGLang installation
python -m sglang.launch_server --help
```

### Megatron-LM (Optional, for Large Scale)

For distributed training with 3D parallelism:

```bash
# Clone Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Install
pip install -e .

# Install Apex for fused kernels (recommended)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
```

### Flash Attention (Recommended)

For faster attention computation:

```bash
pip install flash-attn --no-build-isolation
```

---

## Verify Installation

Run the verification script to check all components:

```bash
flux info
```

Expected output:

```
Flux v0.1.0
============
Python:     3.11.5
PyTorch:    2.2.0
CUDA:       12.4
GPUs:       8x NVIDIA H100 80GB HBM3
SGLang:     0.3.0
Megatron:   Available

All checks passed!
```

Or verify programmatically:

```python
import flux
from flux.utils import check_installation

# Print version
print(f"Flux version: {flux.__version__}")

# Run all checks
status = check_installation()
print(f"Installation status: {'OK' if status.ok else 'FAILED'}")
for check in status.checks:
    print(f"  {check.name}: {check.status}")
```

---

## Troubleshooting

### Common Issues

??? failure "CUDA not found"

    ```
    RuntimeError: CUDA not available
    ```

    **Solution**: Ensure CUDA is properly installed and `nvidia-smi` works.

    ```bash
    # Check CUDA
    nvidia-smi

    # Check PyTorch CUDA
    python -c "import torch; print(torch.cuda.is_available())"
    ```

??? failure "SGLang server won't start"

    ```
    Error: Failed to load model
    ```

    **Solution**: Ensure sufficient GPU memory and correct model path.

    ```bash
    # Check available memory
    nvidia-smi --query-gpu=memory.free --format=csv

    # Try with smaller model
    python -m sglang.launch_server --model-path Qwen/Qwen3-0.5B --port 8000
    ```

??? failure "Import errors"

    ```
    ModuleNotFoundError: No module named 'flux'
    ```

    **Solution**: Ensure you're in the correct environment and Flux is installed.

    ```bash
    # Check environment
    which python

    # Reinstall
    pip uninstall flux-rlhf
    pip install flux-rlhf
    ```

??? failure "Version conflicts"

    ```
    ERROR: Cannot install flux-rlhf because these package versions have conflicting dependencies
    ```

    **Solution**: Use a fresh virtual environment.

    ```bash
    python -m venv fresh_env
    source fresh_env/bin/activate
    pip install flux-rlhf
    ```

---

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

# Install CUDA (if not present)
# Follow NVIDIA's official guide for your distribution
```

### macOS

!!! warning "Limited GPU Support"

    macOS is supported for development but GPU training requires NVIDIA GPUs (Linux).

    ```bash
    # Install for CPU-only development
    pip install flux-rlhf
    ```

### Windows

!!! warning "WSL2 Recommended"

    We recommend using WSL2 for Windows development.

    ```bash
    # In WSL2 Ubuntu
    pip install flux-rlhf
    ```

---

## Development Installation

For contributing to Flux:

```bash
# Clone with submodules
git clone --recursive https://github.com/flux-team/flux.git
cd flux

# Install dev dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check .
black --check .
mypy flux/
```

---

## Next Steps

Installation complete! Continue to the [Quick Start](quickstart.md) guide.
