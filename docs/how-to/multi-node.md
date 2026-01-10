---
title: Scale to Multiple Nodes
description: Distribute training across multiple machines
---

# Scale to Multiple Nodes

Distribute training across multiple machines.

## Prerequisites

- SSH access between nodes
- Shared filesystem for checkpoints
- Fast network (InfiniBand recommended)

## Configuration

```yaml
distributed:
  world_size: 16  # Total GPUs
  tensor_parallel: 4
  pipeline_parallel: 1
  data_parallel: 4
  master_addr: node1
  master_port: 29500
```

## Launch with torchrun

```bash
# On node 1
torchrun --nnodes=2 --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=node1 --master_port=29500 \
    -m flux.cli train --config config.yaml

# On node 2
torchrun --nnodes=2 --node_rank=1 \
    --nproc_per_node=8 \
    --master_addr=node1 --master_port=29500 \
    -m flux.cli train --config config.yaml
```

## SLURM Script

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

srun python -m flux.cli train --config config.yaml
```

## Troubleshooting

### NCCL Timeouts

```bash
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
```

### Network Issues

```bash
export NCCL_SOCKET_IFNAME=eth0
```

## See Also

- [Production Deployment](../tutorials/production.md)
- [Multi-GPU Training](../tutorials/multi-gpu.md)
