"""Distributed training helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedState:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0


def init_distributed() -> DistributedState:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if enabled and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return DistributedState(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def is_main_process(state: DistributedState) -> bool:
    return state.rank == 0


def barrier(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(value: float, state: DistributedState, device: torch.device) -> float:
    if not state.enabled or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= state.world_size
    return float(tensor.item())
