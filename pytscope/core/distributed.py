"""Distributed-rank detection — so DDP runs don't corrupt one shared run file.

Order of truth: an initialized ``torch.distributed`` process group, then the
common launcher env vars. Falls back to single-process rank 0.
"""

from __future__ import annotations

import os


def get_rank() -> int:
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def get_world_size() -> int:
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


def is_main_process() -> bool:
    return get_rank() == 0
