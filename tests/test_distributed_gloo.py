"""End-to-end proof on a REAL distributed system (CPU gloo, multi-process).

Spawns actual OS processes that form a torch.distributed group and do real
gradient all-reduce, then checks pytscope identifies the injected straggler
from the multi-rank critical path. Skipped if torch is unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pytscope.analyzers.distributed import (  # noqa: E402
    analyze_distributed,
    load_multirank,
)


def _worker(rank, world_size, steps, straggler_rank, run_dir):
    import os

    import torch.distributed as dist
    import torch.nn as nn

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29577"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    from pytscope import Profiler

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(rank)
    model = nn.Linear(128, 128)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    prof = Profiler(run_dir, distributed=True, warmup=2)
    prof.start()
    for _ in range(steps + 2):
        x = torch.randn(32, 128)
        with prof.step():
            loss = model(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            prof.mark("backward")
            if rank == straggler_rank:
                b = torch.randn(256, 256)
                for _ in range(10):
                    b = b @ b * 1e-4 + 0.1
            prof.mark("compute")
            with prof.comm():
                for p in model.parameters():
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
            opt.step()
            prof.mark("optimizer")
    prof.finish()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.slow
def test_real_gloo_straggler(tmp_path):
    import torch.multiprocessing as mp

    world_size, steps, straggler = 3, 30, 1
    run_dir = str(tmp_path / "gloo")
    mp.spawn(
        _worker,
        args=(world_size, steps, straggler, run_dir),
        nprocs=world_size,
        join=True,
    )
    ranks = load_multirank(run_dir)
    assert set(ranks) == {0, 1, 2}
    d = analyze_distributed(load_multirank(run_dir))
    assert d is not None
    assert d.straggler is not None
    assert d.straggler.rank == straggler
