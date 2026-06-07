"""Real distributed-data-parallel run on the CPU `gloo` backend.

This spawns N actual OS processes that form a `torch.distributed` process group
and train a tiny model with **real gradient all-reduce** — a genuine distributed
system you can run on a laptop with no GPU. Each rank records its own timeline
with trainscope; the all-reduce is wrapped in ``prof.comm()`` so the analyzer can
separate communication from compute.

Pass ``--straggler-rank K`` to make rank K spend extra compute each step; then
``trainscope analyze`` will identify it as a persistent straggler from the
multi-rank critical path.

Usage::

    pip install -e ".[torch]"
    python examples/ddp_gloo.py --ranks 4 --steps 80 --straggler-rank 2
    trainscope analyze runs/ddp_gloo

Requires only CPU PyTorch.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from trainscope import Profiler


def _worker(rank: int, args) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.ranks)
    dist.init_process_group("gloo", rank=rank, world_size=args.ranks)
    torch.manual_seed(1234 + rank)

    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    prof = Profiler(args.run_dir, name="ddp_gloo", distributed=True, warmup=args.warmup)
    prof.start()
    for _ in range(args.steps + args.warmup):
        x = torch.randn(args.batch, 256)
        y = torch.randn(args.batch, 256)
        with prof.step():
            out = model(x)
            loss = loss_fn(out, y)
            prof.mark("forward")
            opt.zero_grad()
            loss.backward()
            # Injected straggler: this rank does extra (wasted) compute, so it
            # consistently reaches the all-reduce barrier last.
            if rank == args.straggler_rank:
                _busy = torch.randn(512, 512)
                for _ in range(args.straggler_work):
                    _busy = _busy @ _busy * 1e-4 + 0.1
            prof.mark("backward")
            # Real gradient all-reduce across ranks — the synchronization point.
            with prof.comm():
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= args.ranks
            opt.step()
            prof.mark("optimizer")
            prof.log(loss=loss.item())
    prof.finish()
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"Done. {args.ranks} ranks → {args.run_dir}")
        print(f"Now run:  trainscope analyze {args.run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--straggler-rank", type=int, default=2, help="-1 to disable")
    ap.add_argument("--straggler-work", type=int, default=12)
    ap.add_argument("--run-dir", default="runs/ddp_gloo")
    args = ap.parse_args()

    shutil.rmtree(args.run_dir, ignore_errors=True)
    t0 = time.time()
    mp.spawn(_worker, args=(args,), nprocs=args.ranks, join=True)
    print(f"({time.time() - t0:.1f}s wall)")


if __name__ == "__main__":
    main()
