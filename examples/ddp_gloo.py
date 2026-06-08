"""Real distributed-data-parallel run with an injectable straggler.

Two launch modes, same code:

* **Laptop, no GPU** — spawns N processes on the CPU ``gloo`` backend:

      python examples/ddp_gloo.py --ranks 4 --steps 80 --straggler-rank 2
      pytscope analyze runs/ddp_gloo

* **Real multi-GPU** — launched by ``torchrun``, uses the ``nccl`` backend on
  CUDA (this is the path the validation protocol in docs/VALIDATION.md uses):

      torchrun --nproc_per_node=2 examples/ddp_gloo.py --steps 200 --straggler-rank 1
      pytscope analyze runs/ddp_gloo

Each rank records its own timeline; the gradient all-reduce is wrapped in
``prof.comm()`` so the analyzer separates communication from compute. Rank
``--straggler-rank`` does extra compute each step, so pytscope's multi-rank
critical-path analysis should identify it as a persistent straggler.
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

from pytscope import Profiler


def _run_rank(
    rank: int, world_size: int, args, backend: str, clean_first: bool = False
) -> None:
    if clean_first:
        shutil.rmtree(args.run_dir, ignore_errors=True)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # Barrier after init guarantees rank 0's cleanup precedes any rank's start().
    dist.barrier()
    use_cuda = backend == "nccl" and torch.cuda.is_available()
    device = (
        torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        if use_cuda
        else (torch.device("cpu"))
    )
    if use_cuda:
        torch.cuda.set_device(device)
    torch.manual_seed(1234 + rank)

    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    prof = Profiler(
        args.run_dir, name="ddp", distributed=True, warmup=args.warmup, sync=use_cuda
    )
    prof.start()
    for _ in range(args.steps + args.warmup):
        x = torch.randn(args.batch, 256, device=device)
        y = torch.randn(args.batch, 256, device=device)
        with prof.step():
            loss = loss_fn(model(x), y)
            prof.mark("forward")
            opt.zero_grad()
            loss.backward()
            if rank == args.straggler_rank:
                busy = torch.randn(512, 512, device=device)
                for _ in range(args.straggler_work):
                    busy = busy @ busy * 1e-4 + 0.1
            prof.mark("backward")
            with prof.comm():
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world_size
            opt.step()
            prof.mark("optimizer")
            prof.log(loss=loss.item())
    prof.finish()
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"Done ({backend}, {world_size} ranks) → {args.run_dir}")
        print(f"Now run:  pytscope analyze {args.run_dir}")


def _spawn_worker(rank: int, args) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.ranks)
    _run_rank(rank, args.ranks, args, backend="gloo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=int, default=4, help="processes (CPU/gloo mode)")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--straggler-rank", type=int, default=2, help="-1 to disable")
    ap.add_argument("--straggler-work", type=int, default=12)
    ap.add_argument("--run-dir", default="runs/ddp_gloo")
    args = ap.parse_args()

    # Launched by torchrun? Then RANK/WORLD_SIZE are set per process: run one rank
    # with nccl (GPU) or gloo (CPU). Otherwise spawn `--ranks` gloo processes.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        _run_rank(rank, world_size, args, backend=backend, clean_first=(rank == 0))
    else:
        shutil.rmtree(args.run_dir, ignore_errors=True)
        t0 = time.time()
        mp.spawn(_spawn_worker, args=(args,), nprocs=args.ranks, join=True)
        print(f"({time.time() - t0:.1f}s wall)")


if __name__ == "__main__":
    main()
