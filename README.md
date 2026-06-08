<div align="center">

# pytscope

**See *why* your training is slow — not just that it is.**

[![CI](https://github.com/Sumu004/pytscope/actions/workflows/ci.yml/badge.svg)](https://github.com/Sumu004/pytscope/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/pytscope.svg)](https://pypi.org/project/pytscope/)
[![Python versions](https://img.shields.io/pypi/pyversions/pytscope.svg)](https://pypi.org/project/pytscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Install](#install) · [Quickstart](#quickstart) · [What it finds](#what-it-finds) · [Docs](#documentation)

</div>

## What it does

`pytscope` watches your training loop, then tells you in plain language what's
slowing it down and how to fix it — not just a wall of numbers.

```
● TIMING — 95 steps · 23.1 ms/step · 43.3 steps/s
  data       ████████████████░░░░░░░░░░░░░░  52.0%
  forward    █████████░░░░░░░░░░░░░░░░░░░░░  17.3%
  backward   ██████████████░░░░░░░░░░░░░░░░  26.0%
  optimizer  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   4.3%

● FINDINGS (1)
  [HIGH] Input pipeline is a bottleneck  (TIMING.DATALOADER_BOUND)
        52% of step time is spent fetching data. The accelerator is
        stalling on the dataloader.
        -> Raise DataLoader num_workers, set persistent_workers=True
           and pin_memory=True, or move heavy transforms off the hot path.
```

In a real terminal each `●` lights up red, amber, or green — a clean,
glanceable "hardware panel" view of your run.

## Install

```bash
pip install pytscope
```

Pure Python, no required dependencies. PyTorch/Lightning/HF integrations are
optional extras:

```bash
pip install "pytscope[torch,lightning,huggingface]"
```

## Quickstart

**No changes to your loop (recommended):**

```python
from pytscope.auto import AutoProfiler

prof = AutoProfiler("runs/exp1", model, optimizer, warmup=10)
prof.start()
for x, y in loader:                  # <- your loop, untouched
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
prof.finish()
```

**Then read the report:**

```bash
pytscope analyze runs/exp1
pytscope diff runs/exp1 runs/exp2     # compare two runs
```

Prefer manual control, or use Lightning / Hugging Face? See the
[usage guide](docs/usage.md) for those paths.

## What it finds

One aligned timeline, six lenses — each backed by a real, tested analyzer:

| Lens | What it catches |
|------|-----------------|
| **Timing** | Dataloader stalls, slow backward/optimizer, step-time jitter |
| **Memory** | Peak usage, fragmentation, leaks |
| **Convergence** | Loss/grad-norm divergence, spikes |
| **Cross-signal** | Problems that only show up when several signals spike *together* |
| **Distributed** | Stragglers, load imbalance, pipeline bubbles, exposed communication |
| **Efficiency budget** | Where every second of wall time goes, anchored to MFU |

A few examples of what the report looks like:

<details>
<summary><b>A straggler in a 4-rank run</b></summary>

```
● DISTRIBUTED — 4 ranks, 60 aligned steps
  wall lost to imbalance 18.6% · median sync skew 4.7 ms/step
    rank 2:  12.0 ms ·  99% (z=+13.4)  <- straggler

● FINDINGS (1)
  [HIGH] Rank 2 is a persistent straggler  (DIST.STRAGGLER)
        Rank 2 is the slowest rank in 99% of steps (expected 25% by
        chance) — every other rank waits for it on the all-reduce.
        -> Check rank 2's hardware: thermal throttling, a slower GPU,
           NUMA placement, or an unbalanced data shard.
```

Reproduce with real multi-process gloo (CPU, no GPU needed):
```bash
pip install "pytscope[torch]"
python examples/ddp_gloo.py --ranks 4 --straggler-rank 2 && pytscope analyze runs/ddp_gloo
```
</details>

<details>
<summary><b>A finding no single-axis tool can make</b></summary>

```
● FINDINGS (1)
  [HIGH] Correlated instability at steps 70–72  (CROSS.CORRELATED_INSTABILITY)
        3 independent signals spike together (loss, grad_norm, step_time).
        That co-occurrence is strong evidence of a real optimization
        event — not noise.
        -> Check your LR schedule and gradient clipping around steps 70–72.
```

Most tools watch one signal. `pytscope` watches them on the same clock and
flags when several move together — that's usually the real story.
```bash
python examples/cross_signal.py && pytscope analyze runs/cross
```
</details>

<details>
<summary><b>A training efficiency budget, anchored to hardware peak (MFU)</b></summary>

```
● EFFICIENCY BUDGET — wall-time decomposition
  MFU 38.0%  ·  useful compute 38.0% of 142.00s wall
  useful_compute    ███████████░░░░░░░░░░░░░░░░░░  38.0%
  data_stall        ████████░░░░░░░░░░░░░░░░░░░░░  27.0%  (recoverable)
  communication     █████░░░░░░░░░░░░░░░░░░░░░░░░  19.0%  (recoverable)

  [HIGH] MFU is 38% — 62% of wall is recoverable  (EFFICIENCY.LOW_MFU)
        Biggest win: data_stall at 27% of wall.
        -> Start there: raise num_workers, persistent_workers, prefetch.
```

Every line is exact — they sum to your measured wall time, no fudge factor.
```bash
python examples/efficiency_mfu.py && pytscope analyze runs/mfu
```
</details>

<details>
<summary><b>Exposed communication — the metric that decides large-scale efficiency</b></summary>

```
● COMMUNICATION OVERLAP — from kernel trace
  comm 36.0 ms · overlapped 67% · exposed 12.0 ms (20% of wall)

  [HIGH] Communication is not overlapped with compute  (DIST.EXPOSED_COMM)
        Only 67% of your all-reduce is hidden behind compute.
        -> Try DDP gradient bucketing, or increase per-GPU compute so
           backward hides the all-reduce.
```

```bash
pytscope analyze runs/job --trace trace.json   # from torch.profiler
```
</details>

## Try it now (no GPU, no setup)

```bash
python examples/manual_loop.py  && pytscope analyze runs/demo   # timing
python examples/cross_signal.py && pytscope analyze runs/cross  # cross-signal
```

With real PyTorch (CUDA / Apple MPS / CPU, auto-detected):

```bash
pip install "pytscope[torch]"
python examples/pytorch_real.py --leak && pytscope analyze runs/pytorch   # catches a memory leak
```

## How it works

<div align="center">
<img src="docs/figures/architecture.png" alt="pytscope pipeline: training loop → collectors → one aligned timeline → analyzers → diagnosis engine → report" width="520">
</div>

*Figure source: [`docs/figures/architecture.tex`](docs/figures/architecture.tex) (TikZ — rebuild with `pdflatex` + ImageMagick, see the file header).*

Pure-stdlib core, ~3 µs/step overhead — small enough to leave on by default.
Adding a new diagnosis rule is one decorated function.

## Status

Validated on real multi-GPU NCCL hardware (2× T4): straggler detection and
exposed-communication analysis both confirmed on real runs — see the
[full report](docs/validation-runs/2026-06-08-kaggle-2xT4/RESULTS.md) and
[validation matrix](docs/VALIDATION.md). DDP is first-class today;
FSDP / tensor parallelism are on the roadmap.

## Documentation

- [Usage guide](docs/usage.md) — install, instrument, and the CLI
- [Architecture](docs/architecture.md) — the one-timeline design
- [Diagnostics reference](docs/diagnostics.md) — every finding and its fix
- [Validation](docs/VALIDATION.md) — what's proven, and how to reproduce it

## Contributing

Contributions are welcome — adding a diagnosis rule is the easiest first PR.
See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Releases follow [RELEASING.md](RELEASING.md).

## License

[MIT](LICENSE) © 2026 Sumukh Chaluvaraju
