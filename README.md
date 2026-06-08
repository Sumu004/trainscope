# trainscope

[![CI](https://github.com/Sumu004/trainscope/actions/workflows/ci.yml/badge.svg)](https://github.com/Sumu004/trainscope/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/trainscope.svg)](https://pypi.org/project/trainscope/)
[![Python versions](https://img.shields.io/pypi/pyversions/trainscope.svg)](https://pypi.org/project/trainscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An intelligence layer for ML training — go beyond collecting metrics to
> **explaining** them.

Standard profilers hand you a 50 MB trace and leave the "so what do I change?"
to you. `trainscope` captures **timing, memory, convergence signals, and
provenance on one aligned per-step timeline**, then runs a diagnosis engine that
turns the raw numbers into ranked, actionable findings.

```
Run summary — 95 steps, 23.1 ms/step, 43.3 steps/s

Step time breakdown:
  data       ############------------------  52.0%    12.01 ms
  forward    #########---------------------  17.3%     4.00 ms
  backward   ##############----------------  26.0%     6.01 ms
  optimizer  #-----------------------------   4.3%     1.00 ms

Findings (1):
  [HIGH] Input pipeline is a bottleneck  (TIMING.DATALOADER_BOUND)
        52% of step time is spent fetching data (12.0 ms/step). The
        accelerator is stalling on the dataloader.
        -> Raise DataLoader num_workers, set persistent_workers=True and
           pin_memory=True, prefetch, or move heavy transforms off the hot path.
```

## The headline: a Training Efficiency Budget

Most profilers hand you a list of findings. trainscope also gives you a single
**accounting identity** — every second of training, decomposed into named line
items that provably sum to your measured wall time, anchored to hardware peak
(**MFU**, Model FLOPs Utilization):

```
Efficiency budget (wall-time decomposition):
  MFU 38.0%  ·  useful compute 38.0% of 142.0s wall
  useful_compute    ###########-------------------  38.0%   53.96s
  compute_overhead  #####-------------------------  16.0%   22.72s (recoverable)
  data_stall        ########----------------------  27.0%   38.34s (recoverable)
  communication     #####-------------------------  19.0%   26.98s (recoverable)

  [HIGH] MFU is 38% — 62% of wall is recoverable  (EFFICIENCY.LOW_MFU)
        Biggest recoverable line: data_stall at 27% of wall.
        -> Start with data_stall: raise num_workers, persistent_workers, prefetch.
```

Because the phase timeline partitions each step, the decomposition is **exact** —
the line items sum to wall with no fudge factor, which makes the model
falsifiable. And every recoverable line is *seconds you can win back*, so fixes
rank themselves by payoff. FLOPs are counted automatically
(`AutoProfiler(measure_flops=True)`); peak comes from a built-in GPU table or
`--peak-tflops`.

```bash
python examples/efficiency_mfu.py && trainscope analyze runs/mfu
```

## Why it's different

One backbone, four lenses. Every analyzer reads the same `StepRecord` timeline,
so findings can **cross-correlate signals no single existing tool aligns**:

| Vertical | Status |
|----------|--------|
| **Distributed** — multi-rank critical-path, straggler & comm/pipeline-bubble analysis | ✅ |
| **Timing** — attribute step time to data / fwd / bwd / optimizer | ✅ |
| **Convergence** — loss/grad-norm trend, divergence, spikes | ✅ |
| **Memory** — peak attribution, fragmentation, leak/growth | ✅ |
| **Cross-signal** — correlate spikes across all axes on one timeline | ✅ |
| **Reproducibility** — provenance capture + run-vs-run diff & drift diagnosis | ✅ |

The core is **pure-stdlib** — no heavy deps to profile your training.

### The headline: a finding no single-axis tool can make

```
Findings (1):
  [HIGH] Correlated instability at steps 70–72  (CROSS.CORRELATED_INSTABILITY)
        At steps 70–72, 3 independent axes spike simultaneously (grad_norm,
        loss, step_time): loss=3.579, grad_norm=45, step_time=25.6ms.
        Co-occurrence across axes is strong evidence of a real optimization
        event, not noise.
        -> Inspect the LR schedule, gradient clipping, and the batch around
           steps 70–72. A simultaneous loss + grad-norm spike usually means
           the update blew up (LR too high / bad batch).
```

HTA sees only timing; Cockpit only gradients; W&B only logged scalars. trainscope
sees them **on one clock** and reports the correlation. Reproduce it with
`python examples/cross_signal.py && trainscope analyze runs/cross`.

### Distributed: the straggler no single-rank profiler can name

In synchronous data-parallel training every rank waits at the gradient
all-reduce for the **slowest** rank. That idle time is pure waste, and it's
invisible to any single-rank profiler — you only see it by putting all ranks on
one timeline. trainscope does, and uses a **statistical persistence test** (not a
threshold) to tell a genuine bad node from noise:

```
Distributed — 4 ranks, 60 aligned steps:
  wall lost to imbalance 18.6% · median sync skew 4.7 ms/step
    rank 0:  10.0 ms ·   0% (z=-4.5)
    rank 2:  12.0 ms ·  99% (z=+13.4)  <- straggler

Findings:
  [HIGH] Rank 2 is a persistent straggler  (DIST.STRAGGLER)
        Rank 2 is the slowest (critical-path) rank in 99% of steps across 4
        ranks (expected 25% by chance; z=13.4) and runs 20% slower than the
        median rank. Synchronous all-reduce makes every other rank wait for it
        — 18.6% of wall time is lost to this imbalance.
        -> Investigate rank 2's device/host: thermal throttling, a slower GPU,
           NUMA placement, or an unbalanced data shard.
```

This is a **real** distributed system — reproduce it on your laptop (CPU, no GPU)
with genuine multi-process gloo all-reduce:

```bash
pip install -e ".[torch]"
python examples/ddp_gloo.py --ranks 4 --straggler-rank 2
trainscope analyze runs/ddp_gloo
```

For **pipeline parallelism**, trainscope measures the achieved bubble and compares
it to the inherent GPipe minimum `(p-1)/(m+p-1)`, so it flags only the *excess*
bubble you can actually fix — not the bubble that's just the cost of your `p`
and `m`.

### Exposed communication: the metric that decides large-scale efficiency

Gradient all-reduce *can* run concurrently with backward compute — the part that
overlaps is free, the part that doesn't is **exposed** and sits on the critical
path. trainscope ingests a `torch.profiler`/Kineto trace and computes the split
exactly (interval arithmetic over the kernel timeline):

```
Communication overlap (from kernel trace):
  comm 36.0 ms · overlapped 67% · exposed 12.0 ms (20% of wall)

  [HIGH] Communication is not overlapped with compute  (DIST.EXPOSED_COMM)
        20% of wall time is exposed communication. Only 67% of the 36.0 ms of
        communication is hidden behind compute.
        -> DDP gradient bucketing (bucket_cap_mb), overlap optimizer/all-reduce,
           or increase per-GPU compute so backward hides the all-reduce.
```

```bash
trainscope analyze runs/job --trace trace.json   # from torch.profiler
python examples/exposed_comm.py && trainscope analyze runs/trace_demo  # no GPU
```

## Overhead

Measured on `tests/test_overhead.py` (run `pytest -s`):

| Path | Cost |
|------|------|
| Pure instrumentation (begin/mark×3/end) | **~0.7 µs/step** |
| End-to-end incl. JSONL disk write | **~3 µs/step** |
| Disabled DDP rank (no-op) | **~0.06 µs/step** |

On a 50 ms training step that's **~0.006% overhead** — versus trace-dumping
profilers (Kineto/HTA) that add real overhead and emit multi-MB artifacts.
Memory bounded (live writer retains nothing); batched flushes; DDP-safe.

## Install

```bash
pip install -e ".[dev]"          # core + tests
pip install -e ".[torch,lightning,huggingface]"   # framework integrations
```

## Quickstart

**Automatic — zero changes to your loop (recommended):**

```python
from trainscope.auto import AutoProfiler

prof = AutoProfiler("runs/exp1", model, optimizer, warmup=10)
prof.start()
for x, y in loader:                           # <- your loop, untouched
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
prof.finish()
```

`AutoProfiler` registers PyTorch hooks (forward, `optimizer.step`, and
synchronous collectives) to attribute **data / forward / backward / optimizer /
comm** automatically — no `mark()` calls anywhere in your training code. All
hooks/patches are removed on `finish()`.

**Manual loop** (full control, or gradient accumulation):

```python
from trainscope import Profiler

prof = Profiler("runs/exp1", warmup=10)
prof.start()
for batch in prof.iter_data(loader):          # times data fetch
    with prof.step():
        loss = loss_fn(model(batch))
        prof.mark("forward")
        loss.backward();   prof.mark("backward")
        opt.step(); opt.zero_grad(); prof.mark("optimizer")
prof.finish()
```

**Lightning (one line):**

```python
from trainscope.integrations.lightning import TrainScopeCallback
trainer = pl.Trainer(callbacks=[TrainScopeCallback("runs/exp1")])
```

**Hugging Face (one line):**

```python
from trainscope.integrations.huggingface import TrainScopeCallback
trainer = Trainer(..., callbacks=[TrainScopeCallback("runs/exp1")])
```

**Then analyze, or compare two runs:**

```bash
trainscope analyze runs/exp1
trainscope diff runs/exp1 runs/exp2   # reproducibility / drift: why do they differ?
```

## Try the demos

No ML deps:

```bash
python examples/manual_loop.py     && trainscope analyze runs/demo    # timing
python examples/cross_signal.py    && trainscope analyze runs/cross   # cross-signal
```

Real PyTorch (CUDA / Apple MPS / CPU, auto-detected), with real device timing
and memory:

```bash
pip install -e ".[torch]"
python examples/pytorch_real.py            && trainscope analyze runs/pytorch  # healthy
python examples/pytorch_real.py --leak     && trainscope analyze runs/pytorch  # catches the leak
```

The `--leak` run reports `MEMORY.GROWTH [HIGH]` from genuinely captured device
memory. (Memory attribution is most accurate on CUDA, which exposes true in-step
peaks; on MPS we sample resident memory at the step boundary.)

## Architecture

```
training loop → collectors → RunStore (aligned timeline)
                                  ↓
              analyzers (timing | memory | convergence | repro)
                                  ↓
              diagnosis engine (ranked, cross-signal findings)
                                  ↓
                  reporters (CLI | HTML | markdown)
```

Adding a heuristic is one decorated function (`@rule`); adding a vertical is one
analyzer over the existing timeline.

## Status & validation

**v0.1, validated on real multi-GPU NCCL hardware** — straggler attribution and
exposed-comm now have a clean run on 2× T4 (Kaggle, free tier, no paid rental):
an exact pass on straggler detection (`z=14.1`, named the injected rank
correctly) and a directionally-correct exposed-comm read that also surfaced a
genuine finding about PCIe-only interconnects. MFU-on-GPU is the last gap —
unblocked (a demo bug found and fixed) with a rerun pending.
[Full report →](docs/validation-runs/2026-06-08-kaggle-2xT4/RESULTS.md) ·
[Validation matrix & protocol →](docs/VALIDATION.md)

DDP is first-class; FSDP/tensor/pipeline parallelism are not yet.

## Documentation

- [Usage guide](docs/usage.md) — install, instrument, and the CLI.
- [Architecture](docs/architecture.md) — the one-timeline design.
- [Diagnostics reference](docs/diagnostics.md) — every finding and its fix.
- [Validation](docs/VALIDATION.md) — what's proven, and the multi-GPU protocol.

## Contributing

Contributions are welcome — adding a diagnosis rule is the most approachable
first PR. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Releases follow [RELEASING.md](RELEASING.md).

## License

[MIT](LICENSE) © 2026 Sumukh Chaluvaraju
