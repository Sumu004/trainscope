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

## Documentation

- [Usage guide](docs/usage.md) — install, instrument, and the CLI.
- [Architecture](docs/architecture.md) — the one-timeline design.
- [Diagnostics reference](docs/diagnostics.md) — every finding and its fix.

## Contributing

Contributions are welcome — adding a diagnosis rule is the most approachable
first PR. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Releases follow [RELEASING.md](RELEASING.md).

## License

[MIT](LICENSE) © 2026 Sumukh Chaluvaraju
