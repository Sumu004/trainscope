# Usage guide

## Install

```bash
pip install pytscope                              # core (pure-stdlib)
pip install "pytscope[torch,lightning,huggingface]"  # framework integrations
pip install "pytscope[dev]"                       # tests + lint + build tooling
```

The core has **no runtime dependencies**. `torch`, `lightning`, and
`transformers` are optional extras, imported lazily only when you use an
integration or capture device memory.

## Automatic instrumentation (zero loop changes)

`AutoProfiler` captures the whole phase timeline without any `mark()` calls. Wrap
the model + optimizer once and leave your loop exactly as it is:

```python
from pytscope.auto import AutoProfiler

prof = AutoProfiler("runs/exp1", model, optimizer, warmup=10)
prof.start()
for x, y in loader:
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
    prof.log(loss=loss.item())   # optional — records the loss signal
prof.finish()
```

It works by registering a forward pre-hook + forward hook on the model and
wrapping `optimizer.step` (the step boundary). The gap between steps becomes
`data`; forward/backward/optimizer are timed from the hooks. With
`capture_comm=True` (default), **synchronous** `torch.distributed` collectives
are timed into a `comm` phase; asynchronous (overlapped) collectives — e.g.
inside `DistributedDataParallel` — are intentionally left inside backward, since
their wall-cost is genuinely hidden there. For **distributed** runs pass
`distributed=True` (records every rank). Limitation: one forward/backward per
optimizer step; for gradient accumulation use the manual `Profiler` below.

## Instrument a manual loop

```python
from pytscope import Profiler

prof = Profiler("runs/exp1", warmup=10)   # first 10 steps excluded from stats
prof.start()
for batch in prof.iter_data(loader):       # times the data fetch (DATA phase)
    with prof.step():                      # one aligned StepRecord per iteration
        out = model(batch)
        loss = loss_fn(out)
        prof.mark("forward")               # close FORWARD, open the next phase
        loss.backward()
        prof.mark("backward")
        opt.step()
        opt.zero_grad()
        prof.mark("optimizer")
        prof.log(loss=loss.item())         # scalars land on the same timeline
prof.finish()                              # flush + write run.json provenance
```

Key primitives:

| Call | What it does |
|------|--------------|
| `Profiler(run_dir, warmup=…)` | Create a run; `warmup` steps are recorded but excluded from summaries. |
| `prof.iter_data(iterable)` | Wraps your loader so time spent waiting on it is attributed to `DATA`. |
| `prof.step()` | Context manager delimiting one training step. |
| `prof.mark(phase)` | Closes the current phase and opens the next (`forward`, `backward`, `optimizer`, …). |
| `prof.log(**scalars)` | Attaches scalars (loss, grad_norm, lr, …) to the current step. |
| `prof.finish()` | Idempotent; flushes the store and writes environment provenance. |

`Profiler` is **DDP-aware**: on non-zero ranks it no-ops so multiple processes
never corrupt the same run directory.

## Lightning (one line)

```python
import lightning.pytorch as pl
from pytscope.integrations.lightning import PytscopeCallback

trainer = pl.Trainer(callbacks=[PytscopeCallback("runs/exp1")])
```

## Hugging Face `Trainer` (one line)

```python
from transformers import Trainer
from pytscope.integrations.huggingface import PytscopeCallback

trainer = Trainer(..., callbacks=[PytscopeCallback("runs/exp1")])
```

## Distributed (data-parallel) runs

Set `distributed=True` so **every** rank records into `run_dir/rank{k}/`, and
wrap the gradient all-reduce in `comm()` so communication is separated from
compute:

```python
prof = Profiler("runs/ddp", distributed=True, warmup=5)
prof.start()
for batch in loader:
    with prof.step():
        loss = loss_fn(model(batch)); loss.backward()
        prof.mark("backward")
        with prof.comm():                      # attributed to the `comm` phase
            for p in model.parameters():
                dist.all_reduce(p.grad); p.grad /= world_size
        opt.step(); opt.zero_grad(); prof.mark("optimizer")
prof.finish()
```

Then `pytscope analyze runs/ddp` auto-detects the multi-rank layout and reports
critical-path wall loss, the per-rank straggler table, and communication share.
See `examples/ddp_gloo.py` for a complete runnable CPU example (real gloo
multi-process all-reduce, no GPU needed).

## Efficiency budget & MFU

To anchor the wall-time budget to Model FLOPs Utilization, let `AutoProfiler`
count FLOPs from the first batch:

```python
prof = AutoProfiler("runs/exp", model, optimizer, measure_flops=True)
# peak is auto-detected on a recognized GPU; otherwise pass peak_flops=…
```

Then `pytscope analyze runs/exp` prints the budget and MFU. You can also supply
the anchor at analysis time:

```bash
pytscope analyze runs/exp --flops-per-step 6.2e12 --peak-tflops 312
```

The decomposition (useful compute / overhead / data / communication / other)
sums exactly to the attributed wall; recoverable lines are ranked by the seconds
they'd win back. The peak table is approximate spec peak — override with
`--peak-tflops` when precision matters.

## Exposed-communication analysis (from a kernel trace)

Per-step phase timing can't tell you whether your all-reduce overlaps compute —
that needs the kernel timeline. Capture a `torch.profiler` trace and hand it to
pytscope:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        train_one_step()
prof.export_chrome_trace("trace.json")
```

```bash
pytscope analyze runs/job --trace trace.json
# or drop the file in the run dir as trace.json[.gz] and it's auto-detected
```

pytscope classifies NCCL collective kernels vs compute kernels and reports the
overlapped/exposed split and overlap efficiency. (Real overlap numbers require a
multi-GPU trace; the interval math itself is exact and `examples/exposed_comm.py`
demonstrates it with a synthetic trace.)

## Analyze a run

```bash
pytscope analyze runs/exp1
```

Prints the step-time breakdown, per-vertical summaries, and the ranked findings.

## Compare two runs (reproducibility / drift)

```bash
pytscope diff runs/exp1 runs/exp2
```

Diffs provenance (Python/torch/CUDA/cuDNN, seeds, determinism flags), config,
and outcomes; diagnoses likely sources of nondeterminism; and reports the first
step at which the two runs diverge.

## Where data lives

Each run is a directory:

```
runs/exp1/
  steps.jsonl   # one StepRecord per line (append-only, streamed)
  run.json      # environment + provenance captured at finish()
```

`steps.jsonl` is plain newline-delimited JSON — easy to load yourself or feed to
another tool. Corrupt or partial trailing lines are skipped on load.
