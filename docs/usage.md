# Usage guide

## Install

```bash
pip install trainscope                              # core (pure-stdlib)
pip install "trainscope[torch,lightning,huggingface]"  # framework integrations
pip install "trainscope[dev]"                       # tests + lint + build tooling
```

The core has **no runtime dependencies**. `torch`, `lightning`, and
`transformers` are optional extras, imported lazily only when you use an
integration or capture device memory.

## Instrument a manual loop

```python
from trainscope import Profiler

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
from trainscope.integrations.lightning import TrainScopeCallback

trainer = pl.Trainer(callbacks=[TrainScopeCallback("runs/exp1")])
```

## Hugging Face `Trainer` (one line)

```python
from transformers import Trainer
from trainscope.integrations.huggingface import TrainScopeCallback

trainer = Trainer(..., callbacks=[TrainScopeCallback("runs/exp1")])
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

Then `trainscope analyze runs/ddp` auto-detects the multi-rank layout and reports
critical-path wall loss, the per-rank straggler table, and communication share.
See `examples/ddp_gloo.py` for a complete runnable CPU example (real gloo
multi-process all-reduce, no GPU needed).

## Analyze a run

```bash
trainscope analyze runs/exp1
```

Prints the step-time breakdown, per-vertical summaries, and the ranked findings.

## Compare two runs (reproducibility / drift)

```bash
trainscope diff runs/exp1 runs/exp2
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
