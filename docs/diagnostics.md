# Diagnostics reference

Every finding pytscope can emit, what triggers it, and what to do. Findings are
ranked by severity (`high` > `med` > `low`). A wrong or noisy finding is a bug —
please [report it](../CONTRIBUTING.md#reporting-bugs--requesting-features) with
the run details.

## Timing

### `TIMING.DATALOADER_BOUND`
The `DATA` phase dominates step time — the accelerator is stalling on input.
**Fix:** raise `DataLoader` `num_workers`, set `persistent_workers=True` and
`pin_memory=True`, prefetch, or move heavy transforms off the hot path.

### `TIMING.BACKWARD_HEAVY`
Backward is disproportionately expensive relative to forward (fires above ~2.5×;
high tier above ~3.5×). **Fix:** check for retained graphs, unnecessary
`requires_grad`, activation checkpointing trade-offs, or gradient hooks.

### `TIMING.OPTIMIZER_HEAVY`
The optimizer step takes an outsized share. **Fix:** review optimizer choice and
state size, `foreach`/fused implementations, and whether per-parameter Python
loops are involved.

### `TIMING.JITTER`
Step time is highly variable (large p95/p50 ratio / coefficient of variation).
**Fix:** look for periodic stalls — logging, checkpointing, eval inside the train
loop, host–device syncs, or dataloader starvation.

## Memory

### `MEMORY.GROWTH`
Allocated device memory trends upward across steps (robust slope), i.e. a likely
leak. **Fix:** detach tensors kept for logging (`.item()`/`.detach()`), avoid
accumulating graph-attached tensors in lists, and check caches that grow per
step.

### `MEMORY.FRAGMENTATION`
A large gap between reserved and allocated memory, gated to runs whose peak
allocation is meaningful (≥ 64 MB) to avoid false positives on tiny/idle runs.
**Fix:** consider `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, stabilize
batch/shape variation, or call the allocator's empty-cache sparingly.

> Note: memory attribution is most accurate on CUDA, which exposes true in-step
> peaks. On Apple MPS, pytscope samples resident memory at the step boundary,
> so fragmentation/peak figures are approximate.

## Convergence

### `CONVERGENCE.DIVERGED`
Loss became NaN/Inf, or blew up — training diverged. **Fix:** lower the learning
rate, add/strengthen gradient clipping, check for bad batches or numerical issues
(mixed precision overflow, log of zero, etc.).

### `CONVERGENCE.NO_PROGRESS`
Loss shows no meaningful downward trend over the analyzed window. **Fix:** verify
the LR is non-trivial, the data/labels are correct, the loss is wired up, and the
model has capacity / isn't frozen.

## Efficiency budget

### `EFFICIENCY.LOW_MFU` / `EFFICIENCY.RECOVERABLE`
From the Training Efficiency Budget — a decomposition of attributed wall time
into line items that sum exactly to wall. With a FLOPs+peak anchor it reports
**MFU** (Model FLOPs Utilization) and fires `EFFICIENCY.LOW_MFU` when utilization
is low; without an anchor it fires `EFFICIENCY.RECOVERABLE` when a recoverable
line dominates. The finding names the **largest recoverable line** (compute
overhead / data stall / communication / other) and its fix. **Fix:** start with
the top line — AMP/`torch.compile` for compute overhead, more dataloader workers
for data stall, comm overlap for communication. Anchor MFU with
`AutoProfiler(measure_flops=True)` or `analyze --flops-per-step --peak-tflops`.

## Distributed (the headline)

These read the multi-rank critical-path summary built by
`Profiler(distributed=True)`. They are not threshold-only heuristics — the
straggler rule uses a statistical-persistence test, and the pipeline rule
subtracts the inherent GPipe bubble.

### `DIST.STRAGGLER`
One rank is the critical path (slowest to the all-reduce barrier) in
significantly more steps than chance (binomial z-test) **and** is materially
slower than the median rank. Synchronous training makes every other rank wait
for it. **Fix:** investigate that rank's device/host — thermal throttling, a
slower GPU, NUMA/host placement, or an unbalanced data shard.

### `DIST.LOAD_IMBALANCE`
Per-rank compute varies substantially but with no single persistent culprit
(e.g. several ranks alternate as the slowest). **Fix:** even out shard
sizes/sequence lengths, check for variable-cost batches, and avoid
rank-dependent branches in the step.

### `DIST.COMM_BOUND`
Collective communication is a large share of step time — the gradient all-reduce
isn't hidden behind compute. **Fix:** overlap communication with backward (DDP
bucketing / `no_sync` for accumulation), increase per-step compute, enable
gradient compression, or check interconnect bandwidth.

### `DIST.EXPOSED_COMM`
From a `torch.profiler`/Kineto kernel trace: a significant share of wall time is
*exposed* communication — collective (all-reduce/all-gather/…) time that does
**not** overlap any compute kernel, so it sits on the critical path. Overlapped
communication is free; exposed communication is wall-time waste. **Fix:** improve
compute/communication overlap — DDP gradient bucketing (`bucket_cap_mb`), avoid
`find_unused_parameters` stalls, overlap the optimizer/all-reduce, or increase
per-GPU compute so backward lasts long enough to hide the all-reduce. Pass the
trace with `pytscope analyze <run> --trace trace.json`.

### `DIST.PIPELINE_BUBBLE`
Pipeline idle time exceeds the inherent GPipe minimum `(p-1)/(m+p-1)` — i.e.
there is *excess* bubble from scheduling/imbalance, not just from your stage and
microbatch counts. **Fix:** balance per-stage compute, check for a slow stage,
increase microbatches, or use a 1F1B interleaved schedule.

## Cross-signal

### `CROSS.CORRELATED_INSTABILITY`
Two or more independent axes (loss, grad-norm, step-time, memory) spike at the
**same** step(s); consecutive such steps are grouped into one finding.
Co-occurrence across axes on a single timeline is strong evidence of a real
optimization event rather than per-axis noise. **Fix:** inspect the LR schedule,
gradient clipping, and the specific batches around the flagged steps — a
simultaneous loss + grad-norm spike usually means an update blew up (LR too high
or a bad batch).

This is the diagnosis no single-axis profiler can make, because it requires all
axes to share one clock — see [architecture](architecture.md).

## Reproducibility (`pytscope diff A B`)

`diff` is not a single finding but a structured comparison of two runs:

- **Provenance diffs** — Python/torch/CUDA/cuDNN versions, GPU, determinism
  flags, and seed-related environment variables.
- **Config & outcome diffs** — differing hyperparameters and final metrics.
- **Nondeterminism diagnosis** — flags likely causes (e.g. unset
  `PYTHONHASHSEED`, nondeterministic cuDNN, missing
  `torch.use_deterministic_algorithms`).
- **First divergence step** — the earliest step at which the two timelines'
  scalars disagree, narrowing where to look.
