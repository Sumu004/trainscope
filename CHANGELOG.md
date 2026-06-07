# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **AutoProfiler is now correct on real models.** A forward re-entrancy guard
  means gradient accumulation records one step per `optimizer.step` (not per
  micro-batch), and activation checkpointing (forward recomputed during backward)
  no longer corrupts the step structure. Tested with both.
- Added `docs/VALIDATION.md` (what's validated where + a multi-GPU protocol) and
  made `examples/ddp_gloo.py` run under `torchrun` on NCCL/CUDA as well as CPU
  gloo, so the GPU validation is executable.

### Added
- **Training Efficiency Budget (MFU)** — a single accounting identity that
  decomposes attributed wall time into named line items (useful compute /
  compute overhead / data stall / communication / other) that sum **exactly** to
  wall, anchored at the top by Model FLOPs Utilization. FLOPs are counted
  automatically (`AutoProfiler(measure_flops=True)` via torch FlopCounterMode);
  peak comes from a built-in GPU table (`hardware.peak_flops_for`) or
  `--peak-tflops`. Recoverable line items are ranked by payoff; `EFFICIENCY.LOW_MFU`
  (anchored) / `EFFICIENCY.RECOVERABLE` (no anchor) rules point at the biggest win.
  CLI: `analyze --flops-per-step --peak-tflops`.
- **Exposed-communication analysis** — ingest a `torch.profiler`/Kineto trace
  (`analyze --trace`, or auto-detected `trace.json[.gz]` in the run dir) and
  compute, via exact interval arithmetic, how much collective communication
  overlaps compute vs is *exposed* (on the critical path). Reports overlap
  efficiency and a per-collective breakdown; new `DIST.EXPOSED_COMM` rule. The
  overlap math is exact (tested on synthetic traces with known answers); the
  parser is validated against a real `torch.profiler` export.
- **Automatic instrumentation** — `AutoProfiler(run_dir, model, optimizer)`
  captures the full phase timeline (data / forward / backward / optimizer, plus
  synchronous `comm`) with **zero changes to the training loop**, via PyTorch
  forward hooks + an `optimizer.step` wrapper + collective patching. The step is
  held open for post-step `log(loss=…)`. All hooks/patches are restored on
  `finish()`. Assumes one forward/backward per step (use `Profiler` for gradient
  accumulation).
- **Distributed vertical (headline)** — multi-rank critical-path analysis for
  data-parallel training. `Profiler(distributed=True)` records every rank to
  `run_dir/rank{k}/`; the analyzer aligns ranks on one timeline and computes
  critical-path wall loss, communication fraction, sync skew, and load imbalance.
- **Statistical straggler detection** — identifies a *persistent* straggler rank
  via a binomial-persistence test (is one rank consistently the critical path,
  beyond chance?), not a fixed threshold. Rules: `DIST.STRAGGLER`,
  `DIST.LOAD_IMBALANCE`, `DIST.COMM_BOUND`.
- **Pipeline-bubble analyzer** — measures achieved bubble from a per-stage
  schedule and compares it to the inherent GPipe minimum `(p-1)/(m+p-1)`, so it
  flags only *excess* bubble (`DIST.PIPELINE_BUBBLE`). Reproduces the closed form
  exactly (tested across p, m).
- **`comm()` context manager** to attribute collective time to a `comm` phase.
- **Real `examples/ddp_gloo.py`** — runs genuine multi-process gloo DDP (CPU, no
  GPU needed) with an injectable straggler; `trainscope analyze` then identifies
  it. Backed by a real multi-process integration test.
- CLI `analyze` auto-detects multi-rank run directories.

### Changed
- Packaging: version is now single-sourced from `trainscope.__version__` (Hatch
  dynamic version); `docs/` added to the sdist.

## [0.1.0] - 2026-06-06

First public beta. One telemetry backbone feeding four analysis verticals plus a
cross-signal diagnosis engine.

### Added
- **Profiler** — live, integer-nanosecond timing core with `step()` / `mark()`
  primitives, `iter_data()` dataloader timing, scalar logging, and optional
  device-memory capture. ~3 µs/step overhead.
- **Integrations** — one-line PyTorch Lightning and Hugging Face `Trainer`
  callbacks; DDP rank-aware (non-zero ranks no-op by default).
- **Timing vertical** — per-step attribution to data / forward / backward /
  optimizer with median/p95 and rules for dataloader-bound, backward-heavy,
  optimizer-heavy, and jitter.
- **Memory vertical** — CUDA + Apple MPS capture; fragmentation and
  leak/growth detection.
- **Convergence vertical** — loss trend, divergence (NaN/Inf), and robust
  local-window spike detection for loss and grad-norm.
- **Cross-signal rule** — correlates spikes across loss / grad-norm / step-time /
  memory on one aligned timeline (the headline diagnostic).
- **Reproducibility vertical** — `trainscope diff A B` compares provenance,
  config, and outcomes; diagnoses nondeterminism and finds the first divergence
  step.
- **CLI** — `trainscope analyze` and `trainscope diff`.
- Pure-stdlib core; CUDA/MPS/CPU examples; 58 tests.

[Unreleased]: https://github.com/Sumu004/trainscope/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Sumu004/trainscope/releases/tag/v0.1.0
