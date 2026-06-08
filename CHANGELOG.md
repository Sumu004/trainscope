# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-06-08

### Changed
- **Project renamed from `trainscope` to `pytscope`.** Both `trainscope` and
  `tscope` were already taken on PyPI; `pytscope` was available. The import
  path, CLI command, PyPI distribution name, and GitHub repository
  (`Sumu004/pytscope`) are now all `pytscope`. No functional changes —
  this release exists solely to publish under the new name (the v0.2.0 tag
  was cut from the pre-rename tree, so its build still carried the old name).

## [0.2.0] - 2026-06-08

### Added
- **Amber-LED "hardware panel" terminal reports.** `pytscope analyze`/`diff`
  now render every section as a lit panel — a colored `●` indicator (red /
  amber / green by what it's reporting: a stalling step-time breakdown, a
  named straggler, a low MFU, …) in front of each heading — plus gradient
  meter bars rendered as lit/unlit block segments (`█`/`░`), severity-coded
  findings, and unicode sparklines for step-time and loss trends. The whole
  grammar (LED color ↔ severity ↔ bar gradient) is consistent across every
  vertical, and the run summary is condensed onto one heading line (steps ·
  ms/step · throughput · median/p95/CV) with the old double-blank-line
  padding between sections removed — same information, faster to scan.
  Pure ANSI escapes, no new deps: auto-detects whether the terminal can
  render color (honoring `NO_COLOR`/`FORCE_COLOR` and the new `--color
  {auto,always,never}` flag), and degrades to byte-identical plain text when
  piped to a file, redirected in CI, or explicitly disabled — nothing is
  ever written to disk beyond the run itself (`tests/test_cli_report.py`).

### Validated
- **Real multi-GPU NCCL run (Kaggle, 2× T4, 2026-06-08)** — closes two of the
  three "needs hardware" gaps in `docs/VALIDATION.md`:
  - **Straggler attribution: exact pass.** `DIST.STRAGGLER` named the injected
    rank correctly (z=14.1, 100% critical-path share vs 50% expected by chance,
    27.5% wall lost to imbalance).
  - **Exposed communication: directionally confirmed**, plus a genuine finding
    about the hardware itself — on PCIe-only GPU pairs (no NVLink) the
    all-reduce is link-bandwidth-bound, so absolute exposed-comm time stays
    roughly constant across batch sizes (`DIST.EXPOSED_COMM` correctly fired
    HIGH for both the small- and large-batch configs: 72% vs 62% exposed,
    overlap improving in the predicted direction). Documented as an
    interconnect-topology caveat in `docs/VALIDATION.md`.
  - Full report, raw console captures, and analysis:
    [`docs/validation-runs/2026-06-08-kaggle-2xT4/RESULTS.md`](https://github.com/Sumu004/pytscope/blob/main/docs/validation-runs/2026-06-08-kaggle-2xT4/RESULTS.md).

### Fixed
- **`examples/efficiency_mfu.py` never selected CUDA** — its device probe only
  checked `mps`/`cpu`, so on a CUDA box it silently profiled the CPU and
  reported a meaningless ~0% MFU anchored against an A100 peak it never
  touched (caught by the Kaggle validation run above). Now checks `cuda` first
  and, when on CUDA, leaves `peak_flops` unset so `AutoProfiler` looks the
  actual device up in the hardware peak table instead of hard-coding an
  A100 anchor.

### Added
- **`docs/validation-runs/`** — a ready-to-run Kaggle notebook
  (`kaggle_2xT4.ipynb`) plus step-by-step instructions that execute the full
  multi-GPU validation protocol from `docs/VALIDATION.md` (straggler
  attribution, exposed-comm overlap, MFU sanity) on **real NCCL/CUDA, for
  free**, using Kaggle's 2× T4 notebook tier — no paid GPU rental needed.

### Fixed
- **CLI crashed on Windows** (`UnicodeEncodeError: 'charmap' codec can't encode
  character 'Δ'`) whenever a report containing `Δ`/`—`/`•` was printed
  with the console in its default `cp1252` encoding — this is what every CI
  run on `windows-latest` was hitting (`pytscope diff`'s metrics table header
  uses `Δ`). `pytscope.cli.main` now re-points stdout/stderr at a UTF-8
  encoder with `errors="replace"` on entry; a no-op on platforms already UTF-8.

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
  GPU needed) with an injectable straggler; `pytscope analyze` then identifies
  it. Backed by a real multi-process integration test.
- CLI `analyze` auto-detects multi-rank run directories.

### Changed
- Packaging: version is now single-sourced from `pytscope.__version__` (Hatch
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
- **Reproducibility vertical** — `pytscope diff A B` compares provenance,
  config, and outcomes; diagnoses nondeterminism and finds the first divergence
  step.
- **CLI** — `pytscope analyze` and `pytscope diff`.
- Pure-stdlib core; CUDA/MPS/CPU examples; 58 tests.

[Unreleased]: https://github.com/Sumu004/pytscope/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Sumu004/pytscope/releases/tag/v0.2.1
[0.2.0]: https://github.com/Sumu004/pytscope/releases/tag/v0.2.0
[0.1.0]: https://github.com/Sumu004/pytscope/releases/tag/v0.1.0
