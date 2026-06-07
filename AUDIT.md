# Expert audit — bugs & efficiency findings (v0.0.1 → v0.0.2)

Severity: **C**=correctness bug, **E**=efficiency, **R**=robustness.

| # | Sev | Location | Problem | Fix |
|---|-----|----------|---------|-----|
| 1 | **C** | `collectors/memory.snapshot` (default-on) | Called every step with `reset_peak=True`, **mutating the user's global CUDA peak-memory stats** (`torch.cuda.max_memory_allocated()` resets under them). Silent side effect on the user's own instrumentation. | Memory collection now **opt-in** (`collect_memory=False` default); `reset_peak` defaults **off**; documented as state-mutating. |
| 2 | **E** | `RunStore.append` | Retained every `StepRecord` in `self.steps` for the live writer → **unbounded RAM growth** over a long run (a profiler that leaks memory is self-defeating). | Live writer no longer retains; `self.steps` is populated only by `load()` for analysis. |
| 3 | **E** | `RunStore.append` | `self._fh.flush()` **every step** = a syscall on the hot path. | Batched: `flush_every` (default 200) + guaranteed flush on `close()`. |
| 4 | **E** | `StepRecord.to_dict` (via `asdict`) | `dataclasses.asdict` does a **recursive deep-copy** of every dict each step — measurable per-step cost. | Hand-written `to_json_dict()`; compact separators; empty dicts omitted (smaller files too). |
| 5 | **C** | both callbacks (DDP) | Under `DistributedDataParallel`, **every rank writes the same `run_dir`** → interleaved JSONL = corrupt file. | Rank-aware: `only_rank_zero=True` default disables non-zero ranks; else run_dir is namespaced per rank. |
| 6 | **C** | `integrations/lightning` | Manual optimization (`automatic_optimization=False`) never fires `on_before_backward` → the **whole step gets mislabeled as `optimizer`**. | Track whether forward was marked; fall back to a single `compute` phase if not. |
| 7 | **E** | `integrations/lightning` | `float(loss.detach())` every step forces a **device→host sync**, serializing the pipeline just to log loss. | Behind `log_loss` flag (default on, documented); skipped cleanly on failure. |
| 8 | **R** | `RunStore.load` | `read_text().splitlines()` loads the whole file as one string **plus** a full list copy; a crash mid-write leaves a truncated final line that **throws** on parse. | Streamed line-by-line; corrupt trailing line skipped (crash-resilient). |
| 9 | **R** | scalars (convergence) | `json` emits bare `NaN`/`Infinity` for divergent loss — invalid per JSON spec (breaks external readers). Needed *intact* for divergence detection. | Round-trip verified within our loader; documented. (Standards-safe export deferred to HTML/parquet exporter.) |

Net: lower per-step overhead, bounded memory, crash- and DDP-safe, no hidden mutation of the user's CUDA state. Verified by `tests/test_overhead.py` (microbenchmark) and expanded edge-case tests.

## Numerical precision (v0.0.2 → v0.0.3)

| # | Sev | Problem | Fix |
|---|-----|---------|-----|
| P1 | **C** | `perf_counter()` returns float seconds with a large epoch; subtracting two large doubles to measure a sub-µs phase loses low-order bits (catastrophic cancellation). | Clock is now integer **`perf_counter_ns`**; phases accumulate as integers; single exact ns→s division at `end_step`. Verified: 123 ns recovered exactly under a 9.8e15 ns epoch. |
| P2 | **C** | Repeated marks of one phase (grad accumulation) summed floats → drift. | Integer-ns accumulation: 1000×7 ns == 7000 ns exactly. |
| P3 | **C** | `mean_step_time` and `phase_fractions` came from *independent* naive `sum()`s → mutually inconsistent low bits; fractions need not sum to 1. | `math.fsum` everywhere + **one shared grand total**. Fractions sum to 1 within 1e-12; `mean*n == total`. |
| P4 | **E/C** | Naive `sum()` over millions of small step times drifts. | `math.fsum` reductions. 1e6 tiny values exact to 1e-15. |
| P5 | **+** | CV alone misses occasional stragglers. | Added precise linear-interpolation **p50/p95** percentiles + `p95/median` tail signal to the jitter rule and report. |

Verified by `tests/test_precision.py`. Overhead unchanged (~0.9 µs/step pure, ~2.9 µs/step incl disk) — integer arithmetic is no slower than float.
