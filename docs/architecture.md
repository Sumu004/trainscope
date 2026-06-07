# Architecture

trainscope is built around a single idea: **put every signal on one aligned
per-step timeline, then reason across signals.** Standard tools profile one axis
(timing, or gradients, or logged scalars). Because trainscope's analyzers all
read the same `StepRecord` stream, the diagnosis engine can correlate events
that no single-axis tool can see together.

```
training loop
   │  (Profiler primitives: iter_data / step / mark / log)
   ▼
collectors ──► RunStore  (steps.jsonl, append-only, aligned timeline)
                   │
                   ▼
        analyzers  (timing | memory | convergence | repro)
                   │   each: List[StepRecord] → typed summary dataclass
                   ▼
        diagnosis engine  (@rule functions → ranked Findings)
                   │
                   ▼
            reporters  (CLI today; HTML/markdown extensible)
```

## Layers

### 1. Profiler (`profiler.py`)

The live runtime object. Timing uses **integer nanoseconds**
(`time.perf_counter_ns`) end-to-end and converts to seconds exactly once, at the
step boundary — this avoids float catastrophic cancellation on sub-millisecond
phases. It captures optional device memory (CUDA true in-step peak; Apple MPS
resident sample) and is DDP rank-aware. The clock is injectable for
deterministic tests. Overhead is ~0.7 µs/step instrumented, ~3 µs/step including
the JSONL write.

### 2. Telemetry backbone (`core/`)

- `events.py` — the `StepRecord` schema (step index, phase durations, scalars,
  memory, timestamp) and canonical phase ordering.
- `store.py` — `RunStore`, an append-only JSONL writer/reader. The live writer
  retains nothing in memory (bounded footprint), batches flushes, and tolerates
  corrupt trailing lines and a missing/corrupt `run.json` on read.
- `provenance.py` — `capture_environment()` (Python, torch, CUDA, cuDNN flags,
  GPU, seed-related env vars) for reproducibility.
- `distributed.py` — rank detection so only the main process writes.

### 3. Analyzers (`analyzers/`)

Pure functions over `List[StepRecord]` returning typed summaries. All reductions
are numerically careful: `math.fsum` for summation, two-pass variance,
linear-interpolation percentiles, and robust statistics (median/MAD,
local-window spike detection) in `stats.py`.

- **timing** → per-phase attribution, p50/p95, coefficient of variation.
- **memory** → peak, fragmentation, robust growth slope.
- **convergence** → loss/grad-norm trend, divergence (NaN/Inf), local spikes.
- **repro** → `diff_runs(a, b)` comparing provenance, config, and outcomes.
- **distributed** → aligns *multiple ranks* on one timeline and computes the
  critical path, per-rank straggler index (with a binomial-persistence test),
  communication fraction, and load imbalance.
- **pipeline** → bubble fraction from a per-stage schedule, compared to the
  inherent GPipe minimum `(p-1)/(m+p-1)`.

The distributed analyzer is the clearest expression of the "one timeline"
thesis: the wasted time it finds (faster ranks idling at the barrier for the
slowest) does not exist in any single rank's data — it is *only* visible once
the ranks are aligned against each other.

### 4. Diagnosis engine (`diagnosis/`)

`run_diagnosis()` builds a `DiagnosisContext` (the summaries plus the raw steps)
and runs every registered `@rule`. A rule is a single decorated function that
returns zero or more `Finding`s; findings carry a `code`, `severity`, `title`,
`detail` (with real numbers), and a concrete `suggestion`. Rules **must no-op
when their input data is absent** — that keeps the engine safe on partial runs.

The flagship is `CROSS.CORRELATED_INSTABILITY` (`rules_cross.py`): it flags
steps that are anomalous on **two or more** independent axes simultaneously
(loss / grad-norm / step-time / memory) and groups consecutive such steps into
one finding. Co-occurrence across axes on one clock is strong evidence of a real
optimization event rather than noise — and it's only expressible because every
axis shares the same timeline.

### 5. Reporters (`report/`)

Render summaries and findings. The CLI renderer is the default; the `Finding`
shape is reporter-agnostic, so HTML/markdown backends are additive.

## Design invariants

- **One timeline.** Every signal is keyed to the same step index. New verticals
  must align to it rather than introduce a parallel clock.
- **Pure-stdlib core.** Heavy dependencies stay in optional extras and are
  imported lazily. This is enforced socially (CONTRIBUTING) and structurally
  (the core imports nothing third-party).
- **Numerical care is non-negotiable.** Anything on the measurement/stats path
  uses the helpers in `analyzers/stats.py` and is tested against
  exact/near-exact expected values.
- **High-precision findings.** A rule should fire only when a practitioner would
  actually act on it; every rule ships a positive and a negative test.

## Extending it

- **A new heuristic** is one decorated function in `diagnosis/rules_*.py`.
- **A new vertical** is one analyzer over the existing `StepRecord` stream, its
  summary wired into `DiagnosisContext`, the CLI, and a reporter.

See [CONTRIBUTING](../CONTRIBUTING.md) for the step-by-step recipes.
