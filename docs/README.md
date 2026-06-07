# trainscope documentation

An intelligence layer for ML training — capture timing, memory, convergence, and
provenance on one aligned per-step timeline, then turn the raw numbers into
ranked, actionable findings.

## Contents

- [Usage guide](usage.md) — install, instrument a loop, the Lightning/HF
  callbacks, and the `analyze` / `diff` CLI.
- [Architecture](architecture.md) — the telemetry backbone, analyzers, and
  diagnosis engine, and why one shared timeline is the design's core idea.
- [Diagnostics reference](diagnostics.md) — every finding code, what triggers
  it, and what to do about it.

## Project meta

- [Changelog](../CHANGELOG.md)
- [Contributing](../CONTRIBUTING.md)
- [Release process](../RELEASING.md)
