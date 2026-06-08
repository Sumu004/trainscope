# Contributing to pytscope

First off — thank you for taking the time to contribute! 🎉 pytscope is an
intelligence layer for ML training, and it gets better the more eyes are on the
numbers. This guide explains how to propose changes.

By participating, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Running the checks](#running-the-checks)
- [Project layout](#project-layout)
- [Adding a diagnosis rule](#adding-a-diagnosis-rule)
- [Adding an analyzer (new vertical)](#adding-an-analyzer-new-vertical)
- [Coding standards](#coding-standards)
- [Commit & PR guidelines](#commit--pr-guidelines)
- [Reporting bugs & requesting features](#reporting-bugs--requesting-features)

## Ways to contribute

- **Report a bug** or a wrong/noisy finding (see the issue templates).
- **Add or tune a diagnosis rule** — the highest-leverage, most approachable
  contribution. New heuristics are a single decorated function.
- **Add an analyzer** for a new signal.
- **Improve an integration** (Lightning, Hugging Face, or a new framework).
- **Docs, examples, tests** — always welcome.

If your change is large, please open an issue to discuss the design first so we
don't both build the same thing twice.

## Development setup

Requires Python 3.9+.

```bash
git clone https://github.com/Sumu004/pytscope.git
cd pytscope
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + test/lint tooling
# optional, for the framework integrations / real examples:
pip install -e ".[torch,lightning,huggingface]"
```

(Optional) enable the pre-commit hooks so formatting/lint run automatically:

```bash
pre-commit install
```

## Running the checks

Everything CI runs, locally:

```bash
pytest -q                 # full test suite (must pass)
pytest -s -q tests/test_overhead.py   # see per-step overhead numbers
ruff check .              # lint
ruff format --check .     # formatting
python -m build && twine check dist/*   # packaging stays valid
```

**The core has no runtime dependencies and must stay that way** — keep heavy
deps (torch, etc.) optional and import them lazily inside functions.

## Project layout

```
pytscope/
  core/          # event schema, run store, provenance, rank detection
  collectors/    # memory snapshot
  profiler.py    # the live Profiler (timing engine)
  integrations/  # lightning.py, huggingface.py  (one-line callbacks)
  analyzers/     # timing / memory / convergence / repro + shared stats
  diagnosis/     # @rule engine + rule modules (the "intelligence")
  report/        # CLI renderers
  cli.py
tests/           # one file per area; please add tests with every change
examples/        # runnable demos (keep the no-dep ones dependency-free)
```

## Adding a diagnosis rule

Rules turn analyzer output into ranked, actionable findings. Add a function to
the relevant `diagnosis/rules_*.py` and decorate it:

```python
from .engine import DiagnosisContext, Finding, rule

@rule
def my_check(ctx: DiagnosisContext) -> list[Finding]:
    t = ctx.timing
    if not t or some_condition_not_met:
        return []                     # rules must no-op when data is absent
    return [Finding(
        code="TIMING.MY_CHECK",
        severity="med",               # "high" | "med" | "low"
        title="Short, specific title",
        detail="What was measured, with numbers.",
        suggestion="The concrete thing to change.",
    )]
```

Guidelines: prefer **high precision** (a finding a practitioner would actually
act on) over coverage; always guard against missing data; cite real numbers; and
add a positive **and** a negative test (it fires when it should, stays silent
when it shouldn't).

## Adding an analyzer (new vertical)

Analyzers are pure functions over `List[StepRecord]` returning a summary
dataclass. Wire the summary into `DiagnosisContext`, the CLI, and the report.
Keep all reductions numerically careful — use `math.fsum` and the helpers in
`analyzers/stats.py`.

## Coding standards

- **Style:** `ruff format` (Black-compatible), 90-char lines. `ruff check` clean.
- **Types:** public functions are type-annotated; the package ships `py.typed`.
- **Every module** that uses type hints starts with
  `from __future__ import annotations`.
- **Tests:** new behavior needs tests; assert exact/near-exact numbers for
  anything in the measurement or stats path.
- **No new runtime dependencies** in the core.

## Commit & PR guidelines

- Keep PRs focused; one logical change per PR.
- Write a clear description: what, why, and how you verified it.
- Reference any related issue (`Fixes #123`).
- Ensure `pytest`, `ruff check`, and `ruff format --check` all pass.
- Update `CHANGELOG.md` under `[Unreleased]` for user-facing changes.
- Commit messages: imperative mood, e.g. `Add fragmentation rule for MPS`.

We squash-merge, so your PR title becomes the commit — make it descriptive.

## Reporting bugs & requesting features

Use the issue templates. For bugs, include: pytscope version, Python version,
OS/accelerator, a minimal repro, and the actual vs expected finding/output. A
wrong or noisy finding **is** a bug — please report it with the run details.

Thanks again for contributing! 💜
