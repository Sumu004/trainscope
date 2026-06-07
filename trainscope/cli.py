"""`trainscope` command line — post-hoc analysis of a recorded run."""

from __future__ import annotations

import argparse
import sys

from .analyzers.convergence import analyze_convergence
from .analyzers.distributed import analyze_distributed, is_multirank, load_multirank
from .analyzers.memory import analyze_memory
from .analyzers.repro import diff_runs
from .analyzers.timing import analyze_timing
from .core.store import RunStore
from .diagnosis.engine import DiagnosisContext, run_diagnosis
from .report.cli_report import (
    render_convergence,
    render_diff,
    render_distributed,
    render_findings,
    render_memory,
    render_timing,
)


def cmd_analyze(args) -> int:
    # A distributed run is a directory of rank{k}/ subdirs; analyze across ranks
    # and additionally summarize rank 0's own single-rank timeline.
    distributed = None
    if is_multirank(args.run_dir):
        ranks = load_multirank(args.run_dir)
        distributed = analyze_distributed(ranks)
        store = ranks.get(min(ranks)) if ranks else None
    else:
        store = RunStore.load(args.run_dir)

    if store is None or not store.steps:
        if distributed is None:
            print(f"No steps found in {args.run_dir!r}.", file=sys.stderr)
            return 1

    steps = store.steps[args.warmup :] if store else []
    timing = analyze_timing(store.steps, warmup=args.warmup) if store else None
    memory = analyze_memory(steps)
    convergence = analyze_convergence(steps)
    findings = run_diagnosis(
        DiagnosisContext(
            timing=timing,
            memory=memory,
            convergence=convergence,
            steps=steps,
            distributed=distributed,
        )
    )

    name = store.meta.get("name", "run") if store else "run"
    print(f"trainscope — {name}  ({args.run_dir})\n")
    out = ""
    if timing is not None:
        out += render_timing(timing)
        out += render_memory(memory)
        out += render_convergence(convergence)
    out += render_distributed(distributed)
    out += render_findings(findings)
    print(out, end="")
    return 0


def cmd_diff(args) -> int:
    store_a = RunStore.load(args.run_a)
    store_b = RunStore.load(args.run_b)
    if not store_a.steps or not store_b.steps:
        print("Both run directories must contain steps.", file=sys.stderr)
        return 1
    print(f"trainscope diff  ({args.run_a}  vs  {args.run_b})\n")
    print(render_diff(diff_runs(store_a, store_b)), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainscope", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze a recorded run directory")
    p_analyze.add_argument("run_dir", help="Path to a run directory")
    p_analyze.add_argument(
        "--warmup", type=int, default=0, help="Steps to drop from the front"
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_diff = sub.add_parser(
        "diff", help="Compare two runs (reproducibility / drift analysis)"
    )
    p_diff.add_argument("run_a", help="First run directory")
    p_diff.add_argument("run_b", help="Second run directory")
    p_diff.set_defaults(func=cmd_diff)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
