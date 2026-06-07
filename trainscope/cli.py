"""`trainscope` command line — post-hoc analysis of a recorded run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzers.convergence import analyze_convergence
from .analyzers.distributed import analyze_distributed, is_multirank, load_multirank
from .analyzers.efficiency import analyze_efficiency
from .analyzers.memory import analyze_memory
from .analyzers.repro import diff_runs
from .analyzers.timing import analyze_timing
from .analyzers.trace import analyze_trace_file
from .core.store import RunStore
from .diagnosis.engine import DiagnosisContext, run_diagnosis
from .report.cli_report import (
    render_budget,
    render_convergence,
    render_diff,
    render_distributed,
    render_findings,
    render_memory,
    render_timing,
    render_trace,
)

# Trace files trainscope will auto-detect inside a run directory.
_TRACE_NAMES = ("trace.json", "trace.json.gz", "kineto.json", "kineto.json.gz")


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

    trace = _resolve_trace(args)
    if (store is None or not store.steps) and distributed is None and trace is None:
        print(f"No steps or trace found in {args.run_dir!r}.", file=sys.stderr)
        return 1

    steps = store.steps[args.warmup :] if store else []
    timing = analyze_timing(store.steps, warmup=args.warmup) if store else None
    memory = analyze_memory(steps)
    convergence = analyze_convergence(steps)
    efficiency = _resolve_efficiency(args, store, steps)
    findings = run_diagnosis(
        DiagnosisContext(
            timing=timing,
            memory=memory,
            convergence=convergence,
            steps=steps,
            distributed=distributed,
            trace=trace,
            efficiency=efficiency,
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
    out += render_trace(trace)
    out += render_budget(efficiency)
    out += render_findings(findings)
    print(out, end="")
    return 0


def _resolve_efficiency(args, store, steps):
    """Build the efficiency budget; FLOPs/peak from CLI flags or run meta."""
    if not steps:
        return None
    meta = store.meta if store else {}
    flops = getattr(args, "flops_per_step", None)
    if flops is None:
        flops = meta.get("flops_per_step")
    peak = None
    if getattr(args, "peak_tflops", None) is not None:
        peak = args.peak_tflops * 1e12
    elif meta.get("peak_flops"):
        peak = meta.get("peak_flops")
    return analyze_efficiency(steps, flops_per_step=flops, peak_flops=peak)


def _resolve_trace(args):
    """Load a Kineto trace from --trace, else auto-detect one in the run dir."""
    path = getattr(args, "trace", None)
    if path is None:
        for name in _TRACE_NAMES:
            cand = Path(args.run_dir) / name
            if cand.exists():
                path = cand
                break
    if path is None:
        return None
    try:
        return analyze_trace_file(path)
    except (OSError, ValueError) as exc:
        print(f"Could not read trace {path!r}: {exc}", file=sys.stderr)
        return None


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
    p_analyze.add_argument(
        "--trace",
        default=None,
        help="Path to a torch.profiler/Kineto trace (.json/.json.gz) for "
        "exposed-communication analysis. Auto-detected in the run dir if present.",
    )
    p_analyze.add_argument(
        "--flops-per-step",
        type=float,
        default=None,
        dest="flops_per_step",
        help="Training FLOPs per step, to anchor the efficiency budget / MFU.",
    )
    p_analyze.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        dest="peak_tflops",
        help="Device peak throughput in TFLOP/s (overrides the built-in table).",
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
