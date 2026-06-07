"""Exposed-communication analysis from a kernel trace — no GPU required.

Real NCCL overlap numbers come from a multi-GPU ``torch.profiler`` trace. To make
the analysis runnable anywhere, this writes a small **synthetic** Kineto-style
trace with a known compute/communication overlap, then analyzes it::

    python examples/exposed_comm.py && trainscope analyze runs/trace_demo

The synthetic run models 6 steps. Each step: a backward compute kernel of 8 ms
and an all-reduce of 6 ms that overlaps the last 4 ms of backward — so 2 ms per
step (12 ms total) is exposed. trainscope should report ~67% overlap efficiency
and an exposed-communication finding (~20% of wall). With a real trace, pass it
via ``trainscope analyze <run> --trace path/to/trace.json``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _kernel(name, ts_us, dur_us, tid):
    # Minimal Kineto "complete" GPU-kernel event (timestamps in microseconds).
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "ts": ts_us,
        "dur": dur_us,
        "pid": 0,
        "tid": tid,
    }


def main() -> None:
    run_dir = Path("runs/trace_demo")
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)

    events = []
    step_us = 10_000  # 10 ms between step starts
    compute_us = 8_000
    comm_us = 6_000
    for step in range(6):
        base = step * step_us
        # backward compute kernel on the compute stream (tid=7)
        events.append(_kernel("ampere_sgemm_backward", base, compute_us, tid=7))
        # all-reduce on the NCCL comm stream (tid=20). It overlaps the last 4 ms
        # of backward and ends exactly at the step boundary (no bleed into the
        # next step), so 2 ms of its 6 ms is exposed.
        comm_start = base + compute_us - 4_000
        events.append(_kernel("ncclDevKernel_AllReduce_Sum", comm_start, comm_us, tid=20))

    (run_dir / "trace.json").write_text(json.dumps({"traceEvents": events}))
    print(f"Wrote synthetic trace to {run_dir / 'trace.json'}")
    print(f"Now run:  trainscope analyze {run_dir}")


if __name__ == "__main__":
    main()
