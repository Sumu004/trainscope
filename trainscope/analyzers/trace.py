"""Exposed-communication analysis from a PyTorch profiler (Kineto) trace.

In distributed training the gradient all-reduce *can* run concurrently with
backward compute (DDP bucketing overlaps them). The time that overlaps is
effectively free; the communication time that does **not** overlap any compute is
"exposed" — it sits on the critical path and is pure wall-time waste. Exposed
communication is the single most important efficiency metric in large-scale
data-parallel training, and you cannot get it from per-step phase timing alone:
you need the *kernel timeline* with compute and communication on separate streams.

This analyzer ingests a Chrome/Kineto trace (``torch.profiler`` ->
``export_chrome_trace``), classifies GPU kernels as communication (NCCL
collectives) vs compute, and computes — via **exact interval arithmetic** — how
much communication overlaps compute and how much is exposed:

    exposed_comm = | comm_intervals  \\  compute_intervals |
    overlapped   = | comm_intervals | - exposed_comm
    overlap_efficiency = overlapped / total_comm

The interval math is exact and dependency-free; see ``tests/test_trace.py`` for
synthetic traces with known answers. Real NCCL overlap numbers require a
multi-GPU trace, but the parser is validated against a genuine ``torch.profiler``
export.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

Interval = tuple[float, float]

# Substrings (lowercased kernel name) that mark a communication kernel. "nccl"
# catches every NCCL collective (ncclDevKernel_AllReduce_…); the explicit phrases
# catch backend-agnostic namings. We deliberately avoid a bare "reduce" (it would
# match compute reduction kernels).
_COMM_HINTS = (
    "nccl",
    "allreduce",
    "all_reduce",
    "allgather",
    "all_gather",
    "reducescatter",
    "reduce_scatter",
    "all_to_all",
    "alltoall",
    "broadcast",
    "c10d::",
)
# Categories that denote a device-side kernel in a Kineto trace.
_KERNEL_CATS = ("kernel", "gpu_memcpy", "gpu_memset")

_COLLECTIVE_KINDS = (
    ("all_reduce", ("allreduce", "all_reduce")),
    ("all_gather", ("allgather", "all_gather")),
    ("reduce_scatter", ("reducescatter", "reduce_scatter")),
    ("all_to_all", ("all_to_all", "alltoall")),
    ("broadcast", ("broadcast",)),
)


@dataclass
class TraceSummary:
    wall_time: float  # span of the trace (seconds)
    total_comm_time: float  # union length of communication kernels
    total_compute_time: float  # union length of compute kernels
    exposed_comm_time: float  # comm not overlapping any compute
    overlapped_comm_time: float  # comm overlapping compute
    n_comm_kernels: int
    n_compute_kernels: int
    per_collective: dict[str, float] = field(default_factory=dict)

    @property
    def overlap_efficiency(self) -> float:
        """Fraction of communication hidden behind compute, in [0, 1]."""
        return (
            self.overlapped_comm_time / self.total_comm_time
            if self.total_comm_time > 0
            else 0.0
        )

    @property
    def exposed_comm_fraction(self) -> float:
        """Exposed communication as a fraction of wall time."""
        return self.exposed_comm_time / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def has_comm(self) -> bool:
        return self.n_comm_kernels > 0


# --- interval arithmetic (exact) -----------------------------------------


def merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    """Return the disjoint union of intervals, sorted by start."""
    cleaned = [(s, e) for s, e in intervals if e > s]
    if not cleaned:
        return []
    cleaned.sort()
    out: list[Interval] = [cleaned[0]]
    for s, e in cleaned[1:]:
        ls, le = out[-1]
        if s <= le:  # overlapping or touching -> extend
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def subtract_intervals(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    """A minus B, both given as *disjoint sorted* interval lists."""
    out: list[Interval] = []
    j = 0
    nb = len(b)
    for s, e in a:
        cur = s
        k = j
        while k < nb and b[k][0] < e:
            bs, be = b[k]
            if be <= cur:
                k += 1
                continue
            if bs > cur:
                out.append((cur, min(bs, e)))
            cur = max(cur, be)
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def total_length(intervals: Sequence[Interval]) -> float:
    return math.fsum(e - s for s, e in intervals)


# --- loading & classification --------------------------------------------


def load_chrome_trace(path) -> list[dict]:
    """Load a Chrome/Kineto trace (.json or .json.gz) -> list of trace events."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def _is_kernel(ev: dict) -> bool:
    cat = str(ev.get("cat", "")).lower()
    return any(k in cat for k in _KERNEL_CATS)


def _is_comm(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in _COMM_HINTS)


def _collective_kind(name: str) -> str:
    n = name.lower()
    for kind, hints in _COLLECTIVE_KINDS:
        if any(h in n for h in hints):
            return kind
    return "other"


def analyze_trace(events: Sequence[dict]) -> TraceSummary | None:
    """Compute exposed/overlapped communication from Kineto trace events."""
    comm: list[Interval] = []
    compute: list[Interval] = []
    per_collective: dict[str, float] = {}
    lo = math.inf
    hi = -math.inf

    for ev in events:
        if ev.get("ph") != "X":  # only complete (duration) events
            continue
        dur = ev.get("dur")
        ts = ev.get("ts")
        if dur is None or ts is None or dur <= 0:
            continue
        if not _is_kernel(ev):
            continue
        # Kineto timestamps are microseconds; convert to seconds.
        s = ts / 1e6
        e = (ts + dur) / 1e6
        lo = min(lo, s)
        hi = max(hi, e)
        if _is_comm(str(ev.get("name", ""))):
            comm.append((s, e))
            kind = _collective_kind(str(ev.get("name", "")))
            per_collective[kind] = per_collective.get(kind, 0.0) + (e - s)
        else:
            compute.append((s, e))

    if not comm and not compute:
        return None

    comm_u = merge_intervals(comm)
    compute_u = merge_intervals(compute)
    exposed = subtract_intervals(comm_u, compute_u)
    total_comm = total_length(comm_u)
    exposed_comm = total_length(exposed)

    return TraceSummary(
        wall_time=(hi - lo) if hi > lo else 0.0,
        total_comm_time=total_comm,
        total_compute_time=total_length(compute_u),
        exposed_comm_time=exposed_comm,
        overlapped_comm_time=total_comm - exposed_comm,
        n_comm_kernels=len(comm),
        n_compute_kernels=len(compute),
        per_collective=per_collective,
    )


def analyze_trace_file(path) -> TraceSummary | None:
    return analyze_trace(load_chrome_trace(path))
