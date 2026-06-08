"""Timing analyzer — turns the raw step timeline into an aggregate summary.

Numerical care:
- ``math.fsum`` for every reduction, so summing millions of small step times
  does not accumulate floating-point error.
- One shared grand total: ``mean_step_time``, ``phase_seconds`` and
  ``phase_fractions`` are all derived from the *same* fsum'd total, so they are
  mutually consistent (fractions sum to 1, mean*n == total) rather than computed
  from independent summations that disagree in the low bits.
- Two-pass variance (mean first, then fsum of squared deviations) — stable,
  never the catastrophic one-pass sum-of-squares.
- Percentiles via linear interpolation on sorted values (numpy 'linear'
  convention) for robust straggler detection that CV alone misses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.events import StepRecord, order_phases


@dataclass
class TimingSummary:
    n_steps: int
    mean_step_time: float  # seconds
    std_step_time: float
    p50_step_time: float
    p95_step_time: float
    steps_per_sec: float
    total_time: float
    phase_seconds: dict[str, float] = field(default_factory=dict)  # mean per step
    phase_fractions: dict[str, float] = field(default_factory=dict)  # of total time
    phase_order: list[str] = field(default_factory=list)

    @property
    def cv(self) -> float:
        """Coefficient of variation of step time — a jitter signal."""
        return self.std_step_time / self.mean_step_time if self.mean_step_time else 0.0

    @property
    def p95_p50_ratio(self) -> float:
        """Tail/median ratio — robust straggler signal, less fragile than CV."""
        return self.p95_step_time / self.p50_step_time if self.p50_step_time else 0.0


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]) over pre-sorted values."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def analyze_timing(steps: list[StepRecord], warmup: int = 0) -> TimingSummary:
    steps = steps[warmup:]
    n = len(steps)
    if n == 0:
        return TimingSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}, [])

    # Per-step totals via fsum, so each step total is itself exact.
    step_times = [math.fsum(s.phases.values()) for s in steps]

    # Phase totals accumulated exactly with fsum across all steps.
    phase_values: dict[str, list[float]] = {}
    for s in steps:
        for phase, dur in s.phases.items():
            phase_values.setdefault(phase, []).append(dur)
    phase_totals = {p: math.fsum(v) for p, v in phase_values.items()}

    # ONE shared total drives mean and fractions -> mutually consistent.
    grand_total = math.fsum(phase_totals.values())
    denom = grand_total if grand_total > 0.0 else 1.0
    mean_step = grand_total / n

    # Two-pass variance (numerically stable).
    var = math.fsum((t - mean_step) ** 2 for t in step_times) / n
    std_step = math.sqrt(var)

    sorted_times = sorted(step_times)
    p50 = _percentile(sorted_times, 50.0)
    p95 = _percentile(sorted_times, 95.0)

    phase_seconds = {p: phase_totals[p] / n for p in phase_totals}
    phase_fractions = {p: phase_totals[p] / denom for p in phase_totals}
    ordered = order_phases(phase_totals.keys())

    return TimingSummary(
        n_steps=n,
        mean_step_time=mean_step,
        std_step_time=std_step,
        p50_step_time=p50,
        p95_step_time=p95,
        steps_per_sec=(1.0 / mean_step) if mean_step > 0 else 0.0,
        total_time=grand_total,
        phase_seconds={p: phase_seconds[p] for p in ordered},
        phase_fractions={p: phase_fractions[p] for p in ordered},
        phase_order=ordered,
    )
