"""The Training Efficiency Budget — one accounting identity for the whole run.

Every other analyzer produces a *finding*. This one produces a **budget**: it
decomposes the wall time of training into named line items that, by construction,
sum back to the measured wall time. Anchored at the top by ``useful_compute`` —
the time the math would take at the hardware's peak throughput — the rest of the
budget is, line by line, *recoverable* time:

    wall = useful_compute            (irreducible: the FLOPs, at peak)
         + compute_overhead          (your kernels don't hit peak)
         + data_stall                (waiting on the dataloader)
         + communication             (collective time on the timeline)
         + other                     (everything else attributed)

``MFU`` (Model FLOPs Utilization) falls straight out: ``useful_compute / wall``.
Because the phase timeline already partitions each step, the decomposition is
**exact** — the line items sum to the attributed wall with no fudge factor, which
makes the model falsifiable (a wrong term shows up as a non-zero residual).

This turns a profiler into an advisor: each recoverable line is a number of
seconds you could win back, so fixes rank themselves by payoff.

FLOPs and peak are optional. With them you get a true MFU anchor; without them
the budget still decomposes wall time, with ``useful_compute`` falling back to
measured compute (MFU unknown).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.events import (
    COMM,
    COMPUTE,
    DATA,
    OTHER,
    StepRecord,
)

# Phases that count as local compute (where FLOPs are spent).
_COMPUTE_PHASES = ("forward", "backward", "optimizer", COMPUTE)


@dataclass
class BudgetLine:
    name: str
    seconds: float
    fraction: float  # of attributed wall
    recoverable: bool  # is this time you could win back?


@dataclass
class EfficiencyBudget:
    wall: float  # attributed wall (sum of all phases over the run)
    n_steps: int
    compute_measured: float  # forward+backward+optimizer+compute, seconds
    ideal_compute: float | None  # FLOPs-anchored minimum, seconds (if known)
    mfu: float | None  # useful_compute / wall, in [0, 1] (if FLOPs+peak known)
    flops_per_step: float | None
    peak_flops: float | None
    lines: list[BudgetLine] = field(default_factory=list)

    @property
    def efficiency(self) -> float:
        """Useful fraction of wall (== MFU when FLOPs+peak are known)."""
        useful = next(
            (ln.seconds for ln in self.lines if ln.name == "useful_compute"), 0.0
        )
        return useful / self.wall if self.wall > 0 else 0.0

    @property
    def recoverable_lines(self) -> list[BudgetLine]:
        """Recoverable line items, largest first — the ranked fix list."""
        return sorted(
            (ln for ln in self.lines if ln.recoverable and ln.seconds > 0),
            key=lambda ln: ln.seconds,
            reverse=True,
        )

    @property
    def top_recoverable(self) -> BudgetLine | None:
        rec = self.recoverable_lines
        return rec[0] if rec else None


def _phase_total(steps: Sequence[StepRecord], phase: str) -> float:
    return math.fsum(s.phases.get(phase, 0.0) for s in steps)


def analyze_efficiency(
    steps: Sequence[StepRecord],
    *,
    flops_per_step: float | None = None,
    peak_flops: float | None = None,
) -> EfficiencyBudget | None:
    """Build the efficiency budget from the per-step phase timeline.

    ``flops_per_step``: training FLOPs per step (forward+backward; see
    ``hardware.measure_flops``). ``peak_flops``: device peak FLOP/s. Both
    optional — supply them for a true MFU anchor.
    """
    steps = list(steps)
    if not steps:
        return None

    data = _phase_total(steps, DATA)
    comm = _phase_total(steps, COMM)
    other = _phase_total(steps, OTHER)
    compute_measured = math.fsum(_phase_total(steps, p) for p in _COMPUTE_PHASES)
    wall = compute_measured + data + comm + other
    if wall <= 0:
        return None

    n = len(steps)
    ideal_compute: float | None = None
    mfu: float | None = None
    if flops_per_step and peak_flops and peak_flops > 0:
        ideal_compute = flops_per_step * n / peak_flops

    # Useful compute is the irreducible work. If we know the FLOPs-anchored ideal,
    # it caps useful at the measured compute (you can't be "more than 100%
    # efficient"; if the estimate exceeds measured, treat compute as all-useful
    # and surface no negative overhead).
    if ideal_compute is not None:
        useful = min(ideal_compute, compute_measured)
        mfu = useful / wall
    else:
        useful = compute_measured
    overhead = compute_measured - useful

    lines = [
        BudgetLine("useful_compute", useful, useful / wall, recoverable=False),
        BudgetLine("compute_overhead", overhead, overhead / wall, recoverable=True),
        BudgetLine("data_stall", data, data / wall, recoverable=True),
        BudgetLine("communication", comm, comm / wall, recoverable=True),
        BudgetLine("other", other, other / wall, recoverable=True),
    ]
    return EfficiencyBudget(
        wall=wall,
        n_steps=n,
        compute_measured=compute_measured,
        ideal_compute=ideal_compute,
        mfu=mfu,
        flops_per_step=flops_per_step,
        peak_flops=peak_flops,
        lines=lines,
    )
