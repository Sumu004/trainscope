"""Pipeline-parallel bubble analysis.

In pipeline parallelism a batch is split into ``m`` microbatches streamed across
``p`` stages. During warmup (filling the pipe) and cooldown (draining it) some
stages have no work — the **bubble**. For a uniform GPipe schedule the bubble
fraction has a closed form::

    bubble_fraction = (p - 1) / (m + p - 1)

which is why you increase ``m`` to amortize it. This analyzer measures the
*achieved* bubble from the recorded per-stage busy intervals and compares it to
that ideal, so you can tell "is my pipeline badly scheduled?" from "is this just
the inherent bubble for my p and m?".

The analyzer is schedule-based: it consumes per-stage ``(start, end)`` intervals
(in seconds, one timeline). It does not require a GPU — given a real recorded
schedule it reports the real bubble; given the closed-form GPipe schedule it
reproduces the formula exactly (see tests). ``gpipe_schedule`` generates a
physically-correct reference schedule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

# A scheduled unit of work: which stage, and its [start, end) wall interval.
Interval = tuple[int, float, float]  # (stage, start, end)


@dataclass
class PipelineSummary:
    n_stages: int
    makespan: float  # wall span from first start to last end
    total_busy: float  # summed busy time across all stages
    bubble_fraction: float  # idle / (n_stages * makespan), in [0, 1)
    per_stage_bubble: dict[int, float] = field(default_factory=dict)
    n_microbatches: int | None = None
    ideal_bubble_fraction: float | None = None  # (p-1)/(m+p-1) if m known
    busiest_stage: int | None = None
    idlest_stage: int | None = None

    @property
    def excess_bubble(self) -> float | None:
        """Bubble beyond the inherent GPipe minimum (schedule inefficiency)."""
        if self.ideal_bubble_fraction is None:
            return None
        return max(0.0, self.bubble_fraction - self.ideal_bubble_fraction)


def analyze_pipeline(
    intervals: Sequence[Interval],
    n_microbatches: int | None = None,
) -> PipelineSummary | None:
    """Compute the bubble fraction from per-stage busy intervals.

    ``intervals``: iterable of ``(stage, start, end)``. ``n_microbatches``:
    optional ``m`` so the inherent GPipe ideal can be reported for comparison.
    """
    if not intervals:
        return None
    stages = sorted({s for s, _, _ in intervals})
    p = len(stages)
    if p < 2:
        return None

    start = min(s for _, s, _ in intervals)
    end = max(e for _, _, e in intervals)
    makespan = end - start
    if makespan <= 0:
        return None

    busy_by_stage: dict[int, float] = {s: 0.0 for s in stages}
    for stage, s0, s1 in intervals:
        busy_by_stage[stage] += max(0.0, s1 - s0)

    total_busy = math.fsum(busy_by_stage.values())
    bubble = 1.0 - total_busy / (p * makespan)
    bubble = min(max(bubble, 0.0), 1.0)

    per_stage = {s: min(max(1.0 - busy_by_stage[s] / makespan, 0.0), 1.0) for s in stages}
    ideal = None
    if n_microbatches and n_microbatches > 0:
        ideal = (p - 1) / (n_microbatches + p - 1)

    busiest = max(stages, key=lambda s: busy_by_stage[s])
    idlest = min(stages, key=lambda s: busy_by_stage[s])

    return PipelineSummary(
        n_stages=p,
        makespan=makespan,
        total_busy=total_busy,
        bubble_fraction=bubble,
        per_stage_bubble=per_stage,
        n_microbatches=n_microbatches,
        ideal_bubble_fraction=ideal,
        busiest_stage=busiest,
        idlest_stage=idlest,
    )


def gpipe_schedule(
    n_stages: int,
    n_microbatches: int,
    forward: float = 1.0,
    backward: float = 0.0,
) -> list[Interval]:
    """Generate a physically-correct GPipe schedule (reference / testing).

    Each microbatch flows forward through stages 0..p-1; stage ``i`` cannot start
    microbatch ``j`` until both stage ``i-1`` finished feeding it ``j`` and stage
    ``i`` finished its own ``j-1``. With ``backward=0`` this is a forward-only
    pipeline whose bubble fraction is exactly ``(p-1)/(m+p-1)``.
    """
    p, m = n_stages, n_microbatches
    intervals: list[Interval] = []
    # finish[i] = time stage i is free; arrive[i][j] tracked implicitly by
    # forward dependency. We model uniform unit time `forward` per stage.
    stage_free = [0.0] * p
    # done[j] after stage i = end time of microbatch j at stage i.
    prev_stage_end = [0.0] * m  # end time at stage i-1 for each microbatch
    for i in range(p):
        cur_end = [0.0] * m
        for j in range(m):
            ready = max(stage_free[i], prev_stage_end[j] if i > 0 else 0.0)
            s0 = ready
            s1 = s0 + forward
            intervals.append((i, s0, s1))
            stage_free[i] = s1
            cur_end[j] = s1
        prev_stage_end = cur_end
    if backward > 0:
        # Symmetric backward sweep stages p-1..0, microbatches m-1..0.
        bstage_free = [intervals[-1][2]] * p  # start draining after forward done
        prev_end = [0.0] * m
        for idx in range(p):
            i = p - 1 - idx
            cur_end = [0.0] * m
            for jdx in range(m):
                j = m - 1 - jdx
                ready = max(bstage_free[i], prev_end[j] if idx > 0 else 0.0)
                s0 = ready
                s1 = s0 + backward
                intervals.append((i, s0, s1))
                bstage_free[i] = s1
                cur_end[j] = s1
            prev_end = cur_end
    return intervals
