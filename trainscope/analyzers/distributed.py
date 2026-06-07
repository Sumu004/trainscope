"""Distributed (data-parallel) analyzer — the headline vertical.

In synchronous data-parallel training every rank must reach the gradient
all-reduce before *any* rank can proceed. The step is therefore gated by the
**slowest** rank's compute (the critical path); every faster rank sits idle at
the barrier. That idle time is pure waste, and it is invisible to any
single-rank profiler — you only see it by putting all ranks on one timeline.

This module aligns the per-rank step timelines and computes:

- **Critical-path wall loss** — wall time lost because the step waits for the
  slowest rank instead of the average rank.
- **Straggler attribution with a persistence test** — *which* rank is slow and,
  crucially, whether it is a *consistent* straggler (a bad GPU/node) or just
  noise. We test the count of steps a rank is the critical path against the
  null hypothesis that the slowest rank is uniformly random (Binomial(S, 1/N)),
  via a one-sided normal approximation z-score.
- **Load imbalance** — robust coefficient of variation of per-rank compute.
- **Communication fraction** — share of step time spent in collectives.
- **Sync skew** — how far ahead the fastest ranks arrive at the barrier.

Everything is pure-stdlib and numerically careful (``math.fsum``, robust
medians from :mod:`trainscope.analyzers.stats`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from ..core.events import COMM, COMPUTE_PHASES, StepRecord
from ..core.store import RunStore
from .stats import median


@dataclass
class RankStat:
    rank: int
    mean_compute: float  # mean per-step local compute (non-comm), seconds
    median_compute: float
    slowest_count: int  # steps where this rank was the critical path
    slowest_fraction: float
    straggler_z: float  # z-score of slowest_count vs Binomial(S, 1/N)
    rel_slowdown: float  # median(this_rank_compute / step_median_compute) - 1


@dataclass
class DistributedSummary:
    world_size: int
    n_steps: int  # number of aligned steps analyzed
    mean_step_wall: float  # mean critical-path step time (max compute + comm)
    mean_comm_fraction: float  # comm / step wall
    imbalance_cv: float  # median over steps of CV(per-rank compute)
    sync_skew: float  # median over steps of (max - median) compute, seconds
    wall_frac_lost_to_imbalance: float  # (max-mean)/max summed over steps
    ranks: list[RankStat] = field(default_factory=list)
    straggler: RankStat | None = None  # worst *persistent* straggler, if any

    @property
    def has_straggler(self) -> bool:
        return self.straggler is not None


# --- loading -------------------------------------------------------------


def load_multirank(run_dir) -> dict[int, RunStore]:
    """Load a distributed run written by ``Profiler(distributed=True)``.

    Layout is ``run_dir/rank{k}/{steps.jsonl,run.json}``. Returns a mapping of
    rank -> RunStore. Returns ``{}`` if the directory has no per-rank subdirs.
    """
    run_dir = Path(run_dir)
    ranks: dict[int, RunStore] = {}
    if not run_dir.is_dir():
        return ranks
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name.startswith("rank"):
            suffix = child.name[len("rank") :]
            if suffix.isdigit():
                ranks[int(suffix)] = RunStore.load(child)
    return ranks


def is_multirank(run_dir) -> bool:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return False
    return any(
        c.is_dir() and c.name.startswith("rank") and c.name[len("rank") :].isdigit()
        for c in run_dir.iterdir()
    )


# --- helpers -------------------------------------------------------------


def _compute_seconds(rec: StepRecord) -> float:
    """Local compute time for a step: everything that is not communication."""
    return math.fsum(v for p, v in rec.phases.items() if p in COMPUTE_PHASES)


def _comm_seconds(rec: StepRecord) -> float:
    return float(rec.phases.get(COMM, 0.0))


def _mean(xs: list[float]) -> float:
    return math.fsum(xs) / len(xs) if xs else 0.0


def _cv(xs: list[float]) -> float:
    """Coefficient of variation (std / mean), population std. 0 if mean<=0."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    if m <= 0:
        return 0.0
    var = math.fsum((x - m) ** 2 for x in xs) / n
    return math.sqrt(var) / m


# --- the analyzer --------------------------------------------------------


def analyze_distributed(
    ranks: dict[int, RunStore],
    straggler_z: float = 3.0,
    min_rel_slowdown: float = 0.05,
) -> DistributedSummary | None:
    """Analyze aligned per-rank timelines.

    ``straggler_z``: z-score threshold for calling a rank a *persistent*
    straggler (3.0 ≈ p<0.002, one-sided). ``min_rel_slowdown``: also require the
    flagged rank to be at least this much slower than the per-step median, so we
    don't flag a statistically-consistent-but-negligible straggler.
    """
    if len(ranks) < 2:
        return None

    rank_ids = sorted(ranks)
    n = len(rank_ids)
    # Per-rank step index -> record, so we can align by the shared step number.
    by_rank: dict[int, dict[int, StepRecord]] = {
        r: {rec.step: rec for rec in ranks[r].steps} for r in rank_ids
    }
    common_steps = sorted(set.intersection(*[set(by_rank[r]) for r in rank_ids]))
    if not common_steps:
        return None

    # Accumulators.
    step_walls: list[float] = []
    comm_fracs: list[float] = []
    imbalance_cvs: list[float] = []
    sync_skews: list[float] = []
    slowest_count = {r: 0 for r in rank_ids}
    rel_slowdowns: dict[int, list[float]] = {r: [] for r in rank_ids}
    compute_samples: dict[int, list[float]] = {r: [] for r in rank_ids}
    sum_max = 0.0
    sum_lost = 0.0  # sum of (max - mean) compute

    for s in common_steps:
        comps = {r: _compute_seconds(by_rank[r][s]) for r in rank_ids}
        comms = {r: _comm_seconds(by_rank[r][s]) for r in rank_ids}
        cvals = [comps[r] for r in rank_ids]
        cmax = max(cvals)
        cmean = _mean(cvals)
        cmed = median(cvals)
        # Critical path: slowest compute, plus the max comm observed that step.
        wall = cmax + max(comms.values())
        step_walls.append(wall)
        comm_fracs.append((max(comms.values()) / wall) if wall > 0 else 0.0)
        imbalance_cvs.append(_cv(cvals))
        sync_skews.append(cmax - cmed)
        sum_max += cmax
        sum_lost += cmax - cmean

        # Critical rank (argmax compute); ties -> lowest rank id (deterministic).
        crit = min(rank_ids, key=lambda r: (-comps[r], r))
        slowest_count[crit] += 1
        for r in rank_ids:
            compute_samples[r].append(comps[r])
            if cmed > 0:
                rel_slowdowns[r].append(comps[r] / cmed - 1.0)

    s_total = len(common_steps)
    # Binomial(S, 1/N) null for "is rank r the slowest".
    p0 = 1.0 / n
    sd = math.sqrt(s_total * p0 * (1.0 - p0)) or 1.0
    expected = s_total * p0

    rank_stats: list[RankStat] = []
    for r in rank_ids:
        z = (slowest_count[r] - expected) / sd
        rank_stats.append(
            RankStat(
                rank=r,
                mean_compute=_mean(compute_samples[r]),
                median_compute=median(compute_samples[r]),
                slowest_count=slowest_count[r],
                slowest_fraction=slowest_count[r] / s_total,
                straggler_z=z,
                rel_slowdown=median(rel_slowdowns[r]) if rel_slowdowns[r] else 0.0,
            )
        )

    # A *single* persistent straggler exists only if exactly one rank clears both
    # the statistical-persistence bar (z) and the practical-magnitude bar
    # (rel_slowdown). If several ranks do (e.g. two slow nodes alternating as the
    # critical path), that's load imbalance, not one straggler — leave it to the
    # imbalance rule rather than fingering one rank misleadingly.
    significant = [
        rs
        for rs in rank_stats
        if rs.straggler_z >= straggler_z and rs.rel_slowdown >= min_rel_slowdown
    ]
    straggler = significant[0] if len(significant) == 1 else None

    return DistributedSummary(
        world_size=n,
        n_steps=s_total,
        mean_step_wall=_mean(step_walls),
        mean_comm_fraction=_mean(comm_fracs),
        imbalance_cv=median(imbalance_cvs),
        sync_skew=median(sync_skews),
        wall_frac_lost_to_imbalance=(sum_lost / sum_max) if sum_max > 0 else 0.0,
        ranks=rank_stats,
        straggler=straggler,
    )
