"""Distributed-training diagnosis rules — the headline vertical.

These read the multi-rank critical-path summary and the pipeline-bubble summary.
They are *not* threshold-only heuristics: the straggler rule fires on a
statistical-persistence test (is one rank consistently the critical path?), and
the pipeline rule compares achieved bubble against the inherent GPipe minimum,
so it never flags a bubble that is simply the cost of your p and m.
"""

from __future__ import annotations

from .engine import DiagnosisContext, Finding, rule


@rule
def straggler(ctx: DiagnosisContext) -> list[Finding]:
    d = ctx.distributed
    if not d or not d.straggler:
        return []
    s = d.straggler
    lost_pct = d.wall_frac_lost_to_imbalance * 100.0
    sev = "high" if (lost_pct >= 10.0 or s.rel_slowdown >= 0.25) else "med"
    return [
        Finding(
            code="DIST.STRAGGLER",
            severity=sev,
            title=f"Rank {s.rank} is a persistent straggler",
            detail=(
                f"Rank {s.rank} is the slowest (critical-path) rank in "
                f"{s.slowest_fraction * 100:.0f}% of steps across {d.world_size} "
                f"ranks (expected {100.0 / d.world_size:.0f}% by chance; "
                f"z={s.straggler_z:.1f}), and runs {s.rel_slowdown * 100:.0f}% "
                f"slower than the median rank. Synchronous all-reduce makes every "
                f"other rank wait for it — {lost_pct:.1f}% of wall time is lost to "
                f"this imbalance."
            ),
            suggestion=(
                f"Investigate rank {s.rank}'s device/host: thermal throttling, a "
                f"slower GPU, NUMA/host placement, or an unbalanced data shard. "
                f"Check for a hot node and rebalance or replace it."
            ),
        )
    ]


@rule
def load_imbalance(ctx: DiagnosisContext) -> list[Finding]:
    d = ctx.distributed
    if not d:
        return []
    # Imbalance without a single persistent culprit (otherwise STRAGGLER covers it).
    if d.straggler is not None:
        return []
    lost_pct = d.wall_frac_lost_to_imbalance * 100.0
    if d.imbalance_cv < 0.15 or lost_pct < 5.0:
        return []
    return [
        Finding(
            code="DIST.LOAD_IMBALANCE",
            severity="med",
            title="Compute is imbalanced across ranks",
            detail=(
                f"Per-rank compute varies by CV={d.imbalance_cv:.2f} with no single "
                f"persistent straggler. {lost_pct:.1f}% of wall time is lost waiting "
                f"at the all-reduce barrier (median sync skew {d.sync_skew * 1e3:.1f} "
                f"ms/step)."
            ),
            suggestion=(
                "Balance per-rank work: even out shard sizes / sequence lengths, "
                "check for variable-cost batches, and avoid rank-dependent branches "
                "in the step."
            ),
        )
    ]


@rule
def comm_bound(ctx: DiagnosisContext) -> list[Finding]:
    d = ctx.distributed
    if not d:
        return []
    frac = d.mean_comm_fraction
    if frac < 0.25:
        return []
    sev = "high" if frac >= 0.45 else "med"
    return [
        Finding(
            code="DIST.COMM_BOUND",
            severity=sev,
            title="Communication dominates step time",
            detail=(
                f"Collective communication is {frac * 100:.0f}% of step wall time "
                f"across {d.world_size} ranks — the gradient all-reduce is not "
                f"hidden behind compute."
            ),
            suggestion=(
                "Overlap communication with backward (DDP gradient bucketing / "
                "`no_sync` for accumulation), increase the per-step compute (larger "
                "local batch), enable gradient compression, or check interconnect "
                "bandwidth (NVLink/InfiniBand vs Ethernet)."
            ),
        )
    ]


@rule
def pipeline_bubble(ctx: DiagnosisContext) -> list[Finding]:
    p = ctx.pipeline
    if not p:
        return []
    bubble_pct = p.bubble_fraction * 100.0
    excess = p.excess_bubble
    # If we know m, only flag *excess* bubble beyond the inherent GPipe minimum.
    if excess is not None:
        if excess < 0.05:
            return []
        sev = "high" if excess >= 0.15 else "med"
        ideal_pct = p.ideal_bubble_fraction * 100.0
        detail = (
            f"Pipeline idle (bubble) is {bubble_pct:.0f}% across {p.n_stages} "
            f"stages, but the inherent GPipe minimum for m={p.n_microbatches} "
            f"microbatches is only {ideal_pct:.0f}% — {excess * 100:.0f} points of "
            f"*excess* bubble from scheduling/imbalance, not from p and m."
        )
        suggestion = (
            "Reduce excess bubble: balance per-stage compute (split layers more "
            "evenly), check for a slow stage, or switch to an interleaved "
            "(1F1B) schedule."
        )
    else:
        if p.bubble_fraction < 0.25:
            return []
        sev = "high" if p.bubble_fraction >= 0.45 else "med"
        detail = (
            f"Pipeline idle (bubble) is {bubble_pct:.0f}% across {p.n_stages} "
            f"stages (stage {p.idlest_stage} is idlest). Each idle stage is wasted "
            f"accelerator time."
        )
        suggestion = (
            "Increase the number of microbatches to amortize warmup/cooldown "
            "(bubble ≈ (p-1)/(m+p-1)), balance per-stage compute, or use a 1F1B "
            "interleaved schedule."
        )
    return [
        Finding(
            code="DIST.PIPELINE_BUBBLE",
            severity=sev,
            title="Pipeline bubble is wasting accelerator time",
            detail=detail,
            suggestion=suggestion,
        )
    ]
