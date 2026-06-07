"""Timing rules for the diagnosis engine.

Each rule is a small, honest heuristic over the timing summary. Thresholds are
deliberately conservative; the goal is high-precision findings a practitioner
would actually act on, not a wall of speculative warnings.
"""

from __future__ import annotations

from ..core.events import BACKWARD, DATA, FORWARD, OPTIMIZER
from .engine import DiagnosisContext, Finding, rule


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


@rule
def dataloader_bound(ctx: DiagnosisContext) -> list[Finding]:
    t = ctx.timing
    if not t or DATA not in t.phase_fractions:
        return []
    frac = t.phase_fractions[DATA]
    if frac < 0.25:
        return []
    sev = "high" if frac >= 0.40 else "med"
    return [
        Finding(
            code="TIMING.DATALOADER_BOUND",
            severity=sev,
            title="Input pipeline is a bottleneck",
            detail=(
                f"{_pct(frac)} of step time is spent fetching data "
                f"({t.phase_seconds[DATA] * 1e3:.1f} ms/step). The accelerator is "
                f"stalling on the dataloader."
            ),
            suggestion=(
                "Raise DataLoader num_workers, set persistent_workers=True and "
                "pin_memory=True, prefetch, or move heavy transforms off the hot "
                "path / onto the GPU."
            ),
        )
    ]


@rule
def backward_heavy(ctx: DiagnosisContext) -> list[Finding]:
    t = ctx.timing
    if not t:
        return []
    fwd = t.phase_seconds.get(FORWARD, 0.0)
    bwd = t.phase_seconds.get(BACKWARD, 0.0)
    if fwd <= 0 or bwd <= 0:
        return []
    ratio = bwd / fwd
    # Backward ~2x forward is normal; only flag clearly abnormal ratios.
    if ratio < 2.5:
        return []
    sev = "med" if ratio < 3.5 else "high"
    return [
        Finding(
            code="TIMING.BACKWARD_HEAVY",
            severity=sev,
            title="Backward pass disproportionately slow",
            detail=(
                f"Backward is {ratio:.1f}x forward (normal is ~2x). "
                "Often caused by retain_graph=True, unfused elementwise ops, or "
                "activation recomputation you didn't intend."
            ),
            suggestion=(
                "Check for stray retain_graph/create_graph, enable AMP, and "
                "consider torch.compile to fuse backward ops."
            ),
        )
    ]


@rule
def optimizer_heavy(ctx: DiagnosisContext) -> list[Finding]:
    t = ctx.timing
    if not t:
        return []
    frac = t.phase_fractions.get(OPTIMIZER, 0.0)
    if frac < 0.15:
        return []
    return [
        Finding(
            code="TIMING.OPTIMIZER_HEAVY",
            severity="low",
            title="Optimizer step is costly",
            detail=(
                f"{_pct(frac)} of step time is the optimizer. For small models "
                "this can dominate; for large ones it may signal a Python-side "
                "per-parameter loop."
            ),
            suggestion=(
                "Use a fused/foreach optimizer (foreach=True or fused=True) to "
                "batch the parameter updates."
            ),
        )
    ]


@rule
def step_time_jitter(ctx: DiagnosisContext) -> list[Finding]:
    t = ctx.timing
    if not t or t.n_steps < 10:
        return []
    # Trigger on either spread (CV) or a heavy tail (p95/median) — the tail
    # ratio catches occasional stragglers that a modest CV would hide.
    if t.cv < 0.50 and t.p95_p50_ratio < 1.5:
        return []
    return [
        Finding(
            code="TIMING.JITTER",
            severity="low",
            title="High step-time variance",
            detail=(
                f"Step time is uneven (CV={t.cv:.2f}, "
                f"p95/median={t.p95_p50_ratio:.2f}). Stragglers point to "
                "input-pipeline stalls, GC pauses, logging/checkpoint hitches, or "
                "uneven batch sizes."
            ),
            suggestion=(
                "Profile a few slow steps specifically; move checkpointing and "
                "logging off the critical path."
            ),
        )
    ]
