"""Cross-signal rules — the part no single-axis tool can do.

HTA sees only timing; Cockpit only gradients; W&B only logged scalars. Because
trainscope captures timing + memory + convergence on ONE aligned per-step
timeline, a rule can find a step that is anomalous across *several* axes at once
and report the correlation — far stronger evidence than any one signal alone.
"""

from __future__ import annotations

import math

from ..analyzers.stats import local_spikes
from .engine import DiagnosisContext, Finding, rule


def _series(steps, key: str) -> list[float | None]:
    return [s.scalars.get(key) for s in steps]


def _step_time_series(steps) -> list[float]:
    return [math.fsum(s.phases.values()) for s in steps]


def _alloc_series(steps) -> list[float | None]:
    return [s.memory.get("alloc") if s.memory else None for s in steps]


@rule
def correlated_instability(ctx: DiagnosisContext) -> list[Finding]:
    steps = ctx.steps
    if not steps or len(steps) < 8:
        return []

    losses = _series(steps, "loss")
    grads = _series(steps, "grad_norm")
    times = _step_time_series(steps)
    allocs = _alloc_series(steps)

    # Per-axis anomaly positions (index into `steps`).
    signals: dict[str, set] = {}
    if any(v is not None for v in losses):
        nonfinite = {
            i for i, v in enumerate(losses) if v is not None and not math.isfinite(v)
        }
        signals["loss"] = local_spikes(losses) | nonfinite
    if any(v is not None for v in grads):
        signals["grad_norm"] = local_spikes(grads)
    signals["step_time"] = local_spikes(times)
    if any(v is not None for v in allocs):
        signals["memory"] = local_spikes(allocs)

    # Need at least two distinct axes present to claim a correlation.
    present = [name for name, s in signals.items() if s is not None]
    if len(present) < 2:
        return []

    # Count, per step, how many axes flagged it.
    hit_axes: dict[int, list[str]] = {}
    for name, idxs in signals.items():
        for i in idxs:
            hit_axes.setdefault(i, []).append(name)

    correlated = sorted(i for i, axes in hit_axes.items() if len(axes) >= 2)
    if not correlated:
        return []

    # Collapse consecutive correlated steps (gap <= 2) into one event so a 3-step
    # blow-up is reported once as a range, not three near-identical findings.
    groups: list[list[int]] = []
    for i in correlated:
        if groups and i - groups[-1][-1] <= 2:
            groups[-1].append(i)
        else:
            groups.append([i])

    findings: list[Finding] = []
    for group in groups[:3]:  # report up to the first few distinct events
        # Representative = the most extreme step in the group (max loss if any).
        peak = max(
            group,
            key=lambda j: losses[j] if losses[j] is not None else 0.0,
        )
        axes = sorted(set().union(*(hit_axes[j] for j in group)))
        lo, hi = steps[group[0]].step, steps[group[-1]].step
        where = f"step {lo}" if lo == hi else f"steps {lo}–{hi}"

        parts = []
        if "loss" in axes and losses[peak] is not None:
            parts.append(f"loss={losses[peak]:.4g}")
        if "grad_norm" in axes and grads[peak] is not None:
            parts.append(f"grad_norm={grads[peak]:.4g}")
        if "step_time" in axes:
            parts.append(f"step_time={times[peak] * 1e3:.1f}ms")
        if "memory" in axes and allocs[peak] is not None:
            parts.append(f"alloc={allocs[peak] / (1024 * 1024):.0f}MB")

        detail = (
            f"At {where}, {len(axes)} independent axes spike simultaneously "
            f"({', '.join(axes)}): {', '.join(parts)}. Co-occurrence across axes is "
            "strong evidence of a real optimization event, not noise."
        )
        findings.append(
            Finding(
                code="CROSS.CORRELATED_INSTABILITY",
                severity="high",
                title=f"Correlated instability at {where}",
                detail=detail,
                suggestion=(
                    "Inspect the LR schedule, gradient clipping, and the specific "
                    f"batch around {where}. A simultaneous loss + grad-norm spike "
                    "usually means the update blew up (LR too high / bad batch)."
                ),
            )
        )
    return findings
