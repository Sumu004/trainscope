"""Convergence analyzer (vertical #3) — reads per-step ``scalars``.

Consumes loss / grad_norm / lr that the user logs via ``prof.log(...)`` (or the
Lightning callback's loss). Pure functions over the timeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.events import StepRecord
from .stats import local_spikes, median


@dataclass
class ConvergenceSummary:
    has_loss: bool = False
    has_grad_norm: bool = False
    n_steps: int = 0
    final_loss: float | None = None
    best_loss: float | None = None
    loss_trend: str = "unknown"  # improving | plateau | worsening | diverged | unknown
    diverged_at: int | None = None  # step index of first non-finite loss
    loss_spikes: list[int] = field(default_factory=list)  # step indices
    grad_norm_spikes: list[int] = field(default_factory=list)


def _trend(losses: list[float], tol: float = 0.02) -> str:
    n = len(losses)
    if n < 4:
        return "unknown"
    w = max(1, n // 10)
    first = median(losses[:w])
    last = median(losses[-w:])
    denom = abs(first) if first != 0 else 1.0
    rel = (last - first) / denom
    if rel < -tol:
        return "improving"
    if rel > tol:
        return "worsening"
    return "plateau"


def analyze_convergence(steps: list[StepRecord]) -> ConvergenceSummary:
    loss_pairs = [(s.step, s.scalars["loss"]) for s in steps if "loss" in s.scalars]
    grad_series = [s.scalars.get("grad_norm") for s in steps]
    has_grad = any(g is not None for g in grad_series)

    if not loss_pairs:
        return ConvergenceSummary(
            has_loss=False, has_grad_norm=has_grad, n_steps=len(steps)
        )

    steps_idx = [st for st, _ in loss_pairs]
    losses = [lv for _, lv in loss_pairs]

    diverged_at = next((st for st, lv in loss_pairs if not math.isfinite(lv)), None)
    finite = [lv for lv in losses if math.isfinite(lv)]

    # Spike indices map back to the original step numbers.
    loss_spike_pos = local_spikes(losses)
    loss_spikes = [steps_idx[i] for i in sorted(loss_spike_pos)]
    grad_spike_pos = local_spikes(grad_series) if has_grad else set()
    grad_spikes = [steps[i].step for i in sorted(grad_spike_pos)]

    return ConvergenceSummary(
        has_loss=True,
        has_grad_norm=has_grad,
        n_steps=len(steps),
        final_loss=finite[-1] if finite else None,
        best_loss=min(finite) if finite else None,
        loss_trend="diverged" if diverged_at is not None else _trend(finite),
        diverged_at=diverged_at,
        loss_spikes=loss_spikes,
        grad_norm_spikes=grad_spikes,
    )
