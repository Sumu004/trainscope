"""Convergence rules (vertical #3)."""

from __future__ import annotations

from .engine import DiagnosisContext, Finding, rule


@rule
def loss_diverged(ctx: DiagnosisContext) -> list[Finding]:
    c = ctx.convergence
    if not c or not c.has_loss or c.diverged_at is None:
        return []
    return [
        Finding(
            code="CONVERGENCE.DIVERGED",
            severity="high",
            title="Loss diverged (NaN/Inf)",
            detail=(
                f"Loss became non-finite at step {c.diverged_at}. Training is "
                "broken from this point — later metrics are meaningless."
            ),
            suggestion=(
                "Lower the learning rate, add gradient clipping, check for bad "
                "inputs / log(0) / divide-by-zero in the loss, and verify AMP "
                "loss scaling."
            ),
        )
    ]


@rule
def loss_not_improving(ctx: DiagnosisContext) -> list[Finding]:
    c = ctx.convergence
    if not c or not c.has_loss or c.loss_trend not in ("worsening", "plateau"):
        return []
    if c.n_steps < 50:  # too early to call a plateau
        return []
    sev = "high" if c.loss_trend == "worsening" else "low"
    title = "Loss is increasing" if c.loss_trend == "worsening" else "Loss has plateaued"
    return [
        Finding(
            code="CONVERGENCE.NO_PROGRESS",
            severity=sev,
            title=title,
            detail=(
                f"Loss trend over the run is '{c.loss_trend}' "
                f"(best {c.best_loss:.4g}, final {c.final_loss:.4g})."
            ),
            suggestion=(
                "Worsening: likely LR too high or unstable optimizer. Plateau: try "
                "LR decay, more capacity, or check the data is still informative."
            ),
        )
    ]
