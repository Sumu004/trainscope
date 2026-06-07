"""Efficiency-budget rule — turns the wall-time decomposition into advice.

Unlike the single-axis rules, this one reads the whole budget and points at the
*largest recoverable line item*, so the suggestion is ranked by payoff (seconds
you could win back), and reports MFU when the hardware anchor is known.
"""

from __future__ import annotations

from .engine import DiagnosisContext, Finding, rule

_FIX_HINT = {
    "compute_overhead": "enable mixed precision (AMP/bf16), torch.compile, or "
    "fused kernels so compute approaches peak throughput",
    "data_stall": "raise DataLoader num_workers, persistent_workers, pin_memory, "
    "and prefetch; move heavy transforms off the hot path",
    "communication": "overlap the all-reduce with backward (DDP bucketing), or "
    "raise per-GPU compute so communication is hidden",
    "other": "profile the unattributed time — logging, checkpointing, host syncs, "
    "or eval inside the train loop",
}


@rule
def low_efficiency(ctx: DiagnosisContext) -> list[Finding]:
    b = ctx.efficiency
    if not b:
        return []
    top = b.top_recoverable
    if top is None or top.fraction < 0.10:
        return []  # nothing meaningfully recoverable

    if b.mfu is not None:
        # With an MFU anchor, flag when utilization is low.
        if b.mfu >= 0.50:
            return []
        sev = "high" if b.mfu < 0.30 else "med"
        recov = (1 - b.efficiency) * 100
        head = f"MFU is {b.mfu * 100:.0f}% — {recov:.0f}% of wall is recoverable"
        anchor = f"Useful compute is {b.efficiency * 100:.0f}% of wall (FLOPs at peak). "
    else:
        # No anchor: flag a dominant recoverable line.
        if top.fraction < 0.20:
            return []
        sev = "high" if top.fraction >= 0.35 else "med"
        head = f"{top.fraction * 100:.0f}% of wall time is recoverable ({top.name})"
        anchor = ""

    lines = ", ".join(
        f"{ln.name} {ln.fraction * 100:.0f}%" for ln in b.recoverable_lines[:3]
    )
    hint = _FIX_HINT.get(top.name, "")
    return [
        Finding(
            code="EFFICIENCY.LOW_MFU" if b.mfu is not None else "EFFICIENCY.RECOVERABLE",
            severity=sev,
            title=head,
            detail=(
                f"{anchor}Biggest recoverable line: {top.name} at "
                f"{top.fraction * 100:.0f}% of wall ({top.seconds:.2f}s over "
                f"{b.n_steps} steps). Top recoverable: {lines}."
            ),
            suggestion=f"Start with {top.name}: {hint}." if hint else "",
        )
    ]
