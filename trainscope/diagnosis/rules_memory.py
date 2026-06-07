"""Memory rules (vertical #2)."""

from __future__ import annotations

from .engine import DiagnosisContext, Finding, rule

_MB = 1024 * 1024


@rule
def memory_fragmentation(ctx: DiagnosisContext) -> list[Finding]:
    m = ctx.memory
    if not m or not m.has_memory:
        return []
    # Below ~64 MB allocated, the alloc/reserved ratio is dominated by allocator
    # bookkeeping (and on MPS the step-boundary alloc reads near zero) — not a
    # real memory-pressure signal, so don't flag it.
    if m.peak_alloc_bytes < 64 * _MB:
        return []
    if m.fragmentation < 0.20:
        return []
    sev = "med" if m.fragmentation >= 0.40 else "low"
    return [
        Finding(
            code="MEMORY.FRAGMENTATION",
            severity=sev,
            title="High allocator fragmentation",
            detail=(
                f"On average {m.fragmentation * 100:.0f}% of reserved device "
                f"memory is not in use (reserved peak {m.peak_reserved_bytes / _MB:.0f} "
                f"MB vs alloc peak {m.peak_alloc_bytes / _MB:.0f} MB). Fragmentation "
                "wastes capacity and can trigger OOMs despite 'free' memory."
            ),
            suggestion=(
                "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; avoid wildly "
                "varying tensor shapes; call torch.cuda.empty_cache() sparingly."
            ),
        )
    ]


@rule
def memory_growth(ctx: DiagnosisContext) -> list[Finding]:
    m = ctx.memory
    if not m or not m.has_memory or m.n_steps < 20:
        return []
    growth_per_step = m.growth_bytes_per_step
    # Flag only growth that is material over the run (> 1% of peak, > ~1 MB/step).
    total_growth = growth_per_step * m.n_steps
    if growth_per_step <= 0 or total_growth < 0.01 * max(m.peak_alloc_bytes, 1):
        return []
    if growth_per_step < _MB:
        return []
    return [
        Finding(
            code="MEMORY.GROWTH",
            severity="high",
            title="Allocated memory grows steadily (possible leak)",
            detail=(
                f"Allocated memory rises ~{growth_per_step / _MB:.1f} MB/step "
                f"(~{total_growth / _MB:.0f} MB over {m.n_steps} steps). Steady "
                "growth in steady-state training usually means tensors are "
                "retained across steps."
            ),
            suggestion=(
                "Detach/`.item()` metrics before accumulating; ensure you're not "
                "appending graph-attached tensors to a list; check for retained "
                "activations or growing caches."
            ),
        )
    ]
