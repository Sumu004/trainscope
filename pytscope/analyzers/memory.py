"""Memory analyzer (vertical #2) — reads the per-step ``memory`` block.

Operates on whatever the memory collector stored (bytes): ``alloc``,
``reserved``, ``peak_alloc``, ``peak_reserved``. Pure functions over the
timeline, like the timing analyzer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.events import StepRecord
from .stats import robust_slope


@dataclass
class MemorySummary:
    has_memory: bool = False
    n_steps: int = 0
    peak_alloc_bytes: float = 0.0
    peak_reserved_bytes: float = 0.0
    mean_alloc_bytes: float = 0.0
    # Fraction of reserved memory not actually allocated — fragmentation/slack.
    fragmentation: float = 0.0
    # Robust per-step growth of allocated memory — a leak signal.
    growth_bytes_per_step: float = 0.0
    alloc_series: list[float] = field(default_factory=list)


def analyze_memory(steps: list[StepRecord]) -> MemorySummary:
    allocs = [s.memory["alloc"] for s in steps if "alloc" in s.memory]
    if not allocs:
        return MemorySummary(has_memory=False, n_steps=len(steps))

    reserved = [s.memory["reserved"] for s in steps if "reserved" in s.memory]
    peak_alloc = [
        s.memory.get("peak_alloc", s.memory.get("alloc", 0.0)) for s in steps if s.memory
    ]
    peak_reserved = [
        s.memory.get("peak_reserved", s.memory.get("reserved", 0.0))
        for s in steps
        if s.memory
    ]

    frag_samples = [(r - a) / r for a, r in zip(allocs, reserved) if r > 0]
    fragmentation = math.fsum(frag_samples) / len(frag_samples) if frag_samples else 0.0

    return MemorySummary(
        has_memory=True,
        n_steps=len(steps),
        peak_alloc_bytes=max(peak_alloc) if peak_alloc else max(allocs),
        peak_reserved_bytes=max(peak_reserved) if peak_reserved else 0.0,
        mean_alloc_bytes=math.fsum(allocs) / len(allocs),
        fragmentation=fragmentation,
        growth_bytes_per_step=robust_slope(allocs),
        alloc_series=allocs,
    )
