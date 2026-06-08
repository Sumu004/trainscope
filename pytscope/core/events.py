"""Core event schema: the aligned per-step record everything is built on.

A run is a sequence of ``StepRecord``s. Every analyzer (timing, memory,
convergence, repro) reads from this same record, sampled at the training-step
boundary, so signals stay aligned on one timeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Canonical phase names. Collectors may emit others; these just control
# display order and let rules reason about known phases.
DATA = "data"
FORWARD = "forward"
BACKWARD = "backward"
OPTIMIZER = "optimizer"
COMPUTE = "compute"  # coarse fwd+bwd, used when a backend can't split them
COMM = "comm"  # collective communication (all-reduce / barrier) in distributed runs
OTHER = "other"

CANONICAL_ORDER = [DATA, FORWARD, BACKWARD, OPTIMIZER, COMPUTE, COMM, OTHER]

# Phases that represent local compute (everything that is *not* waiting on the
# network). Used by the distributed analyzer to separate compute from comm.
COMPUTE_PHASES = (DATA, FORWARD, BACKWARD, OPTIMIZER, COMPUTE, OTHER)


@dataclass
class StepRecord:
    """One training step, with time attributed to phases plus optional signals."""

    step: int
    phases: dict[str, float] = field(default_factory=dict)  # seconds per phase
    scalars: dict[str, float] = field(default_factory=dict)  # loss, grad_norm, lr...
    memory: dict[str, float] = field(default_factory=dict)  # bytes, for vertical #2
    timestamp: float = field(default_factory=time.time)

    def total(self) -> float:
        """Total attributed wall time for this step (seconds)."""
        return sum(self.phases.values())

    def to_json_dict(self) -> dict[str, Any]:
        """Hot-path serialization. Avoids ``dataclasses.asdict`` (deep-copy) and
        omits empty collections to shrink the file and the work per step."""
        d: dict[str, Any] = {"step": self.step, "timestamp": self.timestamp}
        if self.phases:
            d["phases"] = self.phases
        if self.scalars:
            d["scalars"] = self.scalars
        if self.memory:
            d["memory"] = self.memory
        return d

    # Back-compat alias.
    def to_dict(self) -> dict[str, Any]:
        return self.to_json_dict()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepRecord:
        return cls(
            step=d["step"],
            phases=dict(d.get("phases", {})),
            scalars=dict(d.get("scalars", {})),
            memory=dict(d.get("memory", {})),
            timestamp=d.get("timestamp", 0.0),
        )


def order_phases(names) -> list:
    """Return phase names in canonical order, with unknown ones appended sorted."""
    known = [p for p in CANONICAL_ORDER if p in names]
    rest = sorted(n for n in names if n not in CANONICAL_ORDER)
    return known + rest
