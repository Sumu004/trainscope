"""Diagnosis engine — the layer that makes this more than another profiler.

Analyzers produce summaries; rules read those summaries and emit ranked,
actionable ``Finding``s. Rules register themselves with ``@rule`` so adding a
new heuristic is one decorated function, and cross-signal rules (timing + memory
+ convergence) live in the same registry once those analyzers land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..analyzers.convergence import ConvergenceSummary
from ..analyzers.distributed import DistributedSummary
from ..analyzers.memory import MemorySummary
from ..analyzers.pipeline import PipelineSummary
from ..analyzers.timing import TimingSummary
from ..core.events import StepRecord

SEVERITY_ORDER = {"high": 0, "med": 1, "low": 2}


@dataclass
class Finding:
    code: str
    severity: str  # "high" | "med" | "low"
    title: str
    detail: str
    suggestion: str = ""


@dataclass
class DiagnosisContext:
    """Everything the rules can read. Single-axis rules use the summaries;
    cross-signal rules read the aligned per-step ``steps`` timeline directly."""

    timing: TimingSummary | None = None
    memory: MemorySummary | None = None
    convergence: ConvergenceSummary | None = None
    steps: list[StepRecord] | None = None
    distributed: DistributedSummary | None = None
    pipeline: PipelineSummary | None = None


Rule = Callable[[DiagnosisContext], list[Finding]]
_RULES: list[Rule] = []


def rule(fn: Rule) -> Rule:
    _RULES.append(fn)
    return fn


def run_diagnosis(ctx: DiagnosisContext) -> list[Finding]:
    findings: list[Finding] = []
    for r in _RULES:
        try:
            out = r(ctx)
        except Exception:
            # A buggy rule must never sink the whole report.
            continue
        if out:
            findings.extend(out)
    findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    return findings
