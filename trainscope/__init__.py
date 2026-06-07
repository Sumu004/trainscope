"""trainscope — an intelligence layer for ML training.

One telemetry backbone (aligned per-step records) feeds pluggable analyzers
(timing today; memory, convergence, reproducibility next) and a diagnosis engine
that turns raw numbers into ranked, actionable findings.
"""

from __future__ import annotations

from .analyzers.convergence import ConvergenceSummary, analyze_convergence
from .analyzers.distributed import (
    DistributedSummary,
    analyze_distributed,
    load_multirank,
)
from .analyzers.memory import MemorySummary, analyze_memory
from .analyzers.pipeline import PipelineSummary, analyze_pipeline
from .analyzers.timing import TimingSummary, analyze_timing
from .analyzers.trace import TraceSummary, analyze_trace, analyze_trace_file
from .auto import AutoProfiler
from .core.events import StepRecord
from .core.store import RunStore
from .diagnosis.engine import DiagnosisContext, Finding, run_diagnosis
from .profiler import Profiler

__version__ = "0.1.0"

__all__ = [
    "Profiler",
    "AutoProfiler",
    "RunStore",
    "StepRecord",
    "analyze_timing",
    "TimingSummary",
    "analyze_memory",
    "MemorySummary",
    "analyze_convergence",
    "ConvergenceSummary",
    "analyze_distributed",
    "DistributedSummary",
    "load_multirank",
    "analyze_pipeline",
    "PipelineSummary",
    "analyze_trace",
    "analyze_trace_file",
    "TraceSummary",
    "run_diagnosis",
    "DiagnosisContext",
    "Finding",
    "__version__",
]
