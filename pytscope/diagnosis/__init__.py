# Importing the rule modules registers them via the @rule decorator.
from . import (
    rules,  # noqa: E402,F401  (timing)
    rules_convergence,  # noqa: E402,F401
    rules_cross,  # noqa: E402,F401
    rules_distributed,  # noqa: E402,F401
    rules_efficiency,  # noqa: E402,F401
    rules_memory,  # noqa: E402,F401
)
from .engine import DiagnosisContext, Finding, rule, run_diagnosis

__all__ = ["DiagnosisContext", "Finding", "run_diagnosis", "rule"]
