"""Reproducibility analyzer (vertical #4) — compare two runs and explain drift.

Diffs the captured provenance (environment), the user config, and the outcome
metrics of two runs, then interprets *why* they differ: an expected config
change, an environment change that perturbs numerics, or — the important case —
identical inputs producing different outputs, i.e. nondeterminism.

It also finds the first step where the two loss trajectories diverge, which is
the single most useful number when chasing a reproducibility bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.store import RunStore
from .convergence import analyze_convergence
from .memory import analyze_memory
from .timing import analyze_timing

# Missing-on-one-side sentinel (distinct from a real None value).
MISSING = "<absent>"

# Environment keys whose change can legitimately alter numerics / determinism.
DETERMINISM_KEYS = {
    "torch",
    "cuda",
    "gpu",
    "gpu_count",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "env.PYTHONHASHSEED",
    "env.CUBLAS_WORKSPACE_CONFIG",
}


@dataclass
class FieldDiff:
    key: str
    a: Any
    b: Any


@dataclass
class MetricDiff:
    key: str
    a: float | None
    b: float | None

    @property
    def delta(self) -> float | None:
        if self.a is None or self.b is None:
            return None
        return self.b - self.a


@dataclass
class RunDiff:
    name_a: str
    name_b: str
    env_diffs: list[FieldDiff] = field(default_factory=list)
    config_diffs: list[FieldDiff] = field(default_factory=list)
    metric_diffs: list[MetricDiff] = field(default_factory=list)
    first_divergence_step: int | None = None
    identical_trajectory: bool = False
    notes: list[str] = field(default_factory=list)


def _flatten_env(env: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in env.items():
        if k == "env" and isinstance(v, dict):
            for ek, ev in v.items():
                out[f"env.{ek}"] = ev
        else:
            out[k] = v
    return out


def _dict_diffs(a: dict[str, Any], b: dict[str, Any]) -> list[FieldDiff]:
    diffs = []
    for key in sorted(set(a) | set(b)):
        va = a.get(key, MISSING)
        vb = b.get(key, MISSING)
        if va != vb:
            diffs.append(FieldDiff(key, va, vb))
    return diffs


def _losses(store: RunStore) -> list[float]:
    return [s.scalars["loss"] for s in store.steps if "loss" in s.scalars]


def _first_divergence(la: list[float], lb: list[float]) -> int | None:
    for i in range(min(len(la), len(lb))):
        if la[i] != lb[i]:
            return i
    if len(la) != len(lb):
        return min(len(la), len(lb))
    return None


def diff_runs(store_a: RunStore, store_b: RunStore) -> RunDiff:
    name_a = store_a.meta.get("name", "A")
    name_b = store_b.meta.get("name", "B")

    env_a = _flatten_env(store_a.meta.get("environment", {}))
    env_b = _flatten_env(store_b.meta.get("environment", {}))
    env_diffs = _dict_diffs(env_a, env_b)
    config_diffs = _dict_diffs(
        store_a.meta.get("config", {}), store_b.meta.get("config", {})
    )

    ca, cb = analyze_convergence(store_a.steps), analyze_convergence(store_b.steps)
    ta, tb = analyze_timing(store_a.steps), analyze_timing(store_b.steps)
    ma, mb = analyze_memory(store_a.steps), analyze_memory(store_b.steps)

    metric_diffs = [
        MetricDiff("final_loss", ca.final_loss, cb.final_loss),
        MetricDiff("best_loss", ca.best_loss, cb.best_loss),
        MetricDiff("mean_step_ms", ta.mean_step_time * 1e3, tb.mean_step_time * 1e3),
        MetricDiff("steps_per_sec", ta.steps_per_sec, tb.steps_per_sec),
        MetricDiff(
            "peak_alloc_mb",
            ma.peak_alloc_bytes / (1024 * 1024) if ma.has_memory else None,
            mb.peak_alloc_bytes / (1024 * 1024) if mb.has_memory else None,
        ),
        MetricDiff("n_steps", float(ta.n_steps), float(tb.n_steps)),
    ]

    la, lb = _losses(store_a), _losses(store_b)
    first_div = _first_divergence(la, lb)
    identical = bool(la) and la == lb

    diff = RunDiff(
        name_a=name_a,
        name_b=name_b,
        env_diffs=env_diffs,
        config_diffs=config_diffs,
        metric_diffs=metric_diffs,
        first_divergence_step=first_div,
        identical_trajectory=identical,
    )
    diff.notes = _interpret(diff, env_a, env_b)
    return diff


def _interpret(diff: RunDiff, env_a: dict[str, Any], env_b: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    det_diffs = [d for d in diff.env_diffs if d.key in DETERMINISM_KEYS]

    if diff.identical_trajectory and not diff.env_diffs and not diff.config_diffs:
        notes.append(
            "Reproducible: identical environment, config, and step-for-step loss "
            "trajectory."
        )
        return notes

    # Priority of explanation: config change > environment change > nondeterminism.
    if diff.config_diffs:
        keys = ", ".join(d.key for d in diff.config_diffs)
        notes.append(
            f"Config differs ({keys}); outcome differences are expected, not a "
            "reproducibility problem."
        )
    elif det_diffs:
        keys = ", ".join(d.key for d in det_diffs)
        notes.append(
            f"Environment differs in determinism-relevant fields ({keys}); these "
            "can change numerics across otherwise-identical runs."
        )
    elif not diff.identical_trajectory:
        notes.append(
            "NONDETERMINISM: identical config and environment, but the loss "
            "trajectories differ. Set all seeds, enable "
            "torch.backends.cudnn.deterministic, set CUBLAS_WORKSPACE_CONFIG, and "
            "disable cudnn.benchmark to make runs bit-reproducible."
        )

    # Always flag known nondeterminism switches if either side has them on.
    for env, side in ((env_a, diff.name_a), (env_b, diff.name_b)):
        if env.get("cudnn_benchmark") is True:
            notes.append(
                f"cudnn.benchmark is ON in '{side}' — a common source of "
                "run-to-run nondeterminism."
            )
        if "env.PYTHONHASHSEED" not in env:
            notes.append(
                f"PYTHONHASHSEED is unset in '{side}' — set it for reproducibility."
            )

    if diff.first_divergence_step is not None and not diff.identical_trajectory:
        notes.append(
            f"Loss trajectories first diverge at step {diff.first_divergence_step}."
        )

    # Dedupe identical messages (e.g. same env warning for both sides) in order.
    seen = set()
    deduped = []
    for n in notes:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped
