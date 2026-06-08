"""Distributed analyzer: critical-path, straggler persistence, comm, imbalance."""

from __future__ import annotations

import random

from pytscope.analyzers.distributed import (
    analyze_distributed,
    is_multirank,
    load_multirank,
)
from pytscope.core.events import StepRecord
from pytscope.core.store import RunStore
from pytscope.diagnosis.engine import DiagnosisContext, run_diagnosis


def _make_run(tmp_path, world_size, steps, compute_fn, comm=0.003, seed=0):
    """compute_fn(rank, step) -> local compute seconds. Returns run dir."""
    rng = random.Random(seed)
    base = tmp_path / "run"
    for r in range(world_size):
        st = RunStore(base / f"rank{r}", meta={"rank": r, "world_size": world_size})
        st.open()
        for s in range(steps):
            comp = compute_fn(r, s, rng)
            st.append(
                StepRecord(
                    step=s,
                    phases={"forward": comp * 0.4, "backward": comp * 0.6, "comm": comm},
                )
            )
        st.write_meta()
        st.close()
    return base


def test_is_multirank_and_load(tmp_path):
    base = _make_run(tmp_path, 3, 10, lambda r, s, rng: 0.01)
    assert is_multirank(base)
    ranks = load_multirank(base)
    assert set(ranks) == {0, 1, 2}
    assert all(len(ranks[r].steps) == 10 for r in ranks)


def test_single_rank_is_not_distributed(tmp_path):
    base = _make_run(tmp_path, 1, 10, lambda r, s, rng: 0.01)
    # world_size < 2 → analyzer returns None (nothing to correlate).
    assert analyze_distributed(load_multirank(base)) is None


def test_persistent_straggler_detected(tmp_path):
    # Rank 2 is consistently 25% slower than the others.
    def compute(r, s, rng):
        base = 0.010 + rng.gauss(0, 0.0003)
        return base * (1.25 if r == 2 else 1.0)

    base = _make_run(tmp_path, 4, 100, compute, seed=1)
    d = analyze_distributed(load_multirank(base))
    assert d is not None
    assert d.straggler is not None
    assert d.straggler.rank == 2
    assert d.straggler.straggler_z > 3.0
    assert d.straggler.rel_slowdown > 0.15
    assert d.wall_frac_lost_to_imbalance > 0.0


def test_balanced_run_has_no_straggler(tmp_path):
    # All ranks statistically identical: no persistent straggler, no false alarm.
    def compute(r, s, rng):
        return 0.010 + rng.gauss(0, 0.0005)

    base = _make_run(tmp_path, 4, 120, compute, seed=2)
    d = analyze_distributed(load_multirank(base))
    assert d is not None
    assert d.straggler is None  # noise must not be flagged
    assert d.imbalance_cv < 0.1


def test_straggler_rule_fires(tmp_path):
    def compute(r, s, rng):
        base = 0.010 + rng.gauss(0, 0.0003)
        return base * (1.30 if r == 1 else 1.0)

    base = _make_run(tmp_path, 4, 100, compute, seed=3)
    d = analyze_distributed(load_multirank(base))
    findings = run_diagnosis(DiagnosisContext(distributed=d))
    codes = {f.code for f in findings}
    assert "DIST.STRAGGLER" in codes
    straggler = next(f for f in findings if f.code == "DIST.STRAGGLER")
    assert "Rank 1" in straggler.title


def test_comm_bound_rule_fires(tmp_path):
    # Tiny compute, large comm → communication-bound.
    base = _make_run(tmp_path, 4, 60, lambda r, s, rng: 0.001, comm=0.010, seed=4)
    d = analyze_distributed(load_multirank(base))
    findings = run_diagnosis(DiagnosisContext(distributed=d))
    assert "DIST.COMM_BOUND" in {f.code for f in findings}


def test_load_imbalance_without_single_culprit(tmp_path):
    # Two slow ranks alternate as critical path → imbalance but no one straggler.
    def compute(r, s, rng):
        slow = (r == 0 and s % 2 == 0) or (r == 1 and s % 2 == 1)
        return (0.010 + rng.gauss(0, 0.0002)) * (1.4 if slow else 1.0)

    base = _make_run(tmp_path, 4, 120, compute, comm=0.001, seed=5)
    d = analyze_distributed(load_multirank(base))
    findings = {f.code for f in run_diagnosis(DiagnosisContext(distributed=d))}
    # No single persistent straggler, but imbalance should surface.
    assert d.straggler is None
    assert "DIST.LOAD_IMBALANCE" in findings


def test_critical_path_wall_loss_is_bounded(tmp_path):
    base = _make_run(tmp_path, 4, 50, lambda r, s, rng: 0.01, seed=6)
    d = analyze_distributed(load_multirank(base))
    assert 0.0 <= d.wall_frac_lost_to_imbalance < 1.0
