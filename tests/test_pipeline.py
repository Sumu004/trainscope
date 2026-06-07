"""Pipeline-bubble analyzer: closed-form correctness + excess-bubble rule."""

from __future__ import annotations

import pytest

from trainscope.analyzers.pipeline import analyze_pipeline, gpipe_schedule
from trainscope.diagnosis.engine import DiagnosisContext, run_diagnosis


@pytest.mark.parametrize("p", [2, 4, 8])
@pytest.mark.parametrize("m", [1, 4, 16, 64])
def test_bubble_matches_closed_form(p, m):
    # A uniform forward-only GPipe schedule must reproduce (p-1)/(m+p-1) exactly.
    sched = gpipe_schedule(p, m, forward=1.0)
    s = analyze_pipeline(sched, n_microbatches=m)
    ideal = (p - 1) / (m + p - 1)
    assert abs(s.bubble_fraction - ideal) < 1e-9
    assert abs(s.ideal_bubble_fraction - ideal) < 1e-12
    assert s.excess_bubble < 1e-9  # a perfect schedule has no excess


def test_more_microbatches_shrink_bubble():
    b = [
        analyze_pipeline(gpipe_schedule(4, m), n_microbatches=m).bubble_fraction
        for m in (2, 8, 32)
    ]
    assert b[0] > b[1] > b[2]


def test_excess_bubble_from_slow_stage():
    # Build a schedule then inflate one stage's makespan with idle (a slow stage
    # that delays everyone) → achieved bubble exceeds the inherent minimum.
    p, m = 4, 16
    sched = gpipe_schedule(p, m, forward=1.0)
    # Shift stage-3 intervals later (it starts late / runs slow), stretching span.
    inflated = [
        (st, s0 + (5.0 if st == 3 else 0.0), s1 + (5.0 if st == 3 else 0.0))
        for (st, s0, s1) in sched
    ]
    s = analyze_pipeline(inflated, n_microbatches=m)
    assert s.excess_bubble > 0.05
    findings = {f.code for f in run_diagnosis(DiagnosisContext(pipeline=s))}
    assert "DIST.PIPELINE_BUBBLE" in findings


def test_healthy_pipeline_no_finding():
    p, m = 4, 64  # many microbatches → small inherent bubble, no excess
    s = analyze_pipeline(gpipe_schedule(p, m), n_microbatches=m)
    findings = {f.code for f in run_diagnosis(DiagnosisContext(pipeline=s))}
    assert "DIST.PIPELINE_BUBBLE" not in findings


def test_degenerate_inputs():
    assert analyze_pipeline([]) is None
    assert analyze_pipeline([(0, 0.0, 1.0)]) is None  # single stage
    # zero makespan
    assert analyze_pipeline([(0, 1.0, 1.0), (1, 1.0, 1.0)]) is None
