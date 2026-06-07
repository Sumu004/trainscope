"""Numerical-precision tests for the measurement core.

The tool's value is correct numbers, so these assert exact / near-exact results,
not loose tolerances.
"""

import math

from trainscope.analyzers.timing import _percentile, analyze_timing
from trainscope.core.events import StepRecord
from trainscope.profiler import Profiler


class NsClock:
    """Injected integer-ns clock."""

    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t

    def advance_ns(self, ns):
        self.t += ns


# --- integer-ns timing: no float cancellation, exact accumulation ----------
def test_phase_durations_are_exact_under_large_epoch(tmp_path):
    clk = NsClock()
    clk.t = 9_876_543_210_123_456  # large epoch — would wreck float perf_counter
    prof = Profiler(tmp_path, clock=clk)
    prof.start()

    prof.begin_step()
    clk.advance_ns(123)  # 123 ns forward
    prof.mark("forward")
    clk.advance_ns(456)
    prof.mark("backward")
    prof.end_step()
    prof.finish()

    from trainscope.core.store import RunStore

    rec = RunStore.load(tmp_path).steps[0]
    # 123 ns and 456 ns recovered exactly despite a ~9.8e15 ns epoch.
    assert rec.phases["forward"] == 123 / 1e9
    assert rec.phases["backward"] == 456 / 1e9


def test_gradient_accumulation_marks_add_exactly(tmp_path):
    """Repeated marks of the same phase must sum with no drift."""
    clk = NsClock()
    prof = Profiler(tmp_path, clock=clk)
    prof.start()
    prof.begin_step()
    for _ in range(1000):
        clk.advance_ns(7)  # 7 ns per micro-step
        prof.mark("backward")
        clk.advance_ns(0)
    prof.end_step()
    prof.finish()

    from trainscope.core.store import RunStore

    rec = RunStore.load(tmp_path).steps[0]
    assert rec.phases["backward"] == (1000 * 7) / 1e9  # exact: 7000 ns


# --- analyzer consistency: shared total, fsum reductions -------------------
def test_fractions_sum_to_one_and_mean_consistent():
    phases = {"data": 0.1, "forward": 0.2, "backward": 0.2}
    steps = [StepRecord(step=i, phases=dict(phases)) for i in range(1000)]
    s = analyze_timing(steps)
    assert abs(sum(s.phase_fractions.values()) - 1.0) < 1e-12
    # mean * n == grand total (consistency between the two derived quantities)
    assert abs(s.mean_step_time * s.n_steps - s.total_time) < 1e-9


def test_fsum_beats_naive_on_pathological_sequence():
    # Many tiny values + structure that naive float summation mis-handles.
    tiny = 1e-9
    steps = [StepRecord(step=i, phases={"forward": tiny}) for i in range(1_000_000)]
    s = analyze_timing(steps)
    expected_total = math.fsum([tiny] * 1_000_000)  # == 1e-3 essentially
    assert abs(s.total_time - expected_total) < 1e-15
    assert abs(s.mean_step_time - tiny) < 1e-18


# --- percentile correctness (numpy 'linear' convention) --------------------
def test_percentile_matches_linear_interpolation():
    vals = [10.0, 20.0, 30.0, 40.0]  # sorted
    assert _percentile(vals, 0) == 10.0
    assert _percentile(vals, 100) == 40.0
    assert _percentile(vals, 50) == 25.0  # midpoint of 20 and 30
    # q=75 -> pos=2.25 -> 30 + 0.25*(40-30) = 32.5
    assert _percentile(vals, 75) == 32.5


def test_percentile_single_and_empty():
    assert _percentile([42.0], 95) == 42.0
    assert _percentile([], 50) == 0.0


def test_variance_zero_for_constant_steps():
    steps = [StepRecord(step=i, phases={"forward": 0.01}) for i in range(50)]
    s = analyze_timing(steps)
    assert s.std_step_time == 0.0
    assert s.cv == 0.0
    assert s.p50_step_time == 0.01
    assert s.p95_step_time == 0.01
