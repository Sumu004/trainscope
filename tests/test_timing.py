from pytscope.analyzers.timing import analyze_timing
from pytscope.core.events import StepRecord


def _steps(n, phases):
    return [StepRecord(step=i, phases=dict(phases)) for i in range(n)]


def test_fractions_sum_to_one():
    steps = _steps(20, {"data": 0.1, "forward": 0.2, "backward": 0.2})
    s = analyze_timing(steps)
    assert s.n_steps == 20
    assert abs(sum(s.phase_fractions.values()) - 1.0) < 1e-9
    assert abs(s.phase_fractions["data"] - 0.2) < 1e-9
    assert abs(s.mean_step_time - 0.5) < 1e-9


def test_warmup_dropped():
    steps = _steps(10, {"data": 0.1, "forward": 0.1})
    s = analyze_timing(steps, warmup=4)
    assert s.n_steps == 6


def test_empty():
    s = analyze_timing([])
    assert s.n_steps == 0
    assert s.steps_per_sec == 0.0


def test_phase_order_canonical():
    steps = _steps(3, {"optimizer": 0.1, "data": 0.1, "forward": 0.1})
    s = analyze_timing(steps)
    assert s.phase_order == ["data", "forward", "optimizer"]
