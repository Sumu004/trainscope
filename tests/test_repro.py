"""Reproducibility diff tests (vertical #4)."""

from pytscope.analyzers.repro import diff_runs
from pytscope.core.events import StepRecord
from pytscope.core.store import RunStore


def _store(name, losses, env=None, config=None):
    s = RunStore("/tmp/_unused")  # not opened; we only use it in-memory
    s.meta = {"name": name, "environment": env or {}, "config": config or {}}
    s.steps = [
        StepRecord(step=i, phases={"forward": 0.01}, scalars={"loss": lv})
        for i, lv in enumerate(losses)
    ]
    return s


def _notes(diff):
    return " ".join(diff.notes)


def test_identical_runs_are_reproducible():
    env = {"torch": "2.6.0", "env": {"PYTHONHASHSEED": "0"}}
    a = _store("a", [1.0, 0.5, 0.25], env=env)
    b = _store("b", [1.0, 0.5, 0.25], env=env)
    d = diff_runs(a, b)
    assert d.identical_trajectory
    assert d.first_divergence_step is None
    assert not d.env_diffs
    assert "Reproducible" in _notes(d)


def test_config_difference_is_expected():
    a = _store("a", [1.0, 0.5], config={"lr": 0.1})
    b = _store("b", [1.0, 0.9], config={"lr": 0.5})
    d = diff_runs(a, b)
    assert any(fd.key == "lr" for fd in d.config_diffs)
    assert "expected" in _notes(d)


def test_env_determinism_difference_attributed():
    a = _store("a", [1.0, 0.5], env={"cudnn_benchmark": False, "torch": "2.6.0"})
    b = _store("b", [1.0, 0.6], env={"cudnn_benchmark": True, "torch": "2.6.0"})
    d = diff_runs(a, b)
    assert "determinism-relevant" in _notes(d)
    assert "cudnn.benchmark is ON" in _notes(d)


def test_nondeterminism_when_env_and_config_identical():
    env = {"torch": "2.6.0", "env": {"PYTHONHASHSEED": "0"}}
    a = _store("a", [1.0, 0.5, 0.25], env=env)
    b = _store("b", [1.0, 0.5, 0.27], env=env)  # diverges at step 2
    d = diff_runs(a, b)
    assert not d.identical_trajectory
    assert d.first_divergence_step == 2
    assert "NONDETERMINISM" in _notes(d)


def test_first_divergence_step_detection():
    a = _store("a", [1.0, 0.9, 0.8, 0.7])
    b = _store("b", [1.0, 0.9, 0.85, 0.7])
    d = diff_runs(a, b)
    assert d.first_divergence_step == 2


def test_metric_diffs_present():
    a = _store("a", [1.0, 0.5])
    b = _store("b", [1.0, 0.4])
    d = diff_runs(a, b)
    keys = {m.key for m in d.metric_diffs}
    assert {"final_loss", "best_loss", "mean_step_ms", "n_steps"} <= keys
    final = next(m for m in d.metric_diffs if m.key == "final_loss")
    assert abs(final.delta - (-0.1)) < 1e-9
