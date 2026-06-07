"""Production scenario / stress tests — the messy real-world cases."""

import subprocess
import sys

import pytest

from trainscope import Profiler
from trainscope.core.events import StepRecord
from trainscope.core.store import RUN_META, RunStore


# --- crash resilience ------------------------------------------------------
def test_training_crash_preserves_recorded_steps(tmp_path):
    with pytest.raises(RuntimeError):
        with Profiler(tmp_path, collect_memory=False) as prof:
            for i in range(10):
                with prof.step():
                    prof.mark("forward")
                    if i == 5:
                        raise RuntimeError("boom")
    # __exit__ -> finish() ran even on exception; data is intact and analyzable.
    store = RunStore.load(tmp_path)
    assert len(store.steps) == 6  # steps 0..5 recorded
    assert "wall_time" in store.meta


def test_corrupt_run_json_degrades_gracefully(tmp_path):
    s = RunStore(tmp_path).open()
    s.append(StepRecord(step=0, phases={"forward": 0.1}))
    s.close()
    (tmp_path / RUN_META).write_text("{ this is not json", encoding="utf-8")
    store = RunStore.load(tmp_path)  # must not raise
    assert store.meta == {}
    assert len(store.steps) == 1


# --- degenerate inputs -----------------------------------------------------
def test_zero_duration_steps(tmp_path):
    clock = lambda: 0  # noqa: E731  — time never advances
    prof = Profiler(tmp_path, collect_memory=False, clock=clock)
    prof.start()
    for _ in range(5):
        with prof.step():
            prof.mark("forward")
    prof.finish()
    from trainscope.analyzers.timing import analyze_timing

    s = analyze_timing(RunStore.load(tmp_path).steps)
    assert s.n_steps == 5
    assert s.mean_step_time == 0.0  # no crash, no div-by-zero


def test_single_step_run(tmp_path):
    with Profiler(tmp_path, collect_memory=False) as prof:
        with prof.step():
            prof.mark("forward")
    assert len(RunStore.load(tmp_path).steps) == 1


def test_only_scalars_no_phases(tmp_path):
    with Profiler(tmp_path, collect_memory=False) as prof:
        for i in range(10):
            with prof.step():
                prof.log(loss=1.0 / (i + 1))
    from trainscope.analyzers.convergence import analyze_convergence

    c = analyze_convergence(RunStore.load(tmp_path).steps)
    assert c.has_loss and c.final_loss is not None


# --- misuse safety ---------------------------------------------------------
def test_mark_and_log_before_begin_step_are_noops(tmp_path):
    prof = Profiler(tmp_path, collect_memory=False)
    prof.start()
    prof.mark("forward")  # no active step — must not raise
    prof.log(loss=1.0)
    prof.finish()
    assert RunStore.load(tmp_path).steps == []


def test_double_finish_is_idempotent(tmp_path):
    prof = Profiler(tmp_path, collect_memory=False)
    prof.start()
    with prof.step():
        prof.mark("forward")
    prof.finish()
    prof.finish()  # must not raise or corrupt
    assert len(RunStore.load(tmp_path).steps) == 1


# --- CLI end-to-end via subprocess ----------------------------------------
def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "trainscope.cli", *args],
        capture_output=True,
        text=True,
    )


def _make_run(tmp_path, name, losses):
    s = RunStore(tmp_path)
    s.meta = {"name": name, "environment": {"torch": "2.6.0"}, "config": {}}
    s = s.open()
    for i, lv in enumerate(losses):
        s.append(StepRecord(step=i, phases={"forward": 0.01}, scalars={"loss": lv}))
    s.write_meta()
    s.close()


def test_cli_analyze_succeeds(tmp_path):
    d = tmp_path / "run"
    _make_run(d, "r", [1.0, 0.5, 0.25])
    res = _run_cli("analyze", str(d))
    assert res.returncode == 0
    assert "Run summary" in res.stdout


def test_cli_analyze_empty_dir_fails_cleanly(tmp_path):
    res = _run_cli("analyze", str(tmp_path / "nonexistent"))
    assert res.returncode == 1
    assert "No steps" in res.stderr


def test_cli_diff_succeeds(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _make_run(a, "a", [1.0, 0.5])
    _make_run(b, "b", [1.0, 0.6])
    res = _run_cli("diff", str(a), str(b))
    assert res.returncode == 0
    assert "Reproducibility" in res.stdout


def test_cli_no_command_fails(tmp_path):
    res = _run_cli()
    assert res.returncode != 0


# --- diff robustness -------------------------------------------------------
def test_diff_runs_of_different_lengths(tmp_path):
    from trainscope.analyzers.repro import diff_runs

    a, b = tmp_path / "a", tmp_path / "b"
    _make_run(a, "a", [1.0, 0.5, 0.25, 0.1])
    _make_run(b, "b", [1.0, 0.5])  # shorter
    d = diff_runs(RunStore.load(a), RunStore.load(b))
    assert d.first_divergence_step == 2  # common prefix matches, then length differs
