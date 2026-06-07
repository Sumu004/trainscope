"""Tests for memory + convergence analyzers and the cross-signal rule."""

from trainscope.analyzers.convergence import analyze_convergence
from trainscope.analyzers.memory import analyze_memory
from trainscope.core.events import StepRecord
from trainscope.diagnosis.engine import DiagnosisContext, run_diagnosis


def _codes(findings):
    return {f.code for f in findings}


# --- convergence -----------------------------------------------------------
def test_convergence_detects_divergence():
    steps = [StepRecord(step=i, scalars={"loss": 1.0 / (i + 1)}) for i in range(10)]
    steps.append(StepRecord(step=10, scalars={"loss": float("nan")}))
    c = analyze_convergence(steps)
    assert c.loss_trend == "diverged"
    assert c.diverged_at == 10
    findings = run_diagnosis(DiagnosisContext(convergence=c))
    assert "CONVERGENCE.DIVERGED" in _codes(findings)


def test_convergence_improving_trend():
    steps = [StepRecord(step=i, scalars={"loss": 2.0 - i * 0.01}) for i in range(100)]
    c = analyze_convergence(steps)
    assert c.loss_trend == "improving"
    assert c.best_loss < c.final_loss + 1e-9


def test_convergence_no_loss_is_inert():
    steps = [StepRecord(step=i, phases={"forward": 0.1}) for i in range(10)]
    c = analyze_convergence(steps)
    assert not c.has_loss
    assert run_diagnosis(DiagnosisContext(convergence=c)) == []


# --- memory ----------------------------------------------------------------
def test_memory_growth_flagged():
    mb = 1024 * 1024
    steps = [
        StepRecord(
            step=i,
            memory={"alloc": (100 + i * 5) * mb, "reserved": (110 + i * 5) * mb},
        )
        for i in range(40)
    ]
    m = analyze_memory(steps)
    assert m.has_memory
    assert m.growth_bytes_per_step > 0
    assert "MEMORY.GROWTH" in _codes(run_diagnosis(DiagnosisContext(memory=m)))


def test_memory_fragmentation_flagged():
    mb = 1024 * 1024
    steps = [
        StepRecord(step=i, memory={"alloc": 100 * mb, "reserved": 200 * mb})
        for i in range(30)
    ]
    m = analyze_memory(steps)
    assert abs(m.fragmentation - 0.5) < 1e-9
    assert "MEMORY.FRAGMENTATION" in _codes(run_diagnosis(DiagnosisContext(memory=m)))


def test_memory_absent_is_inert():
    steps = [StepRecord(step=i, phases={"forward": 0.1}) for i in range(10)]
    m = analyze_memory(steps)
    assert not m.has_memory
    assert run_diagnosis(DiagnosisContext(memory=m)) == []


# --- cross-signal (the flagship) ------------------------------------------
def _run_with_event():
    steps = []
    for i in range(40):
        loss = 1.0 / (i + 1)
        grad = 1.0
        phases = {"forward": 0.01, "backward": 0.01}
        if i == 25:  # simultaneous blow-up across three axes
            loss = 50.0
            grad = 80.0
            phases = {"forward": 0.05, "backward": 0.05}
        steps.append(
            StepRecord(step=i, phases=phases, scalars={"loss": loss, "grad_norm": grad})
        )
    return steps


def test_cross_signal_correlated_instability():
    steps = _run_with_event()
    findings = run_diagnosis(DiagnosisContext(steps=steps))
    cross = [f for f in findings if f.code == "CROSS.CORRELATED_INSTABILITY"]
    assert cross, "expected a correlated-instability finding"
    assert "step 25" in cross[0].title
    assert cross[0].severity == "high"


def test_cross_signal_quiet_run_no_finding():
    steps = [
        StepRecord(
            step=i,
            phases={"forward": 0.01, "backward": 0.01},
            scalars={"loss": 1.0 / (i + 1), "grad_norm": 1.0},
        )
        for i in range(40)
    ]
    findings = run_diagnosis(DiagnosisContext(steps=steps))
    assert all(f.code != "CROSS.CORRELATED_INSTABILITY" for f in findings)


def test_cross_signal_single_axis_does_not_correlate():
    # Only loss spikes; no other axis co-occurs -> no correlation claim.
    steps = []
    for i in range(40):
        loss = 1.0 / (i + 1)
        if i == 25:
            loss = 50.0
        steps.append(StepRecord(step=i, phases={"forward": 0.01}, scalars={"loss": loss}))
    findings = run_diagnosis(DiagnosisContext(steps=steps))
    assert all(f.code != "CROSS.CORRELATED_INSTABILITY" for f in findings)
