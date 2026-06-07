from trainscope.analyzers.timing import analyze_timing
from trainscope.core.events import StepRecord
from trainscope.diagnosis.engine import DiagnosisContext, run_diagnosis


def _diagnose(phases, n=20):
    steps = [StepRecord(step=i, phases=dict(phases)) for i in range(n)]
    timing = analyze_timing(steps)
    return run_diagnosis(DiagnosisContext(timing=timing))


def test_dataloader_bound_high():
    findings = _diagnose({"data": 0.5, "forward": 0.25, "backward": 0.25})
    codes = {f.code: f for f in findings}
    assert "TIMING.DATALOADER_BOUND" in codes
    assert codes["TIMING.DATALOADER_BOUND"].severity == "high"


def test_balanced_run_no_dataloader_finding():
    findings = _diagnose({"data": 0.1, "forward": 0.45, "backward": 0.45})
    assert all(f.code != "TIMING.DATALOADER_BOUND" for f in findings)


def test_backward_heavy():
    findings = _diagnose({"forward": 0.1, "backward": 0.3})
    assert any(f.code == "TIMING.BACKWARD_HEAVY" for f in findings)


def test_findings_sorted_by_severity():
    findings = _diagnose({"data": 0.5, "forward": 0.1, "backward": 0.3, "optimizer": 0.2})
    severities = [f.severity for f in findings]
    order = {"high": 0, "med": 1, "low": 2}
    assert severities == sorted(severities, key=lambda s: order[s])
