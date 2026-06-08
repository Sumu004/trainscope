"""HTML report — single self-contained file, valid structure, escapes input,
and degrades gracefully when sections have no data."""

from __future__ import annotations

import html

from trainscope.analyzers.efficiency import analyze_efficiency
from trainscope.analyzers.timing import analyze_timing
from trainscope.core.events import StepRecord
from trainscope.diagnosis.engine import DiagnosisContext, Finding, run_diagnosis
from trainscope.report.html_report import render_html_report


def _steps(n=20):
    out = []
    for i in range(n):
        out.append(
            StepRecord(
                step=i,
                phases={
                    "data": 0.012,
                    "forward": 0.004,
                    "backward": 0.006,
                    "optimizer": 0.001,
                },
                scalars={"loss": 1.0 / (i + 1)},
            )
        )
    return out


def test_render_is_one_self_contained_html_document():
    steps = _steps()
    timing = analyze_timing(steps)
    out = render_html_report("demo", "runs/demo", timing=timing, findings=[])
    assert out.startswith("<!DOCTYPE html>")
    assert out.strip().endswith("</html>")
    assert "<style>" in out  # inline CSS — no external stylesheet
    assert "http://" not in out and "https://" not in out  # no network assets


def test_escapes_user_controlled_text():
    findings = [
        Finding(
            code="TEST.X",
            severity="high",
            title="<script>alert(1)</script>",
            detail="payload & <b>bold</b>",
            suggestion="",
        )
    ]
    out = render_html_report("demo", "runs/demo", findings=findings)
    assert "<script>alert(1)</script>" not in out
    assert html.escape("<script>alert(1)</script>") in out
    assert html.escape("payload & <b>bold</b>") in out


def test_findings_severity_ordering_and_no_issues_state():
    findings = [
        Finding(code="A", severity="low", title="low one", detail="d"),
        Finding(code="B", severity="high", title="high one", detail="d"),
    ]
    out = render_html_report("demo", "runs/demo", findings=findings)
    assert out.index("high one") < out.index("low one")

    out_clean = render_html_report("demo", "runs/demo", findings=[])
    assert "No issues found" in out_clean


def test_empty_summaries_render_nothing_for_that_section():
    out = render_html_report("demo", "runs/demo")
    assert "Timing" not in out
    assert "Memory" not in out
    assert "Convergence" not in out
    assert "No issues found" in out  # findings section always renders


def test_budget_section_present_with_efficiency_data():
    steps = _steps()
    eff = analyze_efficiency(steps, flops_per_step=1e9, peak_flops=1e12)
    out = render_html_report("demo", "runs/demo", efficiency=eff)
    assert "Efficiency budget" in out
    assert "MFU" in out


def test_diagnosis_findings_render_end_to_end():
    steps = _steps()
    timing = analyze_timing(steps)
    findings = run_diagnosis(DiagnosisContext(timing=timing, steps=steps))
    out = render_html_report("demo", "runs/demo", timing=timing, findings=findings)
    assert "ts-finding" in out or "No issues found" in out
