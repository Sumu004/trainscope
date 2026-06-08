"""Chart-based dashboard (`trainscope visualize`) — self-contained, valid SVG
markup, escapes input, and degrades gracefully when there's nothing to chart."""

from __future__ import annotations

from trainscope.analyzers.convergence import analyze_convergence
from trainscope.analyzers.memory import analyze_memory
from trainscope.analyzers.timing import analyze_timing
from trainscope.core.events import StepRecord
from trainscope.report.charts import bar_rows, line_chart
from trainscope.report.visualize_report import render_visualization


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
                scalars={"loss": 1.0 / (i + 1), "grad_norm": 2.0 - i * 0.05},
                memory={"alloc": 1e9 + i * 1e6, "reserved": 1.2e9},
            )
        )
    return out


def test_line_chart_degrades_to_empty_for_short_or_constant_nan_series():
    assert line_chart([1.0]) == ""
    assert line_chart([]) == ""
    assert line_chart([float("nan"), float("nan")]) == ""
    out = line_chart([1.0, 2.0, 3.0], label="trend")
    assert out.startswith('<div class="ts-chart-wrap">')
    assert "<svg" in out and "</svg>" in out
    assert "trend" in out


def test_bar_rows_renders_one_row_per_entry_and_escapes_labels():
    out = bar_rows([("<b>x</b>", 0.5, "50%", "good"), ("y", 0.25, "25%", "bad")])
    assert out.count("ts-bar-row") == 2
    assert "<b>x</b>" not in out
    assert "&lt;b&gt;x&lt;/b&gt;" in out


def test_render_is_one_self_contained_html_document():
    steps = _steps()
    timing = analyze_timing(steps)
    out = render_visualization("demo", "runs/demo", steps=steps, timing=timing)
    assert out.startswith("<!DOCTYPE html>")
    assert out.strip().endswith("</html>")
    assert "<style>" in out
    assert "http://" not in out and "https://" not in out
    assert "<svg" in out


def test_charts_present_for_each_populated_vertical():
    steps = _steps()
    timing = analyze_timing(steps)
    memory = analyze_memory(steps)
    convergence = analyze_convergence(steps)
    out = render_visualization(
        "demo",
        "runs/demo",
        steps=steps,
        timing=timing,
        memory=memory,
        convergence=convergence,
    )
    assert "Timing" in out
    assert "Convergence" in out
    assert "Memory" in out
    assert out.count("<svg") >= 3  # step time, loss, grad-norm (memory optional)


def test_empty_run_degrades_to_no_chartable_data_message():
    out = render_visualization("demo", "runs/demo")
    assert "No chartable data" in out
    assert "<svg" not in out
