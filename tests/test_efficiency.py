"""Training Efficiency Budget: exact decomposition, MFU, ranked recoverables."""

from __future__ import annotations

import math

import pytest

from pytscope.analyzers.efficiency import analyze_efficiency
from pytscope.core.events import StepRecord
from pytscope.diagnosis.engine import DiagnosisContext, run_diagnosis
from pytscope.hardware import peak_flops_for


def _steps(n=10, **phase_secs):
    return [StepRecord(step=i, phases=dict(phase_secs)) for i in range(n)]


def test_budget_sums_to_wall_exactly_no_anchor():
    steps = _steps(10, data=0.001, forward=0.002, backward=0.004, optimizer=0.001)
    b = analyze_efficiency(steps)
    assert b.mfu is None
    total = math.fsum(ln.seconds for ln in b.lines)
    assert abs(total - b.wall) < 1e-12
    # Without an anchor, useful == measured compute.
    useful = next(ln for ln in b.lines if ln.name == "useful_compute")
    assert abs(useful.seconds - b.compute_measured) < 1e-12


def test_budget_sums_to_wall_exactly_with_anchor():
    steps = _steps(
        10, data=0.001, forward=0.002, backward=0.004, optimizer=0.001, comm=0.002
    )
    b = analyze_efficiency(steps, flops_per_step=1e9, peak_flops=1e12)
    total = math.fsum(ln.seconds for ln in b.lines)
    assert abs(total - b.wall) < 1e-12


def test_mfu_math():
    # ideal = 1e9 * 10 / 1e12 = 0.01s; wall = 0.10s -> MFU = 0.10
    steps = _steps(
        10, data=0.001, forward=0.002, backward=0.004, optimizer=0.001, comm=0.002
    )
    b = analyze_efficiency(steps, flops_per_step=1e9, peak_flops=1e12)
    assert abs(b.mfu - 0.10) < 1e-9
    assert abs(b.efficiency - 0.10) < 1e-9


def test_ideal_above_measured_clamps_overhead_to_zero():
    # Overstated FLOPs would imply >100% efficiency; useful is capped at measured.
    steps = _steps(10, forward=0.001, backward=0.001)  # compute = 0.02s total
    b = analyze_efficiency(steps, flops_per_step=1e15, peak_flops=1e12)
    overhead = next(ln for ln in b.lines if ln.name == "compute_overhead")
    useful = next(ln for ln in b.lines if ln.name == "useful_compute")
    assert overhead.seconds >= 0.0
    assert abs(useful.seconds - b.compute_measured) < 1e-12


def test_recoverable_ranking():
    steps = _steps(10, data=0.005, forward=0.001, backward=0.001, comm=0.003)
    b = analyze_efficiency(steps)
    rec = b.recoverable_lines
    # Sorted by seconds desc; data_stall is largest here.
    assert rec[0].name == "data_stall"
    assert all(rec[i].seconds >= rec[i + 1].seconds for i in range(len(rec) - 1))
    assert b.top_recoverable.name == "data_stall"


def test_empty_returns_none():
    assert analyze_efficiency([]) is None
    assert analyze_efficiency(_steps(3)) is None  # all-zero phases -> wall 0


def test_low_mfu_rule_fires():
    steps = _steps(10, data=0.005, forward=0.002, backward=0.003)
    b = analyze_efficiency(steps, flops_per_step=1e8, peak_flops=1e12)  # tiny MFU
    findings = run_diagnosis(DiagnosisContext(efficiency=b))
    assert "EFFICIENCY.LOW_MFU" in {f.code for f in findings}


def test_high_mfu_no_finding():
    # Nearly all wall is useful compute -> high MFU -> no finding.
    steps = _steps(10, forward=0.005, backward=0.005)  # 0.10s compute
    b = analyze_efficiency(steps, flops_per_step=9e10, peak_flops=1e12)  # ideal 0.9*wall
    assert b.mfu > 0.5
    findings = {f.code for f in run_diagnosis(DiagnosisContext(efficiency=b))}
    assert "EFFICIENCY.LOW_MFU" not in findings


def test_recoverable_rule_without_anchor():
    # No FLOPs anchor, but data dominates -> a recoverable finding.
    steps = _steps(10, data=0.008, forward=0.001, backward=0.001)
    b = analyze_efficiency(steps)
    findings = {f.code for f in run_diagnosis(DiagnosisContext(efficiency=b))}
    assert "EFFICIENCY.RECOVERABLE" in findings


def test_peak_table_lookup():
    assert peak_flops_for("NVIDIA A100-SXM4-80GB") == 312e12
    assert peak_flops_for("NVIDIA H100 80GB HBM3") == 989e12
    assert peak_flops_for("Some Unknown GPU") is None
    assert peak_flops_for("") is None


@pytest.mark.slow
def test_measure_flops_real_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from pytscope.hardware import measure_flops

    model = nn.Linear(512, 512)
    x = torch.randn(64, 512)
    # forward FLOPs for a linear = 2 * batch * in * out
    flops = measure_flops(model, x, fwd_bwd_factor=1.0)
    assert abs(flops - 2 * 64 * 512 * 512) < 1.0
