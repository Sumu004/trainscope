"""Exposed-communication analysis: exact interval math + real Kineto ingestion."""

from __future__ import annotations

import gzip
import json

import pytest

from trainscope.analyzers.trace import (
    analyze_trace,
    analyze_trace_file,
    load_chrome_trace,
    merge_intervals,
    subtract_intervals,
    total_length,
)
from trainscope.diagnosis.engine import DiagnosisContext, run_diagnosis


def _ev(name, ts_us, dur_us, cat="kernel"):
    return {"ph": "X", "name": name, "ts": ts_us, "dur": dur_us, "cat": cat}


# --- interval arithmetic --------------------------------------------------


def test_merge_intervals():
    assert merge_intervals([(0, 5), (3, 8), (10, 12)]) == [(0, 8), (10, 12)]
    assert merge_intervals([]) == []
    assert merge_intervals([(5, 5)]) == []  # zero-length dropped


def test_subtract_intervals():
    # [0,10] minus [3,5] -> [0,3],[5,10]
    assert subtract_intervals([(0, 10)], [(3, 5)]) == [(0, 3), (5, 10)]
    # disjoint subtraction leaves A intact
    assert subtract_intervals([(0, 5)], [(10, 20)]) == [(0, 5)]
    # full cover -> empty
    assert subtract_intervals([(0, 5)], [(0, 5)]) == []
    assert abs(total_length([(0, 3), (5, 10)]) - 8) < 1e-12


# --- exposed-comm correctness (known answers) -----------------------------


def test_partial_overlap_exposed_exact():
    # comm [0,100], compute [40,100] -> exposed [0,40] = 40us, overlapped 60us
    s = analyze_trace([_ev("ncclAllReduce", 0, 100), _ev("ampere_gemm", 40, 60)])
    assert abs(s.exposed_comm_time * 1e6 - 40) < 1e-6
    assert abs(s.overlapped_comm_time * 1e6 - 60) < 1e-6
    assert abs(s.overlap_efficiency - 0.6) < 1e-9


def test_multiple_kernels_exposed_exact():
    # comm [0,50],[200,260]; compute [10,40],[210,300]
    # exposed: [0,10]+[40,50]+[200,210] = 30us
    events = [
        _ev("ncclAllReduce", 0, 50),
        _ev("ncclAllGather", 200, 60),
        _ev("conv2d", 10, 30),
        _ev("sgemm", 210, 90),
    ]
    s = analyze_trace(events)
    assert abs(s.exposed_comm_time * 1e6 - 30) < 1e-6
    assert set(s.per_collective) == {"all_reduce", "all_gather"}


def test_fully_overlapped_is_zero_exposed():
    s = analyze_trace([_ev("nccl:all_reduce", 100, 50), _ev("relu", 90, 100)])
    assert abs(s.exposed_comm_time) < 1e-12
    assert abs(s.overlap_efficiency - 1.0) < 1e-12


def test_no_overlap_fully_exposed():
    s = analyze_trace([_ev("ncclAllReduce", 0, 50), _ev("gemm", 100, 50)])
    assert abs(s.exposed_comm_time * 1e6 - 50) < 1e-6
    assert s.overlap_efficiency == 0.0


def test_reduction_kernel_not_misclassified_as_comm():
    # A compute reduction kernel must NOT be counted as communication.
    s = analyze_trace([_ev("reduce_kernel_sum", 0, 50), _ev("gemm", 0, 50)])
    assert s.n_comm_kernels == 0
    assert not s.has_comm


def test_non_kernel_events_ignored():
    s = analyze_trace(
        [_ev("ncclAllReduce", 0, 50), _ev("aten::add", 0, 50, cat="cpu_op")]
    )
    assert s.n_compute_kernels == 0  # cpu_op is not a device kernel


def test_empty_trace_returns_none():
    assert analyze_trace([]) is None


# --- rule -----------------------------------------------------------------


def test_exposed_comm_rule_fires():
    # 50us exposed out of 100us wall -> 50% exposed -> high
    s = analyze_trace([_ev("ncclAllReduce", 0, 50), _ev("gemm", 50, 50)])
    findings = run_diagnosis(DiagnosisContext(trace=s))
    codes = {f.code for f in findings}
    assert "DIST.EXPOSED_COMM" in codes


def test_well_overlapped_comm_no_finding():
    # comm fully hidden -> no exposed-comm finding
    s = analyze_trace([_ev("ncclAllReduce", 10, 40), _ev("gemm", 0, 100)])
    findings = {f.code for f in run_diagnosis(DiagnosisContext(trace=s))}
    assert "DIST.EXPOSED_COMM" not in findings


# --- file loading ---------------------------------------------------------


def test_load_plain_and_gzip(tmp_path):
    events = [_ev("ncclAllReduce", 0, 50), _ev("gemm", 25, 50)]
    doc = {"traceEvents": events}
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(doc))
    s = analyze_trace_file(p)
    assert s.has_comm

    pg = tmp_path / "trace.json.gz"
    with gzip.open(pg, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    sg = analyze_trace_file(pg)
    assert abs(sg.exposed_comm_time - s.exposed_comm_time) < 1e-12


# --- real torch.profiler ingestion (validates the parser, not NCCL numbers) -


@pytest.mark.slow
def test_real_torch_profiler_trace_parses(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from torch.profiler import ProfilerActivity, profile

    model = nn.Linear(128, 128)
    x = torch.randn(64, 128)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            model(x).pow(2).mean().backward()
    out = tmp_path / "trace.json"
    prof.export_chrome_trace(str(out))
    # The parser must consume a genuine Kineto export without error.
    events = load_chrome_trace(out)
    assert len(events) > 0  # real export parsed
    # A CPU-only trace has no device kernels, so there's nothing to analyze for
    # communication overlap → None is the correct, honest result.
    assert analyze_trace_file(out) is None
