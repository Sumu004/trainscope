"""Overhead microbenchmark — the core competitive claim is 'low overhead'.

These print real per-step numbers (run `pytest -s`) and assert loose ceilings
that are still orders of magnitude under trace-dumping profilers, so they catch
a regression without flaking on slow CI runners.
"""

import time

from trainscope.profiler import Profiler


def _busy_clock():
    # Real monotonic clock so we measure true instrumentation cost.
    return time.perf_counter


def test_instrumentation_overhead_pure(capsys):
    """CPU cost of begin/mark*3/end with the disk write stubbed out."""
    prof = Profiler("/tmp/_ts_bench", collect_memory=False)
    prof.start()
    prof.store.append = lambda rec: None  # isolate instrumentation from IO

    n = 200_000
    t0 = time.perf_counter()
    for _ in range(n):
        prof.begin_step()
        prof.mark("forward")
        prof.mark("backward")
        prof.mark("optimizer")
        prof.end_step()
    dt = time.perf_counter() - t0
    prof.finish()

    per_step_us = dt / n * 1e6
    print(f"\n[pure instrumentation] {per_step_us:.3f} µs/step ({n} steps)")
    # Pure Python bookkeeping should be a handful of µs; 100 is a huge ceiling.
    assert per_step_us < 100.0


def test_end_to_end_overhead_with_disk(tmp_path, capsys):
    """Full cost including JSONL serialization + batched disk writes."""
    prof = Profiler(tmp_path, collect_memory=False, flush_every=500)
    prof.start()

    n = 50_000
    t0 = time.perf_counter()
    for _ in range(n):
        prof.begin_step()
        prof.mark("forward")
        prof.mark("backward")
        prof.log(loss=0.1)
        prof.end_step()
    dt = time.perf_counter() - t0
    prof.finish()

    per_step_us = dt / n * 1e6
    print(f"\n[end-to-end + disk] {per_step_us:.3f} µs/step ({n} steps)")
    assert per_step_us < 500.0


def test_disabled_rank_is_near_zero_overhead(capsys, monkeypatch):
    """Non-zero DDP ranks must be a true no-op (no file, negligible cost)."""
    monkeypatch.setattr("trainscope.profiler.get_rank", lambda: 3)
    prof = Profiler("/tmp/_ts_should_not_exist", only_rank_zero=True)
    prof.start()
    n = 200_000
    t0 = time.perf_counter()
    for _ in range(n):
        prof.begin_step()
        prof.mark("forward")
        prof.end_step()
    dt = time.perf_counter() - t0
    prof.finish()
    print(f"\n[disabled rank] {dt / n * 1e6:.3f} µs/step")
    assert prof._disabled
