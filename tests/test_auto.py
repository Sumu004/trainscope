"""AutoProfiler: zero-instrumentation capture + clean teardown. Needs torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from pytscope.auto import AutoProfiler  # noqa: E402
from pytscope.core.events import BACKWARD, FORWARD, OPTIMIZER  # noqa: E402
from pytscope.core.store import RunStore  # noqa: E402


def _train(run_dir, steps=12, warmup=2, **kw):
    model = nn.Linear(32, 32)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    prof = AutoProfiler(run_dir, model, opt, warmup=warmup, **kw)
    prof.start()
    for _ in range(steps):
        x = torch.randn(8, 32)
        y = torch.randn(8, 32)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        prof.log(loss=loss.item())
    prof.finish()
    return model, opt


def test_phases_captured_without_manual_marks(tmp_path):
    run_dir = tmp_path / "auto"
    _train(run_dir, steps=12, warmup=2)
    store = RunStore.load(run_dir)
    assert len(store.steps) == 10  # warmup dropped
    rec = store.steps[-1]
    # Every core phase was attributed automatically.
    for phase in (FORWARD, BACKWARD, OPTIMIZER):
        assert phase in rec.phases
        assert rec.phases[phase] > 0
    assert rec.scalars.get("loss") is not None
    assert rec.total() > 0


def test_data_phase_attributed_after_first_step(tmp_path):
    run_dir = tmp_path / "auto"
    _train(run_dir, steps=12, warmup=0)
    store = RunStore.load(run_dir)
    # Steps after the first should see the inter-step gap as `data`.
    assert any("data" in s.phases for s in store.steps[1:])


def test_optimizer_step_restored_on_finish(tmp_path):
    model = nn.Linear(8, 8)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    original = opt.step
    prof = AutoProfiler(tmp_path / "auto", model, opt)
    prof.start()
    assert opt.step is not original  # wrapped while profiling
    x = torch.randn(4, 8)
    model(x).pow(2).mean().backward()
    opt.step()
    prof.finish()
    assert opt.step == original  # restored


def test_hooks_removed_on_finish(tmp_path):
    model = nn.Linear(8, 8)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    prof = AutoProfiler(tmp_path / "auto", model, opt)
    prof.start()
    prof.finish()
    # No forward hooks should linger on the module.
    assert len(model._forward_pre_hooks) == 0
    assert len(model._forward_hooks) == 0


def test_context_manager(tmp_path):
    model = nn.Linear(8, 8)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    run_dir = tmp_path / "auto"
    with AutoProfiler(run_dir, model, opt, warmup=0) as prof:
        for _ in range(5):
            loss = model(torch.randn(4, 8)).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            prof.log(loss=loss.item())
    assert len(RunStore.load(run_dir).steps) == 5


def test_gradient_accumulation_one_step_per_optimizer_step(tmp_path):
    # Multiple forward/backward per optimizer.step must record ONE step each,
    # not one per micro-batch.
    run_dir = tmp_path / "auto"
    model = nn.Linear(32, 32)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    acc = 3
    n_steps = 5
    prof = AutoProfiler(run_dir, model, opt, warmup=0)
    prof.start()
    for _ in range(n_steps):
        for _ in range(acc):
            loss = model(torch.randn(8, 32)).pow(2).mean() / acc
            loss.backward()
        opt.step()
        opt.zero_grad()
    prof.finish()
    store = RunStore.load(run_dir)
    assert len(store.steps) == n_steps  # not n_steps * acc
    for rec in store.steps:
        assert rec.phases.get(FORWARD, 0) > 0
        assert rec.phases.get(BACKWARD, 0) > 0


def test_activation_checkpointing_does_not_corrupt_steps(tmp_path):
    # Activation checkpointing recomputes forward DURING backward, re-firing the
    # forward hooks. The step structure must stay intact (one step per opt.step).
    from torch.utils.checkpoint import checkpoint

    run_dir = tmp_path / "auto"
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    n_steps = 4
    prof = AutoProfiler(run_dir, model, opt, warmup=0)
    prof.start()
    for _ in range(n_steps):
        x = torch.randn(8, 32, requires_grad=True)
        out = checkpoint(model, x, use_reentrant=False)  # recompute in backward
        out.pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    prof.finish()
    store = RunStore.load(run_dir)
    assert len(store.steps) == n_steps  # recompute must not spawn extra steps
    for rec in store.steps:
        assert rec.total() > 0


def test_no_optimizer_does_not_crash(tmp_path):
    # Without an optimizer there's no step boundary; start/finish must be safe.
    model = nn.Linear(8, 8)
    prof = AutoProfiler(tmp_path / "auto", model)
    prof.start()
    model(torch.randn(4, 8))
    prof.finish()
