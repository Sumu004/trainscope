"""Hugging Face Trainer integration — one-line attach.

Usage::

    from pytscope.integrations.huggingface import PytscopeCallback
    trainer = Trainer(..., callbacks=[PytscopeCallback("runs/exp1")])

The Trainer callback API does not expose the forward/backward boundary, so the
HF integration attributes time at a coarser grain: ``data`` (between steps) and
``compute`` (the step itself). The diagnosis engine handles arbitrary phase
names, so dataloader-bound detection still works.
"""

from __future__ import annotations

from ..core.events import COMPUTE
from ..profiler import Profiler

try:
    from transformers import TrainerCallback  # type: ignore
except Exception:  # pragma: no cover - exercised only with transformers installed
    TrainerCallback = object  # type: ignore


class PytscopeCallback(TrainerCallback):  # type: ignore[misc]
    def __init__(self, run_dir, name: str = "hf-run", **profiler_kwargs):
        self.profiler = Profiler(run_dir, name=name, **profiler_kwargs)
        self._last_step_end = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.profiler.start()

    def on_step_begin(self, args, state, control, **kwargs):
        now = self.profiler._now()
        if self._last_step_end is not None:
            self.profiler._record_data_ns(now - self._last_step_end)
        self.profiler.begin_step()

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.mark(COMPUTE)
        self.profiler.end_step()
        self._last_step_end = self.profiler._now()

    def on_train_end(self, args, state, control, **kwargs):
        self.profiler.finish()
