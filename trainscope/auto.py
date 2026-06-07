"""Automatic instrumentation — profiling with zero changes to your loop.

The manual :class:`~trainscope.profiler.Profiler` asks you to call ``mark()`` at
each phase boundary. ``AutoProfiler`` removes that: it registers PyTorch hooks so
the phase timeline (data / forward / backward / optimizer, plus synchronous
communication) is captured automatically. You wrap the model + optimizer once and
leave your training loop **exactly as it is**::

    from trainscope.auto import AutoProfiler

    prof = AutoProfiler("runs/exp", model, optimizer, warmup=5)
    prof.start()
    for x, y in loader:                 # <- unchanged
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    prof.finish()
    # trainscope analyze runs/exp

How it works (the step boundary is ``optimizer.step``):

- a **forward pre-hook** on the model opens a step and attributes the gap since
  the previous step to ``data`` (the dataloader fetch);
- a **forward hook** closes the ``forward`` phase;
- wrapping **optimizer.step** attributes the time since forward to ``backward``
  (minus any communication captured below), times the optimizer update, and
  closes the step;
- with ``capture_comm`` (default), **synchronous** ``torch.distributed``
  collectives are timed and attributed to ``comm``. Asynchronous (overlapped)
  collectives — e.g. those issued internally by ``DistributedDataParallel`` — are
  deliberately *not* split out, because their cost is genuinely hidden inside
  backward; separating them would double-count wall time.

Assumes one forward + backward per optimizer step (the common case). For gradient
accumulation, use the manual ``Profiler``. Everything is restored on ``finish()``.
"""

from __future__ import annotations

from .core.events import BACKWARD, COMM, FORWARD, OPTIMIZER
from .profiler import Profiler

_COLLECTIVES = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "reduce",
    "broadcast",
    "all_to_all",
    "all_to_all_single",
)


class AutoProfiler:
    def __init__(
        self,
        run_dir,
        model,
        optimizer=None,
        *,
        name: str = "run",
        warmup: int = 0,
        sync: bool = False,
        collect_memory: bool = False,
        capture_comm: bool = True,
        distributed: bool = False,
        flush_every: int = 200,
        config: dict | None = None,
    ):
        self.prof = Profiler(
            run_dir,
            name=name,
            warmup=warmup,
            sync=sync,
            collect_memory=collect_memory,
            flush_every=flush_every,
            distributed=distributed,
            config=config,
        )
        self.model = model
        self.optimizer = optimizer
        self.capture_comm = capture_comm

        self._handles: list = []
        self._orig_step = None
        self._patched: list = []
        self._comm_ns = 0
        self._opt_end_ns: int | None = None
        self._started = False

    # --- lifecycle --------------------------------------------------------
    def start(self) -> AutoProfiler:
        self.prof.start()
        if self.prof._disabled:
            return self
        self._register_hooks()
        if self.optimizer is not None:
            self._wrap_optimizer()
        if self.capture_comm:
            self._patch_collectives()
        self._started = True
        return self

    def finish(self) -> None:
        # The last step is held open for post-step logging; flush it now.
        if self.prof._cur_active:
            self.prof.end_step()
        if self._started:
            self._unregister()
        self.prof.finish()

    def __enter__(self) -> AutoProfiler:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.finish()

    def log(self, **scalars: float) -> None:
        """Attach scalars (loss, grad_norm, lr) to the current step, if open."""
        self.prof.log(**scalars)

    # --- hook registration ------------------------------------------------
    def _register_hooks(self) -> None:
        self._handles.append(self.model.register_forward_pre_hook(self._on_forward_pre))
        self._handles.append(self.model.register_forward_hook(self._on_forward))

    def _wrap_optimizer(self) -> None:
        self._orig_step = self.optimizer.step

        def step_wrapper(*args, **kwargs):
            return self._on_optimizer_step(self._orig_step, *args, **kwargs)

        self.optimizer.step = step_wrapper  # type: ignore[method-assign]

    def _patch_collectives(self) -> None:
        try:
            import torch.distributed as dist  # type: ignore
        except Exception:
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        for fname in _COLLECTIVES:
            orig = getattr(dist, fname, None)
            if orig is None:
                continue
            setattr(dist, fname, self._make_collective_wrapper(orig))
            self._patched.append((dist, fname, orig))

    def _make_collective_wrapper(self, orig):
        def wrapper(*args, **kwargs):
            # Async/overlapped collectives are hidden inside backward — don't
            # attribute them (that would double-count). Only time blocking ones.
            if kwargs.get("async_op", False) or not self.prof._cur_active:
                return orig(*args, **kwargs)
            t0 = self.prof._now()
            result = orig(*args, **kwargs)
            self._comm_ns += self.prof._now() - t0
            return result

        return wrapper

    def _unregister(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        if self._orig_step is not None:
            self.optimizer.step = self._orig_step  # type: ignore[method-assign]
            self._orig_step = None
        for obj, fname, orig in self._patched:
            setattr(obj, fname, orig)
        self._patched.clear()

    # --- the captured state machine ---------------------------------------
    def _on_forward_pre(self, _module, _inputs) -> None:
        now = self.prof._now()
        if self.prof._cur_active:
            # The previous step was held open for post-step logging (loss is
            # usually logged after optimizer.step). Close it now, attributing the
            # gap since that step's optimizer to THIS step's data fetch.
            self.prof.end_step()
            if self._opt_end_ns is not None:
                self.prof._record_data_ns(now - self._opt_end_ns)
        self._comm_ns = 0
        self.prof.begin_step()  # sets _last_mark = now

    def _on_forward(self, _module, _inputs, _output) -> None:
        if self.prof._cur_active:
            self.prof.mark(FORWARD)

    def _on_optimizer_step(self, orig_step, *args, **kwargs):
        if not self.prof._cur_active:
            return orig_step(*args, **kwargs)
        now = self.prof._now()
        # Everything since forward end is backward; carve out captured comm.
        backward_ns = max(0, (now - self.prof._last_mark) - self._comm_ns)
        self.prof._cur_ns[BACKWARD] = self.prof._cur_ns.get(BACKWARD, 0) + backward_ns
        if self._comm_ns:
            self.prof._cur_ns[COMM] = self.prof._cur_ns.get(COMM, 0) + self._comm_ns
        self.prof._last_mark = now

        result = orig_step(*args, **kwargs)

        end = self.prof._now()
        self.prof._cur_ns[OPTIMIZER] = self.prof._cur_ns.get(OPTIMIZER, 0) + (
            end - self.prof._last_mark
        )
        self.prof._last_mark = end
        # Do NOT end the step here — keep it open so a post-step prof.log(loss=…)
        # still lands on this step. It is finalized at the next forward (or
        # finish()). Record when the optimizer finished so the gap until the next
        # forward can be attributed to the next step's data fetch.
        self._opt_end_ns = end
        self._comm_ns = 0
        return result
