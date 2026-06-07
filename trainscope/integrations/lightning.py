"""PyTorch Lightning integration — one-line attach.

Usage::

    from trainscope.integrations.lightning import TrainScopeCallback
    trainer = pl.Trainer(callbacks=[TrainScopeCallback("runs/exp1")])

Lightning's hooks let us split the step into data / forward / backward /
optimizer cleanly:

    on_train_batch_start   <- data time = since previous batch end
    on_before_backward     <- close forward
    on_after_backward      <- close backward
    on_train_batch_end     <- close optimizer, finalize step
"""

from __future__ import annotations

from ..core.events import BACKWARD, COMPUTE, FORWARD, OPTIMIZER
from ..profiler import Profiler

try:  # support both the new `lightning` and legacy `pytorch_lightning`
    from lightning.pytorch.callbacks import Callback  # type: ignore
except Exception:  # pragma: no cover - exercised only with lightning installed
    try:
        from pytorch_lightning.callbacks import Callback  # type: ignore
    except Exception:
        Callback = object  # type: ignore


class TrainScopeCallback(Callback):  # type: ignore[misc]
    def __init__(
        self,
        run_dir,
        name: str = "lightning-run",
        log_loss: bool = True,
        **profiler_kwargs,
    ):
        # log_loss forces a device->host sync per step; on by default but
        # documented. Set False for the absolute-lowest-overhead timing run.
        self.profiler = Profiler(run_dir, name=name, **profiler_kwargs)
        self.log_loss = log_loss
        self._last_batch_end = None
        self._marked_forward = False

    def on_train_start(self, trainer, pl_module):
        self.profiler.start()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = self.profiler._now()
        if self._last_batch_end is not None:
            self.profiler._record_data_ns(now - self._last_batch_end)
        self.profiler.begin_step()
        self._marked_forward = False

    def on_before_backward(self, trainer, pl_module, loss):
        self.profiler.mark(FORWARD)
        self._marked_forward = True
        if self.log_loss:
            try:
                self.profiler.log(loss=float(loss.detach()))
            except Exception:
                pass

    def on_after_backward(self, trainer, pl_module):
        self.profiler.mark(BACKWARD)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # With automatic optimization the tail is the optimizer step. With manual
        # optimization on_before_backward never fired, so the whole step is one
        # opaque compute phase — don't mislabel it all as "optimizer".
        self.profiler.mark(OPTIMIZER if self._marked_forward else COMPUTE)
        self.profiler.end_step()
        self._last_batch_end = self.profiler._now()

    def on_train_end(self, trainer, pl_module):
        self.profiler.finish()
