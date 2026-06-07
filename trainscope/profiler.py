"""The Profiler — the one runtime object the whole package revolves around.

It owns the clock, the phase marking, the optional memory snapshot, and the
RunStore. Framework integrations (Lightning/HF) are thin shims that drive the
same primitives every step:

    begin_step()  ->  mark(phase) ...  ->  end_step()

Manual usage::

    prof = Profiler("runs/exp1", warmup=10)
    prof.start()
    for batch in prof.iter_data(loader):       # times data fetch
        with prof.step():
            out = model(batch); loss = loss_fn(out)
            prof.mark("forward")
            loss.backward();      prof.mark("backward")
            opt.step(); opt.zero_grad(); prof.mark("optimizer")
    prof.finish()

**Numerical design.** Time is measured with a monotonic *integer nanosecond*
clock (``time.perf_counter_ns``). All phase durations accumulate as integers, so
there is no floating-point cancellation when subtracting two large timestamps
and no drift when the same phase is marked many times (e.g. gradient
accumulation). The integer nanosecond totals are converted to float seconds
exactly once, at ``end_step``. A ``clock`` returning integer ns can be injected
for deterministic testing. ``sync=True`` calls ``torch.cuda.synchronize`` around
marks so async GPU work is attributed to the right phase.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from .collectors import memory as memory_collector
from .core.distributed import get_rank, get_world_size
from .core.events import DATA, StepRecord
from .core.provenance import capture_environment
from .core.store import RunStore

_NS_PER_S = 1_000_000_000


class Profiler:
    def __init__(
        self,
        run_dir,
        name: str = "run",
        warmup: int = 0,
        sync: bool = False,
        collect_memory: bool = False,
        flush_every: int = 200,
        only_rank_zero: bool = True,
        distributed: bool = False,
        clock: Callable[[], int] | None = None,
        config: dict | None = None,
    ):
        rank = get_rank()
        world_size = get_world_size()
        # Three capture modes on DDP:
        #   distributed=True      -> EVERY rank records, each into run_dir/rank{k}/.
        #                            This is what the distributed analyzer reads.
        #   only_rank_zero=True   -> only rank 0 records (default; safe, single
        #                            timeline, avoids concurrent appends to one file).
        #   only_rank_zero=False  -> legacy per-rank sibling dirs (run_dir_rank{k}).
        self.distributed = distributed
        if distributed:
            self._disabled = False
            run_dir = Path(run_dir) / f"rank{rank}"
        else:
            self._disabled = only_rank_zero and rank != 0
            if not only_rank_zero and rank != 0:
                run_dir = f"{run_dir}_rank{rank}"

        self.store = RunStore(
            run_dir,
            meta={
                "name": name,
                "config": config or {},
                "rank": rank,
                "world_size": world_size,
            },
            flush_every=flush_every,
        )
        self.warmup = warmup
        self.sync = sync
        self.collect_memory = collect_memory
        self.rank = rank
        # Integer-nanosecond monotonic clock; no float cancellation on deltas.
        self._clock = clock or time.perf_counter_ns

        self._step_idx = 0
        self._pending_data_ns = 0
        self._cur_active = False
        self._cur_ns: dict[str, int] = {}
        self._cur_scalars: dict[str, float] = {}
        self._last_mark = 0
        self._run_t0 = 0
        self._finished = False

    # --- lifecycle --------------------------------------------------------
    def start(self) -> Profiler:
        if self._disabled:
            return self
        self.store.meta["environment"] = capture_environment()
        self.store.meta["started"] = time.time()
        self.store.open()
        self._run_t0 = self._clock()
        return self

    def finish(self) -> None:
        if self._disabled or self._finished:
            return
        self._finished = True
        self.store.meta["wall_time"] = (self._clock() - self._run_t0) / _NS_PER_S
        self.store.meta["n_steps"] = self._step_idx
        self.store.write_meta()
        self.store.close()

    def __enter__(self) -> Profiler:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.finish()

    # --- timing primitives ------------------------------------------------
    def _now(self) -> int:
        if self.sync:
            _maybe_cuda_sync()
        return self._clock()

    def set_data_time(self, seconds: float) -> None:
        """Record data-fetch time (seconds) to attribute to the next step."""
        self._pending_data_ns = max(0, int(round(seconds * _NS_PER_S)))

    def _record_data_ns(self, ns: int) -> None:
        """Internal: attribute an integer-ns data interval to the next step."""
        if not self._disabled:
            self._pending_data_ns = ns if ns > 0 else 0

    def begin_step(self) -> None:
        if self._disabled:
            return
        self._cur_ns = {}
        if self._pending_data_ns:
            self._cur_ns[DATA] = self._pending_data_ns
            self._pending_data_ns = 0
        self._cur_scalars = {}
        self._cur_active = True
        self._last_mark = self._now()

    def mark(self, phase: str) -> None:
        """Attribute time since the previous mark/begin_step to ``phase``.

        Accumulates in integer nanoseconds, so repeated marks of the same phase
        (gradient accumulation) add exactly with no rounding drift.
        """
        if not self._cur_active:
            return
        now = self._now()
        self._cur_ns[phase] = self._cur_ns.get(phase, 0) + (now - self._last_mark)
        self._last_mark = now

    def log(self, **scalars: float) -> None:
        """Attach scalar signals (loss, grad_norm, lr) to the current step."""
        if self._cur_active:
            self._cur_scalars.update({k: float(v) for k, v in scalars.items()})

    def end_step(self) -> None:
        if not self._cur_active:
            return
        self._cur_active = False
        # Single, exact ns -> seconds conversion per phase (division only here).
        phases = {p: ns / _NS_PER_S for p, ns in self._cur_ns.items()}
        rec = StepRecord(step=self._step_idx, phases=phases, scalars=self._cur_scalars)
        if self.collect_memory:
            mem = memory_collector.snapshot()
            if mem:
                rec.memory = mem
        # Skip warmup steps from storage so analyzers see steady-state only.
        if rec.step >= self.warmup:
            self.store.append(rec)
        self._step_idx += 1

    # --- ergonomic helpers ------------------------------------------------
    @contextmanager
    def step(self) -> Iterator[Profiler]:
        self.begin_step()
        try:
            yield self
        finally:
            self.end_step()

    @contextmanager
    def comm(self) -> Iterator[Profiler]:
        """Time a collective-communication block and attribute it to ``comm``.

        Wrap gradient all-reduce / barriers so the distributed analyzer can
        separate network time from compute::

            loss.backward(); prof.mark("backward")
            with prof.comm():
                all_reduce_gradients(model)
            opt.step(); prof.mark("optimizer")
        """
        from .core.events import COMM

        if not self._cur_active:
            yield self
            return
        # Close out time accrued so far to the *previous* phase boundary, then
        # measure only the block, then resume the clock cleanly.
        start = self._now()
        try:
            yield self
        finally:
            now = self._now()
            self._cur_ns[COMM] = self._cur_ns.get(COMM, 0) + (now - start)
            self._last_mark = now

    def iter_data(self, loader: Iterable) -> Iterator:
        """Wrap a dataloader so inter-batch fetch time is attributed to ``data``."""
        if self._disabled:
            yield from loader
            return
        it = iter(loader)
        while True:
            t0 = self._now()
            try:
                batch = next(it)
            except StopIteration:
                return
            self._record_data_ns(self._now() - t0)
            yield batch


def _maybe_cuda_sync() -> None:
    """Block until queued accelerator work finishes, so async device time is
    attributed to the phase that launched it (CUDA and Apple MPS)."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass
