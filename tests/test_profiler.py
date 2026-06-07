from trainscope.core.store import RunStore
from trainscope.profiler import Profiler


class FakeClock:
    """Deterministic monotonic integer-ns clock; tick() advances by seconds."""

    def __init__(self):
        self.t = 0  # nanoseconds

    def __call__(self):
        return self.t

    def tick(self, dt_seconds):
        self.t += int(round(dt_seconds * 1_000_000_000))


def test_profiler_records_phases(tmp_path):
    clk = FakeClock()
    prof = Profiler(tmp_path, warmup=0, collect_memory=False, clock=clk)
    prof.start()

    for _ in range(3):
        prof.set_data_time(0.10)
        prof.begin_step()
        clk.tick(0.20)
        prof.mark("forward")
        clk.tick(0.30)
        prof.mark("backward")
        prof.end_step()

    prof.finish()

    store = RunStore.load(tmp_path)
    assert len(store.steps) == 3
    rec = store.steps[0]
    assert abs(rec.phases["data"] - 0.10) < 1e-9
    assert abs(rec.phases["forward"] - 0.20) < 1e-9
    assert abs(rec.phases["backward"] - 0.30) < 1e-9
    assert store.meta["n_steps"] == 3
    assert "environment" in store.meta


def test_warmup_steps_not_stored(tmp_path):
    clk = FakeClock()
    prof = Profiler(tmp_path, warmup=2, collect_memory=False, clock=clk)
    prof.start()
    for _ in range(5):
        prof.begin_step()
        clk.tick(0.1)
        prof.mark("forward")
        prof.end_step()
    prof.finish()

    store = RunStore.load(tmp_path)
    assert len(store.steps) == 3  # 5 run, first 2 warmup dropped


def test_iter_data_attributes_fetch_time(tmp_path):
    clk = FakeClock()
    prof = Profiler(tmp_path, collect_memory=False, clock=clk)
    prof.start()

    def loader():
        for i in range(3):
            clk.tick(0.05)  # simulate fetch latency
            yield i

    for _ in prof.iter_data(loader()):
        prof.begin_step()
        clk.tick(0.1)
        prof.mark("forward")
        prof.end_step()
    prof.finish()

    store = RunStore.load(tmp_path)
    assert all(abs(s.phases["data"] - 0.05) < 1e-9 for s in store.steps)
