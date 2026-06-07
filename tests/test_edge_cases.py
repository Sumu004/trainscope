"""Edge cases covering each bug from AUDIT.md."""

import json
import math

from trainscope.core.events import StepRecord
from trainscope.core.store import STEPS_FILE, RunStore
from trainscope.integrations.lightning import TrainScopeCallback
from trainscope.profiler import Profiler


# --- #5 DDP rank guard -----------------------------------------------------
def test_nonzero_rank_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("trainscope.profiler.get_rank", lambda: 1)
    prof = Profiler(tmp_path / "run", only_rank_zero=True)
    prof.start()
    for _ in prof.iter_data([1, 2, 3]):  # must still yield batches
        with prof.step():
            prof.mark("forward")
    prof.finish()
    assert prof._disabled
    assert not (tmp_path / "run" / STEPS_FILE).exists()


def test_nonzero_rank_namespaced_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("trainscope.profiler.get_rank", lambda: 2)
    base = tmp_path / "run"
    prof = Profiler(base, only_rank_zero=False)
    assert str(prof.store.run_dir).endswith("_rank2")


# --- #3 batched flush durability ------------------------------------------
def test_data_survives_without_explicit_flush(tmp_path):
    prof = Profiler(tmp_path, collect_memory=False, flush_every=1000)
    prof.start()
    for _ in range(5):
        with prof.step():
            prof.mark("forward")
    prof.finish()  # close() must flush the tail even below flush_every
    assert len(RunStore.load(tmp_path).steps) == 5


# --- #8 crash-resilient load ----------------------------------------------
def test_truncated_final_line_is_skipped(tmp_path):
    p = tmp_path / STEPS_FILE
    good = json.dumps(StepRecord(step=0, phases={"forward": 0.1}).to_json_dict())
    p.write_text(good + "\n" + '{"step":1,"pha', encoding="utf-8")  # truncated
    store = RunStore.load(tmp_path)
    assert len(store.steps) == 1
    assert store.steps[0].step == 0


# --- #4 empty collections omitted -----------------------------------------
def test_empty_dicts_omitted_from_json():
    line = json.dumps(StepRecord(step=0, phases={"forward": 0.1}).to_json_dict())
    assert "scalars" not in line and "memory" not in line
    assert "forward" in line


# --- #9 non-finite loss round-trips (divergence detection needs it) -------
def test_nan_and_inf_scalars_roundtrip(tmp_path):
    store = RunStore(tmp_path).open()
    store.append(StepRecord(step=0, scalars={"loss": float("inf")}))
    store.append(StepRecord(step=1, scalars={"loss": float("nan")}))
    store.close()
    steps = RunStore.load(tmp_path).steps
    assert math.isinf(steps[0].scalars["loss"])
    assert math.isnan(steps[1].scalars["loss"])


# --- #6 Lightning manual-optimization not mislabeled ----------------------
class _FakeLoss:
    def detach(self):
        return 0.5


def test_lightning_manual_opt_labels_compute(tmp_path):
    cb = TrainScopeCallback(tmp_path, collect_memory=False)
    cb.on_train_start(None, None)
    cb.on_train_batch_start(None, None, None, 0)
    # NOTE: no on_before_backward (manual optimization)
    cb.on_train_batch_end(None, None, None, None, 0)
    cb.on_train_end(None, None)

    rec = RunStore.load(tmp_path).steps[0]
    assert "compute" in rec.phases
    assert "optimizer" not in rec.phases


def test_lightning_automatic_opt_splits_phases(tmp_path):
    cb = TrainScopeCallback(tmp_path, collect_memory=False)
    cb.on_train_start(None, None)
    cb.on_train_batch_start(None, None, None, 0)
    cb.on_before_backward(None, None, _FakeLoss())
    cb.on_after_backward(None, None)
    cb.on_train_batch_end(None, None, None, None, 0)
    cb.on_train_end(None, None)

    rec = RunStore.load(tmp_path).steps[0]
    assert {"forward", "backward", "optimizer"} <= set(rec.phases)
    assert rec.scalars.get("loss") == 0.5  # log_loss default on
