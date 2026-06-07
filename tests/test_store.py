from trainscope.core.events import StepRecord
from trainscope.core.store import RunStore


def test_roundtrip(tmp_path):
    store = RunStore(tmp_path, meta={"name": "t"}).open()
    store.append(StepRecord(step=0, phases={"data": 0.1, "forward": 0.2}))
    store.append(StepRecord(step=1, phases={"data": 0.1}, scalars={"loss": 1.5}))
    store.write_meta()
    store.close()

    loaded = RunStore.load(tmp_path)
    assert loaded.meta["name"] == "t"
    assert len(loaded.steps) == 2
    assert loaded.steps[0].phases["forward"] == 0.2
    assert loaded.steps[1].scalars["loss"] == 1.5


def test_load_missing(tmp_path):
    loaded = RunStore.load(tmp_path / "does_not_exist")
    assert loaded.steps == []
    assert loaded.meta == {}
