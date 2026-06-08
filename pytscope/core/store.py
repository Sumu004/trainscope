"""Run store: the timeline on disk.

Layout of a run directory::

    run_dir/
      run.json      # metadata (env, config, name, timing of run)
      steps.jsonl   # one StepRecord per line, append-only

JSONL keeps writes cheap and streaming-friendly during live training. The live
writer never holds records in memory (so long runs don't leak); ``load()``
materializes the timeline for analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .events import StepRecord

RUN_META = "run.json"
STEPS_FILE = "steps.jsonl"

# Compact encoder bound once; separators drop whitespace from every line.
_encode = json.JSONEncoder(separators=(",", ":"), default=str).encode


class RunStore:
    """Append-only writer / loader for a single run's step timeline."""

    def __init__(
        self,
        run_dir,
        meta: dict[str, Any] | None = None,
        flush_every: int = 200,
    ):
        self.run_dir = Path(run_dir)
        self.meta: dict[str, Any] = dict(meta or {})
        self.steps: list[StepRecord] = []  # populated by load(), not by the writer
        self.flush_every = max(1, flush_every)
        self._fh = None
        self._since_flush = 0

    # --- writing (live) ---------------------------------------------------
    def open(self) -> RunStore:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.run_dir / STEPS_FILE, "a", encoding="utf-8")
        return self

    def append(self, rec: StepRecord) -> None:
        if self._fh is None:
            raise RuntimeError("RunStore is not open(); call open() first")
        # Hot path: no deep-copy, no per-step flush, no in-memory retention.
        self._fh.write(_encode(rec.to_json_dict()))
        self._fh.write("\n")
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._fh.flush()
            self._since_flush = 0

    def write_meta(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / RUN_META).write_text(
            json.dumps(self.meta, indent=2, default=str), encoding="utf-8"
        )

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._since_flush = 0

    # --- reading (analysis) ----------------------------------------------
    @classmethod
    def load(cls, run_dir) -> RunStore:
        run_dir = Path(run_dir)
        meta: dict[str, Any] = {}
        meta_path = run_dir / RUN_META
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A crash can leave run.json half-written; degrade gracefully.
                meta = {}

        store = cls(run_dir, meta)
        steps_path = run_dir / STEPS_FILE
        if steps_path.exists():
            with open(steps_path, encoding="utf-8") as fh:
                for line in fh:  # stream — don't slurp the whole file
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        store.steps.append(StepRecord.from_dict(json.loads(line)))
                    except json.JSONDecodeError:
                        # A crash mid-write can truncate the final line; skip it.
                        continue
        return store
