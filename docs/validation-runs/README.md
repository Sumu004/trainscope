# Running the GPU validation protocol for free (Kaggle, 2× T4)

`docs/VALIDATION.md` documents three experiments that need real multi-GPU
NCCL hardware. Kaggle's free notebook tier gives you **two T4 GPUs for up to
30 hours/week** — enough to run all three and produce a real artifact, at
zero cost. `kaggle_2xT4.ipynb` in this directory is ready to run as-is.

## Steps

1. **Sign in** at [kaggle.com](https://www.kaggle.com) (a Google/email account
   is enough — no payment details required for the free GPU tier).

2. **Create a new notebook**: Code → New Notebook.

3. **Upload `kaggle_2xT4.ipynb`** — File → Import Notebook → Upload, and
   select the file from this directory (`docs/validation-runs/kaggle_2xT4.ipynb`).
   Alternatively, open a blank notebook and paste the cells in by hand; the
   `.ipynb` is just there so you don't have to retype anything.

4. **Turn on the accelerator**: in the right sidebar, *Notebook options* →
   **Accelerator** → **GPU T4 × 2**. (If you only see a single-GPU option,
   your weekly free GPU quota may be exhausted — it resets weekly.)

5. **Turn on internet access**: same sidebar → *Internet* → **On** (needed to
   `git clone` the repo and `pip install`).

6. **Run all cells top to bottom** (Run → Run All, or step through with
   Shift+Enter). Total runtime is roughly **10–20 minutes**:
   - Cell 0 confirms 2 GPUs are visible (fails fast with a clear message if not).
   - Cell 1 clones `Sumu004/trainscope` and installs it with the `torch` extra.
   - Cells 2–3 run **Experiment 1** (straggler attribution) via
     `torchrun --standalone --nproc_per_node=2 examples/ddp_gloo.py` on real
     NCCL, then `trainscope analyze` on the result.
   - Cells 4–7 write a small DDP probe script and run **Experiment 2**
     (exposed communication) twice — once with a tiny per-GPU batch (bad
     overlap) and once with a large one (good overlap) — capturing a real
     `torch.profiler` Kineto trace each time.
   - Cells 8–9 run **Experiment 3** (MFU sanity) via `examples/efficiency_mfu.py`
     and anchor it to the T4's published peak (`--peak-tflops 65`).
   - Cell 10 zips every `run.json`, trace, and console capture into
     `gpu_validation_artifacts.zip`.

7. **Check the acceptance criteria** in each cell's output against
   `docs/VALIDATION.md` (e.g. does `DIST.STRAGGLER` name rank 1? is
   `overlap_efficiency` for the large-batch run meaningfully higher than for
   the small-batch run? is MFU within ~15% of the hand-computed value and
   ≤100%?).

8. **Download the artifact**: in the notebook's *Output* panel (bottom-right),
   download `gpu_validation_artifacts.zip`.

9. **Commit it back into the repo**: unzip into a dated subdirectory here, e.g.

   ```bash
   mkdir -p docs/validation-runs/2026-06-08-kaggle-2xT4
   unzip ~/Downloads/gpu_validation_artifacts.zip -d docs/validation-runs/2026-06-08-kaggle-2xT4
   ```

   then add a row to the validation matrix in `docs/VALIDATION.md` linking to
   it, and commit. That turns "validated, here's the artifact" from a promise
   into a fact someone can click through to.

## Why this satisfies the protocol

- **Real hardware, real NCCL** — `torchrun --standalone --nproc_per_node=2`
  launches genuine multi-process CUDA/NCCL, not a simulation; this is exactly
  the code path `examples/ddp_gloo.py` was refactored to support.
- **Known-bad configurations with a priori right answers** — rank 1 is
  *injected* as the straggler, batch size is *deliberately* tiny vs large for
  the overlap experiment — so a pass/fail isn't a vibe, it's a check against
  ground truth.
- **Free and repeatable** — anyone reviewing the project (a hiring committee,
  a contributor) can re-run the exact same notebook and get the exact same
  kind of artifact, which is the whole point of writing the protocol down in
  the first place.
