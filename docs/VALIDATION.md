# Validation status & protocol

pytscope makes quantitative claims (straggler attribution, exposed
communication, MFU). This document states **exactly what is validated, where, and
how** — and gives a reproducible protocol to validate the parts that need real
multi-GPU hardware. Honesty about the boundary is the point: a number you can't
trust is worse than no number.

## What is validated today (and how)

| Claim | Validation | Where it runs |
|-------|-----------|---------------|
| Phase timing (data/fwd/bwd/opt) | Unit + real torch single-process | Any machine ✅ |
| Auto-instrumentation correctness | Real torch, incl. **gradient accumulation** and **activation checkpointing** (re-entrant forward) | Any machine ✅ |
| Straggler detection (statistics) | Synthetic multi-rank + **real multi-process gloo DDP** (CPU) with injected straggler | Any machine ✅ |
| Pipeline-bubble math | Exact vs closed form `(p-1)/(m+p-1)` across p, m | Any machine ✅ |
| Exposed-comm interval math | Synthetic traces with hand-computed answers (exact) | Any machine ✅ |
| Kineto trace parsing | Against a real `torch.profiler` export | Any machine ✅ |
| FLOP counting | Real `torch` FlopCounterMode, exact vs analytic | Any machine ✅ |
| **Exposed-comm on real NCCL** | Real 2×T4 NCCL run, two batch-size configs — see [validation-runs/2026-06-08-kaggle-2xT4](validation-runs/2026-06-08-kaggle-2xT4/RESULTS.md) | ✅ validated (2026-06-08) |
| **MFU vs measured GPU throughput** | Blocked on an example bug (now fixed); rerun pending — see the same report, Experiment 3 | ⛔ pending (unblocked) |
| **Straggler on real NCCL all-reduce** | Real 2×T4 NCCL run, exact pass (z=14.1, named rank correctly) — see [validation-runs/2026-06-08-kaggle-2xT4](validation-runs/2026-06-08-kaggle-2xT4/RESULTS.md) | ✅ validated (2026-06-08) |

The single-node gloo path exercises the *same code* the GPU path uses (same
analyzer, same rules); what the GPU run adds is confidence that the kernel
classification and the NCCL overlap numbers are right on real traces.

## Multi-GPU validation protocol

Goal: confirm the straggler, exposed-comm, and MFU numbers on real hardware
against **known-bad configurations**, where the right answer is known a priori.

### Setup (≈1 GPU-hour, ~2×A10/T4 is enough — and free)

```bash
pip install -e ".[torch]"
# 2-GPU box; adjust nproc_per_node to the GPU count.
```

**Free option:** Kaggle's notebook tier gives 2× T4 GPUs for up to 30
hrs/week at zero cost — enough to run all three experiments below in one
sitting. [`docs/validation-runs/README.md`](validation-runs/README.md) has a
ready-to-run notebook (`kaggle_2xT4.ipynb`) and step-by-step instructions; no
paid rental needed.

### Experiment 1 — Straggler attribution (known answer: rank 1)

Run real NCCL DDP with one rank given extra work, mirroring
`examples/ddp_gloo.py` but on CUDA:

```bash
torchrun --nproc_per_node=2 examples/ddp_gloo.py \
    --steps 200 --straggler-rank 1 --run-dir runs/gpu_straggler
pytscope analyze runs/gpu_straggler
```

**Acceptance:** `DIST.STRAGGLER` fires naming **rank 1**, with `slowest_fraction`
≫ 1/world_size and a positive z-score; `wall_frac_lost_to_imbalance` within a few
points of the injected slowdown.

### Experiment 2 — Exposed communication (known answer: high vs low)

Capture a `torch.profiler` trace for two configs and compare:

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(20):
        train_one_step()
p.export_chrome_trace("runs/gpu_job/trace.json")
```

- **(a) Bad overlap:** tiny per-GPU batch + large model params → all-reduce can't
  hide. Expect `DIST.EXPOSED_COMM` HIGH, low overlap efficiency.
- **(b) Good overlap:** large per-GPU batch, default DDP bucketing. Expect high
  overlap efficiency, no (or LOW) finding.

**Acceptance:** overlap_efficiency(b) ≫ overlap_efficiency(a); exposed-comm
fraction tracks the configuration in the expected direction.

> **Interconnect-topology caveat (confirmed 2026-06-08 on Kaggle 2×T4, PCIe —
> see [validation-runs/2026-06-08-kaggle-2xT4](validation-runs/2026-06-08-kaggle-2xT4/RESULTS.md)):**
> on link-bandwidth-bound hardware (no NVLink/InfiniBand), absolute
> communication time is roughly *constant* across batch sizes — the all-reduce
> is bound by interconnect throughput, not by how much there is to overlap it
> with. `DIST.EXPOSED_COMM` may legitimately fire HIGH for *both* configs there
> (it did: 72% exposed at batch=4, 62% at batch=256). The directional claim
> (`overlap_efficiency(b) > overlap_efficiency(a)`) still held — read it as the
> pass condition on PCIe-only hardware; "(b) absent/LOW" is the expected outcome
> only on NVLink/InfiniBand-connected GPUs, where backward can fully hide a
> faster reduce.

### Experiment 3 — MFU sanity vs a known model

Run a transformer block of known FLOPs at a known precision on a known GPU:

```bash
pytscope analyze runs/gpu_job   # FLOPs auto-counted; peak from the table
```

**Acceptance:** reported MFU is within ~15% of a hand-computed
`6 · N_params · tokens / (step_time · peak_FLOPS)`, and never exceeds 100%. If it
exceeds 100%, the FLOP estimate or the peak table is wrong — that's the
falsifiability check doing its job.

### Recording results

Capture each run's console output and the `run.json`/trace into
`docs/validation-runs/` and link them here. One real NCCL run with the expected
findings turns "unproven at scale" into "validated, here's the artifact."

## Known limitations (be explicit with users)

- **DDP only.** FSDP / tensor / pipeline parallelism are not yet first-class
  (pipeline is schedule-analysis only). The exposed-comm trace analysis works on
  any trace, but the rules are framed for data-parallel.
- **Spec-peak MFU.** The peak table is approximate dense peak; override with
  `--peak-tflops` for a precise anchor.
- **`fwd_bwd_factor=3`** is the standard training-FLOPs approximation, not an
  exact per-kernel count.
