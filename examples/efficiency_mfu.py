"""The Training Efficiency Budget — one number (MFU) + where the rest of wall goes.

    pip install -e ".[torch]"
    python examples/efficiency_mfu.py && trainscope analyze runs/mfu

`AutoProfiler(measure_flops=True)` auto-counts the model's FLOPs from the first
batch. On a recognized GPU the hardware peak is looked up automatically; on other
devices pass a peak (here we set one explicitly so the MFU anchor is defined off
a real GPU). `trainscope analyze` then prints the budget: useful compute (the
FLOPs at peak) vs the recoverable line items that sum to your wall time.
"""

from __future__ import annotations

import shutil

import torch
import torch.nn as nn

from trainscope.auto import AutoProfiler


def main() -> None:
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    shutil.rmtree("runs/mfu", ignore_errors=True)
    # measure_flops=True → FLOPs/step counted automatically from the first batch.
    # On CUDA, AutoProfiler looks the device up in the hardware peak table (e.g.
    # T4 -> 65 TFLOP/s, A100 -> 312 TFLOP/s) — leave peak_flops unset so the
    # anchor matches the GPU you're actually running on. Off-GPU (CPU/MPS) there
    # is no device to look up, so set an explicit anchor (~A100 bf16) so MFU is
    # still defined — it'll legitimately read near-zero there, which is correct:
    # the model isn't running anywhere near accelerator peak.
    prof_kwargs = {} if dev == "cuda" else {"peak_flops": 312e12}
    prof = AutoProfiler(
        "runs/mfu",
        model,
        opt,
        warmup=5,
        sync=True,
        measure_flops=True,
        **prof_kwargs,
    )
    prof.start()
    for _ in range(60):
        x = torch.randn(256, 1024, device=dev)
        y = torch.randn(256, 1024, device=dev)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        prof.log(loss=loss.item())
    prof.finish()
    print(f"Done on {dev}. Run:  trainscope analyze runs/mfu")


if __name__ == "__main__":
    main()
