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
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    shutil.rmtree("runs/mfu", ignore_errors=True)
    # measure_flops=True → FLOPs/step counted automatically from the first batch.
    # peak_flops set explicitly here so MFU is defined even off-GPU (≈ A100 bf16).
    prof = AutoProfiler(
        "runs/mfu",
        model,
        opt,
        warmup=5,
        sync=True,
        measure_flops=True,
        peak_flops=312e12,
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
