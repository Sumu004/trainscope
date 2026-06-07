"""Zero-instrumentation profiling: wrap once, leave the loop untouched.

    pip install -e ".[torch]"
    python examples/auto.py && trainscope analyze runs/auto

Compare this to examples/manual_loop.py — same telemetry, but here the training
loop has *no* trainscope calls at all. AutoProfiler registers PyTorch hooks to
attribute data / forward / backward / optimizer automatically.
"""

from __future__ import annotations

import shutil

import torch
import torch.nn as nn

from trainscope.auto import AutoProfiler


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    shutil.rmtree("runs/auto", ignore_errors=True)
    prof = AutoProfiler("runs/auto", model, opt, warmup=5, sync=True)
    prof.start()
    for _ in range(60):
        x = torch.randn(64, 512, device=dev)
        y = torch.randn(64, 512, device=dev)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        prof.log(loss=loss.item())  # optional: only to record the loss signal
    prof.finish()
    print(f"Done on {dev}. Run:  trainscope analyze runs/auto")


if __name__ == "__main__":
    main()
