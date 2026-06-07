"""Real PyTorch training run with actual device timing + memory capture.

Trains a small CNN on synthetic images (no dataset download) and profiles it
end-to-end: real forward/backward/optimizer timing (device-synchronized) and
real GPU/MPS memory. Runs on CUDA, Apple MPS, or CPU automatically.

    python examples/pytorch_real.py            # honest, healthy run
    python examples/pytorch_real.py --leak     # retains activations -> growth
    trainscope analyze runs/pytorch

Requires: pip install -e ".[torch]"

Memory note: on CUDA we record the true in-step peak (max_memory_allocated). On
Apple MPS there is no peak API, so we sample resident memory at the step
boundary — that captures persistent/leaked memory (run with --leak) but not
transient in-step activation peaks. Memory attribution is most accurate on CUDA.
"""

import argparse

import torch
import torch.nn as nn

from trainscope import Profiler


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--leak", action="store_true", help="retain activations each step")
    args = ap.parse_args()

    device = pick_device()
    model = make_model().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    prof = Profiler(
        "runs/pytorch",
        name=f"pytorch-{device}",
        warmup=10,
        collect_memory=True,
        sync=True,  # device-accurate phase timing
        config={"device": device, "batch_size": args.batch_size},
    )
    prof.start()

    retained = []  # the "leak"
    for _ in range(args.steps):
        with prof.step():
            x = torch.randn(args.batch_size, 3, 32, 32, device=device)
            y = torch.randint(0, 10, (args.batch_size,), device=device)
            prof.mark("data")

            out = model(x)
            loss = loss_fn(out, y)
            prof.mark("forward")

            opt.zero_grad()
            loss.backward()
            prof.mark("backward")

            # Measure the true grad-norm without clipping (huge max_norm).
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
            opt.step()
            prof.mark("optimizer")

            if args.leak:
                # Simulate a retained-activation leak (~2 MB/step) — the kind of
                # bug where graph-attached tensors get appended to a list.
                retained.append(torch.randn(512, 1024, device=device))

            prof.log(loss=float(loss.detach()), grad_norm=float(grad_norm))

    prof.finish()
    print(f"Trained on {device}. Recorded runs/pytorch — now run:")
    print("  trainscope analyze runs/pytorch")


if __name__ == "__main__":
    main()
