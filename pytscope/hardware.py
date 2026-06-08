"""Hardware peak throughput + model FLOP counting, for the MFU anchor.

Two honest caveats, stated loudly:

1. The peak table is **spec peak** (dense, no sparsity), approximate, and varies
   by source/SKU/clocks. It is a *ceiling*, not an achievable number. Always
   override with a measured or vendor-confirmed value via ``--peak-tflops`` when
   precision matters. MFU computed against spec peak is a lower bound on "how
   well am I using this GPU."
2. FLOPs are counted for one forward pass and scaled by ``fwd_bwd_factor`` (3.0
   by default: ~1× forward + ~2× backward, the standard approximation). This is
   an estimate of training FLOPs, not an exact count of every kernel.

Everything degrades gracefully: no torch, unknown device, or no sample input →
return ``None`` and the budget simply omits the MFU anchor.
"""

from __future__ import annotations

# Approximate **dense** tensor-core peak, FLOP/s, at typical bf16/fp16 training
# precision. Sources vary; treat as ceilings and override when it matters.
_PEAK_FLOPS = {
    "h100": 989e12,
    "a100": 312e12,
    "a10": 125e12,
    "l4": 121e12,
    "l40": 181e12,
    "v100": 125e12,
    "t4": 65e12,
    "rtx 4090": 165e12,
    "rtx 3090": 71e12,
    "rtx a6000": 155e12,
    "rtx 4080": 98e12,
}


def peak_flops_for(device_name: str) -> float | None:
    """Look up approximate dense peak FLOP/s by GPU name (substring match)."""
    if not device_name:
        return None
    name = device_name.lower()
    for key, peak in _PEAK_FLOPS.items():
        if key in name:
            return peak
    return None


def current_device_peak() -> tuple[str | None, float | None]:
    """(device_name, peak_flops) for the active CUDA device, if recognized."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name()
            return name, peak_flops_for(name)
    except Exception:
        pass
    return None, None


def measure_flops(model, example_inputs, fwd_bwd_factor: float = 3.0) -> float | None:
    """Estimate training FLOPs/step = forward FLOPs × ``fwd_bwd_factor``.

    Uses ``torch.utils.flop_counter.FlopCounterMode`` (torch >= 2.0). Runs one
    forward pass under the counter. ``example_inputs`` may be a tensor, a tuple of
    positional args, or a dict of kwargs. Returns ``None`` if unavailable.
    """
    try:
        import torch  # type: ignore
        from torch.utils.flop_counter import FlopCounterMode  # type: ignore
    except Exception:
        return None

    try:
        was_training = model.training
        model.eval()
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            if isinstance(example_inputs, dict):
                model(**example_inputs)
            elif isinstance(example_inputs, (tuple, list)):
                model(*example_inputs)
            else:
                model(example_inputs)
        forward_flops = float(counter.get_total_flops())
        if was_training:
            model.train()
    except Exception:
        return None

    if forward_flops <= 0:
        return None
    return forward_flops * fwd_bwd_factor
