"""Memory collector — real device memory for the memory vertical.

Returns a flat dict of bytes-valued stats for the current step, or {} when no
accelerator is available. Supports CUDA and Apple MPS. Kept cheap so it can run
every step during live training.
"""

from __future__ import annotations


def snapshot(reset_peak: bool = False) -> dict[str, float]:
    """Sample device memory. ``reset_peak`` mutates global CUDA peak stats, so it
    is OFF by default — never silently clobber the user's own tracking."""
    try:
        import torch  # type: ignore
    except Exception:
        return {}

    if torch.cuda.is_available():
        stats = {
            "alloc": float(torch.cuda.memory_allocated()),
            "reserved": float(torch.cuda.memory_reserved()),
            "peak_alloc": float(torch.cuda.max_memory_allocated()),
            "peak_reserved": float(torch.cuda.max_memory_reserved()),
        }
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        return stats

    # Apple MPS: no peak API, but current + driver-reserved are available.
    try:
        if torch.backends.mps.is_available():
            return {
                "alloc": float(torch.mps.current_allocated_memory()),
                "reserved": float(torch.mps.driver_allocated_memory()),
            }
    except Exception:
        pass

    return {}
