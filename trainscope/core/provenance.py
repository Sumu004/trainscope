"""Environment capture — the seed of the reproducibility vertical.

Cheap to call once at run start; stored in run.json. Later, `repro` analyzer
diffs two runs' provenance blocks to explain why results differ.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any


def capture_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    # Torch / CUDA details are the high-value repro signals; capture if present.
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        # Determinism-relevant flags
        info["cudnn_deterministic"] = bool(
            getattr(torch.backends.cudnn, "deterministic", False)
        )
        info["cudnn_benchmark"] = bool(getattr(torch.backends.cudnn, "benchmark", False))
    except Exception:  # torch optional — never let capture break a run
        info["torch"] = None

    # Common seed env vars that affect determinism.
    for key in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS"):
        if key in os.environ:
            info.setdefault("env", {})[key] = os.environ[key]

    return info
