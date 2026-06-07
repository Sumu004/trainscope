"""Small, robust statistics used across analyzers.

Median/MAD are used instead of mean/std because training signals (loss,
grad-norm, step time) are heavy-tailed — a single huge spike would inflate a
std-based threshold and hide everything else. MAD is outlier-resistant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Scale factor making MAD a consistent estimator of std for normal data.
_MAD_TO_STD = 1.4826


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def mad(xs: Sequence[float], med: float | None = None) -> float:
    """Median absolute deviation."""
    if not xs:
        return 0.0
    if med is None:
        med = median(xs)
    return median([abs(x - med) for x in xs])


def upward_spikes(
    values: Sequence[float | None], k: float = 5.0, min_n: int = 8
) -> set[int]:
    """Indices of upward outliers: finite, and > median + k·(MAD·1.4826).

    Returns positions into ``values`` (None / non-finite entries are ignored for
    the threshold but never reported as spikes). Empty set if too few points or
    no spread (MAD == 0).
    """
    finite = [
        (i, float(v)) for i, v in enumerate(values) if v is not None and math.isfinite(v)
    ]
    if len(finite) < min_n:
        return set()
    vs = [v for _, v in finite]
    med = median(vs)
    scale = mad(vs, med) * _MAD_TO_STD
    if scale <= 0.0:
        return set()
    thr = med + k * scale
    return {i for i, v in finite if v > thr}


def local_spikes(
    values: Sequence[float | None],
    window: int = 12,
    k: float = 6.0,
    min_hist: int = 6,
    rel: float = 0.5,
) -> set[int]:
    """Indices of *upward* spikes relative to a rolling local baseline.

    Detrending locally (vs the global median) is what makes this correct on
    training signals: a smoothly *decaying* loss is never flagged (its recent
    baseline tracks the decay), while a sudden jump is. When the recent baseline
    is perfectly flat (MAD == 0, e.g. a constant grad-norm), a value exceeding it
    by more than ``rel`` is flagged — so a clean baseline doesn't blind us.

    Non-finite / None entries are skipped (the caller handles divergence).
    """
    spikes: set[int] = set()
    hist: list[float] = []  # recent finite values, oldest first
    for i, raw in enumerate(values):
        if raw is None:
            continue
        v = float(raw)
        if not math.isfinite(v):
            continue
        if len(hist) >= min_hist:
            med = median(hist)
            scale = mad(hist, med) * _MAD_TO_STD
            if scale > 0.0:
                if v > med + k * scale:
                    spikes.add(i)
            else:  # flat recent baseline
                if v > med and (v > med * (1.0 + rel) if med > 0 else v > med):
                    spikes.add(i)
        hist.append(v)
        if len(hist) > window:
            hist.pop(0)
    return spikes


def robust_slope(values: Sequence[float], frac: float = 0.1) -> float:
    """Per-step trend via (median of last ``frac`` − median of first ``frac``)
    divided by the step span. Robust to spikes, unlike least squares."""
    n = len(values)
    if n < 2:
        return 0.0
    w = max(1, int(n * frac))
    first = median(values[:w])
    last = median(values[-w:])
    span = n - 1
    return (last - first) / span if span else 0.0
