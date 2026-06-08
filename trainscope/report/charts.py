"""Pure-stdlib SVG chart primitives — no matplotlib, no deps, no network
fonts. Inline ``<svg>`` strings rendered by string formatting, in the same
amber/LED visual language as ``html_report``. Small enough to embed directly
in a self-contained HTML dashboard.
"""

from __future__ import annotations

import html as _html

_AMBER = "#ffb000"
_GREEN = "#6ee06e"
_RED = "#ff5a4d"
_GRID = "#3a2410"


def _esc(s: object) -> str:
    return _html.escape(str(s))


def line_chart(
    values: list[float],
    *,
    width: int = 640,
    height: int = 140,
    color: str = _AMBER,
    fill: bool = True,
    label: str = "",
) -> str:
    """A filled line/trend chart over a numeric series — the workhorse for
    step-time, loss, grad-norm, and memory-over-time views. Degrades to an
    empty string for too-short or constant-NaN series (nothing misleading)."""
    vals = [v for v in values if v == v]  # drop NaNs
    if len(vals) < 2:
        return ""
    pad = 10
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = len(vals)
    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    def _x(i: int) -> float:
        return pad + i * plot_w / (n - 1)

    def _y(v: float) -> float:
        return pad + plot_h - (v - lo) / span * plot_h

    pts = [(_x(i), _y(v)) for i, v in enumerate(vals)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="ts-chart" '
        'preserveAspectRatio="none" role="img">',
        # Baseline gridlines (low / mid / high) — cheap orientation cues.
        f'<line x1="{pad}" y1="{pad}" x2="{width - pad}" y2="{pad}" '
        f'class="ts-chart-grid" />',
        f'<line x1="{pad}" y1="{height / 2:.1f}" x2="{width - pad}" '
        f'y2="{height / 2:.1f}" class="ts-chart-grid" />',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" '
        f'y2="{height - pad}" class="ts-chart-grid" />',
    ]
    if fill:
        base = height - pad
        area = f"{pad:.1f},{base:.1f} " + poly + f" {pts[-1][0]:.1f},{base:.1f}"
        parts.append(f'<polygon points="{area}" class="ts-chart-fill" />')
    parts.append(f'<polyline points="{poly}" class="ts-chart-line" />')
    parts.append("</svg>")
    svg = "".join(parts)
    cap = (
        f'<div class="ts-chart-cap">{_esc(label)} '
        f"<span class='ts-chart-range'>min {lo:.4g} · max {hi:.4g}</span></div>"
        if label
        else ""
    )
    return f'<div class="ts-chart-wrap">{svg}{cap}</div>'


def bar_rows(
    rows: list[tuple[str, float, str, str]],
    *,
    width: int = 640,
) -> str:
    """A column of horizontal LED-style bars: ``(label, fraction, sub, kind)``.
    ``kind`` is ``"good" | "bad" | "neutral"`` and selects the fill color —
    the same red→yellow→green grammar as the CLI/HTML reports, so all three
    surfaces read consistently."""
    color_for = {"good": _GREEN, "bad": _RED, "neutral": _AMBER}
    out = ['<div class="ts-bars">']
    for label, frac, sub, kind in rows:
        frac = max(0.0, min(1.0, frac))
        color = color_for.get(kind, _AMBER)
        out.append(
            '<div class="ts-bar-row">'
            f'<span class="ts-bar-label">{_esc(label)}</span>'
            '<span class="ts-bar-track">'
            f'<span class="ts-bar-fill" style="width:{frac * 100:.1f}%;'
            f'background:{color};box-shadow:0 0 6px {color}99;"></span>'
            "</span>"
            f'<span class="ts-bar-sub">{_esc(sub)}</span>'
            "</div>"
        )
    out.append("</div>")
    return "".join(out)


CHART_CSS = (
    """
.ts-chart-wrap { margin: 6px 0 14px; }
.ts-chart {
  width: 100%; height: 140px; display: block;
  background: #150a04; border: 1px solid var(--amber-dim); border-radius: 8px;
}
.ts-chart-grid { stroke: """
    + _GRID
    + """; stroke-width: 1; stroke-dasharray: 3 4; }
.ts-chart-line {
  fill: none; stroke: var(--amber); stroke-width: 2;
  filter: drop-shadow(0 0 3px rgba(255,176,0,0.65));
}
.ts-chart-fill { fill: rgba(255,176,0,0.12); stroke: none; }
.ts-chart-cap {
  margin-top: 4px; font-size: 11px; color: var(--amber-dim);
  letter-spacing: 0.06em; text-transform: uppercase;
  display: flex; justify-content: space-between;
}
.ts-chart-range { color: var(--ink); opacity: 0.7; text-transform: none; }
.ts-bars { display: flex; flex-direction: column; gap: 6px; }
.ts-bar-row { display: flex; align-items: center; gap: 10px; font-size: 12px; }
.ts-bar-label {
  width: 120px; flex-shrink: 0; color: var(--ink); letter-spacing: 0.05em;
  text-transform: uppercase; font-size: 11px;
}
.ts-bar-track {
  flex: 1; height: 12px; background: #150a04; border-radius: 3px;
  border: 1px solid """
    + _GRID
    + """; overflow: hidden;
}
.ts-bar-fill { display: block; height: 100%; border-radius: 2px; }
.ts-bar-sub {
  width: 200px; flex-shrink: 0; color: var(--amber-dim); white-space: pre;
  font-size: 11px; text-align: right;
}
"""
)
