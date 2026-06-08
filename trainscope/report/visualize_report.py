"""``trainscope visualize`` — a chart-based dashboard for one run.

Renders trend charts (step time, loss, grad-norm, memory) and breakdown bars
(phase attribution, efficiency budget, per-rank straggler analysis) as a
single self-contained HTML file: inline SVG + CSS, no JS, no network assets,
no new dependencies — keeping the project's pure-stdlib core intact while
giving you something more visual than a terminal report to stare at, page
through, or drop in a CI artifact.
"""

from __future__ import annotations

from ..analyzers.convergence import ConvergenceSummary
from ..analyzers.distributed import DistributedSummary
from ..analyzers.efficiency import EfficiencyBudget
from ..analyzers.memory import MemorySummary
from ..analyzers.timing import TimingSummary
from .charts import CHART_CSS, bar_rows, line_chart
from .html_report import _CSS, _esc, _panel

_PHASE_KIND = {"data": "bad", "comm": "bad"}
_BUDGET_KIND = {"useful_compute": "good"}


def _timing_panel(t: TimingSummary | None, steps: list | None) -> str:
    if not t or t.n_steps == 0:
        return ""
    body = ""
    if steps:
        body += line_chart(
            [s.total() * 1e3 for s in steps],
            label="step time (ms) over the run",
        )
    rows = [
        (
            phase,
            t.phase_fractions[phase],
            f"{t.phase_fractions[phase] * 100:5.1f}%  "
            f"{t.phase_seconds[phase] * 1e3:7.2f} ms",
            _PHASE_KIND.get(phase, "neutral"),
        )
        for phase in t.phase_order
    ]
    body += bar_rows(rows)
    title = f"Timing — {t.n_steps} steps, {t.mean_step_time * 1e3:.1f} ms/step"
    return _panel(title, body)


def _convergence_panel(c: ConvergenceSummary | None, steps: list | None) -> str:
    if not c or not c.has_loss or not steps:
        return ""
    body = line_chart(
        [s.scalars["loss"] for s in steps if "loss" in s.scalars],
        label="loss over the run",
    )
    grad = [s.scalars["grad_norm"] for s in steps if "grad_norm" in s.scalars]
    body += line_chart(grad, label="grad-norm over the run")
    if not body:
        return ""
    return _panel(f"Convergence — trend {c.loss_trend}", body)


def _memory_panel(m: MemorySummary | None) -> str:
    if not m or not m.has_memory or not m.alloc_series:
        return ""
    body = line_chart(
        [v / (1024 * 1024) for v in m.alloc_series],
        label="allocated memory (MB) over the run",
    )
    if not body:
        return ""
    return _panel("Memory — allocation over time", body)


def _distributed_panel(d: DistributedSummary | None) -> str:
    if not d:
        return ""
    rows = []
    for rs in d.ranks:
        is_straggler = bool(d.straggler and rs.rank == d.straggler.rank)
        rows.append(
            (
                f"rank {rs.rank}" + (" *" if is_straggler else ""),
                rs.slowest_fraction,
                f"{rs.mean_compute * 1e3:7.2f} ms · "
                f"{rs.slowest_fraction * 100:4.0f}% crit · z={rs.straggler_z:+.1f}",
                "bad" if is_straggler else "neutral",
            )
        )
    body = bar_rows(rows)
    title = f"Distributed — {d.world_size} ranks · % steps on critical path"
    return _panel(title, body)


def _budget_panel(b: EfficiencyBudget | None) -> str:
    if not b:
        return ""
    rows = [
        (
            ln.name,
            ln.fraction,
            f"{ln.fraction * 100:5.1f}%  {ln.seconds:7.2f}s"
            + ("  recoverable" if ln.recoverable else ""),
            _BUDGET_KIND.get(ln.name, "bad" if ln.recoverable else "neutral"),
        )
        for ln in b.lines
        if ln.seconds > 0 or ln.name == "useful_compute"
    ]
    title = "Efficiency budget — wall-time decomposition"
    if b.mfu is not None:
        title += f" (MFU {b.mfu * 100:.1f}%)"
    return _panel(title, bar_rows(rows))


def render_visualization(
    name: str,
    run_dir: str,
    steps: list | None = None,
    timing: TimingSummary | None = None,
    memory: MemorySummary | None = None,
    convergence: ConvergenceSummary | None = None,
    distributed: DistributedSummary | None = None,
    efficiency: EfficiencyBudget | None = None,
) -> str:
    """Render a complete, self-contained chart dashboard for one run.

    Pure SVG + inline CSS — the same dependency-free, single-file contract as
    ``render_html_report``: open it anywhere, attach it to CI, no build step.
    """
    panels = "".join(
        s
        for s in (
            _timing_panel(timing, steps),
            _convergence_panel(convergence, steps),
            _memory_panel(memory),
            _distributed_panel(distributed),
            _budget_panel(efficiency),
        )
        if s
    )
    if not panels:
        panels = _panel(
            "No chartable data",
            '<div class="ts-led ts-led-low">'
            "● This run has no per-step series to chart "
            "(timing/loss/grad-norm/memory/distributed/budget all empty)."
            "</div>",
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trainscope visualize — {_esc(name)}</title>
<style>{_CSS}{CHART_CSS}</style>
</head>
<body>
<div class="ts-wrap">
  <header class="ts-header">
    <h1 class="ts-title">📈 TRAINSCOPE VISUALIZE — {_esc(name).upper()}</h1>
    <p class="ts-subtitle">{_esc(run_dir)}</p>
  </header>
  {panels}
  <div class="ts-footer">generated by trainscope · charts are inline SVG, no deps</div>
</div>
</body>
</html>
"""
