"""Self-contained HTML report — a "hardware panel" rendering of a run.

One file, no deps, no network fonts/assets: inline CSS only, in a dark/amber
LED-panel aesthetic (segmented-digit displays for headline numbers, LED-style
meters for phase/budget breakdowns, lit indicators for findings by severity).
Built to be **compact** — the whole story of a run on one scrollable screen,
shareable as a single `.html` file or attached to a CI artifact.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass

from ..analyzers.convergence import ConvergenceSummary
from ..analyzers.efficiency import EfficiencyBudget
from ..analyzers.memory import MemorySummary
from ..analyzers.timing import TimingSummary
from ..diagnosis.engine import Finding

_MB = 1024 * 1024

_SEV_ORDER = {"high": 0, "med": 1, "low": 2}
_SEV_LABEL = {"high": "HIGH", "med": "MED", "low": "LOW"}


def _esc(s: object) -> str:
    return _html.escape(str(s))


def _digits(value: str, *, glow: bool = True) -> str:
    """A row of LED/segmented-digit cells — the headline-number look."""
    cls = "ts-digits ts-glow" if glow else "ts-digits"
    cells = "".join(f'<span class="ts-digit">{_esc(ch)}</span>' for ch in value)
    return f'<div class="{cls}">{cells}</div>'


def _meter(label: str, frac: float, *, sub: str = "", kind: str = "neutral") -> str:
    """One LED-bar meter row (label · lit segments · sub-readout)."""
    frac = max(0.0, min(1.0, frac))
    n_lit = round(frac * 20)
    segs = "".join(
        f'<span class="ts-seg {"lit-" + kind if i < n_lit else "unlit"}"></span>'
        for i in range(20)
    )
    return (
        '<div class="ts-meter-row">'
        f'<span class="ts-meter-label">{_esc(label)}</span>'
        f'<span class="ts-meter-bar">{segs}</span>'
        f'<span class="ts-meter-sub">{_esc(sub)}</span>'
        "</div>"
    )


def _panel(title: str, body: str) -> str:
    return (
        '<section class="ts-panel">'
        f'<h2 class="ts-panel-title">{_esc(title)}</h2>'
        f"{body}"
        "</section>"
    )


@dataclass
class _Stat:
    label: str
    value: str
    sub: str = ""


def _stat_row(stats: list[_Stat]) -> str:
    cells = "".join(
        '<div class="ts-stat">'
        f'<div class="ts-stat-label">{_esc(s.label)}</div>'
        f"{_digits(s.value)}"
        + (f'<div class="ts-stat-sub">{_esc(s.sub)}</div>' if s.sub else "")
        + "</div>"
        for s in stats
    )
    return f'<div class="ts-stat-row">{cells}</div>'


_PHASE_KIND = {"data": "amber", "comm": "amber"}
_BUDGET_KIND = {"useful_compute": "green"}


def _timing_section(t: TimingSummary | None) -> str:
    if not t or t.n_steps == 0:
        return ""
    stats = [
        _Stat("STEPS", str(t.n_steps)),
        _Stat("MS/STEP", f"{t.mean_step_time * 1e3:.1f}"),
        _Stat("STEPS/S", f"{t.steps_per_sec:.1f}"),
        _Stat("CV", f"{t.cv:.2f}", "jitter"),
    ]
    rows = "".join(
        _meter(
            phase,
            t.phase_fractions[phase],
            sub=f"{t.phase_fractions[phase] * 100:4.1f}%  "
            f"{t.phase_seconds[phase] * 1e3:6.2f} ms",
            kind=_PHASE_KIND.get(phase, "neutral"),
        )
        for phase in t.phase_order
    )
    return _panel("Timing", _stat_row(stats) + rows)


def _budget_section(b: EfficiencyBudget | None) -> str:
    if not b:
        return ""
    stats = []
    if b.mfu is not None:
        stats.append(_Stat("MFU", f"{b.mfu * 100:.1f}%", f"of {b.wall:.1f}s wall"))
    else:
        stats.append(_Stat("WALL", f"{b.wall:.1f}s", f"{b.n_steps} steps · MFU n/a"))
    rows = "".join(
        _meter(
            ln.name,
            ln.fraction,
            sub=f"{ln.fraction * 100:4.1f}%  {ln.seconds:6.2f}s"
            + ("  recoverable" if ln.recoverable else ""),
            kind=_BUDGET_KIND.get(ln.name, "amber" if ln.recoverable else "neutral"),
        )
        for ln in b.lines
        if ln.seconds > 0 or ln.name == "useful_compute"
    )
    return _panel("Efficiency budget", _stat_row(stats) + rows)


def _memory_section(m: MemorySummary | None) -> str:
    if not m or not m.has_memory:
        return ""
    stats = [
        _Stat("PEAK ALLOC", f"{m.peak_alloc_bytes / _MB:.0f}", "MB"),
        _Stat("PEAK RESV", f"{m.peak_reserved_bytes / _MB:.0f}", "MB"),
    ]
    if m.peak_alloc_bytes >= 64 * _MB:
        stats.append(_Stat("FRAG", f"{m.fragmentation * 100:.0f}%"))
    if m.growth_bytes_per_step > 0:
        stats.append(_Stat("GROWTH", f"{m.growth_bytes_per_step / _MB:.2f}", "MB/step"))
    return _panel("Memory", _stat_row(stats))


def _convergence_section(c: ConvergenceSummary | None) -> str:
    if not c or not c.has_loss:
        return ""
    best = f"{c.best_loss:.4g}" if c.best_loss is not None else "n/a"
    final = f"{c.final_loss:.4g}" if c.final_loss is not None else "n/a"
    stats = [
        _Stat("TREND", c.loss_trend.upper()),
        _Stat("BEST", best),
        _Stat("FINAL", final),
    ]
    extra = ""
    if c.diverged_at is not None:
        extra += f'<div class="ts-alert">⚠ DIVERGED at step {_esc(c.diverged_at)}</div>'
    return _panel("Convergence", _stat_row(stats) + extra)


def _findings_section(findings: list[Finding]) -> str:
    if not findings:
        msg = "No issues found — training looks balanced"
        body = f'<div class="ts-led ts-led-green">● {msg}</div>'
        return _panel("Findings", body)
    ordered = sorted(findings, key=lambda f: _SEV_ORDER.get(f.severity, 9))
    rows = []
    for f in ordered:
        sev = f.severity if f.severity in _SEV_LABEL else "low"
        rows.append(
            '<div class="ts-finding">'
            f'<span class="ts-led ts-led-{_esc(sev)}">● {_SEV_LABEL[sev]}</span>'
            f'<span class="ts-finding-title">{_esc(f.title)}</span>'
            f'<span class="ts-finding-code">{_esc(f.code)}</span>'
            f'<div class="ts-finding-detail">{_esc(f.detail)}</div>'
            + (
                f'<div class="ts-finding-fix">→ {_esc(f.suggestion)}</div>'
                if f.suggestion
                else ""
            )
            + "</div>"
        )
    return _panel(f"Findings ({len(findings)})", "".join(rows))


_CSS = """
:root {
  --bg: #1a0a05; --panel: #2a1208; --amber: #ffb000; --amber-dim: #8a5a1a;
  --green: #6ee06e; --red: #ff5a4d; --yellow: #ffd23f; --ink: #f4d9a8;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 16px; background: var(--bg); color: var(--ink);
  font-family: ui-monospace, SFMono-Regular, "JetBrains Mono", Consolas, monospace;
}
.ts-wrap { max-width: 880px; margin: 0 auto; }
.ts-header {
  border: 1px solid var(--amber-dim); border-radius: 14px; padding: 18px 24px;
  margin-bottom: 22px;
  background: radial-gradient(ellipse at top left, #3a1c0a, var(--panel));
  box-shadow: inset 0 0 24px rgba(255,176,0,0.06);
}
.ts-title {
  margin: 0; color: var(--amber); letter-spacing: 0.12em; font-size: 22px;
  text-shadow: 0 0 10px rgba(255,176,0,0.55);
}
.ts-subtitle {
  margin: 6px 0 0; color: var(--amber-dim); font-size: 13px;
  letter-spacing: 0.06em;
}
.ts-panel {
  border: 1px solid var(--amber-dim); border-radius: 12px; padding: 16px 20px;
  margin-bottom: 16px; background: var(--panel);
}
.ts-panel-title {
  margin: 0 0 12px; font-size: 13px; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--amber-dim);
  border-bottom: 1px solid var(--amber-dim); padding-bottom: 8px;
}
.ts-stat-row { display: flex; flex-wrap: wrap; gap: 22px; margin-bottom: 14px; }
.ts-stat { min-width: 96px; }
.ts-stat-label {
  font-size: 11px; letter-spacing: 0.14em; color: var(--amber-dim);
  margin-bottom: 4px;
}
.ts-stat-sub { font-size: 11px; color: var(--amber-dim); margin-top: 2px; }
.ts-digits { display: inline-flex; gap: 2px; }
.ts-digit {
  display: inline-block; min-width: 16px; padding: 2px 4px; text-align: center;
  font-size: 22px; font-weight: 700; color: var(--amber); background: #150a04;
  border-radius: 4px; border: 1px solid #3a2410;
}
.ts-glow .ts-digit {
  text-shadow: 0 0 8px rgba(255,176,0,0.75), 0 0 2px rgba(255,176,0,0.9);
}
.ts-meter-row {
  display: flex; align-items: center; gap: 10px; font-size: 12px;
  padding: 3px 0;
}
.ts-meter-label {
  width: 110px; color: var(--ink); letter-spacing: 0.06em;
  text-transform: uppercase; flex-shrink: 0;
}
.ts-meter-bar { display: inline-flex; gap: 2px; }
.ts-seg {
  width: 9px; height: 13px; border-radius: 1px; background: #3a2410;
  display: inline-block;
}
.ts-seg.lit-amber {
  background: var(--amber); box-shadow: 0 0 5px rgba(255,176,0,0.85);
}
.ts-seg.lit-green {
  background: var(--green); box-shadow: 0 0 5px rgba(110,224,110,0.85);
}
.ts-seg.lit-neutral {
  background: var(--ink); box-shadow: 0 0 4px rgba(244,217,168,0.6);
}
.ts-meter-sub { color: var(--amber-dim); white-space: pre; margin-left: auto; }
.ts-led {
  display: inline-block; font-size: 11px; letter-spacing: 0.1em;
  padding: 2px 8px; border-radius: 999px; border: 1px solid currentColor;
}
.ts-led-high, .ts-led-red {
  color: var(--red); text-shadow: 0 0 6px rgba(255,90,77,0.7);
}
.ts-led-med, .ts-led-yellow {
  color: var(--yellow); text-shadow: 0 0 6px rgba(255,210,63,0.7);
}
.ts-led-low { color: #6db8ff; text-shadow: 0 0 6px rgba(109,184,255,0.6); }
.ts-led-green {
  color: var(--green); text-shadow: 0 0 6px rgba(110,224,110,0.7);
}
.ts-finding { padding: 10px 0; border-top: 1px solid #3a2410; }
.ts-finding:first-child { border-top: none; }
.ts-finding-title { font-weight: 700; margin-left: 10px; color: var(--ink); }
.ts-finding-code { float: right; color: var(--amber-dim); font-size: 11px; }
.ts-finding-detail {
  margin: 6px 0 0 0; font-size: 12px; color: var(--ink); opacity: 0.85;
}
.ts-finding-fix { margin-top: 4px; font-size: 12px; color: var(--amber); }
.ts-alert { color: var(--red); font-weight: 700; }
.ts-footer {
  text-align: center; color: var(--amber-dim); font-size: 11px;
  margin-top: 20px; letter-spacing: 0.08em;
}
"""


def render_html_report(
    name: str,
    run_dir: str,
    timing: TimingSummary | None = None,
    findings: list[Finding] | None = None,
    memory: MemorySummary | None = None,
    convergence: ConvergenceSummary | None = None,
    efficiency: EfficiencyBudget | None = None,
) -> str:
    """Render a complete, self-contained HTML report for one run.

    Returns a full ``<html>`` document string — write it to a ``.html`` file
    and open it in a browser, no server or build step needed.
    """
    findings = findings or []
    sections = "".join(
        s
        for s in (
            _timing_section(timing),
            _budget_section(efficiency),
            _memory_section(memory),
            _convergence_section(convergence),
            _findings_section(findings),
        )
        if s
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trainscope — {_esc(name)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="ts-wrap">
  <header class="ts-header">
    <h1 class="ts-title">⏻ TRAINSCOPE — {_esc(name).upper()}</h1>
    <p class="ts-subtitle">{_esc(run_dir)}</p>
  </header>
  {sections}
  <div class="ts-footer">generated by trainscope · single-file report, no deps</div>
</div>
</body>
</html>
"""
