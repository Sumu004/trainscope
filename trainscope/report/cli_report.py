"""Terminal report. No color deps (pure ANSI escapes) — auto-detects whether
the terminal can render them, and degrades to identical plain text when piped
to a file, redirected in CI, or when ``NO_COLOR``/``--color=never`` is set. See
https://no-color.org/. Always readable either way."""

from __future__ import annotations

import os
import sys

from ..analyzers.convergence import ConvergenceSummary
from ..analyzers.memory import MemorySummary
from ..analyzers.timing import TimingSummary
from ..diagnosis.engine import Finding

_MB = 1024 * 1024

_SEV_LABEL = {"high": "HIGH", "med": "MED ", "low": "LOW "}
_BAR_WIDTH = 30

# ---------------------------------------------------------------------------
# Color: ANSI SGR codes only (no deps). `set_color_mode` lets the CLI honor an
# explicit `--color {auto,always,never}` flag; absent that, we auto-detect via
# the NO_COLOR / FORCE_COLOR conventions and whether stdout is a real terminal,
# so piping to a file or `| cat` in CI silently yields plain text.
# ---------------------------------------------------------------------------
_RESET = "\x1b[0m"
_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
}

_color_override: bool | None = None


def set_color_mode(mode: str) -> None:
    """``mode`` is one of ``auto`` (default, auto-detect), ``always``, ``never``."""
    global _color_override
    _color_override = {"always": True, "never": False}.get(mode)


def _use_color() -> bool:
    if _color_override is not None:
        return _color_override
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    stream = getattr(sys, "stdout", None)
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _style(text: str, *codes: str) -> str:
    if not text or not _use_color():
        return text
    seq = ";".join(_CODES[c] for c in codes)
    return f"\x1b[{seq}m{text}{_RESET}"


_SEV_STYLE = {"high": ("bold", "red"), "med": ("yellow",), "low": ("cyan",)}


def _severity(sev: str) -> str:
    label = _SEV_LABEL.get(sev, "?")
    return _style(label, *_SEV_STYLE.get(sev, ()))


def _heading(text: str) -> str:
    return _style(text, "bold")


# A lit/unlit "panel indicator" — the same retro hardware-panel visual
# language as a glowing LED, brought into the terminal: a colored dot in
# front of each section heading. Severity colors line up with `_SEV_STYLE`
# so the eye learns one grammar across the whole report.
_LED_COLOR = {
    "high": "red",
    "med": "yellow",
    "low": "cyan",
    "green": "green",
    "amber": "yellow",
    "neutral": "dim",
}


def _led(kind: str = "amber") -> str:
    return _style("●", _LED_COLOR.get(kind, "dim"))


def _panel(title: str, *, led: str = "amber") -> str:
    """A lit-indicator section heading — ``● TITLE`` — the terminal's take on
    the hardware-panel aesthetic (every section is a 'lit panel')."""
    return f"{_led(led)} {_heading(title)}"


# Three-stop gradient (good→ok→bad) by filled fraction. `kind="good"` means a
# high fraction is *desirable* (useful compute, overlap efficiency) so the
# ramp is reversed; `kind="bad"` means high is a problem (overhead, exposed
# comm, data stall); `kind=None` is neutral (e.g. plain timing breakdown).
def _gradient(frac: float, kind: str | None) -> str | None:
    if kind is None:
        return None
    lo, hi = ("red", "green") if kind == "good" else ("green", "red")
    if frac < 1 / 3:
        return lo
    if frac < 2 / 3:
        return "yellow"
    return hi


def _bar(frac: float, kind: str | None = None) -> str:
    """LED-meter-style bar: lit segments (█) vs unlit (░) — same visual
    language as the HTML report's segmented meters, in plain ANSI/unicode."""
    filled = int(round(frac * _BAR_WIDTH))
    fill, empty = "█" * filled, "░" * (_BAR_WIDTH - filled)
    color = _gradient(frac, kind)
    return (_style(fill, color) if color else fill) + _style(empty, "dim")


_SPARK_TICKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], *, color: str | None = None) -> str:
    """Compress a numeric series into one line of unicode block ticks — a
    cheap, dependency-free trend chart that survives copy/paste into a log."""
    vals = [v for v in values if v == v]  # drop NaNs
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    ticks = "".join(
        _SPARK_TICKS[min(len(_SPARK_TICKS) - 1, int((v - lo) / span * len(_SPARK_TICKS)))]
        for v in vals
    )
    return _style(ticks, color) if color else ticks


def render_timing(t: TimingSummary, steps: list | None = None) -> str:
    if t.n_steps == 0:
        return "No steps recorded.\n"

    # Amber by default; red if a stall phase (data/comm) dominates the step —
    # the panel "lights up" the same way a hardware meter would peak into the red.
    stall_frac = sum(t.phase_fractions.get(p, 0.0) for p in ("data", "comm"))
    lines = [
        _panel(
            f"TIMING — {t.n_steps} steps · {t.mean_step_time * 1e3:.1f} ms/step · "
            f"{t.steps_per_sec:.1f} steps/s",
            led="high" if stall_frac >= 0.5 else "amber",
        )
        + _style(
            f"   (median {t.p50_step_time * 1e3:.1f} · "
            f"p95 {t.p95_step_time * 1e3:.1f} ms · CV {t.cv:.2f})",
            "dim",
        )
    ]
    if steps:
        spark = _sparkline([s.total() for s in steps])
        if spark:
            lines.append(f"  step time  {spark}  (low→high)")
    # Data/comm stalls are overhead (red-when-high); compute phases are neutral.
    _kind = {"data": "bad", "comm": "bad"}
    for phase in t.phase_order:
        frac = t.phase_fractions[phase]
        ms = t.phase_seconds[phase] * 1e3
        bar = _bar(frac, _kind.get(phase))
        lines.append(f"  {phase:<10} {bar} {frac * 100:5.1f}%  {ms:7.2f} ms")
    return "\n".join(lines) + "\n"


def render_findings(findings: list[Finding]) -> str:
    if not findings:
        return f"{_led('green')} " + _style(
            "No issues found. Training looks balanced.\n", "green"
        )
    worst = next(
        (s for s in ("high", "med", "low") if any(f.severity == s for f in findings)),
        "low",
    )
    lines = [_panel(f"FINDINGS ({len(findings)})", led=worst)]
    for f in findings:
        lines.append(f"  [{_severity(f.severity)}] {_style(f.title, 'bold')}  ({f.code})")
        lines.append(f"        {f.detail}")
        if f.suggestion:
            lines.append(_style(f"        -> {f.suggestion}", "cyan"))
    return "\n".join(lines) + "\n"


def render_memory(m: MemorySummary) -> str:
    if not m or not m.has_memory:
        return ""
    led = "high" if m.growth_bytes_per_step > 0 else "amber"
    lines = [_panel("MEMORY", led=led)]
    head = (
        f"  peak alloc {m.peak_alloc_bytes / _MB:.0f} MB · "
        f"peak reserved {m.peak_reserved_bytes / _MB:.0f} MB"
    )
    # Fragmentation from a tiny allocation is allocator noise (and on MPS the
    # step-boundary alloc is a resident trough, not the in-step peak) — hide it.
    if m.peak_alloc_bytes >= 64 * _MB:
        head += f" · fragmentation {m.fragmentation * 100:.0f}%"
    lines.append(head)
    if m.growth_bytes_per_step > 0:
        msg = f"  growth {m.growth_bytes_per_step / _MB:.2f} MB/step"
        lines.append(_style(msg, "yellow"))
    return "\n".join(lines) + "\n"


def render_convergence(c: ConvergenceSummary, steps: list | None = None) -> str:
    if not c or not c.has_loss:
        return ""
    led = (
        "high" if (c.diverged_at is not None or c.loss_trend == "worsening") else "amber"
    )
    lines = [_panel("CONVERGENCE", led=led)]
    best = f"{c.best_loss:.4g}" if c.best_loss is not None else "n/a"
    final = f"{c.final_loss:.4g}" if c.final_loss is not None else "n/a"
    trend_color = {"improving": "green", "worsening": "red", "diverged": "red"}.get(
        c.loss_trend
    )
    trend = _style(c.loss_trend, trend_color) if trend_color else c.loss_trend
    lines.append(f"  loss trend {trend} · best {best} · final {final}")
    if steps:
        losses = [s.scalars["loss"] for s in steps if "loss" in s.scalars]
        spark = _sparkline(losses, color="green")
        if spark:
            lines.append(f"  loss   {spark}  (low→high)")
    if c.diverged_at is not None:
        lines.append(_style(f"  DIVERGED at step {c.diverged_at}", "bold", "red"))
    if c.loss_spikes:
        msg = f"  loss spikes at steps {c.loss_spikes[:8]}"
        lines.append(_style(msg, "yellow"))
    if c.grad_norm_spikes:
        msg = f"  grad-norm spikes at steps {c.grad_norm_spikes[:8]}"
        lines.append(_style(msg, "yellow"))
    return "\n".join(lines) + "\n"


def render_distributed(d) -> str:
    """Render a DistributedSummary (multi-rank critical-path analysis)."""
    if not d:
        return ""
    head = f"DISTRIBUTED — {d.world_size} ranks, {d.n_steps} aligned steps"
    lines = [_panel(head, led="high" if d.straggler else "amber")]
    lines.append(
        f"  mean step wall {d.mean_step_wall * 1e3:.1f} ms · "
        f"comm {d.mean_comm_fraction * 100:.0f}% · "
        f"imbalance CV {d.imbalance_cv:.2f}"
    )
    lines.append(
        f"  wall lost to imbalance {d.wall_frac_lost_to_imbalance * 100:.1f}% · "
        f"median sync skew {d.sync_skew * 1e3:.1f} ms/step"
    )
    lines.append("  per-rank compute (ms/step) · % steps on critical path:")
    for rs in d.ranks:
        is_straggler = bool(d.straggler and rs.rank == d.straggler.rank)
        flag = _style("  <- straggler", "bold", "red") if is_straggler else ""
        plain_label = f"rank {rs.rank}"
        rank_label = _style(plain_label, "bold", "red") if is_straggler else plain_label
        lines.append(
            f"    {rank_label}: {rs.mean_compute * 1e3:7.2f} ms · "
            f"{_bar(rs.slowest_fraction, 'bad' if is_straggler else None)} "
            f"{rs.slowest_fraction * 100:4.0f}% (z={rs.straggler_z:+.1f}){flag}"
        )
    return "\n".join(lines) + "\n"


def render_pipeline(p) -> str:
    """Render a PipelineSummary (pipeline-bubble analysis)."""
    if not p:
        return ""
    lines = [_panel(f"PIPELINE — {p.n_stages} stages", led="amber")]
    head = f"  bubble {p.bubble_fraction * 100:.0f}%"
    if p.ideal_bubble_fraction is not None:
        head += (
            f" · inherent min {p.ideal_bubble_fraction * 100:.0f}% "
            f"(m={p.n_microbatches}) · excess {p.excess_bubble * 100:.0f}%"
        )
    lines.append(head)
    return "\n".join(lines) + "\n"


def render_budget(b) -> str:
    """Render an EfficiencyBudget — the wall-time accounting identity + MFU."""
    if not b:
        return ""
    led = "amber"
    if b.mfu is not None:
        led = "green" if b.mfu >= 0.5 else ("med" if b.mfu >= 0.25 else "high")
    lines = [_panel("EFFICIENCY BUDGET — wall-time decomposition", led=led)]
    if b.mfu is not None:
        mfu_color = "green" if b.mfu >= 0.5 else ("yellow" if b.mfu >= 0.25 else "red")
        lines.append(
            f"  MFU {_style(f'{b.mfu * 100:.1f}%', 'bold', mfu_color)}  ·  "
            f"useful compute {b.efficiency * 100:.1f}% of {b.wall:.2f}s wall"
        )
    else:
        lines.append(f"  wall {b.wall:.2f}s over {b.n_steps} steps (MFU: n/a)")
    for ln in b.lines:
        if ln.seconds <= 0 and ln.name != "useful_compute":
            continue
        tag = _style(" (recoverable)", "yellow") if ln.recoverable else ""
        is_useful = ln.name == "useful_compute"
        kind = "good" if is_useful else ("bad" if ln.recoverable else None)
        lines.append(
            f"  {ln.name:<17} {_bar(ln.fraction, kind)} {ln.fraction * 100:5.1f}%  "
            f"{ln.seconds:7.2f}s{tag}"
        )
    return "\n".join(lines) + "\n"


def render_trace(t) -> str:
    """Render a TraceSummary (exposed-communication analysis)."""
    if not t or not t.has_comm:
        return ""
    exp = t.exposed_comm_fraction
    exp_color = "red" if exp >= 0.5 else ("yellow" if exp >= 0.2 else "green")
    led = "high" if exp >= 0.5 else ("med" if exp >= 0.2 else "amber")
    lines = [_panel("COMMUNICATION OVERLAP — from kernel trace", led=led)]
    lines.append(
        f"  comm {t.total_comm_time * 1e3:.1f} ms · "
        f"overlapped {_style(f'{t.overlap_efficiency * 100:.0f}%', 'green')} · "
        f"exposed {_style(f'{t.exposed_comm_time * 1e3:.1f} ms', exp_color)} "
        f"({_style(f'{exp * 100:.0f}%', exp_color)} of wall)"
    )
    if t.per_collective:
        parts = ", ".join(
            f"{k} {v * 1e3:.1f}ms" for k, v in sorted(t.per_collective.items())
        )
        lines.append(f"  by collective: {parts}")
    return "\n".join(lines) + "\n"


def render_diff(d) -> str:
    """Render a RunDiff (reproducibility comparison)."""
    lines = [_heading(f"Comparing  A='{d.name_a}'  vs  B='{d.name_b}'"), ""]

    def _fmt(v):
        return "—" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))

    def _diff_line(fd):
        return f"  {fd.key}: {_fmt(fd.a)}  {_style('->', 'yellow')}  {_fmt(fd.b)}"

    lines.append(_heading("Environment diffs:"))
    if d.env_diffs:
        lines += [_diff_line(fd) for fd in d.env_diffs]
    else:
        lines.append(_style("  (identical)", "dim"))

    lines.append("")
    lines.append(_heading("Config diffs:"))
    if d.config_diffs:
        lines += [_diff_line(fd) for fd in d.config_diffs]
    else:
        lines.append(_style("  (identical)", "dim"))

    lines.append("")
    lines.append(_heading("Metrics:"))
    lines.append(_style(f"  {'metric':<16}{'A':>14}{'B':>14}{'Δ':>14}", "bold"))
    for md in d.metric_diffs:
        # Sign, not magnitude, is what's interesting here — whether A or B came
        # out ahead depends on the metric (loss down is good; throughput down
        # isn't), so we highlight the delta without implying good/bad.
        delta = "—" if md.delta is None else _style(f"{md.delta:+.4g}", "magenta")
        lines.append(f"  {md.key:<16}{_fmt(md.a):>14}{_fmt(md.b):>14}{delta:>14}")

    lines.append("")
    lines.append(_heading("Reproducibility:"))
    for note in d.notes:
        lines.append(f"  {_style('•', 'cyan')} {note}")
    return "\n".join(lines) + "\n"


def render_report(
    t: TimingSummary,
    findings: list[Finding],
    memory: MemorySummary | None = None,
    convergence: ConvergenceSummary | None = None,
    steps: list | None = None,
) -> str:
    sections = [
        render_timing(t, steps),
        render_memory(memory) if memory else "",
        render_convergence(convergence, steps) if convergence else "",
        render_findings(findings),
    ]
    return "\n".join(s for s in sections if s)
