"""Plain-text terminal report. No color deps; readable when piped to a file."""

from __future__ import annotations

from ..analyzers.convergence import ConvergenceSummary
from ..analyzers.memory import MemorySummary
from ..analyzers.timing import TimingSummary
from ..diagnosis.engine import Finding

_MB = 1024 * 1024

_SEV_LABEL = {"high": "HIGH", "med": "MED ", "low": "LOW "}
_BAR_WIDTH = 30


def _bar(frac: float) -> str:
    filled = int(round(frac * _BAR_WIDTH))
    return "#" * filled + "-" * (_BAR_WIDTH - filled)


def render_timing(t: TimingSummary) -> str:
    if t.n_steps == 0:
        return "No steps recorded.\n"

    lines = []
    lines.append(
        f"Run summary — {t.n_steps} steps, "
        f"{t.mean_step_time * 1e3:.1f} ms/step, "
        f"{t.steps_per_sec:.1f} steps/s"
    )
    lines.append(
        f"  step time: median {t.p50_step_time * 1e3:.1f} ms · "
        f"p95 {t.p95_step_time * 1e3:.1f} ms · "
        f"CV {t.cv:.2f}"
    )
    lines.append("")
    lines.append("Step time breakdown:")
    for phase in t.phase_order:
        frac = t.phase_fractions[phase]
        ms = t.phase_seconds[phase] * 1e3
        lines.append(f"  {phase:<10} {_bar(frac)} {frac * 100:5.1f}%  {ms:7.2f} ms")
    return "\n".join(lines) + "\n"


def render_findings(findings: list[Finding]) -> str:
    if not findings:
        return "\nNo issues found. Training looks balanced.\n"
    lines = ["", f"Findings ({len(findings)}):"]
    for f in findings:
        lines.append(f"  [{_SEV_LABEL.get(f.severity, '?')}] {f.title}  ({f.code})")
        lines.append(f"        {f.detail}")
        if f.suggestion:
            lines.append(f"        -> {f.suggestion}")
    return "\n".join(lines) + "\n"


def render_memory(m: MemorySummary) -> str:
    if not m or not m.has_memory:
        return ""
    lines = ["", "Memory:"]
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
        lines.append(f"  growth {m.growth_bytes_per_step / _MB:.2f} MB/step")
    return "\n".join(lines) + "\n"


def render_convergence(c: ConvergenceSummary) -> str:
    if not c or not c.has_loss:
        return ""
    lines = ["", "Convergence:"]
    best = f"{c.best_loss:.4g}" if c.best_loss is not None else "n/a"
    final = f"{c.final_loss:.4g}" if c.final_loss is not None else "n/a"
    lines.append(f"  loss trend {c.loss_trend} · best {best} · final {final}")
    if c.diverged_at is not None:
        lines.append(f"  DIVERGED at step {c.diverged_at}")
    if c.loss_spikes:
        lines.append(f"  loss spikes at steps {c.loss_spikes[:8]}")
    if c.grad_norm_spikes:
        lines.append(f"  grad-norm spikes at steps {c.grad_norm_spikes[:8]}")
    return "\n".join(lines) + "\n"


def render_distributed(d) -> str:
    """Render a DistributedSummary (multi-rank critical-path analysis)."""
    if not d:
        return ""
    lines = ["", f"Distributed — {d.world_size} ranks, {d.n_steps} aligned steps:"]
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
        flag = "  <- straggler" if (d.straggler and rs.rank == d.straggler.rank) else ""
        lines.append(
            f"    rank {rs.rank}: {rs.mean_compute * 1e3:7.2f} ms · "
            f"{rs.slowest_fraction * 100:4.0f}% (z={rs.straggler_z:+.1f}){flag}"
        )
    return "\n".join(lines) + "\n"


def render_pipeline(p) -> str:
    """Render a PipelineSummary (pipeline-bubble analysis)."""
    if not p:
        return ""
    lines = ["", f"Pipeline — {p.n_stages} stages:"]
    head = f"  bubble {p.bubble_fraction * 100:.0f}%"
    if p.ideal_bubble_fraction is not None:
        head += (
            f" · inherent min {p.ideal_bubble_fraction * 100:.0f}% "
            f"(m={p.n_microbatches}) · excess {p.excess_bubble * 100:.0f}%"
        )
    lines.append(head)
    return "\n".join(lines) + "\n"


def render_diff(d) -> str:
    """Render a RunDiff (reproducibility comparison)."""
    lines = [f"Comparing  A='{d.name_a}'  vs  B='{d.name_b}'", ""]

    def _fmt(v):
        return "—" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))

    lines.append("Environment diffs:")
    if d.env_diffs:
        for fd in d.env_diffs:
            lines.append(f"  {fd.key}: {_fmt(fd.a)}  ->  {_fmt(fd.b)}")
    else:
        lines.append("  (identical)")

    lines.append("")
    lines.append("Config diffs:")
    if d.config_diffs:
        for fd in d.config_diffs:
            lines.append(f"  {fd.key}: {_fmt(fd.a)}  ->  {_fmt(fd.b)}")
    else:
        lines.append("  (identical)")

    lines.append("")
    lines.append("Metrics:")
    lines.append(f"  {'metric':<16}{'A':>14}{'B':>14}{'Δ':>14}")
    for md in d.metric_diffs:
        delta = "—" if md.delta is None else f"{md.delta:+.4g}"
        lines.append(f"  {md.key:<16}{_fmt(md.a):>14}{_fmt(md.b):>14}{delta:>14}")

    lines.append("")
    lines.append("Reproducibility:")
    for note in d.notes:
        lines.append(f"  • {note}")
    return "\n".join(lines) + "\n"


def render_report(
    t: TimingSummary,
    findings: list[Finding],
    memory: MemorySummary | None = None,
    convergence: ConvergenceSummary | None = None,
) -> str:
    out = render_timing(t)
    out += render_memory(memory) if memory else ""
    out += render_convergence(convergence) if convergence else ""
    out += render_findings(findings)
    return out
