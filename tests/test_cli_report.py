"""Color/visualization helpers in the CLI report — must degrade to identical
plain text whenever color is off (piped, NO_COLOR, --color=never), and never
break the plain-text contract other renderers rely on."""

from __future__ import annotations

import re

from trainscope.report import cli_report as r

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _reset():
    r.set_color_mode("auto")


def test_style_noop_when_color_disabled():
    _reset()
    r.set_color_mode("never")
    try:
        assert r._style("hello", "red") == "hello"
        assert r._style("", "red") == ""
    finally:
        _reset()


def test_style_wraps_in_ansi_when_forced():
    _reset()
    r.set_color_mode("always")
    try:
        styled = r._style("hello", "bold", "red")
        assert styled != "hello"
        assert _ANSI.sub("", styled) == "hello"
        assert styled.startswith("\x1b[") and styled.endswith("\x1b[0m")
    finally:
        _reset()


def test_bar_strips_to_identical_plain_text():
    _reset()
    plain = r._bar(0.4)
    r.set_color_mode("always")
    try:
        colored = r._bar(0.4, kind="bad")
        assert _ANSI.sub("", colored) == _ANSI.sub("", plain) == plain
        assert len(_ANSI.sub("", colored)) == r._BAR_WIDTH
    finally:
        _reset()


def test_sparkline_basic_properties():
    ticks = r._sparkline([1.0, 2.0, 3.0, 2.0, 1.0])
    assert len(ticks) == 5
    assert all(c in r._SPARK_TICKS for c in ticks)
    # Flat / too-short series produce nothing rather than a misleading flat line.
    assert r._sparkline([1.0]) == ""
    assert r._sparkline([]) == ""


def test_render_findings_plain_when_color_off():
    _reset()
    r.set_color_mode("never")
    try:
        out = r.render_findings([])
        assert "\x1b[" not in out
        assert "No issues found" in out
    finally:
        _reset()


def test_color_mode_always_produces_ansi_in_report_sections():
    _reset()
    r.set_color_mode("always")
    try:
        out = r.render_findings([])
        assert "\x1b[" in out
    finally:
        _reset()
