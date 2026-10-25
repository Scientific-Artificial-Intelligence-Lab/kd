
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import sympy
from kd import LhsSpec, format_pde, render_lhs_label
from kd.viz import style_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kdagent.analysis.sources import load_trace, tool_calls, tree_events

_FRAMES = (("platform_nmse", "platform NMSE", True), ("nmse", "own NMSE", False))


def trajectory_figure(workspace: Path) -> Figure:
    events = tree_events(workspace)
    segments = [event for event in events if event["event"] == "segment"]
    trace = load_trace(workspace) if (workspace / "trace.jsonl").is_file() else []
    origin = _origin(trace, segments)
    points = [_point(row, origin) for row in segments]
    pruned = _pruned(events)
    chosen = _chosen(trace, events, pruned)
    with style_context():
        figure = Figure(figsize=(8.0, 4.5), layout="constrained")
        axes = figure.add_subplot()
        _plot_series(axes, points)
        _plot_marks(axes, points, pruned, chosen)
        axes.set_xlabel("minutes since the run started")
        axes.set_ylabel("segment NMSE")
        axes.set_title(workspace.name)
        if any(point["y"] is not None for point in points):
            axes.set_yscale("log")
        axes.legend(loc="best", fontsize="small")
    return figure


def _origin(trace: list[dict[str, Any]], segments: list[dict[str, Any]]) -> datetime:
    stamps = [row["start_time"] for row in trace] or [
        row["created_at"] for row in segments
    ]
    return min(datetime.fromisoformat(stamp) for stamp in stamps)


def _point(row: dict[str, Any], origin: datetime) -> dict[str, Any]:
    minutes = (datetime.fromisoformat(row["created_at"]) - origin).total_seconds() / 60
    point: dict[str, Any] = {
        "run_id": row["run_id"],
        "instrument": row["instrument"],
        "x": minutes,
        "y": None,
        "frame": None,
    }
    for key, frame, _filled in _FRAMES:
        value = row.get(key)
        if value is not None and math.isfinite(value):
            point.update(y=value, frame=frame)
            break
    return point


def _pruned(events: list[dict[str, Any]]) -> set[str]:
    parent = {
        event["run_id"]: event["parent_run_id"]
        for event in events
        if event["event"] == "segment"
    }
    named = {event["run_id"] for event in events if event["event"] == "prune"}
    pruned = set()
    for run_id in parent:
        current: str | None = run_id
        while current is not None and current not in named:
            current = parent[current]
        if current is not None:
            pruned.add(run_id)
    return pruned


def _chosen(
    trace: list[dict[str, Any]], events: list[dict[str, Any]], pruned: set[str]
) -> tuple[str, dict[str, Any] | None] | None:
    submitted = [
        call
        for call in tool_calls(trace)
        if call["name"] == "submit_answer"
        and call["envelope"] is not None
        and call["envelope"]["status"] == "success"
    ]
    if submitted:
        call = submitted[-1]
        return str(call["inputs"]["run_id"]), call["envelope"]["primary"]["law"]
    selected = [event["run_id"] for event in events if event["event"] == "select"]
    if not selected or selected[-1] in pruned:
        return None
    row = next(
        event
        for event in events
        if event["event"] == "segment" and event["run_id"] == selected[-1]
    )
    return str(row["run_id"]), row["law"]


def _plot_series(axes: Axes, points: list[dict[str, Any]]) -> None:
    for instrument in sorted({point["instrument"] for point in points}):
        color = None
        for _key, frame, filled in _FRAMES:
            mine = [
                p
                for p in points
                if p["instrument"] == instrument and p["frame"] == frame
            ]
            if not mine:
                continue
            (line,) = axes.plot(
                [p["x"] for p in mine],
                [p["y"] for p in mine],
                marker="o",
                linestyle="none",
                color=color,
                markerfacecolor=None if filled else "none",
                label=f"{instrument} ({frame})",
            )
            color = line.get_color()
    numberless = [p["x"] for p in points if p["y"] is None]
    if numberless:
        axes.plot(
            numberless,
            [0.02] * len(numberless),
            marker="|",
            markersize=12,
            linestyle="none",
            color="gray",
            transform=axes.get_xaxis_transform(),
            label="no NMSE",
        )


def _plot_marks(
    axes: Axes,
    points: list[dict[str, Any]],
    pruned: set[str],
    chosen: tuple[str, dict[str, Any] | None] | None,
) -> None:
    crossed = [p for p in points if p["run_id"] in pruned and p["y"] is not None]
    if crossed:
        axes.plot(
            [p["x"] for p in crossed],
            [p["y"] for p in crossed],
            marker="x",
            markersize=10,
            linestyle="none",
            color="black",
            label="pruned",
        )
    if chosen is None:
        return
    run_id, law = chosen
    star = next(
        (p for p in points if p["run_id"] == run_id and p["y"] is not None), None
    )
    if star is not None:
        axes.plot(
            [star["x"]],
            [star["y"]],
            marker="*",
            markersize=16,
            linestyle="none",
            color="gold",
            markeredgecolor="black",
            label=f"chosen: {_law_text(law)}",
        )


def _law_text(law: dict[str, Any] | None) -> str:
    if law is None:
        return "no law"
    label = _lhs_label(law["lhs"])


    coefficients = law.get("coefficients")
    if coefficients is None or None in coefficients:
        terms = ",\\ ".join(
            sympy.latex(format_pde([term], [1.0]).rhs) for term in law["support"]
        )
        lhs = sympy.latex(sympy.Symbol(label))
        return f"${lhs}$ on ${terms}$ (coefficients not all known)"
    equation = format_pde(law["support"], coefficients, lhs=label, sig_figs=3)
    return f"${equation.latex}$"


def _lhs_label(lhs: dict[str, Any] | None) -> str:
    if lhs is None:
        return "0"
    if lhs["order"] == 0:
        return str(lhs["field"])
    return render_lhs_label(LhsSpec(**lhs))


__all__ = ["trajectory_figure"]
