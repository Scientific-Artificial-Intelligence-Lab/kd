
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np

from kd.core.equation import active_law, structure, term_diff
from kd.core.expr import format_pde
from kd.search.recorder import BEST_SCORE_KEY
from kd.search.result import DEFAULT_SCORE_KIND
from kd.viz._result_data import _sketch_fit_note
from kd.viz.axes import _get_axes, integer_ticks, value_bar_scale
from kd.viz.gap_notes import append_subtitle
from kd.viz.plots._comparison_cells import _expression_cell, _render_term_diff_cell
from kd.viz.style import (
    BAND,
    BAND_ALPHA,
    BAR_FILL,
    CONTEXT,
    CONTEXT_WIDTH,
    FONTSIZE,
    INK_2,
    INK_MUTED,
    PANEL_H,
    PANEL_W,
    REF_LINE_WIDTH,
    SERIES,
    SERIES_DASHES,
    SMALL_FONTSIZE,
    draw_table,
    isolated_point_marks,
    style_context,
    table_height,
    wrap_to_width,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from torch import Tensor

    from kd.core.equation import Equation
    from kd.evaluate import ComparisonResult
    from kd.search.result import ExperimentResult


def plot_structure_comparison(
    comparison: ComparisonResult,
    *,
    note: str = "",
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    warnings = [f"{name}: {why}" for name, why in comparison.exclusions.items()]
    table_h = table_height(len(comparison.results))
    plot_h = 1.3 * PANEL_H
    with style_context(style):
        fig = plt.figure(
            figsize=(3 * PANEL_W, table_h + plot_h + 1.0), layout="constrained"
        )
        grid = fig.add_gridspec(2, 2, height_ratios=[table_h, plot_h], hspace=0.08)
        table_ax = fig.add_subplot(grid[0,:])
        progress_ax = fig.add_subplot(grid[1, 0])
        score_ax = fig.add_subplot(grid[1, 1])
        warnings.extend(_normalized_progress(comparison.results, progress_ax))

        order = list(comparison.results)
        names = list(comparison.evaluations)
        values = [comparison.evaluations[name].nmse for name in names]
        positions = np.arange(len(names), dtype=float)
        bars = score_ax.bar(
            positions,
            values,
            color=[SERIES[order.index(name) % len(SERIES)] for name in names],
            width=BAR_FILL,
        )
        score_ax.bar_label(
            bars,
            labels=[f"{value:.2e}" for value in values],
            padding=3,
            fontsize=SMALL_FONTSIZE,
            color=INK_2,
        )
        value_bar_scale(score_ax, values, headroom=0.2)
        score_ax.set(
            ylabel="NMSE (shared structure refit)", title="One ruler: lower is better"
        )
        title = "One dataset, one ruler: structure refits"
        if comparison.reference is not None:
            title += f"\nReference: {comparison.reference}"
        fig.suptitle(title)
        if comparison.fixed:
            pins = ", ".join(f"{t} = {v}" for t, v in comparison.fixed.items())
            note = "\n".join(filter(None, [note, f"Fixed coefficients: {pins}"]))
        if note:
            fig.supxlabel(note, fontsize=SMALL_FONTSIZE, color=INK_2)

        _label_bars(score_ax, positions, names)
        _structure_comparison_table(comparison, table_ax)
    return fig, warnings



_LABEL_SLOT = 0.9


def _label_bars(ax: Axes, positions: np.ndarray, labels: list[str]) -> None:
    ax.set_xticks(positions, labels)
    if not labels:
        return
    cast("Figure", ax.get_figure(root=True)).draw_without_rendering()
    band = _LABEL_SLOT * ax.get_window_extent().width / len(labels)
    font = ax.get_xticklabels()[0]
    ax.set_xticks(positions, [wrap_to_width(label, font, band) for label in labels])


def _structure_comparison_table(comparison: ComparisonResult, ax: Axes) -> None:
    rows = []
    for name, result in comparison.results.items():
        if name in comparison.exclusions:
            cells = [comparison.exclusions[name], "n/a", "n/a"]
        else:
            evaluation = comparison.evaluations[name]
            equation = format_pde(
                cast(list[str], evaluation.terms),
                cast("Tensor", evaluation.coefficients),
                lhs=comparison.lhs_label,
                sig_figs=3,
            )
            cells = [
                f"${equation.latex}$",
                f"{evaluation.nmse:.2e}",
                f"{evaluation.r2:.4f}",
            ]
        rows.append([name, *cells, str(result.iterations)])
    draw_table(
        ax,
        rows,
        ["Run", "Equation (shared refit)", "NMSE", "$R^2$", "Iterations"],
        text_columns=(0, 1),
    )


def _normalized_progress(results: dict[str, ExperimentResult], ax: Axes) -> list[str]:
    notes = []
    for index, (name, result) in enumerate(results.items()):
        scores = np.asarray(result.recorder.get(BEST_SCORE_KEY), dtype=float)
        finite = np.isfinite(scores)
        if not finite.any():
            notes.append(f"{name}: no finite score history; progress omitted")
            continue
        start, end = scores[finite][[0, -1]]
        progress = np.full(scores.shape, np.nan)
        progress[finite] = (
            1.0 if start == end else (scores[finite] - start) / (end - start)
        )
        ax.plot(
            np.arange(len(scores)),
            progress,
            color=SERIES[index % len(SERIES)],
            linestyle=SERIES_DASHES[index % len(SERIES_DASHES)],
            label=name,
            **isolated_point_marks(progress),
        )
    ax.set(
        xlabel="Iteration",
        ylabel="Normalized best-score progress",
        title="Per-run progress from first to last finite score",
    )
    integer_ticks(ax)
    if ax.lines:
        ax.legend(loc="lower right")
    if notes:
        append_subtitle(ax, "\n".join(notes))
    return notes


def _disambiguate_labels(raw_labels: list[str]) -> list[str]:
    counts = Counter(raw_labels)


    used = {label for label, count in counts.items() if count == 1}
    seen: Counter[str] = Counter()
    disambiguated: list[str] = []
    for label in raw_labels:
        if counts[label] == 1:
            disambiguated.append(label)
            continue
        seen[label] += 1
        candidate = f"{label} #{seen[label]}"
        while candidate in used:
            seen[label] += 1
            candidate = f"{label} #{seen[label]}"
        used.add(candidate)
        disambiguated.append(candidate)
    return disambiguated


def _run_labels(results: list[ExperimentResult], labels: list[str] | None) -> list[str]:
    return _disambiguate_labels(
        [
            labels[i] if labels and i < len(labels) else results[i].algorithm_name
            for i in range(len(results))
        ]
    )


def _shared_score_ylabel(results: list[ExperimentResult]) -> str:
    if not results:
        return "Best Score"
    kinds = {r.score_kind for r in results}
    if len(kinds) == 1:
        return f"Best {kinds.pop()}"
    return "Best Score"


def _pooling_identity(result: ExperimentResult) -> tuple[str, str]:
    if result.score_kind == DEFAULT_SCORE_KIND:
        return ("__undeclared__", str(result.config.get("algorithm", "")))
    return (result.score_kind, result.score_direction)


def _band_identity(result: ExperimentResult) -> tuple[str, str, str]:
    return (str(result.config.get("algorithm", "")), *_pooling_identity(result))


def render_overlaid_convergence(
    results: list[ExperimentResult],
    ax: Axes | None = None,
    *,
    labels: list[str] | None = None,
    style: dict[str, Any] | None = None,
) -> list[str]:
    ax = _get_axes(ax, style)
    warnings: list[str] = []
    any_data = False
    drawn_mixed = False
    drawn_cross_algo = False
    all_scores: list[list[float]] = []
    score_identities: list[tuple[str, str]] = []
    band_identities: list[tuple[str, str, str]] = []
    run_labels = _run_labels(results, labels)




    mixed_metrics = len({_pooling_identity(r) for r in results}) > 1

    with style_context(style):
        for i, result in enumerate(results):
            label = run_labels[i]
            if mixed_metrics:
                if result.score_kind == DEFAULT_SCORE_KIND:




                    label = f"{label} (undeclared metric)"
                else:
                    label = f"{label} ({result.score_kind}, {result.score_direction})"
            scores = result.recorder.get(BEST_SCORE_KEY)
            if not scores:
                continue





            finite_scores = [
                float(s)
                if isinstance(s, (int, float)) and np.isfinite(s)
                else float("nan")
                for s in scores
            ]




            if not any(np.isfinite(v) for v in finite_scores):
                warnings.append(
                    f"Run {label!r} has no finite {BEST_SCORE_KEY} data; skipped"
                )
                continue
            any_data = True
            all_scores.append(finite_scores)



            score_identities.append(_pooling_identity(result))
            band_identities.append(_band_identity(result))
            iterations = list(range(len(scores)))
            ax.plot(
                iterations,
                finite_scores,
                label=label,
                color=SERIES[i % len(SERIES)],
                linestyle=SERIES_DASHES[i % len(SERIES_DASHES)],
                **isolated_point_marks(finite_scores),
            )















        if len(all_scores) >= 2:
            if len(set(score_identities)) > 1:
                warnings.append(
                    "Overlaid runs carry incommensurable (or undeclared) score "
                    "metrics (mixed score kind/direction, e.g. SGA AIC vs DLGA "
                    "fitness vs DISCOVER reward); mean/std band suppressed. "
                    "Compare final nmse/r2 across runs instead of the internal "
                    "search score."
                )
                drawn_mixed = True
            elif len(set(band_identities)) > 1:
                warnings.append(
                    "Overlaid runs share one score metric but come from "
                    "different algorithms; cross-algorithm mean/std band "
                    "suppressed (one band would average distinct engines, and "
                    "'iteration' may not mean the same thing across them)."
                )
                drawn_cross_algo = True
            else:
                max_len = max(len(s) for s in all_scores)

                padded = np.full((len(all_scores), max_len), np.nan)
                for j, s in enumerate(all_scores):
                    padded[j,: len(s)] = s
                    if len(s) < max_len:
                        padded[j, len(s):] = s[-1]
                mean = np.nanmean(padded, axis=0)
                std = np.nanstd(padded, axis=0)
                x = np.arange(max_len)
                ax.fill_between(
                    x, mean - std, mean + std, color=BAND, alpha=BAND_ALPHA, linewidth=0
                )
                ax.plot(
                    x,
                    mean,
                    linestyle="--",
                    color=CONTEXT,
                    linewidth=CONTEXT_WIDTH,
                    label="mean",
                )

        if not any_data:
            warnings.append("No convergence data in any result; skipped overlaid plot")
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel(_shared_score_ylabel(results))
        integer_ticks(ax)
        ax.set_title("Convergence Comparison")
        if drawn_mixed:



            append_subtitle(
                ax,
                "Mixed score metrics: (kind, direction) in legend; "
                "curve trends not comparable across runs",
            )
        elif drawn_cross_algo:



            append_subtitle(
                ax,
                "One score metric, multiple algorithms: "
                "cross-algorithm mean/std band suppressed",
            )
        if any_data:
            ax.legend(fontsize=SMALL_FONTSIZE)

    return warnings


def plot_score_bar(
    results: list[ExperimentResult],
    ax: Axes | None = None,
    *,
    labels: list[str] | None = None,
    style: dict[str, Any] | None = None,
) -> list[str]:
    ax = _get_axes(ax, style)
    warnings: list[str] = []
    run_labels = _run_labels(results, labels)
    r2_values: list[float] = []
    invalid_positions: list[int] = []
    for i, r in enumerate(results):
        r2 = r.final_eval.r2
        if not np.isfinite(r2):


            warnings.append(f"Non-finite R2 ({r2}) for {run_labels[i]}; shown as N/A")




            r2_values.append(float("nan"))
            invalid_positions.append(i)
        else:
            r2_values.append(float(r2))





    positions = np.arange(len(results), dtype=float)
    with style_context(style):

        ax.bar(
            positions,
            r2_values,
            color=[SERIES[i % len(SERIES)] for i in range(len(results))],
            width=BAR_FILL,
        )
        for i in invalid_positions:








            ax.text(
                positions[i],
                0.02,
                "N/A",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=SMALL_FONTSIZE,
                color=INK_MUTED,
            )
        ax.set_ylabel("$R^2$")
        ax.set_title("Score Comparison")
        ax.axhline(y=1.0, color=CONTEXT, linewidth=REF_LINE_WIDTH)
        _annotate_sketch_metrics(ax, results, run_labels, warnings)
        _label_bars(ax, positions, run_labels)

    return warnings


def plot_summary_table(
    results: list[ExperimentResult],
    ax: Axes | None = None,
    *,
    labels: list[str] | None = None,
    style: dict[str, Any] | None = None,
) -> list[str]:
    ax = _get_axes(ax, style, figsize=summary_table_figsize(len(results)), fill=True)
    warnings: list[str] = []
    run_labels = _run_labels(results, labels)

    col_labels = ["Run", "Expression", "NMSE", "R2", "Iterations", "Δ Terms vs Run 0"]
    table_data: list[list[Any]] = []

    baseline_law = _validate_baseline_for_term_diff(results, run_labels, warnings)
    _warn_on_mixed_algorithm_vocabularies(results, warnings)

    for i, result in enumerate(results):
        term_diff_text = _summary_term_diff(
            results, i, run_labels, warnings, baseline_law
        )
        table_data.append(
            [
                run_labels[i],
                _expression_cell(result, run_labels[i], warnings),
                _metric_cell(
                    result.final_eval.nmse, ".4g", "NMSE", run_labels[i], warnings
                ),
                _metric_cell(
                    result.final_eval.r2, ".4f", "R2", run_labels[i], warnings
                ),
                str(result.iterations),
                term_diff_text,
            ]
        )

    with style_context(style):
        draw_table(ax, table_data, col_labels, text_columns=(0, 1, 5))
        _annotate_sketch_metrics(ax, results, run_labels, warnings)

    return warnings


def summary_table_figsize(n_runs: int) -> tuple[float, float]:
    return 3 * PANEL_W, table_height(n_runs)


def _annotate_sketch_metrics(
    ax: Axes,
    results: list[ExperimentResult],
    labels: list[str],
    warnings: list[str],
) -> None:
    notes = [
        f"{label}: {note}"
        for result, label in zip(results, labels, strict=True)
        if (note := _sketch_fit_note(result)) is not None
    ]
    if notes:
        ax.set_title(
            "Search-fit metrics (targets may differ)\n"
            "Not a comparison of full-equation quality",
            fontsize=FONTSIZE,
        )
        warnings.extend(notes)


def _metric_cell(
    value: float, fmt: str, metric: str, run_label: str, warnings: list[str]
) -> str:
    if not np.isfinite(value):
        warnings.append(f"Non-finite {metric} ({value}) for {run_label}; shown as N/A")
        return "N/A"
    return format(value, fmt)


def _validate_baseline_for_term_diff(
    results: list[ExperimentResult], run_labels: list[str], warnings: list[str]
) -> Equation | None:
    if len(results) < 2:
        return None
    baseline_equation = results[0].equation
    if baseline_equation is None:
        warnings.append(
            f"Baseline {run_labels[0]} has no equation; term diffs unavailable"
        )
        return None
    try:
        baseline_law = active_law(baseline_equation)
        structure(baseline_law)
    except (ValueError, NotImplementedError) as err:
        warnings.append(
            f"Baseline {run_labels[0]} equation is not canonicalizable; "
            f"term diffs unavailable ({err})"
        )
        return None
    return baseline_law


def _warn_on_mixed_algorithm_vocabularies(
    results: list[ExperimentResult], warnings: list[str]
) -> None:
    algorithms = {
        algorithm
        for result in results
        if (algorithm := result.config.get("algorithm")) is not None
    }
    if len(algorithms) > 1:
        warnings.append(
            "Δ Terms compares canonical IR spelling; term vocabularies may "
            f"differ across algorithms ({', '.join(sorted(map(str, algorithms)))})"
        )


def _summary_term_diff(
    results: list[ExperimentResult],
    index: int,
    run_labels: list[str],
    warnings: list[str],
    baseline_law: Equation | None,
) -> str:
    if index == 0:
        return "baseline"
    if baseline_law is None:
        return "n/a"

    result = results[index]
    if result.equation is None:
        warnings.append(f"{run_labels[index]} has no equation; term diff unavailable")
        return "n/a"

    try:
        delta = term_diff(baseline_law, active_law(result.equation))
    except (ValueError, NotImplementedError) as err:
        warnings.append(
            f"{run_labels[index]} equation is not canonicalizable; "
            f"term diff unavailable ({err})"
        )
        return "n/a"

    return _render_term_diff_cell(delta, run_labels[index], warnings)
