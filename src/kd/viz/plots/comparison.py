
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

from kd.core.equation import structure, term_diff
from kd.search.recorder import BEST_SCORE_KEY
from kd.search.result import DEFAULT_SCORE_KIND
from kd.viz.plots._comparison_cells import _expression_cell, _render_term_diff_cell

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult


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


def render_overlaid_convergence(
    results: list[ExperimentResult],
    ax: Axes,
    *,
    labels: list[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    any_data = False
    all_scores: list[list[float]] = []
    score_identities: list[tuple[str, str]] = []
    run_labels = _run_labels(results, labels)

    for i, result in enumerate(results):
        label = run_labels[i]
        scores = result.recorder.get(BEST_SCORE_KEY)
        if not scores:
            continue





        finite_scores = [
            float(s) if isinstance(s, (int, float)) and np.isfinite(s) else float("nan")
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
        iterations = list(range(len(scores)))
        ax.plot(
            iterations, finite_scores, label=label, marker=".", markersize=2, alpha=0.6
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
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="gray")
            ax.plot(x, mean, "--", color="gray", linewidth=1.5, label="mean")

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
    ax.set_title("Convergence Comparison")
    if any_data:
        ax.legend(fontsize=8)

    return warnings


def plot_score_bar(
    results: list[ExperimentResult],
    ax: Axes,
    *,
    labels: list[str] | None = None,
) -> list[str]:
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
    ax.bar(positions, r2_values, color="steelblue", alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(run_labels)
    for i in invalid_positions:








        ax.text(
            positions[i],
            0.02,
            "N/A",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )
    ax.set_ylabel("$R^2$")
    ax.set_title("Score Comparison")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)

    return warnings


def plot_summary_table(
    results: list[ExperimentResult],
    ax: Axes,
    *,
    labels: list[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    run_labels = _run_labels(results, labels)

    col_labels = ["Run", "Expression", "NMSE", "R2", "Iterations", "Δ Terms vs Run 0"]
    table_data: list[list[Any]] = []

    baseline_ready = _validate_baseline_for_term_diff(results, run_labels, warnings)
    _warn_on_mixed_algorithm_vocabularies(results, warnings)

    for i, result in enumerate(results):
        term_diff_text = _summary_term_diff(
            results, i, run_labels, warnings, baseline_ready
        )
        table_data.append(
            [
                run_labels[i],
                _expression_cell(result, run_labels[i], warnings),
                f"{result.final_eval.nmse:.4g}",
                f"{result.final_eval.r2:.4f}",
                str(result.iterations),
                term_diff_text,
            ]
        )

    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    return warnings


def _validate_baseline_for_term_diff(
    results: list[ExperimentResult], run_labels: list[str], warnings: list[str]
) -> bool:
    if len(results) < 2:
        return False
    baseline = results[0]
    if baseline.equation is None:
        warnings.append(
            f"Baseline {run_labels[0]} has no equation; term diffs unavailable"
        )
        return False
    try:
        structure(baseline.equation)
    except (ValueError, NotImplementedError) as err:
        warnings.append(
            f"Baseline {run_labels[0]} equation is not canonicalizable; "
            f"term diffs unavailable ({err})"
        )
        return False
    return True


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
    baseline_ready: bool,
) -> str:
    if index == 0:
        return "baseline"
    if not baseline_ready:
        return "n/a"

    baseline_equation = results[0].equation
    if baseline_equation is None:

        return "n/a"

    result = results[index]
    if result.equation is None:
        warnings.append(f"{run_labels[index]} has no equation; term diff unavailable")
        return "n/a"

    try:
        delta = term_diff(baseline_equation, result.equation)
    except (ValueError, NotImplementedError) as err:
        warnings.append(
            f"{run_labels[index]} equation is not canonicalizable; "
            f"term diff unavailable ({err})"
        )
        return "n/a"

    return _render_term_diff_cell(delta, run_labels[index], warnings)
