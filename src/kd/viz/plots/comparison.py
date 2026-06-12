
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from kd.search.result import DEFAULT_SCORE_KIND

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)


def _safe_label(
    results: list[ExperimentResult],
    labels: list[str] | None,
    index: int,
) -> str:
    if labels and index < len(labels):
        return labels[index]
    return results[index].algorithm_name


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

    for i, result in enumerate(results):
        label = _safe_label(results, labels, i)
        scores = result.recorder.get("_best_score")
        if not scores:
            continue
        any_data = True
        all_scores.append([float(s) for s in scores])



        score_identities.append(_pooling_identity(result))
        iterations = list(range(len(scores)))
        ax.plot(iterations, scores, label=label, marker=".", markersize=2, alpha=0.6)













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
    run_labels = [_safe_label(results, labels, i) for i in range(len(results))]
    r2_values = []
    for r in results:
        r2 = r.final_eval.r2
        if not np.isfinite(r2):
            warnings.append(f"Non-finite R2 ({r2}) for {r.algorithm_name}")
            r2 = 0.0
        r2_values.append(r2)

    ax.bar(run_labels, r2_values, color="steelblue", alpha=0.7)
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
    run_labels = [_safe_label(results, labels, i) for i in range(len(results))]

    col_labels = ["Run", "Expression", "NMSE", "R2", "Iterations"]
    table_data: list[list[Any]] = []

    for i, result in enumerate(results):
        table_data.append(
            [
                run_labels[i],
                _truncate(result.best_expression, max_len=30),
                f"{result.final_eval.nmse:.4g}",
                f"{result.final_eval.r2:.4f}",
                str(result.iterations),
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


_TRUNCATE_SUFFIX = "..."
_DEFAULT_MAX_LEN = 30


def _truncate(text: str, *, max_len: int = _DEFAULT_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - len(_TRUNCATE_SUFFIX)] + _TRUNCATE_SUFFIX
