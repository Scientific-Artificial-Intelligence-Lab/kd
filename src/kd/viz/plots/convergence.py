
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)


def plot_convergence(
    result: ExperimentResult,
    ax: Axes,
) -> list[str]:
    warnings: list[str] = []
    scores = result.recorder.get("_best_score")


    ylabel = f"Best {result.score_kind}"

    if not scores:
        warnings.append("No _best_score data in recorder; skipping convergence plot")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title("Convergence")
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return warnings

    iterations = list(range(len(scores)))
    ax.plot(iterations, scores, marker=".", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title("Convergence")

    return warnings
