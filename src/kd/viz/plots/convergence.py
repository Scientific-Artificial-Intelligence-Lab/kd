
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)






_FLAT_REL_TOL = 1e-9


def flat_value(values: Sequence[float]) -> float | None:
    finite = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and math.isfinite(v)
    ]
    if len(finite) < 2:
        return None
    lo, hi = min(finite), max(finite)
    if (hi - lo) <= _FLAT_REL_TOL * max(1.0, abs(hi)):
        return hi
    return None


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




    constant = flat_value(scores)
    if constant is None:
        ax.set_title("Convergence")
    else:
        ax.set_title(
            f"Convergence\nbest {result.score_kind} constant at {constant:.4g} "
            "(reached at the first iteration)",
            fontsize="medium",
        )

    return warnings
