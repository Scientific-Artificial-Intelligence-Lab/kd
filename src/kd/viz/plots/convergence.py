
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from kd.search.recorder import BEST_SCORE_KEY

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
    scores = result.recorder.get(BEST_SCORE_KEY)


    ylabel = f"Best {result.score_kind}"

    if not scores:
        warnings.append(
            f"No {BEST_SCORE_KEY} data in recorder; skipping convergence plot"
        )
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





    finite_scores = [
        float(v) if isinstance(v, (int, float)) and math.isfinite(v) else float("nan")
        for v in scores
    ]



    finite_indices = [i for i, v in enumerate(finite_scores) if not math.isnan(v)]
    if not finite_indices:
        warnings.append(
            f"No finite {BEST_SCORE_KEY} data in recorder; skipping convergence plot"
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title("Convergence")
        ax.text(
            0.5, 0.5, "No finite data", transform=ax.transAxes, ha="center", va="center"
        )
        return warnings

    ax.plot(iterations, finite_scores, marker=".", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)




    constant = flat_value(scores)
    if constant is None:
        ax.set_title("Convergence")
    else:


        first_finite = finite_indices[0]
        reached = (
            "the first iteration" if first_finite == 0 else f"iteration {first_finite}"
        )
        ax.set_title(
            f"Convergence\nbest {result.score_kind} constant at {constant:.4g} "
            f"(reached at {reached})",
            fontsize="medium",
        )

    return warnings
