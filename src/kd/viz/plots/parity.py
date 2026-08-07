
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)


def plot_parity(
    result: ExperimentResult,
    ax: Axes,
    *,
    style: dict[str, Any] | None = None,
) -> list[str]:
    warnings: list[str] = []

    actual = result.actual.detach().cpu().numpy()
    predicted = result.predicted.detach().cpu().numpy()

    with style_context(style):

        ax.scatter(actual, predicted, s=8, alpha=0.6, label="Samples", rasterized=True)


        finite_vals = np.concatenate(
            [
                actual[np.isfinite(actual)],
                predicted[np.isfinite(predicted)],
            ]
        )
        if len(finite_vals) > 0:
            lo = float(finite_vals.min())
            hi = float(finite_vals.max())
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
        else:
            warnings.append("All values are non-finite; 45-degree line omitted")


        r2 = result.final_eval.r2
        if math.isfinite(r2):
            r2_text = f"$R^2 = {r2:.4f}$"
        else:
            r2_text = f"$R^2 = $ {r2}"
            warnings.append(f"R2 is non-finite ({r2})")
        ax.text(
            0.05,
            0.95,
            r2_text,
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Parity Plot")

    return warnings
