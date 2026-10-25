
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from kd.viz._result_data import _fit_target_label, _sketch_fit_note
from kd.viz.axes import _get_axes
from kd.viz.style import INK_2, REF_LINE_WIDTH, SERIES, style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)


def plot_parity(
    result: ExperimentResult,
    ax: Axes | None = None,
    *,
    style: dict[str, Any] | None = None,
) -> list[str]:
    ax = _get_axes(ax, style)
    warnings: list[str] = []

    actual = result.actual.detach().cpu().numpy()
    predicted = result.predicted.detach().cpu().numpy()

    with style_context(style):

        ax.scatter(
            actual,
            predicted,
            s=6,
            alpha=0.35,
            linewidths=0,
            color=SERIES[0],
            label="Samples",
            rasterized=True,
        )


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
            ax.plot(
                [lo, hi], [lo, hi], color=INK_2, linewidth=REF_LINE_WIDTH, label="y = x"
            )
            ax.set_aspect("equal", adjustable="box")
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
            color=INK_2,
        )
        ax.grid(True, axis="both")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Parity")
        note = _sketch_fit_note(result)
        if note is not None:
            ax.set_xlabel(f"Actual ({_fit_target_label(result)})")
            ax.set_ylabel(f"Predicted ({_fit_target_label(result)})")
            ax.set_title("Parity (search fit)")
            warnings.append(note)

    return warnings
