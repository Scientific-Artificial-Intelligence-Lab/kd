"""PySR visualisation adapter: equation / parity / residual."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .._contracts import ParityPlotData, ResidualPlotData
from ..core import VizResult
from ..equation_renderer import render_latex_to_image


class PySRVizAdapter:
    """Minimal visualisation adapter for PySR.

    Supported intents:

    * ``'equation'``  – render a LaTeX equation figure;
    * ``'parity'``    – scatter plot of ``y_true`` vs ``y_pred``;
    * ``'residual'``  – residual histogram.
    """

    capabilities: Iterable[str] = {
        "equation",
        "parity",
        "residual",
    }

    def __init__(self, *, subdir: str = "pysr") -> None:
        self._subdir = subdir

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def render(self, request, ctx):  # type: ignore[override]
        model = request.target
        kind = request.kind
        handler = {
            "equation": self._equation,
            "parity": self._parity,
            "residual": self._residual,
        }.get(kind)
        if handler is None:
            return VizResult(
                intent=kind,
                warnings=[f"PySR adapter does not support intent '{kind}'."],
                metadata={"capabilities": sorted(self.capabilities)},
            )
        return handler(model, ctx)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _resolve_output(self, ctx, filename: str) -> Tuple[Path, Path]:
        """Resolve a save directory consistent with other adapters.

        It prefers ``ctx.options['output_dir']`` when provided, otherwise
        falls back to ``ctx.save_path`` and appends the adapter-specific
        subdirectory (e.g. ``'pysr/'``). When no explicit save directory is
        configured, this ultimately falls back to ``CWD/artifacts``.
        """
        base = ctx.options.get("output_dir")
        if base is not None:
            base_path = Path(base)
            if self._subdir:
                base_path = base_path / self._subdir
            base_path.mkdir(parents=True, exist_ok=True)
            return base_path, base_path / filename

        relative = f"{self._subdir}/{filename}" if self._subdir else filename
        path = ctx.save_path(relative)
        return path.parent, path

    def _get_training_data(self, model, ctx):
        """Retrieve training ``(X, y)`` either from the model or the request."""
        X = ctx.options.get("X", None)
        y = ctx.options.get("y", None)
        if X is None or y is None:
            X = getattr(model, "X_train_", None)
            y = getattr(model, "y_train_", None)
        if X is None or y is None:
            return None, None
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).reshape(-1)
        if X_arr.shape[0] != y_arr.shape[0]:
            return None, None
        return X_arr, y_arr

    # ------------------------------------------------------------------
    # intent handlers
    # ------------------------------------------------------------------
    def _equation(self, model, ctx) -> VizResult:
        try:
            latex = model.equation_latex()
        except Exception as exc:
            return VizResult(
                intent="equation",
                warnings=[f"Failed to obtain equation LaTeX from KD_PySR: {exc}"],
            )

        _, path = self._resolve_output(ctx, "equation.png")
        try:
            render_latex_to_image(
                latex,
                output_path=str(path),
                font_size=ctx.options.get("font_size", 18),
                show=False,
            )
        except Exception as exc:
            return VizResult(
                intent="equation",
                warnings=[f"Failed to render PySR LaTeX equation: {exc}"],
            )

        return VizResult(intent="equation", paths=[path], metadata={"latex": latex})

    def _parity(self, model, ctx) -> VizResult:
        X, y_true = self._get_training_data(model, ctx)
        if X is None or y_true is None:
            return VizResult(
                intent="parity",
                warnings=["PySR parity plot requires access to (X, y)."],
            )

        try:
            y_pred = np.asarray(model.predict(X)).reshape(-1)
        except Exception as exc:
            return VizResult(
                intent="parity",
                warnings=[f"KD_PySR.predict failed during parity plot: {exc}"],
            )

        try:
            data = ParityPlotData.from_actual_predicted(
                actual_values=y_true,
                predicted_values=y_pred,
            )
        except Exception as exc:
            return VizResult(
                intent="parity",
                warnings=[f"Failed to build parity plot data: {exc}"],
            )

        fig, ax = plt.subplots(figsize=ctx.options.get("figsize", (5, 5)))
        ax.scatter(data.actual_values, data.predicted_values, alpha=0.6, s=10)
        lims = [
            min(data.actual_values.min(), data.predicted_values.min()),
            max(data.actual_values.max(), data.predicted_values.max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(ctx.options.get("title", "PySR Parity Plot"))
        ax.grid(True, linestyle="--", alpha=0.3)

        _, path = self._resolve_output(ctx, "parity.png")
        fig.savefig(str(path), dpi=ctx.options.get("dpi", 300), bbox_inches="tight")
        plt.close(fig)

        return VizResult(intent="parity", paths=[path])

    def _residual(self, model, ctx) -> VizResult:
        X, y_true = self._get_training_data(model, ctx)
        if X is None or y_true is None:
            return VizResult(
                intent="residual",
                warnings=["PySR residual plot requires access to (X, y)."],
            )

        try:
            y_pred = np.asarray(model.predict(X)).reshape(-1)
        except Exception as exc:
            return VizResult(
                intent="residual",
                warnings=[f"KD_PySR.predict failed during residual plot: {exc}"],
            )

        try:
            data = ResidualPlotData.from_actual_predicted(
                actual=y_true,
                predicted=y_pred,
            )
        except Exception as exc:
            return VizResult(
                intent="residual",
                warnings=[f"Failed to build residual data: {exc}"],
            )

        fig, ax = plt.subplots(figsize=ctx.options.get("figsize", (6, 4)))
        ax.hist(
            data.residuals,
            bins=ctx.options.get("bins", 30),
            alpha=0.8,
            edgecolor="black",
        )
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.set_title(ctx.options.get("title", "PySR Residual Histogram"))
        ax.grid(True, linestyle="--", alpha=0.3)

        _, path = self._resolve_output(ctx, "residual.png")
        fig.savefig(str(path), dpi=ctx.options.get("dpi", 300), bbox_inches="tight")
        plt.close(fig)

        metadata = {
            "residual_mean": float(np.mean(data.residuals)),
            "residual_std": float(np.std(data.residuals)),
        }
        return VizResult(intent="residual", paths=[path], metadata=metadata)


__all__ = ["PySRVizAdapter"]
