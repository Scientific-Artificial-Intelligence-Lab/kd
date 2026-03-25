"""Discover Regression model adapter for the unified visualization facade."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .._contracts import (
    ParityPlotData,
    ResidualPlotData,
    RewardEvolutionData,
)
from ..core import VizResult
from ..discover_eq2latex import regression_program_to_latex
from ..equation_renderer import render_latex_to_image

logger = logging.getLogger(__name__)


class DiscoverRegressionVizAdapter:
    """Visualization adapter for KD_Discover_Regression models.

    Supports regression-specific intents (equation, parity, residual)
    and reuses Discover search visualization logic (search_evolution,
    density, tree).
    """

    capabilities: Iterable[str] = {
        "equation",
        "parity",
        "residual",
        "search_evolution",
        "density",
        "tree",
    }

    def render(self, request: Any, ctx: Any) -> VizResult:
        handler = {
            "equation": self._equation,
            "parity": self._parity,
            "residual": self._residual,
            "search_evolution": self._search_evolution,
            "density": self._density,
            "tree": self._tree,
        }.get(request.kind)

        if handler is None:
            return VizResult(
                intent=request.kind,
                warnings=[
                    f"DiscoverRegressionVizAdapter does not support "
                    f"intent '{request.kind}'."
                ],
                metadata={"capabilities": sorted(self.capabilities)},
            )

        return handler(request.target, ctx)

    # ------------------------------------------------------------------
    # Equation rendering
    # ------------------------------------------------------------------
    def _equation(self, model: Any, ctx: Any) -> VizResult:
        program = self._get_best_program(model)
        if program is None:
            return VizResult(
                intent="equation",
                warnings=["No best program available for equation rendering."],
            )

        var_names = self._get_var_names(model)
        target_name = ctx.options.get("target_name", "y")

        try:
            latex = regression_program_to_latex(
                program, var_names=var_names, target_name=target_name,
            )
        except Exception as exc:
            return VizResult(
                intent="equation",
                warnings=[f"Failed to convert program to LaTeX: {exc}"],
            )

        _, path = self._resolve_output(ctx, "equation.png")

        # Try PNG file rendering (may silently fail on some environments)
        png_ok = False
        try:
            render_latex_to_image(
                latex,
                output_path=str(path),
                font_size=ctx.options.get("font_size", 16),
                show=False,
            )
            png_ok = path.exists()
        except Exception as exc:
            logger.warning("Failed to render equation PNG: %s", exc)

        # Always create in-memory figure (decoupled from PNG success)
        fig = self._latex_figure(latex, ctx)

        return VizResult(
            intent="equation",
            figure=fig,
            paths=[path] if png_ok else [],
            metadata={"latex": latex},
        )

    @staticmethod
    def _latex_figure(latex: str, ctx: Any) -> Any:
        """Create in-memory matplotlib figure with rendered LaTeX equation."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis("off")
        try:
            ax.text(
                0.5, 0.5, latex, fontsize=16,
                ha="center", va="center", transform=ax.transAxes,
            )
            fig.tight_layout()
        except Exception:
            plt.close(fig)
            return None
        return fig

    # ------------------------------------------------------------------
    # Parity plot
    # ------------------------------------------------------------------
    def _parity(self, model: Any, ctx: Any) -> VizResult:
        y_true, y_pred = self._get_actual_predicted(model)
        if y_true is None:
            return VizResult(
                intent="parity",
                warnings=["Cannot build parity plot: model not fitted or data unavailable."],
            )

        data = ParityPlotData.from_actual_predicted(y_true, y_pred)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=ctx.options.get("figsize", (7, 7)))
        ax.scatter(
            data.actual_values, data.predicted_values,
            alpha=0.35, s=20, label="Prediction vs Actual",
        )

        min_val = float(min(data.actual_values.min(), data.predicted_values.min()))
        max_val = float(max(data.actual_values.max(), data.predicted_values.max()))
        ref = np.linspace(min_val, max_val, 100)
        ax.plot(ref, ref, "r--", linewidth=1.5, label="y = x")

        ax.set_xlabel(ctx.options.get("xlabel", "Actual"))
        ax.set_ylabel(ctx.options.get("ylabel", "Predicted"))
        ax.set_title(ctx.options.get("title", "Parity Plot"))
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_aspect("equal", "box")

        _, path = self._resolve_output(ctx, "parity_plot.png")
        fig.savefig(
            str(path), dpi=ctx.options.get("dpi", 300), bbox_inches="tight",
        )

        residual_stats = {
            "mean_residual": float(np.nanmean(data.residuals)),
            "max_abs_residual": float(np.nanmax(np.abs(data.residuals))),
            "rmse": float(np.sqrt(np.nanmean(np.square(data.residuals)))),
        }
        return VizResult(
            intent="parity",
            figure=fig,
            paths=[path],
            metadata={"parity": data, "summary": residual_stats},
        )

    # ------------------------------------------------------------------
    # Residual analysis
    # ------------------------------------------------------------------
    def _residual(self, model: Any, ctx: Any) -> VizResult:
        y_true, y_pred = self._get_actual_predicted(model)
        if y_true is None:
            return VizResult(
                intent="residual",
                warnings=["Cannot build residual plot: model not fitted or data unavailable."],
            )

        data = ResidualPlotData.from_actual_predicted(y_true, y_pred)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            1, 2, figsize=ctx.options.get("figsize", (12, 5)),
        )

        # Left: scatter of predicted vs residual
        axes[0].scatter(
            data.predicted, data.residuals, alpha=0.5, s=15,
        )
        axes[0].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # Right: residual histogram
        axes[1].hist(
            data.residuals,
            bins=int(ctx.options.get("bins", 30)),
            color="#1f77b4",
            alpha=0.7,
            edgecolor="black",
            density=True,
        )
        axes[1].axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        axes[1].set_xlabel("Residual Value")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Residual Distribution")

        fig.tight_layout()

        _, path = self._resolve_output(ctx, "residual_analysis.png")
        fig.savefig(
            str(path), dpi=ctx.options.get("dpi", 300), bbox_inches="tight",
        )

        summary = {
            "mean": float(np.nanmean(data.residuals)),
            "std": float(np.nanstd(data.residuals)),
            "max_abs": float(np.nanmax(np.abs(data.residuals))),
            "count": int(data.residuals.size),
        }
        return VizResult(
            intent="residual",
            figure=fig,
            paths=[path],
            metadata={"residual_data": data, "summary": summary},
        )

    # ------------------------------------------------------------------
    # Search evolution (reuse Discover logic)
    # ------------------------------------------------------------------
    def _search_evolution(self, model: Any, ctx: Any) -> VizResult:
        data = self._build_reward_evolution(model)
        if data is None:
            return VizResult(
                intent="search_evolution",
                warnings=["No reward history available for search evolution plot."],
            )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=ctx.options.get("figsize", (10, 6)))
        ax.plot(
            data.steps, data.best_reward,
            color="black", linewidth=2, label="Best Reward",
        )
        if data.max_reward is not None:
            ax.plot(
                data.steps, data.max_reward,
                color="#F47E62", linestyle="-.", linewidth=2,
                label="Maximum Reward",
            )
        if data.mean_reward is not None:
            ax.plot(
                data.steps, data.mean_reward,
                color="#4F8FBA", linestyle="--", linewidth=2,
                label="Average Reward",
            )

        ax.set_xlabel("Iteration Count")
        ax.set_ylabel("Reward Value")
        ax.set_title(ctx.options.get("title", "Reward Evolution"))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=False)

        _, path = self._resolve_output(ctx, "reward_evolution.png")
        fig.savefig(
            str(path), dpi=ctx.options.get("dpi", 300), bbox_inches="tight",
        )

        return VizResult(
            intent="search_evolution",
            figure=fig,
            paths=[path],
            metadata={"reward_evolution": data},
        )

    # ------------------------------------------------------------------
    # Density plot (reuse Discover logic)
    # ------------------------------------------------------------------
    def _density(self, model: Any, ctx: Any) -> VizResult:
        try:
            import seaborn as sns
        except ImportError:
            return VizResult(
                intent="density",
                warnings=["seaborn is required for reward density plots."],
            )

        history = getattr(getattr(model, "searcher", None), "r_history", None)
        if not history:
            return VizResult(
                intent="density",
                warnings=["No reward history available for density plot."],
            )

        import matplotlib.pyplot as plt

        epoches = ctx.options.get("epoches")
        max_curves = max(1, ctx.options.get("max_curves", 10))
        if epoches is None and len(history) > max_curves:
            step = max(1, len(history) // max_curves)
            epoches = list(range(0, len(history), step))
            if (len(history) - 1) not in epoches:
                epoches.append(len(history) - 1)
        data = (
            history
            if epoches is None
            else [history[i] for i in epoches if i < len(history)]
        )
        if not data:
            return VizResult(
                intent="density",
                warnings=["No data available for requested epochs."],
            )

        fig, ax = plt.subplots(figsize=ctx.options.get("figsize", (8, 5)))
        for idx, rewards in enumerate(data):
            sns.kdeplot(
                np.asarray(rewards), ax=ax,
                label=f"Epoch {epoches[idx] if epoches else idx}",
                fill=False,
            )

        ax.set_xlabel("Reward")
        ax.set_ylabel("Density")
        ax.set_title(ctx.options.get("title", "Reward Density Evolution"))
        ax.grid(True, linestyle="--", alpha=0.4)
        if len(data) > 1:
            ax.legend(loc="best", frameon=False)

        _, path = self._resolve_output(ctx, "reward_density.png")
        fig.savefig(
            str(path), dpi=ctx.options.get("dpi", 300), bbox_inches="tight",
        )

        metadata = {
            "epoches": epoches if epoches is not None else list(range(len(data))),
        }
        return VizResult(intent="density", figure=fig, paths=[path], metadata=metadata)

    # ------------------------------------------------------------------
    # Expression tree
    # ------------------------------------------------------------------
    def _tree(self, model: Any, ctx: Any) -> VizResult:
        try:
            import graphviz  # noqa: F401
        except ImportError:
            return VizResult(
                intent="tree",
                warnings=["graphviz is required for expression tree rendering."],
            )

        program = self._get_best_program(model)
        if program is None:
            return VizResult(
                intent="tree",
                warnings=["No best program available for tree rendering."],
            )

        searcher = getattr(model, "searcher", None)
        plotter = getattr(searcher, "plotter", None)
        if plotter is None or not hasattr(plotter, "tree_plot"):
            return VizResult(
                intent="tree",
                warnings=["Tree plotter is unavailable on the model searcher."],
            )

        graph = plotter.tree_plot(program)
        if graph is None:
            return VizResult(
                intent="tree",
                warnings=["Tree plotter did not return a graph object."],
            )

        graph.attr(dpi=str(ctx.options.get("dpi", 200)))
        try:
            png_bytes = graph.pipe(format="png")
        except Exception as exc:
            return VizResult(
                intent="tree",
                warnings=[f"Failed to render expression tree: {exc}"],
            )

        _, path = self._resolve_output(ctx, "expression_tree.png")
        path.write_bytes(png_bytes)

        metadata = {
            "expression": getattr(program, "str_expression", None),
        }
        return VizResult(intent="tree", paths=[path], metadata=metadata)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_best_program(self, model: Any) -> Any:
        """Retrieve the best program from a regression model.

        For regression models, ``best_program_`` is the canonical attribute
        set by ``fit()``. If it is explicitly ``None``, the model has not
        been fitted — return ``None`` without fallback.
        """
        if hasattr(model, "best_program_"):
            return model.best_program_
        # Fallback for non-standard models
        best = getattr(model, "best_p", None)
        if isinstance(best, list):
            best = best[0] if best else None
        return best

    def _get_var_names(self, model: Any) -> Optional[list]:
        """Retrieve variable names from the model."""
        result = getattr(model, "result_", None)
        if result is not None and isinstance(result, dict):
            return result.get("var_names")
        data_class = getattr(model, "data_class", None)
        if data_class is not None:
            return getattr(data_class, "var_names", None)
        return None

    def _get_actual_predicted(self, model: Any) -> tuple:
        """Return (y_true, y_pred) or (None, None) if unavailable."""
        data_class = getattr(model, "data_class", None)
        if data_class is None:
            return None, None

        y_true = getattr(data_class, "y", None)
        if y_true is None:
            return None, None

        X = getattr(data_class, "X", None)
        if X is None:
            return None, None

        try:
            y_pred = model.predict(X)
        except Exception as exc:
            logger.warning("predict() failed during visualization: %s", exc)
            return None, None

        y_true_arr = np.asarray(y_true).reshape(-1)
        y_pred_arr = np.asarray(y_pred).reshape(-1)

        if not np.all(np.isfinite(y_pred_arr)):
            logger.warning(
                "predict() returned non-finite values (NaN/Inf count: %d/%d)",
                int(np.sum(~np.isfinite(y_pred_arr))),
                y_pred_arr.size,
            )

        return y_true_arr, y_pred_arr

    def _build_reward_evolution(
        self, model: Any,
    ) -> Optional[RewardEvolutionData]:
        """Build reward evolution data from model's searcher."""
        searcher = getattr(model, "searcher", None)
        if searcher is None or not getattr(searcher, "r_train", None):
            return None

        rewards = searcher.r_train
        if not rewards:
            return None

        steps = np.arange(1, len(rewards) + 1)
        max_vals = []
        mean_vals = []
        best_vals = []
        cumulative_best = -np.inf
        for batch in rewards:
            arr = np.asarray(batch)
            max_val = float(np.nanmax(arr))
            mean_val = float(np.nanmean(arr))
            cumulative_best = max(cumulative_best, max_val)
            max_vals.append(max_val)
            mean_vals.append(mean_val)
            best_vals.append(cumulative_best)

        return RewardEvolutionData(
            steps=steps,
            best_reward=np.asarray(best_vals),
            max_reward=np.asarray(max_vals),
            mean_reward=np.asarray(mean_vals),
        )

    def _resolve_output(
        self, ctx: Any, filename: str,
    ) -> tuple:
        """Resolve the output path for a file."""
        base = ctx.options.get("output_dir")
        if base is not None:
            base_path = Path(base) / "discover_regression"
            base_path.mkdir(parents=True, exist_ok=True)
            return base_path, base_path / filename
        path = ctx.save_path(f"discover_regression/{filename}")
        return path.parent, path
