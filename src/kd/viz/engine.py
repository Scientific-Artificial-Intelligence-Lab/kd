"""VizEngine: orchestration layer for rendering experiment visualizations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from kd.core.equation import DEFAULT_LHS_LABEL, Form
from kd.data.schema import DataTopology
from kd.viz.extension import HomogeneousVizExtension, VizExtension
from kd.viz.integration_assembly import build_integration_result
from kd.viz.plots.animation import plot_field_animation
from kd.viz.plots.coefficient import plot_coefficient_bar
from kd.viz.plots.comparison import (
    plot_score_bar,
    plot_summary_table,
    render_overlaid_convergence,
)
from kd.viz.plots.convergence import plot_convergence
from kd.viz.plots.equation import plot_equation
from kd.viz.plots.equation_tree import plot_equation_tree
from kd.viz.plots.error_heatmap import plot_error_heatmap
from kd.viz.plots.field import plot_field_comparison
from kd.viz.plots.parity import plot_parity
from kd.viz.plots.pde_residual import plot_pde_residual_field
from kd.viz.plots.residual import plot_residual
from kd.viz.plots.time_slices import plot_time_slices
from kd.viz.report import FigureSpec, ReportResult, generate_report
from kd.viz.style import style_context

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kd.core.integrator import IntegrationResult
    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150
_SVG_FORMAT = "svg"
_UNIVERSAL_FIGSIZE = (8, 5)
_FIELD_FIGSIZE = (15, 4)
_COMPARISON_FIGSIZE = (10, 5)
_SUMMARY_FIGSIZE = (10, 3)
_PLUGIN_FIGSIZE = (8, 5)








_AUTOGRAD_DOMAIN_NOTE = (
    "Domain note: this run fitted derivatives in an autograd / NN-smoothed "
    "domain. Forward time integration necessarily applies "
    "finite-difference spatial derivatives to the evolved states (the same "
    " stencils as the data pipeline; no autograd surrogate exists for "
    "integrator states). Integrating autograd-fitted coefficients with "
    "finite-difference derivatives is an inherent approximation of this plot: "
    "final_eval (fit quality in the autograd domain) and field-comparison "
    "metrics (physical recovery in the finite-difference domain) measure "
    "different things and may disagree on noisy data. This is expected, not a "
    "discovery error."
)





_INTEGRATION_DEGRADED_NOTE = (
    "Integration-dependent plots (field comparison, time slices, error "
    "heatmap) are rendered in degraded form; other plots and metrics are "
    "unaffected."
)






_NO_DATASET_NOTE = (
    "no dataset provided: dataset-dependent plots skipped (coefficient bar, "
    "field comparison, PDE residual field, time slices, error heatmap)"
)
_NO_ALGORITHM_NOTE = "no algorithm provided: plugin and form-specific plots skipped"




_ANIMATION_TOPOLOGY_NOTE = (
    "field animation requires a 2D-spatial+time dataset; skipping field_animation.gif"
)


def _scatter_aware_field_shape(
    dataset: Any,
) -> tuple[tuple[int, ...] | None, bool]:
    """Resolve ``(field_shape, is_scatter)`` for the residual-panel plots.

    Single source for ``render_all``'s residual path and
    ``_render_pde_residual``:
    a SCATTERED dataset has no grid, so ``get_shape()`` is a 1-D ``(N,)``
    point count that never matches the (primary coeff-grid) residual length --
    passing it makes the spatial-residual panel emit a spurious "does not
    match data size" warning. Treat SCATTERED (and a missing dataset) as
    shapeless (``field_shape=None``) AND forbid the square-shape guess
    (callers pass ``infer_grid=not is_scatter``) so the panels degrade to a
    DISCLOSED fallback ("No spatial data" / 1D line panels) rather than
    fabricating a square heatmap from a coincidentally-square residual count.
    """

    is_scatter = getattr(dataset, "topology", None) == DataTopology.SCATTERED
    try:
        field_shape = None if (dataset is None or is_scatter) else dataset.get_shape()
    except (ValueError, AttributeError):
        field_shape = None
    return field_shape, is_scatter


class VizEngine:
    """Orchestrates rendering of universal, plugin, and comparison plots.

    The engine creates figures, applies styling, delegates to individual
    plot functions, saves output, and closes figures to avoid resource
    leaks. After rendering plots it generates an HTML report with
    inline SVG figures (LaTeX in the equation block is rendered by
    MathJax loaded from a public CDN).

    Args:
        output_dir: Directory for rendered output. Created if missing.
        style: Extra matplotlib rcParams merged on top of DEFAULT_STYLE.
    """

    def __init__(
        self,
        output_dir: Path,
        style: dict[str, Any] | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._style = dict(style) if style else {}






        with style_context(self._style):
            pass





    def render_all(
        self,
        result: ExperimentResult,
        *,
        algorithm: Any | None = None,
        dataset: Any | None = None,
        animate: bool = False,
    ) -> ReportResult:
        """Render universal plots, plugin plots, and HTML report.

        Args:
            result: Completed experiment result.
            algorithm: If it implements VizExtension, plugin plots
                are rendered with per-plot error isolation.
            dataset: If provided (PDEDataset), enables field comparison
                (u-field True/Predicted/Residual) and PDE residual field.
            animate: When True, render a 2D field animation GIF for
                2D-spatial+time datasets. Defaults to False.

        Returns:
            ReportResult with generated figure paths, HTML report
            path, and warnings. Omitting ``dataset`` / ``algorithm``
            (or requesting ``animate`` on non-2D data) skips the
            corresponding figure families; each skip is disclosed in
            ``report.warnings`` rather than silently shrinking the
            figure set.
        """


        field_shape, is_scatter = _scatter_aware_field_shape(dataset)

        report = self.render_universal(
            result, field_shape=field_shape, infer_grid=not is_scatter
        )


        if dataset is not None:
            self._render_field_comparison(result, dataset, report, animate=animate)
        else:
            notes = [_NO_DATASET_NOTE]
            if animate:
                notes.append(_ANIMATION_TOPOLOGY_NOTE)
            self._merge_warnings(report, notes)


        universal_figures = list(report.figures)

        if algorithm is None:
            self._merge_warnings(report, [_NO_ALGORITHM_NOTE])



        homogeneous_specs = self._render_homogeneous_plots(result, algorithm, report)


        plugin_specs = homogeneous_specs + self._render_plugin_plots(algorithm, report)


        html_path = self._output_dir / "report.html"
        generate_report(
            result,
            universal_figures,
            html_path,
            plugin_figures=plugin_specs,
            warnings=report.warnings,
        )
        report.report = html_path

        return report

    def render_universal(
        self,
        result: ExperimentResult,
        *,
        field_shape: tuple[int, ...] | None = None,
        infer_grid: bool = True,
    ) -> ReportResult:
        """Render universal plots from an ExperimentResult.

        Tier 1 (engine creates fig+ax): convergence, parity, equation.
        Tier 2 (plot creates its own figure): residual (histogram + heatmap).

        ``coefficient_bar`` and field-comparison Tier 2 plots require an
        explicit dataset and are handled by ``render_all``.

        Args:
            result: Completed experiment result.
            field_shape: Optional field shape (e.g., ``(nx, nt)``) used to
                reshape 1D residuals into a 2D heatmap. When omitted the
                spatial-residual panel falls back to a square-shape guess
                and shows "No spatial data" if that fails.
            infer_grid: When ``field_shape`` is omitted, whether to allow the
                square-shape guess. ``False`` (set by ``render_all`` for
                SCATTERED data) forces "No spatial data" instead of a
                fabricated grid.

        Returns:
            ReportResult with generated figure paths and warnings.
        """
        report = ReportResult()


        tier1_specs = [
            ("convergence", plot_convergence, _UNIVERSAL_FIGSIZE),
            ("parity", plot_parity, _UNIVERSAL_FIGSIZE),
            ("equation", plot_equation, _UNIVERSAL_FIGSIZE),
            ("equation_tree", plot_equation_tree, _UNIVERSAL_FIGSIZE),
        ]

        for name, plot_fn, figsize in tier1_specs:
            path, warnings = self._render_one(
                name,
                plot_fn,
                result,
                figsize=figsize,
            )
            if path is not None:
                report.figures.append(path)
            self._merge_warnings(report, warnings)


        path, warnings = self._render_tier2(
            "residual",
            plot_residual,
            result=result,
            field_shape=field_shape,
            infer_grid=infer_grid,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)

        return report

    def render_comparison(
        self,
        results: list[ExperimentResult],
        *,
        labels: list[str] | None = None,
    ) -> ReportResult:
        """Render multi-run comparison plots.

        Generates:
        - Overlaid convergence curves
        - Score bar chart
        - Summary table (expression, metrics, iterations)

        Args:
            results: List of completed experiment results.
            labels: Custom labels for each run. Falls back to
                ``algorithm_name`` from each result.

        Returns:
            ReportResult with generated figure paths and warnings.
        """
        report = ReportResult()


        path, warnings = self._render_comparison_one(
            "comparison_convergence",
            render_overlaid_convergence,
            results,
            labels=labels,
            figsize=_COMPARISON_FIGSIZE,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)


        path, warnings = self._render_comparison_one(
            "comparison_scores",
            plot_score_bar,
            results,
            labels=labels,
            figsize=_COMPARISON_FIGSIZE,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)


        path, warnings = self._render_comparison_one(
            "comparison_summary",
            plot_summary_table,
            results,
            labels=labels,
            figsize=_SUMMARY_FIGSIZE,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)

        return report





    def _render_homogeneous_plots(
        self,
        result: ExperimentResult,
        algorithm: Any | None,
        report: ReportResult,
    ) -> list[FigureSpec]:
        """Render producer-owned plots only for an explicit HOMOGENEOUS form.

        Returns:
            One ``FigureSpec`` per successfully rendered plot, carrying the
            title and description the producer declared on its ``PlotInfo``.
        """
        if algorithm is None or not isinstance(algorithm, HomogeneousVizExtension):
            return []
        declared = algorithm.list_homogeneous_plots()
        if not declared:


            return []
        equation = result.equation
        if equation is None or equation.form is not Form.HOMOGENEOUS:



            reason = (
                "result carries no equation (invalid run)"
                if equation is None
                else f"equation form is {equation.form.name}, not HOMOGENEOUS"
            )
            self._merge_warnings(
                report,
                [f"Steady-state plots ({len(declared)}) skipped: {reason}"],
            )
            return []

        specs: list[FigureSpec] = []
        for plot_info in declared:
            subplot_kw = (
                {"projection": plot_info.projection}
                if plot_info.projection is not None
                else None
            )


            with style_context(self._style):
                fig, ax = plt.subplots(
                    figsize=_PLUGIN_FIGSIZE,
                    dpi=_DEFAULT_DPI,
                    subplot_kw=subplot_kw,
                )
            try:
                with style_context(self._style):
                    plugin_warnings = algorithm.render_homogeneous_plot(
                        plot_info.name, ax, result
                    )



                if plugin_warnings:
                    self._merge_warnings(report, plugin_warnings)
                path = self._output_dir / (
                    f"homogeneous_{plot_info.name}.{_SVG_FORMAT}"
                )
                fig.savefig(path, format=_SVG_FORMAT, bbox_inches="tight")
                specs.append(
                    FigureSpec(
                        path=path,
                        title=plot_info.title,
                        description=plot_info.description,
                    )
                )
                report.figures.append(path)
            except Exception as exc:
                msg = f"Homogeneous plot '{plot_info.name}' failed: {exc}"
                logger.warning(msg)
                report.warnings.append(msg)
            finally:
                plt.close(fig)
        return specs

    def _render_plugin_plots(
        self,
        algorithm: Any | None,
        report: ReportResult,
    ) -> list[FigureSpec]:
        """Render plugin plots if algorithm implements VizExtension.

        Per-plot error isolation: a failing plugin plot does not prevent
        other plots or the report from being generated.

        Returns:
            One ``FigureSpec`` per successfully rendered plugin figure,
            carrying the title and description the plugin declared on its
            ``PlotInfo`` descriptor.
        """
        plugin_specs: list[FigureSpec] = []
        if algorithm is None:
            return plugin_specs




        if not isinstance(algorithm, VizExtension):
            msg = (
                f"Algorithm {type(algorithm).__name__} implements no "
                "VizExtension; 0 plugin plots rendered"
            )
            logger.warning(msg)
            report.warnings.append(msg)
            return plugin_specs
        plot_infos = algorithm.list_plots()
        if not plot_infos:
            msg = (
                f"Algorithm {type(algorithm).__name__} implements VizExtension "
                "but declared zero plots; 0 plugin plots rendered"
            )
            logger.warning(msg)
            report.warnings.append(msg)
            return plugin_specs

        for plot_info in plot_infos:



            with style_context(self._style):
                fig, ax = plt.subplots(
                    figsize=_PLUGIN_FIGSIZE,
                    dpi=_DEFAULT_DPI,
                )
            try:
                with style_context(self._style):
                    plugin_warnings = algorithm.render_plot(plot_info.name, ax)






                if plugin_warnings:
                    self._merge_warnings(report, plugin_warnings)
                path = self._output_dir / f"plugin_{plot_info.name}.{_SVG_FORMAT}"
                fig.savefig(path, format=_SVG_FORMAT, bbox_inches="tight")
                plugin_specs.append(
                    FigureSpec(
                        path=path,
                        title=plot_info.title,
                        description=plot_info.description,
                    )
                )
                report.figures.append(path)
            except Exception as exc:
                msg = f"Plugin plot '{plot_info.name}' failed: {exc}"
                logger.warning(msg)
                report.warnings.append(msg)
            finally:
                plt.close(fig)

        return plugin_specs

    def _render_field_comparison(
        self,
        result: ExperimentResult,
        dataset: PDEDataset,
        report: ReportResult,
        *,
        animate: bool = False,
    ) -> None:
        """Render field comparison and Tier 2 plots that need dataset."""



        lhs_note = self._maybe_lhs_assumption_note(result, dataset)
        if lhs_note:
            self._merge_warnings(report, [lhs_note])


        path, warnings = self._render_one(
            "coefficient_bar",
            plot_coefficient_bar,
            result,
            figsize=_UNIVERSAL_FIGSIZE,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)










        if getattr(dataset, "topology", None) == DataTopology.SCATTERED:
            notes = ["scattered data: field-grid plots skipped (no grid topology)"]



            if animate:
                notes.append(_ANIMATION_TOPOLOGY_NOTE)
            self._merge_warnings(report, notes)
            return


        try:
            integration_result, prune_notes = build_integration_result(
                result, dataset
            )
        except ValueError as exc:





            from kd.core.integrator import IntegrationResult

            integration_result = IntegrationResult(success=False, warning=str(exc))
            prune_notes = []
        self._merge_warnings(report, prune_notes)





        if not integration_result.success:
            self._merge_warnings(
                report,
                [
                    integration_result.warning or "Integration failed",
                    _INTEGRATION_DEGRADED_NOTE,
                ],
            )





        autograd_note = self._maybe_autograd_domain_note(result)
        if autograd_note:
            self._merge_warnings(report, [autograd_note])

        path, warnings = self._render_tier2(
            "field_comparison",
            plot_field_comparison,
            dataset=dataset,
            integration_result=integration_result,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)


        if animate:
            path, warnings = self._render_animation(
                dataset=dataset,
                integration_result=integration_result,
            )
            if path is not None:
                report.data_files.append(path)
            self._merge_warnings(report, warnings)


        self._render_pde_residual(result, dataset, report)


        path, warnings = self._render_tier2(
            "time_slices",
            plot_time_slices,
            dataset=dataset,
            integration_result=integration_result,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)


        path, warnings = self._render_tier2(
            "error_heatmap",
            plot_error_heatmap,
            dataset=dataset,
            integration_result=integration_result,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)

    @staticmethod
    def _merge_warnings(report: ReportResult, warnings: Iterable[str]) -> None:
        """Append warnings to the report, skipping exact duplicates.

        Several Tier 2 plots consume the same IntegrationResult and each
        forwards its warning — correct for standalone plot calls, but at
        report level a shared failure is a single fact and is reported
        once. Identical strings carry no extra information when repeated.
        """
        for msg in warnings:
            if msg not in report.warnings:
                report.warnings.append(msg)

    @staticmethod
    def _maybe_autograd_domain_note(result: ExperimentResult) -> str | None:
        """Return the autograd-domain note, or ``None`` if not applicable.

        The note applies when result config says the fit used an autograd
        provider (``provider_kind == "autograd"``) or the SGA-specific internal
        autograd path (``use_autograd is True``). The latter remains necessary
        because SGA builds its own internal provider while its platform-level
        derivative requirement stays finite-difference.
        """
        config = result.config
        uses_autograd_domain = (
            config.get("provider_kind") == "autograd"
            or config.get("use_autograd") is True
        )
        if not uses_autograd_domain:
            return None
        return _AUTOGRAD_DOMAIN_NOTE

    @staticmethod
    def _maybe_lhs_assumption_note(
        result: ExperimentResult, dataset: Any
    ) -> str | None:
        """Return a disclosure note when ``result.lhs_label`` is an assumption.

        Mirrors the no-declaration condition of ``Runner._lhs_label``'s final
        fallback: when neither the algorithm (``final_eval.lhs_name``) nor the
        dataset (``lhs_field``/``lhs_axis``) declared the regression target,
        the label reaching equation and residual panels is the hardcoded
        ``DEFAULT_LHS_LABEL``, and the report must say so.
        The label itself is the first gate: a
        non-default label — a homogeneous result's ``"0"``, a directly
        constructed custom label — cannot be the hardcoded guess this note
        claims it is. Do NOT key this on ``final_eval.form``: that field is a
        transient dispatch signal that does not survive save/load (review
        finding, 2026-08-06).
        """

        if result.lhs_label != DEFAULT_LHS_LABEL:
            return None
        lhs_name = result.final_eval.lhs_name
        if isinstance(lhs_name, str) and lhs_name:
            return None
        lhs_field = getattr(dataset, "lhs_field", "")
        lhs_axis = getattr(dataset, "lhs_axis", "")
        if (
            isinstance(lhs_field, str)
            and lhs_field
            and isinstance(lhs_axis, str)
            and lhs_axis
        ):
            return None
        return (
            f"LHS label {result.lhs_label!r} is a default assumption: neither "
            "the algorithm (final_eval.lhs_name) nor the dataset "
            "(lhs_field/lhs_axis) declared the regression target"
        )

    def _render_pde_residual(
        self,
        result: ExperimentResult,
        dataset: PDEDataset,
        report: ReportResult,
    ) -> None:
        """Render PDE residual field (u_t actual vs predicted)."""



        field_shape, is_scatter = _scatter_aware_field_shape(dataset)

        path, warnings = self._render_tier2(
            "pde_residual_field",
            plot_pde_residual_field,
            result=result,
            field_shape=field_shape,
            dataset=dataset,
            infer_grid=not is_scatter,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)

    def _render_tier2(
        self,
        name: str,
        plot_fn: Any,
        **kwargs: Any,
    ) -> tuple[Path | None, list[str]]:
        """Render a Tier 2 plot (creates its own Figure).

        Returns:
            Tuple of (saved file path or None on error, warnings).
        """
        figs_before = set(plt.get_fignums())
        fig = None
        try:
            fig, warnings = plot_fn(**kwargs, style=self._style)
            path = self._output_dir / f"{name}.{_SVG_FORMAT}"
            fig.savefig(
                path,
                format=_SVG_FORMAT,
                bbox_inches="tight",
                dpi=_DEFAULT_DPI,
            )
            return path, warnings
        except Exception as exc:

            for num in set(plt.get_fignums()) - figs_before:
                plt.close(num)
            msg = f"Plot '{name}' failed: {exc}"
            logger.warning(msg)
            return None, [msg]
        finally:
            if fig is not None:
                plt.close(fig)

    def _render_animation(
        self,
        *,
        dataset: PDEDataset,
        integration_result: IntegrationResult,
    ) -> tuple[Path | None, list[str]]:
        """Render the optional 2D field animation GIF."""
        if not self._can_render_animation(dataset):
            logger.warning(_ANIMATION_TOPOLOGY_NOTE)
            return None, [_ANIMATION_TOPOLOGY_NOTE]
        if not PillowWriter.isAvailable():
            msg = "PillowWriter unavailable; skipping field_animation.gif"
            logger.warning(msg)
            return None, [msg]

        figs_before = set(plt.get_fignums())
        animation = None
        try:
            animation, warnings = plot_field_animation(
                dataset,
                integration_result,
                style=self._style,
            )
            if animation is None:
                return None, warnings
            path = self._output_dir / "field_animation.gif"
            writer = PillowWriter(fps=8)
            animation.save(path, writer=writer)
            return path, warnings
        except Exception as exc:
            for num in set(plt.get_fignums()) - figs_before:
                plt.close(num)
            msg = f"Animation 'field_animation' failed: {exc}"
            logger.warning(msg)
            return None, [msg]
        finally:
            if animation is not None:
                fig = getattr(animation, "_fig", None)
                if fig is not None:
                    plt.close(fig)

    @staticmethod
    def _can_render_animation(dataset: PDEDataset) -> bool:
        """Return True for 2D-spatial+time datasets."""
        return (
            dataset.axis_order is not None
            and dataset.lhs_axis in dataset.axis_order
            and len(dataset.spatial_axes) == 2
        )

    def _render_one(
        self,
        name: str,
        plot_fn: Any,
        result: ExperimentResult,
        *,
        figsize: tuple[float, float] = _UNIVERSAL_FIGSIZE,
    ) -> tuple[Path | None, list[str]]:
        """Create figure, apply style, call plot_fn, save, close.

        Per-plot error isolation: a failing plot does not prevent other
        plots or the report from being generated.

        The Axes is created INSIDE the style context: axis/tick label sizes
        fix at Axes creation, so a figure born outside the context keeps
        default sizes no matter what the plot function later applies
        (023-D3 revision). The engine's context exits before ``plot_fn``
        runs; the fn then applies the SAME merged rcParams itself via its
        ``style=`` parameter (sequential, not nested — the param exists so
        direct engine-less callers get styled artists too).

        Returns:
            Tuple of (saved file path or None on error, warnings).
        """
        with style_context(self._style):
            fig, ax = plt.subplots(figsize=figsize, dpi=_DEFAULT_DPI)
        try:
            warnings = plot_fn(result, ax, style=self._style)
            path = self._output_dir / f"{name}.{_SVG_FORMAT}"
            fig.savefig(path, format=_SVG_FORMAT, bbox_inches="tight")
        except Exception as exc:
            msg = f"Universal plot '{name}' failed: {exc}"
            logger.warning(msg)
            return None, [msg]
        finally:
            plt.close(fig)

        return path, warnings

    def _render_comparison_one(
        self,
        name: str,
        plot_fn: Any,
        results: list[ExperimentResult],
        *,
        labels: list[str] | None = None,
        figsize: tuple[float, float] = _COMPARISON_FIGSIZE,
    ) -> tuple[Path | None, list[str]]:
        """Create figure, apply style, call comparison plot_fn, save, close.

        Per-plot error isolation: a failing comparison plot does not
        prevent other comparison plots from being generated.

        Axes creation and style plumbing follow ``_render_one`` (023-D3
        revision): figure born inside the style context, plot fn re-wraps
        with the same ``style=`` for direct callers.

        Returns:
            Tuple of (saved file path or None on error, warnings).
        """
        with style_context(self._style):
            fig, ax = plt.subplots(figsize=figsize, dpi=_DEFAULT_DPI)
        try:
            warnings = plot_fn(results, ax, labels=labels, style=self._style)
            path = self._output_dir / f"{name}.{_SVG_FORMAT}"
            fig.savefig(path, format=_SVG_FORMAT, bbox_inches="tight")
        except Exception as exc:
            msg = f"Comparison plot '{name}' failed: {exc}"
            logger.warning(msg)
            return None, [msg]
        finally:
            plt.close(fig)

        return path, warnings
