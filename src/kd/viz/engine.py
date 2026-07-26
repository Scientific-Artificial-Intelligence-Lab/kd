"""VizEngine: orchestration layer for rendering experiment visualizations."""

from __future__ import annotations

import ast
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from kd.core.equation import Form
from kd.data.schema import DataTopology
from kd.viz.extension import HomogeneousVizExtension, VizExtension
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
    from collections.abc import Iterable, Sequence

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






_PROTECTED_SEMANTICS_NOTE = (
    "Protected-operator note: the integrated RHS contains exp/log, which "
    "the platform evaluates with protected semantics (safe_exp/safe_log "
    "clamping) — the same semantics under which the equation was scored "
    "during search. Trajectories that would diverge under bare operators "
    "may remain bounded."
)



_PROTECTED_OPERATORS = frozenset({"exp", "log"})









_INTEGRATION_PRUNE_RTOL = 1e-12





_INTEGRATION_DEGRADED_NOTE = (
    "Integration-dependent plots (field comparison, time slices, error "
    "heatmap) are rendered in degraded form; other plots and metrics are "
    "unaffected."
)


def _prune_near_zero_terms(
    terms: Sequence[str],
    coefficients: Sequence[float],
    active: list[int],
) -> tuple[list[int], list[str]]:
    """Drop active terms whose coefficient is numerically zero.

    A term is pruned when ``|c| < _INTEGRATION_PRUNE_RTOL * max|c|`` over
    the active, finite coefficients. Non-finite coefficients are never
    pruned: they flow on to ``_assemble_integration_rhs``, which rejects
    them explicitly with a ValueError naming the offending term —
    fail-loud beats silent drop. The largest-|c| term always survives,
    so a non-empty selection stays non-empty.

    Returns:
        Tuple of (surviving indices, disclosure notes — one summary line
        when anything was pruned, empty otherwise).
    """
    finite_magnitudes = [
        abs(coefficients[i]) for i in active if math.isfinite(coefficients[i])
    ]
    if not finite_magnitudes:
        return active, []
    threshold = _INTEGRATION_PRUNE_RTOL * max(finite_magnitudes)
    dropped = [
        i
        for i in active
        if math.isfinite(coefficients[i]) and abs(coefficients[i]) < threshold
    ]
    if not dropped:
        return active, []
    keep = [i for i in active if i not in set(dropped)]
    detail = ", ".join(f"'{terms[i]}' (coeff {coefficients[i]:.3g})" for i in dropped)
    note = (
        f"Near-zero term(s) excluded from time integration: {detail} "
        f"(|coeff| < {_INTEGRATION_PRUNE_RTOL:g} * max|coeff|); "
        "reported equation and metrics keep the full term list."
    )
    return keep, [note]


def _assemble_integration_rhs(
    terms: Sequence[str],
    coefficients: Sequence[float],
    keep: Sequence[int],
) -> str:
    """Assemble the integrable RHS IR string from surviving terms (-4).

    ``"(c0)*(term0) + (c1)*(term1) + ..."`` with coefficients serialized
    via ``repr()`` (float64 round-trip). An empty selection yields ``"0"``
    so ``integrate_pde``'s scalar-broadcast branch keeps the field at its
    initial condition.

    Exactly-zero coefficients are omitted without disclosure: dropping
    ``(0.0)*(term)`` is mathematically lossless and preserves the old
    sympy path's ``0*x -> 0`` canonicalization (an all-zero equation must
    integrate as u_t = 0, not fail on an unintegrable zero-weighted
    term). Near-zero-but-nonzero terms are handled — and disclosed — by
    ``_prune_near_zero_terms`` instead.

    Raises:
        ValueError: If any surviving coefficient is non-finite. A NaN/Inf
            refit coefficient is an upstream pipeline defect, not a
            property of the RHS terms — it must fail loud HERE with an
            accurate attribution (the old path's ``format_pde`` raised
            'Coefficients must be finite'), not serialize via ``repr()``
            into a bare ``nan``/``inf`` symbol that the integrator's
            classifier would misreport as an unrecognised RHS symbol.
            Assembly runs before ``_get_integration_result``'s try/except,
            so this propagates to the caller by design.
    """
    non_finite = [i for i in keep if not math.isfinite(coefficients[i])]
    if non_finite:
        detail = ", ".join(
            f"term '{terms[i]}' has coefficient {coefficients[i]!r}"
            for i in non_finite
        )
        raise ValueError(
            f"Cannot assemble integration RHS: non-finite coefficient(s) — "
            f"{detail}. Coefficients must be finite; a NaN/Inf here points "
            "at a degenerate upstream fit/refit, not at the RHS terms."
        )
    survivors = [i for i in keep if coefficients[i] != 0.0]
    if not survivors:
        return "0"
    return " + ".join(f"({coefficients[i]!r})*({terms[i]})" for i in survivors)


def _protected_semantics_note(rhs: str) -> str | None:
    """Return the protected-operator disclosure when the RHS needs it.

    -5: when the integrable RHS calls exp/log, disclose that the
    registry evaluates them as safe_exp/safe_log (clamped). Uses an AST
    walk (not substring matching) so e.g. a hypothetical ``myexp(...)``
    does not false-positive; unparseable RHS strings yield no note —
    ``integrate_pde`` reports those on its own.
    """
    try:
        tree = ast.parse(rhs, mode="eval")
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _PROTECTED_OPERATORS
        ):
            return _PROTECTED_SEMANTICS_NOTE
    return None


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
            path, and warnings.
        """








        is_scatter = getattr(dataset, "topology", None) == DataTopology.SCATTERED
        try:
            field_shape = (
                None if (dataset is None or is_scatter) else dataset.get_shape()
            )
        except (ValueError, AttributeError):
            field_shape = None

        report = self.render_universal(
            result, field_shape=field_shape, infer_grid=not is_scatter
        )


        if dataset is not None:
            self._render_field_comparison(result, dataset, report, animate=animate)


        universal_figures = list(report.figures)



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
            report.warnings.extend(warnings)


        path, warnings = self._render_tier2(
            "residual",
            plot_residual,
            result=result,
            field_shape=field_shape,
            infer_grid=infer_grid,
        )
        if path is not None:
            report.figures.append(path)
        report.warnings.extend(warnings)

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
        report.warnings.extend(warnings)


        path, warnings = self._render_comparison_one(
            "comparison_scores",
            plot_score_bar,
            results,
            labels=labels,
            figsize=_COMPARISON_FIGSIZE,
        )
        if path is not None:
            report.figures.append(path)
        report.warnings.extend(warnings)


        path, warnings = self._render_comparison_one(
            "comparison_summary",
            plot_summary_table,
            results,
            labels=labels,
            figsize=_SUMMARY_FIGSIZE,
        )
        if path is not None:
            report.figures.append(path)
        report.warnings.extend(warnings)

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
        equation = result.equation
        if (
            equation is None
            or equation.form is not Form.HOMOGENEOUS
            or algorithm is None
            or not isinstance(algorithm, HomogeneousVizExtension)
        ):
            return []

        specs: list[FigureSpec] = []
        for plot_info in algorithm.list_homogeneous_plots():
            subplot_kw = (
                {"projection": plot_info.projection}
                if plot_info.projection is not None
                else None
            )
            fig, ax = plt.subplots(
                figsize=_PLUGIN_FIGSIZE,
                dpi=_DEFAULT_DPI,
                subplot_kw=subplot_kw,
            )
            try:
                with style_context(self._style):
                    algorithm.render_homogeneous_plot(plot_info.name, ax, result)
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
            fig, ax = plt.subplots(
                figsize=_PLUGIN_FIGSIZE,
                dpi=_DEFAULT_DPI,
            )
            try:
                with style_context(self._style):
                    algorithm.render_plot(plot_info.name, ax)
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
            self._merge_warnings(
                report,
                ["scattered data: field-grid plots skipped (no grid topology)"],
            )
            return


        integration_result, prune_notes = self._get_integration_result(result, dataset)
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
            result=result,
            dataset=dataset,
            integration_result=integration_result,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)


        if animate:
            path, warnings = self._render_animation(
                result=result,
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
            result=result,
            dataset=dataset,
            integration_result=integration_result,
        )
        if path is not None:
            report.figures.append(path)
        self._merge_warnings(report, warnings)


        path, warnings = self._render_tier2(
            "error_heatmap",
            plot_error_heatmap,
            result=result,
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

    def _get_integration_result(
        self,
        result: ExperimentResult,
        dataset: PDEDataset,
    ) -> tuple[IntegrationResult, list[str]]:
        """Compute integration result for field_comparison/time_slices/error_heatmap.

        Near-zero coefficients are pruned from the integrable RHS first
        (see ``_prune_near_zero_terms``); the returned notes disclose what
        was pruned and must reach the report exactly once
        (``_render_field_comparison`` owns that).

        The integrable RHS is the platform IR string assembled directly
        from the pruned terms + coefficients (-4):
        ``"(c0)*(term0) + (c1)*(term1) + ..."`` with ``repr()``
        coefficients for float64 round-trip, or ``"0"`` when everything
        is pruned/deselected. ``format_pde``/sympy no longer sit on the
        integration path — they serve LaTeX display only, so nested
        open-form derivative terms reach ``integrate_pde`` losslessly.

        Only ``integrate_pde()`` is wrapped in try/except (it may fail for
        legitimate scientific reasons). Attribute access and the string
        assembly are programmer-level steps whose errors should propagate
        normally.

        Why no autograd-note annotation here: the autograd-domain
        warning is emitted ONCE engine-side by ``_render_field_comparison``
        (see the dedup check around ``autograd_note``). Mutating
        ``IntegrationResult.warning`` would let Tier 2 plots forward the
        annotated note 4x (one per plot), bypassing the dedup guard.

        Returns:
            Tuple of (integration result, disclosure notes).
        """
        from kd.core.integrator import IntegrationResult, integrate_pde

        terms = result.final_eval.terms
        coeffs = result.final_eval.coefficients
        if terms is None or coeffs is None:
            return (
                IntegrationResult(
                    success=False,
                    warning="Missing terms or coefficients in final_eval",
                ),
                [],
            )
        coeff_values = [float(c) for c in coeffs]
        selected = result.final_eval.selected_indices
        active = list(selected) if selected is not None else list(range(len(terms)))
        keep, notes = _prune_near_zero_terms(terms, coeff_values, active)
        rhs = _assemble_integration_rhs(terms, coeff_values, keep)
        protected_note = _protected_semantics_note(rhs)
        if protected_note is not None:
            notes = [*notes, protected_note]
        try:
            return integrate_pde(rhs, dataset), notes
        except Exception as exc:
            return (
                IntegrationResult(
                    success=False,
                    warning=f"Integration failed: {exc}",
                ),
                notes,
            )

    @staticmethod
    def _maybe_autograd_domain_note(result: ExperimentResult) -> str | None:
        """Return the autograd-domain note, or ``None`` if not applicable.

        The note applies when result config says the fit used an autograd
        provider (``provider_kind == "autograd"``) or the SGA-specific internal
        autograd path (``use_autograd is True``). The latter remains necessary
        because SGA builds its own internal provider while its platform-level
        derivative requirement stays finite-difference.
        """
        config = getattr(result, "config", None)
        if not isinstance(config, dict):
            return None
        uses_autograd_domain = (
            config.get("provider_kind") == "autograd"
            or config.get("use_autograd") is True
        )
        if not uses_autograd_domain:
            return None
        return _AUTOGRAD_DOMAIN_NOTE

    def _render_pde_residual(
        self,
        result: ExperimentResult,
        dataset: PDEDataset,
        report: ReportResult,
    ) -> None:
        """Render PDE residual field (u_t actual vs predicted)."""
        try:
            field_shape = dataset.get_shape()
        except (ValueError, AttributeError):
            field_shape = None

        path, warnings = self._render_tier2(
            "pde_residual_field",
            plot_pde_residual_field,
            result=result,
            field_shape=field_shape,
            dataset=dataset,
        )
        if path is not None:
            report.figures.append(path)
        report.warnings.extend(warnings)

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
        result: ExperimentResult,
        dataset: PDEDataset,
        integration_result: IntegrationResult,
    ) -> tuple[Path | None, list[str]]:
        """Render the optional 2D field animation GIF."""
        if not self._can_render_animation(dataset):
            return None, []
        if not PillowWriter.isAvailable():
            msg = "PillowWriter unavailable; skipping field_animation.gif"
            logger.warning(msg)
            return None, [msg]

        figs_before = set(plt.get_fignums())
        animation = None
        try:
            animation, warnings = plot_field_animation(
                result,
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

        Returns:
            Tuple of (saved file path or None on error, warnings).
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=_DEFAULT_DPI)
        try:
            with style_context(self._style):
                warnings = plot_fn(result, ax)
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

        Returns:
            Tuple of (saved file path or None on error, warnings).
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=_DEFAULT_DPI)
        try:
            with style_context(self._style):
                warnings = plot_fn(results, ax, labels=labels)
            path = self._output_dir / f"{name}.{_SVG_FORMAT}"
            fig.savefig(path, format=_SVG_FORMAT, bbox_inches="tight")
        except Exception as exc:
            msg = f"Comparison plot '{name}' failed: {exc}"
            logger.warning(msg)
            return None, [msg]
        finally:
            plt.close(fig)

        return path, warnings
