
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from kd.search.result import ExperimentResult
from kd.viz.plots.comparison import (
    plot_score_bar,
    plot_summary_table,
    render_overlaid_convergence,
)
from kd.viz.plots.convergence import plot_convergence
from kd.viz.plots.equation import plot_equation
from kd.viz.plots.parity import plot_parity
from kd.viz.plots.residual import plot_residual




class TestPlotConvergence:

    def test_renders_line(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_convergence(mock_experiment_result, ax)

        assert len(ax.get_lines()) >= 1
        plt.close(fig)

    def test_labels_set(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_convergence(mock_experiment_result, ax)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_empty_recorder_warns(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        from kd.search.recorder import VizRecorder

        mock_experiment_result.recorder = VizRecorder()
        fig, ax = plt.subplots()
        warnings = plot_convergence(mock_experiment_result, ax)
        assert len(warnings) > 0
        plt.close(fig)

    def test_flat_curve_gets_explanatory_subtitle(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        from kd.search.recorder import VizRecorder

        rec = VizRecorder()
        for _ in range(5):
            rec.log("_best_score", 0.9486)
        mock_experiment_result.recorder = rec
        fig, ax = plt.subplots()
        plot_convergence(mock_experiment_result, ax)
        assert "constant" in ax.get_title().lower()

        assert not any("constant" in t.get_text().lower() for t in ax.texts)
        plt.close(fig)

    def test_varied_curve_has_plain_title(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        from kd.search.recorder import VizRecorder

        rec = VizRecorder()
        for score in [0.90, 0.93, 0.95]:
            rec.log("_best_score", score)
        mock_experiment_result.recorder = rec
        fig, ax = plt.subplots()
        plot_convergence(mock_experiment_result, ax)
        assert ax.get_title() == "Convergence"
        plt.close(fig)





class TestPlotParity:

    def test_renders_scatter(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)

        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_has_45_degree_line(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)

        assert len(ax.get_lines()) >= 1
        plt.close(fig)

    def test_r2_annotation(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)

        texts = [t.get_text() for t in ax.texts]
        assert any("R" in t for t in texts)
        plt.close(fig)

    def test_labels_set(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_nan_values_handled(self, mock_experiment_result: ExperimentResult) -> None:
        mock_experiment_result.actual[0] = float("nan")
        mock_experiment_result.predicted[1] = float("nan")
        fig, ax = plt.subplots()
        warnings = plot_parity(mock_experiment_result, ax)

        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_constant_values(self, mock_experiment_result: ExperimentResult) -> None:
        mock_experiment_result.actual = torch.ones(10)
        mock_experiment_result.predicted = torch.ones(10)
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)

        assert len(ax.get_lines()) >= 1
        plt.close(fig)





class TestPlotResidual:

    def test_renders_histogram(self, mock_experiment_result: ExperimentResult) -> None:
        fig, warnings = plot_residual(mock_experiment_result)

        ax = fig.axes[0]
        assert len(ax.patches) >= 1
        plt.close(fig)

    def test_has_stats_annotation(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        fig, warnings = plot_residual(mock_experiment_result)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]

        assert any("mean" in t.lower() or "std" in t.lower() for t in texts)
        plt.close(fig)

    def test_no_residuals_warns(self, mock_experiment_result: ExperimentResult) -> None:
        mock_experiment_result.final_eval.residuals = None
        fig, warnings = plot_residual(mock_experiment_result)
        assert len(warnings) > 0
        plt.close(fig)

    def test_nan_residuals_filtered(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        residuals = torch.randn(20)
        residuals[0] = float("nan")
        residuals[5] = float("inf")
        mock_experiment_result.final_eval.residuals = residuals
        fig, warnings = plot_residual(mock_experiment_result)
        assert any("non-finite" in w.lower() for w in warnings)

        ax = fig.axes[0]
        assert len(ax.patches) >= 1
        plt.close(fig)

    def test_all_nan_residuals(self, mock_experiment_result: ExperimentResult) -> None:
        mock_experiment_result.final_eval.residuals = torch.full((10,), float("nan"))
        fig, warnings = plot_residual(mock_experiment_result)
        assert any("non-finite" in w.lower() for w in warnings)
        plt.close(fig)

    def test_labels_set(self, mock_experiment_result: ExperimentResult) -> None:
        fig, warnings = plot_residual(mock_experiment_result)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)





class TestPlotEquation:

    def test_renders_text(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_equation(mock_experiment_result, ax)

        assert len(ax.texts) >= 1
        plt.close(fig)

    def test_axis_off(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_equation(mock_experiment_result, ax)

        assert not ax.axison
        plt.close(fig)

    def test_empty_expression_warns(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        mock_experiment_result.best_expression = ""
        fig, ax = plt.subplots()
        warnings = plot_equation(mock_experiment_result, ax)
        assert len(warnings) > 0
        plt.close(fig)

    def test_renders_latex_not_raw_ir(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        mock_experiment_result.best_expression = "add(u, add(u_x, u_xx))"
        fig, ax = plt.subplots()
        plot_equation(mock_experiment_result, ax)
        texts = [t.get_text() for t in ax.texts]
        assert len(texts) >= 1
        rendered = texts[0]

        assert rendered.strip().startswith("$")
        assert rendered.strip().endswith("$")

        inner = rendered.strip("$ ")
        assert "add(" not in inner, f"Raw IR 'add(' found in rendered text: {rendered}"
        assert "mul(" not in inner, f"Raw IR 'mul(' found in rendered text: {rendered}"
        plt.close(fig)





