
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots.coefficient import plot_coefficient_bar






def _make_result_with_coefficients(
    coeffs: list[float],
    terms: list[str] | None = None,
    selected_indices: list[int] | None = None,
) -> ExperimentResult:
    n_terms = len(coeffs)
    if terms is None:
        terms = [f"term_{i}" for i in range(n_terms)]
    if selected_indices is None:
        selected_indices = list(range(n_terms))
    n_samples = 50
    actual = torch.randn(n_samples)
    predicted = actual + torch.randn(n_samples) * 0.1
    recorder = VizRecorder()
    recorder.log("_best_score", 1.0)
    recorder.log("_best_expr", "dummy")
    recorder.log("_n_candidates", 10)
    return ExperimentResult(
        best_expression="dummy",
        best_score=1.0,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=0.95,
            score=-100.0,
            complexity=n_terms,
            coefficients=torch.tensor(coeffs),
            is_valid=True,
            error_message="",
            selected_indices=selected_indices,
            residuals=predicted - actual,
            terms=terms,
            expression="dummy",
        ),
        actual=actual,
        predicted=predicted,
        dataset_name="test",
        algorithm_name="SGA",
        config={},
        recorder=recorder,
    )







class TestCoefficientBarSmoke:

    def test_callable_returns_list(self) -> None:
        result = _make_result_with_coefficients([1.0, -0.5, 0.3])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert isinstance(warnings, list)
            assert all(isinstance(w, str) for w in warnings)
        finally:
            plt.close(fig)

    def test_accepts_ground_truth_kwarg(self) -> None:
        result = _make_result_with_coefficients([1.0, -0.5])
        gt = torch.tensor([1.1, -0.6])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax, ground_truth=gt)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)







class TestCoefficientBarHappyPath:

    def test_has_bars_for_each_coefficient(self) -> None:
        coeffs = [1.0, -0.5, 0.3, 0.0]
        result = _make_result_with_coefficients(coeffs)
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)

            assert len(ax.patches) >= len(coeffs)
        finally:
            plt.close(fig)

    def test_bars_include_negative_values(self) -> None:
        coeffs = [2.0, -3.0]
        result = _make_result_with_coefficients(coeffs)
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)

            ymin, ymax = ax.get_ylim()
            assert ymin < 0, "y-axis should extend below zero for negative coefficients"
        finally:
            plt.close(fig)

    def test_term_labels_on_xaxis(self) -> None:
        terms = ["u", "u_x", "u_xx"]
        result = _make_result_with_coefficients([1.0, -0.5, 0.3], terms=terms)
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            tick_labels = [t.get_text() for t in ax.get_xticklabels()]

            found = any(term in label for label in tick_labels for term in terms)
            assert found, f"Expected term labels in {tick_labels}"
        finally:
            plt.close(fig)

    def test_ground_truth_adds_visual_elements(self) -> None:
        coeffs = [1.0, -0.5]
        result = _make_result_with_coefficients(coeffs)

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        try:
            plot_coefficient_bar(result, ax1)
            plot_coefficient_bar(result, ax2, ground_truth=torch.tensor([1.1, -0.6]))


            elems_without = len(ax1.patches) + len(ax1.get_lines())
            elems_with = len(ax2.patches) + len(ax2.get_lines())
            assert elems_with > elems_without, (
                f"Ground truth should add visual elements: {elems_without} vs {elems_with}"
            )
        finally:
            plt.close(fig1)
            plt.close(fig2)

    def test_title_is_set(self) -> None:
        result = _make_result_with_coefficients([1.0])
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            assert ax.get_title() != ""
        finally:
            plt.close(fig)







class TestCoefficientBarEdgeCases:

    def test_none_coefficients_warns(self) -> None:
        result = _make_result_with_coefficients([1.0])
        result.final_eval.coefficients = None
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_none_terms_warns(self) -> None:
        result = _make_result_with_coefficients([1.0, -0.5])
        result.final_eval.terms = None
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)

            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_nan_coefficient_draws_no_bar_and_marks_na(self) -> None:
        import math

        result = _make_result_with_coefficients([1.0, float("nan"), 0.5])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert any("N/A" in w for w in warnings)
            assert math.isnan(ax.patches[1].get_height())
            assert any(t.get_text() == "N/A" for t in ax.texts)
        finally:
            plt.close(fig)

    def test_inf_coefficient_handled(self) -> None:
        import math

        result = _make_result_with_coefficients([1.0, float("inf"), 0.5])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert any("N/A" in w for w in warnings)
            assert math.isnan(ax.patches[1].get_height())
        finally:
            plt.close(fig)

    def test_nan_ground_truth_marks_na_and_warns(self) -> None:
        import math

        result = _make_result_with_coefficients([1.0, -0.5])
        gt = torch.tensor([1.1, float("nan")])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax, ground_truth=gt)
            assert any("ground-truth" in w and "N/A" in w for w in warnings)

            assert math.isnan(ax.patches[3].get_height())
            assert any(t.get_text() == "N/A" for t in ax.texts)
        finally:
            plt.close(fig)

    def test_wide_range_nan_never_labelled_zero(self) -> None:
        result = _make_result_with_coefficients([1000.0, float("nan"), 0.5])
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            texts = [t.get_text() for t in ax.texts]
            assert "0" not in texts
            assert "N/A" in texts
        finally:
            plt.close(fig)

    def test_unparseable_term_label_is_marker_not_raw_ir(self) -> None:
        raw_ir = "add(u,"
        result = _make_result_with_coefficients([1.0, 0.5], terms=["u", raw_ir])
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            labels = [t.get_text() for t in ax.get_xticklabels()]
            assert raw_ir not in labels
            assert "unrenderable" in labels
        finally:
            plt.close(fig)

    def test_empty_coefficients(self) -> None:
        result = _make_result_with_coefficients([])
        result.final_eval.coefficients = torch.tensor([])
        result.final_eval.terms = []
        result.final_eval.selected_indices = []
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_ground_truth_length_mismatch(self) -> None:
        result = _make_result_with_coefficients([1.0, -0.5])
        gt = torch.tensor([1.0])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax, ground_truth=gt)
            assert any(
                "mismatch" in w.lower() or "length" in w.lower() for w in warnings
            )
        finally:
            plt.close(fig)

    def test_ground_truth_wrong_length_skips_overlay(self) -> None:
        result = _make_result_with_coefficients(
            [0.0, -4.3, 0.0, 0.0, 0.0], selected_indices=[1]
        )
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(
                result, ax, ground_truth=torch.tensor([-1.0, 0.1, 0.0])
            )
            assert any("length" in w.lower() for w in warnings)
            assert len(ax.patches) == 1
        finally:
            plt.close(fig)

    def test_ground_truth_full_length_projects_onto_selected(self) -> None:
        result = _make_result_with_coefficients(
            [0.0, -4.3, 0.0], selected_indices=[1]
        )
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(
                result, ax, ground_truth=torch.tensor([7.0, -1.0, 9.0])
            )
            assert warnings == []
            heights = [p.get_height() for p in ax.patches]
            assert len(heights) == 2
            assert heights[1] == pytest.approx(-1.0)
        finally:
            plt.close(fig)

    def test_single_coefficient(self) -> None:
        result = _make_result_with_coefficients([42.0])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert len(ax.patches) >= 1
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        result = _make_result_with_coefficients([1.0, -0.5])
        figs_before = len(plt.get_fignums())
        fig, ax = plt.subplots()
        plot_coefficient_bar(result, ax)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before







class TestCoefficientBarWideRange:

    _WIDE = [0.8240275, 1.595041e-04, -2.611047e-05]
    _WIDE_TERMS = ["u_x", "u_xxx", "diff2_x(mul(u, u_x))"]

    def test_wide_range_switches_to_symlog(self) -> None:
        result = _make_result_with_coefficients(self._WIDE, terms=self._WIDE_TERMS)
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            assert ax.get_yscale() == "symlog"
            assert "symlog" in ax.get_ylabel().lower()
        finally:
            plt.close(fig)

    def test_wide_range_labels_every_bar_with_its_value(self) -> None:
        result = _make_result_with_coefficients(self._WIDE, terms=self._WIDE_TERMS)
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            texts = " ".join(t.get_text() for t in ax.texts)

            assert "0.00016" in texts
            assert "2.61e-05" in texts
        finally:
            plt.close(fig)

    def test_narrow_range_stays_linear_and_unlabelled(self) -> None:
        result = _make_result_with_coefficients([2.0, -3.0])
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            assert ax.get_yscale() == "linear"
            assert ax.get_ylabel() == "Coefficient"
            assert len(ax.texts) == 0
            assert ax.get_ylim()[0] < 0
        finally:
            plt.close(fig)

    def test_single_dominant_term_stays_linear(self) -> None:
        result = _make_result_with_coefficients([5.0, 0.0, 0.0])
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            assert ax.get_yscale() == "linear"
        finally:
            plt.close(fig)
