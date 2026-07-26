
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import dataclasses
import logging

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots.comparison import (
    plot_score_bar,
    plot_summary_table,
    render_overlaid_convergence,
)


_RAW_IR_MARKERS = ("add(", "mul(", "diff")


def _make_result(
    name: str,
    *,
    r2: float = 0.9,
    nmse: float = 0.05,
    terms: list[str] | None = None,
    coefficients: torch.Tensor | None = None,
    selected_indices: list[int] | None = None,
    best_expression: str = "add(u, add(u_x, u_xx))",
    lhs_label: str = "u_t",
) -> ExperimentResult:
    n_samples = 20
    recorder = VizRecorder()
    for i in range(5):
        recorder.log("_best_score", 1.0 / (i + 1))
        recorder.log("_best_expr", f"expr_{i}")

    resolved_terms = ["u", "u_x", "u_xx"] if terms is None else terms
    resolved_coefficients = (
        torch.tensor([1.0, -0.5, 0.3]) if coefficients is None else coefficients
    )
    return ExperimentResult(
        best_expression=best_expression,
        best_score=nmse,
        iterations=5,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=nmse * 2,
            nmse=nmse,
            r2=r2,
            score=-50.0,
            complexity=len(resolved_terms) if resolved_terms else 0,
            coefficients=resolved_coefficients,
            is_valid=True,
            error_message="",
            selected_indices=selected_indices,
            residuals=torch.randn(n_samples) * 0.1,
            terms=resolved_terms if resolved_terms else None,
            expression=best_expression,
        ),
        actual=torch.randn(n_samples),
        predicted=torch.randn(n_samples),
        dataset_name="test",
        algorithm_name=name,
        config={"max_iter": 5},
        recorder=recorder,
        lhs_label=lhs_label,
    )


def _bar_geometry(ax: Axes) -> list[tuple[float, float]]:
    return [
        (patch.get_x() + patch.get_width() / 2.0, patch.get_height())
        for patch in ax.patches
        if isinstance(patch, Rectangle)
    ]


def _annotation_axes_fraction(ax: Axes, text: Text) -> tuple[float, float]:
    ax.figure.canvas.draw()
    display = text.get_transform().transform(text.get_position())
    x_fraction, y_fraction = ax.transAxes.inverted().transform(display)
    return float(x_fraction), float(y_fraction)


def _na_annotations(ax: Axes) -> list[Text]:
    return [t for t in ax.texts if t.get_text() == "N/A"]


def _table_cell_texts(ax: Axes) -> list[str]:
    tables = ax.tables
    assert tables, "plot_summary_table must render a table"
    return [cell.get_text().get_text() for cell in tables[0].get_celld().values()]


def _expression_cells(ax: Axes) -> list[str]:
    table = ax.tables[0]
    return [
        cell.get_text().get_text()
        for (row, col), cell in table.get_celld().items()
        if col == 1 and row > 0
    ]


class TestScoreBarLabelCollisions:

    @pytest.mark.unit
    def test_duplicate_labels_keep_one_bar_per_run(self) -> None:
        results = [
            _make_result("SGA", r2=0.9),
            _make_result("SGA", r2=0.5),
            _make_result("DLGA", r2=0.7),
        ]
        fig, ax = plt.subplots()
        try:
            plot_score_bar(results, ax)
            geometry = _bar_geometry(ax)
            assert len(geometry) == 3, geometry
            x_positions = [x for x, _ in geometry]
            assert len(set(x_positions)) == 3, (
                f"each run needs its own x position; got {x_positions}"
            )
            assert [height for _, height in geometry] == [0.9, 0.5, 0.7]
            tick_labels = [t.get_text() for t in ax.get_xticklabels()]
            assert tick_labels == ["SGA #1", "SGA #2", "DLGA"], tick_labels
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_unique_labels_are_left_untouched(self) -> None:
        results = [_make_result("SGA"), _make_result("DLGA")]
        fig, ax = plt.subplots()
        try:
            plot_score_bar(results, ax)
            tick_labels = [t.get_text() for t in ax.get_xticklabels()]
            assert tick_labels == ["SGA", "DLGA"], tick_labels
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_explicit_labels_are_still_disambiguated(self) -> None:
        results = [_make_result("A"), _make_result("B")]
        fig, ax = plt.subplots()
        try:
            plot_score_bar(results, ax, labels=["seed", "seed"])
            tick_labels = [t.get_text() for t in ax.get_xticklabels()]
            assert tick_labels == ["seed #1", "seed #2"], tick_labels
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_caller_label_already_matching_the_ordinal_pattern_stays_unique(
        self,
    ) -> None:
        results = [_make_result("A"), _make_result("B"), _make_result("C")]
        fig, ax = plt.subplots()
        try:
            plot_score_bar(results, ax, labels=["SGA #1", "SGA", "SGA"])
            tick_labels = [t.get_text() for t in ax.get_xticklabels()]
            assert len(set(tick_labels)) == 3, tick_labels

            assert tick_labels[0] == "SGA #1", tick_labels
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_ordinal_collision_with_a_later_label_is_avoided(self) -> None:
        results = [_make_result(name) for name in ("A", "B", "C", "D")]
        fig, ax = plt.subplots()
        try:
            plot_score_bar(results, ax, labels=["A #2", "A", "A", "A"])
            tick_labels = [t.get_text() for t in ax.get_xticklabels()]
            assert len(set(tick_labels)) == 4, tick_labels
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_na_annotation_sits_at_its_own_run_position(self) -> None:
        results = [
            _make_result("SGA", r2=0.9),
            _make_result("SGA", r2=0.4),
            _make_result("SGA", r2=float("nan")),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = plot_score_bar(results, ax)
            na_texts = _na_annotations(ax)
            assert len(na_texts) == 1, [t.get_text() for t in ax.texts]
            annotation_x = na_texts[0].get_position()[0]
            third_bar_x = _bar_geometry(ax)[-1][0]
            assert annotation_x == pytest.approx(third_bar_x), (
                f"N/A must sit on run 3's x position {third_bar_x}, got {annotation_x}"
            )
            x_min, x_max = ax.get_xlim()
            assert x_min <= annotation_x <= x_max
            x_fraction, y_fraction = _annotation_axes_fraction(ax, na_texts[0])
            assert 0.0 <= x_fraction <= 1.0, x_fraction
            assert 0.0 <= y_fraction <= 1.0, y_fraction
            assert any("Non-finite" in w for w in warnings), warnings
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_na_annotations_stay_in_view_when_every_run_is_invalid(self) -> None:
        results = [
            _make_result(f"SGA{i}", r2=bad)
            for i, bad in enumerate((float("nan"), float("inf"), float("-inf")))
        ]
        fig, ax = plt.subplots()
        try:
            warnings = plot_score_bar(results, ax)
            na_texts = _na_annotations(ax)
            assert len(na_texts) == 3, [t.get_text() for t in ax.texts]
            for text in na_texts:
                x_fraction, y_fraction = _annotation_axes_fraction(ax, text)
                assert 0.0 <= x_fraction <= 1.0, (text.get_position(), x_fraction)
                assert 0.0 <= y_fraction <= 1.0, (text.get_position(), y_fraction)
            assert len([w for w in warnings if "Non-finite" in w]) == 3, warnings
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_non_finite_warning_names_the_disambiguated_run(self) -> None:
        results = [
            _make_result("SGA", r2=float("nan")),
            _make_result("SGA", r2=0.4),
            _make_result("SGA", r2=float("nan")),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = plot_score_bar(results, ax)
            assert any("SGA #1" in w for w in warnings), warnings
            assert any("SGA #3" in w for w in warnings), warnings
            assert not any("SGA #2" in w for w in warnings), warnings
        finally:
            plt.close(fig)


class TestSharedLabelConvention:

    @pytest.mark.unit
    def test_legend_entries_are_distinct_for_duplicate_labels(self) -> None:
        results = [_make_result("SGA"), _make_result("SGA")]
        fig, ax = plt.subplots()
        try:
            render_overlaid_convergence(results, ax)
            run_labels = [
                line.get_label() for line in ax.lines if line.get_label() != "mean"
            ]
            assert run_labels == ["SGA #1", "SGA #2"], run_labels
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_summary_table_uses_the_same_labels(self) -> None:
        results = [_make_result("SGA"), _make_result("SGA")]
        fig, ax = plt.subplots()
        try:
            plot_summary_table(results, ax)
            texts = _table_cell_texts(ax)
            assert "SGA #1" in texts, texts
            assert "SGA #2" in texts, texts
        finally:
            plt.close(fig)


class TestSummaryTableExpressionColumn:

    @pytest.mark.unit
    def test_structured_final_eval_renders_the_fitted_equation(self) -> None:
        from kd.core.expr.sympy_bridge import format_pde

        results = [
            _make_result(
                "SGA",
                terms=["u", "u_x", "u_xx"],
                coefficients=torch.tensor([1.0, -0.5, 0.3]),
                selected_indices=[0, 1, 2],
            )
        ]
        expected = format_pde(
            ["u", "u_x", "u_xx"],
            torch.tensor([1.0, -0.5, 0.3]),
            lhs="u_t",
            selected_indices=[0, 1, 2],
        ).unicode
        fig, ax = plt.subplots()
        try:
            warnings = plot_summary_table(results, ax)
            cells = _expression_cells(ax)
            assert cells == [expected], (cells, expected)
            assert not any(marker in cells[0] for marker in _RAW_IR_MARKERS), cells
            assert warnings == []
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_selected_indices_prune_the_rendered_equation(self) -> None:
        results = [
            _make_result(
                "SGA",
                terms=["u", "u_x", "u_xx"],
                coefficients=torch.tensor([1.0, -0.5, 0.3]),
                selected_indices=[0],
            )
        ]
        fig, ax = plt.subplots()
        try:
            plot_summary_table(results, ax)
            cell = _expression_cells(ax)[0]
            assert "0.3" not in cell, cell
            assert "0.5" not in cell, cell
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_missing_terms_fall_back_without_exposing_raw_ir(self) -> None:
        results = [
            _make_result(
                "SGA",
                terms=[],
                coefficients=None,
                best_expression="add(u, add(u_x, u_xx))",
            )
        ]
        fig, ax = plt.subplots()
        try:
            plot_summary_table(results, ax)
            cell = _expression_cells(ax)[0]
            assert not any(marker in cell for marker in _RAW_IR_MARKERS), cell
            assert "u" in cell
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_structure_fallback_is_marked_and_warned(self) -> None:
        healthy = _make_result("SGA")
        failed = _make_result("SGA")
        degraded = dataclasses.replace(
            failed,
            final_eval=dataclasses.replace(failed.final_eval, coefficients=None),
        )
        fig, ax = plt.subplots()
        try:
            warnings = plot_summary_table([healthy, degraded], ax)
            fitted_cell, degraded_cell = _expression_cells(ax)
            assert "(structure only)" not in fitted_cell, fitted_cell
            assert "(structure only)" in degraded_cell, degraded_cell
            assert len(degraded_cell) <= 48, degraded_cell
            assert not any(marker in degraded_cell for marker in _RAW_IR_MARKERS)
            assert any("structure" in w for w in warnings), warnings
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_unrenderable_expression_keeps_raw_ir_out_of_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        results = [
            _make_result(
                "SGA",
                terms=[],
                coefficients=None,
                best_expression="add(u,",
            )
        ]
        fig, ax = plt.subplots()
        try:
            with caplog.at_level(logging.WARNING):
                warnings = plot_summary_table(results, ax)
            cell = _expression_cells(ax)[0]
            assert not any(marker in cell for marker in _RAW_IR_MARKERS), cell
            assert len(cell) <= 20, cell
            assert any("not renderable" in w for w in warnings), warnings
            for warning in warnings:
                assert "add(u," not in warning, warning
            assert "add(u," in caplog.text
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_out_of_range_selected_indices_degrade_one_cell_not_the_table(
        self,
    ) -> None:
        results = [
            _make_result("SGA", r2=0.9),
            _make_result(
                "SGA",
                terms=["u", "u_x"],
                coefficients=torch.tensor([1.0, -0.5]),
                selected_indices=[0, 5],
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = plot_summary_table(results, ax)
            cells = _expression_cells(ax)
            assert len(cells) == 2, cells
            assert "(structure only)" in cells[1], cells
            assert not any(marker in cells[1] for marker in _RAW_IR_MARKERS), cells
            assert any("structure" in w for w in warnings), warnings
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_long_equation_is_truncated_with_full_text_in_warnings(self) -> None:
        terms = [f"u_{'x' * (i + 1)}" for i in range(8)]
        results = [
            _make_result(
                "SGA",
                terms=terms,
                coefficients=torch.tensor([0.123456 * (i + 1) for i in range(8)]),
                selected_indices=list(range(8)),
            )
        ]
        fig, ax = plt.subplots()
        try:
            warnings = plot_summary_table(results, ax)
            cell = _expression_cells(ax)[0]
            assert cell.endswith("..."), cell
            full = [w for w in warnings if "full text:" in w]
            assert len(full) == 1, warnings
            assert cell.rstrip(".") in full[0], (cell, full)
        finally:
            plt.close(fig)
