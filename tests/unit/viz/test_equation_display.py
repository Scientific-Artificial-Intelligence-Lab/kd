
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from kd.core.evaluator import EvaluationResult
from kd.search.result import ExperimentResult
from kd.viz.equation_display import (
    STRUCTURE_ONLY_NOTE,
    UNRENDERABLE_MARKER,
    latex_display,
)
from kd.viz.plots.equation import plot_equation
from kd.viz.report import generate_report

pytestmark = pytest.mark.unit




_UNPARSEABLE = "add(u,"
_UNKNOWN_OP = "weird_op(u, u_x)"

_IR_FRAGMENTS = ("add(", "mul(", "weird_op(", "diff(")


def _without_structured_eval(result: ExperimentResult) -> ExperimentResult:
    return replace(
        result,
        final_eval=replace(result.final_eval, terms=None, coefficients=None),
    )


def _report_body(result: ExperimentResult, tmp_path: Path) -> str:
    output = tmp_path / "report.html"
    generate_report(result, [], output)
    return output.read_text(encoding="utf-8").split('<pre class="json-summary">')[0]


class TestDegradationLadder:

    def test_structured_final_eval_renders_the_fitted_equation(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        display = latex_display(mock_experiment_result, label="SGA")

        assert display.is_math
        assert display.note is None
        assert not display.degraded

        assert "=" in display.text

    def test_missing_coefficients_degrade_to_a_marked_structure(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        display = latex_display(
            _without_structured_eval(mock_experiment_result), label="SGA"
        )

        assert display.is_math
        assert display.note == STRUCTURE_ONLY_NOTE
        assert display.degraded

    @pytest.mark.parametrize("expression", [_UNPARSEABLE, _UNKNOWN_OP])
    def test_unrenderable_expression_yields_a_marker_not_math(
        self, mock_experiment_result: ExperimentResult, expression: str
    ) -> None:
        result = replace(
            _without_structured_eval(mock_experiment_result),
            best_expression=expression,
        )

        display = latex_display(result, label="SGA")

        assert not display.is_math
        assert display.text == UNRENDERABLE_MARKER
        assert display.degraded

    @pytest.mark.parametrize("expression", [_UNPARSEABLE, _UNKNOWN_OP])
    def test_no_rung_ever_returns_raw_ir(
        self, mock_experiment_result: ExperimentResult, expression: str
    ) -> None:
        result = replace(
            _without_structured_eval(mock_experiment_result),
            best_expression=expression,
        )

        display = latex_display(result, label="SGA")

        rendered = display.text + (display.note or "")
        assert not any(fragment in rendered for fragment in _IR_FRAGMENTS)


class TestHtmlReportExpressionRow:

    @pytest.mark.parametrize("expression", [_UNPARSEABLE, _UNKNOWN_OP])
    def test_unrenderable_expression_does_not_reach_the_html(
        self,
        mock_experiment_result: ExperimentResult,
        tmp_path: Path,
        expression: str,
    ) -> None:
        result = replace(
            _without_structured_eval(mock_experiment_result),
            best_expression=expression,
        )

        body = _report_body(result, tmp_path)

        assert expression not in body
        assert not any(fragment in body for fragment in _IR_FRAGMENTS)
        assert UNRENDERABLE_MARKER in body

    def test_structure_only_row_is_marked_as_such(
        self, mock_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        body = _report_body(_without_structured_eval(mock_experiment_result), tmp_path)

        assert STRUCTURE_ONLY_NOTE in body

    def test_fitted_row_carries_no_degradation_note(
        self, mock_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        body = _report_body(mock_experiment_result, tmp_path)

        assert STRUCTURE_ONLY_NOTE not in body
        assert UNRENDERABLE_MARKER not in body

        assert "\\(" in body


class TestEquationPanel:

    @pytest.mark.parametrize("expression", [_UNPARSEABLE, _UNKNOWN_OP])
    def test_unrenderable_expression_is_not_drawn_as_raw_ir(
        self, mock_experiment_result: ExperimentResult, expression: str
    ) -> None:
        result = replace(
            _without_structured_eval(mock_experiment_result),
            best_expression=expression,
        )
        fig, ax = plt.subplots()
        try:
            warnings = plot_equation(result, ax)
            drawn = [text.get_text() for text in ax.texts]
            title = ax.get_title()
        finally:
            plt.close(fig)

        assert all(
            not any(fragment in text for fragment in _IR_FRAGMENTS) for text in drawn
        )
        assert UNRENDERABLE_MARKER in " ".join(drawn)

        assert "not renderable" in title
        assert warnings

    def test_structure_only_panel_announces_the_degrade(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        fig, ax = plt.subplots()
        try:
            warnings = plot_equation(
                _without_structured_eval(mock_experiment_result), ax
            )
            title = ax.get_title()
        finally:
            plt.close(fig)

        assert STRUCTURE_ONLY_NOTE in title
        assert any(STRUCTURE_ONLY_NOTE in warning for warning in warnings)

    def test_fitted_panel_is_unchanged(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        fig, ax = plt.subplots()
        try:
            warnings = plot_equation(mock_experiment_result, ax)
            title = ax.get_title()
            drawn = [text.get_text() for text in ax.texts]
        finally:
            plt.close(fig)

        assert title == "Best Expression"
        assert warnings == []
        assert drawn and drawn[0].startswith("$")


def test_evaluation_result_fixture_is_the_fitted_shape(
    mock_evaluation_result: EvaluationResult,
) -> None:
    assert mock_evaluation_result.terms is not None
    assert mock_evaluation_result.coefficients is not None
