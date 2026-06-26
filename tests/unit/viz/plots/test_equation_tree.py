
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import pytest

from kd.core.evaluator import EvaluationResult
from kd.core.expr.sympy_bridge import to_sympy
from kd.search.result import ExperimentResult
from kd.viz.plots.equation_tree import plot_equation_tree, sympy_to_render


def _labels(ax) -> set[str]:
    return {t.get_text() for t in ax.texts}


def _child_labels(node) -> set[str]:
    return {c.label for c in node.children}


def _sole(node, label):
    matches = [c for c in node.children if c.label == label]
    assert len(matches) == 1, f"expected one {label!r} child, got {len(matches)}"
    return matches[0]


class TestSympyToRender:

    def test_symbol_leaf(self) -> None:
        node = sympy_to_render(to_sympy("u"))
        assert node.label == "u"
        assert node.children == ()
        assert node.kind == "var"

    def test_product_of_factors(self) -> None:
        node = sympy_to_render(to_sympy("mul(u, diff_x(u))"))
        assert node.label == "*"
        assert _child_labels(node) == {"u", "u_x"}

    def test_strips_numeric_coefficient(self) -> None:
        node = sympy_to_render(to_sympy("mul(0.5, diff_x(u))"))
        assert node.label == "u_x"
        assert node.children == ()

    def test_unary_minus_stripped(self) -> None:
        node = sympy_to_render(to_sympy("mul(-1.0, u)"))
        assert node.label == "u"

    def test_power_exponent_preserved(self) -> None:
        node = sympy_to_render(to_sympy("n2(u)"))

        assert node.label == "^"
        assert [c.label for c in node.children] == ["u", "2"]

    def test_addition_tree(self) -> None:
        node = sympy_to_render(to_sympy("add(mul(u, diff_x(u)), diff2_x(u))"))
        assert node.label == "+"
        assert _child_labels(node) == {"*", "u_xx"}

    def test_subtraction_is_structural_not_addition(self) -> None:
        minus = sympy_to_render(to_sympy("sub(u, diff_x(u))"))
        plus = sympy_to_render(to_sympy("add(u, diff_x(u))"))
        assert minus.label == plus.label == "+"
        assert "-" in _child_labels(minus)
        assert "-" not in _child_labels(plus)
        neg = next(c for c in minus.children if c.label == "-")
        assert [c.label for c in neg.children] == ["u_x"]

    def test_nested_subtraction_inside_product_preserved(self) -> None:
        sub_inner = _sole(
            sympy_to_render(to_sympy("mul(u, sub(diff_x(u), diff2_x(u)))")), "+"
        )
        add_inner = _sole(
            sympy_to_render(to_sympy("mul(u, add(diff_x(u), diff2_x(u)))")), "+"
        )
        assert "-" in _child_labels(sub_inner)
        assert "-" not in _child_labels(add_inner)
        assert _child_labels(sub_inner) != _child_labels(add_inner)

    def test_nonlinear_derivative_is_deriv_kind(self) -> None:
        node = sympy_to_render(to_sympy("diff_x(sin(u))"))
        assert node.label == "diff_x"
        assert node.kind == "deriv"

    def test_non_finite_number_does_not_crash(self) -> None:
        import sympy

        from kd.viz.plots.equation_tree import _format_number

        for const in (sympy.oo, sympy.zoo, sympy.nan):
            label = _format_number(const)
            assert isinstance(label, str) and label


def _result_with(
    terms: list[str] | None,
    selected_indices: list[int] | None,
    best_expression: str,
) -> ExperimentResult:
    import torch

    from kd.search.recorder import VizRecorder

    final_eval = EvaluationResult(
        mse=0.01,
        nmse=0.005,
        r2=0.95,
        aic=-100.0,
        complexity=len(selected_indices) if selected_indices is not None else 0,
        coefficients=None,
        is_valid=True,
        selected_indices=selected_indices,
        terms=terms,
        expression=best_expression,
    )
    return ExperimentResult(
        best_expression=best_expression,
        best_score=0.3,
        iterations=1,
        early_stopped=False,
        final_eval=final_eval,
        actual=torch.zeros(4),
        predicted=torch.zeros(4),
        dataset_name="t",
        algorithm_name="SGA",
        config={},
        recorder=VizRecorder(),
    )


class TestPlotEquationTree:

    def test_renders_all_selected_terms(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        fig, ax = plt.subplots()
        warnings = plot_equation_tree(mock_experiment_result, ax)
        assert warnings == []


        assert _labels(ax) == {"u", "u_x", "u_xx", "+"}
        assert len(ax.texts) == 4
        plt.close(fig)

    def test_sparse_selection_excludes_unselected_terms(self) -> None:
        result = _result_with(
            terms=["u", "u_x", "u_xx"],
            selected_indices=[0, 2],
            best_expression="add(u, add(u_x, u_xx))",
        )
        fig, ax = plt.subplots()
        plot_equation_tree(result, ax)
        labels = _labels(ax)
        assert "u" in labels
        assert "u_xx" in labels
        assert "u_x" not in labels
        plt.close(fig)

    def test_empty_terms_list_does_not_render_best_expression(self) -> None:
        result = _result_with(
            terms=[],
            selected_indices=[],
            best_expression="add(u, add(u_x, u_xx))",
        )
        fig, ax = plt.subplots()
        warnings = plot_equation_tree(result, ax)
        labels = _labels(ax)
        assert "u_x" not in labels
        assert "u_xx" not in labels
        assert len(warnings) > 0
        plt.close(fig)

    def test_out_of_range_selected_index_warns(self) -> None:
        result = _result_with(
            terms=["u", "u_x"],
            selected_indices=[0, 5],
            best_expression="u",
        )
        fig, ax = plt.subplots()
        warnings = plot_equation_tree(result, ax)
        assert any("out of range" in w for w in warnings)
        assert _labels(ax) == {"u"}
        plt.close(fig)

    def test_single_term_no_virtual_root(self) -> None:
        result = _result_with(
            terms=["u_x"], selected_indices=[0], best_expression="u_x"
        )
        fig, ax = plt.subplots()
        plot_equation_tree(result, ax)

        assert _labels(ax) == {"u_x"}
        plt.close(fig)

    def test_fallback_to_best_expression_when_terms_none(self) -> None:
        result = _result_with(
            terms=None, selected_indices=None, best_expression="add(u, diff_x(u))"
        )
        fig, ax = plt.subplots()
        warnings = plot_equation_tree(result, ax)
        assert warnings == []
        assert {"u", "u_x", "+"} <= _labels(ax)
        plt.close(fig)

    def test_empty_expression_warns(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        empty = dataclasses.replace(
            mock_experiment_result,
            best_expression="",
            final_eval=dataclasses.replace(
                mock_experiment_result.final_eval, terms=None, selected_indices=None
            ),
        )
        fig, ax = plt.subplots()
        warnings = plot_equation_tree(empty, ax)
        assert len(warnings) > 0
        plt.close(fig)


@pytest.mark.smoke
def test_smoke_equation_tree(mock_experiment_result: ExperimentResult) -> None:
    fig, ax = plt.subplots()
    plot_equation_tree(mock_experiment_result, ax)
    plt.close(fig)
