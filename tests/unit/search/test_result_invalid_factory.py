
from __future__ import annotations

import inspect
import math

import pytest

from kd.core.evaluator import EvaluationResult

pytestmark = pytest.mark.unit


def _assert_same_invalid_fields(
    actual: EvaluationResult, expected: EvaluationResult
) -> None:
    assert actual.mse == expected.mse
    assert actual.nmse == expected.nmse
    assert actual.r2 == expected.r2
    assert actual.score == expected.score
    assert actual.complexity == expected.complexity
    assert actual.coefficients is None and expected.coefficients is None
    assert actual.is_valid == expected.is_valid
    assert actual.error_message == expected.error_message
    assert actual.selected_indices == expected.selected_indices
    assert actual.residuals is None and expected.residuals is None
    assert actual.terms == expected.terms
    assert actual.expression == expected.expression







def test_factory_is_importable_from_kd_search_result() -> None:
    from kd.search.result import invalid_evaluation_result

    assert callable(invalid_evaluation_result)


def test_score_is_required_keyword_only() -> None:
    from kd.search.result import invalid_evaluation_result

    with pytest.raises(TypeError):
        invalid_evaluation_result("boom")


def test_score_may_not_be_passed_positionally() -> None:
    from kd.search.result import invalid_evaluation_result

    with pytest.raises(TypeError):
        invalid_evaluation_result("boom", 1.5)







def test_residuals_always_none() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=float("inf"))
    assert result.residuals is None


def test_is_valid_is_false() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=float("inf"))
    assert result.is_valid is False


def test_error_metrics_are_worst_case() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=float("inf"))
    assert result.mse == math.inf
    assert result.nmse == math.inf
    assert result.r2 == -math.inf


def test_coefficients_is_none() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=float("inf"))
    assert result.coefficients is None


def test_selected_indices_is_none() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=float("inf"))
    assert result.selected_indices is None


def test_selected_indices_is_none_on_every_call() -> None:
    from kd.search.result import invalid_evaluation_result

    first = invalid_evaluation_result("a", score=float("inf"))
    second = invalid_evaluation_result("b", score=float("inf"))
    assert first.selected_indices is None
    assert second.selected_indices is None







def test_score_lands_in_score_field() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=-3.5)
    assert result.score == -3.5


def test_score_none_is_passed_through() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=None)
    assert result.score is None


def test_score_inf_is_passed_through() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=float("inf"))
    assert result.score == math.inf







def test_error_message_pass_through() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("empty candidate pool", score=None)
    assert result.error_message == "empty candidate pool"


def test_expression_defaults_to_empty_string() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=None)
    assert result.expression == ""


def test_expression_pass_through() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result(
        "boom", score=None, expression="u_t = 1.0*u_xx"
    )
    assert result.expression == "u_t = 1.0*u_xx"







def test_complexity_is_zero_when_terms_absent() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=None)
    assert result.complexity == 0


def test_terms_default_is_none_when_absent() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=None)
    assert result.terms is None


def test_terms_empty_list_is_normalized_to_none() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=None, terms=[])
    assert result.terms is None
    assert result.complexity == 0


def test_complexity_equals_len_terms_when_given() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result(
        "boom", score=None, terms=["u", "u_x", "u_xx"]
    )
    assert result.complexity == 3


def test_terms_are_defensively_copied() -> None:
    from kd.search.result import invalid_evaluation_result

    caller_terms = ["u", "u_x"]
    result = invalid_evaluation_result("boom", score=None, terms=caller_terms)
    caller_terms.append("u_xx")
    assert result.terms == ["u", "u_x"]
    assert result.complexity == 2


def test_terms_do_not_leak_between_calls() -> None:
    from kd.search.result import invalid_evaluation_result

    first = invalid_evaluation_result("a", score=None, terms=["u"])
    second = invalid_evaluation_result("b", score=None, terms=["u"])
    assert first.terms is not None
    first.terms.append("leak")
    assert second.terms == ["u"]








def test_sga_invalid_final_result_delegates_to_factory() -> None:
    from kd.search.sga.config import SGAConfig
    from kd.search.sga.plugin import SGAPlugin

    plugin = SGAPlugin(config=SGAConfig())
    result = plugin._invalid_final_result("no population")

    assert result.is_valid is False
    assert result.residuals is None
    assert result.score == math.inf
    assert result.error_message == "no population"


def test_dlga_invalid_result_delegates_to_factory() -> None:
    from kd.search.dlga.plugin import DLGAPlugin

    result = DLGAPlugin._invalid_result("u_t", "surrogate failed")

    assert result.is_valid is False
    assert result.residuals is None
    assert result.score == math.inf

    assert result.selected_indices is None
    assert result.expression == "u_t"
    assert result.error_message == "surrogate failed"


def test_eqgpt_invalid_final_result_delegates_to_factory() -> None:
    from kd.search.eqgpt import _scoring

    result = _scoring.invalid_final_result("no candidates", best_reward=0.42)

    assert result.is_valid is False
    assert result.selected_indices is None
    assert result.score == 0.42
    assert result.residuals is None


def test_eqgpt_invalid_final_result_terms_populate_complexity() -> None:
    from kd.search.eqgpt import _scoring

    result = _scoring.invalid_final_result(
        "malformed best", best_reward=-1.0, terms=["u", "u_x"]
    )
    assert result.terms == ["u", "u_x"]
    assert result.complexity == 2










def test_runner_invalid_final_eval_matches_factory() -> None:
    from kd.search.result import invalid_evaluation_result
    from kd.search.runner import ExperimentRunner
    from tests.unit.search._runner_mocks import RecordingAlgorithm

    runner = ExperimentRunner(algorithm=RecordingAlgorithm())
    best_expr = runner._algorithm.best_expression

    actual = runner._invalid_final_eval("boom")
    expected = invalid_evaluation_result(
        "boom", score=float("inf"), expression=best_expr
    )
    _assert_same_invalid_fields(actual, expected)


def test_item6_invalid_final_eval_reason_is_unclassified() -> None:



    from kd.search.runner import ExperimentRunner
    from tests.unit.search._runner_mocks import RecordingAlgorithm

    runner = ExperimentRunner(algorithm=RecordingAlgorithm())
    assert runner._invalid_final_eval("boom").invalid_reason == "unclassified"


def test_sga_invalid_result_matches_factory() -> None:
    from kd.search.result import invalid_evaluation_result
    from kd.search.sga.config import SGAConfig
    from kd.search.sga.plugin import SGAPlugin

    plugin = SGAPlugin(config=SGAConfig())
    actual = plugin._invalid_result("u_t = c*u_xx")
    expected = invalid_evaluation_result(
        "No corresponding PDE for evaluation",
        score=float("inf"),
        expression="u_t = c*u_xx",
    )
    _assert_same_invalid_fields(actual, expected)


def test_sga_evaluation_failed_result_overrides_only_error_message() -> None:
    from kd.search.result import invalid_evaluation_result
    from kd.search.sga.config import SGAConfig
    from kd.search.sga.plugin import SGAPlugin

    plugin = SGAPlugin(config=SGAConfig())
    actual = plugin._evaluation_failed_result("u_t = c*u_xx")


    expected = invalid_evaluation_result(
        actual.error_message,
        score=float("inf"),
        expression="u_t = c*u_xx",
    )
    _assert_same_invalid_fields(actual, expected)

    assert actual.error_message != "No corresponding PDE for evaluation"









def test_eqgpt_invalid_final_result_signature_drops_target() -> None:
    from kd.search.eqgpt import _scoring

    params = inspect.signature(_scoring.invalid_final_result).parameters
    assert "target" not in params
    assert "best_reward" in params
    assert "message" in params


def test_eqgpt_invalid_final_result_rejects_third_positional() -> None:
    from kd.search.eqgpt import _scoring

    with pytest.raises(TypeError):
        _scoring.invalid_final_result("boom", 0.0, 0.0)









def test_factory_invalid_result_serializes_terms_indices_as_null() -> None:
    import json

    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=float("inf"))
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["selected_indices"] is None
    assert payload["terms"] is None


def test_eqgpt_invalid_final_result_serializes_terms_indices_as_null() -> None:
    import json

    from kd.search.eqgpt import _scoring

    result = _scoring.invalid_final_result("no candidates", best_reward=0.0)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["terms"] is None
    assert payload["selected_indices"] is None










def test_factory_invalid_terms_truthiness_equivalent_to_empty() -> None:
    from kd.search.result import invalid_evaluation_result

    result = invalid_evaluation_result("boom", score=float("inf"))
    assert list(result.terms or []) == []
    assert list(result.selected_indices or []) == []
