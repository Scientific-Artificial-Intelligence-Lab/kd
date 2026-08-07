
from __future__ import annotations

import json
import math
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.core.linear_solve.base import SolveResult
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.data import generate_advection_data

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset






_WAVE_SPEED = 1.7

_WAVE_NUMBER = 1.0


_GRID_NX = 64
_GRID_NT = 48

_REL_TOL = 0.15

_NMSE_CEILING = 1e-2





_SOLVER_SEAM = LeastSquaresSolver


def _build_advection_dataset() -> PDEDataset:
    return generate_advection_data(
        speeds=(_WAVE_SPEED,),
        waves=(_WAVE_NUMBER,),
        grid_sizes=(_GRID_NX,),
        nt=_GRID_NT,
        seed=0,
    )


def _evaluate_api() -> ModuleType:
    import kd.evaluate as evaluate_module

    return evaluate_module


def _within_rel(actual: float, expected: float, rel: float = _REL_TOL) -> bool:
    return abs(actual - expected) <= rel * abs(expected)


def _make_invalid_solve(theta: torch.Tensor) -> SolveResult:
    return SolveResult(
        coefficients=torch.zeros(theta.shape[1], dtype=theta.dtype),
        residual=float("nan"),
        r2=float("nan"),
        condition_number=float("inf"),
        selected_indices=None,
        is_valid=False,
        error_message="forced solver failure (test seam)",
    )







@pytest.mark.smoke
def test_a_happy_path_recovers_advection_coefficient() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    result = api.evaluate_terms(ds, ["diff_x(u)"])

    assert result.is_valid is True
    assert result.error_message == ""
    assert result.coefficients is not None
    assert result.coefficients.numel() == 1
    coef = float(result.coefficients.flatten()[0])

    assert coef < 0.0
    assert _within_rel(coef, -_WAVE_SPEED)
    assert result.nmse < _NMSE_CEILING







def test_b_strict_reports_all_bad_terms_and_fits_nothing() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    good = "diff_x(u)"
    bad_parse = "nonexistent_op(u)"




    bad_zero = "mul(u, sub(u, u))"

    with pytest.raises(api.InvalidTermsError) as exc_info:
        api.evaluate_terms(ds, [good, bad_parse, bad_zero])

    err = exc_info.value
    rejected = err.rejected
    rejected_terms = {rej.term for rej in rejected}

    assert bad_parse in rejected_terms
    assert bad_zero in rejected_terms

    assert good not in rejected_terms

    reason_by_term = {rej.term: rej.reason for rej in rejected}

    assert all(isinstance(r, str) and r for r in reason_by_term.values())

    assert reason_by_term[bad_parse] != reason_by_term[bad_zero]


    assert "nonexistent_op" in reason_by_term[bad_parse]







def test_c_lenient_fits_survivors_with_single_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    good = "diff_x(u)"

    bad_zero = "mul(u, sub(u, u))"

    with caplog.at_level("WARNING", logger="kd.evaluate"):
        result = api.evaluate_terms(ds, [good, bad_zero], skip_invalid=True)

    assert result.is_valid is True

    assert result.terms == [good]
    assert result.coefficients is not None
    assert result.coefficients.numel() == 1

    warning_records = [rec for rec in caplog.records if rec.levelname == "WARNING"]
    assert len(warning_records) == 1
    warning_text = warning_records[0].getMessage()

    assert bad_zero in warning_text
    assert "zero" in warning_text.lower()


def test_c_lenient_two_drops_emit_single_combined_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    good = "diff_x(u)"
    bad_parse = "nonexistent_op(u)"
    bad_zero = "mul(u, sub(u, u))"

    with caplog.at_level("WARNING", logger="kd.evaluate"):
        result = api.evaluate_terms(ds, [good, bad_parse, bad_zero], skip_invalid=True)

    assert result.is_valid is True
    assert result.terms == [good]

    warning_records = [rec for rec in caplog.records if rec.levelname == "WARNING"]

    assert len(warning_records) == 1
    warning_text = warning_records[0].getMessage()

    assert bad_parse in warning_text
    assert bad_zero in warning_text

    assert "nonexistent_op" in warning_text
    assert "zero" in warning_text.lower()







def test_d_all_bad_lenient_still_raises() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    with pytest.raises(api.InvalidTermsError) as exc_info:
        api.evaluate_terms(ds, ["nonexistent_op(u)", "sub(u, u)"], skip_invalid=True)

    rejected_terms = {rej.term for rej in exc_info.value.rejected}
    assert rejected_terms == {"nonexistent_op(u)", "sub(u, u)"}







def test_e_sentinel_never_returned_on_solver_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    def fake_solve(
        self: LeastSquaresSolver, theta: torch.Tensor, y: torch.Tensor
    ) -> SolveResult:
        return _make_invalid_solve(theta)

    monkeypatch.setattr(_SOLVER_SEAM, "solve", fake_solve)

    with pytest.raises(api.EvaluationFailedError) as exc_info:
        returned = api.evaluate_terms(ds, ["diff_x(u)"])


        assert returned.is_valid is True
        assert returned.mse != 1e10
        pytest.fail("evaluate_terms returned instead of raising on solver failure")


    assert "forced solver failure" in str(exc_info.value)


def test_e_no_sentinel_object_escapes(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    def fake_solve(
        self: LeastSquaresSolver, theta: torch.Tensor, y: torch.Tensor
    ) -> SolveResult:
        return _make_invalid_solve(theta)

    monkeypatch.setattr(_SOLVER_SEAM, "solve", fake_solve)

    returned: EvaluationResult | None = None
    raised = False
    try:
        returned = api.evaluate_terms(ds, ["diff_x(u)"])
    except api.EvaluationFailedError:
        raised = True



    if not raised:
        assert returned is not None
        assert returned.is_valid is True
        assert returned.mse != 1e10







def test_f_validate_terms_partitions_and_is_json_dumpable() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    good = "diff_x(u)"
    bad_parse = "nonexistent_op(u)"
    bad_zero = "mul(u, sub(u, u))"

    report = api.validate_terms(ds, [good, bad_parse, bad_zero])


    assert report.ok is False
    assert report.valid == [good]
    rejected_terms = {rej.term for rej in report.rejected}
    assert rejected_terms == {bad_parse, bad_zero}


    reason_by_term = {rej.term: rej.reason for rej in report.rejected}
    assert reason_by_term[bad_parse] != reason_by_term[bad_zero]
    assert "nonexistent_op" in reason_by_term[bad_parse]


    payload = report.to_dict()
    dumped = json.dumps(payload)
    assert isinstance(dumped, str)
    round_tripped = json.loads(dumped)
    assert round_tripped == payload


def test_f_validate_terms_all_good_reports_ok() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    report = api.validate_terms(ds, ["diff_x(u)", "mul(u, diff_x(u))"])

    assert report.ok is True
    assert report.rejected == []
    assert report.valid == ["diff_x(u)", "mul(u, diff_x(u))"]







def test_g_max_order_passthrough_changes_rejection_set() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()


    with pytest.raises(api.InvalidTermsError) as exc_info:
        api.evaluate_terms(ds, ["u_xx"], max_order=1)
    rejected_terms = {rej.term for rej in exc_info.value.rejected}
    assert "u_xx" in rejected_terms


    result = api.evaluate_terms(ds, ["u_xx"], max_order=2)
    assert result.is_valid is True
    assert result.coefficients is not None


def test_g_lhs_order_two_targets_u_tt() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    result = api.evaluate_terms(ds, ["diff2_x(u)"], lhs_order=2)

    assert result.is_valid is True
    assert result.coefficients is not None
    assert result.coefficients.numel() == 1
    coef = float(result.coefficients.flatten()[0])
    expected = _WAVE_SPEED**2

    assert coef > 0.0
    assert _within_rel(coef, expected)
    assert result.nmse < _NMSE_CEILING







def test_g2_scalar_term_classified_with_shape_reason() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    report = api.validate_terms(ds, ["1"])

    assert report.ok is False
    rejected_terms = {rej.term for rej in report.rejected}
    assert rejected_terms == {"1"}
    reason = report.rejected[0].reason

    assert "shape" in reason.lower()


def test_g2_strict_evaluate_raises_on_scalar_term() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    with pytest.raises(api.InvalidTermsError) as exc_info:
        api.evaluate_terms(ds, ["1", "diff_x(u)"])

    rejected_terms = {rej.term for rej in exc_info.value.rejected}
    assert "1" in rejected_terms
    assert "diff_x(u)" not in rejected_terms


def test_g2_lenient_drops_scalar_and_fits_survivor() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    result = api.evaluate_terms(ds, ["1", "diff_x(u)"], skip_invalid=True)

    assert result.is_valid is True
    assert result.terms == ["diff_x(u)"]
    assert result.coefficients is not None
    assert result.coefficients.numel() == 1







def test_g3_lhs_tautology_rejected_under_default_order() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    report = api.validate_terms(ds, ["diff_t(u)"])
    assert report.ok is False
    assert {rej.term for rej in report.rejected} == {"diff_t(u)"}
    reason = report.rejected[0].reason
    assert "tautolog" in reason.lower()

    with pytest.raises(api.InvalidTermsError) as exc_info:
        api.evaluate_terms(ds, ["diff_t(u)"])
    assert "diff_t(u)" in {rej.term for rej in exc_info.value.rejected}


def test_g3_lhs_tautology_relative_to_lhs_order() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    report = api.validate_terms(ds, ["diff_t(u)"], lhs_order=2)
    assert report.ok is True
    assert report.valid == ["diff_t(u)"]

    result = api.evaluate_terms(ds, ["diff_t(u)"], lhs_order=2)
    assert result.is_valid is True
    assert result.coefficients is not None


def test_g3_non_tautological_term_never_rejected_by_guard() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    report = api.validate_terms(ds, ["diff_x(u)"])
    assert report.ok is True
    assert report.valid == ["diff_x(u)"]


    result = api.evaluate_terms(ds, ["diff_x(u)"])
    assert result.is_valid is True







def test_g4_infix_term_rejected_with_syntax_reason() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    infix = "u + u_xx"

    report = api.validate_terms(ds, [infix])
    assert report.ok is False
    assert {rej.term for rej in report.rejected} == {infix}
    assert "syntax" in report.rejected[0].reason.lower()

    with pytest.raises(api.InvalidTermsError) as exc_info:
        api.evaluate_terms(ds, [infix])
    assert infix in {rej.term for rej in exc_info.value.rejected}


def test_g4_composite_term_rejected_with_split_suggestion() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    composite = "add(u, u_xx)"

    report = api.validate_terms(ds, [composite])
    assert report.ok is False
    reason = report.rejected[0].reason
    assert "composite" in reason.lower()

    assert "u_xx" in reason
    assert "'u'" in reason


def test_g4_canonical_single_terms_pass_syntax_layer() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    report = api.validate_terms(ds, ["mul(u, diff_x(u))", "u_xx"])

    assert report.ok is True
    assert report.valid == ["mul(u, diff_x(u))", "u_xx"]







def test_g5_terminal_order_over_max_gets_retry_hint() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    report = api.validate_terms(ds, ["u_xxx"])

    assert report.ok is False
    reason = report.rejected[0].reason
    assert "max_order=3" in reason

    bumped = api.validate_terms(ds, ["u_xxx"], max_order=3)
    assert bumped.ok is True


def test_g5_unknown_symbol_gets_no_derivative_hint() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    report = api.validate_terms(ds, ["v"])

    assert report.ok is False
    reason = report.rejected[0].reason
    assert "execution" in reason.lower()

    assert "max_order" not in reason
    assert "terminal derivative" not in reason


def test_g5_terminal_order_over_fd_cap_uses_cap_wording() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    report = api.validate_terms(ds, ["u_xxxx"])

    assert report.ok is False
    reason = report.rejected[0].reason
    assert "max order 3" in reason.lower()

    assert "retry with max_order" not in reason







def test_g6_oom_during_term_execution_releases_and_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kd._evaluate_classify as classify_module
    from kd.core.expr.executor import PythonExecutor

    api = _evaluate_api()
    ds = _build_advection_dataset()

    release_calls = {"count": 0}

    def fake_release() -> None:
        release_calls["count"] += 1


    monkeypatch.setattr(classify_module, "release_cuda_memory", fake_release)

    real_execute = PythonExecutor.execute

    def oom_execute(self: PythonExecutor, code: str, context: Any, **kwargs: Any):
        if code == "u_xx":
            raise torch.cuda.OutOfMemoryError("forced OOM (test seam)")
        return real_execute(self, code, context, **kwargs)

    monkeypatch.setattr(PythonExecutor, "execute", oom_execute)

    report = api.validate_terms(ds, ["u_xx"])

    assert report.ok is False
    reason = report.rejected[0].reason
    assert "cuda out of memory" in reason.lower()

    assert release_calls["count"] == 1







def test_h_stateless_identical_calls_and_no_mutation() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()


    lhs_field_before = ds.lhs_field
    lhs_axis_before = ds.lhs_axis

    terms = ["diff_x(u)", "mul(u, diff_x(u))"]
    first = api.evaluate_terms(ds, terms)
    second = api.evaluate_terms(ds, terms)

    assert first.coefficients is not None
    assert second.coefficients is not None
    first_coefs = first.coefficients.flatten().tolist()
    second_coefs = second.coefficients.flatten().tolist()
    assert len(first_coefs) == len(second_coefs)






    for a, b in zip(first_coefs, second_coefs, strict=True):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


    assert ds.lhs_field == lhs_field_before
    assert ds.lhs_axis == lhs_axis_before


def test_h_user_dataset_lhs_field_unset_stays_unset() -> None:
    api = _evaluate_api()
    base = _build_advection_dataset()



    import dataclasses

    ds = dataclasses.replace(base, lhs_field="", lhs_axis="")
    assert ds.lhs_field == ""
    assert ds.lhs_axis == ""

    result = api.evaluate_terms(ds, ["diff_x(u)"])
    assert result.is_valid is True


    assert ds.lhs_field == ""
    assert ds.lhs_axis == ""







def _valid_result_with_tensors() -> EvaluationResult:
    return EvaluationResult(
        mse=0.0123,
        nmse=0.0456,
        r2=0.987,
        score=-12.5,
        complexity=2,
        coefficients=torch.tensor([-1.7, 0.5], dtype=torch.float64),
        is_valid=True,
        error_message="",
        selected_indices=[0, 1],
        residuals=torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64),
        terms=["diff_x(u)", "mul(u, diff_x(u))"],
        expression="add(mul(c0, diff_x(u)), mul(c1, mul(u, diff_x(u))))",
        lhs_name="u_t",
    )


def _invalid_result_with_neg_inf() -> EvaluationResult:
    return EvaluationResult(
        mse=1e10,
        nmse=1e10,
        r2=-float("inf"),
        score=float("inf"),
        complexity=0,
        coefficients=None,
        is_valid=False,
        error_message="Solver returned invalid result",
        selected_indices=None,
        residuals=None,
        terms=None,
        expression="",
        lhs_name=None,
    )


def _result_with_none_score() -> EvaluationResult:
    return EvaluationResult(
        mse=0.5,
        nmse=0.6,
        r2=0.4,
        score=None,
        complexity=1,
        coefficients=torch.tensor([2.0], dtype=torch.float64),
        is_valid=True,
        error_message="",
        selected_indices=None,
        residuals=torch.tensor([0.01, 0.02], dtype=torch.float64),
        terms=["u_xx"],
        expression="",
        lhs_name=None,
    )


def _result_with_zero_score() -> EvaluationResult:
    return EvaluationResult(
        mse=0.3,
        nmse=0.3,
        r2=0.7,
        score=0.0,
        complexity=1,
        coefficients=torch.tensor([1.5], dtype=torch.float64),
        is_valid=True,
        error_message="",
        selected_indices=None,
        residuals=torch.tensor([0.0, 0.0], dtype=torch.float64),
        terms=["diff_x(u)"],
        expression="",
        lhs_name=None,
    )









_EVALUATION_RESULT_KEYS = frozenset(
    {
        "mse",
        "nmse",
        "r2",
        "score",
        "complexity",
        "coefficients",
        "is_valid",
        "error_message",
        "invalid_reason",
        "selected_indices",
        "residuals",
        "terms",
        "expression",
        "lhs_name",
        "condition_number",
        "condition_number_computed",
    }
)






_GOLDEN_TO_DICT: list[tuple[Any, dict[str, Any]]] = [
    (
        _valid_result_with_tensors,
        {
            "mse": 0.0123,
            "nmse": 0.0456,
            "r2": 0.987,
            "score": -12.5,
            "complexity": 2,
            "coefficients": [-1.7, 0.5],
            "is_valid": True,
            "error_message": "",
            "invalid_reason": None,
            "selected_indices": [0, 1],
            "residuals": [0.1, -0.2, 0.05],
            "terms": ["diff_x(u)", "mul(u, diff_x(u))"],
            "expression": "add(mul(c0, diff_x(u)), mul(c1, mul(u, diff_x(u))))",
            "lhs_name": "u_t",
            "condition_number": None,
            "condition_number_computed": False,
        },
    ),
    (
        _invalid_result_with_neg_inf,
        {
            "mse": 1e10,
            "nmse": 1e10,
            "r2": None,
            "score": None,
            "complexity": 0,
            "coefficients": None,
            "is_valid": False,
            "error_message": "Solver returned invalid result",
            "invalid_reason": None,
            "selected_indices": None,
            "residuals": None,
            "terms": None,
            "expression": "",
            "lhs_name": None,
            "condition_number": None,
            "condition_number_computed": False,
        },
    ),
    (
        _result_with_none_score,
        {
            "mse": 0.5,
            "nmse": 0.6,
            "r2": 0.4,
            "score": None,
            "complexity": 1,
            "coefficients": [2.0],
            "is_valid": True,
            "error_message": "",
            "invalid_reason": None,
            "selected_indices": None,
            "residuals": [0.01, 0.02],
            "terms": ["u_xx"],
            "expression": "",
            "lhs_name": None,
            "condition_number": None,
            "condition_number_computed": False,
        },
    ),
    (
        _result_with_zero_score,
        {
            "mse": 0.3,
            "nmse": 0.3,
            "r2": 0.7,
            "score": 0.0,
            "complexity": 1,
            "coefficients": [1.5],
            "is_valid": True,
            "error_message": "",
            "invalid_reason": None,
            "selected_indices": None,
            "residuals": [0.0, 0.0],
            "terms": ["diff_x(u)"],
            "expression": "",
            "lhs_name": None,
            "condition_number": None,
            "condition_number_computed": False,
        },
    ),
]


@pytest.mark.parametrize(("factory", "expected"), _GOLDEN_TO_DICT)
def test_i_to_dict_matches_inline_golden(
    factory: Any, expected: dict[str, Any]
) -> None:
    result = factory()

    produced = result.to_dict()

    assert produced == expected

    assert set(produced.keys()) == _EVALUATION_RESULT_KEYS


def test_i_to_dict_zero_score_stays_zero_not_none() -> None:
    payload = _result_with_zero_score().to_dict()

    assert payload["score"] == 0.0
    assert payload["score"] is not None


@pytest.mark.parametrize(
    "factory",
    [
        _valid_result_with_tensors,
        _invalid_result_with_neg_inf,
        _result_with_none_score,
        _result_with_zero_score,
    ],
)
def test_i_to_dict_json_round_trips(factory: Any) -> None:
    result = factory()

    payload = result.to_dict()
    dumped = json.dumps(payload)
    reloaded = json.loads(dumped)

    assert reloaded == payload


def test_i_to_dict_score_none_preserved() -> None:
    result = _result_with_none_score()

    payload = result.to_dict()

    assert "score" in payload
    assert payload["score"] is None


def test_i_to_dict_neg_inf_r2_sanitized_to_none() -> None:
    result = _invalid_result_with_neg_inf()

    payload = result.to_dict()

    assert payload["r2"] is None
    assert payload["is_valid"] is False


def test_i_to_dict_include_residuals_false_nulls_residuals_only() -> None:
    result = _valid_result_with_tensors()

    full = result.to_dict(include_residuals=True)
    trimmed = result.to_dict(include_residuals=False)


    assert set(trimmed.keys()) == set(full.keys())
    assert "residuals" in trimmed
    assert trimmed["residuals"] is None

    assert full["residuals"] is not None


    for key in full:
        if key == "residuals":
            continue
        assert trimmed[key] == full[key]







def test_j_top_level_exports_present() -> None:
    import kd

    expected_exports = (
        "evaluate_terms",
        "validate_terms",
        "EvaluationResult",
        "TermValidationReport",
        "TermRejection",
        "InvalidTermsError",
        "EvaluationFailedError",
    )
    for name in expected_exports:
        assert hasattr(kd, name), f"kd.{name} is not exported"


def test_j_exports_listed_in_all_and_sorted() -> None:
    import kd

    expected_exports = {
        "evaluate_terms",
        "validate_terms",
        "EvaluationResult",
        "TermValidationReport",
        "TermRejection",
        "InvalidTermsError",
        "EvaluationFailedError",
    }
    all_names = set(kd.__all__)
    assert expected_exports.issubset(all_names)

    assert list(kd.__all__) == sorted(kd.__all__)


def test_j_callable_via_module_alias() -> None:
    import kd

    ds = _build_advection_dataset()
    result = kd.evaluate_terms(ds, ["diff_x(u)"])
    assert result.is_valid is True

















def _build_advection_dataset_order2() -> PDEDataset:
    from kd.data.schema import PDEDataset as _PDEDataset

    base = _build_advection_dataset()
    x = base.get_coords("x")
    t = base.get_coords("t")
    u = base.get_field("u")
    return _PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u},
        lhs="u_tt",
        periodic={"x"},
        name="advection_u_tt",
    )


def test_k_default_none_derives_order_two_from_dataset() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset_order2()

    assert ds.lhs_order == 2

    result = api.evaluate_terms(ds, ["diff2_x(u)"])

    assert result.is_valid is True
    assert result.coefficients is not None
    assert result.coefficients.numel() == 1
    coef = float(result.coefficients.flatten()[0])
    expected = _WAVE_SPEED**2
    assert coef > 0.0
    assert _within_rel(coef, expected)
    assert result.nmse < _NMSE_CEILING


def test_k_default_none_derives_order_one_advection() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()
    assert ds.lhs_order == 1

    result = api.evaluate_terms(ds, ["diff_x(u)"])

    assert result.is_valid is True
    assert result.coefficients is not None
    coef = float(result.coefficients.flatten()[0])
    assert coef < 0.0
    assert _within_rel(coef, -_WAVE_SPEED)
    assert result.nmse < _NMSE_CEILING


def test_k_default_none_validate_derives_tautology_from_dataset() -> None:
    api = _evaluate_api()

    ds_order2 = _build_advection_dataset_order2()
    assert ds_order2.lhs_order == 2
    report2 = api.validate_terms(ds_order2, ["diff_t(u)"])
    assert report2.ok is True
    assert report2.valid == ["diff_t(u)"]

    ds_order1 = _build_advection_dataset()
    assert ds_order1.lhs_order == 1
    report1 = api.validate_terms(ds_order1, ["diff_t(u)"])
    assert report1.ok is False
    assert {rej.term for rej in report1.rejected} == {"diff_t(u)"}


def test_k_explicit_kwarg_overrides_dataset_order() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()

    assert ds.lhs_order == 1

    result = api.evaluate_terms(ds, ["diff2_x(u)"], lhs_order=2)

    assert result.is_valid is True
    assert result.coefficients is not None
    assert result.coefficients.numel() == 1
    coef = float(result.coefficients.flatten()[0])
    assert coef > 0.0
    assert _within_rel(coef, _WAVE_SPEED**2)
    assert result.nmse < _NMSE_CEILING


def test_k_explicit_kwarg_override_accepts_diff_t() -> None:
    api = _evaluate_api()
    ds = _build_advection_dataset()
    assert ds.lhs_order == 1

    report = api.validate_terms(ds, ["diff_t(u)"], lhs_order=2)
    assert report.ok is True
    assert report.valid == ["diff_t(u)"]

    result = api.evaluate_terms(ds, ["diff_t(u)"], lhs_order=2)
    assert result.is_valid is True
    assert result.coefficients is not None
