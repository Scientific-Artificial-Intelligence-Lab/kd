
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import Tensor

from kd.core.equation import Equation, Form, LhsSpec, Scalar, make_evolution
from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult






def _recorder() -> VizRecorder:
    rec = VizRecorder()
    rec.log("loss", 0.5)
    rec.log("loss", 0.1)
    return rec


def _valid_final_eval(lhs_name: str | None = None) -> EvaluationResult:
    return EvaluationResult(
        mse=0.01,
        nmse=0.02,
        r2=0.98,
        score=-50.0,
        complexity=3,
        coefficients=torch.tensor([1.0, -6.0, 1.0]),
        is_valid=True,
        selected_indices=[0, 1, 2],
        residuals=torch.zeros(8),
        terms=["u", "mul(u, u_x)", "u_xx"],
        expression="add(u, add(mul(u, u_x), u_xx))",
        lhs_name=lhs_name,
    )


def _experiment_result(
    final_eval: EvaluationResult,
    *,
    lhs_label: str = "u_t",
) -> ExperimentResult:
    return ExperimentResult(
        best_expression="add(u, add(mul(u, u_x), u_xx))",
        best_score=0.02,
        iterations=7,
        early_stopped=False,
        final_eval=final_eval,
        actual=torch.zeros(8),
        predicted=torch.zeros(8),
        dataset_name="burgers_1d",
        algorithm_name="sga",
        config={"algorithm": "sga"},
        recorder=_recorder(),
        lhs_label=lhs_label,
    )


def _write_legacy_json(result: ExperimentResult, path: Path) -> None:
    result.save(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.pop("equation", None)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)







class TestLegacyLoadDerivesEquation:
    @pytest.mark.unit
    def test_legacy_dict_without_equation_derives_equation(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "legacy.json"
        _write_legacy_json(_experiment_result(_valid_final_eval()), path)

        loaded = ExperimentResult.load(path)

        assert loaded.equation is not None
        assert loaded.equation.form is Form.EVOLUTION
        term_irs = [ir for ir, _coef in loaded.equation.terms]
        assert term_irs == ["u", "mul(u, u_x)", "u_xx"]
        coeffs = [coef for _ir, coef in loaded.equation.terms]
        assert coeffs == [Scalar(1.0), Scalar(-6.0), Scalar(1.0)]

        assert loaded.lhs_label == "u_t"

    @pytest.mark.unit
    def test_legacy_lhs_name_derives_second_order_lhs(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy_u_tt.json"
        result = _experiment_result(
            _valid_final_eval(lhs_name="u_tt"),
            lhs_label="u_tt",
        )
        _write_legacy_json(result, path)

        loaded = ExperimentResult.load(path)

        assert loaded.equation is not None
        assert loaded.equation.lhs_spec == LhsSpec("u", "t", 2)
        assert loaded.lhs_label == "u_tt"

    @pytest.mark.unit
    def test_legacy_lhs_label_derives_named_field_lhs(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy_phi_x.json"
        result = _experiment_result(
            _valid_final_eval(lhs_name=None),
            lhs_label="phi_x",
        )
        _write_legacy_json(result, path)

        loaded = ExperimentResult.load(path)

        assert loaded.equation is not None
        assert loaded.equation.lhs_spec == LhsSpec("phi", "x", 1)
        assert loaded.lhs_label == "phi_x"

    @pytest.mark.unit
    def test_legacy_unparseable_lhs_name_degrades_to_none(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "legacy_mixed_lhs.json"
        result = _experiment_result(
            _valid_final_eval(lhs_name="u_x_y"),
            lhs_label="u_t",
        )
        _write_legacy_json(result, path)

        loaded = ExperimentResult.load(path)

        assert loaded.equation is None
        assert loaded.lhs_label == "u_t"

    @pytest.mark.unit
    def test_legacy_invalid_result_loads_with_none_equation(
        self, tmp_path: Path
    ) -> None:
        invalid = EvaluationResult(
            mse=float("inf"),
            nmse=float("inf"),
            r2=-float("inf"),
            score=None,
            complexity=0,
            coefficients=None,
            is_valid=False,
            error_message="boom",
            selected_indices=None,
            residuals=None,
            terms=None,
            expression="",
        )
        path = tmp_path / "legacy_invalid.json"
        _write_legacy_json(_experiment_result(invalid), path)

        loaded = ExperimentResult.load(path)

        assert loaded.equation is None







class TestNewPayloadRoundTrip:
    @pytest.mark.unit
    def test_new_result_with_equation_round_trips(self, tmp_path: Path) -> None:
        equation = make_evolution(
            LhsSpec(field="u", axis="t", order=1),
            (("u_x", Scalar(1.0)), ("u_xx", Scalar(-0.5))),
        )
        result = _experiment_result_with_equation(_valid_final_eval(), equation)

        path = tmp_path / "new.json"
        result.save(path)
        loaded = ExperimentResult.load(path)

        assert loaded.equation == equation

    @pytest.mark.unit
    def test_new_payload_writes_non_null_equation_key(self, tmp_path: Path) -> None:
        equation = make_evolution(
            LhsSpec(field="u", axis="t", order=1),
            (("u_x", Scalar(1.0)),),
        )
        result = _experiment_result_with_equation(_valid_final_eval(), equation)

        path = tmp_path / "new.json"
        result.save(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        assert data["equation"] is not None
        assert data["equation"]["form"] == "EVOLUTION"

    @pytest.mark.unit
    def test_new_payload_with_null_equation_stays_null(self, tmp_path: Path) -> None:
        path = tmp_path / "new_null_equation.json"
        result = _experiment_result(_valid_final_eval(), lhs_label="u_t")

        result.save(path)
        loaded = ExperimentResult.load(path)

        assert loaded.equation is None


def _experiment_result_with_equation(
    final_eval: EvaluationResult, equation: Equation
) -> ExperimentResult:
    actual: Tensor = torch.zeros(8)
    return ExperimentResult(
        best_expression="u_x + u_xx",
        best_score=0.02,
        iterations=7,
        early_stopped=False,
        final_eval=final_eval,
        actual=actual,
        predicted=actual.clone(),
        dataset_name="burgers_1d",
        algorithm_name="sga",
        config={"algorithm": "sga"},
        recorder=_recorder(),
        lhs_label="u_t",
        equation=equation,
    )
