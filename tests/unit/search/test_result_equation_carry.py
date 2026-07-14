
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from kd.core.equation import Equation, Form, LhsSpec, Scalar
from kd.core.evaluator import EvaluationResult
from kd.data.schema import PDEDataset
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import RecordingAlgorithm






def _evolution_dataset(lhs: str) -> PDEDataset:
    nx, nt = 6, 5
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.outer(x, t)
    return PDEDataset.from_arrays(coords={"x": x, "t": t}, fields={"u": u}, lhs=lhs)


def _real_components(dataset: PDEDataset) -> PlatformComponents:
    return PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
    )


def _terms_algorithm() -> RecordingAlgorithm:
    algo = RecordingAlgorithm(
        score_sequence=[0.0], expression_sequence=["u_x + u_xx"]
    )
    algo.result_target = torch.ones(4, dtype=torch.float64)
    algo.final_eval_result = EvaluationResult(
        mse=0.0,
        nmse=0.0,
        r2=1.0,
        score=0.0,
        complexity=2,
        coefficients=torch.tensor([1.0, -0.5], dtype=torch.float64),
        is_valid=True,
        selected_indices=[0, 1],
        residuals=torch.zeros(4, dtype=torch.float64),
        terms=["u_x", "u_xx"],
        expression="u_x + u_xx",
        lhs_name=None,
    )
    return algo


def _result_without_equation() -> ExperimentResult:
    rec = VizRecorder()
    rec.log("loss", 0.5)
    actual = torch.zeros(8)
    return ExperimentResult(
        best_expression="u_x + u_xx",
        best_score=0.02,
        iterations=7,
        early_stopped=False,
        final_eval=EvaluationResult(mse=0.1, nmse=0.1, r2=0.9),
        actual=actual,
        predicted=actual.clone(),
        dataset_name="burgers_1d",
        algorithm_name="sga",
        config={"algorithm": "sga"},
        recorder=rec,
        lhs_label="u_t",
    )







class TestEquationCarry:
    @pytest.mark.unit
    def test_run_populates_evolution_equation(self) -> None:
        result = ExperimentRunner(
            _terms_algorithm(), max_iterations=1, batch_size=1
        ).run(_real_components(_evolution_dataset("u_t")))

        assert result.equation is not None
        assert isinstance(result.equation, Equation)
        assert result.equation.form is Form.EVOLUTION

    @pytest.mark.unit
    def test_equation_lhs_spec_maps_from_dataset(self) -> None:
        result = ExperimentRunner(
            _terms_algorithm(), max_iterations=1, batch_size=1
        ).run(_real_components(_evolution_dataset("u_t")))

        assert result.equation is not None
        assert result.equation.lhs_spec == LhsSpec(field="u", axis="t", order=1)

        assert result.lhs_label == "u_t"

    @pytest.mark.unit
    def test_equation_terms_match_final_eval_in_order(self) -> None:
        result = ExperimentRunner(
            _terms_algorithm(), max_iterations=1, batch_size=1
        ).run(_real_components(_evolution_dataset("u_t")))

        assert result.equation is not None
        term_irs = [ir for ir, _coef in result.equation.terms]
        assert term_irs == ["u_x", "u_xx"]
        assert term_irs == result.final_eval.terms
        coeffs = [coef for _ir, coef in result.equation.terms]
        assert coeffs == [Scalar(1.0), Scalar(-0.5)]

    @pytest.mark.unit
    def test_non_finite_coefficient_degrades_equation_and_still_saves(
        self, tmp_path: Path
    ) -> None:
        algo = _terms_algorithm()
        algo.final_eval_result.coefficients = torch.tensor(
            [float("nan"), -0.5], dtype=torch.float64
        )

        result = ExperimentRunner(
            algo, max_iterations=1, batch_size=1
        ).run(_real_components(_evolution_dataset("u_t")))

        assert result.final_eval.is_valid is True
        assert result.equation is None
        json.dumps(result.to_dict(), allow_nan=False)
        result.save(tmp_path / "nan_coefficient.json")

    @pytest.mark.unit
    def test_invalid_final_eval_degrades_equation(self) -> None:
        algo = _terms_algorithm()
        algo.final_eval_result.is_valid = False

        result = ExperimentRunner(
            algo, max_iterations=1, batch_size=1
        ).run(_real_components(_evolution_dataset("u_t")))

        assert result.equation is None
        assert result.lhs_label == "u_t"







class TestEquationDoubleWrite:
    @pytest.mark.unit
    def test_to_dict_absent_equation_emits_null_key(self) -> None:
        payload = _result_without_equation().to_dict()
        assert "equation" in payload
        assert payload["equation"] is None

    @pytest.mark.unit
    def test_to_dict_present_equation_emits_dict(self) -> None:
        result = ExperimentRunner(
            _terms_algorithm(), max_iterations=1, batch_size=1
        ).run(_real_components(_evolution_dataset("u_t")))

        payload = result.to_dict()
        assert "equation" in payload
        assert isinstance(payload["equation"], dict)
        assert payload["equation"]["form"] == "EVOLUTION"







class TestEquationThroughputGuard:
    @pytest.mark.unit
    def test_equation_built_once_per_result(
        self, mock_components: PlatformComponents, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert hasattr(ExperimentRunner, "_build_equation"), (
            "1c must add ExperimentRunner._build_equation as the single "
            "result-boundary Equation construction hook (D1-6)"
        )
        original = ExperimentRunner._build_equation
        calls = {"n": 0}

        def counting(self: ExperimentRunner, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(ExperimentRunner, "_build_equation", counting)

        ExperimentRunner(
            RecordingAlgorithm(), max_iterations=3, batch_size=4
        ).run(mock_components)

        assert calls["n"] == 1
