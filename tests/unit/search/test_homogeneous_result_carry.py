
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from kd.core.equation import Equation, Form, Scalar, make_homogeneous
from kd.core.evaluator import EvaluationResult
from kd.data.schema import PDEDataset
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import RecordingAlgorithm






def _homogeneous_dataset() -> PDEDataset:
    n = 8
    x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    y = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    u = x * y
    return PDEDataset.from_scatter(coords={"x": x, "y": y}, fields={"u": u}, lhs="")


def _real_components(dataset: PDEDataset) -> PlatformComponents:
    return PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
    )


def _homogeneous_algorithm() -> RecordingAlgorithm:
    algo = RecordingAlgorithm(
        score_sequence=[0.0], expression_sequence=["diff2_x(u) + diff2_y(u)"]
    )
    algo.result_target = torch.zeros(4, dtype=torch.float64)
    algo.final_eval_result = EvaluationResult(
        mse=0.0,
        nmse=0.0,
        r2=1.0,
        score=0.0,
        complexity=2,
        coefficients=torch.tensor([1.0, 1.0], dtype=torch.float64),
        is_valid=True,
        selected_indices=[0, 1],
        residuals=torch.zeros(4, dtype=torch.float64),
        terms=["diff2_x(u)", "diff2_y(u)"],
        expression="diff2_x(u) + diff2_y(u)",
        lhs_name=None,
        form=Form.HOMOGENEOUS,
    )
    return algo


def _homogeneous_result_with_equation() -> ExperimentResult:
    rec = VizRecorder()
    rec.log("loss", 0.5)
    actual = torch.zeros(8)
    return ExperimentResult(
        best_expression="diff2_x(u) + diff2_y(u)",
        best_score=0.02,
        iterations=3,
        early_stopped=False,
        final_eval=EvaluationResult(mse=0.0, nmse=0.0, r2=1.0),
        actual=actual,
        predicted=actual.clone(),
        dataset_name="eqgpt-laplacian-smile",
        algorithm_name="eqgpt",
        config={"algorithm": "eqgpt"},
        recorder=rec,
        equation=make_homogeneous(
            [("diff2_x(u)", Scalar(1.0)), ("diff2_y(u)", Scalar(1.0))]
        ),
    )







class TestEvaluationResultForm:
    @pytest.mark.unit
    def test_form_defaults_to_evolution(self) -> None:

        assert EvaluationResult(mse=0.0, nmse=0.0, r2=1.0).form is Form.EVOLUTION

    @pytest.mark.unit
    def test_form_carries_homogeneous(self) -> None:
        result = EvaluationResult(mse=0.0, nmse=0.0, r2=1.0, form=Form.HOMOGENEOUS)
        assert result.form is Form.HOMOGENEOUS







class TestRunnerDispatch:
    @pytest.mark.unit
    def test_homogeneous_form_yields_homogeneous_equation(self) -> None:
        result = ExperimentRunner(
            _homogeneous_algorithm(), max_iterations=1, batch_size=1
        ).run(_real_components(_homogeneous_dataset()))

        assert result.equation is not None
        assert isinstance(result.equation, Equation)
        assert result.equation.form is Form.HOMOGENEOUS

    @pytest.mark.unit
    def test_pivot_index_zero_pinned_and_terms_in_order(self) -> None:
        result = ExperimentRunner(
            _homogeneous_algorithm(), max_iterations=1, batch_size=1
        ).run(_real_components(_homogeneous_dataset()))

        assert result.equation is not None
        term_irs = [ir for ir, _c in result.equation.terms]
        assert term_irs == ["diff2_x(u)", "diff2_y(u)"]

        assert result.equation.terms[0][1] == Scalar(1.0)







class TestCarrierRoundTrip:
    @pytest.mark.unit
    def test_to_dict_emits_homogeneous_equation(self) -> None:
        payload = _homogeneous_result_with_equation().to_dict()
        assert isinstance(payload["equation"], dict)
        assert payload["equation"]["form"] == "HOMOGENEOUS"

    @pytest.mark.unit
    def test_save_load_round_trips_homogeneous_equation(self, tmp_path: Path) -> None:
        original = _homogeneous_result_with_equation()
        original.save(tmp_path / "homog.json")
        loaded = ExperimentResult.load(tmp_path / "homog.json")

        assert loaded.equation is not None
        assert loaded.equation.form is Form.HOMOGENEOUS
        assert loaded.equation == original.equation







def _sparse_homogeneous_algorithm() -> RecordingAlgorithm:
    algo = RecordingAlgorithm(
        score_sequence=[0.0],
        expression_sequence=["diff2_x(u) + diff2_z(u)"],
    )
    algo.result_target = torch.zeros(4, dtype=torch.float64)
    algo.final_eval_result = EvaluationResult(
        mse=0.0,
        nmse=0.0,
        r2=1.0,
        score=0.0,
        complexity=2,
        coefficients=torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        is_valid=True,
        selected_indices=[0, 2],
        residuals=torch.zeros(4, dtype=torch.float64),
        terms=["diff2_x(u)", "diff2_y(u)", "diff2_z(u)"],
        expression="diff2_x(u) + diff2_z(u)",
        lhs_name=None,
        form=Form.HOMOGENEOUS,
    )
    return algo


class TestHomogeneousActiveSupportCarry:
    @pytest.mark.unit
    def test_homogeneous_support_is_pivot_inclusive(self) -> None:
        result = ExperimentRunner(
            _sparse_homogeneous_algorithm(), max_iterations=1, batch_size=1
        ).run(_real_components(_homogeneous_dataset()))

        assert result.equation is not None
        assert result.equation.active_indices == (0, 2)
        active_terms = [
            result.equation.terms[i][0] for i in result.equation.active_indices
        ]
        assert active_terms == ["diff2_x(u)", "diff2_z(u)"]
