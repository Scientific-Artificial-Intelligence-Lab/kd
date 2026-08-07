
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

import kd
from kd.api import Model
from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.core.linear_solve.stridge import STRidgeSolver
from kd.core.linear_solve.svd_null_space import SVDNullSpaceSolver
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import FieldData, PDEDataset
from kd.data.synthetic import generate_burgers_data
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult, _deserialize_evaluation_result

_NX = 32
_NT = 16
_NU = 0.1
_TERMS = ["u_x", "u_xx"]

_REPO_ROOT = Path(__file__).resolve().parents[2]






_COMMITTED_RESULT_FILES = (
    "examples/out/burgers.json",
    "examples/out/pde_divide.json",
    "examples/out/09_compare/discover.json",
    "examples/out/09_compare/dlga.json",
    "examples/out/09_compare/eqgpt.json",
    "examples/out/09_compare/llm4ed.json",
    "examples/out/09_compare/sga.json",
)


_THETA = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
_Y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)


@pytest.fixture
def dataset() -> PDEDataset:
    return generate_burgers_data(nx=_NX, nt=_NT, nu=_NU, seed=0)


@pytest.fixture
def collinear_dataset() -> PDEDataset:
    data = generate_burgers_data(nx=_NX, nt=_NT, nu=_NU, seed=0)
    assert data.fields is not None
    u = data.fields["u"].values
    data.fields["v"] = FieldData(name="v", values=(2.0 * u).clone())
    return data


def _components(data: PDEDataset, **kwargs: Any) -> PlatformComponents:
    reqs = DerivativeReqs(max_atomic_order=2, lhs_order=1)
    return PlatformBuilder(data, reqs, **kwargs).build()


def _result_with(condition_number: float | None) -> EvaluationResult:
    return EvaluationResult(
        mse=0.5,
        nmse=0.6,
        r2=0.4,
        score=-1.0,
        complexity=1,
        coefficients=torch.tensor([2.0], dtype=torch.float64),
        is_valid=True,
        terms=["u_xx"],
        condition_number=condition_number,
    )







def test_evaluate_terms_reports_a_finite_condition_number(
    dataset: PDEDataset,
) -> None:
    result = kd.evaluate_terms(dataset, ["u_x", "mul(u, u_x)", "u_xx"])

    assert result.condition_number is not None
    assert math.isfinite(result.condition_number)
    assert result.condition_number >= 1.0


def test_evaluate_terms_flags_a_collinear_library(
    collinear_dataset: PDEDataset,
) -> None:
    healthy = kd.evaluate_terms(collinear_dataset, ["u", "u_xx"])
    degenerate = kd.evaluate_terms(collinear_dataset, ["u", "v", "u_xx"])

    assert healthy.condition_number is not None
    assert degenerate.condition_number is not None
    assert degenerate.condition_number > 1e12
    assert degenerate.condition_number > healthy.condition_number


def test_evaluate_terms_payload_carries_the_measured_pair(
    dataset: PDEDataset,
) -> None:
    payload = kd.evaluate_terms(dataset, _TERMS).to_dict(include_residuals=False)

    assert payload["condition_number_computed"] is True
    assert isinstance(payload["condition_number"], float)
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload







def test_builder_default_leaves_both_gates_closed(dataset: PDEDataset) -> None:
    components = _components(dataset)
    evaluator = components.evaluator
    assert evaluator is not None

    result = evaluator.evaluate_terms(_TERMS)

    assert evaluator.solver.compute_condition_number is False
    assert result.condition_number is None
    assert result.to_dict()["condition_number_computed"] is False


def test_builder_opt_in_opens_both_gates(dataset: PDEDataset) -> None:
    components = _components(dataset, compute_condition_number=True)
    evaluator = components.evaluator
    assert evaluator is not None

    result = evaluator.evaluate_terms(_TERMS)

    assert evaluator.solver.compute_condition_number is True
    assert result.condition_number is not None
    assert math.isfinite(result.condition_number)


def test_model_fit_result_leaves_the_field_unmeasured(dataset: PDEDataset) -> None:
    model = Model(
        algorithm="sga",
        generations=2,
        population=5,
        depth=3,
        width=3,
        seed=0,
        verbose=False,
    )

    model.fit(dataset)

    assert model.result_ is not None
    assert model.result_.final_eval.condition_number is None







@pytest.mark.parametrize(
    "solver_factory",
    [
        lambda: STRidgeSolver(compute_condition_number=True),
        SVDNullSpaceSolver,
    ],
    ids=["stridge", "svd_null_space"],
)
def test_wrapping_solvers_report_nothing_by_default(
    dataset: PDEDataset, solver_factory: Any
) -> None:
    components = _components(dataset)
    base = components.evaluator
    assert base is not None
    evaluator = Evaluator(
        executor=components.executor,
        solver=solver_factory(),
        context=components.context,
        lhs=base.lhs_target,
    )

    result = evaluator.evaluate_terms(_TERMS)

    assert result.is_valid is True
    assert result.condition_number is None


def test_report_gate_is_what_forwards_the_value(dataset: PDEDataset) -> None:
    components = _components(dataset)
    base = components.evaluator
    assert base is not None
    evaluator = Evaluator(
        executor=components.executor,
        solver=STRidgeSolver(compute_condition_number=True),
        context=components.context,
        lhs=base.lhs_target,
        report_condition_number=True,
    )

    result = evaluator.evaluate_terms(_TERMS)

    assert result.condition_number is not None


def test_invalid_result_reports_not_measured(dataset: PDEDataset) -> None:
    components = _components(dataset, compute_condition_number=True)
    evaluator = components.evaluator
    assert evaluator is not None

    result = evaluator.evaluate_terms(["definitely_not_a_variable"])

    assert result.is_valid is False
    assert result.condition_number is None







@pytest.mark.parametrize(
    ("value", "expected_value", "expected_computed"),
    [
        (12.5, 12.5, True),
        (float("inf"), None, True),
        (None, None, False),
    ],
    ids=["measured_finite", "measured_degenerate", "not_measured"],
)
def test_to_dict_encodes_three_states(
    value: float | None, expected_value: float | None, expected_computed: bool
) -> None:
    payload = _result_with(value).to_dict()

    assert payload["condition_number"] == expected_value
    assert payload["condition_number_computed"] is expected_computed


@pytest.mark.parametrize(
    "value", [12.5, float("inf"), None], ids=["finite", "inf", "none"]
)
def test_payload_round_trips_through_the_loader(value: float | None) -> None:
    payload = _result_with(value).to_dict()

    restored = _deserialize_evaluation_result(json.loads(json.dumps(payload)))

    assert restored.condition_number == value


def test_legacy_payload_without_the_keys_loads_as_unmeasured() -> None:
    payload = _result_with(12.5).to_dict()
    del payload["condition_number"]
    del payload["condition_number_computed"]

    restored = _deserialize_evaluation_result(payload)

    assert restored.condition_number is None


def test_experiment_result_file_round_trips_a_degenerate_library(
    tmp_path: Path,
) -> None:
    result = ExperimentResult(
        best_expression="u_xx",
        best_score=0.02,
        iterations=1,
        early_stopped=False,
        final_eval=_result_with(float("inf")),
        actual=torch.zeros(4),
        predicted=torch.zeros(4),
        dataset_name="probe",
        algorithm_name="sga",
        config={},
        recorder=VizRecorder(),
    )
    path = tmp_path / "result.json"

    result.save(path)
    loaded = ExperimentResult.load(path)

    assert loaded.final_eval.condition_number == float("inf")


@pytest.mark.parametrize("corpus", _COMMITTED_RESULT_FILES)
def test_committed_final_eval_payloads_still_deserialize(corpus: str) -> None:
    path = _REPO_ROOT / corpus
    assert path.exists(), f"committed corpus is missing: {corpus}"
    payload = json.loads(path.read_text(encoding="utf-8"))

    restored = _deserialize_evaluation_result(payload["final_eval"])

    assert restored.condition_number is None


@pytest.mark.parametrize("corpus", _COMMITTED_RESULT_FILES)
def test_committed_result_files_still_load(corpus: str) -> None:
    path = _REPO_ROOT / corpus
    assert path.exists(), f"committed corpus is missing: {corpus}"

    loaded = ExperimentResult.load(path)

    assert loaded.final_eval.condition_number is None







def test_lapack_failure_is_reported_as_degenerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def _fail(_matrix: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("linalg.cond: LAPACK failure")

    monkeypatch.setattr(torch.linalg, "cond", _fail)
    solver = LeastSquaresSolver(compute_condition_number=True)

    result = solver.solve(_THETA, _Y)

    assert result.condition_number == float("inf")


def test_allocation_failure_is_not_reported_as_degenerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def _oom(_matrix: torch.Tensor) -> torch.Tensor:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(torch.linalg, "cond", _oom)
    solver = LeastSquaresSolver(compute_condition_number=True)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        solver.solve(_THETA, _Y)


def test_cpu_allocation_failure_is_not_reported_as_degenerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def _oom(_matrix: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "[enforce fail at alloc_cpu.cpp:117]. DefaultCPUAllocator: "
            "can't allocate memory: you tried to allocate 8000000000000 bytes."
        )

    monkeypatch.setattr(torch.linalg, "cond", _oom)
    solver = LeastSquaresSolver(compute_condition_number=True)

    with pytest.raises(RuntimeError, match="DefaultCPUAllocator"):
        solver.solve(_THETA, _Y)
