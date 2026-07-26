
from __future__ import annotations

import math

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.record_assembly import (
    active_support_and_coefficients,
    assemble_run_record,
    residual_stats_from,
)
from kd.search.records import ResidualStats, RunCost, RunRecord
from kd.search.run_spec import RunSpec






@pytest.mark.unit
def test_residual_stats_from_none_returns_none() -> None:
    assert residual_stats_from(None) is None


@pytest.mark.unit
def test_residual_stats_from_computes_scalar_summary() -> None:
    stats = residual_stats_from(torch.tensor([1.0, -3.0, 2.0]))

    assert isinstance(stats, ResidualStats)
    assert stats is not None
    assert stats.n == 3
    assert stats.max_abs == pytest.approx(3.0)
    assert stats.mean == pytest.approx(0.0)

    assert stats.std is not None and math.isfinite(stats.std)


@pytest.mark.unit
def test_residual_stats_from_empty_tensor_does_not_crash() -> None:
    stats = residual_stats_from(torch.empty(0))

    assert stats == ResidualStats(mean=None, std=None, max_abs=None, n=0)


@pytest.mark.unit
def test_residual_stats_empty_summary_is_json_safe() -> None:
    stats = residual_stats_from(torch.empty(0))
    assert stats is not None
    assert ResidualStats.from_dict(stats.to_dict()) == stats







def _assemble(final_eval: EvaluationResult, *, best_score: float = 0.5) -> RunRecord:
    return assemble_run_record(
        instrument="sga",
        dataset_name="burgers_1d",
        dataset_cache_fingerprint="abc123",
        seed=0,
        final_eval=final_eval,
        equation=None,
        best_expression="u_x",
        best_score=best_score,
        score_kind="Score",
        score_direction="min",
        headline_coefficient_source="native",
        run_spec=RunSpec(
            kd_version="0.1.0",
            config={"algorithm": "sga"},
            dataset_cache_fingerprint="abc123",
        ),
        cost=RunCost(
            wallclock_seconds=1.0,
            search_seconds=1.0,
            boundary_results=0,
            boundary_invalid_results=0,
        ),
    )


@pytest.mark.unit
def test_invalid_eval_records_complexity_as_none() -> None:
    invalid = EvaluationResult(
        mse=float("inf"),
        nmse=float("inf"),
        r2=float("-inf"),
        score=None,
        complexity=0,
        coefficients=None,
        is_valid=False,
        selected_indices=None,
        residuals=None,
        terms=None,
        expression="",
    )

    record = _assemble(invalid, best_score=0.0)

    assert record.evidence.complexity is None
    assert record.evidence.is_valid is False


@pytest.mark.unit
def test_valid_eval_preserves_measured_complexity() -> None:
    valid = EvaluationResult(
        mse=0.1,
        nmse=0.2,
        r2=0.9,
        complexity=2,
        is_valid=True,
        terms=["u_x", "u_xx"],
        expression="u_x + u_xx",
    )

    assert _assemble(valid).evidence.complexity == 2


@pytest.mark.unit
def test_metric_int_and_float_hash_identically() -> None:
    shared = {
        "nmse": 0.0,
        "r2": 1.0,
        "complexity": 1,
        "is_valid": True,
        "terms": ["u_x"],
        "expression": "u_x",
    }
    eval_int = EvaluationResult(mse=0, **shared)
    eval_float = EvaluationResult(mse=0.0, **shared)

    record_int = _assemble(eval_int, best_score=0)
    record_float = _assemble(eval_float, best_score=0.0)

    assert record_int.evidence.mse == 0.0
    assert isinstance(record_int.evidence.mse, float)
    assert record_int.evidence_hash == record_float.evidence_hash







@pytest.mark.unit
def test_active_support_sparse_fit_records_only_selected() -> None:
    sparse = EvaluationResult(
        mse=0.1,
        nmse=0.2,
        r2=0.9,
        complexity=1,
        is_valid=True,
        coefficients=torch.tensor([0.0, 2.0, 0.0]),
        selected_indices=[1],
        terms=["u", "u_x", "u_xx"],
        expression="u_x",
    )

    evidence = _assemble(sparse).evidence

    assert evidence.support == ["u_x"]
    assert evidence.coefficients == [2.0]


@pytest.mark.unit
def test_active_support_dense_fit_keeps_all_columns() -> None:
    dense = EvaluationResult(
        mse=0.1,
        nmse=0.2,
        r2=0.9,
        complexity=3,
        is_valid=True,
        coefficients=torch.tensor([1.0, 2.0, 3.0]),
        selected_indices=None,
        terms=["u", "u_x", "u_xx"],
        expression="u + u_x + u_xx",
    )

    evidence = _assemble(dense).evidence

    assert evidence.support == ["u", "u_x", "u_xx"]
    assert evidence.coefficients == [1.0, 2.0, 3.0]


@pytest.mark.unit
def test_active_support_and_coefficients_stay_index_aligned() -> None:
    result = EvaluationResult(
        mse=0.1,
        nmse=0.2,
        r2=0.9,
        is_valid=True,
        coefficients=torch.tensor([5.0, 0.0, -1.0, 0.0]),
        selected_indices=[0, 2],
        terms=["a", "b", "c", "d"],
        expression="5a - c",
    )

    support, coefficients = active_support_and_coefficients(result)

    assert support == ["a", "c"]
    assert coefficients == [5.0, -1.0]
    assert support is not None and coefficients is not None
    assert len(support) == len(coefficients)


@pytest.mark.unit
def test_active_support_terms_none_returns_none_pair() -> None:
    invalid = EvaluationResult(
        mse=float("inf"),
        nmse=float("inf"),
        r2=float("-inf"),
        is_valid=False,
        coefficients=None,
        selected_indices=None,
        terms=None,
        expression="",
    )

    assert active_support_and_coefficients(invalid) == (None, None)


@pytest.mark.unit
def test_active_support_corrupt_picklist_drops_wholesale() -> None:
    corrupt = EvaluationResult(
        mse=0.1,
        nmse=0.2,
        r2=0.9,
        is_valid=True,
        coefficients=torch.tensor([5.0, 0.0, -1.0]),
        selected_indices=[0, 7],
        terms=["a", "b", "c"],
        expression="5a",
    )

    assert active_support_and_coefficients(corrupt) == (None, None)


@pytest.mark.unit
def test_active_support_short_coefficients_drop_to_none() -> None:
    misaligned = EvaluationResult(
        mse=0.1,
        nmse=0.2,
        r2=0.9,
        is_valid=True,
        coefficients=torch.tensor([5.0, -1.0]),
        selected_indices=[0, 2],
        terms=["a", "b", "c"],
        expression="5a - c",
    )

    support, coefficients = active_support_and_coefficients(misaligned)

    assert support == ["a", "c"]
    assert coefficients is None


@pytest.mark.unit
def test_active_support_short_coefficients_dense_also_drop() -> None:
    misaligned = EvaluationResult(
        mse=0.1,
        nmse=0.2,
        r2=0.9,
        is_valid=True,
        coefficients=torch.tensor([1.0, 2.0]),
        selected_indices=None,
        terms=["a", "b", "c"],
        expression="a + 2b",
    )

    support, coefficients = active_support_and_coefficients(misaligned)

    assert support == ["a", "b", "c"]
    assert coefficients is None
