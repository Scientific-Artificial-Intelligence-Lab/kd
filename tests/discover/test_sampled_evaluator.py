
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch




import kd.search.discover.tokens.prior
from kd.core.linear_solve.least_squares import (
    LeastSquaresSolver,
)
from kd.search.discover.runners.sampled_evaluator import SampledEvaluator

if TYPE_CHECKING:
    from kd.core.evaluator import Evaluator






@dataclass
class _FakeExecResult:
    value: torch.Tensor


class _FakeExecutor:

    def __init__(self, term_columns: dict[str, torch.Tensor]) -> None:
        self._cols = term_columns
        self.registry = None

    def execute(self, term: str, context: Any) -> _FakeExecResult:
        if term not in self._cols:
            raise KeyError(f"unknown term: {term}")
        return _FakeExecResult(value=self._cols[term])


class _FakeBase:

    def __init__(
        self,
        lhs: torch.Tensor,
        term_columns: dict[str, torch.Tensor],
    ) -> None:
        self.lhs_target = lhs
        self.executor = _FakeExecutor(term_columns)
        self.solver = LeastSquaresSolver()
        self.context = None


def _make_evaluator(
    lhs: torch.Tensor,
    term_columns: dict[str, torch.Tensor],
    *,
    rank_check: bool,
    n_points: int | None = None,
) -> SampledEvaluator:



    base = cast("Evaluator", _FakeBase(lhs=lhs, term_columns=term_columns))
    if n_points is None:
        n_points = lhs.shape[0]
    indices = torch.arange(n_points, dtype=torch.int64)
    return SampledEvaluator(base, indices, rank_check=rank_check)







@pytest.mark.unit
def test_default_no_rank_check_back_compat() -> None:
    n = 50
    lhs = torch.randn(n, dtype=torch.float64)
    a = torch.randn(n, dtype=torch.float64)
    cols = {"a": a, "b": 2.0 * a}
    evaluator = SampledEvaluator(
        cast("Evaluator", _FakeBase(lhs, cols)),
        torch.arange(n, dtype=torch.int64)
    )


    result = evaluator.evaluate_terms(["a", "b"])
    assert result.is_valid


@pytest.mark.unit
def test_default_rank_check_kwarg_is_false() -> None:
    n = 30
    lhs = torch.randn(n, dtype=torch.float64)
    a = torch.randn(n, dtype=torch.float64)
    cols = {"a": a, "b": a.clone()}
    evaluator = _make_evaluator(lhs, cols, rank_check=False)
    result = evaluator.evaluate_terms(["a", "b"])
    assert result.is_valid







@pytest.mark.unit
def test_rank_check_raises_on_identical_columns() -> None:
    n = 40
    lhs = torch.randn(n, dtype=torch.float64)
    a = torch.randn(n, dtype=torch.float64)
    cols = {"col_x": a, "col_y": a.clone()}
    evaluator = _make_evaluator(lhs, cols, rank_check=True)
    result = evaluator.evaluate_terms(["col_x", "col_y"])
    assert not result.is_valid
    assert "rank" in (result.error_message or "").lower()


@pytest.mark.unit
def test_rank_check_raises_on_collinear_columns() -> None:
    n = 40
    lhs = torch.randn(n, dtype=torch.float64)
    a = torch.randn(n, dtype=torch.float64)
    cols = {"col_x": a, "col_y": -3.7 * a}
    evaluator = _make_evaluator(lhs, cols, rank_check=True)
    result = evaluator.evaluate_terms(["col_x", "col_y"])
    assert not result.is_valid
    assert "rank" in (result.error_message or "").lower()


@pytest.mark.unit
def test_rank_check_raises_on_near_collinear_columns_at_machine_precision() -> None:
    n = 100
    lhs = torch.randn(n, dtype=torch.float64)
    a = torch.randn(n, dtype=torch.float64)

    perturbation = 1e-13 * torch.randn(n, dtype=torch.float64)
    cols = {"diff2_x": a, "diff2_y": a + perturbation}
    evaluator = _make_evaluator(lhs, cols, rank_check=True)
    result = evaluator.evaluate_terms(["diff2_x", "diff2_y"])
    assert not result.is_valid
    assert "rank" in (result.error_message or "").lower()







@pytest.mark.unit
def test_rank_check_passes_on_full_rank_theta() -> None:
    n = 100
    torch.manual_seed(0)
    lhs = torch.randn(n, dtype=torch.float64)
    cols = {
        "a": torch.randn(n, dtype=torch.float64),
        "b": torch.randn(n, dtype=torch.float64),
        "c": torch.randn(n, dtype=torch.float64),
    }
    evaluator = _make_evaluator(lhs, cols, rank_check=True)
    result = evaluator.evaluate_terms(["a", "b", "c"])
    assert result.is_valid


@pytest.mark.unit
def test_rank_check_passes_on_single_column() -> None:
    n = 50
    lhs = torch.randn(n, dtype=torch.float64)
    cols = {"a": torch.randn(n, dtype=torch.float64)}
    evaluator = _make_evaluator(lhs, cols, rank_check=True)
    result = evaluator.evaluate_terms(["a"])
    assert result.is_valid







@pytest.mark.unit
def test_build_theta_matrix_raises_on_rank_deficient_with_check() -> None:
    n = 40
    lhs = torch.randn(n, dtype=torch.float64)
    a = torch.randn(n, dtype=torch.float64)
    cols = {"x": a, "y": a.clone()}
    evaluator = _make_evaluator(lhs, cols, rank_check=True)
    with pytest.raises(ValueError, match="rank"):
        evaluator.build_theta_matrix(["x", "y"])


@pytest.mark.unit
def test_build_theta_matrix_succeeds_when_check_disabled() -> None:
    n = 40
    lhs = torch.randn(n, dtype=torch.float64)
    a = torch.randn(n, dtype=torch.float64)
    cols = {"x": a, "y": a.clone()}
    evaluator = _make_evaluator(lhs, cols, rank_check=False)
    theta, terms = evaluator.build_theta_matrix(["x", "y"])
    assert theta.shape == (n, 2)
    assert terms == ["x", "y"]







def _make_evaluator_with_magnitude(
    lhs: torch.Tensor,
    term_columns: dict[str, torch.Tensor],
    *,
    magnitude_filter: bool,
) -> SampledEvaluator:

    base = cast("Evaluator", _FakeBase(lhs=lhs, term_columns=term_columns))
    indices = torch.arange(lhs.shape[0], dtype=torch.int64)
    return SampledEvaluator(base, indices, magnitude_filter=magnitude_filter)


@pytest.mark.unit
def test_magnitude_filter_default_off_back_compat() -> None:
    n = 50
    torch.manual_seed(0)
    a = torch.randn(n, dtype=torch.float64)

    lhs = 1e-8 * a
    cols = {"a": a}
    evaluator = _make_evaluator_with_magnitude(
        lhs, cols, magnitude_filter=False,
    )
    result = evaluator.evaluate_terms(["a"])
    assert result.is_valid


@pytest.mark.unit
def test_magnitude_filter_rejects_below_min_threshold() -> None:
    n = 100
    torch.manual_seed(0)
    a = torch.randn(n, dtype=torch.float64)

    lhs = 1e-8 * a
    cols = {"a": a}
    evaluator = _make_evaluator_with_magnitude(
        lhs, cols, magnitude_filter=True,
    )
    result = evaluator.evaluate_terms(["a"])
    assert not result.is_valid
    assert "magnitude" in (result.error_message or "").lower()


@pytest.mark.unit
def test_magnitude_filter_rejects_above_max_threshold() -> None:
    n = 100
    torch.manual_seed(1)
    a = torch.randn(n, dtype=torch.float64)

    lhs = 1e6 * a
    cols = {"a": a}
    evaluator = _make_evaluator_with_magnitude(
        lhs, cols, magnitude_filter=True,
    )
    result = evaluator.evaluate_terms(["a"])
    assert not result.is_valid
    assert "magnitude" in (result.error_message or "").lower()


@pytest.mark.unit
def test_magnitude_filter_passes_on_healthy_unit_coefficients() -> None:
    n = 100
    torch.manual_seed(2)
    a = torch.randn(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)

    lhs = 0.1 * a + 1.0 * b
    cols = {"a": a, "b": b}
    evaluator = _make_evaluator_with_magnitude(
        lhs, cols, magnitude_filter=True,
    )
    result = evaluator.evaluate_terms(["a", "b"])
    assert result.is_valid


@pytest.mark.unit
def test_magnitude_filter_rejects_when_one_of_many_coefs_is_tiny() -> None:
    n = 200
    torch.manual_seed(3)
    a = torch.randn(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)
    c = torch.randn(n, dtype=torch.float64)

    lhs = 1.0 * a + 1.0 * b + 1e-7 * c
    cols = {"a": a, "b": b, "c": c}
    evaluator = _make_evaluator_with_magnitude(
        lhs, cols, magnitude_filter=True,
    )
    result = evaluator.evaluate_terms(["a", "b", "c"])
    assert not result.is_valid
    assert "magnitude" in (result.error_message or "").lower()
