
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.executor import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve import LeastSquaresSolver
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)

if TYPE_CHECKING:
    from kd.core.executor.context import ExecutionContext as _Ctx







_TERM_SEQUENCE: tuple[list[str], ...] = (
    ["u_x", "u_xx"],
    ["u_xx", "u"],
    ["u_x", "u_x"],
    ["u", "u_x", "u_xx"],
)









class SpyExecutor:

    def __init__(self, inner: PythonExecutor) -> None:
        self._inner = inner
        self.calls: Counter[str] = Counter()

    def execute(
        self, code: str, context: ExecutionContext, force_diff_path: bool = False
    ):
        self.calls[code] += 1
        return self._inner.execute(code, context, force_diff_path)

    @property
    def registry(self) -> FunctionRegistry:
        return self._inner.registry








def _finite_diff_context() -> tuple[ExecutionContext, Tensor]:
    n_x, n_t = 32, 16
    x = torch.linspace(0.0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt)
    dataset = PDEDataset(
        name="fd_2d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    lhs = context.get_derivative("u", "t", 1)
    return context, lhs


class _QuadTModel(nn.Module):

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        return x**2 + t**2


def _autograd_context() -> tuple[ExecutionContext, Tensor]:
    n = 24
    x_1d = torch.linspace(0.1, 2.0, n, dtype=torch.float64)
    t_1d = torch.linspace(0.1, 1.0, n // 2, dtype=torch.float64)
    xx, tt = torch.meshgrid(x_1d, t_1d, indexing="ij")
    u_grid = xx**2 + tt**2
    x_flat = xx.reshape(-1).clone().detach().requires_grad_(True)
    t_flat = tt.reshape(-1).clone().detach().requires_grad_(True)
    dataset = PDEDataset(
        name="ag_2d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x_1d),
            "t": AxisInfo(name="t", values=t_1d),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_grid)},
        lhs_field="u",
        lhs_axis="t",
    )
    model = _QuadTModel().double()
    provider = AutogradProvider(
        model=model, coords={"x": x_flat, "t": t_flat}, dataset=dataset
    )
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    lhs = provider.get_derivative("u", "t", order=1)
    return context, lhs


_PROVIDER_BUILDERS = {
    "finite_diff": _finite_diff_context,
    "autograd": _autograd_context,
}


def _make_evaluator(
    context: ExecutionContext,
    lhs: Tensor,
    executor: PythonExecutor | SpyExecutor,
    *,
    enable_term_cache: bool,
) -> Evaluator:
    return Evaluator(
        executor=executor,
        solver=LeastSquaresSolver(),
        context=context,
        lhs=lhs,
        enable_term_cache=enable_term_cache,
    )







def _floats_identical(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _tensors_identical(a: Tensor | None, b: Tensor | None) -> bool:
    if a is None or b is None:
        return a is b
    return bool(torch.equal(a, b))


def _assert_result_bitwise_equal(
    on: EvaluationResult, off: EvaluationResult, *, label: str
) -> None:
    assert on.is_valid == off.is_valid, f"{label}: is_valid differs"
    assert _floats_identical(on.mse, off.mse), f"{label}: mse differs"
    assert _floats_identical(on.nmse, off.nmse), f"{label}: nmse differs"
    assert _floats_identical(on.r2, off.r2), f"{label}: r2 differs"
    assert _floats_identical(on.score, off.score), f"{label}: aic differs"
    assert _tensors_identical(
        on.coefficients, off.coefficients
    ), f"{label}: coefficients differ"
    assert _tensors_identical(on.residuals, off.residuals), f"{label}: residuals differ"








@pytest.fixture
def deterministic_lstsq() -> Iterator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:



        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)







@pytest.mark.equivalence
@pytest.mark.parametrize("provider_kind", ["finite_diff", "autograd"])
def test_cache_on_off_bitwise_identical_over_sequence(
    provider_kind: str, deterministic_lstsq: None
) -> None:
    context, lhs = _PROVIDER_BUILDERS[provider_kind]()
    reg = FunctionRegistry.create_default()
    eval_on = _make_evaluator(
        context, lhs, PythonExecutor(reg), enable_term_cache=True
    )
    eval_off = _make_evaluator(
        context, lhs, PythonExecutor(reg), enable_term_cache=False
    )

    for i, terms in enumerate(_TERM_SEQUENCE):
        res_on = eval_on.evaluate_terms(list(terms))
        res_off = eval_off.evaluate_terms(list(terms))
        _assert_result_bitwise_equal(
            res_on, res_off, label=f"{provider_kind} call#{i} terms={terms}"
        )


@pytest.mark.equivalence
@pytest.mark.parametrize("provider_kind", ["finite_diff", "autograd"])
def test_build_theta_columns_bitwise_identical(provider_kind: str) -> None:
    context, lhs = _PROVIDER_BUILDERS[provider_kind]()
    reg = FunctionRegistry.create_default()
    eval_on = _make_evaluator(
        context, lhs, PythonExecutor(reg), enable_term_cache=True
    )
    eval_off = _make_evaluator(
        context, lhs, PythonExecutor(reg), enable_term_cache=False
    )

    for terms in _TERM_SEQUENCE:
        with torch.no_grad():
            theta_on, valid_on = eval_on._build_theta(list(terms))
            theta_off, valid_off = eval_off._build_theta(list(terms))
        assert valid_on == valid_off
        assert torch.equal(theta_on, theta_off), f"theta differs for terms={terms}"







@pytest.mark.unit
def test_cache_on_reeval_triggers_zero_executions() -> None:
    context, lhs = _finite_diff_context()
    spy = SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(context, lhs, spy, enable_term_cache=True)

    evaluator.evaluate_terms(["u_x", "u_xx"])
    first = Counter(spy.calls)
    assert first["u_x"] == 1 and first["u_xx"] == 1

    evaluator.evaluate_terms(["u_x", "u_xx"])
    assert spy.calls["u_x"] == 1, "cached term re-executed"
    assert spy.calls["u_xx"] == 1, "cached term re-executed"


@pytest.mark.unit
def test_cache_on_duplicate_in_one_list_executes_once() -> None:
    context, lhs = _finite_diff_context()
    spy = SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(context, lhs, spy, enable_term_cache=True)

    evaluator.evaluate_terms(["u_x", "u_x"])
    assert spy.calls["u_x"] == 1


@pytest.mark.unit
def test_cache_off_counts_scale_with_occurrences() -> None:
    context, lhs = _finite_diff_context()
    spy = SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(context, lhs, spy, enable_term_cache=False)

    evaluator.evaluate_terms(["u_x", "u_x"])
    evaluator.evaluate_terms(["u_x"])
    assert spy.calls["u_x"] == 3


@pytest.mark.unit
def test_invalidate_forces_reexecution() -> None:
    context, lhs = _finite_diff_context()
    spy = SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(context, lhs, spy, enable_term_cache=True)

    evaluator.evaluate_terms(["u_x"])
    assert spy.calls["u_x"] == 1
    evaluator.invalidate_term_cache()
    evaluator.evaluate_terms(["u_x"])
    assert spy.calls["u_x"] == 2, "invalidate did not force re-execution"







@pytest.mark.numerical
def test_execution_error_not_cached_sibling_still_cached() -> None:
    context, lhs = _finite_diff_context()
    spy = SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(context, lhs, spy, enable_term_cache=True)


    terms = ["u_x", "does_not_exist"]
    r1 = evaluator.evaluate_terms(terms, skip_invalid=True)
    r2 = evaluator.evaluate_terms(terms, skip_invalid=True)

    assert r1.is_valid and r2.is_valid
    assert spy.calls["u_x"] == 1, "valid sibling should be cached"
    assert spy.calls["does_not_exist"] == 2, "failing term must NOT be cached"


@pytest.mark.numerical
@pytest.mark.parametrize("enable_term_cache", [True, False])
@pytest.mark.parametrize("skip_invalid", [True, False])
def test_error_behavior_identical_regardless_of_cache(
    enable_term_cache: bool, skip_invalid: bool
) -> None:
    context, lhs = _finite_diff_context()
    spy = SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(
        context, lhs, spy, enable_term_cache=enable_term_cache
    )

    result = evaluator.evaluate_terms(["does_not_exist"], skip_invalid=skip_invalid)
    assert not result.is_valid

    evaluator.evaluate_terms(["does_not_exist"], skip_invalid=skip_invalid)
    assert spy.calls["does_not_exist"] == 2


def _nan_registry() -> FunctionRegistry:
    reg = FunctionRegistry.create_default()
    reg.register("nanop", lambda a: a * float("nan"), arity=1)
    return reg


@pytest.mark.numerical
@pytest.mark.parametrize("enable_term_cache", [True, False])
def test_nan_column_skipped_identically(enable_term_cache: bool) -> None:
    context, lhs = _finite_diff_context()
    evaluator = _make_evaluator(
        context, lhs, PythonExecutor(_nan_registry()),
        enable_term_cache=enable_term_cache,
    )

    result = evaluator.evaluate_terms(["u_x", "nanop(u)"], skip_invalid=True)
    assert result.is_valid
    assert result.terms == ["u_x"], "NaN term must be skipped, sibling kept"


@pytest.mark.numerical
@pytest.mark.parametrize("enable_term_cache", [True, False])
def test_all_zero_column_skipped_identically(enable_term_cache: bool) -> None:
    context, lhs = _finite_diff_context()
    evaluator = _make_evaluator(
        context, lhs, PythonExecutor(FunctionRegistry.create_default()),
        enable_term_cache=enable_term_cache,
    )

    result = evaluator.evaluate_terms(["u_x", "sub(u, u)"], skip_invalid=True)
    assert result.is_valid
    assert result.terms == ["u_x"], "all-zero term must be skipped"







@pytest.mark.numerical
def test_cached_autograd_column_is_detached() -> None:
    context, lhs = _autograd_context()
    evaluator = _make_evaluator(
        context, lhs, PythonExecutor(FunctionRegistry.create_default()),
        enable_term_cache=True,
    )

    evaluator.evaluate_terms(["u_x"])
    cached = evaluator._term_cache.get("u_x")
    assert cached is not None, "term was not cached"
    assert cached.requires_grad is False, "cached column keeps an autograd graph"
    assert torch.isfinite(cached).all()







@pytest.mark.smoke
def test_default_cache_enabled_and_invalidate_exists() -> None:
    context, lhs = _finite_diff_context()
    evaluator = Evaluator(
        executor=PythonExecutor(FunctionRegistry.create_default()),
        solver=LeastSquaresSolver(),
        context=context,
        lhs=lhs,
    )
    assert hasattr(evaluator, "invalidate_term_cache")
    evaluator.invalidate_term_cache()
