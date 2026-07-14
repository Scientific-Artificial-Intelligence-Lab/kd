
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import Evaluator
from kd.core.executor import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve import LeastSquaresSolver
from kd.data.derivatives.base import DerivativeProvider
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)

if TYPE_CHECKING:
    from kd.core.expr.registry import FunctionRegistry as _Reg


_N_X, _N_T = 4, 3


class _GenerationProvider(DerivativeProvider):

    def __init__(self, base: Tensor) -> None:
        self._base = base
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def _bump_generation(self) -> None:
        self._generation += 1

    def get_derivative(self, field: str, axis: str, order: int) -> Tensor:
        return self._base + float(self._generation)

    def diff(self, expression: Tensor, axis: str, order: int) -> Tensor:
        raise NotImplementedError

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        return [("u", "x", 1)]


class _GenerationReadCountingProvider(_GenerationProvider):

    def __init__(self, base: Tensor) -> None:
        super().__init__(base)
        self.generation_reads = 0

    @property
    def generation(self) -> int:
        self.generation_reads += 1
        return self._generation


class _SpyExecutor:

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


def _context(base: Tensor) -> tuple[ExecutionContext, _GenerationProvider, Tensor]:
    x = torch.linspace(0.0, 1.0, _N_X, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, _N_T, dtype=torch.float64)
    u = base.clone()
    dataset = PDEDataset(
        name="gen_cache",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )
    provider = _GenerationProvider(base)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    lhs = base.flatten() * 0.5
    return context, provider, lhs


def _make_evaluator(
    context: ExecutionContext, lhs: Tensor, executor: _SpyExecutor
) -> Evaluator:
    return Evaluator(
        executor=executor,
        solver=LeastSquaresSolver(),
        context=context,
        lhs=lhs,
        enable_term_cache=True,
    )


@pytest.mark.unit
def test_generation_bump_forces_recomputation() -> None:
    base = torch.arange(_N_X * _N_T, dtype=torch.float64).reshape(_N_X, _N_T)
    context, provider, lhs = _context(base)
    spy = _SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(context, lhs, spy)

    with torch.no_grad():
        evaluator._build_theta(["u_x"])
        assert spy.calls["u_x"] == 1
        evaluator._build_theta(["u_x"])
        assert spy.calls["u_x"] == 1, "stable-generation re-eval must hit the cache"

        provider._bump_generation()
        evaluator._build_theta(["u_x"])

    assert spy.calls["u_x"] == 2, (
        "generation bump must miss the stale column and re-execute"
    )


@pytest.mark.numerical
def test_generation_bump_serves_fresh_column_not_stale() -> None:
    base = torch.arange(_N_X * _N_T, dtype=torch.float64).reshape(_N_X, _N_T)
    context, provider, lhs = _context(base)
    spy = _SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(context, lhs, spy)

    with torch.no_grad():
        theta_gen0, _ = evaluator._build_theta(["u_x"])
        col_gen0 = theta_gen0[:, 0].clone()

        provider._bump_generation()
        theta_gen1, _ = evaluator._build_theta(["u_x"])
        col_gen1 = theta_gen1[:, 0].clone()

    expected_gen1 = (base + 1.0).flatten()
    torch.testing.assert_close(col_gen0, base.flatten(), rtol=0, atol=0)
    torch.testing.assert_close(col_gen1, expected_gen1, rtol=0, atol=0)
    assert not torch.equal(col_gen0, col_gen1), "stale column reused after retrain"


@pytest.mark.unit
def test_generation_read_once_per_build_theta_not_per_term() -> None:
    base = torch.arange(_N_X * _N_T, dtype=torch.float64).reshape(_N_X, _N_T)
    x = torch.linspace(0.0, 1.0, _N_X, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, _N_T, dtype=torch.float64)
    dataset = PDEDataset(
        name="gen_read_count",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=base.clone())},
        lhs_field="u",
        lhs_axis="t",
    )
    provider = _GenerationReadCountingProvider(base)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    spy = _SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(context, base.flatten() * 0.5, spy)

    with torch.no_grad():

        evaluator._build_theta(["u_x", "u_xx", "u_xxx"])

    assert provider.generation_reads == 1, (
        "generation must be read once per _build_theta call, not once per term"
    )


@pytest.mark.numerical
def test_provider_object_swap_invalidates_memo() -> None:
    base_a = torch.arange(_N_X * _N_T, dtype=torch.float64).reshape(_N_X, _N_T)
    base_b = base_a + 100.0
    context, provider_a, lhs = _context(base_a)
    provider_b = _GenerationProvider(base_b)
    assert provider_a.generation == 0 and provider_b.generation == 0
    spy = _SpyExecutor(PythonExecutor(FunctionRegistry.create_default()))
    evaluator = _make_evaluator(context, lhs, spy)

    with torch.no_grad():
        theta_a, _ = evaluator._build_theta(["u_x"])
        col_a = theta_a[:, 0].clone()
        assert spy.calls["u_x"] == 1


        context.derivative_provider = provider_b
        theta_b, _ = evaluator._build_theta(["u_x"])
        col_b = theta_b[:, 0].clone()

    assert spy.calls["u_x"] == 2, (
        "provider object swap must re-execute; stale memo served"
    )
    torch.testing.assert_close(col_a, base_a.flatten(), rtol=0, atol=0)
    torch.testing.assert_close(col_b, base_b.flatten(), rtol=0, atol=0)
    assert not torch.equal(col_a, col_b), "stale provider-A column served after swap"
