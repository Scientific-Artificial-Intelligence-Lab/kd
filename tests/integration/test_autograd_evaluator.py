
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd.core.evaluator import Evaluator
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






class LinearFieldModel(nn.Module):

    def forward(self, *, x: Tensor) -> Tensor:
        return 2.0 * x


class QuadraticFieldModel(nn.Module):

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        return x**2 + t






_N = 32


@pytest.fixture
def registry() -> FunctionRegistry:
    return FunctionRegistry.create_default()


@pytest.fixture
def executor(registry: FunctionRegistry) -> PythonExecutor:
    return PythonExecutor(registry)


@pytest.fixture
def solver() -> LeastSquaresSolver:
    return LeastSquaresSolver()


def _make_1d_autograd_components() -> tuple[nn.Module, dict[str, Tensor], PDEDataset]:
    x_vals = torch.linspace(0.1, 2.0, _N, dtype=torch.float64)
    x = x_vals.clone().detach().requires_grad_(True)

    dataset = PDEDataset(
        name="linear_1d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x_vals)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=2.0 * x_vals)},
        lhs_field="u",
        lhs_axis="x",
    )
    model = LinearFieldModel().double()
    coords = {"x": x}
    return model, coords, dataset


def _make_2d_autograd_components() -> tuple[nn.Module, dict[str, Tensor], PDEDataset]:
    x_1d = torch.linspace(0.1, 2.0, _N, dtype=torch.float64)
    t_1d = torch.linspace(0.0, 1.0, _N // 2, dtype=torch.float64)
    X, T = torch.meshgrid(x_1d, t_1d, indexing="ij")
    u_grid = X**2 + T

    x_flat = X.reshape(-1).clone().detach().requires_grad_(True)
    t_flat = T.reshape(-1).clone().detach().requires_grad_(True)

    dataset = PDEDataset(
        name="quad_2d",
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
    model = QuadraticFieldModel().double()
    coords = {"x": x_flat, "t": t_flat}
    return model, coords, dataset







class TestAutogradEvaluatorIntegration:

    @pytest.mark.integration
    def test_autograd_evaluator_computes_derivative(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        model, coords, dataset = _make_2d_autograd_components()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        context = ExecutionContext(dataset=dataset, derivative_provider=provider)


        lhs = provider.get_derivative("u", "t", order=1)
        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )

        result = evaluator.evaluate_terms(["u_x"])


        assert result.is_valid, (
            f"Evaluator + AutogradProvider failed: {result.error_message}"
        )
        assert result.coefficients is not None
        assert torch.isfinite(result.coefficients).all(), "Coefficients must be finite"

    @pytest.mark.integration
    def test_autograd_evaluator_no_grad_does_not_block(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        model, coords, dataset = _make_2d_autograd_components()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        context = ExecutionContext(dataset=dataset, derivative_provider=provider)


        lhs = provider.get_derivative("u", "t", order=1)
        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )



        result = evaluator.evaluate_terms(["u_xx"])

        assert result.is_valid, f"no_grad blocked autograd: {result.error_message}"
        assert result.coefficients is not None





        torch.testing.assert_close(
            result.coefficients.flatten(),
            torch.tensor([0.5], dtype=result.coefficients.dtype),
            rtol=1e-4,
            atol=1e-6,
        )

    @pytest.mark.integration
    def test_finitediff_evaluator_still_works(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        n_x, n_t = 64, 32
        x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
        t = torch.linspace(0, 1, n_t, dtype=torch.float64)
        X, T = torch.meshgrid(x, t, indexing="ij")
        u = torch.sin(X) * torch.exp(-T)

        dataset = PDEDataset(
            name="sinexp",
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
        lhs = provider.get_derivative("u", "t", order=1)

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )


        result = evaluator.evaluate_terms(["u"])

        assert result.is_valid, f"FiniteDiff + Evaluator broke: {result.error_message}"
        assert result.coefficients is not None


        coeff = result.coefficients.flatten()[0].item()
        assert abs(coeff - (-1.0)) < 0.05, f"Expected coefficient ~ -1.0, got {coeff}"


        assert result.r2 > 0.95, f"R^2 too low: {result.r2}"


class TestAutogradCoordinateDerivative:

    @pytest.mark.integration
    def test_diff_of_coordinate_is_one_1d(self, executor: PythonExecutor) -> None:
        model, coords, dataset = _make_1d_autograd_components()
        provider = AutogradProvider(model, coords, dataset, max_order=1)
        ctx = ExecutionContext(dataset=dataset, derivative_provider=provider)
        result = executor.execute("diff_x(x)", ctx)
        torch.testing.assert_close(
            result.value,
            torch.ones(_N, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.integration
    def test_diff_of_coordinate_is_one_2d(self, executor: PythonExecutor) -> None:
        model, coords, dataset = _make_2d_autograd_components()
        provider = AutogradProvider(model, coords, dataset, max_order=1)
        ctx = ExecutionContext(dataset=dataset, derivative_provider=provider)

        n_pts = _N * (_N // 2)
        result = executor.execute("diff_x(x)", ctx)
        torch.testing.assert_close(
            result.value,
            torch.ones(n_pts, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        result_t = executor.execute("diff_t(t)", ctx)
        torch.testing.assert_close(
            result_t.value,
            torch.ones(n_pts, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
