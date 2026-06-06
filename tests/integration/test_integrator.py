
from __future__ import annotations

import pytest
import sympy
import torch

from kd.core.integrator import IntegrationResult, integrate_pde
from kd.data.schema import DataTopology, PDEDataset
from kd.data.synthetic import (
    generate_advection_data,
    generate_burgers_data,
    generate_diffusion_data,
)






@pytest.fixture
def advection_dataset() -> PDEDataset:
    return generate_advection_data(
        speeds=(1.0,),
        waves=(1.0,),
        grid_sizes=(128,),
        nt=51,
        seed=42,
    )


@pytest.fixture
def diffusion_dataset() -> PDEDataset:
    return generate_diffusion_data(
        alpha=0.1,
        waves=(1.0,),
        grid_sizes=(128,),
        nt=51,
        seed=42,
    )


@pytest.fixture
def burgers_dataset() -> PDEDataset:
    return generate_burgers_data(
        nx=128,
        nt=51,
        nu=0.1,
        seed=42,
    )







class TestAdvectionIntegration:

    def test_advection_returns_success(self, advection_dataset: PDEDataset) -> None:

        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert isinstance(result, IntegrationResult)
        assert result.success is True
        assert result.predicted_field is not None

    def test_advection_output_is_tensor(self, advection_dataset: PDEDataset) -> None:
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert isinstance(result.predicted_field, torch.Tensor)

    def test_advection_output_shape(self, advection_dataset: PDEDataset) -> None:
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert result.predicted_field is not None
        assert result.predicted_field.shape == advection_dataset.get_shape()

    def test_advection_correlation_with_analytical(
        self, advection_dataset: PDEDataset
    ) -> None:
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert result.predicted_field is not None

        analytical = advection_dataset.get_field("u").to(torch.float64)
        predicted = result.predicted_field.to(torch.float64)


        a_flat = analytical.flatten()
        p_flat = predicted.flatten()
        a_centered = a_flat - a_flat.mean()
        p_centered = p_flat - p_flat.mean()
        corr = (a_centered * p_centered).sum() / (a_centered.norm() * p_centered.norm())
        assert corr.item() > 0.95, f"Correlation {corr.item():.4f} < 0.95"

    def test_advection_output_is_finite(self, advection_dataset: PDEDataset) -> None:
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert result.predicted_field is not None
        assert torch.isfinite(result.predicted_field).all()







class TestDiffusionIntegration:

    def test_diffusion_returns_success(self, diffusion_dataset: PDEDataset) -> None:
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert isinstance(result, IntegrationResult)
        assert result.success is True
        assert result.predicted_field is not None

    def test_diffusion_output_shape(self, diffusion_dataset: PDEDataset) -> None:
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert result.success
        assert result.predicted_field is not None
        assert result.predicted_field.shape == diffusion_dataset.get_shape()

    def test_diffusion_correlation_with_analytical(
        self, diffusion_dataset: PDEDataset
    ) -> None:
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert result.success
        assert result.predicted_field is not None

        analytical = diffusion_dataset.get_field("u").to(torch.float64)
        predicted = result.predicted_field.to(torch.float64)

        a_flat = analytical.flatten()
        p_flat = predicted.flatten()
        a_centered = a_flat - a_flat.mean()
        p_centered = p_flat - p_flat.mean()
        corr = (a_centered * p_centered).sum() / (a_centered.norm() * p_centered.norm())
        assert corr.item() > 0.95, f"Correlation {corr.item():.4f} < 0.95"

    def test_diffusion_decays_over_time(self, diffusion_dataset: PDEDataset) -> None:
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert result.success
        assert result.predicted_field is not None

        field = result.predicted_field.to(torch.float64)
        norm_first = field[:, 0].norm()
        norm_last = field[:, -1].norm()
        assert norm_last <= norm_first * 1.05







class TestBurgersIntegration:

    def test_burgers_does_not_diverge(self, burgers_dataset: PDEDataset) -> None:
        u = sympy.Symbol("u")
        u_x = sympy.Symbol("u_x")
        u_xx = sympy.Symbol("u_xx")
        rhs = -u * u_x + 0.1 * u_xx
        result = integrate_pde(rhs, burgers_dataset)
        assert isinstance(result, IntegrationResult)
        assert result.success, f"Burgers integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        assert torch.isfinite(result.predicted_field).all()

    def test_burgers_output_shape(self, burgers_dataset: PDEDataset) -> None:
        u = sympy.Symbol("u")
        u_x = sympy.Symbol("u_x")
        u_xx = sympy.Symbol("u_xx")
        rhs = -u * u_x + 0.1 * u_xx
        result = integrate_pde(rhs, burgers_dataset)
        assert result.success, f"Burgers integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        assert result.predicted_field.shape == burgers_dataset.get_shape()







class TestScatteredTopology:

    def test_scattered_returns_failure(self) -> None:
        nx, nt = 32, 10
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
        u = torch.randn(nx, nt, dtype=torch.float64)
        from kd.data.schema import AxisInfo, FieldData, TaskType

        dataset = PDEDataset(
            name="scattered-reject",
            task_type=TaskType.PDE,
            topology=DataTopology.SCATTERED,
            axes={
                "x": AxisInfo(name="x", values=x),
                "t": AxisInfo(name="t", values=t),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        rhs = sympy.Symbol("u_x")
        result = integrate_pde(rhs, dataset)
        assert result.success is False
        assert result.predicted_field is None
        assert "SCATTERED" in result.warning or "scattered" in result.warning.lower()







class TestWrongEquations:

    def test_wrong_equation_does_not_crash(self, advection_dataset: PDEDataset) -> None:

        u_xx = sympy.Symbol("u_xx")
        rhs = 100.0 * u_xx
        result = integrate_pde(rhs, advection_dataset)
        assert isinstance(result, IntegrationResult)


    def test_unknown_symbols_in_rhs(self, advection_dataset: PDEDataset) -> None:
        u_y = sympy.Symbol("u_y")
        rhs = u_y
        result = integrate_pde(rhs, advection_dataset)
        assert isinstance(result, IntegrationResult)








class TestInitialCondition:

    def test_ic_matches_at_t0(self, advection_dataset: PDEDataset) -> None:
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert result.predicted_field is not None

        ic_dataset = advection_dataset.get_field("u")[:, 0].to(torch.float64)
        ic_predicted = result.predicted_field[:, 0].to(torch.float64)
        torch.testing.assert_close(ic_predicted, ic_dataset, rtol=1e-5, atol=1e-8)

    def test_diffusion_ic_matches_at_t0(self, diffusion_dataset: PDEDataset) -> None:
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert result.success
        assert result.predicted_field is not None

        ic_dataset = diffusion_dataset.get_field("u")[:, 0].to(torch.float64)
        ic_predicted = result.predicted_field[:, 0].to(torch.float64)
        torch.testing.assert_close(ic_predicted, ic_dataset, rtol=1e-5, atol=1e-8)
