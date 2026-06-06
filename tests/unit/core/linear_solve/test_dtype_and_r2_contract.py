
from __future__ import annotations

import pytest
import torch

from kd.core.linear_solve import (
    LeastSquaresSolver,
    SolveResult,
    SparseSolver,
    STRidgeSolver,
    SVDNullSpaceSolver,
)


def _ill_conditioned_float32_system() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, 40, dtype=torch.float32)
    theta = torch.stack([torch.ones_like(x), x, x + 1e-5 * x**2], dim=1)
    coef = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32)
    return theta, theta @ coef


@pytest.fixture(
    params=[
        LeastSquaresSolver(),
        STRidgeSolver(tol=0.0, normalize=0),
        SVDNullSpaceSolver(eps=1e-12),
    ],
    ids=["lstsq", "stridge", "svd_null_space"],
)
def solver(request: pytest.FixtureRequest) -> SparseSolver:
    return request.param


@pytest.mark.unit
def test_float32_coefficients_return_float32_but_match_float64_internal_solve(
    solver: SparseSolver,
) -> None:
    theta, lhs = _ill_conditioned_float32_system()

    result = solver.solve(theta, lhs)
    expected = solver.solve(theta.to(torch.float64), lhs.to(torch.float64))

    assert result.is_valid is True
    assert expected.is_valid is True
    assert result.coefficients.dtype == torch.float32
    torch.testing.assert_close(
        result.coefficients,
        expected.coefficients.to(torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.unit
def test_r2_is_consistent_for_constant_target_perfect_fit() -> None:
    gen = torch.Generator().manual_seed(7)
    theta = torch.column_stack(
        [
            torch.ones(20, dtype=torch.float64),
            torch.randn(20, dtype=torch.float64, generator=gen),
        ]
    )
    lhs = torch.full((20,), 3.0, dtype=torch.float64)

    solvers: list[SparseSolver] = [
        LeastSquaresSolver(),
        STRidgeSolver(tol=0.0, normalize=0),
        SVDNullSpaceSolver(eps=1e-12),
    ]

    results: list[SolveResult] = [solver.solve(theta, lhs) for solver in solvers]

    assert all(result.is_valid for result in results)
    assert [result.r2 for result in results] == pytest.approx([1.0, 1.0, 1.0])











def _well_conditioned_float32_system() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float32)
    theta = torch.stack([torch.ones_like(x), x, x**2], dim=1)
    coef = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32)
    return theta, theta @ coef


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS backend not available on this machine",
)
def test_solver_accepts_mps_float32_inputs(solver: SparseSolver) -> None:
    mps_device = torch.device("mps")
    theta_cpu, lhs_cpu = _well_conditioned_float32_system()
    theta_mps = theta_cpu.to(device=mps_device)
    lhs_mps = lhs_cpu.to(device=mps_device)

    result = solver.solve(theta_mps, lhs_mps)

    assert result.is_valid is True
    assert result.coefficients.device.type == "mps"
    assert result.coefficients.dtype == torch.float32


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS backend not available on this machine",
)
def test_solver_mps_result_matches_cpu_float64(solver: SparseSolver) -> None:
    mps_device = torch.device("mps")
    theta_cpu, lhs_cpu = _well_conditioned_float32_system()
    theta_mps = theta_cpu.to(device=mps_device)
    lhs_mps = lhs_cpu.to(device=mps_device)

    expected = solver.solve(theta_cpu.to(torch.float64), lhs_cpu.to(torch.float64))
    actual = solver.solve(theta_mps, lhs_mps)

    torch.testing.assert_close(
        actual.coefficients.detach().cpu().to(torch.float64),
        expected.coefficients.detach().cpu(),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS backend not available on this machine",
)
def test_compute_r2_and_squared_residual_handle_mps() -> None:
    from kd.core.linear_solve._helpers import compute_r2, squared_residual

    mps_device = torch.device("mps")
    theta_cpu, lhs_cpu = _well_conditioned_float32_system()
    theta_mps = theta_cpu.to(device=mps_device)
    lhs_mps = lhs_cpu.to(device=mps_device)
    coef_mps = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32, device=mps_device)


    r2 = compute_r2(theta_mps, coef_mps, lhs_mps)
    res = squared_residual(theta_mps, coef_mps, lhs_mps)

    assert r2 == pytest.approx(1.0, abs=1e-4)
    assert res == pytest.approx(0.0, abs=1e-4)
