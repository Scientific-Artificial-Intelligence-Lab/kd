
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
def test_r2_helpers_are_public_package_exports() -> None:
    from kd.core.linear_solve import (
        R2_EPS_RES,
        R2_EPS_TOT,
        compute_r2,
        r2_score,
    )

    assert R2_EPS_TOT == 1e-15
    assert R2_EPS_RES == 1e-10
    assert callable(compute_r2)
    assert callable(r2_score)


@pytest.mark.unit
def test_r2_score_matches_compute_r2_on_generic_system() -> None:
    from kd.core.linear_solve import compute_r2, r2_score

    theta, lhs = _ill_conditioned_float32_system()
    coef = torch.tensor([0.9, 2.1, -2.5], dtype=torch.float32)
    y_pred_64 = theta.to(torch.float64) @ coef.to(torch.float64)

    assert r2_score(y_pred_64, lhs) == pytest.approx(
        compute_r2(theta, coef, lhs), rel=1e-12
    )


@pytest.mark.unit
def test_r2_score_float32_constant_target_perfect_fit() -> None:
    from kd.core.linear_solve import r2_score

    lhs = torch.full((50,), 1.7, dtype=torch.float32)
    y_pred = lhs + 2e-7

    assert r2_score(y_pred, lhs) == pytest.approx(1.0)


@pytest.mark.unit
def test_r2_score_accepts_column_vector_prediction() -> None:
    from kd.core.linear_solve import compute_r2, r2_score

    gen = torch.Generator().manual_seed(11)
    lhs = torch.randn(30, dtype=torch.float64, generator=gen)
    y_pred = lhs + 0.1 * torch.randn(30, dtype=torch.float64, generator=gen)
    expected = r2_score(y_pred, lhs)

    assert r2_score(y_pred.unsqueeze(-1), lhs) == pytest.approx(expected, rel=1e-12)
    assert r2_score(y_pred.unsqueeze(-1), lhs.unsqueeze(-1)) == pytest.approx(
        expected, rel=1e-12
    )

    theta, lhs32 = _ill_conditioned_float32_system()
    coef = torch.tensor([0.9, 2.1, -2.5], dtype=torch.float32)
    assert compute_r2(theta, coef.unsqueeze(-1), lhs32) == pytest.approx(
        compute_r2(theta, coef, lhs32), rel=1e-12
    )


@pytest.mark.unit
def test_r2_score_rejects_broadcastable_length_mismatch() -> None:
    from kd.core.linear_solve import r2_score

    with pytest.raises(ValueError, match="shape"):
        r2_score(
            torch.ones(1, dtype=torch.float64),
            torch.ones(20, dtype=torch.float64),
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
