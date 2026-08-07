
import logging

import pytest
import torch

from kd.core.linear_solve.base import SolveResult, SparseSolver






def _make_sparse_system(
    n: int = 100,
    d: int = 5,
    true_coeffs: list[float] | None = None,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if true_coeffs is None:
        true_coeffs = [2.0, 0.0, 3.0, 0.0, 0.0]
    true_xi = torch.tensor(true_coeffs, dtype=torch.float64)
    assert true_xi.shape[0] == d, f"true_coeffs length {len(true_coeffs)} != d={d}"

    rng = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, d, dtype=torch.float64, generator=rng)
    y = theta @ true_xi
    if noise_std > 0:
        y = y + noise_std * torch.randn(n, dtype=torch.float64, generator=rng)
    return theta, y, true_xi


def _make_multiscale_system(
    n: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(123)
    x0 = 1e3 * torch.randn(n, 1, dtype=torch.float64, generator=gen)
    x1 = torch.randn(n, 1, dtype=torch.float64, generator=gen)
    x2 = 1e-3 * torch.randn(n, 1, dtype=torch.float64, generator=gen)
    theta = torch.cat([x0, x1, x2], dim=1)
    true_xi = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y = theta @ true_xi
    return theta, y, true_xi







@pytest.mark.unit
class TestSTRidgeSolver:


    def test_sparse_recovery_exact(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=42)
        result = solver.solve(theta, y)

        assert isinstance(result, SolveResult)
        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-5, atol=1e-8)

        assert result.selected_indices is not None
        assert set(result.selected_indices) == {0, 2}


    def test_lam_zero_uses_ols(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1, lam=0.0)
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=43)
        result = solver.solve(theta, y)


        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-5, atol=1e-8)

    def test_lam_nonzero_uses_ridge(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver_ols = STRidgeSolver(tol=0.1, lam=0.0)
        solver_ridge = STRidgeSolver(tol=0.1, lam=1.0)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=44)

        result_ols = solver_ols.solve(theta, y)
        result_ridge = solver_ridge.solve(theta, y)




        assert result_ols.selected_indices is not None
        assert result_ridge.selected_indices is not None


    def test_column_normalization_multiscale(self) -> None:
        from kd.core.linear_solve import STRidgeSolver



        solver = STRidgeSolver(tol=0.01, normalize=2)
        theta, y, true_xi = _make_multiscale_system(n=200)
        result = solver.solve(theta, y)



        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-4, atol=1e-6)
        assert result.selected_indices is not None
        assert set(result.selected_indices) == {0, 1, 2}

    def test_no_normalization(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1, normalize=0)

        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=45)
        result = solver.solve(theta, y)


        assert isinstance(result, SolveResult)
        assert result.coefficients.shape == (5,)


    def test_zero_column_no_crash(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(
            n=100, d=5, true_coeffs=[2.0, 0.0, 3.0, 0.0, 0.0], seed=46
        )

        theta[:, 1] = 0.0
        theta[:, 3] = 0.0

        result = solver.solve(theta, y)

        assert isinstance(result, SolveResult)
        assert result.coefficients.shape == (5,)

        assert abs(result.coefficients[1].item()) < 1e-10
        assert abs(result.coefficients[3].item()) < 1e-10

    def test_zero_column_selected_indices_map_to_original(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(
            n=100, d=5, true_coeffs=[2.0, 0.0, 3.0, 0.0, 0.0], seed=47
        )

        theta[:, 1] = 0.0
        theta[:, 3] = 0.0

        result = solver.solve(theta, y)



        assert result.selected_indices is not None
        assert set(result.selected_indices) == {0, 2}


    def test_biginds_empty_j0_returns_initial(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=1e6, lam=0.0, max_iter=10)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=48)
        result = solver.solve(theta, y)


        assert isinstance(result, SolveResult)
        assert result.coefficients.shape == (5,)


        assert result.coefficients.abs().sum().item() > 0

    def test_biginds_empty_after_j0_uses_previous(self) -> None:
        from kd.core.linear_solve import STRidgeSolver



        theta, y, _ = _make_sparse_system(
            n=100, d=5, true_coeffs=[0.5, 0.0, 0.5, 0.0, 0.0], seed=49
        )


        solver = STRidgeSolver(tol=0.4, max_iter=20)
        result = solver.solve(theta, y)


        assert isinstance(result, SolveResult)


    def test_early_stopping_support_unchanged(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1, max_iter=100)
        theta, y, true_xi = _make_sparse_system(n=200, d=5, seed=50)
        result = solver.solve(theta, y)



        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-5, atol=1e-8)


    def test_debiased_ols_final_coefficients(self) -> None:
        from kd.core.linear_solve import STRidgeSolver


        solver = STRidgeSolver(tol=0.1, lam=0.5, max_iter=10)
        theta, y, true_xi = _make_sparse_system(n=200, d=5, seed=51)
        result = solver.solve(theta, y)




        assert result.selected_indices is not None
        for idx in result.selected_indices:
            torch.testing.assert_close(
                result.coefficients[idx],
                true_xi[idx],
                rtol=1e-4,
                atol=1e-6,
            )


    def test_solve_with_tol_different_sparsity(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(
            n=200,
            d=10,
            true_coeffs=[3.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            seed=52,
        )

        result_low_tol = solver.solve_with_tol(theta, y, tol=0.01)
        result_high_tol = solver.solve_with_tol(theta, y, tol=2.5)


        assert result_low_tol.selected_indices is not None
        assert result_high_tol.selected_indices is not None
        assert len(result_low_tol.selected_indices) >= len(
            result_high_tol.selected_indices
        )

    def test_solve_with_tol_overrides_default(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=1e6)
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=53)


        result = solver.solve_with_tol(theta, y, tol=0.1)
        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-5, atol=1e-8)

    def test_solve_uses_default_tol(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=54)

        result_solve = solver.solve(theta, y)
        result_with_tol = solver.solve_with_tol(theta, y, tol=0.1)


        torch.testing.assert_close(
            result_solve.coefficients,
            result_with_tol.coefficients,
            rtol=1e-10,
            atol=1e-10,
        )


    def test_tol_zero_keeps_all_coefficients(self) -> None:
        rng = torch.Generator().manual_seed(200)
        theta = torch.randn(100, 5, generator=rng, dtype=torch.float64)
        true_w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        y = theta @ true_w

        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.0)
        result = solver.solve(theta, y)

        assert result.selected_indices is not None
        assert len(result.selected_indices) == 5, "tol=0 should select all terms"
        torch.testing.assert_close(result.coefficients, true_w, rtol=1e-5, atol=1e-8)


    def test_fat_matrix_no_crash(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        n, d = 10, 50
        rng = torch.Generator().manual_seed(55)
        theta = torch.randn(n, d, dtype=torch.float64, generator=rng)
        true_xi = torch.zeros(d, dtype=torch.float64)
        true_xi[0] = 2.0
        true_xi[5] = 3.0
        y = theta @ true_xi

        result = solver.solve(theta, y)

        assert isinstance(result, SolveResult)
        assert result.coefficients.shape == (d,)


    def test_nan_theta_raises(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.tensor([[1.0, float("nan")], [3.0, 4.0]], dtype=torch.float64)
        y = torch.tensor([1.0, 2.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="NaN"):
            solver.solve(theta, y)

    def test_nan_y_raises(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        y = torch.tensor([1.0, float("nan")], dtype=torch.float64)

        with pytest.raises(ValueError, match="NaN"):
            solver.solve(theta, y)

    def test_inf_theta_raises(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.tensor([[1.0, float("inf")], [3.0, 4.0]], dtype=torch.float64)
        y = torch.tensor([1.0, 2.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)


    def test_high_condition_number_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.01, compute_condition_number=True)
        n = 100

        rng = torch.Generator().manual_seed(56)
        theta = torch.randn(n, 5, dtype=torch.float64, generator=rng)

        theta[:, 4] = theta[:, 0] + 1e-12 * theta[:, 1]
        y = torch.randn(n, dtype=torch.float64, generator=rng)

        with caplog.at_level(logging.WARNING):
            solver.solve(theta, y)


        assert any(
            "condition" in record.message.lower() for record in caplog.records
        ), (
            f"Expected condition number warning, got: {[r.message for r in caplog.records]}"
        )


    def test_lapack_failure_is_reported_as_degenerate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kd.core.linear_solve import STRidgeSolver

        def _fail(_matrix: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("linalg.cond: LAPACK failure")

        monkeypatch.setattr(torch.linalg, "cond", _fail)
        solver = STRidgeSolver(compute_condition_number=True)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=61)

        result = solver.solve(theta, y)

        assert result.condition_number == float("inf")

    def test_allocation_failure_is_not_reported_as_degenerate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kd.core.linear_solve import STRidgeSolver

        def _oom(_matrix: torch.Tensor) -> torch.Tensor:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        monkeypatch.setattr(torch.linalg, "cond", _oom)
        solver = STRidgeSolver(compute_condition_number=True)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=62)

        with pytest.raises(torch.cuda.OutOfMemoryError):
            solver.solve(theta, y)

    def test_cpu_allocation_failure_is_not_reported_as_degenerate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kd.core.linear_solve import STRidgeSolver

        def _oom(_matrix: torch.Tensor) -> torch.Tensor:
            raise RuntimeError(
                "[enforce fail at alloc_cpu.cpp:117]. DefaultCPUAllocator: "
                "can't allocate memory: you tried to allocate "
                "8000000000000 bytes."
            )

        monkeypatch.setattr(torch.linalg, "cond", _oom)
        solver = STRidgeSolver(compute_condition_number=True)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=63)

        with pytest.raises(RuntimeError, match="DefaultCPUAllocator"):
            solver.solve(theta, y)


    def test_inherits_sparse_solver(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        assert issubclass(STRidgeSolver, SparseSolver)

    def test_solve_returns_solve_result(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=57)
        result = solver.solve(theta, y)

        assert isinstance(result, SolveResult)
        assert isinstance(result.coefficients, torch.Tensor)
        assert isinstance(result.residual, float)
        assert isinstance(result.r2, float)
        assert result.condition_number is None

    def test_condition_number_returned_when_enabled(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(compute_condition_number=True)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=59)
        result = solver.solve(theta, y)

        assert isinstance(result.condition_number, float)
        assert result.condition_number > 0
        assert result.condition_number < float("inf")

    def test_result_has_selected_indices(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=58)
        result = solver.solve(theta, y)

        assert result.selected_indices is not None
        assert isinstance(result.selected_indices, list)
        assert all(isinstance(i, int) for i in result.selected_indices)

    def test_r2_close_to_one_for_clean_data(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(n=200, d=5, noise_std=0.0, seed=59)
        result = solver.solve(theta, y)

        assert result.r2 > 0.999

    def test_residual_near_zero_for_clean_data(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(n=200, d=5, noise_std=0.0, seed=60)
        result = solver.solve(theta, y)

        assert result.residual < 1e-10

    def test_y_2d_accepted(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=61)

        result_1d = solver.solve(theta, y)
        result_2d = solver.solve(theta, y.unsqueeze(1))

        torch.testing.assert_close(
            result_1d.coefficients, result_2d.coefficients, rtol=1e-10, atol=1e-10
        )

    def test_default_parameters(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        assert solver.tol == 0.1
        assert solver.lam == 0.0
        assert solver.max_iter == 10
        assert solver.normalize == 2


@pytest.mark.unit
class TestSTRidgeSolverEdgeCases:

    def test_single_column(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.01)
        rng = torch.Generator().manual_seed(62)
        theta = torch.randn(50, 1, dtype=torch.float64, generator=rng)
        y = 3.0 * theta.squeeze()
        result = solver.solve(theta, y)

        assert abs(result.coefficients[0].item() - 3.0) < 1e-4

    def test_all_zero_y(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        rng = torch.Generator().manual_seed(63)
        theta = torch.randn(50, 5, dtype=torch.float64, generator=rng)
        y = torch.zeros(50, dtype=torch.float64)
        result = solver.solve(theta, y)

        assert result.coefficients.abs().max().item() < 1e-10

    def test_selected_indices_empty_list_not_none(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        rng = torch.Generator().manual_seed(64)
        theta = torch.randn(50, 5, dtype=torch.float64, generator=rng)
        y = torch.zeros(50, dtype=torch.float64)
        result = solver.solve(theta, y)


        assert result.selected_indices is not None
        assert isinstance(result.selected_indices, list)
        assert len(result.selected_indices) == 0

    def test_dimension_mismatch_raises(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.randn(10, 5, dtype=torch.float64)
        y = torch.randn(20, dtype=torch.float64)

        with pytest.raises(ValueError, match="dimension"):
            solver.solve(theta, y)

    def test_empty_theta_raises(self) -> None:
        from kd.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.zeros(0, 5, dtype=torch.float64)
        y = torch.zeros(0, dtype=torch.float64)

        with pytest.raises(ValueError):
            solver.solve(theta, y)
