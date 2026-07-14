
from __future__ import annotations

import pytest
import torch

from kd.core.linear_solve import LeastSquaresSolver, SolveResult, SparseSolver






@pytest.fixture
def solver() -> LeastSquaresSolver:
    return LeastSquaresSolver()


@pytest.fixture
def solver_with_rcond() -> LeastSquaresSolver:
    return LeastSquaresSolver(rcond=1e-10)


@pytest.fixture
def simple_overdetermined() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    n_samples = 10
    n_features = 2

    theta = torch.randn(n_samples, n_features, dtype=torch.float64)
    true_coef = torch.tensor([2.0, 3.0], dtype=torch.float64)
    y = theta @ true_coef

    return theta, y, true_coef


@pytest.fixture
def noisy_overdetermined() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    n_samples = 100
    n_features = 2

    theta = torch.randn(n_samples, n_features, dtype=torch.float64)
    true_coef = torch.tensor([2.0, 3.0], dtype=torch.float64)
    noise = 0.1 * torch.randn(n_samples, dtype=torch.float64)
    y = theta @ true_coef + noise

    return theta, y, true_coef


@pytest.fixture
def underdetermined() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    n_samples = 5
    n_features = 10

    theta = torch.randn(n_samples, n_features, dtype=torch.float64)
    y = torch.randn(n_samples, dtype=torch.float64)

    return theta, y


@pytest.fixture
def square_system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    n = 5


    theta = torch.randn(n, n, dtype=torch.float64)
    theta = theta + torch.eye(n, dtype=torch.float64) * 2
    true_coef = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    y = theta @ true_coef

    return theta, y, true_coef


@pytest.fixture
def ill_conditioned() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    n_samples = 20


    x1 = torch.randn(n_samples, 1, dtype=torch.float64)
    x2 = x1 + 1e-8 * torch.randn(n_samples, 1, dtype=torch.float64)
    x3 = torch.randn(n_samples, 1, dtype=torch.float64)

    theta = torch.cat([x1, x2, x3], dim=1)
    y = 2 * x1.squeeze() + 3 * x3.squeeze()

    return theta, y







@pytest.mark.smoke
class TestSolverSmoke:

    def test_solve_result_can_be_created(self) -> None:
        result = SolveResult(
            coefficients=torch.tensor([1.0, 2.0]),
            residual=0.1,
            r2=0.99,
            condition_number=10.0,
        )
        assert result.coefficients is not None
        assert result.residual == 0.1
        assert result.r2 == 0.99
        assert result.condition_number == 10.0
        assert result.selected_indices is None

    def test_solve_result_condition_number_defaults_to_none(self) -> None:
        result = SolveResult(
            coefficients=torch.tensor([1.0, 2.0]),
            residual=0.1,
            r2=0.99,
        )

        assert result.condition_number is None

    def test_solve_result_with_selected_indices(self) -> None:
        result = SolveResult(
            coefficients=torch.tensor([1.0, 0.0, 2.0]),
            residual=0.0,
            r2=1.0,
            condition_number=1.0,
            selected_indices=[0, 2],
        )
        assert result.selected_indices == [0, 2]

    def test_sparse_solver_is_abc(self) -> None:
        with pytest.raises(TypeError):
            SparseSolver()

    def test_least_squares_solver_can_be_created(self) -> None:
        solver = LeastSquaresSolver()
        assert solver is not None
        assert solver.rcond is None
        assert solver.compute_condition_number is False

    def test_least_squares_solver_can_opt_into_condition_number(self) -> None:
        solver = LeastSquaresSolver(compute_condition_number=True)
        assert solver.compute_condition_number is True

    def test_least_squares_solver_with_rcond(self) -> None:
        solver = LeastSquaresSolver(rcond=1e-10)
        assert solver.rcond == 1e-10

    def test_least_squares_solver_is_sparse_solver(self) -> None:
        assert issubclass(LeastSquaresSolver, SparseSolver)
        solver = LeastSquaresSolver()
        assert isinstance(solver, SparseSolver)

    def test_least_squares_solver_has_solve_method(self) -> None:
        solver = LeastSquaresSolver()
        assert hasattr(solver, "solve")
        assert callable(solver.solve)







@pytest.mark.unit
class TestBasicSolving:

    def test_solve_returns_solve_result(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)
        assert isinstance(result, SolveResult)

    def test_solve_recovers_exact_coefficients(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, true_coef = simple_overdetermined
        result = solver.solve(theta, y)

        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_solve_coefficients_shape(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)

        n_terms = theta.shape[1]
        assert result.coefficients.shape == (n_terms,)

    def test_solve_handles_1d_y(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, true_coef = simple_overdetermined
        assert y.dim() == 1

        result = solver.solve(theta, y)
        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_solve_handles_2d_y(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, true_coef = simple_overdetermined
        y_2d = y.unsqueeze(1)
        assert y_2d.dim() == 2

        result = solver.solve(theta, y_2d)
        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_solve_square_system(
        self,
        solver: LeastSquaresSolver,
        square_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, true_coef = square_system
        result = solver.solve(theta, y)

        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )







@pytest.mark.unit
class TestOverdeterminedSystems:

    def test_overdetermined_minimizes_residual(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)



        y_pred = theta @ result.coefficients
        residual = ((y - y_pred) ** 2).sum().item()


        perturbed_coef = result.coefficients + 0.1 * torch.randn_like(
            result.coefficients
        )
        y_pred_perturbed = theta @ perturbed_coef
        residual_perturbed = ((y - y_pred_perturbed) ** 2).sum().item()

        assert residual <= residual_perturbed

    def test_overdetermined_coefficients_close_to_true(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, true_coef = noisy_overdetermined
        result = solver.solve(theta, y)


        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=0.1,
            atol=0.2,
        )







@pytest.mark.unit
class TestUnderdeterminedSystems:

    def test_underdetermined_returns_solution(
        self,
        solver: LeastSquaresSolver,
        underdetermined: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y = underdetermined
        result = solver.solve(theta, y)


        y_pred = theta @ result.coefficients
        torch.testing.assert_close(
            y_pred,
            y,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_underdetermined_minimum_norm(
        self,
        solver: LeastSquaresSolver,
        underdetermined: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y = underdetermined
        result = solver.solve(theta, y)

        solution_norm = torch.linalg.norm(result.coefficients).item()





        assert solution_norm < 1e6







@pytest.mark.unit
class TestMetrics:

    def test_residual_for_exact_solution(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)


        assert result.residual < 1e-10

    def test_residual_for_noisy_data(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)


        assert result.residual > 0.0

    def test_residual_matches_computed(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)

        y_pred = theta @ result.coefficients
        expected_residual = ((y - y_pred) ** 2).sum().item()

        assert abs(result.residual - expected_residual) < 1e-10

    def test_r2_for_perfect_fit(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)

        assert result.r2 > 0.9999

    def test_r2_for_noisy_data(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)

        assert 0.0 < result.r2 < 1.0

    def test_r2_formula_correct(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)


        y_pred = theta @ result.coefficients
        ss_res = ((y - y_pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        expected_r2 = 1.0 - ss_res / ss_tot

        assert abs(result.r2 - expected_r2) < 1e-10

    def test_condition_number_default_not_computed(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)

        assert result.condition_number is None

    def test_condition_number_returned_when_enabled(
        self,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = simple_overdetermined
        solver = LeastSquaresSolver(compute_condition_number=True)
        result = solver.solve(theta, y)

        assert isinstance(result.condition_number, float)
        assert result.condition_number > 0
        assert result.condition_number < float("inf")

    def test_condition_number_high_for_ill_conditioned(
        self,
        ill_conditioned: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y = ill_conditioned
        solver = LeastSquaresSolver(compute_condition_number=True)
        result = solver.solve(theta, y)


        assert result.condition_number > 1e6







@pytest.mark.unit
class TestDataTypes:

    def test_float64_input(self, solver: LeastSquaresSolver) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        true_coef = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        y = theta @ true_coef

        result = solver.solve(theta, y)
        assert result.coefficients.dtype == torch.float64

    def test_float32_input(self, solver: LeastSquaresSolver) -> None:
        theta = torch.randn(10, 3, dtype=torch.float32)
        true_coef = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = theta @ true_coef

        result = solver.solve(theta, y)
        assert result.coefficients.dtype == torch.float32

    def test_output_matches_input_dtype(self, solver: LeastSquaresSolver) -> None:
        for dtype in [torch.float32, torch.float64]:
            theta = torch.randn(10, 3, dtype=dtype)
            y = torch.randn(10, dtype=dtype)

            result = solver.solve(theta, y)
            assert result.coefficients.dtype == dtype







@pytest.mark.numerical
class TestNumericalStability:

    def test_handles_very_small_values(self, solver: LeastSquaresSolver) -> None:
        theta = torch.randn(20, 3, dtype=torch.float64)
        true_coef = torch.tensor([1e-10, 2e-10, 3e-10], dtype=torch.float64)
        y = theta @ true_coef

        result = solver.solve(theta, y)
        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-3,
            atol=1e-12,
        )

    def test_handles_very_large_values(self, solver: LeastSquaresSolver) -> None:
        theta = torch.randn(20, 3, dtype=torch.float64)
        true_coef = torch.tensor([1e6, 2e6, 3e6], dtype=torch.float64)
        y = theta @ true_coef

        result = solver.solve(theta, y)
        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-2,
        )

    def test_handles_mixed_scale_values(self, solver: LeastSquaresSolver) -> None:
        theta = torch.randn(50, 3, dtype=torch.float64)
        true_coef = torch.tensor([1e-6, 1.0, 1e6], dtype=torch.float64)
        y = theta @ true_coef

        result = solver.solve(theta, y)

        diff = (result.coefficients - true_coef).abs()
        relative_error = diff / (true_coef.abs() + 1e-10)
        assert relative_error.max().item() < 0.01

    def test_result_is_finite(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)

        assert torch.isfinite(result.coefficients).all()

    def test_result_finite_for_ill_conditioned(
        self,
        solver: LeastSquaresSolver,
        ill_conditioned: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        theta, y = ill_conditioned
        result = solver.solve(theta, y)

        assert torch.isfinite(result.coefficients).all()







@pytest.mark.unit
class TestEdgeCases:

    def test_single_sample_single_feature(self, solver: LeastSquaresSolver) -> None:
        theta = torch.tensor([[2.0]], dtype=torch.float64)
        y = torch.tensor([6.0], dtype=torch.float64)

        result = solver.solve(theta, y)

        torch.testing.assert_close(
            result.coefficients,
            torch.tensor([3.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_single_feature_multiple_samples(self, solver: LeastSquaresSolver) -> None:
        theta = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        y = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)

        result = solver.solve(theta, y)

        torch.testing.assert_close(
            result.coefficients,
            torch.tensor([2.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_multiple_features_few_samples(self, solver: LeastSquaresSolver) -> None:

        theta = torch.randn(3, 5, dtype=torch.float64)
        y = torch.randn(3, dtype=torch.float64)

        result = solver.solve(theta, y)


        y_pred = theta @ result.coefficients
        torch.testing.assert_close(y_pred, y, rtol=1e-5, atol=1e-8)

    def test_all_zeros_y(self, solver: LeastSquaresSolver) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.zeros(10, dtype=torch.float64)

        result = solver.solve(theta, y)


        assert torch.allclose(
            result.coefficients,
            torch.zeros(3, dtype=torch.float64),
            atol=1e-10,
        )

        assert result.r2 == 1.0

    def test_constant_y(self, solver: LeastSquaresSolver) -> None:

        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[:, 0] = 1.0


        y = torch.full((10,), 5.0, dtype=torch.float64)

        result = solver.solve(theta, y)


        y_pred = theta @ result.coefficients
        torch.testing.assert_close(y_pred, y, rtol=1e-5, atol=1e-8)

    def test_r2_with_constant_y(self, solver: LeastSquaresSolver) -> None:

        theta = torch.ones(10, 1, dtype=torch.float64)
        y = torch.full((10,), 5.0, dtype=torch.float64)

        result = solver.solve(theta, y)

        assert result.r2 == 1.0







@pytest.mark.unit
class TestInputValidation:

    def test_dimension_mismatch_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(5, dtype=torch.float64)

        with pytest.raises(ValueError, match="dimension"):
            solver.solve(theta, y)

    def test_empty_theta_zero_rows_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.empty(0, 3, dtype=torch.float64)
        y = torch.empty(0, dtype=torch.float64)

        with pytest.raises(ValueError, match="empty"):
            solver.solve(theta, y)

    def test_empty_theta_zero_cols_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.empty(10, 0, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="empty"):
            solver.solve(theta, y)

    def test_theta_1d_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="2D"):
            solver.solve(theta, y)

    def test_y_wrong_shape_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, 2, dtype=torch.float64)

        with pytest.raises(ValueError):
            solver.solve(theta, y)







@pytest.mark.numerical
class TestNumericalEdgeCases:

    def test_nan_in_theta_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[5, 1] = float("nan")
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="NaN"):
            solver.solve(theta, y)

    def test_nan_in_y_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)
        y[3] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            solver.solve(theta, y)

    def test_inf_in_theta_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[2, 0] = float("inf")
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)

    def test_neg_inf_in_theta_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[7, 2] = float("-inf")
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)

    def test_inf_in_y_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)
        y[8] = float("inf")

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)

    def test_neg_inf_in_y_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)
        y[0] = float("-inf")

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)

    def test_all_zero_theta_returns_result_with_inf_condition(
        self,
    ) -> None:
        theta = torch.zeros(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)
        solver = LeastSquaresSolver(compute_condition_number=True)

        result = solver.solve(theta, y)


        assert torch.isfinite(result.coefficients).all()

        assert result.condition_number == float("inf")

    def test_single_zero_column_in_theta(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[:, 1] = 0.0
        y = theta[:, 0] * 2.0 + theta[:, 2] * 3.0

        result = solver.solve(theta, y)


        assert torch.isfinite(result.coefficients).all()









@pytest.mark.unit
class TestRcond:

    def test_rcond_affects_ill_conditioned_solution(self) -> None:

        n_samples = 20
        theta = torch.randn(n_samples, 3, dtype=torch.float64)

        theta[:, 1] = theta[:, 0] + 1e-12 * torch.randn(n_samples, dtype=torch.float64)
        y = torch.randn(n_samples, dtype=torch.float64)

        solver_no_rcond = LeastSquaresSolver(rcond=None)
        solver_with_rcond = LeastSquaresSolver(rcond=1e-6)

        result_no_rcond = solver_no_rcond.solve(theta, y)
        result_with_rcond = solver_with_rcond.solve(theta, y)


        assert torch.isfinite(result_no_rcond.coefficients).all()
        assert torch.isfinite(result_with_rcond.coefficients).all()







@pytest.mark.unit
class TestBurgersCoefficients:

    def test_recover_burgers_coefficients(self, solver: LeastSquaresSolver) -> None:
        torch.manual_seed(42)
        n_samples = 100
        nu = 0.1




        theta = torch.randn(n_samples, 2, dtype=torch.float64)


        true_coef = torch.tensor([-1.0, nu], dtype=torch.float64)


        y = theta @ true_coef

        result = solver.solve(theta, y)

        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_recover_burgers_with_noise(self, solver: LeastSquaresSolver) -> None:
        torch.manual_seed(42)
        n_samples = 500
        nu = 0.1

        theta = torch.randn(n_samples, 2, dtype=torch.float64)
        true_coef = torch.tensor([-1.0, nu], dtype=torch.float64)


        noise_level = 0.05
        y_clean = theta @ true_coef
        noise_scale = noise_level * y_clean.std()
        noise = noise_scale * torch.randn(n_samples, dtype=torch.float64)
        y = y_clean + noise

        result = solver.solve(theta, y)


        relative_error = (result.coefficients - true_coef).abs() / true_coef.abs()
        assert relative_error.max().item() < 0.1







@pytest.mark.unit
class TestHighPriorityFixes:

    def test_mixed_dtype_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float32)

        with pytest.raises(ValueError, match="dtype"):
            solver.solve(theta, y)

    def test_mixed_dtype_both_directions(
        self, solver: LeastSquaresSolver
    ) -> None:

        theta_32 = torch.randn(10, 3, dtype=torch.float32)
        y_64 = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="dtype"):
            solver.solve(theta_32, y_64)


        theta_64 = torch.randn(10, 3, dtype=torch.float64)
        y_32 = torch.randn(10, dtype=torch.float32)

        with pytest.raises(ValueError, match="dtype"):
            solver.solve(theta_64, y_32)

    def test_0d_y_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.tensor(1.0, dtype=torch.float64)

        assert y.dim() == 0

        with pytest.raises(ValueError, match="0D"):
            solver.solve(theta, y)

    def test_near_constant_y_r2_bounded(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.randn(10, 3, dtype=torch.float64)

        y = torch.full((10,), 5.0, dtype=torch.float64)
        y = y + 1e-14 * torch.randn(10, dtype=torch.float64)

        result = solver.solve(theta, y)


        assert -1.0 <= result.r2 <= 1.0

    def test_exactly_constant_y_r2_one_for_perfect_fit(
        self, solver: LeastSquaresSolver
    ) -> None:
        theta = torch.ones(10, 1, dtype=torch.float64)
        y = torch.full((10,), 5.0, dtype=torch.float64)

        result = solver.solve(theta, y)

        assert result.r2 == 1.0
