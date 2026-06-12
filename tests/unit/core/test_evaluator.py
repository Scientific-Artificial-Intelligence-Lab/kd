
from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.executor import ExecutionContext
from kd.core.expr import ExecutorResult, FunctionRegistry, PythonExecutor
from kd.core.linear_solve import LeastSquaresSolver
from kd.core.linear_solve.base import SolveResult, SparseSolver
from kd.data import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.data.derivatives import FiniteDiffProvider
from kd.data.synthetic import generate_burgers_data






@pytest.fixture
def registry() -> FunctionRegistry:
    return FunctionRegistry.create_default()


@pytest.fixture
def executor(registry: FunctionRegistry) -> PythonExecutor:
    return PythonExecutor(registry)


@pytest.fixture
def solver() -> LeastSquaresSolver:
    return LeastSquaresSolver()


@pytest.fixture
def simple_2d_dataset() -> PDEDataset:
    n_x = 32
    n_t = 16

    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)

    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt)

    return PDEDataset(
        name="test_2d",
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


@pytest.fixture
def derivative_provider(simple_2d_dataset: PDEDataset) -> FiniteDiffProvider:
    return FiniteDiffProvider(simple_2d_dataset, max_order=2)


@pytest.fixture
def context(
    simple_2d_dataset: PDEDataset,
    derivative_provider: FiniteDiffProvider,
) -> ExecutionContext:
    return ExecutionContext(
        dataset=simple_2d_dataset,
        derivative_provider=derivative_provider,
        constants={"nu": 0.1},
    )


@pytest.fixture
def lhs_tensor(context: ExecutionContext) -> Tensor:
    return context.get_derivative("u", "t", 1)


@pytest.fixture
def evaluator(
    executor: PythonExecutor,
    solver: LeastSquaresSolver,
    context: ExecutionContext,
    lhs_tensor: Tensor,
) -> Evaluator:
    return Evaluator(
        executor=executor,
        solver=solver,
        context=context,
        lhs=lhs_tensor,
    )







@pytest.fixture
def burgers_dataset() -> PDEDataset:
    return generate_burgers_data(
        nx=64,
        nt=32,
        nu=0.1,
        noise_level=0.0,
        seed=42,
    )


@pytest.fixture
def burgers_provider(burgers_dataset: PDEDataset) -> FiniteDiffProvider:
    return FiniteDiffProvider(burgers_dataset, max_order=2)


@pytest.fixture
def burgers_context(
    burgers_dataset: PDEDataset,
    burgers_provider: FiniteDiffProvider,
) -> ExecutionContext:
    return ExecutionContext(
        dataset=burgers_dataset,
        derivative_provider=burgers_provider,
        constants={"nu": 0.1},
    )


@pytest.fixture
def burgers_evaluator(
    executor: PythonExecutor,
    solver: LeastSquaresSolver,
    burgers_context: ExecutionContext,
    burgers_provider: FiniteDiffProvider,
) -> Evaluator:
    u_t = burgers_context.get_derivative("u", "t", 1)
    return Evaluator(
        executor=executor,
        solver=solver,
        context=burgers_context,
        lhs=u_t,
    )







@pytest.mark.smoke
class TestEvaluatorSmoke:

    def test_evaluation_result_can_be_created(self) -> None:
        result = EvaluationResult(
            mse=0.01,
            nmse=0.001,
            r2=0.99,
        )
        assert result.mse == 0.01
        assert result.nmse == 0.001
        assert result.r2 == 0.99

    def test_evaluation_result_default_values(self) -> None:
        result = EvaluationResult(mse=0.0, nmse=0.0, r2=1.0)
        assert result.aic is None
        assert result.complexity == 0
        assert result.coefficients is None
        assert result.is_valid is True
        assert result.error_message == ""

    def test_evaluator_can_be_created(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )
        assert evaluator is not None

    def test_evaluator_with_penalty(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
            penalty_value=1e8,
        )
        assert evaluator is not None

    def test_evaluator_has_evaluate_terms(self, evaluator: Evaluator) -> None:
        assert hasattr(evaluator, "evaluate_terms")
        assert callable(evaluator.evaluate_terms)

    def test_evaluator_has_evaluate_expression(self, evaluator: Evaluator) -> None:
        assert hasattr(evaluator, "evaluate_expression")
        assert callable(evaluator.evaluate_expression)







@pytest.mark.unit
class TestEvaluatorDefensiveDetach:

    def test_evaluator_detaches_lhs_input(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
    ) -> None:
        source = torch.arange(6, dtype=torch.float64, requires_grad=True)
        lhs = (source * 2.0).reshape(2, 3)

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )

        assert evaluator._lhs.grad_fn is None
        assert evaluator._lhs.requires_grad is False
        assert evaluator._lhs_flat.grad_fn is None







@pytest.mark.unit
class TestEvaluatorPublicAccessors:

    def test_executor_property_returns_injected_executor(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        ev = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )
        assert ev.executor is executor

    def test_solver_property_returns_injected_solver(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        ev = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )
        assert ev.solver is solver

    def test_context_property_returns_injected_context(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        ev = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )
        assert ev.context is context

    def test_public_accessors_are_read_only(self, evaluator: Evaluator) -> None:
        with pytest.raises(AttributeError):
            evaluator.executor = evaluator.executor
        with pytest.raises(AttributeError):
            evaluator.solver = evaluator.solver
        with pytest.raises(AttributeError):
            evaluator.context = evaluator.context







@pytest.mark.unit
class TestEvaluatorBuildThetaMatrix:

    def test_build_theta_matrix_matches_internal_path(
        self,
        evaluator: Evaluator,
    ) -> None:
        terms = ["u", "u_xx"]

        theta_public, valid_public = evaluator.build_theta_matrix(terms)
        theta_private, valid_private = evaluator._build_theta(terms)

        assert valid_public == valid_private
        assert torch.allclose(theta_public, theta_private)

    def test_build_theta_matrix_skip_invalid_filters_same_as_evaluate_terms(
        self,
        evaluator: Evaluator,
    ) -> None:
        terms = ["u", "sub(u, u)"]

        theta, valid_terms = evaluator.build_theta_matrix(terms, skip_invalid=True)
        result = evaluator.evaluate_terms(terms, skip_invalid=True)

        assert valid_terms == ["u"]
        assert result.is_valid is True
        assert result.terms == valid_terms
        assert theta.shape[1] == 1

    def test_build_theta_matrix_does_not_change_evaluate_expression_behavior(
        self,
        evaluator: Evaluator,
    ) -> None:
        expression = "add(u, u_xx)"

        before = evaluator.evaluate_expression(expression)
        evaluator.build_theta_matrix(["u", "u_xx"])
        after = evaluator.evaluate_expression(expression)

        assert before.is_valid is True
        assert after.is_valid is True
        assert before.terms == after.terms
        assert before.coefficients is not None
        assert after.coefficients is not None
        assert torch.allclose(before.coefficients, after.coefficients)







@pytest.mark.unit
class TestEvaluatorScorerInjection:

    def test_default_scorer_produces_standard_aic(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u"])
        assert result.is_valid


        assert result.aic is not None
        assert math.isfinite(result.aic)

    def test_custom_scorer_is_called(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:

        sentinel = -999.0

        def custom_scorer(mse: float, k: int) -> float:
            del mse, k
            return sentinel

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
            scorer=custom_scorer,
        )
        result = evaluator.evaluate_terms(["u"])
        assert result.is_valid
        assert result.aic == sentinel

    def test_custom_scorer_receives_correct_args(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        captured: list[tuple[float, int]] = []

        def tracking_scorer(mse: float, k: int) -> float:
            captured.append((mse, k))
            return mse + k

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
            scorer=tracking_scorer,
        )
        result = evaluator.evaluate_terms(["u"])
        assert result.is_valid
        assert len(captured) == 1
        mse_arg, k_arg = captured[0]
        assert mse_arg == result.mse
        assert k_arg == result.complexity

    def test_bic_scorer_injection(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        from kd.core.metrics import make_bic_scorer

        n_samples = lhs_tensor.flatten().shape[0]
        bic_scorer = make_bic_scorer(n_samples)

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
            scorer=bic_scorer,
        )
        result = evaluator.evaluate_terms(["u"])
        assert result.is_valid

        default_eval = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )
        default_result = default_eval.evaluate_terms(["u"])

        assert result.mse == pytest.approx(default_result.mse)
        assert result.aic != default_result.aic







@pytest.mark.unit
class TestEvaluateTermsBasic:

    def test_evaluate_terms_returns_result(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u"])
        assert isinstance(result, EvaluationResult)

    def test_evaluate_single_term(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u"])

        assert result.coefficients is not None
        assert result.coefficients.shape == (1,)
        assert result.is_valid is True

    def test_evaluate_multiple_terms(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u", "u_x", "u_xx"])

        assert result.coefficients is not None
        assert result.coefficients.shape == (3,)

    def test_evaluate_terms_result_has_metrics(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u"])

        assert isinstance(result.mse, float)
        assert isinstance(result.nmse, float)
        assert isinstance(result.r2, float)
        assert 0.0 <= result.mse < 1e-2
        assert 0.0 <= result.nmse < 1e-2

    def test_evaluate_terms_with_complex_expression(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["mul(u, u_x)", "u_xx"])

        assert result.coefficients is not None
        assert result.coefficients.shape == (2,)
        assert result.is_valid is True







@pytest.mark.unit
class TestEvaluateExpression:

    def test_evaluate_expression_single_term(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_expression("u")

        assert result.is_valid is True
        assert result.coefficients is not None
        assert result.coefficients.shape == (1,)

    def test_evaluate_expression_splits_add(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_expression("add(u, u_xx)")

        assert result.is_valid is True
        assert result.coefficients is not None

        assert result.coefficients.shape == (2,)

    def test_evaluate_expression_complex(self, evaluator: Evaluator) -> None:

        result = evaluator.evaluate_expression("add(mul(u, u_x), u_xx)")

        assert result.is_valid is True
        assert result.coefficients is not None
        assert result.coefficients.shape == (2,)







@pytest.mark.unit
class TestMetricsComputation:

    def test_mse_is_small_for_exact_relation(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u"])
        assert 0.0 <= result.mse < 1e-2

    def test_nmse_is_small_for_exact_relation(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u"])
        assert 0.0 <= result.nmse < 1e-2

    def test_r2_bounded(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u", "u_x", "u_xx"])

        assert result.r2 <= 1.0

    def test_perfect_fit_high_r2(self, context: ExecutionContext) -> None:
        registry = FunctionRegistry.create_default()
        executor = PythonExecutor(registry)
        solver = LeastSquaresSolver()
        u_t = context.get_derivative("u", "t", 1)

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=u_t,
        )


        result = evaluator.evaluate_terms(["u"])


        assert result.r2 > 0.95

        assert result.coefficients is not None
        assert abs(result.coefficients[0].item() + 1.0) < 0.1

    def test_lhs_var_is_population_variance(
        self, evaluator: Evaluator, lhs_tensor: Tensor
    ) -> None:
        expected = lhs_tensor.flatten().var(correction=0).item()
        assert evaluator._lhs_var == pytest.approx(
            expected, rel=1e-12
        )

    def test_nmse_equals_one_minus_r2(self, evaluator: Evaluator) -> None:
        for terms in (["u"], ["u_x"], ["u", "u_x", "u_xx"]):
            result = evaluator.evaluate_terms(terms)
            assert result.is_valid, f"fit failed for {terms}"
            assert result.nmse == pytest.approx(1.0 - result.r2, rel=1e-5, abs=1e-9), (
                f"nmse != 1 - r2 for {terms}"
            )

    def test_aic_with_complexity(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u", "u_xx"])



        assert result.complexity >= 0











@pytest.mark.unit
class TestInvalidExpressions:

    def test_empty_terms_returns_invalid(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms([])

        assert result.is_valid is False
        assert result.mse >= evaluator._penalty_value or result.r2 < 0

    def test_invalid_expression_returns_penalty(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
            penalty_value=1e10,
        )


        result = evaluator.evaluate_terms(["unknown_var"])

        assert result.is_valid is False
        assert result.mse == 1e10

    def test_syntax_error_returns_invalid(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["add(u, )"])

        assert result.is_valid is False
        assert result.error_message != ""

    def test_execution_error_returns_invalid(self, evaluator: Evaluator) -> None:


        result = evaluator.evaluate_terms(["this_is_not_valid!!!"])

        assert result.is_valid is False







@pytest.mark.integration
class TestBurgersCoeffientRecovery:

    def test_burgers_coefficient_recovery(
        self,
        burgers_evaluator: Evaluator,
    ) -> None:
        result = burgers_evaluator.evaluate_terms(["mul(u, u_x)", "u_xx"])

        assert result.is_valid is True
        assert result.coefficients is not None


        coef_convection = result.coefficients[0].item()
        coef_diffusion = result.coefficients[1].item()


        assert abs(coef_convection - (-1.0)) < 0.05, f"Got {coef_convection}"
        assert abs(coef_diffusion - 0.1) < 0.05, f"Got {coef_diffusion}"

    def test_burgers_r2_high(self, burgers_evaluator: Evaluator) -> None:
        result = burgers_evaluator.evaluate_terms(["mul(u, u_x)", "u_xx"])

        assert result.r2 > 0.99

    def test_burgers_via_evaluate_expression(
        self,
        burgers_evaluator: Evaluator,
    ) -> None:
        result = burgers_evaluator.evaluate_expression("add(mul(u, u_x), u_xx)")

        assert result.is_valid is True
        assert result.r2 > 0.99
        assert result.coefficients is not None


        coef_convection = result.coefficients[0].item()
        coef_diffusion = result.coefficients[1].item()

        assert abs(coef_convection - (-1.0)) < 0.05
        assert abs(coef_diffusion - 0.1) < 0.05

    def test_burgers_wrong_terms_lower_r2(
        self,
        burgers_evaluator: Evaluator,
    ) -> None:

        correct_result = burgers_evaluator.evaluate_terms(["mul(u, u_x)", "u_xx"])


        wrong_result = burgers_evaluator.evaluate_terms(["u_xx"])


        assert correct_result.r2 > wrong_result.r2







@pytest.mark.numerical
class TestNumericalStability:

    def test_result_coefficients_finite(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u", "u_x", "u_xx"])
        assert result.is_valid, f"Evaluation must succeed: {result.error_message}"
        assert result.coefficients is not None
        assert torch.isfinite(result.coefficients).all()

    def test_metrics_finite(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u"])
        assert result.is_valid, f"Evaluation must succeed: {result.error_message}"
        assert math.isfinite(result.mse)
        assert math.isfinite(result.nmse)
        assert math.isfinite(result.r2)

    def test_handles_zero_variance_lhs(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
    ) -> None:

        shape = context.dataset.get_shape()
        constant_lhs = torch.ones(shape, dtype=torch.float64)

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=constant_lhs,
        )

        result = evaluator.evaluate_terms(["u"])


        assert math.isfinite(result.mse)


    def test_handles_large_values(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
    ) -> None:
        shape = context.dataset.get_shape()
        large_lhs = torch.ones(shape, dtype=torch.float64) * 1e6

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=large_lhs,
        )

        result = evaluator.evaluate_terms(["u"])


        assert math.isfinite(result.mse)
        assert result.is_valid, f"Evaluation must succeed: {result.error_message}"
        assert result.coefficients is not None
        assert torch.isfinite(result.coefficients).all()







@pytest.mark.unit
class TestEdgeCases:

    def test_small_dataset(self) -> None:

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        t = torch.linspace(0, 1, 5, dtype=torch.float64)
        xx, tt = torch.meshgrid(x, t, indexing="ij")
        u = xx + tt

        dataset = PDEDataset(
            name="minimal",
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

        provider = FiniteDiffProvider(dataset, max_order=1)
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=provider,
            constants={},
        )

        registry = FunctionRegistry.create_default()
        executor = PythonExecutor(registry)
        solver = LeastSquaresSolver()
        lhs = torch.ones_like(u)

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )


        result = evaluator.evaluate_terms(["u"])
        assert isinstance(result, EvaluationResult)

    def test_many_terms(self, evaluator: Evaluator) -> None:
        terms = ["u", "u_x", "u_xx", "n2(u)", "n3(u)", "sin(u)", "cos(u)"]
        result = evaluator.evaluate_terms(terms)
        assert result.is_valid, f"Evaluation must succeed: {result.error_message}"
        assert result.coefficients is not None
        assert result.coefficients.shape == (len(terms),)

    def test_duplicate_terms(self, evaluator: Evaluator) -> None:
        result = evaluator.evaluate_terms(["u", "u"])


        assert isinstance(result, EvaluationResult)









class _SparseMockSolver(SparseSolver):

    def __init__(self, selected_indices: list[int] | None) -> None:
        self._selected_indices = selected_indices

    def solve(self, theta: torch.Tensor, y: torch.Tensor) -> SolveResult:

        result = torch.linalg.lstsq(theta, y.unsqueeze(1) if y.dim() == 1 else y)
        coefficients = result.solution.squeeze()
        if coefficients.dim() == 0:
            coefficients = coefficients.unsqueeze(0)


        if self._selected_indices is not None:
            mask = torch.zeros_like(coefficients)
            for idx in self._selected_indices:
                mask[idx] = 1.0
            coefficients = coefficients * mask

        y_1d = y.squeeze(-1) if y.dim() == 2 else y
        y_pred = theta @ coefficients
        ss_res = ((y_1d - y_pred) ** 2).sum().item()
        ss_tot = ((y_1d - y_1d.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

        return SolveResult(
            coefficients=coefficients,
            residual=ss_res,
            r2=r2,
            condition_number=1.0,
            selected_indices=self._selected_indices,
        )


@pytest.mark.unit
class TestAICComplexityKFix:

    def test_dense_solver_complexity_equals_num_terms(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        dense_solver = _SparseMockSolver(selected_indices=None)
        evaluator = Evaluator(
            executor=executor,
            solver=dense_solver,
            context=context,
            lhs=lhs_tensor,
        )

        terms = ["u", "u_x", "u_xx"]
        result = evaluator.evaluate_terms(terms)

        assert result.is_valid is True
        assert result.complexity == len(terms)

    def test_sparse_solver_complexity_equals_selected_count(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        sparse_solver = _SparseMockSolver(selected_indices=[0, 2])
        evaluator = Evaluator(
            executor=executor,
            solver=sparse_solver,
            context=context,
            lhs=lhs_tensor,
        )

        terms = ["u", "u_x", "u_xx", "n2(u)", "sin(u)"]
        result = evaluator.evaluate_terms(terms)

        assert result.is_valid is True

        assert result.complexity == 2

    def test_sparse_solver_empty_selection_complexity_zero(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        sparse_solver = _SparseMockSolver(selected_indices=[])
        evaluator = Evaluator(
            executor=executor,
            solver=sparse_solver,
            context=context,
            lhs=lhs_tensor,
        )

        terms = ["u", "u_x", "u_xx"]
        result = evaluator.evaluate_terms(terms)

        assert result.is_valid is True

        assert result.complexity == 0

    def test_aic_uses_correct_complexity(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        sparse_solver = _SparseMockSolver(selected_indices=[0, 2])
        evaluator = Evaluator(
            executor=executor,
            solver=sparse_solver,
            context=context,
            lhs=lhs_tensor,
        )

        terms = ["u", "u_x", "u_xx"]
        result = evaluator.evaluate_terms(terms)

        assert result.is_valid is True

        assert result.mse > 1e-15, f"MSE too small ({result.mse}), AIC would be -inf"
        assert result.aic is not None


        n_samples = lhs_tensor.numel()
        expected_k = 2
        expected_aic = n_samples * math.log(result.mse) + 2 * expected_k


        assert result.aic == pytest.approx(expected_aic, rel=1e-10)

    def test_dense_regression_unchanged(
        self,
        evaluator: Evaluator,
    ) -> None:
        terms = ["u", "u_xx"]
        result = evaluator.evaluate_terms(terms)

        assert result.is_valid is True

        assert result.complexity == len(terms)






        assert result.mse > 1e-15, f"MSE too small ({result.mse}), AIC would be -inf"

        n_samples = evaluator._lhs_flat.shape[0]
        expected_aic = n_samples * math.log(result.mse) + 2 * len(terms)
        assert result.aic == pytest.approx(expected_aic, rel=1e-10)







@pytest.mark.unit
class TestEvaluatorIntermediateResults:

    def test_new_fields_have_defaults(self) -> None:
        result = EvaluationResult(mse=0.1, nmse=0.01, r2=0.9)
        assert result.selected_indices is None
        assert result.residuals is None
        assert result.terms is None
        assert result.expression == ""

    def test_selected_indices_from_sparse_solver(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        sparse_solver = _SparseMockSolver(selected_indices=[0, 2])
        evaluator = Evaluator(
            executor=executor,
            solver=sparse_solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(["u", "u_x", "u_xx"])

        assert result.is_valid is True
        assert result.selected_indices == [0, 2]

    def test_selected_indices_none_for_dense(
        self,
        evaluator: Evaluator,
    ) -> None:
        result = evaluator.evaluate_terms(["u", "u_x"])

        assert result.is_valid is True
        assert result.selected_indices is None

    def test_residuals_computed_and_detached(
        self,
        evaluator: Evaluator,
    ) -> None:
        result = evaluator.evaluate_terms(["u"])

        assert result.residuals is not None
        assert not result.residuals.requires_grad

        assert result.residuals.dim() == 1

    def test_residuals_shape_matches_lhs(
        self,
        evaluator: Evaluator,
    ) -> None:
        result = evaluator.evaluate_terms(["u", "u_x"])

        assert result.residuals is not None
        assert result.residuals.shape == evaluator._lhs_flat.shape

    def test_residuals_value_correct(
        self,
        evaluator: Evaluator,
    ) -> None:
        result = evaluator.evaluate_terms(["u"])

        assert result.residuals is not None
        assert result.coefficients is not None

        assert torch.isfinite(result.residuals).all()

    def test_residuals_perfect_fit_near_zero(
        self,
        context: ExecutionContext,
    ) -> None:
        registry = FunctionRegistry.create_default()
        executor = PythonExecutor(registry)
        solver = LeastSquaresSolver()
        u_t = context.get_derivative("u", "t", 1)

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=u_t,
        )

        result = evaluator.evaluate_terms(["u"])

        assert result.residuals is not None

        assert result.residuals.abs().max().item() < 0.1

    def test_terms_field_populated(
        self,
        evaluator: Evaluator,
    ) -> None:
        terms = ["u", "u_x", "u_xx"]
        result = evaluator.evaluate_terms(terms)

        assert result.terms == terms

    def test_terms_field_is_copy(
        self,
        evaluator: Evaluator,
    ) -> None:
        terms = ["u", "u_x"]
        result = evaluator.evaluate_terms(terms)
        terms.append("u_xx")

        assert result.terms == ["u", "u_x"]

    def test_expression_set_by_evaluate_expression(
        self,
        evaluator: Evaluator,
    ) -> None:
        expr = "add(u, u_xx)"
        result = evaluator.evaluate_expression(expr)

        assert result.expression == expr

    def test_expression_empty_for_evaluate_terms(
        self,
        evaluator: Evaluator,
    ) -> None:
        result = evaluator.evaluate_terms(["u"])

        assert result.expression == ""

    def test_invalid_result_has_none_fields(
        self,
        evaluator: Evaluator,
    ) -> None:
        result = evaluator.evaluate_terms([])

        assert result.is_valid is False
        assert result.selected_indices is None
        assert result.residuals is None
        assert result.terms is None







@pytest.mark.unit
class TestEncapsulationFix:

    def test_executor_has_registry_property(
        self,
        executor: PythonExecutor,
    ) -> None:
        assert hasattr(executor, "registry")
        assert isinstance(executor.registry, FunctionRegistry)

    def test_registry_property_matches_internal(
        self,
        executor: PythonExecutor,
    ) -> None:
        assert executor.registry is executor._registry







class _NaNInjectingExecutor:

    def __init__(
        self,
        real_executor: PythonExecutor,
        nan_terms: list[str] | None = None,
        inf_terms: list[str] | None = None,
        error_terms: list[str] | None = None,
        zero_terms: list[str] | None = None,
    ) -> None:
        self._real = real_executor
        self._nan_terms: set[str] = set(nan_terms or [])
        self._inf_terms: set[str] = set(inf_terms or [])
        self._error_terms: set[str] = set(error_terms or [])
        self._zero_terms: set[str] = set(zero_terms or [])

    @property
    def registry(self) -> FunctionRegistry:
        return self._real.registry

    def execute(
        self,
        code: str,
        context: ExecutionContext,
    ) -> ExecutorResult:
        if code in self._error_terms:
            raise RuntimeError(f"Injected error for '{code}'")

        result = self._real.execute(code, context)

        if code in self._nan_terms:
            val = result.value.clone()
            val.flatten()[0] = float("nan")
            return ExecutorResult(value=val, used_diff=result.used_diff)

        if code in self._inf_terms:
            val = result.value.clone()
            val.flatten()[0] = float("inf")
            return ExecutorResult(value=val, used_diff=result.used_diff)

        if code in self._zero_terms:
            val = torch.zeros_like(result.value)
            return ExecutorResult(value=val, used_diff=result.used_diff)

        return result







@pytest.mark.unit
class TestSkipInvalid:

    def test_skip_exception_term(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            error_terms=["u_x"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx"],
            skip_invalid=True,
        )

        assert result.is_valid is True
        assert result.terms == ["u", "u_xx"]
        assert result.coefficients is not None
        assert result.coefficients.shape == (2,)

    def test_skip_nan_term(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            nan_terms=["u_x"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx"],
            skip_invalid=True,
        )

        assert result.is_valid is True
        assert result.terms == ["u", "u_xx"]
        assert result.coefficients is not None
        assert result.coefficients.shape == (2,)

    def test_skip_inf_term(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            inf_terms=["u"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx"],
            skip_invalid=True,
        )

        assert result.is_valid is True
        assert result.terms == ["u_x", "u_xx"]
        assert result.coefficients is not None
        assert result.coefficients.shape == (2,)

    def test_skip_all_zero_term(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            zero_terms=["u_xx"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx"],
            skip_invalid=True,
        )

        assert result.is_valid is True
        assert result.terms == ["u", "u_x"]
        assert result.coefficients is not None
        assert result.coefficients.shape == (2,)

    def test_skip_all_fail(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            error_terms=["u"],
            nan_terms=["u_x"],
            zero_terms=["u_xx"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx"],
            skip_invalid=True,
        )

        assert result.is_valid is False

    def test_skip_false_exception_fails(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            error_terms=["u_x"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx"],
            skip_invalid=False,
        )

        assert result.is_valid is False

    def test_skip_false_nan_fails(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            nan_terms=["u_x"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx"],
            skip_invalid=False,
        )

        assert result.is_valid is False

    def test_result_terms_alignment(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            error_terms=["u_x"],
            nan_terms=["u_xx"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )


        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx", "n2(u)", "sin(u)"],
            skip_invalid=True,
        )

        assert result.is_valid is True
        assert result.terms is not None
        assert result.coefficients is not None

        assert len(result.terms) == result.coefficients.shape[0]

        assert result.terms == ["u", "n2(u)", "sin(u)"]

    def test_selected_indices_relative_to_filtered(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            error_terms=["u_x"],
        )

        sparse_solver = _SparseMockSolver(selected_indices=[0, 1])
        evaluator = Evaluator(
            executor=bad_executor,
            solver=sparse_solver,
            context=context,
            lhs=lhs_tensor,
        )


        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx"],
            skip_invalid=True,
        )

        assert result.is_valid is True
        assert result.terms == ["u", "u_xx"]

        assert result.selected_indices == [0, 1]

    def test_no_skip_unchanged(
        self,
        evaluator: Evaluator,
    ) -> None:
        terms = ["u", "u_x", "u_xx"]


        result_explicit = evaluator.evaluate_terms(terms, skip_invalid=False)

        result_default = evaluator.evaluate_terms(terms)

        assert result_explicit.is_valid == result_default.is_valid
        assert result_explicit.mse == pytest.approx(result_default.mse, rel=1e-10)
        assert result_explicit.r2 == pytest.approx(result_default.r2, rel=1e-10)
        assert result_explicit.terms == result_default.terms

    def test_mixed_failures(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            error_terms=["u_x"],
            nan_terms=["n2(u)"],
            zero_terms=["sin(u)"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(
            ["u", "u_x", "u_xx", "n2(u)", "sin(u)"],
            skip_invalid=True,
        )

        assert result.is_valid is True

        assert result.terms == ["u", "u_xx"]
        assert result.coefficients is not None
        assert result.coefficients.shape == (2,)







class _OOMInjectingExecutor:

    def __init__(
        self,
        real_executor: PythonExecutor,
        oom_terms: list[str] | None = None,
    ) -> None:
        self._real = real_executor
        self._oom_terms: set[str] = set(oom_terms or [])

    @property
    def registry(self) -> FunctionRegistry:
        return self._real.registry

    def execute(
        self,
        code: str,
        context: ExecutionContext,
    ) -> ExecutorResult:
        if code in self._oom_terms:
            raise torch.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 526.00 MiB.",
            )
        return self._real.execute(code, context)


@pytest.mark.unit
class TestAutogradOOM:

    def test_evaluate_terms_oom_returns_invalid_result(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        oom_executor = _OOMInjectingExecutor(
            executor,
            oom_terms=["u_xx"],
        )
        evaluator = Evaluator(
            executor=oom_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(["u", "u_xx"])

        assert result.is_valid is False
        assert "autograd OOM" in result.error_message
        assert math.isinf(result.aic or 0.0)

    def test_evaluate_expression_oom_returns_invalid_result(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        oom_executor = _OOMInjectingExecutor(
            executor,
            oom_terms=["u_xx"],
        )
        evaluator = Evaluator(
            executor=oom_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_expression("add(u, u_xx)")

        assert result.is_valid is False
        assert "autograd OOM" in result.error_message
        assert result.expression == "add(u, u_xx)"

    def test_evaluate_terms_oom_logs_warning(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        oom_executor = _OOMInjectingExecutor(
            executor,
            oom_terms=["u_xx"],
        )
        evaluator = Evaluator(
            executor=oom_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        with caplog.at_level("WARNING", logger="kd.core.evaluator"):
            evaluator.evaluate_terms(["u", "u_xx"])

        oom_records = [r for r in caplog.records if "OOM" in r.getMessage()]
        assert oom_records, "expected an OOM warning log record"
        assert oom_records[0].levelname == "WARNING"

    def test_non_oom_runtime_error_not_misreported_as_oom(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
        context: ExecutionContext,
        lhs_tensor: Tensor,
    ) -> None:
        bad_executor = _NaNInjectingExecutor(
            executor,
            error_terms=["u_xx"],
        )
        evaluator = Evaluator(
            executor=bad_executor,
            solver=solver,
            context=context,
            lhs=lhs_tensor,
        )

        result = evaluator.evaluate_terms(["u", "u_xx"])

        assert result.is_valid is False
        assert "autograd OOM" not in result.error_message
        assert "Execution error" in result.error_message
