
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from kd.core.executor import ExecutionContext
from kd.core.expr.executor import ExecutorResult, PythonExecutor
from kd.core.expr.registry import FunctionRegistry
from kd.core.linear_solve import LeastSquaresSolver, SolveResult
from kd.data import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.data.derivatives import DerivativeProvider, FiniteDiffProvider

if TYPE_CHECKING:
    pass







@pytest.fixture
def simple_1d_dataset() -> PDEDataset:
    n_x = 32
    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    u = torch.sin(x)
    v = torch.cos(x)

    return PDEDataset(
        name="test_1d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={
            "u": FieldData(name="u", values=u),
            "v": FieldData(name="v", values=v),
        },
        lhs_field="u",
        lhs_axis="x",
    )


@pytest.fixture
def simple_2d_dataset() -> PDEDataset:
    n_x = 32
    n_t = 16

    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)

    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt)
    v = torch.cos(xx) * torch.exp(-tt)

    return PDEDataset(
        name="test_2d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={
            "u": FieldData(name="u", values=u),
            "v": FieldData(name="v", values=v),
        },
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def derivative_provider_1d(simple_1d_dataset: PDEDataset) -> FiniteDiffProvider:
    return FiniteDiffProvider(simple_1d_dataset, max_order=2)


@pytest.fixture
def derivative_provider_2d(simple_2d_dataset: PDEDataset) -> FiniteDiffProvider:
    return FiniteDiffProvider(simple_2d_dataset, max_order=3)


@pytest.fixture
def execution_context_1d(
    simple_1d_dataset: PDEDataset,
    derivative_provider_1d: FiniteDiffProvider,
) -> ExecutionContext:
    return ExecutionContext(
        dataset=simple_1d_dataset,
        derivative_provider=derivative_provider_1d,
        constants={"pi": math.pi, "C": 1.5},
    )


@pytest.fixture
def execution_context_2d(
    simple_2d_dataset: PDEDataset,
    derivative_provider_2d: FiniteDiffProvider,
) -> ExecutionContext:
    return ExecutionContext(
        dataset=simple_2d_dataset,
        derivative_provider=derivative_provider_2d,
        constants={"pi": math.pi, "nu": 0.1},
    )


@pytest.fixture
def default_registry() -> FunctionRegistry:
    return FunctionRegistry.create_default()


@pytest.fixture
def python_executor(default_registry: FunctionRegistry) -> PythonExecutor:
    return PythonExecutor(default_registry)







@pytest.mark.integration
class TestPythonExecutorWithExecutionContext:

    def test_execute_with_context_variable(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:

        expected_u = execution_context_1d.get_variable("u")



        result = python_executor.execute("u", execution_context_1d)

        assert isinstance(result, ExecutorResult)
        assert result.value is not None
        torch.testing.assert_close(result.value, expected_u, rtol=1e-10, atol=1e-10)

    def test_execute_with_context_coordinate(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        expected_x = execution_context_1d.get_variable("x")

        result = python_executor.execute("x", execution_context_1d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected_x, rtol=1e-10, atol=1e-10)

    def test_execute_expression_with_context_variables(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        u = execution_context_1d.get_variable("u")
        v = execution_context_1d.get_variable("v")
        expected = u + v

        result = python_executor.execute("add(u, v)", execution_context_1d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_with_terminal_derivatives(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:

        expected_u_x = execution_context_1d.get_derivative("u", "x", 1)


        result = python_executor.execute("u_x", execution_context_1d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected_u_x, rtol=1e-5, atol=1e-8)

    def test_execute_expression_with_derivatives(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:
        u = execution_context_2d.get_variable("u")
        u_x = execution_context_2d.get_derivative("u", "x", 1)
        u_xx = execution_context_2d.get_derivative("u", "x", 2)


        expected = u * u_x + u_xx

        result = python_executor.execute("add(mul(u, u_x), u_xx)", execution_context_2d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_with_constant(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:
        u = execution_context_2d.get_variable("u")
        nu = execution_context_2d.get_constant("nu")


        expected = nu * u

        result = python_executor.execute("mul(nu, u)", execution_context_2d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)







@pytest.mark.integration
class TestDerivativeResolution:

    def test_terminal_derivative_from_context(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        result = python_executor.execute("u_x", execution_context_1d)

        assert result.value is not None
        assert result.used_diff is False

    def test_open_form_diff_uses_provider(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        result = python_executor.execute("diff_x(u)", execution_context_1d)
        assert result.value is not None
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_compound_terminal_matches_nested_call_numerically(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        compound = python_executor.execute("u_x_x", execution_context_1d)
        nested = python_executor.execute("diff_x(diff_x(u))", execution_context_1d)

        assert compound.value is not None
        assert nested.value is not None
        torch.testing.assert_close(compound.value, nested.value, rtol=0.0, atol=0.0)

    def test_terminal_and_open_form_equivalence_with_mock_provider(
        self,
        simple_1d_dataset: PDEDataset,
        default_registry: FunctionRegistry,
    ) -> None:

        class MockDerivativeProvider(DerivativeProvider):

            def __init__(self, dataset: PDEDataset) -> None:
                self._dataset = dataset
                self._cache: dict[tuple[str, str, int], Tensor] = {}

                if dataset.axes is not None:
                    x = dataset.axes["x"].values
                    u = torch.sin(x)
                    u_x = torch.cos(x)
                    u_xx = -torch.sin(x)

                    self._field_cache = {"u": u}
                    self._cache[("u", "x", 1)] = u_x
                    self._cache[("u", "x", 2)] = u_xx

            def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
                key = (field, axis, order)
                if key not in self._cache:
                    raise KeyError(f"Derivative {key} not found")
                return self._cache[key].clone()

            def diff(
                self, expression: torch.Tensor, axis: str, order: int
            ) -> torch.Tensor:
                if self._dataset.axes is None:
                    raise ValueError("No axes")


                u = self._field_cache.get("u")
                if u is not None and torch.allclose(
                    expression, u, rtol=1e-10, atol=1e-10
                ):

                    key = ("u", axis, order)
                    if key in self._cache:
                        return self._cache[key].clone()


                raise NotImplementedError(
                    f"diff() only supports base field u for testing, got expression with shape {expression.shape}"
                )

            def available_derivatives(self) -> list[tuple[str, str, int]]:
                return list(self._cache.keys())

        provider = MockDerivativeProvider(simple_1d_dataset)
        context = ExecutionContext(
            dataset=simple_1d_dataset,
            derivative_provider=provider,
            constants={},
        )

        executor = PythonExecutor(default_registry)


        terminal_result = executor.execute("u_x", context)

        diff_result = executor.execute("diff_x(u)", context)


        assert terminal_result.value is not None
        assert diff_result.value is not None



        torch.testing.assert_close(
            terminal_result.value, diff_result.value, rtol=1e-10, atol=1e-10
        )

    def test_open_form_diff_on_expression(
        self,
        simple_1d_dataset: PDEDataset,
        default_registry: FunctionRegistry,
    ) -> None:

        class AnalyticalDiffProvider(DerivativeProvider):

            def __init__(self, dataset: PDEDataset) -> None:
                self._dataset = dataset
                if dataset.axes is None:
                    raise ValueError("No axes")
                x = dataset.axes["x"].values
                self._dx = (x[-1] - x[0]).item() / (len(x) - 1)

            def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
                raise KeyError(f"No precomputed {field}_{axis}")

            def diff(
                self, expression: torch.Tensor, axis: str, order: int
            ) -> torch.Tensor:
                if order != 1:
                    raise NotImplementedError

                result = torch.zeros_like(expression)
                result[1:-1] = (expression[2:] - expression[:-2]) / (2 * self._dx)
                result[0] = (expression[1] - expression[0]) / self._dx
                result[-1] = (expression[-1] - expression[-2]) / self._dx
                return result

            def available_derivatives(self) -> list[tuple[str, str, int]]:
                return []

        provider = AnalyticalDiffProvider(simple_1d_dataset)
        context = ExecutionContext(
            dataset=simple_1d_dataset,
            derivative_provider=provider,
            constants={},
        )

        executor = PythonExecutor(default_registry)


        result = executor.execute("diff_x(mul(u, u))", context)

        assert result.value is not None
        assert result.used_diff is True


        if simple_1d_dataset.axes is not None:
            x = simple_1d_dataset.axes["x"].values
            expected = torch.sin(2 * x)


            torch.testing.assert_close(
                result.value[1:-1], expected[1:-1], rtol=0.15, atol=0.15
            )







@pytest.mark.integration
class TestPrefixConversionExecution:

    def test_simple_prefix_to_execution(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
        default_registry: FunctionRegistry,
    ) -> None:
        from kd.core.compat.prefix import prefix_to_python


        prefix_tokens = ["add", "u", "v"]
        python_expr = prefix_to_python(prefix_tokens, default_registry)

        assert python_expr == "add(u, v)"


        result = python_executor.execute(python_expr, execution_context_2d)


        u = execution_context_2d.get_variable("u")
        v = execution_context_2d.get_variable("v")
        expected = u + v

        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_nested_prefix_to_execution(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
        default_registry: FunctionRegistry,
    ) -> None:
        from kd.core.compat.prefix import prefix_to_python



        prefix_tokens = ["add", "mul", "u", "u_x", "u_xx"]
        python_expr = prefix_to_python(prefix_tokens, default_registry)

        assert python_expr == "add(mul(u, u_x), u_xx)"

        result = python_executor.execute(python_expr, execution_context_2d)

        u = execution_context_2d.get_variable("u")
        u_x = execution_context_2d.get_derivative("u", "x", 1)
        u_xx = execution_context_2d.get_derivative("u", "x", 2)
        expected = u * u_x + u_xx

        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_roundtrip_python_to_prefix_to_python(
        self,
        default_registry: FunctionRegistry,
    ) -> None:
        from kd.core.compat.prefix import prefix_to_python, python_to_prefix

        original = "add(mul(u, v), sin(u))"


        prefix = python_to_prefix(original)

        reconstructed = prefix_to_python(prefix, default_registry)



        assert reconstructed == original

    def test_prefix_execution_matches_direct_execution(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
        default_registry: FunctionRegistry,
    ) -> None:
        from kd.core.compat.prefix import prefix_to_python


        direct_expr = "mul(sin(u), cos(v))"


        from kd.core.compat.prefix import python_to_prefix

        prefix = python_to_prefix(direct_expr)
        converted_expr = prefix_to_python(prefix, default_registry)


        direct_result = python_executor.execute(direct_expr, execution_context_2d)
        converted_result = python_executor.execute(converted_expr, execution_context_2d)

        torch.testing.assert_close(
            direct_result.value, converted_result.value, rtol=1e-10, atol=1e-10
        )







@pytest.mark.smoke
class TestSolverIntegration:

    def test_expression_result_as_solver_input(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:

        expressions = ["u", "u_x", "u_xx", "mul(u, u_x)"]
        results = []

        for expr in expressions:
            result = python_executor.execute(expr, execution_context_2d)
            assert result.value is not None

            results.append(result.value.flatten())


        theta = torch.stack(results, dim=1)


        u_t = execution_context_2d.get_derivative("u", "t", 1)
        y = u_t.flatten()


        assert theta.shape[0] == y.shape[0]
        assert theta.shape[1] == len(expressions)


        assert isinstance(theta, Tensor)
        assert isinstance(y, Tensor)

    def test_solver_with_expression_results(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:

        expressions = ["u", "u_xx"]
        results = []

        for expr in expressions:
            result = python_executor.execute(expr, execution_context_2d)
            results.append(result.value.flatten())

        theta = torch.stack(results, dim=1)


        u_t = execution_context_2d.get_derivative("u", "t", 1)
        y = u_t.flatten()


        solver = LeastSquaresSolver()
        solve_result = solver.solve(theta, y)


        assert isinstance(solve_result, SolveResult)
        assert solve_result.coefficients.shape == (len(expressions),)
        assert isinstance(solve_result.r2, float)
        assert isinstance(solve_result.residual, float)

    def test_solver_with_numerical_stability(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:

        result = python_executor.execute("div(u, v)", execution_context_2d)
        assert result.value is not None
        assert torch.isfinite(result.value).all()


        theta = result.value.flatten().unsqueeze(1)
        y = torch.randn_like(theta[:, 0])

        solver = LeastSquaresSolver()
        solve_result = solver.solve(theta, y)


        assert torch.isfinite(solve_result.coefficients).all()







@pytest.mark.numerical
class TestIntegrationNumericalEdgeCases:

    def test_expression_with_zero_values(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:

        result = python_executor.execute("div(v, u)", execution_context_1d)
        assert result.value is not None

        assert torch.isfinite(result.value).all()

    def test_expression_with_small_values(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:

        result = python_executor.execute("exp(neg(n2(u)))", execution_context_1d)
        assert result.value is not None
        assert torch.isfinite(result.value).all()

    def test_derivative_at_boundary(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        result = python_executor.execute("u_x", execution_context_1d)
        assert result.value is not None
        assert torch.isfinite(result.value).all()


        assert torch.isfinite(result.value[0])
        assert torch.isfinite(result.value[-1])

    def test_nested_operations_numerical_stability(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:

        result = python_executor.execute("sin(cos(exp(neg(u))))", execution_context_2d)
        assert result.value is not None
        assert torch.isfinite(result.value).all()


        assert (result.value >= -1.0).all()
        assert (result.value <= 1.0).all()

    def test_empty_expression_error(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:

        with pytest.raises((ValueError, SyntaxError)):
            python_executor.execute("", execution_context_1d)


        with pytest.raises((ValueError, SyntaxError)):
            python_executor.execute(" ", execution_context_1d)


        with pytest.raises((ValueError, SyntaxError)):
            python_executor.execute("\t\n", execution_context_1d)

    def test_extreme_values(
        self,
        default_registry: FunctionRegistry,
    ) -> None:
        from kd.data.derivatives import DerivativeProvider


        n = 10
        x = torch.linspace(0, 1, n, dtype=torch.float64)
        u = torch.sin(x * math.pi)

        dataset = PDEDataset(
            name="extreme_test",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

        class MinimalProvider(DerivativeProvider):
            def get_derivative(self, field: str, axis: str, order: int) -> Tensor:
                raise KeyError(f"No derivative for {field}_{axis}")

            def diff(self, expr: Tensor, axis: str, order: int) -> Tensor:
                raise NotImplementedError("No diff support")

            def available_derivatives(self) -> list[tuple[str, str, int]]:
                return []

        provider = MinimalProvider()
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=provider,
            constants={},
        )

        executor = PythonExecutor(default_registry)





        result_large = executor.execute("exp(mul(u, add(u, u)))", context)
        assert result_large.value is not None

        assert not torch.isnan(result_large.value).any()




        result_div = executor.execute("div(u, mul(u, u))", context)
        assert result_div.value is not None

        assert torch.isfinite(result_div.value).all()



        result_n2 = executor.execute("n2(u)", context)
        assert result_n2.value is not None
        assert not torch.isnan(result_n2.value).any()
        assert torch.isfinite(result_n2.value).all()



        result_nested = executor.execute("exp(neg(exp(neg(u))))", context)
        assert result_nested.value is not None
        assert torch.isfinite(result_nested.value).all()





        add_fn = default_registry.get_func("add")
        div_fn = default_registry.get_func("div")


        extreme_tensor = torch.tensor([1e100, 1e-100, 0.0], dtype=torch.float64)
        result_extreme_add = add_fn(extreme_tensor, extreme_tensor)

        assert torch.isfinite(result_extreme_add).all() or result_extreme_add[
            0
        ] == float("inf")


        zero_tensor = torch.zeros(3, dtype=torch.float64)
        one_tensor = torch.ones(3, dtype=torch.float64)
        result_safe_div = div_fn(one_tensor, zero_tensor)

        assert torch.isfinite(result_safe_div).all()
