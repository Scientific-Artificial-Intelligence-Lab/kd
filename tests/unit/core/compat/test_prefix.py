
import pytest

from kd.core.compat.prefix import prefix_to_python, python_to_prefix
from kd.core.expr.registry import FunctionRegistry







@pytest.fixture
def default_registry() -> FunctionRegistry:
    return FunctionRegistry.create_default()


@pytest.fixture
def extended_registry() -> FunctionRegistry:
    reg = FunctionRegistry.create_default()

    reg.register("sqrt", lambda x: x**0.5, arity=1)
    reg.register("pow", lambda x, y: x**y, arity=2)
    return reg







class TestPythonToPrefixSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(python_to_prefix)

    @pytest.mark.smoke
    def test_simple_binary_expression(self) -> None:
        result = python_to_prefix("add(x, y)")
        assert result == ["add", "x", "y"]

    @pytest.mark.smoke
    def test_simple_unary_expression(self) -> None:
        result = python_to_prefix("sin(x)")
        assert result == ["sin", "x"]


class TestPrefixToPythonSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(prefix_to_python)

    @pytest.mark.smoke
    def test_simple_binary_expression(self, default_registry: FunctionRegistry) -> None:
        result = prefix_to_python(["add", "x", "y"], default_registry)
        assert result == "add(x, y)"

    @pytest.mark.smoke
    def test_simple_unary_expression(self, default_registry: FunctionRegistry) -> None:
        result = prefix_to_python(["sin", "x"], default_registry)
        assert result == "sin(x)"







class TestPythonToPrefixHappyPath:

    @pytest.mark.unit
    def test_nested_binary_operations(self) -> None:
        result = python_to_prefix("add(mul(a, b), sub(c, d))")
        assert result == ["add", "mul", "a", "b", "sub", "c", "d"]

    @pytest.mark.unit
    def test_deeply_nested_expression(self) -> None:

        result = python_to_prefix("add(mul(u, u_x), mul(C, u_xx))")
        assert result == ["add", "mul", "u", "u_x", "mul", "C", "u_xx"]

    @pytest.mark.unit
    def test_mixed_unary_binary(self) -> None:
        result = python_to_prefix("add(sin(x), cos(y))")
        assert result == ["add", "sin", "x", "cos", "y"]

    @pytest.mark.unit
    def test_chained_unary(self) -> None:
        result = python_to_prefix("sin(cos(exp(x)))")
        assert result == ["sin", "cos", "exp", "x"]

    @pytest.mark.unit
    def test_three_level_nesting(self) -> None:
        result = python_to_prefix("add(mul(sin(x), y), z)")
        assert result == ["add", "mul", "sin", "x", "y", "z"]

    @pytest.mark.unit
    def test_multiple_variables(self) -> None:
        result = python_to_prefix("add(add(a, b), add(c, d))")
        assert result == ["add", "add", "a", "b", "add", "c", "d"]

    @pytest.mark.unit
    def test_derivative_style_names(self) -> None:
        result = python_to_prefix("mul(u_x, u_xx)")
        assert result == ["mul", "u_x", "u_xx"]







class TestPrefixToPythonHappyPath:

    @pytest.mark.unit
    def test_nested_binary_operations(self, default_registry: FunctionRegistry) -> None:
        tokens = ["add", "mul", "a", "b", "sub", "c", "d"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "add(mul(a, b), sub(c, d))"

    @pytest.mark.unit
    def test_deeply_nested_expression(self, default_registry: FunctionRegistry) -> None:
        tokens = ["add", "mul", "u", "u_x", "mul", "C", "u_xx"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "add(mul(u, u_x), mul(C, u_xx))"

    @pytest.mark.unit
    def test_mixed_unary_binary(self, default_registry: FunctionRegistry) -> None:
        tokens = ["add", "sin", "x", "cos", "y"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "add(sin(x), cos(y))"

    @pytest.mark.unit
    def test_chained_unary(self, default_registry: FunctionRegistry) -> None:
        tokens = ["sin", "cos", "exp", "x"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "sin(cos(exp(x)))"

    @pytest.mark.unit
    def test_derivative_style_names(self, default_registry: FunctionRegistry) -> None:
        tokens = ["mul", "u_x", "u_xx"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "mul(u_x, u_xx)"







class TestSingleElement:

    @pytest.mark.unit
    def test_python_to_prefix_single_variable(self) -> None:
        result = python_to_prefix("x")
        assert result == ["x"]

    @pytest.mark.unit
    def test_python_to_prefix_single_variable_with_underscore(self) -> None:
        result = python_to_prefix("u_x")
        assert result == ["u_x"]

    @pytest.mark.unit
    def test_prefix_to_python_single_variable(
        self, default_registry: FunctionRegistry
    ) -> None:
        result = prefix_to_python(["x"], default_registry)
        assert result == "x"

    @pytest.mark.unit
    def test_prefix_to_python_single_variable_with_underscore(
        self, default_registry: FunctionRegistry
    ) -> None:
        result = prefix_to_python(["u_x"], default_registry)
        assert result == "u_x"







class TestConstants:

    @pytest.mark.unit
    def test_python_to_prefix_integer(self) -> None:
        result = python_to_prefix("42")
        assert result == ["42"]

    @pytest.mark.unit
    def test_python_to_prefix_float(self) -> None:
        result = python_to_prefix("3.14")
        assert result == ["3.14"]

    @pytest.mark.unit
    def test_python_to_prefix_negative_integer(self) -> None:
        result = python_to_prefix("-5")
        assert result == ["-5"]

    @pytest.mark.unit
    def test_python_to_prefix_negative_float(self) -> None:
        result = python_to_prefix("-3.14")
        assert result == ["-3.14"]

    @pytest.mark.unit
    def test_python_to_prefix_scientific_notation(self) -> None:
        result = python_to_prefix("1e-10")
        assert result == ["1e-10"]

    @pytest.mark.unit
    def test_python_to_prefix_constant_in_expression(self) -> None:
        result = python_to_prefix("mul(2.5, x)")
        assert result == ["mul", "2.5", "x"]

    @pytest.mark.unit
    def test_prefix_to_python_integer(self, default_registry: FunctionRegistry) -> None:
        result = prefix_to_python(["42"], default_registry)
        assert result == "42"

    @pytest.mark.unit
    def test_prefix_to_python_float(self, default_registry: FunctionRegistry) -> None:
        result = prefix_to_python(["3.14"], default_registry)
        assert result == "3.14"

    @pytest.mark.unit
    def test_prefix_to_python_negative(self, default_registry: FunctionRegistry) -> None:
        result = prefix_to_python(["-5"], default_registry)
        assert result == "-5"

    @pytest.mark.unit
    def test_prefix_to_python_constant_in_expression(
        self, default_registry: FunctionRegistry
    ) -> None:
        result = prefix_to_python(["mul", "2.5", "x"], default_registry)
        assert result == "mul(2.5, x)"







class TestEmptyAndMinimal:

    @pytest.mark.unit
    def test_python_to_prefix_empty_raises(self) -> None:
        with pytest.raises((SyntaxError, ValueError)):
            python_to_prefix("")

    @pytest.mark.unit
    def test_python_to_prefix_whitespace_only_raises(self) -> None:
        with pytest.raises((SyntaxError, ValueError)):
            python_to_prefix(" ")

    @pytest.mark.unit
    def test_prefix_to_python_empty_raises(
        self, default_registry: FunctionRegistry
    ) -> None:
        with pytest.raises(ValueError):
            prefix_to_python([], default_registry)







class TestRoundTrip:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "expr",
        [
            "x",
            "add(x, y)",
            "mul(a, b)",
            "sin(x)",
            "cos(y)",
            "exp(z)",
            "add(mul(x, y), z)",
            "sin(cos(x))",
            "add(sin(x), cos(y))",
            "mul(u_x, u_xx)",
            "add(mul(u, u_x), mul(C, u_xx))",
            "div(sub(a, b), add(c, d))",
            "neg(x)",
            "n2(x)",
            "n3(y)",
        ],
    )
    def test_round_trip_preserves_expression(
        self, default_registry: FunctionRegistry, expr: str
    ) -> None:
        prefix = python_to_prefix(expr)
        restored = prefix_to_python(prefix, default_registry)
        assert restored == expr

    @pytest.mark.unit
    def test_round_trip_with_constants(
        self, default_registry: FunctionRegistry
    ) -> None:
        expr = "add(mul(2, x), 3.14)"
        prefix = python_to_prefix(expr)
        restored = prefix_to_python(prefix, default_registry)
        assert restored == expr

    @pytest.mark.unit
    def test_round_trip_complex_pde_term(
        self, default_registry: FunctionRegistry
    ) -> None:

        expr = "add(mul(u, u_x), mul(nu, u_xx))"
        prefix = python_to_prefix(expr)
        restored = prefix_to_python(prefix, default_registry)
        assert restored == expr

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "expr,expected_prefix",
        [

            ("-5", ["-5"]),
            ("-3.14", ["-3.14"]),
            ("-1e-10", ["-1e-10"]),

            ("add(x, -5)", ["add", "x", "-5"]),
            ("mul(-2.5, y)", ["mul", "-2.5", "y"]),
            ("add(-1, -2)", ["add", "-1", "-2"]),
        ],
    )
    def test_round_trip_negative_constants(
        self, default_registry: FunctionRegistry, expr: str, expected_prefix: list[str]
    ) -> None:

        prefix = python_to_prefix(expr)
        assert prefix == expected_prefix


        restored = prefix_to_python(prefix, default_registry)
        prefix_again = python_to_prefix(restored)
        assert prefix_again == expected_prefix







class TestPythonToPrefixErrors:

    @pytest.mark.unit
    def test_invalid_syntax_raises(self) -> None:
        with pytest.raises(SyntaxError):
            python_to_prefix("add x y")

    @pytest.mark.unit
    def test_incomplete_expression_raises(self) -> None:
        with pytest.raises(SyntaxError):
            python_to_prefix("add(x")

    @pytest.mark.unit
    def test_statement_not_expression_raises(self) -> None:
        with pytest.raises(SyntaxError):
            python_to_prefix("x = 5")

    @pytest.mark.unit
    def test_multiple_expressions_raises(self) -> None:
        with pytest.raises(SyntaxError):
            python_to_prefix("x; y")

    @pytest.mark.unit
    def test_method_call_raises(self) -> None:

        with pytest.raises((ValueError, AttributeError)):
            python_to_prefix("x.sin()")

    @pytest.mark.unit
    def test_subscript_raises(self) -> None:
        with pytest.raises(ValueError):
            python_to_prefix("x[0]")

    @pytest.mark.unit
    def test_binary_operator_raises(self) -> None:

        with pytest.raises(ValueError):
            python_to_prefix("x + y")

    @pytest.mark.unit
    def test_list_literal_raises(self) -> None:
        with pytest.raises(ValueError):
            python_to_prefix("[1, 2, 3]")







class TestPrefixToPythonErrors:

    @pytest.mark.unit
    def test_unbalanced_too_few_args(self, default_registry: FunctionRegistry) -> None:

        with pytest.raises(ValueError):
            prefix_to_python(["add", "x"], default_registry)

    @pytest.mark.unit
    def test_unbalanced_extra_tokens(self, default_registry: FunctionRegistry) -> None:

        with pytest.raises(ValueError):
            prefix_to_python(["sin", "x", "y"], default_registry)

    @pytest.mark.unit
    def test_unknown_function_used_as_variable(
        self, default_registry: FunctionRegistry
    ) -> None:

        result = prefix_to_python(["add", "unknown_var", "x"], default_registry)
        assert result == "add(unknown_var, x)"

    @pytest.mark.unit
    def test_deeply_unbalanced(self, default_registry: FunctionRegistry) -> None:

        with pytest.raises(ValueError):
            prefix_to_python(["add", "add", "x", "y"], default_registry)







class TestExtendedRegistry:

    @pytest.mark.unit
    def test_custom_unary_function(self, extended_registry: FunctionRegistry) -> None:
        result = prefix_to_python(["sqrt", "x"], extended_registry)
        assert result == "sqrt(x)"

    @pytest.mark.unit
    def test_custom_binary_function(self, extended_registry: FunctionRegistry) -> None:
        result = prefix_to_python(["pow", "x", "2"], extended_registry)
        assert result == "pow(x, 2)"

    @pytest.mark.unit
    def test_mixed_default_and_custom(
        self, extended_registry: FunctionRegistry
    ) -> None:
        tokens = ["add", "sqrt", "x", "pow", "y", "2"]
        result = prefix_to_python(tokens, extended_registry)
        assert result == "add(sqrt(x), pow(y, 2))"







class TestDISCOVERCompatibility:

    @pytest.mark.unit
    def test_discover_style_expression(self) -> None:

        code = "add(mul(u, u_x), mul(C, u_xx))"
        prefix = python_to_prefix(code)
        expected = ["add", "mul", "u", "u_x", "mul", "C", "u_xx"]
        assert prefix == expected

    @pytest.mark.unit
    def test_discover_reconstruction(
        self, default_registry: FunctionRegistry
    ) -> None:

        prefix_from_discover = ["add", "mul", "u", "u_x", "mul", "C", "u_xx"]
        result = prefix_to_python(prefix_from_discover, default_registry)
        assert result == "add(mul(u, u_x), mul(C, u_xx))"

    @pytest.mark.unit
    def test_burgers_equation_term(self) -> None:
        code = "add(mul(u, u_x), mul(nu, u_xx))"
        prefix = python_to_prefix(code)
        assert prefix == ["add", "mul", "u", "u_x", "mul", "nu", "u_xx"]

    @pytest.mark.unit
    def test_heat_equation_term(self) -> None:
        code = "mul(alpha, u_xx)"
        prefix = python_to_prefix(code)
        assert prefix == ["mul", "alpha", "u_xx"]

    @pytest.mark.unit
    def test_wave_equation_term(self) -> None:
        code = "mul(n2(c), u_xx)"
        prefix = python_to_prefix(code)
        assert prefix == ["mul", "n2", "c", "u_xx"]
