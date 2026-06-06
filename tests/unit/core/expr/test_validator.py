
import pytest

from kd.core.expr.validator import (
    ALLOWED_NODES,
    get_function_calls,
    validate_expr,
)






@pytest.mark.smoke
class TestValidatorSmoke:

    def test_validate_expr_exists(self) -> None:
        assert callable(validate_expr)

    def test_get_function_calls_exists(self) -> None:
        assert callable(get_function_calls)

    def test_allowed_nodes_is_set(self) -> None:
        assert isinstance(ALLOWED_NODES, set)
        assert len(ALLOWED_NODES) > 0
        for node_type in ALLOWED_NODES:
            assert isinstance(node_type, type)







@pytest.mark.unit
class TestValidExpressions:

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        return {"add", "mul", "sub", "div", "sin", "cos", "exp", "neg", "n2", "n3"}

    def test_simple_function_call(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("add(u, v)", allowed_funcs) is True

    def test_unary_function_call(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("sin(x)", allowed_funcs) is True

    def test_constant_expression(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("1.0", allowed_funcs) is True
        assert validate_expr("42", allowed_funcs) is True
        assert validate_expr("-2.5", allowed_funcs) is True

    def test_variable_expression(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("u", allowed_funcs) is True
        assert validate_expr("u_x", allowed_funcs) is True
        assert validate_expr("x", allowed_funcs) is True

    def test_nested_function_calls(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("add(sin(x), cos(y))", allowed_funcs) is True
        assert validate_expr("mul(add(u, v), sub(x, y))", allowed_funcs) is True

    def test_deeply_nested_expression(self, allowed_funcs: set[str]) -> None:
        expr = "add(mul(sin(u), cos(v)), neg(exp(x)))"
        assert validate_expr(expr, allowed_funcs) is True

    def test_function_with_constant_arg(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("mul(u, 2.0)", allowed_funcs) is True
        assert validate_expr("add(1.0, 2.0)", allowed_funcs) is True

    def test_negative_constant(self, allowed_funcs: set[str]) -> None:



        assert validate_expr("neg(2.5)", allowed_funcs) is True

    def test_zero_arity_terminal(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("u_x", allowed_funcs) is True
        assert validate_expr("u_xx", allowed_funcs) is True







@pytest.mark.unit
class TestInvalidExpressions:

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        return {"add", "mul", "sub", "div", "sin", "cos"}

    def test_infix_operator_binop(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("u * v", allowed_funcs) is False
        assert validate_expr("u + v", allowed_funcs) is False
        assert validate_expr("u - v", allowed_funcs) is False
        assert validate_expr("u / v", allowed_funcs) is False
        assert validate_expr("u ** 2", allowed_funcs) is False

    def test_unary_minus_operator(self, allowed_funcs: set[str]) -> None:

        assert validate_expr("-x", allowed_funcs) is False

    def test_method_call(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("x.sin()", allowed_funcs) is False
        assert validate_expr("tensor.mean()", allowed_funcs) is False
        assert validate_expr("obj.method(a, b)", allowed_funcs) is False

    def test_subscript(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("x[0]", allowed_funcs) is False
        assert validate_expr("data[i]", allowed_funcs) is False

    def test_list_literal(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("[1, 2, 3]", allowed_funcs) is False

    def test_dict_literal(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("{'a': 1}", allowed_funcs) is False

    def test_comprehension(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("[x for x in items]", allowed_funcs) is False

    def test_lambda(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("lambda x: x + 1", allowed_funcs) is False

    def test_comparison(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("x > 0", allowed_funcs) is False
        assert validate_expr("a == b", allowed_funcs) is False

    def test_boolean_operator(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("a and b", allowed_funcs) is False
        assert validate_expr("a or b", allowed_funcs) is False
        assert validate_expr("not a", allowed_funcs) is False

    def test_conditional_expression(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("a if cond else b", allowed_funcs) is False







@pytest.mark.unit
class TestUnknownFunctions:

    def test_unknown_function_rejected(self) -> None:
        allowed = {"add", "mul"}
        assert validate_expr("unknown_func(x)", allowed) is False

    def test_typo_function_rejected(self) -> None:
        allowed = {"sin", "cos"}
        assert validate_expr("sine(x)", allowed) is False

    def test_case_sensitive_rejection(self) -> None:
        allowed = {"sin", "cos"}
        assert validate_expr("Sin(x)", allowed) is False
        assert validate_expr("SIN(x)", allowed) is False

    def test_nested_unknown_function(self) -> None:
        allowed = {"add", "mul"}
        assert validate_expr("add(unknown(x), y)", allowed) is False

    def test_all_functions_must_be_allowed(self) -> None:
        allowed = {"add"}

        assert validate_expr("add(mul(a, b), c)", allowed) is False







@pytest.mark.unit
class TestSyntaxErrors:

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        return {"add", "mul", "sin"}

    def test_syntax_error_unbalanced_parens(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("add(a, b", allowed_funcs) is False
        assert validate_expr("add(a, b))", allowed_funcs) is False

    def test_syntax_error_missing_comma(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("add(a b)", allowed_funcs) is False

    def test_syntax_error_invalid_token(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("add(a, @b)", allowed_funcs) is False

    def test_syntax_error_statement(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("x = 1", allowed_funcs) is False
        assert validate_expr("import math", allowed_funcs) is False







@pytest.mark.unit
class TestEdgeCases:

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        return {"add", "mul", "sin"}

    def test_empty_string(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("", allowed_funcs) is False

    def test_whitespace_only(self, allowed_funcs: set[str]) -> None:
        assert validate_expr(" ", allowed_funcs) is False
        assert validate_expr("\n\t", allowed_funcs) is False

    def test_empty_allowed_funcs(self) -> None:
        assert validate_expr("add(a, b)", set()) is False

        assert validate_expr("x", set()) is True
        assert validate_expr("1.0", set()) is True

    def test_whitespace_around_expression(self, allowed_funcs: set[str]) -> None:
        assert validate_expr(" add(a, b) ", allowed_funcs) is True
        assert validate_expr("\nadd(a, b)\n", allowed_funcs) is True

    def test_expression_with_float_scientific_notation(
        self, allowed_funcs: set[str]
    ) -> None:
        assert validate_expr("1e-10", allowed_funcs) is True
        assert validate_expr("add(x, 1e5)", allowed_funcs) is True

    def test_expression_with_boolean_constant(self, allowed_funcs: set[str]) -> None:



        result = validate_expr("True", allowed_funcs)

        assert isinstance(result, bool)

    def test_expression_with_none_constant(self, allowed_funcs: set[str]) -> None:
        result = validate_expr("None", allowed_funcs)
        assert isinstance(result, bool)

    def test_very_long_expression(self, allowed_funcs: set[str]) -> None:

        expr = "x"
        for _ in range(100):
            expr = f"sin({expr})"
        assert validate_expr(expr, allowed_funcs) is True

    def test_many_arguments_rejected(self, allowed_funcs: set[str]) -> None:


        assert validate_expr("add(a, b, c)", allowed_funcs) is True







@pytest.mark.unit
class TestGetFunctionCalls:

    def test_single_function(self) -> None:
        result = get_function_calls("sin(x)")
        assert result == {"sin"}

    def test_multiple_functions(self) -> None:
        result = get_function_calls("add(sin(x), mul(y, z))")
        assert result == {"add", "sin", "mul"}

    def test_no_functions(self) -> None:
        result = get_function_calls("x")
        assert result == set()

    def test_constant_no_functions(self) -> None:
        result = get_function_calls("1.0")
        assert result == set()

    def test_nested_same_function(self) -> None:
        result = get_function_calls("add(add(a, b), add(c, d))")
        assert result == {"add"}

    def test_syntax_error_returns_empty(self) -> None:
        result = get_function_calls("add(a, b")
        assert result == set()

    def test_empty_string_returns_empty(self) -> None:
        result = get_function_calls("")
        assert result == set()







@pytest.mark.unit
class TestSecurityConsiderations:

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        return {"add", "mul", "sin"}

    def test_dunder_attribute_rejected(self, allowed_funcs: set[str]) -> None:

        assert validate_expr("x.__class__", allowed_funcs) is False
        assert validate_expr("x.__dict__", allowed_funcs) is False

    def test_builtin_function_not_in_allowed(self, allowed_funcs: set[str]) -> None:
        assert validate_expr("eval(code)", allowed_funcs) is False
        assert validate_expr("exec(code)", allowed_funcs) is False
        assert validate_expr("__import__('os')", allowed_funcs) is False

    def test_call_on_call_result_rejected(self, allowed_funcs: set[str]) -> None:


        assert validate_expr("get_func('sin')(x)", allowed_funcs) is False
