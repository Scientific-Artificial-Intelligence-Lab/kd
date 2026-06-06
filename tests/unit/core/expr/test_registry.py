
import pytest
import torch

from kd.core.expr.registry import FunctionRegistry






@pytest.mark.smoke
class TestFunctionRegistrySmoke:

    def test_registry_can_be_instantiated(self) -> None:
        reg = FunctionRegistry()
        assert reg is not None

    def test_create_default_exists(self) -> None:
        assert hasattr(FunctionRegistry, "create_default")
        assert callable(FunctionRegistry.create_default)

    def test_create_default_returns_registry(self) -> None:
        reg = FunctionRegistry.create_default()
        assert isinstance(reg, FunctionRegistry)

    def test_required_methods_exist(self) -> None:
        required_methods = [
            "register",
            "get_context",
            "is_commutative",
            "get_arity",
            "get_func",
            "has",
            "list_names",
            "get_by_arity",
        ]
        for method in required_methods:
            assert hasattr(FunctionRegistry, method), f"Missing method: {method}"







@pytest.mark.unit
class TestFunctionRegistryRegistration:

    def test_register_and_query(self) -> None:
        reg = FunctionRegistry()
        reg.register("my_add", lambda a, b: a + b, arity=2, commutative=True)

        assert reg.has("my_add")
        assert reg.get_arity("my_add") == 2
        assert reg.is_commutative("my_add") is True

    def test_register_multiple_functions(self) -> None:
        reg = FunctionRegistry()
        reg.register("op1", lambda x: x, arity=1)
        reg.register("op2", lambda x, y: x + y, arity=2)
        reg.register("op3", lambda x, y: x * y, arity=2, commutative=True)

        assert reg.has("op1")
        assert reg.has("op2")
        assert reg.has("op3")
        assert len(reg.list_names()) == 3

    def test_register_duplicate_raises(self) -> None:
        reg = FunctionRegistry()
        reg.register("my_op", lambda x: x, arity=1)

        with pytest.raises(ValueError, match="already exists"):
            reg.register("my_op", lambda x: x * 2, arity=1)

    def test_has_returns_false_for_unregistered(self) -> None:
        reg = FunctionRegistry()
        assert reg.has("nonexistent") is False

    def test_query_unregistered_raises_keyerror(self) -> None:
        reg = FunctionRegistry()

        with pytest.raises(KeyError):
            reg.get_arity("nonexistent")

        with pytest.raises(KeyError):
            reg.is_commutative("nonexistent")

        with pytest.raises(KeyError):
            reg.get_func("nonexistent")







@pytest.mark.unit
class TestGetFunc:

    def test_get_func_returns_callable(self) -> None:
        reg = FunctionRegistry()

        def my_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        reg.register("my_add", my_func, arity=2)

        retrieved = reg.get_func("my_add")
        assert callable(retrieved)

    def test_get_func_returns_correct_function(self) -> None:
        reg = FunctionRegistry()

        def custom_op(x: torch.Tensor) -> torch.Tensor:
            return x * 3

        reg.register("triple", custom_op, arity=1)

        retrieved = reg.get_func("triple")
        test_input = torch.tensor([1.0, 2.0])
        expected = torch.tensor([3.0, 6.0])

        torch.testing.assert_close(retrieved(test_input), expected)

    def test_get_func_for_default_operators(self) -> None:
        reg = FunctionRegistry.create_default()

        add_func = reg.get_func("add")
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])

        result = add_func(a, b)
        expected = torch.tensor([4.0, 6.0])
        torch.testing.assert_close(result, expected)

    def test_get_func_vs_get_context(self) -> None:
        reg = FunctionRegistry.create_default()


        mul_direct = reg.get_func("mul")


        ctx = reg.get_context()
        mul_context = ctx["mul"]


        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])

        torch.testing.assert_close(mul_direct(a, b), mul_context(a, b))







@pytest.mark.unit
class TestDefaultRegistry:

    @pytest.fixture
    def default_reg(self) -> FunctionRegistry:
        return FunctionRegistry.create_default()

    def test_binary_operators_exist(self, default_reg: FunctionRegistry) -> None:
        binary_ops = ["add", "mul", "sub", "div"]
        for name in binary_ops:
            assert default_reg.has(name), f"Missing binary operator: {name}"
            assert default_reg.get_arity(name) == 2, f"{name} should have arity 2"

    def test_unary_operators_exist(self, default_reg: FunctionRegistry) -> None:
        unary_ops = ["sin", "cos", "exp", "log", "neg", "n2", "n3", "lap"]
        for name in unary_ops:
            assert default_reg.has(name), f"Missing unary operator: {name}"
            assert default_reg.get_arity(name) == 1, f"{name} should have arity 1"

    def test_commutative_operators(self, default_reg: FunctionRegistry) -> None:
        assert default_reg.is_commutative("add") is True
        assert default_reg.is_commutative("mul") is True
        assert default_reg.is_commutative("sub") is False
        assert default_reg.is_commutative("div") is False
        assert default_reg.is_commutative("sin") is False

    def test_commutative_math_property_add(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])

        result_ab = ctx["add"](a, b)
        result_ba = ctx["add"](b, a)

        torch.testing.assert_close(result_ab, result_ba)

    def test_commutative_math_property_mul(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])

        result_ab = ctx["mul"](a, b)
        result_ba = ctx["mul"](b, a)

        torch.testing.assert_close(result_ab, result_ba)

    def test_non_commutative_math_property_sub(
        self, default_reg: FunctionRegistry
    ) -> None:
        ctx = default_reg.get_context()
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])

        result_ab = ctx["sub"](a, b)
        result_ba = ctx["sub"](b, a)


        assert not torch.allclose(result_ab, result_ba)

    def test_operator_count(self, default_reg: FunctionRegistry) -> None:



        names = default_reg.list_names()
        assert len(names) == 12

    def test_lap_stub_prevents_registry_execution(
        self, default_reg: FunctionRegistry
    ) -> None:
        lap = default_reg.get_func("lap")

        with pytest.raises(NotImplementedError, match="context-aware executor"):
            lap(torch.tensor([1.0, 2.0]))

    def test_get_by_arity(self, default_reg: FunctionRegistry) -> None:
        unary = default_reg.get_by_arity(1)
        binary = default_reg.get_by_arity(2)

        assert len(unary) == 8
        assert len(binary) == 4


        for name in unary:
            assert default_reg.get_arity(name) == 1
        for name in binary:
            assert default_reg.get_arity(name) == 2

    def test_get_by_arity_empty(self, default_reg: FunctionRegistry) -> None:
        result = default_reg.get_by_arity(3)
        assert result == []







@pytest.mark.unit
class TestContextGeneration:

    def test_get_context_returns_dict(self) -> None:
        reg = FunctionRegistry()
        reg.register("my_op", lambda x: x * 2, arity=1)

        ctx = reg.get_context()
        assert isinstance(ctx, dict)

    def test_get_context_contains_functions(self) -> None:
        reg = FunctionRegistry()
        my_func = lambda x: x * 2
        reg.register("double", my_func, arity=1)

        ctx = reg.get_context()
        assert "double" in ctx
        assert callable(ctx["double"])

    def test_context_usable_with_eval_simple(self) -> None:
        reg = FunctionRegistry()
        reg.register("add", lambda a, b: a + b, arity=2)
        reg.register("mul", lambda a, b: a * b, arity=2)

        ctx = reg.get_context()

        ctx["x"] = 2
        ctx["y"] = 3

        result = eval("add(mul(x, y), x)", ctx)
        assert result == 8

    def test_context_usable_with_eval_tensor(self) -> None:
        reg = FunctionRegistry.create_default()
        ctx = reg.get_context()


        ctx["x"] = torch.tensor([1.0, 2.0, 3.0])
        ctx["y"] = torch.tensor([4.0, 5.0, 6.0])

        result = eval("add(x, y)", ctx)
        expected = torch.tensor([5.0, 7.0, 9.0])
        torch.testing.assert_close(result, expected)

    def test_context_usable_for_nested_expressions(self) -> None:
        reg = FunctionRegistry.create_default()
        ctx = reg.get_context()
        ctx["u"] = torch.tensor([1.0, 2.0])


        result = eval("sin(add(u, mul(u, u)))", ctx)
        expected = torch.sin(ctx["u"] + ctx["u"] * ctx["u"])
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)







@pytest.mark.unit
class TestDiffOperators:

    def test_register_diff_x_operator(self) -> None:
        reg = FunctionRegistry()


        def diff_x_func(u: torch.Tensor) -> torch.Tensor:
            return u

        reg.register("diff_x", diff_x_func, arity=1)

        assert reg.has("diff_x")
        assert reg.get_arity("diff_x") == 1

    def test_register_diff2_x_operator(self) -> None:
        reg = FunctionRegistry()

        def diff2_x_func(u: torch.Tensor) -> torch.Tensor:
            return u

        reg.register("diff2_x", diff2_x_func, arity=1)

        assert reg.has("diff2_x")
        assert reg.get_arity("diff2_x") == 1

    def test_register_multiple_diff_operators(self) -> None:
        reg = FunctionRegistry()


        reg.register("diff_x", lambda u: u, arity=1)
        reg.register("diff2_x", lambda u: u, arity=1)
        reg.register("diff_t", lambda u: u, arity=1)


        assert reg.has("diff_x")
        assert reg.has("diff2_x")
        assert reg.has("diff_t")


        unary_ops = reg.get_by_arity(1)
        assert "diff_x" in unary_ops
        assert "diff2_x" in unary_ops
        assert "diff_t" in unary_ops

    def test_diff_operator_in_expression(self) -> None:
        reg = FunctionRegistry()

        reg.register("diff_x", lambda u: u * 2, arity=1)

        ctx = reg.get_context()
        ctx["u"] = torch.tensor([1.0, 2.0, 3.0])

        result = eval("diff_x(u)", ctx)
        expected = torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(result, expected)







@pytest.mark.unit
class TestParameterValidation:

    def test_register_negative_arity_raises(self) -> None:
        reg = FunctionRegistry()

        with pytest.raises(ValueError, match="arity"):
            reg.register("bad_op", lambda x: x, arity=-1)

    def test_register_none_func_raises(self) -> None:
        reg = FunctionRegistry()

        with pytest.raises((TypeError, ValueError)):
            reg.register("bad_op", None, arity=1)

    def test_register_empty_name_raises(self) -> None:
        reg = FunctionRegistry()

        with pytest.raises(ValueError, match="name"):
            reg.register("", lambda x: x, arity=1)

    def test_register_non_callable_raises(self) -> None:
        reg = FunctionRegistry()

        with pytest.raises(TypeError):
            reg.register("bad_op", "not a function", arity=1)







@pytest.mark.unit
class TestDangerousNames:

    def test_reject_builtins_name(self) -> None:
        reg = FunctionRegistry()

        with pytest.raises(ValueError, match="reserved|dangerous|forbidden"):
            reg.register("__builtins__", lambda x: x, arity=1)

    def test_reject_dunder_names(self) -> None:
        reg = FunctionRegistry()
        dangerous_names = [
            "__class__",
            "__import__",
            "__globals__",
            "__code__",
            "__dict__",
        ]

        for name in dangerous_names:
            with pytest.raises(ValueError, match="reserved|dangerous|forbidden"):
                reg.register(name, lambda x: x, arity=1)

    def test_reject_eval_exec_names(self) -> None:
        reg = FunctionRegistry()

        with pytest.raises(ValueError, match="reserved|dangerous|forbidden"):
            reg.register("eval", lambda x: x, arity=1)

        with pytest.raises(ValueError, match="reserved|dangerous|forbidden"):
            reg.register("exec", lambda x: x, arity=1)

    def test_allow_normal_names(self) -> None:
        reg = FunctionRegistry()


        reg.register("my_func", lambda x: x, arity=1)
        reg.register("add2", lambda a, b: a + b, arity=2)
        reg.register("diff_x", lambda u: u, arity=1)
        reg.register("u_xx", lambda: 1.0, arity=0)

        assert reg.has("my_func")
        assert reg.has("add2")
        assert reg.has("diff_x")
        assert reg.has("u_xx")







@pytest.mark.numerical
class TestDefaultOperatorsSafety:

    @pytest.fixture
    def default_reg(self) -> FunctionRegistry:
        return FunctionRegistry.create_default()

    def test_div_by_zero_is_safe(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["a"] = torch.tensor([1.0, 2.0, 3.0])
        ctx["b"] = torch.tensor([0.0, 0.0, 0.0])

        result = eval("div(a, b)", ctx)
        assert torch.isfinite(result).all(), "div should never produce NaN/Inf"

    def test_div_near_zero_is_safe(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["a"] = torch.tensor([1.0])
        ctx["b"] = torch.tensor([1e-15])

        result = eval("div(a, b)", ctx)
        assert torch.isfinite(result).all(), "div should handle near-zero"

    def test_div_preserves_sign_positive_over_positive(
        self, default_reg: FunctionRegistry
    ) -> None:
        ctx = default_reg.get_context()
        ctx["a"] = torch.tensor([1.0, 2.0, 3.0])
        ctx["b"] = torch.tensor([1e-20, 1e-10, 1e-5])

        result = eval("div(a, b)", ctx)
        assert torch.isfinite(result).all()
        assert (result > 0).all(), "positive / positive should be positive"

    def test_div_preserves_sign_negative_over_positive(
        self, default_reg: FunctionRegistry
    ) -> None:
        ctx = default_reg.get_context()
        ctx["a"] = torch.tensor([-1.0, -2.0, -3.0])
        ctx["b"] = torch.tensor([1e-20, 1e-10, 1e-5])

        result = eval("div(a, b)", ctx)
        assert torch.isfinite(result).all()
        assert (result < 0).all(), "negative / positive should be negative"

    def test_div_preserves_sign_positive_over_negative(
        self, default_reg: FunctionRegistry
    ) -> None:
        ctx = default_reg.get_context()
        ctx["a"] = torch.tensor([1.0, 2.0, 3.0])
        ctx["b"] = torch.tensor([-1e-20, -1e-10, -1e-5])

        result = eval("div(a, b)", ctx)
        assert torch.isfinite(result).all()
        assert (result < 0).all(), "positive / negative should be negative"

    def test_div_magnitude_reasonable(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["a"] = torch.tensor([6.0])
        ctx["b"] = torch.tensor([2.0])

        result = eval("div(a, b)", ctx)

        torch.testing.assert_close(result, torch.tensor([3.0]), rtol=1e-5, atol=1e-8)

    def test_exp_large_input_is_safe(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["x"] = torch.tensor([100.0, 500.0, 1000.0])

        result = eval("exp(x)", ctx)
        assert torch.isfinite(result).all(), "exp should clamp large inputs"

    def test_neg_correctness(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["x"] = torch.tensor([1.0, -2.0, 0.0])

        result = eval("neg(x)", ctx)
        expected = torch.tensor([-1.0, 2.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_n2_correctness(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["x"] = torch.tensor([2.0, 3.0, -4.0])

        result = eval("n2(x)", ctx)
        expected = torch.tensor([4.0, 9.0, 16.0])
        torch.testing.assert_close(result, expected)

    def test_n3_correctness(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["x"] = torch.tensor([2.0, 3.0, -2.0])

        result = eval("n3(x)", ctx)
        expected = torch.tensor([8.0, 27.0, -8.0])
        torch.testing.assert_close(result, expected)

    def test_trig_functions_correctness(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["x"] = torch.tensor([0.0, torch.pi / 2, torch.pi])

        sin_result = eval("sin(x)", ctx)
        cos_result = eval("cos(x)", ctx)


        torch.testing.assert_close(
            sin_result, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=1e-5
        )

        torch.testing.assert_close(
            cos_result, torch.tensor([1.0, 0.0, -1.0]), atol=1e-6, rtol=1e-5
        )

    def test_add_sub_mul_correctness(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["a"] = torch.tensor([1.0, 2.0, 3.0])
        ctx["b"] = torch.tensor([4.0, 5.0, 6.0])

        add_result = eval("add(a, b)", ctx)
        sub_result = eval("sub(a, b)", ctx)
        mul_result = eval("mul(a, b)", ctx)

        torch.testing.assert_close(add_result, torch.tensor([5.0, 7.0, 9.0]))
        torch.testing.assert_close(sub_result, torch.tensor([-3.0, -3.0, -3.0]))
        torch.testing.assert_close(mul_result, torch.tensor([4.0, 10.0, 18.0]))

    def test_nested_expression_safety(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["x"] = torch.tensor([0.0, 1e-15, 1e10])


        result = eval("exp(div(x, add(x, x)))", ctx)
        assert torch.isfinite(result).all(), "Nested expressions should be safe"







@pytest.mark.numerical
class TestExtremeValues:

    @pytest.fixture
    def default_reg(self) -> FunctionRegistry:
        return FunctionRegistry.create_default()

    def test_very_large_values(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["x"] = torch.tensor([1e30, 1e35, 1e38])
        ctx["y"] = torch.tensor([1e30, 1e35, 1e38])


        result = eval("add(x, y)", ctx)
        assert torch.isfinite(result).all()

    def test_very_small_values(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["x"] = torch.tensor([1e-30, 1e-35, 1e-38])


        ctx["one"] = torch.tensor([1.0])
        result = eval("div(one, x)", ctx)
        assert torch.isfinite(result).all()

    def test_mixed_extreme_values(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["big"] = torch.tensor([1e20])
        ctx["small"] = torch.tensor([1e-20])

        result = eval("mul(div(big, small), small)", ctx)

        assert torch.isfinite(result).all()

    def test_zero_handling(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["zero"] = torch.tensor([0.0])
        ctx["one"] = torch.tensor([1.0])


        assert torch.isfinite(eval("add(zero, one)", ctx)).all()
        assert torch.isfinite(eval("mul(zero, one)", ctx)).all()
        assert torch.isfinite(eval("div(zero, one)", ctx)).all()
        assert torch.isfinite(eval("div(one, zero)", ctx)).all()
        assert torch.isfinite(eval("n2(zero)", ctx)).all()
        assert torch.isfinite(eval("exp(zero)", ctx)).all()

    def test_empty_tensor_handling(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["empty"] = torch.tensor([])


        result = eval("sin(empty)", ctx)
        assert result.shape == torch.Size([0])







@pytest.mark.numerical
class TestLogOperator:

    @pytest.fixture
    def default_reg(self) -> FunctionRegistry:
        return FunctionRegistry.create_default()

    def test_log_registered(self, default_reg: FunctionRegistry) -> None:
        assert default_reg.has("log")
        assert default_reg.get_arity("log") == 1

    def test_log_correctness(self, default_reg: FunctionRegistry) -> None:
        log_fn = default_reg.get_func("log")
        x = torch.tensor([1.0, torch.e, torch.e**2])
        result = log_fn(x)
        expected = torch.tensor([0.0, 1.0, 2.0])
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_log_zero_safe(self, default_reg: FunctionRegistry) -> None:
        log_fn = default_reg.get_func("log")
        result = log_fn(torch.tensor([0.0]))
        assert torch.isfinite(result).all(), "log(0) must not produce -inf"

    def test_log_negative_safe(self, default_reg: FunctionRegistry) -> None:
        log_fn = default_reg.get_func("log")
        result = log_fn(torch.tensor([-1.0, -0.5]))
        assert torch.isfinite(result).all(), "log(negative) must not produce NaN"

    def test_log_in_expression(self, default_reg: FunctionRegistry) -> None:
        ctx = default_reg.get_context()
        ctx["x"] = torch.tensor([1.0, torch.e])
        result = eval("log(x)", ctx)
        expected = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)







@pytest.mark.numerical
class TestProtectedN2N3:

    @pytest.fixture
    def default_reg(self) -> FunctionRegistry:
        return FunctionRegistry.create_default()



    def test_n2_normal_input_unaffected(self, default_reg: FunctionRegistry) -> None:
        n2 = default_reg.get_func("n2")
        x = torch.tensor([3.0, -4.0, 0.0, 1.0, -1.0, 100.0])
        expected = torch.tensor([9.0, 16.0, 0.0, 1.0, 1.0, 10000.0])
        torch.testing.assert_close(n2(x), expected)

    def test_n2_clamps_large_input(self, default_reg: FunctionRegistry) -> None:
        n2 = default_reg.get_func("n2")
        result = n2(torch.tensor([1e7]))
        expected = torch.tensor([1e12])
        torch.testing.assert_close(result, expected)

    def test_n2_clamps_large_negative_input(
        self, default_reg: FunctionRegistry
    ) -> None:
        n2 = default_reg.get_func("n2")
        result = n2(torch.tensor([-1e7]))
        expected = torch.tensor([1e12])
        torch.testing.assert_close(result, expected)

    def test_n2_output_always_finite(self, default_reg: FunctionRegistry) -> None:
        n2 = default_reg.get_func("n2")
        x = torch.tensor([1e10, 1e20, 1e30, -1e10, -1e20])
        result = n2(x)
        assert torch.isfinite(result).all(), "n2 must always produce finite output"

    def test_n2_at_clamp_boundary(self, default_reg: FunctionRegistry) -> None:
        n2 = default_reg.get_func("n2")
        result = n2(torch.tensor([1e6]))
        expected = torch.tensor([1e12])
        torch.testing.assert_close(result, expected)



    def test_n3_normal_input_unaffected(self, default_reg: FunctionRegistry) -> None:
        n3 = default_reg.get_func("n3")
        x = torch.tensor([2.0, -2.0, 0.0, 1.0, -1.0])
        expected = torch.tensor([8.0, -8.0, 0.0, 1.0, -1.0])
        torch.testing.assert_close(n3(x), expected)

    def test_n3_clamps_large_input(self, default_reg: FunctionRegistry) -> None:
        n3 = default_reg.get_func("n3")
        result = n3(torch.tensor([1e7]))
        expected = torch.tensor([1e18])
        torch.testing.assert_close(result, expected)

    def test_n3_clamps_large_negative_preserves_sign(
        self, default_reg: FunctionRegistry
    ) -> None:
        n3 = default_reg.get_func("n3")
        result = n3(torch.tensor([-1e7]))
        expected = torch.tensor([-1e18])
        torch.testing.assert_close(result, expected)

    def test_n3_output_always_finite(self, default_reg: FunctionRegistry) -> None:
        n3 = default_reg.get_func("n3")
        x = torch.tensor([1e10, 1e20, 1e30, -1e10, -1e20])
        result = n3(x)
        assert torch.isfinite(result).all(), "n3 must always produce finite output"



    def test_n2_gradient_normal_input(self, default_reg: FunctionRegistry) -> None:
        n2 = default_reg.get_func("n2")
        x = torch.tensor([2.0], requires_grad=True)
        y = n2(x)
        y.backward()
        torch.testing.assert_close(x.grad, torch.tensor([4.0]))

    def test_n2_gradient_clamped_input(self, default_reg: FunctionRegistry) -> None:
        n2 = default_reg.get_func("n2")
        x = torch.tensor([1e7], requires_grad=True)
        y = n2(x)
        y.backward()
        torch.testing.assert_close(x.grad, torch.tensor([0.0]))

    def test_n3_gradient_normal_input(self, default_reg: FunctionRegistry) -> None:
        n3 = default_reg.get_func("n3")
        x = torch.tensor([2.0], requires_grad=True)
        y = n3(x)
        y.backward()
        torch.testing.assert_close(x.grad, torch.tensor([12.0]))

    def test_n3_gradient_clamped_input(self, default_reg: FunctionRegistry) -> None:
        n3 = default_reg.get_func("n3")
        x = torch.tensor([1e7], requires_grad=True)
        y = n3(x)
        y.backward()
        torch.testing.assert_close(x.grad, torch.tensor([0.0]))
