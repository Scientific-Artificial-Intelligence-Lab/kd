
import ast

import pytest

from kd.core.expr.canonicalizer import (
    DEFAULT_MAX_DEPTH,
    canonical_hash,
    canonicalize,
    canonicalize_code,
)
from kd.core.expr.registry import FunctionRegistry






@pytest.fixture
def mock_registry() -> FunctionRegistry:



    return FunctionRegistry.create_default()







@pytest.mark.smoke
class TestCanonicalizerSmoke:

    def test_canonicalize_exists(self) -> None:
        assert callable(canonicalize)

    def test_canonical_hash_exists(self) -> None:
        assert callable(canonical_hash)

    def test_canonicalize_code_exists(self) -> None:
        assert callable(canonicalize_code)







@pytest.mark.unit
class TestCommutativeReordering:

    def test_add_arguments_sorted(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("add(b, a)", mock_registry)
        assert result == "add(a, b)"

    def test_add_already_sorted(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("add(a, b)", mock_registry)
        assert result == "add(a, b)"

    def test_mul_arguments_sorted(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("mul(y, x)", mock_registry)
        assert result == "mul(x, y)"

    def test_sorting_is_lexicographic(self, mock_registry: FunctionRegistry) -> None:

        result = canonicalize_code("add(c, a)", mock_registry)
        assert result == "add(a, c)"

    def test_nested_commutative_outer(self, mock_registry: FunctionRegistry) -> None:

        result = canonicalize_code("add(mul(x, y), add(a, b))", mock_registry)



        tree1 = ast.parse(result, mode="eval")
        assert isinstance(tree1.body, ast.Call)
        assert tree1.body.func.id == "add"

    def test_nested_commutative_inner(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("add(mul(b, a), x)", mock_registry)

        assert "mul(a, b)" in result

    def test_deeply_nested_sorting(self, mock_registry: FunctionRegistry) -> None:

        result = canonicalize_code("add(mul(b, a), add(d, c))", mock_registry)



        assert "mul(a, b)" in result
        assert "add(c, d)" in result







@pytest.mark.unit
class TestNonCommutative:

    def test_sub_unchanged(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("sub(a, b)", mock_registry)
        assert result == "sub(a, b)"

    def test_sub_reversed_unchanged(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("sub(b, a)", mock_registry)
        assert result == "sub(b, a)"

    def test_div_unchanged(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("div(a, b)", mock_registry)
        assert result == "div(a, b)"

    def test_mixed_commutative_noncommutative(
        self, mock_registry: FunctionRegistry
    ) -> None:

        result = canonicalize_code("sub(mul(b, a), x)", mock_registry)

        assert result == "sub(mul(a, b), x)"







@pytest.mark.unit
class TestUnaryFunctions:

    def test_unary_unchanged(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("sin(x)", mock_registry)
        assert result == "sin(x)"

    def test_unary_nested_sorted(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("sin(add(b, a))", mock_registry)
        assert result == "sin(add(a, b))"

    def test_chain_of_unary(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("sin(cos(exp(x)))", mock_registry)
        assert result == "sin(cos(exp(x)))"







@pytest.mark.unit
class TestTerminals:

    def test_variable_unchanged(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("x", mock_registry)
        assert result == "x"

    def test_constant_unchanged(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("1.0", mock_registry)
        assert result == "1.0"

    def test_negative_constant_unchanged(self, mock_registry: FunctionRegistry) -> None:

        tree = ast.parse("-1.0", mode="eval")


        canonical = canonicalize(tree.body, mock_registry)
        assert canonical is not None







@pytest.mark.unit
class TestHashEquivalence:

    def test_same_expression_same_hash(self, mock_registry: FunctionRegistry) -> None:
        h1 = canonical_hash("add(a, b)", mock_registry)
        h2 = canonical_hash("add(a, b)", mock_registry)
        assert h1 == h2

    def test_commutative_reorder_same_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        h1 = canonical_hash("add(a, b)", mock_registry)
        h2 = canonical_hash("add(b, a)", mock_registry)
        assert h1 == h2

    def test_mul_commutative_same_hash(self, mock_registry: FunctionRegistry) -> None:
        h1 = canonical_hash("mul(x, y)", mock_registry)
        h2 = canonical_hash("mul(y, x)", mock_registry)
        assert h1 == h2

    def test_nested_commutative_same_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        h1 = canonical_hash("add(mul(a, b), mul(c, d))", mock_registry)
        h2 = canonical_hash("add(mul(d, c), mul(b, a))", mock_registry)
        assert h1 == h2

    def test_deeply_nested_same_hash(self, mock_registry: FunctionRegistry) -> None:
        h1 = canonical_hash("add(add(a, b), add(c, d))", mock_registry)
        h2 = canonical_hash("add(add(d, c), add(b, a))", mock_registry)
        assert h1 == h2







@pytest.mark.unit
class TestHashDifference:

    def test_different_expressions_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        h1 = canonical_hash("add(a, b)", mock_registry)
        h2 = canonical_hash("mul(a, b)", mock_registry)
        assert h1 != h2

    def test_sub_not_commutative_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        h1 = canonical_hash("sub(a, b)", mock_registry)
        h2 = canonical_hash("sub(b, a)", mock_registry)
        assert h1 != h2

    def test_div_not_commutative_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        h1 = canonical_hash("div(a, b)", mock_registry)
        h2 = canonical_hash("div(b, a)", mock_registry)
        assert h1 != h2

    def test_different_variables_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        h1 = canonical_hash("add(x, y)", mock_registry)
        h2 = canonical_hash("add(a, b)", mock_registry)
        assert h1 != h2

    def test_different_constants_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        h1 = canonical_hash("add(x, 1.0)", mock_registry)
        h2 = canonical_hash("add(x, 2.0)", mock_registry)
        assert h1 != h2







@pytest.mark.unit
class TestHashFormat:

    def test_hash_is_string(self, mock_registry: FunctionRegistry) -> None:
        h = canonical_hash("add(a, b)", mock_registry)
        assert isinstance(h, str)

    def test_hash_is_hexadecimal(self, mock_registry: FunctionRegistry) -> None:
        h = canonical_hash("add(a, b)", mock_registry)

        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_length_is_16(self, mock_registry: FunctionRegistry) -> None:
        h = canonical_hash("add(a, b)", mock_registry)
        assert len(h) == 16

    def test_hash_deterministic(self, mock_registry: FunctionRegistry) -> None:
        results = [canonical_hash("sin(add(x, y))", mock_registry) for _ in range(10)]
        assert len(set(results)) == 1







@pytest.mark.unit
class TestErrorHandling:

    def test_canonical_hash_syntax_error(self, mock_registry: FunctionRegistry) -> None:
        with pytest.raises(SyntaxError):
            canonical_hash("add(a, b", mock_registry)

    def test_canonicalize_code_syntax_error(
        self, mock_registry: FunctionRegistry
    ) -> None:
        with pytest.raises(SyntaxError):
            canonicalize_code("add(a, b", mock_registry)







@pytest.mark.unit
class TestRecursionDepthLimit:

    def test_default_max_depth_constant_exists(self) -> None:
        assert DEFAULT_MAX_DEPTH == 1000

    def test_shallow_expression_succeeds(
        self, mock_registry: FunctionRegistry
    ) -> None:
        result = canonicalize_code("add(a, b)", mock_registry, max_depth=10)
        assert result == "add(a, b)"

    def test_exceeds_max_depth_raises_recursion_error(
        self, mock_registry: FunctionRegistry
    ) -> None:

        expr = "sin(sin(sin(x)))"


        with pytest.raises(RecursionError) as exc_info:
            canonicalize_code(expr, mock_registry, max_depth=2)
        assert "exceeds max_depth" in str(exc_info.value)

    def test_exact_max_depth_succeeds(
        self, mock_registry: FunctionRegistry
    ) -> None:

        expr = "sin(x)"
        result = canonicalize_code(expr, mock_registry, max_depth=1)
        assert result == "sin(x)"

    def test_canonical_hash_respects_max_depth(
        self, mock_registry: FunctionRegistry
    ) -> None:
        expr = "sin(sin(sin(x)))"
        with pytest.raises(RecursionError):
            canonical_hash(expr, mock_registry, max_depth=2)

    def test_canonicalize_direct_respects_max_depth(
        self, mock_registry: FunctionRegistry
    ) -> None:
        tree = ast.parse("sin(sin(sin(x)))", mode="eval")
        with pytest.raises(RecursionError):
            canonicalize(tree.body, mock_registry, max_depth=2)

    def test_deeply_nested_within_limit_succeeds(
        self, mock_registry: FunctionRegistry
    ) -> None:

        expr = "add(add(add(add(add(x, y), z), w), v), u)"

        result = canonicalize_code(expr, mock_registry)
        assert "add" in result

    def test_error_message_includes_depth_info(
        self, mock_registry: FunctionRegistry
    ) -> None:
        expr = "sin(sin(x))"
        with pytest.raises(RecursionError) as exc_info:
            canonicalize_code(expr, mock_registry, max_depth=1)
        error_msg = str(exc_info.value)
        assert "2" in error_msg
        assert "1" in error_msg







@pytest.mark.unit
class TestEdgeCases:

    def test_single_variable(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("x", mock_registry)
        assert result == "x"

    def test_single_constant(self, mock_registry: FunctionRegistry) -> None:
        result = canonicalize_code("42", mock_registry)
        assert result == "42"

    def test_whitespace_preserved_or_normalized(
        self, mock_registry: FunctionRegistry
    ) -> None:
        result = canonicalize_code("add( a, b )", mock_registry)

        assert result == "add(a, b)"

    def test_many_commutative_arguments(
        self, mock_registry: FunctionRegistry
    ) -> None:


        result = canonicalize_code("add(add(c, b), a)", mock_registry)


        assert "add(b, c)" in result

    def test_complex_real_world_expression(
        self, mock_registry: FunctionRegistry
    ) -> None:

        expr = "add(mul(u, u_x), mul(C, u_xx))"
        result = canonicalize_code(expr, mock_registry)


        tree = ast.parse(result, mode="eval")
        assert isinstance(tree.body, ast.Call)

    def test_equivalent_complex_expressions_same_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:


        h1 = canonical_hash("add(mul(u, u_x), mul(C, u_xx))", mock_registry)
        h2 = canonical_hash("add(mul(u_xx, C), mul(u_x, u))", mock_registry)
        assert h1 == h2







@pytest.mark.unit
class TestASTNodeHandling:

    def test_canonicalize_name_node(self, mock_registry: FunctionRegistry) -> None:
        tree = ast.parse("x", mode="eval")
        result = canonicalize(tree.body, mock_registry)
        assert isinstance(result, ast.Name)
        assert result.id == "x"

    def test_canonicalize_constant_node(self, mock_registry: FunctionRegistry) -> None:
        tree = ast.parse("1.0", mode="eval")
        result = canonicalize(tree.body, mock_registry)
        assert isinstance(result, ast.Constant)
        assert result.value == 1.0

    def test_canonicalize_call_node(self, mock_registry: FunctionRegistry) -> None:
        tree = ast.parse("sin(x)", mode="eval")
        result = canonicalize(tree.body, mock_registry)
        assert isinstance(result, ast.Call)

    def test_canonicalize_returns_valid_ast(
        self, mock_registry: FunctionRegistry
    ) -> None:
        tree = ast.parse("add(mul(b, a), x)", mode="eval")
        result = canonicalize(tree.body, mock_registry)


        code = ast.unparse(result)
        assert isinstance(code, str)
        assert len(code) > 0


        reparsed = ast.parse(code, mode="eval")
        assert reparsed is not None
