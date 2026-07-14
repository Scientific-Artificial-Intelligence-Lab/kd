
from __future__ import annotations

import ast
import hashlib

import pytest




pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from kd.core.equation.canonical import canonicalize_expression
from kd.core.expr.canonicalizer import (
    canonical_hash,
    canonicalize,
    canonicalize_code,
)
from kd.core.expr.registry import FunctionRegistry


@pytest.fixture
def registry() -> FunctionRegistry:
    return FunctionRegistry.create_default()


class TestPublicSurfacePreserved:

    def test_names_importable_from_package_root(self) -> None:
        from kd.core.expr import (
            canonical_hash as h,
        )
        from kd.core.expr import (
            canonicalize as c,
        )
        from kd.core.expr import (
            canonicalize_code as cc,
        )

        assert callable(c) and callable(h) and callable(cc)


class TestCanonicalizeCodeDelegates:

    def test_commutative_sort_uses_survivor_semantics(
        self, registry: FunctionRegistry
    ) -> None:

        assert canonicalize_code("add(b, a)", registry) == "add(a,b)"

    def test_name_vs_call_sibling_uses_string_sort(
        self, registry: FunctionRegistry
    ) -> None:

        assert canonicalize_code("add(c, mul(a, b))", registry) == "add(c,mul(a,b))"

    @pytest.mark.parametrize(
        "expression",
        [
            "mul(u, diff_x(u))",
            "sub(diff2_x(u), mul(u, diff_x(u)))",
            "add(add(c, b), a)",
            "sin(add(b, a))",
            "u",
        ],
    )
    def test_matches_survivor(
        self, registry: FunctionRegistry, expression: str
    ) -> None:
        assert canonicalize_code(expression, registry) == canonicalize_expression(
            expression
        )


class TestLegacyParametersAreDead:

    def test_registry_is_ignored(self) -> None:
        empty = FunctionRegistry()
        assert canonicalize_code("add(b, a)", empty) == "add(a,b)"

    def test_max_depth_is_not_enforced(self, registry: FunctionRegistry) -> None:

        result = canonicalize_code("sin(sin(sin(x)))", registry, max_depth=2)
        assert result == "sin(sin(sin(x)))"

    def test_canonical_hash_accepts_max_depth(
        self, registry: FunctionRegistry
    ) -> None:
        h = canonical_hash("add(b, a)", registry, max_depth=2)
        assert h == canonical_hash("add(a, b)", registry)


class TestCanonicalHash:

    def test_hash_is_sha256_of_survivor_string(
        self, registry: FunctionRegistry
    ) -> None:
        canonical = canonicalize_expression("add(b, a)")
        expected = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        assert canonical_hash("add(b, a)", registry) == expected

    def test_commutative_equivalents_share_hash(
        self, registry: FunctionRegistry
    ) -> None:
        assert canonical_hash("add(a, b)", registry) == canonical_hash(
            "add(b, a)", registry
        )

    def test_non_commutative_orders_differ(self, registry: FunctionRegistry) -> None:
        assert canonical_hash("sub(a, b)", registry) != canonical_hash(
            "sub(b, a)", registry
        )

    def test_hash_format(self, registry: FunctionRegistry) -> None:
        h = canonical_hash("add(a, b)", registry)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestCanonicalizeAstDelegates:

    def test_call_node_sorted_per_survivor(self, registry: FunctionRegistry) -> None:
        node = ast.parse("add(c, mul(a, b))", mode="eval").body
        result = canonicalize(node, registry)
        expected = ast.parse(
            canonicalize_expression("add(c, mul(a, b))"), mode="eval"
        ).body
        assert ast.dump(result) == ast.dump(expected)

    def test_name_node_passes_through(self, registry: FunctionRegistry) -> None:
        node = ast.parse("x", mode="eval").body
        result = canonicalize(node, registry)
        assert isinstance(result, ast.Name)
        assert result.id == "x"

    def test_result_is_unparsable_and_reparsable(
        self, registry: FunctionRegistry
    ) -> None:
        node = ast.parse("add(mul(b, a), x)", mode="eval").body
        code = ast.unparse(canonicalize(node, registry))
        assert ast.parse(code, mode="eval") is not None


class TestConvergedErrorContract:

    def test_numeric_constants_rejected(self, registry: FunctionRegistry) -> None:

        with pytest.raises(ValueError, match="constants"):
            canonicalize_code("mul(2, u)", registry)

    def test_constant_ast_node_rejected(self, registry: FunctionRegistry) -> None:
        node = ast.parse("1.0", mode="eval").body
        with pytest.raises(ValueError, match="constants"):
            canonicalize(node, registry)

    def test_keyword_arguments_rejected(self, registry: FunctionRegistry) -> None:

        with pytest.raises(ValueError, match="Keyword arguments"):
            canonicalize_code("add(a, b=1)", registry)

    def test_syntax_error_becomes_value_error(
        self, registry: FunctionRegistry
    ) -> None:

        with pytest.raises(ValueError, match="Invalid IR syntax"):
            canonicalize_code("add(a, b", registry)


class TestDeprecationWarning:

    def test_canonicalize_code_warns(self, registry: FunctionRegistry) -> None:
        with pytest.warns(DeprecationWarning, match="kd.core.equation.canonical"):
            canonicalize_code("add(b, a)", registry)

    def test_canonical_hash_warns(self, registry: FunctionRegistry) -> None:
        with pytest.warns(DeprecationWarning, match="kd.core.equation.canonical"):
            canonical_hash("add(b, a)", registry)

    def test_canonicalize_warns(self, registry: FunctionRegistry) -> None:
        node = ast.parse("add(b, a)", mode="eval").body
        with pytest.warns(DeprecationWarning, match="kd.core.equation.canonical"):
            canonicalize(node, registry)
