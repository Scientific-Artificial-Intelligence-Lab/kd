
import ast
import hashlib
import warnings

from kd.core.expr.registry import FunctionRegistry


DEFAULT_MAX_DEPTH: int = 1000

_DEPRECATION_MESSAGE: str = (
    "kd.core.expr.canonicalizer is deprecated; use "
    "kd.core.equation.canonical.canonicalize_expression instead."
)


def _warn_deprecated() -> None:


    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


def canonicalize(
    node: ast.expr,
    registry: FunctionRegistry,
    _depth: int = 0,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> ast.expr:
    del registry, _depth, max_depth
    _warn_deprecated()
    from kd.core.equation.canonical import canonicalize_expression

    canonical = canonicalize_expression(ast.unparse(node))
    return ast.parse(canonical, mode="eval").body


def canonical_hash(
    code: str,
    registry: FunctionRegistry,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str:
    del registry, max_depth
    _warn_deprecated()
    from kd.core.equation.canonical import canonicalize_expression

    return hashlib.sha256(canonicalize_expression(code).encode()).hexdigest()[:16]


def canonicalize_code(
    code: str,
    registry: FunctionRegistry,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str:
    del registry, max_depth
    _warn_deprecated()
    from kd.core.equation.canonical import canonicalize_expression

    return canonicalize_expression(code)
