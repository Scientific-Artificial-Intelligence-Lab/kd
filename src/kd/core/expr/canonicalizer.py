
import ast
import hashlib

from kd.core.expr.registry import FunctionRegistry


DEFAULT_MAX_DEPTH: int = 1000


def canonicalize(
    node: ast.expr,
    registry: FunctionRegistry,
    _depth: int = 0,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> ast.expr:
    if _depth > max_depth:
        raise RecursionError(
            f"Expression depth {_depth} exceeds max_depth {max_depth}"
        )

    if isinstance(node, ast.Call):

        new_args = [
            canonicalize(arg, registry, _depth + 1, max_depth)
            for arg in node.args
        ]


        if isinstance(node.func, ast.Name):
            func_name = node.func.id



            if registry.has(func_name) and registry.is_commutative(func_name):
                new_args = sorted(new_args, key=lambda a: ast.dump(a))


        return ast.Call(func=node.func, args=new_args, keywords=[])


    return node


def canonical_hash(
    code: str,
    registry: FunctionRegistry,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str:

    tree = ast.parse(code, mode="eval")


    canonical = canonicalize(tree.body, registry, max_depth=max_depth)


    canonical_code = ast.unparse(canonical)


    return hashlib.sha256(canonical_code.encode()).hexdigest()[:16]


def canonicalize_code(
    code: str,
    registry: FunctionRegistry,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str:

    tree = ast.parse(code, mode="eval")


    canonical = canonicalize(tree.body, registry, max_depth=max_depth)


    return ast.unparse(canonical)
