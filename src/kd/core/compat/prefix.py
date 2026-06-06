
import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kd.core.expr.registry import FunctionRegistry


def python_to_prefix(code: str) -> list[str]:

    if not code or not code.strip():
        raise ValueError("Empty expression")


    tree = ast.parse(code, mode="eval")
    return _traverse(tree.body)


def _traverse(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Call):

        if not isinstance(node.func, ast.Name):
            raise ValueError(
                f"Only simple function calls supported, got {type(node.func).__name__}"
            )
        tokens = [node.func.id]
        for arg in node.args:
            tokens.extend(_traverse(arg))
        return tokens

    elif isinstance(node, ast.Name):

        return [node.id]

    elif isinstance(node, ast.Constant):

        return [_format_constant(node.value)]

    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):

        if isinstance(node.operand, ast.Constant):
            value = node.operand.value
            if isinstance(value, (int, float)):
                return [_format_constant(-value)]
            raise ValueError(f"Unary minus not supported on {type(value).__name__}")

        raise ValueError("Unary minus on non-constant not supported; use neg()")

    elif isinstance(node, ast.BinOp):
        raise ValueError(
            "Binary operators (+, -, *, /) not supported; use function calls"
        )

    elif isinstance(node, ast.Subscript):
        raise ValueError("Subscript operations not supported")

    elif isinstance(node, ast.Attribute):
        raise ValueError("Attribute access not supported")

    elif isinstance(node, ast.List):
        raise ValueError("List literals not supported")

    elif isinstance(node, ast.Dict):
        raise ValueError("Dict literals not supported")

    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def _format_constant(value: object) -> str:
    if isinstance(value, float):

        formatted = repr(value)

        return formatted
    return str(value)


def prefix_to_python(tokens: list[str], registry: "FunctionRegistry") -> str:
    if not tokens:
        raise ValueError("Empty token list")

    stack: list[str] = []


    for token in reversed(tokens):
        if registry.has(token):
            arity = registry.get_arity(token)
            if arity == 0:

                stack.append(token)
            else:

                if len(stack) < arity:
                    raise ValueError(
                        f"Not enough arguments for '{token}': "
                        f"need {arity}, have {len(stack)}"
                    )
                args = [stack.pop() for _ in range(arity)]
                expr = f"{token}({', '.join(args)})"
                stack.append(expr)
        else:

            stack.append(token)


    if len(stack) != 1:
        raise ValueError(
            f"Unbalanced expression: {len(stack)} elements remain on stack"
        )

    return stack[0]
