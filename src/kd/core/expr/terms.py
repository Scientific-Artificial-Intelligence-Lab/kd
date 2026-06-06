
from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kd.core.expr.registry import FunctionRegistry


def _reject_infix_nodes(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            raise ValueError(
                "Infix operators (e.g., 'a + b') are not allowed. "
                "Use function-call IR: add(a, b), mul(a, b), etc."
            )
        if isinstance(node, ast.UnaryOp):


            if isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(
                node.operand, ast.Constant
            ):
                continue
            raise ValueError(
                "Unary operators (e.g., '-x') are not allowed. "
                "Use function-call IR: neg(x)."
            )
        if isinstance(node, ast.BoolOp):
            raise ValueError("Boolean operators are not allowed in kd IR.")


def split_terms(expr: str, registry: FunctionRegistry) -> list[str]:

    if not expr or not expr.strip():
        raise ValueError("Expression cannot be empty")


    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in expression: {e}") from e


    _reject_infix_nodes(tree)



    terms = _collect_terms(tree.body, negated=False)


    result = []
    for node, is_negated in terms:
        term_str = ast.unparse(node)
        if is_negated:
            term_str = f"neg({term_str})"
        result.append(term_str)

    return result


def _collect_terms(
    node: ast.expr,
    negated: bool,
) -> list[tuple[ast.expr, bool]]:

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id

        if func_name == "add":

            if len(node.args) != 2:
                raise ValueError(
                    f"add() requires exactly 2 arguments, got {len(node.args)}"
                )



            left_terms = _collect_terms(node.args[0], negated)
            right_terms = _collect_terms(node.args[1], negated)
            return left_terms + right_terms

        elif func_name == "sub":

            if len(node.args) != 2:
                raise ValueError(
                    f"sub() requires exactly 2 arguments, got {len(node.args)}"
                )




            left_terms = _collect_terms(node.args[0], negated)
            right_terms = _collect_terms(node.args[1], not negated)
            return left_terms + right_terms


    return [(node, negated)]
