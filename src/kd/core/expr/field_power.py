
from __future__ import annotations

import ast
from collections.abc import Collection
from dataclasses import dataclass

_PROTECTED_DEGREES = {"n2": 2, "n3": 3}


@dataclass(frozen=True)
class FieldPower:

    field: str
    degree: int
    protected: bool


def field_power(node: ast.expr, fields: Collection[str]) -> FieldPower | None:
    if isinstance(node, ast.Name):
        return FieldPower(node.id, 1, False) if node.id in fields else None
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.keywords
    ):
        return None
    head, args = node.func.id, node.args
    if head in _PROTECTED_DEGREES and len(args) == 1:
        argument = args[0]
        if isinstance(argument, ast.Name) and argument.id in fields:
            return FieldPower(argument.id, _PROTECTED_DEGREES[head], True)
        return None
    if head != "mul" or len(args) != 2:
        return None
    left, right = (field_power(argument, fields) for argument in args)
    if left is None or right is None or left.protected or right.protected:
        return None
    if left.field != right.field:
        return None
    return FieldPower(left.field, left.degree + right.degree, False)


__all__ = ["FieldPower", "field_power"]
