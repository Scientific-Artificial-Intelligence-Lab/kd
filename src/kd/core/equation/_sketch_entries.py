
from __future__ import annotations

import ast
import functools
import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass

from kd.core.equation.canonical import skeletonize_constants
from kd.core.equation.signature import (
    _regression_entry,
    _rewrite_aliases,
    law_term_entry,
)
from kd.core.expr.executor import UNITY_TOKEN
from kd.core.expr.term_key import numeric_scalar_value, split_scalar_factor

_COMMUTATIVE = frozenset({"add", "mul"})
_POWERS = {"n2": 2, "n3": 3}


@dataclass(frozen=True)
class SketchEntry:

    key: str
    value: float
    spelling: str
    from_power: bool = False


def sketch_entries(
    term_ir: str, coefficient: float, *, regression: bool
) -> list[SketchEntry]:
    if not regression:
        key, value = law_term_entry(term_ir, coefficient)
        return [SketchEntry(key, value, key)]
    return [_entry(term_ir, coefficient, piece) for piece in _pieces(_parse(term_ir))]


def whole_entry(term_ir: str, coefficient: float) -> SketchEntry:
    scale, node = split_scalar_factor(_parse(term_ir))
    return _entry(term_ir, coefficient, (scale, node, False))


def whole_term_key(term_ir: str) -> str:
    spelling = _spelling(_rewrite_aliases(_parse(term_ir)))
    return _regression_entry(spelling, 1.0)[0]


def aggregate_entries(entries: Iterable[SketchEntry]) -> dict[str, SketchEntry]:
    values: dict[str, list[float]] = {}
    spellings: dict[str, str] = {}
    for entry in entries:
        known = spellings.setdefault(entry.key, entry.spelling)
        if known != entry.spelling:
            raise ValueError(
                f"law term {entry.key!r} is spelled with different fitted "
                f"constants ({known!r} and {entry.spelling!r}); a sketch term "
                "is one column"
            )
        values.setdefault(entry.key, []).append(entry.value)
    return {
        key: SketchEntry(key, math.fsum(summands), spellings[key])
        for key, summands in values.items()
    }


def _parse(term_ir: str) -> ast.expr:
    try:
        return ast.parse(term_ir.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid IR syntax: {exc}") from exc


def _call(node: ast.expr, head: str, arity: int) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == head
        and len(node.args) == arity
        and not node.keywords
    )



_Piece = tuple[float, ast.expr | None, bool]


def _pieces(node: ast.expr) -> list[_Piece]:
    literal = numeric_scalar_value(node)
    if literal is not None:
        return [(literal, None, False)]
    if isinstance(node, ast.Name) and node.id == UNITY_TOKEN:
        return [(1.0, None, False)]
    if not isinstance(node, ast.Call):
        return [(1.0, node, False)]
    if _call(node, "add", 2):
        return _pieces(node.args[0]) + _pieces(node.args[1])
    if _call(node, "sub", 2):
        return _pieces(node.args[0]) + _negated(_pieces(node.args[1]))
    if _call(node, "neg", 1):
        return _negated(_pieces(node.args[0]))
    if _call(node, "mul", 2):
        return [
            (left_scale * right_scale, _product(left, right), left_power or right_power)
            for left_scale, left, left_power in _pieces(node.args[0])
            for right_scale, right, right_power in _pieces(node.args[1])
        ]
    if _call(node, "div", 2):
        return _quotient(node.args[0], node.args[1])
    for head, exponent in _POWERS.items():
        if _call(node, head, 1):
            return _power(node, exponent)
    return [(1.0, node, False)]


def _entry(term_ir: str, coefficient: float, piece: _Piece) -> SketchEntry:
    scale, node, from_power = piece
    spelling = UNITY_TOKEN if node is None else _spelling(_rewrite_aliases(node))
    key, value = _regression_entry(spelling, coefficient * scale)
    if not math.isfinite(value):
        raise ValueError(f"term {term_ir!r} gives {key!r} a non-finite coefficient")
    return SketchEntry(key, value, spelling, from_power)


def _negated(pieces: list[_Piece]) -> list[_Piece]:
    return [(-scale, piece, power) for scale, piece, power in pieces]


def _power(node: ast.Call, exponent: int) -> list[_Piece]:
    base = _pieces(node.args[0])
    if len(base) == 1:
        return [(1.0, node, False)]
    return [
        (
            math.prod(scale for scale, _piece, _power in factors),
            functools.reduce(_product, (piece for _scale, piece, _power in factors)),
            True,
        )
        for factors in itertools.product(base, repeat=exponent)
    ]


def _quotient(numerator: ast.expr, denominator: ast.expr) -> list[_Piece]:
    divisor = numeric_scalar_value(denominator)
    if divisor == 0.0:
        raise ValueError("term divides by the literal 0")
    if divisor is not None:
        return [
            (scale / divisor, piece, power)
            for scale, piece, power in _pieces(numerator)
        ]
    reciprocal = ast.Call(func=ast.Name(id="recip"), args=[denominator], keywords=[])
    return [
        (scale, _product(piece, reciprocal), power)
        for scale, piece, power in _pieces(numerator)
    ]


def _product(left: ast.expr | None, right: ast.expr | None) -> ast.expr | None:
    if left is None:
        return right
    if right is None:
        return left
    return ast.Call(func=ast.Name(id="mul"), args=[left, right], keywords=[])


def _spelling(node: ast.expr) -> str:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return ast.unparse(node)
    if node.keywords:
        raise ValueError("Keyword arguments are not allowed in IR.")
    if _call(node, "mul", 2):
        factors = sorted(map(_spelling, _factors(node)), key=_factor_order)
        return functools.reduce(lambda left, right: f"mul({left},{right})", factors)
    arguments = [_spelling(argument) for argument in node.args]
    if node.func.id in _COMMUTATIVE:
        arguments.sort()
    return f"{node.func.id}({','.join(arguments)})"


def _factors(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.Call) and _call(node, "mul", 2):
        return _factors(node.args[0]) + _factors(node.args[1])
    return [node]


def _factor_order(spelling: str) -> tuple[str, str]:
    return skeletonize_constants(spelling), spelling


__all__ = [
    "SketchEntry",
    "aggregate_entries",
    "sketch_entries",
    "whole_entry",
    "whole_term_key",
]
