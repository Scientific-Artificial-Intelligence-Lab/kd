
from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import TypeAlias

import sympy
from sympy import Expr

from kd.core.expr.executor import DIFF_OPERATOR_PATTERN, UNITY_TOKEN
from kd.core.expr.naming import build_derivative_name, parse_compound_derivative
from kd.core.expr.term_features import TermVocabulary

Monomial: TypeAlias = Expr

JetExpansion: TypeAlias = frozenset[tuple[Monomial, Expr]]


_EPS = sympy.Rational(1, 10**10)



_CACHE_SIZE = 65536

_Multiindex = tuple[tuple[str, int], ...]


def _recip(denominator: Expr) -> Expr:
    if sympy.expand(denominator) == 0:
        return 1 / _EPS
    return 1 / denominator


def _log(argument: Expr) -> Expr:
    if sympy.expand(argument) == 0:
        return sympy.log(_EPS)
    return sympy.log(argument)


def _divide(numerator: Expr, denominator: Expr) -> Expr:
    return numerator * _recip(denominator)




_OPERATIONS: dict[tuple[str, int], Callable[..., Expr]] = {
    ("add", 2): operator.add,
    ("sub", 2): operator.sub,
    ("mul", 2): operator.mul,
    ("div", 2): _divide,
    ("neg", 1): operator.neg,
    ("n2", 1): lambda base: base**2,
    ("n3", 1): lambda base: base**3,
    ("recip", 1): _recip,
    ("sin", 1): sympy.sin,
    ("cos", 1): sympy.cos,
    ("exp", 1): sympy.exp,
    ("log", 1): _log,
    ("sqrt", 1): sympy.sqrt,
}



_BINARY: dict[type[ast.operator], Callable[[Expr, Expr], Expr]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: _divide,
}


def _multiindex(orders: dict[str, int]) -> _Multiindex:
    return tuple(sorted((axis, order) for axis, order in orders.items() if order))


def _jet_name(field: str, multiindex: _Multiindex) -> str:


    return ",".join([field, *(f"{axis}:{order}" for axis, order in multiindex)])


def _kd_name(jet_name: str) -> str:
    field, *segments = jet_name.split(",")
    name = field
    for segment in segments:
        axis, order = segment.split(":")
        name = build_derivative_name(name, axis, int(order))
    return name


class _Jet:

    def __init__(self, vocabulary: TermVocabulary) -> None:
        self.vocabulary = vocabulary
        self.coordinates = {name: sympy.Symbol(name) for name in vocabulary.coordinates}
        self.index: dict[Expr, tuple[str, _Multiindex]] = {}

    def symbol(self, field: str, multiindex: _Multiindex) -> Expr:
        symbol = sympy.Symbol(_jet_name(field, multiindex))
        self.index[symbol] = (field, multiindex)
        return symbol

    def name(self, name: str) -> Expr:
        if name == UNITY_TOKEN:
            return sympy.Integer(1)
        if name in self.coordinates:
            return self.coordinates[name]
        if name in self.vocabulary.fields:
            return self.symbol(name, ())
        parsed = parse_compound_derivative(
            name,
            known_fields=set(self.vocabulary.fields),
            known_axes=set(self.vocabulary.coordinates),
        )
        if parsed is None:
            raise ValueError(f"Unresolvable term symbol {name!r}")
        field, segments = parsed
        orders: dict[str, int] = {}
        for axis, order in segments:
            orders[axis] = orders.get(axis, 0) + order
        return self.symbol(field, _multiindex(orders))

    def total_derivative(self, expression: Expr, axis: str) -> Expr:
        result = sympy.diff(expression, self.coordinates[axis])
        for symbol in list(expression.free_symbols & self.index.keys()):
            field, multiindex = self.index[symbol]
            orders = dict(multiindex)
            orders[axis] = orders.get(axis, 0) + 1
            raised = self.symbol(field, _multiindex(orders))
            result += sympy.diff(expression, symbol) * raised
        return result


def _derivative(
    head: str, match: re.Match[str], arguments: list[Expr], jet: _Jet
) -> Expr:
    if len(arguments) != 1:
        raise ValueError(f"Open-form derivative {head} requires exactly one argument")
    axis = match.group(2)
    if axis not in jet.coordinates:
        raise ValueError(
            f"Derivative axis {axis!r} of {head!r} is outside the vocabulary"
        )
    order = int(match.group(1) or "1")
    if order < 1:
        raise ValueError(f"Diff order must be >= 1, got {order} in {head!r}")
    expression = arguments[0]
    for _ in range(order):
        expression = jet.total_derivative(expression, axis)
    return expression


def _exponent(node: ast.expr) -> Expr:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return sympy.Integer(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_exponent(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return sympy.Rational(_exponent(node.left), _exponent(node.right))
    raise ValueError(
        f"Exponent {ast.unparse(node)!r} is not an integer or a ratio of "
        "integer literals"
    )


def _infix(node: ast.expr, jet: _Jet) -> Expr:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):

        return sympy.Rational(repr(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        operand = _expression(node.operand, jet)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        base, exponent = _expression(node.left, jet), _exponent(node.right)

        return _recip(base**-exponent) if exponent < 0 else base**exponent
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _expression(node.left, jet)
        return _BINARY[type(node.op)](left, _expression(node.right, jet))
    raise ValueError(
        f"Unsupported term syntax {type(node).__name__} in {ast.unparse(node)!r}"
    )


def _expression(node: ast.expr, jet: _Jet) -> Expr:
    if isinstance(node, ast.Name):
        return jet.name(node.id)
    if not isinstance(node, ast.Call):
        return _infix(node, jet)
    if not isinstance(node.func, ast.Name):
        raise ValueError(f"Unsupported term syntax: {type(node).__name__}")
    head = node.func.id
    arguments = [_expression(argument, jet) for argument in node.args]
    match = DIFF_OPERATOR_PATTERN.fullmatch(head)
    if match is not None:
        return _derivative(head, match, arguments, jet)
    operation = _OPERATIONS.get((head, len(arguments)))
    if operation is None:
        raise ValueError(
            f"Term head {head!r} with {len(arguments)} argument(s) has no jet expansion"
        )
    return operation(*arguments)


def _addends(expression: Expr) -> Iterator[tuple[Expr, Monomial]]:
    for addend in sympy.Add.make_args(sympy.expand(expression)):
        if addend == 0:
            continue
        symbols = addend.free_symbols
        if not symbols:
            yield addend, sympy.Integer(1)
            continue
        yield addend.as_independent(*symbols, as_Add=False)


@lru_cache(maxsize=_CACHE_SIZE)
def jet_expansion(term_ir: str, vocabulary: TermVocabulary) -> JetExpansion:

    from kd.core.equation.signature import law_term_key

    parsed = ast.parse(law_term_key(term_ir), mode="eval").body
    summed: dict[Monomial, Expr] = {}
    for coefficient, monomial in _addends(_expression(parsed, _Jet(vocabulary))):
        summed[monomial] = summed.get(monomial, sympy.Integer(0)) + coefficient
    return frozenset(
        (monomial, coefficient)
        for monomial, coefficient in summed.items()
        if coefficient != 0
    )


@lru_cache(maxsize=_CACHE_SIZE)
def jet_monomials(term_ir: str, vocabulary: TermVocabulary) -> frozenset[Monomial]:
    return frozenset(monomial for monomial, _ in jet_expansion(term_ir, vocabulary))


def jet_polynomial(
    expression: str, vocabulary: TermVocabulary
) -> dict[Monomial, float]:
    parsed = ast.parse(expression, mode="eval").body
    polynomial: dict[Monomial, float] = {}
    for coefficient, monomial in _addends(_expression(parsed, _Jet(vocabulary))):
        polynomial[monomial] = polynomial.get(monomial, 0.0) + float(coefficient)
    return polynomial


def monomial_label(monomial: Monomial) -> str:
    names = {
        symbol: sympy.Symbol(_kd_name(symbol.name))
        for symbol in monomial.free_symbols
        if "," in symbol.name
    }
    return str(monomial.xreplace(names))


__all__ = [
    "JetExpansion",
    "Monomial",
    "jet_expansion",
    "jet_monomials",
    "jet_polynomial",
    "monomial_label",
]
