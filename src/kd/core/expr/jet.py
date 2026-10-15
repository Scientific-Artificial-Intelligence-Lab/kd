
from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from functools import lru_cache
from typing import TypeAlias

import sympy
from sympy import Expr

from kd.core.expr.executor import DIFF_OPERATOR_PATTERN, UNITY_TOKEN
from kd.core.expr.naming import parse_compound_derivative
from kd.core.expr.term_features import TermVocabulary

Monomial: TypeAlias = Expr


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




_OPERATIONS: dict[tuple[str, int], Callable[..., Expr]] = {
    ("add", 2): operator.add,
    ("sub", 2): operator.sub,
    ("mul", 2): operator.mul,
    ("neg", 1): operator.neg,
    ("n2", 1): lambda base: base**2,
    ("n3", 1): lambda base: base**3,
    ("recip", 1): _recip,
    ("sin", 1): sympy.sin,
    ("cos", 1): sympy.cos,
    ("exp", 1): sympy.exp,
    ("log", 1): _log,
}


def _multiindex(orders: dict[str, int]) -> _Multiindex:
    return tuple(sorted((axis, order) for axis, order in orders.items() if order))


class _Jet:

    def __init__(self, vocabulary: TermVocabulary) -> None:
        self.vocabulary = vocabulary
        self.coordinates = {name: sympy.Symbol(name) for name in vocabulary.coordinates}
        self.index: dict[Expr, tuple[str, _Multiindex]] = {}

    def symbol(self, field: str, multiindex: _Multiindex) -> Expr:


        name = ",".join([field, *(f"{axis}:{order}" for axis, order in multiindex)])
        symbol = sympy.Symbol(name)
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


def _expression(node: ast.expr, jet: _Jet) -> Expr:
    if isinstance(node, ast.Name):
        return jet.name(node.id)
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
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


@lru_cache(maxsize=_CACHE_SIZE)
def jet_monomials(term_ir: str, vocabulary: TermVocabulary) -> frozenset[Monomial]:

    from kd.core.equation.signature import law_term_key

    parsed = ast.parse(law_term_key(term_ir), mode="eval").body
    expanded = sympy.expand(_expression(parsed, _Jet(vocabulary)))
    monomials: set[Monomial] = set()
    for addend in sympy.Add.make_args(expanded):
        if addend == 0:
            continue
        symbols = addend.free_symbols
        if not symbols:
            monomials.add(sympy.Integer(1))
            continue
        _coefficient, monomial = addend.as_independent(*symbols, as_Add=False)
        monomials.add(monomial)
    return frozenset(monomials)


__all__ = ["Monomial", "jet_monomials"]
