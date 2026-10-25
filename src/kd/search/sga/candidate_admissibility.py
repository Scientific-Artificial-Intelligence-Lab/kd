
from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Any

import sympy as sp

from kd.core.expr.executor import UNITY_TOKEN
from kd.core.expr.sympy_bridge import from_sympy
from kd.search.sga.config import TABULAR_FUNCTIONS
from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node

_Coefficient = int | Fraction
_Polynomial = dict[tuple[str, ...], _Coefficient]
_Fraction = tuple[_Polynomial, _Polynomial]
_Monomial = tuple[int, ...]


def _order(monomial: _Monomial) -> tuple[int, tuple[int, ...]]:
    return sum(monomial), tuple(-power for power in monomial)


def _exponents(monomial: tuple[str, ...], names: tuple[str, ...]) -> _Monomial:
    return tuple(monomial.count(name) for name in names)


def _fold(operator: str, terms: list[str]) -> str:
    expression = terms[-1]
    for term in reversed(terms[:-1]):
        expression = f"{operator}({term}, {expression})"
    return expression


def _monomial_ir(monomial: _Monomial, names: tuple[str, ...]) -> str:
    factors = [
        from_sympy(sp.Symbol(name) ** power)
        for name, power in zip(names, monomial, strict=True)
        if power
    ]
    return _fold("mul", factors) if factors else "one"


def _literal(value: Any) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return repr(value.numerator / value.denominator)


def _poly_ir(terms: Iterable[tuple[_Monomial, Any]], names: tuple[str, ...]) -> str:
    rendered = []
    for monomial, coefficient in sorted(terms, key=lambda item: _order(item[0])):
        term = _monomial_ir(monomial, names)
        if coefficient != 1:
            literal = _literal(coefficient)
            term = literal if not any(monomial) else f"mul({literal}, {term})"
        rendered.append(term)
    return _fold("add", rendered)


def _primitive(poly: Any) -> Any:
    _, integer = poly.clear_denoms(convert=True)
    _, primitive = integer.primitive()
    first = min(primitive.terms(), key=lambda item: _order(item[0]))[1]
    return (-primitive if first < 0 else primitive).to_field()


def _first(poly: Any) -> Any:
    return min(poly.terms(), key=lambda item: _order(item[0]))[1]


def _atoms(*polynomials: _Polynomial) -> tuple[str, ...]:
    return tuple(
        sorted({atom for poly in polynomials for monomial in poly for atom in monomial})
    )


def _function_atom(
    name: str, numerator: _Polynomial, denominator: _Polynomial
) -> str | None:
    names = _atoms(numerator, denominator)
    if all(not monomial for monomial in denominator):
        if all(not monomial for monomial in numerator):
            return None
        scale = denominator[()]
        terms = [
            (_exponents(monomial, names), Fraction(value) / scale)
            for monomial, value in numerator.items()
        ]
        return f"{name}({_poly_ir(terms, names)})"
    symbols = tuple(sp.Symbol(atom) for atom in names)
    n, d = (
        sp.Poly.from_dict(
            {_exponents(monomial, names): value for monomial, value in poly.items()},
            *symbols,
            domain=sp.QQ,
        )
        for poly in (numerator, denominator)
    )
    common = n.gcd(d)
    n, d = n.exquo(common), d.exquo(common)
    primitive = _primitive(d)
    n = n.mul_ground(_first(primitive) / _first(d))
    if primitive.total_degree() == 0:
        if n.total_degree() == 0:
            return None
        return f"{name}({_poly_ir(n.terms(), names)})"
    return (
        f"{name}(div({_poly_ir(n.terms(), names)}, "
        f"{_poly_ir(primitive.terms(), names)}))"
    )


def _add(left: _Polynomial, right: _Polynomial, sign: int = 1) -> _Polynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        value = result.get(monomial, 0) + sign * coefficient
        if value:
            result[monomial] = value
        else:
            result.pop(monomial, None)
    return result


def _multiply(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    result: _Polynomial = {}
    for first, a in left.items():
        for second, b in right.items():
            monomial = tuple(sorted((*first, *second)))
            result[monomial] = result.get(monomial, 0) + a * b
    return {
        monomial: coefficient for monomial, coefficient in result.items() if coefficient
    }


def _leaf(name: str) -> _Fraction:
    if name == UNITY_TOKEN:
        return {(): 1}, {(): 1}
    if not name.isidentifier():
        return {(): Fraction(float(name))}, {(): 1}
    return {(name,): 1}, {(): 1}


def _fraction(node: Node) -> _Fraction | None:
    if node.is_leaf:
        return _leaf(node.name)
    left = _fraction(node.children[0])
    if left is None:
        return None
    numerator, denominator = left
    if node.name in ("^2", "^3"):
        power = int(node.name[1:])
        n, d = numerator, denominator
        for _ in range(power - 1):
            n, d = _multiply(n, numerator), _multiply(d, denominator)
        return n, d
    if node.name in TABULAR_FUNCTIONS:
        atom = _function_atom(node.name, numerator, denominator)
        return None if atom is None else ({(atom,): 1}, {(): 1})
    right = _fraction(node.children[1])
    if right is None:
        return None
    other_numerator, other_denominator = right
    if node.name in ("+", "-"):
        combined = _add(
            _multiply(numerator, other_denominator),
            _multiply(other_numerator, denominator),
            1 if node.name == "+" else -1,
        )
        return (
            (combined, _multiply(denominator, other_denominator)) if combined else None
        )
    if node.name == "*":
        return _multiply(numerator, other_numerator), _multiply(
            denominator, other_denominator
        )
    if node.name == "/":
        return _multiply(numerator, other_denominator), _multiply(
            denominator, other_numerator
        )
    raise ValueError(f"Unsupported SGA tabular operator {node.name!r}")


def tabular_candidate_admissible(pde: PDE) -> bool:
    return all(_fraction(term.root) is not None for term in pde.terms)
