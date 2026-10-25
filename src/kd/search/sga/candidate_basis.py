
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from typing import Any

import sympy as sp

from kd.search.sga.candidate_admissibility import (
    _atoms,
    _exponents,
    _fraction,
    _Monomial,
    _monomial_ir,
    _order,
    _poly_ir,
    _primitive,
)
from kd.search.sga.convert import tree_to_kd_expr
from kd.search.sga.pde import PDE

_CacheKey = tuple[tuple[str, ...], tuple[str, ...]]
_CACHE_SIZE = 2048


def _poly_key(poly: Any) -> tuple[tuple[_Monomial, int], ...]:
    return tuple((powers, int(value)) for powers, value in poly.terms())


def _roots(
    pde: PDE, features: tuple[str, ...]
) -> tuple[list[_Monomial], list[tuple[Any, Any]], tuple[str, ...]] | None:
    fractions = []
    for tree in pde.terms:
        fraction = _fraction(tree.root)
        if fraction is None:
            return None
        fractions.append(fraction)
    atoms = _atoms(*(poly for fraction in fractions for poly in fraction))
    names = (*features, *(atom for atom in atoms if atom not in features))
    symbols = tuple(sp.Symbol(name) for name in names)
    monomials: dict[_Monomial, None] = {}
    rational = []
    seen = set()
    for numerator, denominator in fractions:
        if all(not term for term in denominator):
            powers = [_exponents(term, names) for term in numerator]
        else:
            n, d = [
                sp.Poly.from_dict(
                    {_exponents(term, names): value for term, value in poly.items()},
                    *symbols,
                    domain=sp.QQ,
                )
                for poly in (numerator, denominator)
            ]
            common = n.gcd(d)
            n, d = n.exquo(common), d.exquo(common)
            if d.total_degree() == 0:
                powers = n.monoms()
            else:
                n, d = _primitive(n), _primitive(d)
                key = (_poly_key(n), _poly_key(d))
                if key not in seen:
                    rational.append((n, d))
                    seen.add(key)
                continue
        for monomial in sorted(powers, key=_order):
            monomials.setdefault(monomial, None)
    return list(monomials), rational, names


def _independent_columns(
    monomials: list[_Monomial], rational: list[tuple[Any, Any]], names: tuple[str, ...]
) -> tuple[str, ...]:
    polynomial_ir = [_monomial_ir(monomial, names) for monomial in monomials]
    if not rational:
        return tuple(polynomial_ir)
    symbols = tuple(sp.Symbol(name) for name in names)
    unity = sp.Poly(1, *symbols, domain=sp.QQ)
    common = unity
    for _, denominator in rational:
        common = common.lcm(denominator)
    columns = [
        (sp.Poly.from_dict({monomial: 1}, *symbols, domain=sp.QQ), unity, ir)
        for monomial, ir in zip(monomials, polynomial_ir, strict=True)
    ]
    columns.extend(
        (n, d, f"div({_poly_ir(n.terms(), names)}, {_poly_ir(d.terms(), names)})")
        for n, d in rational
    )
    pivots: list[tuple[_Monomial, dict[_Monomial, Fraction]]] = []
    retained = []
    for numerator, denominator, ir in columns:
        poly = numerator * common.exquo(denominator)
        vector = {powers: Fraction(int(c.p), int(c.q)) for powers, c in poly.terms()}
        for pivot, previous in pivots:
            factor = vector.get(pivot, Fraction(0))
            if factor:
                for powers, value in previous.items():
                    updated = vector.get(powers, Fraction(0)) - factor * value
                    if updated:
                        vector[powers] = updated
                    else:
                        vector.pop(powers, None)
        if vector:
            pivot = min(vector, key=_order)
            scale = vector[pivot]
            pivots.append((pivot, {powers: c / scale for powers, c in vector.items()}))
            retained.append(ir)
    return tuple(retained)


class TabularBasisCompiler:

    def __init__(self) -> None:
        self._cache: OrderedDict[_CacheKey, tuple[str, ...] | None] = OrderedDict()

    def clear(self) -> None:
        self._cache.clear()

    def compile(
        self, pde: PDE, feature_order: tuple[str, ...]
    ) -> tuple[str, ...] | None:
        key = feature_order, tuple(tree_to_kd_expr(tree) for tree in pde.terms)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        roots = _roots(pde, feature_order)
        result = None if roots is None else _independent_columns(*roots)
        self._cache[key] = result
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return result
