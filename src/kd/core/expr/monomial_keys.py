
from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import sympy
from sympy.core.function import AppliedUndef

from kd.core.expr.executor import DIFF_OPERATOR_PATTERN
from kd.core.expr.sympy_bridge import to_sympy



_CACHE_SIZE = 65536



_UNDEFINED = (sympy.zoo, sympy.nan, sympy.oo, -sympy.oo)


@dataclass(frozen=True)
class _Frame:

    fields: frozenset[str]
    coordinates: frozenset[str]
    time_axis: str | None


@lru_cache(maxsize=_CACHE_SIZE)
def _axis_readings(
    suffix: str, coordinates: frozenset[str]
) -> frozenset[tuple[str, ...]]:
    readings: set[tuple[str, ...]] = set()
    for name in coordinates:
        if not suffix.startswith(name):
            continue
        rest = suffix[len(name):]
        tails: set[tuple[str, ...]] = set() if rest else {()}
        if rest:
            tails |= _axis_readings(rest, coordinates)
        if rest.startswith("_") and len(rest) > 1:
            tails |= _axis_readings(rest[1:], coordinates)
        readings |= {tuple(sorted((name, *tail))) for tail in tails}
    return frozenset(readings)


@lru_cache(maxsize=_CACHE_SIZE)
def _derivative_readings(
    name: str, frame: _Frame
) -> frozenset[tuple[str, tuple[str, ...]]]:
    variables = frame.fields | frame.coordinates
    return frozenset(
        (name[:index], axes)
        for index, char in enumerate(name)
        if char == "_" and name[:index] in variables
        for axes in _axis_readings(name[index + 1:], frame.coordinates)
    )


def _spell(readings: frozenset[tuple[str, tuple[str, ...]]]) -> str:
    return "; ".join(
        f"{base} along {', '.join(axes)}" for base, axes in sorted(readings)
    )


def _reading(name: str, frame: _Frame) -> tuple[str, tuple[str, ...]] | None:
    if name in frame.fields or name in frame.coordinates:
        return name, ()
    readings = _derivative_readings(name, frame)
    if len(readings) > 1:
        raise ValueError(
            f"{name!r} has more than one reading in this input "
            f"({_spell(readings)}), so it is refused rather than guessed"
        )
    return next(iter(readings), None)


def _derivative_symbol(base: str, axes: tuple[str, ...], frame: _Frame) -> Any:
    if base in frame.coordinates:
        return sympy.Integer(1 if axes == (base,) else 0)
    axes = tuple(sorted(axes))
    name = "_".join((base, *axes))
    readings = _derivative_readings(name, frame)
    if name in frame.fields or name in frame.coordinates or len(readings) > 1:
        taken = _spell(readings - {(base, axes)}) or f"the variable {name}"
        raise ValueError(
            f"{base} along {', '.join(axes)} is spelled {name!r}, which has "
            f"another reading in this input ({taken}), so it is refused "
            "rather than given that key"
        )
    return sympy.Symbol(name)


def _rename_derivatives(expr: Any, frame: _Frame) -> Any:
    mapping: dict[Any, Any] = {}
    for symbol in expr.free_symbols:
        reading = _reading(symbol.name, frame)
        if reading is not None and reading[1]:
            mapping[symbol] = _derivative_symbol(*reading, frame)
    return expr.xreplace(mapping) if mapping else expr


def _total_derivative(expr: Any, axis: str, frame: _Frame) -> Any:
    total = sympy.Integer(0)
    for symbol in expr.free_symbols:
        reading = _reading(symbol.name, frame)
        if reading is None:
            raise ValueError(
                f"{symbol.name!r} is not a field or coordinate of this input "
                f"(fields: {sorted(frame.fields)}, coordinates: "
                f"{sorted(frame.coordinates)}), so its derivative along "
                f"{axis!r} has no reading"
            )
        base, axes = reading
        inner = _derivative_symbol(base, (*axes, axis), frame)
        if inner != 0:
            total += sympy.diff(expr, symbol) * inner
    return total


def _expand_operators(expr: Any, frame: _Frame) -> Any:
    if not expr.args:
        return expr
    args = [_expand_operators(arg, frame) for arg in expr.args]
    rebuilt = expr.func(*args) if args != list(expr.args) else expr
    if not isinstance(rebuilt, AppliedUndef):
        return rebuilt
    name = rebuilt.func.__name__
    (argument,) = rebuilt.args
    match = DIFF_OPERATOR_PATTERN.match(name)
    if match is not None:
        axis, order = match.group(2), int(match.group(1) or 1)
        result = argument
        for _ in range(order):
            result = _total_derivative(result, axis, frame)
        return result
    if name == "lap":
        result = sympy.Integer(0)
        for axis in sorted(
            axis for axis in frame.coordinates if axis != frame.time_axis
        ):
            result += _total_derivative(
                _total_derivative(argument, axis, frame), axis, frame
            )
        return result
    return rebuilt


def _check_operators(term: str, frame: _Frame) -> None:
    for node in ast.walk(ast.parse(term.replace("^", "**"), mode="eval")):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        match = DIFF_OPERATOR_PATTERN.match(node.func.id)
        if match is not None and match.group(2) not in frame.coordinates:
            raise ValueError(
                f"{term!r}: {node.func.id} differentiates along "
                f"{match.group(2)!r}, which is not a coordinate of this input "
                f"(coordinates: {sorted(frame.coordinates)})"
            )
        if node.func.id == "lap" and all(
            axis == frame.time_axis for axis in frame.coordinates
        ):
            raise ValueError(
                f"{term!r}: lap needs a spatial coordinate and this input has none"
            )


def _number(node: Any) -> Any:
    value = node.evalf()
    if value.is_real and float(value).is_integer():
        return sympy.Integer(int(value))
    return value


def _one_number_spelling(expr: Any) -> Any:
    return expr.replace(
        lambda node: (
            isinstance(node, sympy.Expr)
            and not node.is_Integer
            and not node.free_symbols
            and node.is_finite is True
        ),
        _number,
    )


@lru_cache(maxsize=_CACHE_SIZE)
def _keys(term: str, frame: _Frame) -> frozenset[str]:
    raw = to_sympy(term, fold_derivatives=False)
    _check_operators(term, frame)
    expr = _expand_operators(_rename_derivatives(raw, frame), frame)
    sums: dict[str, complex] = {}
    for part in sympy.Add.make_args(sympy.expand(_one_number_spelling(expr))):


        constant: list[Any] = []
        rest: list[Any] = []
        for factor in sympy.Mul.make_args(_one_number_spelling(part)):
            (constant if factor.is_Number and factor.is_finite else rest).append(factor)
        key = str(sympy.Mul(*rest))
        sums[key] = sums.get(key, 0) + complex(sympy.Mul(*constant))
    return frozenset(key for key, value in sums.items() if value != 0)


def term_monomial_keys(
    term: str,
    *,
    fields: Sequence[str],
    coordinates: Sequence[str],
    time_axis: str | None,
) -> frozenset[str]:
    if time_axis is not None and time_axis not in coordinates:
        raise ValueError(
            f"time_axis {time_axis!r} is not one of the coordinates {list(coordinates)}"
        )
    both = sorted(set(fields) & set(coordinates))
    if both:
        raise ValueError(
            f"{', '.join(map(repr, both))} is both a field and a coordinate, "
            "so a derivative of it has no single reading"
        )
    return _keys(term, _Frame(frozenset(fields), frozenset(coordinates), time_axis))


@lru_cache(maxsize=_CACHE_SIZE)
def _undefined(term: str) -> bool:
    if to_sympy(term).has(*_UNDEFINED):
        return True
    return any(
        to_sympy(ast.unparse(node)).has(*_UNDEFINED)
        for node in ast.walk(ast.parse(term.replace("^", "**"), mode="eval"))
        if isinstance(node, (ast.Call, ast.BinOp, ast.UnaryOp))
    )


def undefined_terms(terms: Sequence[str]) -> list[str]:
    return [term for term in terms if _undefined(term)]


__all__ = ["term_monomial_keys", "undefined_terms"]
