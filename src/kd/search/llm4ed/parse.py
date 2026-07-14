
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import sympy
from sympy.parsing.sympy_parser import (
    _token_splittable,
    convert_xor,
    implicit_multiplication,
    parse_expr,
    split_symbols_custom,
    standard_transformations,
)


_ALLOWED_POWER_ORDERS = frozenset({"2", "3", "4", "5"})
_POWER_ORDER_PATTERN = re.compile(r"\^(\d+)")


_POSITIVE_POW_IR = {
    2: "n2({base})",
    3: "n3({base})",
    4: "n2(n2({base}))",
    5: "mul(n2(n2({base})), {base})",
}


class Llm4edParseError(ValueError):
    pass


class UndefinedOperandError(Llm4edParseError):
    pass


class UndefinedOperatorError(Llm4edParseError):
    pass


class InexpressibleTermError(Llm4edParseError):
    pass


@dataclass(frozen=True)
class ParsedTerm:

    sympy_term: sympy.Expr
    term_str: str
    ir: str
    base_ir: str
    coeff: float


@dataclass(frozen=True)
class ParsedEquation:

    equation: str
    sympy_expr: sympy.Expr
    terms: tuple[ParsedTerm, ...]


def equation_to_sympy(equation: str, operands: Sequence[str]) -> sympy.Expr:
    operand_set = frozenset(operands)

    def _can_split(symbol: str) -> bool:


        if symbol not in operand_set:
            return bool(_token_splittable(symbol))
        return False

    transformations = standard_transformations + (
        split_symbols_custom(_can_split),
        convert_xor,
        implicit_multiplication,
    )
    try:
        expr = parse_expr(equation, transformations=transformations)
    except Exception as exc:
        raise Llm4edParseError(
            f"cannot parse equation {equation!r}: {type(exc).__name__}: {exc}"
        ) from exc
    return sympy.expand(expr)


def _check_power_orders(equation: str) -> None:
    for order in _POWER_ORDER_PATTERN.findall(equation):
        if order not in _ALLOWED_POWER_ORDERS:
            raise UndefinedOperatorError(
                f"power order ^{order} outside the allowed set "
                f"{sorted(_ALLOWED_POWER_ORDERS)} in {equation!r}"
            )


def _check_operands(expr: sympy.Expr, operands: Sequence[str]) -> None:
    operand_set = frozenset(operands)
    unknown = sorted(
        str(symbol) for symbol in expr.free_symbols if str(symbol) not in operand_set
    )
    if unknown:
        raise UndefinedOperandError(
            f"operands {unknown} are not in the dataset vocabulary "
            f"{sorted(operand_set)}"
        )


def _number_literal(node: sympy.Expr) -> str:
    return repr(float(node))


def _fold(op: str, parts: list[str]) -> str:
    result = parts[0]
    for part in parts[1:]:
        result = f"{op}({result}, {part})"
    return result


def _convert_pow(node: sympy.Expr) -> str:
    base, exponent = node.args
    if not exponent.is_Integer:
        raise InexpressibleTermError(
            f"non-integer exponent {exponent} in {node} is not expressible"
        )
    k = int(exponent)
    base_ir = _convert_node(base)
    if k in _POSITIVE_POW_IR:
        return _POSITIVE_POW_IR[k].format(base=base_ir)
    if k == -1:
        return f"recip({base_ir})"
    if -k in _POSITIVE_POW_IR:
        return f"recip({_POSITIVE_POW_IR[-k].format(base=base_ir)})"
    raise InexpressibleTermError(
        f"exponent {k} in {node} is outside the expressible range "
        "(-5..-1, 2..5)"
    )


def _convert_node(node: sympy.Expr) -> str:
    if isinstance(node, sympy.Symbol):
        return str(node)
    if node.is_Number:
        return _number_literal(node)
    if isinstance(node, sympy.Add):
        return _fold("add", [_convert_node(arg) for arg in node.args])
    if isinstance(node, sympy.Mul):
        return _fold("mul", [_convert_node(arg) for arg in node.args])
    if isinstance(node, sympy.Pow):
        return _convert_pow(node)
    raise InexpressibleTermError(
        f"cannot express {node} (sympy {type(node).__name__}) in kd IR"
    )


def term_to_ir(term: sympy.Expr) -> str:
    if term.is_Number:
        raise InexpressibleTermError(
            f"pure constant term {term} has no design-matrix column "
            "(EDL PDE mode has no constant term; candidate is dropped)"
        )
    return _convert_node(term)


def term_base_ir_and_coeff(term: sympy.Expr) -> tuple[str, float]:
    coeff, base = term.as_coeff_Mul()
    return term_to_ir(base), float(coeff)


def _split_terms(expr: sympy.Expr) -> tuple[sympy.Expr, ...]:
    if isinstance(expr, sympy.Add):
        return tuple(expr.args)
    return (expr,)


def _term_str(term: sympy.Expr) -> str:
    return str(term).replace("**", "^")


def parse_equation(equation: str, operands: Sequence[str]) -> ParsedEquation:
    expr = equation_to_sympy(equation, operands)


    _check_power_orders(equation)
    _check_operands(expr, operands)
    terms = tuple(_build_parsed_term(term) for term in _split_terms(expr))
    return ParsedEquation(equation=equation, sympy_expr=expr, terms=terms)


def _build_parsed_term(term: sympy.Expr) -> ParsedTerm:
    base_ir, coeff = term_base_ir_and_coeff(term)
    return ParsedTerm(
        sympy_term=term,
        term_str=_term_str(term),
        ir=term_to_ir(term),
        base_ir=base_ir,
        coeff=coeff,
    )
