
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

from kd.core.expr.executor import UNITY_TOKEN
from kd.core.interrupt import SearchInterrupted


_ALLOWED_POWER_ORDERS = frozenset({"2", "3", "4", "5"})
_POWER_ORDER_PATTERN = re.compile(r"\^(\d+)")


_MAX_POWER = 5


_POSITIVE_POW_IR = {
    2: "n2({base})",
    3: "n3({base})",
    4: "n2(n2({base}))",
    5: "mul(n2(n2({base})), {base})",
}



_FUNCTION_IR: dict[type, str] = {
    sympy.sin: "sin",
    sympy.cos: "cos",
    sympy.exp: "exp",
    sympy.log: "log",
}

_PLACEHOLDER_PATTERN = re.compile(r"c[1-9][0-9]*")


def is_placeholder(name: str) -> bool:
    return _PLACEHOLDER_PATTERN.fullmatch(name) is not None


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

    term_str: str
    ir: str
    base_ir: str
    coeff: float


@dataclass(frozen=True)
class ParsedEquation:

    equation: str
    sympy_expr: sympy.Expr
    terms: tuple[ParsedTerm, ...]


def equation_to_sympy(
    equation: str,
    operands: Sequence[str],
    *,
    placeholders: bool = False,
    target: str | None = None,
) -> sympy.Expr:
    operand_set = frozenset(operands)

    def _can_split(symbol: str) -> bool:


        if symbol in operand_set or (placeholders and is_placeholder(symbol)):
            return False
        return bool(_token_splittable(symbol))

    transformations = standard_transformations + (
        split_symbols_custom(_can_split),
        convert_xor,
        implicit_multiplication,
    )



    bound = operand_set if target is None else operand_set | {target}
    symbols = {name: sympy.Symbol(name) for name in bound}
    try:
        expr = parse_expr(equation, local_dict=symbols, transformations=transformations)




        expanded = sympy.expand(expr)
    except SearchInterrupted:



        raise
    except Exception as exc:
        raise Llm4edParseError(
            f"cannot parse equation {equation!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(expanded, sympy.Expr):


        raise Llm4edParseError(
            f"equation {equation!r} is a {type(expanded).__name__}, not an expression"
        )
    return expanded


def reads_as_symbol(name: str) -> bool:
    probe = f"{name} + 2*c1 + 0.5"
    expected = sympy.Symbol(name) + 2 * sympy.Symbol("c1") + sympy.Float(0.5)
    try:
        return bool(equation_to_sympy(probe, (name,), placeholders=True) == expected)
    except Llm4edParseError:
        return False


def _check_power_orders(equation: str) -> None:
    for order in _POWER_ORDER_PATTERN.findall(equation):
        if order not in _ALLOWED_POWER_ORDERS:
            raise UndefinedOperatorError(
                f"power order ^{order} outside the allowed set "
                f"{sorted(_ALLOWED_POWER_ORDERS)} in {equation!r}"
            )


def _check_operands(
    expr: sympy.Expr, operands: Sequence[str], *, placeholders: bool
) -> None:
    operand_set = frozenset(operands)
    unknown = sorted(
        str(symbol)
        for symbol in expr.free_symbols
        if str(symbol) not in operand_set
        and not (placeholders and is_placeholder(str(symbol)))
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


def _convert_pow(node: sympy.Expr, functions: frozenset[str]) -> str:
    base, exponent = node.args
    if exponent.is_Rational and exponent.q == 2 and "sqrt" in functions:


        if abs(exponent) > _MAX_POWER:
            raise InexpressibleTermError(
                f"exponent {exponent} in {node} is outside the expressible range "
                f"(|p/2| <= {_MAX_POWER})"
            )
        return _sqrt_power_ir(_convert_node(base, functions), int(exponent.p))
    if not exponent.is_Integer:
        raise InexpressibleTermError(
            f"non-integer exponent {exponent} in {node} is not expressible"
        )
    k = int(exponent)
    base_ir = _convert_node(base, functions)
    if k in _POSITIVE_POW_IR:
        return _POSITIVE_POW_IR[k].format(base=base_ir)
    if k == -1:
        return f"recip({base_ir})"
    if -k in _POSITIVE_POW_IR:
        return f"recip({_POSITIVE_POW_IR[-k].format(base=base_ir)})"
    raise InexpressibleTermError(
        f"exponent {k} in {node} is outside the expressible range (-5..-1, 2..5)"
    )


def _sqrt_power_ir(base_ir: str, p: int) -> str:
    power = _positive_power_ir(f"sqrt({base_ir})", abs(p))
    return power if p > 0 else f"recip({power})"


def _positive_power_ir(base_ir: str, k: int) -> str:
    if k == 1:
        return base_ir
    if k in _POSITIVE_POW_IR:
        return _POSITIVE_POW_IR[k].format(base=base_ir)
    fifth = _POSITIVE_POW_IR[5].format(base=base_ir)
    return f"mul({_positive_power_ir(base_ir, k - 5)}, {fifth})"


def _convert_node(node: sympy.Expr, functions: frozenset[str]) -> str:
    if isinstance(node, sympy.Symbol):
        return str(node)
    if node.is_Number:
        return _number_literal(node)
    if isinstance(node, sympy.Add):
        return _fold("add", [_convert_node(arg, functions) for arg in node.args])
    if isinstance(node, sympy.Mul):
        return _fold("mul", [_convert_node(arg, functions) for arg in node.args])
    if isinstance(node, sympy.Pow):
        return _convert_pow(node, functions)
    name = _FUNCTION_IR.get(type(node))
    if name is not None and name in functions:
        return f"{name}({_convert_node(node.args[0], functions)})"
    raise InexpressibleTermError(
        f"cannot express {node} (sympy {type(node).__name__}) in kd IR"
    )


def term_to_ir(
    term: sympy.Expr,
    *,
    functions: frozenset[str] = frozenset(),
    constant_term: bool = False,
) -> str:
    if term.is_Number:

        if constant_term and term.is_finite and float(term) != 0.0:
            if term == 1:
                return UNITY_TOKEN
            return f"mul({_number_literal(term)}, {UNITY_TOKEN})"
        if constant_term:
            raise InexpressibleTermError(
                f"constant term {term} has no finite nonzero value"
            )
        raise InexpressibleTermError(
            f"pure constant term {term} has no design-matrix column "
            "(EDL PDE mode has no constant term; candidate is dropped)"
        )
    ir = _convert_node(term, functions)
    if not term.free_symbols:
        raise InexpressibleTermError(
            f"constant term {term} is not a number and has no design-matrix column"
        )
    return ir


def term_base_ir_and_coeff(
    term: sympy.Expr,
    *,
    functions: frozenset[str] = frozenset(),
    constant_term: bool = False,
) -> tuple[str, float]:
    coeff, base = term.as_coeff_Mul()
    ir = term_to_ir(base, functions=functions, constant_term=constant_term)
    return ir, float(coeff)


def _split_terms(expr: sympy.Expr) -> tuple[sympy.Expr, ...]:
    if isinstance(expr, sympy.Add):
        return tuple(expr.args)
    return (expr,)


def _term_str(term: sympy.Expr) -> str:
    return str(term).replace("**", "^")


def parse_equation(
    equation: str,
    operands: Sequence[str],
    *,
    functions: frozenset[str] = frozenset(),
    placeholders: bool = False,
    constant_term: bool = False,
    target: str | None = None,
) -> ParsedEquation:
    expr = equation_to_sympy(
        equation, operands, placeholders=placeholders, target=target
    )


    _check_power_orders(equation)
    _check_operands(expr, operands, placeholders=placeholders)
    terms = tuple(
        _build_parsed_term(term, functions, constant_term)
        for term in _split_terms(expr)
    )
    return ParsedEquation(equation=equation, sympy_expr=expr, terms=terms)


def _build_parsed_term(
    term: sympy.Expr, functions: frozenset[str], constant_term: bool
) -> ParsedTerm:
    base_ir, coeff = term_base_ir_and_coeff(
        term, functions=functions, constant_term=constant_term
    )
    return ParsedTerm(
        term_str=_term_str(term),
        ir=term_to_ir(term, functions=functions, constant_term=constant_term),
        base_ir=base_ir,
        coeff=coeff,
    )
