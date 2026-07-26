
from __future__ import annotations

import ast
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import sympy
from sympy import Expr
from sympy.core.relational import Equality
from torch import Tensor

from kd.core.expr.naming import build_derivative_name

logger = logging.getLogger(__name__)

_DEFAULT_LHS_LABEL = "u_t"
_DIFF_NAME_PATTERN = re.compile(r"^diff([0-9]*)_([a-z]+)$")
_LAP_FUNC = sympy.Function("lap")

_KD_OPS: dict[str, Callable[..., Any]] = {
    "add": lambda left, right: left + right,
    "mul": lambda left, right: left * right,
    "sub": lambda left, right: left - right,
    "div": lambda left, right: left / right,
    "neg": lambda value: -value,
    "n2": lambda value: value**2,
    "n3": lambda value: value**3,
    "recip": lambda value: 1 / value,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "exp": sympy.exp,
    "log": sympy.log,
    "lap": _LAP_FUNC,
}


_SYMPY_FUNC_TO_IR: dict[type, str] = {
    sympy.sin: "sin",
    sympy.cos: "cos",
    sympy.exp: "exp",
    sympy.log: "log",
    _LAP_FUNC: "lap",
}


@dataclass
class FormattedEquation:

    latex: str
    unicode: str
    sympy_expr: Equality
    lhs: str
    rhs: Expr


def _fallback_symbol(code: str) -> Expr:
    return cast(Expr, sympy.Symbol(code))


def _parse_expression(code: str) -> ast.Expression:
    if not code.strip():
        raise ValueError("Expression cannot be empty")
    try:
        return ast.parse(code, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {code!r}") from exc


def _extract_names(tree: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _derivative_symbol_name(expr: Expr, axis: str, order: int) -> str:
    if isinstance(expr, sympy.Symbol):
        return build_derivative_name(expr.name, axis, order)

    if isinstance(expr, sympy.Function):
        base_str = str(expr)
        axis_suffix = axis * order
        return f"{base_str}_{axis_suffix}"
    return f"d{order}_{axis}"


def _make_diff_callable(name: str) -> Callable[[Expr], Expr] | None:
    match = _DIFF_NAME_PATTERN.match(name)
    if match is None:
        return None
    order_str, axis = match.groups()
    order = int(order_str) if order_str else 1
    diff_func = sympy.Function(name)

    def _diff(expr: Expr) -> Expr:


        expr = cast(Expr, sympy.sympify(expr))
        if expr.is_number:
            return cast(Expr, sympy.Integer(0))
        if isinstance(expr, sympy.Symbol):
            return cast(Expr, sympy.Symbol(_derivative_symbol_name(expr, axis, order)))
        if isinstance(expr, sympy.Add):
            return cast(Expr, sympy.Add(*(_diff(arg) for arg in expr.args)))
        if isinstance(expr, sympy.Mul):
            numeric: list[Expr] = []
            symbolic: list[Expr] = []
            for arg in expr.args:
                if arg.is_number:
                    numeric.append(cast(Expr, arg))
                else:
                    symbolic.append(cast(Expr, arg))
            if not symbolic:
                return cast(Expr, sympy.Integer(0))
            if len(symbolic) == 1:
                scalar = (
                    cast(Expr, sympy.Mul(*numeric)) if numeric else sympy.Integer(1)
                )
                return cast(Expr, scalar * _diff(symbolic[0]))
        return cast(Expr, diff_func(expr))

    return _diff


def _build_namespace(names: set[str]) -> dict[str, Any]:
    namespace: dict[str, Any] = dict(_KD_OPS)
    for name in names:
        if name in namespace:
            continue
        diff_callable = _make_diff_callable(name)
        if diff_callable is not None:
            namespace[name] = diff_callable
            continue
        namespace[name] = sympy.Symbol(name)
    return namespace


def to_sympy(code: str, *, strict: bool = True) -> Expr:
    try:
        tree = _parse_expression(code)
        compiled = compile(tree, "<kd-sympy>", "eval")
        namespace = _build_namespace(_extract_names(tree))
        result = eval(compiled, {"__builtins__": {}}, namespace)
        return cast(Expr, sympy.sympify(result))
    except Exception as exc:
        if strict:
            message = f"Failed to convert expression to SymPy: {code!r}"
            raise ValueError(message) from exc
        logger.debug("Falling back to Symbol(%r) after SymPy conversion failure", code)
        return _fallback_symbol(code)


def to_latex(code: str, *, strict: bool = True) -> str:
    return str(sympy.latex(to_sympy(code, strict=strict)))


def to_unicode(code: str, *, strict: bool = True) -> str:
    return str(
        sympy.pretty(to_sympy(code, strict=strict), use_unicode=True, wrap_line=False)
    )


def _coefficients_to_list(coefficients: Tensor | Sequence[float]) -> list[float]:
    if isinstance(coefficients, Tensor):
        values = coefficients.detach().cpu().flatten().tolist()
        return [float(value) for value in values]
    return [float(value) for value in coefficients]


def _selected_indices(
    n_terms: int,
    selected_indices: Sequence[int] | None,
) -> list[int]:
    if selected_indices is None:
        return list(range(n_terms))
    resolved = list(selected_indices)
    invalid = [i for i in resolved if not 0 <= i < n_terms]
    if invalid:
        raise ValueError(
            f"selected_indices out of range for {n_terms} terms: {invalid}"
        )
    return resolved


def _round_sig(value: float, sig_figs: int) -> float:
    if value == 0.0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    return round(value, sig_figs - 1 - magnitude)


def _term_contribution(term: str, coefficient: float, *, sig_figs: int = 0) -> Expr:
    if not math.isfinite(coefficient):
        raise ValueError("Coefficients must be finite")
    if coefficient == 0.0:
        return cast(Expr, sympy.Integer(0))
    c = _round_sig(coefficient, sig_figs) if sig_figs > 0 else coefficient

    if c == int(c) and abs(c) < 1e12:
        coeff_sym: Expr = cast(Expr, sympy.Integer(int(c)))
    else:
        coeff_sym = cast(Expr, sympy.Float(c))
    return cast(Expr, coeff_sym * to_sympy(term))


def _build_rhs(
    terms: Sequence[str],
    coefficient_values: Sequence[float],
    indices: Sequence[int],
    *,
    sig_figs: int = 0,
) -> Expr:
    rhs = cast(Expr, sympy.Integer(0))
    for index in indices:
        rhs += _term_contribution(
            terms[index], coefficient_values[index], sig_figs=sig_figs
        )
    return cast(Expr, sympy.expand(rhs))


_DEFAULT_SIG_FIGS = 4


def format_pde(
    terms: Sequence[str],
    coefficients: Tensor | Sequence[float],
    *,
    lhs: str = _DEFAULT_LHS_LABEL,
    selected_indices: Sequence[int] | None = None,
    sig_figs: int = _DEFAULT_SIG_FIGS,
) -> FormattedEquation:
    coefficient_values = _coefficients_to_list(coefficients)
    if len(terms) != len(coefficient_values):
        raise ValueError("terms and coefficients must have the same length")
    indices = _selected_indices(len(terms), selected_indices)
    rhs = _build_rhs(terms, coefficient_values, indices, sig_figs=sig_figs)
    lhs_symbol = cast(Expr, sympy.Symbol(lhs))
    equation = sympy.Eq(lhs_symbol, rhs, evaluate=False)
    return FormattedEquation(
        latex=str(sympy.latex(equation)),
        unicode=str(sympy.pretty(equation, use_unicode=True, wrap_line=False)),
        sympy_expr=equation,
        lhs=lhs,
        rhs=rhs,
    )


def are_equivalent(expr_a: str, expr_b: str) -> bool:
    return bool(sympy.expand(to_sympy(expr_a) - to_sympy(expr_b)) == 0)


def _fold_funcall(name: str, args: Sequence[str]) -> str:
    if not args:
        raise ValueError(f"Cannot fold empty argument list for {name}")
    if len(args) == 1:
        return args[0]
    result = args[-1]
    for arg in reversed(args[:-1]):
        result = f"{name}({arg}, {result})"
    return result


def _serialize_rational(expr: sympy.Rational) -> str:
    if expr.q == 1:
        return str(expr.p)
    return f"div({expr.p}, {expr.q})"


def _repeat_mul(base: str, exponent: int) -> str:
    terms = [base] * exponent
    return _fold_funcall("mul", terms)


def _serialize_pow(base: Expr, exponent: int) -> str:
    base_ir = _sympy_to_ir(base)
    if exponent == 2:
        return f"n2({base_ir})"
    if exponent == 3:
        return f"n3({base_ir})"
    if exponent > 0:
        return _repeat_mul(base_ir, exponent)
    if exponent == -1:
        return f"recip({base_ir})"




    return f"recip({_serialize_pow(base, -exponent)})"


def _sympy_to_ir(expr: Expr) -> str:
    if isinstance(expr, sympy.Symbol):
        return cast(str, expr.name)
    if isinstance(expr, sympy.Integer):
        return str(int(expr))
    if isinstance(expr, sympy.Float):
        return str(float(expr))
    if isinstance(expr, sympy.Rational):
        return _serialize_rational(expr)
    if isinstance(expr, sympy.Add):
        terms = [_sympy_to_ir(arg) for arg in expr.as_ordered_terms()]
        return _fold_funcall("add", terms)
    if isinstance(expr, sympy.Mul):
        return _fold_funcall(
            "mul",
            [_sympy_to_ir(arg) for arg in expr.as_ordered_factors()],
        )
    if isinstance(expr, sympy.Pow) and expr.exp.is_Integer:
        return _serialize_pow(expr.base, int(expr.exp))
    if isinstance(expr, sympy.Function) and _DIFF_NAME_PATTERN.match(
        type(expr).__name__
    ):
        if len(expr.args) != 1:
            raise ValueError(
                f"{type(expr).__name__}() expects 1 argument, got {len(expr.args)}"
            )
        return f"{type(expr).__name__}({_sympy_to_ir(expr.args[0])})"

    func_name = _SYMPY_FUNC_TO_IR.get(type(expr))
    if func_name is not None:
        if len(expr.args) != 1:
            raise ValueError(f"{func_name}() expects 1 argument, got {len(expr.args)}")
        return f"{func_name}({_sympy_to_ir(expr.args[0])})"
    raise ValueError(f"Unsupported SymPy expression for kd IR: {expr!r}")


def symbolic_diff(code: str, var: str) -> str:
    if not var:
        raise ValueError("var must be a non-empty symbol name")
    derivative = cast(
        Expr,
        sympy.expand(sympy.diff(to_sympy(code), sympy.Symbol(var))),
    )
    return _sympy_to_ir(derivative)


def from_sympy(expr: Expr) -> str:
    return _sympy_to_ir(expr)


__all__ = [
    "FormattedEquation",
    "are_equivalent",
    "format_pde",
    "from_sympy",
    "symbolic_diff",
    "to_latex",
    "to_sympy",
    "to_unicode",
]
