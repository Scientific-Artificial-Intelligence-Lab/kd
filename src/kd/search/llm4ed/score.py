
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
import sympy
import torch

from kd.core.equation.gauge import regression_term_gauge
from kd.core.expr.constant_fit import (
    fit_placeholders,
    placeholder_names,
    substitute_placeholders,
)
from kd.core.expr.executor import UNITY_TOKEN
from kd.search.llm4ed.columns import build_columns, build_llm4ed_registry
from kd.search.llm4ed.fd import OPERAND_ORDER
from kd.search.llm4ed.parse import (
    InexpressibleTermError,
    Llm4edParseError,
    ParsedEquation,
    ParsedTerm,
    UndefinedOperandError,
    UndefinedOperatorError,
    is_placeholder,
    parse_equation,
)
from kd.search.llm4ed.reward import DEFAULT_COMPLEXITY_WEIGHT, rounded_sparse_reward
from kd.search.llm4ed.stridge import (
    DEFAULT_L0_PENALTY,
    ERROR_ABNORMAL_COEF,
    ERROR_LSTSQ,
    sparse_solve,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


MAX_TERMS: Final[int] = 5




_DUPLICATE_RTOL: Final[float] = 1e-9



_REGISTRY: Final = build_llm4ed_registry()




ERROR_UNDEFINED_OPERANDS: Final[str] = "undefined operands"
ERROR_UNDEFINED_OPERATORS: Final[str] = "undefined operators"
ERROR_INEXPRESSIBLE: Final[str] = "inexpressible"
ERROR_PARSE: Final[str] = "parse_error"
ERROR_NON_FINITE: Final[str] = "non_finite_column"
ERROR_SKETCH: Final[str] = "sketch_reject"



__all__ = [
    "ERROR_ABNORMAL_COEF",
    "ERROR_INEXPRESSIBLE",
    "ERROR_LSTSQ",
    "ERROR_NON_FINITE",
    "ERROR_PARSE",
    "ERROR_SKETCH",
    "ERROR_UNDEFINED_OPERANDS",
    "ERROR_UNDEFINED_OPERATORS",
    "MAX_TERMS",
    "EquationScore",
    "TermGate",
    "remove_redundants",
    "score_equation",
]


class TermGate(Protocol):

    def refusal(self, term_irs: Sequence[str]) -> str | None:
        pass

    def protected(self, term_irs: Sequence[str]) -> tuple[int, ...]:
        pass


@dataclass(frozen=True)
class EquationScore:

    valid: bool
    reward: float | None
    error_type: str | None
    coefficients: FloatArray | None
    n_terms: int
    term_strs: tuple[str, ...]
    term_irs: tuple[str, ...]
    term_coeffs: tuple[float, ...]
    y_hat: FloatArray | None


def _invalid(error_type: str, term_strs: tuple[str, ...] = ()) -> EquationScore:
    return EquationScore(
        valid=False,
        reward=None,
        error_type=error_type,
        coefficients=None,
        n_terms=0,
        term_strs=term_strs,
        term_irs=(),
        term_coeffs=(),
        y_hat=None,
    )


def _abs_sum_duplicate(current: FloatArray, compare: FloatArray) -> bool:
    delta = float(np.sum(np.abs(compare) - np.abs(current)))
    return abs(delta) < 1e-5


def _elementwise_duplicate(current: FloatArray, compare: FloatArray) -> bool:
    scale = max(float(np.max(np.abs(current))), float(np.max(np.abs(compare))))
    tolerance = _DUPLICATE_RTOL * scale
    return bool(
        np.all(np.abs(compare - current) <= tolerance)
        or np.all(np.abs(compare + current) <= tolerance)
    )


def remove_redundants(
    columns: Sequence[FloatArray],
    term_strs: Sequence[str],
    *,
    elementwise: bool = False,
) -> tuple[list[FloatArray], list[str], bool]:
    is_duplicate = _elementwise_duplicate if elementwise else _abs_sum_duplicate
    terms = list(columns)
    tokens = list(term_strs)
    unique_values: list[FloatArray] = []
    unique_tokens: list[str] = []
    duplicate = False
    for i in range(len(terms)):
        arr_current = terms[i]
        str_current = tokens[i]
        duplicate_found = False
        for j in range(i + 1, len(terms)):
            arr_compare = terms[j]
            str_compare = tokens[j]
            if is_duplicate(arr_current, arr_compare):
                duplicate_found = True
                duplicate = True
                if len(str_current) < len(str_compare):
                    tokens[j] = tokens[i]
                    terms[j] = terms[i]
        if not duplicate_found:
            unique_values.append(arr_current)
            unique_tokens.append(tokens[i])
    return unique_values, unique_tokens, duplicate


def _parse_error_label(exc: Llm4edParseError) -> str:
    if isinstance(exc, UndefinedOperandError):
        return ERROR_UNDEFINED_OPERANDS
    if isinstance(exc, UndefinedOperatorError):
        return ERROR_UNDEFINED_OPERATORS
    if isinstance(exc, InexpressibleTermError):
        return ERROR_INEXPRESSIBLE
    return ERROR_PARSE


def _has_non_finite_literal(parsed: ParsedEquation) -> bool:
    return any(
        not math.isfinite(float(number))
        for number in parsed.sympy_expr.atoms(sympy.Number)
    )


def _active_skeletons_collide(
    term_irs: Sequence[str], coefficients: FloatArray
) -> bool:
    skeletons = [
        regression_term_gauge(term_ir)[0]
        for term_ir, coefficient in zip(term_irs, np.ravel(coefficients), strict=True)
        if coefficient != 0
    ]
    return len(set(skeletons)) != len(skeletons)


def _has_placeholder(parsed: ParsedEquation) -> bool:
    return any(is_placeholder(str(symbol)) for symbol in parsed.sympy_expr.free_symbols)


def _is_feature_free(expr: sympy.Expr) -> bool:
    symbols = [str(symbol) for symbol in expr.free_symbols]
    return bool(symbols) and all(is_placeholder(symbol) for symbol in symbols)


def _intercept_terms(parsed: ParsedEquation) -> list[ParsedTerm]:
    exprs = sympy.Add.make_args(parsed.sympy_expr)
    has_constant_term = any(expr.is_Number for expr in exprs)
    terms: list[ParsedTerm] = []
    for term, expr in zip(parsed.terms, exprs, strict=True):
        if not _is_feature_free(expr):
            terms.append(term)
        elif not has_constant_term:
            terms.append(replace(term, ir=UNITY_TOKEN, base_ir=UNITY_TOKEN, coeff=1.0))
    return terms


def _fit_placeholders(
    parsed: ParsedEquation, lhs: FloatArray, features: Mapping[str, FloatArray]
) -> ParsedEquation:
    terms = _intercept_terms(parsed)
    term_irs = [term.ir for term in terms]
    columns = {
        name: torch.from_numpy(np.ascontiguousarray(column, dtype=np.float64))
        for name, column in features.items()
    }
    target = torch.from_numpy(np.ascontiguousarray(lhs, dtype=np.float64).reshape(-1))
    fitted_irs, constants = fit_placeholders(
        term_irs, columns, target, registry=_REGISTRY
    )
    values = dict(zip(placeholder_names(term_irs), constants.tolist(), strict=True))
    fitted = tuple(
        replace(term, ir=ir, base_ir=substitute_placeholders(term.base_ir, values))
        for term, ir in zip(terms, fitted_irs, strict=True)
    )
    return replace(parsed, terms=fitted)


def score_equation(
    equation: str,
    lhs: FloatArray,
    features: Mapping[str, FloatArray],
    *,
    operands: Sequence[str] = OPERAND_ORDER,
    functions: frozenset[str] = frozenset(),
    placeholders: bool = False,
    constant_term: bool = False,
    elementwise_dedup: bool = False,
    target: str | None = None,
    distinct_skeletons: bool = False,
    l0_penalty: float = DEFAULT_L0_PENALTY,
    complexity_weight: float = DEFAULT_COMPLEXITY_WEIGHT,
    max_terms: int = MAX_TERMS,
    gate: TermGate | None = None,
) -> EquationScore:
    try:
        parsed = parse_equation(
            equation,
            operands,
            functions=functions,
            placeholders=placeholders,
            constant_term=constant_term,
            target=target,
        )
    except Llm4edParseError as exc:
        logger.debug("parse-invalid candidate %r: %s", equation, exc)
        return _invalid(_parse_error_label(exc))
    if _has_non_finite_literal(parsed):
        logger.debug("non-finite literal in candidate %r", equation)
        return _invalid(
            ERROR_INEXPRESSIBLE, tuple(term.term_str for term in parsed.terms)
        )
    if placeholders and _has_placeholder(parsed):
        parsed = _fit_placeholders(parsed, lhs, features)

    column_result = build_columns(parsed, features)
    term_strs = column_result.term_strs
    if not column_result.valid:
        return _invalid(ERROR_NON_FINITE, term_strs)

    columns, dedup_strs, _duplicate = remove_redundants(
        list(column_result.columns), list(term_strs), elementwise=elementwise_dedup
    )





    term_meta = {term.term_str: (term.base_ir, term.coeff) for term in parsed.terms}
    dedup_irs = tuple(term_meta[name][0] for name in dedup_strs)
    dedup_coeffs = tuple(term_meta[name][1] for name in dedup_strs)
    protected: tuple[int, ...] = ()
    if gate is not None:
        refusal = gate.refusal(dedup_irs)
        if refusal is not None:
            logger.debug("sketch refuses candidate %r: %s", equation, refusal)
            return _invalid(ERROR_SKETCH, tuple(dedup_strs))
        protected = gate.protected(dedup_irs)
    theta = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    lhs64 = np.ascontiguousarray(lhs, dtype=np.float64)

    try:
        solve = sparse_solve(theta, lhs64, l0_penalty=l0_penalty, protected=protected)
    except np.linalg.LinAlgError as exc:



        logger.debug("STRidge failed on candidate %r: %s", equation, exc)
        return _invalid(ERROR_LSTSQ, tuple(dedup_strs))
    if not solve.valid:
        assert solve.error_type is not None
        return _invalid(solve.error_type, tuple(dedup_strs))

    assert solve.coefficients is not None and solve.y_hat is not None
    if distinct_skeletons and _active_skeletons_collide(dedup_irs, solve.coefficients):
        logger.debug("constant-skeleton collision in candidate %r", equation)
        return _invalid(ERROR_INEXPRESSIBLE, tuple(dedup_strs))
    n_terms = min(int(np.count_nonzero(solve.coefficients)), max_terms)
    reward = rounded_sparse_reward(
        lhs64.reshape(-1, 1),
        solve.y_hat,
        n_terms,
        complexity_weight=complexity_weight,
    )
    return EquationScore(
        valid=True,
        reward=reward,
        error_type=None,
        coefficients=solve.coefficients,
        n_terms=n_terms,
        term_strs=tuple(dedup_strs),
        term_irs=dedup_irs,
        term_coeffs=dedup_coeffs,
        y_hat=solve.y_hat,
    )
