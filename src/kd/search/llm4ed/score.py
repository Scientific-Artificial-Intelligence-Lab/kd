
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from kd.search.llm4ed.columns import build_columns
from kd.search.llm4ed.fd import OPERAND_ORDER
from kd.search.llm4ed.parse import (
    InexpressibleTermError,
    Llm4edParseError,
    UndefinedOperandError,
    UndefinedOperatorError,
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




ERROR_UNDEFINED_OPERANDS: Final[str] = "undefined operands"
ERROR_UNDEFINED_OPERATORS: Final[str] = "undefined operators"
ERROR_INEXPRESSIBLE: Final[str] = "inexpressible"
ERROR_PARSE: Final[str] = "parse_error"
ERROR_NON_FINITE: Final[str] = "non_finite_column"



__all__ = [
    "ERROR_ABNORMAL_COEF",
    "ERROR_INEXPRESSIBLE",
    "ERROR_LSTSQ",
    "ERROR_NON_FINITE",
    "ERROR_PARSE",
    "ERROR_UNDEFINED_OPERANDS",
    "ERROR_UNDEFINED_OPERATORS",
    "MAX_TERMS",
    "EquationScore",
    "remove_redundants",
    "score_equation",
]


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


def remove_redundants(
    columns: Sequence[FloatArray], term_strs: Sequence[str]
) -> tuple[list[FloatArray], list[str], bool]:
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
            delta = float(np.sum(np.abs(arr_compare) - np.abs(arr_current)))
            if abs(delta) < 1e-5:
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


def score_equation(
    equation: str,
    lhs: FloatArray,
    features: Mapping[str, FloatArray],
    *,
    operands: Sequence[str] = OPERAND_ORDER,
    l0_penalty: float = DEFAULT_L0_PENALTY,
    complexity_weight: float = DEFAULT_COMPLEXITY_WEIGHT,
    max_terms: int = MAX_TERMS,
) -> EquationScore:
    try:
        parsed = parse_equation(equation, operands)
    except Llm4edParseError as exc:
        logger.debug("parse-invalid candidate %r: %s", equation, exc)
        return _invalid(_parse_error_label(exc))

    column_result = build_columns(parsed, features)
    term_strs = column_result.term_strs
    if not column_result.valid:
        return _invalid(ERROR_NON_FINITE, term_strs)

    columns, dedup_strs, _duplicate = remove_redundants(
        list(column_result.columns), list(term_strs)
    )
    theta = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    lhs64 = np.ascontiguousarray(lhs, dtype=np.float64)

    solve = sparse_solve(theta, lhs64, l0_penalty=l0_penalty)
    if not solve.valid:
        assert solve.error_type is not None
        return _invalid(solve.error_type, tuple(dedup_strs))

    assert solve.coefficients is not None and solve.y_hat is not None
    n_terms = min(int(np.count_nonzero(solve.coefficients)), max_terms)
    reward = rounded_sparse_reward(
        lhs64.reshape(-1, 1),
        solve.y_hat,
        n_terms,
        complexity_weight=complexity_weight,
    )





    term_meta = {term.term_str: (term.base_ir, term.coeff) for term in parsed.terms}
    dedup_irs = tuple(term_meta[name][0] for name in dedup_strs)
    dedup_coeffs = tuple(term_meta[name][1] for name in dedup_strs)
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
