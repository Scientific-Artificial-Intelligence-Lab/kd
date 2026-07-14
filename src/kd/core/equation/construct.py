
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence

from torch import Tensor

from kd.core.equation.types import (
    Equation,
    EquationAttrs,
    Evolution,
    Homogeneous,
    LhsSpec,
    Scalar,
    Term,
)

logger = logging.getLogger(__name__)


def build_equation(
    term_irs: Sequence[str] | None,
    coefficients: Tensor | Sequence[float] | None,
    lhs_spec: LhsSpec | None,
    *,
    is_valid: bool = True,
) -> Equation | None:
    if not is_valid:
        logger.debug("Skipping equation derivation: final evaluation is invalid")
        return None
    if lhs_spec is None:
        logger.debug("Skipping equation derivation: missing lhs_spec")
        return None
    if term_irs is None or len(term_irs) == 0:
        logger.debug("Skipping equation derivation: no terms")
        return None
    if coefficients is None:
        logger.debug("Skipping equation derivation: missing coefficients")
        return None

    coefficient_values = _coefficient_values(coefficients)
    if len(term_irs) != len(coefficient_values):
        logger.debug(
            "Skipping equation derivation: term/coefficient length mismatch "
            "(terms=%d, coefficients=%d)",
            len(term_irs),
            len(coefficient_values),
        )
        return None

    for index, coefficient in enumerate(coefficient_values):
        if not math.isfinite(coefficient):
            logger.debug(
                "Skipping equation derivation: non-finite coefficient at index %d",
                index,
            )
            return None

    terms = tuple(
        (term_ir, Scalar(coefficient))
        for term_ir, coefficient in zip(term_irs, coefficient_values, strict=True)
    )
    return make_evolution(lhs_spec, terms)


def _coefficient_values(coefficients: Tensor | Sequence[float]) -> list[float]:
    if isinstance(coefficients, Tensor):
        return [float(value) for value in coefficients.detach().flatten().tolist()]
    return [float(value) for value in coefficients]


def build_homogeneous(
    term_irs: Sequence[str] | None,
    coefficients: Tensor | Sequence[float] | None,
    *,
    is_valid: bool = True,
) -> Equation | None:
    if not is_valid:
        logger.debug("Skipping equation derivation: final evaluation is invalid")
        return None
    if term_irs is None or len(term_irs) == 0:
        logger.debug("Skipping equation derivation: no terms")
        return None
    if coefficients is None:
        logger.debug("Skipping equation derivation: missing coefficients")
        return None

    coefficient_values = _coefficient_values(coefficients)
    if len(term_irs) != len(coefficient_values):
        logger.debug(
            "Skipping equation derivation: term/coefficient length mismatch "
            "(terms=%d, coefficients=%d)",
            len(term_irs),
            len(coefficient_values),
        )
        return None

    for index, coefficient in enumerate(coefficient_values):
        if not math.isfinite(coefficient):
            logger.debug(
                "Skipping equation derivation: non-finite coefficient at index %d",
                index,
            )
            return None

    terms = tuple(
        (term_ir, Scalar(1.0 if index == 0 else coefficient_values[index]))
        for index, term_ir in enumerate(term_irs)
    )
    return make_homogeneous(terms)


def make_evolution(lhs_spec: LhsSpec | None, terms: Iterable[Term]) -> Evolution:
    if lhs_spec is None:
        raise ValueError("EVOLUTION equations require lhs_spec")

    term_tuple = tuple(terms)
    if not term_tuple:
        raise ValueError("EVOLUTION equations require at least one term")

    if any(term_ir == "" for term_ir, _coefficient in term_tuple):
        raise ValueError("EVOLUTION equation term IR strings must be non-empty")

    return Evolution(
        lhs_spec=lhs_spec,
        terms=term_tuple,
        attrs=EquationAttrs(),
    )


def make_homogeneous(terms: Iterable[Term]) -> Homogeneous:
    term_tuple = tuple(terms)
    if not term_tuple:
        raise ValueError("HOMOGENEOUS equations require at least one term")

    if any(term_ir == "" for term_ir, _coefficient in term_tuple):
        raise ValueError("HOMOGENEOUS equation term IR strings must be non-empty")

    return Homogeneous(
        terms=term_tuple,
        attrs=EquationAttrs(),
    )
