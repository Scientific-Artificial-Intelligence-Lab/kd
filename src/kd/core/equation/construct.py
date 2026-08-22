
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from typing import Final, cast

from torch import Tensor

from kd.core.equation.gauge import regression_term_gauge
from kd.core.equation.types import (
    Equation,
    EquationAttrs,
    Evolution,
    Homogeneous,
    LhsSpec,
    Regression,
    Scalar,
    Term,
)

logger = logging.getLogger(__name__)

PIVOT_UNITY_RTOL: Final[float] = 1e-6


def build_equation(
    term_irs: Sequence[str] | None,
    coefficients: Tensor | Sequence[float] | None,
    lhs_spec: LhsSpec | None,
    *,
    active_indices: Sequence[int] | None = None,
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

        logger.warning(
            "Skipping equation derivation: term/coefficient length mismatch "
            "(terms=%d, coefficients=%d)",
            len(term_irs),
            len(coefficient_values),
        )
        return None

    for index, coefficient in enumerate(coefficient_values):
        if not math.isfinite(coefficient):

            logger.warning(
                "Skipping equation derivation: non-finite coefficient at index %d",
                index,
            )
            return None

    terms = tuple(
        (term_ir, Scalar(coefficient))
        for term_ir, coefficient in zip(term_irs, coefficient_values, strict=True)
    )
    return make_evolution(
        lhs_spec,
        terms,
        active_indices=_degrade_active_indices(active_indices, len(terms)),
    )


def _coefficient_values(coefficients: Tensor | Sequence[float]) -> list[float]:
    if isinstance(coefficients, Tensor):
        return [float(value) for value in coefficients.detach().flatten().tolist()]
    return [float(value) for value in coefficients]


def _validate_active_indices(
    active_indices: Sequence[int] | None, n_terms: int
) -> tuple[int, ...] | None:
    if active_indices is None:
        return None
    indices = tuple(active_indices)
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"active_indices entries must be int, got {index!r}")
        if not 0 <= index < n_terms:
            raise ValueError(
                f"active_indices entry {index} out of range for {n_terms} terms"
            )
    return indices


def _degrade_active_indices(
    active_indices: Sequence[int] | None, n_terms: int
) -> tuple[int, ...] | None:
    try:
        return _validate_active_indices(active_indices, n_terms)
    except ValueError as exc:
        logger.warning("Dropping active_indices metadata: %s", exc)
        return None


def build_homogeneous(
    term_irs: Sequence[str] | None,
    coefficients: Tensor | Sequence[float] | None,
    *,
    active_indices: Sequence[int] | None = None,
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

        logger.warning(
            "Skipping equation derivation: term/coefficient length mismatch "
            "(terms=%d, coefficients=%d)",
            len(term_irs),
            len(coefficient_values),
        )
        return None

    for index, coefficient in enumerate(coefficient_values):
        if not math.isfinite(coefficient):

            logger.warning(
                "Skipping equation derivation: non-finite coefficient at index %d",
                index,
            )
            return None

    if not math.isclose(
        coefficient_values[0],
        1.0,
        rel_tol=PIVOT_UNITY_RTOL,
        abs_tol=PIVOT_UNITY_RTOL,
    ):
        logger.warning(
            "Skipping homogeneous equation derivation: pivot coefficient must "
            "be pre-normalized by the producer to [1.0, *w] (got %r)",
            coefficient_values[0],
        )
        return None

    terms = tuple(
        (term_ir, Scalar(1.0 if index == 0 else coefficient_values[index]))
        for index, term_ir in enumerate(term_irs)
    )
    return make_homogeneous(
        terms,
        active_indices=_degrade_active_indices(active_indices, len(terms)),
    )


def build_regression(
    term_irs: Sequence[str] | None,
    coefficients: Tensor | Sequence[float] | None,
    lhs_spec: LhsSpec | None,
    *,
    active_indices: Sequence[int] | None = None,
    is_valid: bool = True,
) -> Equation | None:
    if lhs_spec is not None and (
        not lhs_spec.field or lhs_spec.axis != "" or lhs_spec.order != 0
    ):
        logger.debug("Skipping regression equation derivation: invalid lhs_spec")
        return None
    equation = build_equation(
        term_irs,
        coefficients,
        lhs_spec,
        active_indices=active_indices,
        is_valid=is_valid,
    )
    if equation is None:
        return None
    evolution = cast(Evolution, equation)
    return make_regression(
        evolution.lhs_spec,
        evolution.terms,
        active_indices=evolution.active_indices,
    )


def make_evolution(
    lhs_spec: LhsSpec | None,
    terms: Iterable[Term],
    *,
    active_indices: Sequence[int] | None = None,
) -> Evolution:
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
        active_indices=_validate_active_indices(active_indices, len(term_tuple)),
    )


def make_homogeneous(
    terms: Iterable[Term],
    *,
    active_indices: Sequence[int] | None = None,
) -> Homogeneous:
    term_tuple = tuple(terms)
    if not term_tuple:
        raise ValueError("HOMOGENEOUS equations require at least one term")

    if any(term_ir == "" for term_ir, _coefficient in term_tuple):
        raise ValueError("HOMOGENEOUS equation term IR strings must be non-empty")

    return Homogeneous(
        terms=term_tuple,
        attrs=EquationAttrs(),
        active_indices=_validate_active_indices(active_indices, len(term_tuple)),
    )


def make_regression(
    lhs_spec: LhsSpec,
    terms: Iterable[Term],
    *,
    active_indices: Sequence[int] | None = None,
) -> Regression:
    if not lhs_spec.field:
        raise ValueError("REGRESSION equations require a non-empty target field")
    if lhs_spec.axis != "" or lhs_spec.order != 0:
        raise ValueError("REGRESSION lhs_spec must have axis='' and order=0")

    term_tuple = tuple(terms)
    if not term_tuple:
        raise ValueError("REGRESSION equations require at least one term")
    if any(term_ir == "" for term_ir, _coefficient in term_tuple):
        raise ValueError("REGRESSION equation term IR strings must be non-empty")
    indices = _validate_active_indices(active_indices, len(term_tuple))













    active_terms = (
        term_tuple
        if indices is None
        else [term_tuple[index] for index in sorted(set(indices))]
    )
    skeletons = [
        regression_term_gauge(term_ir)[0] for term_ir, _coefficient in active_terms
    ]
    if len(set(skeletons)) != len(skeletons):
        raise ValueError(
            "REGRESSION equation terms must have distinct constant skeletons"
        )

    return Regression(
        lhs_spec=lhs_spec,
        terms=term_tuple,
        attrs=EquationAttrs(),
        active_indices=indices,
    )
