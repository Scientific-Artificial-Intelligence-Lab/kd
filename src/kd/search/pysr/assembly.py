
from __future__ import annotations

import logging
from typing import Any

import sympy

from kd.core.expr.sympy_bridge import are_equivalent
from kd.search.pysr.backend import HOFEntry, PySRBackend
from kd.search.pysr.convert import (
    pysr_sympy_to_kd_terms,
    pysr_sympy_to_tabular_term,
)
from kd.search.term_utils import fold_add as fold_add
from kd.search.term_utils import infer_max_atomic_order as infer_max_atomic_order

logger = logging.getLogger(__name__)


def convert_best(
    backend: PySRBackend,
    valid_terms: list[str],
    feature_names: list[str],
) -> str:
    try:
        kd_terms = pysr_sympy_to_kd_terms(
            backend.best_sympy(), valid_terms, feature_names
        )
    except ValueError as exc:
        raise RuntimeError(
            "PySR best expression is not convertible to kd IR "
            f"(outside the supported subset): {exc}"
        ) from exc
    if not kd_terms:
        raise RuntimeError(
            "PySR best expression reduces to a pure constant with no "
            "structural terms; kd Theta has no intercept column, so this "
            "degenerate best is rejected (honest hard fail, no degrade)."
        )
    return fold_add(kd_terms)


def try_convert_entry(
    entry: HOFEntry,
    valid_terms: list[str],
    feature_names: list[str],
) -> str | None:
    try:
        kd_terms = pysr_sympy_to_kd_terms(entry.sympy_expr, valid_terms, feature_names)
    except ValueError:
        return None
    if not kd_terms:
        return None
    return fold_add(kd_terms)


def convert_hall_of_fame(
    backend: PySRBackend,
    valid_terms: list[str],
    feature_names: list[str],
) -> tuple[list[str], list[tuple[int, float]]]:
    candidates: list[str] = []
    meta: list[tuple[int, float]] = []
    skipped = 0
    for entry in backend.hall_of_fame():
        converted = try_convert_entry(entry, valid_terms, feature_names)
        if converted is None:
            skipped += 1
            continue
        candidates.append(converted)
        meta.append((int(entry.complexity), float(entry.loss)))
    if skipped:
        logger.debug("Skipped %d unconvertible hall-of-fame entries", skipped)
    return candidates, meta


def convert_best_tabular(
    backend: PySRBackend,
    valid_terms: list[str],
    feature_names: list[str],
) -> str:
    best = backend.best_sympy()
    feature_symbols = {sympy.Symbol(name) for name in feature_names}
    if not (best.free_symbols & feature_symbols):
        raise RuntimeError(
            "PySR tabular best is a constant model with no feature symbols; "
            "kd has no intercept-only semantics for a tabular target"
        )
    try:
        return pysr_sympy_to_tabular_term(best, valid_terms, feature_names)
    except ValueError as exc:
        raise RuntimeError(
            "PySR tabular best is not convertible to kd IR (outside the "
            f"supported subset): {exc}"
        ) from exc


def convert_hall_of_fame_tabular(
    backend: PySRBackend,
    valid_terms: list[str],
    feature_names: list[str],
) -> tuple[list[str], list[tuple[int, float]]]:
    candidates: list[str] = []
    meta: list[tuple[int, float]] = []
    skipped = 0
    for entry in backend.hall_of_fame():
        try:
            converted = pysr_sympy_to_tabular_term(
                entry.sympy_expr, valid_terms, feature_names
            )
        except ValueError:
            skipped += 1
            continue
        candidates.append(converted)
        meta.append((int(entry.complexity), float(entry.loss)))
    if skipped:
        logger.debug("Skipped %d unconvertible tabular HOF entries", skipped)
    return candidates, meta


def match_selected_entry(
    best_expression: str,
    candidates: list[str],
    meta: list[tuple[int, float]],
) -> tuple[int | None, float | None]:
    if not best_expression:
        return None, None
    for candidate, (complexity, loss) in zip(candidates, meta, strict=True):
        if are_equivalent(candidate, best_expression):
            return complexity, loss
    return None, None


def reserved_names(dataset: Any) -> set[str]:
    reserved: set[str] = set()
    if dataset is None:
        return reserved
    fields = getattr(dataset, "fields", None)
    if fields is not None:
        reserved |= set(fields)
    axes = getattr(dataset, "axes", None)
    if axes is not None:
        reserved |= set(axes)
    return reserved
