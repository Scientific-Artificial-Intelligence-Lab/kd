
from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import TYPE_CHECKING

from kd.core.equation._sketch_terms import lower_term
from kd.core.equation.sketch import (
    AnchoredTerm,
    PinnedTerm,
    Sketch,
    SketchMatchPolicy,
    TermConstraint,
    TermHole,
)
from kd.core.equation.types import LhsSpec
from kd.core.expr.executor import DIFF_OPERATOR_PATTERN
from kd.core.expr.registry import FunctionRegistry
from kd.core.expr.term_features import TermVocabulary, analyze_term

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset

ADDITIONAL_HOLE_ID = "additional"

ZERO_MATCH_POLICY = SketchMatchPolicy(
    coeff_atol=0.0, coeff_rtol=0.0, support_threshold=0.0
)


def sketch_from_anchors(
    dataset: PDEDataset,
    anchors: Sequence[str],
    *,
    min_additional_terms: int,
    max_additional_terms: int,
    max_derivative_order: int,
) -> Sketch:
    vocabulary = dataset_vocabulary(dataset)
    irs = _evaluable_anchors(anchors, vocabulary)
    return dataset_sketch(
        dataset,
        vocabulary=vocabulary,
        pinned=(),
        anchored=tuple(AnchoredTerm(term) for term in irs),
        holes=_additional_hole(
            min_additional_terms, max_additional_terms, max_derivative_order
        ),
    )


def dataset_vocabulary(
    dataset: PDEDataset, override: TermVocabulary | None = None
) -> TermVocabulary:
    vocabulary = TermVocabulary(
        fields=frozenset(dataset.fields or ()),
        coordinates=frozenset(dataset.axes or ()),
    )
    if override is None:
        return vocabulary
    fields = sorted(override.fields - vocabulary.fields)
    axes = sorted(override.coordinates - vocabulary.coordinates)
    if fields or axes:
        raise ValueError(
            f"vocabulary names fields {fields} and axes {axes} the dataset does "
            f"not have (fields {sorted(vocabulary.fields)}, axes "
            f"{sorted(vocabulary.coordinates)})"
        )
    return override


def dataset_sketch(
    dataset: PDEDataset,
    *,
    vocabulary: TermVocabulary,
    pinned: tuple[PinnedTerm, ...],
    anchored: tuple[AnchoredTerm, ...],
    holes: tuple[TermHole, ...],
    lhs_spec: LhsSpec | None = None,
    match_policy: SketchMatchPolicy | None = None,
) -> Sketch:
    bound = LhsSpec(dataset.lhs_field, dataset.lhs_axis, dataset.lhs_order)
    if lhs_spec is not None and lhs_spec != bound:
        raise ValueError(
            f"lhs_spec {lhs_spec!r} differs from the dataset's LHS {bound!r}; "
            "retarget the dataset (dataset.retarget(...)) to change the target"
        )
    return Sketch(
        lhs_spec=bound,
        vocabulary=vocabulary,
        pinned=pinned,
        anchored=anchored,
        holes=holes,
        match_policy=ZERO_MATCH_POLICY if match_policy is None else match_policy,
    )


def term_refusal(
    term: str, vocabulary: TermVocabulary, registry: FunctionRegistry
) -> str | None:
    try:
        analyze_term(term, vocabulary)
    except ValueError as exc:
        return str(exc)
    return _operator_misuse(term, registry)


def _evaluable_anchors(anchors: Sequence[str], vocabulary: TermVocabulary) -> list[str]:
    registry = FunctionRegistry.create_default()
    irs: list[str] = []
    refused: list[str] = []
    for term in anchors:
        try:
            ir = lower_term(term)
        except ValueError as exc:
            refused.append(f"{term!r} ({exc})")
            continue
        reason = term_refusal(ir, vocabulary, registry)
        if reason is not None:
            refused.append(f"{term!r} ({reason})")
        irs.append(ir)
    if refused:
        raise ValueError(
            "anchors kd cannot evaluate on this dataset (fields "
            f"{sorted(vocabulary.fields)}, coordinates "
            f"{sorted(vocabulary.coordinates)}): {'; '.join(refused)}"
        )
    return irs


def _operator_misuse(term: str, registry: FunctionRegistry) -> str | None:
    for node in ast.walk(ast.parse(term, mode="eval")):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        head = node.func.id
        if DIFF_OPERATOR_PATTERN.fullmatch(head):
            continue
        if not registry.has(head):
            return f"operator {head!r} is not in the operator registry"
        arity = registry.get_arity(head)
        if len(node.args) != arity:
            return f"{head!r} takes {arity} arguments, got {len(node.args)}"
    return None


def _additional_hole(
    min_additional_terms: int,
    max_additional_terms: int,
    max_derivative_order: int,
) -> tuple[TermHole, ...]:
    if not 0 <= min_additional_terms <= max_additional_terms:
        raise ValueError(
            "min_additional_terms and max_additional_terms must satisfy "
            f"0 <= min_additional_terms <= max_additional_terms; got "
            f"{min_additional_terms} and {max_additional_terms}"
        )
    if max_additional_terms == 0:
        return ()
    return (
        TermHole(
            id=ADDITIONAL_HOLE_ID,
            min_count=min_additional_terms,
            max_count=max_additional_terms,
            constraint=TermConstraint(max_deriv_order=max_derivative_order),
        ),
    )


__all__ = [
    "ADDITIONAL_HOLE_ID",
    "ZERO_MATCH_POLICY",
    "dataset_sketch",
    "dataset_vocabulary",
    "sketch_from_anchors",
    "term_refusal",
]
